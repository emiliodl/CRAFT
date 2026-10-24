"""Arquitetura no molde do TF-C, com um encoder por vista do agregado.

No TF-C original ha dois ramos (tempo e frequencia), cada um com seu encoder G,
seu projetor R, sua perda contrastiva propria, e uma perda de consistencia que
aproxima os dois espacos projetados. Aqui os ramos sao TRES, e nenhum deles e
frequencia: sao tres transformacoes da mesma serie temporal do agregado (base
removida, z robusto, arcsinh). O resto do esqueleto e o mesmo.

A tarefa e seq2seq: entra um dia inteiro (288 janelas de 5 min), saem 288
rotulos, um por janela de 5 min. Entao cada encoder devolve a sequencia de
tokens (nao so um vetor), e:

  - a FUSAO e a CABECA trabalham token a token, para produzir os 288 rotulos;
  - as PERDAS CONTRASTIVAS usam o resumo h_v (media dos tokens), que e o mesmo
    papel do h_i do artigo.

    x --(3 vistas)--> G_v --> H_v (tokens) --> fusao --> cabeca --> 288 logits
                               |
                               +--> h_v = media --> R_v --> z_v --> L_v, L_C

Uma instancia deste modelo classifica UM aparelho. Sao tres modelos
independentes, um por carga - nao ha cabeca multirrotulo.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CodificadorVista(nn.Module):
    """G_v: Transformer sobre uma vista (1 canal, L passos) -> tokens (B, n, d)."""

    def __init__(self, largura: int, d: int = 96, camadas: int = 3, cabecas: int = 4,
                 patch: int = 4, dropout: float = 0.1):
        super().__init__()
        if largura % patch:
            raise ValueError("largura deve ser multipla do patch")
        self.patch, self.dim = patch, d
        n = largura // patch
        self.entrada = nn.Linear(patch, d)
        self.pos = nn.Parameter(torch.zeros(1, n, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        camada = nn.TransformerEncoderLayer(
            d_model=d, nhead=cabecas, dim_feedforward=4 * d, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.corpo = nn.TransformerEncoder(camada, num_layers=camadas)
        self.norma = nn.LayerNorm(d)

    def forward(self, x: Tensor) -> Tensor:              # (B, L) -> (B, n, d)
        b, comprimento = x.shape
        p = x.view(b, comprimento // self.patch, self.patch)
        return self.norma(self.corpo(self.entrada(p) + self.pos))


class Projetor(nn.Module):
    """R_v: leva o resumo h_v ao espaco onde as perdas contrastivas vivem."""

    def __init__(self, d: int, saida: int = 32):
        super().__init__()
        self.rede = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.GELU(), nn.Linear(d, saida))

    def forward(self, h: Tensor) -> Tensor:
        return F.normalize(self.rede(h), dim=-1)


class FusaoPorta(nn.Module):
    """Media ponderada dos tokens das vistas, com pesos previstos token a token.

    Fusao padrao por ser a unica legivel: da para reportar quanto cada vista
    pesou, por aparelho e ate por instante do dia. Concatenacao e soma nao
    entregam isso.
    """

    def __init__(self, d: int, n_vistas: int):
        super().__init__()
        self.porta = nn.Sequential(nn.Linear(n_vistas * d, d), nn.GELU(), nn.Linear(d, n_vistas))

    def forward(self, tokens: list[Tensor]) -> tuple[Tensor, Tensor]:
        emp = torch.stack(tokens, dim=2)                  # (B, n, V, d)
        peso = torch.softmax(self.porta(emp.flatten(2)), dim=-1)      # (B, n, V)
        return (emp * peso.unsqueeze(-1)).sum(dim=2), peso


class Classificador(nn.Module):
    """V encoders, V projetores, fusao token a token e cabeca seq2seq."""

    def __init__(self, vistas: list[str], largura: int, d: int = 96, camadas: int = 3,
                 fusao: str = "porta", patch: int = 4, proj: int = 32, dropout: float = 0.1):
        super().__init__()
        self.vistas, self.fusao, self.patch, self.largura = list(vistas), fusao, patch, largura
        v = len(vistas)
        self.encoders = nn.ModuleList(
            [CodificadorVista(largura, d, camadas, patch=patch, dropout=dropout) for _ in range(v)])
        self.projetores = nn.ModuleList([Projetor(d, proj) for _ in range(v)])

        if fusao == "concat":
            dim, self.porta = v * d, None
        elif fusao == "soma":
            dim, self.porta = d, None
        elif fusao == "porta":
            self.porta, dim = FusaoPorta(d, v), d
        else:
            raise ValueError(f"fusao desconhecida: {fusao}")

        self.cabeca = nn.Sequential(
            nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, d), nn.GELU(),
            nn.Linear(d, patch))

    def representar(self, x: Tensor) -> tuple[Tensor, list[Tensor], Tensor | None]:
        """x: (B, V, L) -> (tokens fundidos, [h_v resumidos], pesos da porta)."""
        tokens = [enc(x[:, i]) for i, enc in enumerate(self.encoders)]
        resumos = [t.mean(dim=1) for t in tokens]
        if self.fusao == "concat":
            return torch.cat(tokens, dim=-1), resumos, None
        if self.fusao == "soma":
            return torch.stack(tokens, 0).sum(0), resumos, None
        fundido, peso = self.porta(tokens)
        return fundido, resumos, peso

    def forward(self, x: Tensor) -> Tensor:               # (B, V, L) -> (B, L)
        fundido, _, _ = self.representar(x)
        return self.cabeca(fundido).reshape(x.shape[0], self.largura)

    def pesos_das_vistas(self, x: Tensor) -> Tensor | None:
        _, _, peso = self.representar(x)
        return None if peso is None else peso.mean(dim=(0, 1))

    def projetar(self, resumos: list[Tensor]) -> list[Tensor]:
        return [p(h) for p, h in zip(self.projetores, resumos)]


# --------------------------------------------------------------------------------------
# Perdas do TF-C
# --------------------------------------------------------------------------------------


def nt_xent(a: Tensor, b: Tensor, tau: float = 0.2) -> Tensor:
    """NT-Xent entre dois conjuntos alinhados de embeddings normalizados."""
    n = a.shape[0]
    z = torch.cat([a, b], dim=0)
    sim = (z @ z.t()) / tau
    sim.fill_diagonal_(-1e4)
    alvo = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(a.device)
    return F.cross_entropy(sim, alvo)


def perda_contrastiva(modelo: Classificador, x: Tensor, x_aug: Tensor,
                      tau: float = 0.2) -> tuple[Tensor, Tensor]:
    """L_v somada sobre as vistas, e L_C entre pares de vistas.

    L_v : cada vista contra sua propria versao aumentada (o L_T / L_F do artigo).
    L_C : para cada par de vistas, o mesmo dia e positivo e os demais dias do
          lote sao negativos - generalizacao direta da perda de consistencia do
          TF-C de dois para tres ramos.
    """
    _, hs, _ = modelo.representar(x)
    _, hs_aug, _ = modelo.representar(x_aug)
    zs, zs_aug = modelo.projetar(hs), modelo.projetar(hs_aug)

    l_vista = sum(nt_xent(z, za, tau) for z, za in zip(zs, zs_aug)) / len(zs)
    if len(zs) == 1:
        return l_vista, torch.zeros((), device=x.device)
    pares = [(i, j) for i in range(len(zs)) for j in range(i + 1, len(zs))]
    l_cons = sum(nt_xent(zs[i], zs[j], tau) for i, j in pares) / len(pares)
    return l_vista, l_cons


def aumentar(x: Tensor, ruido: float = 0.1, escala: float = 0.1, mascara: float = 0.15) -> Tensor:
    """Jitter + escala + mascara temporal, por vista.

    Aumentos genericos de serie temporal, sem nada especifico de NILM: a hipotese
    do TF-C e que a invariancia util emerge da consistencia entre os ramos, e nao
    de aumentos desenhados a mao.
    """
    b, v, comprimento = x.shape
    y = x + ruido * torch.randn_like(x)
    y = y * (1 + escala * torch.randn(b, v, 1, device=x.device))
    return y * (torch.rand(b, v, comprimento, device=x.device) > mascara).float()


def contar_parametros(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
