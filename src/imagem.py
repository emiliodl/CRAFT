"""Transformadas imagem-de-serie na GPU, e um Vision Transformer sobre elas.

Seq2point: uma janela de contexto vira uma imagem, e sai UM rotulo. A janela de
24 h (288 amostras a 5 min) e comprimida por PAA antes de virar imagem, senao a
matriz seria 288x288 - e o padrao nos trabalhos de GAF e justamente aplicar PAA
primeiro.

Canais disponiveis:

  gasf  cos(phi_i + phi_j), phi = arccos(x~), x~ reescalado para [-1,1]
  gadf  sin(phi_i - phi_j)
  mtf   Markov Transition Field, faixas por quantil da janela
  rp    |x_i - x_j| sobre o sinal padronizado (nao reescalado)
  amp   produto externo x_i * x_j sobre o sinal padronizado

gasf, gadf e mtf reescalam cada janela pelo proprio min-max ou pelos proprios
quantis, entao sao invariantes a escala da casa - o que ataca a falha de
transferencia REFIT -> ECO - mas descartam a amplitude absoluta, que e o que
distingue uma maquina de lavar de 2 kW. `rp` e `amp` existem para reinjetar
essa amplitude, e ficam como ablacao.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

CANAIS = ("gasf", "gadf", "mtf", "rp", "amp")


def paa(x: Tensor, n: int) -> Tensor:
    """Piecewise aggregate approximation: (B, L) -> (B, n), L multiplo de n."""
    b, comprimento = x.shape
    if comprimento == n:
        return x
    if comprimento % n:
        raise ValueError(f"comprimento {comprimento} nao e multiplo de {n}")
    return x.view(b, n, comprimento // n).mean(dim=2)


def _reescalar(x: Tensor) -> Tensor:
    lo = x.min(dim=1, keepdim=True).values
    hi = x.max(dim=1, keepdim=True).values
    faixa = (hi - lo).clamp_min(1e-9)
    return (((x - hi) + (x - lo)) / faixa).clamp(-1.0, 1.0)


def _mtf(x: Tensor, faixas: int = 8) -> Tensor:
    """(B, n) -> (B, n, n). Quantis, matriz de transicao e leitura em (q_i, q_j)."""
    b, n = x.shape
    cortes = torch.quantile(x, torch.linspace(0, 1, faixas + 1, device=x.device)[1:-1], dim=1)
    q = (x.unsqueeze(0) > cortes.unsqueeze(-1)).sum(dim=0)          # (B, n) em [0, faixas-1]
    par = q[:, :-1] * faixas + q[:, 1:]                             # (B, n-1)
    conta = torch.zeros(b, faixas * faixas, device=x.device)
    conta.scatter_add_(1, par, torch.ones_like(par, dtype=conta.dtype))
    w = conta.view(b, faixas, faixas)
    w = w / w.sum(dim=2, keepdim=True).clamp_min(1.0)
    linha = w.gather(1, q.unsqueeze(-1).expand(b, n, faixas))       # (B, n, faixas)
    return linha.gather(2, q.unsqueeze(1).expand(b, n, n))


def _fatiar(x: Tensor, comprimento: int, alinhamento: str) -> Tensor:
    """Recorta `comprimento` amostras da janela, ancorado no alvo."""
    total = x.shape[1]
    if comprimento >= total:
        return x
    if alinhamento == "fim":              # alvo na borda direita (modo causal)
        return x[:, total - comprimento:]
    meio = total // 2                      # alvo no centro
    ini = meio - comprimento // 2
    return x[:, ini:ini + comprimento]


def imagens(x: Tensor, canais: list[str], tamanho: int, faixas: int = 8,
            escalas: list[int] | None = None, alinhamento: str = "centro") -> Tensor:
    """(B, V, L) -> (B, V*len(escalas)*len(canais), tamanho, tamanho).

    `escalas` sao comprimentos de janela em amostras, todos ancorados no mesmo
    alvo e todos comprimidos por PAA para `tamanho`. Combinar escalas resolve o
    conflito da janela longa: uma semana a 5 min sao 2016 amostras, e comprimir
    isso em 96 pixels da 105 min por pixel - o instante que se quer classificar
    fica enterrado dentro de um pixel. A escala curta devolve a resolucao fina,
    a longa traz o contexto semanal, e o ViT ve as duas como canais da mesma
    imagem.
    """
    b, v, comprimento_total = x.shape
    escalas = escalas or [comprimento_total]
    saidas = []
    for i in range(v):
      for escala in escalas:
        s = paa(_fatiar(x[:, i], escala, alinhamento), tamanho)
        precisa_ang = any(c in canais for c in ("gasf", "gadf"))
        if precisa_ang:
            r = _reescalar(s)
            raiz = (1 - r**2).clamp_min(0).sqrt()
        for c in canais:
            if c == "gasf":
                saidas.append(r.unsqueeze(2) * r.unsqueeze(1) - raiz.unsqueeze(2) * raiz.unsqueeze(1))
            elif c == "gadf":
                saidas.append(raiz.unsqueeze(2) * r.unsqueeze(1) - r.unsqueeze(2) * raiz.unsqueeze(1))
            elif c == "mtf":
                saidas.append(_mtf(s, faixas))
            elif c == "rp":
                saidas.append(-(s.unsqueeze(2) - s.unsqueeze(1)).abs())
            elif c == "amp":
                saidas.append(s.unsqueeze(2) * s.unsqueeze(1))
            else:
                raise ValueError(f"canal desconhecido: {c}")
    return torch.stack(saidas, dim=1)


class ViT(nn.Module):
    """Vision Transformer pequeno; entra a imagem, sai UM logito."""

    def __init__(self, n_canais: int, tamanho: int = 96, patch: int = 8, d: int = 192,
                 camadas: int = 4, cabecas: int = 4, dropout: float = 0.1):
        super().__init__()
        if tamanho % patch:
            raise ValueError("tamanho deve ser multiplo do patch")
        n = (tamanho // patch) ** 2
        self.recorte = nn.Conv2d(n_canais, d, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, d))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        camada = nn.TransformerEncoderLayer(
            d_model=d, nhead=cabecas, dim_feedforward=4 * d, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.corpo = nn.TransformerEncoder(camada, num_layers=camadas)
        self.norma = nn.LayerNorm(d)
        self.cabeca = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 1))

    def representar(self, img: Tensor) -> Tensor:        # (B, C, H, W) -> (B, d)
        t = self.recorte(img).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], dim=1) + self.pos
        return self.norma(self.corpo(t))[:, 0]

    def forward(self, img: Tensor) -> Tensor:            # (B, C, H, W) -> (B, 1)
        return self.cabeca(self.representar(img))


class ClassificadorImagem(nn.Module):
    """Junta as duas metades: serie -> imagem -> ViT -> logito."""

    def __init__(self, n_vistas: int, canais: list[str], tamanho: int = 96, patch: int = 8,
                 d: int = 192, camadas: int = 4, dropout: float = 0.1, faixas: int = 8,
                 escalas: list[int] | None = None, alinhamento: str = "centro"):
        super().__init__()
        self.canais, self.tamanho, self.faixas = list(canais), tamanho, faixas
        self.escalas, self.alinhamento = escalas, alinhamento
        n_canais = n_vistas * len(canais) * (len(escalas) if escalas else 1)
        self.vit = ViT(n_canais, tamanho, patch, d, camadas, dropout=dropout)

    def _imagem(self, x: Tensor) -> Tensor:
        return imagens(x, self.canais, self.tamanho, self.faixas,
                       self.escalas, self.alinhamento)

    def representar(self, x: Tensor) -> Tensor:          # (B, V, L) -> (B, d)
        return self.vit.representar(self._imagem(x))

    def forward(self, x: Tensor) -> Tensor:              # (B, V, L) -> (B, 1)
        return self.vit(self._imagem(x))
