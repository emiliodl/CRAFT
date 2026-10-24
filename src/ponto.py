"""Seq2point de verdade: uma janela de contexto, UM rotulo.

O que o projeto vinha fazendo era seq2seq - 288 amostras entram, 288 rotulos
saem - com janelas sobrepostas e media das probabilidades na inferencia. Isso
NAO e seq2point: o modelo seq2seq e treinado para prever todas as posicoes, e
portanto nao especializa a representacao para nenhuma. Mediar saidas depois nao
recupera isso.

Aqui cada exemplo tem um alvo so, sempre na mesma posicao relativa da janela.
E o que Seq2Point (Zhang et al., AAAI 2018) e SGN (Shin et al., AAAI 2019)
fazem - os dois metodos que vencem no artigo de referencia.

Dois formatos de contexto:

  centrado  janela [X - L/2, X + L/2), alvo no meio. E o que o Seq2Point
            original faz, e o que torna a comparacao com o artigo justa.
  causal    janela [X - (L-1-d), X + d], ou seja quase tudo passado, com `d`
            amostras de folga no futuro. E o formato usavel em tempo real.

A interface devolve rotulo e indice com forma (B, K), entao o laco de treino,
a avaliacao e a previsao do resto do projeto funcionam sem alteracao: para o
codigo generico, e um seq2seq de comprimento K.

Com `saida=1` (padrao) isso e o seq2point estrito. Com K > 1 o modelo recebe K
rotulos CENTRADOS no alvo como supervisao auxiliar - o meio-termo que o SGN usa
(32 rotulos para 200 amostras de entrada). A inferencia continua sendo de um
rotulo so: `SoCentro` corta a coluna central, entao a metrica e exatamente a
mesma de K=1 e a comparacao e justa. K nao e uma saida seq2seq - a janela de
saida e uma fracao pequena da de entrada, e nunca se rotula o dia inteiro de uma
vez.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from dados import Serie
from dominio import grupos_tempo
from modelo import CodificadorVista
from treino import DISPOSITIVO


def faixas_da_particao(s: Serie, n: int) -> np.ndarray:
    """Mascara das posicoes que pertencem a ESTA particao, lida de `s.centros`.

    `dividir` separa treino e validacao mexendo so em `centros` - as duas Series
    continuam compartilhando `vistas`, `rotulo` e `t`. Sem esta mascara,
    `alvos_validos` reconstruiria os alvos a partir de `t` e devolveria o MESMO
    conjunto para as duas, o que torna a validacao uma copia do treino (e foi
    exatamente o bug que existia aqui: 100% de interseccao).
    """
    m = np.zeros(n, dtype=bool)
    c = s.centros
    if len(c) == 0:
        return m
    passo = int(np.min(np.diff(c))) if len(c) > 1 else 1
    for g in np.split(c, np.flatnonzero(np.diff(c) > passo) + 1):
        m[g[0]:g[-1] + 1] = True
    return m


def alvos_validos(s: Serie, antes: int, depois: int, passo: int = 1) -> np.ndarray:
    """Indices X cujo contexto [X-antes, X+depois] cabe numa corrida contigua
    E que pertencem a particao desta Serie."""
    t = s.t
    taxa = int(np.median(np.diff(t))) if len(t) > 1 else 300
    quebras = np.flatnonzero(np.diff(t) != taxa) + 1
    limites = np.concatenate(([0], quebras, [len(t)]))
    alvos = []
    for a, b in zip(limites[:-1], limites[1:]):
        ini, fim = a + antes, b - depois
        if fim > ini:
            alvos.append(np.arange(ini, fim))
    if not alvos:
        return np.zeros(0, np.int64)
    a = np.concatenate(alvos)
    a = a[faixas_da_particao(s, len(t))[a]]      # so o que e desta particao
    return a[::passo].astype(np.int64)           # subamostra DEPOIS de filtrar


class FontePonto:
    """Mantem vistas e rotulo na GPU; cada lote e (contexto, 1 rotulo)."""

    def __init__(self, s: Serie, indices_vistas: list[int], antes: int, depois: int,
                 media: np.ndarray, desvio: np.ndarray, passo: int = 1, saida: int = 1,
                 esquema_grupo: str | None = None):
        v = s.vistas[indices_vistas]
        m, dv = media[indices_vistas, None], desvio[indices_vistas, None]
        self.vistas = torch.from_numpy((v - m) / dv).to(DISPOSITIVO)
        self.rotulo = torch.from_numpy(s.rotulo).to(DISPOSITIVO)
        # os K rotulos ficam centrados no alvo, entao o alvo precisa de margem
        # tambem para eles - senao o ultimo exemplo de cada corrida contigua
        # indexaria para fora dela.
        self.saida = saida
        self.desloc_saida = torch.arange(-(saida // 2), saida - saida // 2,
                                         device=DISPOSITIVO)
        margem_a, margem_d = max(antes, saida // 2), max(depois, saida - saida // 2 - 1)
        self.alvos = torch.from_numpy(
            alvos_validos(s, margem_a, margem_d, passo)).to(DISPOSITIVO)
        self.desloc = torch.arange(-antes, depois + 1, device=DISPOSITIVO)
        self.largura = antes + depois + 1
        self.nome = s.nome
        # subdominio de calendario por instante, para group-DRO; None desliga
        self.grupo = (torch.from_numpy(grupos_tempo(s.t, esquema_grupo)).to(DISPOSITIVO)
                      if esquema_grupo else None)

    def __len__(self) -> int:
        return len(self.alvos)

    def lote(self, pos: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        alvo = self.alvos[pos]                                  # (B,)
        idx = alvo[:, None] + self.desloc[None, :]              # (B, L)
        x = self.vistas[:, idx].permute(1, 0, 2)                # (B, V, L)
        saida = alvo[:, None] + self.desloc_saida[None, :]      # (B, K)
        return x, self.rotulo[saida], saida


class ClassificadorPonto(nn.Module):
    """Encoders por vista, tokens agregados por atencao, um logito."""

    def __init__(self, vistas: list[str], largura: int, d: int = 96, camadas: int = 3,
                 patch: int = 4, dropout: float = 0.1, saida: int = 1):
        super().__init__()
        self.vistas, self.largura, self.saida = list(vistas), largura, saida
        self.centro = saida // 2
        self.encoders = nn.ModuleList(
            [CodificadorVista(largura, d, camadas, patch=patch, dropout=dropout)
             for _ in vistas])
        # agregacao por atencao em vez de media: o alvo esta numa posicao fixa da
        # janela, entao faz diferenca o modelo poder pesar os tokens proximos
        # dela mais que os das bordas.
        self.pontuacao = nn.Linear(d, 1)
        self.cabeca = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, saida))

    def representar(self, x: Tensor) -> Tensor:                 # (B, V, L) -> (B, d)
        tokens = torch.stack([e(x[:, i]) for i, e in enumerate(self.encoders)], 0).sum(0)
        peso = torch.softmax(self.pontuacao(tokens), dim=1)     # (B, n, 1)
        return (tokens * peso).sum(dim=1)

    def forward(self, x: Tensor) -> Tensor:                     # (B, V, L) -> (B, K)
        return self.cabeca(self.representar(x))


class SoCentro(nn.Module):
    """Corta a coluna central de um modelo de K saidas, para avaliar como K=1.

    Treinar com K rotulos e avaliar com K rotulos sobrepostos seriam duas
    mudancas de uma vez, e a segunda ja se sabe que mexe na metrica (mediar
    previsoes vizinhas suaviza). Aqui so a supervisao muda; a previsao continua
    sendo a do instante alvo.
    """

    def __init__(self, modelo: nn.Module):
        super().__init__()
        self.modelo = modelo
        self.centro = getattr(modelo, "centro", 0)

    def forward(self, x: Tensor) -> Tensor:
        return self.modelo(x)[:, self.centro:self.centro + 1]
