"""SGN (Shin et al., AAAI 2019) em modo classificacao.

O SGN original tem duas sub-redes identicas sobre a mesma entrada: uma de
REGRESSAO, que estima a potencia do aparelho, e uma de CLASSIFICACAO, que estima
o estado on/off. A saida e o produto das duas, `y = p * sigmoid(s)`, e a perda e
o MSE contra a potencia real - a sub-rede de estado e supervisionada so
indiretamente, pela porta.

Aqui nao ha desagregacao: nao existe alvo de potencia. O que se porta e a
ESTRUTURA - duas sub-redes e uma porta multiplicativa - com a saida virando
logito e a perda virando BCE.

Um detalhe que nao e cosmetico. No SGN a porta fechada leva a saida a 0 W, que
significa "desligado". Em logito, 0 significa p = 0,5, que nao significa nada.
Por isso a saida e

    logito(t) = vies + evidencia(t) * porta(t)

com `vies` aprendido e inicializado em logit(prevalencia). Porta fechada devolve
o vies, ou seja, a taxa-base - que e o analogo correto de "desligado".

Modos, para separar o efeito da porta do efeito de ter o dobro de parametros:

  completo    duas sub-redes independentes, saida com porta   (o SGN)
  cabeca      uma sub-rede, duas cabecas, saida com porta     (porta barata)
  duplo_soma  duas sub-redes independentes, saidas somadas    (controle de
              capacidade: mesmos parametros do `completo`, sem porta)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from modelo import CodificadorVista


class SubRede(nn.Module):
    """Um encoder por vista, tokens somados, mais uma cabeca por instante."""

    def __init__(self, n_vistas: int, largura: int, d: int, camadas: int,
                 patch: int, dropout: float):
        super().__init__()
        self.encoders = nn.ModuleList(
            [CodificadorVista(largura, d, camadas, patch=patch, dropout=dropout)
             for _ in range(n_vistas)])
        self.cabeca = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, patch))
        self.largura = largura

    def forward(self, x: Tensor) -> Tensor:            # (B, V, L) -> (B, L)
        tokens = torch.stack([e(x[:, i]) for i, e in enumerate(self.encoders)], 0).sum(0)
        return self.cabeca(tokens).reshape(x.shape[0], self.largura)


class ClassificadorSGN(nn.Module):
    def __init__(self, vistas: list[str], largura: int, d: int = 96, camadas: int = 3,
                 patch: int = 4, dropout: float = 0.1, modo: str = "completo",
                 prevalencia: float = 0.05):
        super().__init__()
        if modo not in ("completo", "cabeca", "duplo_soma"):
            raise ValueError(f"modo desconhecido: {modo}")
        self.vistas, self.largura, self.modo = list(vistas), largura, modo
        v = len(vistas)

        def nova() -> SubRede:
            return SubRede(v, largura, d, camadas, patch, dropout)

        if modo == "cabeca":
            self.tronco = nn.ModuleList(
                [CodificadorVista(largura, d, camadas, patch=patch, dropout=dropout)
                 for _ in range(v)])
            self.cab_evidencia = nn.Sequential(
                nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, d), nn.GELU(),
                nn.Linear(d, patch))
            self.cab_porta = nn.Sequential(
                nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, d), nn.GELU(),
                nn.Linear(d, patch))
        else:
            self.evidencia = nova()
            self.porta = nova()

        p = min(max(prevalencia, 1e-4), 1 - 1e-4)
        self.vies = nn.Parameter(torch.tensor(math.log(p / (1 - p))))

    def _partes(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self.modo == "cabeca":
            tokens = torch.stack([e(x[:, i]) for i, e in enumerate(self.tronco)], 0).sum(0)
            b = x.shape[0]
            return (self.cab_evidencia(tokens).reshape(b, self.largura),
                    self.cab_porta(tokens).reshape(b, self.largura))
        return self.evidencia(x), self.porta(x)

    def forward(self, x: Tensor) -> Tensor:
        ev, pt = self._partes(x)
        if self.modo == "duplo_soma":
            return self.vies + ev + pt
        return self.vies + ev * torch.sigmoid(pt)

    @torch.no_grad()
    def abertura_da_porta(self, x: Tensor) -> float:
        """Media de sigmoid(porta) - diagnostico: a porta esta fazendo algo?"""
        _, pt = self._partes(x)
        return float(torch.sigmoid(pt).mean())
