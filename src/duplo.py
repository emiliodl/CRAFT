"""Dois ramos com forcas medidas e complementares, fundidos.

O que a medicao mostrou na maquina de lavar a 5 min (3 sementes):

                    intra F1   intra AUC   cross F1   cross AUC
  seq2point 1D        0,639      0,920       0,144      0,590
  ViT sobre imagem    0,498      0,939       0,209      0,745

O ramo 1D ve a amplitude absoluta e por isso vence dentro da casa de treino. O
ramo de imagem reescala cada janela pelo proprio min-max (GASF/GADF/MTF), o que
o torna invariante a escala do domicilio e faz a AUC no cruzado saltar de 0,59
para 0,75 - mas descarta justamente a amplitude.

A fusao concatena as duas representacoes antes da cabeca. O controle honesto
nao e "um ramo so": e o CONJUNTO, isto e, treinar os dois separadamente e mediar
as probabilidades. Se o modelo treinado em conjunto nao vencer o conjunto
simples, a arquitetura nao acrescenta nada alem do que a media ja dava.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from imagem import ClassificadorImagem
from ponto import ClassificadorPonto


class ClassificadorDuplo(nn.Module):
    def __init__(self, vistas: list[str], largura: int, canais: list[str],
                 tamanho: int = 96, escalas: list[int] | None = None,
                 alinhamento: str = "centro", d1: int = 96, d2: int = 192,
                 dropout: float = 0.1):
        super().__init__()
        self.ramo_serie = ClassificadorPonto(vistas, largura, d=d1, dropout=dropout)
        self.ramo_imagem = ClassificadorImagem(len(vistas), canais, tamanho=tamanho,
                                               d=d2, escalas=escalas,
                                               alinhamento=alinhamento, dropout=dropout)
        self.cabeca = nn.Sequential(
            nn.LayerNorm(d1 + d2), nn.Dropout(dropout), nn.Linear(d1 + d2, d2), nn.GELU(),
            nn.Linear(d2, 1))

    def forward(self, x: Tensor) -> Tensor:              # (B, V, L) -> (B, 1)
        r = torch.cat([self.ramo_serie.representar(x), self.ramo_imagem.representar(x)], dim=-1)
        return self.cabeca(r)
