"""Vistas do agregado, janelamento e divisao treino/validacao/teste.

Tres vistas, todas derivadas do agregado (a unica entrada disponivel na pratica):

  base     x - mediana movel de 2 h      tira a carga de base, mantem o patamar
  z        (x - mediana) / (1.4826*MAD)  adimensional, transfere entre casas
  arcsinh  arcsinh(x / 50)               comprime a faixa dinamica

Mais a vista `bruto` (o proprio x), usada nos controles da ablacao.

Regra que vale para tudo aqui: os dias mantidos em unificado_5min NAO sao
consecutivos. Toda vista e toda janela e construida DENTRO de uma corrida
contigua (diferenca de t igual a 300 s), nunca atravessando um buraco.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

RAIZ = Path(__file__).resolve().parents[1]
PASTA = RAIZ / "datasets" / "unificado_5min"

VISTAS = ("bruto", "base", "z", "arcsinh", "casa")
K_BASE = 24          # janelas de 5 min -> 2 h de mediana movel
ESCALA_ARCSINH = 50.0


def corridas(t: np.ndarray, taxa_s: int = 300) -> list[tuple[int, int]]:
    quebras = np.flatnonzero(np.diff(t) != taxa_s) + 1
    limites = np.concatenate(([0], quebras, [len(t)]))
    return list(zip(limites[:-1].tolist(), limites[1:].tolist()))


def calcular_vistas(x: np.ndarray, trechos: list[tuple[int, int]]) -> np.ndarray:
    """(5, N) float32 com bruto/base/z/arcsinh/casa, calculadas por corrida.

    `casa` e a normalizacao POR CASA: divide o agregado pelo p95 da propria casa.
    Ao contrario de `z` (que divide pelo MAD de uma janela movel de 2 h), a escala
    e uma constante do domicilio, entao a estrutura de amplitude dentro do dia
    sobrevive. Era `z` que derrubava o AUC de carga alta para 0,67: dividir por
    uma escala local apaga justamente a amplitude, que e o sinal inteiro de um
    lava-loucas de 2 kW.

    O p95 usa so o agregado da propria casa, que esta disponivel em operacao sem
    nenhum submedidor - entao calcula-lo tambem para a casa-alvo nao e vazamento.
    """
    x = x.astype(np.float64)
    v = np.zeros((len(VISTAS), len(x)), dtype=np.float32)
    v[0] = x
    v[3] = np.arcsinh(x / ESCALA_ARCSINH)
    v[4] = x / max(float(np.percentile(x, 95)), 1.0)
    for a, b in trechos:
        seg = x[a:b]
        k = min(K_BASE, max(3, len(seg)))
        med = median_filter(seg, size=k, mode="nearest")
        mad = median_filter(np.abs(seg - med), size=k, mode="nearest")
        v[1, a:b] = seg - med
        v[2, a:b] = (seg - med) / (1.4826 * np.maximum(mad, 1.0))
    return v


@dataclass
class Serie:
    """Uma casa, um aparelho: vistas, rotulo e os centros de janela validos."""
    vistas: np.ndarray      # (5, N) float32
    rotulo: np.ndarray      # (N,) float32, 1 = ligado
    centros: np.ndarray     # (M,) int64, indices com janela inteira dentro de uma corrida
    nome: str
    t: np.ndarray = None    # (N,) int64, segundos - so para achar os buracos no tempo

    @property
    def prevalencia(self) -> float:
        return float(self.rotulo[self.centros].mean())


def carregar(dataset: str, aparelho: str, casa: str, largura: int, passo: int = 1) -> Serie:
    arq = PASTA / f"{dataset}__{aparelho.replace(' ', '_')}__{casa.replace(' ', '_')}.npz"
    d = np.load(arq)
    ap = aparelho.replace(" ", "_")
    t, x, st = d["t"], d["main"], d[f"status_{ap}"].astype(np.float32)

    trechos = corridas(t)
    vistas = calcular_vistas(x, trechos)

    meia = largura // 2
    centros = []
    for a, b in trechos:
        if b - a < largura:
            continue
        centros.append(np.arange(a + meia, b - (largura - meia) + 1, passo))
    centros = np.concatenate(centros) if centros else np.zeros(0, np.int64)
    return Serie(vistas, st, centros.astype(np.int64), arq.stem, t.astype(np.int64))


def listar(aparelho: str, minimo_dias: float = 0.0) -> list[tuple[str, str, float]]:
    """(dataset, casa, dias) de todas as series daquele aparelho, maior primeiro."""
    import pandas as pd
    d = pd.read_csv(PASTA / "indice.csv")
    d = d[(d.aparelho == aparelho) & (d.dias >= minimo_dias)].sort_values("dias", ascending=False)
    return [(r.dataset, r.casa, float(r.dias)) for r in d.itertuples()]


def juntar(series: list[Serie], nome: str) -> Serie:
    """Empilha varias casas numa serie so, deslocando os centros.

    Nenhuma janela cruza a fronteira entre casas: os centros ja foram calculados
    dentro das corridas de cada casa antes do deslocamento.
    """
    vistas = np.concatenate([s.vistas for s in series], axis=1)
    rotulo = np.concatenate([s.rotulo for s in series])
    centros, tempos, desloc = [], [], 0
    for i, s in enumerate(series):
        centros.append(s.centros + desloc)
        # deslocamento gigante por casa: a fronteira entre duas casas nunca pode
        # parecer duas amostras consecutivas para quem procura buracos no tempo
        tempos.append(s.t + i * 10**10)
        desloc += s.vistas.shape[1]
    return Serie(vistas, rotulo, np.concatenate(centros), nome, np.concatenate(tempos))


def dividir_no_tempo(s: Serie, fracao_val: float = 0.2, largura: int = 48) -> tuple[Serie, Serie]:
    """Ultimos `fracao_val` de CADA corrida viram validacao.

    Divisao por corrida, e nao um corte unico no fim da serie, porque as corridas
    estao espalhadas por anos: um corte unico deixaria a validacao concentrada em
    uma estacao so. A largura da janela e descontada na fronteira para que nenhuma
    janela de treino toque um ponto de validacao.
    """
    c = s.centros
    passo = int(np.min(np.diff(c))) if len(c) > 1 else 1
    grupos = np.split(c, np.flatnonzero(np.diff(c) > passo) + 1)
    tr, va = [], []
    for g in grupos:
        corte = int(len(g) * (1 - fracao_val))
        folga = max(1, largura // passo)          # nenhuma janela de treino toca a validacao
        tr.append(g[: max(corte - folga, 0)])
        va.append(g[corte:])
    return (Serie(s.vistas, s.rotulo, np.concatenate(tr), s.nome + "[treino]", s.t),
            Serie(s.vistas, s.rotulo, np.concatenate(va), s.nome + "[val]", s.t))


def normalizador(s: Serie, largura: int) -> tuple[np.ndarray, np.ndarray]:
    """Media e desvio de cada vista, medidos SO nos pontos de treino."""
    meia = largura // 2
    idx = np.unique(np.clip(
        s.centros[:, None] + np.arange(-meia, largura - meia)[None, :], 0, s.vistas.shape[1] - 1))
    sub = s.vistas[:, idx]
    return sub.mean(axis=1).astype(np.float32), np.maximum(sub.std(axis=1), 1e-6).astype(np.float32)
