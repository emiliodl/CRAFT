"""Decisao por quantil do v3 (docs/v3.md §2.4, §3.6, §8).

Transfere o PONTO DE OPERACAO na ROC, nao o valor do limiar. A ROC nao depende
da razao entre as classes enquanto as distribuicoes de escore dentro de cada
classe se mantem (Fawcett, 2006); o valor do limiar depende tambem da
calibracao de cada casa, que e justamente o que muda entre casas.

  postos      cada membro do conjunto vira posto percentual sobre TODOS os
              intervalos da casa-alvo; o conjunto e a media dos postos.
              Invariante a qualquer transformacao monotona de cada membro.
  decisao     positivos = os round(r * N) intervalos de maior posto, com
              r = min(1, k * pi); r >= 0,95 -> tudo positivo (regime trivial).
  k           mediana, sobre as dobras do LOHO, de r* / pi, com r* a fracao que
              o limiar oraculo preve positiva na casa retida.
  pi          (a) mediana das prevalencias das casas fonte
              (b) Median Sweep (Forman, 2008) com TPR/FPR estimados no LOHO
              (c) media de (a) e (b)

Usa a distribuicao NAO ROTULADA do alvo (postos e quantil): transdutivo leve,
faixas A-t e A+-t. A variante indutiva (limiar da origem) e reportada ao lado.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from decisao import limiar_oraculo

TRIVIAL = 0.95


def posto_percentual(s: np.ndarray) -> np.ndarray:
    """Posto medio (empates recebem a media) dividido por N: valores em (0, 1]."""
    s = np.asarray(s, float)
    return rankdata(s, method="average") / max(len(s), 1)


def media_postos(escores: list[np.ndarray]) -> np.ndarray:
    return np.mean([posto_percentual(s) for s in escores], axis=0)


def decidir_quantil(posto: np.ndarray, r: float, semente: int = 0) -> np.ndarray:
    """Exatamente round(r * N) positivos (os de maior posto); empates sorteados com
    semente fixa. r >= 0,95 -> tudo positivo."""
    n = len(posto)
    if r >= TRIVIAL:
        return np.ones(n, bool)
    k = int(round(float(np.clip(r, 0.0, 1.0)) * n))
    desempate = np.random.default_rng(semente).random(n)
    ordem = np.lexsort((desempate, -np.asarray(posto, float)))
    pred = np.zeros(n, bool)
    pred[ordem[:k]] = True
    return pred


def fracao_oraculo(p: np.ndarray, y: np.ndarray) -> float:
    """r*: fracao que o limiar oraculo (maximo F1) preve positiva."""
    lim = limiar_oraculo(p, y)[0]
    return float((np.asarray(p) >= lim).mean())


def k_de_dobras(dobras: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """k = mediana de r*/pi sobre as dobras (escore, rotulo) de casas retidas."""
    razoes = [fracao_oraculo(p, y) / y.mean() for p, y in dobras if 0 < y.mean() < 1]
    return float(np.median(razoes)) if razoes else 1.0


def median_sweep(p_alvo: np.ndarray, p_cal: np.ndarray, y_cal: np.ndarray,
                 n_limiares: int = 99, dif_min: float = 0.25) -> float | None:
    """Median Sweep (Forman, 2008): contagem ajustada em muitos limiares,
    mantendo so os com TPR - FPR > dif_min, e a mediana das estimativas.
    `p_cal`, `y_cal`: escores e rotulos das casas retidas (LOHO), na mesma escala
    de `p_alvo`."""
    p_cal, y_cal = np.asarray(p_cal, float), np.asarray(y_cal) == 1
    p_alvo = np.asarray(p_alvo, float)
    grade = np.unique(np.quantile(p_cal, np.linspace(0.01, 0.99, n_limiares)))
    est = []
    for t in grade:
        tpr = (p_cal[y_cal] >= t).mean()
        fpr = (p_cal[~y_cal] >= t).mean()
        if tpr - fpr > dif_min:
            est.append(np.clip(((p_alvo >= t).mean() - fpr) / (tpr - fpr), 0.0, 1.0))
    return float(np.median(est)) if est else None


def estimar_pi(modo: str, prev_fontes: list[float], p_alvo: np.ndarray,
               p_cal: np.ndarray | None = None, y_cal: np.ndarray | None = None) -> float:
    """modo: 'fonte' (a), 'ms' (b), 'media' (c). Sem estimativa do MS, cai em (a)."""
    pa = float(np.median(prev_fontes))
    if modo == "fonte":
        return pa
    pb = median_sweep(p_alvo, p_cal, y_cal) if p_cal is not None else None
    if pb is None:
        return pa
    return pb if modo == "ms" else 0.5 * (pa + pb)


def decisao_quantil(escores: list[np.ndarray], k: float, pi: float,
                    semente: int = 0) -> tuple[np.ndarray, float]:
    """Conjunto de escores (um por membro) -> binarias e r usado."""
    r = min(1.0, k * pi)
    return decidir_quantil(media_postos(escores), r, semente), r
