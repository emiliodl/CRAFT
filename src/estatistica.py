"""Protocolo estatistico do v2 (docs/v2.md §5, §9.A5, A7, A8).

Duas fontes de variancia: treino (sementes) e amostragem do teste (quais dias
cairam no mes). A segunda domina com poucas ativacoes e NAO e capturada por
sementes. Daqui:

  - unidade de reamostragem = bloco de um dia comecando ao meio-dia (nao corta
    os ciclos noturnos da maquina de lavar);
  - em cada replica, reamostrar dias E sementes;
  - F1 das contagens agregadas dos dias sorteados, nunca media de F1 diarios;
  - pareado: o mesmo sorteio de dias para os dois metodos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from taxa import TAXA


def blocos_dia(t_s: np.ndarray, inicio_h: int = 12) -> np.ndarray:
    """Id do bloco diario (comecando as `inicio_h` horas) de cada instante."""
    return ((np.asarray(t_s, np.int64) - inicio_h * 3600) // 86400).astype(np.int64)


def _contagens(y, pred, grupos):
    """(vp, n_pred, n_pos) por dia, para agregar replicas sem recalcular tudo."""
    y = np.asarray(y) == 1
    pred = np.atleast_2d(pred).astype(bool)
    vp = np.stack([(pred[:, g] & y[g]).sum(1) for g in grupos], 1)      # (S, D)
    npred = np.stack([pred[:, g].sum(1) for g in grupos], 1)
    npos = np.array([y[g].sum() for g in grupos])                       # (D,)
    return vp, npred, npos


def _f1_replica(vp, npred, npos, dias, sementes):
    """Media sobre as sementes sorteadas do F1 das contagens agregadas dos dias."""
    v = vp[sementes][:, dias].sum(1)
    p = npred[sementes][:, dias].sum(1)
    n = npos[dias].sum()
    f = np.where(v > 0, 2 * v / np.maximum(p + n, 1), 0.0)
    return float(f.mean())


def bootstrap_f1(y, pred, dias, n: int = 2000, semente: int = 0,
                 pred_b=None) -> dict:
    """IC95% de F1(pred) e, se `pred_b` for dado, de F1(pred) - F1(pred_b), com o
    MESMO sorteio de dias. `pred`/`pred_b`: (amostras,) ou (sementes, amostras).
    Para 'vencer o trivial': pred_b = np.ones(len(y))."""
    rng = np.random.default_rng(semente)
    ud, inv = np.unique(dias, return_inverse=True)
    grupos = [np.flatnonzero(inv == g) for g in range(len(ud))]
    A = _contagens(y, pred, grupos)
    B = _contagens(y, pred_b, grupos) if pred_b is not None else None
    sa, sb = A[0].shape[0], (B[0].shape[0] if B else 1)
    fa, dif = np.empty(n), np.empty(n)
    for i in range(n):
        d = rng.integers(0, len(ud), len(ud))
        fa[i] = _f1_replica(*A, d, rng.integers(0, sa, sa))
        if B:
            dif[i] = fa[i] - _f1_replica(*B, d, rng.integers(0, sb, sb))
    out = dict(f1_ic=tuple(np.percentile(fa, [2.5, 97.5]).round(4)), meia_largura=
               round(float(np.diff(np.percentile(fa, [2.5, 97.5]))[0] / 2), 4), dias=len(ud))
    if B:
        out["dif_ic"] = tuple(np.percentile(dif, [2.5, 50, 97.5]).round(4))
        out["dp_dif"] = round(float(dif.std()), 4)
        out["minimo_detectavel"] = round(2.8 * float(dif.std()), 4)
    return out


def ativacoes(y: np.ndarray, t_s: np.ndarray, taxa_s: int = TAXA) -> int:
    """Numero de sessoes ligadas (corridas de 1 contiguas no tempo)."""
    y = np.asarray(y).astype(int)
    contig = np.r_[False, np.diff(t_s) == taxa_s]
    inicio = (y == 1) & ~(np.r_[0, y[:-1]].astype(bool) & contig)
    return int(inicio.sum())


def f1_binormal(auc: float, p: float) -> float:
    """F1 no limiar otimo sob modelo binormal de variancias iguais (ordem de grandeza)."""
    d = np.sqrt(2) * norm.ppf(auc)
    t = np.linspace(-4, 6, 20001)
    fpr, tpr = norm.sf(t), norm.sf(t - d)
    return float((2 * p * tpr / (p * tpr + p + (1 - p) * fpr)).max())


def holm(pvalores: dict[str, float]) -> pd.DataFrame:
    """Correcao de Holm dentro de uma familia de comparacoes."""
    itens = sorted(pvalores.items(), key=lambda kv: kv[1])
    m, acum, linhas = len(itens), 0.0, []
    for i, (k, p) in enumerate(itens):
        acum = max(acum, min(1.0, (m - i) * p))
        linhas.append(dict(comparacao=k, p=p, p_holm=acum))
    return pd.DataFrame(linhas)


def p_bootstrap(dif_amostras: np.ndarray) -> float:
    """p bilateral a partir das replicas de uma diferenca."""
    d = np.asarray(dif_amostras)
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
