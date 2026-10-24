"""Estagio 6 do v2 (docs/v2.md §3.6, §9.A3-A4): decisao otima para F1 sem rotulos do alvo.

Fundamentos (Lipton et al., 2014): com probabilidades calibradas o limiar otimo
para F1 e METADE do F1 otimo, e com classificador nao informativo o otimo e
prever tudo positivo. Daqui o procedimento:

  1. calibrar (BCTS, que no caso binario e Platt: sigmoide(z/T + b)) em dados
     rotulados que nao sao de treino;
  2. estimar a prevalencia do alvo por EM (Saerens et al., 2002), conferindo
     com a contagem ajustada (Forman, 2008);
  3. TRAVA de identificabilidade: se o classificador e pouco informativo a
     verossimilhanca da prevalencia e plana e o EM deriva para a fronteira -
     nesse caso prever tudo positivo e marcar "sem deteccao";
  4. ajustar as posteriores para a prevalencia estimada e escolher o limiar que
     maximiza o F1 ESPERADO nos escores do alvo.

Regras comparadas em E1.1:

  a   limiar otimo da origem (pratica atual)
  b   EM + limiar 0,5
  c   BCTS -> EM com trava -> limiar de F1 esperado (transdutiva)
  d   indutiva: limiar otimo na escala calibrada, medido na calibracao
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

EPS = 1e-7


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


# --------------------------------------------------------------------------------------
# A3 - calibracao, EM e trava
# --------------------------------------------------------------------------------------


def ajustar_bcts(z_cal: np.ndarray, y_cal: np.ndarray):
    """p = sigmoide(z/T + b) por verossimilhanca. Devolve (funcao, T, b)."""
    z, y = np.asarray(z_cal, float), np.asarray(y_cal, float)

    def nll(th):
        s = z / np.exp(th[0]) + th[1]
        return np.mean(np.logaddexp(0, s) - y * s)

    log_t, b = minimize(nll, x0=[0.0, 0.0], method="L-BFGS-B").x
    t = float(np.exp(log_t))
    return (lambda zz: 1 / (1 + np.exp(-(np.asarray(zz, float) / t + b)))), t, float(b)


def em_prevalencia(p: np.ndarray, pi_fonte: float, tol: float = 1e-7,
                   max_iter: int = 10_000) -> tuple[float, np.ndarray]:
    """EM de Saerens et al.: prevalencia do alvo e posteriores ajustadas."""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    pi_fonte = float(np.clip(pi_fonte, EPS, 1 - EPS))
    pi = pi_fonte
    for _ in range(max_iter):
        w1, w0 = pi / pi_fonte, (1 - pi) / (1 - pi_fonte)
        novo = float((w1 * p / (w1 * p + w0 * (1 - p))).mean())
        if abs(novo - pi) < tol:
            pi = novo
            break
        pi = novo
    w1, w0 = pi / pi_fonte, (1 - pi) / (1 - pi_fonte)
    return pi, w1 * p / (w1 * p + w0 * (1 - p))


def taxas_na_fonte(p_cal: np.ndarray, y_cal: np.ndarray, limiar: float | None = None
                   ) -> tuple[float, float, float]:
    """TPR, FPR e o limiar usados na contagem ajustada.

    Desvio do apendice A3 do v2, que fixava o limiar em 0,5: numa classe rara a
    probabilidade CALIBRADA quase nunca passa de 0,5, entao TPR ~ 0,06, a trava
    "TPR-FPR < 0,1" dispara com AUC 0,86 e a decisao vira "tudo positivo" - F1
    abaixo do trivial na maquina de lavar. Aqui o limiar e o que maximiza
    TPR - FPR na calibracao (metodo "Max" de Forman, 2008), que minimiza a
    variancia da contagem ajustada."""
    y = np.asarray(y_cal) == 1
    p = np.asarray(p_cal, float)
    if limiar is None:
        grade = np.unique(np.quantile(p, np.linspace(0.005, 0.995, 400)))
        j = [(p[y] >= t).mean() - (p[~y] >= t).mean() for t in grade]
        limiar = float(grade[int(np.argmax(j))])
    pr = p >= limiar
    tpr = float(pr[y].mean()) if y.any() else 0.0
    fpr = float(pr[~y].mean()) if (~y).any() else 0.0
    return tpr, fpr, limiar


def prevalencia_contagem_ajustada(p_alvo, limiar, tpr, fpr):
    den = tpr - fpr
    if den < 0.1:
        return None
    r = float(np.mean(np.asarray(p_alvo) >= limiar))
    return float(np.clip((r - fpr) / den, 0.0, 1.0))


def prevalencia_alvo(p_alvo, pi_fonte, auc_retida, tpr, fpr, limiar=0.5, auc_min=0.6,
                     margem=0.02, discord_max=0.5):
    """EM com trava. `auc_retida=None` desliga o criterio de AUC (faixa A, em que
    nao ha casa retida e a AUC do holdout superestima a do alvo)."""
    pi_em, q = em_prevalencia(p_alvo, pi_fonte)
    pi_ca = prevalencia_contagem_ajustada(p_alvo, limiar, tpr, fpr)
    motivos = []
    if auc_retida is not None and auc_retida < auc_min:
        motivos.append("AUC retida baixa")
    if pi_em < margem or pi_em > 1 - margem:
        motivos.append("EM na fronteira")
    if pi_ca is None:
        motivos.append("TPR-FPR pequeno")
    elif abs(pi_em - pi_ca) > discord_max * max(pi_em, pi_ca):
        motivos.append("EM e contagem discordam")
    info = dict(pi_em=pi_em, pi_ca=pi_ca)
    if motivos:
        return pi_fonte, np.clip(np.asarray(p_alvo, float), EPS, 1 - EPS), motivos, info
    return pi_em, q, [], info


# --------------------------------------------------------------------------------------
# A4 - limiar por F1 esperado, decisao e oraculo
# --------------------------------------------------------------------------------------


def limiar_f1_esperado(q: np.ndarray) -> tuple[float, float]:
    """Limiar que maximiza E[F1] ~ 2*sum(q_top-k) / (k + sum(q)). k = N da o trivial."""
    qs = np.sort(np.asarray(q, float))[::-1]
    k = np.arange(1, len(qs) + 1)
    ef1 = 2 * np.cumsum(qs) / (k + qs.sum())
    i = int(ef1.argmax())
    return float(qs[i]), float(ef1[i])


def limiar_oraculo(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Limiar empirico otimo e F1 correspondente (busca exata sobre todos os cortes).
    Usos: variante indutiva e F1 oraculo DIAGNOSTICO - nunca para selecionar."""
    o = np.argsort(-np.asarray(p, float), kind="stable")
    ps, ys = np.asarray(p, float)[o], np.asarray(y)[o]
    if ys.sum() == 0:
        return 1.0, 0.0
    tp = np.cumsum(ys)
    f1 = 2 * tp / (np.arange(1, len(ys) + 1) + ys.sum())
    # so cortes entre valores distintos sao realizaveis por um limiar
    fim_grupo = np.r_[ps[1:] != ps[:-1], True]
    f1 = np.where(fim_grupo, f1, -1)
    i = int(f1.argmax())
    return float(ps[i]), float(f1[i])


def f1(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = np.asarray(y) == 1, np.asarray(pred, bool)
    vp = float((y & pred).sum())
    return 0.0 if vp == 0 else 2 * vp / (float(pred.sum()) + float(y.sum()))


class Decisor:
    """Calibracao + regras de decisao, ajustado num conjunto rotulado que NAO e
    de treino (holdout temporal na faixa A; casas retidas na faixa A+).

    `auc_retida` e a AUC na(s) casa(s) retida(s); None na faixa A.
    """

    def __init__(self, p_cal: np.ndarray, y_cal: np.ndarray, auc_retida: float | None = None,
                 limiar_origem: float | None = None):
        self.cal, self.T, self.b = ajustar_bcts(logit(p_cal), y_cal)
        q = self.cal(logit(p_cal))
        self.pi_fonte = float(q.mean())
        self.tpr, self.fpr, self.limiar_ca = taxas_na_fonte(q, y_cal)
        self.auc_retida = auc_retida
        # (a): limiar otimo na escala ORIGINAL; (d): na escala calibrada
        self.limiar_a = (limiar_origem if limiar_origem is not None
                         else limiar_oraculo(p_cal, y_cal)[0])
        self.limiar_d = limiar_oraculo(q, y_cal)[0]

    def calibrar(self, p: np.ndarray) -> np.ndarray:
        return self.cal(logit(p))

    def decidir(self, p_alvo: np.ndarray) -> dict:
        """Aplica as quatro regras. Devolve predicoes binarias e o log da decisao."""
        q0 = self.calibrar(p_alvo)
        pi, q, motivos, info = prevalencia_alvo(q0, self.pi_fonte, self.auc_retida,
                                                self.tpr, self.fpr, limiar=self.limiar_ca)
        pi_em_bruto, q_em = em_prevalencia(q0, self.pi_fonte)
        if motivos:
            pred_c = np.ones(len(q), bool)
            lim_c, ef1 = np.nan, 2 * pi / (1 + pi)
        else:
            lim_c, ef1 = limiar_f1_esperado(q)
            pred_c = q >= lim_c
        # (e): variante declarada da (c). A trava do v2 trata qualquer disparo como
        # "classificador nao informativo" e preve tudo positivo. Quando o que falha
        # e so a estimativa da prevalencia (EM na fronteira ou discordando da
        # contagem ajustada) mas o classificador e informativo, o otimo de F1 nao e
        # "tudo positivo": cai-se para a regra indutiva (d).
        nao_informativo = any(m in ("AUC retida baixa", "TPR-FPR pequeno") for m in motivos)
        if not motivos:
            pred_e = pred_c
        elif nao_informativo:
            pred_e = np.ones(len(q), bool)
        else:
            pred_e = q0 >= self.limiar_d
        return dict(
            e=pred_e,
            a=np.asarray(p_alvo) >= self.limiar_a,
            b=q_em >= 0.5,
            c=pred_c,
            d=q0 >= self.limiar_d,
            log=dict(pi_em=round(info["pi_em"], 4),
                     pi_ca=None if info["pi_ca"] is None else round(info["pi_ca"], 4),
                     trava=";".join(motivos) or "", sem_deteccao=bool(motivos),
                     limiar_c=lim_c, f1_esperado=round(ef1, 4), T=round(self.T, 3),
                     b=round(self.b, 3), pi_fonte=round(self.pi_fonte, 4)))


def auc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
