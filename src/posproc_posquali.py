"""Pos-quali: pos-processamento temporal das decisoes do CRAFT, so na validacao.

Hoje cada intervalo e decidido sozinho. Tres formas de usar a vizinhanca no tempo,
todas com parametros e limiar escolhidos SEM a casa avaliada:

  suav    media movel centrada (largura w) do logit do conjunto antes da regra d
  hmm     decodificacao de 2 estados (forward-backward): emissao = q/pi e (1-q)/(1-pi)
          com q calibrado (Platt), transicoes estimadas dos rotulos das casas de
          calibracao e multiplicadas por kappa; limiar F1 sobre a posterior
  filtro  sobre a decisao d: preenche desligados de ate g intervalos e apaga ligados
          de ate r intervalos

Proxy do S2: REFIT 3/7/20 em LOHO (parametros e limiar nas outras duas casas).
Proxy do S1: holdout da casa 8 (4 semanas), 1a metade calibra e escolhe, 2a avalia.
Sequencias quebram em qualquer lacuna (intervalos nao consecutivos).

Uso:
    python src/posproc_posquali.py
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decisao import ajustar_bcts, f1, limiar_oraculo, logit      # noqa: E402
from taxa import TAXA                                            # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
V2, V3 = RAIZ / "resultados" / "v2" / "pred", RAIZ / "resultados" / "v3" / "pred"
KS_MODELO = (5, 7, 9, 15, 25)
SEMENTES = range(5)
E = ("escada_A", "A2")
TAGS = {"television": "ta_tv", "washing machine": "ta_wm", "freezer": "ta_fz"}
LARGURAS = (1, 3, 5, 7, 9, 13, 19, 25)
KAPPAS = (1.0, 0.5, 0.25, 0.1)
GR = (0, 1, 2, 3, 4, 6, 9, 12)


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


# --------------------------------------------------------------------------------------
# leitura (so chaves de validacao)
# --------------------------------------------------------------------------------------


def ler(arq: Path) -> dict | None:
    if not arq.exists():
        return None
    d = np.load(arq)
    r = {}
    for k in d.files:
        a, c = k.rsplit("|", 1)
        if a.startswith("cv:") or a == "val":
            r.setdefault(a, {})[c] = d[k]
    for a in r:
        if "z" not in r[a]:
            r[a]["z"] = logit(r[a]["p"])
    return r


def conjunto(ap: str, s: int) -> dict | None:
    """{alvo: (y, z, t)}: media dos logits de G5..G25 e, nas casas cv, tambem do E
    (no 'val' o E do v2 tem outros instantes e fica de fora)."""
    slug = ap.replace(" ", "_")
    ms = [ler(V3 / TAGS[ap] / f"{slug}__G{k}__s{s}.npz") for k in KS_MODELO]
    if not all(ms):
        return None
    e = ler(V2 / E[0] / f"{slug}__{E[1]}__s{s}.npz")
    out = {}
    for a in ms[0]:
        zs = [m[a]["z"] for m in ms]
        if a.startswith("cv:") and e is not None and a in e and len(e[a]["z"]) == len(zs[0]):
            zs.append(e[a]["z"])
        out[a] = (ms[0][a]["y"].astype(int), np.mean(zs, 0), ms[0][a]["t"].astype(np.int64))
    return out


# --------------------------------------------------------------------------------------
# operacoes temporais
# --------------------------------------------------------------------------------------


def segmentos(t: np.ndarray) -> np.ndarray:
    """Id do trecho continuo (intervalos consecutivos) de cada amostra."""
    return np.concatenate([[0], np.cumsum(np.diff(t) != TAXA)])


def suavizar(z: np.ndarray, seg: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return z
    out = np.empty_like(z)
    for s in np.unique(seg):
        i = np.flatnonzero(seg == s)
        v = z[i]
        k = min(w, len(v))
        c = np.convolve(np.ones(len(v)), np.ones(k), "same")
        out[i] = np.convolve(v, np.ones(k), "same") / c
    return out


def transicoes(y: np.ndarray, seg: np.ndarray) -> tuple[float, float]:
    """P(desl -> lig) e P(lig -> desl) por intervalo, dentro dos trechos."""
    mesmo = seg[1:] == seg[:-1]
    a, b = y[:-1][mesmo], y[1:][mesmo]
    p01 = ((a == 0) & (b == 1)).sum() / max((a == 0).sum(), 1)
    p10 = ((a == 1) & (b == 0)).sum() / max((a == 1).sum(), 1)
    return float(max(p01, 1e-4)), float(max(p10, 1e-4))


def posterior_hmm(q: np.ndarray, seg: np.ndarray, pi: float, p01: float, p10: float
                  ) -> np.ndarray:
    """Forward-backward de 2 estados com verossimilhanca escalada q/pi, (1-q)/(1-pi).
    Em cada inicio de trecho o estado volta a priori pi."""
    eps = 1e-6
    q = np.clip(q, eps, 1 - eps)
    L = np.stack([(1 - q) / (1 - pi), q / pi], 1)                     # (N, 2)
    A = np.array([[1 - p01, p01], [p10, 1 - p10]])
    prior = np.array([1 - pi, pi])
    n = len(q)
    inicio = np.r_[True, seg[1:] != seg[:-1]]
    fim = np.r_[seg[1:] != seg[:-1], True]
    al = np.empty((n, 2))
    a = prior * L[0]
    al[0] = a / a.sum()
    for i in range(1, n):
        a = (prior if inicio[i] else al[i - 1] @ A) * L[i]
        al[i] = a / a.sum()
    be = np.empty((n, 2))
    be[-1] = 1.0
    for i in range(n - 2, -1, -1):
        b = np.ones(2) if fim[i] else A @ (L[i + 1] * be[i + 1])
        be[i] = b / b.sum()
    g = al * be
    return g[:, 1] / g.sum(1)


def filtrar(d: np.ndarray, seg: np.ndarray, g: int, r: int) -> np.ndarray:
    """Fecha buracos de desligado <= g e depois apaga ligados <= r, por trecho."""
    d = d.astype(bool).copy()
    for alvo, lim in ((False, g), (True, r)):
        if lim <= 0:
            continue
        for s in np.unique(seg):
            i = np.flatnonzero(seg == s)
            v = d[i]
            mud = np.flatnonzero(np.diff(np.r_[~v[0], v, ~v[-1]].astype(int)) != 0)
            for a, b in zip(mud[:-1], mud[1:]):
                interno = a > 0 and b < len(v)            # so corridas com vizinhos
                if v[a] == alvo and b - a <= lim and (interno or alvo):
                    v[a:b] = not alvo
            d[i] = v
    return d


# --------------------------------------------------------------------------------------
# uma divisao calibracao -> avaliacao
# --------------------------------------------------------------------------------------


def aplicar(cal: list[tuple], alvo: tuple) -> dict:
    """`cal`: lista de (y, z, t) de calibracao; `alvo`: (y, z, t) avaliado. Devolve o F1
    no alvo de cada forma, com parametros escolhidos so na calibracao."""
    yc = np.concatenate([c[0] for c in cal])
    segc = [segmentos(c[2]) for c in cal]
    y, z, t = alvo
    seg = segmentos(t)
    out = {}

    # base: Platt + limiar F1 na calibracao (regra d)
    def regra_d(zc_list, za):
        zc = np.concatenate(zc_list)
        cal_f, _, _ = ajustar_bcts(zc, yc)
        lim = limiar_oraculo(cal_f(zc), yc)[0]
        return cal_f, lim, cal_f(za) >= lim

    cal_f, lim, d = regra_d([c[1] for c in cal], z)
    out["base"] = (f1(y, d), float(d.mean()), "")
    out["oraculo"] = (limiar_oraculo(sig(z), y)[1], np.nan, "")

    # suavizacao: w escolhido pelo F1 (regra d) na propria calibracao
    melhor = (-1, None)
    for w in LARGURAS:
        zs = [suavizar(c[1], sc, w) for c, sc in zip(cal, segc)]
        cf, lm, _ = regra_d(zs, np.zeros(1))
        f = f1(yc, cf(np.concatenate(zs)) >= lm)
        melhor = max(melhor, (f, w), key=lambda v: v[0])
    w = melhor[1]
    _, _, d = regra_d([suavizar(c[1], sc, w) for c, sc in zip(cal, segc)], suavizar(z, seg, w))
    out["suav"] = (f1(y, d), float(d.mean()), f"w={w}")

    # hmm: transicoes dos rotulos de calibracao, kappa e limiar escolhidos na calibracao
    p01, p10 = transicoes(yc, np.concatenate([sc + 10_000 * i for i, sc in enumerate(segc)]))
    qc = [cal_f(c[1]) for c in cal]
    pi = float(yc.mean())
    melhor = (-1, None, None)
    for kap in KAPPAS:
        gc = np.concatenate([posterior_hmm(q, sc, pi, kap * p01, kap * p10)
                             for q, sc in zip(qc, segc)])
        lm, f = limiar_oraculo(gc, yc)
        melhor = max(melhor, (f, kap, lm), key=lambda v: v[0])
    _, kap, lm = melhor
    d = posterior_hmm(cal_f(z), seg, pi, kap * p01, kap * p10) >= lm
    out["hmm"] = (f1(y, d), float(d.mean()), f"kappa={kap}")

    # filtro de duracao minima sobre a decisao d, (g, r) escolhidos na calibracao
    dc = [cal_f(c[1]) >= lim for c in cal]
    melhor = (-1, None)
    for g, r in product(GR, GR):
        f = f1(yc, np.concatenate([filtrar(dd, sc, g, r) for dd, sc in zip(dc, segc)]))
        melhor = max(melhor, (f, (g, r)), key=lambda v: v[0])
    g, r = melhor[1]
    d = filtrar(cal_f(z) >= lim, seg, g, r)
    out["filtro"] = (f1(y, d), float(d.mean()), f"g={g},r={r}")
    return out


def main() -> None:
    linhas = []
    for ap in TAGS:
        for s in SEMENTES:
            c = conjunto(ap, s)
            if c is None:
                continue
            cvs = sorted(a for a in c if a.startswith("cv:"))
            divisoes = [("S2 proxy", h.split()[-1], [c[o] for o in cvs if o != h], c[h])
                        for h in cvs]
            if "val" in c:                               # S1 proxy: casa 8, metades
                y, z, t = c["val"]
                dia = (t - t.min()) // 86400
                meio = dia < (dia.max() + 1) // 2
                divisoes.append(("S1 proxy", "8", [(y[meio], z[meio], t[meio])],
                                 (y[~meio], z[~meio], t[~meio])))
            for cen, casa, cal, alvo in divisoes:
                for forma, (f, prev, par) in aplicar(cal, alvo).items():
                    linhas.append(dict(aparelho=ap, semente=s, cenario=cen, casa=casa,
                                       forma=forma, f1=f, prevista=prev, parametro=par,
                                       prevalencia=float(alvo[0].mean())))
            print(f"  {ap} s{s} ok", flush=True)
    d = pd.DataFrame(linhas)
    d.to_csv(RAIZ / "resultados" / "v3" / "posproc_posquali__linhas.csv", index=False)
    base = d[d.forma == "base"].set_index(["aparelho", "semente", "cenario", "casa"]).f1
    res = []
    for (ap, cen, forma), g in d.groupby(["aparelho", "cenario", "forma"], sort=False):
        dif = (g.set_index(["aparelho", "semente", "cenario", "casa"]).f1 - base).dropna()
        res.append(dict(aparelho=ap, cenario=cen, forma=forma, f1=g.f1.mean(),
                        delta=dif.mean(), positivos=f"{int((dif > 1e-9).sum())}/{len(dif)}",
                        negativos=int((dif < -1e-9).sum())))
    r = pd.DataFrame(res).round(3)
    r.to_csv(RAIZ / "resultados" / "v3" / "posproc_posquali__resumo.csv", index=False)
    pd.set_option("display.width", 200)
    print(r.to_string(index=False))
    print(d[d.forma.isin(["suav", "hmm", "filtro"])].groupby(["aparelho", "forma"]).parametro
          .agg(lambda v: v.value_counts().head(3).to_dict()).to_string())


if __name__ == "__main__":
    main()
