"""Pos-quali: transfer learning para a casa de destino, so na validacao (nenhum teste).

Em cada casa de validacao h (REFIT 3, 7, 20), os primeiros k dias com dado sao os
dias de ADAPTACAO (rotulados, como se plugues fossem instalados por k dias) e os
dias depois do 14o sao os de AVALIACAO, os mesmos para todo k e toda forma.

Formas (docs/v3_resultados.md §17):
  dL          regra atual: Platt + limiar nas outras duas casas, nada de h
  recal_k     forma 2: Platt + limiar so nos k dias rotulados de h
  misto_k     forma 2: outras duas casas + k dias de h, cada parte com peso 1/2
  adabn       forma 1: estatisticas de BatchNorm reestimadas no agregado (sem rotulo)
              dos 14 primeiros dias de h; decisao dL (chaves 'cvbn:')
  ft_k        forma 3: ajuste fino com os k dias rotulados de h + recal_k
              (chaves 'cvft<k>:')
  oraculo     teto: melhor limiar nos proprios dias de avaliacao

Sem positivo ou sem negativo nos k dias, recal/misto/ft caem para dL (registrado).

Uso:
    python src/adaptacao_posquali.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decisao import Decisor, f1, limiar_oraculo, logit            # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
V2, V3 = RAIZ / "resultados" / "v2" / "pred", RAIZ / "resultados" / "v3" / "pred"
KS_MODELO = (5, 7, 9, 15, 25)
KS_DIAS = (1, 3, 7, 14)
N_ADAPT = 14
SEMENTES = range(5)
E = ("escada_A", "A2")
TAGS = {"television": "craft_A", "washing machine": "q_wm_base", "freezer": "q_fz_base"}
TAGS_ADAPT = {"television": "ta_tv", "washing machine": "ta_wm", "freezer": "ta_fz"}


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def ler(arq: Path, prefixos=("cv",)) -> dict | None:
    """So chaves de validacao (cv, cvbn, cvft<k>); nada de teste e lido."""
    if not arq.exists():
        return None
    d = np.load(arq)
    r = {}
    for k in d.files:
        a, c = k.rsplit("|", 1)
        if a.split(":")[0].startswith(prefixos):
            r.setdefault(a, {})[c] = d[k]
    for a in r:
        if "z" not in r[a]:
            r[a]["z"] = logit(r[a]["p"])
    return r


def conjunto(ap: str, s: int, tag: str, com_e: bool = True) -> dict | None:
    """{chave: (y, z medio, t)} do CRAFT (G5..G25 [+ E]) da semente s."""
    slug = ap.replace(" ", "_")
    ms = [ler(V3 / tag / f"{slug}__G{k}__s{s}.npz") for k in KS_MODELO]
    if not all(ms):
        return None
    e = ler(V2 / E[0] / f"{slug}__{E[1]}__s{s}.npz") if com_e else None
    out = {}
    for a in ms[0]:
        zs = [m[a]["z"] for m in ms]
        if e is not None:
            base = a.split(":", 1)[1]                  # o E nao e adaptado: chave cv
            ze = e.get(f"cv:{base}", {}).get("z")
            if ze is not None and len(ze) == len(zs[0]):
                zs.append(ze)
        out[a] = (ms[0][a]["y"], np.mean(zs, 0), ms[0][a]["t"])
    return out


def dias_de(t: np.ndarray) -> np.ndarray:
    """Indice do dia com dado (0, 1, 2, ...) de cada intervalo."""
    d = t // 86400
    _, inv = np.unique(d, return_inverse=True)
    return inv


def decidir(p_cal, y_cal, p_alvo, peso_alvo=None):
    """Platt + limiar F1 (regra d) num conjunto de calibracao; None se degenerado."""
    if y_cal.sum() == 0 or y_cal.sum() == len(y_cal):
        return None
    return Decisor(p_cal, y_cal).decidir(p_alvo)["d"]


def avaliar(c: dict, prefixo_alvo: str = "cv") -> list[dict]:
    """Uma linha por (casa, forma, k). `c` tem as chaves 'cv:' (base) e as adaptadas."""
    cvs = sorted(a for a in c if a.startswith("cv:"))
    linhas = []
    for h in cvs:
        casa = h.split()[-1]
        outros = [o for o in cvs if o != h]
        ys = np.concatenate([c[o][0] for o in outros])
        ps = sig(np.concatenate([c[o][1] for o in outros]))
        y, z, t = c[h]
        dia = dias_de(t)
        av = dia >= N_ADAPT
        ya = y[av]

        def lin(forma, k, pred, caiu=False, **extra):
            linhas.append(dict(casa=casa, forma=forma, k=k, f1=f1(ya, pred),
                               prev=float(ya.mean()), prevista=float(np.mean(pred)),
                               caiu_para_dL=caiu, **extra))

        dl = decidir(ps, ys, sig(z[av]))
        lin("dL", 0, dl)
        lin("oraculo", 0, sig(z[av]) >= limiar_oraculo(sig(z[av]), ya)[0])
        for k in KS_DIAS:
            ad = dia < k
            pa, yk = sig(z[ad]), y[ad]
            r = decidir(pa, yk, sig(z[av]))
            lin("recal", k, dl if r is None else r, r is None, pos_adapt=int(yk.sum()))
            # misto: replica os k dias para pesarem o mesmo que as duas casas fonte
            rep = max(1, int(round(len(ys) / max(len(yk), 1))))
            r = (decidir(np.concatenate([ps, np.tile(pa, rep)]),
                         np.concatenate([ys, np.tile(yk, rep)]), sig(z[av]))
                 if 0 < yk.sum() < len(yk) else None)
            lin("misto", k, dl if r is None else r, r is None, pos_adapt=int(yk.sum()))
        # formas 1 e 3, se as chaves existirem
        hb = f"cvbn:{h.split(':', 1)[1]}"
        if hb in c:
            yb, zb, tb = c[hb]
            assert np.array_equal(tb, t)
            # dL com as outras casas tambem adaptadas (cada casa com o seu BN)
            ob = [f"cvbn:{o.split(':', 1)[1]}" for o in outros]
            psb = sig(np.concatenate([c[o][1] for o in ob]))
            lin("adabn", 0, decidir(psb, ys, sig(zb[av])))
        for k in KS_DIAS:
            hf = f"cvft{k}:{h.split(':', 1)[1]}"
            if hf not in c:
                continue
            yf, zf, tf = c[hf]
            assert np.array_equal(tf, t)
            ad = dia < k
            r = decidir(sig(zf[ad]), y[ad], sig(zf[av]))
            lin("ft", k, dl if r is None else r, r is None, pos_adapt=int(y[ad].sum()))
    return linhas


def main() -> None:
    linhas = []
    for ap in TAGS:
        for s in SEMENTES:
            # a rodada com adaptacao tem as suas proprias predicoes base (chaves cv:):
            # todas as formas saem dos MESMOS modelos; sem ela, so a forma 2 (base)
            c = conjunto(ap, s, TAGS_ADAPT[ap])
            origem = "adapt"
            if c is None:
                c, origem = conjunto(ap, s, TAGS[ap]), "base"
            if c is None:
                continue
            for l in avaliar(c):
                linhas.append(dict(aparelho=ap, semente=s, origem=origem, **l))
    d = pd.DataFrame(linhas)
    d.to_csv(RAIZ / "resultados" / "v3" / "adaptacao_posquali__linhas.csv", index=False)
    r = (d.groupby(["aparelho", "forma", "k"], sort=False)
         .agg(f1=("f1", "mean"), prevista=("prevista", "mean"), prev=("prev", "mean"),
              caiu=("caiu_para_dL", "mean"), n=("f1", "size")).round(3))
    r.to_csv(RAIZ / "resultados" / "v3" / "adaptacao_posquali__resumo.csv")
    pd.set_option("display.width", 200)
    print(r.to_string())
    # por casa, para ver se o ganho e geral
    print(d.pivot_table(index=["aparelho", "forma", "k"], columns="casa", values="f1",
                        aggfunc="mean", sort=False).round(3).to_string())


if __name__ == "__main__":
    main()
