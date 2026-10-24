"""Closure of the UK-DALE scenarios S4 (train house 1 -> test house 1, later month) and
S5 (train house 1 -> test house 2), pre-registered in resultados/v3/preregistro_ukdale_s.json.

CRAFT (internal name CRAFT+E) of seed s = mean logit of G5..G25 (resultados/v3/pred/
ukdale_s_<ap>) and of the A2 evidence branch (resultados/v2/pred/ukdale_s_ev). Rules:
d_A in S4 (Platt + F1-optimal threshold on the last four weeks of the house 1 training
period), d_L in S5 (calibrated on REFIT houses 3, 7 and 20). There is no published
number for these scenarios, so the report gives F1 of the "on" class with its 95%
day-block bootstrap interval, precision, recall and prevalence. The ResNets-only
ensemble is reported alongside.

S6 (train UK-DALE house 1 -> test REDD house 1, pre-registered in
resultados/v3/preregistro_ukdale_redd.json) uses the same analysis with rule d_L, from the
runs tagged ukdale_r_<ap> / ukdale_r_ev.

Usage:
    python src/final_ukdale.py                  # S4 and S5
    python src/final_ukdale.py --cenario s6     # S6
    python src/final_ukdale.py --cenario casa5  # second cross-house test (house 5)
    NILM_TAXA=900 python src/final_ukdale.py --cenario 15   # house 1 and house 5 at 15 min
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decisao import Decisor, f1                                     # noqa: E402
from estatistica import ativacoes, blocos_dia                       # noqa: E402
from final_craftE import conjunto, membros_da_semente, replicas, sig  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
APARELHOS = ("fridge", "dishwasher", "washing machine")
CELULAS = (("S4 intra (house 1)", "intra: ukdale casa 1", "d_A"),
           ("S5 cross-house (house 2)", "cross: ukdale casa 2", "d_L"))
CENARIOS = {  # tag prefix, cells, output name
    "s45": ("ukdale_s", CELULAS, "final_ukdale_s__celulas.csv"),
    "s6": ("ukdale_r", (("S6 cross-dataset (REDD house 1)", "cross: redd casa 1", "d_L"),),
           "final_ukdale_r__celulas.csv"),
    "casa5": ("ukdale_c5", (("cross-house (house 5)", "cross: ukdale casa 5", "d_L"),),
              "final_ukdale_c5__celulas.csv"),
    # 15 min (run with NILM_TAXA=900): intra-house and cross-house to house 5
    "15": ("ukdale_15", (("intra-house (house 1)", "intra: ukdale casa 1", "d_A"),
                         ("cross-house (house 5)", "cross: ukdale casa 5", "d_L")),
           "final_ukdale_15__celulas.csv"),
}


def celula(ap: str, alvo: str, regra: str, com_e: bool, n_boot: int = 2000,
           pref: str = "ukdale_s") -> dict | None:
    slug = ap.replace(" ", "_")
    por_sem = {}
    for s in range(10):
        ms = membros_da_semente(ap, [f"{pref}_{slug}"], f"{pref}_ev" if com_e else None, s)
        if ms is not None and alvo in ms[0]:
            por_sem[s] = ms
    if not por_sem:
        return None
    preds, y, t = [], None, None
    for s, ms in por_sem.items():
        yy, tt, _, z = conjunto(ms, alvo)
        if y is None:
            y, t = yy, tt
        assert len(yy) == len(y) and (yy == y).all()
        if regra == "d_A":
            yv, _, _, zv = conjunto(ms, "val")
            dec = Decisor(sig(zv), yv)
        else:
            cc = [conjunto(ms, k) for k in sorted(k for k in ms[0] if k.startswith("cv:"))]
            dec = Decisor(sig(np.concatenate([c[3] for c in cc])),
                          np.concatenate([c[0] for c in cc]))
        preds.append(dec.decidir(sig(z))["d"])
    B = np.stack(preds)
    dias = blocos_dia(t)
    fa, dif = replicas(y, B, dias, n_boot)        # dif = F1 - F1(always on), same replicas
    tp = [(b & (y == 1)).sum() for b in B]
    prec = np.mean([tp_ / max(b.sum(), 1) for tp_, b in zip(tp, B)])
    rec = np.mean([tp_ / max(y.sum(), 1) for tp_ in tp])
    return dict(sementes=len(por_sem), dias=len(np.unique(dias)), n=len(y),
                ativacoes=ativacoes(y, t), prevalencia=round(float(y.mean()), 3),
                f1=round(float(np.mean([f1(y, b) for b in B])), 3),
                ic95=tuple(float(v) for v in np.percentile(fa, [2.5, 97.5]).round(3)),
                allon=round(float(2 * y.mean() / (1 + y.mean())), 3),
                vs_allon_ic=tuple(float(v) for v in np.percentile(dif, [2.5, 97.5]).round(3)),
                precisao=round(float(prec), 3), revocacao=round(float(rec), 3),
                fracao_prevista=round(float(B.mean()), 3))


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--cenario", default="s45", choices=sorted(CENARIOS))
    pref, celulas, saida = CENARIOS[a.parse_args().cenario]
    linhas = []
    for ap in APARELHOS:
        for nome, alvo, regra in celulas:
            for modelo, com_e in (("CRAFT", True), ("ResNets only", False)):
                r = celula(ap, alvo, regra, com_e, pref=pref)
                if r is None:
                    print("  missing:", ap, nome, modelo)
                    continue
                linhas.append(dict(aparelho=ap, cenario=nome, regra=regra, modelo=modelo, **r))
                print(f"  {modelo:12s} {ap:16s} {nome:26s} {regra} F1={r['f1']:.3f} "
                      f"IC{r['ic95']} P={r['precisao']:.3f} R={r['revocacao']:.3f} "
                      f"prev={r['prevalencia']:.3f} ativ={r['ativacoes']}", flush=True)
    pd.DataFrame(linhas).to_csv(RAIZ / "resultados" / "v3" / saida, index=False)


if __name__ == "__main__":
    main()
