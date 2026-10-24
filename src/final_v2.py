"""P4 do v2 (docs/v2.md §3.7, §5.3): relatorio por celula de teste do protocolo.

Para cada aparelho e celula (intra = mes da casa 8; cross = ECO, casas empilhadas
como no artigo), com as predicoes de todas as sementes da rodada final:

  F1 [IC95%]      bootstrap por blocos de dia (meio-dia) x sementes, F1 das
                  contagens agregadas; media das sementes = comparavel ao artigo
  vs trivial      IC da diferenca pareada contra "tudo ligado"
  vs artigo       "supera" so se o limite inferior do IC passar o numero deles
  AUC, AP, oraculo (diagnostico), ativacoes, trava

Regras: d_A (faixa A estrita: calibrada no holdout da casa 8), d_L (calibrada nas
casas REFIT 3/7/20 - usa rotulos de casas extras), mais a, c e e.

Uso:
    python src/final_v2.py --tag final_A
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decisao import Decisor, auc, limiar_oraculo                # noqa: E402
from estatistica import ativacoes, blocos_dia, bootstrap_f1     # noqa: E402

SAIDA = Path(__file__).resolve().parents[1] / "resultados" / "v2"
ARTIGO = {("washing machine", "intra"): 0.63, ("washing machine", "cross"): 0.30,
          ("freezer", "intra"): 0.46, ("freezer", "cross"): 0.78,
          ("television", "intra"): 0.68, ("television", "cross"): 0.54}
REGRAS = [("A", "d"), ("L", "d"), ("A", "a"), ("A", "c"), ("A", "e"), ("L", "e")]


def f1b(y, pred) -> float:
    y, pred = np.asarray(y) == 1, np.asarray(pred, bool)
    vp = (y & pred).sum()
    return 0.0 if vp == 0 else 2 * vp / (pred.sum() + y.sum())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="final_A")
    p.add_argument("--n_boot", type=int, default=2000)
    args = p.parse_args()
    pasta = SAIDA / "pred" / args.tag
    linhas = []
    for arq_ap in sorted({a.name.split("__")[0] for a in pasta.glob("*.npz")}):
        ap = arq_ap.replace("_", " ")
        arqs = sorted(pasta.glob(f"{arq_ap}__*.npz"))
        modelo = arqs[0].name.split("__")[1]
        preds = []
        for a in arqs:
            d = np.load(a)
            P = {}
            for k in d.files:
                alvo, campo = k.rsplit("|", 1)
                P.setdefault(alvo, {})[campo] = d[k]
            preds.append(P)
        cvs = [k for k in preds[0] if k.startswith("cv:")]
        celulas = {"intra": [k for k in preds[0] if k.startswith("intra")],
                   "cross": [k for k in preds[0] if k.startswith("cross")]}
        for cel, alvos in celulas.items():
            if not alvos:
                continue
            y = np.concatenate([preds[0][a]["y"] for a in alvos])
            t = [preds[0][a]["t"] for a in alvos]
            # dias de casas diferentes sao blocos diferentes
            dias = np.concatenate([blocos_dia(tt) + 100000 * i for i, tt in enumerate(t)])
            n_at = sum(ativacoes(preds[0][a]["y"], preds[0][a]["t"]) for a in alvos)
            ps = [np.concatenate([P[a]["p"] for a in alvos]) for P in preds]
            prev = float(y.mean())
            base = dict(aparelho=ap, celula=cel, alvos="+".join(a.split(": ")[1] for a in alvos),
                        modelo=modelo, sementes=len(preds), n=len(y), dias=len(np.unique(dias)),
                        ativacoes=n_at, prevalencia=round(prev, 4),
                        trivial=round(2 * prev / (1 + prev), 4), artigo=ARTIGO[(ap, cel)],
                        auc=round(float(np.mean([auc(y, q) for q in ps])), 3),
                        ap=round(float(np.mean([average_precision_score(y, q) for q in ps])), 3),
                        oraculo=round(float(np.mean([limiar_oraculo(q, y)[1] for q in ps])), 3))
            for calib, regra in REGRAS:
                bins, travas = [], []
                for P, q in zip(preds, ps):
                    if calib == "A":
                        dec = Decisor(P["val"]["p"], P["val"]["y"], auc_retida=None)
                    else:
                        yo = np.concatenate([P[k]["y"] for k in cvs])
                        po = np.concatenate([P[k]["p"] for k in cvs])
                        dec = Decisor(po, yo, auc_retida=np.nanmean(
                            [auc(P[k]["y"], P[k]["p"]) for k in cvs]))
                    dd = dec.decidir(q)
                    bins.append(dd[regra])
                    travas.append(bool(dd["log"]["sem_deteccao"]))
                B = np.stack(bins)
                bt = bootstrap_f1(y, B, dias, n=args.n_boot, pred_b=np.ones(len(y)))
                f1m = float(np.mean([f1b(y, b) for b in B]))
                lin = dict(**base, regra=f"{regra}_{calib}", f1=round(f1m, 3),
                           f1_dp=round(float(np.std([f1b(y, b) for b in B])), 3),
                           ic95=bt["f1_ic"], vs_trivial_ic=bt["dif_ic"],
                           supera_artigo=bool(bt["f1_ic"][0] > ARTIGO[(ap, cel)]),
                           perde_artigo=bool(bt["f1_ic"][1] < ARTIGO[(ap, cel)]),
                           vence_trivial=bool(bt["dif_ic"][0] > 0),
                           trava=round(float(np.mean(travas)), 2))
                linhas.append(lin)
                print(f"  {ap:16s} {cel:5s} {lin['regra']:4s} F1={f1m:.3f} IC{bt['f1_ic']} "
                      f"artigo={ARTIGO[(ap, cel)]:.2f} trivial={base['trivial']:.3f} "
                      f"AUC={base['auc']:.3f} ora={base['oraculo']:.3f} trava={lin['trava']}",
                      flush=True)
    d = pd.DataFrame(linhas)
    d.to_csv(SAIDA / f"{args.tag}__celulas.csv", index=False)


if __name__ == "__main__":
    main()
