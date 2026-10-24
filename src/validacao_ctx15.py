"""15 min, so validacao: contexto do CRAFT de 4 h (16 amostras) x 12 h (48 amostras).

CRAFT da semente s = media dos logits de G5..G25. Nas casas REFIT 3/7/20: AUC
por casa e F1 da regra d_L em LOHO (Platt + limiar nas outras duas). Le SO as
chaves 'cv:' das predicoes; nenhuma chave de teste e aberta.

Criterio (fixado antes de rodar): maior AUC media sobre casas e aparelhos; se a
diferenca for < 0,01, fica o contexto de 4 h (mesma duracao do CRAFT a 5 min).

Uso:
    python src/validacao_ctx15.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decisao import Decisor, f1, logit                                 # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
V3 = RAIZ / "resultados" / "v3" / "pred"
KS = (5, 7, 9, 15, 25)
APS = ("television", "washing machine", "freezer")


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def cv_de(arq: Path) -> dict:
    d = np.load(arq)
    r = {}
    for k in d.files:
        a, c = k.rsplit("|", 1)
        if a.startswith("cv:"):
            r.setdefault(a, {})[c] = d[k]
    for a in r:
        r[a].setdefault("z", logit(r[a]["p"]))
    return r


def main() -> None:
    linhas = []
    for ctx in (16, 48):
        for ap in APS:
            slug = ap.replace(" ", "_")
            pasta = V3 / f"val15_c{ctx}_A_{slug}"
            for s in range(3):
                arqs = [pasta / f"{slug}__G{k}__s{s}.npz" for k in KS]
                if not all(a.exists() for a in arqs):
                    continue
                ms = [cv_de(a) for a in arqs]
                casas = sorted(ms[0])
                z = {c: np.mean([m[c]["z"] for m in ms], 0) for c in casas}
                y = {c: ms[0][c]["y"] for c in casas}
                for c in casas:
                    outros = [o for o in casas if o != c]
                    dec = Decisor(sig(np.concatenate([z[o] for o in outros])),
                                  np.concatenate([y[o] for o in outros])).decidir(sig(z[c]))
                    linhas.append(dict(ctx=ctx, aparelho=ap, semente=s, casa=c,
                                       auc=roc_auc_score(y[c], z[c]), f1_dL=f1(y[c], dec["d"])))
    d = pd.DataFrame(linhas)
    d.to_csv(RAIZ / "resultados" / "v3" / "validacao_ctx15__sementes.csv", index=False)
    r = d.groupby(["aparelho", "ctx"])[["auc", "f1_dL"]].mean().unstack("ctx").round(3)
    print(r.to_string())
    print("\nmedia geral:\n", d.groupby("ctx")[["auc", "f1_dL"]].mean().round(4).to_string())
    print("\nsementes completas:", d.groupby(["ctx", "aparelho"]).semente.nunique().to_dict())


if __name__ == "__main__":
    main()
