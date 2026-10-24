"""Melhoria 1 pos-v3: regra de decisao calibrada em varias casas (docs/v3_resultados.md §8).

No F4 do v3 a regra d_A (limiar calibrado no holdout temporal da casa 8) nao
transferiu: TV cross faixa A com AUC 0,95 e F1 0,42 (preve 8% para 21%). Aqui
as regras sao comparadas SO no LOHO das casas de validacao REFIT 3/7/20, com as
predicoes da escada do v3 (conjunto R5..R25 = B5, 5 sementes):

  d_A   Platt + limiar oraculo no holdout temporal das fontes (a do F4, faixa A)
  d_L   Platt + limiar oraculo nas OUTRAS casas de validacao, empilhadas
  d_M   Platt nas outras casas; limiar = MEDIANA dos limiares otimos de cada casa
        (uma casa grande ou atipica nao domina)
  q     media de postos + quantil (k das outras casas, pi = mediana das fontes)

Metrica: F1 / oraculo por casa retida. Portao (o mesmo do G1): adota-se a regra
de maior mediana, se superar d_A em pelo menos 2/3 das casas nas duas faixas.

Uso:
    python src/experimento_decisao_v3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decisao import Decisor, f1, limiar_oraculo                    # noqa: E402
from experimento_quantil import FONTES_AMAIS, prev_fonte           # noqa: E402
from quantil import decidir_quantil, k_de_dobras, posto_percentual  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
V3 = RAIZ / "resultados" / "v3"
KS = (5, 7, 9, 15, 25)


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


class DecisorMediana:
    """Platt nas casas de calibracao empilhadas; limiar = mediana dos limiares
    otimos (F1) de cada casa, na escala calibrada."""

    def __init__(self, casas: list[tuple[np.ndarray, np.ndarray]]):
        yo = np.concatenate([y for _, y in casas])
        zo = np.concatenate([z for z, _ in casas])
        self.base = Decisor(sig(zo), yo)
        lims = [limiar_oraculo(self.base.calibrar(sig(z)), y)[0] for z, y in casas
                if 0 < y.sum() < len(y)]
        self.limiar = float(np.median(lims))

    def decidir(self, z: np.ndarray) -> np.ndarray:
        return self.base.calibrar(sig(z)) >= self.limiar


def carregar(tag: str, aparelho: str, prefixo: str = "R") -> dict[int, list[dict]]:
    pasta = V3 / "pred" / tag
    por_sem: dict[int, list[dict]] = {}
    for k in KS:
        for arq in sorted(pasta.glob(f"{aparelho.replace(' ', '_')}__{prefixo}{k}__s*.npz")):
            s = int(arq.stem.rsplit("__s", 1)[1])
            d = np.load(arq)
            r = {}
            for kk in d.files:
                alvo, campo = kk.rsplit("|", 1)
                r.setdefault(alvo, {})[campo] = d[kk]
            por_sem.setdefault(s, []).append(r)
    return {s: v for s, v in por_sem.items() if len(v) == len(KS)}


def conj(ms, alvo):
    y = ms[0][alvo]["y"]
    return (y, np.mean([m[alvo]["z"] for m in ms], 0),
            np.mean([posto_percentual(m[alvo]["z"]) for m in ms], 0))


def avaliar(tag: str, aparelho: str, faixa: str, prefixo: str = "R") -> list[dict]:
    casas_f = [8] if faixa == "A" else FONTES_AMAIS[aparelho]
    pi = float(np.median([prev_fonte(aparelho, c) for c in casas_f]))
    linhas = []
    for s, ms in carregar(tag, aparelho, prefixo).items():
        cvs = sorted(k for k in ms[0] if k.startswith("cv:"))
        C = {k: conj(ms, k) for k in cvs}
        yv, zv, _ = conj(ms, "val")
        for h in cvs:
            y, z, posto = C[h]
            outros = [o for o in cvs if o != h]
            yo = np.concatenate([C[o][0] for o in outros])
            zo = np.concatenate([C[o][1] for o in outros])
            k = k_de_dobras([(C[o][2], C[o][0]) for o in outros])
            pred = dict(
                d_A=Decisor(sig(zv), yv).decidir(sig(z))["d"],
                d_L=Decisor(sig(zo), yo).decidir(sig(z))["d"],
                d_M=DecisorMediana([(C[o][1], C[o][0]) for o in outros]).decidir(z),
                q=decidir_quantil(posto, min(1.0, k * pi), semente=s))
            ora = limiar_oraculo(z, y)[1]
            lin = dict(aparelho=aparelho, faixa=faixa, tag=tag, semente=s, casa=h,
                       oraculo=ora, prev=float(y.mean()))
            for r, p in pred.items():
                lin[f"f1_{r}"] = f1(y, p)
                lin[f"razao_{r}"] = f1(y, p) / ora
                lin[f"frac_{r}"] = float(p.mean())
            linhas.append(lin)
    return linhas


def main() -> None:
    casos = [("escada_A_tv", "television", "A"), ("escada_A_wm", "washing machine", "A"),
             ("escada_Amais_tv", "television", "A+"), ("escada_Amais_wm", "washing machine", "A+")]
    d = pd.DataFrame([l for c in casos for l in avaliar(*c)])
    d.to_csv(V3 / "decisao_v3__sementes.csv", index=False)
    regras = ["d_A", "d_L", "d_M", "q"]
    pc = d.groupby(["aparelho", "faixa", "casa"])[[f"razao_{r}" for r in regras]
                                                  + [f"f1_{r}" for r in regras]
                                                  + ["oraculo", "prev"]
                                                  + [f"frac_{r}" for r in regras]].mean()
    print(pc.round(3).to_string())
    res = []
    for (ap, fx), g in pc.groupby(level=[0, 1]):
        for r in regras:
            res.append(dict(aparelho=ap, faixa=fx, regra=r,
                            mediana_razao=round(float(g[f"razao_{r}"].median()), 3),
                            f1_medio=round(float(g[f"f1_{r}"].mean()), 3),
                            vence_d_A=f"{int((g[f'f1_{r}'] > g['f1_d_A']).sum())}/{len(g)}"))
    res = pd.DataFrame(res)
    res.to_csv(V3 / "decisao_v3__portao.csv", index=False)
    print("\n" + res.to_string(index=False))


if __name__ == "__main__":
    main()
