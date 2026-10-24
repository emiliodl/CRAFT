"""F1 do v3 (docs/v3.md §6, E1.1, E1.2) e diagnostico D3, sobre as predicoes SALVAS do v2.

So as casas de validacao cross (REFIT 3/7/20) sao lidas - nunca o ECO nem o mes
de teste da casa 8 (D3: a olhada no teste do v2 ja foi gasta).

Para cada casa retida h e cada semente, tudo calibrado nas OUTRAS casas de
validacao (LOHO aninhado, como o `L` do v2):

  a     limiar da origem (oraculo nas outras casas, escala original)   = v2 (a)
  d     limiar oraculo na escala calibrada                           = v2 (d)
  qa    quantil, pi = mediana das prevalencias das casas fonte
  qb    quantil, pi = Median Sweep com TPR/FPR das outras casas
  qc    quantil, pi = media de qa e qb
  k     mediana de r*/pi nas outras casas (E1.2)

Metrica: F1 / oraculo por casa retida (media das sementes).
D3: fracao prevista positiva por (a) e (d) contra a do oraculo, por casa.

Portao G1: adotar a melhor regra q* se mediana(F1/oraculo) >= 0,95 E q* > (a)
em pelo menos 2/3 das casas.

Uso:
    python src/experimento_quantil.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dados_v2 import arquivo                                   # noqa: E402
from decisao import Decisor, f1, limiar_oraculo                # noqa: E402
from quantil import (decidir_quantil, estimar_pi, fracao_oraculo,  # noqa: E402
                     k_de_dobras)

RAIZ = Path(__file__).resolve().parents[1]
V2 = RAIZ / "resultados" / "v2"
SAIDA = RAIZ / "resultados" / "v3"
PROTO = RAIZ / "datasets" / "protocolo_artigo"

# modelos finais do v2 (pre-registros) + as referencias da escada, com as fontes
FONTES_AMAIS = {
    "television": [8, 1, 2, 5, 13, 15, 16, 19, 21],
    "washing machine": [8, 1, 2, 4, 5, 6, 9, 10, 11, 13, 15, 16, 17, 18, 19, 21],
    "freezer": [8, 1, 4, 6, 13, 17, 18],
}
CASOS = [
    # (aparelho, tag, modelo, faixa)
    ("television", "final_A", "A0_4hcf", "A"),
    ("television", "final_Amais", "A0_4h", "A+"),
    ("washing machine", "final_A", "A2", "A"),
    ("washing machine", "final_Amais", "A5", "A+"),
    ("television", "escada_Amais2", "A0_4hcf", "A+"),
    ("washing machine", "escada_Amais2", "A0_4h", "A+"),
    ("television", "escada_A", "A0cf", "A"),
    ("washing machine", "escada_A", "A0", "A"),
]


def prev_fonte(aparelho: str, casa: int) -> float:
    d = np.load(PROTO / arquivo("refit", casa, aparelho, "treino"))
    k = next(k for k in d.files if k.startswith("status_") and not k.endswith("nativo"))
    return float(d[k].mean())


def carregar(tag: str, aparelho: str, modelo: str) -> list[dict]:
    out = []
    for arq in sorted((V2 / "pred" / tag).glob(f"{aparelho.replace(' ', '_')}__{modelo}__s*.npz")):
        d = np.load(arq)
        cv = {}
        for k in d.files:
            alvo, campo = k.rsplit("|", 1)
            if alvo.startswith("cv:"):            # NUNCA le as chaves de teste
                cv.setdefault(alvo, {})[campo] = d[k]
        out.append(cv)
    return out


def avaliar(aparelho: str, tag: str, modelo: str, faixa: str) -> list[dict]:
    casas_f = [8] if faixa == "A" else FONTES_AMAIS[aparelho]
    prevs = [prev_fonte(aparelho, c) for c in casas_f]
    linhas = []
    for s, P in enumerate(carregar(tag, aparelho, modelo)):
        for h in P:
            y, p = P[h]["y"], P[h]["p"]
            outros = [o for o in P if o != h]
            yo = np.concatenate([P[o]["y"] for o in outros])
            po = np.concatenate([P[o]["p"] for o in outros])
            dec = Decisor(po, yo).decidir(p)
            k = k_de_dobras([(P[o]["p"], P[o]["y"]) for o in outros])
            ora = limiar_oraculo(p, y)[1]
            lin = dict(aparelho=aparelho, tag=tag, modelo=modelo, faixa=faixa, semente=s,
                       casa=h, prev=round(float(y.mean()), 4), oraculo=round(ora, 4),
                       r_oraculo=round(fracao_oraculo(p, y), 4), k=round(k, 3),
                       f1_a=f1(y, dec["a"]), f1_d=f1(y, dec["d"]),
                       r_a=float(dec["a"].mean()), r_d=float(dec["d"].mean()))
            for modo, nome in (("fonte", "qa"), ("ms", "qb"), ("media", "qc")):
                pi = estimar_pi(modo, prevs, p, po, yo)
                pred = decidir_quantil(p, min(1.0, k * pi), semente=s)
                lin[f"pi_{nome}"] = round(pi, 4)
                lin[f"f1_{nome}"] = f1(y, pred)
                lin[f"r_{nome}"] = float(pred.mean())
            linhas.append(lin)
    return linhas


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame([l for c in CASOS for l in avaliar(*c)])
    regras = ["a", "d", "qa", "qb", "qc"]
    for r in regras:
        d[f"razao_{r}"] = d[f"f1_{r}"] / d["oraculo"]
    d.to_csv(SAIDA / "f1_quantil__sementes.csv", index=False)
    g = d.groupby(["aparelho", "tag", "modelo", "faixa", "casa"])
    por_casa = g[[f"f1_{r}" for r in regras] + ["oraculo", "prev", "r_oraculo", "r_a", "r_d",
                                              "r_qa", "r_qb", "r_qc", "pi_qa", "pi_qb", "k"]
                 ].mean().round(4).reset_index()
    for r in regras:
        por_casa[f"razao_{r}"] = (por_casa[f"f1_{r}"] / por_casa["oraculo"]).round(3)
    por_casa.to_csv(SAIDA / "f1_quantil__casas.csv", index=False)
    print(por_casa.to_string())
    # portao G1 por caso e por regra
    res = []
    for (ap, tag, m, fx), gg in por_casa.groupby(["aparelho", "tag", "modelo", "faixa"]):
        for r in ["qa", "qb", "qc", "d"]:
            vence = int((gg[f"f1_{r}"] > gg["f1_a"]).sum())
            res.append(dict(aparelho=ap, tag=tag, modelo=m, faixa=fx, regra=r,
                            mediana_razao=round(float(gg[f"razao_{r}"].median()), 3),
                            mediana_razao_a=round(float(gg["razao_a"].median()), 3),
                            casas_vence_a=f"{vence}/{len(gg)}",
                            f1_medio=round(float(gg[f"f1_{r}"].mean()), 4),
                            f1_medio_a=round(float(gg["f1_a"].mean()), 4)))
    res = pd.DataFrame(res)
    res.to_csv(SAIDA / "f1_quantil__portao.csv", index=False)
    print("\n" + res.to_string())
    # D3
    d3 = por_casa[["aparelho", "tag", "modelo", "casa", "prev", "r_oraculo", "r_a", "r_d",
                   "razao_a", "razao_d"]]
    d3.to_csv(SAIDA / "d3_fracao_prevista.csv", index=False)


if __name__ == "__main__":
    main()
