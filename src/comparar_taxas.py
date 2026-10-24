"""CRAFT a 5 min x 15 min nas celulas do artigo (descritivo, sem teste novo).

Le os dois fechamentos ja calculados:
    resultados/v3/final_craftE__celulas.csv     5 min  (preregistro_craftE.json)
    resultados/v3/final_craftE15__celulas.csv   15 min (preregistro_craftE15.json)

Uma linha por faixa x aparelho x celula, com a regra pre-registrada (d_L no
cross, d_A no intra) do modelo final ("CRAFT+E" nos CSVs = CRAFT no artigo).
O rotulo depende da taxa (limiar sobre a media reamostrada), entao a
prevalencia e o F1 trivial mudam; F1 - trivial e AUC comparam melhor as taxas.

Uso:
    python src/comparar_taxas.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
V3 = RAIZ / "resultados" / "v3"
REGRA = {"cross": "d_L", "intra": "d_A"}


def ler(arq: str, taxa: str) -> pd.DataFrame:
    d = pd.read_csv(V3 / arq)
    d = d[(d.modelo == "CRAFT+E") & (d.regra == d.celula.map(REGRA))]
    d = d[~((d.faixa == "A+") & (d.celula == "intra"))]
    d = d.assign(f1_menos_trivial=(d.f1 - d.trivial).round(3),
                 veredito=d.apply(lambda r: "supera" if r.supera_artigo else
                                  ("perde" if r.perde_artigo else "empate"), axis=1))
    cols = ["prevalencia", "trivial", "artigo", "auc", "f1", "ic95", "f1_menos_trivial",
            "veredito", "familia", "p_holm"]
    return d.set_index(["faixa", "aparelho", "celula"])[cols].add_suffix(f"_{taxa}")


def main() -> None:
    d = ler("final_craftE__celulas.csv", "5").join(ler("final_craftE15__celulas.csv", "15"))
    d = d.reset_index().sort_values(["faixa", "celula", "aparelho"], ascending=[False, True, True])
    d.to_csv(V3 / "comparacao_5x15__celulas.csv", index=False)
    cols = ["faixa", "aparelho", "celula"] + [f"{c}_{t}" for c in
            ("artigo", "f1", "veredito", "auc", "trivial", "f1_menos_trivial") for t in ("5", "15")]
    pd.set_option("display.width", 250)
    print(d[cols].to_string(index=False))


if __name__ == "__main__":
    main()
