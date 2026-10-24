"""E2.3 do v3 (docs/v3.md §6): selecao gulosa de casas fonte para a faixa A+.

Comeca pela casa 8; a cada passo, treina o modelo de referencia com cada casa
candidata acrescentada e fica com a que mais sobe a AUC media nas casas de
validacao cross (REFIT 3/7/20). Para quando o ganho fica abaixo de `--ganho_min`
(0,005, [extrapolacao: criterio] no v3) ou acabam as candidatas.

Custo: um modelo so (ResNet k = 9 com a receita do ReCo, `--modelo R9`), poucas
sementes e epocas curtas - a selecao compara casas, nao modelos.

Uso:
    python src/selecao_casas_v3.py --aparelho television
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experimento_reco as er                                    # noqa: E402

SAIDA = er.SAIDA


def avaliar(aparelho: str, extras: list[int], base_args: dict, tag: str) -> float:
    args = argparse.Namespace(**{**base_args, "casas_extra": ",".join(map(str, extras)),
                                 "fonte": "A+" if extras else "A", "tag": tag})
    linhas = er.rodar(aparelho, args)
    d = pd.DataFrame(linhas)
    return float(d.groupby("semente")["auc"].mean().mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aparelho", default="television")
    p.add_argument("--modelo", default="R9")
    p.add_argument("--sementes", type=int, default=2)
    p.add_argument("--epocas", type=int, default=15)
    p.add_argument("--lotes_epoca", type=int, default=250)
    p.add_argument("--ganho_min", type=float, default=0.005)
    p.add_argument("--max_passos", type=int, default=8)
    args = p.parse_args()
    base = dict(aparelhos=args.aparelho, modelos=args.modelo, casas_val="3,7,20",
                sementes=args.sementes, semente0=0, epocas=args.epocas, paciencia=4, passo=8,
                lote=256, lotes_epoca=args.lotes_epoca, taxa=2e-3, ema=0.99, ctx=er.W5,
                frac_real=None, escala=None, perda="", intra=False, olhar_teste=False,
                verboso=False)
    slug = args.aparelho.replace(" ", "_")
    tag = f"e23_{slug}"
    atuais: list[int] = []
    cand = list(er.POOL[args.aparelho])
    hist = []
    t0 = time.time()
    auc_atual = avaliar(args.aparelho, atuais, base, tag)
    hist.append(dict(passo=0, casas="8", candidata=None, auc=auc_atual, escolhida=True))
    print(f"\n[E2.3] {args.aparelho}: so a casa 8 -> AUC {auc_atual:.4f}", flush=True)
    for passo in range(1, args.max_passos + 1):
        if not cand:
            break
        res = {}
        for c in cand:
            res[c] = avaliar(args.aparelho, atuais + [c], base, tag)
            hist.append(dict(passo=passo, casas=",".join(map(str, [8] + atuais + [c])),
                             candidata=c, auc=res[c], escolhida=False))
            print(f"[E2.3] passo {passo}: + casa {c:2d} -> AUC {res[c]:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            pd.DataFrame(hist).to_csv(SAIDA / f"e23_selecao__{slug}.csv", index=False)
        melhor = max(res, key=res.get)
        ganho = res[melhor] - auc_atual
        print(f"[E2.3] passo {passo}: melhor + casa {melhor} (AUC {res[melhor]:.4f}, "
              f"ganho {ganho:+.4f})", flush=True)
        if ganho < args.ganho_min:
            break
        atuais.append(melhor)
        cand.remove(melhor)
        auc_atual = res[melhor]
        for h in hist:
            if h["passo"] == passo and h["candidata"] == melhor:
                h["escolhida"] = True
    pd.DataFrame(hist).to_csv(SAIDA / f"e23_selecao__{slug}.csv", index=False)
    final = dict(aparelho=args.aparelho, casas=[8] + atuais, auc=auc_atual)
    (SAIDA / f"e23_selecao__{slug}.json").write_text(json.dumps(final, indent=1))
    print(f"\n[E2.3] {args.aparelho}: casas selecionadas {[8] + atuais} (AUC {auc_atual:.4f})")


if __name__ == "__main__":
    main()
