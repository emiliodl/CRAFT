"""Relatorio do v2 (docs/v2.md §3.7, §5.3): medias por semente, decisao e bootstrap.

Le resultados/v2/<tag>.csv e as predicoes em resultados/v2/pred/<tag>/ e produz:

  <tag>__resumo.csv       media e dp entre sementes, por aparelho x alvo x modelo
  <tag>__pareado.csv      bootstrap pareado por dias (e sementes) de F1 e AUC
                          entre um modelo e a referencia, com a MESMA decisao
  <tag>__decisao.csv      E1.1: regras a/b/c/d contra o oraculo, lacuna fechada

Uso:
    python src/relatorio_v2.py --tag escada_A --ref A0 --comparar A2,A4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decisao import Decisor, auc                               # noqa: E402
from estatistica import blocos_dia, bootstrap_f1                # noqa: E402

SAIDA = Path(__file__).resolve().parents[1] / "resultados" / "v2"


def carregar_pred(tag: str, aparelho: str, modelo: str, semente: int) -> dict:
    arq = SAIDA / "pred" / tag / f"{aparelho.replace(' ', '_')}__{modelo}__s{semente}.npz"
    d = np.load(arq)
    out = {}
    for k in d.files:
        alvo, campo = k.rsplit("|", 1)
        out.setdefault(alvo, {})[campo] = d[k]
    return out


def decisoes(pred: dict, alvo: str, calib: str) -> dict:
    """Binarias das regras a-d para `alvo`, calibrando em `calib`:
    'A' = holdout temporal da fonte; 'L' = outras casas de cross-val (ou todas,
    para alvos de teste)."""
    y, p = pred[alvo]["y"], pred[alvo]["p"]
    if calib == "A":
        dec = Decisor(pred["val"]["p"], pred["val"]["y"], auc_retida=None)
    else:
        cvs = [k for k in pred if k.startswith("cv:") and k != alvo]
        yo = np.concatenate([pred[k]["y"] for k in cvs])
        po = np.concatenate([pred[k]["p"] for k in cvs])
        dec = Decisor(po, yo, auc_retida=np.nanmean([auc(pred[k]["y"], pred[k]["p"])
                                                     for k in cvs]))
    return dec.decidir(p)


def recomputar(tag: str, d: pd.DataFrame) -> pd.DataFrame:
    """Refaz as colunas de decisao a partir das predicoes salvas, com o decisao.py
    ATUAL (os processos longos podem ter rodado com uma versao anterior)."""
    from decisao import limiar_oraculo
    linhas = []
    for (ap, m, s), g in d.groupby(["aparelho", "modelo", "semente"]):
        try:
            P = carregar_pred(tag, ap, m, s)
        except FileNotFoundError:
            continue
        for alvo in P:
            if alvo == "val":
                continue
            y, p = P[alvo]["y"], P[alvo]["p"]
            lin = dict(aparelho=ap, modelo=m, semente=s, alvo=alvo)
            for calib in ("A", "L"):
                dd = decisoes(P, alvo, calib)
                for r in "abcde":
                    lin[f"{calib}_f1_{r}"] = f1b(y, dd[r])
                lin[f"{calib}_trava"] = dd["log"]["trava"]
            linhas.append(lin)
    novo = pd.DataFrame(linhas)
    velhas = [c for c in d.columns if c.startswith(("A_", "L_"))]
    return d.drop(columns=velhas).merge(novo, on=["aparelho", "modelo", "semente", "alvo"])


def resumo(d: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("auc", "ap", "f1_oraculo", "trivial", "A_f1_a", "A_f1_b", "A_f1_c",
                        "A_f1_d", "A_f1_e", "L_f1_a", "L_f1_c", "L_f1_d", "L_f1_e", "delta_rem", "delta_add",
                        "delta_add_madrugada", "delta_add_dia", "epocas") if c in d]
    g = d.groupby(["aparelho", "alvo", "modelo"])
    m = g[cols].mean().round(3)
    s = g[[c for c in ("auc", "A_f1_e", "L_f1_e") if c in d]].std().round(3).add_suffix("_dp")
    return m.join(s).reset_index()


def pareado(tag: str, d: pd.DataFrame, ref: str, outros: list[str], regra: str,
            calib: str, n_boot: int) -> pd.DataFrame:
    linhas = []
    for (ap, alvo), g in d.groupby(["aparelho", "alvo"]):
        sem = sorted(g.semente.unique())
        if ref not in set(g.modelo):
            continue
        P = {m: [carregar_pred(tag, ap, m, s) for s in sem] for m in [ref] + outros
             if m in set(g.modelo)}
        y = P[ref][0][alvo]["y"]
        dias = blocos_dia(P[ref][0][alvo]["t"])
        bin_ = {m: np.stack([decisoes(p, alvo, calib)[regra] for p in ps]) for m, ps in P.items()}
        for m in P:
            if m == ref:
                continue
            b = bootstrap_f1(y, bin_[m], dias, n=n_boot, pred_b=bin_[ref])
            bt = bootstrap_f1(y, bin_[m], dias, n=n_boot, pred_b=np.ones(len(y)))
            linhas.append(dict(aparelho=ap, alvo=alvo, modelo=m, ref=ref, regra=f"{calib}_{regra}",
                               f1=round(float(np.mean([f1b(y, x) for x in bin_[m]])), 4),
                               f1_ref=round(float(np.mean([f1b(y, x) for x in bin_[ref]])), 4),
                               f1_ic=b["f1_ic"], dif_ic=b["dif_ic"],
                               minimo_detectavel=b["minimo_detectavel"],
                               vs_trivial_ic=bt["dif_ic"], dias=b["dias"]))
            print(f"  {ap:16s} {alvo:22s} {m:5s} vs {ref}: F1 {linhas[-1]['f1']:.3f} vs "
                  f"{linhas[-1]['f1_ref']:.3f}  dif IC95 {b['dif_ic'][0]:+.3f}..{b['dif_ic'][2]:+.3f}"
                  f"  vs trivial {bt['dif_ic'][0]:+.3f}..{bt['dif_ic'][2]:+.3f}", flush=True)
    return pd.DataFrame(linhas)


def f1b(y, pred) -> float:
    y, pred = np.asarray(y) == 1, np.asarray(pred, bool)
    vp = (y & pred).sum()
    return 0.0 if vp == 0 else 2 * vp / (pred.sum() + y.sum())


def decisao_e11(d: pd.DataFrame) -> pd.DataFrame:
    """E1.1 / H3: fracao da lacuna (oraculo - regra a) fechada por cada regra,
    nas casas de cross-val (LOHO) - onde a trava tem todos os criterios."""
    cv = d[d.alvo.str.startswith("cv:")].copy()
    for r in "abcde":
        cv[f"regra_{r}"] = cv[f"L_f1_{r}"]
    cv["lacuna_a"] = cv.f1_oraculo - cv.regra_a
    cv["fecha_c"] = (cv.regra_c - cv.regra_a) / cv.lacuna_a.where(cv.lacuna_a > 0.01)
    cv["fecha_e"] = (cv.regra_e - cv.regra_a) / cv.lacuna_a.where(cv.lacuna_a > 0.01)
    cv["c_abaixo_trivial"] = cv.regra_c < cv.trivial - 1e-3
    cv["e_abaixo_trivial"] = cv.regra_e < cv.trivial - 1e-3
    return cv.groupby(["aparelho", "modelo"]).agg(
        auc=("auc", "mean"), oraculo=("f1_oraculo", "mean"), a=("regra_a", "mean"),
        b=("regra_b", "mean"), c=("regra_c", "mean"), d=("regra_d", "mean"),
        trivial=("trivial", "mean"), fecha_c=("fecha_c", "median"),
        c_abaixo_trivial=("c_abaixo_trivial", "mean")).round(3).reset_index()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="escada_A")
    p.add_argument("--ref", default="A0")
    p.add_argument("--comparar", default="A2,A4")
    p.add_argument("--regra", default="e")
    p.add_argument("--calib", default="L")
    p.add_argument("--n_boot", type=int, default=1000)
    args = p.parse_args()
    arqs = sorted(SAIDA.glob(f"{args.tag}__*.csv"))
    arqs = [a for a in arqs if not any(x in a.name for x in ("__resumo", "__pareado",
                                                              "__decisao", "__args", "__recomputado"))]
    d = pd.concat([pd.read_csv(a) for a in arqs]) if arqs else pd.read_csv(SAIDA / f"{args.tag}.csv")
    d = recomputar(args.tag, d)
    d.to_csv(SAIDA / f"{args.tag}__recomputado.csv", index=False)
    r = resumo(d)
    r.to_csv(SAIDA / f"{args.tag}__resumo.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(r.to_string(index=False))
    e = decisao_e11(d)
    e.to_csv(SAIDA / f"{args.tag}__decisao.csv", index=False)
    print("\n=== E1.1 (casas de cross-val, calibracao LOHO) ===")
    print(e.to_string(index=False))
    if args.comparar:
        print(f"\n=== pareado por dias x sementes, regra {args.calib}_{args.regra} ===")
        pr = pareado(args.tag, d, args.ref, args.comparar.split(","), args.regra, args.calib,
                     args.n_boot)
        pr.to_csv(SAIDA / f"{args.tag}__pareado_{args.calib}{args.regra}.csv", index=False)


if __name__ == "__main__":
    main()
