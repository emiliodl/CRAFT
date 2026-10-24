"""Tabelas e figuras a partir de resultados/ablacao.csv."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "resultados"
FIG = RAIZ / "figuras"

ORDEM = ["piso", "bruto", "base", "z", "arcsinh", "3x_bruto", "3v_concat", "3v_soma",
         "3v_porta", "4v_porta", "3v_porta_tfc", "3v_porta_conj"]
GRUPO = {"piso": "chao", "bruto": "1 vista", "base": "1 vista", "z": "1 vista",
         "arcsinh": "1 vista", "3x_bruto": "controle", "3v_concat": "fusao",
         "3v_soma": "fusao", "3v_porta": "fusao", "4v_porta": "fusao",
         "3v_porta_tfc": "TF-C", "3v_porta_conj": "TF-C"}
CORES = {"chao": "#9AA0A6", "1 vista": "#2C5EA8", "controle": "#B35C00",
         "fusao": "#3F8F5B", "TF-C": "#7B4FA8"}


def resumo(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["aparelho", "braco"]).agg(
        n=("f1", "size"), f1=("f1", "mean"), f1_dp=("f1", "std"),
        f1_oraculo=("f1_oraculo", "mean"), ap=("ap", "mean"), auc=("auc", "mean"),
        f1_val=("f1_val", "mean"), epocas=("epocas", "mean"),
        parametros=("parametros", "mean"), segundos=("segundos", "mean")).reset_index()
    g["ordem"] = g.braco.map({b: i for i, b in enumerate(ORDEM)})
    return g.sort_values(["aparelho", "ordem"]).drop(columns="ordem").round(3)


def tabela_md(g: pd.DataFrame) -> str:
    linhas = []
    for ap, bloco in g.groupby("aparelho"):
        prev = 100 * float(bloco.iloc[0].get("prevalencia", np.nan)) if "prevalencia" in bloco else np.nan
        linhas.append(f"\n### {ap}" + (f"  (prevalência no alvo: {prev:.1f}%)" if prev == prev else ""))
        linhas.append("\n| braço | F1 | ± | F1 oráculo | AP | AUC | F1 val (origem) | épocas | params |")
        linhas.append("|---|---|---|---|---|---|---|---|---|")
        melhor = bloco.f1.max()
        for _, r in bloco.iterrows():
            nome = f"**{r.braco}**" if r.f1 == melhor else r.braco
            dp = "—" if r.f1_dp != r.f1_dp else f"{r.f1_dp:.3f}"
            linhas.append(f"| {nome} | {r.f1:.3f} | {dp} | {r.f1_oraculo:.3f} | {r.ap:.3f} | "
                          f"{r.auc:.3f} | {r.f1_val:.3f} | {r.epocas:.0f} | {r.parametros / 1000:.0f}k |")
    return "\n".join(linhas)


def figura(d: pd.DataFrame, g: pd.DataFrame, saida: Path) -> None:
    aparelhos = sorted(d.aparelho.unique())
    fig, eixos = plt.subplots(1, len(aparelhos), figsize=(5.2 * len(aparelhos), 6.2), sharey=False)
    eixos = np.atleast_1d(eixos)
    for a, aparelho in zip(eixos, aparelhos):
        bloco = g[g.aparelho == aparelho]
        y = np.arange(len(bloco))
        cores = [CORES[GRUPO[b]] for b in bloco.braco]
        a.barh(y, bloco.f1, color=cores, height=0.72,
               xerr=bloco.f1_dp.fillna(0), error_kw=dict(lw=1, ecolor="#444444", capsize=2))
        a.barh(y, bloco.f1_oraculo, color="none", edgecolor="#444444", height=0.72,
               lw=0.8, ls=":")
        for i, (f1, ora) in enumerate(zip(bloco.f1, bloco.f1_oraculo)):
            a.text(max(f1, ora) + 0.012, i, f"{f1:.3f}", va="center", fontsize=8.5)
        chao = float(bloco[bloco.braco == "piso"].f1.iloc[0]) if (bloco.braco == "piso").any() else None
        if chao is not None:
            a.axvline(chao, color="#9AA0A6", lw=1.2, ls="--", zorder=0)
        a.set_yticks(y, bloco.braco, fontsize=9)
        a.invert_yaxis()
        prev = 100 * float(d[d.aparelho == aparelho].prevalencia.mean())
        a.set_title(f"{aparelho}\nprevalência no alvo: {prev:.1f}%", fontsize=11, fontweight="bold")
        a.set_xlabel("F1 da classe positiva (UK-DALE casa 2)")
        a.set_xlim(0, min(1.0, max(bloco.f1_oraculo.max(), bloco.f1.max()) * 1.22))
        a.grid(axis="x", alpha=0.2, lw=0.6)
        for lado in ("top", "right"):
            a.spines[lado].set_visible(False)
    marcas = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CORES.values()]
    marcas.append(plt.Line2D([], [], color="#444444", ls=":", lw=1))
    fig.legend(marcas, list(CORES) + ["teto (limiar oráculo no alvo)"],
               loc="lower center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Treino: UK-DALE casa 1   ·   Teste: UK-DALE casa 2   ·   um modelo por aparelho",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    FIG.mkdir(exist_ok=True)
    fig.savefig(saida, dpi=150, bbox_inches="tight", facecolor="white")


def pesos(d: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for _, r in d[d.pesos_vistas.notna()].iterrows():
        p = r.pesos_vistas
        p = ast.literal_eval(p) if isinstance(p, str) else p
        if not isinstance(p, list):
            continue
        for vista, peso in zip(str(r.vistas).split("+"), p):
            linhas.append(dict(aparelho=r.aparelho, braco=r.braco, vista=vista, peso=peso))
    if not linhas:
        return pd.DataFrame()
    return (pd.DataFrame(linhas).groupby(["braco", "aparelho", "vista"]).peso.mean()
            .unstack().round(3))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--entrada", default="ablacao.csv")
    args = p.parse_args()

    d = pd.read_csv(RES / args.entrada)
    g = resumo(d)
    prev = d.groupby("aparelho").prevalencia.mean()
    g["prevalencia"] = g.aparelho.map(prev)

    print(g.to_string(index=False))
    print("\n=== pesos medios da porta ===")
    print(pesos(d).to_string())

    figura(d, g, FIG / "ablacao_3ramos.png")
    (RES / "tabelas.md").write_text(tabela_md(g), encoding="utf-8")
    g.to_csv(RES / "resumo.csv", index=False, encoding="utf-8")
    print(f"\n-> {FIG / 'ablacao_3ramos.png'}\n-> {RES / 'tabelas.md'}")


if __name__ == "__main__":
    main()
