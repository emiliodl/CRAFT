"""Scoreboard figure (English): F1 of CRAFT vs the best benchmark method at 5 and 15 min.

CRAFT trained on REFIT house 8 only (band A, the benchmark's training data). Two panels,
one per sampling interval, same rows: cross-dataset (REFIT 8 -> ECO) and intra-dataset
(following month of house 8). Dumbbell: open diamond = benchmark, dot = CRAFT (mean of 10
seeds) over its 95% CI as a light band; the right column gives the difference, bold when
significant after Holm correction within its pre-registered family. A dagger marks a
cell where CRAFT does not beat the all-on classifier.
"""
import ast
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
from estatistica import holm  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
AXIS, GRID, SURFACE = "#c3c2b7", "#ecebe7", "#ffffff"
ACCENT, ACCENT_CI = "#2a78d6", "#a9c8ee"
REGRA = {"cross": "d_L", "intra": "d_A"}
NOME = {"television": "TV", "washing machine": "Washing machine", "freezer": "Freezer"}
GRUPOS = [("S1, intra-dataset", "intra"), ("S2, cross-dataset", "cross")]


def celulas(arq: str) -> pd.DataFrame:
    """Band A, pre-registered rule; p_Holm within each family (intra: as pre-registered;
    cross-dataset band A: its own family of three)."""
    d = pd.read_csv(RAIZ / "resultados/v3" / arq)
    # in the result files the final model keeps its internal name "CRAFT+E"
    d = d[(d.modelo == "CRAFT+E") & (d.faixa == "A") & (d.regra == d.celula.map(REGRA))].copy()
    cr = d[d.celula == "cross"]
    h = holm(dict(zip(cr.aparelho, cr.p_artigo))).set_index("comparacao").p_holm
    d["p_fam"] = np.where(d.celula == "cross", d.aparelho.map(h), d.p_holm)
    d["sig"] = d.p_fam < 0.05
    d["trivial_ok"] = d.vs_trivial_ic.map(lambda s: ast.literal_eval(s)[0] > 0)
    return d


PAINEIS = [("5 min", celulas("final_craftE__celulas.csv")),
           ("15 min", celulas("final_craftE15__celulas.csv"))]

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
                     "font.size": 8.2, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.35), sharey=True,
                        gridspec_kw={"wspace": 0.22})
for k, (ax, (taxa, d)) in enumerate(zip(axs, PAINEIS)):
    y, rotulos = 0.0, []
    for titulo, cel in GRUPOS:
        rotulos.append((y, titulo, True))
        y -= 0.95
        for ap in ("television", "washing machine", "freezer"):
            r = d[(d.celula == cel) & (d.aparelho == ap)].iloc[0]
            lo, hi = ast.literal_eval(r.ic95)
            ax.plot([r.artigo, r.f1], [y, y], color=AXIS, lw=1.4, solid_capstyle="round",
                    zorder=1)
            ax.plot([lo, hi], [y, y], color=ACCENT_CI, lw=3.2, solid_capstyle="round", zorder=2)
            ax.plot(r.artigo, y, "D", ms=4.4, mfc=SURFACE, mec=INK2, mew=1.0, zorder=3)
            ax.plot(r.f1, y, "o", ms=5.4, color=ACCENT, mec=SURFACE, mew=0.8, zorder=4)
            txt = f"{r.f1 - r.artigo:+.2f}".replace("-", "−") + ("" if r.trivial_ok else "†")
            ax.text(1.03, y, txt, transform=ax.get_yaxis_transform(), ha="left", va="center",
                    color=INK if r.sig else MUTED, fontweight="bold" if r.sig else "normal")
            rotulos.append((y, NOME[ap], False))
            y -= 1
        y -= 0.25
    if k == 0:
        for yy, t, titulo in rotulos:
            ax.text(-0.42 if titulo else -0.03, yy, t, transform=ax.get_yaxis_transform(),
                    ha="left" if titulo else "right", va="center",
                    color=INK if titulo else INK2, fontweight="bold" if titulo else "normal")
    ax.text(1.03, 0.0, "Δ F1", transform=ax.get_yaxis_transform(), ha="left", va="center",
            color=MUTED)
    ax.set_title(taxa, loc="center", color=INK, fontweight="bold", pad=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(y + 0.55, 0.55)
    ax.set_yticks([])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("F1 of the “on” class", color=INK2, labelpad=2)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(colors=MUTED, labelcolor=INK2, width=0.5, length=2)
h = [Line2D([], [], color=ACCENT, marker="o", lw=0, ms=5.4, mec=SURFACE, label="CRAFT"),
     Line2D([], [], color=ACCENT_CI, lw=3.2, solid_capstyle="round", label="95% CI"),
     Line2D([], [], color=INK2, marker="D", lw=0, ms=4.4, mfc=SURFACE, mew=1.0,
            label="Best benchmark method")]
fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False,
           handletextpad=0.4, columnspacing=1.4, handlelength=1.4)
out = Path(__file__).with_name("placar_taxas_en")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor=SURFACE)
fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300, facecolor=SURFACE)
print("ok")
