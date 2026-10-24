"""Scoreboard figure (English): F1 of CRAFT vs the best benchmark method.

Only the pre-registered primary family: cross-dataset (REFIT -> ECO, band A+) and
intra-dataset (following month of house 8, band A). Dumbbell: open diamond =
benchmark, dot = CRAFT (mean of 10 seeds) over its 95% CI as a light band; the right
column gives the difference, in gray where the pre-registered test was a tie.
"""
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
AXIS, GRID, SURFACE = "#c3c2b7", "#ecebe7", "#ffffff"
ACCENT, ACCENT_CI = "#2a78d6", "#a9c8ee"

d = pd.read_csv(RAIZ / "resultados/v3/final_craftE__celulas.csv")
# in the result files the final model keeps its internal name "CRAFT+E" (the paper calls it
# CRAFT); "CRAFT" there is the ResNets-only variant
d = d[(d.modelo == "CRAFT+E") & (d.familia == "primaria")]
NOME = {"television": "TV", "washing machine": "Washing machine", "freezer": "Freezer"}
GRUPOS = [("Cross-dataset (REFIT → ECO)", "cross"), ("Intra-dataset (REFIT house 8)", "intra")]

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
                     "font.size": 8.2, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.5, 2.3))
y, rotulos = 0.0, []
for titulo, cel in GRUPOS:
    rotulos.append((y, titulo, True))
    y -= 0.95
    for ap in ("television", "washing machine", "freezer"):
        r = d[(d.celula == cel) & (d.aparelho == ap)].iloc[0]
        lo, hi = ast.literal_eval(r.ic95)
        ax.plot([r.artigo, r.f1], [y, y], color=AXIS, lw=1.4, solid_capstyle="round", zorder=1)
        ax.plot([lo, hi], [y, y], color=ACCENT_CI, lw=3.2, solid_capstyle="round", zorder=2)
        ax.plot(r.artigo, y, "D", ms=4.4, mfc=SURFACE, mec=INK2, mew=1.0, zorder=3)
        ax.plot(r.f1, y, "o", ms=5.4, color=ACCENT, mec=SURFACE, mew=0.8, zorder=4)
        dif = r.f1 - r.artigo
        txt = f"{dif:+.2f}".replace("-", "−")
        ax.text(1.03, y, txt, transform=ax.get_yaxis_transform(), ha="left", va="center",
                color=INK if r.supera_artigo else MUTED,
                fontweight="bold" if r.supera_artigo else "normal")
        rotulos.append((y, NOME[ap], False))
        y -= 1
    y -= 0.25
for yy, t, titulo in rotulos:
    ax.text(-0.36 if titulo else -0.03, yy, t, transform=ax.get_yaxis_transform(),
            ha="left" if titulo else "right", va="center",
            color=INK if titulo else INK2, fontweight="bold" if titulo else "normal")
ax.text(1.03, 0.0, "Δ F1", transform=ax.get_yaxis_transform(), ha="left", va="center",
        color=MUTED)
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
ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3, frameon=False,
          handletextpad=0.4, columnspacing=1.1, handlelength=1.4)
out = Path(__file__).with_name("placar_en")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor=SURFACE)
fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300, facecolor=SURFACE)
print("ok")
