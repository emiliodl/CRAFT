"""Figure: two test days of the TV in ECO house 2 (5 min, trained on REFIT house 8 only, seed 0): aggregate, true state
and the CRAFT decision (rule d_L). Uses predictions already analysed in the
pre-registered closure (no new test query). Same visual language as the pipeline figure.

The window is representative, not the best one: the two consecutive days whose F1 is
closest to the F1 of the whole cell for this seed.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
from decisao import Decisor  # noqa: E402
from final_craftE import conjunto, membros_da_semente, sig  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
AXIS, BOX, SURFACE = "#c3c2b7", "#f0efec", "#ffffff"
ACCENT = "#2a78d6"
FS = 1.3


def f1(yy, pp):
    tp = (pp & (yy == 1)).sum()
    return 2 * tp / max(pp.sum() + yy.sum(), 1)


ms = membros_da_semente("television", ["craftF_A"], "craftE_A_ev", 0)   # band A
y, t, _, z = conjunto(ms, "cross: eco casa 2")
cc = [conjunto(ms, k) for k in sorted(k for k in ms[0] if k.startswith("cv:"))]
dec = Decisor(sig(np.concatenate([c[3] for c in cc])), np.concatenate([c[0] for c in cc]))
pred = dec.decidir(sig(z))["d"].astype(bool)
d = np.load(RAIZ / "datasets/protocolo_artigo/eco_casa_2__television__5min__cross.npz")
agg = dict(zip(d["t"].astype(np.int64), d["main"]))
x = np.array([agg.get(int(v), np.nan) for v in t])

dia = (t - t[0]) // 86400
f_cel = f1(y, pred)
cands = [(abs(f1(y[m], pred[m]) - f_cel), k) for k in range(dia.max())
         if (m := (dia >= k) & (dia <= k + 1)).any() and y[m].sum() > 0]
k0 = min(cands)[1]
sel = (dia >= k0) & (dia <= k0 + 1)
hh = (t[sel] - t[sel][0]) / 3600
print("F1 cell:", round(f_cel, 3), "window:", round(f1(y[sel], pred[sel]), 3))

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42})
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 2.45), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.25]})
ys, ps = y[sel] == 1, pred[sel]
# shade the true on-periods behind the aggregate
a1.fill_between(hh, 0, 1, where=ys, step="mid", color=BOX, lw=0,
                transform=a1.get_xaxis_transform())
a1.plot(hh, x[sel] / 1000, color=INK2, lw=0.7, drawstyle="steps-mid")
a1.set_ylabel("kW", fontsize=5.4 * FS, color=INK2, labelpad=1)
a1.text(0.99, 0.97, "aggregate, ECO house 2", transform=a1.transAxes, ha="right", va="top",
        fontsize=5.0 * FS, color=INK2)
for ax in (a1, a2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.5)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=5.0 * FS, width=0.5, length=2)
for y0, estado, cor in ((1.1, ys, INK2), (0.1, ps, ACCENT)):
    a2.fill_between(hh, y0, y0 + 0.75, color=BOX, lw=0, step="mid")
    a2.fill_between(hh, y0, y0 + 0.75, where=estado, color=cor, lw=0, step="mid")
a2.set_yticks([0.475, 1.475])
a2.set_yticklabels(["CRAFT", "true state"])
a2.get_yticklabels()[0].set_color(ACCENT)
a2.get_yticklabels()[0].set_fontweight("bold")
a2.set_ylim(0, 1.95)
a2.spines["left"].set_visible(False)
a2.tick_params(axis="y", length=0)
a2.set_xlabel("time (h)", fontsize=5.4 * FS, color=INK2, labelpad=1)
a2.set_xlim(0, 48)
a2.set_xticks([0, 12, 24, 36, 48])
fig.tight_layout(h_pad=0.2)
out = Path(__file__).with_name("exemplo_teste_en")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor=SURFACE)
fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300, facecolor=SURFACE)
