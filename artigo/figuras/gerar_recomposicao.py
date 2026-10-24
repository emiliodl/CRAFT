"""Figure: the five recomposition operations decomposed into their components (English).

For the television (REFIT houses 8 and 15, training period), each panel shows what the
training generator of src/reco.py sums on the 30-s grid, after the count-weighted
resampling to 5 min: the background (grey area), the target appliance (accent area) or a
distractor (hatched grey area), and the resulting aggregate the model sees (line). R2
shows the removed target as a dashed accent line. Same visual language as the pipeline
figure.

Usage: python artigo/figuras/gerar_recomposicao.py [--A]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
from reco import Gerador, carregar_casa, definir_periodo  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
AXIS, BOX, SURFACE = "#c3c2b7", "#f0efec", "#ffffff"
ACCENT, ACCENT_WASH = "#2a78d6", "#e6effb"
FS = 1.3
LIM = 10.0                                   # TV label threshold (W)
# "--A": house 8 only (band A, the benchmark training data) -> recomposicao_A_en
CASAS = (8,) if "--A" in sys.argv else (8, 15)


def componentes(seed=11):
    torch.manual_seed(seed)
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    casas = []
    for k in CASAS:
        cf = carregar_casa(k, "television")
        cf.periodo = definir_periodo(cf, "cross")
        casas.append(cf)
    g = Gerador(casas, "television", "treino", disp, passo=4)
    gen = torch.Generator(device=disp).manual_seed(seed)
    ct = g.antes
    # a real window with the TV clearly on at the centre and a moderate background
    cand = g.centros[g.idx_on]
    p = g.res(g.P, cand)[:, ct]
    a_ = g.res(g.A, cand)
    ok = (p > 50) & (a_.max(1).values < 700)
    c_on = cand[ok][torch.randint(0, int(ok.sum()), (1,), device=disp, generator=gen)]

    def fundo():
        while True:
            f = g.sorteia_fundo(1, gen, evitar=c_on)
            b = g.res(g.B, f)
            if float(b.max()) < 700:
                return b

    A0, P0 = g.res(g.A, c_on), g.res(g.P, c_on)
    B3 = fundo()
    B1 = fundo()
    while True:
        a1, _, _ = g.injetar(g.ativ, g._serie_alvo, 1, gen)
        if float(a1[0, ct]) > 40 and int((a1[0] >= LIM).sum()) >= 8:
            break
    B4 = fundo()
    while True:
        d4, _, _ = g.injetar(g.dist, g._serie_dist, 1, gen)
        if 40 < float(d4[0, ct]) < 500 and int((d4[0] >= LIM).sum()) >= 3:
            break
    np_ = lambda v: v[0].detach().cpu().numpy()    # noqa: E731
    A0, P0, B3, B1, a1, B4, d4 = map(np_, (A0, P0, B3, B1, a1, B4, d4))
    return {
        "R0": dict(fundo=A0 - P0, alvo=P0, dist=None, removido=None, on=True),
        "R3": dict(fundo=B3, alvo=P0, dist=None, removido=None, on=True),
        "R1": dict(fundo=B1, alvo=a1, dist=None, removido=None, on=a1[ct] >= LIM),
        "R2": dict(fundo=A0 - P0, alvo=None, dist=None, removido=P0, on=False),
        "R4": dict(fundo=B4, alvo=None, dist=d4, removido=None, on=False),
    }


def main():
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
                         "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "dejavusans",
                         "hatch.linewidth": 0.5})
    comp = componentes()
    h = (np.arange(48) - 24) * 5 / 60
    titulos = {"R0": ("R0  real", "$A_h$"),
               "R3": ("R3  background swap", "$B_k + P_h$"),
               "R1": ("R1  injection", "$B_k + s\\,a_j$"),
               "R2": ("R2  removal", "$A_h - P_h$"),
               "R4": ("R4  distractor", "$B_k + s\\,d_j$")}
    fig, axs = plt.subplots(1, 5, figsize=(7.16, 1.95), sharey=True)
    ymax = max(float((c["fundo"] + (c["alvo"] if c["alvo"] is not None else 0)
                      + (c["dist"] if c["dist"] is not None else 0)).max()) for c in comp.values())
    for ax, op in zip(axs, ("R0", "R3", "R1", "R2", "R4")):
        c = comp[op]
        b = c["fundo"] / 1000
        topo = b.copy()
        ax.fill_between(h, 0, b, step="mid", color=BOX, lw=0)
        if c["alvo"] is not None:
            topo = b + c["alvo"] / 1000
            ax.fill_between(h, b, topo, step="mid", color=ACCENT, alpha=0.85, lw=0)
        if c["dist"] is not None:
            topo = b + c["dist"] / 1000
            ax.fill_between(h, b, topo, step="mid", color=MUTED, alpha=0.55, lw=0)
        ax.plot(h, topo, color=INK, lw=0.8, drawstyle="steps-mid")
        if c["removido"] is not None:
            ax.plot(h, b + c["removido"] / 1000, color=ACCENT, lw=0.9, ls=(0, (2.5, 1.5)),
                    drawstyle="steps-mid")
        ax.axvspan(-2.5 / 60, 2.5 / 60, color=AXIS, alpha=0.55, lw=0, zorder=0)
        titulo, formula = titulos[op]
        ax.set_title(titulo, fontsize=6.0 * FS, color=INK, fontweight="bold", pad=17)
        ax.text(0.5, 1.03, formula, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=5.4 * FS, color=INK2)
        ax.text(0.97, 0.95, "label on" if c["on"] else "label off", transform=ax.transAxes,
                ha="right", va="top", fontsize=5.3 * FS, color=ACCENT if c["on"] else MUTED,
                fontweight="bold" if c["on"] else "normal")
        ax.set_xlim(h[0], h[-1])
        ax.set_ylim(0, ymax / 1000 * 1.12)
        ax.set_xticks([-2, 0, 2])
        ax.set_xticklabels(["−2", "0", "2 h"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
            ax.spines[s].set_linewidth(0.5)
        ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=4.8 * FS, width=0.5, length=2,
                       pad=1)
    axs[0].set_ylabel("kW", fontsize=5.2 * FS, color=INK2, labelpad=1)
    # shared legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    leg = [Patch(facecolor=BOX, label="background"),
           Patch(facecolor=ACCENT, alpha=0.85, label="TV (target)"),
           Patch(facecolor=MUTED, alpha=0.55, label="distractor"),
           Line2D([], [], color=ACCENT, lw=0.9, ls=(0, (2.5, 1.5)), label="removed TV"),
           Line2D([], [], color=INK, lw=0.8, label="aggregate seen by the model"),
           Patch(facecolor=AXIS, alpha=0.55, label="interval to classify")]
    fig.legend(handles=leg, loc="lower center", ncol=6, frameon=False, fontsize=5.0 * FS,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.5)
    out = Path(__file__).with_name("recomposicao_A_en" if CASAS == (8,) else "recomposicao_en")
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    fig.savefig(out.with_suffix(".png"), dpi=300, facecolor=SURFACE)
    print("ok")


if __name__ == "__main__":
    main()
