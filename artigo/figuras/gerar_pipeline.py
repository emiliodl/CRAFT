"""Figure: CRAFT pipeline with real data at every stage (English version).

Visual language shared with the companion paper (fir_mobilenet, fig. 4): neutral ink,
borderless grey boxes, one accent hue. Here the accent marks two things only: the
recomposition generator (the main contribution, on a light wash) and the target
appliance wherever it appears (plug trace, injected activation, "on" states).

Top lane, training on source houses: a real 4-h window of REFIT house 8 on the 30-s grid
(aggregate and TV plug) -> one real example of each recomposition operation, produced by
the training generator and resampled to 5 min -> training of the ensemble.
Bottom lane, inference on a target house: a real 4-h window of ECO house 2 (test set,
already analysed in the pre-registered closure) -> the four input channels computed from
it -> five ResNets + evidence branch (edges of the same window) -> calibrated threshold ->
the decisions of the model (band A+, seed 0) against the true state.

Usage: python artigo/figuras/gerar_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
from decisao import Decisor  # noqa: E402
from final_craftE import conjunto, membros_da_semente, sig  # noqa: E402
from reco import OPS, SUB, Gerador, carregar_casa, definir_periodo  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
AXIS, BOX, SURFACE = "#c3c2b7", "#f0efec", "#ffffff"
ACCENT, ACCENT_WASH = "#2a78d6", "#e6effb"
FS = 1.45
W, H = 7.16, 3.35                      # IEEE text width (in)
PREF_TV = 70.0                         # median TV power when on, band A+ training (W)


# ------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------

def janela_treino():
    """A real 4-h window of REFIT house 8 on the 30-s grid, centred on a TV switch-on."""
    c = carregar_casa(8, "television")
    n = 48 * SUB
    p5 = c.P[: len(c.P) // SUB * SUB].reshape(-1, SUB).mean(1)
    on = p5 >= 10
    ini = np.flatnonzero(on[1:] & ~on[:-1]) + 1
    melhor, escolha = np.inf, None
    for i in ini:
        a, b = (i - 22) * SUB, (i + 26) * SUB
        if a <= 0 or b >= len(c.A) or not c.ok[a:b].all() or not on[i:i + 12].all():
            continue
        if p5[i:i + 12].mean() - p5[i - 12:i].mean() < 50:   # a TV-sized plug step
            continue
        faixa = c.A[a:b].max() - c.A[a:b].min()
        if faixa < melhor:                        # the quietest house around a TV switch-on
            melhor, escolha = faixa, (a, b)
    a, b = escolha
    return np.arange(n) * 30 / 3600, c.A[a:b], c.P[a:b]


def exemplos_recomposicao():
    torch.manual_seed(3)
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    casas = []
    for k in (8, 15):
        cf = carregar_casa(k, "television")
        cf.periodo = definir_periodo(cf, "cross")
        casas.append(cf)
    g = Gerador(casas, "television", "treino", disp, passo=4)
    gen = torch.Generator(device=disp).manual_seed(7)
    base = g.base[torch.randint(0, len(g.base), (400,), device=disp, generator=gen)]
    L = g.lote(base, gen)
    x, y, op, a = (L[k].cpu().numpy() for k in ("x", "y", "op", "a"))
    rng = np.random.default_rng(0)
    out = {}
    for nome, on in (("R0", 1), ("R3", 1), ("R1", 1), ("R2", 0), ("R4", 0)):
        idx = np.flatnonzero((op == OPS.index(nome)) & (y == on))
        if on:
            forte = idx[a[idx, x.shape[1] // 2] > 40]
            idx = forte if len(forte) else idx
        i = idx[rng.integers(len(idx))]
        out[nome] = (x[i], a[i], on)
    return out


def janela_teste():
    """A real 4-h window of ECO house 2 (TV test cell) with true state and decisions."""
    ms = membros_da_semente("television", ["craftE_Amais_tv"], "craftE_Amais_ev", 0)
    y, t, _, z = conjunto(ms, "cross: eco casa 2")
    cc = [conjunto(ms, k) for k in sorted(k for k in ms[0] if k.startswith("cv:"))]
    dec = Decisor(sig(np.concatenate([c[3] for c in cc])), np.concatenate([c[0] for c in cc]))
    pred = dec.decidir(sig(z))["d"].astype(bool)
    d = np.load(RAIZ / "datasets/protocolo_artigo/eco_casa_2__television__5min__cross.npz")
    agg = dict(zip(d["t"].astype(np.int64), d["main"]))
    x = np.array([agg.get(int(v), np.nan) for v in t])
    # a switch-on whose step is legible in the aggregate: quiet background before it, the
    # model right on at least 90% of the window; chosen by the step-to-noise ratio
    cont = np.r_[False, np.diff(t) == 300]
    ini = np.flatnonzero((y[1:] == 1) & (y[:-1] == 0)) + 1
    melhor, escolha = -1.0, None
    for i in ini:
        a, b = i - 22, i + 26
        if a < 0 or b > len(t) or not cont[a + 1:b].all() or np.isnan(x[a:b]).any():
            continue
        if (pred[a:b] == y[a:b].astype(bool)).mean() < 0.9:
            continue
        degrau = x[i:i + 6].mean() - x[i - 6:i].mean()      # sustained for 30 min
        faixa = x[a:b].max() - x[a:b].min()
        if not (80 < degrau < 400) or faixa > 700:          # a TV-sized step, no big loads
            continue
        ruido = x[a:i - 1].std() + np.abs(np.diff(x[i + 1:b])).max()
        if degrau / (ruido + 30) > melhor:
            melhor, escolha = degrau / (ruido + 30), (a, b)
    if escolha is None:
        raise RuntimeError("no window")
    a, b = escolha
    return x[a:b], y[a:b], pred[a:b], np.nanmedian(x)


# ------------------------------------------------------------------------------------
# drawing helpers (figure coordinates in inches)
# ------------------------------------------------------------------------------------

def ax_in(fig, x, y, w, h):
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.set_zorder(5)                   # always above the boxes and cards
    ax.patch.set_alpha(0)
    return ax


def limpa(ax, eixos=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in eixos)
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.5)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=4.8 * FS, width=0.5, length=2, pad=1)


def caixa(fig, x, y, w, h, fill=BOX, raio=0.06, z=0):
    fig.patches.append(FancyBboxPatch((x / W, y / H), w / W, h / H, transform=fig.transFigure,
                                      boxstyle=f"round,pad=0,rounding_size={raio / W}",
                                      facecolor=fill, edgecolor="none", zorder=z))


def seta(fig, x0, y0, x1, y1, cor=MUTED, estilo="-", txt=None, dy=0.05, conn=None):
    fig.patches.append(FancyArrowPatch((x0 / W, y0 / H), (x1 / W, y1 / H),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=7, lw=0.9, color=cor, linestyle=estilo,
                                       shrinkA=0, shrinkB=0, connectionstyle=conn))
    if txt:
        fig.text((x0 + x1) / 2 / W, (max(y0, y1) + dy) / H, txt, ha="center", va="bottom",
                 fontsize=5.6 * FS, color=INK2, linespacing=1.0)


def texto(fig, x, y, s, size=6.0, cor=INK2, bold=False, ha="center", va="center", **kw):
    fig.text(x / W, y / H, s, ha=ha, va=va, fontsize=size * FS, color=cor,
             fontweight="bold" if bold else "normal", **kw)


def verifica_vazamentos(fig, folga=0.04):
    """Prints every text whose centre lies inside a box or card but whose extent leaves
    it (with a margin of `folga` inches). Boxes are the rounded patches of the figure."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dpi = fig.dpi
    caixas = []
    for p in fig.patches:
        if isinstance(p, FancyBboxPatch):
            b = p.get_window_extent(r)
            caixas.append((b.x0 / dpi, b.y0 / dpi, b.x1 / dpi, b.y1 / dpi))
    problemas = 0
    for t in fig.texts:
        if not t.get_text().strip():
            continue
        b = t.get_window_extent(r)
        x0, y0, x1, y1 = b.x0 / dpi, b.y0 / dpi, b.x1 / dpi, b.y1 / dpi
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        dentro = [c for c in caixas if c[0] <= cx <= c[2] and c[1] <= cy <= c[3]]
        if not dentro:
            continue
        c = dentro[-1]                  # the box drawn last is the one on top (front card)
        if x0 < c[0] + folga or x1 > c[2] - folga or y0 < c[1] + folga or y1 > c[3] - folga:
            problemas += 1
            print(f"VAZA: {t.get_text()[:45]!r}  texto x[{x0:.2f},{x1:.2f}] y[{y0:.2f},{y1:.2f}]"
                  f"  bloco x[{c[0]:.2f},{c[2]:.2f}] y[{c[1]:.2f},{c[3]:.2f}]")
    print("vazamentos:", problemas)


def faixa_estado(ax, estado, cor, y0, h=0.62):
    ax.add_patch(Rectangle((-0.5, y0), len(estado), h, facecolor=BOX, edgecolor="none"))
    for i, v in enumerate(estado):
        if v:
            ax.add_patch(Rectangle((i - 0.5, y0), 1, h, facecolor=cor, edgecolor="none"))


# ------------------------------------------------------------------------------------

def main() -> None:
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
                         "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "dejavusans"})
    ht, A30, P30 = janela_treino()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gerar_recomposicao import componentes          # same real windows as the R0-R4 figure
    r1 = componentes()["R1"]
    xw, yw, pw, sigma_h = janela_teste()
    fig = plt.figure(figsize=(W, H))
    h5 = (np.arange(48) - 24) * 5 / 60

    # ============================ top lane: training ============================
    texto(fig, 0.08, 3.22, "Training  ·  REFIT source houses", 6.4, INK, True, ha="left")
    # (1) source window on the 30-s grid
    ax = ax_in(fig, 0.36, 2.02, 1.02, 0.78)
    ax.plot(ht, A30 / 1000, color=INK2, lw=0.55)
    ax.plot(ht, P30 / 1000, color=ACCENT, lw=1.0)
    ax.set_xlim(0, 4)
    ax.set_xticks([0, 2, 4])
    ax.set_xticklabels(["0", "2", "4 h"])
    limpa(ax)
    ax.set_ylabel("kW", fontsize=5.0 * FS, color=INK2, labelpad=1)
    ymax = ax.get_ylim()[1]
    ax.text(0.08, ymax * 0.99, "aggregate", color=INK2, fontsize=5.0 * FS, ha="left", va="top")
    ax.text(4, P30.max() / 1000 + ymax * 0.05, "TV plug", color=ACCENT, fontsize=5.0 * FS,
            ha="right", va="bottom", fontweight="bold")
    texto(fig, 0.87, 1.76, "Source house, 30-s grid", 5.6)

    # (2) recomposition panel (accent wash)
    px0, px1 = 1.62, 5.28
    caixa(fig, px0, 1.78, px1 - px0, 1.34, ACCENT_WASH, 0.08, z=-1)
    texto(fig, (px0 + px1) / 2, 2.99, "Aggregate recomposition", 7.0, INK, True)
    seta(fig, 1.43, 2.41, 1.58, 2.41)
    # one operation as a visual equation with real data: background of house k + a real TV
    # activation = recomposed window (the injection, R1); the five operations have their
    # own figure
    b = r1["fundo"] / 1000
    p = r1["alvo"] / 1000
    ytop = float((b + p).max()) * 1.15
    mw, mh, my = 0.84, 0.52, 2.22
    xs_ = [2.02, 3.12, 4.22]
    partes = [("background of house $k$", "$B_k$"), ("TV activation", "$s\,a_j$"),
              ("recomposed window", "label on")]
    for j, (x_, (tit, rod)) in enumerate(zip(xs_, partes)):
        ax = ax_in(fig, x_, my, mw, mh)
        if j in (0, 2):
            ax.fill_between(h5, 0, b, step="mid", color=SURFACE, lw=0)
            ax.fill_between(h5, 0, b, step="mid", color=AXIS, alpha=0.45, lw=0)
        if j == 1:
            ax.fill_between(h5, 0, p, step="mid", color=ACCENT, alpha=0.9, lw=0)
        if j == 2:
            ax.fill_between(h5, b, b + p, step="mid", color=ACCENT, alpha=0.9, lw=0)
            ax.plot(h5, b + p, color=INK, lw=0.8, drawstyle="steps-mid")
        if j == 0:
            ax.plot(h5, b, color=INK2, lw=0.6, drawstyle="steps-mid")
        ax.axvline(0, color=INK2, lw=0.6, ls=(0, (1, 1.5)))
        ax.set_xlim(h5[0], h5[-1])
        ax.set_ylim(0, ytop)
        ax.set_xticks([])
        ax.set_yticks([])
        limpa(ax, ("bottom",))
        cx = x_ + mw / 2
        texto(fig, cx, my + mh + 0.08, tit, 5.4, INK, j == 2)
        texto(fig, cx, my - 0.1, rod, 5.4 if j < 2 else 5.6,
              ACCENT if j == 2 else INK2, bold=(j == 2))
    texto(fig, xs_[0] + mw + (xs_[1] - xs_[0] - mw) / 2, my + mh / 2, "+", 11, INK2)
    texto(fig, xs_[1] + mw + (xs_[2] - xs_[1] - mw) / 2, my + mh / 2, "=", 11, INK2)
    texto(fig, (px0 + px1) / 2, 1.93,
          "30-s sum  →  count-weighted 5-min mean  →  "
          "label recomputed ($\\bar{p}_t \\geq \\theta$)", 5.4)

    # (3) training box
    seta(fig, px1 + 0.04, 2.41, px1 + 0.22, 2.41)
    bx0 = px1 + 0.26
    caixa(fig, bx0, 1.98, W - 0.08 - bx0, 0.86)
    texto(fig, bx0 + (W - 0.08 - bx0) / 2, 2.62, "Train the ensemble", 6.4, INK, True)
    texto(fig, bx0 + (W - 0.08 - bx0) / 2, 2.31,
          "5 ResNets: BCE on\nreal + recomposed\nevidence branch: real", 5.4, linespacing=1.15)

    # ============================ bottom lane: inference ============================
    texto(fig, 0.08, 1.5, "Inference  ·  target house, 5-min smart meter", 6.4, INK, True, ha="left")
    # (4) 4-h window of the target house
    ax = ax_in(fig, 0.36, 0.42, 0.98, 0.72)
    ax.plot(h5, xw / 1000, color=INK2, lw=0.8, drawstyle="steps-mid")
    ax.axvspan(-2.5 / 60, 2.5 / 60, color=AXIS, alpha=0.6, lw=0)
    ax.set_xlim(h5[0], h5[-1])
    ax.set_xticks([-2, 0, 2])
    ax.set_xticklabels(["−2", "0", "2 h"])
    limpa(ax)
    ax.set_ylabel("kW", fontsize=5.0 * FS, color=INK2, labelpad=1)
    texto(fig, 0.85, 0.14, "4-h window, ECO house 2", 5.6)

    # (5) four input channels
    seta(fig, 1.42, 0.78, 1.58, 0.78)
    dx = np.r_[0, np.diff(xw)]
    v1 = np.asinh(xw / sigma_h)
    canais = [((v1 - v1.mean()) / (v1.std() + 1e-9), "level"),
              (np.clip(xw / PREF_TV, -10, 10), r"$x/P_{\mathrm{ref}}$"),
              (np.clip(np.maximum(dx, 0) / PREF_TV, 0, 10), "rises"),
              (np.clip(np.maximum(-dx, 0) / PREF_TV, 0, 10), "falls")]
    cx0, cw, ch = 1.98, 0.62, 0.2
    for k, (v, nome) in enumerate(canais):
        yb = 1.12 - k * 0.255
        ax = ax_in(fig, cx0, yb, cw, ch)
        ax.plot(h5, v, color=INK2, lw=0.6, drawstyle="steps-mid")
        ax.set_xlim(h5[0], h5[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        limpa(ax, ())
        texto(fig, cx0 - 0.04, yb + ch / 2, nome, 5.2, ha="right")
    texto(fig, cx0 + cw / 2 - 0.12, 0.14, "4 input channels", 5.6)

    # (6) ensemble
    seta(fig, 2.64, 0.78, 2.78, 0.78)
    ex0, ex1 = 2.82, 4.82
    caixa(fig, ex0, 0.18, ex1 - ex0, 1.16)
    texto(fig, (ex0 + ex1) / 2, 1.2, "Ensemble of 6 members", 6.4, INK, True)
    cw_, ch_, step = 0.82, 0.56, 0.035                     # deck of five ResNet cards
    for k in range(5):
        caixa(fig, ex0 + 0.08 + step * k, 0.36 + step * k, cw_, ch_, SURFACE, 0.04, z=1)
    fx, fy = ex0 + 0.08 + step * 4, 0.36 + step * 4        # front card
    texto(fig, fx + cw_ / 2, fy + 0.41, "1D ResNet \u00d7 5", 5.0, INK, True)
    texto(fig, fx + cw_ / 2, fy + 0.2, "$k$ = 5, 7, 9,\n15, 25", 4.9, linespacing=1.05)
    # evidence branch card with the real edges of the window
    ecx, ecw = ex0 + 1.08, 0.86
    caixa(fig, ecx, 0.36, ecw, 0.72, SURFACE, 0.04, z=1)
    texto(fig, ecx + ecw / 2, 0.98, "Evidence branch", 4.8, INK, True)
    ax = ax_in(fig, ecx + 0.08, 0.55, ecw - 0.16, 0.31)
    up = np.maximum(dx, 0) / PREF_TV
    dn = -np.maximum(-dx, 0) / PREF_TV
    lim = max(up.max(), -dn.min()) * 1.15
    ax.bar(h5, up, width=0.07, color=ACCENT, lw=0)
    ax.bar(h5, dn, width=0.07, color=MUTED, lw=0)
    ax.axhline(0, xmin=0.2, color=AXIS, lw=0.5)
    ax.set_xlim(h5[0] - 1.0, h5[-1])                # room on the left for the labels
    ax.set_ylim(-lim, lim)
    ax.set_xticks([])
    ax.set_yticks([])
    limpa(ax, ())
    ax.text(h5[0] - 1.0, lim * 0.45, "$a^+$", fontsize=4.8 * FS, color=ACCENT, va="center")
    ax.text(h5[0] - 1.0, -lim * 0.45, "$a^-$", fontsize=4.8 * FS, color=MUTED, va="center")
    texto(fig, ecx + ecw / 2, 0.47, "edges → state", 4.8)
    # trained weights from the top lane
    seta(fig, bx0 + (W - 0.08 - bx0) / 2, 1.98, (ex0 + ex1) / 2 + 0.3, 1.36, cor=ACCENT,
         estilo=(0, (3, 2)), conn="angle,angleA=-90,angleB=0,rad=0")
    texto(fig, bx0 + (W - 0.08 - bx0) / 2 - 0.06, 1.62, "trained weights", 5.3, ACCENT, ha="right")

    # (7) calibrated threshold
    seta(fig, ex1 + 0.04, 0.78, ex1 + 0.24, 0.78)
    texto(fig, ex1 + 0.14, 0.6, "mean\n$z$", 5.0, linespacing=1.0)
    dx0, dw = ex1 + 0.28, 1.1
    caixa(fig, dx0, 0.18, dw, 1.16)
    texto(fig, dx0 + dw / 2, 1.2, "Decision", 6.4, INK, True)
    ax = ax_in(fig, dx0 + 0.2, 0.58, dw - 0.34, 0.46)
    zz = np.linspace(-6, 6, 200)
    ax.plot(zz, 1 / (1 + np.exp(-zz / 1.6)), color=INK2, lw=0.9)
    tau = 0.62
    ax.axhline(tau, color=ACCENT, lw=0.8, ls=(0, (3, 2)))
    ax.text(-5.8, tau + 0.04, r"$\tau^{\ast}$", color=ACCENT, fontsize=5.4 * FS, ha="left",
            va="bottom")
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([0, 1])
    limpa(ax)
    ax.set_xlabel("$z$", fontsize=5.0 * FS, color=INK2, labelpad=0)
    ax.set_ylabel("$q$", fontsize=5.0 * FS, color=INK2, labelpad=0, rotation=0)
    texto(fig, dx0 + dw / 2, 0.37, "Platt + F1-optimal $\\tau^{\\ast}$\non REFIT 3, 7, 20", 4.6,
          linespacing=1.05)

    # (8) output: on/off per 5-min interval, decisions vs true state
    seta(fig, dx0 + dw + 0.03, 0.78, dx0 + dw + 0.17, 0.78)
    ox = dx0 + dw + 0.21
    ow = W - 0.1 - ox
    ax = ax_in(fig, ox, 0.46, ow, 0.62)
    faixa_estado(ax, yw.astype(bool), INK2, 1.1)
    faixa_estado(ax, pw, ACCENT, 0.1)
    for y0 in (0.1, 1.1):                      # central interval, only across the strips
        ax.plot([23.5, 23.5], [y0, y0 + 0.62], color=SURFACE, lw=0.6)
        ax.plot([24.5, 24.5], [y0, y0 + 0.62], color=SURFACE, lw=0.6)
    ax.set_xlim(-0.5, 47.5)
    ax.set_ylim(0, 2.1)
    ax.set_xticks([])
    ax.set_yticks([])
    limpa(ax, ())
    ax.text(-0.5, 1.76, "true state", color=INK2, fontsize=4.8 * FS, va="bottom")
    ax.text(-0.5, 0.76, "CRAFT", color=ACCENT, fontsize=4.8 * FS, va="bottom",
            fontweight="bold")
    texto(fig, ox + ow / 2, 1.2, "On/off", 6.4, INK, True)
    texto(fig, ox + ow / 2, 0.3, "per 5-min\ninterval", 4.9, linespacing=1.05)

    verifica_vazamentos(fig)
    out = Path(__file__).with_name("pipeline_en")
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    fig.savefig(out.with_suffix(".png"), dpi=300, facecolor=SURFACE)
    print("ok")


if __name__ == "__main__":
    main()
