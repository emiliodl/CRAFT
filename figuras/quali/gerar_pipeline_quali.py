"""Figura da qualificação: o CRAFT de ponta a ponta, com dados reais, em português.

Três faixas, de cima para baixo:
  1  Treinamento: janela real da casa 8 da REFIT na grade de 30 s e um exemplo real de
     cada uma das cinco operações de recomposição (faixa A, só a casa 8, como no artigo
     de referência), produzidos pelo gerador de src/reco.py.
  2  Inferência: janela real de 4 h da casa 2 da ECO, os quatro canais de entrada, o
     conjunto, a decisão calibrada e as decisões do modelo da faixa A (semente 0) contra o
     estado real.
  3  Dentro do modelo: um membro ResNet 1D e o ramo de evidência.

Mesma linguagem visual das figuras do artigo (artigo/figuras/gerar_pipeline.py): tinta
neutra, caixas cinza sem borda e uma única cor de destaque, reservada ao aparelho-alvo e
à recomposição.

Uso: python figuras/quali/gerar_pipeline_quali.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "artigo" / "figuras"))
import gerar_recomposicao as rec  # noqa: E402
from decisao import Decisor  # noqa: E402
from final_craftE import conjunto, membros_da_semente, sig  # noqa: E402
from gerar_pipeline import janela_treino  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
AXIS, BOX, SURFACE = "#c3c2b7", "#f0efec", "#ffffff"
ACCENT, ACCENT_WASH = "#2a78d6", "#e6effb"
W, H = 6.3, 8.6                        # largura do texto em A4 com margens ABNT (16 cm)
FS = 1.0
PREF_TV = 133.0                        # potência mediana da TV ligada, casa 8 (W)


# ------------------------------------------------------------------------------------
# dados
# ------------------------------------------------------------------------------------

def janela_teste():
    """Janela real de 4 h da casa 2 da ECO (célula da TV no cenário S2), com o estado real
    e as decisões do CRAFT da faixa A (treino só na casa 8), regra d_L."""
    ms = membros_da_semente("television", ["craftF_A"], "craftE_A_ev", 0)
    y, t, _, z = conjunto(ms, "cross: eco casa 2")
    cc = [conjunto(ms, k) for k in sorted(k for k in ms[0] if k.startswith("cv:"))]
    dec = Decisor(sig(np.concatenate([c[3] for c in cc])), np.concatenate([c[0] for c in cc]))
    pred = dec.decidir(sig(z))["d"].astype(bool)
    d = np.load(RAIZ / "datasets/protocolo_artigo/eco_casa_2__television__5min__cross.npz")
    agg = dict(zip(d["t"].astype(np.int64), d["main"]))
    x = np.array([agg.get(int(v), np.nan) for v in t])
    cont = np.r_[False, np.diff(t) == 300]
    ini = np.flatnonzero((y[1:] == 1) & (y[:-1] == 0)) + 1
    for minimo in (0.9, 0.85, 0.8):
        melhor, escolha = -1.0, None
        for i in ini:
            a, b = i - 22, i + 26
            if a < 0 or b > len(t) or not cont[a + 1:b].all() or np.isnan(x[a:b]).any():
                continue
            if (pred[a:b] == y[a:b].astype(bool)).mean() < minimo:
                continue
            degrau = x[i:i + 6].mean() - x[i - 6:i].mean()
            faixa = x[a:b].max() - x[a:b].min()
            if not (80 < degrau < 400) or faixa > 700:
                continue
            ruido = x[a:i - 1].std() + np.abs(np.diff(x[i + 1:b])).max()
            if degrau / (ruido + 30) > melhor:
                melhor, escolha = degrau / (ruido + 30), (a, b)
        if escolha is not None:
            a, b = escolha
            print(f"janela de teste: acerto mínimo {minimo}")
            return x[a:b], y[a:b], pred[a:b], np.nanmedian(x)
    raise RuntimeError("nenhuma janela")


# ------------------------------------------------------------------------------------
# desenho (coordenadas da figura em polegadas)
# ------------------------------------------------------------------------------------

def ax_in(fig, x, y, w, h):
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.set_zorder(5)
    ax.patch.set_alpha(0)
    return ax


def limpa(ax, eixos=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in eixos)
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=6.6, width=0.6, length=2.2, pad=1.5)


def caixa(fig, x, y, w, h, fill=BOX, raio=0.07, z=0, borda=None):
    fig.patches.append(FancyBboxPatch((x / W, y / H), w / W, h / H, transform=fig.transFigure,
                                      boxstyle=f"round,pad=0,rounding_size={raio / W}",
                                      facecolor=fill, edgecolor=borda or "none",
                                      lw=0.8 if borda else 0, zorder=z))


def seta(fig, x0, y0, x1, y1, cor=MUTED, estilo="-", conn=None, lw=1.0, ms=8):
    fig.patches.append(FancyArrowPatch((x0 / W, y0 / H), (x1 / W, y1 / H),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=ms, lw=lw, color=cor, linestyle=estilo,
                                       shrinkA=0, shrinkB=0, connectionstyle=conn, zorder=6))


def linha(fig, xs, ys, cor=MUTED, estilo="-", lw=0.8):
    fig.lines.append(plt.Line2D([v / W for v in xs], [v / H for v in ys],
                                transform=fig.transFigure, color=cor, ls=estilo, lw=lw,
                                zorder=6))


def texto(fig, x, y, s, size=7.5, cor=INK2, bold=False, ha="center", va="center", **kw):
    kw.setdefault("zorder", 7)
    fig.text(x / W, y / H, s, ha=ha, va=va, fontsize=size * FS, color=cor,
             fontweight="bold" if bold else "normal", **kw)


def titulo_faixa(fig, y, num, s, sub):
    fig.patches.append(plt.Circle((0.19, y), 0.115, transform=fig.dpi_scale_trans,
                                  facecolor=INK, edgecolor="none", zorder=6))
    texto(fig, 0.19, y - 0.005, num, 8.5, SURFACE, True)
    texto(fig, 0.4, y, s, 10.5, INK, True, ha="left")
    texto(fig, W - 0.12, y, sub, 8.0, MUTED, ha="right")


def faixa_estado(ax, estado, cor, y0, h=0.62):
    ax.add_patch(Rectangle((-0.5, y0), len(estado), h, facecolor=BOX, edgecolor="none"))
    for i, v in enumerate(estado):
        if v:
            ax.add_patch(Rectangle((i - 0.5, y0), 1, h, facecolor=cor, edgecolor="none"))


def verifica_vazamentos(fig, folga=0.03):
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
        if x0 < 0.02 or x1 > W - 0.02 or y0 < 0.02 or y1 > H - 0.02:
            problemas += 1
            print(f"FORA DA FIGURA: {t.get_text()[:45]!r}")
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        dentro = [c for c in caixas if c[0] <= cx <= c[2] and c[1] <= cy <= c[3]]
        if not dentro:
            continue
        c = dentro[-1]
        if x0 < c[0] + folga or x1 > c[2] - folga or y0 < c[1] + folga or y1 > c[3] - folga:
            problemas += 1
            print(f"VAZA: {t.get_text()[:45]!r} x[{x0:.2f},{x1:.2f}] y[{y0:.2f},{y1:.2f}]"
                  f" caixa x[{c[0]:.2f},{c[2]:.2f}] y[{c[1]:.2f},{c[3]:.2f}]")
    print("vazamentos:", problemas)


# ------------------------------------------------------------------------------------

def faixa_treino(fig, ht, A30, P30, comp, h5):
    titulo_faixa(fig, 8.42, "1", "Treinamento com agregados recompostos", "casa fonte: REFIT 8")

    # janela real da casa fonte, grade de 30 s
    ax = ax_in(fig, 0.42, 6.62, 1.18, 1.1)
    ax.plot(ht, A30 / 1000, color=INK2, lw=0.6)
    ax.plot(ht, P30 / 1000, color=ACCENT, lw=1.1)
    ax.set_xlim(0, 4)
    ax.set_xticks([0, 2, 4])
    ax.set_xticklabels(["0", "2", "4 h"])
    limpa(ax)
    ax.set_ylabel("kW", fontsize=7, color=INK2, labelpad=1)
    ymax = ax.get_ylim()[1]
    ax.text(0.1, ymax * 0.99, "agregado", color=INK2, fontsize=7, ha="left", va="top")
    ax.text(4, P30.max() / 1000 + ymax * 0.05, "TV", color=ACCENT, fontsize=7.5,
            ha="right", va="bottom", fontweight="bold")
    texto(fig, 1.0, 6.32, "grade de 30 s", 7.5)
    seta(fig, 1.72, 7.17, 1.9, 7.17)

    # as cinco operações (faixa de destaque)
    px0, px1, py0, py1 = 1.98, 6.2, 6.06, 8.18
    caixa(fig, px0, py0, px1 - px0, py1 - py0, ACCENT_WASH, 0.09, z=-1)
    texto(fig, (px0 + px1) / 2, py1 - 0.17, "Recomposição do agregado", 9.5, INK, True)
    ops = [("R0", "real", "$A_h$"), ("R3", "troca de fundo", "$B_k + P_h$"),
           ("R1", "injeção", "$B_k + s\\,a_j$"), ("R2", "remoção", "$A_h - P_h$"),
           ("R4", "distrator", "$B_k + s\\,d_j$")]
    ymax = max(float((c["fundo"] + (c["alvo"] if c["alvo"] is not None else 0)
                      + (c["dist"] if c["dist"] is not None else 0)).max())
               for c in comp.values()) / 1000 * 1.12
    mw, mh, gap = 0.7, 0.62, 0.122
    x0 = px0 + 0.14
    my = 6.84
    for j, (op, nome, formula) in enumerate(ops):
        c = comp[op]
        ax = ax_in(fig, x0 + j * (mw + gap), my, mw, mh)
        b = c["fundo"] / 1000
        topo = b.copy()
        ax.fill_between(h5, 0, b, step="mid", color=SURFACE, lw=0)
        ax.fill_between(h5, 0, b, step="mid", color=AXIS, alpha=0.45, lw=0)
        if c["alvo"] is not None:
            topo = b + c["alvo"] / 1000
            ax.fill_between(h5, b, topo, step="mid", color=ACCENT, alpha=0.9, lw=0)
        if c["dist"] is not None:
            topo = b + c["dist"] / 1000
            ax.fill_between(h5, b, topo, step="mid", color=MUTED, alpha=0.7, lw=0,
                            hatch="////", edgecolor=SURFACE)
        ax.plot(h5, topo, color=INK, lw=0.8, drawstyle="steps-mid")
        if c["removido"] is not None:
            ax.plot(h5, b + c["removido"] / 1000, color=ACCENT, lw=0.9, ls=(0, (2.2, 1.4)),
                    drawstyle="steps-mid")
        ax.axvspan(-2.5 / 60, 2.5 / 60, color=INK2, alpha=0.18, lw=0, zorder=0)
        ax.set_xlim(h5[0], h5[-1])
        ax.set_ylim(0, ymax)
        ax.set_xticks([])
        ax.set_yticks([])
        limpa(ax, ("bottom",))
        cx = x0 + j * (mw + gap) + mw / 2
        texto(fig, cx, my + mh + 0.34, f"{op} · {nome}", 7.2, INK, True)
        texto(fig, cx, my + mh + 0.14, formula, 7.4)
        on = bool(c["on"])
        texto(fig, cx, my - 0.13, "ligado" if on else "desligado", 7.4,
              ACCENT if on else MUTED, bold=on)
    # legenda compacta
    ly = 6.37
    itens = [("fundo", AXIS, 0.45, None), ("TV", ACCENT, 0.9, None),
             ("distrator", MUTED, 0.7, "////")]
    lx = px0 + 0.2
    for nome, cor, alfa, hatch in itens:
        fig.patches.append(Rectangle(((lx) / W, (ly - 0.05) / H), 0.16 / W, 0.1 / H,
                                     transform=fig.transFigure, facecolor=cor, alpha=alfa,
                                     hatch=hatch, edgecolor=SURFACE if hatch else "none",
                                     lw=0, zorder=6))
        texto(fig, lx + 0.21, ly, nome, 7.0, ha="left")
        lx += 0.21 + 0.06 * len(nome) + 0.3
    linha(fig, [lx, lx + 0.2], [ly, ly], ACCENT, (0, (2.2, 1.4)), 0.9)
    texto(fig, lx + 0.25, ly, "TV removida", 7.0, ha="left")
    texto(fig, (px0 + px1) / 2, 6.17,
          "soma na grade de 30 s  →  média de 5 min  →  rótulo recalculado "
          "($\\bar{p}_t \\geq \\theta$)", 7.4)

    # treino do conjunto
    caixa(fig, 0.12, 5.6, W - 0.24, 0.36)
    texto(fig, W / 2, 5.78, "Treino:  5 ResNets com lotes reais + recompostos (BCE)   ·   "
          "ramo de evidência só com janelas reais", 7.6, INK)
    seta(fig, (px0 + px1) / 2, py0 - 0.01, (px0 + px1) / 2, 5.97)


def faixa_inferencia(fig, xw, yw, pw, sigma_h, h5):
    titulo_faixa(fig, 5.27, "2", "Inferência na casa de destino", "ECO casa 2 · medidor de 5 min")
    yb, ytop = 3.35, 4.95

    # janela de 4 h
    ax = ax_in(fig, 0.42, 3.72, 0.98, 1.0)
    ax.plot(h5, xw / 1000, color=INK2, lw=0.9, drawstyle="steps-mid")
    ax.axvspan(-2.5 / 60, 2.5 / 60, color=INK2, alpha=0.18, lw=0)
    ax.set_xlim(h5[0], h5[-1])
    ax.set_xticks([-2, 0, 2])
    ax.set_xticklabels(["−2", "0", "2 h"])
    limpa(ax)
    ax.set_ylabel("kW", fontsize=7, color=INK2, labelpad=1)
    texto(fig, 0.92, 3.4, "janela de 4 h", 7.5)
    seta(fig, 1.46, 4.22, 1.6, 4.22)

    # quatro canais
    dx = np.r_[0, np.diff(xw)]
    v1 = np.asinh(xw / sigma_h)
    canais = [((v1 - v1.mean()) / (v1.std() + 1e-9), "nível"),
              (np.clip(xw / PREF_TV, -10, 10), r"$x/P_{\mathrm{ref}}$"),
              (np.clip(np.maximum(dx, 0) / PREF_TV, 0, 10), "subidas"),
              (np.clip(np.maximum(-dx, 0) / PREF_TV, 0, 10), "descidas")]
    cx0, cw, ch = 2.12, 0.62, 0.21
    for k, (v, nome) in enumerate(canais):
        y0 = 4.48 - k * 0.27
        ax = ax_in(fig, cx0, y0, cw, ch)
        ax.plot(h5, v, color=ACCENT if k >= 2 else INK2, lw=0.7, drawstyle="steps-mid")
        ax.set_xlim(h5[0], h5[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        limpa(ax, ())
        texto(fig, cx0 - 0.05, y0 + ch / 2, nome, 7.0, ha="right")
    texto(fig, cx0 + cw / 2 - 0.2, 3.4, "4 canais de entrada", 7.5)
    seta(fig, 2.8, 4.22, 2.94, 4.22)

    # o modelo (detalhado na faixa 3)
    mx0, mx1 = 2.98, 4.18
    caixa(fig, mx0, yb + 0.12, mx1 - mx0, 1.38, SURFACE, 0.07, borda=INK2)
    texto(fig, (mx0 + mx1) / 2, 4.62, "CRAFT", 9.5, INK, True)
    texto(fig, (mx0 + mx1) / 2, 4.22, "5 ResNets 1D\n+ ramo de\nevidência", 7.4, linespacing=1.15)
    texto(fig, (mx0 + mx1) / 2, 3.72, "$z$ = média dos\n6 logits", 7.2, linespacing=1.1)
    seta(fig, mx1 + 0.04, 4.22, mx1 + 0.18, 4.22)

    # decisão calibrada
    dx0, dw = 4.3, 1.0
    ax = ax_in(fig, dx0 + 0.16, 3.82, dw - 0.2, 0.86)
    zz = np.linspace(-6, 6, 200)
    ax.plot(zz, 1 / (1 + np.exp(-zz / 1.6)), color=INK2, lw=1.0)
    tau = 0.62
    ax.axhline(tau, color=ACCENT, lw=0.9, ls=(0, (3, 2)))
    ax.text(-5.8, tau + 0.05, r"$\tau^{\ast}$", color=ACCENT, fontsize=8, ha="left", va="bottom")
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([0, 1])
    limpa(ax)
    ax.set_xlabel("$z$", fontsize=7.5, color=INK2, labelpad=0)
    ax.text(-6, 1.08, "$q$", fontsize=7.5, color=INK2, ha="center", va="bottom")
    texto(fig, dx0 + dw / 2 + 0.05, 3.4, "Platt + limiar $\\tau^{\\ast}$", 7.5)
    seta(fig, dx0 + dw + 0.04, 4.22, dx0 + dw + 0.18, 4.22)

    # saída
    ox = dx0 + dw + 0.24
    ow = W - 0.12 - ox
    ax = ax_in(fig, ox, 3.82, ow, 0.86)
    faixa_estado(ax, yw.astype(bool), INK2, 1.15)
    faixa_estado(ax, pw, ACCENT, 0.05)
    ax.set_xlim(-0.5, 47.5)
    ax.set_ylim(0, 2.3)
    ax.set_xticks([])
    ax.set_yticks([])
    limpa(ax, ())
    ax.text(-0.5, 1.82, "estado real", color=INK2, fontsize=7.0, va="bottom")
    ax.text(-0.5, 0.72, "CRAFT", color=ACCENT, fontsize=7.0, va="bottom", fontweight="bold")
    texto(fig, ox + ow / 2, 3.4, "ligado/desligado\na cada 5 min", 7.5, linespacing=1.1)
    return mx0, mx1, yb + 0.12


def faixa_modelo(fig, xw, mx0, mx1, ymod):
    titulo_faixa(fig, 2.95, "3", "Dentro do CRAFT", "um dos 5 membros ResNet e o ramo de evidência")

    # ---- membro ResNet 1D ----
    rx0, rx1, ry0, ry1 = 0.12, 3.62, 0.12, 2.62
    caixa(fig, rx0, ry0, rx1 - rx0, ry1 - ry0)
    texto(fig, (rx0 + rx1) / 2, ry1 - 0.2, "ResNet 1D  (×5, kernel $k$ = 5, 7, 9, 15, 25)",
          8.2, INK, True)
    # fluxo: entrada -> 3 blocos -> média global -> linear -> z_m
    yc = 1.62
    blocos = [("entrada", "4 × 48"), ("bloco 1", "32 canais"), ("bloco 2", "64 canais"),
              ("bloco 3", "64 canais"), ("média\nglobal", ""), ("linear", "")]
    bw, bh, bgap = 0.44, 0.5, 0.11
    bx = rx0 + 0.16
    xs = []
    for j, (nome, sub) in enumerate(blocos):
        destaque = 1 <= j <= 3
        caixa(fig, bx, yc - bh / 2, bw, bh, SURFACE if destaque else BOX, 0.05, z=1,
              borda=INK2 if destaque else AXIS)
        texto(fig, bx + bw / 2, yc + (0.06 if sub else 0), nome, 6.8, INK, destaque,
              linespacing=1.0)
        if sub:
            texto(fig, bx + bw / 2, yc - 0.13, sub, 6.0, MUTED)
        xs.append(bx)
        if j < len(blocos) - 1:
            seta(fig, bx + bw + 0.01, yc, bx + bw + bgap - 0.01, yc, lw=0.8, ms=6)
        bx += bw + bgap
    texto(fig, bx - bgap + 0.08, yc, "$z_m$", 8.5, INK, ha="left")
    # bloco residual expandido
    ex0, ex1, ey0, ey1 = rx0 + 0.3, rx1 - 0.3, ry0 + 0.06, 1.2
    linha(fig, [xs[1], ex0 + 0.05], [yc - bh / 2, ey1], AXIS, (0, (2, 2)), 0.7)
    linha(fig, [xs[3] + bw, ex1 - 0.05], [yc - bh / 2, ey1], AXIS, (0, (2, 2)), 0.7)
    caixa(fig, ex0, ey0, ex1 - ex0, ey1 - ey0, SURFACE, 0.05, z=1)
    texto(fig, ex0 + 0.1, ey1 - 0.14, "bloco residual", 6.8, INK, True, ha="left")
    convs = ["conv $k$", "conv 5", "conv 3"]
    cy = 0.7
    cbw = 0.62
    cx = ex0 + 0.32
    for j, c in enumerate(convs):
        caixa(fig, cx, cy - 0.15, cbw, 0.3, BOX, 0.04, z=2)
        texto(fig, cx + cbw / 2, cy, c, 6.8, INK)
        if j < 2:
            seta(fig, cx + cbw + 0.01, cy, cx + cbw + 0.13, cy, lw=0.7, ms=5)
        cx += cbw + 0.14
    # soma e atalho
    sx = cx + 0.05
    texto(fig, sx + 0.06, cy, "+", 10, INK, True)
    seta(fig, cx - 0.13, cy, sx - 0.02, cy, lw=0.7, ms=5)
    linha(fig, [ex0 + 0.18, ex0 + 0.18, sx + 0.06, sx + 0.06], [cy, cy - 0.3, cy - 0.3, cy - 0.12],
          INK2, "-", 0.7)
    seta(fig, ex0 + 0.12, cy, ex0 + 0.31, cy, lw=0.7, ms=5)
    texto(fig, (ex0 + sx) / 2 + 0.05, cy - 0.42, "atalho (conv 1×1)", 6.4, MUTED)

    # ---- ramo de evidência ----
    vx0, vx1, vy0, vy1 = 3.76, W - 0.12, 0.12, 2.62
    caixa(fig, vx0, vy0, vx1 - vx0, vy1 - vy0)
    texto(fig, (vx0 + vx1) / 2, vy1 - 0.2, "Ramo de evidência", 8.2, INK, True)
    dx = np.r_[0, np.diff(xw)]
    up = np.maximum(dx, 0) / PREF_TV
    dn = np.maximum(-dx, 0) / PREF_TV
    lim = max(up.max(), dn.max()) * 1.1
    h5 = (np.arange(48) - 24) * 5 / 60
    for k, (v, nome, cor) in enumerate([(up, "$a^{+}$: subidas", ACCENT),
                                        (dn, "$a^{-}$: descidas", MUTED)]):
        y0 = 1.76 - k * 0.66
        ax = ax_in(fig, vx0 + 0.18, y0, 1.1, 0.34)
        ax.bar(h5, v, width=0.07, color=cor, lw=0)
        ax.axhline(0, color=AXIS, lw=0.6)
        ax.set_xlim(h5[0], h5[-1])
        ax.set_ylim(0, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        limpa(ax, ())
        texto(fig, vx0 + 0.18, y0 + 0.44, nome, 6.8, cor if k == 0 else INK2, ha="left")
    texto(fig, vx0 + 0.73, 0.68, "detectores restritos em sinal:\numa queda nunca vira \"ligou\"",
          6.4, MUTED, linespacing=1.1)
    # acumulador e saída
    ax0 = vx0 + 1.42
    seta(fig, vx0 + 1.3, 1.62, ax0 + 0.02, 1.62, lw=0.8, ms=6)
    linha(fig, [vx0 + 1.3, vx0 + 1.3], [1.27, 1.93], AXIS, "-", 0.7)
    caixa(fig, ax0 + 0.04, 1.27, 0.86, 0.7, SURFACE, 0.05, z=1, borda=INK2)
    texto(fig, ax0 + 0.47, 1.77, "acumulador", 6.8, INK, True)
    texto(fig, ax0 + 0.47, 1.5, "$s_t$: para frente\n$b_t$: para trás", 6.4, INK2,
          linespacing=1.15)
    seta(fig, ax0 + 0.47, 1.26, ax0 + 0.47, 0.98, lw=0.8, ms=6)
    texto(fig, ax0 + 0.47, 0.84, "$z_E$", 8.5, INK)
    texto(fig, ax0 + 0.47, 0.5, "6º membro\ndo conjunto", 6.4, MUTED, linespacing=1.1)


def main() -> None:
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
                         "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "dejavusans",
                         "hatch.linewidth": 0.5})
    ht, A30, P30 = janela_treino()
    rec.CASAS = (8,)                       # faixa A: só a casa 8, como no artigo de referência
    comp = rec.componentes()
    xw, yw, pw, sigma_h = janela_teste()
    h5 = (np.arange(48) - 24) * 5 / 60
    fig = plt.figure(figsize=(W, H))
    faixa_treino(fig, ht, A30, P30, comp, h5)
    mx0, mx1, ymod = faixa_inferencia(fig, xw, yw, pw, sigma_h, h5)
    seta(fig, (mx0 + mx1) / 2, 5.6, (mx0 + mx1) / 2, 4.98, cor=ACCENT, estilo=(0, (3, 2)))
    texto(fig, (mx0 + mx1) / 2 + 0.08, 5.42, "pesos treinados", 7.2, ACCENT, ha="left")
    faixa_modelo(fig, xw, mx0, mx1, ymod)
    verifica_vazamentos(fig)
    out = Path(__file__).with_name("pipeline_quali")
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    fig.savefig(out.with_suffix(".png"), dpi=250, facecolor=SURFACE)
    print("ok", out)


if __name__ == "__main__":
    main()
