"""Figura da qualificação (v2): o CRAFT como um fluxo linear, com o modelo no centro.

Seis etapas numeradas da esquerda para a direita, à maneira das figuras de pipeline da
literatura (o conjunto de ResNets em ramos paralelos, como no CamAL; a cadeia janela ->
camadas -> saída, como no Seq2Point):

  1 Dados        casa fonte (REFIT 8, grade de 30 s) e casa de destino (ECO 2, 5 min)
  2 Recomposição só no caminho de treino: um exemplo real de cada operação
  3 Entrada      os 4 canais; os caminhos de treino e de inferência se juntam aqui
  4 CRAFT        5 ResNets 1D (kernel k) e o ramo de evidência em paralelo -> média
  5 Decisão      Platt + limiar tau*, calibrados em REFIT 3, 7 e 20
  6 Saída        ligado/desligado a cada 5 min, contra o estado real

Dados reais em todos os painéis (faixa A: treino só na casa 8). Formato paisagem, para
uma página girada.

Uso: python figuras/quali/gerar_pipeline_quali_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "artigo" / "figuras"))
sys.path.insert(0, str(AQUI))
import gerar_recomposicao as rec  # noqa: E402
from gerar_pipeline import janela_treino  # noqa: E402
from gerar_pipeline_quali import janela_teste  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
AXIS, BOX, SURFACE = "#c3c2b7", "#f0efec", "#ffffff"
ACCENT, ACCENT_WASH = "#2a78d6", "#e6effb"
W, H = 9.7, 6.2                        # área útil de uma página A4 girada (ABNT)
PREF_TV = 133.0
FONTE = 1.15                           # fator de escala de todos os textos


# ------------------------------------------------------------------------------------
# primitivas (polegadas)
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
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=6.6 * FONTE, width=0.6, length=2.2, pad=1.5)


def caixa(fig, x, y, w, h, fill=BOX, raio=0.07, z=0, borda=None, lw=0.8):
    fig.patches.append(FancyBboxPatch((x / W, y / H), w / W, h / H, transform=fig.transFigure,
                                      boxstyle=f"round,pad=0,rounding_size={raio / W}",
                                      facecolor=fill, edgecolor=borda or "none",
                                      lw=lw if borda else 0, zorder=z))


def seta(fig, x0, y0, x1, y1, cor=MUTED, estilo="-", conn=None, lw=1.0, ms=8):
    fig.patches.append(FancyArrowPatch((x0 / W, y0 / H), (x1 / W, y1 / H),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=ms, lw=lw, color=cor, linestyle=estilo,
                                       shrinkA=0, shrinkB=0, connectionstyle=conn, zorder=6))


def linha(fig, xs, ys, cor=MUTED, estilo="-", lw=0.8, z=6):
    fig.lines.append(plt.Line2D([v / W for v in xs], [v / H for v in ys],
                                transform=fig.transFigure, color=cor, ls=estilo, lw=lw,
                                zorder=z))


def texto(fig, x, y, s, size=7.5, cor=INK2, bold=False, ha="center", va="center", **kw):
    kw.setdefault("zorder", 7)
    fig.text(x / W, y / H, s, ha=ha, va=va, fontsize=size * FONTE, color=cor,
             fontweight="bold" if bold else "normal", **kw)


def etapa(fig, x, num, nome, y=5.98):
    fig.patches.append(Circle((x - 0.02, y), 0.105, transform=fig.dpi_scale_trans,
                              facecolor=INK, edgecolor="none", zorder=6))
    texto(fig, x - 0.02, y - 0.004, num, 8, SURFACE, True)
    texto(fig, x + 0.14, y, nome, 9.5, INK, True, ha="left")


def faixa_estado(ax, estado, cor, y0, h=0.62):
    ax.add_patch(Rectangle((-0.5, y0), len(estado), h, facecolor=BOX, edgecolor="none"))
    for i, v in enumerate(estado):
        if v:
            ax.add_patch(Rectangle((i - 0.5, y0), 1, h, facecolor=cor, edgecolor="none"))


def verifica_vazamentos(fig, folga=0.03):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dpi = fig.dpi
    caixas = [tuple(v / dpi for v in (b.x0, b.y0, b.x1, b.y1))
              for p in fig.patches if isinstance(p, FancyBboxPatch)
              for b in [p.get_window_extent(r)]]
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
        if dentro:
            c = dentro[-1]
            if x0 < c[0] + folga or x1 > c[2] - folga or y0 < c[1] + folga or y1 > c[3] - folga:
                problemas += 1
                print(f"VAZA: {t.get_text()[:45]!r} x[{x0:.2f},{x1:.2f}] y[{y0:.2f},{y1:.2f}]"
                      f" caixa x[{c[0]:.2f},{c[2]:.2f}] y[{c[1]:.2f},{c[3]:.2f}]")
    print("vazamentos:", problemas)


# ------------------------------------------------------------------------------------

def main() -> None:
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
                         "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "dejavusans",
                         "hatch.linewidth": 0.5})
    ht, A30, P30 = janela_treino()
    rec.CASAS = (8,)
    comp = rec.componentes()
    xw, yw, pw, sigma_h = janela_teste()
    h5 = (np.arange(48) - 24) * 5 / 60
    fig = plt.figure(figsize=(W, H))

    # faixas de treino (em cima) e inferência (embaixo), atrás de tudo
    yT0, yT1 = 3.78, 5.62
    yI0, yI1 = 0.3, 2.14
    for (y0, y1), nome in (((yT0, yT1), "TREINO"), ((yI0, yI1), "INFERÊNCIA")):
        caixa(fig, 0.06, y0, W - 0.12, y1 - y0, "#f7f6f3", 0.1, z=-3)
        texto(fig, 0.2, (y0 + y1) / 2, nome, 7.5, MUTED, True, rotation=90)

    # colunas
    mx0, mx1 = 4.78, 7.82                       # CRAFT
    dx0 = 8.02                                  # decisão
    ox = 8.98                                   # saída
    for x, num, nome in ((0.42, "1", "Dados"), (2.0, "2", "Recomposição"),
                         (3.72, "3", "Entrada"), ((mx0 + mx1) / 2 - 0.42, "4", "CRAFT"),
                         (dx0, "5", "Decisão"), (ox - 0.02, "6", "Saída")):
        etapa(fig, x, num, nome)

    # ============================ 1 Dados ============================
    ax = ax_in(fig, 0.68, 4.25, 1.02, 1.05)
    ax.plot(ht, A30 / 1000, color=INK2, lw=0.6)
    ax.plot(ht, P30 / 1000, color=ACCENT, lw=1.1)
    ax.set_xlim(0, 4)
    ax.set_xticks([0, 2, 4])
    ax.set_xticklabels(["0", "2", "4 h"])
    limpa(ax)
    ax.set_ylabel("kW", fontsize=7 * FONTE, color=INK2, labelpad=1)
    ym = ax.get_ylim()[1]
    ax.text(0.1, ym * 0.99, "agregado", color=INK2, fontsize=7 * FONTE, ha="left", va="top")
    ax.text(4, P30.max() / 1000 + ym * 0.05, "TV", color=ACCENT, fontsize=7.5 * FONTE, ha="right",
            va="bottom", fontweight="bold")
    texto(fig, 1.17, 3.98, "casa fonte · REFIT 8 · 30 s", 6.8)

    ax = ax_in(fig, 0.68, 0.8, 1.02, 1.05)
    ax.plot(h5, xw / 1000, color=INK2, lw=0.9, drawstyle="steps-mid")
    ax.axvspan(-2.5 / 60, 2.5 / 60, color=INK2, alpha=0.18, lw=0)
    ax.set_xlim(h5[0], h5[-1])
    ax.set_xticks([-2, 0, 2])
    ax.set_xticklabels(["−2", "0", "2 h"])
    limpa(ax)
    ax.set_ylabel("kW", fontsize=7 * FONTE, color=INK2, labelpad=1)
    texto(fig, 1.17, 0.5, "casa de destino · ECO 2 · 5 min", 6.8)

    # ============================ 2 Recomposição ============================
    rx0, rx1 = 1.95, 3.5
    caixa(fig, rx0, yT0 + 0.08, rx1 - rx0, yT1 - yT0 - 0.16, ACCENT_WASH, 0.08, z=-1)
    seta(fig, 1.76, 4.78, rx0 - 0.04, 4.78)
    ops = [("R0", "real"), ("R3", "troca de fundo"), ("R1", "injeção"),
           ("R2", "remoção"), ("R4", "distrator")]
    ymax = max(float((c["fundo"] + (c["alvo"] if c["alvo"] is not None else 0)
                      + (c["dist"] if c["dist"] is not None else 0)).max())
               for c in comp.values()) / 1000 * 1.1
    ph, pg = 0.25, 0.08
    for j, (op, nome) in enumerate(ops):
        c = comp[op]
        y0 = yT1 - 0.42 - j * (ph + pg)
        texto(fig, rx0 + 0.09, y0 + ph / 2, op, 7.0, INK, True, ha="left")
        texto(fig, rx0 + 0.34, y0 + ph / 2, nome, 6.5, INK2, ha="left")
        ax = ax_in(fig, rx0 + 1.06, y0, 0.34, ph)
        b = c["fundo"] / 1000
        topo = b.copy()
        ax.fill_between(h5, 0, b, step="mid", color=SURFACE, lw=0)
        ax.fill_between(h5, 0, b, step="mid", color=AXIS, alpha=0.5, lw=0)
        if c["alvo"] is not None:
            topo = b + c["alvo"] / 1000
            ax.fill_between(h5, b, topo, step="mid", color=ACCENT, alpha=0.9, lw=0)
        if c["dist"] is not None:
            topo = b + c["dist"] / 1000
            ax.fill_between(h5, b, topo, step="mid", color=MUTED, alpha=0.7, lw=0,
                            hatch="////", edgecolor=SURFACE)
        ax.plot(h5, topo, color=INK, lw=0.6, drawstyle="steps-mid")
        if c["removido"] is not None:
            ax.plot(h5, b + c["removido"] / 1000, color=ACCENT, lw=0.7, ls=(0, (2, 1.2)),
                    drawstyle="steps-mid")
        ax.set_xlim(h5[0], h5[-1])
        ax.set_ylim(0, ymax)
        ax.set_xticks([])
        ax.set_yticks([])
        limpa(ax, ("bottom",))
        on = bool(c["on"])
        fig.patches.append(Circle((rx1 - 0.06, y0 + ph / 2), 0.035,
                                  transform=fig.dpi_scale_trans, zorder=6,
                                  facecolor=ACCENT if on else SURFACE,
                                  edgecolor=ACCENT if on else MUTED, lw=0.8))
    for xx, on, nome in ((rx0 + 0.42, True, "ligado"), (rx0 + 0.92, False, "desligado")):
        fig.patches.append(Circle((xx, 3.66), 0.035, transform=fig.dpi_scale_trans, zorder=6,
                                  facecolor=ACCENT if on else SURFACE,
                                  edgecolor=ACCENT if on else MUTED, lw=0.8))
        texto(fig, xx + 0.07, 3.66, nome, 6.4, MUTED, ha="left")
    texto(fig, (rx0 + rx1) / 2, 3.47, "rótulo recalculado após a média de 5 min", 6.4, MUTED)

    # caminho de inferência: sem recomposição
    linha(fig, [1.76, 3.45], [1.32, 1.32], MUTED, "-", 1.0)
    texto(fig, 2.6, 1.45, "janela real, sem recomposição", 6.5, MUTED)

    # ============================ 3 Entrada ============================
    ex0 = 3.7
    dx = np.r_[0, np.diff(xw)]
    v1 = np.asinh(xw / sigma_h)
    canais = [((v1 - v1.mean()) / (v1.std() + 1e-9), "nível"),
              (np.clip(xw / PREF_TV, -10, 10), r"$x/P_{\mathrm{ref}}$"),
              (np.clip(np.maximum(dx, 0) / PREF_TV, 0, 10), "subidas"),
              (np.clip(np.maximum(-dx, 0) / PREF_TV, 0, 10), "descidas")]
    for k, (v, nome) in enumerate(canais):
        y0 = 3.27 - k * 0.27
        ax = ax_in(fig, ex0 + 0.42, y0, 0.48, 0.2)
        ax.plot(h5, v, color=ACCENT if k >= 2 else INK2, lw=0.7, drawstyle="steps-mid")
        ax.set_xlim(h5[0], h5[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        limpa(ax, ())
        texto(fig, ex0 + 0.38, y0 + 0.1, nome, 6.5, ha="right")
    ent_y = 2.96
    seta(fig, rx1 + 0.02, 4.3, ex0 + 0.66, 3.52, conn="angle3,angleA=0,angleB=-90")
    seta(fig, 3.45, 1.32, ex0 + 0.66, 2.4, conn="angle3,angleA=0,angleB=90")
    seta(fig, ex0 + 0.94, ent_y, mx0 - 0.03, ent_y)

    # ============================ 4 CRAFT ============================
    my0, my1 = 0.42, 5.5
    caixa(fig, mx0, my0, mx1 - mx0, my1 - my0, SURFACE, 0.1, z=-1, borda=INK2, lw=1.1)
    bus = mx0 + 0.16
    node_x, node_y = mx1 - 0.32, 3.75
    ks = (5, 7, 9, 15, 25)
    row_y = [4.98 - j * 0.42 for j in range(5)]
    texto(fig, bus + 1.3, 5.3, "5 ResNets 1D, kernels diferentes", 7.8, INK, True)
    for k, yy in zip(ks, row_y):
        linha(fig, [bus, bus + 0.1], [yy, yy], MUTED, "-", 0.8)
        texto(fig, bus + 0.33, yy, f"$k$={k}", 6.8, INK)
        x = bus + 0.6
        larg = 0.06 + 0.0085 * k
        for b in range(3):
            caixa(fig, x, yy - 0.12, larg, 0.24, ACCENT_WASH if b == 0 else BOX, 0.03, z=1,
                  borda=ACCENT if b == 0 else AXIS, lw=0.6)
            x += larg + 0.05
        caixa(fig, x + 0.01, yy - 0.09, 0.27, 0.18, BOX, 0.03, z=1, borda=AXIS, lw=0.6)
        texto(fig, x + 0.145, yy, "GAP", 5.8, INK2)
        linha(fig, [x + 0.29, node_x - 0.12], [yy, node_y], AXIS, "-", 0.7)
    # ramo de evidência
    ev_y = 2.42
    texto(fig, bus + 1.3, 2.95, "ramo de evidência", 7.8, INK, True)
    linha(fig, [bus, bus + 0.1], [ev_y, ev_y], MUTED, "-", 0.8)
    ax = ax_in(fig, bus + 0.14, ev_y - 0.28, 0.82, 0.56)
    up = np.maximum(dx, 0) / PREF_TV
    dn = -np.maximum(-dx, 0) / PREF_TV
    lim = max(up.max(), -dn.min()) * 1.1
    ax.bar(h5, up, width=0.07, color=ACCENT, lw=0)
    ax.bar(h5, dn, width=0.07, color=MUTED, lw=0)
    ax.axhline(0, color=AXIS, lw=0.5)
    ax.set_xlim(h5[0], h5[-1])
    ax.set_ylim(-lim, lim)
    ax.set_xticks([])
    ax.set_yticks([])
    limpa(ax, ())
    ax.text(h5[0], lim * 0.95, "$a^{+}$", color=ACCENT, fontsize=7 * FONTE, ha="left", va="top")
    ax.text(h5[0], -lim * 0.95, "$a^{-}$", color=INK2, fontsize=7 * FONTE, ha="left", va="bottom")
    seta(fig, bus + 1.0, ev_y, bus + 1.14, ev_y, lw=0.7, ms=5)
    caixa(fig, bus + 1.16, ev_y - 0.2, 0.7, 0.4, BOX, 0.04, z=1, borda=AXIS, lw=0.6)
    texto(fig, bus + 1.51, ev_y + 0.07, "acumulador", 6.4, INK)
    texto(fig, bus + 1.51, ev_y - 0.09, "$s_t$ e $b_t$", 6.4, INK2)
    linha(fig, [bus + 1.88, node_x - 0.12], [ev_y, node_y], AXIS, "-", 0.7)
    texto(fig, bus + 1.05, 1.98, "detectores restritos em sinal", 6.3, MUTED)
    linha(fig, [bus, bus], [ev_y, row_y[0]], MUTED, "-", 0.8)
    # nó de média
    fig.patches.append(Circle((node_x, node_y), 0.12, transform=fig.dpi_scale_trans,
                              facecolor=INK, edgecolor="none", zorder=7))
    texto(fig, node_x, node_y + 0.005, "μ", 9, SURFACE, True, zorder=8)
    texto(fig, node_x, node_y - 0.3, "média dos\n6 logits", 6.3, INK2, linespacing=1.1)
    # bloco residual (detalhe de um bloco)
    by0, by1 = 0.56, 1.62
    caixa(fig, mx0 + 0.16, by0, mx1 - mx0 - 0.32, by1 - by0, BOX, 0.05, z=0)
    texto(fig, mx0 + 0.3, by1 - 0.15, "cada bloco residual", 6.8, INK, True, ha="left")
    cx, cy = mx0 + 0.46, 1.08
    for j, c in enumerate(["conv $k$", "conv 5", "conv 3"]):
        caixa(fig, cx, cy - 0.13, 0.52, 0.26, SURFACE, 0.03, z=1,
              borda=ACCENT if j == 0 else AXIS, lw=0.6)
        texto(fig, cx + 0.26, cy, c, 6.6, INK)
        if j < 2:
            seta(fig, cx + 0.53, cy, cx + 0.67, cy, lw=0.6, ms=5)
        cx += 0.68
    seta(fig, cx - 0.15, cy, cx + 0.03, cy, lw=0.6, ms=5)
    texto(fig, cx + 0.11, cy, "+", 9, INK, True)
    linha(fig, [mx0 + 0.33, mx0 + 0.33, cx + 0.11, cx + 0.11], [cy, cy - 0.3, cy - 0.3, cy - 0.11],
          INK2, "-", 0.6)
    seta(fig, mx0 + 0.3, cy, mx0 + 0.45, cy, lw=0.6, ms=5)
    texto(fig, (mx0 + cx) / 2 + 0.12, cy - 0.38, "atalho 1×1", 6.2, MUTED)

    # saída do modelo
    seta(fig, node_x + 0.13, node_y, mx1 + 0.12, node_y)
    texto(fig, mx1 + 0.05, node_y + 0.13, "$z$", 8, INK)

    # ============================ treino: perda e atualização ============================
    lx0, lx1 = 8.1, W - 0.16
    caixa(fig, lx0, 4.22, lx1 - lx0, 0.82, BOX, 0.06)
    texto(fig, (lx0 + lx1) / 2, 4.82, "perda BCE", 7.8, INK, True)
    texto(fig, (lx0 + lx1) / 2, 4.5, "com o rótulo\nrecalculado", 6.5, linespacing=1.1)
    seta(fig, mx1 + 0.12, node_y, lx0 + 0.25, 4.2, conn="angle3,angleA=0,angleB=90")
    seta(fig, (lx0 + lx1) / 2, 5.06, mx1 - 0.3, 5.42, cor=ACCENT, estilo=(0, (3, 2)),
         conn="angle3,angleA=90,angleB=0")
    texto(fig, (lx0 + lx1) / 2 + 0.07, 5.27, "atualiza\nos pesos", 6.6, ACCENT, ha="left",
          linespacing=1.05)

    # ============================ 5 Decisão ============================
    ax = ax_in(fig, dx0 + 0.14, 0.98, 0.7, 0.8)
    zz = np.linspace(-6, 6, 200)
    ax.plot(zz, 1 / (1 + np.exp(-zz / 1.6)), color=INK2, lw=1.0)
    tau = 0.62
    ax.axhline(tau, color=ACCENT, lw=0.9, ls=(0, (3, 2)))
    ax.text(-5.8, tau + 0.05, r"$\tau^{\ast}$", color=ACCENT, fontsize=8 * FONTE, ha="left", va="bottom")
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([0, 1])
    limpa(ax)
    ax.set_xlabel("$z$", fontsize=7.5 * FONTE, color=INK2, labelpad=0)
    ax.text(-6, 1.06, "$q$", fontsize=7.5 * FONTE, color=INK2, ha="center", va="bottom")
    seta(fig, mx1 + 0.12, node_y, dx0 + 0.45, 1.98, conn="angle3,angleA=0,angleB=-90")
    texto(fig, dx0 + 0.48, 0.55, "Platt + $\\tau^{\\ast}$,\nem REFIT 3, 7, 20", 6.4,
          linespacing=1.1)

    # ============================ 6 Saída ============================
    seta(fig, dx0 + 0.9, 1.38, ox - 0.03, 1.38)
    ax = ax_in(fig, ox, 0.98, W - 0.16 - ox, 0.82)
    faixa_estado(ax, yw.astype(bool), INK2, 1.15)
    faixa_estado(ax, pw, ACCENT, 0.05)
    ax.set_xlim(-0.5, 47.5)
    ax.set_ylim(0, 2.35)
    ax.set_xticks([])
    ax.set_yticks([])
    limpa(ax, ())
    ax.text(-0.5, 1.84, "real", color=INK2, fontsize=6.8 * FONTE, va="bottom")
    ax.text(-0.5, 0.74, "CRAFT", color=ACCENT, fontsize=6.8 * FONTE, va="bottom", fontweight="bold")
    texto(fig, ox + (W - 0.16 - ox) / 2, 0.55, "estado a cada\n5 min", 6.4,
          linespacing=1.1)

    verifica_vazamentos(fig)
    out = AQUI / "pipeline_quali"
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    fig.savefig(out.with_suffix(".png"), dpi=220, facecolor=SURFACE)
    print("ok", out)


if __name__ == "__main__":
    main()
