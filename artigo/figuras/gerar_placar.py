"""Figura do placar (artigo): F1 do CRAFT x melhor metodo do artigo de referencia.

So a familia primaria pre-registrada: cross-dataset (REFIT -> ECO, faixa A+) e
intra-dataset (mes seguinte da casa 8, faixa A). Haltere: losango aberto = artigo,
circulo = CRAFT (media de 10 sementes) com o IC95% em traco fino; a coluna da
direita da a diferenca, em cinza quando o teste pre-registrado deu empate.
"""
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
TINTA, TINTA2, APAGADO = "#0b0b0b", "#52514e", "#898781"
EIXO, GRADE, FUNDO = "#c3c2b7", "#ecebe7", "#ffffff"
AZUL, AZUL_IC = "#2a78d6", "#a9c8ee"

d = pd.read_csv(RAIZ / "resultados/v3/final_craftE__celulas.csv")
# nos CSVs o modelo final se chama "CRAFT+E"; no artigo, CRAFT
d = d[(d.modelo == "CRAFT+E") & (d.familia == "primaria")]
NOME = {"television": "TV", "washing machine": "Máq. de lavar", "freezer": "Freezer"}
GRUPOS = [("Entre conjuntos (REFIT → ECO)", "cross"), ("Mesma casa (REFIT 8)", "intra")]

plt.rcParams.update({"font.family": "serif", "font.size": 8.2, "pdf.fonttype": 42,
                     "ps.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.3, 2.25))
y, rotulos = 0.0, []
for titulo, cel in GRUPOS:
    rotulos.append((y, titulo, True))
    y -= 0.95
    for ap in ("television", "washing machine", "freezer"):
        r = d[(d.celula == cel) & (d.aparelho == ap)].iloc[0]
        lo, hi = ast.literal_eval(r.ic95)
        ax.plot([r.artigo, r.f1], [y, y], color=EIXO, lw=1.4, solid_capstyle="round", zorder=1)
        ax.plot([lo, hi], [y, y], color=AZUL_IC, lw=3.2, solid_capstyle="round", zorder=2)
        ax.plot(r.artigo, y, "D", ms=4.4, mfc=FUNDO, mec=TINTA2, mew=1.0, zorder=3)
        ax.plot(r.f1, y, "o", ms=5.4, color=AZUL, mec=FUNDO, mew=0.8, zorder=4)
        dif = r.f1 - r.artigo
        txt = f"{dif:+.2f}".replace("-", "−")
        ax.text(1.03, y, txt, transform=ax.get_yaxis_transform(), ha="left", va="center",
                color=TINTA if r.supera_artigo else APAGADO,
                fontweight="bold" if r.supera_artigo else "normal")
        rotulos.append((y, NOME[ap], False))
        y -= 1
    y -= 0.25
for yy, t, titulo in rotulos:
    ax.text(-0.03 if not titulo else -0.33, yy, t, transform=ax.get_yaxis_transform(),
            ha="right" if not titulo else "left", va="center",
            color=TINTA if titulo else TINTA2, fontweight="bold" if titulo else "normal")
ax.text(1.03, 0.0, "Δ F1", transform=ax.get_yaxis_transform(), ha="left", va="center",
        color=APAGADO)
ax.set_xlim(0, 1)
ax.set_ylim(y + 0.55, 0.55)
ax.set_yticks([])
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel("F1 da classe ligado", color=TINTA2, labelpad=2)
ax.grid(axis="x", color=GRADE, lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(EIXO)
ax.spines["bottom"].set_linewidth(0.5)
ax.tick_params(colors=APAGADO, labelcolor=TINTA2, width=0.5, length=2)
h = [Line2D([], [], color=AZUL, marker="o", lw=0, ms=5.4, mec=FUNDO, label="CRAFT"),
     Line2D([], [], color=AZUL_IC, lw=3.2, solid_capstyle="round", label="IC95%"),
     Line2D([], [], color=TINTA2, marker="D", lw=0, ms=4.4, mfc=FUNDO, mew=1.0,
            label="Melhor método da ref.")]
ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3, frameon=False,
          handletextpad=0.4, columnspacing=1.1, handlelength=1.4)
fig.savefig(Path(__file__).with_name("placar.pdf"), bbox_inches="tight", facecolor=FUNDO)
fig.savefig(Path(__file__).with_name("placar.png"), bbox_inches="tight", dpi=300,
            facecolor=FUNDO)
print("ok")
