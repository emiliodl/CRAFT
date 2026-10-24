"""Graficos da semana extraida do UK-DALE (casa 2, 6 s): agregada e geladeira.

Gera em figuras/:
  ukdale_casa2_semana_6s.png   semana inteira, agregada e geladeira em paineis separados
  ukdale_casa2_dia_6s.png      um dia (21/09/2013)
  ukdale_casa2_3h_6s.png       3 h, agregada e geladeira no mesmo eixo (mesma unidade)

Horario em UTC, como no CSV. O eixo da geladeira nos dois primeiros e cortado em 250 W
para os ciclos (~90 W) ficarem visiveis; os picos de partida (ate 1826 W) sao indicados.

Uso:
    python scripts/graficos_semana_ukdale.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CSV = RAIZ / "datasets" / "ukdale_casa2_semana_20130918_6s.csv"
FIG = RAIZ / "figuras"
AZUL, LARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"
TETO_GEL = 250.0

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA2,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRADE,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 200,
})


def paineis(w: pd.DataFrame, titulo: str, fmt: str, loc, saida: Path, lw: float) -> None:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True,
                                 gridspec_kw={"height_ratios": [1.4, 1]})
    a1.plot(w.index, w.aggregate_W, color=AZUL, lw=lw)
    a1.set_ylabel("Agregada (VA)")
    a1.set_title(titulo, loc="left", color=TINTA, fontsize=10)
    a2.plot(w.index, w.fridge_W.clip(upper=TETO_GEL), color=LARANJA, lw=lw)
    a2.set_ylim(0, TETO_GEL * 1.04)
    a2.set_ylabel("Geladeira (W)")
    picos = w.fridge_W[w.fridge_W > TETO_GEL]
    if len(picos):
        a2.plot(picos.index, [TETO_GEL] * len(picos), "v", ms=3, color=TINTA2, ls="none")
        a2.text(1.0, 1.02, f"▼ {len(picos)} picos de partida acima de {TETO_GEL:.0f} W "
                f"(max. {picos.max():.0f} W), cortados", transform=a2.transAxes,
                ha="right", va="bottom", fontsize=8, color=TINTA2)
    a2.xaxis.set_major_locator(loc)
    a2.xaxis.set_major_formatter(mdates.DateFormatter(fmt, tz="UTC"))
    a2.set_xlabel("Horario (UTC)")
    a2.set_xlim(w.index[0], w.index[-1])
    fig.tight_layout()
    fig.savefig(saida)
    plt.close(fig)
    print(saida)


def sobreposto(w: pd.DataFrame, titulo: str, saida: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(w.index, w.aggregate_W, color=AZUL, lw=1.0, label="Agregada (VA)")
    ax.plot(w.index, w.fridge_W, color=LARANJA, lw=1.0, label="Geladeira (W)")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Potencia")
    ax.set_title(titulo, loc="left", color=TINTA, fontsize=10)
    ax.legend(loc="upper left", frameon=False, labelcolor=TINTA2)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30], tz="UTC"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz="UTC"))
    ax.set_xlabel("Horario (UTC), 21/09/2013")
    ax.set_xlim(w.index[0], w.index[-1])
    fig.tight_layout()
    fig.savefig(saida)
    plt.close(fig)
    print(saida)


def main() -> None:
    w = pd.read_csv(CSV, index_col=0, parse_dates=True)
    paineis(w, "UK-DALE casa 2, 18 a 24/09/2013, amostras de 6 s", "%d/%m",
            mdates.DayLocator(tz="UTC"), FIG / "ukdale_casa2_semana_6s.png", lw=0.35)
    dia = w.loc["2013-09-21"]
    paineis(dia, "UK-DALE casa 2, 21/09/2013, amostras de 6 s", "%H:%M",
            mdates.HourLocator(byhour=range(0, 24, 3), tz="UTC"),
            FIG / "ukdale_casa2_dia_6s.png", lw=0.6)
    tres = w.loc["2013-09-21 17:00":"2013-09-21 19:59:59"]
    sobreposto(tres, "UK-DALE casa 2, 21/09/2013 17h-20h UTC, amostras de 6 s",
               FIG / "ukdale_casa2_3h_6s.png")


if __name__ == "__main__":
    main()
