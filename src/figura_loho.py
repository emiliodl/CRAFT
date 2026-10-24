"""Figura do antes/depois: 1 casa de origem contra 18, e a dispersao entre folds."""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RES, FIG = RAIZ / "resultados", RAIZ / "figuras"
APS = ["fridge", "dish washer", "washing machine"]
ROTULO = {"fridge": "geladeira", "dish washer": "lava-loucas",
          "washing machine": "maquina de lavar"}
BRACOS = ["piso", "bruto", "3v_antigo", "3v_novo", "4v_novo"]
CORES = {"piso": "#9AA0A6", "bruto": "#2C5EA8", "3v_antigo": "#3F8F5B",
         "3v_novo": "#B35C00", "4v_novo": "#7B4FA8"}


def carregar():
    loho = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / "loho_*.csv"))
                      if "alvo_casa2" not in f])
    casa2 = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / "loho_alvo_casa2*.csv"))])
    antigo = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / "ablacao_*.csv"))])
    piso = pd.read_csv(RES / "piso_corrigido.csv")
    antigo = pd.concat([antigo[antigo.braco != "piso"], piso])
    return loho, casa2, antigo


def main() -> None:
    loho, casa2, antigo = carregar()
    fig, eixos = plt.subplots(2, 3, figsize=(15, 9),
                              gridspec_kw=dict(height_ratios=[1, 1.35], hspace=0.38))

    # --- linha 1: mesmo alvo (uk_dale casa 2), 1 casa de origem contra 18 ------------
    for a, ap in zip(eixos[0], APS):
        ant = antigo[antigo.aparelho == ap].groupby("braco").f1.mean()
        nov = casa2[casa2.aparelho == ap].set_index("braco").f1
        pares = [("piso", ant.get("piso"), nov.get("piso")),
                 ("bruto", ant.get("bruto"), nov.get("bruto")),
                 ("3 vistas", ant.get("3v_porta"), nov.get("3v_antigo"))]
        x = np.arange(len(pares))
        a.bar(x - 0.2, [p[1] for p in pares], 0.38, color="#C8CDD4", label="1 casa de origem")
        a.bar(x + 0.2, [p[2] for p in pares], 0.38, color="#2C5EA8", label="18 casas (LOHO)")
        for i, (_, v0, v1) in enumerate(pares):
            for dx, v in ((-0.2, v0), (0.2, v1)):
                if v == v:
                    a.text(i + dx, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5)
        a.set_xticks(x, [p[0] for p in pares], fontsize=9.5)
        a.set_ylim(0, 1.0)
        a.set_title(ROTULO[ap], fontsize=11.5, fontweight="bold")
        if ap == APS[0]:
            a.set_ylabel("F1 na UK-DALE casa 2")
            a.legend(frameon=False, fontsize=8.5, loc="upper left")
        a.grid(axis="y", alpha=0.2, lw=0.6)
        for lado in ("top", "right"):
            a.spines[lado].set_visible(False)

    # --- linha 2: leave-one-house-out, media e dispersao entre folds ----------------
    for a, ap in zip(eixos[1], APS):
        b = loho[loho.aparelho == ap]
        for i, braco in enumerate(BRACOS):
            v = b[b.braco == braco].f1.to_numpy()
            if not len(v):
                continue
            a.bar(i, v.mean(), 0.62, color=CORES[braco], alpha=0.85)
            a.scatter(np.full(len(v), i) + np.random.uniform(-0.16, 0.16, len(v)), v,
                      s=22, color="#222222", zorder=3, alpha=0.75, lw=0)
            a.text(i, v.mean() + 0.02, f"{v.mean():.3f}", ha="center", fontsize=9,
                   fontweight="bold")
        a.set_xticks(range(len(BRACOS)), BRACOS, fontsize=8.5, rotation=20)
        a.set_ylim(0, 0.9)
        a.set_title(f"{ROTULO[ap]} — 8 folds", fontsize=11, fontweight="bold")
        if ap == APS[0]:
            a.set_ylabel("F1 (cada ponto = uma casa-alvo)")
        a.grid(axis="y", alpha=0.2, lw=0.6)
        for lado in ("top", "right"):
            a.spines[lado].set_visible(False)

    fig.suptitle("Leave-one-house-out: treinar em ~18 casas contra treinar em uma  ·  "
                 "um modelo por aparelho", fontsize=13, y=0.975)
    FIG.mkdir(exist_ok=True)
    saida = FIG / "loho_antes_depois.png"
    fig.savefig(saida, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"-> {saida}")


if __name__ == "__main__":
    main()
