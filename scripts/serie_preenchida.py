"""A mesma serie, preenchida em vez de tracada - para comparar com as imagens.

Tres colunas por janela:

  1  area preenchida com eixos: agregado em bloco solido, aparelho por cima
  2  silhueta pura do agregado, sem eixos - e o que um modelo de visao veria se
     a gente simplesmente rasterizasse o grafico, sem GASF nem nada
  3  silhueta do aparelho (a verdade), para contraste

As janelas sao sorteadas com a mesma regra e a mesma semente de
imagens_janela.py, entao com `--semente 0` saem as MESMAS janelas da figura das
transformadas, e as duas figuras podem ser postas lado a lado.

Uso:
    python scripts/serie_preenchida.py
    python scripts/serie_preenchida.py --fonte eco_casa_1 --n 3 --largura 288
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imagens_janela import PASTA, sortear                      # noqa: E402

FIGURAS = Path(__file__).resolve().parents[1] / "figuras"

AGREGADO = "#1F3F6E"      # azul escuro
APARELHO = "#E8890C"      # laranja
LIGADO = "#F7D774"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fonte", default="refit_casa_8")
    p.add_argument("--aparelho", default="washing_machine")
    p.add_argument("--taxa", default="5min")
    p.add_argument("--particao", default="treino")
    p.add_argument("--largura", type=int, default=96)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--semente", type=int, default=0)
    args = p.parse_args()

    arq = PASTA / f"{args.fonte}__{args.aparelho}__{args.taxa}__{args.particao}.npz"
    if not arq.exists():
        raise SystemExit(f"nao encontrei {arq.name}")
    d = np.load(arq)
    t, principal = d["t"], d["main"]
    pot, st = d[args.aparelho], d[f"status_{args.aparelho}"]
    taxa_s = 900 if "15min" in args.taxa else 300

    rng = np.random.default_rng(args.semente)
    centros = sortear(t, st, args.largura, args.n, rng, taxa_s)
    if not centros:
        raise SystemExit("nenhuma janela com ativacao")

    meia = args.largura // 2
    fig, eixos = plt.subplots(len(centros), 3, figsize=(15, 2.9 * len(centros)),
                              gridspec_kw=dict(width_ratios=[2.4, 1, 1], wspace=0.18,
                                               hspace=0.45))
    eixos = np.atleast_2d(eixos)

    for linha, c in enumerate(centros):
        fatia = slice(c - meia, c - meia + args.largura)
        x, y, s = principal[fatia].astype(float), pot[fatia].astype(float), st[fatia]
        hora = np.arange(len(x)) * taxa_s / 3600
        inicio = datetime.fromtimestamp(int(t[fatia][0]), tz=timezone.utc).replace(tzinfo=None)

        a = eixos[linha, 0]
        bordas = np.diff(np.concatenate(([0], s, [0])).astype(int))
        for i, f in zip(np.flatnonzero(bordas == 1), np.flatnonzero(bordas == -1)):
            a.axvspan(hora[i], hora[min(f, len(hora) - 1)], color=LIGADO, alpha=0.45, lw=0,
                      zorder=0)
        a.fill_between(hora, 0, x, color=AGREGADO, lw=0, zorder=2, label="agregado")
        a.fill_between(hora, 0, y, color=APARELHO, lw=0, zorder=3, alpha=0.95,
                       label=args.aparelho.replace("_", " "))
        a.set_xlim(hora[0], hora[-1])
        a.set_ylim(0, None)
        a.set_ylabel("W")
        a.set_xlabel("horas")
        a.set_title(f"{inicio:%Y-%m-%d %H:%M}  ·  pico {x.max():.0f} W  ·  "
                    f"{int(s.sum())} janelas ligado", fontsize=10, loc="left",
                    fontweight="bold")
        a.legend(loc="upper right", frameon=False, fontsize=8.5)
        for lado in ("top", "right"):
            a.spines[lado].set_visible(False)

        for col, (serie, cor, nome) in enumerate(
                [(x, "#111111", "silhueta do agregado"),
                 (y, APARELHO, "silhueta do aparelho (verdade)")], start=1):
            e = eixos[linha, col]
            e.fill_between(hora, 0, serie, color=cor, lw=0)
            e.set_xlim(hora[0], hora[-1])
            e.set_ylim(0, max(float(x.max()), 1.0))      # mesma escala nas duas
            e.set_xticks([]); e.set_yticks([])
            e.set_title(nome, fontsize=9.5, fontweight="bold")
            for lado in e.spines.values():
                lado.set_visible(False)

    fig.suptitle(f"{args.fonte} · {args.aparelho.replace('_', ' ')} · {args.taxa} · "
                 f"janelas de {args.largura} amostras "
                 f"({args.largura * taxa_s / 3600:.0f} h)  —  serie preenchida",
                 fontsize=13, y=0.995)
    FIGURAS.mkdir(exist_ok=True)
    saida = FIGURAS / f"preenchida_{args.fonte}_{args.aparelho}_{args.taxa}_{args.largura}.png"
    fig.savefig(saida, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"-> {saida}")


if __name__ == "__main__":
    main()
