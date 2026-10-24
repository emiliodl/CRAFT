"""A mesma janela vista no tempo e na frequencia.

Tres colunas por janela:

  1  TEMPO       area preenchida: agregado em bloco solido, aparelho por cima,
                 e o periodo ligado sombreado ao fundo
  2  FREQUENCIA  espectro de amplitude contra PERIODO (nao contra indice de
                 raia), em log-log, para agregado e aparelho separados. A faixa
                 que `src/dominio.py::randomizar_fourier` perturba fica
                 sombreada, e a banda tipica de um ciclo de lavagem fica marcada
  3  ENERGIA     fracao acumulada da energia abaixo de cada periodo, para os
                 dois sinais - quantifica o que a coluna 2 mostra de forma
                 qualitativa

Para que serve: a randomizacao de Fourier parte da premissa de que a baixa
frequencia (nivel da casa + envoltoria diurna) e especifica do dominio e a banda
media carrega a assinatura do aparelho. Esta figura e o teste visual dessa
premissa - se as duas curvas da coluna 3 se separarem em torno do corte, a
premissa se sustenta; se subirem juntas, o corte esta apagando sinal.

As janelas usam a mesma regra e a mesma semente de imagens_janela.py, entao com
`--semente 0` saem as MESMAS janelas das outras figuras e tudo pode ser posto
lado a lado.

Uso:
    python scripts/tempo_frequencia.py
    python scripts/tempo_frequencia.py --largura 288 --n 3
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
CORTE = "#C7CDD4"         # faixa randomizada
TINTA = "#2B3037"

CORTE_RAIAS = 8           # o mesmo padrao de randomizar_fourier


def espectro(x: np.ndarray, taxa_s: int) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude por raia e o periodo (em horas) de cada raia, sem a DC."""
    n = len(x)
    amp = np.abs(np.fft.rfft(x - x.mean())) / n
    k = np.arange(len(amp))
    with np.errstate(divide="ignore"):
        periodo = np.where(k > 0, n * taxa_s / 3600 / np.maximum(k, 1), np.inf)
    return periodo[1:], amp[1:]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fonte", default="refit_casa_8")
    p.add_argument("--aparelho", default="washing_machine")
    p.add_argument("--taxa", default="5min")
    p.add_argument("--particao", default="treino")
    p.add_argument("--largura", type=int, default=288)
    p.add_argument("--n", type=int, default=3)
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

    nome_ap = args.aparelho.replace("_", " ")
    meia = args.largura // 2
    # periodo correspondente ao corte: raia CORTE_RAIAS de uma janela de L amostras
    periodo_corte = args.largura * taxa_s / 3600 / CORTE_RAIAS

    fig, eixos = plt.subplots(len(centros), 3, figsize=(16, 3.3 * len(centros)),
                              gridspec_kw=dict(width_ratios=[1.5, 1.15, 1.0],
                                               wspace=0.26, hspace=0.55))
    eixos = np.atleast_2d(eixos)

    for linha, c in enumerate(centros):
        fatia = slice(c - meia, c - meia + args.largura)
        x = principal[fatia].astype(float)
        y = pot[fatia].astype(float)
        s = st[fatia]
        hora = np.arange(len(x)) * taxa_s / 3600
        inicio = datetime.fromtimestamp(int(t[fatia][0]), tz=timezone.utc).replace(tzinfo=None)

        # ---------------------------------------------------------------- tempo
        a = eixos[linha, 0]
        bordas = np.diff(np.concatenate(([0], s, [0])).astype(int))
        for i, f in zip(np.flatnonzero(bordas == 1), np.flatnonzero(bordas == -1)):
            a.axvspan(hora[i], hora[min(f, len(hora) - 1)], color=LIGADO, alpha=0.45,
                      lw=0, zorder=0)
        a.fill_between(hora, 0, x, color=AGREGADO, lw=0, zorder=2, label="agregado")
        a.fill_between(hora, 0, y, color=APARELHO, lw=0, alpha=0.95, zorder=3, label=nome_ap)
        a.set_xlim(hora[0], hora[-1])
        a.set_ylim(0, None)
        a.set_xlabel("horas")
        a.set_ylabel("W")
        a.set_title(f"tempo  ·  {inicio:%Y-%m-%d %H:%M}  ·  pico {x.max():.0f} W  ·  "
                    f"{int(s.sum())} amostras ligado",
                    fontsize=10, loc="left", fontweight="bold", color=TINTA)
        a.legend(loc="upper right", frameon=False, fontsize=8.5)

        # ----------------------------------------------------------- frequencia
        b = eixos[linha, 1]
        per_x, amp_x = espectro(x, taxa_s)
        per_y, amp_y = espectro(y, taxa_s)
        b.axvspan(periodo_corte, per_x.max() * 1.6, color=CORTE, alpha=0.55, lw=0, zorder=0)
        b.text(0.02, 0.04, f"randomizado (periodo $\\geq$ {periodo_corte:.0f} h)",
               transform=b.transAxes, fontsize=8, ha="left", va="bottom",
               color="#5A626B", zorder=5)
        b.axvspan(1.0, 2.0, color=APARELHO, alpha=0.16, lw=0, zorder=0)
        b.text(1.41, 0.97, "ciclo de lavagem", transform=b.get_xaxis_transform(),
               fontsize=8, ha="center", va="top", color="#9A5B05", zorder=5)
        b.loglog(per_x, amp_x, color=AGREGADO, lw=2, label="agregado", zorder=3)
        b.loglog(per_y, amp_y, color=APARELHO, lw=2, label=nome_ap, zorder=4)
        b.set_xlim(per_x.max() * 1.6, per_x.min())        # periodo longo a esquerda
        b.set_xlabel("periodo (horas)  —  baixa frequencia a esquerda")
        b.set_ylabel("amplitude (W)")
        b.set_title("frequencia  ·  espectro de amplitude", fontsize=10, loc="left",
                    fontweight="bold", color=TINTA)
        b.legend(loc="lower left", frameon=False, fontsize=8.5,
                 bbox_to_anchor=(0.0, 0.10))

        # -------------------------------------------------------------- energia
        e = eixos[linha, 2]
        for per, amp, cor, nome in ((per_x, amp_x, AGREGADO, "agregado"),
                                    (per_y, amp_y, APARELHO, nome_ap)):
            ordem = np.argsort(per)                       # do periodo curto ao longo
            pot_ord = amp[ordem] ** 2
            acum = np.cumsum(pot_ord[::-1])[::-1]         # energia em periodos >= p
            total = acum.max() if acum.max() > 0 else 1.0
            e.semilogx(per[ordem], acum / total, color=cor, lw=2, label=nome)
        e.axvline(periodo_corte, color="#5A626B", lw=1.4, ls="--", zorder=5)
        e.text(periodo_corte * 1.06, 0.03, f"corte {periodo_corte:.0f} h", fontsize=8,
               rotation=90, va="bottom", color="#5A626B")
        e.set_xlim(per_x.max() * 1.6, per_x.min())
        e.set_ylim(0, 1.02)
        e.set_xlabel("periodo (horas)")
        e.set_ylabel("fracao da energia acima do periodo")
        e.set_title("energia acumulada", fontsize=10, loc="left", fontweight="bold",
                    color=TINTA)
        e.legend(loc="upper left", frameon=False, fontsize=8.5)

        for ax in eixos[linha]:
            ax.grid(True, which="major", color="#E6E9ED", lw=0.7, zorder=1)
            for lado in ("top", "right"):
                ax.spines[lado].set_visible(False)
            ax.set_axisbelow(True)

    fig.suptitle(f"{args.fonte} · {nome_ap} · {args.taxa} · janelas de {args.largura} "
                 f"amostras ({args.largura * taxa_s / 3600:.0f} h)  —  tempo vs frequencia",
                 fontsize=13, y=0.997, color=TINTA)
    FIGURAS.mkdir(exist_ok=True)
    saida = (FIGURAS / f"tempo_freq_{args.fonte}_{args.aparelho}_{args.taxa}_"
                       f"{args.largura}.png")
    fig.savefig(saida, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"-> {saida}")


if __name__ == "__main__":
    main()
