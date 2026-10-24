"""Sete transformacoes do agregado na MESMA janela, para comparar vistas.

Complementa vistas_janela.py (bruto / 1a diff / 2a diff / desvio padrao movel)
com transformacoes que preservam a DURACAO do evento, e nao so as bordas:

  base removida    x - mediana movel longa: tira a carga de base, mantem o pulso
  top-hat          x - abertura morfologica: so o que e estreito e positivo
  multi-escala     x[t] - x[t-k] para k = 1, 3, 12 (5 min, 15 min, 1 h)
  z robusto        (x - mediana movel) / (1.4826 * MAD movel), adimensional
  CUSUM bilateral  soma acumulada com reset, detector de mudanca de patamar
  patamares        ajuste constante por partes (segmentacao binaria)
  arcsinh          compressao de escala: geladeira e lava-loucas no mesmo eixo

Usa o mesmo sorteio de vistas_janela.py, entao a mesma semente da a mesma janela.

Uso:
    python scripts/transformacoes_janela.py --semente 3
    python scripts/transformacoes_janela.py --serie refit__fridge__casa_9.npz
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import grey_opening, median_filter

from vistas_janela import FIGURAS, PASTA, marcar_ligado, sortear_janela

COR, COR2, COR3 = "#2C5EA8", "#B35C00", "#3F8F5B"


def mediana_movel(x: np.ndarray, k: int) -> np.ndarray:
    return median_filter(x, size=k, mode="nearest")


def mad_movel(x: np.ndarray, k: int) -> np.ndarray:
    med = mediana_movel(x, k)
    return mediana_movel(np.abs(x - med), k)


def cusum_bilateral(x: np.ndarray, folga: float) -> tuple[np.ndarray, np.ndarray]:
    """Somas acumuladas positiva e negativa com reset em zero (Page)."""
    d = np.diff(x, prepend=x[0])
    s_mais = np.zeros_like(d)
    s_menos = np.zeros_like(d)
    a = b = 0.0
    for i, v in enumerate(d):
        a = max(0.0, a + v - folga)
        b = min(0.0, b + v + folga)
        s_mais[i], s_menos[i] = a, b
    return s_mais, s_menos


def patamares(x: np.ndarray, penalidade: float, minimo: int = 2) -> np.ndarray:
    """Ajuste constante por partes por segmentacao binaria."""
    def custo(a: int, b: int) -> float:
        return float(((x[a:b] - x[a:b].mean()) ** 2).sum())

    cortes: list[int] = []
    fila = [(0, len(x))]
    while fila:
        a, b = fila.pop()
        if b - a < 2 * minimo:
            continue
        base = custo(a, b)
        melhor, ganho = -1, penalidade
        for c in range(a + minimo, b - minimo + 1):
            g = base - custo(a, c) - custo(c, b)
            if g > ganho:
                melhor, ganho = c, g
        if melhor > 0:
            cortes.append(melhor)
            fila += [(a, melhor), (melhor, b)]

    limites = np.array([0, *sorted(cortes), len(x)])
    y = np.empty_like(x)
    for a, b in zip(limites[:-1], limites[1:]):
        y[a:b] = x[a:b].mean()
    return y


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--serie", default=None)
    p.add_argument("--janela", type=int, default=288)
    p.add_argument("--semente", type=int, default=3)
    p.add_argument("--base", type=int, default=24, help="mediana movel da base, em janelas (24 = 2 h)")
    p.add_argument("--saida", default=None)
    args = p.parse_args()

    rng = np.random.default_rng(args.semente)
    arquivos = sorted(PASTA.glob("*.npz"))
    if args.serie:
        arquivos = [PASTA / args.serie]
    else:
        rng.shuffle(arquivos)
    for arq in arquivos:
        r = sortear_janela(arq, args.janela, rng)
        if r is not None:
            break
    else:
        raise SystemExit("nenhuma janela com ativacao encontrada")

    t, x, pot, status, aparelho = r
    x = x.astype(np.float64)
    hora = np.arange(len(t)) * 5 / 60.0
    inicio = datetime.fromtimestamp(int(t[0]), tz=timezone.utc).replace(tzinfo=None)
    dataset, _, casa = arq.stem.split("__")

    k = args.base
    base = mediana_movel(x, k)
    sem_base = x - base
    tophat = x - grey_opening(x, size=k // 2, mode="nearest")
    mad = mad_movel(x, k)
    z = (x - base) / (1.4826 * np.maximum(mad, 1.0))
    ruido = float(np.median(np.abs(np.diff(x))))
    s_mais, s_menos = cusum_bilateral(x, folga=2.0 * ruido)
    passos = patamares(x, penalidade=20.0 * max(ruido, 1.0) ** 2)
    comprimido = np.arcsinh(x / 50.0)

    fig, eixos = plt.subplots(8, 1, figsize=(12, 18), sharex=True,
                              gridspec_kw=dict(hspace=0.22))

    a = eixos[0]
    a.plot(hora, x, color=COR, lw=1.3, label="agregado")
    a.plot(hora, base, color=COR2, lw=1.4, ls="--", label=f"mediana movel ({k * 5} min)")
    a.plot(hora, pot, color="#999999", lw=1.0, alpha=0.8, label=f"{aparelho} (referencia)")
    a.set_ylabel("W")
    a.set_title("0. bruto, com a carga de base estimada", loc="left", fontsize=11, fontweight="bold")
    a.legend(loc="upper right", frameon=False, fontsize=8.5, ncol=3)

    a = eixos[1]
    a.plot(hora, sem_base, color=COR, lw=1.3)
    a.axhline(0, color="#999999", lw=0.8, zorder=0)
    a.set_ylabel("W")
    a.set_title("1. base removida: agregado - mediana movel  (mantem o PATAMAR do evento, "
                "nao so as bordas)", loc="left", fontsize=11, fontweight="bold")

    a = eixos[2]
    a.plot(hora, tophat, color=COR, lw=1.3)
    a.axhline(0, color="#999999", lw=0.8, zorder=0)
    a.set_ylabel("W")
    a.set_title(f"2. top-hat morfologico: agregado - abertura ({k // 2 * 5} min)  "
                "(so pulsos positivos mais curtos que o elemento)",
                loc="left", fontsize=11, fontweight="bold")

    a = eixos[3]
    for lag, cor, rot in ((1, COR, "5 min"), (3, COR2, "15 min"), (12, COR3, "1 h")):
        a.plot(hora, x - np.concatenate((np.full(lag, x[0]), x[:-lag])),
               color=cor, lw=1.1, alpha=0.9, label=f"lag {rot}")
    a.axhline(0, color="#999999", lw=0.8, zorder=0)
    a.set_ylabel("delta W")
    a.set_title("3. diferencas multi-escala: x[t] - x[t-k]  (lag longo enxerga o ciclo inteiro)",
                loc="left", fontsize=11, fontweight="bold")
    a.legend(loc="upper right", frameon=False, fontsize=8.5, ncol=3)

    a = eixos[4]
    a.plot(hora, z, color=COR, lw=1.3)
    for nivel in (3, -3):
        a.axhline(nivel, color=COR2, lw=0.9, ls=":")
    a.set_ylabel("z")
    a.set_title("4. z robusto: (x - mediana) / (1,4826 x MAD)  "
                "(ADIMENSIONAL - o mesmo limiar serve em casas diferentes)",
                loc="left", fontsize=11, fontweight="bold")

    a = eixos[5]
    a.plot(hora, s_mais, color=COR, lw=1.3, label="CUSUM+")
    a.plot(hora, s_menos, color=COR2, lw=1.3, label="CUSUM-")
    a.axhline(0, color="#999999", lw=0.8, zorder=0)
    a.set_ylabel("W acumulados")
    a.set_title(f"5. CUSUM bilateral com reset (folga = {2 * ruido:.0f} W)  "
                "(acumula so enquanto a mudanca persiste)",
                loc="left", fontsize=11, fontweight="bold")
    a.legend(loc="upper right", frameon=False, fontsize=8.5, ncol=2)

    a = eixos[6]
    a.plot(hora, x, color="#BBBBBB", lw=1.0, zorder=0)
    a.plot(hora, passos, color=COR, lw=1.6)
    n_passos = int((np.diff(passos) != 0).sum())
    a.set_ylabel("W")
    a.set_title(f"6. patamares: ajuste constante por partes ({n_passos} degraus)  "
                "(a 5 min o agregado E quase constante por partes)",
                loc="left", fontsize=11, fontweight="bold")

    a = eixos[7]
    a.plot(hora, comprimido, color=COR, lw=1.3)
    a.set_ylabel("arcsinh(W/50)")
    a.set_title("7. compressao arcsinh  (geladeira de 50 W e lava-loucas de 2 kW "
                "no mesmo eixo, sem estourar o zero)",
                loc="left", fontsize=11, fontweight="bold")
    a.set_xlabel(f"horas desde {inicio:%Y-%m-%d %H:%M}")

    for a in eixos:
        marcar_ligado(a, hora, status)
        a.grid(alpha=0.18, lw=0.6)
        a.margins(x=0.01)
        for lado in ("top", "right"):
            a.spines[lado].set_visible(False)

    fig.suptitle(f"{dataset.upper()} - {casa.replace('_', ' ')} - {aparelho} - "
                 f"{len(t)} janelas de 5 min  (faixa amarela = aparelho ligado)",
                 fontsize=13, y=0.905)

    FIGURAS.mkdir(exist_ok=True)
    saida = Path(args.saida) if args.saida else FIGURAS / f"transformacoes_{arq.stem}.png"
    fig.savefig(saida, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"serie  : {arq.name}")
    print(f"janela : {inicio:%Y-%m-%d %H:%M} + {len(t) * 5 / 60:.1f} h "
          f"({int(status.sum())} janelas ligado)")
    print(f"ruido  : mediana de |diff| = {ruido:.1f} W")
    print(f"figura : {saida}")


if __name__ == "__main__":
    main()
