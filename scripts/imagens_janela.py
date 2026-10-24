"""Transformadas imagem-de-serie de janelas com ativacao da maquina de lavar.

Gera, para cada janela sorteada, o agregado bruto e quatro representacoes 2D:

  GASF  Gramian Angular Summation Field, cos(phi_i + phi_j)
  GADF  Gramian Angular Difference Field, sin(phi_i - phi_j)
  MTF   Markov Transition Field
  RP    matriz de distancias (recurrence plot sem limiar)

As tres primeiras vem de Wang & Oates (AAAI-15 workshop); a quarta e o grafico
de recorrencia classico, sem binarizar.

Um detalhe que vale notar: GASF, GADF e MTF reescalam cada janela pelo proprio
min-max ou pelos proprios quantis. Isso as torna INVARIANTES A ESCALA DA CASA
por construcao - exatamente o problema que derrubou a transferencia REFIT ->
ECO, onde o agregado suico e ~2,4x menor. O preco e que a amplitude absoluta,
que e o que distingue uma maquina de lavar de 2 kW, e descartada junto.

Uso:
    python scripts/imagens_janela.py
    python scripts/imagens_janela.py --fonte eco_casa_1 --taxa 5min --n 4
    python scripts/imagens_janela.py --largura 64 --semente 7
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
PASTA = RAIZ / "datasets" / "protocolo_artigo"
FIGURAS = RAIZ / "figuras"


# --------------------------------------------------------------------------------------
# Transformadas
# --------------------------------------------------------------------------------------


def _reescalar(x: np.ndarray) -> np.ndarray:
    """Para [-1, 1] pelo min-max da propria janela, como no artigo original."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return ((x - hi) + (x - lo)) / (hi - lo)


def gaf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(GASF, GADF). phi = arccos(x~); GASF=cos(phi_i+phi_j), GADF=sin(phi_i-phi_j)."""
    v = np.clip(_reescalar(x), -1.0, 1.0)
    raiz = np.sqrt(np.clip(1.0 - v**2, 0.0, None))
    gasf = np.outer(v, v) - np.outer(raiz, raiz)
    gadf = np.outer(raiz, v) - np.outer(v, raiz)
    return gasf, gadf


def mtf(x: np.ndarray, faixas: int = 8) -> np.ndarray:
    """Markov Transition Field: MTF[i,j] = P(faixa de x_j | faixa de x_i)."""
    n = len(x)
    cortes = np.quantile(x, np.linspace(0, 1, faixas + 1)[1:-1])
    q = np.searchsorted(cortes, x, side="right")
    w = np.zeros((faixas, faixas))
    np.add.at(w, (q[:-1], q[1:]), 1)
    soma = w.sum(axis=1, keepdims=True)
    w = np.divide(w, soma, out=np.zeros_like(w), where=soma > 0)
    return w[np.ix_(q, q)].reshape(n, n)


def recorrencia(x: np.ndarray) -> np.ndarray:
    """Matriz de distancias |x_i - x_j|, sem binarizar."""
    return np.abs(x[:, None] - x[None, :])


# --------------------------------------------------------------------------------------
# Selecao de janelas
# --------------------------------------------------------------------------------------


def corridas(t: np.ndarray, taxa_s: int) -> list[tuple[int, int]]:
    quebras = np.flatnonzero(np.diff(t) != taxa_s) + 1
    limites = np.concatenate(([0], quebras, [len(t)]))
    return list(zip(limites[:-1].tolist(), limites[1:].tolist()))


def sortear(t, status, largura, n, rng, taxa_s):
    """Janelas centradas numa borda de subida, sem cruzar buraco no tempo."""
    meia = largura // 2
    candidatos = []
    for a, b in corridas(t, taxa_s):
        if b - a < largura:
            continue
        subidas = np.flatnonzero(np.diff(status[a:b]) == 1) + 1 + a
        candidatos += [int(s) for s in subidas if a + meia <= s <= b - (largura - meia)]
    if not candidatos:
        return []
    rng.shuffle(candidatos)
    escolhidos, usados = [], []
    for c in candidatos:                       # evita janelas quase iguais
        if all(abs(c - u) >= largura for u in usados):
            escolhidos.append(c)
            usados.append(c)
        if len(escolhidos) == n:
            break
    return escolhidos


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fonte", default="refit_casa_8")
    p.add_argument("--taxa", default="5min")
    p.add_argument("--particao", default="treino")
    p.add_argument("--largura", type=int, default=96, help="amostras da janela (96 = 8 h a 5 min)")
    p.add_argument("--n", type=int, default=4, help="quantas janelas")
    p.add_argument("--semente", type=int, default=0)
    args = p.parse_args()

    arq = PASTA / f"{args.fonte}__washing_machine__{args.taxa}__{args.particao}.npz"
    if not arq.exists():
        raise SystemExit(f"nao encontrei {arq.name}\ndisponiveis: "
                         + ", ".join(sorted(a.name for a in PASTA.glob('*washing*'))))
    d = np.load(arq)
    t, principal, pot, st = d["t"], d["main"], d["washing_machine"], d["status_washing_machine"]
    taxa_s = 900 if "15min" in args.taxa else 300

    rng = np.random.default_rng(args.semente)
    centros = sortear(t, st, args.largura, args.n, rng, taxa_s)
    if not centros:
        raise SystemExit("nenhuma janela com ativacao")

    meia = args.largura // 2
    mapas = [("GASF", "coolwarm", None), ("GADF", "coolwarm", None),
             ("MTF", "viridis", None), ("RP  |xi-xj|", "magma", None)]
    fig, eixos = plt.subplots(len(centros), 5, figsize=(17.5, 3.5 * len(centros)),
                              gridspec_kw=dict(width_ratios=[1.9, 1, 1, 1, 1], wspace=0.22,
                                               hspace=0.35))
    eixos = np.atleast_2d(eixos)

    for linha, c in enumerate(centros):
        fatia = slice(c - meia, c - meia + args.largura)
        x, y, s = principal[fatia].astype(float), pot[fatia].astype(float), st[fatia]
        hora = np.arange(len(x)) * taxa_s / 3600
        inicio = datetime.fromtimestamp(int(t[fatia][0]), tz=timezone.utc).replace(tzinfo=None)

        a = eixos[linha, 0]
        bordas = np.diff(np.concatenate(([0], s, [0])).astype(int))
        for i, f in zip(np.flatnonzero(bordas == 1), np.flatnonzero(bordas == -1)):
            a.axvspan(hora[i], hora[min(f, len(hora) - 1)], color="#F0B429", alpha=0.20, lw=0)
        a.plot(hora, x, color="#2C5EA8", lw=1.3, label="agregado")
        a.plot(hora, y, color="#B35C00", lw=1.1, alpha=0.85, label="maq. lavar")
        a.set_ylabel("W")
        a.set_xlabel("horas")
        a.set_title(f"{inicio:%Y-%m-%d %H:%M}  ·  {int(s.sum())} janelas ligado",
                    fontsize=10, loc="left", fontweight="bold")
        a.legend(loc="upper right", frameon=False, fontsize=8)
        a.grid(alpha=0.18, lw=0.5)
        for lado in ("top", "right"):
            a.spines[lado].set_visible(False)

        gasf, gadf = gaf(x)
        imagens = [gasf, gadf, mtf(x), recorrencia(x)]
        for col, (img, (nome, cmapa, _)) in enumerate(zip(imagens, mapas), start=1):
            e = eixos[linha, col]
            lim = np.abs(img).max()
            kw = dict(cmap=cmapa, vmin=-lim, vmax=lim) if nome.startswith(("GASF", "GADF")) \
                else dict(cmap=cmapa)
            im = e.imshow(img, origin="lower", aspect="equal", **kw)
            e.set_title(nome, fontsize=10, fontweight="bold")
            e.set_xticks([]); e.set_yticks([])
            plt.colorbar(im, ax=e, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)

    fig.suptitle(f"{args.fonte} · maquina de lavar · {args.taxa} · janelas de "
                 f"{args.largura} amostras ({args.largura * taxa_s / 3600:.0f} h)  —  "
                 "transformadas do AGREGADO (faixa amarela = aparelho ligado)",
                 fontsize=13, y=0.995)
    FIGURAS.mkdir(exist_ok=True)
    saida = FIGURAS / f"imagens_{args.fonte}_{args.taxa}_{args.largura}.png"
    fig.savefig(saida, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"janelas em: {[str(datetime.fromtimestamp(int(t[c]), tz=timezone.utc).replace(tzinfo=None)) for c in centros]}")
    print(f"-> {saida}")


if __name__ == "__main__":
    main()
