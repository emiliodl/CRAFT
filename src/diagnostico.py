"""Por que a F1 cai na casa 2: decompoe a queda em causas mensuraveis.

Treina um braco, preve na casa 2 e responde tres perguntas:

  1. quanto da queda e TAXA-BASE?  A mesma curva ROC, avaliada na prevalencia da
     casa de origem, daria que F1? (precisao = p*TPR / (p*TPR + (1-p)*FPR))
  2. quanto e POTENCIA BAIXA?  Recall separado por decil da potencia real do
     aparelho: o modelo erra o ciclo todo ou so a parte fraca dele?
  3. quanto e LIMIAR?  F1 no limiar da origem contra F1 no limiar de oraculo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dados import VISTAS, carregar, dividir_no_tempo, normalizador      # noqa: E402
from experimento import BRACOS, LARGURA, PASSO_AVAL, PASSO_TREINO       # noqa: E402
from modelo import Classificador                                        # noqa: E402
from treino import Config, Fonte, f1_positiva, melhor_limiar, prever, treinar   # noqa: E402


def f1_na_prevalencia(y: np.ndarray, p: np.ndarray, prev: float) -> float:
    """Melhor F1 que esta ROC daria se a prevalencia fosse `prev`."""
    ordem = np.argsort(-p)
    yo = y[ordem]
    tp = np.cumsum(yo)
    fp = np.cumsum(1 - yo)
    tpr = tp / max(y.sum(), 1)
    fpr = fp / max((1 - y).sum(), 1)
    prec = prev * tpr / np.maximum(prev * tpr + (1 - prev) * fpr, 1e-12)
    f1 = 2 * prec * tpr / np.maximum(prec + tpr, 1e-12)
    return float(np.nanmax(f1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aparelho", default="washing machine")
    p.add_argument("--braco", default="3v_concat")
    p.add_argument("--semente", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.semente)
    np.random.seed(args.semente)
    spec = BRACOS[args.braco]
    vistas = spec["vistas"]
    idx = [VISTAS.index(v) for v in vistas]

    origem = carregar("uk_dale", args.aparelho, "casa 1", LARGURA, PASSO_TREINO)
    alvo = carregar("uk_dale", args.aparelho, "casa 2", LARGURA, PASSO_AVAL)
    tr, va = dividir_no_tempo(origem, 0.2, LARGURA)
    media, desvio = normalizador(tr, LARGURA)
    f_tr, f_va = Fonte(tr, idx, LARGURA, media, desvio), Fonte(va, idx, LARGURA, media, desvio)
    f_te = Fonte(alvo, idx, LARGURA, media, desvio)

    modelo = Classificador(vistas, LARGURA, fusao=spec["fusao"]).to(f_tr.vistas.device)
    info = treinar(modelo, f_tr, f_va, Config())
    y, prob = prever(modelo, f_te)
    limiar = info["limiar"]

    # potencia real do aparelho nos mesmos instantes avaliados
    arq = np.load(f"datasets/unificado_5min/uk_dale__{args.aparelho.replace(' ', '_')}__casa_2.npz")
    pot_total = arq[args.aparelho.replace(" ", "_")]
    meia = LARGURA // 2
    cobertos = np.unique(np.clip(alvo.centros[:, None] + np.arange(-meia, LARGURA - meia)[None, :],
                                 0, len(pot_total) - 1))
    pot = pot_total[cobertos][:len(y)]

    prev_alvo = float(y.mean())
    prev_origem = float(origem.rotulo[origem.centros].mean())
    _, f1_ora = melhor_limiar(y, prob)

    print(f"\n=== {args.aparelho} / {args.braco} ===")
    print(f"prevalencia: origem {100 * prev_origem:.2f}%   alvo {100 * prev_alvo:.2f}%")
    print(f"\n1) LIMIAR    F1 no limiar da origem   = {f1_positiva(y, prob, limiar):.3f}")
    print(f"             F1 no limiar de oraculo   = {f1_ora:.3f}"
          f"   (perda por calibracao: {f1_ora - f1_positiva(y, prob, limiar):.3f})")
    print(f"\n2) TAXA-BASE F1 desta ROC na prev. do alvo   ({100 * prev_alvo:.2f}%) = "
          f"{f1_na_prevalencia(y, prob, prev_alvo):.3f}")
    print(f"             F1 desta ROC na prev. da origem ({100 * prev_origem:.2f}%) = "
          f"{f1_na_prevalencia(y, prob, prev_origem):.3f}")
    print(f"             F1 desta ROC a 20%% de prevalencia            = "
          f"{f1_na_prevalencia(y, prob, 0.20):.3f}")

    print("\n3) POTENCIA  recall por faixa da potencia real do aparelho "
          f"(limiar {limiar:.3f})")
    ligado = y == 1
    faixas = [(0, 100), (100, 300), (300, 800), (800, 1500), (1500, 1e9)]
    for a, b in faixas:
        sel = ligado & (pot >= a) & (pot < b)
        if sel.sum() == 0:
            continue
        rec = float((prob[sel] >= limiar).mean())
        print(f"   {a:5.0f}-{b if b < 1e8 else 9999:5.0f} W  n={int(sel.sum()):6d} "
              f"({100 * sel.sum() / ligado.sum():5.1f}% do tempo ligado)  recall={rec:.3f}")


if __name__ == "__main__":
    main()
