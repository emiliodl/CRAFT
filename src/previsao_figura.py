"""Treina um braco e desenha a previsao sobre a casa de teste, dia a dia."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dados import VISTAS, carregar, dividir_no_tempo, normalizador      # noqa: E402
from experimento import BRACOS, LARGURA, PASSO_AVAL, PASSO_TREINO       # noqa: E402
from modelo import Classificador                                        # noqa: E402
from treino import Config, Fonte, prever, treinar                       # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figuras"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aparelho", default="dish washer")
    p.add_argument("--braco", default="3v_porta")
    p.add_argument("--epocas", type=int, default=60)
    p.add_argument("--dias", type=int, default=5)
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
    info = treinar(modelo, f_tr, f_va, Config(epocas=args.epocas, paciencia=10))
    y, prob = prever(modelo, f_te)
    limiar = info["limiar"]

    # o trecho com mais ativacoes, para a figura mostrar alguma coisa
    n = args.dias * 288
    soma = np.convolve(y, np.ones(n), mode="valid")
    i0 = int(np.argmax(soma))
    fatia = slice(i0, i0 + n)
    hora = np.arange(n) * 5 / 60 / 24

    fig, eixos = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                              gridspec_kw=dict(height_ratios=[2, 1], hspace=0.12))
    a = eixos[0]
    a.plot(hora, alvo.vistas[0][:len(y)][fatia], color="#2C5EA8", lw=0.9, label="agregado")
    a.set_ylabel("W")
    a.legend(loc="upper right", frameon=False, fontsize=9)
    a.set_title(f"{args.aparelho} · {args.braco} · UK-DALE casa 2 (nunca vista no treino)",
                loc="left", fontsize=12, fontweight="bold")

    a = eixos[1]
    a.fill_between(hora, 0, y[fatia], color="#F0B429", alpha=0.55, lw=0, label="verdade")
    a.plot(hora, prob[fatia], color="#7B4FA8", lw=1.1, label="probabilidade prevista")
    a.axhline(limiar, color="#B35C00", lw=1.0, ls="--", label=f"limiar da origem ({limiar:.2f})")
    a.set_ylim(-0.03, 1.03)
    a.set_ylabel("P(ligado)")
    a.set_xlabel("dias")
    a.legend(loc="upper right", frameon=False, fontsize=9, ncol=3)

    for a in eixos:
        a.grid(alpha=0.18, lw=0.6)
        a.margins(x=0.005)
        for lado in ("top", "right"):
            a.spines[lado].set_visible(False)

    FIG.mkdir(exist_ok=True)
    saida = FIG / f"previsao_{args.aparelho.replace(' ', '_')}_{args.braco}.png"
    fig.savefig(saida, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"limiar={limiar:.3f}  melhor_epoca={info['melhor_epoca']}  -> {saida}")


if __name__ == "__main__":
    main()
