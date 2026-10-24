"""Perda de treino alinhada com a F1, sobre o leave-one-house-out.

Todos os experimentos anteriores treinaram com BCE pura. Com 2-5% de janelas
positivas, a BCE e dominada pelos negativos e o modelo aprende a ser
conservador - o recall medido foi 0,395 contra precisao 0,396, com 41% dos
ciclos nunca tocados. Este experimento troca so a perda, mantendo arquitetura,
dados, protocolo e criterio de parada.

  bce          referencia (o que rodamos ate aqui)
  bce_peso     pos_weight = neg/pos
  focal        gama = 2, alfa da prevalencia
  soft_f1      1 - F1 diferenciavel
  bce_soft_f1  soma das duas

Reporta o limiar cru e o fechamento morfologico, que e o pos-processamento que
sobreviveu ao teste anterior (+0,007, positivo em 7 de 8 folds).

Uso:
    python src/experimento_perda.py --aparelhos "washing machine" --folds 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dados import VISTAS, carregar, dividir_no_tempo, juntar, listar, normalizador  # noqa: E402
from decodificador import duracoes, grupos_contiguos, morfologico                   # noqa: E402
from diagnostico import f1_na_prevalencia                                           # noqa: E402
from experimento_loho import LARGURA, PASSO_AVAL, PASSO_TREINO, PREV_REFERENCIA     # noqa: E402
from experimento_pos import ajustar_morf                                            # noqa: E402
from modelo import Classificador                                                    # noqa: E402
from treino import Config, Fonte, f1_positiva, prever_com_indices, treinar          # noqa: E402

SAIDA = Path(__file__).resolve().parents[1] / "resultados"
VISTA = ["bruto"]
PERDAS = ["bce", "bce_peso", "focal", "soft_f1", "bce_soft_f1"]


def ciclos(y: np.ndarray, pred: np.ndarray, fatias: list[slice]) -> tuple[int, float, float]:
    """(numero de ciclos, fracao tocada, cobertura media dentro dos tocados)."""
    tocados, cobertura, total = 0, [], 0
    for f in fatias:
        yv, pv = y[f], pred[f]
        if len(yv) == 0:
            continue
        lim = np.concatenate(([0], np.flatnonzero(np.diff(yv.astype(int))) + 1, [len(yv)]))
        for a, b in zip(lim[:-1], lim[1:]):
            if yv[a] != 1:
                continue
            total += 1
            if pv[a:b].any():
                tocados += 1
                cobertura.append(float(pv[a:b].mean()))
    return total, tocados / max(total, 1), float(np.mean(cobertura)) if cobertura else 0.0


def rodar(aparelho: str, alvo: tuple[str, str], fontes: list[tuple[str, str]],
          perda: str, cfg_base: Config) -> list[dict]:
    torch.manual_seed(0)
    np.random.seed(0)

    s_alvo = carregar(alvo[0], aparelho, alvo[1], LARGURA, PASSO_AVAL)
    s_fontes = juntar([carregar(d, aparelho, c, LARGURA, PASSO_TREINO) for d, c in fontes],
                      f"{len(fontes)} casas")
    tr, va = dividir_no_tempo(s_fontes, 0.2, LARGURA)

    idx = [VISTAS.index(v) for v in VISTA]
    media, desvio = normalizador(tr, LARGURA)
    f_tr = Fonte(tr, idx, LARGURA, media, desvio)
    f_va = Fonte(va, idx, LARGURA, media, desvio)
    f_te = Fonte(s_alvo, idx, LARGURA, media, desvio)

    cfg = Config(**cfg_base.__dict__)
    cfg.perda = perda
    cfg.prevalencia = float(tr.rotulo[tr.centros].mean())

    t0 = time.time()
    modelo = Classificador(VISTA, LARGURA, fusao="soma").to(f_tr.vistas.device)
    info = treinar(modelo, f_tr, f_va, cfg)
    y_va, p_va, i_va = prever_com_indices(modelo, f_va)
    y_te, p_te, i_te = prever_com_indices(modelo, f_te)
    fatias_va = grupos_contiguos(i_va, s_fontes.t)
    fatias_te = grupos_contiguos(i_te, s_alvo.t)
    segundos = time.time() - t0

    limiar = info["limiar"]
    dur_on, _ = duracoes(y_va, fatias_va)
    fechar, abrir = ajustar_morf(y_va, p_va, fatias_va, limiar, dur_on)

    saidas = {
        "cru": (p_te >= limiar).astype(float),
        "morf": morfologico(p_te >= limiar, fatias_te, fechar, abrir).astype(float),
    }
    n_ciclos, frac, cob = ciclos(y_te, saidas["cru"], fatias_te)

    linhas = []
    for modo, pred in saidas.items():
        vp = float(((pred == 1) & (y_te == 1)).sum())
        linhas.append(dict(
            aparelho=aparelho, alvo=f"{alvo[0]}/{alvo[1]}", perda=perda, modo=modo,
            f1=f1_positiva(y_te, pred, 0.5),
            precisao=vp / max(pred.sum(), 1), recall=vp / max(y_te.sum(), 1),
            f1_padronizada=f1_na_prevalencia(y_te, p_te, PREV_REFERENCIA),
            f1_origem=info["f1_val"], ap_origem=info["ap_val"],
            prevalencia=float(y_te.mean()), n_ciclos=n_ciclos,
            ciclos_tocados=round(frac, 3), cobertura_no_ciclo=round(cob, 3),
            limiar=round(limiar, 4), epocas=len(info["historico"]),
            segundos=round(segundos, 1)))
    return linhas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aparelhos", default="washing machine")
    p.add_argument("--perdas", default=",".join(PERDAS))
    p.add_argument("--folds", type=int, default=8)
    p.add_argument("--minimo_dias", type=float, default=100.0)
    p.add_argument("--epocas", type=int, default=300)
    p.add_argument("--paciencia", type=int, default=20)
    p.add_argument("--saida", default="perda.csv")
    args = p.parse_args()

    cfg = Config(epocas=args.epocas, paciencia=args.paciencia)
    SAIDA.mkdir(exist_ok=True)
    caminho = SAIDA / args.saida
    linhas = []

    for aparelho in args.aparelhos.split(","):
        aparelho = aparelho.strip()
        todas = listar(aparelho, args.minimo_dias)
        alvos = [(d, c) for d, c, _ in todas[:args.folds]]
        print(f"\n### {aparelho}: {len(alvos)} folds sobre {len(todas)} casas", flush=True)
        for alvo in alvos:
            fontes = [(d, c) for d, c, _ in todas if (d, c) != alvo]
            for perda in args.perdas.split(","):
                novas = rodar(aparelho, alvo, fontes, perda.strip(), cfg)
                linhas += novas
                r = {x["modo"]: x for x in novas}
                print(f"  {novas[0]['alvo']:20s} {perda:12s} cru={r['cru']['f1']:.3f} "
                      f"morf={r['morf']['f1']:.3f}  P/R={r['cru']['precisao']:.2f}/"
                      f"{r['cru']['recall']:.2f}  tocados={100*novas[0]['ciclos_tocados']:.0f}%  "
                      f"ep={novas[0]['epocas']:3d}  {novas[0]['segundos']:.0f}s", flush=True)
                pd.DataFrame(linhas).to_csv(caminho, index=False, encoding="utf-8")

    d = pd.DataFrame(linhas)
    print("\n=== media sobre os folds ===")
    print(d.groupby(["aparelho", "perda", "modo"]).agg(
        folds=("f1", "size"), f1=("f1", "mean"), dp=("f1", "std"),
        precisao=("precisao", "mean"), recall=("recall", "mean"),
        tocados=("ciclos_tocados", "mean"), f1_origem=("f1_origem", "mean")).round(3).to_string())
    print(f"\n-> {caminho}")


if __name__ == "__main__":
    main()
