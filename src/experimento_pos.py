"""Pos-processamento com prior de duracao, sobre o melhor braco (agregado cru).

Para cada fold do leave-one-house-out treina UMA vez e compara tres formas de
transformar as probabilidades por instante em rotulos:

  cru      limiar por instante (o que ja tinhamos)
  morf     fechamento morfologico com tamanhos vindos das casas de origem
  viterbi  HMM de dois estados, transicoes contadas nas casas de origem

Todos os parametros - limiar, tamanhos do fechamento, vies da emissao - sao
escolhidos na validacao das casas de ORIGEM. A casa-alvo so e tocada na medicao.

Uso:
    python src/experimento_pos.py --aparelhos "washing machine" --folds 8
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
from decodificador import (duracoes, emissao_logit, grupos_contiguos,               # noqa: E402
                           matriz_transicao, morfologico, viterbi)
from diagnostico import f1_na_prevalencia                                           # noqa: E402
from experimento_loho import LARGURA, PASSO_AVAL, PASSO_TREINO, PREV_REFERENCIA      # noqa: E402
from modelo import Classificador                                                     # noqa: E402
from treino import Config, Fonte, f1_positiva, melhor_limiar, prever_com_indices, treinar  # noqa: E402

SAIDA = Path(__file__).resolve().parents[1] / "resultados"
VISTA = ["bruto"]        # o braco vencedor da ablacao LOHO


def ajustar_morf(y: np.ndarray, p: np.ndarray, fatias: list[slice],
                 limiar: float, dur_on: np.ndarray) -> tuple[int, int]:
    """Escolhe (fechar, abrir) na validacao de origem, partindo da duracao real."""
    base = p >= limiar
    candidatos_f = sorted({1, 2, 3, 4, 6, int(np.median(dur_on)) if len(dur_on) else 6})
    candidatos_a = sorted({1, 2, 3, int(np.percentile(dur_on, 25)) if len(dur_on) else 2})
    melhor, arg = -1.0, (1, 1)
    for f in candidatos_f:
        for a in candidatos_a:
            f1 = f1_positiva(y, morfologico(base, fatias, f, a).astype(float), 0.5)
            if f1 > melhor:
                melhor, arg = f1, (f, a)
    return arg


def ajustar_vies(y: np.ndarray, p: np.ndarray, fatias: list[slice], A: np.ndarray,
                 prior: float) -> float:
    """Desloca a emissao para maximizar F1 na validacao de origem."""
    melhor, arg = -1.0, 0.0
    for vies in np.linspace(-4, 4, 33):
        caminho = viterbi(emissao_logit(p, prior, vies), fatias, A)
        f1 = f1_positiva(y, caminho.astype(float), 0.5)
        if f1 > melhor:
            melhor, arg = f1, float(vies)
    return arg


def rodar(aparelho: str, alvo: tuple[str, str], fontes: list[tuple[str, str]],
          semente: int, cfg: Config) -> list[dict]:
    torch.manual_seed(semente)
    np.random.seed(semente)

    s_alvo = carregar(alvo[0], aparelho, alvo[1], LARGURA, PASSO_AVAL)
    s_fontes = juntar([carregar(d, aparelho, c, LARGURA, PASSO_TREINO) for d, c in fontes],
                      f"{len(fontes)} casas")
    tr, va = dividir_no_tempo(s_fontes, 0.2, LARGURA)

    idx = [VISTAS.index(v) for v in VISTA]
    media, desvio = normalizador(tr, LARGURA)
    f_tr = Fonte(tr, idx, LARGURA, media, desvio)
    f_va = Fonte(va, idx, LARGURA, media, desvio)
    f_te = Fonte(s_alvo, idx, LARGURA, media, desvio)

    t0 = time.time()
    modelo = Classificador(VISTA, LARGURA, fusao="soma").to(f_tr.vistas.device)
    info = treinar(modelo, f_tr, f_va, cfg)

    y_va, p_va, i_va = prever_com_indices(modelo, f_va)
    y_te, p_te, i_te = prever_com_indices(modelo, f_te)
    fatias_va = grupos_contiguos(i_va, s_fontes.t)
    fatias_te = grupos_contiguos(i_te, s_alvo.t)

    # tudo estimado nas casas de origem
    prior = float(y_va.mean())
    limiar = info["limiar"]
    dur_on, _ = duracoes(y_va, fatias_va)
    A = matriz_transicao(y_va, fatias_va)
    fechar, abrir = ajustar_morf(y_va, p_va, fatias_va, limiar, dur_on)
    vies = ajustar_vies(y_va, p_va, fatias_va, A, prior)
    segundos = time.time() - t0

    # versoes de ORACULO: os mesmos decodificadores, mas com o parametro ajustado
    # no proprio alvo. Nao sao resultado - servem so para separar "o metodo nao
    # ajuda" de "o parametro nao transferiu".
    limiar_ora, _ = melhor_limiar(y_te, p_te)
    vies_ora = ajustar_vies(y_te, p_te, fatias_te, A, prior)
    fechar_ora, abrir_ora = ajustar_morf(y_te, p_te, fatias_te, limiar_ora, dur_on)

    saidas = {
        "cru": (p_te >= limiar).astype(float),
        "morf": morfologico(p_te >= limiar, fatias_te, fechar, abrir).astype(float),
        "viterbi": viterbi(emissao_logit(p_te, prior, vies), fatias_te, A).astype(float),
        "cru_oraculo": (p_te >= limiar_ora).astype(float),
        "morf_oraculo": morfologico(p_te >= limiar_ora, fatias_te,
                                    fechar_ora, abrir_ora).astype(float),
        "viterbi_oraculo": viterbi(emissao_logit(p_te, prior, vies_ora),
                                   fatias_te, A).astype(float),
    }
    # o mesmo na validacao de origem, para ver se o decodificador ajuda em casa
    val = {
        "cru": (p_va >= limiar).astype(float),
        "morf": morfologico(p_va >= limiar, fatias_va, fechar, abrir).astype(float),
        "viterbi": viterbi(emissao_logit(p_va, prior, vies), fatias_va, A).astype(float),
    }

    # O prior de duracao so pode ajudar se o modelo TOCAR o ciclo: se ele acerta
    # alguma janela dentro do ciclo, a continuidade preenche o resto. Se o ciclo
    # inteiro passa despercebido, nao ha de onde propagar. Esta e a medida que
    # diz qual dos dois casos domina.
    def por_ciclo(pred: np.ndarray) -> tuple[int, float, float]:
        tocados, cobertura, total = 0, [], 0
        for f in fatias_te:
            yv, pv = y_te[f], pred[f]
            if len(yv) == 0:
                continue
            troca = np.flatnonzero(np.diff(yv.astype(int))) + 1
            lim = np.concatenate(([0], troca, [len(yv)]))
            for a, b in zip(lim[:-1], lim[1:]):
                if yv[a] != 1:
                    continue
                total += 1
                if pv[a:b].any():
                    tocados += 1
                    cobertura.append(float(pv[a:b].mean()))
        return total, tocados / max(total, 1), float(np.mean(cobertura)) if cobertura else 0.0

    n_ciclos, frac_tocados, cobertura = por_ciclo(saidas["cru"])

    linhas = []
    for modo, pred in saidas.items():
        vp = float(((pred == 1) & (y_te == 1)).sum())
        linhas.append(dict(
            aparelho=aparelho, alvo=f"{alvo[0]}/{alvo[1]}", modo=modo, semente=semente,
            f1=f1_positiva(y_te, pred, 0.5),
            precisao=vp / max(pred.sum(), 1), recall=vp / max(y_te.sum(), 1),
            f1_padronizada=(f1_na_prevalencia(y_te, p_te, PREV_REFERENCIA)
                            if modo == "cru" else np.nan),
            f1_origem=(f1_positiva(y_va, val[modo], 0.5) if modo in val else np.nan),
            prevalencia=float(y_te.mean()),
            n_ciclos=n_ciclos, ciclos_tocados=round(frac_tocados, 3),
            cobertura_no_ciclo=round(cobertura, 3),
            duracao_mediana_on=float(np.median(dur_on)) if len(dur_on) else np.nan,
            p_permanecer_ligado=round(float(A[1, 1]), 4),
            fechar=fechar, abrir=abrir, vies=round(vies, 2),
            vies_oraculo=round(vies_ora, 2),
            epocas=len(info["historico"]), segundos=round(segundos, 1)))
    return linhas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aparelhos", default="washing machine")
    p.add_argument("--folds", type=int, default=8)
    p.add_argument("--minimo_dias", type=float, default=100.0)
    p.add_argument("--epocas", type=int, default=300)
    p.add_argument("--paciencia", type=int, default=20)
    p.add_argument("--saida", default="pos.csv")
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
            novas = rodar(aparelho, alvo, fontes, 0, cfg)
            linhas += novas
            r = {x["modo"]: x for x in novas}
            print(f"  {novas[0]['alvo']:22s} cru={r['cru']['f1']:.3f}  "
                  f"morf={r['morf']['f1']:.3f}  viterbi={r['viterbi']['f1']:.3f}   "
                  f"(P/R viterbi {r['viterbi']['precisao']:.2f}/{r['viterbi']['recall']:.2f}, "
                  f"ciclos tocados={100*novas[0]['ciclos_tocados']:.0f}% de "
                  f"{novas[0]['n_ciclos']}, cobertura dentro="
                  f"{100*novas[0]['cobertura_no_ciclo']:.0f}%)", flush=True)
            pd.DataFrame(linhas).to_csv(caminho, index=False, encoding="utf-8")

    d = pd.DataFrame(linhas)
    print("\n=== media sobre os folds ===")
    print(d.groupby(["aparelho", "modo"]).agg(
        folds=("f1", "size"), f1=("f1", "mean"), dp=("f1", "std"),
        precisao=("precisao", "mean"), recall=("recall", "mean"),
        f1_origem=("f1_origem", "mean"), ciclos_tocados=("ciclos_tocados", "mean"),
        cobertura=("cobertura_no_ciclo", "mean")).round(3).to_string())
    print(f"\n-> {caminho}")


if __name__ == "__main__":
    main()
