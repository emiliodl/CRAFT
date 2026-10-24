"""Fusao dos dois ramos, contra os ramos isolados e contra o conjunto simples.

Quatro linhas por semente, todas da MESMA divisao e da MESMA inicializacao:

  1d        seq2point sobre a serie          (vence dentro da casa)
  vit       ViT sobre GASF/GADF/MTF/RP       (vence entre casas, na AUC)
  duplo     os dois ramos concatenados, treinados juntos
  conjunto  media das probabilidades de `1d` e `vit` treinados separadamente

O `conjunto` e o controle que importa. Se `duplo` nao vencer a media simples,
a arquitetura nao acrescentou nada - apenas reproduziu o que mediar ja dava, e
por um custo maior.

Uso:
    python src/experimento_duplo.py --aparelhos "washing machine" --taxas 5min
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

from dados import VISTAS                                                    # noqa: E402
from duplo import ClassificadorDuplo                                        # noqa: E402
from experimento_artigo import JANELAS_POR_DIA, carregar, dividir, normalizador  # noqa: E402
from imagem import ClassificadorImagem                                      # noqa: E402
from ponto import ClassificadorPonto, FontePonto                            # noqa: E402
from treino import (Config, contar_parametros_seguro, f1_positiva,          # noqa: E402
                    melhor_limiar, prever_com_indices, treinar)
from sklearn.metrics import roc_auc_score                                   # noqa: E402

SAIDA = Path(__file__).resolve().parents[1] / "resultados"


def construir(nome: str, vista, largura, canais, tamanho, escalas, alinhamento):
    if nome == "1d":
        return ClassificadorPonto(vista, largura)
    if nome == "vit":
        return ClassificadorImagem(len(vista), canais, tamanho=tamanho,
                                   escalas=escalas, alinhamento=alinhamento)
    if nome == "duplo":
        return ClassificadorDuplo(vista, largura, canais, tamanho=tamanho,
                                  escalas=escalas, alinhamento=alinhamento)
    raise ValueError(nome)


def metricas(y, p, limiar):
    return dict(f1=f1_positiva(y, p, limiar), f1_oraculo=melhor_limiar(y, p)[1],
                auc=float(roc_auc_score(y, p)) if y.min() != y.max() else float("nan"),
                prevalencia=round(float(y.mean()), 4))


def rodar(aparelho: str, taxa: str, formato: str, canais: list[str], vista: list[str],
          cfg: Config, semente: int, passo_treino: int, tamanho: int, folga: int) -> list[dict]:
    ap = aparelho.replace(" ", "_")
    largura = JANELAS_POR_DIA[taxa]
    if formato == "centrado":
        antes, depois, alinhamento = largura // 2, largura - largura // 2 - 1, "centro"
    else:
        antes, depois, alinhamento = largura - 1 - folga, folga, "fim"

    treino_total = carregar(f"refit_casa_8__{ap}__{taxa}__treino.npz", "media", largura, 1)
    tr, va = dividir(treino_total, 0.15, largura)
    alvos = {}
    intra = carregar(f"refit_casa_8__{ap}__{taxa}__teste.npz", "media", largura, 1)
    if intra is not None:
        alvos["intra: refit casa 8"] = intra
    for n in (1, 2, 5):
        s = carregar(f"eco_casa_{n}__{ap}__{taxa}__cross.npz", "media", largura, 1)
        if s is not None:
            alvos[f"cross: eco casa {n}"] = s

    idx = [VISTAS.index(v) for v in vista]
    media, desvio = normalizador(tr, largura)
    f_tr = FontePonto(tr, idx, antes, depois, media, desvio, passo_treino)
    f_va = FontePonto(va, idx, antes, depois, media, desvio, max(1, passo_treino // 2))
    fontes_alvo = {n: FontePonto(s, idx, antes, depois, media, desvio, 1)
                   for n, s in alvos.items()}

    linhas, guardado = [], {}
    for nome in ("1d", "vit", "duplo"):
        torch.manual_seed(semente)
        np.random.seed(semente)
        m = construir(nome, vista, antes + depois + 1, canais, tamanho,
                      [antes + depois + 1], alinhamento).to(f_tr.vistas.device)
        t0 = time.time()
        info = treinar(m, f_tr, f_va, cfg)
        seg = time.time() - t0
        y_va, p_va, _ = prever_com_indices(m, f_va)
        guardado[nome] = dict(val=(y_va, p_va), limiar=info["limiar"], alvo={})
        for n, f in fontes_alvo.items():
            y, p, _ = prever_com_indices(m, f)
            guardado[nome]["alvo"][n] = (y, p)
            linhas.append(dict(aparelho=aparelho, taxa=taxa, formato=formato, modelo=nome,
                               alvo=n, semente=semente, limiar=round(info["limiar"], 4),
                               parametros=contar_parametros_seguro(m),
                               epocas=len(info["historico"]), segundos=round(seg, 1),
                               **metricas(y, p, info["limiar"])))
        print(f"    {nome:6s} pronto em {seg:.0f}s ({len(info['historico'])} epocas)", flush=True)

    # conjunto: media das probabilidades dos dois ramos treinados separadamente
    y_va = guardado["1d"]["val"][0]
    p_va = (guardado["1d"]["val"][1] + guardado["vit"]["val"][1]) / 2
    limiar, _ = melhor_limiar(y_va, p_va)
    for n in fontes_alvo:
        y = guardado["1d"]["alvo"][n][0]
        p = (guardado["1d"]["alvo"][n][1] + guardado["vit"]["alvo"][n][1]) / 2
        linhas.append(dict(aparelho=aparelho, taxa=taxa, formato=formato, modelo="conjunto",
                           alvo=n, semente=semente, limiar=round(limiar, 4),
                           parametros=(guardado["1d"]["alvo"] and 0), epocas=0, segundos=0.0,
                           **metricas(y, p, limiar)))
    return linhas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aparelhos", default="washing machine")
    p.add_argument("--taxas", default="5min")
    p.add_argument("--formatos", default="centrado")
    p.add_argument("--canais", default="gasf,gadf,mtf,rp")
    p.add_argument("--vista", default="casa")
    p.add_argument("--tamanho", type=int, default=96)
    p.add_argument("--folga", type=int, default=1)
    p.add_argument("--passo_treino", type=int, default=4)
    p.add_argument("--sementes", type=int, default=3)
    p.add_argument("--epocas", type=int, default=300)
    p.add_argument("--paciencia", type=int, default=12)
    p.add_argument("--saida", default="duplo.csv")
    args = p.parse_args()

    cfg = Config(epocas=args.epocas, paciencia=args.paciencia)
    SAIDA.mkdir(exist_ok=True)
    vista = [v.strip() for v in args.vista.split("+")]
    canais = [c.strip() for c in args.canais.split(",")]
    linhas = []
    for aparelho in args.aparelhos.split(","):
        for taxa in args.taxas.split(","):
            for formato in args.formatos.split(","):
                for s in range(args.sementes):
                    print(f"  {aparelho} / {taxa} / {formato} / semente {s}", flush=True)
                    linhas += rodar(aparelho.strip(), taxa.strip(), formato.strip(), canais,
                                    vista, cfg, s, args.passo_treino, args.tamanho, args.folga)
                    pd.DataFrame(linhas).to_csv(SAIDA / args.saida, index=False,
                                                encoding="utf-8")

    d = pd.DataFrame(linhas)
    print("\n=== media sobre as sementes ===")
    print(d.groupby(["aparelho", "alvo", "taxa", "modelo"]).agg(
        f1=("f1", "mean"), dp=("f1", "std"), ora=("f1_oraculo", "mean"),
        auc=("auc", "mean")).round(3).to_string())
    print(f"\n-> {SAIDA / args.saida}")


if __name__ == "__main__":
    main()
