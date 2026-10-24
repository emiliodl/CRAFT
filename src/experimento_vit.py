"""Vision Transformer sobre transformadas imagem-de-serie, no protocolo do artigo.

Mesma tarefa, mesmos dados, mesmo criterio de parada e de limiar do seq2point 1D
([src/ponto.py](ponto.py)) - a unica diferenca e que a janela vira imagem
(GASF/GADF/MTF/RP) e o encoder e um ViT.

Comparacao direta: `--modelo 1d` roda o seq2point de sempre, `--modelo vit` roda
este. Assim a linha de base sai da mesma execucao, com a mesma divisao e as
mesmas sementes.

Uso:
    python src/experimento_vit.py --aparelhos "washing machine,freezer"
    python src/experimento_vit.py --canais gasf,gadf --modelos vit
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
from experimento_artigo import JANELAS_POR_DIA, carregar, dividir, normalizador  # noqa: E402
from imagem import ClassificadorImagem                                      # noqa: E402
from ponto import ClassificadorPonto, FontePonto                            # noqa: E402
from treino import (Config, avaliar, contar_parametros_seguro, f1_positiva,  # noqa: E402
                    melhor_limiar, prever_com_indices, treinar)

SAIDA = Path(__file__).resolve().parents[1] / "resultados"


def rodar(aparelho: str, taxa: str, formato: str, modelo_nome: str, canais: list[str],
          vista: list[str], cfg: Config, semente: int, passo_treino: int,
          tamanho: int, folga: int, dias: int, escalas_dias: list[int]) -> list[dict]:
    torch.manual_seed(semente)
    np.random.seed(semente)
    ap = aparelho.replace(" ", "_")
    por_dia = JANELAS_POR_DIA[taxa]
    largura = por_dia                      # janela usada pelo carregador/divisao
    ctx = dias * por_dia                   # contexto real do seq2point
    escalas = [d * por_dia for d in escalas_dias]
    for e in escalas:
        if e % tamanho:
            raise SystemExit(f"escala de {e} amostras nao e multipla de {tamanho}")
    if formato == "centrado":
        antes, depois = ctx // 2, ctx - ctx // 2 - 1
        alinhamento = "centro"
    else:
        antes, depois = ctx - 1 - folga, folga
        alinhamento = "fim"

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

    t0 = time.time()
    if modelo_nome == "vit":
        m = ClassificadorImagem(len(vista), canais, tamanho=tamanho,
                                escalas=escalas, alinhamento=alinhamento)
    else:
        m = ClassificadorPonto(vista, antes + depois + 1)
    m = m.to(f_tr.vistas.device)
    info = treinar(m, f_tr, f_va, cfg)
    segundos = time.time() - t0

    linhas = []
    for nome, s in alvos.items():
        f_te = FontePonto(s, idx, antes, depois, media, desvio, 1)
        if len(f_te) == 0:
            continue
        y, p, _ = prever_com_indices(m, f_te)
        linhas.append(dict(
            aparelho=aparelho, taxa=taxa, formato=formato, modelo=modelo_nome,
            canais="+".join(canais) if modelo_nome == "vit" else "-",
            dias=dias, escalas="+".join(str(d) for d in escalas_dias),
            vista="+".join(vista), alvo=nome, semente=semente,
            f1=f1_positiva(y, p, info["limiar"]), f1_oraculo=melhor_limiar(y, p)[1],
            auc=avaliar(m, f_te, limiar=info["limiar"]).auc,
            prevalencia=round(float(y.mean()), 4), limiar=round(info["limiar"], 4),
            parametros=contar_parametros_seguro(m), n_alvos=len(f_te),
            epocas=len(info["historico"]), segundos=round(segundos, 1)))
    return linhas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aparelhos", default="washing machine,freezer")
    p.add_argument("--taxas", default="5min,15min")
    p.add_argument("--formatos", default="centrado")
    p.add_argument("--modelos", default="vit,1d")
    p.add_argument("--canais", default="gasf,gadf,mtf,rp")
    p.add_argument("--vista", default="casa")
    p.add_argument("--tamanho", type=int, default=96, help="lado da imagem apos PAA")
    p.add_argument("--dias", type=int, default=1, help="contexto total em dias")
    p.add_argument("--escalas", default=None,
                   help="escalas em dias, ex '1,7'; padrao = so o contexto inteiro")
    p.add_argument("--folga", type=int, default=1)
    p.add_argument("--passo_treino", type=int, default=4)
    p.add_argument("--sementes", type=int, default=3)
    p.add_argument("--epocas", type=int, default=300)
    p.add_argument("--paciencia", type=int, default=12)
    p.add_argument("--saida", default="vit.csv")
    args = p.parse_args()

    cfg = Config(epocas=args.epocas, paciencia=args.paciencia)
    SAIDA.mkdir(exist_ok=True)
    vista = [v.strip() for v in args.vista.split("+")]
    canais = [c.strip() for c in args.canais.split(",")]
    escalas_dias = ([int(x) for x in args.escalas.split(",")] if args.escalas
                    else [args.dias])
    linhas = []

    for aparelho in args.aparelhos.split(","):
        for taxa in args.taxas.split(","):
            for formato in args.formatos.split(","):
                for modelo in args.modelos.split(","):
                    for s in range(args.sementes):
                        novas = rodar(aparelho.strip(), taxa.strip(), formato.strip(),
                                      modelo.strip(), canais, vista, cfg, s,
                                      args.passo_treino, args.tamanho, args.folga,
                                      args.dias, escalas_dias)
                        linhas += novas
                        for r in novas:
                            print(f"  {aparelho:16s} {taxa:6s} {modelo:4s} s{s} "
                                  f"{r['alvo']:22s} F1={r['f1']:.3f} "
                                  f"(ora {r['f1_oraculo']:.3f}) AUC={r['auc']:.3f} "
                                  f"ep={r['epocas']:3d} {r['segundos']:.0f}s", flush=True)
                        pd.DataFrame(linhas).to_csv(SAIDA / args.saida, index=False,
                                                    encoding="utf-8")

    d = pd.DataFrame(linhas)
    print("\n=== media sobre as sementes ===")
    print(d.groupby(["aparelho", "alvo", "taxa", "modelo", "escalas"]).agg(
        f1=("f1", "mean"), dp=("f1", "std"), ora=("f1_oraculo", "mean"),
        auc=("auc", "mean"), params=("parametros", "mean")).round(3).to_string())
    print(f"\n-> {SAIDA / args.saida}")


if __name__ == "__main__":
    main()
