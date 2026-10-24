"""Varias casas fonte: o ataque a causa do colapso em transferencia.

Todo o protocolo do artigo treina numa casa so (REFIT 8) e testa na Suica. Esse
e o regime de single-source domain generalization, o mais dificil que existe, e
foi onde todos os nossos metodos bateram: com uma casa so, nao ha contraste que
permita distinguir o que e propriedade do APARELHO do que e peculiaridade
daquela instalacao.

O caso da televisao mostrou o mecanismo de forma nitida: na REFIT casa 8, ligar
a TV faz o agregado CAIR 75 W (correlacao -0,03); na ECO casa 2, ligar faz subir
282 W (+0,36). O modelo aprendeu a regra certa da casa errada, e a AUC de
transferencia ficou em 0,235 - pior que o acaso. Com dez casas fonte, uma
correlacao invertida numa delas vira ruido em vez de regra.

O que muda e SO o treino:

  treino   REFIT casa 8 + K outras casas, mesma janela de datas, empilhadas
  teste    identicos aos do artigo - mes da casa 8 (intra) e ECO (cross)

Como os conjuntos de teste nao mudam, as duas colunas continuam comparaveis com
a tabela do artigo; o que deixa de valer e a clausula "treinado numa casa so",
e isso precisa ser declarado como desvio consciente do protocolo.

Uso:
    python src/experimento_multicasa.py --casas_extra 1,3,4,7,13,17,18,20 --k 0,2,4,8
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

from dados import VISTAS, juntar                                              # noqa: E402
from experimento_artigo import JANELAS_POR_DIA, carregar, dividir, normalizador  # noqa: E402
from ponto import ClassificadorPonto, FontePonto                              # noqa: E402
from treino import (Config, DISPOSITIVO, avaliar, f1_positiva, melhor_limiar,  # noqa: E402
                    prever_com_indices, treinar)

SAIDA = Path(__file__).resolve().parents[1] / "resultados"


def rodar(aparelho: str, taxa: str, casas: list[int], vista: list[str], cfg: Config,
          semente: int, passo_treino: int, horas_ctx: float) -> list[dict]:
    torch.manual_seed(semente)
    np.random.seed(semente)
    ap = aparelho.replace(" ", "_")
    por_dia = JANELAS_POR_DIA[taxa]
    ctx = max(4, int(round(horas_ctx * por_dia / 24 / 4)) * 4)
    antes, depois = ctx // 2, ctx - ctx // 2 - 1

    # --- treino: casa 8 sempre, mais as extras ---
    partes_tr, partes_va, usadas = [], [], []
    for casa in casas:
        s = carregar(f"refit_casa_{casa}__{ap}__{taxa}__treino.npz", "media", por_dia, 1)
        if s is None:
            continue
        t, v = dividir(s, 0.15, por_dia)
        partes_tr.append(t)
        partes_va.append(v)
        usadas.append(casa)
    if not partes_tr:
        return []
    tr = juntar(partes_tr, "treino") if len(partes_tr) > 1 else partes_tr[0]
    va = juntar(partes_va, "val") if len(partes_va) > 1 else partes_va[0]

    # --- alvos: identicos aos do artigo, nao mudam com o numero de casas ---
    alvos = {}
    intra = carregar(f"refit_casa_8__{ap}__{taxa}__teste.npz", "media", por_dia, 1)
    if intra is not None:
        alvos["intra: refit casa 8"] = intra
    eco = []
    for n in (1, 2, 5):
        s = carregar(f"eco_casa_{n}__{ap}__{taxa}__cross.npz", "media", por_dia, 1)
        if s is not None:
            alvos[f"cross: eco casa {n}"] = s
            eco.append(s)
    if len(eco) > 1:
        alvos["cross: eco agregada"] = juntar(eco, "eco agregada")

    idx = [VISTAS.index(v) for v in vista]
    media, desvio = normalizador(tr, por_dia)
    f_tr = FontePonto(tr, idx, antes, depois, media, desvio, passo_treino)
    f_va = FontePonto(va, idx, antes, depois, media, desvio, max(1, passo_treino // 2))

    t0 = time.time()
    modelo = ClassificadorPonto(vista, ctx).to(DISPOSITIVO)
    info = treinar(modelo, f_tr, f_va, cfg)
    segundos = time.time() - t0

    linhas = []
    for nome, s in alvos.items():
        f_te = FontePonto(s, idx, antes, depois, media, desvio, 1)
        if len(f_te) == 0:
            continue
        y, p, _ = prever_com_indices(modelo, f_te)
        linhas.append(dict(
            aparelho=aparelho, taxa=taxa, n_casas=len(usadas), casas="+".join(map(str, usadas)),
            ctx=ctx, vista="+".join(vista), alvo=nome, semente=semente,
            f1=f1_positiva(y, p, info["limiar"]), f1_meio=f1_positiva(y, p, 0.5),
            f1_oraculo=melhor_limiar(y, p)[1],
            auc=avaliar(modelo, f_te, limiar=info["limiar"]).auc,
            prevalencia=round(float(y.mean()), 4), limiar=round(info["limiar"], 4),
            n_alvos=len(f_te), n_treino=len(f_tr), epocas=len(info["historico"]),
            segundos=round(segundos, 1)))
    return linhas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aparelhos", default="television,washing machine,freezer")
    p.add_argument("--taxa", default="5min")
    p.add_argument("--casas_extra", default="1,3,4,7,13,17,18,20")
    p.add_argument("--k", default="0,2,4,8", help="quantas casas EXTRA entram no treino")
    p.add_argument("--vista", default="casa")
    p.add_argument("--horas_ctx", type=float, default=24.0)
    p.add_argument("--passo_treino", type=int, default=4)
    p.add_argument("--sementes", type=int, default=3)
    p.add_argument("--epocas", type=int, default=200)
    p.add_argument("--paciencia", type=int, default=15)
    p.add_argument("--saida", default="multicasa.csv")
    args = p.parse_args()

    cfg = Config(epocas=args.epocas, paciencia=args.paciencia)
    vista = [v.strip() for v in args.vista.split("+")]
    extras = [int(c) for c in args.casas_extra.split(",")]
    linhas = []
    for aparelho in args.aparelhos.split(","):
        for k in [int(x) for x in args.k.split(",")]:
            casas = [8] + extras[:k]
            for semente in range(args.sementes):
                novas = rodar(aparelho.strip(), args.taxa, casas, vista, cfg,
                              semente, args.passo_treino, args.horas_ctx)
                linhas += novas
                for r in novas:
                    print(f"  {aparelho:16s} k={k} ({r['n_casas']} casas) s{semente} "
                          f"{r['alvo']:22s} F1={r['f1']:.3f} (ora {r['f1_oraculo']:.3f}) "
                          f"AUC={r['auc']:.3f} n_tr={r['n_treino']} "
                          f"ep={r['epocas']:3d} {r['segundos']:.0f}s", flush=True)
                SAIDA.mkdir(exist_ok=True)
                pd.DataFrame(linhas).to_csv(SAIDA / args.saida, index=False, encoding="utf-8")

    d = pd.DataFrame(linhas)
    print("\n=== media sobre as sementes ===")
    print(d.groupby(["aparelho", "alvo", "n_casas"]).agg(
        f1=("f1", "mean"), dp=("f1", "std"), ora=("f1_oraculo", "mean"),
        auc=("auc", "mean")).round(3).to_string())
    print(f"\n-> {SAIDA / args.saida}")


if __name__ == "__main__":
    main()
