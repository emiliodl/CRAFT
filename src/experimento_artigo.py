"""Protocolo de Chatterjee & Heer (2025), so classificacao.

Treina na casa 8 do REFIT (2014-03-07 a 2015-03-06) e testa em duas condicoes:

  intra   casa 8 do REFIT, 2015-03-07 a 2015-04-06 (o mes seguinte)
  cross   ECO casas 1, 2 e 5, 2012-06-02 a 2012-07-01

Aparelhos: freezer e washing machine (a intersecao entre o artigo e o recorte
do projeto). Taxas: 5 min e 15 min. Metrica: F1 da classe positiva.

Duas definicoes de rotulo, ambas medidas:

  media    limiariza a potencia ja reamostrada, como o nilmtk faz. E o que
           torna o numero comparavel com a tabela publicada.
  nativo   limiariza na taxa original e marca a janela se >= 50% das amostras
           nativas estavam ligadas. E o alvo estavel entre taxas - sem ele,
           "5 min contra 15 min" compara alvos diferentes, nao resolucoes.

Nada do alvo entra em treino, parada ou escolha de limiar.

Uso:
    python src/experimento_artigo.py
    python src/experimento_artigo.py --aparelhos freezer --taxas 5min
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

from dados import Serie, calcular_vistas, corridas                      # noqa: E402
from dados import VISTAS                                                # noqa: E402
from decodificador import duracoes, grupos_contiguos, morfologico        # noqa: E402
from experimento_pos import ajustar_morf                                 # noqa: E402
from modelo import Classificador                                         # noqa: E402
from modelo_sgn import ClassificadorSGN                                  # noqa: E402
from treino import (Config, Fonte, avaliar, f1_positiva, melhor_limiar,  # noqa: E402
                    prever_com_indices, treinar)

RAIZ = Path(__file__).resolve().parents[1]
PASTA = RAIZ / "datasets" / "protocolo_artigo"
SAIDA = RAIZ / "resultados"

TAXA_S = {"5min": 300, "15min": 900}
JANELAS_POR_DIA = {"5min": 288, "15min": 96}
VISTA = ["bruto"]          # sobrescrito por --vista


def carregar(arquivo: str, rotulo: str, largura: int, passo: int) -> Serie | None:
    """Le um npz do protocolo e monta a Serie, respeitando os buracos no tempo."""
    caminho = PASTA / arquivo
    if not caminho.exists():
        return None
    d = np.load(caminho)
    ap = next(k for k in d.files
              if k not in ("t", "main") and not k.startswith("status_"))
    chave = f"status_{ap}" + ("_nativo" if rotulo == "nativo" else "")
    t, x, st = d["t"], d["main"], d[chave].astype(np.float32)

    taxa = TAXA_S["15min" if "15min" in arquivo else "5min"]
    trechos = corridas(t, taxa)
    vistas = calcular_vistas(x, trechos)

    meia = largura // 2
    centros = [np.arange(a + meia, b - (largura - meia) + 1, passo)
               for a, b in trechos if b - a >= largura]
    if not centros:
        return None
    return Serie(vistas, st, np.concatenate(centros).astype(np.int64),
                 caminho.stem, t.astype(np.int64))


def dividir(s: Serie, fracao_val: float, largura: int) -> tuple[Serie, Serie]:
    """Ultimos `fracao_val` de cada corrida viram validacao de origem."""
    c = s.centros
    passo = int(np.min(np.diff(c))) if len(c) > 1 else 1
    grupos = np.split(c, np.flatnonzero(np.diff(c) > passo) + 1)
    tr, va = [], []
    for g in grupos:
        corte = int(len(g) * (1 - fracao_val))
        folga = max(1, largura // passo)
        tr.append(g[: max(corte - folga, 0)])
        va.append(g[corte:])
    return (Serie(s.vistas, s.rotulo, np.concatenate(tr), s.nome + "[tr]", s.t),
            Serie(s.vistas, s.rotulo, np.concatenate(va), s.nome + "[va]", s.t))


def normalizador(s: Serie, largura: int):
    meia = largura // 2
    idx = np.unique(np.clip(s.centros[:, None] + np.arange(-meia, largura - meia)[None, :],
                            0, s.vistas.shape[1] - 1))
    sub = s.vistas[:, idx]
    return sub.mean(1).astype(np.float32), np.maximum(sub.std(1), 1e-6).astype(np.float32)


def piso_reglog(tr: Serie, va: Serie, alvos: dict[str, Serie], largura: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    meia = largura // 2

    def matriz(s: Serie):
        idx = np.unique(np.clip(s.centros[:, None] + np.arange(-meia, largura - meia)[None, :],
                                0, s.vistas.shape[1] - 1))
        return s.vistas[:, idx].T, s.rotulo[idx]

    xtr, ytr = matriz(tr)
    xva, yva = matriz(va)
    m, dv = xtr.mean(0), xtr.std(0) + 1e-6
    rl = LogisticRegression(max_iter=300).fit((xtr - m) / dv, ytr)
    limiar, _ = melhor_limiar(yva, rl.predict_proba((xva - m) / dv)[:, 1])
    fora = {}
    for nome, s in alvos.items():
        xte, yte = matriz(s)
        p = rl.predict_proba((xte - m) / dv)[:, 1]
        fora[nome] = dict(f1=f1_positiva(yte, p, limiar), f1_oraculo=melhor_limiar(yte, p)[1],
                          prevalencia=float(yte.mean()))
    return fora


def rodar(aparelho: str, taxa: str, rotulo: str, cfg: Config, semente: int,
          vista: list[str] | None = None, patch: int = 4, arq: str = "linear") -> list[dict]:
    vista = vista or VISTA
    torch.manual_seed(semente)
    np.random.seed(semente)
    ap = aparelho.replace(" ", "_")
    largura = JANELAS_POR_DIA[taxa]
    passo_tr = largura // 8
    passo_te = largura // 4

    treino_total = carregar(f"refit_casa_8__{ap}__{taxa}__treino.npz", rotulo, largura, passo_tr)
    if treino_total is None:
        raise SystemExit(f"falta o treino de {aparelho}/{taxa}")
    tr, va = dividir(treino_total, 0.15, largura)

    alvos: dict[str, Serie] = {}
    intra = carregar(f"refit_casa_8__{ap}__{taxa}__teste.npz", rotulo, largura, passo_te)
    if intra is not None:
        alvos["intra: refit casa 8"] = intra
    for n in (1, 2, 5):
        s = carregar(f"eco_casa_{n}__{ap}__{taxa}__cross.npz", rotulo, largura, passo_te)
        if s is not None:
            alvos[f"cross: eco casa {n}"] = s
    if not alvos:
        print(f"  {aparelho}/{taxa}/{rotulo}: nenhum alvo disponivel")
        return []

    idx = [VISTAS.index(v) for v in vista]
    media, desvio = normalizador(tr, largura)
    f_tr, f_va = Fonte(tr, idx, largura, media, desvio), Fonte(va, idx, largura, media, desvio)

    t0 = time.time()
    prev_tr = float(tr.rotulo[tr.centros].mean())
    if arq == "linear":
        modelo = Classificador(vista, largura, fusao="soma", patch=patch)
    else:
        modelo = ClassificadorSGN(vista, largura, patch=patch, modo=arq,
                                  prevalencia=prev_tr)
    modelo = modelo.to(f_tr.vistas.device)
    info = treinar(modelo, f_tr, f_va, cfg)
    y_va, p_va, i_va = prever_com_indices(modelo, f_va)
    fatias_va = grupos_contiguos(i_va, treino_total.t, TAXA_S[taxa])
    dur_on, _ = duracoes(y_va, fatias_va)
    fechar, abrir = ajustar_morf(y_va, p_va, fatias_va, info["limiar"], dur_on)
    segundos = time.time() - t0

    pisos = piso_reglog(tr, va, alvos, largura)

    linhas = []
    for nome, s in alvos.items():
        f_te = Fonte(s, idx, largura, media, desvio)
        y, p, i = prever_com_indices(modelo, f_te)
        fatias = grupos_contiguos(i, s.t, TAXA_S[taxa])
        pred_morf = morfologico(p >= info["limiar"], fatias, fechar, abrir).astype(float)
        base = dict(aparelho=aparelho, taxa=taxa, rotulo=rotulo, alvo=nome,
                    vista="+".join(vista), patch=patch, arq=arq,
                    prevalencia=round(float(y.mean()), 4), limiar=round(info["limiar"], 4),
                    epocas=len(info["historico"]), segundos=round(segundos, 1))
        linhas.append(dict(**base, modelo="rede", f1=f1_positiva(y, p, info["limiar"]),
                           f1_oraculo=melhor_limiar(y, p)[1],
                           auc=avaliar(modelo, f_te, limiar=info["limiar"]).auc))
        linhas.append(dict(**base, modelo="rede+morf", f1=f1_positiva(y, pred_morf, 0.5),
                           f1_oraculo=np.nan, auc=np.nan))
        linhas.append(dict(**base, modelo="piso", f1=pisos[nome]["f1"],
                           f1_oraculo=pisos[nome]["f1_oraculo"], auc=np.nan))
        # referencia trivial: prever SEMPRE ligado. Com prevalencia p a F1 da
        # classe positiva e 2p/(1+p) - qualquer numero abaixo disso nao carrega
        # informacao sobre o modelo, so sobre a taxa-base.
        pv = float(y.mean())
        linhas.append(dict(**base, modelo="tudo_ligado", f1=2 * pv / (1 + pv),
                           f1_oraculo=np.nan, auc=np.nan))
    return linhas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aparelhos", default="freezer,washing machine")
    p.add_argument("--taxas", default="5min,15min")
    p.add_argument("--rotulos", default="media,nativo")
    p.add_argument("--sementes", type=int, default=3)
    p.add_argument("--epocas", type=int, default=300)
    p.add_argument("--paciencia", type=int, default=20)
    p.add_argument("--vista", default="bruto", help="vista de entrada (bruto, casa, z, arcsinh)")
    p.add_argument("--patches", default="4", help="tamanhos de patch a comparar, ex 1,2,4")
    p.add_argument("--arqs", default="linear",
                   help="linear | completo | cabeca | duplo_soma (ver modelo_sgn.py)")
    p.add_argument("--saida", default="artigo.csv")
    args = p.parse_args()

    cfg = Config(epocas=args.epocas, paciencia=args.paciencia)
    SAIDA.mkdir(exist_ok=True)
    linhas = []
    for aparelho in args.aparelhos.split(","):
        for taxa in args.taxas.split(","):
            for rotulo in args.rotulos.split(","):
                for arq in args.arqs.split(","):
                 for patch in [int(x) for x in args.patches.split(",")]:
                  for s in range(args.sementes):
                    novas = rodar(aparelho.strip(), taxa.strip(), rotulo.strip(), cfg, s,
                                  vista=[v.strip() for v in args.vista.split("+")], patch=patch,
                                  arq=arq.strip())
                    linhas += novas
                    for r in novas:
                        if r["modelo"] == "rede":
                            print(f"  {aparelho:16s} {taxa:6s} {arq:10s} p{patch} s{s} "
                                  f"{r['alvo']:22s} F1={r['f1']:.3f} "
                                  f"(ora {r['f1_oraculo']:.3f})  prev={100*r['prevalencia']:.1f}%  "
                                  f"ep={r['epocas']:3d}", flush=True)
                    pd.DataFrame(linhas).to_csv(SAIDA / args.saida, index=False, encoding="utf-8")

    d = pd.DataFrame(linhas)
    print("\n=== F1 media sobre as sementes ===")
    print(d.groupby(["aparelho", "taxa", "rotulo", "alvo", "modelo"]).agg(
        f1=("f1", "mean"), dp=("f1", "std"), prev=("prevalencia", "mean")
    ).round(3).to_string())
    print(f"\n-> {SAIDA / args.saida}")


if __name__ == "__main__":
    main()
