"""Leave-one-house-out: treina em todas as casas menos uma, testa na que sobrou.

Corrige as duas causas medidas no experimento casa 1 -> casa 2:

  1. UMA casa de origem nao deixa a rede separar "o que e lava-loucas" de "o que
     e o lava-loucas desta casa". Aqui sao ~15-20 casas de origem por fold.
  2. A vista `z` normaliza por MAD de janela movel e apaga a amplitude, o que
     derruba o AUC de carga alta para 0,67. A vista `casa` troca isso por uma
     escala constante do domicilio (p95 do proprio agregado), que tira a escala
     da casa sem tirar a amplitude do evento.

Bracos:
  piso        regressao logistica nas vistas do instante (o chao)
  bruto       1 encoder, agregado cru
  3v_antigo   base + z + arcsinh          <- o conjunto do experimento anterior
  3v_novo     base + casa + arcsinh       <- z trocado pela normalizacao por casa
  4v_novo     bruto + base + casa + arcsinh

Um modelo por aparelho, como sempre. A casa-alvo nao entra em nenhuma decisao:
nem no treino, nem na parada, nem na escolha do limiar. O unico uso do alvo
antes da medicao e o p95 do seu proprio agregado, que nao precisa de submedidor.

Uso:
    python src/experimento_loho.py --aparelhos fridge --folds 8
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

from dados import VISTAS, carregar, dividir_no_tempo, juntar, listar, normalizador   # noqa: E402
from modelo import Classificador, contar_parametros                                  # noqa: E402
from treino import (Config, Fonte, avaliar, f1_positiva, melhor_limiar,              # noqa: E402
                    treinar)
from diagnostico import f1_na_prevalencia                                            # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "resultados"

LARGURA = 288
PASSO_TREINO = 144     # 2 janelas por dia: ha ~5x mais dias que antes
PASSO_AVAL = 72
PREV_REFERENCIA = 0.05  # prevalencia em que a F1 padronizada e reportada

BRACOS: dict[str, dict] = {
    "piso":      dict(tipo="reglog"),
    "bruto":     dict(vistas=["bruto"], fusao="soma"),
    "3v_antigo": dict(vistas=["base", "z", "arcsinh"], fusao="porta"),
    "3v_novo":   dict(vistas=["base", "casa", "arcsinh"], fusao="porta"),
    "4v_novo":   dict(vistas=["bruto", "base", "casa", "arcsinh"], fusao="porta"),
}


def piso_reglog(tr, va, te) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    meia = LARGURA // 2
    def matriz(s):
        idx = np.unique(np.clip(s.centros[:, None] + np.arange(-meia, LARGURA - meia)[None, :],
                                0, s.vistas.shape[1] - 1))
        return s.vistas[:, idx].T, s.rotulo[idx]

    xtr, ytr = matriz(tr)
    xva, yva = matriz(va)
    xte, yte = matriz(te)
    m, dv = xtr.mean(0), xtr.std(0) + 1e-6
    rl = LogisticRegression(max_iter=300).fit((xtr - m) / dv, ytr)
    pva = rl.predict_proba((xva - m) / dv)[:, 1]
    pte = rl.predict_proba((xte - m) / dv)[:, 1]
    limiar, _ = melhor_limiar(yva, pva)
    _, ora = melhor_limiar(yte, pte)
    return dict(f1=f1_positiva(yte, pte, limiar), f1_oraculo=ora,
                ap=float(average_precision_score(yte, pte)),
                auc=float(roc_auc_score(yte, pte)), prevalencia=float(yte.mean()),
                f1_padronizada=f1_na_prevalencia(yte, pte, PREV_REFERENCIA),
                limiar=limiar, parametros=int(xtr.shape[1] + 1), epocas=0, f1_val=np.nan)


def rodar(aparelho: str, alvo: tuple[str, str], fontes: list[tuple[str, str]],
          braco: str, semente: int, cfg: Config) -> dict:
    spec = BRACOS[braco]
    torch.manual_seed(semente)
    np.random.seed(semente)

    s_alvo = carregar(alvo[0], aparelho, alvo[1], LARGURA, PASSO_AVAL)
    s_fontes = juntar([carregar(d, aparelho, c, LARGURA, PASSO_TREINO) for d, c in fontes],
                      f"{len(fontes)} casas")
    tr, va = dividir_no_tempo(s_fontes, 0.2, LARGURA)

    t0 = time.time()
    if spec.get("tipo") == "reglog":
        r = piso_reglog(tr, va, s_alvo)
    else:
        vistas = spec["vistas"]
        idx = [VISTAS.index(v) for v in vistas]
        media, desvio = normalizador(tr, LARGURA)
        f_tr = Fonte(tr, idx, LARGURA, media, desvio)
        f_va = Fonte(va, idx, LARGURA, media, desvio)
        f_te = Fonte(s_alvo, idx, LARGURA, media, desvio)

        modelo = Classificador(vistas, LARGURA, fusao=spec["fusao"]).to(f_tr.vistas.device)
        info = treinar(modelo, f_tr, f_va, cfg)
        from treino import prever
        y, p = prever(modelo, f_te)
        _, ora = melhor_limiar(y, p)
        r_te = avaliar(modelo, f_te, limiar=info["limiar"])
        r = dict(f1=r_te.f1, f1_oraculo=ora, ap=r_te.ap, auc=r_te.auc,
                 prevalencia=r_te.prevalencia,
                 f1_padronizada=f1_na_prevalencia(y, p, PREV_REFERENCIA),
                 limiar=info["limiar"], parametros=contar_parametros(modelo),
                 epocas=len(info["historico"]), f1_val=info["f1_val"])

    r.update(aparelho=aparelho, braco=braco, semente=semente,
             alvo=f"{alvo[0]}/{alvo[1]}", n_casas_treino=len(fontes),
             dias_treino=round(len(tr.centros) * PASSO_TREINO / 288, 0),
             segundos=round(time.time() - t0, 1))
    return r


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aparelhos", default="dish washer,fridge,washing machine")
    p.add_argument("--bracos", default=",".join(BRACOS))
    p.add_argument("--folds", type=int, default=8, help="quantas casas-alvo (as com mais dias)")
    p.add_argument("--alvos", default=None,
                   help="casas-alvo explicitas, ex 'uk_dale/casa 2' - ignora --folds")
    p.add_argument("--minimo_dias", type=float, default=100.0)
    p.add_argument("--sementes", type=int, default=1)
    p.add_argument("--epocas", type=int, default=300)
    p.add_argument("--paciencia", type=int, default=20)
    p.add_argument("--saida", default="loho.csv")
    args = p.parse_args()

    cfg = Config(epocas=args.epocas, paciencia=args.paciencia)
    SAIDA.mkdir(exist_ok=True)
    caminho = SAIDA / args.saida
    linhas = []

    for aparelho in args.aparelhos.split(","):
        aparelho = aparelho.strip()
        todas = listar(aparelho, args.minimo_dias)
        if args.alvos:
            pedidos = [tuple(a.strip().split("/")) for a in args.alvos.split(",")]
            alvos = [p for p in pedidos if p in [(d, c) for d, c, _ in todas]]
            if len(alvos) != len(pedidos):
                raise SystemExit(f"alvo fora da lista de {aparelho}: "
                                 f"{set(pedidos) - set(alvos)}")
        else:
            alvos = [(d, c) for d, c, _ in todas[:args.folds]]
        print(f"\n### {aparelho}: {len(todas)} casas com >= {args.minimo_dias:.0f} dias, "
              f"{len(alvos)} folds", flush=True)
        for alvo in alvos:
            fontes = [(d, c) for d, c, _ in todas if (d, c) != alvo]
            for braco in args.bracos.split(","):
                for s in range(args.sementes):
                    r = rodar(aparelho, alvo, fontes, braco.strip(), s, cfg)
                    linhas.append(r)
                    print(f"  {r['alvo']:22s} {braco:10s} F1={r['f1']:.3f} "
                          f"(ora {r['f1_oraculo']:.3f}, pad {r['f1_padronizada']:.3f})  "
                          f"AP={r['ap']:.3f} AUC={r['auc']:.3f}  prev={100*r['prevalencia']:.1f}%  "
                          f"ep={r['epocas']:3d}  {r['segundos']:.0f}s", flush=True)
                    pd.DataFrame(linhas).to_csv(caminho, index=False, encoding="utf-8")

    d = pd.DataFrame(linhas)
    print("\n=== media sobre os folds ===")
    print(d.groupby(["aparelho", "braco"]).agg(
        folds=("f1", "size"), f1=("f1", "mean"), dp=("f1", "std"),
        f1_pad=("f1_padronizada", "mean"), ap=("ap", "mean"), auc=("auc", "mean")
    ).round(3).to_string())
    print(f"\n-> {caminho}")


if __name__ == "__main__":
    main()
