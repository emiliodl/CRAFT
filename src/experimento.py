"""Ablacao: treina na casa 1 do UK-DALE, testa na casa 2, um modelo por aparelho.

Bracos (todos seq2seq, janela de 1 dia, rotulo a cada 5 min):

  piso            regressao logistica sobre as 4 vistas no proprio instante,
                  sem contexto e sem rede - o chao que a arquitetura tem que bater
  bruto           1 encoder, so o agregado cru
  base            1 encoder, so a base removida
  z               1 encoder, so o z robusto
  arcsinh         1 encoder, so o arcsinh
  3x_bruto        3 encoders, TODOS no agregado cru, fusao por porta
                  (controle de capacidade: isola "3 vistas" de "3x parametros")
  3v_concat       3 vistas, fusao por concatenacao
  3v_soma         3 vistas, fusao por soma
  3v_porta        3 vistas, fusao por porta          <- modelo principal
  4v_porta        3 vistas + o bruto, fusao por porta
  3v_porta_tfc    3v_porta com pre-treino contrastivo (L_v + lambda L_C) antes
  3v_porta_conj   3v_porta com L_v + lambda L_C somadas a BCE durante todo o treino

Uso:
    python src/experimento.py --rapido            # 1 semente, 15 epocas
    python src/experimento.py                     # 3 sementes, sem teto util de epocas
    python src/experimento.py --aparelhos fridge --bracos 3v_porta,3v_porta_tfc
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dados import VISTAS, carregar, dividir_no_tempo, normalizador          # noqa: E402
from modelo import Classificador, contar_parametros                          # noqa: E402
from treino import Config, Fonte, avaliar, melhor_limiar, pre_treinar, prever, treinar  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "resultados"

APARELHOS = ("dish washer", "fridge", "washing machine")
LARGURA = 288          # 1 dia
PASSO_TREINO = 36      # 8 janelas por dia no treino
PASSO_AVAL = 72        # 4 previsoes por instante na avaliacao

TRES = ["base", "z", "arcsinh"]

BRACOS: dict[str, dict] = {
    "piso":          dict(tipo="reglog"),
    "bruto":         dict(vistas=["bruto"], fusao="soma"),
    "base":          dict(vistas=["base"], fusao="soma"),
    "z":             dict(vistas=["z"], fusao="soma"),
    "arcsinh":       dict(vistas=["arcsinh"], fusao="soma"),
    "3x_bruto":      dict(vistas=["bruto", "bruto", "bruto"], fusao="porta"),
    "3v_concat":     dict(vistas=TRES, fusao="concat"),
    "3v_soma":       dict(vistas=TRES, fusao="soma"),
    "3v_porta":      dict(vistas=TRES, fusao="porta"),
    "4v_porta":      dict(vistas=["bruto"] + TRES, fusao="porta"),
    "3v_porta_tfc":  dict(vistas=TRES, fusao="porta", pre=10),
    "3v_porta_conj": dict(vistas=TRES, fusao="porta", conj=0.1),
}


def indices_de(vistas: list[str]) -> list[int]:
    return [VISTAS.index(v) for v in vistas]


def piso_reglog(tr, va, te, largura: int) -> dict:
    """Regressao logistica nas 4 vistas do instante, sem contexto nenhum."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from treino import f1_positiva

    def matriz(s):
        # os MESMOS instantes que as redes veem: tudo que alguma janela cobre,
        # e nao so os centros - senao a comparacao e contra outro conjunto
        meia = largura // 2
        idx = np.unique(np.clip(s.centros[:, None] + np.arange(-meia, largura - meia)[None, :],
                                0, s.vistas.shape[1] - 1))
        return s.vistas[:, idx].T, s.rotulo[idx]

    xtr, ytr = matriz(tr)
    xva, yva = matriz(va)
    xte, yte = matriz(te)
    m, dv = xtr.mean(0), xtr.std(0) + 1e-6
    rl = LogisticRegression(max_iter=1000).fit((xtr - m) / dv, ytr)
    pva = rl.predict_proba((xva - m) / dv)[:, 1]
    pte = rl.predict_proba((xte - m) / dv)[:, 1]
    limiar, _ = melhor_limiar(yva, pva)
    _, f1_ora = melhor_limiar(yte, pte)
    return dict(f1=f1_positiva(yte, pte, limiar), f1_oraculo=f1_ora,
                ap=float(average_precision_score(yte, pte)),
                auc=float(roc_auc_score(yte, pte)), prevalencia=float(yte.mean()),
                limiar=limiar, parametros=int(xtr.shape[1] + 1))


def rodar(aparelho: str, braco: str, semente: int, cfg_base: Config) -> dict:
    spec = BRACOS[braco]
    torch.manual_seed(semente)
    np.random.seed(semente)

    origem = carregar("uk_dale", aparelho, "casa 1", LARGURA, PASSO_TREINO)
    alvo = carregar("uk_dale", aparelho, "casa 2", LARGURA, PASSO_AVAL)
    tr, va = dividir_no_tempo(origem, 0.2, LARGURA)

    if spec.get("tipo") == "reglog":
        r = piso_reglog(tr, va, alvo, LARGURA)
        r.update(aparelho=aparelho, braco=braco, semente=semente, epocas=0, segundos=0.0)
        return r

    vistas = spec["vistas"]
    idx = indices_de(vistas)
    media, desvio = normalizador(tr, LARGURA)
    f_tr = Fonte(tr, idx, LARGURA, media, desvio)
    f_va = Fonte(va, idx, LARGURA, media, desvio)
    f_te = Fonte(alvo, idx, LARGURA, media, desvio)

    modelo = Classificador(vistas, LARGURA, fusao=spec["fusao"]).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    cfg = Config(**{**cfg_base.__dict__})
    cfg.epocas_contrastivas = spec.get("pre", 0)
    cfg.peso_contrastivo_fino = spec.get("conj", 0.0)

    t0 = time.time()
    hist_pre = pre_treinar(modelo, f_tr, cfg) if cfg.epocas_contrastivas else []
    info = treinar(modelo, f_tr, f_va, cfg)
    r_te = avaliar(modelo, f_te, limiar=info["limiar"])
    r_va = avaliar(modelo, f_va, limiar=info["limiar"])
    segundos = time.time() - t0

    pesos = None
    if spec["fusao"] == "porta":
        x, _, _ = f_te.lote(torch.arange(min(256, len(f_te)), device=f_te.vistas.device))
        with torch.no_grad():
            pesos = modelo.pesos_das_vistas(x).cpu().numpy().round(3).tolist()

    return dict(
        aparelho=aparelho, braco=braco, semente=semente, vistas="+".join(vistas),
        fusao=spec["fusao"], f1=r_te.f1, f1_oraculo=r_te.f1_oraculo, ap=r_te.ap, auc=r_te.auc,
        prevalencia=r_te.prevalencia, limiar=info["limiar"],
        f1_val=info["f1_val"], ap_val=info["ap_val"], f1_val_limiar=r_va.f1,
        melhor_epoca=info["melhor_epoca"], epocas=len(info["historico"]),
        parametros=contar_parametros(modelo), pesos_vistas=pesos, segundos=round(segundos, 1),
        l_consistencia_final=(hist_pre[-1]["l_consistencia"] if hist_pre else None),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aparelhos", default=",".join(APARELHOS))
    p.add_argument("--bracos", default=",".join(BRACOS))
    p.add_argument("--sementes", type=int, default=3)
    p.add_argument("--epocas", type=int, default=300)
    p.add_argument("--paciencia", type=int, default=30)
    p.add_argument("--rapido", action="store_true")
    p.add_argument("--saida", default="ablacao.csv")
    p.add_argument("--verboso", action="store_true")
    args = p.parse_args()

    if args.rapido:
        args.sementes, args.epocas, args.paciencia = 1, 15, 4

    cfg = Config(epocas=args.epocas, paciencia=args.paciencia, verboso=args.verboso)
    SAIDA.mkdir(exist_ok=True)
    caminho = SAIDA / args.saida

    linhas = []
    for aparelho in args.aparelhos.split(","):
        for braco in args.bracos.split(","):
            for s in range(args.sementes):
                r = rodar(aparelho.strip(), braco.strip(), s, cfg)
                linhas.append(r)
                print(f"{aparelho:16s} {braco:14s} s{s}  F1={r['f1']:.3f} "
                      f"(ora {r['f1_oraculo']:.3f})  AP={r['ap']:.3f}  AUC={r['auc']:.3f}  "
                      f"ep={r['epocas']:3d}  {r['segundos']:.0f}s", flush=True)
                pd.DataFrame(linhas).to_csv(caminho, index=False, encoding="utf-8")

    d = pd.DataFrame(linhas)
    print("\n=== media sobre as sementes ===")
    print(d.groupby(["aparelho", "braco"]).agg(
        f1=("f1", "mean"), f1_dp=("f1", "std"), ap=("ap", "mean"),
        auc=("auc", "mean")).round(3).to_string())
    (caminho.with_suffix(".json")).write_text(json.dumps(linhas, indent=1, default=str), encoding="utf-8")
    print(f"\n-> {caminho}")


if __name__ == "__main__":
    main()
