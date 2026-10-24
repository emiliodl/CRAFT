"""TimesFM com mistura de camadas, o nosso seq2point, e o conjunto dos dois.

Motivado pela varredura anterior: os dois modelos erram em lugares diferentes.
O TimesFM triplica a F1 de transferencia da televisao (0,461 contra 0,159) e
perde na maquina de lavar (0,100 contra 0,194). A televisao e consumo sustentado
com padrao diario - parecido com o que o TimesFM viu no pre-treino; a maquina de
lavar e evento raro e abrupto, mais proximo de anomalia que de padrao previsivel.

Quatro linhas por semente:

  nosso      ClassificadorPonto, janela centrada, treinado do zero
  timesfm    backbone congelado + SondaMistura sobre varias camadas
  conjunto   media das probabilidades dos dois
  (cada um com f1, f1 pelo limiar do EM de prevalencia, e oraculo)

Os dois modelos usam alinhamentos diferentes (centrado vs causal), entao os
conjuntos de alvos validos NAO coincidem. Aqui as previsoes sao indexadas pela
posicao na serie e o conjunto e calculado apenas na INTERSECCAO - caso
contrario a media compararia instantes diferentes.

Uso:
    python src/experimento_conjunto.py --aparelhos "television,washing machine"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dados import VISTAS                                                       # noqa: E402
from dominio import limiar_por_prevalencia, prevalencia_em                     # noqa: E402
from experimento_artigo import JANELAS_POR_DIA, carregar, dividir, normalizador  # noqa: E402
from experimento_timesfm import janelas_de, prever as prever_sonda             # noqa: E402
from ponto import ClassificadorPonto, FontePonto, alvos_validos                # noqa: E402
from timesfm_sonda import (FonteMultiCamada, PATCH, SondaMistura,              # noqa: E402
                           carregar_backbone, extrair)
from treino import (Config, DISPOSITIVO, f1_positiva, melhor_limiar,           # noqa: E402
                    prever_com_indices, treinar)

SAIDA = Path(__file__).resolve().parents[1] / "resultados"


def treinar_mistura(f_tr, f_va, cfg: Config, n_camadas: int, pooling: str):
    modelo = SondaMistura(n_camadas, pooling=pooling).to(DISPOSITIVO)
    opt = torch.optim.AdamW(modelo.parameters(), lr=cfg.taxa, weight_decay=cfg.decaimento)
    agenda = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                        patience=10)
    melhor, estado, espera = -1.0, None, 0
    for _ in range(cfg.epocas):
        modelo.train()
        ordem = torch.randperm(len(f_tr), device=DISPOSITIVO)
        for i in range(0, len(ordem) - cfg.lote + 1, cfg.lote):
            x, y, _ = f_tr.lote(ordem[i:i + cfg.lote])
            perda = torch.nn.functional.binary_cross_entropy_with_logits(modelo(x), y)
            opt.zero_grad(set_to_none=True)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
        y, p = prever_sonda(modelo, f_va)
        f1 = melhor_limiar(y, p)[1]
        agenda.step(f1)
        if f1 > melhor:
            melhor, espera = f1, 0
            estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            espera += 1
            if espera >= cfg.paciencia:
                break
    if estado is not None:
        modelo.load_state_dict(estado)
    return modelo


def metricas(y, p, limiar, prev_origem):
    """F1 no limiar da origem, no limiar do EM de prevalencia, e no oraculo."""
    prev = prevalencia_em(p, prev_origem)
    return dict(f1=f1_positiva(y, p, limiar),
                f1_em=f1_positiva(y, p, limiar_por_prevalencia(p, prev)),
                f1_oraculo=melhor_limiar(y, p)[1], prev_em=round(prev, 4),
                auc=float(roc_auc_score(y, p)) if y.min() != y.max() else float("nan"),
                prevalencia=round(float(y.mean()), 4), n=len(y))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aparelhos", default="television,washing machine,freezer")
    p.add_argument("--taxa", default="5min")
    p.add_argument("--horas_ctx", type=float, default=24.0)
    p.add_argument("--camadas", default="5,10,15,20")
    p.add_argument("--pooling", default="ultimo")
    p.add_argument("--passo_treino", type=int, default=4)
    p.add_argument("--sementes", type=int, default=3)
    p.add_argument("--epocas", type=int, default=200)
    p.add_argument("--paciencia", type=int, default=15)
    p.add_argument("--saida", default="conjunto.csv")
    args = p.parse_args()

    por_dia = JANELAS_POR_DIA[args.taxa]
    ctx = max(PATCH, int(round(args.horas_ctx * por_dia / 24 / PATCH)) * PATCH)
    camadas = [int(c) for c in args.camadas.split(",")]
    cfg = Config(epocas=args.epocas, paciencia=args.paciencia)
    # centrado para o nosso modelo, causal para o TimesFM (decoder-only)
    c_antes, c_depois = ctx // 2, ctx - ctx // 2 - 1
    t_antes, t_depois = ctx - 1, 0

    print(f"ctx={ctx} ({ctx // PATCH} patches) | camadas {camadas}", flush=True)
    net = carregar_backbone(DISPOSITIVO)
    linhas = []

    for aparelho in args.aparelhos.split(","):
        aparelho = aparelho.strip()
        ap = aparelho.replace(" ", "_")
        total = carregar(f"refit_casa_8__{ap}__{args.taxa}__treino.npz", "media", por_dia, 1)
        tr, va = dividir(total, 0.15, por_dia)
        conjuntos = {"treino": tr, "val": va}
        intra = carregar(f"refit_casa_8__{ap}__{args.taxa}__teste.npz", "media", por_dia, 1)
        if intra is not None:
            conjuntos["intra: refit casa 8"] = intra
        for n in (1, 2, 5):
            s = carregar(f"eco_casa_{n}__{ap}__{args.taxa}__cross.npz", "media", por_dia, 1)
            if s is not None:
                conjuntos[f"cross: eco casa {n}"] = s

        # ---- embeddings do TimesFM (causal) ----
        t0 = time.time()
        fontes_tfm, alvos_tfm = {}, {}
        for nome, s in conjuntos.items():
            passo = args.passo_treino if nome == "treino" else (
                max(1, args.passo_treino // 2) if nome == "val" else 1)
            x, y = janelas_de(s, t_antes, t_depois, passo)
            if len(x) == 0:
                continue
            emb = extrair(net, x, camadas, dispositivo=DISPOSITIVO)
            fontes_tfm[nome] = FonteMultiCamada(emb, y, DISPOSITIVO)
            alvos_tfm[nome] = alvos_validos(s, t_antes, t_depois, passo)
        print(f"  {aparelho}: embeddings em {time.time() - t0:.0f}s", flush=True)

        # ---- fontes do nosso modelo (centrado) ----
        idx_v = [VISTAS.index("casa")]
        media, desvio = normalizador(tr, por_dia)
        fontes_nos = {}
        for nome, s in conjuntos.items():
            passo = args.passo_treino if nome == "treino" else (
                max(1, args.passo_treino // 2) if nome == "val" else 1)
            f = FontePonto(s, idx_v, c_antes, c_depois, media, desvio, passo)
            if len(f) > 0:
                fontes_nos[nome] = f

        for semente in range(args.sementes):
            torch.manual_seed(semente)
            np.random.seed(semente)
            nosso = ClassificadorPonto(["casa"], c_antes + c_depois + 1).to(DISPOSITIVO)
            info = treinar(nosso, fontes_nos["treino"], fontes_nos["val"], cfg)
            torch.manual_seed(semente)
            np.random.seed(semente)
            sonda = treinar_mistura(fontes_tfm["treino"], fontes_tfm["val"], cfg,
                                    len(camadas), args.pooling)
            pesos = torch.softmax(sonda.pesos.detach(), 0).cpu().numpy().round(3)

            # limiares e prevalencia de origem, os dois vindos SO da validacao
            yv_n, pv_n, iv_n = prever_com_indices(nosso, fontes_nos["val"])
            lim_n = melhor_limiar(yv_n, pv_n)[0]
            yv_t, pv_t = prever_sonda(sonda, fontes_tfm["val"])
            lim_t = melhor_limiar(yv_t, pv_t)[0]
            prev_origem = float(yv_n.mean())
            # conjunto: media na interseccao dos alvos da validacao
            comum_v, ia, ib = np.intersect1d(iv_n, alvos_tfm["val"],
                                             return_indices=True)
            lim_c = melhor_limiar(yv_n[ia], (pv_n[ia] + pv_t[ib]) / 2)[0]

            for nome in fontes_tfm:
                if nome in ("treino", "val"):
                    continue
                y_n, p_n, i_n = prever_com_indices(nosso, fontes_nos[nome])
                y_t, p_t = prever_sonda(sonda, fontes_tfm[nome])
                comum, ja, jb = np.intersect1d(i_n, alvos_tfm[nome], return_indices=True)
                casos = {
                    "nosso": (y_n, p_n, lim_n),
                    "timesfm": (y_t, p_t, lim_t),
                    # o conjunto e medido SO na interseccao; para nao comparar
                    # populacoes diferentes, os dois isolados tambem entram
                    # restritos a ela nas linhas de sufixo `(int)`
                    "conjunto": (y_n[ja], (p_n[ja] + p_t[jb]) / 2, lim_c),
                    "nosso (int)": (y_n[ja], p_n[ja], lim_n),
                    "timesfm (int)": (y_t[jb], p_t[jb], lim_t),
                }
                for modelo, (y, pr, lim) in casos.items():
                    if len(y) == 0 or y.min() == y.max():
                        continue
                    linhas.append(dict(
                        modelo=modelo, aparelho=aparelho, taxa=args.taxa, ctx=ctx,
                        alvo=nome, semente=semente, limiar=round(float(lim), 4),
                        pesos_camada=str(pesos), **metricas(y, pr, lim, prev_origem)))
                print(f"  {aparelho:16s} s{semente} {nome:20s} "
                      + "  ".join(f"{m}={[r for r in linhas if r['modelo'] == m][-1]['f1']:.3f}"
                                  for m in ("nosso", "timesfm", "conjunto")), flush=True)
            SAIDA.mkdir(exist_ok=True)
            pd.DataFrame(linhas).to_csv(SAIDA / args.saida, index=False, encoding="utf-8")

    d = pd.DataFrame(linhas)
    print("\n=== media sobre as sementes ===")
    print(d.groupby(["aparelho", "alvo", "modelo"]).agg(
        f1=("f1", "mean"), dp=("f1", "std"), em=("f1_em", "mean"),
        ora=("f1_oraculo", "mean"), auc=("auc", "mean")).round(3).to_string())
    print(f"\n-> {SAIDA / args.saida}")


if __name__ == "__main__":
    main()
