"""TimesFM congelado + sonda, contra o nosso seq2point, no protocolo do artigo.

Mesmos alvos, mesma divisao treino/validacao, mesmo criterio de parada e de
limiar (F1 na validacao da casa de origem; o alvo nunca entra em decisao). A
unica coisa que muda e de onde vem a representacao.

  timesfm   backbone congelado de 231M + sonda de ~130 mil parametros
  nosso     ClassificadorPonto, 353 mil parametros, treinado do zero

Uso:
    python src/experimento_timesfm.py --aparelhos "washing machine" --horas_ctx 24
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

from experimento_artigo import JANELAS_POR_DIA, carregar, dividir              # noqa: E402
from ponto import alvos_validos                                               # noqa: E402
from timesfm_sonda import (FonteEmbedding, PATCH, Sonda, carregar_backbone,    # noqa: E402
                           extrair)
from treino import DISPOSITIVO, f1_positiva, melhor_limiar                    # noqa: E402

SAIDA = Path(__file__).resolve().parents[1] / "resultados"


def janelas_de(s, antes: int, depois: int, passo: int) -> tuple[np.ndarray, np.ndarray]:
    """(M, L) janelas do agregado em W e o rotulo do instante alvo."""
    alvos = alvos_validos(s, antes, depois, passo)
    if len(alvos) == 0:
        return np.zeros((0, antes + depois + 1), np.float32), np.zeros(0, np.float32)
    desloc = np.arange(-antes, depois + 1)
    idx = alvos[:, None] + desloc[None, :]
    # vista 0 = `bruto`, o agregado em W: o TimesFM normaliza por janela, entao
    # nao se deve entregar a ele uma serie ja padronizada pelo treino.
    return s.vistas[0][idx].astype(np.float32), s.rotulo[alvos].astype(np.float32)


def treinar_sonda(f_tr, f_va, epocas: int, paciencia: int, taxa: float,
                  lote: int, pooling: str = "atencao") -> tuple[Sonda, float, int]:
    modelo = Sonda(pooling=pooling).to(DISPOSITIVO)
    opt = torch.optim.AdamW(modelo.parameters(), lr=taxa, weight_decay=1e-4)
    agenda = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                        patience=10)
    melhor, estado, espera, ep = -1.0, None, 0, 0
    for ep in range(epocas):
        modelo.train()
        ordem = torch.randperm(len(f_tr), device=DISPOSITIVO)
        for i in range(0, len(ordem) - lote + 1, lote):
            x, y, _ = f_tr.lote(ordem[i:i + lote])
            perda = torch.nn.functional.binary_cross_entropy_with_logits(modelo(x), y)
            opt.zero_grad(set_to_none=True)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
        y, p = prever(modelo, f_va)
        f1 = melhor_limiar(y, p)[1]
        agenda.step(f1)
        if f1 > melhor:
            melhor, espera = f1, 0
            estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            espera += 1
            if espera >= paciencia:
                break
    if estado is not None:
        modelo.load_state_dict(estado)
    return modelo, melhor, ep + 1


@torch.no_grad()
def prever(modelo: Sonda, fonte: FonteEmbedding, lote: int = 512):
    modelo.eval()
    saidas = []
    for i in range(0, len(fonte), lote):
        pos = torch.arange(i, min(i + lote, len(fonte)), device=DISPOSITIVO)
        x, _, _ = fonte.lote(pos)
        saidas.append(torch.sigmoid(modelo(x)).squeeze(1))
    return fonte.rotulo.cpu().numpy(), torch.cat(saidas).cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aparelhos", default="washing machine")
    p.add_argument("--taxa", default="5min")
    p.add_argument("--horas_ctx", type=float, default=24.0)
    p.add_argument("--camadas", default="5,10,15,20",
                   help="camadas do TimesFM a sondar (1-20); 20 e a saida final")
    p.add_argument("--alinhamentos", default="centrado,causal")
    p.add_argument("--poolings", default="atencao,ultimo")
    p.add_argument("--passo_treino", type=int, default=4)
    p.add_argument("--sementes", type=int, default=3)
    p.add_argument("--epocas", type=int, default=200)
    p.add_argument("--paciencia", type=int, default=15)
    p.add_argument("--taxa_lr", type=float, default=1e-3)
    p.add_argument("--lote", type=int, default=128)
    p.add_argument("--saida", default="timesfm.csv")
    args = p.parse_args()

    por_dia = JANELAS_POR_DIA[args.taxa]
    ctx = max(PATCH, int(round(args.horas_ctx * por_dia / 24 / PATCH)) * PATCH)
    camadas = [int(c) for c in args.camadas.split(",")]
    print(f"contexto = {ctx} amostras, {ctx // PATCH} patches | camadas {camadas}",
          flush=True)
    print("carregando TimesFM...", flush=True)
    net = carregar_backbone(DISPOSITIVO)
    linhas = []

    for aparelho in args.aparelhos.split(","):
        aparelho = aparelho.strip()
        ap = aparelho.replace(" ", "_")
        treino_total = carregar(f"refit_casa_8__{ap}__{args.taxa}__treino.npz",
                                "media", por_dia, 1)
        tr, va = dividir(treino_total, 0.15, por_dia)
        conjuntos = {"treino": tr, "val": va}
        intra = carregar(f"refit_casa_8__{ap}__{args.taxa}__teste.npz", "media", por_dia, 1)
        if intra is not None:
            conjuntos["intra: refit casa 8"] = intra
        for n in (1, 2, 5):
            s_ = carregar(f"eco_casa_{n}__{ap}__{args.taxa}__cross.npz", "media", por_dia, 1)
            if s_ is not None:
                conjuntos[f"cross: eco casa {n}"] = s_

        for alinhamento in args.alinhamentos.split(","):
            alinhamento = alinhamento.strip()
            if alinhamento == "centrado":
                antes, depois = ctx // 2, ctx - ctx // 2 - 1
            else:                       # causal: a janela TERMINA no alvo
                antes, depois = ctx - 1, 0

            emb_por_conj, rot_por_conj, t0 = {}, {}, time.time()
            for nome, s_ in conjuntos.items():
                passo = args.passo_treino if nome == "treino" else (
                    max(1, args.passo_treino // 2) if nome == "val" else 1)
                x, y = janelas_de(s_, antes, depois, passo)
                if len(x) == 0:
                    continue
                emb_por_conj[nome] = extrair(net, x, camadas, dispositivo=DISPOSITIVO)
                rot_por_conj[nome] = y
                print(f"  [{alinhamento}] {nome:22s} {len(x):7d} janelas "
                      f"({time.time() - t0:.0f}s)", flush=True)

            for camada in camadas:
                fontes = {n: FonteEmbedding(e[camada], rot_por_conj[n], DISPOSITIVO)
                          for n, e in emb_por_conj.items()}
                for pooling in args.poolings.split(","):
                    pooling = pooling.strip()
                    for semente in range(args.sementes):
                        torch.manual_seed(semente)
                        np.random.seed(semente)
                        t1 = time.time()
                        modelo, _, epocas = treinar_sonda(
                            fontes["treino"], fontes["val"], args.epocas,
                            args.paciencia, args.taxa_lr, args.lote, pooling)
                        y_va, p_va = prever(modelo, fontes["val"])
                        limiar = melhor_limiar(y_va, p_va)[0]
                        seg = time.time() - t1
                        for nome, f in fontes.items():
                            if nome in ("treino", "val"):
                                continue
                            y, p = prever(modelo, f)
                            linhas.append(dict(
                                modelo="timesfm", aparelho=aparelho, taxa=args.taxa,
                                alinhamento=alinhamento, camada=camada, pooling=pooling,
                                ctx=ctx, alvo=nome, semente=semente,
                                f1=f1_positiva(y, p, limiar),
                                f1_oraculo=melhor_limiar(y, p)[1],
                                auc=(float(roc_auc_score(y, p))
                                     if y.min() != y.max() else float("nan")),
                                prevalencia=round(float(y.mean()), 4),
                                limiar=round(limiar, 4), n_alvos=len(f),
                                epocas=epocas, segundos=round(seg, 1)))
                    ult = [r for r in linhas[-2 * args.sementes:]
                           if r["alvo"].startswith("intra")]
                    if ult:
                        print(f"    cam{camada:2d} {pooling:8s} intra F1="
                              f"{np.mean([r['f1'] for r in ult]):.3f}", flush=True)
                del fontes
                torch.cuda.empty_cache()
            SAIDA.mkdir(exist_ok=True)
            pd.DataFrame(linhas).to_csv(SAIDA / args.saida, index=False, encoding="utf-8")

    d = pd.DataFrame(linhas)
    print("\n=== melhores combinacoes por celula ===")
    print(d.groupby(["aparelho", "alvo", "alinhamento", "camada", "pooling"]).agg(
        f1=("f1", "mean"), auc=("auc", "mean")).round(3).sort_values(
        "f1", ascending=False).head(25).to_string())
    print(f"\n-> {SAIDA / args.saida}")


if __name__ == "__main__":
    main()
