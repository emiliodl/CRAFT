"""Seq2point no protocolo de Chatterjee & Heer, contra o seq2seq atual.

Mesmos dados, mesmas vistas, mesmo criterio de parada e de limiar. A unica
diferenca e a forma da tarefa:

  seq2seq    288 amostras entram, 288 rotulos saem; na inferencia, janelas
             sobrepostas com media (o que o projeto vinha fazendo)
  centrado   janela [X-L/2, X+L/2), um rotulo em X  (o Seq2Point original)
  causal     janela quase toda passada, com `--folga` amostras de futuro

Uso:
    python src/experimento_ponto.py --formatos centrado,causal --folga 1
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

from dados import VISTAS, juntar                                           # noqa: E402
from dominio import limiar_por_prevalencia, prevalencia_em                 # noqa: E402
from experimento_artigo import JANELAS_POR_DIA, carregar, dividir, normalizador  # noqa: E402
from ponto import ClassificadorPonto, FontePonto, SoCentro                 # noqa: E402
from treino import Config, avaliar, f1_positiva, melhor_limiar, prever_com_indices, treinar  # noqa: E402

SAIDA = Path(__file__).resolve().parents[1] / "resultados"


def rodar(aparelho: str, taxa: str, formato: str, folga: int, vista: list[str],
          patch: int, cfg: Config, semente: int, passo_treino: int,
          saida: int = 1, horas_ctx: float = 24.0,
          esquema_grupo: str | None = None) -> list[dict]:
    torch.manual_seed(semente)
    np.random.seed(semente)
    ap = aparelho.replace(" ", "_")
    largura = JANELAS_POR_DIA[taxa]          # amostras por dia; base da divisao treino/val
    # Contexto do seq2point, desacoplado do dia. Encurta-lo e o teste barato da
    # hipotese de fase: com 24 h o modelo consegue se localizar no dia pela
    # forma da curva de carga e aprender "maquina de lavar so de madrugada" -
    # regra que vale na REFIT casa 8 e nao vale na ECO. Com 2-4 h essa pista
    # some. Se o intra cair e o cross subir, a hipotese esta confirmada.
    ctx = max(patch, int(round(horas_ctx * largura / 24 / patch)) * patch)
    if formato == "centrado":
        antes, depois = ctx // 2, ctx - ctx // 2 - 1
    elif formato == "causal":
        antes, depois = ctx - 1 - folga, folga
    else:
        raise ValueError(formato)

    # o carregamento reaproveita o do seq2seq so para montar vistas/rotulo/t;
    # os alvos sao recalculados pela FontePonto com as margens certas
    treino_total = carregar(f"refit_casa_8__{ap}__{taxa}__treino.npz", "media", largura, 1)
    tr, va = dividir(treino_total, 0.15, largura)
    alvos: dict[str, object] = {}
    intra = carregar(f"refit_casa_8__{ap}__{taxa}__teste.npz", "media", largura, 1)
    if intra is not None:
        alvos["intra: refit casa 8"] = intra
    eco = []
    for n in (1, 2, 5):
        s = carregar(f"eco_casa_{n}__{ap}__{taxa}__cross.npz", "media", largura, 1)
        if s is not None:
            alvos[f"cross: eco casa {n}"] = s
            eco.append(s)
    # O artigo reporta UM numero para REFIT -> ECO, medido nas casas 1, 2 e 5
    # juntas. Este alvo reproduz isso: as series sao empilhadas e a F1 e
    # calculada sobre o conjunto todo, e nao por casa. `juntar` desloca o tempo
    # de cada casa para que a fronteira entre elas nunca pareca contigua.
    if len(eco) > 1:
        alvos["cross: eco agregada"] = juntar(eco, "eco agregada")

    idx = [VISTAS.index(v) for v in vista]
    media, desvio = normalizador(tr, largura)
    f_tr = FontePonto(tr, idx, antes, depois, media, desvio, passo_treino, saida,
                      esquema_grupo)
    f_va = FontePonto(va, idx, antes, depois, media, desvio, max(1, passo_treino // 2), saida)

    t0 = time.time()
    modelo = ClassificadorPonto(vista, antes + depois + 1, patch=patch,
                                saida=saida).to(f_tr.vistas.device)
    info = treinar(modelo, f_tr, f_va, cfg)
    segundos = time.time() - t0

    # A previsao continua sendo de um rotulo so. Com K > 1 o limiar tambem e
    # recalculado numa validacao de um rotulo so: senao duas coisas mudariam de
    # uma vez (a supervisao E a calibracao sobre previsoes vizinhas mediadas), e
    # nao daria para dizer qual moveu a F1.
    avaliado = SoCentro(modelo) if saida > 1 else modelo
    f_va1 = FontePonto(va, idx, antes, depois, media, desvio, max(1, passo_treino // 2), 1)
    y_va, p_va, _ = prever_com_indices(avaliado, f_va1)
    limiar = melhor_limiar(y_va, p_va)[0] if saida > 1 else info["limiar"]
    # Taxa de positivos que o modelo emite na validacao da FONTE, no seu limiar.
    # E uma propriedade so do modelo e da origem - nenhum rotulo do alvo entra.
    taxa_origem = float((p_va >= limiar).mean())

    linhas = []
    for nome, s in alvos.items():
        f_te = FontePonto(s, idx, antes, depois, media, desvio, 1, 1)
        if len(f_te) == 0:
            continue
        y, p, _ = prever_com_indices(avaliado, f_te)
        linhas.append(dict(
            aparelho=aparelho, taxa=taxa, formato=formato, folga=folga,
            vista="+".join(vista), patch=patch, saida=saida, horas_ctx=horas_ctx,
            ctx=ctx, aumento=cfg.aumento, dro=cfg.dro, grupos=esquema_grupo or "-",
            alvo=nome, semente=semente,
            f1=f1_positiva(y, p, limiar),
            # Coluna sem calibracao nenhuma, para espelhar o procedimento do
            # artigo: eles limiarizam a POTENCIA prevista em 10 W / 20 W fixos e
            # nao ajustam nada. O analogo para um classificador e o limiar
            # neutro de 0,5. Atencao: nao sao equivalentes em dificuldade - um
            # regressor que acerta a potencia cruza 20 W naturalmente quando o
            # aparelho liga, enquanto p >= 0,5 com 4% de prevalencia e uma
            # exigencia bem maior. A coluna serve como piso, nao como empate.
            f1_meio=f1_positiva(y, p, 0.5),
            # Limiar por casamento de taxa: escolhe no ALVO o corte que produz a
            # mesma fracao de positivos que o modelo emite na origem. Usa apenas
            # as probabilidades do alvo, nunca os rotulos dele, entao continua
            # sendo um procedimento honesto - e resolve o caso em que a AUC sobe
            # mas a F1 nao acompanha porque a calibracao da fonte nao transfere.
            f1_quantil=f1_positiva(y, p, float(np.quantile(p, 1 - taxa_origem))
                                   if 0 < taxa_origem < 1 else 0.5),
            # Casar a taxa da fonte falhou porque a taxa nao e propriedade do
            # modelo: ela mistura qualidade do classificador com prevalencia do
            # dominio, e a prevalencia vai de 4,5% (REFIT) para 7,4% (ECO). O EM
            # de prior shift estima a prevalencia do alvo pelas probabilidades,
            # sem rotulo nenhum, e o limiar sai dela.
            f1_em=f1_positiva(y, p, limiar_por_prevalencia(p, prev_em := prevalencia_em(
                p, float(y_va.mean())))),
            prev_em=round(prev_em, 4),
            taxa_origem=round(taxa_origem, 4),
            f1_oraculo=melhor_limiar(y, p)[1],
            auc=avaliar(avaliado, f_te, limiar=limiar).auc,
            prevalencia=round(float(y.mean()), 4), limiar=round(limiar, 4),
            n_alvos=len(f_te), n_treino=len(f_tr), rotulos_treino=len(f_tr) * saida,
            epocas=len(info["historico"]), segundos=round(segundos, 1)))
    return linhas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aparelhos", default="washing machine")
    p.add_argument("--taxas", default="5min,15min")
    p.add_argument("--formatos", default="centrado,causal")
    p.add_argument("--folga", type=int, default=1, help="amostras de futuro no modo causal")
    p.add_argument("--vista", default="casa")
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--passo_treino", type=int, default=4,
                   help="1 usa todos os alvos; 4 subamostra para o treino caber")
    p.add_argument("--horas_ctx", default="24",
                   help="contexto em horas, ex '24,4,2'; encurtar apaga a pista de hora do dia")
    p.add_argument("--aumento", default="nenhum", choices=["nenhum", "fourier"],
                   help="fourier randomiza a envoltoria de baixa frequencia")
    p.add_argument("--forca", type=float, default=0.5)
    p.add_argument("--corte", type=int, default=8, help="raias randomizadas (8 = periodos >= 3 h)")
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--dro", action="store_true", help="group-DRO sobre subdominios do calendario")
    p.add_argument("--grupos", default="estacao_dia",
                   choices=["estacao_dia", "estacao", "dia", "hora"])
    p.add_argument("--saidas", default="1",
                   help="K rotulos centrados no alvo como supervisao, ex '1,8,16'. "
                        "A previsao continua sendo de um rotulo so.")
    p.add_argument("--sementes", type=int, default=3)
    p.add_argument("--epocas", type=int, default=300)
    p.add_argument("--paciencia", type=int, default=15)
    p.add_argument("--saida", default="ponto.csv")
    args = p.parse_args()

    cfg = Config(epocas=args.epocas, paciencia=args.paciencia, aumento=args.aumento,
                 forca_aumento=args.forca, corte_fourier=args.corte,
                 jitter_fourier=args.jitter, dro=args.dro)
    esquema = args.grupos if args.dro else None
    SAIDA.mkdir(exist_ok=True)
    vista = [v.strip() for v in args.vista.split("+")]
    linhas = []
    for aparelho in args.aparelhos.split(","):
        for taxa in args.taxas.split(","):
            for formato in args.formatos.split(","):
              for k in [int(x) for x in args.saidas.split(",")]:
               for h in [float(x) for x in args.horas_ctx.split(",")]:
                for s in range(args.sementes):
                    novas = rodar(aparelho.strip(), taxa.strip(), formato.strip(), args.folga,
                                  vista, args.patch, cfg, s, args.passo_treino, k, h, esquema)
                    linhas += novas
                    for r in novas:
                        print(f"  {aparelho:16s} {taxa:6s} {h:>4.1f}h s{s} {r['alvo']:22s} "
                              f"F1={r['f1']:.3f} (ora {r['f1_oraculo']:.3f}) AUC={r['auc']:.3f} "
                              f"n={r['n_alvos']} ep={r['epocas']:3d} {r['segundos']:.0f}s",
                              flush=True)
                    pd.DataFrame(linhas).to_csv(SAIDA / args.saida, index=False, encoding="utf-8")

    d = pd.DataFrame(linhas)
    print("\n=== media sobre as sementes ===")
    chaves = [c for c in ("formato", "saida", "horas_ctx", "aumento", "dro")
              if d[c].nunique() > 1]
    print(d.groupby(["aparelho", "alvo", "taxa"] + (chaves or ["saida"])).agg(
        f1=("f1", "mean"), dp=("f1", "std"), ora=("f1_oraculo", "mean"),
        auc=("auc", "mean"), seg=("segundos", "mean"),
        rot=("rotulos_treino", "mean")).round(3).to_string())
    print(f"\n-> {SAIDA / args.saida}")


if __name__ == "__main__":
    main()
