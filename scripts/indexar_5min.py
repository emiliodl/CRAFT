"""Reindexa datasets/unificado_5min apos a limpeza e escreve o LEIAME."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
PASTA = RAIZ / "datasets" / "unificado_5min"
LIMIAR_ON = {"fridge": 50.0, "dish washer": 10.0, "washing machine": 20.0}


def corridas(t: np.ndarray, taxa_s: int = 300) -> list[int]:
    """Comprimentos das corridas de amostras consecutivas (sem buraco no tempo)."""
    if len(t) == 0:
        return []
    quebras = np.flatnonzero(np.diff(t) != taxa_s) + 1
    limites = np.concatenate(([0], quebras, [len(t)]))
    return list(np.diff(limites))


def main() -> None:
    linhas = []
    for arq in sorted(PASTA.glob("*.npz")):
        d = np.load(arq)
        ap = next(k for k in d.files if k not in ("t", "main") and not k.startswith("status_"))
        st = f"status_{ap}"
        t, status = d["t"], d[st]
        if len(t) == 0:
            continue
        comp = corridas(t)
        aparelho = ap.replace("_", " ")
        dataset, _, casa = arq.stem.split("__")
        linhas.append(dict(
            arquivo=arq.name, dataset=dataset, aparelho=aparelho, casa=casa.replace("_", " "),
            taxa_s=300, n=len(t), dias=round(len(t) / 288, 1),
            inicio=str(datetime.fromtimestamp(int(t[0]), tz=timezone.utc).replace(tzinfo=None)),
            fim=str(datetime.fromtimestamp(int(t[-1]), tz=timezone.utc).replace(tzinfo=None)),
            corridas=len(comp), maior_corrida_dias=round(max(comp) / 288, 1),
            limiar_on_W=LIMIAR_ON[aparelho], pct_ligado=round(100 * float(status.mean()), 2),
            main_mediano_W=round(float(np.median(d["main"])), 1),
            ap_medio_ligado_W=round(float(d[ap][status == 1].mean()), 1) if status.any() else None,
        ))

    d = pd.DataFrame(linhas).sort_values(["dataset", "aparelho", "casa"])
    d.to_csv(PASTA / "indice.csv", index=False, encoding="utf-8")

    pd.set_option("display.width", 200)
    print(d.groupby(["dataset", "aparelho"]).agg(
        casas=("casa", "nunique"), dias=("dias", "sum"),
        pct_ligado=("pct_ligado", "mean")).round(1).to_string())
    print(f"\ntotal: {len(d)} series, {d.casas.nunique() if 'casas' in d else d.casa.nunique()} casas, "
          f"{d.dias.sum():.0f} dias")

    (PASTA / "LEIAME.txt").write_text(f"""Datasets de NILM unificados em 5 min
====================================
Gerado por scripts/preparar_5min.py + limpar_5min.py + indexar_5min.py
{datetime.now():%Y-%m-%d}

Um arquivo .npz por (dataset, aparelho, casa), todos com as mesmas colunas:

    t        int64    segundos desde a epoca, horario LOCAL de parede (sem fuso)
    main     float32  agregado da casa, em W
    <ap>     float32  potencia do aparelho, em W
    status_<ap>  int8  1 quando a potencia >= limiar de ON

Leitura:
    import numpy as np
    d = np.load('uk_dale__fridge__casa_1.npz')
    t, main, pot, st = d['t'], d['main'], d['fridge'], d['status_fridge']

Aparelhos: dish washer, fridge, washing machine.
Limiares de ON: lava-loucas 10 W, geladeira 50 W, maquina de lavar 20 W.

Fontes e canais
---------------
UK-DALE  3 casas, submedidores a 6 s reamostrados para 5 min
REDD     ate 5 casas, idem
REFIT    ate 20 casas, CSV a ~8 s. Canal escolhido pelo MetaData_Tables.xlsx:
         'Fridge'/'Fridge-Freezer' contam como geladeira; 'Freezer' sozinho nao;
         'Washer Dryer' NAO conta como maquina de lavar. Quando ha mais de um
         candidato na casa, o primeiro foi usado (coluna `canal` do indice antigo).
ECO      casa 02, 1 Hz. Agregado = coluna 1 do medidor (soma das tres fases);
         plug 02 = lava-loucas, plug 04 = geladeira. Nao ha maquina de lavar.

Limpeza aplicada
----------------
Na reamostragem: janela de 5 min valida com >= 50% das amostras esperadas, e
apenas dias inteiros (288 janelas validas no agregado E no aparelho).

Por dia, em limpar_5min.py:
  travamento       agregado repetindo o mesmo valor por mais de 1 h
  incoerencia      potencia do aparelho > agregado em mais de 1% das janelas
  faixa            valores negativos, agregado > 30 kW ou aparelho > 15 kW
  agregado morto   mediana do agregado no dia < 5 W
  canal morto      (so geladeira) potencia maxima do dia < 5 W

Series inteiras sem nenhum evento (p99 da potencia abaixo do limiar de ON) foram
movidas para descartadas/ - nao apagadas. O relatorio esta em
limpeza_aplicada.csv, com a contagem por regra e por serie.

ATENCAO: os dias mantidos NAO sao necessariamente consecutivos. Ao montar janelas
longas, segmente por corridas contiguas (diferenca de t igual a 300 s), senao
blocos vao costurar periodos separados por semanas. A coluna `corridas` e
`maior_corrida_dias` do indice.csv ajudam a dimensionar isso.

ATENCAO 2: o limiar de ON e fixo por aparelho e nao transfere bem entre datasets.
Geladeiras do UK-DALE puxam 84-102 W; algumas do REFIT puxam 36-56 W, ou seja
rente ao limiar de 50 W. Ver a coluna limiar_suspeito em limpeza_aplicada.csv.
""", encoding="utf-8")
    print(f"indice.csv e LEIAME.txt escritos em {PASTA}")


if __name__ == "__main__":
    main()
