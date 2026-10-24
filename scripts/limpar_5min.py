"""Descarta dias irregulares das series unificadas de 5 min.

A conversao ja exigiu janelas com ao menos 50% das amostras e dias inteiros de
288 janelas. Este passo remove os dias que passam nesse filtro mas continuam
suspeitos, por regra explicita e auditavel:

  travamento   o agregado repete o mesmo valor por mais de 1 h seguida
               (medidor travado, nao casa parada: ate a geladeira oscila)
  incoerencia  a potencia do aparelho excede o agregado em mais de 1% das
               janelas do dia - fisicamente impossivel, indica submedidor e
               agregado dessincronizados ou canal trocado
  faixa        valor negativo, agregado acima de 30 kW ou aparelho acima de 15 kW
  agregado morto  mediana do agregado no dia abaixo de 5 W (medidor desligado)
  canal morto  so para a geladeira: potencia maxima do dia abaixo de 5 W. A
               geladeira cicla o dia todo, entao um canal que nunca sai de zero
               esta morto. A regra olha a POTENCIA, nao o status: usar o limiar
               de ON aqui descartaria geladeiras de baixa potencia (as do REFIT
               puxam 36-56 W contra 84-102 W do UK-DALE) por erro de rotulo, e
               nao por defeito do dado. Lava-loucas e maquina de lavar podem
               legitimamente passar o dia desligados, entao nao se aplica a eles.

Uso:
    python scripts/limpar_5min.py --relatorio     # so mostra o que seria removido
    python scripts/limpar_5min.py                 # aplica e reescreve os .npz
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
PASTA = RAIZ / "datasets" / "unificado_5min"
JANELAS_POR_DIA = 288

MAX_REPETICAO = 12      # janelas iguais seguidas (1 h) no agregado
TOL_COERENCIA = 5.0     # W de folga antes de considerar aparelho > agregado
FRACAO_INCOERENTE = 0.01
MAIN_MAX, AP_MAX, MAIN_MIN_MEDIANA = 30000.0, 15000.0, 5.0


def _maior_repeticao(x: np.ndarray) -> int:
    if len(x) == 0:
        return 0
    troca = np.flatnonzero(np.diff(x) != 0)
    limites = np.concatenate(([-1], troca, [len(x) - 1]))
    return int(np.diff(limites).max())


def avaliar_dia(main: np.ndarray, pot: np.ndarray, status: np.ndarray, aparelho: str) -> str | None:
    """Devolve o nome da regra violada, ou None se o dia esta bom."""
    if np.any(main < 0) or np.any(pot < 0) or np.any(main > MAIN_MAX) or np.any(pot > AP_MAX):
        return "faixa"
    if np.median(main) < MAIN_MIN_MEDIANA:
        return "agregado morto"
    if _maior_repeticao(main) > MAX_REPETICAO:
        return "travamento"
    if np.mean(pot > main + TOL_COERENCIA) > FRACAO_INCOERENTE:
        return "incoerencia"
    if aparelho == "fridge" and np.max(pot) < MAIN_MIN_MEDIANA:
        return "canal morto"
    return None


def processar(arq: Path, aplicar: bool) -> dict:
    d = dict(np.load(arq))
    ap = next(k for k in d if k not in ("t", "main") and not k.startswith("status_"))
    st = f"status_{ap}"
    aparelho = ap.replace("_", " ")

    dias = (d["t"] // 86400).astype(np.int64)
    manter = np.ones(len(d["t"]), dtype=bool)
    contagem: dict[str, int] = {}
    total_dias = 0

    for dia in np.unique(dias):
        sel = dias == dia
        if sel.sum() != JANELAS_POR_DIA:
            manter[sel] = False
            contagem["dia incompleto"] = contagem.get("dia incompleto", 0) + 1
            continue
        total_dias += 1
        regra = avaliar_dia(d["main"][sel], d[ap][sel], d[st][sel], aparelho)
        if regra:
            manter[sel] = False
            contagem[regra] = contagem.get(regra, 0) + 1

    dias_removidos = sum(contagem.values())
    if aplicar and dias_removidos:
        for k in list(d):
            d[k] = d[k][manter]
        np.savez_compressed(arq, **d)

    # o limiar de ON e adequado para esta serie? (diagnostico, nao descarta)
    p99 = float(np.percentile(d[ap], 99)) if len(d[ap]) else 0.0
    limiar = {"fridge": 50.0, "dish washer": 10.0, "washing machine": 20.0}[aparelho]

    partes = arq.stem.split("__")
    return dict(
        arquivo=arq.name, dataset=partes[0], aparelho=partes[1], casa=partes[2],
        dias_antes=total_dias, dias_removidos=dias_removidos,
        dias_depois=total_dias - dias_removidos,
        p99_potencia_W=round(p99, 1), limiar_on_W=limiar,
        limiar_suspeito=bool(p99 < 1.5 * limiar),
        **{f"por_{k.replace(' ', '_')}": v for k, v in contagem.items()},
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--relatorio", action="store_true", help="nao reescreve nada")
    args = p.parse_args()

    linhas = [processar(a, aplicar=not args.relatorio) for a in sorted(PASTA.glob("*.npz"))]
    d = pd.DataFrame(linhas).fillna(0)
    colunas_regra = [c for c in d.columns if c.startswith("por_")]

    print("=== dias removidos por regra ===")
    resumo = d[colunas_regra].sum().astype(int).sort_values(ascending=False)
    total_antes, total_depois = int(d.dias_antes.sum()), int(d.dias_depois.sum())
    for regra, n in resumo.items():
        if n:
            print(f"  {regra.replace('por_', '').replace('_', ' '):20s} {n:6d} dias  "
                  f"({100 * n / max(total_antes, 1):5.1f}%)")
    print(f"  {'TOTAL':20s} {total_antes - total_depois:6d} de {total_antes} dias "
          f"({100 * (total_antes - total_depois) / max(total_antes, 1):.1f}%)")

    print("\n=== por dataset ===")
    print(d.groupby("dataset").agg(series=("arquivo", "count"), dias_antes=("dias_antes", "sum"),
                                   dias_depois=("dias_depois", "sum")).to_string())

    piores = d.sort_values("dias_removidos", ascending=False).head(10)
    print("\n=== series mais afetadas ===")
    print(piores[["arquivo", "dias_antes", "dias_removidos", "dias_depois"]].to_string(index=False))

    suspeitas = d[d.limiar_suspeito & (d.dias_antes > 0)]
    if len(suspeitas):
        print("\n=== limiar de ON possivelmente inadequado (p99 da potencia < 1,5x o limiar) ===")
        print(suspeitas[["arquivo", "p99_potencia_W", "limiar_on_W", "dias_antes"]].to_string(index=False))

    vazias = d[d.dias_depois == 0]
    if len(vazias):
        print(f"\n=== {len(vazias)} series ficariam sem nenhum dia ===")
        print(vazias[["arquivo", "dias_antes"] + colunas_regra].to_string(index=False))

    # series sem evento nenhum: o canal do aparelho nunca sobe. Nao servem para
    # treino nem avaliacao (F1 indefinido), entao saem do conjunto - movidas, nao
    # apagadas, com o motivo registrado no relatorio.
    sem_evento = d[d.p99_potencia_W < d.limiar_on_W]
    if len(sem_evento):
        print("\n=== series sem nenhum evento (p99 da potencia abaixo do limiar de ON) ===")
        print(sem_evento[["arquivo", "p99_potencia_W", "limiar_on_W", "dias_depois"]].to_string(index=False))
        if not args.relatorio:
            destino = PASTA / "descartadas"
            destino.mkdir(exist_ok=True)
            for nome in sem_evento.arquivo:
                (PASTA / nome).rename(destino / nome)
            print(f"  -> movidas para {destino}")
    d["descartada_sem_evento"] = d.p99_potencia_W < d.limiar_on_W

    d.to_csv(PASTA / ("relatorio_limpeza.csv" if args.relatorio else "limpeza_aplicada.csv"),
             index=False, encoding="utf-8")
    print(f"\n{'(relatorio, nada foi alterado)' if args.relatorio else '(aplicado aos .npz)'}")


if __name__ == "__main__":
    main()
