"""Grade quase-nativa (30 s) das casas REFIT, com TODOS os plugues (docs/v3.md §3.2, D4).

O gerador de recomposicao do v3 injeta e subtrai na resolucao nativa e so depois
reamostra para 5 min. A resolucao nativa do REFIT e irregular (~6-8 s), entao
guardamos uma grade regular de 30 s, alinhada a epoca: cada caixa de 5 min do
protocolo contem exatamente 10 caixas de 30 s.
A fase de uma ativacao injetada pode entao cair em qualquer um dos 10 pontos
do intervalo, que e o que "injetar antes de reamostrar" quer preservar.

A media de 5 min e ponderada pela contagem de amostras nativas de cada caixa
(`cont`): sum(cont * x) / sum(cont) reproduz EXATAMENTE a media do protocolo
(`scripts/preparar_artigo.py::reamostrar`). Na recomposicao, cada componente
(fundo, ativacao) e reamostrado com a sua propria contagem - e a media de 5 min
do sinal que cada medidor teria registrado.

Lacuna = linha ausente (v2 E0.2). Caixas vazias isoladas (ate 2, 60 s) sao
interpoladas; o resto fica NaN e a janela que tocar NaN nao entra no gerador.

Saida: datasets/v3_fino/refit_casa_<h>.npz com
    t0        inicio (s, epoca) da primeira caixa
    x         (10, N) float32: agregado + Appliance1..9, NaN onde falta
    cont      (N,) amostras nativas por caixa (1 nas interpoladas)
    nomes     nomes dos canais no metadado

Uso:
    python scripts/preparar_v3.py                  # todas as casas
    python scripts/preparar_v3.py --casas 8,3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
PASTA = RAIZ / "datasets" / "refit" / "CLEAN_REFIT_081116"
SAIDA = RAIZ / "datasets" / "v3_fino"
PASSO = 30
INICIO, FIM = "2014-03-07", "2015-03-07"          # ano de treino do protocolo
CASAS = [c for c in range(1, 22) if c != 14]


def nomes_canais(casa: int) -> list[str]:
    meta = pd.ExcelFile(PASTA / "MetaData_Tables.xlsx").parse(f"House {casa}")
    nomes = {}
    for _, lin in meta.iterrows():
        if not pd.isna(lin.iloc[0]):
            try:
                nomes[int(lin.iloc[0])] = str(lin.iloc[1]).strip()
            except ValueError:
                pass
    return ["Aggregate"] + [nomes.get(i, f"Appliance{i}") for i in range(1, 10)]


def grade(casa: int) -> dict:
    cols = ["Unix", "Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]
    df = pd.read_csv(PASTA / f"CLEAN_House{casa}.csv", usecols=cols)
    t0 = int(pd.Timestamp(INICIO).timestamp())
    t1 = int(pd.Timestamp(FIM).timestamp())
    df = df[(df.Unix >= t0) & (df.Unix < t1)]
    n = (t1 - t0) // PASSO
    caixa = ((df.Unix.to_numpy() - t0) // PASSO).astype(np.int64)
    cont = np.bincount(caixa, minlength=n)[:n]
    x = np.full((10, n), np.nan, np.float32)
    for k, c in enumerate(cols[1:]):
        s = np.bincount(caixa, weights=df[c].to_numpy(np.float64), minlength=n)[:n]
        with np.errstate(invalid="ignore", divide="ignore"):
            x[k] = np.where(cont > 0, s / np.maximum(cont, 1), np.nan)
    # interpola buracos de ate 2 caixas
    falta = cont == 0
    d = np.diff(np.r_[0, falta.astype(np.int8), 0])
    ini, fim = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    curtos = [(a, b) for a, b in zip(ini, fim) if b - a <= 2 and a > 0 and b < n]
    for a, b in curtos:
        for k in range(10):
            esq, dir_ = x[k, a - 1], x[k, b]
            w = (np.arange(a, b) - (a - 1)) / (b - (a - 1))
            x[k, a:b] = esq + w * (dir_ - esq)
        cont[a:b] = 1                     # peso minimo para a caixa interpolada
    return dict(t0=t0, x=x, cont=cont.astype(np.uint8 if cont.max() < 256 else np.uint16),
                nomes=np.array(nomes_canais(casa)), n_interp=sum(b - a for a, b in curtos))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--casas", default=",".join(map(str, CASAS)))
    args = p.parse_args()
    SAIDA.mkdir(parents=True, exist_ok=True)
    for c in [int(x) for x in args.casas.split(",")]:
        t = time.time()
        g = grade(c)
        np.savez_compressed(SAIDA / f"refit_casa_{c}.npz", t0=g["t0"], x=g["x"],
                            cont=g["cont"], nomes=g["nomes"])
        ok = np.isfinite(g["x"][0])
        print(f"casa {c:2d}: {ok.mean() * 100:5.1f}% das caixas de 30 s com dado, "
              f"{g['n_interp']} interpoladas, {time.time() - t:.0f}s | "
              f"{list(g['nomes'][1:])}", flush=True)


if __name__ == "__main__":
    main()
