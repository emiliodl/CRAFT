"""Unifica todos os datasets em 5 min, para os tres aparelhos de interesse.

Saida: datasets/unificado_5min/<DATASET>__<aparelho>__<casa>.npz, todos com as
mesmas colunas, o mesmo limiar de ON e a mesma regra de limpeza:

    t       int64   segundos desde a epoca, horario LOCAL de parede (sem fuso)
    main    float32 agregado da casa em W
    <ap>    float32 potencia do aparelho em W
    status  int8    1 se a potencia do aparelho >= limiar de ON

Regra de limpeza, herdada dos pickles de 5 min ja existentes:
  - janela de 5 min valida se tiver ao menos 50% da contagem tipica de amostras;
  - so dias inteiros (288 janelas validas no agregado E no aparelho) sao mantidos;
  - os dias mantidos nao sao necessariamente consecutivos.

Fontes:
  UK-DALE, REDD  ja convertidos em datasets/cache (5 min), apenas copiados
  REFIT          CSV por casa a ~8 s, canais escolhidos pelo MetaData_Tables.xlsx
  ECO            CSV diario a 1 Hz, agregado em 02_sm_csv e plugs em 02_plugs_csv

Uso:
    python scripts/preparar_5min.py                # tudo
    python scripts/preparar_5min.py --fontes refit # so um
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "datasets"
SAIDA = DADOS / "unificado_5min"

APARELHOS = ("dish washer", "fridge", "washing machine")
LIMIAR_ON = {"fridge": 50.0, "dish washer": 10.0, "washing machine": 20.0}
JANELA = "5min"
JANELAS_POR_DIA = 288


def _slug(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


# --------------------------------------------------------------------------------------
# Reamostragem e limpeza, iguais para todas as fontes
# --------------------------------------------------------------------------------------


def para_5min(df: pd.DataFrame, aparelho: str) -> pd.DataFrame | None:
    """df com indice datetime local e colunas main/<aparelho> -> 5 min limpo."""
    col = aparelho
    agrupado = df.resample(JANELA)
    medias = agrupado.mean()
    contagens = agrupado.count()
    if len(medias) == 0:
        return None

    tipica = float(contagens["main"].median())
    if not np.isfinite(tipica) or tipica <= 0:
        return None
    valido = (contagens["main"] >= 0.5 * tipica) & (contagens[col] >= 0.5 * tipica)
    medias = medias[valido.reindex(medias.index, fill_value=False)]
    medias = medias.dropna()
    if len(medias) == 0:
        return None

    # so dias completos
    por_dia = medias.groupby(medias.index.normalize()).size()
    dias_ok = set(por_dia[por_dia == JANELAS_POR_DIA].index)
    medias = medias[medias.index.normalize().isin(dias_ok)]
    return medias if len(medias) else None


def salvar(dataset: str, aparelho: str, casa: str, df: pd.DataFrame, extra: dict) -> dict:
    SAIDA.mkdir(parents=True, exist_ok=True)
    limiar = LIMIAR_ON[aparelho]
    status = (df[aparelho].to_numpy() >= limiar).astype(np.int8)
    t = df.index.view("int64") // 10**9
    arq = SAIDA / f"{_slug(dataset)}__{_slug(aparelho)}__{_slug(casa)}.npz"
    np.savez_compressed(
        arq, t=t.astype(np.int64), main=df["main"].to_numpy(np.float32),
        **{_slug(aparelho): df[aparelho].to_numpy(np.float32)},
        **{f"status_{_slug(aparelho)}": status},
    )
    linha = dict(
        arquivo=arq.name, dataset=dataset, aparelho=aparelho, casa=casa, taxa_s=300,
        n=int(len(df)), dias=round(len(df) / JANELAS_POR_DIA, 1),
        inicio=str(df.index[0]), fim=str(df.index[-1]),
        limiar_on_W=limiar, pct_ligado=round(100 * float(status.mean()), 2),
        main_mediano_W=round(float(df["main"].median()), 1),
        ap_medio_ligado_W=round(float(df[aparelho][status == 1].mean()), 1) if status.any() else None,
        **extra,
    )
    print(f"  {arq.name:48s} {linha['dias']:7.1f} dias  {linha['pct_ligado']:5.2f}% ligado")
    return linha


# --------------------------------------------------------------------------------------
# UK-DALE e REDD: ja estao em 5 min no cache
# --------------------------------------------------------------------------------------


def fonte_cache() -> list[dict]:
    cache = DADOS / "cache"
    if not (cache / "indice.json").exists():
        print("  cache nao encontrado, pulando")
        return []
    SAIDA.mkdir(parents=True, exist_ok=True)
    linhas = []
    for e in json.loads((cache / "indice.json").read_text(encoding="utf-8")):
        if e["aparelho"] not in APARELHOS or e["taxa_s"] != 300:
            continue
        origem = cache / e["arquivo"]
        destino = SAIDA / f"{_slug(e['dataset'])}__{_slug(e['aparelho'])}__{_slug(e['casa'])}.npz"
        shutil.copyfile(origem, destino)
        d = np.load(destino)
        ap, st = _slug(e["aparelho"]), f"status_{_slug(e['aparelho'])}"
        status = d[st]
        linhas.append(dict(
            arquivo=destino.name, dataset=e["dataset"], aparelho=e["aparelho"], casa=e["casa"],
            taxa_s=300, n=int(len(d["t"])), dias=round(len(d["t"]) / JANELAS_POR_DIA, 1),
            inicio=e["inicio"], fim=e["fim"], limiar_on_W=LIMIAR_ON[e["aparelho"]],
            pct_ligado=round(100 * float(status.mean()), 2),
            main_mediano_W=round(float(np.median(d["main"])), 1),
            ap_medio_ligado_W=round(float(d[ap][status == 1].mean()), 1) if status.any() else None,
            canal="(do cache 5 min ja existente)",
        ))
        print(f"  {destino.name:48s} {linhas[-1]['dias']:7.1f} dias  {linhas[-1]['pct_ligado']:5.2f}% ligado")
    return linhas


# --------------------------------------------------------------------------------------
# REFIT
# --------------------------------------------------------------------------------------

PADROES = {
    "fridge": re.compile(r"^fridge(\b|[- ]freezer|\(|$)", re.I),
    "washing machine": re.compile(r"^washing machine", re.I),   # exclui "Washer Dryer"
    "dish washer": re.compile(r"^dishwasher", re.I),
}


def mapa_refit(pasta: Path) -> dict[str, dict[str, tuple[int, str]]]:
    """casa -> aparelho -> (numero do canal, nome original)."""
    x = pd.ExcelFile(pasta / "MetaData_Tables.xlsx")
    mapa: dict[str, dict[str, tuple[int, str]]] = {}
    for aba in [s for s in x.sheet_names if s.startswith("House")]:
        casa = f"casa {aba.split()[1]}"
        d = x.parse(aba)
        escolhas: dict[str, tuple[int, str]] = {}
        for _, linha in d.iterrows():
            num, nome = linha.iloc[0], str(linha.iloc[1]).strip()
            if pd.isna(num):
                continue
            for aparelho, padrao in PADROES.items():
                if aparelho not in escolhas and padrao.match(nome):  # primeira ocorrencia
                    escolhas[aparelho] = (int(num), nome)
        mapa[casa] = escolhas
    return mapa


def fonte_refit() -> list[dict]:
    pasta = DADOS / "refit" / "CLEAN_REFIT_081116"
    if not pasta.exists():
        print("  refit nao encontrado, pulando")
        return []
    mapa = mapa_refit(pasta)
    linhas = []
    for arq in sorted(pasta.glob("CLEAN_House*.csv"), key=lambda p: int(re.findall(r"\d+", p.stem)[0])):
        numero = re.findall(r"\d+", arq.stem)[0]
        casa = f"casa {numero}"
        escolhas = mapa.get(casa, {})
        if not escolhas:
            continue
        colunas = ["Time", "Aggregate", "Issues"] + [f"Appliance{n}" for n, _ in escolhas.values()]
        df = pd.read_csv(arq, usecols=colunas, parse_dates=["Time"])
        df = df[df["Issues"] == 0].drop(columns=["Issues"])
        df = df.set_index("Time").sort_index()
        for aparelho, (num, nome) in escolhas.items():
            sub = df[["Aggregate", f"Appliance{num}"]].rename(
                columns={"Aggregate": "main", f"Appliance{num}": aparelho}
            )
            limpo = para_5min(sub, aparelho)
            if limpo is None:
                print(f"  casa {casa} {aparelho}: sem dias completos, pulando")
                continue
            linhas.append(salvar("REFIT", aparelho, casa, limpo, dict(canal=f"Appliance{num} ({nome})")))
    return linhas


# --------------------------------------------------------------------------------------
# ECO
# --------------------------------------------------------------------------------------

PLUGS_ECO = {"dish washer": "02", "fridge": "04"}  # casa 02; conferir 0X_doc.txt de outras


def fonte_eco() -> list[dict]:
    pasta = DADOS / "ECO"
    if not pasta.exists():
        print("  ECO nao encontrado, pulando")
        return []
    linhas = []
    for sm in sorted(pasta.glob("*_sm_csv")):
        casa_id = sm.name.split("_")[0]
        casa = f"casa {int(casa_id)}"
        dir_sm = sm / casa_id
        dir_plugs = pasta / f"{casa_id}_plugs_csv" / casa_id
        if not dir_sm.exists() or not dir_plugs.exists():
            continue
        for aparelho, plug in PLUGS_ECO.items():
            dir_ap = dir_plugs / plug
            if not dir_ap.exists():
                continue
            partes = []
            for arq_sm in sorted(dir_sm.glob("*.csv")):
                arq_ap = dir_ap / arq_sm.name
                if not arq_ap.exists():
                    continue
                # 1 Hz, um dia por arquivo; -1 marca amostra ausente
                main = pd.read_csv(arq_sm, header=None, usecols=[0]).iloc[:, 0].to_numpy(np.float32)
                pot = pd.read_csv(arq_ap, header=None).iloc[:, 0].to_numpy(np.float32)
                n = min(len(main), len(pot))
                if n == 0:
                    continue
                main, pot = main[:n], pot[:n]
                main[main < 0] = np.nan
                pot[pot < 0] = np.nan
                idx = pd.date_range(arq_sm.stem, periods=n, freq="1s")
                partes.append(pd.DataFrame({"main": main, aparelho: pot}, index=idx))
            if not partes:
                continue
            bruto = pd.concat(partes).sort_index()
            limpo = para_5min(bruto, aparelho)
            if limpo is None:
                print(f"  {casa} {aparelho}: sem dias completos, pulando")
                continue
            linhas.append(salvar("ECO", aparelho, casa, limpo, dict(canal=f"plug {plug}")))
    return linhas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fontes", default="cache,refit,eco")
    args = p.parse_args()

    linhas = []
    for fonte in args.fontes.split(","):
        print(f"\n=== {fonte} ===")
        linhas += {"cache": fonte_cache, "refit": fonte_refit, "eco": fonte_eco}[fonte]()

    if linhas:
        indice = pd.DataFrame(linhas).sort_values(["dataset", "aparelho", "casa"])
        indice.to_csv(SAIDA / "indice.csv", index=False, encoding="utf-8")
        print(f"\n{len(indice)} series em {SAIDA}")
        print(indice.groupby(["dataset", "aparelho"]).agg(
            casas=("casa", "nunique"), dias=("dias", "sum")).round(0).to_string())


if __name__ == "__main__":
    main()
