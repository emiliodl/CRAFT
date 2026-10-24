"""Prepara os dados no protocolo de Chatterjee & Heer (2025).

Ver docs/baseline_chatterjee2025.md. Resumo do que este script reproduz:

  casas       REFIT casa 8 (treino e teste intra), ECO casas 1, 2 e 5 (teste cross)
  aparelhos   freezer e washing machine (a intersecao com o nosso recorte)
  taxas       5 min e 15 min
  janelas     treino  REFIT 2014-03-07 a 2015-03-06
              teste   REFIT 2015-03-07 a 2015-04-06
              cross   ECO   2012-06-02 a 2012-07-01

Diferencas conscientes em relacao ao pipeline de datasets/unificado_5min:

  - NAO exige dias completos. O mes de teste tem 27-31 dias; descartar dias
    mudaria o denominador da metrica e tornaria o numero incomparavel com o do
    artigo. Cada janela de 5/15 min e valida se tiver >= 50% das amostras
    esperadas; o resto vira buraco, e o janelamento respeita os buracos.
  - NAO aplica as regras de limpeza por dia (travamento, incoerencia etc.),
    pelo mesmo motivo.

Limiares de ON: o artigo nao os reporta no texto, mas eles estao no codigo
(nilmtk/losses.py do NeuralNILM_Pytorch). Sao freezer 10 W (padrao, por nao
estar na tabela) e washing machine 20 W. Ver a constante LIMIAR_ON abaixo.

Uso:
    python scripts/preparar_artigo.py                 # tudo que estiver em disco
    python scripts/preparar_artigo.py --fontes refit
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "datasets"
SAIDA = DADOS / "protocolo_artigo"

# Os SEIS aparelhos do artigo. A casa 8 do REFIT tem todos; na ECO cada casa tem
# um subconjunto, e o cruzamento so existe onde o aparelho aparece nos dois lados.
APARELHOS = ("freezer", "washing machine", "kettle", "computer", "microwave",
             "television")

# Limiares EXATOS do artigo, lidos de nilmtk/losses.py do NeuralNILM_Pytorch
# (github.com/Ming-er/NeuralNILM_Pytorch), que e o que a API do nilmtk usa para
# calcular f1score/recall/precision:
#     on_threhold = {'fridge':50, 'kettle':2000, 'dish washer':10,
#                    'washing machine':20, 'drill':0}
#     threshold = on_threhold.get(app_name, 10)
# 'freezer' NAO esta no dicionario, entao cai no padrao de 10 W - e nao nos 50 W
# que o resto deste projeto usa para geladeira.
# kettle e o unico dos seis que esta na tabela alem da maquina de lavar; os
# demais caem no padrao de 10 W.
LIMIAR_ON = {"freezer": 10.0, "washing machine": 20.0, "kettle": 2000.0,
             "computer": 10.0, "microwave": 10.0, "television": 10.0}
TAXAS = {"5min": 300, "15min": 900}

JANELAS = {
    "refit_treino": ("2014-03-07", "2015-03-06"),
    "refit_teste": ("2015-03-07", "2015-04-06"),
    "eco_teste": ("2012-06-02", "2012-07-01"),
}

CASA_REFIT = 8
CASAS_ECO = (1, 2, 5)

# Na casa 8 do REFIT: 'Washer Dryer' NAO conta como maquina de lavar, e o canal
# 'Freezer' e distinto de 'Fridge'.
# Nomes equivalentes entre REFIT e ECO. Na ECO o "computer" aparece como PC
# (casas 1 e 5) e Laptops (casa 2), e a televisao como TV (casa 2). A casa 5 tem
# "Entertainment", que NAO e mapeado para televisao por ser ambiguo - pode ser
# um conjunto de equipamentos de sala, e o metadado nao desambigua.
PADROES = {
    "freezer": re.compile(r"^freezer\b", re.I),
    "washing machine": re.compile(r"^washing machine", re.I),
    "kettle": re.compile(r"^kettle\b", re.I),
    "computer": re.compile(r"^(computer|pc|laptops)\b", re.I),
    "microwave": re.compile(r"^microwave", re.I),
    "television": re.compile(r"^(television|tv)\b", re.I),
}


def _slug(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


def reamostrar(df: pd.DataFrame, aparelho: str, taxa: str) -> pd.DataFrame | None:
    """Media por janela, exigindo >= 50% das amostras tipicas. Sem filtro de dia.

    Devolve tambem `fracao_ligado`: a fracao das amostras NATIVAS da janela em
    que a potencia do aparelho passou o limiar. Isso permite duas definicoes de
    rotulo, e a diferenca entre elas nao e cosmetica:

      media   limiariza a potencia ja reamostrada (o que o nilmtk faz). O rotulo
              passa a depender da taxa: o freezer da ECO puxa 52 W contra limiar
              de 50 W, e a media de 15 min o afunda - a prevalencia cai de 45%
              para 18% sem que nada tenha mudado no aparelho.
      nativo   limiariza na taxa original e marca a janela como ligada se >= 50%
              das amostras nativas estavam ligadas. O alvo fica estavel entre
              taxas, e "5 min contra 15 min" passa a comparar resolucao de
              ENTRADA com o mesmo alvo, que e o que se quer medir.
    """
    limiar = LIMIAR_ON[aparelho]
    df = df.assign(_ligado=(df[aparelho] >= limiar).astype(float))
    g = df.resample(taxa)
    medias, contagens = g.mean(), g.count()
    if len(medias) == 0:
        return None
    tipica = float(contagens["main"].median())
    if not np.isfinite(tipica) or tipica <= 0:
        return None
    ok = (contagens["main"] >= 0.5 * tipica) & (contagens[aparelho] >= 0.5 * tipica)
    medias = medias[ok.reindex(medias.index, fill_value=False)].dropna()
    return medias.rename(columns={"_ligado": "fracao_ligado"}) if len(medias) else None


def salvar(fonte: str, casa: str, aparelho: str, taxa: str, particao: str,
           df: pd.DataFrame) -> dict:
    SAIDA.mkdir(parents=True, exist_ok=True)
    limiar = LIMIAR_ON[aparelho]
    status = (df[aparelho].to_numpy() >= limiar).astype(np.int8)          # definicao "media"
    status_nativo = (df["fracao_ligado"].to_numpy() >= 0.5).astype(np.int8)  # definicao "nativo"
    ap = _slug(aparelho)
    arq = SAIDA / f"{fonte}_{_slug(casa)}__{ap}__{taxa}__{particao}.npz"
    np.savez_compressed(
        arq, t=(df.index.view("int64") // 10**9).astype(np.int64),
        main=df["main"].to_numpy(np.float32),
        **{ap: df[aparelho].to_numpy(np.float32)},
        **{f"status_{ap}": status, f"status_{ap}_nativo": status_nativo})
    linha = dict(arquivo=arq.name, fonte=fonte, casa=casa, aparelho=aparelho, taxa=taxa,
                 particao=particao, n=len(df), dias=round(len(df) * TAXAS[taxa] / 86400, 1),
                 inicio=str(df.index[0]), fim=str(df.index[-1]), limiar_on_W=limiar,
                 pct_ligado=round(100 * float(status.mean()), 2),
                 pct_ligado_nativo=round(100 * float(status_nativo.mean()), 2),
                 ap_medio_ligado_W=(round(float(df[aparelho][status == 1].mean()), 1)
                                    if status.any() else None))
    print(f"  {arq.name:52s} {linha['dias']:6.1f} d  "
          f"{linha['pct_ligado']:5.2f}% ligado (media) / "
          f"{linha['pct_ligado_nativo']:5.2f}% (nativo)")
    return linha


# --------------------------------------------------------------------------------------
# REFIT casa 8
# --------------------------------------------------------------------------------------


def fonte_refit(manter_issues: bool = True, casas: tuple[int, ...] | None = None
                ) -> list[dict]:
    pasta = DADOS / "refit" / "CLEAN_REFIT_081116"
    linhas_todas = []
    for casa in (casas or (CASA_REFIT,)):
        linhas_todas += _uma_casa_refit(pasta, casa, manter_issues)
    return linhas_todas


def _uma_casa_refit(pasta: Path, CASA_REFIT: int, manter_issues: bool) -> list[dict]:
    """Uma casa do REFIT. Casas != 8 entram so como TREINO adicional: o teste
    intra do artigo e sempre o mes da casa 8, para a metrica continuar
    comparavel."""
    arq = pasta / f"CLEAN_House{CASA_REFIT}.csv"
    if not arq.exists():
        print(f"  casa {CASA_REFIT} nao encontrada, pulando")
        return []

    meta = pd.ExcelFile(pasta / "MetaData_Tables.xlsx").parse(f"House {CASA_REFIT}")
    canais: dict[str, int] = {}
    for _, linha in meta.iterrows():
        num, nome = linha.iloc[0], str(linha.iloc[1]).strip()
        if pd.isna(num):
            continue
        for aparelho, padrao in PADROES.items():
            if aparelho not in canais and padrao.match(nome):
                canais[aparelho] = int(num)
    if not canais:
        print(f"  casa {CASA_REFIT}: nenhum aparelho de interesse")
        return []
    print(f"  casa {CASA_REFIT}: {canais}")

    colunas = ["Time", "Aggregate", "Issues"] + [f"Appliance{n}" for n in canais.values()]
    df = pd.read_csv(arq, usecols=colunas, parse_dates=["Time"])
    # O conversor REFIT do nilmtk NAO le a coluna Issues (ela nem esta no usecols
    # dele), entao o pipeline do artigo mantem as linhas sinalizadas. Filtra-las
    # encolhe o mes de teste de 31 para 23 dias e muda o denominador da metrica.
    # Padrao aqui: manter, para o conjunto de teste ser o mesmo que o deles.
    if not manter_issues:
        df = df[df["Issues"] == 0]
    df = df.drop(columns=["Issues"]).set_index("Time").sort_index()

    linhas = []
    for aparelho, num in canais.items():
        sub = df[["Aggregate", f"Appliance{num}"]].rename(
            columns={"Aggregate": "main", f"Appliance{num}": aparelho})
        particoes = ((("treino", "refit_treino"), ("teste", "refit_teste"))
                     if CASA_REFIT == 8 else (("treino", "refit_treino"),))
        for particao, chave in particoes:
            a, b = JANELAS[chave]
            fatia = sub.loc[a:b]
            if fatia.empty:
                print(f"  {aparelho}/{particao}: intervalo vazio, pulando")
                continue
            for taxa in TAXAS:
                limpo = reamostrar(fatia, aparelho, taxa)
                if limpo is not None:
                    linhas.append(salvar("refit", f"casa {CASA_REFIT}", aparelho, taxa,
                                         particao, limpo))
    return linhas


# --------------------------------------------------------------------------------------
# ECO
# --------------------------------------------------------------------------------------


# Mapeamento de plugs quando o 0X_doc.txt nao veio junto com o download.
# Fonte: nilmtk/dataset_converters/eco/metadata/building{1,2,5}.yaml, que e o
# mesmo metadado que o pipeline do artigo usa. Os arquivos listam `meters: [N]`,
# onde N e o id de medidor do nilmtk; os medidores 1-3 sao as tres fases do
# smart meter, entao PLUG = N - 3. Conferido contra o 02_doc.txt que temos em
# disco: building2.yaml da Tablet=4 e Freezer=9, e o doc da 01: Tablet e
# 06: Freezer. Bate.
PLUGS_CONHECIDOS = {
    1: {"01": "Fridge", "02": "Dryer", "03": "Coffee machine", "04": "Kettle",
        "05": "Washing machine", "06": "PC", "07": "Freezer"},
    2: {"01": "Tablet", "02": "Dishwasher", "03": "Air exhaust", "04": "Fridge",
        "05": "Entertainment", "06": "Freezer", "07": "Kettle", "08": "Lamp",
        "09": "Laptops", "10": "Stove", "11": "TV", "12": "Stereo"},
    5: {"01": "Tablet", "02": "Coffee machine", "03": "Fountain", "04": "Microwave",
        "05": "Fridge", "06": "Entertainment", "07": "PC", "08": "Kettle"},
}


def plugs_da_casa(casa: int, doc: Path) -> dict[str, str]:
    """aparelho -> numero do plug. Prefere o 0X_doc.txt; senao usa a tabela."""
    nomes: dict[str, str] = {}
    if doc.exists():
        for linha in doc.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(r"\s*(\d{2}):\s*([A-Za-z][A-Za-z /]*)", linha)
            if m:
                nomes[m.group(1)] = m.group(2).strip()
    if not nomes:
        nomes = PLUGS_CONHECIDOS.get(casa, {})
        if nomes:
            print(f"    (sem {doc.name}; usando o mapa do metadado do nilmtk)")

    mapa: dict[str, str] = {}
    for numero, nome in sorted(nomes.items()):
        for aparelho, padrao in PADROES.items():
            if aparelho not in mapa and padrao.match(nome):
                mapa[aparelho] = numero
    return mapa


def fonte_eco() -> list[dict]:
    pasta = DADOS / "ECO"
    if not pasta.exists():
        print("  ECO nao encontrado, pulando")
        return []
    a, b = JANELAS["eco_teste"]
    linhas = []

    for n in CASAS_ECO:
        cid = f"{n:02d}"
        doc = pasta / f"{cid}_doc.txt"
        dir_sm = pasta / f"{cid}_sm_csv" / cid
        dir_plugs = pasta / f"{cid}_plugs_csv" / cid
        if not (dir_sm.exists() and dir_plugs.exists()):
            print(f"  ECO casa {n}: nao esta em disco, pulando")
            continue
        mapa = plugs_da_casa(n, doc)
        if not mapa:
            print(f"  ECO casa {n}: nenhum aparelho de interesse no doc")
            continue
        print(f"  ECO casa {n}: {mapa}")

        for aparelho, plug in mapa.items():
            dir_ap = dir_plugs / plug
            if not dir_ap.exists():
                print(f"    plug {plug} ausente em disco")
                continue
            partes = []
            for arq_sm in sorted(dir_sm.glob("*.csv")):
                if not (a <= arq_sm.stem <= b):
                    continue
                arq_ap = dir_ap / arq_sm.name
                if not arq_ap.exists():
                    continue
                main = pd.read_csv(arq_sm, header=None, usecols=[0]).iloc[:, 0].to_numpy(np.float32)
                pot = pd.read_csv(arq_ap, header=None).iloc[:, 0].to_numpy(np.float32)
                k = min(len(main), len(pot))
                if k == 0:
                    continue
                main, pot = main[:k].copy(), pot[:k].copy()
                main[main < 0] = np.nan          # -1 marca amostra ausente
                pot[pot < 0] = np.nan
                partes.append(pd.DataFrame(
                    {"main": main, aparelho: pot},
                    index=pd.date_range(arq_sm.stem, periods=k, freq="1s")))
            if not partes:
                print(f"    {aparelho}: nenhum dia no intervalo do artigo")
                continue
            bruto = pd.concat(partes).sort_index()
            for taxa in TAXAS:
                limpo = reamostrar(bruto, aparelho, taxa)
                if limpo is not None:
                    linhas.append(salvar("eco", f"casa {n}", aparelho, taxa, "cross", limpo))
    return linhas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fontes", default="refit,eco")
    p.add_argument("--aparelhos", default="", help="subconjunto de APARELHOS; vazio = todos")
    p.add_argument("--taxas", default="", help="subconjunto de TAXAS; vazio = todas")
    p.add_argument("--casas_refit", default="8",
                   help="casas REFIT a processar, ex '8,1,3,4'; so a 8 gera teste intra")
    args = p.parse_args()

    if args.aparelhos:
        manter = {a.strip() for a in args.aparelhos.split(",")}
        for k in list(PADROES):
            if k not in manter:
                del PADROES[k]
    if args.taxas:
        manter_t = {t.strip() for t in args.taxas.split(",")}
        for k in list(TAXAS):
            if k not in manter_t:
                del TAXAS[k]

    linhas = []
    for fonte in args.fontes.split(","):
        print(f"\n=== {fonte} ===")
        if fonte.strip() == "refit":
            linhas += fonte_refit(casas=tuple(int(c) for c in args.casas_refit.split(",")))
        else:
            linhas += fonte_eco()

    if not linhas:
        return
    idx = pd.DataFrame(linhas).sort_values(["fonte", "aparelho", "taxa", "particao"])
    SAIDA.mkdir(parents=True, exist_ok=True)
    idx.to_csv(SAIDA / "indice.csv", index=False, encoding="utf-8")
    print(f"\n{len(idx)} arquivos em {SAIDA}")
    print(idx[["arquivo", "dias", "pct_ligado", "pct_ligado_nativo",
               "ap_medio_ligado_W"]].to_string(index=False))


if __name__ == "__main__":
    main()
