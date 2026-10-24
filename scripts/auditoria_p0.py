"""P0 do v2 (docs/v2.md §4, E0.1 e E0.2): auditoria de bordas, lag e lacunas.

Le cada casa UMA vez na resolucao nativa (REFIT ~8 s, ECO 1 s) e produz:

  resultados/p0_auditoria.csv    por casa x aparelho: bordas, lag de maior |corr|,
                                 correlacao no lag 0, degrau do aparelho e do
                                 agregado, residuo +- EP, e a mesma media disparada
                                 SEM as bordas que caem perto de lacuna
  resultados/p0_lacunas.csv      fracao de amostras em lacuna por canal e mes
  resultados/p0_curvas.csv       curvas disparadas por evento (para figura)
  datasets/protocolo_artigo/lacunas/<arquivo>.npz
                                 fracao de lacuna por intervalo, alinhada ao `t`
                                 do npz do protocolo (agregado e aparelho)

Regras do portao (v2 §3.0 item 2): sinalizar se o lag de maior |corr| != 0, se a
correlacao no lag 0 for negativa, ou se |residuo| > 30% do degrau do aparelho.

Uso:
    python scripts/auditoria_p0.py
    python scripts/auditoria_p0.py --casas_refit 8 --eco 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from preparar_artigo import (JANELAS, LIMIAR_ON, PADROES, TAXAS, _slug,  # noqa: E402
                             plugs_da_casa)
from saneamento import (auditoria_bordas, bordas_de_sessao, corridas_de_zero, fracao_lacuna,   # noqa: E402
                        melhor_lag)

DADOS = RAIZ / "datasets"
PROTO = DADOS / "protocolo_artigo"
SAIDA = RAIZ / "resultados"
APARELHOS = ("freezer", "washing machine", "television")
MEIA = 20                      # amostras nativas de cada lado da borda na media disparada


def resumo_bordas(agg, app, t_s, lac, aparelho, rotulo_fonte):
    """Linha de auditoria + curvas. `lac` = mascara nativa de lacuna (agregado OU
    aparelho). A segunda media disparada exclui bordas com lacuna em +-MEIA."""
    limiar = LIMIAR_ON[aparelho]
    passo = float(np.median(np.diff(t_s)))
    tab, n = auditoria_bordas(agg, app, limiar, MEIA)
    lag, c_lag, c0 = melhor_lag(agg, app, max_lag=120)
    # mesma media, sem bordas perto de lacuna
    bordas = bordas_de_sessao(app, limiar, MEIA)
    perto = np.array([lac[b - MEIA:b + MEIA + 1].any() for b in bordas], bool)
    tab2, n2 = auditoria_bordas(agg, app, limiar, MEIA, excluir=lac)

    def degraus(t):
        depois = t[(t.lag >= 1) & (t.lag <= 5)]
        return (float(depois.aparelho.mean()), float(depois.agregado.mean()),
                float(depois.residuo.mean()), float(depois.ep_residuo.mean()))

    d_ap, d_ag, res, ep = degraus(tab)
    d_ap2, d_ag2, res2, ep2 = degraus(tab2)
    alertas = []
    if lag != 0:
        alertas.append(f"lag={lag}")
    if c0 < 0:
        alertas.append("corr0<0")
    if np.isfinite(res) and np.isfinite(d_ap) and abs(res) > 0.3 * abs(d_ap):
        alertas.append("residuo>30%")
    linha = dict(fonte=rotulo_fonte, aparelho=aparelho, passo_s=round(passo, 1),
                 bordas=n, lag=lag, lag_s=round(lag * passo, 0), corr_lag=round(c_lag, 3),
                 corr0=round(c0, 3), degrau_aparelho=round(d_ap, 1),
                 degrau_agregado=round(d_ag, 1), residuo=round(res, 1), ep_residuo=round(ep, 1),
                 bordas_sem_lacuna=n2, bordas_em_lacuna=int(perto.sum()),
                 degrau_agregado_sem_lacuna=round(d_ag2, 1), residuo_sem_lacuna=round(res2, 1),
                 fracao_lacuna=round(float(lac.mean()), 4),
                 alertas=";".join(alertas) or "ok")
    curvas = tab.assign(fonte=rotulo_fonte, aparelho=aparelho, versao="todas")
    curvas2 = tab2.assign(fonte=rotulo_fonte, aparelho=aparelho, versao="sem_lacuna")
    return linha, pd.concat([curvas, curvas2])


def degrau_5min(t5, agg5, st5, lac5):
    """A leitura que o registro fez a 5 min: Delta do agregado na transicao
    desligado->ligado do rotulo, com e sem intervalos em lacuna."""
    contig = np.diff(t5) == 300
    sobe = (st5[:-1] == 0) & (st5[1:] == 1) & contig
    d = np.diff(agg5)
    limpo = sobe & (lac5[:-1] == 0) & (lac5[1:] == 0)
    return (float(d[sobe].mean()) if sobe.any() else np.nan, int(sobe.sum()),
            float(d[limpo].mean()) if limpo.any() else np.nan, int(limpo.sum()))


def lacunas_por_mes(t_s, masc, rotulo_fonte, canal):
    mes = pd.to_datetime(t_s, unit="s").to_period("M").astype(str)
    return (pd.DataFrame({"mes": mes, "m": masc}).groupby("mes")["m"].mean()
            .rename("fracao").reset_index().assign(fonte=rotulo_fonte, canal=canal))


def salvar_lacunas(arquivo_npz: str, t_nat, lac_agg, lac_app):
    caminho = PROTO / arquivo_npz
    if not caminho.exists():
        return None
    t5 = np.load(caminho)["t"]
    taxa = TAXAS["15min" if "15min" in arquivo_npz else "5min"]
    fa = fracao_lacuna(t_nat, lac_agg, t5, taxa)
    fp = fracao_lacuna(t_nat, lac_app, t5, taxa)
    (PROTO / "lacunas").mkdir(exist_ok=True)
    np.savez_compressed(PROTO / "lacunas" / arquivo_npz, t=t5, agregado=fa, aparelho=fp)
    return t5, fa, fp


def refit(casa: int, linhas, curvas, meses, degraus5):
    pasta = DADOS / "refit" / "CLEAN_REFIT_081116"
    meta = pd.ExcelFile(pasta / "MetaData_Tables.xlsx").parse(f"House {casa}")
    canais = {}
    for _, r in meta.iterrows():
        if pd.isna(r.iloc[0]):
            continue
        for ap in APARELHOS:
            if ap not in canais and PADROES[ap].match(str(r.iloc[1]).strip()):
                canais[ap] = int(r.iloc[0])
    cols = ["Time", "Aggregate", "Issues"] + [f"Appliance{n}" for n in canais.values()]
    df = pd.read_csv(pasta / f"CLEAN_House{casa}.csv", usecols=cols, parse_dates=["Time"])
    df = df.set_index("Time").sort_index()
    a = JANELAS["refit_treino"][0]
    b = JANELAS["refit_teste"][1] if casa == 8 else JANELAS["refit_treino"][1]
    df = df.loc[a:b]
    t_s = (df.index.view("int64") // 10**9).astype(np.int64)
    agg = df["Aggregate"].to_numpy(float)
    lac_agg = corridas_de_zero(agg, t_s)
    meses.append(lacunas_por_mes(t_s, lac_agg, f"refit {casa}", "agregado"))
    print(f"  refit {casa}: {len(df)} linhas, lacuna no agregado {100 * lac_agg.mean():.2f}%, "
          f"issues {100 * df['Issues'].mean():.1f}%", flush=True)
    for ap, n in canais.items():
        app = df[f"Appliance{n}"].to_numpy(float)
        lac_app = corridas_de_zero(app, t_s, minimo_s=1e9)   # plugue: zero e legitimo
        # no plugue zero e "desligado"; lacuna do plugue so aparece como buraco no tempo
        buraco = np.zeros(len(t_s), bool)
        salto = np.flatnonzero(np.diff(t_s) > 120) + 1
        buraco[salto] = True
        lac = lac_agg | buraco
        linha, cv = resumo_bordas(agg, app, t_s, lac, ap, f"refit {casa}")
        bs = bordas_de_sessao(app, LIMIAR_ON[ap], MEIA)
        linha["issues_nas_bordas"] = round(float(df["Issues"].to_numpy()[bs].mean()), 3) if len(bs) else None
        for part in (("treino", "teste") if casa == 8 else ("treino",)):
            arq = f"refit_casa_{casa}__{_slug(ap)}__5min__{part}.npz"
            r = salvar_lacunas(arq, t_s, lac_agg, lac_app | buraco)
            salvar_lacunas(arq.replace("5min", "15min"), t_s, lac_agg, lac_app | buraco)
            if r is not None and part == "treino":
                d = np.load(PROTO / arq)
                st = d[f"status_{_slug(ap)}"]
                m, nm, ml, nml = degrau_5min(r[0], d["main"].astype(float), st, r[1] > 0)
                degraus5.append(dict(fonte=f"refit {casa}", aparelho=ap, particao=part,
                                     d_agregado_5min=round(m, 1), n=nm,
                                     d_agregado_5min_sem_lacuna=round(ml, 1), n_sem_lacuna=nml,
                                     intervalos_em_lacuna=round(float((r[1] > 0).mean()), 4)))
        linhas.append(linha)
        curvas.append(cv)
        print(f"    {ap:16s} bordas={linha['bordas']:5d} lag={linha['lag']:+4d} "
              f"corr0={linha['corr0']:+.3f} degrau ap={linha['degrau_aparelho']:7.1f} "
              f"agg={linha['degrau_agregado']:7.1f} res={linha['residuo']:7.1f}"
              f"+-{linha['ep_residuo']:.1f}  [{linha['alertas']}]", flush=True)


def eco(casa: int, linhas, curvas, meses, degraus5):
    pasta = DADOS / "ECO"
    cid = f"{casa:02d}"
    dir_sm, dir_pl = pasta / f"{cid}_sm_csv" / cid, pasta / f"{cid}_plugs_csv" / cid
    if not dir_sm.exists():
        return
    mapa = {k: v for k, v in plugs_da_casa(casa, pasta / f"{cid}_doc.txt").items()
            if k in APARELHOS}
    a, b = JANELAS["eco_teste"]
    for ap, plug in mapa.items():
        partes = []
        for arq_sm in sorted(dir_sm.glob("*.csv")):
            if not (a <= arq_sm.stem <= b) or not (dir_pl / plug / arq_sm.name).exists():
                continue
            main = pd.read_csv(arq_sm, header=None, usecols=[0]).iloc[:, 0].to_numpy(float)
            pot = pd.read_csv(dir_pl / plug / arq_sm.name, header=None).iloc[:, 0].to_numpy(float)
            k = min(len(main), len(pot))
            main, pot = main[:k], pot[:k]
            main[main < 0] = np.nan
            pot[pot < 0] = np.nan
            t0 = int(pd.Timestamp(arq_sm.stem).value // 10**9)
            partes.append((t0 + np.arange(k), main, pot))
        if not partes:
            continue
        t_s = np.concatenate([p[0] for p in partes])
        agg = np.concatenate([p[1] for p in partes])
        app = np.concatenate([p[2] for p in partes])
        lac_agg = corridas_de_zero(agg, t_s)
        lac_app = ~np.isfinite(app)
        meses.append(lacunas_por_mes(t_s, lac_agg, f"eco {casa}", "agregado"))
        agg_f = np.where(np.isfinite(agg), agg, np.nan)
        linha, cv = resumo_bordas(agg_f, np.nan_to_num(app, nan=0.0), t_s, lac_agg | lac_app,
                                  ap, f"eco {casa}")
        linhas.append(linha)
        curvas.append(cv)
        arq = f"eco_casa_{casa}__{_slug(ap)}__5min__cross.npz"
        r = salvar_lacunas(arq, t_s, lac_agg, lac_app)
        salvar_lacunas(arq.replace("5min", "15min"), t_s, lac_agg, lac_app)
        if r is not None:
            d = np.load(PROTO / arq)
            m, nm, ml, nml = degrau_5min(r[0], d["main"].astype(float),
                                         d[f"status_{_slug(ap)}"], r[1] > 0)
            degraus5.append(dict(fonte=f"eco {casa}", aparelho=ap, particao="cross",
                                 d_agregado_5min=round(m, 1), n=nm,
                                 d_agregado_5min_sem_lacuna=round(ml, 1), n_sem_lacuna=nml,
                                 intervalos_em_lacuna=round(float((r[1] > 0).mean()), 4)))
        print(f"  eco {casa} {ap:16s} bordas={linha['bordas']:5d} lag={linha['lag']:+4d} "
              f"corr0={linha['corr0']:+.3f} degrau ap={linha['degrau_aparelho']:7.1f} "
              f"agg={linha['degrau_agregado']:7.1f} res={linha['residuo']:7.1f}"
              f"+-{linha['ep_residuo']:.1f}  [{linha['alertas']}]", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--casas_refit", default="8,1,3,4,7,13,17,18,20")
    p.add_argument("--eco", default="1,2,5")
    p.add_argument("--sufixo", default="", help="sufixo dos CSVs de saida (nao sobrescreve)")
    args = p.parse_args()
    linhas, curvas, meses, degraus5 = [], [], [], []
    for c in [int(x) for x in args.casas_refit.split(",") if x]:
        refit(c, linhas, curvas, meses, degraus5)
    for c in [int(x) for x in args.eco.split(",") if x]:
        eco(c, linhas, curvas, meses, degraus5)
    SAIDA.mkdir(exist_ok=True)
    pd.DataFrame(linhas).to_csv(SAIDA / f"p0_auditoria{args.sufixo}.csv", index=False)
    pd.concat(curvas).to_csv(SAIDA / f"p0_curvas{args.sufixo}.csv", index=False)
    pd.concat(meses).to_csv(SAIDA / f"p0_lacunas{args.sufixo}.csv", index=False)
    pd.DataFrame(degraus5).to_csv(SAIDA / f"p0_degraus_5min{args.sufixo}.csv", index=False)
    print("\n" + pd.DataFrame(degraus5).to_string(index=False))


if __name__ == "__main__":
    main()
