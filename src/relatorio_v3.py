"""Relatorio da escada do v3 (docs/v3.md §5, §6 E2.1/E2.4, portao G2).

Monta os modelos da escada a partir das predicoes salvas:

  B0   A0_4hcf do v2 (CNN 4 h + 25% de pares)          resultados/v2/pred/...
  B1-B3, B2r, Q9                                        resultados/v3/pred/<tag>/
  B4   R9
  B5   media de postos de R5, R7, R9, R15, R25 (mesma semente) = ReCo
  B6   media de postos de A5, ..., A25 (auxiliares)
  B7   B5 + o ramo so-evidencia A2 do v2 (mesma semente)
  CRAFT  media de postos de G5..G25: o B5 sem L_inv, L_sens e auxiliares (pos-v3,
       os componentes que a escada mostrou sem efeito foram descartados)

Por casa de validacao (REFIT 3/7/20) e modelo, media das sementes: AUC, AP,
oraculo e duas decisoes, ambas calibradas nas OUTRAS casas de validacao:

  d    indutiva (regra congelada no portao G1): media dos logits dos membros,
       Platt + limiar oraculo na escala calibrada (a `d` do v2)
  q    transdutiva (A-t / A+-t): media de postos + quantil, k das outras casas,
       pi = mediana das prevalencias das casas fonte

Comparacoes pareadas contra B0 (bootstrap por blocos de dia, 1000 replicas):
  dAUC  media sobre as 3 casas da AUC (postos medios das sementes), dias
        reamostrados dentro de cada casa
  dF1   F1 das contagens agregadas das 3 casas (regra d), dias x sementes

Uso:
    python src/relatorio_v3.py --fonte A --tags escada_A_tv,escada_A_wm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decisao import Decisor, f1, limiar_oraculo, logit           # noqa: E402
from estatistica import blocos_dia, bootstrap_f1                 # noqa: E402
from experimento_quantil import FONTES_AMAIS, prev_fonte         # noqa: E402
from quantil import decidir_quantil, k_de_dobras, posto_percentual  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
V2 = RAIZ / "resultados" / "v2" / "pred"
V3 = RAIZ / "resultados" / "v3"
KS = (5, 7, 9, 15, 25)
B0_V2 = {"A": ("escada_A2", "A0_4hcf"), "A+": ("escada_Amais2", "A0_4hcf")}
EVID_V2 = {"A": [("escada_A", "A2")], "A+": [("escada_Amais2", "A2"), ("escada_Amais", "A2")]}


def ler(pasta: Path, aparelho: str, modelo: str) -> dict[int, dict]:
    out = {}
    for arq in sorted(pasta.glob(f"{aparelho.replace(' ', '_')}__{modelo}__s*.npz")):
        s = int(arq.stem.rsplit("__s", 1)[1])
        d = np.load(arq)
        r = {}
        for k in d.files:
            alvo, campo = k.rsplit("|", 1)
            if alvo.startswith("cv:"):
                r.setdefault(alvo, {})[campo] = d[k]
        for a in r:
            if "z" not in r[a]:
                r[a]["z"] = logit(r[a]["p"])
        out[s] = r
    return out


def membros(aparelho: str, fonte: str, tags: list[str]) -> dict[str, dict[int, list[dict]]]:
    """nome da escada -> semente -> lista de membros (cada um {alvo: {y, z, t}})."""
    v3 = {}
    for tag in tags:
        pasta = V3 / "pred" / tag
        for arq in pasta.glob(f"{aparelho.replace(' ', '_')}__*__s*.npz"):
            m = arq.stem.split("__")[1]
            v3.setdefault(m, None)
    for m in list(v3):
        for tag in tags:
            r = ler(V3 / "pred" / tag, aparelho, m)
            if r:
                v3[m] = r
                break
    b0 = ler(V2 / B0_V2[fonte][0], aparelho, B0_V2[fonte][1])
    ev = {}
    for tag, m in EVID_V2[fonte]:          # a TV A+ tem o A2 so na primeira rodada A+
        ev = ev or ler(V2 / tag, aparelho, m)
    esc: dict[str, dict[int, list[dict]]] = {"B0": {s: [r] for s, r in b0.items()}}
    for m in ("B1", "B2", "B3", "B2r", "Q9"):
        if v3.get(m):
            esc[m] = {s: [r] for s, r in v3[m].items()}
    if v3.get("R9"):
        esc["B4"] = {s: [r] for s, r in v3["R9"].items()}
    for nome, pref in (("B5", "R"), ("B6", "A"), ("CRAFT", "G"), ("FUSAO", "F")):
        ms = [v3.get(f"{pref}{k}") for k in KS]
        if all(ms):
            sem = set.intersection(*[set(m) for m in ms])
            esc[nome] = {s: [m[s] for m in ms] for s in sorted(sem)}
    if "B5" in esc and ev:
        esc["B7"] = {s: esc["B5"][s] + [ev[s]] for s in esc["B5"] if s in ev}
    # pos-v3: CRAFT + o ramo so-evidencia do v2 como 6o membro, e os campeoes do v2
    if "CRAFT" in esc and ev:
        esc["CRAFT+E"] = {s: esc["CRAFT"][s] + [ev[s]] for s in esc["CRAFT"] if s in ev}
    if ev:
        esc["A2v2"] = {s: [r] for s, r in ev.items()}
    a5 = ler(V2 / ("escada_A" if fonte == "A" else "escada_Amais2"), aparelho, "A5")
    if a5:
        esc["A5v2"] = {s: [r] for s, r in a5.items()}
    return esc


def combinar(ms: list[dict], alvo: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(y, t, posto medio, logit medio) de um conjunto numa casa."""
    y, t = ms[0][alvo]["y"], ms[0][alvo]["t"]
    for m in ms[1:]:
        assert len(m[alvo]["y"]) == len(y) and (m[alvo]["y"] == y).all(), "alvos diferentes"
    posto = np.mean([posto_percentual(m[alvo]["z"]) for m in ms], 0)
    z = np.mean([m[alvo]["z"] for m in ms], 0)
    return y, t, posto, z


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def avaliar(esc: dict, aparelho: str, fonte: str) -> tuple[pd.DataFrame, dict]:
    casas_f = [8] if fonte == "A" else FONTES_AMAIS[aparelho]
    pi = float(np.median([prev_fonte(aparelho, c) for c in casas_f]))
    linhas, bin_ = [], {}
    for nome, sems in esc.items():
        for s, ms in sems.items():
            alvos = sorted(ms[0])
            comb = {a: combinar(ms, a) for a in alvos}
            for a in alvos:
                y, t, posto, z = comb[a]
                outros = [o for o in alvos if o != a]
                yo = np.concatenate([comb[o][0] for o in outros])
                po = sig(np.concatenate([comb[o][3] for o in outros]))
                pred_d = Decisor(po, yo).decidir(sig(z))["d"]
                k = k_de_dobras([(comb[o][2], comb[o][0]) for o in outros])
                pred_q = decidir_quantil(posto, min(1.0, k * pi), semente=s)
                bin_[(nome, s, a)] = (pred_d, pred_q, posto)
                linhas.append(dict(aparelho=aparelho, fonte=fonte, modelo=nome, semente=s,
                                   alvo=a, membros=len(ms), prev=round(float(y.mean()), 4),
                                   auc=roc_auc_score(y, posto),
                                   auc_logit=roc_auc_score(y, z),
                                   ap=average_precision_score(y, posto),
                                   oraculo=limiar_oraculo(posto, y)[1],
                                   f1_d=f1(y, pred_d), f1_q=f1(y, pred_q),
                                   r_q=float(pred_q.mean()), k=k, pi=pi))
    return pd.DataFrame(linhas), bin_


def auc_rapida(y: np.ndarray, s: np.ndarray) -> float:
    r = posto_percentual(s) * len(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def pareado(esc, bin_, d: pd.DataFrame, ref: str = "B0", n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    linhas = []
    alvos = sorted(d.alvo.unique())
    sem_ref = sorted(esc[ref])
    y = {a: esc[ref][sem_ref[0]][0][a]["y"] for a in alvos}
    t = {a: esc[ref][sem_ref[0]][0][a]["t"] for a in alvos}
    dias = {a: blocos_dia(t[a]) for a in alvos}
    grupos = {a: [np.flatnonzero(dias[a] == u) for u in np.unique(dias[a])] for a in alvos}
    for nome in esc:
        if nome == ref:
            continue
        sems = sorted(set(esc[nome]) & set(esc[ref]))
        if not sems:
            continue
        # escore medio das sementes (postos) para a AUC
        sm = {a: np.mean([bin_[(nome, s, a)][2] for s in sems], 0) for a in alvos}
        sr = {a: np.mean([bin_[(ref, s, a)][2] for s in sems], 0) for a in alvos}
        dif = np.empty(n)
        for i in range(n):
            vals = []
            for a in alvos:
                idx = np.concatenate([grupos[a][j] for j in
                                      rng.integers(0, len(grupos[a]), len(grupos[a]))])
                vals.append(auc_rapida(y[a][idx], sm[a][idx]) - auc_rapida(y[a][idx], sr[a][idx]))
            dif[i] = np.nanmean(vals)
        dauc = np.mean([auc_rapida(y[a], sm[a]) - auc_rapida(y[a], sr[a]) for a in alvos])
        # F1 agregado das 3 casas, regra d e regra q, dias (unicos por casa) x sementes
        yy = np.concatenate([y[a] for a in alvos])
        dd = np.concatenate([dias[a] + 100_000 * i for i, a in enumerate(alvos)])
        out = dict(modelo=nome, ref=ref, sementes=len(sems), dAUC=round(dauc, 4),
                   dAUC_ic=tuple(np.percentile(dif, [2.5, 97.5]).round(4)))
        for j, regra in ((0, "d"), (1, "q")):
            pm = np.stack([np.concatenate([bin_[(nome, s, a)][j] for a in alvos]) for s in sems])
            pr = np.stack([np.concatenate([bin_[(ref, s, a)][j] for a in alvos]) for s in sems])
            b = bootstrap_f1(yy, pm, dd, n=n, pred_b=pr)
            out[f"F1_{regra}"] = round(float(np.mean([f1(yy, x) for x in pm])), 4)
            out[f"F1_{regra}_ref"] = round(float(np.mean([f1(yy, x) for x in pr])), 4)
            out[f"dF1_{regra}_ic"] = (b["dif_ic"][0], b["dif_ic"][2])
        linhas.append(out)
        print(f"  {nome:4s} vs {ref}: dAUC {dauc:+.3f} [{out['dAUC_ic'][0]:+.3f}; "
              f"{out['dAUC_ic'][1]:+.3f}]  F1d {out['F1_d']:.3f} vs {out['F1_d_ref']:.3f} "
              f"[{out['dF1_d_ic'][0]:+.3f}; {out['dF1_d_ic'][1]:+.3f}]  F1q {out['F1_q']:.3f}",
              flush=True)
    return pd.DataFrame(linhas)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fonte", default="A")
    p.add_argument("--tags", required=True)
    p.add_argument("--aparelhos", default="television,washing machine")
    p.add_argument("--saida", default="")
    p.add_argument("--n_boot", type=int, default=1000)
    args = p.parse_args()
    tags = args.tags.split(",")
    nome = args.saida or f"escada_{args.fonte.replace('+', 'mais')}"
    res, par = [], []
    for ap in args.aparelhos.split(","):
        esc = membros(ap, args.fonte, tags)
        d, bin_ = avaliar(esc, ap, args.fonte)
        res.append(d)
        print(f"\n=== {ap} | faixa {args.fonte} | modelos {sorted(esc)}")
        g = d.groupby(["modelo", "alvo"])[["auc", "ap", "oraculo", "f1_d", "f1_q"]].mean()
        print(g.unstack("alvo").round(3).to_string())
        m = d.groupby(["modelo", "semente"])[["auc", "ap", "oraculo", "f1_d", "f1_q"]].mean()
        print(m.groupby("modelo").mean().round(3).to_string())
        pp = pareado(esc, bin_, d, n=args.n_boot)
        pp.insert(0, "aparelho", ap)
        par.append(pp)
    pd.concat(res).to_csv(V3 / f"{nome}__sementes.csv", index=False)
    pd.concat(par).to_csv(V3 / f"{nome}__pareado.csv", index=False)
    r = pd.concat(res).groupby(["aparelho", "modelo", "semente"])[
        ["auc", "ap", "oraculo", "f1_d", "f1_q"]].mean().groupby(["aparelho", "modelo"]).mean()
    r.round(4).to_csv(V3 / f"{nome}__resumo.csv")


if __name__ == "__main__":
    main()
