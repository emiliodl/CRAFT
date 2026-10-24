"""Pos-quali: variantes do CRAFT so na validacao (nenhum teste).

Item 2 (maquina de lavar): R1 com ativacao deformada (esticamento, ganho por trecho,
emenda de ciclos); variante com mais R1 no lote.
Item 3 (freezer): R1 como trem de ciclos, caracteristicas globais de periodicidade e
parada antecipada por AUC.
Saida de regressao (todos): saida extra que regride log1p(P/P_ref) no centro, BCE + 0,5 MSE.

Para cada variante e semente, o conjunto e a media dos logits de G5..G25 (CRAFT so
ResNets) e, como linha secundaria, com o ramo de evidencia A2 do v2 como 6o membro
(o mesmo em todas as variantes). Dois conjuntos de validacao:

  cv   REFIT 3/7/20 (proxy do S2): AUC, F1 da regra d_L em LOHO e F1 - F1(sempre
       ligado) da propria casa, media das tres casas
  val  holdout temporal da casa 8 (proxy do S1): AUC e F1 com o melhor limiar
       (a d_A e calibrada nele, entao nao ha F1 decidido)

Diferencas pareadas por semente contra a referencia retreinada com o mesmo codigo.

Uso:
    python src/validacao_posquali.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decisao import Decisor, f1, limiar_oraculo, logit            # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
V2, V3 = RAIZ / "resultados" / "v2" / "pred", RAIZ / "resultados" / "v3" / "pred"
KS = (5, 7, 9, 15, 25)
SEMENTES = range(5)
E = ("escada_A", "A2")                    # ramo de evidencia do v2, faixa A
VARIANTES = {
    "television": [("ta_tv", "referencia"), ("q_tv_reg", "regressao da potencia"),
                   ("q_tv_dif", "entrada so diferencas"), ("q_tv_inv1", "invariancia lam=1"),
                   ("q_tv_inv3", "invariancia lam=3")],
    "washing machine": [("q_wm_base", "referencia"), ("q_wm_def", "R1 deformada"),
                        ("q_wm_def30", "R1 deformada, 30% R1"),
                        ("q_wm_reg", "regressao da potencia"),
                        ("q_wm_dif", "entrada so diferencas"), ("q_wm_inv1", "invariancia lam=1"),
                        ("q_wm_inv3", "invariancia lam=3")],
    "freezer": [("q_fz_base", "referencia"), ("q_fz_trem", "trem de ciclos"),
                ("q_fz_glob", "periodicidade"), ("q_fz_tg", "trem + periodicidade"),
                ("q_fz_tgauc", "trem + periodicidade + parada AUC"),
                ("q_fz_reg", "regressao da potencia"),
                ("q_fz_dif", "entrada so diferencas"), ("q_fz_inv1", "invariancia lam=1"),
                ("q_fz_inv3", "invariancia lam=3")],
}
METRICAS = ["cv_auc", "cv_oraculo", "cv_f1dL", "cv_f1dL_menos_trivial", "val_auc",
            "val_oraculo"]


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def ler(arq: Path) -> dict | None:
    """So as chaves de validacao (cv: e val); nada de teste e lido."""
    if not arq.exists():
        return None
    d = np.load(arq)
    r = {}
    for k in d.files:
        a, c = k.rsplit("|", 1)
        if a.startswith("cv:") or a == "val":
            r.setdefault(a, {})[c] = d[k]
    for a in r:
        if "z" not in r[a]:
            r[a]["z"] = logit(r[a]["p"])
    return r


def alinhar(ms: list[dict], t_val: np.ndarray) -> dict:
    """{alvo: (y, z medio)}, com o 'val' restrito aos instantes de `t_val`."""
    out = {}
    for a in ms[0]:
        if a == "val":
            idx = [np.flatnonzero(np.isin(m[a]["t"], t_val)) for m in ms]
            for m, i in zip(ms, idx):
                assert np.array_equal(m[a]["t"][i], t_val), "val desalinhado"
        else:
            idx = [np.arange(len(m[a]["y"])) for m in ms]
        y = ms[0][a]["y"][idx[0]]
        for m, i in zip(ms[1:], idx[1:]):
            assert (m[a]["y"][i] == y).all(), f"rotulos diferentes em {a}"
        out[a] = (y, np.mean([m[a]["z"][i] for m, i in zip(ms, idx)], 0))
    return out


def medir(c: dict) -> dict:
    cvs = sorted(a for a in c if a.startswith("cv:"))
    aucs, f1s, ganhos, oras = [], [], [], []
    for h in cvs:
        y, z = c[h]
        outros = [o for o in cvs if o != h]
        yo = np.concatenate([c[o][0] for o in outros])
        po = sig(np.concatenate([c[o][1] for o in outros]))
        f = f1(y, Decisor(po, yo).decidir(sig(z))["d"])
        pi = float(y.mean())
        aucs.append(roc_auc_score(y, z))
        oras.append(limiar_oraculo(sig(z), y)[1])
        f1s.append(f)
        ganhos.append(f - 2 * pi / (1 + pi))
    y, z = c["val"]
    return dict(cv_auc=np.mean(aucs), cv_oraculo=np.mean(oras), cv_f1dL=np.mean(f1s),
                cv_f1dL_menos_trivial=np.mean(ganhos),
                val_auc=roc_auc_score(y, z), val_oraculo=limiar_oraculo(sig(z), y)[1],
                **{f"f1dL_{h.split()[-1]}": v for h, v in zip(cvs, f1s)},
                **{f"auc_{h.split()[-1]}": v for h, v in zip(cvs, aucs)})


def rodar(ap: str) -> pd.DataFrame:
    slug = ap.replace(" ", "_")
    carregados = {}
    for tag, _ in VARIANTES[ap]:
        for s in SEMENTES:
            ms = [ler(V3 / tag / f"{slug}__G{k}__s{s}.npz") for k in KS]
            if all(ms):
                carregados[(tag, s)] = ms
    es = {s: ler(V2 / E[0] / f"{slug}__{E[1]}__s{s}.npz") for s in SEMENTES}
    # 'val' comum a todas as variantes e ao E: mesma amostra de comparacao
    ts = [m["val"]["t"] for ms in carregados.values() for m in ms]
    ts += [e["val"]["t"] for e in es.values() if e is not None]
    t_val = ts[0]
    for t in ts[1:]:
        t_val = np.intersect1d(t_val, t)
    linhas = []
    for (tag, s), ms in carregados.items():
        nome = dict(VARIANTES[ap])[tag]
        for conj, membros in (("ResNets", ms), ("ResNets+E", ms + [es[s]] if es[s] else None)):
            if membros is None:
                continue
            linhas.append(dict(aparelho=ap, variante=nome, tag=tag, conjunto=conj, semente=s,
                               **medir(alinhar(membros, t_val))))
    return pd.DataFrame(linhas)


def resumir(d: pd.DataFrame) -> pd.DataFrame:
    """Media por variante e diferenca pareada (por semente) contra a referencia."""
    out = []
    for (ap, conj), g in d.groupby(["aparelho", "conjunto"], sort=False):
        ref = g[g.variante == "referencia"].set_index("semente")
        for var, gv in g.groupby("variante", sort=False):
            gv = gv.set_index("semente")
            comuns = gv.index.intersection(ref.index)
            lin = dict(aparelho=ap, conjunto=conj, variante=var, sementes=len(gv))
            for m in METRICAS:
                lin[m] = gv[m].mean()
                if var != "referencia" and len(comuns):
                    dif = gv.loc[comuns, m] - ref.loc[comuns, m]
                    lin[f"d_{m}"] = dif.mean()
                    lin[f"d_{m}_pos"] = f"{int((dif > 0).sum())}/{len(dif)}"
            out.append(lin)
    return pd.DataFrame(out)


def intervencoes() -> pd.DataFrame:
    """|Delta logit| medio nas casas cv: remocao do alvo, adicao, troca de fundo e a razao
    fundo/remocao (menor = mais dependente da assinatura). Cada variante e comparada a
    referencia nos MESMOS (modelo, semente), porque rodadas divididas em processos
    gravam so parte dos modelos no csv."""
    c = ["delta_rem", "delta_add", "delta_fundo"]
    out = []
    for ap, vs in VARIANTES.items():
        slug = ap.replace(" ", "_")
        tabs = {}
        for tag, nome in vs:
            arq = RAIZ / "resultados" / "v3" / f"{tag}__{slug}.csv"
            if arq.exists():
                d = pd.read_csv(arq)
                d = d[d.alvo.str.startswith("cv:")].set_index(["modelo", "semente", "alvo"])
                if set(c) <= set(d.columns):
                    tabs[nome] = d[c].abs()
        ref = tabs.get("referencia")
        if ref is None:
            continue
        for nome, d in tabs.items():
            comum = d.index.intersection(ref.index)
            if not len(comum):
                continue
            for lado, t in (("referencia", ref), (nome, d)):
                m = t.loc[comum].mean()
                out.append(dict(aparelho=ap, variante=nome, lado=lado,
                                modelos=",".join(sorted({i[0] for i in comum})), n=len(comum),
                                **m.round(3).to_dict(),
                                razao=round(m.delta_fundo / m.delta_rem, 2)))
    return pd.DataFrame(out).drop_duplicates(["aparelho", "variante", "lado"])


def main() -> None:
    d = pd.concat([rodar(ap) for ap in VARIANTES], ignore_index=True)
    d.to_csv(RAIZ / "resultados" / "v3" / "validacao_posquali__sementes.csv", index=False)
    r = resumir(d)
    r.to_csv(RAIZ / "resultados" / "v3" / "validacao_posquali__resumo.csv", index=False)
    pd.set_option("display.width", 250)
    for ap in VARIANTES:
        print(f"\n=== {ap}")
        rr = r[r.aparelho == ap]
        print(rr[["conjunto", "variante", "sementes", *METRICAS]].round(3).to_string(index=False))
        cols = [c for c in rr.columns if c.startswith("d_") and not c.endswith("_pos")]
        print(rr[rr.variante != "referencia"][["conjunto", "variante", *cols]].round(3)
              .to_string(index=False))
        print(rr[rr.variante != "referencia"][["conjunto", "variante",
                                              *[c for c in rr.columns if c.endswith("_pos")]]]
              .to_string(index=False))
    iv = intervencoes()
    iv.to_csv(RAIZ / "resultados" / "v3" / "validacao_posquali__intervencoes.csv", index=False)
    print()
    print("=== intervencoes (|Delta logit| nas casas cv; razao = fundo / remocao)")
    print(iv.to_string(index=False))


if __name__ == "__main__":
    main()
