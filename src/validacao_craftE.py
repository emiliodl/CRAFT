"""Passo 1 do fechamento unificado: CRAFT x CRAFT+E so na validacao.

CRAFT   = media dos logits de G5..G25 (mesma semente)
CRAFT+E = CRAFT + o ramo so-evidencia A2 do v2 como 6o membro (peso 1/6)

Dois conjuntos de validacao, nenhum de teste:

  cv   casas REFIT 3/7/20 (proxy do cross): AUC e F1 da regra d_L em LOHO
       (Platt + limiar nas outras duas casas)
  val  so na faixa A: holdout temporal da casa 8 (4 ultimas semanas; proxy do
       intra), alinhado pelos instantes comuns ao v2 e ao v3: AUC e oraculo.
       Sem F1 decidido: a d_A e calibrada nele. Na A+ o 'val' mistura fontes
       que diferem entre v2 e v3 e fica de fora.

Uso:
    python src/validacao_craftE.py
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
CASOS = [  # (aparelho, faixa, pastas do CRAFT, (pasta, modelo) do E, (pasta, modelo) do B0)
    ("television", "A", ["craft_A"], ("escada_A", "A2"), ("escada_A2", "A0_4hcf")),
    ("washing machine", "A", ["craft_A"], ("escada_A", "A2"), ("escada_A2", "A0_4hcf")),
    ("freezer", "A", ["craft_fz2_A"], ("escada_A", "A2"), ("escada_A2", "A0_4hcf")),
    ("television", "A+", ["craft_tv815"], ("escada_Amais", "A2"), ("escada_Amais2", "A0_4hcf")),
    ("washing machine", "A+", ["craft_Amais_wm"], ("escada_Amais2", "A2"),
     ("escada_Amais2", "A0_4hcf")),
    ("freezer", "A+", ["craft_fz2_Amais"], ("escada_Amais2", "A2"), ("escada_Amais2", "A0_4hcf")),
]


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def ler(arq: Path) -> dict | None:
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


def media(ms: list[dict]) -> dict:
    """{alvo: (y, z medio)}. As casas cv sao identicas entre v2 e v3; o 'val' nao
    (o v3 exige a janela inteira coberta na grade de 30 s), entao ele e alinhado
    pela intersecao dos instantes."""
    out = {}
    for a in ms[0]:
        ts = [m[a]["t"] for m in ms]
        comum = ts[0]
        for t in ts[1:]:
            comum = np.intersect1d(comum, t)
        idx = [np.intersect1d(t, comum, return_indices=True)[1] for t in ts]
        y = ms[0][a]["y"][idx[0]]
        for m, i in zip(ms[1:], idx[1:]):
            assert (m[a]["y"][i] == y).all(), f"rotulos diferentes em {a}"
        out[a] = (y, np.mean([m[a]["z"][i] for m, i in zip(ms, idx)], 0))
    return out


def restringir(ms: list[dict], ref: list[dict]) -> list[dict]:
    """Restringe o 'val' de `ms` aos instantes comuns com `ref` (comparacao justa)."""
    t_ref = ref[0]["val"]["t"]
    for m in ref[1:]:
        t_ref = np.intersect1d(t_ref, m["val"]["t"])
    out = []
    for m in ms:
        i = np.isin(m["val"]["t"], t_ref)
        out.append(dict(m, val={k: v[i] for k, v in m["val"].items()}))
    return out


def medir(nome: str, c: dict) -> dict:
    cvs = sorted(a for a in c if a.startswith("cv:"))
    lin = {"modelo": nome}
    aucs, f1s = [], []
    for h in cvs:
        y, z = c[h]
        outros = [o for o in cvs if o != h]
        yo = np.concatenate([c[o][0] for o in outros])
        po = sig(np.concatenate([c[o][1] for o in outros]))
        aucs.append(roc_auc_score(y, z))
        f1s.append(f1(y, Decisor(po, yo).decidir(sig(z))["d"]))
    lin.update(cv_auc=np.mean(aucs), cv_f1dL=np.mean(f1s))
    if "val" not in c:
        return lin
    y, z = c["val"]
    lin.update(val_auc=roc_auc_score(y, z) if 0 < y.sum() < len(y) else np.nan,
               val_oraculo=limiar_oraculo(z, y)[1], val_prev=float(y.mean()))
    return lin


def rodar(ap, faixa, tags_c, ev, b0) -> list[dict]:
    slug = ap.replace(" ", "_")
    linhas = []
    for s in range(5):
        cr = []
        for k in KS:
            m = next((r for t in tags_c if (r := ler(V3 / t / f"{slug}__G{k}__s{s}.npz"))), None)
            cr.append(m)
        e = ler(V2 / ev[0] / f"{slug}__{ev[1]}__s{s}.npz")
        b = ler(V2 / b0[0] / f"{slug}__{b0[1]}__s{s}.npz")
        if not all(cr) or e is None:
            continue
        todos = cr + [e] + ([b] if b is not None else [])
        for nome, ms in (("CRAFT", cr), ("CRAFT+E", cr + [e]), ("A2v2", [e]), ("B0", [b])):
            if ms[-1] is None:
                continue
            if faixa == "A":
                ms = restringir(ms, todos)      # mesmo 'val' (casa 8) para os quatro
            else:                               # A+: 'val' de fontes diferentes v2 x v3
                ms = [{k: v for k, v in m.items() if k != "val"} for m in ms]
            try:
                linhas.append(dict(aparelho=ap, faixa=faixa, semente=s, **medir(nome, media(ms))))
            except AssertionError as err:
                print(f"  {ap} {faixa} s{s} {nome}: {err}")
    return linhas


def main() -> None:
    d = pd.DataFrame([l for c in CASOS for l in rodar(*c)])
    d.to_csv(RAIZ / "resultados" / "v3" / "validacao_craftE__sementes.csv", index=False)
    cols = ["cv_auc", "cv_f1dL", "val_auc", "val_oraculo"]
    r = d.groupby(["aparelho", "faixa", "modelo"], sort=False)[cols].mean().round(3)
    r["sementes"] = d.groupby(["aparelho", "faixa", "modelo"], sort=False).size()
    r.to_csv(RAIZ / "resultados" / "v3" / "validacao_craftE__resumo.csv")
    print(r.to_string())


if __name__ == "__main__":
    main()
