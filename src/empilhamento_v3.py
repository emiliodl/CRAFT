"""Juncao CRAFT + ramo de evidencia (1): empilhamento com peso escolhido no LOHO.

Escore = w * E + (1 - w) * C, com E e C os logits do ramo so-evidencia (A2 do
v2) e do CRAFT (media dos logits de G5..G25), cada um PADRONIZADO pela media e
desvio no holdout temporal das fontes ('val', sem rotulo de casa-alvo). O peso w
(grade 0, 0,1, ..., 1) e escolhido por aparelho e faixa em LOHO ANINHADO: para a
casa retida h, o w que maximiza a AUC media das outras casas de validacao; h so
e usada para medir. A decisao e a d_L sobre o escore combinado.

Uso:
    python src/empilhamento_v3.py
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
GRADE = np.round(np.linspace(0, 1, 11), 2)
CASOS = [  # (aparelho, faixa, tag do CRAFT na validacao, tags do A2 do v2)
    ("television", "A", "craft_A", ["escada_A"]),
    ("washing machine", "A", "craft_A", ["escada_A"]),
    ("television", "A+", "craft_tv815u5", ["escada_Amais2", "escada_Amais"]),
    ("washing machine", "A+", "craft_Amais_wm", ["escada_Amais2"]),
    ("freezer", "A", "craft_fz_A", ["escada_A"]),
    ("freezer", "A+", "craft_fz_Amais", ["escada_Amais2"]),
]


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def ler(arq: Path) -> dict:
    d = np.load(arq)
    r = {}
    for k in d.files:
        a, c = k.rsplit("|", 1)
        r.setdefault(a, {})[c] = d[k]
    for a in r:
        if "z" not in r[a]:
            r[a]["z"] = logit(r[a]["p"])
    return r


def padronizado(r: dict) -> dict:
    """{alvo: (y, z padronizado pelo 'val')}"""
    mu, dp = r["val"]["z"].mean(), r["val"]["z"].std() + 1e-9
    return {a: (v["y"], (v["z"] - mu) / dp) for a, v in r.items()}


def rodar(ap: str, faixa: str, tag: str, tags_e: list[str]) -> list[dict]:
    slug = ap.replace(" ", "_")
    linhas = []
    for s in range(5):
        arqs = [V3 / tag / f"{slug}__G{k}__s{s}.npz" for k in KS]
        arq_e = next((V2 / t / f"{slug}__A2__s{s}.npz" for t in tags_e
                      if (V2 / t / f"{slug}__A2__s{s}.npz").exists()), None)
        if not all(a.exists() for a in arqs) or arq_e is None:
            continue
        ms = [ler(a) for a in arqs]
        C = {a: (ms[0][a]["y"], np.mean([m[a]["z"] for m in ms], 0)) for a in ms[0]}
        mu, dp = C["val"][1].mean(), C["val"][1].std() + 1e-9
        C = {a: (y, (z - mu) / dp) for a, (y, z) in C.items()}
        E = padronizado(ler(arq_e))
        cvs = sorted(a for a in C if a.startswith("cv:"))
        for a in cvs:
            assert len(E[a][0]) == len(C[a][0]) and (E[a][0] == C[a][0]).all()
        comb = lambda a, w: w * E[a][1] + (1 - w) * C[a][1]          # noqa: E731
        for h in cvs:
            outros = [o for o in cvs if o != h]
            aucs = [np.mean([roc_auc_score(C[o][0], comb(o, w)) for o in outros]) for w in GRADE]
            w = float(GRADE[int(np.argmax(aucs))])
            y = C[h][0]
            yo = np.concatenate([C[o][0] for o in outros])
            lin = dict(aparelho=ap, faixa=faixa, semente=s, casa=h, w=w)
            for nome, ww in (("CRAFT", 0.0), ("evidencia", 1.0), ("empilhado", w)):
                zo = np.concatenate([comb(o, ww) for o in outros])
                z = comb(h, ww)
                lin[f"auc_{nome}"] = roc_auc_score(y, z)
                lin[f"f1d_{nome}"] = f1(y, Decisor(sig(zo), yo).decidir(sig(z))["d"])
                lin[f"ora_{nome}"] = limiar_oraculo(z, y)[1]
            linhas.append(lin)
    return linhas


def main() -> None:
    d = pd.DataFrame([l for c in CASOS for l in rodar(*c)])
    d.to_csv(RAIZ / "resultados" / "v3" / "empilhamento__sementes.csv", index=False)
    cols = [f"{m}_{n}" for m in ("auc", "f1d") for n in ("CRAFT", "evidencia", "empilhado")]
    r = d.groupby(["aparelho", "faixa"])[cols + ["w"]].mean().round(3)
    r.to_csv(RAIZ / "resultados" / "v3" / "empilhamento__resumo.csv")
    print(r.to_string())


if __name__ == "__main__":
    main()
