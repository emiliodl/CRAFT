"""Fechamento unificado do CRAFT+E (pos-v3.1): uma configuracao, todas as celulas.

CRAFT+E da semente s = media dos logits de G5, G7, G9, G15, G25 (v3, pred/<tag>)
e do A2 do v2 (pred/<tag_e>), seis membros com peso igual. Mesmo modelo para TV,
maquina de lavar e freezer; um treino por aparelho e faixa serve as celulas cross
(ECO) e intra (mes de teste da casa 8), como no protocolo do artigo.

Os membros do v2 e do v3 cobrem as mesmas janelas de teste e as mesmas casas de
validacao; o holdout 'val' da casa 8 difere (o v3 exige a janela inteira na grade
de 30 s) e e alinhado pela intersecao dos instantes.

Regras (fixadas no pre-registro):

  d_L  cross: Platt + limiar otimo nas casas REFIT 3/7/20
  d_A  intra: Platt + limiar otimo no holdout temporal da casa 8 (so faixa A)
  q    transdutiva, secundaria (media de postos + quantil)

F1 [IC95%] por bootstrap de dias (meio-dia) x sementes; p bilateral de
F1 - artigo nas mesmas replicas; Holm dentro da familia primaria. "Supera" =
limite inferior do IC acima do artigo; "vence o trivial" = IC de F1 - F1(tudo
positivo) exclui zero. O CRAFT sem o E sai ao lado, como linha secundaria.

Uso:
    python src/final_craftE.py --preregistro resultados/v3/preregistro_craftE.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decisao import Decisor, f1, limiar_oraculo, logit                  # noqa: E402
from estatistica import _contagens, _f1_replica, ativacoes, blocos_dia, holm  # noqa: E402
from experimento_quantil import FONTES_AMAIS, prev_fonte                # noqa: E402
from final_v2 import ARTIGO                                             # noqa: E402
from quantil import decidir_quantil, fracao_oraculo, k_de_dobras, posto_percentual  # noqa: E402
from taxa import TAXA                                                   # noqa: E402

# melhor algoritmo de cada celula no artigo a 15 min (docs/baseline_chatterjee2025.md)
ARTIGO_15 = {("washing machine", "intra"): 0.35, ("washing machine", "cross"): 0.30,
             ("freezer", "intra"): 0.55, ("freezer", "cross"): 0.84,
             ("television", "intra"): 0.53, ("television", "cross"): 0.36}
if TAXA == 900:
    ARTIGO = ARTIGO_15

RAIZ = Path(__file__).resolve().parents[1]
V2, V3 = RAIZ / "resultados" / "v2" / "pred", RAIZ / "resultados" / "v3" / "pred"
KS = (5, 7, 9, 15, 25)


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def ler(arq: Path) -> dict | None:
    if not arq.exists():
        return None
    d = np.load(arq)
    r = {}
    for k in d.files:
        a, c = k.rsplit("|", 1)
        r.setdefault(a, {})[c] = d[k]
    for a in r:
        if "z" not in r[a]:
            r[a]["z"] = logit(r[a]["p"])
    return r


def membros_da_semente(ap: str, tags_c: list[str], tag_e: str | None, s: int) -> list[dict] | None:
    slug = ap.replace(" ", "_")
    ms = []
    for k in KS:
        m = next((r for t in tags_c if (r := ler(V3 / t / f"{slug}__G{k}__s{s}.npz"))), None)
        if m is None:
            return None
        ms.append(m)
    if tag_e:
        e = ler(V2 / tag_e / f"{slug}__A2__s{s}.npz")
        if e is None:
            return None
        ms.append(e)
    return ms


def conjunto(ms: list[dict], alvo: str):
    """(y, t, posto medio, logit medio) nos instantes comuns a todos os membros."""
    ts = [m[alvo]["t"] for m in ms]
    comum = ts[0]
    for t in ts[1:]:
        comum = np.intersect1d(comum, t)
    idx = [np.intersect1d(t, comum, return_indices=True)[1] for t in ts]
    y = ms[0][alvo]["y"][idx[0]]
    for m, i in zip(ms[1:], idx[1:]):
        assert (m[alvo]["y"][i] == y).all(), f"rotulos diferentes em {alvo}"
    posto = np.mean([posto_percentual(m[alvo]["z"][i]) for m, i in zip(ms, idx)], 0)
    z = np.mean([m[alvo]["z"][i] for m, i in zip(ms, idx)], 0)
    return y, comum, posto, z


def replicas(y, B, dias, n: int, semente: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """F1 e F1 - F1(tudo positivo) por replica (dias x sementes), mesmo sorteio."""
    rng = np.random.default_rng(semente)
    ud, inv = np.unique(dias, return_inverse=True)
    grupos = [np.flatnonzero(inv == g) for g in range(len(ud))]
    A = _contagens(y, B, grupos)
    T = _contagens(y, np.ones(len(y)), grupos)
    fa, dif = np.empty(n), np.empty(n)
    for i in range(n):
        d = rng.integers(0, len(ud), len(ud))
        fa[i] = _f1_replica(*A, d, rng.integers(0, A[0].shape[0], A[0].shape[0]))
        dif[i] = fa[i] - _f1_replica(*T, d, np.zeros(1, int))
    return fa, dif


def celula(ap: str, cel: str, faixa: str, cfg: dict, com_e: bool, n_boot: int) -> list[dict]:
    tag_e = cfg["tag_e"] if com_e else None
    sems = [s for s in range(cfg["sementes"])
            if membros_da_semente(ap, cfg["tags_craft"], tag_e, s) is not None]
    if len(sems) < cfg["sementes"]:
        print(f"  aviso: {ap} {faixa} {cel}: {len(sems)} de {cfg['sementes']} sementes")
    if not sems:
        return []
    por_sem = {s: membros_da_semente(ap, cfg["tags_craft"], tag_e, s) for s in sems}
    ex = por_sem[sems[0]]
    alvos = sorted(k for k in ex[0] if k.startswith(cel))
    cvs = sorted(k for k in ex[0] if k.startswith("cv:"))
    casas_f = cfg.get("fontes") or ([8] if faixa == "A" else FONTES_AMAIS[ap])
    pi = float(np.median([prev_fonte(ap, c) for c in casas_f]))
    regras = {r: [] for r in ("d_A", "d_L", "q")}
    aucs, oras, rs = [], [], []
    y = dias = t_all = None
    for s in sems:
        ms = por_sem[s]
        partes = [conjunto(ms, a) for a in alvos]
        yc = np.concatenate([p[0] for p in partes])
        if y is None:
            y, t_all = yc, [p[1] for p in partes]
            dias = np.concatenate([blocos_dia(p[1]) + 100000 * i for i, p in enumerate(partes)])
        assert len(yc) == len(y) and (yc == y).all()
        z_c = np.concatenate([p[3] for p in partes])
        aucs.append(roc_auc_score(y, z_c))
        oras.append(limiar_oraculo(z_c, y)[1])
        if faixa == "A":
            yv, _, pv, zv = conjunto(ms, "val")
            regras["d_A"].append(Decisor(sig(zv), yv).decidir(sig(z_c))["d"])
        cc = [conjunto(ms, k) for k in cvs]
        yo = np.concatenate([c[0] for c in cc])
        po = sig(np.concatenate([c[3] for c in cc]))
        regras["d_L"].append(Decisor(po, yo).decidir(sig(z_c))["d"])
        r = min(1.0, k_de_dobras([(c[2], c[0]) for c in cc]) * pi)
        rs.append(r)
        regras["q"].append(np.concatenate([decidir_quantil(p[2], r, semente=s) for p in partes]))
    prev = float(y.mean())
    art = ARTIGO[(ap, cel)]
    base = dict(modelo="CRAFT+E" if com_e else "CRAFT", aparelho=ap, celula=cel, faixa=faixa,
                sementes=len(sems), n=len(y), dias=len(np.unique(dias)),
                ativacoes=sum(ativacoes(p_y, p_t) for p_y, p_t in
                              zip(np.split(y, np.cumsum([len(t) for t in t_all])[:-1]), t_all)),
                prevalencia=round(prev, 4), trivial=round(2 * prev / (1 + prev), 4), artigo=art,
                auc=round(float(np.mean(aucs)), 3), oraculo=round(float(np.mean(oras)), 3))
    linhas = []
    for nome, bins in regras.items():
        if not bins:
            continue
        B = np.stack(bins)
        fa, dif = replicas(y, B, dias, n_boot)
        f1m = float(np.mean([f1(y, b) for b in B]))
        d = fa - art
        p = max(float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))), 0.5 / n_boot)
        lin = dict(**base, regra=nome, primaria=nome == cfg["primaria"][cel] and com_e,
                   f1=round(f1m, 3), ic95=tuple(float(v) for v in np.percentile(fa, [2.5, 97.5]).round(3)),
                   p_artigo=p, vs_trivial_ic=tuple(float(v) for v in np.percentile(dif, [2.5, 97.5]).round(3)),
                   supera_artigo=bool(np.percentile(fa, 2.5) > art),
                   perde_artigo=bool(np.percentile(fa, 97.5) < art),
                   vence_trivial=bool(np.percentile(dif, 2.5) > 0),
                   fracao_prevista=round(float(B.mean()), 4))
        linhas.append(lin)
        print(f"  {base['modelo']:7s} {faixa:2s} {ap:16s} {cel:5s} {nome:3s}"
              f"{'*' if lin['primaria'] else ' '} F1={f1m:.3f} IC{lin['ic95']} artigo={art:.2f} "
              f"trivial={base['trivial']:.3f} AUC={base['auc']:.3f} ora={base['oraculo']:.3f}",
              flush=True)
    return linhas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preregistro", required=True)
    p.add_argument("--n_boot", type=int, default=2000)
    args = p.parse_args()
    pre = json.loads(Path(args.preregistro).read_text(encoding="utf-8"))
    linhas = []
    for faixa, aps in pre["execucoes"].items():
        for ap, cfg in aps.items():
            cfg = dict(cfg, sementes=pre["sementes"], primaria=pre["regra_primaria"])
            for cel in cfg["celulas"]:
                for com_e in (True, False):
                    linhas += celula(ap, cel, faixa, cfg, com_e, args.n_boot)
    d = pd.DataFrame(linhas)
    chave = d.faixa + "|" + d.aparelho + "|" + d.celula
    fam = d.primaria & chave.isin(pre["familia_primaria"])
    h = holm(dict(zip(chave[fam], d.p_artigo[fam]))).set_index("comparacao").p_holm
    d["familia"] = np.where(fam, "primaria", "secundaria")
    d["p_holm"] = np.where(fam, chave.map(h), np.nan)
    nome = Path(args.preregistro).stem.replace("preregistro_", "final_")
    d.to_csv(RAIZ / "resultados" / "v3" / f"{nome}__celulas.csv", index=False)
    print("\nFamilia primaria (Holm):")
    print(d[d.familia == "primaria"][["faixa", "aparelho", "celula", "regra", "f1", "ic95",
                                      "artigo", "p_artigo", "p_holm", "supera_artigo",
                                      "vence_trivial"]].to_string(index=False))


if __name__ == "__main__":
    main()
