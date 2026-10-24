"""F4 do v3 (docs/v3.md §6 F4, §7): relatorio por celula de teste, a partir de um
pre-registro.

O pre-registro (json) diz, por aparelho e celula, de que rodada vem cada modelo e
quais membros formam o conjunto:

  {"faixa": "A+", "celulas": {"television": {"cross": {"tag": "final_Amais_tv",
   "membros": ["R5", "R7", "R9", "R15", "R25"], "primaria": "d_L"}, ...}}}

Para cada celula, com todas as sementes (o conjunto da semente s = membros da
semente s):

  d_A   indutiva, media dos logits, Platt + limiar no holdout temporal das fontes
        (na variante intra, o holdout dos 15% finais da casa 8)
  d_L   indutiva, calibrada nas casas REFIT 3/7/20 (usa rotulos de casas extras)
  q     transdutiva (A-t/A+-t): media de postos + quantil; k das casas 3/7/20
        (ou do holdout, no intra); pi = mediana das prevalencias das fontes

F1 [IC95%] por bootstrap de dias (meio-dia) x sementes; "supera" = limite
inferior acima do artigo; "vence o trivial" = IC da diferenca exclui zero.

Uso:
    python src/final_v3.py --preregistro resultados/v3/preregistro_Amais.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decisao import Decisor, f1, limiar_oraculo, logit           # noqa: E402
from estatistica import ativacoes, blocos_dia, bootstrap_f1      # noqa: E402
from experimento_quantil import FONTES_AMAIS, prev_fonte         # noqa: E402
from final_v2 import ARTIGO as _ARTIGO                            # noqa: E402

ARTIGO = dict(_ARTIGO)
ARTIGO.update({("television", "novo"): float("nan"), ("washing machine", "novo"): float("nan")})
from quantil import decidir_quantil, fracao_oraculo, k_de_dobras, posto_percentual  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
V3 = RAIZ / "resultados" / "v3"


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def ler_todas(tag: str, aparelho: str, modelo: str) -> dict[int, dict]:
    out = {}
    for arq in sorted((V3 / "pred" / tag).glob(f"{aparelho.replace(' ', '_')}__{modelo}__s*.npz")):
        s = int(arq.stem.rsplit("__s", 1)[1])
        d = np.load(arq)
        r = {}
        for k in d.files:
            alvo, campo = k.rsplit("|", 1)
            r.setdefault(alvo, {})[campo] = d[k]
        for a in r:
            r[a].setdefault("z", logit(r[a]["p"]) if "p" in r[a] else None)
        out[s] = r
    return out


def conjunto(ms: list[dict], alvo: str):
    y, t = ms[0][alvo]["y"], ms[0][alvo]["t"]
    posto = np.mean([posto_percentual(m[alvo]["z"]) for m in ms], 0)
    z = np.mean([m[alvo]["z"] for m in ms], 0)
    return y, t, posto, z


def celula(aparelho: str, cel: str, cfg: dict, faixa: str, n_boot: int) -> list[dict]:
    membros = {m: ler_todas(cfg["tag"], aparelho, m) for m in cfg["membros"]}
    sems = sorted(set.intersection(*[set(v) for v in membros.values()]))
    por_sem = {s: [membros[m][s] for m in cfg["membros"]] for s in sems}
    ex = por_sem[sems[0]][0]
    alvos = [k for k in ex if k.startswith(cel)]
    cvs = [k for k in ex if k.startswith("cv:")]
    casas_f = cfg.get("fontes") or ([8] if faixa == "A" else FONTES_AMAIS[aparelho])
    pi = float(np.median([prev_fonte(aparelho, c) for c in casas_f if c < 200]))   # UK-DALE: sem arquivo do protocolo
    y = np.concatenate([ex[a]["y"] for a in alvos])
    dias = np.concatenate([blocos_dia(ex[a]["t"]) + 100000 * i for i, a in enumerate(alvos)])
    prev = float(y.mean())
    regras = {"d_A": [], "d_L": [], "q": []}
    aucs, aps, oras, rs = [], [], [], []
    for s in sems:
        ms = por_sem[s]
        partes = [conjunto(ms, a) for a in alvos]
        posto_c = np.concatenate([p[2] for p in partes])
        z_c = np.concatenate([p[3] for p in partes])
        aucs.append(roc_auc_score(y, z_c) if 0 < y.sum() < len(y) else np.nan)
        aps.append(average_precision_score(y, z_c))
        oras.append(limiar_oraculo(z_c, y)[1])
        yv, _, pv, zv = conjunto(ms, "val")
        regras["d_A"].append(Decisor(sig(zv), yv).decidir(sig(z_c))["d"])
        if cvs:
            cc = [conjunto(ms, k) for k in cvs]
            yo = np.concatenate([c[0] for c in cc])
            po = sig(np.concatenate([c[3] for c in cc]))
            regras["d_L"].append(Decisor(po, yo).decidir(sig(z_c))["d"])
            k = k_de_dobras([(c[2], c[0]) for c in cc])
        else:
            k = fracao_oraculo(pv, yv) / max(yv.mean(), 1e-9)
        # quantil por casa-alvo (os postos sao da casa), com o mesmo r
        r = min(1.0, k * pi)
        rs.append(r)
        regras["q"].append(np.concatenate([decidir_quantil(p[2], r, semente=s) for p in partes]))
    base = dict(aparelho=aparelho, celula=cel, faixa=faixa, tag=cfg["tag"],
                membros="+".join(cfg["membros"]), sementes=len(sems), n=len(y),
                dias=len(np.unique(dias)),
                ativacoes=sum(ativacoes(ex[a]["y"], ex[a]["t"]) for a in alvos),
                prevalencia=round(prev, 4), trivial=round(2 * prev / (1 + prev), 4),
                artigo=ARTIGO[(aparelho, cel)], auc=round(float(np.nanmean(aucs)), 3),
                ap=round(float(np.mean(aps)), 3), oraculo=round(float(np.mean(oras)), 3),
                r_quantil=round(float(np.mean(rs)), 4))
    linhas = []
    for nome, bins in regras.items():
        if not bins:
            continue
        B = np.stack(bins)
        bt = bootstrap_f1(y, B, dias, n=n_boot, pred_b=np.ones(len(y)))
        f1m = float(np.mean([f1(y, b) for b in B]))
        lin = dict(**base, regra=nome, primaria=nome == cfg["primaria"], f1=round(f1m, 3),
                   f1_dp=round(float(np.std([f1(y, b) for b in B])), 3), ic95=bt["f1_ic"],
                   vs_trivial_ic=bt["dif_ic"],
                   supera_artigo=bool(bt["f1_ic"][0] > ARTIGO[(aparelho, cel)]),
                   perde_artigo=bool(bt["f1_ic"][1] < ARTIGO[(aparelho, cel)]),
                   vence_trivial=bool(bt["dif_ic"][0] > 0),
                   fracao_prevista=round(float(B.mean()), 4))
        linhas.append(lin)
        print(f"  {aparelho:16s} {cel:5s} {nome:3s}{'*' if lin['primaria'] else ' '} "
              f"F1={f1m:.3f} IC{bt['f1_ic']} artigo={ARTIGO[(aparelho, cel)]:.2f} "
              f"trivial={base['trivial']:.3f} AUC={base['auc']:.3f} ora={base['oraculo']:.3f}",
              flush=True)
    return linhas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preregistro", required=True)
    p.add_argument("--n_boot", type=int, default=2000)
    args = p.parse_args()
    pre = json.loads(Path(args.preregistro).read_text())
    linhas = []
    for ap, cels in pre["celulas"].items():
        for cel, cfg in cels.items():
            f = pre.get("fontes", {}).get(ap)
            cfg = dict(cfg, fontes=f if isinstance(f, list) else None)
            linhas += celula(ap, cel, cfg, cfg.get("faixa", pre["faixa"]), args.n_boot)
    nome = Path(args.preregistro).stem.replace("preregistro_", "final_")
    pd.DataFrame(linhas).to_csv(V3 / f"{nome}__celulas.csv", index=False)


if __name__ == "__main__":
    main()
