"""E2.0 / teste 7 do v3 (docs/v3.md §3.2, §8): discriminador real x recomposto.

Para cada operacao (R1, R2, R3, R4 e o fundo sozinho Rf), uma ResNet 1D (k = 9,
as mesmas 4 entradas do ReCo) e treinada para separar janelas recompostas de
janelas reais R0 COM O MESMO ROTULO do alvo (R1 contra R0 ligadas; R2, R4, Rf
contra R0 desligadas; R3 contra os seus parceiros R0). Treino no periodo de
treino, AUC no holdout temporal (bancos reconstruidos so com o holdout).

`--controle` (declarado, desenhado depois da primeira rodada): escala s = 1 e
negativos reais so de janelas limpas, como os fundos. Separa o artefato de
subtracao/reamostragem das diferencas intencionais (escala, distratores raros,
co-ocorrencia quebrada), que qualquer discriminador tambem detecta.

Criterio do v3: AUC < 0,6 para seguir. Acima disso, o gerador deixa uma marca
que o modelo pode aprender no lugar do aparelho.

Uso:
    python src/discriminador_v3.py --fonte A
    python src/discriminador_v3.py --fonte A+
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decisao import auc                                              # noqa: E402
from reco import (ANTES, OPS, Gerador, ResNet1D, carregar_casa,       # noqa: E402
                  definir_periodo, entradas, stats_casa)
from treino import DISPOSITIVO                                       # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "resultados" / "v3"
POOL = {"television": [1, 2, 5, 13, 15, 16, 19, 21],
        "washing machine": [1, 2, 4, 5, 6, 9, 10, 11, 13, 15, 16, 17, 18, 19, 21]}


def coletar(g: Gerador, n_por_op: int, semente: int, casar_limpo: bool = False) -> dict:
    """{op: (x recomposto, casa, x real pareado por rotulo, casa real)}"""
    ger = torch.Generator(device=DISPOSITIVO).manual_seed(semente)
    acc = {op: [] for op in ("R1", "R2", "R3", "R4", "Rf")}
    reais = {op: [] for op in acc}
    while min(sum(len(a[0]) for a in acc[op]) for op in acc) < n_por_op:
        base = g.centros[torch.randint(0, len(g.centros), (102,), device=DISPOSITIVO,
                                       generator=ger)]
        L = g.lote(base, ger)
        for op in acc:
            m = L["op"] == OPS.index(op)
            if op == "R3":
                i, j = L["pares_inv"].T
                acc[op].append((L["x"][j], L["casa"][j]))
                reais[op].append((L["x"][i], L["casa"][i]))
                continue
            if m.any():
                acc[op].append((L["x"][m], L["casa"][m]))
    out = {}
    for op in acc:
        x = torch.cat([a[0] for a in acc[op]])[:n_por_op]
        c = torch.cat([a[1] for a in acc[op]])[:n_por_op]
        if op == "R3":
            xr = torch.cat([a[0] for a in reais[op]])[:n_por_op]
            cr = torch.cat([a[1] for a in reais[op]])[:n_por_op]
        else:
            lig = op == "R1"
            pool = g.idx_on if lig else (g.idx_limpo if casar_limpo
                                         else torch.nonzero(~g.on).squeeze(1))
            cc = g.centros[pool[torch.randint(0, len(pool), (n_por_op,), device=DISPOSITIVO,
                                              generator=ger)]]
            L = g.lote_real(cc)
            xr, cr = L["x"], L["casa"]
        out[op] = (x, c, xr, cr)
    return out


def rodar(aparelho: str, fonte: str, n: int, epocas: int, semente: int,
          controle: bool = False) -> list[dict]:
    casas_ids = [8] + (POOL[aparelho] if fonte == "A+" else [])
    casas = []
    for c in casas_ids:
        cf = carregar_casa(c, aparelho)
        if cf is not None:
            cf.periodo = definir_periodo(cf, "cross")
            casas.append(cf)
    esc = (1.0, 1.0) if controle else None
    g_tr = Gerador(casas, aparelho, "treino", DISPOSITIVO, escala=esc)
    g_te = Gerador(casas, aparelho, "val", DISPOSITIVO, escala=esc)
    st = []
    for c in casas:
        n5 = len(c.A) // 10
        w = c.W[:n5 * 10].reshape(n5, 10)
        x5 = (c.A[:n5 * 10].reshape(n5, 10) * w).sum(1) / np.maximum(w.sum(1), 1e-6)
        st.append(stats_casa(x5[c.ok[:n5 * 10].reshape(n5, 10).all(1)]))
    stt = {k: torch.tensor([s[k] for s in st], device=DISPOSITIVO) for k in st[0]}
    p5 = g_tr.res(g_tr.P, g_tr.centros[::7])[:, ANTES]
    pref = float(p5[p5 >= g_tr.lim].median())

    def ent(x, casa):
        return entradas(x, stt["s"][casa], stt["m"][casa], stt["d"][casa], pref)

    tr = coletar(g_tr, n, semente, controle)
    te = coletar(g_te, max(n // 4, 500), semente + 1, controle)
    linhas = []
    for op in tr:
        torch.manual_seed(semente)
        x, c, xr, cr = tr[op]
        X = torch.cat([ent(x, c), ent(xr, cr)])
        Y = torch.cat([torch.ones(len(x)), torch.zeros(len(xr))]).to(DISPOSITIVO)
        rede = ResNet1D(4, 9).to(DISPOSITIVO)
        opt = torch.optim.AdamW(rede.parameters(), lr=2e-3, weight_decay=1e-4)
        for _ in range(epocas):
            rede.train()
            ordem = torch.randperm(len(X), device=DISPOSITIVO)
            for i in range(0, len(X), 256):
                b = ordem[i:i + 256]
                perda = F.binary_cross_entropy_with_logits(rede(X[b])[:, 0], Y[b])
                opt.zero_grad()
                perda.backward()
                opt.step()
        rede.eval()
        x, c, xr, cr = te[op]
        with torch.no_grad():
            s = torch.cat([rede(ent(x, c))[:, 0], rede(ent(xr, cr))[:, 0]]).cpu().numpy()
        y = np.r_[np.ones(len(x)), np.zeros(len(xr))]
        a = auc(y, s)
        linhas.append(dict(aparelho=aparelho, fonte=fonte, controle=controle, op=op,
                           n_treino=len(X),
                           n_teste=len(y), auc=round(a, 3), passa=a < 0.6))
        print(f"  {aparelho:16s} {fonte:2s} {op}: AUC real x recomposto = {a:.3f} "
              f"{'ok' if a < 0.6 else 'ACIMA DE 0,6'}", flush=True)
    return linhas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aparelhos", default="television,washing machine")
    p.add_argument("--fonte", default="A")
    p.add_argument("--n", type=int, default=8000)
    p.add_argument("--epocas", type=int, default=6)
    p.add_argument("--semente", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--controle", action="store_true",
                   help="escala s = 1 e R0 negativos so de janelas limpas: isola o "
                        "artefato de subtracao/reamostragem das mudancas intencionais")
    args = p.parse_args()
    SAIDA.mkdir(parents=True, exist_ok=True)
    linhas = []
    for ap in args.aparelhos.split(","):
        linhas += rodar(ap.strip(), args.fonte, args.n, args.epocas, args.semente,
                        args.controle)
    nome = f"e20_discriminador_{args.fonte.replace('+', 'mais')}{args.tag}.csv"
    pd.DataFrame(linhas).to_csv(SAIDA / nome, index=False)


if __name__ == "__main__":
    main()
