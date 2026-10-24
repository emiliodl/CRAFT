"""F0 do v3 (docs/v3.md §6): diagnosticos sem treino D1 e D2.

D1  confundidor da maquina de lavar. Em cada INICIO DE SESSAO do plugue (grade
    de 30 s; plugue desligado nos 10 min anteriores), na casa 8 e nas casas
    fonte/validacao com maquina de lavar:
      dR  = resíduo (agregado - plugue) medio nos 20 min seguintes menos o dos
            10 min anteriores
      dQ  = o mesmo para cada um dos OUTROS plugues da casa
    Reporta dR medio, dQ medio por plugue, a fracao de dR explicada pela soma
    dos plugues, a correlacao entre dR e dQ sobre as bordas, e dR por hora do
    dia (00-08 contra o resto). Tarifa: o metadado do REFIT nao traz tarifa;
    fica registrado.
D2  teto do rotulo: fracao dos intervalos POSITIVOS de 5 min em que a potencia
    media do plugue fica abaixo de 50 W e de 100 W, por aparelho e por casa
    (anos de treino do protocolo; nunca o mes de teste).

Uso:
    python src/diagnostico_v3.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dados_v2 import LIMIAR_ON, arquivo                       # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
FINO = RAIZ / "datasets" / "v3_fino"
PROTO = RAIZ / "datasets" / "protocolo_artigo"
SAIDA = RAIZ / "resultados" / "v3"
PADRAO_WM = re.compile(r"^washing machine", re.I)


def carregar_fino(casa: int) -> dict:
    d = np.load(FINO / f"refit_casa_{casa}.npz")
    return dict(t0=int(d["t0"]), x=d["x"], cont=d["cont"], nomes=[str(n) for n in d["nomes"]])


def inicios_sessao(p: np.ndarray, limiar: float, antes: int) -> np.ndarray:
    on = np.nan_to_num(p, nan=0.0) >= limiar
    b = np.flatnonzero(~on[:-1] & on[1:]) + 1
    b = b[(b >= antes) & (b < len(p) - 60)]
    soma = np.convolve(on.astype(int), np.ones(antes, int), "full")[:len(on)]
    return b[soma[b - 1] == 0]


def d1(casas: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    linhas, horas = [], []
    antes, depois = 20, 40                         # 10 min antes, 20 min depois (30 s)
    for c in casas:
        f = carregar_fino(c)
        k_wm = next((i for i, n in enumerate(f["nomes"]) if PADRAO_WM.match(n)), None)
        if k_wm is None:
            continue
        x = f["x"]
        A, P = x[0], x[k_wm]
        R = A - P
        bordas = inicios_sessao(P, LIMIAR_ON["washing machine"], antes)
        idx_a = bordas[:, None] + np.arange(-antes, 0)[None, :]
        idx_d = bordas[:, None] + np.arange(0, depois)[None, :]

        def delta(s):
            return np.nanmean(s[idx_d], 1) - np.nanmean(s[idx_a], 1)

        ok = np.isfinite(delta(A))
        dR, dP = delta(R)[ok], delta(P)[ok]
        hora = (((f["t0"] + bordas[ok] * 30) % 86400) // 3600).astype(int)
        base = dict(casa=c, bordas=int(ok.sum()), dP=round(float(np.mean(dP)), 1),
                    dR=round(float(np.mean(dR)), 1),
                    dR_madrugada=round(float(np.mean(dR[hora < 8])), 1) if (hora < 8).any()
                    else np.nan,
                    dR_dia=round(float(np.mean(dR[hora >= 8])), 1) if (hora >= 8).any()
                    else np.nan,
                    frac_madrugada=round(float((hora < 8).mean()), 3))
        soma_q = np.zeros(ok.sum())
        for k in range(1, 10):
            if k == k_wm:
                continue
            dq = np.nan_to_num(delta(x[k])[ok])
            soma_q += dq
            r = float(np.corrcoef(dR, dq)[0, 1]) if dq.std() > 0 and dR.std() > 0 else 0.0
            linhas.append(dict(**base, plugue=f["nomes"][k], dQ=round(float(dq.mean()), 1),
                               corr_dR_dQ=round(r, 3),
                               frac_dR=round(float(dq.mean() / dR.mean()), 3)
                               if dR.mean() != 0 else np.nan))
        linhas.append(dict(**base, plugue="SOMA DOS PLUGUES", dQ=round(float(soma_q.mean()), 1),
                           corr_dR_dQ=round(float(np.corrcoef(dR, soma_q)[0, 1]), 3),
                           frac_dR=round(float(soma_q.mean() / dR.mean()), 3)))
        for h in range(24):
            m = hora == h
            if m.any():
                horas.append(dict(casa=c, hora=h, bordas=int(m.sum()),
                                  dR=round(float(dR[m].mean()), 1)))
    return pd.DataFrame(linhas), pd.DataFrame(horas)


def d2() -> pd.DataFrame:
    linhas = []
    for arq in sorted(PROTO.glob("refit_casa_*__5min__treino.npz")):
        m = re.match(r"refit_casa_(\d+)__(.+)__5min__treino", arq.stem)
        casa, ap = int(m.group(1)), m.group(2).replace("_", " ")
        if ap not in ("television", "washing machine", "freezer"):
            continue
        d = np.load(arq)
        a = d[ap.replace(" ", "_")]
        y = d[f"status_{ap.replace(' ', '_')}"] > 0
        if not y.any():
            continue
        linhas.append(dict(aparelho=ap, casa=casa, positivos=int(y.sum()),
                           prev=round(float(y.mean()), 4),
                           abaixo_50W=round(float((a[y] < 50).mean()), 3),
                           abaixo_100W=round(float((a[y] < 100).mean()), 3),
                           mediana_W=round(float(np.median(a[y])), 1)))
    return pd.DataFrame(linhas)


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    casas = [8, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20, 21]
    t1, t2 = d1(casas)
    t1.to_csv(SAIDA / "d1_confundidor_mlavar.csv", index=False)
    t2.to_csv(SAIDA / "d1_residuo_por_hora.csv", index=False)
    top = t1[t1.plugue != "SOMA DOS PLUGUES"].sort_values("frac_dR", ascending=False)
    print("D1 - por casa (soma dos plugues):")
    print(t1[t1.plugue == "SOMA DOS PLUGUES"].drop(columns=["plugue"]).to_string(index=False))
    print("\nD1 - plugues que mais explicam dR (frac_dR > 0,05):")
    print(top[top.frac_dR > 0.05][["casa", "plugue", "dR", "dQ", "frac_dR",
                                    "corr_dR_dQ"]].to_string(index=False))
    t3 = d2()
    t3.to_csv(SAIDA / "d2_teto_rotulo.csv", index=False)
    print("\nD2 - teto do rotulo:")
    print(t3.to_string(index=False))
    print(t3.groupby("aparelho")[["abaixo_50W", "abaixo_100W"]].median())


if __name__ == "__main__":
    main()
