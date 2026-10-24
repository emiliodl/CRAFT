"""UK-DALE na grade de 30 s do v3 (docs/v3.md D5; melhorias pos-v3: mais fontes e casas-alvo novas).

Mesmo formato de scripts/preparar_v3.py (x = [agregado, plugues...], cont, nomes),
em datasets/v3_fino/ukdale_casa_<b>.npz, e arquivos de 5 min no formato do
protocolo (t, main, <aparelho>, status_<aparelho>) em datasets/protocolo_artigo/
para as casas-alvo, com a mesma regra de validade de scripts/preparar_artigo.py
(>= 50% das amostras tipicas no intervalo).

Grandezas (declarado, §3.9): o agregado do UK-DALE (meter1) e potencia APARENTE;
os plugues sao potencia ativa. No REFIT o agregado vem de pinça de corrente.
Nao ha conversao: a diferenca fica no dominio, como no ECO.

Papeis (fixados antes de qualquer treino com o UK-DALE):
  casa 1  alvo novo da TV (jan-mar 2015)
  casa 2  alvo novo da maquina de lavar (periodo inteiro)
  casa 4  fonte da TV (canal TV + set-top box; excluida no D5: 100% ligada);
          alvo novo do freezer (pos-v3, fixado antes de treinar o CRAFT do freezer)
  casa 5  fonte da TV
Washer dryer (casas 1 e 5) NAO e maquina de lavar no protocolo: vira categoria
"dryer" (distrator/auxiliar).

Uso:
    python scripts/preparar_ukdale_v3.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
H5 = RAIZ / "datasets" / "ukdale.h5"
FINO = RAIZ / "datasets" / "v3_fino"
PROTO = RAIZ / "datasets" / "protocolo_artigo"
PASSO = 30
sys.path.insert(0, str(RAIZ / "src"))

# tipo do metadado -> nome do canal (casado pelas regex de src/reco.py)
NOMES = {"television": "Television", "washing machine": "Washing Machine",
         "washer dryer": "Washer Dryer", "dish washer": "Dishwasher", "kettle": "Kettle",
         "microwave": "Microwave", "computer": "Computer", "desktop computer": "Computer",
         "laptop computer": "Computer", "audio system": "Hi-Fi", "audio amplifier": "Hi-Fi",
         "games console": "Games Console", "freezer": "Freezer",
         "fridge freezer": "Fridge-Freezer", "fridge": "Fridge"}
CASAS = {1: ("2015-01-01", "2015-04-01"), 2: (None, None), 4: (None, None), 5: (None, None)}
ALVOS = {1: ["television"], 2: ["washing machine"], 4: ["freezer"]}
LIMIAR = {"television": 10.0, "washing machine": 20.0, "freezer": 10.0}


def medidores(s: pd.HDFStore, b: int) -> list[tuple[str, int]]:
    md = s.root[f"building{b}"]._v_attrs.metadata
    out = []
    for a in md["appliances"]:
        if a["type"] in NOMES and len(a["meters"]) == 1 and a["meters"][0] > 1:
            # medidor compartilhado por varios aparelhos: fica o nome do de interesse
            out.append((NOMES[a["type"]], a["meters"][0]))
    visto, uniq = set(), []
    for n, m in out:
        if m not in visto:
            visto.add(m)
            uniq.append((n, m))
    return uniq


def serie(s, b, m, t0, t1):
    k = f"/building{b}/elec/meter{m}"
    d = s.select(k) if t0 is None else s.select(k, where=f"index >= '{t0}' & index < '{t1}'")
    col = [c for c in d.columns if c[0] == "power"]
    col = next((c for c in col if c[1] == "active"), col[0])
    return d.index.asi8 // 10**9, d[col].to_numpy(np.float64)


def grade(b: int) -> dict:
    s = pd.HDFStore(H5, "r")
    t0, t1 = CASAS[b]
    ta, xa = serie(s, b, 1, t0, t1)
    ini = int(ta.min() // 300 * 300)
    n = int((ta.max() - ini) // PASSO + 1)
    n -= n % 10

    def binar(t, x):
        i = ((t - ini) // PASSO).astype(np.int64)
        ok = (i >= 0) & (i < n) & np.isfinite(x)
        c = np.bincount(i[ok], minlength=n)[:n]
        sm = np.bincount(i[ok], weights=x[ok], minlength=n)[:n]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c > 0, sm / np.maximum(c, 1), np.nan), c

    agg, cont = binar(ta, xa)
    canais, nomes = [agg], ["Aggregate"]
    for nome, m in medidores(s, b):
        t, x = serie(s, b, m, t0, t1)
        v, _ = binar(t, x)
        canais.append(v)
        nomes.append(nome)
    s.close()
    x = np.stack(canais).astype(np.float32)
    return dict(t0=ini, x=x, cont=cont.astype(np.uint16), nomes=np.array(nomes))


def protocolo(b: int, g: dict, aparelho: str) -> None:
    """5 min no formato do protocolo, para a casa-alvo."""
    nomes = [str(n) for n in g["nomes"]]
    k = nomes.index(NOMES[aparelho])
    n5 = g["x"].shape[1] // 10
    w = g["cont"][:n5 * 10].reshape(n5, 10).astype(np.float64)
    a = g["x"][0][:n5 * 10].reshape(n5, 10)
    p = g["x"][k][:n5 * 10].reshape(n5, 10)
    okf = np.isfinite(a) & np.isfinite(p)
    ww = np.where(okf, w, 0)
    tip = np.median(w.sum(1)[w.sum(1) > 0])
    main = (np.nan_to_num(a) * ww).sum(1) / np.maximum(ww.sum(1), 1e-9)
    ap = (np.nan_to_num(p) * ww).sum(1) / np.maximum(ww.sum(1), 1e-9)
    ok = ww.sum(1) >= 0.5 * tip
    t = g["t0"] + np.arange(n5) * 300
    slug = aparelho.replace(" ", "_")
    arq = PROTO / f"ukdale_casa_{b}__{slug}__5min__alvo.npz"
    np.savez_compressed(arq, t=t[ok].astype(np.int64), main=main[ok].astype(np.float32),
                        **{slug: ap[ok].astype(np.float32),
                           f"status_{slug}": (ap[ok] >= LIMIAR[aparelho]).astype(np.int8)})
    y = ap[ok] >= LIMIAR[aparelho]
    print(f"   {arq.name}: {ok.sum() * 300 / 86400:.1f} dias, prevalencia {y.mean():.3f}")


def main() -> None:
    FINO.mkdir(parents=True, exist_ok=True)
    for b in CASAS:
        t = time.time()
        g = grade(b)
        np.savez_compressed(FINO / f"ukdale_casa_{b}.npz", **g)
        ok = np.isfinite(g["x"][0])
        print(f"ukdale {b}: {g['x'].shape[1] * 30 / 86400:.0f} dias, {ok.mean() * 100:.1f}% com "
              f"agregado, canais {list(g['nomes'][1:])} ({time.time() - t:.0f}s)", flush=True)
        for ap in ALVOS.get(b, []):
            protocolo(b, g, ap)


if __name__ == "__main__":
    main()
