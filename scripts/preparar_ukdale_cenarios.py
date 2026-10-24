"""Data for the UK-DALE scenarios S4 and S5 of the paper (train on house 1; test on
house 1 in a later period, and on house 2).

Periods (fixed before any training on these scenarios):
  house 1, training   2013-03-01 .. 2014-03-01 (12 months, as the benchmark protocol)
  house 1, test (S4)  2014-03-01 .. 2014-04-01 (the following month)
  house 2, test (S5)  the whole recording (2013-02-17 .. 2013-10-10)
Appliances: fridge (house 1: fridge freezer, meter 12; house 2: fridge, meter 14),
dishwasher (meters 6 and 13) and washing machine (house 2: meter 12; house 1 has no
washing machine, only a washer dryer, meter 5, which is used as the washing machine
target in these scenarios - declared in the paper).

Outputs, in the formats of the rest of the pipeline:
  datasets/v3_fino/ukdale_s_casa_1.npz       30-s grid of house 1, TRAINING PERIOD ONLY
                                             (src/reco.py: house id 301)
  datasets/protocolo_artigo/ukdale_casa_1__<ap>__5min__treino.npz
  datasets/protocolo_artigo/ukdale_casa_1__<ap>__5min__teste.npz
  datasets/protocolo_artigo/ukdale_casa_2__<ap>__5min__cross.npz
  datasets/protocolo_artigo/refit_casa_{3,7,20}__{fridge,dishwasher}__5min__treino.npz
                                             (held-out validation / calibration houses,
                                             from the REFIT 30-s grids)
5-min means use the sample counts, with the validity rule of scripts/preparar_artigo.py
(>= 50% of the typical number of samples in the interval).

House 5 (second cross-house test, added 2026-10-02): the whole recording, with its fridge
freezer (meter 19), dishwasher (meter 22) and washer dryer (meter 24, the same kind of
appliance as the house 1 training target):
  datasets/protocolo_artigo/ukdale_casa_5__<ap>__5min__cross.npz

Usage:
    python scripts/preparar_ukdale_cenarios.py            # houses 1, 2 and REFIT validation
    python scripts/preparar_ukdale_cenarios.py --casa5    # house 5 only
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
from preparar_ukdale_v3 import H5, PASSO, medidores, serie  # noqa: E402

FINO = RAIZ / "datasets" / "v3_fino"
PROTO = RAIZ / "datasets" / "protocolo_artigo"
TREINO = ("2013-03-01", "2014-03-01")
TESTE1 = ("2014-03-01", "2014-04-01")
LIMIAR = {"fridge": 50.0, "dishwasher": 10.0, "washing machine": 20.0}
# channel names written by preparar_ukdale_v3.medidores / present in the REFIT grids
CANAL = {
    ("ukdale", 1): {"fridge": "Fridge-Freezer", "dishwasher": "Dishwasher",
                    "washing machine": "Washer Dryer"},
    ("ukdale", 2): {"fridge": "Fridge", "dishwasher": "Dishwasher",
                    "washing machine": "Washing Machine"},
    ("ukdale", 5): {"fridge": "Fridge-Freezer", "dishwasher": "Dishwasher",
                    "washing machine": "Washer Dryer"},
    ("refit", 3): {"fridge": "Fridge-Freezer", "dishwasher": "Dishwasher"},
    ("refit", 7): {"fridge": "Fridge", "dishwasher": "Dishwasher"},
    ("refit", 20): {"fridge": "Fridge", "dishwasher": "Dishwasher"},
}


def grade(b: int, t0: str | None, t1: str | None) -> dict:
    s = pd.HDFStore(H5, "r")
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
    return dict(t0=ini, x=np.stack(canais).astype(np.float32), cont=cont.astype(np.uint16),
                nomes=np.array(nomes))


def recorte(g: dict, t0: str, t1: str) -> dict:
    a = int(pd.Timestamp(t0, tz="UTC").timestamp())
    b = int(pd.Timestamp(t1, tz="UTC").timestamp())
    i0 = max(0, (a - g["t0"]) // PASSO)
    i1 = min(g["x"].shape[1], (b - g["t0"]) // PASSO)
    i0 += (-i0) % 10                       # keep 5-min alignment
    i1 -= (i1 - i0) % 10
    return dict(t0=g["t0"] + i0 * PASSO, x=g["x"][:, i0:i1], cont=g["cont"][i0:i1],
                nomes=g["nomes"])


def protocolo(g: dict, canal: str, aparelho: str, arq: Path) -> None:
    nomes = [str(n) for n in g["nomes"]]
    k = nomes.index(canal)
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
    np.savez_compressed(arq, t=t[ok].astype(np.int64), main=main[ok].astype(np.float32),
                        **{slug: ap[ok].astype(np.float32),
                           f"status_{slug}": (ap[ok] >= LIMIAR[aparelho]).astype(np.int8)})
    y = ap[ok] >= LIMIAR[aparelho]
    print(f"   {arq.name}: {ok.sum() * 300 / 86400:.1f} days"
          + ("" if SEM_ROTULO else f", prevalence {y.mean():.3f}"), flush=True)


SEM_ROTULO = "--casa5" in sys.argv          # test house of a pending pre-registration


def main() -> None:
    t = time.time()
    if SEM_ROTULO:
        g5 = grade(5, None, None)
        print(f"ukdale 5: channels {list(g5['nomes'][1:])}", flush=True)
        for ap, canal in CANAL[("ukdale", 5)].items():
            protocolo(g5, canal, ap,
                      PROTO / f"ukdale_casa_5__{ap.replace(' ', '_')}__5min__cross.npz")
        return
    # house 1: training + test month in one read, then split
    g1 = grade(1, TREINO[0], TESTE1[1])
    tr1, te1 = recorte(g1, *TREINO), recorte(g1, *TESTE1)
    fino = dict(tr1)
    fino["nomes"] = np.array(["Washing Machine" if n == "Washer Dryer" else str(n)
                              for n in tr1["nomes"]])
    np.savez_compressed(FINO / "ukdale_s_casa_1.npz", **fino)
    print(f"ukdale 1: channels {list(g1['nomes'][1:])} ({time.time() - t:.0f}s)", flush=True)
    for ap, canal in CANAL[("ukdale", 1)].items():
        slug = ap.replace(" ", "_")
        protocolo(tr1, canal, ap, PROTO / f"ukdale_casa_1__{slug}__5min__treino.npz")
        protocolo(te1, canal, ap, PROTO / f"ukdale_casa_1__{slug}__5min__teste.npz")
    # house 2: whole recording as the cross-house test
    g2 = grade(2, None, None)
    print(f"ukdale 2: channels {list(g2['nomes'][1:])}", flush=True)
    for ap, canal in CANAL[("ukdale", 2)].items():
        protocolo(g2, canal, ap, PROTO / f"ukdale_casa_2__{ap.replace(' ', '_')}__5min__cross.npz")
    # REFIT validation houses for the appliances that had no protocol file
    for c in (3, 7, 20):
        d = np.load(FINO / f"refit_casa_{c}.npz")
        g = dict(t0=int(d["t0"]), x=d["x"], cont=d["cont"], nomes=d["nomes"])
        for ap, canal in CANAL[("refit", c)].items():
            protocolo(g, canal, ap, PROTO / f"refit_casa_{c}__{ap}__5min__treino.npz")


if __name__ == "__main__":
    main()
