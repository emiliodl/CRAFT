"""Scenario S6 of the paper: train on UK-DALE house 1, test on one REDD house.

REDD (US, 2011) has no stand-alone washing machine: every house has a "washer dryer",
sometimes on two meters (the two legs of a 240-V circuit), which are summed. It is used
as the "washing machine" target, as the washer dryer of UK-DALE house 1 is in training.
Fridge and dishwasher come from their own meters. The aggregate is the sum of the two
mains legs (meters 1 and 2).

House choice, fixed before any label is computed: among the houses that have the three
appliances (fridge, dishwasher, washer dryer), the one with the most 5-min intervals in
which the aggregate and the three plugs all have data. Labels are not used.

    python scripts/preparar_redd_cenario.py --inventario    # coverage only, no labels
    python scripts/preparar_redd_cenario.py --casa N        # protocol files for house N

Outputs (format of scripts/preparar_artigo.py, label = 5-min mean >= threshold):
    datasets/protocolo_artigo/redd_casa_<N>__<ap>__5min__cross.npz
Timestamps are local wall time (US/Eastern), like the REFIT and ECO protocol files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
H5 = RAIZ / "datasets" / "redd.h5"
PROTO = RAIZ / "datasets" / "protocolo_artigo"
LIMIAR = {"fridge": 50.0, "dishwasher": 10.0, "washing machine": 20.0}
TIPO = {"fridge": "fridge", "dishwasher": "dish washer", "washing machine": "washer dryer"}
JANELA = "5min"


def medidores(s: pd.HDFStore, b: int) -> dict[str, list[int]]:
    md = s.root[f"building{b}"]._v_attrs.metadata
    out = {}
    for ap, tipo in TIPO.items():
        for a in md["appliances"]:
            if a["type"] == tipo and ap not in out:
                out[ap] = list(a["meters"])
    return out


def potencia(s: pd.HDFStore, b: int, m: int) -> pd.Series:
    d = s.select(f"/building{b}/elec/meter{m}")
    col = [c for c in d.columns if c[0] == "power"]
    col = next((c for c in col if c[1] == "active"), col[0])
    x = d[col]
    x.index = x.index.tz_convert("US/Eastern").tz_localize(None)
    return x.astype(np.float64)


def cinco_min(xs: list[pd.Series]) -> tuple[pd.Series, pd.Series]:
    """Sum of the 5-min means of each meter, and validity (every meter with >= 50% of its
    typical sample count in the interval)."""
    medias, oks = [], []
    for x in xs:
        g = x.resample(JANELA)
        n = g.count()
        tip = float(n[n > 0].median())
        medias.append(g.mean())
        oks.append(n >= 0.5 * tip)
    idx = medias[0].index
    for m in medias[1:]:
        idx = idx.union(m.index)
    soma = sum(m.reindex(idx) for m in medias)
    ok = np.logical_and.reduce([o.reindex(idx, fill_value=False) for o in oks])
    return soma, pd.Series(ok, index=idx) & soma.notna()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inventario", action="store_true")
    p.add_argument("--casa", type=int, default=0)
    args = p.parse_args()
    s = pd.HDFStore(H5, "r")
    casas = sorted(int(k.split("/")[1][8:]) for k in s.keys() if k.endswith("meter1"))
    if args.inventario:
        for b in casas:
            md = medidores(s, b)
            if len(md) < 3:
                print(f"house {b}: lacks {set(TIPO) - set(md)}, excluded")
                continue
            agg, ok = cinco_min([potencia(s, b, 1), potencia(s, b, 2)])
            for ap, ms in md.items():
                _, ok_ap = cinco_min([potencia(s, b, m) for m in ms])
                ok = ok & ok_ap.reindex(ok.index, fill_value=False)
            print(f"house {b}: meters {md}; {int(ok.sum())} valid 5-min intervals "
                  f"({ok.sum() * 300 / 86400:.1f} days)", flush=True)
    if args.casa:
        b = args.casa
        md = medidores(s, b)
        agg, ok_agg = cinco_min([potencia(s, b, 1), potencia(s, b, 2)])
        for ap, ms in md.items():
            x, ok_ap = cinco_min([potencia(s, b, m) for m in ms])
            ok = ok_agg & ok_ap.reindex(ok_agg.index, fill_value=False)
            t = ok.index[ok.to_numpy()]
            main_ = agg.reindex(t).to_numpy(np.float32)
            ap_ = x.reindex(t).to_numpy(np.float32)
            slug = ap.replace(" ", "_")
            arq = PROTO / f"redd_casa_{b}__{slug}__5min__cross.npz"
            np.savez_compressed(arq, t=(t.asi8 // 10**9).astype(np.int64), main=main_,
                                **{slug: ap_, f"status_{slug}": (ap_ >= LIMIAR[ap]).astype(np.int8)})
            print(f"  {arq.name}: {len(t) * 300 / 86400:.1f} days", flush=True)
    s.close()


if __name__ == "__main__":
    main()
