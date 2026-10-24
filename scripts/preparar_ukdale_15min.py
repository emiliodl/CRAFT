"""15-min protocol files for the UK-DALE scenarios (train house 1; test house 1 the following
month and house 5), as scripts/preparar_ukdale_cenarios.py does at 5 min.

  house 1, training  from the saved 30-s training grid (datasets/v3_fino/ukdale_s_casa_1.npz),
                     so that the 15-min intervals line up with the grid the ResNets train on
  house 1, test      2014-03-01 .. 2014-04-01
  house 5, test      the whole recording
  REFIT 3, 7, 20     fridge and dishwasher (validation / calibration), from their 30-s grids

15-min means use the sample counts, and an interval is valid with >= 50% of the typical
count, as at 5 min. Every series starts on a multiple of 900 s. Prevalences are not
printed (test houses of a pending pre-registration).

Usage:
    python scripts/preparar_ukdale_15min.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
from preparar_ukdale_cenarios import CANAL, LIMIAR, TESTE1, grade  # noqa: E402

FINO = RAIZ / "datasets" / "v3_fino"
PROTO = RAIZ / "datasets" / "protocolo_artigo"
SUB, TAXA = 30, 900                      # 30-s bins per 15-min interval


def alinhar(g: dict) -> dict:
    """Drop leading bins so that the series starts on a multiple of 900 s."""
    i0 = int((-g["t0"]) % TAXA // 30)
    n = (g["x"].shape[1] - i0) // SUB * SUB
    return dict(t0=g["t0"] + i0 * 30, x=g["x"][:, i0:i0 + n], cont=g["cont"][i0:i0 + n],
                nomes=g["nomes"])


def protocolo(g: dict, canal: str, aparelho: str, arq: Path) -> None:
    assert g["t0"] % TAXA == 0
    nomes = [str(n) for n in g["nomes"]]
    k = nomes.index(canal)
    n = g["x"].shape[1] // SUB
    w = g["cont"][:n * SUB].reshape(n, SUB).astype(np.float64)
    a = g["x"][0][:n * SUB].reshape(n, SUB)
    p = g["x"][k][:n * SUB].reshape(n, SUB)
    ww = np.where(np.isfinite(a) & np.isfinite(p), w, 0)
    tip = np.median(w.sum(1)[w.sum(1) > 0])
    main = (np.nan_to_num(a) * ww).sum(1) / np.maximum(ww.sum(1), 1e-9)
    ap = (np.nan_to_num(p) * ww).sum(1) / np.maximum(ww.sum(1), 1e-9)
    ok = ww.sum(1) >= 0.5 * tip
    t = g["t0"] + np.arange(n) * TAXA
    slug = aparelho.replace(" ", "_")
    np.savez_compressed(arq, t=t[ok].astype(np.int64), main=main[ok].astype(np.float32),
                        **{slug: ap[ok].astype(np.float32),
                           f"status_{slug}": (ap[ok] >= LIMIAR[aparelho]).astype(np.int8)})
    print(f"   {arq.name}: {ok.sum() * TAXA / 86400:.1f} days", flush=True)


def main() -> None:
    d = np.load(FINO / "ukdale_s_casa_1.npz")
    tr1 = dict(t0=int(d["t0"]), x=d["x"], cont=d["cont"], nomes=d["nomes"])
    # the saved training grid renamed the washer dryer to "Washing Machine"
    canal_tr = dict(CANAL[("ukdale", 1)], **{"washing machine": "Washing Machine"})
    for ap, canal in canal_tr.items():
        protocolo(tr1, canal, ap,
                  PROTO / f"ukdale_casa_1__{ap.replace(' ', '_')}__15min__treino.npz")
    te1 = alinhar(grade(1, *TESTE1))
    for ap, canal in CANAL[("ukdale", 1)].items():
        protocolo(te1, canal, ap,
                  PROTO / f"ukdale_casa_1__{ap.replace(' ', '_')}__15min__teste.npz")
    g5 = alinhar(grade(5, None, None))
    for ap, canal in CANAL[("ukdale", 5)].items():
        protocolo(g5, canal, ap,
                  PROTO / f"ukdale_casa_5__{ap.replace(' ', '_')}__15min__cross.npz")
    for c in (3, 7, 20):
        e = np.load(FINO / f"refit_casa_{c}.npz")
        g = dict(t0=int(e["t0"]), x=e["x"], cont=e["cont"], nomes=e["nomes"])
        for ap, canal in CANAL[("refit", c)].items():
            protocolo(g, canal, ap, PROTO / f"refit_casa_{c}__{ap}__15min__treino.npz")


if __name__ == "__main__":
    main()
