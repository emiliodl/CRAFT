"""Mostra a MESMA janela em quatro vistas: bruta, 1a diferenca, 2a diferenca e
desvio padrao movel. A janela e sorteada entre as que contem uma ativacao do
aparelho (uma borda de subida do status dentro de uma corrida contigua).

Uso:
    python scripts/vistas_janela.py                       # janela aleatoria
    python scripts/vistas_janela.py --semente 7
    python scripts/vistas_janela.py --serie refit__dish_washer__casa_10.npz
    python scripts/vistas_janela.py --janela 288 --std 6  # tamanhos em janelas de 5 min
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
PASTA = RAIZ / "datasets" / "unificado_5min"
FIGURAS = RAIZ / "figuras"


def corridas(t: np.ndarray, taxa_s: int = 300) -> list[tuple[int, int]]:
    """(inicio, fim) de cada trecho de amostras consecutivas."""
    quebras = np.flatnonzero(np.diff(t) != taxa_s) + 1
    limites = np.concatenate(([0], quebras, [len(t)]))
    return list(zip(limites[:-1], limites[1:]))


def sortear_janela(arq: Path, largura: int, rng: np.random.Generator):
    """Devolve (t, main, pot, status, aparelho) de uma janela com ativacao."""
    d = np.load(arq)
    ap = next(k for k in d.files if k not in ("t", "main") and not k.startswith("status_"))
    t, main, pot, st = d["t"], d["main"], d[ap], d[f"status_{ap}"]

    # candidatos: bordas de subida com largura suficiente de cada lado, sem
    # atravessar buraco temporal
    candidatos = []
    for i, f in corridas(t):
        if f - i < largura:
            continue
        subidas = np.flatnonzero(np.diff(st[i:f]) == 1) + 1 + i
        for s in subidas:
            ini = s - largura // 3          # ativacao no primeiro terco da janela
            if ini >= i and ini + largura <= f:
                candidatos.append(ini)
    if not candidatos:
        return None
    ini = int(rng.choice(candidatos))
    sl = slice(ini, ini + largura)
    return t[sl], main[sl], pot[sl], st[sl], ap.replace("_", " ")


def marcar_ligado(eixo, x, status):
    """Faixas verticais onde o aparelho esta ligado."""
    bordas = np.diff(np.concatenate(([0], status, [0])).astype(int))
    for a, b in zip(np.flatnonzero(bordas == 1), np.flatnonzero(bordas == -1)):
        eixo.axvspan(x[a], x[min(b, len(x) - 1)], color="#F0B429", alpha=0.18, lw=0, zorder=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--serie", default=None, help="nome do .npz; padrao: sorteia")
    p.add_argument("--janela", type=int, default=288, help="largura em janelas de 5 min (288 = 1 dia)")
    p.add_argument("--std", type=int, default=6, help="janela do desvio padrao movel (6 = 30 min)")
    p.add_argument("--semente", type=int, default=None)
    p.add_argument("--saida", default=None)
    args = p.parse_args()

    rng = np.random.default_rng(args.semente)
    arquivos = sorted(PASTA.glob("*.npz"))
    if args.serie:
        arquivos = [PASTA / args.serie]
    else:
        rng.shuffle(arquivos)

    for arq in arquivos:
        r = sortear_janela(arq, args.janela, rng)
        if r is not None:
            break
    else:
        raise SystemExit("nenhuma janela com ativacao encontrada")

    t, principal, pot, status, aparelho = r
    x = np.arange(len(t)) * 5 / 60.0                       # horas desde o inicio
    inicio = datetime.fromtimestamp(int(t[0]), tz=timezone.utc).replace(tzinfo=None)
    dataset, _, casa = arq.stem.split("__")

    # as quatro vistas, todas do AGREGADO (a entrada do classificador)
    d1 = np.diff(principal, prepend=principal[0])
    d2 = np.diff(d1, prepend=d1[0])
    k = args.std
    pad = np.pad(principal, (k - 1, 0), mode="edge")
    movel = np.lib.stride_tricks.sliding_window_view(pad, k)
    dpm = movel.std(axis=1)

    cor_main, cor_ap = "#2C5EA8", "#B35C00"
    fig, eixos = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                              gridspec_kw=dict(hspace=0.18))

    a = eixos[0]
    a.plot(x, principal, color=cor_main, lw=1.4, label="agregado")
    a.plot(x, pot, color=cor_ap, lw=1.2, alpha=0.85, label=f"{aparelho} (referencia)")
    a.set_ylabel("potencia (W)")
    a.set_title("1. serie bruta", loc="left", fontsize=11, fontweight="bold")
    a.legend(loc="upper right", frameon=False, fontsize=9)

    a = eixos[1]
    a.plot(x, d1, color=cor_main, lw=1.2)
    a.axhline(0, color="#999999", lw=0.8, zorder=0)
    a.set_ylabel("Δ potencia (W)")
    a.set_title("2. 1a diferenca do agregado", loc="left", fontsize=11, fontweight="bold")

    a = eixos[2]
    a.plot(x, d2, color=cor_main, lw=1.2)
    a.axhline(0, color="#999999", lw=0.8, zorder=0)
    a.set_ylabel("Δ² potencia (W)")
    a.set_title("3. 2a diferenca do agregado", loc="left", fontsize=11, fontweight="bold")

    a = eixos[3]
    a.plot(x, dpm, color=cor_main, lw=1.4)
    a.set_ylabel("desvio padrao (W)")
    a.set_title(f"4. desvio padrao movel do agregado ({k} janelas = {k * 5} min)",
                loc="left", fontsize=11, fontweight="bold")
    a.set_xlabel(f"horas desde {inicio:%Y-%m-%d %H:%M}")

    for a in eixos:
        marcar_ligado(a, x, status)
        a.grid(alpha=0.18, lw=0.6)
        a.margins(x=0.01)
        for lado in ("top", "right"):
            a.spines[lado].set_visible(False)

    fig.suptitle(f"{dataset.upper()} · {casa.replace('_', ' ')} · {aparelho} · "
                 f"{len(t)} janelas de 5 min  (faixa amarela = aparelho ligado)",
                 fontsize=12.5, y=0.955)

    FIGURAS.mkdir(exist_ok=True)
    saida = Path(args.saida) if args.saida else FIGURAS / f"vistas_{arq.stem}.png"
    fig.savefig(saida, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"serie   : {arq.name}")
    print(f"janela  : {inicio:%Y-%m-%d %H:%M} + {len(t) * 5 / 60:.1f} h "
          f"({int(status.sum())} janelas ligado, {100 * status.mean():.1f}%)")
    print(f"agregado: mediana {np.median(principal):.0f} W, pico {principal.max():.0f} W")
    print(f"aparelho: pico {pot.max():.0f} W")
    print(f"figura  : {saida}")


if __name__ == "__main__":
    main()
