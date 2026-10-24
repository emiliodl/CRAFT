"""Dados do v2: series em watts, com aparelho, lacunas e p95 por casa.

O EviCon precisa do agregado em WATTS (a amplitude identifica o aparelho), da
potencia do aparelho (P_ref, L_pot, pares contrafactuais) e da mascara de
lacunas (v2 §3.0). Tudo isso ja esta nos npz do protocolo e em
`protocolo_artigo/lacunas/` (gerado por scripts/auditoria_p0.py).

As quatro linhas de `Serie.vistas` aqui NAO sao as vistas do resto do projeto:

  0 agregado (W)   1 aparelho (W)   2 lacuna (fracao do intervalo)   3 p95 da casa (W)

Reusar `Serie` permite reusar `juntar`, `alvos_validos` e as divisoes sem
reescrever nada; o p95 vai como linha constante para sobreviver ao `juntar`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dados import Serie, corridas
from ponto import alvos_validos
from taxa import ROTULO, TAXA, amostras
from treino import DISPOSITIVO

RAIZ = Path(__file__).resolve().parents[1]
PROTO = RAIZ / "datasets" / "protocolo_artigo"
LIMIAR_ON = {"freezer": 10.0, "washing machine": 20.0, "television": 10.0,
             "kettle": 2000.0, "computer": 10.0, "microwave": 10.0,
             "fridge": 50.0, "dishwasher": 10.0}


def arquivo(fonte: str, casa: int, aparelho: str, particao: str, taxa: str = ROTULO) -> str:
    return f"{fonte}_casa_{casa}__{aparelho.replace(' ', '_')}__{taxa}__{particao}.npz"


def carregar_w(nome: str, p95: float | None = None, casa_id: int = 0) -> Serie | None:
    """npz do protocolo -> Serie com [agregado, aparelho, lacuna, p95] em W.

    `p95=None` calcula o p95 do proprio agregado do arquivo (permitido: e so o
    agregado da casa, disponivel em operacao). Para o mes de teste da casa 8 o
    v2 manda usar o p95 do ano de treino - quem chama passa o valor.
    """
    caminho = PROTO / nome
    if not caminho.exists():
        return None
    d = np.load(caminho)
    ap = next(k for k in d.files if k not in ("t", "main") and not k.startswith("status_"))
    t, x, a, y = d["t"].astype(np.int64), d["main"].astype(np.float32), \
        d[ap].astype(np.float32), d[f"status_{ap}"].astype(np.float32)
    lac = np.zeros(len(t), np.float32)
    arq_lac = PROTO / "lacunas" / nome
    if arq_lac.exists():
        dl = np.load(arq_lac)
        if len(dl["t"]) == len(t) and (dl["t"] == t).all():
            lac = np.maximum(dl["agregado"], dl["aparelho"]).astype(np.float32)
    if p95 is None:
        p95 = float(np.percentile(x, 95))
    v = np.stack([x, a, lac, np.full(len(t), max(p95, 1.0), np.float32)])
    trechos = corridas(t, TAXA)
    centros = np.concatenate([np.arange(i, j) for i, j in trechos]).astype(np.int64)
    s = Serie(v, y, centros, caminho.stem, t)
    s.casa_id = casa_id
    return s


def p_ref(series: list[Serie], limiar: float) -> float:
    """Potencia mediana do aparelho LIGADO nas casas fonte (so centros de treino)."""
    v = np.concatenate([s.vistas[1][s.centros] for s in series])
    return float(np.median(v[v >= limiar])) if (v >= limiar).any() else limiar


def duracoes(y: np.ndarray, t: np.ndarray, taxa_s: int = TAXA) -> np.ndarray:
    """Duracao (em intervalos) de cada sessao ligada."""
    y = np.asarray(y).astype(int)
    out, n = [], 0
    for i in range(len(y)):
        cont = i > 0 and t[i] - t[i - 1] == taxa_s
        if y[i] and (n == 0 or cont):
            n += 1
        elif y[i]:
            out.append(n)
            n = 1
        else:
            if n:
                out.append(n)
            n = 0
    if n:
        out.append(n)
    return np.asarray(out, int)


class Banco:
    """Ativacoes reais (corridas ligadas + 1 intervalo de cada lado), em W, na GPU."""

    def __init__(self, series: list[Serie], limiar: float, lmax: int = amostras(288)):
        ativ, casas = [], []
        for s in series:
            y, a, t = s.rotulo, s.vistas[1], s.t
            m = np.zeros(len(y), bool)
            m[s.centros] = True
            yy = (y > 0) & m
            d = np.diff(np.r_[0, yy.astype(int), 0])
            for i, j in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                i0, j0 = max(i - 1, 0), min(j + 1, len(y))
                if (np.diff(t[i0:j0]) != TAXA).any() or j0 - i0 > lmax:
                    continue
                ativ.append(a[i0:j0].copy())
                casas.append(getattr(s, "casa_id", 0))
        self.n = len(ativ)
        self.lmax = max([len(x) for x in ativ], default=1)
        pad = np.zeros((max(self.n, 1), self.lmax), np.float32)
        for k, x in enumerate(ativ):
            pad[k, :len(x)] = x
        self.ativ = torch.from_numpy(pad).to(DISPOSITIVO)
        self.comp = torch.tensor([len(x) for x in ativ] or [1], device=DISPOSITIVO)
        self.casa = torch.tensor(casas or [0], device=DISPOSITIVO)
        self.limiar = limiar


class Fonte:
    """Agregado/aparelho/lacuna/p95 na GPU; lotes de janelas CRUAS em W.

    Uma janela de contexto de largura w_c (centrada) e montada para cada alvo; a
    janela de evidencia e o recorte central de largura w_e dela.
    """

    def __init__(self, s: Serie, w_c: int, passo: int = 1, limiar: float = 10.0):
        self.antes, self.depois = w_c // 2, w_c - w_c // 2 - 1
        self.v = torch.from_numpy(s.vistas).to(DISPOSITIVO)
        self.y = torch.from_numpy(s.rotulo).to(DISPOSITIVO)
        self.t = s.t
        self.alvos_np = alvos_validos(s, self.antes, self.depois, passo)
        self.alvos = torch.from_numpy(self.alvos_np).to(DISPOSITIVO)
        self.desloc = torch.arange(-self.antes, self.depois + 1, device=DISPOSITIVO)
        self.nome = s.nome
        self.limiar = limiar
        cid = getattr(s, "casa_id_por_amostra", None)
        self.casa = (torch.from_numpy(cid).to(DISPOSITIVO) if cid is not None else None)

    def __len__(self) -> int:
        return len(self.alvos)

    def lote(self, pos: torch.Tensor) -> dict:
        alvo = self.alvos[pos]
        idx = alvo[:, None] + self.desloc[None, :]
        return dict(x=self.v[0][idx], app=self.v[1][idx], lac=self.v[2][idx],
                    p95=self.v[3][alvo], y=self.y[alvo], yj=self.y[idx], alvo=alvo)


def juntar_w(series: list[Serie], nome: str) -> Serie:
    """Como dados.juntar, guardando a casa de cada amostra (para o banco e o LOHO)."""
    from dados import juntar
    s = juntar(series, nome)
    s.casa_id_por_amostra = np.concatenate(
        [np.full(x.vistas.shape[1], getattr(x, "casa_id", 0), np.int64) for x in series])
    return s
