"""ReCo: Recomposicao + Consistencia (docs/v3.md §3).

Gerador de recomposicao (§3.2, §8), backbone ResNet 1D (§3.3) e perda (§3.5).

Dados: grade de 30 s (scripts/preparar_v3.py) com agregado, os 9 plugues e a
contagem de amostras nativas de cada caixa. Cada componente de uma janela
recomposta (fundo, ativacao, distrator) e reamostrado para 5 min com a SUA
contagem - a media que o seu medidor teria registrado -, e so entao somado:

    reamostrar(x) = sum(cont * x) / sum(cont)   por intervalo de 5 min

Para uma serie real isso reproduz exatamente a media do protocolo; para uma soma
com os mesmos pesos, reamostrar(B + a) = reamostrar(B) + reamostrar(a) (teste 3).
A fase da ativacao injetada cai em qualquer das 10 caixas de 30 s do intervalo
central: e o que "injetar na resolucao nativa e reamostrar depois" preserva.

Operacoes (§3.2), todas com rotulo recalculado pelo protocolo sobre o alvo
reamostrado (media >= limiar):

  R0 real       A_h
  R1 injecao    B_k + s * a_j,   ativacao cobrindo o centro
  R2 remocao    A_h - P_h,       numa janela em que o alvo estava ligado no centro
  R3 troca      B_k + P_h,       o mesmo alvo, fundo de outra casa ou outro dia
  R4 distrator  B_k + s * d_j,   alvo ausente

B_k = A_k - (todos os plugues da categoria do alvo na casa k). Decisoes de
implementacao declaradas:

  - fundos so de janelas "limpas" (alvo desligado a janela inteira, nos 5 min):
    minimiza o artefato de subtracao do canal da TV (o agregado registra 40-60%
    do degrau do plugue, v2 E0.1);
  - fundo da mesma casa so de OUTRO dia (|dt| >= 1 dia), sem casar a hora;
  - o corte em zero vale so para o agregado recomposto final;
  - janelas com qualquer caixa de 30 s ausente nao entram (lacuna = linha
    ausente); caixas vazias isoladas (<= 60 s) foram interpoladas na grade.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

RAIZ = Path(__file__).resolve().parents[1]
FINO = RAIZ / "datasets" / "v3_fino"
from taxa import SUB, TAXA, amostras   # noqa: E402  (SUB = 10 a 5 min, 30 a 15 min)
W5 = amostras(48)          # janela de 4 h (contexto curto do E1.3 do v2)
ANTES = W5 // 2            # centro na posicao 24 (antes 24, depois 23), como o v2

CATEGORIAS = {
    "television": r"^(television|tv)\b",
    "washing machine": r"^washing machine",
    "dryer": r"^(tumble dryer|washer dryer)",
    "dishwasher": r"^(dishwasher|dish washer)",
    "kettle": r"^kettle",
    "microwave": r"^microwave",
    "computer": r"^(computer|desktop computer|mjy computer|pgm computer)",
    "hifi": r"^(hi-fi|games console)",
    # pos-v3: freezer (padrao do protocolo) e geladeira/combinado (o distrator dele)
    "freezer": r"^freezer\b",
    "fridge": r"^(fridge|chest freezer)",
}
LIMIAR_CAT = {"washing machine": 20.0, "kettle": 2000.0, "fridge": 50.0}
# limiares de "ligado" do codigo de referencia do artigo (nilmtk losses.py):
# fridge 50 W, dish washer 10 W, washing machine 20 W, demais 10 W
LIMIAR_ALVO = {"television": 10.0, "washing machine": 20.0, "freezer": 10.0,
               "fridge": 50.0, "dishwasher": 10.0}
# §3.8 e portao G0 (D1): distratores confundiveis por aparelho
DISTRATORES = {"television": ["computer", "hifi"],
               "washing machine": ["dryer", "dishwasher"],
               "freezer": [],      # pos-v3: a geladeira confunde mais do que ensina
               "fridge": [],       # cenarios UK-DALE: como o freezer (ciclico)
               "dishwasher": ["washing machine", "dryer"]}
# aparelho ciclico (liga a cada poucos minutos): quase nao ha janela de 4 h com ele
# desligado; o fundo e A - P em qualquer janela, sem exigir janela "limpa"
FUNDO_LIMPO = {"freezer": False, "fridge": False}
PROPORCOES = {   # §3.2, fracoes iniciais
    "washing machine": dict(R0=0.40, R3=0.20, R1=0.15, R2=0.10, R4=0.15),
    "television": dict(R0=0.40, R3=0.25, R1=0.15, R2=0.10, R4=0.10),
    "freezer": dict(R0=0.40, R3=0.25, R1=0.15, R2=0.10, R4=0.10),
    "fridge": dict(R0=0.40, R3=0.25, R1=0.15, R2=0.10, R4=0.10),
    "dishwasher": dict(R0=0.40, R3=0.20, R1=0.15, R2=0.10, R4=0.15),
}
ESCALA = {"television": (0.5, 3.0), "washing machine": (0.7, 1.5), "freezer": (0.7, 1.5),
          "fridge": (0.7, 1.5), "dishwasher": (0.7, 1.5)}
OPS = ("R0", "R1", "R2", "R3", "R4", "Rf")
# pos-quali, item 2: diversidade de ciclos da maquina de lavar na R1. Estica/comprime
# a ativacao no tempo (fator r), modula a potencia por trechos de `seg` caixas de 30 s
# (ganho log-uniforme) e, com prob. `p_emenda`, emenda o inicio de um ciclo ao fim de
# outro num ponto sorteado do progresso. O rotulo continua recalculado depois.
DEFORM = dict(r=(0.7, 1.4), ganho=(0.6, 1.6), seg=20, p_emenda=0.5)
# pos-quali, item 3: R1 do freezer como TREM de ciclos reais. Um ciclo do banco,
# esticado por r, repetido com periodo = ligado * (1 + razao desligado/ligado), fase
# sorteada; cobre a janela inteira, como um compressor de verdade.
TREM = dict(r=(0.7, 1.4), razao=(0.6, 2.5))     # Rf = fundo sozinho, parceiro "sem" do R1


def aux_de(aparelho: str) -> list[str]:
    return [c for c in CATEGORIAS if c != aparelho]


def canais(nomes: list[str], categoria: str) -> list[int]:
    pad = re.compile(CATEGORIAS[categoria], re.I)
    return [i for i, n in enumerate(nomes) if i > 0 and pad.match(n.strip())]


# --------------------------------------------------------------------------------------
# casas na grade fina
# --------------------------------------------------------------------------------------


@dataclass
class CasaFina:
    casa: int
    t0: int
    A: np.ndarray            # agregado (N,)
    P: np.ndarray            # canal do protocolo do alvo (N,)
    Pcat: np.ndarray         # soma dos canais da categoria do alvo (N,)
    Q: np.ndarray            # (K, N) soma por categoria auxiliar
    qmask: np.ndarray        # (K,) categoria submedida nesta casa
    W: np.ndarray            # contagem de amostras (N,)
    ok: np.ndarray           # caixa com dado (N,)
    periodo: np.ndarray = field(default=None)   # (N,) caixa utilizavel neste periodo


def arquivo_fino(casa: int) -> Path:
    """REFIT: id da casa; UK-DALE: 200 + numero do predio (201, 202, ...);
    cenarios UK-DALE (scripts/preparar_ukdale_cenarios.py): 300 + predio, so o
    periodo de treino do cenario."""
    if casa > 300:
        return FINO / f"ukdale_s_casa_{casa - 300}.npz"
    return (FINO / f"ukdale_casa_{casa - 200}.npz" if casa > 200
            else FINO / f"refit_casa_{casa}.npz")


def carregar_casa(casa: int, aparelho: str) -> CasaFina | None:
    arq = arquivo_fino(casa)
    if not arq.exists():
        return None
    d = np.load(arq)
    nomes = [str(n) for n in d["nomes"]]
    x = d["x"]
    ch = canais(nomes, aparelho)
    if not ch:
        return None
    # no REFIT todos os canais vem da mesma linha; no UK-DALE cada medidor tem a sua
    # cobertura: exige agregado e alvo, e os demais canais ausentes contam como 0
    ok = np.isfinite(x[0]) & np.isfinite(x[ch]).all(0)
    xz = np.nan_to_num(x)
    aux = aux_de(aparelho)
    Q = np.zeros((len(aux), x.shape[1]), np.float32)
    qm = np.zeros(len(aux), bool)
    for k, c in enumerate(aux):
        cc = canais(nomes, c)
        if cc:
            Q[k] = xz[cc].sum(0)
            qm[k] = True
    return CasaFina(casa=casa, t0=int(d["t0"]), A=xz[0], P=xz[ch[0]], Pcat=xz[ch].sum(0),
                    Q=Q, qmask=qm, W=d["cont"].astype(np.float32), ok=ok)


def definir_periodo(c: CasaFina, modo: str, semanas: float = 4.0, frac: float = 0.15,
                    embargo_dias: float = 1.0, janela: int = W5) -> dict:
    """Mascaras 'treino'/'val' por caixa fina.

    modo='cross': val = ultimas `semanas` com dado (holdout temporal do v2);
    modo='intra': val = ultimos `frac` do ano (§3.7). Em ambos, embargo de
    `embargo_dias` + uma janela entre o fim do treino e o inicio da val."""
    idx = np.flatnonzero(c.ok)
    fim = idx.max()
    if modo == "cross":
        corte = fim - int(semanas * 7 * 86400 / 30)
    else:
        corte = idx.min() + int((1 - frac) * (fim - idx.min()))
    corte -= corte % SUB
    emb = int(embargo_dias * 86400 / 30) + janela * SUB
    n = len(c.A)
    tr = np.zeros(n, bool)
    va = np.zeros(n, bool)
    tr[:max(corte - emb, 0)] = True
    va[corte:] = True
    return dict(treino=tr & c.ok, val=va & c.ok)


# --------------------------------------------------------------------------------------
# reamostragem
# --------------------------------------------------------------------------------------


def reamostrar(v: Tensor, w: Tensor) -> Tensor:
    """(B, L*SUB) valores e pesos -> (B, L) medias ponderadas por intervalo."""
    b = v.shape[0]
    num = (v * w).view(b, -1, SUB).sum(-1)
    den = w.view(b, -1, SUB).sum(-1).clamp_min(1e-6)
    return num / den


def rotulo(a5: Tensor, limiar: float) -> Tensor:
    return (a5 >= limiar).float()


# --------------------------------------------------------------------------------------
# gerador
# --------------------------------------------------------------------------------------


class Gerador:
    """Bancos de fundos, ativacoes e distratores de UM conjunto de casas e UM
    periodo, na GPU, e a montagem dos lotes recompostos com os pares.

    Tudo e indexado numa serie plana (casas concatenadas); uma janela e
    identificada pelo indice fino do INICIO do intervalo central."""

    def __init__(self, casas: list[CasaFina], aparelho: str, periodo: str, disp,
                 proporcoes: dict | None = None, escala: tuple | None = None,
                 margem: int = amostras(2), lmax: int = amostras(96), passo: int = 1, frac_real: float | None = None,
                 w5: int = W5, deform: dict | None = None, trem: dict | None = None):
        self.aparelho, self.disp = aparelho, disp
        # pos-quali: R1 com ativacao deformada (maquina de lavar) ou com trem de
        # ciclos (freezer); None = injecao original
        self.deform = dict(DEFORM, **deform) if deform is not None else None
        self.trem = dict(TREM, **trem) if trem is not None else None
        self.w5, self.antes = w5, w5 // 2
        self.lim = LIMIAR_ALVO[aparelho]
        self.aux = aux_de(aparelho)
        self.lim_aux = torch.tensor([LIMIAR_CAT.get(c, 10.0) for c in self.aux], device=disp)
        self.casas_ids = [c.casa for c in casas]
        prop = dict(proporcoes or PROPORCOES[aparelho])
        if frac_real is not None:                    # E2.2: reescala as nao reais
            resto = 1 - prop["R0"]
            prop = {k: (frac_real if k == "R0" else v * (1 - frac_real) / resto)
                    for k, v in prop.items()}
        self.prop = prop
        self.escala = escala or ESCALA[aparelho]
        cat = lambda xs: torch.from_numpy(np.concatenate(xs)).to(disp)   # noqa: E731
        self.A = cat([c.A for c in casas])
        self.P = cat([c.P for c in casas])
        self.B = cat([c.A - c.Pcat for c in casas])
        self.W = cat([c.W for c in casas])
        self.Q = torch.from_numpy(np.concatenate([c.Q for c in casas], 1)).to(disp)
        self.qmask = torch.from_numpy(np.stack([c.qmask for c in casas])).to(disp)   # (H, K)
        ofs = np.cumsum([0] + [len(c.A) for c in casas])
        self.ofs = ofs
        self.casa_de = torch.from_numpy(np.concatenate(
            [np.full(len(c.A), i, np.int64) for i, c in enumerate(casas)])).to(disp)
        self.dia = torch.from_numpy(np.concatenate(
            [(c.t0 + np.arange(len(c.A)) * 30) // 86400 for c in casas])).to(disp)
        self.tsec = torch.from_numpy(np.concatenate(
            [c.t0 + np.arange(len(c.A), dtype=np.int64) * 30 for c in casas])).to(disp)
        self.desloc = torch.arange(-self.antes * SUB, (self.w5 - self.antes) * SUB, device=disp)
        # ---- janelas validas (inicio do intervalo central) ----
        centros, on, limpo = [], [], []
        for i, c in enumerate(casas):
            per = c.periodo[periodo] if isinstance(c.periodo, dict) else c.ok
            n = len(c.A)
            n5 = n // SUB
            ok5 = per[:n5 * SUB].reshape(n5, SUB).all(1)
            w = c.W[:n5 * SUB].reshape(n5, SUB)
            p5 = (c.P[:n5 * SUB].reshape(n5, SUB) * w).sum(1) / np.maximum(w.sum(1), 1e-6)
            pc5 = (c.Pcat[:n5 * SUB].reshape(n5, SUB) * w).sum(1) / np.maximum(w.sum(1), 1e-6)
            cs = np.concatenate([[0], np.cumsum(~ok5)])
            cs_on = np.concatenate([[0], np.cumsum(pc5 >= self.lim)])
            j = np.arange(self.antes, n5 - (self.w5 - self.antes))
            j = j[(cs[j + self.w5 - self.antes] - cs[j - self.antes]) == 0]
            # so os intervalos alinhados a 5 min da epoca (t0 alinhado: j*SUB)
            centros.append(ofs[i] + j * SUB)
            on.append(p5[j] >= self.lim)
            limpo.append((cs_on[j + self.w5 - self.antes] - cs_on[j - self.antes]) == 0)
        self.centros = torch.from_numpy(np.concatenate(centros)).to(disp)
        self.on = torch.from_numpy(np.concatenate(on)).to(disp)
        self.limpo = torch.from_numpy(np.concatenate(limpo)).to(disp)
        self.idx_on = torch.nonzero(self.on).squeeze(1)
        self.idx_limpo = torch.nonzero(self.limpo).squeeze(1)
        self.idx_fundo = (self.idx_limpo if FUNDO_LIMPO.get(aparelho, True)
                          else torch.arange(len(self.centros), device=disp))
        self.base = self.centros[::passo] if passo > 1 else self.centros
        # ---- bancos de ativacoes do alvo e de distratores (indices na serie plana) ----
        self.ativ = self._banco(casas, [None], periodo, margem, lmax)
        dis = [self.aux.index(c) for c in DISTRATORES.get(aparelho, []) if c in self.aux]
        self.dist = self._banco(casas, dis, periodo, margem, lmax)
        self.pad_w = float(torch.median(self.W[self.W > 0]))

    def _banco(self, casas, cats, periodo, margem, lmax) -> dict:
        """Corridas LIGADAS (5 min) + `margem` intervalos de cada lado, sem lacuna e
        dentro do periodo. cats=[None] -> o alvo; senao indices de Q."""
        ini, fim, cat, ini_on, fim_on, nivel = [], [], [], [], [], []
        for i, c in enumerate(casas):
            per = c.periodo[periodo] if isinstance(c.periodo, dict) else c.ok
            n5 = len(c.A) // SUB
            w = c.W[:n5 * SUB].reshape(n5, SUB)
            ok5 = per[:n5 * SUB].reshape(n5, SUB).all(1)
            for k in cats:
                if k is not None and not c.qmask[k]:
                    continue
                s = c.P if k is None else c.Q[k]
                lim = self.lim if k is None else float(self.lim_aux[k])
                base = 0.0
                if k is not None:
                    # distrator: ativacao relativa ao nivel de repouso do plugue (o
                    # computador da casa 8 fica em 18 W o tempo todo); limiar =
                    # max(limiar da categoria, mediana + 20 W), e o repouso e
                    # subtraido na injecao
                    base = float(np.median(s[c.ok]))
                    lim = max(lim, base + 20.0)
                s5 = (s[:n5 * SUB].reshape(n5, SUB) * w).sum(1) / np.maximum(w.sum(1), 1e-6)
                lig = (s5 >= lim) & ok5
                d = np.diff(np.r_[0, lig.astype(np.int8), 0])
                for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                    a0, b0 = a - margem, b + margem
                    if a0 < 0 or b0 > n5 or b0 - a0 > lmax or not ok5[a0:b0].all():
                        continue
                    if (s5[a0:a] >= lim).any() or (s5[b:b0] >= lim).any():
                        continue
                    ini.append(self.ofs[i] + a0 * SUB)
                    fim.append(self.ofs[i] + b0 * SUB)
                    ini_on.append(self.ofs[i] + a * SUB)
                    fim_on.append(self.ofs[i] + b * SUB)
                    cat.append(-1 if k is None else k)
                    nivel.append(base)
        t = lambda v: torch.tensor(v, dtype=torch.long, device=self.disp)   # noqa: E731
        return dict(ini=t(ini), fim=t(fim), ini_on=t(ini_on), fim_on=t(fim_on), cat=t(cat),
                    nivel=torch.tensor(nivel, dtype=torch.float32, device=self.disp),
                    n=len(ini))

    # ---------------- primitivas ----------------

    def janela(self, serie: Tensor, c: Tensor) -> Tensor:
        return serie[c[:, None] + self.desloc[None, :]]

    def res(self, serie: Tensor, c: Tensor) -> Tensor:
        """Serie reamostrada com a contagem da PROPRIA janela."""
        return reamostrar(self.janela(serie, c), self.janela(self.W, c))

    def aux_res(self, c: Tensor) -> tuple[Tensor, Tensor]:
        """Rotulos auxiliares no centro e mascara (categoria submedida na casa)."""
        idx = c[:, None] + self.desloc[None, :]
        w = self.W[idx]
        q = self.Q[:, idx]                                             # (K, B, L)
        q5 = ((q * w[None]).view(q.shape[0], q.shape[1], -1, SUB).sum(-1)
              / w.view(w.shape[0], -1, SUB).sum(-1).clamp_min(1e-6)[None])
        y = (q5[:, :, self.antes] >= self.lim_aux[:, None]).float().T      # (B, K)
        return y, self.qmask[self.casa_de[c]].float()

    def sorteia_fundo(self, n: int, g: torch.Generator, evitar: Tensor | None = None) -> Tensor:
        """Janelas limpas; se `evitar` (centros base) for dado, fundo da mesma casa so
        de outro dia. Sem casar a hora do dia, de proposito."""
        pool = self.idx_fundo
        f = self.centros[pool[torch.randint(0, len(pool), (n,), device=self.disp, generator=g)]]
        if evitar is not None:
            for _ in range(10):
                ruim = ((self.casa_de[f] == self.casa_de[evitar])
                        & ((self.dia[f] - self.dia[evitar]).abs() < 1))
                if not ruim.any():
                    break
                novo = self.centros[pool[torch.randint(0, len(pool), (n,), device=self.disp,
                                                       generator=g)]]
                f = torch.where(ruim, novo, f)
        return f

    def injetar(self, banco: dict, serie_de, n: int, g: torch.Generator
                ) -> tuple[Tensor, Tensor, Tensor]:
        """Ativacao do banco, reamostrada com a propria contagem, cobrindo o centro
        com fase sorteada. Devolve (a5 (n, W5), indice no banco, escala)."""
        j = torch.randint(0, banco["n"], (n,), device=self.disp, generator=g)
        ini, fim = banco["ini"][j], banco["fim"][j]
        a_on, b_on = banco["ini_on"][j], banco["fim_on"][j]
        u = a_on + (torch.rand(n, device=self.disp, generator=g) * (b_on - a_on)).long()
        fase = torch.randint(0, SUB, (n,), device=self.disp, generator=g)
        # posicao fina u da ativacao cai na caixa ANTES*SUB + fase da janela
        pos = u[:, None] + (self.desloc[None, :] - fase[:, None])
        dentro = (pos >= ini[:, None]) & (pos < fim[:, None])
        posc = pos.clamp(0, len(self.A) - 1)
        s = serie_de(banco["cat"][j])                   # funcao: categoria -> serie por linha
        v = torch.where(dentro, (s(posc) - banco["nivel"][j][:, None]).clamp_min(0.0),
                        torch.zeros((), device=self.disp))
        w = torch.where(dentro, self.W[posc], torch.full((), self.pad_w, device=self.disp))
        esc = torch.exp(torch.empty(n, device=self.disp).uniform_(
            math.log(self.escala[0]), math.log(self.escala[1]), generator=g))
        return reamostrar(v, w) * esc[:, None], j, esc

    def injetar_deformado(self, n: int, g: torch.Generator) -> Tensor:
        """Como `injetar` no banco do alvo, com esticamento, ganho por trecho e emenda
        de dois ciclos (DEFORM). Com r=1, ganho=1 e sem emenda, e identica a `injetar`."""
        cfg, b, d = self.deform, self.ativ, self.disp
        u01 = lambda lo, hi, *sh: torch.exp(torch.empty(*sh, device=d).uniform_(   # noqa: E731
            math.log(lo), math.log(hi), generator=g))
        j = torch.randint(0, b["n"], (n,), device=d, generator=g)
        ini, fim, a_on, b_on = b["ini"][j], b["fim"][j], b["ini_on"][j], b["fim_on"][j]
        u = a_on + (torch.rand(n, device=d, generator=g) * (b_on - a_on)).long()
        fase = torch.randint(0, SUB, (n,), device=d, generator=g)
        r = u01(*cfg["r"], n)
        off = torch.floor((self.desloc[None, :] - fase[:, None]).float() / r[:, None]).long()
        pos = u[:, None] + off
        dentro = (pos >= ini[:, None]) & (pos < fim[:, None])
        # emenda: a partir do progresso rho do ciclo j, le o ciclo j2 no mesmo progresso
        j2 = torch.randint(0, b["n"], (n,), device=d, generator=g)
        ini2, fim2, a2, b2 = b["ini"][j2], b["fim"][j2], b["ini_on"][j2], b["fim_on"][j2]
        rho = torch.empty(n, device=d).uniform_(0.2, 0.8, generator=g)
        emenda = torch.rand(n, device=d, generator=g) < cfg["p_emenda"]
        prog = (pos - a_on[:, None]).float() / (b_on - a_on).clamp_min(1)[:, None].float()
        pos2 = torch.where(pos < b_on[:, None],
                           a2[:, None] + torch.round(prog * (b2 - a2)[:, None].float()).long(),
                           b2[:, None] + (pos - b_on[:, None]))
        usa2 = emenda[:, None] & (prog >= rho[:, None])
        pos = torch.where(usa2, pos2, pos)
        dentro = torch.where(usa2, (pos2 >= ini2[:, None]) & (pos2 < fim2[:, None]), dentro)
        posc = pos.clamp(0, len(self.A) - 1)
        v = torch.where(dentro, self.P[posc].clamp_min(0.0), torch.zeros((), device=d))
        nseg = -(-v.shape[1] // cfg["seg"])
        ganho = u01(*cfg["ganho"], n, nseg).repeat_interleave(cfg["seg"], 1)[:, :v.shape[1]]
        w = torch.where(dentro, self.W[posc], torch.full((), self.pad_w, device=d))
        esc = u01(*self.escala, n)
        return reamostrar(v * ganho, w) * esc[:, None]

    def injetar_trem(self, n: int, g: torch.Generator) -> Tensor:
        """Trem periodico de um ciclo real do banco do alvo cobrindo a janela (TREM)."""
        cfg, b, d = self.trem, self.ativ, self.disp
        u01 = lambda lo, hi: torch.exp(torch.empty(n, device=d).uniform_(   # noqa: E731
            math.log(lo), math.log(hi), generator=g))
        j = torch.randint(0, b["n"], (n,), device=d, generator=g)
        a_on, b_on = b["ini_on"][j], b["fim_on"][j]
        r = u01(*cfg["r"])
        lig = ((b_on - a_on).float() * r).clamp_min(1.0)               # caixas ligado
        per = lig * (1.0 + u01(*cfg["razao"]))
        fase0 = torch.rand(n, device=d, generator=g) * per
        q = torch.arange(self.w5 * SUB, device=d).float()
        rel = torch.remainder(q[None, :] + fase0[:, None], per[:, None])
        on = rel < lig[:, None]
        src = (a_on[:, None] + torch.floor(rel / r[:, None]).long()).clamp(
            max=(b_on - 1)[:, None]).clamp(0, len(self.A) - 1)
        v = torch.where(on, self.P[src].clamp_min(0.0), torch.zeros((), device=d))
        w = torch.where(on, self.W[src], torch.full((), self.pad_w, device=d))
        esc = u01(*self.escala)
        return reamostrar(v, w) * esc[:, None]

    def _serie_alvo(self, cats):
        return lambda posc: self.P[posc]

    def _serie_dist(self, cats):
        return lambda posc: self.Q[cats[:, None].clamp_min(0), posc]

    # ---------------- lote ----------------

    def lote(self, base: Tensor, g: torch.Generator) -> dict:
        """`base`: centros das janelas reais (R0) deste lote. O tamanho nominal do
        lote e len(base) / prop['R0']; as outras operacoes seguem as proporcoes.
        Pares: (R0, R3) para L_inv; (R1, fundo) e (R0 ligado, R2) para L_sens."""
        n0 = len(base)
        tot = n0 / self.prop["R0"]
        n = {k: int(round(tot * v)) for k, v in self.prop.items()}
        partes, pares_inv, pares_sens = [], [], []
        ofs = 0

        def add(x5, y, yaux, maux, op, centro, a5=None):
            # `centro`: janela que da o nivel da casa e a hora (a base em R0/R2,
            # o fundo em R1/R3/R4/Rf)
            nonlocal ofs
            partes.append(dict(x=x5.clamp_min(0.0), y=y, yaux=yaux, maux=maux,
                               op=torch.full((len(y),), OPS.index(op), device=self.disp),
                               casa=self.casa_de[centro], centro=centro,
                               a=a5 if a5 is not None else torch.zeros_like(x5)))
            ofs += len(y)
            return ofs - len(y)

        # R0 (inclui os parceiros de R3 e R2)
        n2 = min(n["R2"], len(self.idx_on))
        on_c = (self.centros[self.idx_on[torch.randint(0, len(self.idx_on), (n2,),
                                                       device=self.disp, generator=g)]]
                if n2 else base[:0])
        c0 = torch.cat([base, on_c])
        p5 = self.res(self.P, c0)
        ya, ma = self.aux_res(c0)
        i0 = add(self.res(self.A, c0), rotulo(p5[:, self.antes], self.lim), ya, ma, "R0",
                 c0, p5)
        # R3: B_k + P_h para os primeiros n3 da base
        n3 = min(n["R3"], n0)
        if n3:
            cb = base[:n3]
            f = self.sorteia_fundo(n3, g, evitar=cb)
            p5b = p5[:n3]
            ya, ma = self.aux_res(f)
            i3 = add(self.res(self.B, f) + p5b, rotulo(p5b[:, self.antes], self.lim), ya, ma, "R3",
                     f, p5b)
            ar = torch.arange(n3, device=self.disp)
            pares_inv.append(torch.stack([i0 + ar, i3 + ar], 1))
        # R2: A_h - P_h nas janelas ligadas
        if n2:
            ya, ma = self.aux_res(on_c)
            x2 = self.res(self.B, on_c)
            i2 = add(x2, torch.zeros(n2, device=self.disp), ya, ma, "R2", on_c)
            ar = torch.arange(n2, device=self.disp)
            pares_sens.append(torch.stack([i0 + n0 + ar, i2 + ar], 1))
        # R1: fundo + ativacao, e o proprio fundo como parceiro "sem"
        n1 = n["R1"] if self.ativ["n"] else 0
        if n1:
            f = self.sorteia_fundo(n1, g)
            b5 = self.res(self.B, f)
            if self.trem is not None:
                a5 = self.injetar_trem(n1, g)
            elif self.deform is not None:
                a5 = self.injetar_deformado(n1, g)
            else:
                a5, _, _ = self.injetar(self.ativ, self._serie_alvo, n1, g)
            ya, ma = self.aux_res(f)
            i1 = add(b5 + a5, rotulo(a5[:, self.antes], self.lim), ya, ma, "R1", f, a5)
            ib = add(b5, torch.zeros(n1, device=self.disp), ya, ma, "Rf", f)
            ar = torch.arange(n1, device=self.disp)
            pares_sens.append(torch.stack([i1 + ar, ib + ar], 1))
        # R4: fundo + distrator
        n4 = n["R4"] if self.dist["n"] else 0
        if n4:
            f = self.sorteia_fundo(n4, g)
            d5, j, _ = self.injetar(self.dist, self._serie_dist, n4, g)
            ya, ma = self.aux_res(f)
            k = self.dist["cat"][j]
            ya = ya.clone()
            ya[torch.arange(n4, device=self.disp), k] = torch.maximum(
                ya[torch.arange(n4, device=self.disp), k],
                (d5[:, self.antes] >= self.lim_aux[k]).float())
            add(self.res(self.B, f) + d5, torch.zeros(n4, device=self.disp), ya, ma, "R4",
                f)
        out = {k: torch.cat([p[k] for p in partes]) for k in partes[0]}
        out["pares_inv"] = (torch.cat(pares_inv) if pares_inv
                            else torch.zeros(0, 2, dtype=torch.long, device=self.disp))
        out["pares_sens"] = (torch.cat(pares_sens) if pares_sens
                             else torch.zeros(0, 2, dtype=torch.long, device=self.disp))
        out["t_centro"] = None
        return out

    def hora(self, c: Tensor) -> Tensor:
        """(B, W5) hora do dia (UTC) de cada intervalo da janela."""
        t = self.tsec[c][:, None] + (torch.arange(self.w5, device=self.disp) - self.antes)[None] * TAXA
        return ((t % 86400).float() / 3600.0)

    def lote_real(self, c: Tensor) -> dict:
        """So R0, para validacao e diagnostico."""
        p5 = self.res(self.P, c)
        ya, ma = self.aux_res(c)
        return dict(x=self.res(self.A, c), y=rotulo(p5[:, self.antes], self.lim), yaux=ya, maux=ma,
                    casa=self.casa_de[c], a=p5)


# --------------------------------------------------------------------------------------
# estatisticas por casa e entradas do modelo
# --------------------------------------------------------------------------------------


def stats_casa(x5: np.ndarray) -> dict:
    """Escalas da casa a partir SO do agregado de 5 min (disponivel em operacao)."""
    x5 = np.asarray(x5, float)
    s = max(float(np.median(x5)), 1.0)
    v = np.arcsinh(x5 / s)
    return dict(s=s, m=float(v.mean()), d=float(v.std() + 1e-6),
                p95=max(float(np.percentile(x5, 95)), 1.0))


def entradas(x5: Tensor, s: Tensor, m: Tensor, d: Tensor, pref: float,
             hora: Tensor | None = None, corte: float = 10.0) -> Tensor:
    """4 canais (§3.3): arcsinh(x/s_h) normalizado pela casa; x/P_ref; subidas e
    descidas em P_ref; cortados em +-10. `hora` (B, W5) em horas -> +2 canais."""
    v1 = (torch.asinh(x5 / s[:, None]) - m[:, None]) / d[:, None]
    dx = torch.zeros_like(x5)
    dx[:, 1:] = x5[:, 1:] - x5[:, :-1]
    v2 = (x5 / pref).clamp(-corte, corte)
    v3 = (F.relu(dx) / pref).clamp(max=corte)
    v4 = (F.relu(-dx) / pref).clamp(max=corte)
    ch = [v1.clamp(-corte, corte), v2, v3, v4]
    if hora is not None:
        ang = 2 * math.pi * hora / 24.0
        ch += [torch.sin(ang), torch.cos(ang)]
    return torch.stack(ch, 1)


# --------------------------------------------------------------------------------------
# backbone (Wang et al., 2017; CamAL usa o mesmo) e conjunto por kernel
# --------------------------------------------------------------------------------------


class BlocoRes(nn.Module):
    def __init__(self, cin: int, cout: int, k: int):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, padding=k // 2)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 5, padding=2)
        self.b2 = nn.BatchNorm1d(cout)
        self.c3 = nn.Conv1d(cout, cout, 3, padding=1)
        self.b3 = nn.BatchNorm1d(cout)
        self.atalho = nn.Sequential(nn.Conv1d(cin, cout, 1), nn.BatchNorm1d(cout))

    def forward(self, x: Tensor) -> Tensor:
        h = F.relu(self.b1(self.c1(x)))
        h = F.relu(self.b2(self.c2(h)))
        h = self.b3(self.c3(h))
        return F.relu(h + self.atalho(x))


class ResNet1D(nn.Module):
    """3 blocos residuais, GAP, linear -> logit do alvo + logits auxiliares.
    O kernel `k` e o da primeira convolucao de cada bloco (§3.3)."""

    def __init__(self, entrada: int = 4, k: int = 9, larguras=(32, 64, 64), n_aux: int = 0):
        super().__init__()
        cs = [entrada, *larguras]
        self.blocos = nn.Sequential(*[BlocoRes(cs[i], cs[i + 1], k) for i in range(3)])
        self.saida = nn.Linear(cs[-1], 1 + n_aux)
        self.k = k

    def forward(self, x: Tensor) -> Tensor:            # (B, C, L) -> (B, 1 + K)
        return self.saida(self.blocos(x).mean(-1))


def periodicidade(x5: Tensor) -> Tensor:
    """(B, L) -> (B, L//2 + L//2): autocorrelacao da primeira diferenca nos atrasos
    1..L//2 e espectro normalizado da janela sem a media (pos-quali, item 3). Invariante
    a escala: descreve o ritmo de liga-desliga, nao o nivel."""
    L = x5.shape[1]
    dx = x5[:, 1:] - x5[:, :-1]
    dx = dx - dx.mean(1, keepdim=True)
    den = (dx * dx).sum(1).clamp_min(1e-6)
    acf = torch.stack([(dx[:, k:] * dx[:, :-k]).sum(1) / den for k in range(1, L // 2 + 1)], 1)
    xc = x5 - x5.mean(1, keepdim=True)
    esp = torch.fft.rfft(xc, dim=1).abs()[:, 1:L // 2 + 1]
    esp = esp / esp.sum(1, keepdim=True).clamp_min(1e-6)
    return torch.cat([acf, esp], 1)


class ResNetGlob(ResNet1D):
    """ResNet1D + vetor global (periodicidade) por uma MLP pequena, concatenado ao GAP."""

    def __init__(self, entrada: int = 4, k: int = 9, n_glob: int = 48, oculto: int = 16,
                 larguras=(32, 64, 64), n_aux: int = 0):
        super().__init__(entrada, k, larguras, n_aux)
        self.glob = nn.Sequential(nn.Linear(n_glob, oculto), nn.ReLU())
        self.saida = nn.Linear(larguras[-1] + oculto, 1 + n_aux)

    def forward(self, x: Tensor, glob: Tensor) -> Tensor:
        return self.saida(torch.cat([self.blocos(x).mean(-1), self.glob(glob)], 1))


# --------------------------------------------------------------------------------------
# perda (§3.5)
# --------------------------------------------------------------------------------------


@dataclass
class ConfigPerda:
    lam_aux: float = 0.3
    lam_inv: float = 0.5
    lam_sens: float = 0.5
    m_inv: float = 0.5
    m_sens: float = 2.0
    g: float = 0.3


def perda_reco(z: Tensor, zaux: Tensor | None, lote: dict, cfg: ConfigPerda,
               usar_aux: bool, usar_inv: bool, usar_sens: bool) -> tuple[Tensor, dict]:
    """L = BCE + lam_aux * BCE_m(aux) + lam_inv * L_inv + lam_sens * L_sens."""
    bce = F.binary_cross_entropy_with_logits(z, lote["y"])
    total = bce
    partes = dict(bce=float(bce.detach()))
    if usar_aux and zaux is not None and zaux.shape[1]:
        m = lote["maux"]
        la = (F.binary_cross_entropy_with_logits(zaux, lote["yaux"], reduction="none") * m
              ).sum() / m.sum().clamp_min(1.0)
        total = total + cfg.lam_aux * la
        partes["aux"] = float(la.detach())
    if usar_inv and len(lote["pares_inv"]):
        i, j = lote["pares_inv"].T
        zi, zj = z[i], z[j]
        porta = (((torch.sigmoid(zi) - 0.5).abs() > cfg.g)
                 & ((torch.sigmoid(zj) - 0.5).abs() > cfg.g)).float()
        li = (F.relu((zi - zj).abs() - cfg.m_inv) * porta).sum() / porta.sum().clamp_min(1.0)
        total = total + cfg.lam_inv * li
        partes["inv"] = float(li.detach())
        partes["porta"] = float(porta.mean())
    if usar_sens and len(lote["pares_sens"]):
        i, j = lote["pares_sens"].T
        ls = F.relu(cfg.m_sens - (z[i] - z[j])).mean()
        total = total + cfg.lam_sens * ls
        partes["sens"] = float(ls.detach())
    return total, partes


# --------------------------------------------------------------------------------------
# pos-v3: fusao com o ramo de evidencia do v2 (canais de evidencia na ResNet)
# --------------------------------------------------------------------------------------


class EvidenciaSeq(nn.Module):
    """O ramo so-evidencia do v2 (src/evicon.py) devolvendo SEQUENCIAS, para virar
    canais da ResNet: detector de subida a+(t) e de descida a-(t), com restricao
    de sinal (uma queda nunca alimenta a+), e os estados acumulados para frente
    s(t) = clip(lam*s + a+ - a-) e para tras b(t) = clip(lam*b + a- - a+)."""

    def __init__(self, oculto: int = 16, meia_vida: float = 12.0, corte: float = 10.0):
        super().__init__()
        from evicon import Detector
        self.det_on = Detector(6, oculto)
        self.det_off = Detector(6, oculto)
        lam = 2 ** (-1 / meia_vida)
        self.lam_logit = nn.Parameter(torch.tensor(math.log(lam / (1 - lam))))
        self.corte = corte

    def forward(self, x5: Tensor, pref: float) -> dict:
        from evicon import ste_clip, vizinhanca
        d1 = torch.zeros_like(x5)
        d1[:, 1:] = x5[:, 1:] - x5[:, :-1]
        d2 = torch.zeros_like(x5)
        d2[:, 1:-1] = x5[:, 2:] - x5[:, :-2]
        d1 = (d1 / pref).clamp(-self.corte, self.corte)
        d2 = (d2 / pref).clamp(-self.corte, self.corte)
        a_on = self.det_on(vizinhanca(F.relu(d1), F.relu(d2)))
        a_off = self.det_off(vizinhanca(F.relu(-d1), F.relu(-d2)))
        lam = torch.sigmoid(self.lam_logit)
        T = x5.shape[1]
        s, b = torch.zeros_like(x5[:, 0]), torch.zeros_like(x5[:, 0])
        ss, bs = [], [None] * T
        for t in range(T):
            s = ste_clip(lam * s + a_on[:, t] - a_off[:, t])
            ss.append(s)
        for t in range(T - 1, -1, -1):
            bs[t] = b
            b = ste_clip(lam * b + a_off[:, t] - a_on[:, t])
        return dict(canais=torch.stack([a_on, a_off, torch.stack(ss, 1), torch.stack(bs, 1)], 1),
                    a_on=a_on, a_off=a_off)


class ResNetEvid(nn.Module):
    """ResNet1D sobre [4 canais do CRAFT (+ hora)] + [4 canais de evidencia]."""

    def __init__(self, entrada: int = 4, k: int = 9, n_aux: int = 0):
        super().__init__()
        self.evid = EvidenciaSeq()
        self.rede = ResNet1D(entrada + 4, k, n_aux=n_aux)
        self.ultimo = None

    def forward(self, xin: Tensor, x5: Tensor, pref: float) -> Tensor:
        ev = self.evid(x5, pref)
        self.ultimo = ev                      # para a perda auxiliar de transicoes
        return self.rede(torch.cat([xin, ev["canais"]], 1))


def perda_transicoes(ev: dict, yj: Tensor, peso_pos: float = 15.0) -> Tensor:
    """A perda auxiliar do v2 (perda_evicon): a+ deve acender nas subidas do rotulo
    e a- nas descidas, com peso nos (raros) positivos."""
    yp = yj[:, 1:] * (1 - yj[:, :-1])
    ym = yj[:, :-1] * (1 - yj[:, 1:])
    a_on = ev["a_on"][:, 1:].clamp(1e-6, 1 - 1e-6)
    a_off = ev["a_off"][:, 1:].clamp(1e-6, 1 - 1e-6)
    return (F.binary_cross_entropy(a_on, yp, weight=1 + (peso_pos - 1) * yp)
            + F.binary_cross_entropy(a_off, ym, weight=1 + (peso_pos - 1) * ym))
