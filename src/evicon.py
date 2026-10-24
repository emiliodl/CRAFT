"""EviCon: evidencia fisica com contexto restrito (docs/v2.md §2 e §9.C).

Dois ramos separados pela informacao que cada um pode ver.

  evidencia  so degraus do agregado, em watts / P_ref, por detectores que por
             construcao nao podem tratar uma queda como "ligou", e um
             acumulador de estado bidirecional que transforma bordas em
             "ligado agora". Algumas centenas de parametros.
  contexto   janela longa com o centro mascarado, gargalo de 2-4 dimensoes, e
             so pode modular a evidencia (fator em [1-gama, 1+gama]) ou
             deslocar o logit em no maximo +-c.

      z = b0 + (1 + gama*tanh m(u)) * e + c*tanh a(u),      e >= 0

Garantias por construcao (v2 §2.6): sem transicao no entorno, e = 0 e o
contexto sozinho nunca leva o logit alem de b0 + c (G1); uma queda so produz
a-, entao "queda => ligado depois" nao e representavel (G2); z e nao
decrescente em s, b, l com o contexto fixo (G3); mudar so o contexto altera z
em no maximo 2*gama*e + 2c (G4).

Ablacoes (v2 §2.12), todas pelo mesmo construtor:

  A1  acumulador=False, contexto="nenhum"   leitura so no centro
  A2  acumulador=True,  contexto="nenhum"
  A3  A2 com sinal=False                    detectores veem degraus com sinal
  A4  A2 + contexto="restrito"              o EviCon
  A5  A2 + contexto="livre"                 sem mascara, sem gargalo, aditivo livre

Decisoes de implementacao que o v2 deixa em aberto, declaradas:

  - o clip do acumulador e um clip com gradiente reto (straight-through): o
    valor e exatamente clip(., 0, 1), mas o gradiente passa como identidade.
    O v2 preve trocar por versao suave se o gradiente nulo atrapalhar; o
    straight-through mantem o valor exato, que os testes de propriedade exigem.
  - A1 ("leitura so no centro"): s = a+ no centro, b = a- no intervalo seguinte.
  - o contexto recebe a janela bruta em W e calcula as vistas (arcsinh(x/50) e
    x/p95 da casa) dentro do modelo, para que pares contrafactuais possam ser
    montados em W e passar pela mesma conta.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class ConfigEviCon:
    w_e: int = 48                  # janela de evidencia (amostras), centrada
    w_c: int = 288                 # janela de contexto (amostras), centrada
    mascara: int = 12              # meia-largura M do centro mascarado no contexto
    meia_vida: float = 24.0        # H em intervalos; lambda inicial = 2^(-1/H)
    gama: float = 0.5
    c: float = 1.0
    k: int = 2                     # dimensao do gargalo
    beta: float = 1e-3             # peso do KL
    oculto: int = 16               # unidades dos detectores
    acumulador: bool = True
    sinal: bool = True             # restricao de sinal nos detectores
    contexto: str = "restrito"     # nenhum | restrito | livre
    hora: bool = False             # seno/cosseno da hora no contexto (so intra)
    corte: float = 10.0            # diferencas cortadas em +-corte * P_ref
    prior: float = 0.05            # inicializa b0 = logit(prior)


def ste_clip(x: Tensor, lo: float = 0.0, hi: float = 1.0) -> Tensor:
    """clip exato no valor, identidade no gradiente."""
    return x + (x.clamp(lo, hi) - x).detach()


class Detector(nn.Module):
    """a = relu(tanh(h(p) - h(0))), h um MLP pequeno. Com p = 0, a = 0 exatamente."""

    def __init__(self, entrada: int = 6, oculto: int = 16):
        super().__init__()
        self.h = nn.Sequential(nn.Linear(entrada, oculto), nn.GELU(), nn.Linear(oculto, 1))

    def forward(self, p: Tensor) -> Tensor:                      # (B, T, 6) -> (B, T)
        zero = self.h(torch.zeros(1, p.shape[-1], device=p.device, dtype=p.dtype))
        return F.relu(torch.tanh(self.h(p) - zero)).squeeze(-1)


def vizinhanca(d1: Tensor, d2: Tensor) -> Tensor:
    """(B, T) x2 -> (B, T, 6): d1 e d2 em t-1, t, t+1, com zero nas bordas."""
    def desloca(x, k):
        if k == 0:
            return x
        z = torch.zeros_like(x[:, :abs(k)])
        return torch.cat([z, x[:, :-k]], 1) if k > 0 else torch.cat([x[:, -k:], z], 1)
    return torch.stack([desloca(d, k) for d in (d1, d2) for k in (1, 0, -1)], -1)


class Evidencia(nn.Module):
    def __init__(self, cfg: ConfigEviCon):
        super().__init__()
        self.cfg = cfg
        self.t0 = cfg.w_e // 2
        self.det_on = Detector(6, cfg.oculto)
        self.det_off = Detector(6, cfg.oculto)
        lam = 2 ** (-1 / cfg.meia_vida)
        self.lam_logit = nn.Parameter(torch.tensor(math.log(lam / (1 - lam))))
        self.k_l = nn.Parameter(torch.tensor(1.0))
        self.theta_l = nn.Parameter(torch.tensor(0.5))
        self.w = nn.Parameter(torch.full((4,), -1.0))            # softplus(-1) ~ 0.31
        self.w.data[0] = self.w.data[1] = 0.5

    def diferencas(self, x: Tensor, p_ref: Tensor, lacuna: Tensor | None):
        """d1[t] = x[t]-x[t-1], d2[t] = x[t+1]-x[t-1], em P_ref, cortadas e zeradas em lacuna."""
        pr = p_ref.view(-1, 1)
        d1 = torch.zeros_like(x)
        d1[:, 1:] = x[:, 1:] - x[:, :-1]
        d2 = torch.zeros_like(x)
        d2[:, 1:-1] = x[:, 2:] - x[:, :-2]
        d1 = (d1 / pr).clamp(-self.cfg.corte, self.cfg.corte)
        d2 = (d2 / pr).clamp(-self.cfg.corte, self.cfg.corte)
        if lacuna is not None:
            # uma diferenca que toca um intervalo em lacuna nao e evidencia de nada
            lac = lacuna > 0
            l1 = lac.clone()
            l1[:, 1:] |= lac[:, :-1]
            l2 = l1.clone()
            l2[:, :-1] |= lac[:, 1:]
            d1, d2 = d1.masked_fill(l1, 0.0), d2.masked_fill(l2, 0.0)
        return d1, d2

    def transicoes(self, x: Tensor, p_ref: Tensor, lacuna: Tensor | None = None):
        d1, d2 = self.diferencas(x, p_ref, lacuna)
        if self.cfg.sinal:
            p_on = vizinhanca(F.relu(d1), F.relu(d2))
            p_off = vizinhanca(F.relu(-d1), F.relu(-d2))
        else:                                    # A3: os dois detectores veem o sinal
            p_on = p_off = vizinhanca(d1, d2)
        return self.det_on(p_on), self.det_off(p_off)

    def forward(self, x: Tensor, p_ref: Tensor, lacuna: Tensor | None = None) -> dict:
        a_on, a_off = self.transicoes(x, p_ref, lacuna)
        t0 = self.t0
        if self.cfg.acumulador:
            lam = torch.sigmoid(self.lam_logit)
            s = torch.zeros_like(x[:, 0])
            for t in range(0, t0 + 1):
                s = ste_clip(lam * s + a_on[:, t] - a_off[:, t])
            b = torch.zeros_like(x[:, 0])
            for t in range(x.shape[1] - 1, t0, -1):      # termina em b_{t0}
                b = ste_clip(lam * b + a_off[:, t] - a_on[:, t])
        else:                                    # A1: leitura so no centro
            s, b = a_on[:, t0], a_off[:, t0 + 1]
        # nivel no centro acima de uma linha de base robusta da janela
        base = torch.quantile(x, 0.1, dim=1)
        r = (x[:, t0] - base) / p_ref
        ell = torch.sigmoid(F.softplus(self.k_l) * (r - self.theta_l))
        w = F.softplus(self.w)
        e = w[0] * s + w[1] * b + w[2] * s * b + w[3] * ell * torch.maximum(s, b)
        return dict(e=e, s=s, b=b, ell=ell, r=r, a_on=a_on, a_off=a_off)


class Contexto(nn.Module):
    """CNN pequena sobre a janela longa (com o centro mascarado), gargalo estocastico."""

    def __init__(self, cfg: ConfigEviCon, canais: int = 3, largura: int = 32):
        super().__init__()
        self.cfg = cfg
        livre = cfg.contexto == "livre"
        self.k = 32 if livre else cfg.k
        extra = 2 if cfg.hora else 0
        self.cnn = nn.Sequential(
            nn.Conv1d(canais, largura, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(largura, largura, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(largura, largura, 5, stride=2, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten())
        self.mu = nn.Linear(4 * largura + extra, self.k)
        self.logvar = nn.Linear(4 * largura + extra, self.k)
        self.m = nn.Linear(self.k, 1)
        self.a = nn.Linear(self.k, 1)
        for lin in (self.m, self.a):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, xc: Tensor, hora: Tensor | None = None):
        h = self.cnn(xc)
        if self.cfg.hora and hora is not None:
            h = torch.cat([h, hora], 1)
        mu = self.mu(h)
        if self.cfg.contexto == "livre":
            return mu, torch.zeros((), device=xc.device)
        logvar = self.logvar(h).clamp(-8, 4)
        u = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(1).mean()
        return u, kl


class EviCon(nn.Module):
    def __init__(self, cfg: ConfigEviCon):
        super().__init__()
        self.cfg = cfg
        self.evidencia = Evidencia(cfg)
        p = min(max(cfg.prior, 1e-3), 1 - 1e-3)
        self.b0 = nn.Parameter(torch.tensor(math.log(p / (1 - p))))
        self.contexto = Contexto(cfg) if cfg.contexto != "nenhum" else None
        # gama e c sao hiperparametros fixos; `rampa` em [0,1] implementa o curriculo
        self.register_buffer("rampa", torch.tensor(1.0))
        # cabeca auxiliar L_pot: potencia do aparelho no centro a partir da evidencia
        self.pot = nn.Linear(6, 1)

    # ---- contexto: vistas calculadas dentro do modelo a partir de W ----
    def entrada_contexto(self, xc: Tensor, p95: Tensor, lacuna_c: Tensor | None) -> Tensor:
        v1 = torch.asinh(xc / 50.0)
        v1 = (v1 - 5.0) / 1.5                          # escala fixa, ~ padronizada
        v2 = xc / p95.view(-1, 1) - 0.5
        masc = torch.zeros_like(xc)
        if self.cfg.contexto == "restrito":
            c0, m = xc.shape[1] // 2, self.cfg.mascara
            masc[:, c0 - m:c0 + m + 1] = 1.0
        if lacuna_c is not None:
            masc = torch.maximum(masc, (lacuna_c > 0).float())
        v1, v2 = v1 * (1 - masc), v2 * (1 - masc)
        return torch.stack([v1, v2, masc], 1)

    def forward(self, xe: Tensor, p_ref: Tensor, xc: Tensor | None = None,
                p95: Tensor | None = None, lacuna_e: Tensor | None = None,
                lacuna_c: Tensor | None = None, hora: Tensor | None = None) -> dict:
        ev = self.evidencia(xe, p_ref, lacuna_e)
        e = ev["e"]
        mod = torch.zeros_like(e)
        desl = torch.zeros_like(e)
        kl = torch.zeros((), device=e.device)
        if self.contexto is not None:
            u, kl = self.contexto(self.entrada_contexto(xc, p95, lacuna_c), hora)
            if self.cfg.contexto == "livre":
                mod = self.cfg.gama * torch.tanh(self.contexto.m(u).squeeze(1))
                desl = self.contexto.a(u).squeeze(1)                   # aditivo sem limite
            else:
                g, c = self.cfg.gama * self.rampa, self.cfg.c * self.rampa
                mod = g * torch.tanh(self.contexto.m(u).squeeze(1))
                desl = c * torch.tanh(self.contexto.a(u).squeeze(1))
        z = self.b0 + (1 + mod) * e + desl
        t0 = self.evidencia.t0
        feats = torch.stack([ev["s"], ev["b"], ev["ell"], ev["a_on"][:, t0],
                             ev["a_off"][:, t0], ev["r"].clamp(-5, 5)], 1)
        return dict(z=z, kl=kl, mod=mod, desl=desl, pot=self.pot(feats).squeeze(1), **ev)


def perda_evicon(saida: dict, y: Tensor, y_janela: Tensor, cfg: ConfigEviCon,
                 lam_ev: float = 0.3, peso_pos: float = 15.0, lam_p: float = 0.1,
                 pot_alvo: Tensor | None = None) -> tuple[Tensor, dict]:
    """L = BCE(z, y) + lam_ev*[BCE_w(a+, y+) + BCE_w(a-, y-)] + lam_p*L_pot + beta*KL.
    L_par (pares contrafactuais) e somada fora, porque precisa de dois passes."""
    bce = F.binary_cross_entropy_with_logits(saida["z"], y)
    yp = y_janela[:, 1:] * (1 - y_janela[:, :-1])            # transicao para ligado em t
    ym = y_janela[:, :-1] * (1 - y_janela[:, 1:])            # desligou em t
    a_on = saida["a_on"][:, 1:].clamp(1e-6, 1 - 1e-6)
    a_off = saida["a_off"][:, 1:].clamp(1e-6, 1 - 1e-6)
    l_on = F.binary_cross_entropy(a_on, yp, weight=1 + (peso_pos - 1) * yp)
    l_off = F.binary_cross_entropy(a_off, ym, weight=1 + (peso_pos - 1) * ym)
    total = bce + lam_ev * (l_on + l_off) + cfg.beta * saida["kl"]
    partes = dict(bce=float(bce.detach()), trans=float((l_on + l_off).detach()),
                  kl=float(saida["kl"].detach()))
    if pot_alvo is not None and lam_p > 0:
        l_p = F.smooth_l1_loss(saida["pot"], pot_alvo)
        total = total + lam_p * l_p
        partes["pot"] = float(l_p.detach())
    return total, partes
