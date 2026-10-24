"""Pares contrafactuais e teste de intervencao (docs/v2.md §3.1, §3.4, §9.A6).

  fundo       = agregado - aparelho, nas casas onde o aparelho e submedido
  positivo    = fundo + ativacao sorteada do banco, com escala de amplitude,
                cobrindo o centro da janela
  negativo    = fundo em janelas em que o aparelho real estava LIGADO no centro:
                mesmo contexto comportamental, aparelho ausente. E este braco
                que desmonta "agregado cai => TV liga".

Desvio declarado em relacao ao v2: a injecao e feita na serie de 5 min, nao na
resolucao nativa. Para a MEDIA de cada intervalo isso e exato (a media de uma
soma e a soma das medias) e o rotulo do protocolo e a media do aparelho >=
limiar, entao rotulo e agregado obedecem a mesma aritmetica do teste. O que se
perde e a fase da ativacao dentro do intervalo de 5 min: ela so entra alinhada a
grade.

Teste de intervencao (diagnostico, nunca selecao):

  delta_rem = escore(agregado) - escore(agregado - aparelho)   em janelas ligadas
  delta_add = escore(fundo + ativacao) - escore(fundo)          em janelas desligadas

Modelo fisicamente fundamentado -> deltas grandes; modelo de contexto -> ~0.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from dados_v2 import Banco


def injetar(fundo: Tensor, banco: Banco, gerador: torch.Generator,
            escala: tuple[float, float] = (0.6, 1.6), excluir_casa: Tensor | None = None
            ) -> tuple[Tensor, Tensor]:
    """Soma uma ativacao do banco a cada linha de `fundo` (B, L), cobrindo o centro.
    Devolve (agregado sintetico, aparelho sintetico)."""
    b, largura = fundo.shape
    dev = fundo.device
    j = torch.randint(0, banco.n, (b,), device=dev, generator=gerador)
    if excluir_casa is not None and banco.casa.unique().numel() > 1:
        # "outras casas": resorteia quem caiu na propria casa (algumas tentativas)
        for _ in range(8):
            mesma = banco.casa[j] == excluir_casa
            if not mesma.any():
                break
            j = torch.where(mesma, torch.randint(0, banco.n, (b,), device=dev,
                                                 generator=gerador), j)
    comp = banco.comp[j]
    # indice da ativacao que cai no centro: sorteado dentro do trecho ligado
    # (evita as bordas de 1 intervalo acrescentadas ao banco)
    u = torch.rand(b, device=dev, generator=gerador)
    o = 1 + (u * (comp - 2).clamp_min(1)).long()
    c0 = largura // 2
    pos = torch.arange(largura, device=dev)[None, :] - c0 + o[:, None]      # indice na ativacao
    ok = (pos >= 0) & (pos < comp[:, None])
    a = banco.ativ[j[:, None], pos.clamp(0, banco.lmax - 1)] * ok
    s = torch.empty(b, 1, device=dev).uniform_(*escala, generator=gerador)
    a = a * s
    return fundo + a, a


def lote_sintetico(lote: dict, banco: Banco, gerador: torch.Generator, frac: float,
                   escala=(0.6, 1.6), casa: Tensor | None = None) -> tuple[dict, dict | None]:
    """Troca uma fracao `frac` do lote por exemplos contrafactuais.

    Metade vira positivo (fundo + ativacao), metade negativo (fundo em janelas
    onde o aparelho estava ligado; se nao houver, fundo qualquer). Devolve o lote
    novo e o par (com, sem) dos positivos, para a perda L_par - com contexto
    IDENTICO nos dois lados.
    """
    if frac <= 0 or banco.n == 0:
        return lote, None
    b = lote["x"].shape[0]
    n = int(round(frac * b))
    if n == 0:
        return lote, None
    dev = lote["x"].device
    ordem = torch.randperm(b, device=dev, generator=gerador)
    ligados = ordem[lote["y"][ordem] > 0]
    n_pos = n // 2
    n_neg = n - n_pos
    neg = ligados[:n_neg]
    resto = ordem[~torch.isin(ordem, neg)]
    pos = resto[:n_pos]
    novo = {k: v.clone() for k, v in lote.items()}
    fundo = (lote["x"] - lote["app"]).clamp_min(0.0)
    lim = banco.limiar
    par = None
    if len(pos):
        cas = casa[lote["alvo"][pos]] if casa is not None else None
        x_s, a_s = injetar(fundo[pos], banco, gerador, escala, cas)
        c0 = x_s.shape[1] // 2
        novo["x"][pos] = x_s
        novo["app"][pos] = a_s
        novo["yj"][pos] = (a_s >= lim).float()
        novo["y"][pos] = (a_s[:, c0] >= lim).float()
        par = dict(idx=pos, com=x_s, sem=fundo[pos])
    if len(neg):
        novo["x"][neg] = fundo[neg]
        novo["app"][neg] = 0.0
        novo["yj"][neg] = 0.0
        novo["y"][neg] = 0.0
    return novo, par


@torch.no_grad()
def intervencao(escore, fonte, banco: Banco, n: int = 2000, semente: int = 0,
                escala=(0.85, 1.15), filtro_hora: tuple[int, int] | None = None) -> dict:
    """Delta_add e Delta_rem medios (em logits), sobre ate n janelas de `fonte`.
    `escore(lote) -> logits`. `filtro_hora=(h0, h1)` restringe o centro a essa faixa
    de hora (UTC da serie) - o teste de fase da maquina de lavar (dia x madrugada)."""
    g = torch.Generator(device=fonte.alvos.device).manual_seed(semente)
    y = fonte.y[fonte.alvos]
    hora = torch.from_numpy((fonte.t[fonte.alvos_np] % 86400) // 3600).to(y.device)
    ok_h = torch.ones_like(y, dtype=torch.bool)
    if filtro_hora is not None:
        h0, h1 = filtro_hora
        ok_h = (hora >= h0) & (hora < h1)
    out = {}
    lig = torch.nonzero((y > 0) & ok_h).squeeze(1)
    des = torch.nonzero((y == 0) & ok_h).squeeze(1)
    if len(lig):
        pos = lig[torch.randperm(len(lig), device=y.device, generator=g)[:n]]
        lote = fonte.lote(pos)
        z1 = escore(lote)
        sem = dict(lote)
        sem["x"] = (lote["x"] - lote["app"]).clamp_min(0.0)
        z0 = escore(sem)
        out["delta_rem"] = float((z1 - z0).mean())
        out["n_rem"] = int(len(pos))
    if len(des) and banco.n:
        pos = des[torch.randperm(len(des), device=y.device, generator=g)[:n]]
        lote = fonte.lote(pos)
        fundo = (lote["x"] - lote["app"]).clamp_min(0.0)
        x_s, a_s = injetar(fundo, banco, g, escala)
        c0 = x_s.shape[1] // 2
        valido = a_s[:, c0] >= banco.limiar
        com, sem = dict(lote), dict(lote)
        com["x"], sem["x"] = x_s, fundo
        d = escore(com) - escore(sem)
        out["delta_add"] = float(d[valido].mean()) if valido.any() else float("nan")
        out["n_add"] = int(valido.sum())
    return out


def gerador(semente: int, dispositivo) -> torch.Generator:
    return torch.Generator(device=dispositivo).manual_seed(int(semente) + 12345)


def resumo_banco(banco: Banco) -> str:
    if banco.n == 0:
        return "banco vazio"
    c = banco.comp.cpu().numpy()
    return (f"{banco.n} ativacoes, duracao mediana {np.median(c):.0f} intervalos, "
            f"{len(np.unique(banco.casa.cpu().numpy()))} casa(s)")
