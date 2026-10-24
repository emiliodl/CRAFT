"""Testes do v2: garantias do EviCon (E2.0, §9.C), decisao (§3.6) e vazamento (E0.3).

    python -m pytest tests/test_v2.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decisao import Decisor, em_prevalencia, limiar_f1_esperado, limiar_oraculo  # noqa: E402
from evicon import ConfigEviCon, EviCon, perda_evicon                             # noqa: E402
from saneamento import amostras_compartilhadas, auditoria_bordas, melhor_lag      # noqa: E402

torch.manual_seed(0)
B, WE, WC = 16, 48, 288


def modelo(**kw) -> EviCon:
    cfg = ConfigEviCon(w_e=WE, w_c=WC, **kw)
    m = EviCon(cfg).eval()
    # pesos aleatorios nao triviais nos detectores e no contexto
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return m


def janela_contexto(escala=1.0):
    return (torch.rand(B, WC) * 3000 * escala).abs()


def entradas(xe, escala_ctx=1.0):
    return dict(xe=xe, p_ref=torch.full((B,), 100.0), xc=janela_contexto(escala_ctx),
                p95=torch.full((B,), 2000.0))


# ---- 1: janela so com descidas -> a+ = 0 exatamente e a- > 0 ------------------------
def descidas():
    xe = torch.full((B, WE), 1000.0)
    for k in range(5, WE, 7):
        xe[:, k:] -= 50.0
    return xe


def test_1_descidas_nao_geram_ligar():
    m = modelo(contexto="restrito")
    out = m(**entradas(descidas()))
    assert torch.equal(out["a_on"], torch.zeros_like(out["a_on"]))
    assert (out["a_off"] > 0).any()


# ---- 2: sem transicao -> e = 0 e |z - b0| <= c, para contexto de qualquer escala ----
@pytest.mark.parametrize("escala", [0.0, 1.0, 100.0])
def test_2_sem_evidencia_contexto_limitado(escala):
    m = modelo(contexto="restrito", c=0.7)
    out = m(**entradas(torch.full((B, WE), 800.0), escala))
    assert torch.equal(out["e"], torch.zeros_like(out["e"]))
    assert (out["z"] - m.b0).abs().max() <= 0.7 + 1e-6


# ---- 3: acumulador com lambda = 1 ---------------------------------------------------
def test_3_acumulador():
    m = modelo(contexto="nenhum")
    ev = m.evidencia
    ev.lam_logit.data.fill_(50.0)            # lambda = 1
    t0 = ev.t0
    a_on = torch.zeros(2, WE)
    a_off = torch.zeros(2, WE)
    a_on[0, t0 - 5], a_off[0, t0 + 6] = 1.0, 1.0      # liga antes, desliga depois
    a_on[1, t0 + 3], a_off[1, t0 + 9] = 1.0, 1.0      # sessao toda depois do centro
    ev.transicoes = lambda x, p, l=None: (a_on, a_off)
    out = ev(torch.zeros(2, WE), torch.ones(2))
    assert out["s"][0] == 1 and out["b"][0] == 1
    assert out["s"][1] == 0 and out["b"][1] == 0


# ---- 4: resto aditivo do contexto limitado por c ------------------------------------
def test_4_resto_aditivo():
    m = modelo(contexto="restrito", c=0.5, gama=0.5)
    xe = torch.cumsum(torch.randn(B, WE) * 300, 1).abs()
    out = m(**entradas(xe, 5.0))
    resto = out["z"] - m.b0 - (1 + out["mod"]) * out["e"]
    assert resto.abs().max() <= 0.5 + 1e-5
    assert out["mod"].abs().max() <= 0.5 + 1e-6


# ---- 5: gradientes finitos ----------------------------------------------------------
def test_5_gradientes_finitos():
    m = modelo(contexto="restrito").train()
    xe = torch.cumsum(torch.randn(B, WE) * 300, 1).abs()
    out = m(**entradas(xe))
    y = (torch.rand(B) > 0.5).float()
    yj = (torch.rand(B, WE) > 0.5).float()
    perda, _ = perda_evicon(out, y, yj, m.cfg, pot_alvo=torch.rand(B))
    perda.backward()
    for nome, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), nome


# ---- 6: sem restricao de sinal (A3) o teste 1 deve FALHAR ---------------------------
def test_6_a3_quebra_a_garantia():
    torch.manual_seed(3)
    m = modelo(contexto="nenhum", sinal=False)
    out = m(**entradas(descidas()))
    assert (out["a_on"] > 0).any(), "A3 deveria poder tratar queda como ligar"


# ---- G2: queda nao gera estado ligado depois dela ------------------------------------
def test_g2_queda_nao_liga():
    m = modelo(contexto="nenhum")
    xe = torch.full((B, WE), 1000.0)
    xe[:, WE // 2 - 3:] -= 120.0              # o agregado CAI antes do centro
    out = m(**entradas(xe))
    assert torch.equal(out["s"], torch.zeros_like(out["s"]))


# ---- decisao --------------------------------------------------------------------------
def test_em_recupera_prevalencia():
    rng = np.random.default_rng(0)
    def amostra(n, pi):
        y = rng.random(n) < pi
        z = np.where(y, rng.normal(2.0, 1, n), rng.normal(0, 1, n))
        return y.astype(int), z
    ys, zs = amostra(40000, 0.045)
    yt, zt = amostra(40000, 0.074)
    from decisao import ajustar_bcts
    cal, _, _ = ajustar_bcts(zs, ys)
    pi, _ = em_prevalencia(cal(zt), cal(zs).mean())
    assert abs(pi - yt.mean()) < 0.01


def test_limiar_f1_esperado_trivial_e_metade():
    q = np.full(1000, 0.7)
    lim, ef1 = limiar_f1_esperado(q)
    assert abs(ef1 - 2 * 0.7 / 1.7) < 1e-9          # k = N reproduz o trivial
    rng = np.random.default_rng(1)
    q = rng.beta(0.3, 3, 20000)
    lim, ef1 = limiar_f1_esperado(q)
    assert abs(lim - ef1 / 2) < 0.02                # Lipton et al.


def test_trava_nao_informativo():
    rng = np.random.default_rng(2)
    y_cal = (rng.random(5000) < 0.69).astype(int)
    p_cal = rng.random(5000)
    dec = Decisor(p_cal, y_cal, auc_retida=0.5)
    y = (rng.random(5000) < 0.69).astype(int)
    d = dec.decidir(rng.random(5000))
    assert d["log"]["sem_deteccao"] and d["c"].all()
    f1 = 2 * (d["c"] & (y == 1)).sum() / (d["c"].sum() + y.sum())
    assert abs(f1 - 2 * y.mean() / (1 + y.mean())) < 1e-9


def test_limiar_oraculo_exato():
    rng = np.random.default_rng(3)
    y = (rng.random(3000) < 0.1).astype(int)
    p = np.clip(y * 0.3 + rng.random(3000) * 0.8, 0, 1)
    lim, f = limiar_oraculo(p, y)
    pred = p >= lim
    assert abs(f - 2 * (pred & (y == 1)).sum() / (pred.sum() + y.sum())) < 1e-12


# ---- saneamento -----------------------------------------------------------------------
def test_auditoria_sintetica():
    rng = np.random.default_rng(4)
    n = 60000
    app = np.zeros(n)
    for i in rng.choice(np.arange(100, n - 200), 150, replace=False):
        app[i:i + 60] = 75.0
    fundo = 300 + np.cumsum(rng.normal(0, 3, n)).clip(-250, 2000)
    lag, c, c0 = melhor_lag(fundo + app, app, 40)
    assert lag == 0 and c0 > 0
    atras = np.r_[np.zeros(15), app[:-15]]
    assert melhor_lag(fundo + atras, app, 40)[0] == 15
    tab, nb = auditoria_bordas(fundo + app, app, 10.0, 20)
    res = tab[(tab.lag >= 1) & (tab.lag <= 5)].residuo.mean()
    assert abs(res) < 15 and nb > 50


def test_vazamento_divisao_aleatoria_falha_e_temporal_passa():
    alvos = np.arange(0, 100000)
    rng = np.random.default_rng(5)
    va = rng.random(len(alvos)) < 0.15
    assert amostras_compartilhadas(alvos[~va], alvos[va], 144, 143) > 0
    corte = 85000
    tr, vv = alvos[alvos < corte - 288], alvos[alvos >= corte]
    assert amostras_compartilhadas(tr, vv, 144, 143) == 0
