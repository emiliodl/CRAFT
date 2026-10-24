"""Testes do v3 (docs/v3.md §8), escritos antes de treinar.

    python -m pytest tests/test_v3.py -q

1  R3 preserva o rotulo do alvo ponto a ponto depois da reamostragem
2  R2 zera o rotulo quando nao ha outra ativacao do alvo na janela
3  recompor e reamostrar comutam (antes do corte em zero)
4  nenhuma amostra de casa de validacao/teste aparece nos bancos
5  media de postos invariante a transformacoes monotonas por membro
6  decisao por quantil preve exatamente round(r * N) positivos
(7, discriminador real x recomposto, e de integracao: src/discriminador_v3.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantil import decidir_quantil, media_postos                    # noqa: E402
from reco import (ANTES, SUB, W5, CasaFina, Gerador, aux_de, reamostrar,  # noqa: E402
                  rotulo)

DISP = torch.device("cpu")


def casa_sintetica(casa: int, dias: int = 6, semente: int = 0) -> CasaFina:
    """Agregado = fundo aleatorio + TV com sessoes + um computador (distrator)."""
    rng = np.random.default_rng(semente)
    n = dias * 2880
    t0 = 1_400_000_000 - 1_400_000_000 % 86400
    fundo = 200 + 50 * rng.random(n)
    tv = np.zeros(n, np.float32)
    for ini in rng.integers(0, n - 400, 12):
        tv[ini:ini + rng.integers(60, 300)] = 90 + 10 * rng.random()
    pc = np.zeros(n, np.float32)
    for ini in rng.integers(0, n - 400, 12):
        pc[ini:ini + rng.integers(60, 300)] = 110
    aux = aux_de("television")
    Q = np.zeros((len(aux), n), np.float32)
    Q[aux.index("computer")] = pc
    qm = np.zeros(len(aux), bool)
    qm[aux.index("computer")] = True
    A = (fundo + tv + pc).astype(np.float32)
    W = rng.integers(3, 6, n).astype(np.float32)
    c = CasaFina(casa=casa, t0=t0, A=A, P=tv, Pcat=tv.copy(), Q=Q, qmask=qm, W=W,
                 ok=np.ones(n, bool))
    c.periodo = dict(treino=np.ones(n, bool), val=np.ones(n, bool))
    return c


@pytest.fixture(scope="module")
def ger():
    casas = [casa_sintetica(1, semente=1), casa_sintetica(2, semente=2)]
    return Gerador(casas, "television", "treino", DISP)


def test_1_r3_preserva_rotulo(ger):
    g = torch.Generator().manual_seed(0)
    base = ger.centros[torch.randint(0, len(ger.centros), (64,), generator=g)]
    lote = ger.lote(base, g)
    i, j = lote["pares_inv"].T
    assert len(i) > 0
    # o rotulo de R3 e o do protocolo sobre P_h reamostrado com a contagem de h
    assert torch.equal(lote["y"][i], lote["y"][j])
    assert torch.equal(lote["a"][i], lote["a"][j])
    # e bate com o rotulo recalculado do zero
    y_ref = rotulo(ger.res(ger.P, base[:len(i)])[:, ANTES], ger.lim)
    assert torch.equal(lote["y"][j], y_ref)


def test_2_r2_zera_rotulo(ger):
    g = torch.Generator().manual_seed(1)
    base = ger.centros[torch.randint(0, len(ger.centros), (64,), generator=g)]
    lote = ger.lote(base, g)
    r2 = lote["op"] == 2
    assert r2.any()
    assert (lote["y"][r2] == 0).all()
    # o parceiro R0 de cada R2 estava ligado no centro
    i, j = lote["pares_sens"].T
    par_r2 = lote["op"][j] == 2
    assert (lote["y"][i[par_r2]] == 1).all()
    # e sobra zero do alvo na janela removida
    c = ger.centros[ger.idx_on[:8]]
    assert torch.allclose(ger.res(ger.A, c) - ger.res(ger.P, c), ger.res(ger.B, c), atol=1e-3)


def test_3_recompor_e_reamostrar_comutam():
    torch.manual_seed(0)
    b = torch.rand(5, W5 * SUB) * 1000
    a = torch.rand(5, W5 * SUB) * 200 - 50            # inclui valores negativos: antes do corte
    w = torch.randint(1, 6, (5, W5 * SUB)).float()
    assert torch.allclose(reamostrar(b + a, w), reamostrar(b, w) + reamostrar(a, w), atol=1e-3)


def test_4_bancos_sem_casas_retidas():
    treino = [casa_sintetica(8, semente=3), casa_sintetica(1, semente=4)]
    g = Gerador(treino, "television", "treino", DISP)
    retidas = {3, 7, 20, 101, 102, 105}
    ids = set(g.casas_ids)
    assert ids.isdisjoint(retidas)
    casas_banco = {g.casas_ids[int(h)] for h in g.casa_de[g.ativ["ini"]].unique()}
    casas_fundo = {g.casas_ids[int(h)] for h in g.casa_de[g.centros[g.idx_limpo]].unique()}
    casas_dist = {g.casas_ids[int(h)] for h in g.casa_de[g.dist["ini"]].unique()}
    assert (casas_banco | casas_fundo | casas_dist) <= ids


def test_4b_periodo_respeitado():
    c = casa_sintetica(8, semente=5)
    n = len(c.A)
    tr = np.zeros(n, bool)
    tr[: n // 2] = True
    c.periodo = dict(treino=tr, val=~tr)
    g = Gerador([c], "television", "treino", DISP)
    fim = g.centros + (W5 - ANTES) * SUB
    assert int(fim.max()) <= n // 2
    assert int(g.ativ["fim"].max()) <= n // 2


def test_5_media_postos_monotona():
    rng = np.random.default_rng(0)
    m1, m2 = rng.normal(size=500), rng.normal(size=500)
    a = media_postos([m1, m2])
    b = media_postos([np.exp(3 * m1) + 7, 1 / (1 + np.exp(-m2))])
    assert np.allclose(a, b)


@pytest.mark.parametrize("r", [0.0, 0.013, 0.2, 0.5, 0.94])
def test_6_quantil_conta_exata(r):
    posto = np.repeat(np.arange(50), 10).astype(float)        # muitos empates
    pred = decidir_quantil(posto, r)
    assert pred.sum() == int(round(r * len(posto)))


def test_6b_regime_trivial():
    assert decidir_quantil(np.arange(100.0), 0.96).all()


# --------------------------------------------------------------------------------------
# pos-quali: R1 deformada (item 2) e trem de ciclos + periodicidade (item 3)
# --------------------------------------------------------------------------------------


def test_8_deformacao_neutra_reproduz_injecao():
    casas = [casa_sintetica(1, semente=1), casa_sintetica(2, semente=2)]
    neutra = dict(r=(1.0, 1.0), ganho=(1.0, 1.0), p_emenda=0.0)
    gd = Gerador(casas, "television", "treino", DISP, escala=(1.0, 1.0), deform=neutra)
    a = gd.injetar(gd.ativ, gd._serie_alvo, 32, torch.Generator().manual_seed(7))[0]
    b = gd.injetar_deformado(32, torch.Generator().manual_seed(7))
    assert torch.allclose(a, b, atol=1e-4)


def test_9_r1_deformada_rotulo_recalculado():
    casas = [casa_sintetica(1, semente=1), casa_sintetica(2, semente=2)]
    gd = Gerador(casas, "television", "treino", DISP, deform={})
    g = torch.Generator().manual_seed(3)
    lote = gd.lote(gd.centros[torch.randint(0, len(gd.centros), (64,), generator=g)], g)
    r1 = lote["op"] == 1
    assert r1.any()
    a = lote["a"][r1]
    assert torch.isfinite(a).all() and (a >= 0).all()
    assert torch.equal(lote["y"][r1], rotulo(a[:, ANTES], gd.lim))


def test_10_trem_cobre_a_janela():
    casas = [casa_sintetica(1, semente=1), casa_sintetica(2, semente=2)]
    gt = Gerador(casas, "television", "treino", DISP, trem=dict(r=(0.3, 0.3), razao=(1.0, 1.0)))
    a5 = gt.injetar_trem(64, torch.Generator().manual_seed(5))
    assert a5.shape == (64, W5) and torch.isfinite(a5).all() and (a5 >= 0).all()
    # ciclos curtos (r=0,3) com razao 1: ligado ~metade do tempo, em mais de um trecho
    lig = (a5 > 1.0).float()
    assert 0.2 < float(lig.mean()) < 0.8
    assert float((lig[:, 1:] != lig[:, :-1]).sum(1).float().mean()) >= 2


def test_11_periodicidade_invariante_a_escala():
    from reco import periodicidade
    torch.manual_seed(0)
    x = 200 + 80 * (torch.sin(torch.arange(W5).float() / 2)[None] > 0).float() \
        + torch.rand(4, W5)
    p1, p2 = periodicidade(x), periodicidade(3 * x)
    assert p1.shape == (4, 2 * (W5 // 2))
    assert torch.allclose(p1, p2, atol=1e-4)
