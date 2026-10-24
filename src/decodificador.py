"""Pos-processamento das probabilidades por instante, com prior de duracao.

A rede devolve 288 probabilidades quase independentes. Um ciclo de maquina de
lavar e um bloco contiguo de 9 a 18 janelas, e o diagnostico mostrou que 61% do
tempo ligado roda abaixo de 300 W, onde o recall e 0,22-0,30, enquanto o pico de
aquecimento e detectado (recall 0,39-0,42). A ideia e usar a continuidade do
ciclo para o pico puxar o resto junto.

Dois decodificadores:

  morf     fechamento morfologico: preenche buracos menores que `fechar` e
           remove corridas menores que `abrir`. Bruto, mas quase de graca.
  viterbi  HMM de dois estados. A matriz de transicao vem da distribuicao de
           duracao observada nas casas de TREINO; a emissao e o proprio logit da
           rede, descontado o prior. E a versao principiada da mesma ideia.

Todo parametro e estimado nas casas de origem. A casa-alvo nao entra.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------------------
# Segmentacao: nada e decodificado atravessando um buraco no tempo
# --------------------------------------------------------------------------------------


def grupos_contiguos(idx: np.ndarray, t: np.ndarray, taxa_s: int = 300) -> list[slice]:
    """Fatias de `idx` em que as amostras sao consecutivas no indice E no tempo."""
    if len(idx) == 0:
        return []
    quebra = (np.diff(idx) != 1) | (np.diff(t[idx]) != taxa_s)
    limites = np.concatenate(([0], np.flatnonzero(quebra) + 1, [len(idx)]))
    return [slice(a, b) for a, b in zip(limites[:-1], limites[1:])]


# --------------------------------------------------------------------------------------
# Prior de duracao, medido nas casas de origem
# --------------------------------------------------------------------------------------


def duracoes(y: np.ndarray, fatias: list[slice]) -> tuple[np.ndarray, np.ndarray]:
    """Comprimentos das corridas de 1 e de 0, dentro de cada fatia."""
    ligado, desligado = [], []
    for f in fatias:
        s = y[f].astype(int)
        if len(s) == 0:
            continue
        troca = np.flatnonzero(np.diff(s)) + 1
        limites = np.concatenate(([0], troca, [len(s)]))
        for a, b in zip(limites[:-1], limites[1:]):
            (ligado if s[a] == 1 else desligado).append(b - a)
    return np.array(ligado), np.array(desligado)


def matriz_transicao(y: np.ndarray, fatias: list[slice]) -> np.ndarray:
    """A[i, j] = P(estado j no instante seguinte | estado i), contada nas fontes."""
    c = np.ones((2, 2))            # suavizacao de Laplace
    for f in fatias:
        s = y[f].astype(int)
        if len(s) < 2:
            continue
        np.add.at(c, (s[:-1], s[1:]), 1)
    return c / c.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------------------
# Decodificadores
# --------------------------------------------------------------------------------------


def _corridas(s: np.ndarray) -> list[tuple[int, int, int]]:
    troca = np.flatnonzero(np.diff(s)) + 1
    limites = np.concatenate(([0], troca, [len(s)]))
    return [(a, b, int(s[a])) for a, b in zip(limites[:-1], limites[1:])]


def morfologico(binario: np.ndarray, fatias: list[slice], fechar: int, abrir: int) -> np.ndarray:
    """Preenche buracos < `fechar` e apaga corridas de ligado < `abrir`."""
    y = binario.astype(np.int8).copy()
    for f in fatias:
        s = y[f]
        if len(s) == 0:
            continue
        for a, b, v in _corridas(s):
            if v == 0 and a > 0 and b < len(s) and (b - a) < fechar:
                s[a:b] = 1
        for a, b, v in _corridas(s):
            if v == 1 and (b - a) < abrir:
                s[a:b] = 0
        y[f] = s
    return y


def viterbi(emissao: np.ndarray, fatias: list[slice], A: np.ndarray) -> np.ndarray:
    """Caminho mais provavel. `emissao` e o log da razao a favor do estado 1."""
    logA = np.log(np.maximum(A, 1e-12))
    estacionaria = np.array([A[1, 0], A[0, 1]])
    estacionaria = estacionaria / estacionaria.sum()
    log_ini = np.log(np.maximum(estacionaria, 1e-12))

    saida = np.zeros(len(emissao), dtype=np.int8)
    for f in fatias:
        e = emissao[f]
        n = len(e)
        if n == 0:
            continue
        delta = np.empty((n, 2))
        psi = np.zeros((n, 2), dtype=np.int8)
        delta[0] = log_ini + np.array([0.0, e[0]])
        for i in range(1, n):
            for j in (0, 1):
                cand = delta[i - 1] + logA[:, j]
                psi[i, j] = int(np.argmax(cand))
                delta[i, j] = cand[psi[i, j]] + (e[i] if j == 1 else 0.0)
        caminho = np.empty(n, dtype=np.int8)
        caminho[-1] = int(np.argmax(delta[-1]))
        for i in range(n - 2, -1, -1):
            caminho[i] = psi[i + 1, caminho[i + 1]]
        saida[f] = caminho
    return saida


def emissao_logit(p: np.ndarray, prior: float, vies: float = 0.0) -> np.ndarray:
    """log P(x|ligado) - log P(x|desligado), a partir de P(ligado|x) da rede."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    prior = float(np.clip(prior, 1e-6, 1 - 1e-6))
    return np.log(p / (1 - p)) - np.log(prior / (1 - prior)) + vies
