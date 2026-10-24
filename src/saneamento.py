"""Estagio 0 do v2 (docs/v2.md §3.0, §9.A1-A2): saneamento de dados e de divisao.

Tres suspeitas a eliminar antes de modelar (v2 §1.3):

  (a) consistencia fisica nas bordas: num agregado aditivo e alinhado, quando o
      aparelho liga o agregado sobe o que o aparelho subiu. Os numeros do
      registro (freezer -5 W, TV -75 W) nao fecham com isso. `auditoria_bordas`
      e `melhor_lag` separam fisico, desalinhamento e confusao comportamental.
  (b) lacunas zeradas: a versao limpa do REFIT zera lacunas longas, e a 5 min
      tres minutos de zero derrubam a media do intervalo em 60% - uma "queda do
      agregado" que nao existiu. `fracao_lacuna` mede isso na resolucao nativa.
  (c) vazamento na divisao: `test_sem_vazamento` verifica sobre AMOSTRAS BRUTAS
      que nenhuma janela de treino compartilha instante com uma de validacao.
      Teria pego o bug da §9 (interseccao de 100%).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dados import Serie


# --------------------------------------------------------------------------------------
# A1 - auditoria de bordas e lag (resolucao nativa)
# --------------------------------------------------------------------------------------


def auditoria_bordas(agg: np.ndarray, app: np.ndarray, limiar: float,
                     meia_janela: int = 20, excluir: np.ndarray | None = None
                     ) -> tuple[pd.DataFrame, int]:
    """Media disparada por evento nas bordas de subida do aparelho.

    Devolve a tabela por lag com agregado, aparelho, residuo = agregado - aparelho
    e o erro-padrao do residuo (linha de base = media antes da borda), e o numero
    de bordas. `excluir` (mascara por amostra) descarta bordas com qualquer
    amostra marcada na vizinhanca de +-meia_janela. Leitura: residuo ~ 0 com degrau no lag 0 -> fisico e alinhado;
    residuo negativo espalhado -> cargas que desligam junto; degrau do agregado
    fora do lag 0 -> desalinhamento entre medidor e plugue.
    """
    a, p = np.asarray(agg, float), np.asarray(app, float)
    bordas = bordas_de_sessao(p, limiar, meia_janela)
    lags = np.arange(-meia_janela, meia_janela + 1)
    if excluir is not None and len(bordas):
        bordas = bordas[~excluir[bordas[:, None] + lags[None, :]].any(1)]
    if len(bordas) < 2:
        vazio = np.full(len(lags), np.nan)
        return pd.DataFrame({"lag": lags, "agregado": vazio, "aparelho": vazio,
                             "residuo": vazio, "ep_residuo": vazio}), len(bordas)
    idx = bordas[:, None] + lags[None, :]
    A, P = a[idx], p[idx]
    A = A - A[:, :meia_janela].mean(1, keepdims=True)
    P = P - P[:, :meia_janela].mean(1, keepdims=True)
    R = A - P
    ok = np.isfinite(R).all(1)
    A, P, R = A[ok], P[ok], R[ok]
    return pd.DataFrame({"lag": lags, "agregado": A.mean(0), "aparelho": P.mean(0),
                         "residuo": R.mean(0),
                         "ep_residuo": R.std(0, ddof=1) / np.sqrt(len(R))}), int(ok.sum())


def bordas_de_sessao(p: np.ndarray, limiar: float, meia_janela: int) -> np.ndarray:
    """Inicios de SESSAO: cruzamento para cima com o aparelho desligado em toda a
    meia-janela anterior. Na resolucao nativa o motor da maquina de lavar cruza o
    limiar milhares de vezes dentro de um ciclo; sem essa exigencia a "borda"
    media e um pulso intra-ciclo, e o degrau do aparelho sai negativo."""
    p = np.nan_to_num(np.asarray(p, float), nan=0.0)
    on = p >= limiar
    b = np.flatnonzero(~on[:-1] & on[1:]) + 1
    b = b[(b >= meia_janela) & (b < len(p) - meia_janela)]
    if len(b) == 0:
        return b
    desl = np.convolve(on.astype(int), np.ones(meia_janela, int), "full")[:len(on)]
    # desl[i] = soma de on[i-meia+1 .. i]; queremos on[b-meia .. b-1] todos 0
    return b[desl[b - 1] == 0]


def melhor_lag(agg: np.ndarray, app: np.ndarray, max_lag: int = 120) -> tuple[int, float, float]:
    """Lag de maior |correlacao| entre as diferencas primeiras do agregado e do
    aparelho, a correlacao COM sinal nesse lag e a correlacao no lag 0.
    Lag > 0 = agregado atrasado em relacao ao plugue."""
    da, dp = np.diff(np.asarray(agg, float)), np.diff(np.asarray(app, float))
    ok = np.isfinite(da) & np.isfinite(dp)
    da, dp = np.where(ok, da, 0.0), np.where(ok, dp, 0.0)

    def corr(l: int) -> float:
        x, y = (da[l:], dp[:len(dp) - l]) if l >= 0 else (da[:l], dp[-l:])
        if x.std() == 0 or y.std() == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    lags = np.arange(-max_lag, max_lag + 1)
    c = np.array([corr(int(l)) for l in lags])
    i = int(np.abs(c).argmax())
    return int(lags[i]), float(c[i]), float(c[max_lag])


# --------------------------------------------------------------------------------------
# Lacunas
# --------------------------------------------------------------------------------------


def corridas_de_zero(x: np.ndarray, t_s: np.ndarray, minimo_s: float = 120.0) -> np.ndarray:
    """Mascara das amostras nativas dentro de uma sequencia de ZEROS EXATOS (ou NaN)
    que dura pelo menos `minimo_s` segundos. E como a versao limpa do REFIT marca
    lacunas longas, e como o ECO marca falta (-1, que aqui ja chega como NaN)."""
    z = (np.asarray(x) == 0) | ~np.isfinite(np.asarray(x, float))
    m = np.zeros(len(z), bool)
    if not z.any():
        return m
    d = np.diff(np.concatenate(([0], z.astype(np.int8), [0])))
    ini, fim = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    for a, b in zip(ini, fim):
        if t_s[b - 1] - t_s[a] >= minimo_s:
            m[a:b] = True
    return m


def fracao_lacuna(t_nativo: np.ndarray, mascara: np.ndarray, t_alvo: np.ndarray,
                  taxa_s: int) -> np.ndarray:
    """Para cada intervalo [t, t + taxa_s) da serie reamostrada, a fracao das
    amostras nativas que estavam em lacuna. Alinha pelo carimbo de inicio, que e
    o que `DataFrame.resample` usa."""
    caixa = (np.asarray(t_nativo, np.int64) // taxa_s) * taxa_s
    df = pd.DataFrame({"c": caixa, "m": mascara.astype(float)}).groupby("c")["m"].mean()
    return df.reindex(np.asarray(t_alvo, np.int64)).fillna(0.0).to_numpy(np.float32)


# --------------------------------------------------------------------------------------
# A2 - divisao sem vazamento
# --------------------------------------------------------------------------------------


def holdout_temporal(s: Serie, semanas: float = 4.0, embargo: int = 288
                     ) -> tuple[Serie, Serie]:
    """Validacao = as ultimas `semanas` do periodo da serie; treino = o resto,
    menos `embargo` amostras antes do corte (nenhuma janela de treino alcanca a
    validacao). Para o intra, espelha "o mes seguinte" do protocolo."""
    c = s.centros
    t_c = s.t[c]
    corte_t = t_c.max() - int(semanas * 7 * 86400)
    i_corte = int(np.searchsorted(s.t, corte_t))
    va = c[c >= i_corte]
    tr = c[c < i_corte - embargo]
    return (Serie(s.vistas, s.rotulo, tr, s.nome + "[tr]", s.t),
            Serie(s.vistas, s.rotulo, va, s.nome + "[va]", s.t))


def cobertura(alvos: np.ndarray, antes: int, depois: int) -> np.ndarray:
    """Indices brutos cobertos pelas janelas [alvo - antes, alvo + depois]."""
    if len(alvos) == 0:
        return np.zeros(0, np.int64)
    m = np.zeros(int(alvos.max()) + depois + 1, bool)
    for d in range(-antes, depois + 1):
        i = alvos + d
        m[i[i >= 0]] = True
    return np.flatnonzero(m)


def amostras_compartilhadas(alvos_tr: np.ndarray, alvos_va: np.ndarray,
                            antes: int, depois: int) -> int:
    return int(np.intersect1d(cobertura(alvos_tr, antes, depois),
                              cobertura(alvos_va, antes, depois)).size)


def test_sem_vazamento(alvos_tr: np.ndarray, alvos_va: np.ndarray,
                       antes: int, depois: int) -> None:
    n = amostras_compartilhadas(alvos_tr, alvos_va, antes, depois)
    assert n == 0, f"{n} amostras brutas compartilhadas entre treino e validacao"
