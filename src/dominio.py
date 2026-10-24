"""Generalizacao de dominio com uma casa fonte so.

O diagnostico que motiva este modulo: na REFIT casa 8, 97,8% do tempo com a
maquina de lavar ligada cai entre 00 h e 08 h, e das 10 h as 23 h a prevalencia
e exatamente zero. Na ECO casa 1 a atividade esta espalhada pelo dia. Uma janela
centrada de 24 h permite ao modelo se localizar no dia pela forma da curva de
carga, entao ele pode aprender "nunca durante o dia" - regra que acerta 100% na
fonte e erra metade dos ciclos no alvo.

E uma nao-estacionariedade de FASE, e a vista `casa` nao a corrige: `casa`
normaliza NIVEL. Daqui saem os dois ataques baratos:

  randomizar_fourier  embaralha a envoltoria de baixa frequencia (nivel da casa
                      + ciclo diurno) preservando a banda do degrau do aparelho.
                      E a ideia central da familia PhASER / Fourier Domain
                      Adaptation: a baixa frequencia e "estilo", especifico do
                      dominio; a banda media e "conteudo", o que transfere.

  grupos_tempo        subdominios latentes JA rotulados (estacao x fim de
                      semana), para group-DRO. E o nucleo do DIVERSIFY sem o
                      laco min-max adversarial: em vez de descobrir os
                      subdominios, usa os que o calendario ja da de graca.

Nao implementa CauDiTS (disentanglement causal precisa de varios dominios fonte
para identificar o que e invariante - com uma casa so, tudo que e constante no
treino parece invariante, inclusive o vies das 00-08 h) nem o gerador por
difusao do KDD 2025 (responde a mesma pergunta que `randomizar_fourier`, por um
custo tres ordens de grandeza maior; so faz sentido se a versao barata mover a
agulha).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


# --------------------------------------------------------------------------------------
# Randomizacao no espaco de Fourier
# --------------------------------------------------------------------------------------


def randomizar_fourier(x: Tensor, forca: float = 0.5, corte: int = 8,
                       jitter: float = 0.5) -> Tensor:
    """Perturba amplitude e fase das `corte` primeiras raias de cada janela.

    x tem forma (B, V, L) e ja vem normalizado. A raia k corresponde a um
    periodo de L/k amostras: com L = 288 a 5 min, k=1 e o ciclo de 24 h, k=8 e
    o de 3 h. Um ciclo de maquina de lavar dura 1-2 h, ou seja k entre 12 e 24,
    entao `corte=8` mexe na envoltoria diurna e deixa a assinatura do aparelho
    intacta.

    A componente DC (k=0) entra junto: e o nivel absoluto da casa, o mesmo que
    a vista `casa` normaliza - randomiza-lo aqui e redundante mas inofensivo, e
    cobre o caso de se usar outra vista.
    """
    if forca <= 0 and jitter <= 0:
        return x
    n = x.shape[-1]
    X = torch.fft.rfft(x, dim=-1)
    k = min(corte, X.shape[-1] - 1)
    amp, fase = X.abs(), X.angle()

    escala = torch.empty(x.shape[0], x.shape[1], k + 1, device=x.device).uniform_(
        1 - forca, 1 + forca)
    desvio = torch.empty(x.shape[0], x.shape[1], k + 1, device=x.device).uniform_(
        -np.pi * jitter, np.pi * jitter)
    desvio[..., 0] = 0.0              # a fase da DC nao existe; mexer nela inverte o sinal

    amp = amp.clone()
    fase = fase.clone()
    amp[..., :k + 1] = amp[..., :k + 1] * escala
    fase[..., :k + 1] = fase[..., :k + 1] + desvio
    return torch.fft.irfft(torch.polar(amp, fase), n=n, dim=-1).to(x.dtype)


def fazer_aumento(nome: str, forca: float, corte: int, jitter: float):
    """Devolve a funcao de aumento usada no laco de treino, ou None."""
    if nome in ("nenhum", "", None):
        return None
    if nome == "fourier":
        return lambda x: randomizar_fourier(x, forca, corte, jitter)
    raise ValueError(f"aumento desconhecido: {nome}")


# --------------------------------------------------------------------------------------
# Subdominios latentes do calendario
# --------------------------------------------------------------------------------------

N_GRUPOS = {"estacao_dia": 8, "estacao": 4, "dia": 2, "hora": 4}


def grupos_tempo(t: np.ndarray, esquema: str = "estacao_dia") -> np.ndarray:
    """Um id de subdominio por instante, a partir do carimbo de tempo.

    `estacao_dia` = estacao (4) x fim-de-semana (2). Oito grupos e um numero
    pensado para o lote de 128: com 24 (mes x dia) cada grupo cairia para ~5
    exemplos por lote e a perda por grupo viraria ruido.
    """
    d = np.asarray(t, dtype="int64").astype("datetime64[s]")
    mes = d.astype("datetime64[M]").astype(int) % 12 + 1
    estacao = (mes % 12) // 3                       # 0 DJF, 1 MAM, 2 JJA, 3 SON
    dias = np.asarray(t, dtype="int64") // 86400
    fim_de_semana = (((dias + 3) % 7) >= 5).astype(int)   # 1970-01-01 foi quinta
    hora = (np.asarray(t, dtype="int64") % 86400) // 3600

    if esquema == "estacao_dia":
        return (estacao * 2 + fim_de_semana).astype(np.int64)
    if esquema == "estacao":
        return estacao.astype(np.int64)
    if esquema == "dia":
        return fim_de_semana.astype(np.int64)
    if esquema == "hora":
        return (hora // 6).astype(np.int64)
    raise ValueError(f"esquema desconhecido: {esquema}")


class PesosDRO:
    """Group-DRO por gradiente exponenciado (Sagawa et al., 2020).

    q_g <- q_g * exp(eta * L_g), renormalizado; a perda do lote e sum_g q_g L_g.
    O grupo que vai pior ganha peso, entao o treino otimiza o pior subdominio em
    vez da media - que e exatamente o que se quer quando o teste e um dominio
    que nao esta no treino.
    """

    def __init__(self, n: int, taxa: float = 0.01, dispositivo=None):
        self.q = torch.ones(n, device=dispositivo) / n
        self.taxa = taxa

    def combinar(self, perdas: dict[int, Tensor]) -> Tensor:
        with torch.no_grad():
            for g, l in perdas.items():
                self.q[g] = self.q[g] * torch.exp(self.taxa * l.detach())
            self.q /= self.q.sum()
        return sum(self.q[g] * l for g, l in perdas.items())


# --------------------------------------------------------------------------------------
# Desvio de prevalencia entre dominios
# --------------------------------------------------------------------------------------


def prevalencia_em(p: np.ndarray, prev_origem: float, iteracoes: int = 100,
                   tol: float = 1e-8) -> float:
    """Estima a prevalencia do ALVO so a partir das probabilidades previstas.

    EM de Saerens, Latinne & Decaestecker (2002) para `prior shift`: assume que
    p(x|y) e a mesma nos dois dominios e que so p(y) muda - exatamente o caso
    aqui, onde a maquina de lavar liga com frequencia diferente na Suica e no
    Reino Unido, mas um ciclo de lavagem continua parecendo um ciclo de lavagem.

    Motivacao: casar a TAXA de positivos entre fonte e alvo falhou (F1 de cross
    caiu de 0,147 para 0,104) porque a taxa nao e propriedade transferivel do
    modelo - ela mistura a qualidade do classificador com a prevalencia do
    dominio, e a prevalencia mudou de 4,5% para 7,4%. Estimar a prevalencia do
    alvo ataca justamente esse termo, sem usar nenhum rotulo do alvo.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    a = float(np.clip(prev_origem, 1e-6, 1 - 1e-6))
    atual = a
    for _ in range(iteracoes):
        # reescala as probabilidades da razao de priores e renormaliza
        num = (atual / a) * p
        den = num + ((1 - atual) / (1 - a)) * (1 - p)
        pos = num / den
        novo = float(pos.mean())
        if abs(novo - atual) < tol:
            break
        atual = novo
    return atual


def limiar_por_prevalencia(p: np.ndarray, prev: float) -> float:
    """Corte que faz a fracao de positivos no alvo bater `prev`."""
    prev = float(np.clip(prev, 1e-6, 1 - 1e-6))
    return float(np.quantile(np.asarray(p), 1 - prev))
