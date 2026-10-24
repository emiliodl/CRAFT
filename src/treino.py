"""Treino, inferencia e metricas.

Inferencia com janelas SOBREPOSTAS e media das probabilidades: cada instante e
previsto por varias janelas, em posicoes diferentes dentro delas, entao na media
todo instante recebe contexto dos dois lados - a condicao do seq2point, com uma
fracao do custo (uma passada por janela em vez de uma passada por rotulo).

O criterio de parada e de limiar SO olha a validacao da casa de origem. A casa
de teste nunca entra em nenhuma decisao: em generalizacao entre casas, a F1 na
origem sobe de forma monotona enquanto a do alvo tem pico e cai, entao usar o
alvo para parar inventa um resultado que nao existe na pratica.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor

from dados import Serie
from dominio import PesosDRO, fazer_aumento
from modelo import Classificador, aumentar, perda_contrastiva

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------------------
# Lotes
# --------------------------------------------------------------------------------------


class Fonte:
    """Mantem as vistas e o rotulo na GPU e monta lotes por indexacao."""

    def __init__(self, s: Serie, indices_vistas: list[int], largura: int,
                 media: np.ndarray, desvio: np.ndarray):
        v = s.vistas[indices_vistas]
        m, dv = media[indices_vistas, None], desvio[indices_vistas, None]
        self.vistas = torch.from_numpy((v - m) / dv).to(DISPOSITIVO)
        self.rotulo = torch.from_numpy(s.rotulo).to(DISPOSITIVO)
        self.centros = torch.from_numpy(s.centros).to(DISPOSITIVO)
        self.largura = largura
        self.desloc = torch.arange(-(largura // 2), largura - largura // 2, device=DISPOSITIVO)
        self.nome = s.nome

    def __len__(self) -> int:
        return len(self.centros)

    def lote(self, pos: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        idx = self.centros[pos][:, None] + self.desloc[None, :]     # (B, L)
        return self.vistas[:, idx].permute(1, 0, 2), self.rotulo[idx], idx


# --------------------------------------------------------------------------------------
# Metricas
# --------------------------------------------------------------------------------------


def f1_positiva(y: np.ndarray, p: np.ndarray, limiar: float) -> float:
    prev = p >= limiar
    vp = float((prev & (y == 1)).sum())
    if vp == 0:
        return 0.0
    precisao = vp / float(prev.sum())
    recall = vp / float((y == 1).sum())
    return 2 * precisao * recall / (precisao + recall)


def melhor_limiar(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    grade = np.unique(np.quantile(p, np.linspace(0.001, 0.999, 300)))
    f1s = [f1_positiva(y, p, t) for t in grade]
    i = int(np.argmax(f1s))
    return float(grade[i]), float(f1s[i])


@dataclass
class Resultado:
    f1: float = 0.0             # no limiar escolhido na validacao de origem
    f1_oraculo: float = 0.0     # teto: melhor limiar no proprio alvo
    ap: float = 0.0
    auc: float = 0.0
    prevalencia: float = 0.0
    limiar: float = 0.0
    extras: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (f"F1={self.f1:.3f} (oraculo {self.f1_oraculo:.3f})  AP={self.ap:.3f}  "
                f"AUC={self.auc:.3f}  prev={100 * self.prevalencia:.1f}%")


@torch.no_grad()
def prever(modelo: Classificador, fonte: Fonte, lote: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidade media por instante. Devolve (rotulo, probabilidade)."""
    modelo.eval()
    n = fonte.vistas.shape[1]
    soma = torch.zeros(n, device=DISPOSITIVO)
    conta = torch.zeros(n, device=DISPOSITIVO)
    for i in range(0, len(fonte), lote):
        pos = torch.arange(i, min(i + lote, len(fonte)), device=DISPOSITIVO)
        x, _, idx = fonte.lote(pos)
        p = torch.sigmoid(modelo(x))
        soma.index_add_(0, idx.reshape(-1), p.reshape(-1))
        conta.index_add_(0, idx.reshape(-1), torch.ones_like(p).reshape(-1))
    cobertos = conta > 0
    return (fonte.rotulo[cobertos].cpu().numpy(),
            (soma[cobertos] / conta[cobertos]).cpu().numpy())


@torch.no_grad()
def prever_com_indices(modelo: Classificador, fonte: Fonte,
                       lote: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Como `prever`, mas devolve tambem quais posicoes da serie foram cobertas.

    O pos-processamento precisa disso: decodificar com prior de duracao exige
    saber onde estao os buracos no tempo, para nao costurar periodos separados.
    """
    modelo.eval()
    n = fonte.vistas.shape[1]
    soma = torch.zeros(n, device=DISPOSITIVO)
    conta = torch.zeros(n, device=DISPOSITIVO)
    for i in range(0, len(fonte), lote):
        pos = torch.arange(i, min(i + lote, len(fonte)), device=DISPOSITIVO)
        x, _, idx = fonte.lote(pos)
        p = torch.sigmoid(modelo(x))
        soma.index_add_(0, idx.reshape(-1), p.reshape(-1))
        conta.index_add_(0, idx.reshape(-1), torch.ones_like(p).reshape(-1))
    cobertos = torch.nonzero(conta > 0, as_tuple=True)[0]
    return (fonte.rotulo[cobertos].cpu().numpy(),
            (soma[cobertos] / conta[cobertos]).cpu().numpy(),
            cobertos.cpu().numpy())


def avaliar(modelo: Classificador, fonte: Fonte, limiar: float | None = None) -> Resultado:
    y, p = prever(modelo, fonte)
    r = Resultado(prevalencia=float(y.mean()))
    if y.min() == y.max():
        return r
    r.ap = float(average_precision_score(y, p))
    r.auc = float(roc_auc_score(y, p))
    r.limiar, r.f1_oraculo = melhor_limiar(y, p)
    if limiar is not None:
        r.limiar = limiar
    r.f1 = f1_positiva(y, p, r.limiar)
    return r


# --------------------------------------------------------------------------------------
# Treino
# --------------------------------------------------------------------------------------


def fazer_perda(nome: str, prevalencia: float, gama: float = 2.0,
                lamb: float = 1.0):
    """Perda de treino. A BCE pura e dominada pelos negativos quando so 2-5% das
    janelas sao positivas, e o modelo aprende a ser conservador. As alternativas
    mexem nesse desequilibrio de formas diferentes:

      bce         referencia, sem nenhuma correcao
      bce_peso    BCE com pos_weight = neg/pos: cada positivo pesa como ~30 negativos
      focal       foca nos exemplos dificeis, (1-p)^gama, com alfa da prevalencia
      soft_f1     otimiza diretamente 1 - F1 diferenciavel (tp/fp/fn contínuos)
      bce_soft_f1 soma das duas: a BCE mantem o gradiente estavel no inicio,
                  a soft-F1 alinha o objetivo com a metrica
    """
    p = float(np.clip(prevalencia, 1e-4, 1 - 1e-4))
    peso = torch.tensor((1 - p) / p, device=DISPOSITIVO)

    def soft_f1(logit: Tensor, y: Tensor) -> Tensor:
        pr = torch.sigmoid(logit)
        vp = (pr * y).sum()
        fp = (pr * (1 - y)).sum()
        fn = ((1 - pr) * y).sum()
        return 1 - (2 * vp + 1e-6) / (2 * vp + fp + fn + 1e-6)

    def focal(logit: Tensor, y: Tensor) -> Tensor:
        pr = torch.sigmoid(logit)
        pt = pr * y + (1 - pr) * (1 - y)
        alfa = (1 - p) * y + p * (1 - y)
        return -(alfa * (1 - pt).pow(gama) * torch.log(pt.clamp_min(1e-8))).mean()

    fn = {
        "bce": lambda l, y: F.binary_cross_entropy_with_logits(l, y),
        "bce_peso": lambda l, y: F.binary_cross_entropy_with_logits(l, y, pos_weight=peso),
        "focal": focal,
        "soft_f1": soft_f1,
        "bce_soft_f1": lambda l, y: F.binary_cross_entropy_with_logits(l, y) + lamb * soft_f1(l, y),
    }
    if nome not in fn:
        raise ValueError(f"perda desconhecida: {nome}")
    return fn[nome]


@dataclass
class Config:
    perda: str = "bce"               # ver fazer_perda
    prevalencia: float = 0.05        # usada por bce_peso e focal
    epocas: int = 300                # teto alto de proposito: quem para e a paciencia
    epocas_contrastivas: int = 0     # > 0 liga o pre-treino no molde do TF-C
    lote: int = 128
    taxa: float = 1e-3
    decaimento: float = 1e-4
    paciencia: int = 30
    peso_consistencia: float = 0.3   # lambda de L_C durante o pre-treino
    peso_contrastivo_fino: float = 0.0   # > 0 mantem L_v + L_C junto com a BCE
    verboso: bool = False
    # generalizacao de dominio (ver src/dominio.py)
    aumento: str = "nenhum"          # "fourier" randomiza a envoltoria de baixa frequencia
    forca_aumento: float = 0.5
    corte_fourier: int = 8
    jitter_fourier: float = 0.5
    dro: bool = False                # group-DRO sobre os subdominios do calendario
    dro_taxa: float = 0.01


def pre_treinar(modelo: Classificador, treino: Fonte, cfg: Config) -> list[dict]:
    """Fase auto-supervisionada: so L_v + lambda * L_C, sem rotulo nenhum."""
    opt = torch.optim.AdamW(modelo.parameters(), lr=cfg.taxa, weight_decay=cfg.decaimento)
    historico = []
    for ep in range(cfg.epocas_contrastivas):
        modelo.train()
        ordem = torch.randperm(len(treino), device=DISPOSITIVO)
        acum = np.zeros(2)
        n = 0
        for i in range(0, len(ordem) - cfg.lote + 1, cfg.lote):
            x, _, _ = treino.lote(ordem[i:i + cfg.lote])
            l_v, l_c = perda_contrastiva(modelo, x, aumentar(x))
            perda = l_v + cfg.peso_consistencia * l_c
            opt.zero_grad(set_to_none=True)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            acum += [float(l_v), float(l_c)]
            n += 1
        historico.append(dict(epoca=ep, l_vista=acum[0] / n, l_consistencia=acum[1] / n))
        if cfg.verboso:
            print(f"    pre {ep:3d}  L_v={acum[0] / n:.3f}  L_C={acum[1] / n:.3f}")
    return historico


def treinar(modelo: Classificador, treino: Fonte, val: Fonte, cfg: Config) -> dict:
    """Fase supervisionada, com parada e limiar decididos na validacao de origem.

    Nao ha teto util de epocas: o treino segue ate a F1 de validacao ficar
    `cfg.paciencia` epocas sem melhorar, e a taxa de aprendizado cai pela metade
    a cada 10 epocas de estagnacao. O criterio e a F1 da classe positiva (no
    melhor limiar da validacao), e nao a AP, porque e a F1 que se quer maximizar.
    """
    opt = torch.optim.AdamW(modelo.parameters(), lr=cfg.taxa, weight_decay=cfg.decaimento)
    agenda = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10)
    calcular_perda = fazer_perda(cfg.perda, cfg.prevalencia)
    aumento = fazer_aumento(cfg.aumento, cfg.forca_aumento, cfg.corte_fourier,
                            cfg.jitter_fourier)
    # o aumento e o DRO so valem no treino; a validacao decide parada e limiar e
    # tem de continuar sendo a distribuicao real da casa de origem.
    grupo = getattr(treino, "grupo", None)
    dro = (PesosDRO(int(grupo.max()) + 1, cfg.dro_taxa, DISPOSITIVO)
           if cfg.dro and grupo is not None else None)
    melhor, melhor_estado, melhor_ep, espera = -1.0, None, -1, 0
    historico = []

    for ep in range(cfg.epocas):
        modelo.train()
        ordem = torch.randperm(len(treino), device=DISPOSITIVO)
        soma, n = 0.0, 0
        for i in range(0, len(ordem) - cfg.lote + 1, cfg.lote):
            x, y, idx = treino.lote(ordem[i:i + cfg.lote])
            if aumento is not None:
                x = aumento(x)
            saida = modelo(x)
            if dro is None:
                perda = calcular_perda(saida, y)
            else:
                g = grupo[idx[:, 0]]
                perda = dro.combinar({int(gi): calcular_perda(saida[g == gi], y[g == gi])
                                      for gi in torch.unique(g)})
            if cfg.peso_contrastivo_fino > 0:
                l_v, l_c = perda_contrastiva(modelo, x, aumentar(x))
                perda = perda + cfg.peso_contrastivo_fino * (l_v + cfg.peso_consistencia * l_c)
            opt.zero_grad(set_to_none=True)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            soma += float(perda.detach())
            n += 1

        r = avaliar(modelo, val)
        agenda.step(r.f1_oraculo)
        historico.append(dict(epoca=ep, perda=soma / max(n, 1), ap_val=r.ap, f1_val=r.f1_oraculo))
        if cfg.verboso:
            print(f"    ep {ep:3d}  perda={soma / max(n, 1):.4f}  AP_val={r.ap:.3f}  "
                  f"F1_val={r.f1_oraculo:.3f}")
        if r.f1_oraculo > melhor:
            melhor, melhor_ep, espera = r.f1_oraculo, ep, 0
            melhor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            espera += 1
            if espera >= cfg.paciencia:
                break

    if melhor_estado is not None:
        modelo.load_state_dict(melhor_estado)
    r_val = avaliar(modelo, val)
    return dict(historico=historico, melhor_epoca=melhor_ep, ap_val=r_val.ap,
                limiar=r_val.limiar, f1_val=r_val.f1_oraculo)


def contar_parametros_seguro(m) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
