"""TimesFM (Google) congelado + sonda treinavel, para classificacao de estado.

Motivacao: o nosso gargalo e generalizacao de dominio com UMA casa fonte. Uma
rede de 350 mil parametros treinada so na REFIT casa 8 nao tem como aprender o
que e invariante entre residencias - nao ha contraste. Um modelo pre-treinado em
bilhoes de pontos de series diversas ja traz esse tipo de representacao.

O TimesFM e um decoder-only de PREVISAO (Das et al., ICML 2024), nao um
classificador: nao ha cabeca de classificacao nem API para estados internos.
Mas o `forward` do modulo interno devolve `output_embeddings` diretamente, entao
da para usa-lo como backbone congelado e treinar so uma sonda em cima.

  entrada    janela de L amostras do agregado, em W
  normaliza  media/desvio DA PROPRIA JANELA - e o que o TimesFM faz, e de quebra
             remove o nivel da casa, que e metade do desvio de dominio
  patches    L/32 patches de 32 amostras (o tamanho nativo do 2.5)
  backbone   20 camadas, d=1280, 231M parametros, CONGELADO
  sonda      atencao sobre os patches + MLP -> 1 logito

A sonda tem a mesma estrutura de cabeca do nosso `ClassificadorPonto`, entao a
comparacao isola o backbone: mesma tarefa, mesmos alvos, mesma divisao, mesmo
criterio de limiar.

Como o backbone e congelado, os embeddings sao extraidos UMA vez e guardados;
o treino da sonda depois roda em segundos.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

PATCH = 32          # tamanho de patch nativo do TimesFM 2.5
DIM = 1280          # d_model do checkpoint de 200M
REPO = "google/timesfm-2.5-200m-pytorch"


def carregar_backbone(dispositivo: torch.device) -> nn.Module:
    """Modulo interno do TimesFM, em eval e com gradiente desligado."""
    import timesfm

    C = timesfm.TimesFM_2p5_200M_torch
    m = C.from_pretrained(C.DEFAULT_REPO_ID or REPO, torch_compile=False)
    net = m.model.eval().to(dispositivo)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


N_CAMADAS = 20      # profundidade do stacked_xf no checkpoint de 200M


@torch.no_grad()
def extrair(net: nn.Module, janelas: np.ndarray, camadas: list[int] | None = None,
            lote: int = 64, dispositivo: torch.device | None = None
            ) -> dict[int, np.ndarray]:
    """(M, L) janelas em W -> {camada: (M, L/32, 1280)} em float16.

    A normalizacao e por janela, nao pelo conjunto de treino: e o que o TimesFM
    espera, e faz com que o nivel absoluto da casa nunca chegue ao backbone.

    `camadas` sao indices de 1 a 20; 20 e a saida final (o mesmo
    `output_embeddings` que o forward devolve). Camadas intermediarias costumam
    dar sondas melhores que a ultima, porque a ultima ja esta especializada na
    tarefa de PREVISAO para a qual o modelo foi treinado - e a nossa tarefa e
    outra. Todas as camadas pedidas saem de UMA passada, via hooks.
    """
    dispositivo = dispositivo or next(net.parameters()).device
    camadas = sorted(camadas or [N_CAMADAS])
    m, comprimento = janelas.shape
    if comprimento % PATCH:
        raise ValueError(f"comprimento {comprimento} nao e multiplo de {PATCH}")
    n = comprimento // PATCH
    saidas = {c: np.empty((m, n, DIM), dtype=np.float16) for c in camadas}

    capturado: dict[int, Tensor] = {}

    def fazer_hook(c: int):
        def hook(_mod, _entrada, saida):
            capturado[c] = saida[0] if isinstance(saida, tuple) else saida
        return hook

    alcas = [net.stacked_xf[c - 1].register_forward_hook(fazer_hook(c)) for c in camadas]
    try:
        for i in range(0, m, lote):
            x = torch.from_numpy(janelas[i:i + lote]).float().to(dispositivo)
            med = x.mean(dim=1, keepdim=True)
            dev = x.std(dim=1, keepdim=True).clamp_min(1e-3)
            x = ((x - med) / dev).reshape(-1, n, PATCH)
            mascara = torch.zeros_like(x)           # 0 = amostra valida
            net(x, mascara)
            for c in camadas:
                saidas[c][i:i + lote] = capturado[c].to(torch.float16).cpu().numpy()
    finally:
        for a in alcas:
            a.remove()
    return saidas


class Sonda(nn.Module):
    """Pooling sobre os patches + MLP. Mesma cabeca do ClassificadorPonto."""

    def __init__(self, dim: int = DIM, oculto: int = 96, dropout: float = 0.1,
                 pooling: str = "atencao"):
        super().__init__()
        self.pooling = pooling
        self.norma = nn.LayerNorm(dim)
        self.pontuacao = nn.Linear(dim, 1)
        self.cabeca = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(dim, oculto), nn.GELU(),
            nn.Linear(oculto, 1))

    def forward(self, e: Tensor) -> Tensor:          # (B, n, dim) -> (B, 1)
        e = self.norma(e)
        if self.pooling == "ultimo":
            # No alinhamento causal o alvo e a ultima amostra, e o modelo e
            # decoder-only: o ultimo patch e o unico que viu a janela inteira
            # ate o alvo. Mediar com os anteriores so dilui.
            return self.cabeca(e[:, -1])
        peso = torch.softmax(self.pontuacao(e), dim=1)
        return self.cabeca((e * peso).sum(dim=1))


class FonteEmbedding:
    """Mesma interface de FontePonto, mas servindo embeddings ja extraidos."""

    def __init__(self, emb: np.ndarray, rotulo: np.ndarray, dispositivo: torch.device):
        self.emb = torch.from_numpy(emb).to(dispositivo)
        self.rotulo = torch.from_numpy(rotulo.astype(np.float32)).to(dispositivo)
        self.nome = ""

    def __len__(self) -> int:
        return len(self.rotulo)

    def lote(self, pos: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return (self.emb[pos].float(), self.rotulo[pos][:, None], pos[:, None])


class SondaMistura(nn.Module):
    """Sonda sobre VARIAS camadas, com mistura escalar aprendida.

    A varredura mostrou um conflito que nenhuma camada isolada resolve: no intra
    a camada 5 vence (F1 0,606 contra 0,545 da camada 20) e no cross e o
    inverso (0,148 da 20 contra 0,114 da 5). A ultima camada ja esta
    especializada na tarefa de PREVISAO do pre-treino; as rasas guardam mais do
    estado instantaneo. Em vez de escolher uma - o que exigiria olhar o teste -
    a sonda aprende pesos por softmax sobre as camadas, junto com o resto.
    E a mistura escalar do ELMo (Peters et al., 2018).
    """

    def __init__(self, n_camadas: int, dim: int = DIM, oculto: int = 96,
                 dropout: float = 0.1, pooling: str = "ultimo"):
        super().__init__()
        self.pooling = pooling
        self.pesos = nn.Parameter(torch.zeros(n_camadas))   # softmax -> uniforme no inicio
        self.escala = nn.Parameter(torch.ones(1))
        self.norma = nn.LayerNorm(dim)
        self.pontuacao = nn.Linear(dim, 1)
        self.cabeca = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(dim, oculto), nn.GELU(),
            nn.Linear(oculto, 1))

    def forward(self, e: Tensor) -> Tensor:          # (B, C, n, dim) -> (B, 1)
        w = torch.softmax(self.pesos, dim=0)
        e = (e * w[None, :, None, None]).sum(dim=1) * self.escala
        e = self.norma(e)
        if self.pooling == "ultimo":
            return self.cabeca(e[:, -1])
        peso = torch.softmax(self.pontuacao(e), dim=1)
        return self.cabeca((e * peso).sum(dim=1))


class FonteMultiCamada:
    """Serve (B, C, n, dim): as C camadas empilhadas, para a SondaMistura."""

    def __init__(self, emb: dict[int, np.ndarray], rotulo: np.ndarray,
                 dispositivo: torch.device):
        camadas = sorted(emb)
        pilha = np.stack([emb[c] for c in camadas], axis=1)     # (M, C, n, dim)
        self.emb = torch.from_numpy(pilha).to(dispositivo)      # fica em float16
        self.rotulo = torch.from_numpy(rotulo.astype(np.float32)).to(dispositivo)
        self.camadas = camadas
        self.nome = ""

    def __len__(self) -> int:
        return len(self.rotulo)

    def lote(self, pos: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return (self.emb[pos].float(), self.rotulo[pos][:, None], pos[:, None])
