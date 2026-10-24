"""Taxa de amostragem do protocolo (5 ou 15 min), escolhida pela variavel NILM_TAXA.

O pipeline foi escrito para 5 min (300 s). A 15 min (900 s) do artigo, os
tamanhos que estavam em AMOSTRAS de 5 min passam por `amostras`, que preserva a
duracao em tempo: 288 (24 h) -> 96, 48 (4 h) -> 16. Com NILM_TAXA ausente ou 300
tudo fica exatamente como antes.

Uso:
    NILM_TAXA=900 python src/experimento_reco.py ...
"""

from __future__ import annotations

import os

TAXA = int(os.environ.get("NILM_TAXA", "300"))
assert TAXA in (300, 900), f"NILM_TAXA={TAXA}: so 300 (5 min) ou 900 (15 min)"
ROTULO = {300: "5min", 900: "15min"}[TAXA]
SUB = TAXA // 30                   # caixas de 30 s da grade fina por intervalo


def amostras(n5: int, minimo: int = 1) -> int:
    """Um tamanho definido em amostras de 5 min, na taxa corrente (mesma duracao)."""
    return n5 if TAXA == 300 else max(minimo, int(round(n5 * 300 / TAXA)))
