"""E0.4 / E0.5 do v2: inventario das celulas de teste e ordem de grandeza do poder.

So rotulos (nenhum modelo): numero de ativacoes, prevalencia, trivial 2p/(1+p),
dias e, pela simulacao da §5.2, a meia-largura esperada do IC95% de um F1.
Conta como olhada DIAGNOSTICA no ECO e fica registrada.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dados_v2 import arquivo, carregar_w, duracoes          # noqa: E402
from estatistica import ativacoes, blocos_dia                 # noqa: E402
from experimento_evicon import registrar_olhada, SAIDA       # noqa: E402

ARTIGO = {("washing machine", "intra"): 0.63, ("washing machine", "cross"): 0.30,
          ("freezer", "intra"): 0.46, ("freezer", "cross"): 0.78,
          ("television", "intra"): 0.68, ("television", "cross"): 0.54}


def main():
    registrar_olhada("E0.4 inventario de rotulos das celulas de teste (sem modelo)")
    linhas = []
    for ap in ("washing machine", "freezer", "television"):
        celulas = {"intra": [("refit", 8, "teste")],
                   "cross": [("eco", n, "cross") for n in (1, 2, 5)]}
        for cel, fontes in celulas.items():
            ys, ts, casas = [], [], []
            for f, c, part in fontes:
                s = carregar_w(arquivo(f, c, ap, part))
                if s is None:
                    continue
                ys.append(s.rotulo)
                ts.append(s.t)
                casas.append(f"{f} {c}")
            if not ys:
                continue
            y = np.concatenate(ys)
            n_at = sum(ativacoes(a, b) for a, b in zip(ys, ts))
            dur = np.concatenate([duracoes(a, b) for a, b in zip(ys, ts)])
            dias = sum(len(np.unique(blocos_dia(b))) for b in ts)
            p = float(y.mean())
            linhas.append(dict(aparelho=ap, celula=cel, casas="+".join(casas), n=len(y),
                               dias=dias, prevalencia=round(p, 4), ativacoes=n_at,
                               duracao_mediana_min=int(np.median(dur) * 5) if len(dur) else 0,
                               trivial=round(2 * p / (1 + p), 3),
                               artigo=ARTIGO[(ap, cel)],
                               artigo_menos_trivial=round(ARTIGO[(ap, cel)] - 2 * p / (1 + p), 3)))
    d = pd.DataFrame(linhas)
    d.to_csv(SAIDA / "p0_inventario.csv", index=False)
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()
