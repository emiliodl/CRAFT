"""P2 do v2 (docs/v2.md §2.12, §4): escada de ablacoes do EviCon, com decisao e intervencao.

Modelos (v2 §2.12):

  A0    CNN seq2point (ClassificadorPonto, vista casa, 24 h centrada) - referencia
  A0cf  A0 treinado com 25% de pares contrafactuais (a infraestrutura da §3.1 na CNN)
  A1    evidencia sem acumulador, gama = c = 0
  A2    evidencia com acumulador, gama = c = 0
  A3    A2 sem restricao de sinal
  A4    EviCon: A2 + contexto restrito (mascara, gargalo, limites)
  A5    A2 + contexto livre
  A6    A4 em dois estagios (PoE)
  A7    A4 + 25% de pares contrafactuais + L_par

Divisoes (v2 §3.4), sem olhar o ECO:

  treino/val   holdout temporal: ultimas 4 semanas de cada casa fonte, com embargo
               de uma janela inteira (teste de vazamento sobre amostras brutas)
  cross-val    casas REFIT FIXAS nunca usadas no treino (padrao 3, 7, 20): o
               "conjunto fixo de validacao cross" de E1.2
  teste        so com --olhar_teste: mes da casa 8 e ECO 1/2/5. Cada olhada fica
               registrada em resultados/v2/olhadas.csv (v2 §5.3 item 10)

Estagio de decisao (v2 §3.6), regras a/b/c/d com duas calibracoes:

  faixa A   calibra no holdout temporal da(s) casa(s) fonte; trava sem o criterio
            de AUC (nao ha casa retida)
  LOHO      em cada casa de cross-val, calibra nas OUTRAS casas de cross-val, com
            a AUC delas como AUC retida (trava completa)

Uso:
    python src/experimento_evicon.py --aparelhos television --modelos A0,A2,A4
    python src/experimento_evicon.py --fonte A+ --olhar_teste
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contrafactual import gerador, intervencao, lote_sintetico, resumo_banco   # noqa: E402
from dados_v2 import (LIMIAR_ON, Banco, Fonte, arquivo, carregar_w, duracoes,  # noqa: E402
                      juntar_w, p_ref)
from decisao import Decisor, auc, f1, limiar_oraculo                           # noqa: E402
from estatistica import ativacoes                                              # noqa: E402
from evicon import ConfigEviCon, EviCon, perda_evicon                          # noqa: E402
from ponto import ClassificadorPonto                                           # noqa: E402
from saneamento import amostras_compartilhadas, holdout_temporal               # noqa: E402
from taxa import amostras as _a                                                # noqa: E402
from treino import DISPOSITIVO                                                 # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "resultados" / "v2"
W_C = _a(288)              # 24 h

# Fontes aprovadas no portao G0 (scripts/auditoria_p0.py): degrau do agregado > 0 na
# transicao do rotulo a 5 min e prevalencia < 70%. O criterio "|residuo| > 30%" do
# v2 NAO e aplicado: ele reprovaria a propria casa 8 (ver docs/v2_resultados.md).
POOL = {
    "television": [1, 2, 5, 13, 15, 16, 19, 21],
    "washing machine": [1, 2, 4, 5, 6, 9, 10, 11, 13, 15, 16, 17, 18, 19, 21],
    "freezer": [1, 4, 6, 13, 17, 18],
}

MODELOS = {
    "A0": dict(tipo="cnn"),
    "A0cf": dict(tipo="cnn", frac_cf=0.25),
    "A1": dict(tipo="evicon", acumulador=False, contexto="nenhum"),
    "A2": dict(tipo="evicon", contexto="nenhum"),
    "A3": dict(tipo="evicon", contexto="nenhum", sinal=False),
    "A4": dict(tipo="evicon", contexto="restrito"),
    "A5": dict(tipo="evicon", contexto="livre"),
    "A6": dict(tipo="evicon", contexto="restrito", poe=True),
    "A7": dict(tipo="evicon", contexto="restrito", frac_cf=0.25, lam_par=0.5),
    "A4cf": dict(tipo="evicon", contexto="restrito", frac_cf=0.25),
    "A2cf": dict(tipo="evicon", contexto="nenhum", frac_cf=0.25, lam_par=0.5),
    # desenhada DEPOIS da escada A (validacao cross, nao teste): no A4 o deslocamento
    # +-c ordena as janelas sem evidencia (e = 0) so pelo contexto, e o atalho volta
    # pela AUC. Com c = 0 o contexto so multiplica e: essas janelas empatam em b0.
    "A4m": dict(tipo="evicon", contexto="restrito", c=0.0),
    "A4mcf": dict(tipo="evicon", contexto="restrito", c=0.0, frac_cf=0.25, lam_par=0.5),
    # E1.3: re-execucoes que o bug da §9 pode ter invertido (contexto e K rotulos)
    "A0_2h": dict(tipo="cnn", ctx=24), "A0_4h": dict(tipo="cnn", ctx=48),
    "A0_8h": dict(tipo="cnn", ctx=96),
    "A0_4hcf": dict(tipo="cnn", ctx=48, frac_cf=0.25),
    "A0_K8": dict(tipo="cnn", saida=8), "A0_K32": dict(tipo="cnn", saida=32),
}


# --------------------------------------------------------------------------------------
# hiperparametros por aparelho (v2 §2.8), a partir de D90 e P_ref medidos na fonte
# --------------------------------------------------------------------------------------


def hiper(aparelho: str, d90: int, prev: float) -> dict:
    par = lambda v: int(2 * round(v / 2))                         # noqa: E731
    if aparelho in ("freezer", "fridge"):              # ciclicos
        return dict(w_e=_a(24), mascara=_a(12), meia_vida=max(d90, _a(2)), c=0.0, gama=0.5)
    # limites em amostras de 5 min, convertidos para a taxa corrente (mesma duracao)
    w_e = par(min(max(d90, _a(48)), _a(96)))
    m = int(min(max(d90, _a(12)), W_C // 2 - _a(24)))
    c = 1.0 if aparelho == "washing machine" else 0.3
    return dict(w_e=w_e, mascara=m, meia_vida=float(max(d90, _a(6))), c=c, gama=0.5)


# --------------------------------------------------------------------------------------
# adaptadores: lote cru (W) -> logits
# --------------------------------------------------------------------------------------


class Adaptador:
    def __init__(self, modelo, tipo: str, cfg: ConfigEviCon | None, pref: float,
                 norm_casa: tuple[float, float] = (0.0, 1.0), ctx: int = W_C, saida: int = 1):
        self.modelo, self.tipo, self.cfg, self.pref = modelo, tipo, cfg, pref
        self.mu, self.dp = norm_casa
        self.ctx, self.saida = ctx, saida

    def __call__(self, lote: dict) -> dict:
        x = lote["x"]
        if self.tipo == "cnn":
            c0 = x.shape[1] // 2
            x = x[:, c0 - self.ctx // 2:c0 - self.ctx // 2 + self.ctx]
            v = (x / lote["p95"][:, None] - self.mu) / self.dp
            zk = self.modelo(v[:, None, :])
            return dict(z=zk[:, self.saida // 2], zk=zk)
        c0, h = x.shape[1] // 2, self.cfg.w_e // 2
        sl = slice(c0 - h, c0 - h + self.cfg.w_e)
        pr = torch.full((x.shape[0],), self.pref, device=x.device)
        return self.modelo(x[:, sl], pr, x, lote["p95"], lote["lac"][:, sl], lote["lac"])

    def escore(self, lote: dict) -> torch.Tensor:
        return self(lote)["z"]


@torch.no_grad()
def prever(adap: Adaptador, fonte: Fonte, lote: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    adap.modelo.eval()
    zs = []
    for i in range(0, len(fonte), lote):
        pos = torch.arange(i, min(i + lote, len(fonte)), device=DISPOSITIVO)
        zs.append(adap.escore(fonte.lote(pos)).float())
    z = torch.cat(zs).cpu().numpy()
    return fonte.y[fonte.alvos].cpu().numpy(), 1 / (1 + np.exp(-z))


def atualizar_ema(ema, modelo, d: float):
    with torch.no_grad():
        for pe, p in zip(ema.parameters(), modelo.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)
        for be, b in zip(ema.buffers(), modelo.buffers()):
            be.copy_(b)


def treinar(adap: Adaptador, f_tr: Fonte, f_va: Fonte, spec: dict, banco: Banco,
            semente: int, epocas: int, paciencia: int, lote: int = 256,
            taxa: float = 2e-3, ema_d: float = 0.99, lam_ev: float = 0.3,
            lam_p: float = 0.1, margem: float = 2.0, verboso: bool = False) -> dict:
    modelo = adap.modelo
    evi = adap.tipo == "evicon"
    frac_cf, lam_par = spec.get("frac_cf", 0.0), spec.get("lam_par", 0.0)
    g = gerador(semente, DISPOSITIVO)
    ema = copy.deepcopy(modelo)
    adap_ema = copy.copy(adap)
    adap_ema.modelo = ema
    aquecimento = 2 if evi else 0               # curriculo: so transicoes nas primeiras
    rampa_fim = aquecimento + 4
    params = list(modelo.parameters())
    if spec.get("poe"):
        params = [p for n, p in modelo.named_parameters()
                  if not (n.startswith("contexto.") and not n.startswith("contexto.m."))]
    opt = torch.optim.AdamW(params, lr=taxa, weight_decay=1e-4)
    melhor, estado, espera, hist = -1.0, None, 0, []
    for ep in range(epocas):
        modelo.train()
        if evi and modelo.contexto is not None:
            modelo.rampa.fill_(float(np.clip((ep - aquecimento + 1) / (rampa_fim - aquecimento),
                                             0, 1)))
            ema.rampa.copy_(modelo.rampa)
        ordem = torch.randperm(len(f_tr), device=DISPOSITIVO, generator=g)
        soma = 0.0
        for i in range(0, len(ordem) - lote + 1, lote):
            b = f_tr.lote(ordem[i:i + lote])
            b, par = lote_sintetico(b, banco, g, frac_cf, casa=f_tr.casa)
            out = adap(b)
            if evi:
                c0, h = b["x"].shape[1] // 2, adap.cfg.w_e // 2
                yj = b["yj"][:, c0 - h:c0 - h + adap.cfg.w_e]
                pot = torch.asinh(b["app"][:, c0] / adap.pref)
                perda, _ = perda_evicon(out, b["y"], yj, adap.cfg, lam_ev=lam_ev,
                                        lam_p=lam_p, pot_alvo=pot)
                if ep < aquecimento:
                    perda = perda - F.binary_cross_entropy_with_logits(out["z"], b["y"])
                if lam_par > 0 and par is not None:
                    # contexto IDENTICO: so a janela de evidencia perde a ativacao
                    xs = b["x"].clone()
                    xs[par["idx"]] = par["sem"]
                    z_sem = _z_par(adap, b, xs)
                    d = out["z"][par["idx"]] - z_sem[par["idx"]]
                    perda = perda + lam_par * F.relu(margem - d).mean()
            elif adap.saida > 1:
                c0, k = b["x"].shape[1] // 2, adap.saida
                yk = b["yj"][:, c0 - k // 2:c0 - k // 2 + k]
                perda = F.binary_cross_entropy_with_logits(out["zk"], yk)
            else:
                perda = F.binary_cross_entropy_with_logits(out["z"], b["y"])
            opt.zero_grad(set_to_none=True)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            atualizar_ema(ema, modelo, ema_d)
            soma += float(perda.detach())
        y, p = prever(adap_ema, f_va)
        f1v = limiar_oraculo(p, y)[1]
        hist.append(dict(ep=ep, perda=soma, f1_val=f1v))
        if verboso:
            print(f"      ep {ep:2d} perda={soma:.2f} F1val={f1v:.3f}", flush=True)
        if ep < rampa_fim if evi and modelo.contexto is not None else ep < aquecimento:
            continue
        if f1v > melhor:
            melhor, espera = f1v, 0
            estado = copy.deepcopy(ema.state_dict())
        else:
            espera += 1
            if espera >= paciencia:
                break
    if estado is not None:
        ema.load_state_dict(estado)
    adap_ema.modelo = ema.eval()
    return dict(adap=adap_ema, historico=hist, f1_val=melhor, epocas=len(hist))


def _z_par(adap: Adaptador, b: dict, x_evid: torch.Tensor) -> torch.Tensor:
    """Logit com a janela de EVIDENCIA trocada por `x_evid` e o contexto do lote
    original: realiza o par (com, sem) de contexto identico da perda L_par."""
    x = b["x"]
    c0, h = x.shape[1] // 2, adap.cfg.w_e // 2
    sl = slice(c0 - h, c0 - h + adap.cfg.w_e)
    pr = torch.full((x.shape[0],), adap.pref, device=x.device)
    return adap.modelo(x_evid[:, sl], pr, x, b["p95"], b["lac"][:, sl], b["lac"])["z"]


def treinar_poe(adap: Adaptador, f_tr, f_va, spec, banco, semente, epocas, paciencia) -> dict:
    """A6: (1) so o ramo de contexto com b0 (evidencia desligada); (2) congela o
    encoder e a cabeca aditiva e treina a evidencia dentro da formula completa."""
    m = adap.modelo
    evid_orig = m.evidencia.forward

    def sem_evidencia(x, pr, lac=None):
        out = evid_orig(x, pr, lac)
        return {k: (v * 0 if k in ("e", "s", "b") else v) for k, v in out.items()}

    m.evidencia.forward = sem_evidencia
    s1 = dict(spec, poe=False)
    r1 = treinar(adap, f_tr, f_va, s1, banco, semente, max(4, epocas // 3), paciencia)
    m.load_state_dict(r1["adap"].modelo.state_dict())
    m.evidencia.forward = evid_orig
    for n, p in m.named_parameters():
        if n.startswith("contexto.") and not n.startswith("contexto.m."):
            p.requires_grad_(False)
    return treinar(adap, f_tr, f_va, spec, banco, semente, epocas, paciencia)


# --------------------------------------------------------------------------------------
# avaliacao
# --------------------------------------------------------------------------------------


def metricas_base(y, p, t) -> dict:
    prev = float(y.mean())
    return dict(n=len(y), prevalencia=round(prev, 4), ativacoes=ativacoes(y, t),
                trivial=round(2 * prev / (1 + prev), 4), auc=round(auc(y, p), 4),
                ap=round(float(average_precision_score(y, p)) if 0 < y.sum() < len(y)
                         else float("nan"), 4),
                f1_oraculo=round(limiar_oraculo(p, y)[1], 4))


def com_decisao(prefixo: str, dec: Decisor, y, p) -> dict:
    d = dec.decidir(p)
    out = {f"{prefixo}_f1_{r}": round(f1(y, d[r]), 4) for r in "abcde"}
    out.update({f"{prefixo}_{k}": v for k, v in d["log"].items()
                if k in ("pi_em", "pi_ca", "trava", "f1_esperado", "limiar_c")})
    return out


def registrar_olhada(descricao: str):
    SAIDA.mkdir(parents=True, exist_ok=True)
    arq = SAIDA / "olhadas.csv"
    linha = pd.DataFrame([dict(quando=time.strftime("%Y-%m-%d %H:%M:%S"), o_que=descricao)])
    linha.to_csv(arq, mode="a", header=not arq.exists(), index=False)


# --------------------------------------------------------------------------------------
# um aparelho
# --------------------------------------------------------------------------------------


def preparar(aparelho: str, fonte: str, extras: list[int], val: list[int], olhar: bool,
             passo: int, fonte_ds: str = "refit", fonte_casa: int = 8,
             testes: str = "") -> dict:
    lim = LIMIAR_ON[aparelho]
    casas_fonte = [fonte_casa] + (extras if fonte == "A+" else [])
    trs, vas, ok = [], [], []
    for c in casas_fonte:
        ds = fonte_ds if c == fonte_casa else "refit"
        s = carregar_w(arquivo(ds, c, aparelho, "treino"), casa_id=c)
        if s is None:
            continue
        tr, va = holdout_temporal(s, semanas=4, embargo=W_C + 1)
        # E0.3 - bloqueante: nenhuma amostra bruta compartilhada
        from ponto import alvos_validos
        a_tr = alvos_validos(tr, W_C // 2, W_C - W_C // 2 - 1)
        a_va = alvos_validos(va, W_C // 2, W_C - W_C // 2 - 1)
        n = amostras_compartilhadas(a_tr, a_va, W_C // 2, W_C - W_C // 2 - 1)
        assert n == 0, f"vazamento na casa {c}: {n} amostras"
        tr.casa_id = va.casa_id = c
        trs.append(tr)
        vas.append(va)
        ok.append(c)
    tr = juntar_w(trs, "treino") if len(trs) > 1 else trs[0]
    va = juntar_w(vas, "val") if len(vas) > 1 else vas[0]
    if len(trs) == 1:
        tr.casa_id_por_amostra = np.full(tr.vistas.shape[1], fonte_casa, np.int64)
    pref = p_ref(trs, lim)
    dur = np.concatenate([duracoes(s.rotulo[s.centros], s.t[s.centros]) for s in trs])
    d90 = int(np.percentile(dur, 90)) if len(dur) else 12
    prev = float(np.mean(np.concatenate([s.rotulo[s.centros] for s in trs])))
    banco = Banco(trs, lim)
    cv = {}
    for c in val:
        s = carregar_w(arquivo("refit", c, aparelho, "treino"), casa_id=c)
        if s is not None and 0 < s.rotulo.mean() < 1:
            cv[f"cv: refit {c}"] = s
    teste = {}
    if olhar and testes:
        # alvos explicitos "chave@arquivo@modo;..." (mesmo formato de experimento_reco.py)
        p95_f = float(np.percentile(trs[0].vistas[0][trs[0].centros], 95))
        slug = aparelho.replace(" ", "_")
        for item in testes.split(";"):
            chave, nome, modo = (v.strip() for v in item.split("@"))
            s = carregar_w(nome.replace("{ap}", slug), p95=p95_f if modo == "fonte" else None,
                           casa_id=9000 + len(teste))
            if s is not None:
                teste[chave] = s
    elif olhar:
        p95_8 = float(np.percentile(trs[0].vistas[0][trs[0].centros], 95))
        s = carregar_w(arquivo("refit", 8, aparelho, "teste"), p95=p95_8, casa_id=8)
        if s is not None:
            teste["intra: refit casa 8"] = s
        for n in (1, 2, 5):
            s = carregar_w(arquivo("eco", n, aparelho, "cross"), casa_id=100 + n)
            if s is not None:
                teste[f"cross: eco casa {n}"] = s
    x_casa = tr.vistas[0][tr.centros] / tr.vistas[3][tr.centros]
    return dict(tr=Fonte(tr, W_C, passo, lim), va=Fonte(va, W_C, 1, lim),
                va_serie=va, cv={k: Fonte(s, W_C, 1, lim) for k, s in cv.items()},
                teste={k: Fonte(s, W_C, 1, lim) for k, s in teste.items()},
                pref=pref, d90=d90, prev=prev, banco=banco, casas=ok,
                norm_casa=(float(x_casa.mean()), float(x_casa.std() + 1e-6)))


def construir(nome: str, spec: dict, aparelho: str, D: dict) -> Adaptador:
    if spec["tipo"] == "cnn":
        ctx, k = spec.get("ctx", W_C), spec.get("saida", 1)
        m = ClassificadorPonto(["casa"], ctx, saida=k).to(DISPOSITIVO)
        return Adaptador(m, "cnn", None, D["pref"], D["norm_casa"], ctx=ctx, saida=k)
    h = hiper(aparelho, D["d90"], D["prev"])
    cfg = ConfigEviCon(w_e=h["w_e"], w_c=W_C, mascara=h["mascara"], meia_vida=h["meia_vida"],
                       gama=h["gama"], c=spec.get("c", h["c"]), prior=D["prev"],
                       **{k: v for k, v in spec.items()
                          if k in ("acumulador", "sinal", "contexto")})
    return Adaptador(EviCon(cfg).to(DISPOSITIVO), "evicon", cfg, D["pref"])


def rodar_aparelho(aparelho: str, args) -> list[dict]:
    t0 = time.time()
    pool = list(POOL.get(aparelho, []))
    if args.sorteio > 0:            # E1.2: separa "quantas casas" de "quais casas"
        pool = list(np.random.default_rng(args.sorteio).permutation(pool))
    extras = ([int(c) for c in pool[:args.n_extra]] if args.casas_extra == "auto"
              else [int(c) for c in args.casas_extra.split(",") if c])
    D = preparar(aparelho, args.fonte, extras,
                 [int(c) for c in args.casas_val.split(",") if c], args.olhar_teste,
                 args.passo, args.fonte_ds, args.fonte_casa, args.testes)
    print(f"\n=== {aparelho} | fonte {args.fonte} (casas {D['casas']}) | P_ref={D['pref']:.0f} W "
          f"| D90={D['d90']} intervalos | prev={100 * D['prev']:.1f}% | "
          f"{resumo_banco(D['banco'])} | {len(D['tr'])} janelas de treino "
          f"({time.time() - t0:.0f}s)", flush=True)
    print(f"    hiper EviCon: {hiper(aparelho, D['d90'], D['prev'])}", flush=True)
    if args.olhar_teste:
        registrar_olhada(f"{aparelho} fonte={args.fonte} casa_fonte={args.fonte_ds} "
                         f"{args.fonte_casa} modelos={args.modelos} sementes={args.sementes} "
                         f"testes={args.testes or 'padrao'} tag={args.tag}")
    linhas = []
    pasta_pred = SAIDA / "pred" / args.tag
    pasta_pred.mkdir(parents=True, exist_ok=True)
    for nome in args.modelos.split(","):
        spec = MODELOS[nome.strip()]
        for semente in range(args.sementes):
            torch.manual_seed(semente)
            np.random.seed(semente)
            adap = construir(nome, spec, aparelho, D)
            t1 = time.time()
            fn = treinar_poe if spec.get("poe") else treinar
            r = fn(adap, D["tr"], D["va"], spec, D["banco"], semente, args.epocas,
                   args.paciencia)
            ad = r["adap"]
            seg = time.time() - t1
            # calibracao faixa A: holdout temporal da fonte
            y_va, p_va = prever(ad, D["va"])
            dec_a = Decisor(p_va, y_va, auc_retida=None)
            preds = {"val": (y_va, p_va, D["va"].t[D["va"].alvos_np])}
            for k, f in {**D["cv"], **D["teste"]}.items():
                y, p = prever(ad, f)
                preds[k] = (y, p, f.t[f.alvos_np])
            base = dict(aparelho=aparelho, fonte=args.fonte, modelo=nome, semente=semente,
                        epocas=r["epocas"], f1_val=round(r["f1_val"], 4), segundos=round(seg, 1))
            # intervencao (diagnostico) no conjunto de cross-val
            itv = {}
            if D["cv"]:
                f_cv = next(iter(D["cv"].values()))
                itv = intervencao(ad.escore, f_cv, D["banco"], n=2000, semente=semente)
                if aparelho == "washing machine":
                    for rot, faixa in (("madrugada", (0, 8)), ("dia", (10, 20))):
                        x = intervencao(ad.escore, f_cv, D["banco"], n=2000, semente=semente,
                                        filtro_hora=faixa)
                        itv[f"delta_add_{rot}"] = x.get("delta_add")
            for k, (y, p, t) in preds.items():
                if k == "val":
                    continue
                linha = dict(**base, alvo=k, **metricas_base(y, p, t),
                             **com_decisao("A", dec_a, y, p),
                             **{kk: (round(v, 3) if isinstance(v, float) else v)
                                for kk, v in itv.items()})
                if k.startswith("cv:"):
                    outros = [kk for kk in D["cv"] if kk != k]
                    if outros:
                        yo = np.concatenate([preds[o][0] for o in outros])
                        po = np.concatenate([preds[o][1] for o in outros])
                        au = np.nanmean([auc(preds[o][0], preds[o][1]) for o in outros])
                        linha.update(com_decisao("L", Decisor(po, yo, auc_retida=au), y, p))
                elif D["cv"]:
                    yo = np.concatenate([preds[o][0] for o in D["cv"]])
                    po = np.concatenate([preds[o][1] for o in D["cv"]])
                    au = np.nanmean([auc(preds[o][0], preds[o][1]) for o in D["cv"]])
                    linha.update(com_decisao("L", Decisor(po, yo, auc_retida=au), y, p))
                linhas.append(linha)
            np.savez_compressed(
                pasta_pred / f"{aparelho.replace(' ', '_')}__{nome}__s{semente}.npz",
                **{f"{k}|{c}": v for k, (y, p, t) in preds.items()
                   for c, v in (("y", y), ("p", p), ("t", t))})
            cvs = [l for l in linhas[-len(preds) + 1:] if l["alvo"].startswith("cv")]
            msg = "  ".join(f"{l['alvo'][4:]}: auc={l['auc']:.3f} c={l.get('L_f1_c', np.nan):.3f}"
                            for l in cvs)
            tst = [l for l in linhas[-len(preds) + 1:] if not l["alvo"].startswith("cv")]
            msg2 = "  ".join(f"{l['alvo']}: auc={l['auc']:.3f} a={l['A_f1_a']:.3f} "
                             f"c={l['A_f1_c']:.3f}" for l in tst)
            print(f"  {nome:5s} s{semente} ep={r['epocas']:2d} F1val={r['f1_val']:.3f} "
                  f"{seg:4.0f}s | {msg} | drem={itv.get('delta_rem', np.nan):+.2f} "
                  f"dadd={itv.get('delta_add', np.nan):+.2f} {('| ' + msg2) if msg2 else ''}",
                  flush=True)
            pd.DataFrame(linhas).to_csv(SAIDA / f"{args.tag}__{aparelho.replace(' ', '_')}.csv",
                                        index=False)
    return linhas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aparelhos", default="television,washing machine,freezer")
    p.add_argument("--modelos", default="A0,A1,A2,A3,A4,A5")
    p.add_argument("--fonte", default="A", choices=["A", "A+"])
    p.add_argument("--casas_extra", default="auto",
                   help="'auto' = POOL do portao G0, ou lista explicita")
    p.add_argument("--n_extra", type=int, default=99, help="quantas casas do POOL (E1.2)")
    p.add_argument("--sorteio", type=int, default=0, help="> 0 embaralha o POOL com essa semente")
    p.add_argument("--casas_val", default="3,7,20")
    p.add_argument("--fonte_ds", default="refit", help="conjunto da casa-fonte principal")
    p.add_argument("--fonte_casa", type=int, default=8, help="casa-fonte principal")
    p.add_argument("--testes", default="",
                   help="alvos de teste 'chave@arquivo@modo;...' (padrao: REFIT 8 + ECO)")
    p.add_argument("--sementes", type=int, default=5)
    p.add_argument("--epocas", type=int, default=40)
    p.add_argument("--paciencia", type=int, default=6)
    p.add_argument("--passo", type=int, default=4)
    p.add_argument("--olhar_teste", action="store_true")
    p.add_argument("--tag", default="escada")
    args = p.parse_args()
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / f"{args.tag}__args.json").write_text(json.dumps(vars(args), indent=1))
    todas = []
    for ap in args.aparelhos.split(","):
        todas += rodar_aparelho(ap.strip(), args)
    d = pd.DataFrame(todas)
    d.to_csv(SAIDA / f"{args.tag}.csv", index=False)
    cols = [c for c in ("auc", "ap", "f1_oraculo", "A_f1_a", "A_f1_c", "L_f1_c", "L_f1_d",
                        "delta_rem", "delta_add") if c in d]
    print(d.groupby(["aparelho", "alvo", "modelo"])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
