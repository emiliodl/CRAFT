"""F2/F3 do v3 (docs/v3.md §5, §6): escada B1-B6 do ReCo e variante intra.

Modelos (B0 e B7 vem das predicoes salvas do v2 - ver src/relatorio_v3.py):

  B1   CNN do B0 (ClassificadorPonto, 4 h) com o gerador R0-R4, sem termos de par
  B2   B1 + L_inv
  B3   B2 + L_sens
  R<k> ResNet 1D de kernel k com a receita do B3; B4 = R9, B5 = media de postos
       de R5, R7, R9, R15, R25
  A<k> R<k> + cabecas auxiliares (B6 = conjunto dos A<k>)
  I<k> variante intra (§3.7): ResNet k, so R0/R1/R2, hora do dia, contexto `--ctx`
  B2r, Q<k>  so R0 + R3 (+ L_inv): as operacoes que passam no controle do
       discriminador (E2.0); CNN e ResNet
  G<k> ResNet com o gerador completo e sem termos de par. Pos-v3 e o modelo
       adotado (CRAFT = conjunto G5..G25): L_inv, L_sens e as cabecas auxiliares
       nao tiveram efeito resolvivel na escada (HR2, HR7) e foram descartados do
       metodo; B2/B3/R/A ficam aqui so para reproduzir o v3

Divisoes (iguais as do v2): treino/val = holdout temporal (4 ultimas semanas de
cada casa fonte, embargo de 1 dia + 1 janela); validacao cross = REFIT 3/7/20
fixas, nunca no treino nem nos bancos (teste 4). Na variante intra, a val e o
holdout dos 15% finais do ano da casa 8 (§3.7) e nada e avaliado no cross.

As predicoes nas casas de validacao usam os MESMOS alvos do v2 (janela de 24 h
valida, recorte central de 4 h), para o bootstrap pareado contra B0/B7.

Uso:
    python src/experimento_reco.py --aparelhos television --modelos B1,B2,B3,R9 --fonte A
    python src/experimento_reco.py --aparelhos television --modelos I9 --intra --fonte A
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dados_v2 import Fonte, arquivo, carregar_w                        # noqa: E402
from decisao import Decisor, auc, f1, limiar_oraculo                   # noqa: E402
from estatistica import ativacoes                                      # noqa: E402
from ponto import ClassificadorPonto                                   # noqa: E402
from reco import (W5, ConfigPerda, Gerador, ResNet1D, ResNetEvid, ResNetGlob,  # noqa: E402
                  carregar_casa, periodicidade, perda_transicoes,  # noqa: E402
                  definir_periodo, entradas, perda_reco, stats_casa)
from taxa import ROTULO, SUB, TAXA, amostras                                   # noqa: E402
from treino import DISPOSITIVO                                         # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "resultados" / "v3"
W_C = amostras(288)        # 24 h
POOL = {   # portao G0 do v2 (fontes aprovadas)
    "television": [1, 2, 5, 13, 15, 16, 19, 21],
    "washing machine": [1, 2, 4, 5, 6, 9, 10, 11, 13, 15, 16, 17, 18, 19, 21],
    "freezer": [1, 4, 6, 13, 17, 18],
}
KERNELS = (5, 7, 9, 15, 25)


# R0 + R3 apenas: as operacoes que passam no controle do discriminador (E2.0)
SO_TROCA = dict(R0=0.60, R3=0.40, R1=0.0, R2=0.0, R4=0.0)


def spec_modelo(nome: str) -> dict:
    if nome in ("B1", "B2", "B3"):
        return dict(tipo="cnn", inv=nome in ("B2", "B3"), sens=nome == "B3", aux=False)
    if nome == "B2r":
        return dict(tipo="cnn", inv=True, sens=False, aux=False, ops=SO_TROCA)
    if nome == "C0":             # referencia sem gerador (so janelas reais)
        return dict(tipo="cnn", inv=False, sens=False, aux=False,
                    ops=dict(R0=1.0, R1=0.0, R2=0.0, R3=0.0, R4=0.0))
    tipo, k = nome[0], int(nome[1:])
    if tipo == "R":
        return dict(tipo="resnet", k=k, inv=True, sens=True, aux=False)
    if tipo == "A":
        return dict(tipo="resnet", k=k, inv=True, sens=True, aux=True)
    if tipo == "I":
        return dict(tipo="resnet", k=k, inv=False, sens=True, aux=False, intra=True)
    if tipo == "G":              # gerador sem termos de par, ResNet (E2.3, barato)
        return dict(tipo="resnet", k=k, inv=False, sens=False, aux=False)
    if tipo == "F":              # pos-v3: CRAFT + canais do ramo de evidencia (fusao)
        return dict(tipo="fusao", k=k, inv=False, sens=False, aux=False)
    if tipo == "Q":              # ResNet so com R0 + R3 e L_inv
        return dict(tipo="resnet", k=k, inv=True, sens=False, aux=False, ops=SO_TROCA)
    raise ValueError(nome)


# --------------------------------------------------------------------------------------
# dados
# --------------------------------------------------------------------------------------


def preparar(aparelho: str, fonte: str, extras: list[int], val: list[int], args) -> dict:
    intra = args.intra
    f0 = args.fonte_fina                       # casa-fonte principal (padrao: REFIT 8)
    casas_f = [f0] + (extras if fonte == "A+" else [])
    casas = []
    for c in casas_f:
        cf = carregar_casa(c, aparelho)
        if cf is None:
            continue
        cf.periodo = definir_periodo(cf, "intra" if intra and c == f0 else "cross",
                                     janela=max(args.ctx, W5))
        if intra and c != f0:                  # na intra, as outras casas so treinam
            cf.periodo["val"] = np.zeros_like(cf.ok)
        casas.append(cf)
    assert not ({c.casa for c in casas} & set(val)), "casa de validacao na fonte"
    ops = json.loads(args.prop) if args.prop else None
    if intra:                                  # §3.7: so R0, R1 e R2
        ops = dict(R0=0.55, R1=0.25, R2=0.20, R3=0.0, R4=0.0)
    g_tr = Gerador(casas, aparelho, "treino", DISPOSITIVO, proporcoes=ops,
                   passo=args.passo, frac_real=args.frac_real,
                   escala=tuple(args.escala) if args.escala else None, w5=args.ctx,
                   deform=json.loads(args.deform) if args.deform else None,
                   trem=json.loads(args.trem) if args.trem else None)
    g_va = Gerador([c for c in casas if c.periodo["val"].any()], aparelho, "val",
                   DISPOSITIVO, w5=args.ctx)
    # estatisticas por casa (so agregado) e P_ref (mediana do alvo ligado, treino)
    st = []
    for c in casas:
        m = c.periodo["treino"]
        n5 = len(c.A) // SUB
        w = c.W[:n5 * SUB].reshape(n5, SUB)
        x5 = (c.A[:n5 * SUB].reshape(n5, SUB) * w).sum(1) / np.maximum(w.sum(1), 1e-6)
        ok5 = m[:n5 * SUB].reshape(n5, SUB).all(1)
        st.append(stats_casa(x5[ok5]))
    stt = {k: torch.tensor([s[k] for s in st], device=DISPOSITIVO) for k in st[0]}
    p5 = g_tr.res(g_tr.P, g_tr.centros[::7])[:, g_tr.antes]
    pref = float(p5[p5 >= g_tr.lim].median()) if (p5 >= g_tr.lim).any() else g_tr.lim
    # normalizacao do B0/B1 (vista casa do v2): media/dp de x/p95 no treino
    xb = g_tr.res(g_tr.A, g_tr.centros[::7]) / stt["p95"][g_tr.casa_de[g_tr.centros[::7]]][:, None]
    norm = (float(xb.mean()), float(xb.std() + 1e-6))
    # validacao cross e teste pelo protocolo de 5 min (mesmos alvos do v2)
    cv = {}
    if not intra:
        for c in val:
            s = carregar_w(arquivo("refit", c, aparelho, "treino"), casa_id=c)
            if s is not None and 0 < s.rotulo.mean() < 1:
                cv[f"cv: refit {c}"] = s
    teste = {}
    if args.olhar_teste and args.testes:
        # alvos explicitos: "chave@arquivo@modo;..." com {ap} = aparelho no nome do
        # arquivo; modo "fonte" usa as estatisticas da casa-fonte (mesma casa, intra),
        # "proprio" as do agregado do alvo (outra casa)
        slug = aparelho.replace(" ", "_")
        for item in args.testes.split(";"):
            chave, nome, modo = (v.strip() for v in item.split("@"))
            s = carregar_w(nome.replace("{ap}", slug),
                           p95=float(st[0]["p95"]) if modo == "fonte" else None,
                           casa_id=9000 + len(teste))
            if s is not None:
                teste[chave] = (s, st[0] if modo == "fonte" else None)
    elif args.olhar_teste:
        s8 = carregar_w(arquivo("refit", 8, aparelho, "teste"), p95=float(st[0]["p95"]),
                        casa_id=8)
        if s8 is not None:
            teste["intra: refit casa 8"] = (s8, st[0])
        if not intra:
            for n in (1, 2, 5):
                s = carregar_w(arquivo("eco", n, aparelho, "cross"), casa_id=100 + n)
                if s is not None:
                    teste[f"cross: eco casa {n}"] = (s, None)
            # casas-alvo novas do UK-DALE (scripts/preparar_ukdale_v3.py): nunca fonte
            for b in (1, 2):
                nome = f"ukdale_casa_{b}__{aparelho.replace(' ', '_')}__{ROTULO}__alvo.npz"
                s = carregar_w(nome, casa_id=200 + b)
                if s is not None:
                    teste[f"novo: ukdale casa {b}"] = (s, None)
    alvos = {}
    for k, s in cv.items():
        alvos[k] = (Fonte(s, W_C, 1, g_tr.lim), stats_casa(s.vistas[0][s.centros]))
    for k, (s, stc) in teste.items():
        alvos[k] = (Fonte(s, W_C, 1, g_tr.lim), stc or stats_casa(s.vistas[0][s.centros]))
    return dict(g_tr=g_tr, g_va=g_va, st=stt, pref=pref, norm=norm, alvos=alvos,
                casas=[c.casa for c in casas])


# --------------------------------------------------------------------------------------
# modelo + entradas
# --------------------------------------------------------------------------------------


class Modelo:
    def __init__(self, spec: dict, D: dict, ctx: int = W5):
        self.spec, self.D, self.ctx = spec, D, ctx
        self.n_aux = len(D["g_tr"].aux) if spec.get("aux") else 0
        self.hora = bool(spec.get("intra"))
        self.glob = bool(spec.get("glob"))
        # pos-quali: saida extra (a ULTIMA) que regride log1p(P/P_ref) no intervalo central
        self.reg = bool(spec.get("reg"))
        n_saidas = self.n_aux + self.reg
        # pos-quali: entrada so com subidas e descidas (sem os canais de nivel)
        self.dif = bool(spec.get("dif"))
        n_in = (2 if self.dif else 4) + 2 * self.hora
        if self.glob:
            self.rede = ResNetGlob(n_in, spec["k"], n_glob=2 * (ctx // 2),
                                   n_aux=n_saidas).to(DISPOSITIVO)
        elif spec["tipo"] == "cnn":
            self.rede = ClassificadorPonto(["casa"], ctx).to(DISPOSITIVO)
        elif spec["tipo"] == "fusao":
            self.rede = ResNetEvid(4 + 2 * self.hora, spec["k"], n_aux=self.n_aux).to(DISPOSITIVO)
        else:
            self.rede = ResNet1D(n_in, spec["k"], n_aux=n_saidas).to(DISPOSITIVO)

    def logits(self, x5: torch.Tensor, s, m, d, p95, hora=None) -> torch.Tensor:
        if self.spec["tipo"] == "cnn":
            mu, dp = self.D["norm"]
            return self.rede(((x5 / p95[:, None] - mu) / dp)[:, None, :])
        xin = entradas(x5, s, m, d, self.D["pref"], hora if self.hora else None)
        if self.dif:
            xin = xin[:, [c for c in range(xin.shape[1]) if c not in (0, 1)]]
        if self.spec["tipo"] == "fusao":
            return self.rede(xin, x5, self.D["pref"])
        if self.glob:
            return self.rede(xin, periodicidade(x5))
        return self.rede(xin)

    def de_lote_gerado(self, lote: dict, hora=None) -> torch.Tensor:
        st = self.D["st"]
        h = lote["casa"]
        return self.logits(lote["x"], st["s"][h], st["m"][h], st["d"][h], st["p95"][h], hora)

    def de_fonte(self, x: torch.Tensor, stc: dict, hora=None) -> torch.Tensor:
        """x (B, 288) do protocolo -> recorte central de ctx amostras."""
        c0 = x.shape[1] // 2
        x5 = x[:, c0 - self.ctx // 2:c0 - self.ctx // 2 + self.ctx]
        b = x5.shape[0]
        full = lambda v: torch.full((b,), float(v), device=x.device)   # noqa: E731
        return self.logits(x5, full(stc["s"]), full(stc["m"]), full(stc["d"]), full(stc["p95"]),
                           hora)


def hora_fonte(fonte: Fonte, pos: torch.Tensor, ctx: int) -> torch.Tensor:
    t = torch.from_numpy(fonte.t[fonte.alvos_np[pos.cpu().numpy()]]).to(DISPOSITIVO)
    rel = torch.arange(ctx, device=DISPOSITIVO) - ctx // 2
    return ((t[:, None] + rel[None] * TAXA) % 86400).float() / 3600.0


@torch.no_grad()
def prever_fonte(mod: Modelo, fonte: Fonte, stc: dict, lote: int = 2048) -> np.ndarray:
    mod.rede.eval()
    zs = []
    for i in range(0, len(fonte), lote):
        pos = torch.arange(i, min(i + lote, len(fonte)), device=DISPOSITIVO)
        h = hora_fonte(fonte, pos, mod.ctx) if mod.hora else None
        zs.append(mod.de_fonte(fonte.lote(pos)["x"], stc, h)[:, 0].float())
    return torch.cat(zs).cpu().numpy()


@torch.no_grad()
def prever_gerado(mod: Modelo, g: Gerador, lote: int = 2048) -> tuple[np.ndarray, np.ndarray,
                                                                      np.ndarray]:
    mod.rede.eval()
    zs, ys = [], []
    for i in range(0, len(g.centros), lote):
        c = g.centros[i:i + lote]
        L = g.lote_real(c)
        L["casa"] = casa_global(mod.D, g, c)
        zs.append(mod.de_lote_gerado(L, g.hora(c) if mod.hora else None)[:, 0].float())
        ys.append(L["y"])
    t = (g.tsec[g.centros]).cpu().numpy()
    return torch.cat(ys).cpu().numpy(), torch.cat(zs).cpu().numpy(), t


def casa_global(D: dict, g: Gerador, c: torch.Tensor) -> torch.Tensor:
    """Indice da casa no vetor de estatisticas de D (o de treino) para centros de `g`."""
    mapa = torch.tensor([D["casas"].index(h) for h in g.casas_ids], device=DISPOSITIVO)
    return mapa[g.casa_de[c]]


def atualizar_ema(ema, rede, d: float):
    with torch.no_grad():
        for pe, p in zip(ema.parameters(), rede.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)
        for be, b in zip(ema.buffers(), rede.buffers()):
            be.copy_(b)


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def treinar(mod: Modelo, D: dict, semente: int, args, cfg: ConfigPerda) -> dict:
    g_tr, g_va = D["g_tr"], D["g_va"]
    ger = torch.Generator(device=DISPOSITIVO).manual_seed(10_000 + semente)
    rede = mod.rede
    ema = copy.deepcopy(rede)
    mod_ema = copy.copy(mod)
    mod_ema.rede = ema
    opt = torch.optim.AdamW(rede.parameters(), lr=args.taxa, weight_decay=1e-4)
    spec = mod.spec
    prop_orig = g_tr.prop
    if spec.get("ops"):
        g_tr.prop = spec["ops"]
    n0 = max(8, int(round(args.lote * g_tr.prop["R0"])))
    melhor, estado, espera, hist = -1.0, None, 0, []
    for ep in range(args.epocas):
        rede.train()
        ordem = g_tr.base[torch.randperm(len(g_tr.base), device=DISPOSITIVO, generator=ger)]
        if args.lotes_epoca:
            ordem = ordem[:args.lotes_epoca * n0]
        soma, nb, partes_ep = 0.0, 0, {}
        for i in range(0, len(ordem) - n0 + 1, n0):
            L = g_tr.lote(ordem[i:i + n0], ger)
            hora = None
            if mod.hora:
                # janelas recompostas: hora do FUNDO (R1, R4, Rf) ou da base (R0, R2)
                hora = hora_lote(g_tr, L)
            z = mod.de_lote_gerado(L, hora)
            zaux = z[:, 1:1 + mod.n_aux]
            perda, partes = perda_reco(z[:, 0], zaux, L, cfg, spec["aux"], spec["inv"],
                                       spec["sens"])
            if mod.reg:
                alvo_r = torch.log1p(L["a"][:, g_tr.antes].clamp_min(0.0) / mod.D["pref"])
                lr_ = F.mse_loss(z[:, -1], alvo_r)
                perda = perda + args.lam_reg * lr_
                partes["reg"] = float(lr_.detach())
            if spec["tipo"] == "fusao":
                lt = perda_transicoes(rede.ultimo, (L["a"] >= g_tr.lim).float())
                perda = perda + 0.3 * lt
                partes["trans"] = float(lt.detach())
            opt.zero_grad(set_to_none=True)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(rede.parameters(), 1.0)
            opt.step()
            atualizar_ema(ema, rede, args.ema)
            soma += float(perda.detach())
            nb += 1
            for k, v in partes.items():
                partes_ep[k] = partes_ep.get(k, 0.0) + v
        y, z, _ = prever_gerado(mod_ema, g_va)
        # criterio de parada: F1 com o melhor limiar (padrao) ou AUC (pos-quali, item 3:
        # no freezer o F1 fica colado no trivial e nao separa as epocas)
        f1v = auc(y, sig(z)) if args.parada == "auc" else limiar_oraculo(sig(z), y)[1]
        hist.append(dict(ep=ep, perda=soma / max(nb, 1), f1_val=f1v,
                         **{k: v / max(nb, 1) for k, v in partes_ep.items()}))
        if args.verboso:
            print(f"      ep {ep:2d} perda={soma / max(nb, 1):.3f} F1val={f1v:.3f} "
                  f"{ {k: round(v / max(nb, 1), 3) for k, v in partes_ep.items()} }", flush=True)
        if f1v > melhor:
            melhor, espera = f1v, 0
            estado = copy.deepcopy(ema.state_dict())
        else:
            espera += 1
            if espera >= args.paciencia:
                break
    g_tr.prop = prop_orig
    if estado is not None:
        ema.load_state_dict(estado)
    ema.eval()
    return dict(mod=mod_ema, hist=hist, f1_val=melhor, epocas=len(hist))


def hora_lote(g: Gerador, L: dict) -> torch.Tensor:
    """A hora de cada janela gerada: o lote guarda os centros em L['centro']."""
    return g.hora(L["centro"])


# --------------------------------------------------------------------------------------
# pos-quali: adaptacao a casa de destino (so casas de validacao; src/adaptacao_posquali.py)
# --------------------------------------------------------------------------------------

N_ADAPT = 14               # dias iniciais da casa reservados a adaptacao


def dias_fonte(f: Fonte) -> np.ndarray:
    _, inv = np.unique(f.t[f.alvos_np] // 86400, return_inverse=True)
    return inv


def copia(mod: Modelo) -> Modelo:
    m = copy.copy(mod)
    m.rede = copy.deepcopy(mod.rede)
    return m


@torch.no_grad()
def adabn(mod: Modelo, f: Fonte, stc: dict, pos: torch.Tensor, lote: int = 1024) -> Modelo:
    """Forma 1: zera e reestima as estatisticas de BatchNorm com o agregado (sem
    rotulo) das janelas `pos` da casa de destino."""
    m = copia(mod)
    for b in m.rede.modules():
        if isinstance(b, torch.nn.BatchNorm1d):
            b.reset_running_stats()
            b.momentum = None                  # media cumulativa
    m.rede.train()
    for i in range(0, len(pos), lote):
        m.de_fonte(f.lote(pos[i:i + lote])["x"], stc)
    m.rede.eval()
    return m


def ajuste_fino(mod: Modelo, f: Fonte, stc: dict, pos: torch.Tensor, args, semente: int
                ) -> Modelo:
    """Forma 3: ajuste fino de toda a rede (BatchNorm congelado) com as janelas
    rotuladas `pos` da casa de destino, BCE, taxa baixa."""
    m = copia(mod)
    m.rede.eval()                              # BN congelado
    opt = torch.optim.AdamW(m.rede.parameters(), lr=args.ft_taxa, weight_decay=1e-4)
    g = torch.Generator(device=DISPOSITIVO).manual_seed(20_000 + semente)
    y = f.y[f.alvos]
    for _ in range(args.ft_passos):
        j = pos[torch.randint(0, len(pos), (min(128, len(pos)),), device=DISPOSITIVO,
                              generator=g)]
        z = m.de_fonte(f.lote(j)["x"], stc)[:, 0]
        perda = F.binary_cross_entropy_with_logits(z, y[j].float())
        opt.zero_grad(set_to_none=True)
        perda.backward()
        torch.nn.utils.clip_grad_norm_(m.rede.parameters(), 1.0)
        opt.step()
    m.rede.eval()
    return m


def adaptar(mod: Modelo, D: dict, args, semente: int, preds: dict) -> None:
    """Acrescenta a `preds` as chaves cvbn: (AdaBN) e cvft<k>: (ajuste fino com k dias)
    de cada casa de validacao. Sem positivo nos k dias, cvft<k> repete a base."""
    for k, (f, stc) in D["alvos"].items():
        if not k.startswith("cv:"):
            continue
        h = k.split(":", 1)[1]
        dia = torch.from_numpy(dias_fonte(f)).to(DISPOSITIVO)
        yb, zb, tb = preds[k]
        p14 = torch.nonzero(dia < N_ADAPT).squeeze(1)
        preds[f"cvbn:{h}"] = (yb, prever_fonte(adabn(mod, f, stc, p14), f, stc), tb)
        y = f.y[f.alvos]
        for kd in (1, 3, 7, 14):
            pk = torch.nonzero(dia < kd).squeeze(1)
            if y[pk].sum() == 0:
                preds[f"cvft{kd}:{h}"] = (yb, zb, tb)
                continue
            preds[f"cvft{kd}:{h}"] = (yb, prever_fonte(ajuste_fino(mod, f, stc, pk, args,
                                                                    semente), f, stc), tb)


# --------------------------------------------------------------------------------------
# intervencao (E2.4)
# --------------------------------------------------------------------------------------


@torch.no_grad()
def intervencao(mod: Modelo, D: dict, n: int = 2000, semente: int = 0) -> dict:
    """Por casa de validacao: Delta_rem, Delta_add e Delta_fundo (logits).
    Delta_fundo: janelas ligadas da casa h com o fundo trocado pelo de uma janela
    limpa de OUTRA casa de validacao; so o fundo muda, o alvo e o mesmo."""
    g = torch.Generator(device=DISPOSITIVO).manual_seed(semente)
    alvos = {k: v for k, v in D["alvos"].items() if k.startswith("cv:")}
    out = {}
    fundos = {}
    for k, (f, stc) in alvos.items():
        y = f.y[f.alvos]
        lim = D["g_tr"].lim
        des = torch.nonzero(y == 0).squeeze(1)
        des = des[torch.randperm(len(des), device=DISPOSITIVO, generator=g)[:n]]
        L = f.lote(des)
        limpo = (L["app"] < lim).all(1)
        fundos[k] = (L["x"] - L["app"])[limpo]
    for k, (f, stc) in alvos.items():
        y = f.y[f.alvos]
        lig = torch.nonzero(y > 0).squeeze(1)
        lig = lig[torch.randperm(len(lig), device=DISPOSITIVO, generator=g)[:n]]
        des = torch.nonzero(y == 0).squeeze(1)
        des = des[torch.randperm(len(des), device=DISPOSITIVO, generator=g)[:n]]
        z = lambda x: mod.de_fonte(x, stc)[:, 0]                        # noqa: E731
        r = {}
        if len(lig):
            L = f.lote(lig)
            z1 = z(L["x"])
            r["delta_rem"] = float((z1 - z((L["x"] - L["app"]).clamp_min(0))).mean())
            outros = [fundos[o] for o in fundos if o != k and len(fundos[o])]
            if outros:
                fb = torch.cat(outros)
                j = torch.randint(0, len(fb), (len(lig),), device=DISPOSITIVO, generator=g)
                x2 = (fb[j] + L["app"]).clamp_min(0)
                r["delta_fundo"] = float((z1 - z(x2)).abs().mean())
                r["delta_fundo_med"] = float((z1 - z(x2)).mean())
        if len(des):
            L = f.lote(des)
            fundo = (L["x"] - L["app"]).clamp_min(0)
            gt = D["g_tr"]
            c0 = W_C // 2
            a5, _, _ = gt.injetar(gt.ativ, gt._serie_alvo, len(des), g)
            a = torch.zeros_like(fundo)
            a[:, c0 - gt.antes:c0 - gt.antes + gt.w5] = a5
            ok = a5[:, gt.antes] >= gt.lim
            d = z(fundo + a) - z(fundo)
            r["delta_add"] = float(d[ok].mean()) if ok.any() else float("nan")
        out[k] = r
    return out


# --------------------------------------------------------------------------------------
# um aparelho
# --------------------------------------------------------------------------------------


def metricas(y, z, t) -> dict:
    p = sig(z)
    prev = float(y.mean())
    return dict(n=len(y), prevalencia=round(prev, 4), ativacoes=ativacoes(y, t),
                auc=round(auc(y, p), 4), f1_oraculo=round(limiar_oraculo(p, y)[1], 4))


def rodar(aparelho: str, args) -> list[dict]:
    t0 = time.time()
    extras = ([int(c) for c in args.casas_extra.split(",") if c] if args.casas_extra
              else POOL.get(aparelho, []))
    val = [int(c) for c in args.casas_val.split(",") if c]
    D = preparar(aparelho, args.fonte, extras, val, args)
    gt = D["g_tr"]
    print(f"\n=== {aparelho} | fonte {args.fonte} casas {D['casas']} | P_ref={D['pref']:.0f} W"
          f" | {len(gt.base)} janelas base, {gt.ativ['n']} ativacoes, {gt.dist['n']} "
          f"distratores, {len(gt.idx_limpo)} fundos limpos | prop {gt.prop} "
          f"({time.time() - t0:.0f}s)", flush=True)
    if args.olhar_teste and not args.continuacao:
        from experimento_evicon import registrar_olhada
        registrar_olhada(f"v3 {aparelho} fonte={args.fonte} casa_fonte={args.fonte_fina} "
                         f"modelos={args.modelos} sementes={args.sementes} intra={args.intra} "
                         f"testes={args.testes or 'padrao'} tag={args.tag}")
    cfg = ConfigPerda(**json.loads(args.perda)) if args.perda else ConfigPerda()
    pasta = SAIDA / "pred" / args.tag
    pasta.mkdir(parents=True, exist_ok=True)
    linhas = []
    for nome in args.modelos.split(","):
        spec = spec_modelo(nome.strip())
        if args.glob:
            spec = dict(spec, glob=True)
        if args.reg:
            spec = dict(spec, reg=True)
        if args.entrada == "dif":
            spec = dict(spec, dif=True)
        if args.inv:
            spec = dict(spec, inv=True)
        for semente in range(args.semente0, args.semente0 + args.sementes):
            torch.manual_seed(semente)
            np.random.seed(semente)
            mod = Modelo(spec, D, ctx=args.ctx)
            t1 = time.time()
            r = treinar(mod, D, semente, args, cfg)
            m = r["mod"]
            preds = {}
            y, z, t = prever_gerado(m, D["g_va"])
            preds["val"] = (y, z, t)
            for k, (f, stc) in D["alvos"].items():
                preds[k] = (f.y[f.alvos].cpu().numpy(), prever_fonte(m, f, stc),
                            f.t[f.alvos_np])
            if args.adaptar:
                adaptar(m, D, args, semente, preds)
            itv = intervencao(m, D, semente=semente) if not args.intra else {}
            np.savez_compressed(
                pasta / f"{aparelho.replace(' ', '_')}__{nome}__s{semente}.npz",
                **{f"{k}|{c}": v for k, (y, z, t) in preds.items()
                   for c, v in (("y", y), ("z", z), ("p", sig(z)), ("t", t))})
            base = dict(aparelho=aparelho, fonte=args.fonte, modelo=nome, semente=semente,
                        epocas=r["epocas"], f1_val=round(r["f1_val"], 4),
                        segundos=round(time.time() - t1, 1))
            msg = []
            for k, (y, z, t) in preds.items():
                if (k == "val" and not args.intra) or k.startswith(("cvbn:", "cvft")):
                    continue
                lin = dict(**base, alvo=k, **metricas(y, z, t),
                           **{kk: round(v, 3) for kk, v in itv.get(k, {}).items()})
                if k.startswith("cv:"):
                    outros = [o for o in preds if o.startswith("cv:") and o != k]
                    yo = np.concatenate([preds[o][0] for o in outros])
                    po = sig(np.concatenate([preds[o][1] for o in outros]))
                    dec = Decisor(po, yo).decidir(sig(z))
                    lin.update(L_f1_a=round(f1(y, dec["a"]), 4), L_f1_d=round(f1(y, dec["d"]), 4))
                elif k != "val":
                    dec = Decisor(sig(preds["val"][1]), preds["val"][0]).decidir(sig(z))
                    lin.update(A_f1_d=round(f1(y, dec["d"]), 4))
                linhas.append(lin)
                msg.append(f"{k.split(':')[-1].strip()}: auc={lin['auc']:.3f} "
                           f"orac={lin['f1_oraculo']:.3f} "
                           f"dfun={lin.get('delta_fundo', np.nan):.2f}")
            print(f"  {nome:4s} s{semente} ep={r['epocas']:2d} F1val={r['f1_val']:.3f} "
                  f"{base['segundos']:4.0f}s | " + " | ".join(msg), flush=True)
            pd.DataFrame(linhas).to_csv(SAIDA / f"{args.tag}__{aparelho.replace(' ', '_')}.csv",
                                        index=False)
    return linhas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aparelhos", default="television,washing machine")
    p.add_argument("--modelos", default="B1,B2,B3")
    p.add_argument("--fonte", default="A", choices=["A", "A+"])
    p.add_argument("--casas_extra", default="")
    p.add_argument("--casas_val", default="3,7,20")
    p.add_argument("--fonte_fina", type=int, default=8,
                   help="casa-fonte principal na grade fina (REFIT: id; UK-DALE cenario: 300+predio)")
    p.add_argument("--testes", default="",
                   help="alvos de teste 'chave@arquivo@modo;...' (padrao: REFIT 8 + ECO + UK-DALE)")
    p.add_argument("--sementes", type=int, default=5)
    p.add_argument("--semente0", type=int, default=0)
    p.add_argument("--epocas", type=int, default=30)
    p.add_argument("--paciencia", type=int, default=5)
    p.add_argument("--passo", type=int, default=4)
    p.add_argument("--lote", type=int, default=256)
    p.add_argument("--lotes_epoca", type=int, default=0, help="0 = passada completa")
    p.add_argument("--taxa", type=float, default=2e-3)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--ctx", type=int, default=W5)
    p.add_argument("--frac_real", type=float, default=None)
    p.add_argument("--escala", type=float, nargs=2, default=None)
    p.add_argument("--perda", default="", help="json com campos de ConfigPerda")
    p.add_argument("--prop", default="", help="json com as proporcoes R0..R4 do gerador")
    p.add_argument("--deform", default="",
                   help="json (pode ser '{}') : R1 com ativacao deformada (reco.DEFORM)")
    p.add_argument("--trem", default="",
                   help="json (pode ser '{}') : R1 com trem de ciclos (reco.TREM)")
    p.add_argument("--glob", action="store_true",
                   help="ResNet com caracteristicas globais de periodicidade")
    p.add_argument("--parada", default="f1", choices=["f1", "auc"])
    p.add_argument("--entrada", default="padrao", choices=["padrao", "dif"])
    p.add_argument("--inv", action="store_true",
                   help="liga o termo de invariancia (pares R0/R3); forca via --perda")
    p.add_argument("--reg", action="store_true",
                   help="pos-quali: saida extra de regressao da potencia do alvo")
    p.add_argument("--lam_reg", type=float, default=0.5)
    p.add_argument("--adaptar", action="store_true",
                   help="pos-quali: AdaBN e ajuste fino nas casas de validacao")
    p.add_argument("--ft_passos", type=int, default=100)
    p.add_argument("--ft_taxa", type=float, default=1e-4)
    p.add_argument("--intra", action="store_true")
    p.add_argument("--olhar_teste", action="store_true")
    p.add_argument("--continuacao", action="store_true",
                   help="parte de uma rodada final ja registrada (dividida em processos): "
                        "nao registra nova olhada")
    p.add_argument("--verboso", action="store_true")
    p.add_argument("--tag", default="escada")
    args = p.parse_args()
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / f"{args.tag}__args.json").write_text(json.dumps(vars(args), indent=1))
    todas = []
    for ap in args.aparelhos.split(","):
        todas += rodar(ap.strip(), args)
    d = pd.DataFrame(todas)
    d.to_csv(SAIDA / f"{args.tag}.csv", index=False)
    cols = [c for c in ("auc", "f1_oraculo", "L_f1_d", "A_f1_d", "delta_rem", "delta_add",
                        "delta_fundo") if c in d]
    print(d.groupby(["aparelho", "alvo", "modelo"])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
