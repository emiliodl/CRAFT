# Resultados do v2 — implementação e execução do plano de `docs/v2.md`

Documento vivo. Cada seção diz o que foi implementado, o que rodou e o que decidiu. Referências
[v2 §x] apontam para o plano; [reg §x] para `docs/experimentos.md`.

## 0. O que foi implementado

| peça do v2 | arquivo | estado |
|---|---|---|
| A1 auditoria de bordas e lag (E0.1) | `src/saneamento.py`, `scripts/auditoria_p0.py` | rodou: 20 casas REFIT + ECO 1/2/5 |
| lacunas na resolução nativa (E0.2) | idem; `datasets/protocolo_artigo/lacunas/` | rodou |
| A2 holdout temporal com embargo + teste de vazamento (E0.3) | `src/saneamento.py`, `tests/test_v2.py` | teste bloqueante em toda execução |
| inventário das células (E0.4) | `src/inventario_p0.py` | rodou (olhada diagnóstica registrada) |
| A3/A4 calibração, EM, trava, limiar de F1 esperado | `src/decisao.py` | com dois desvios declarados (§3) |
| A5/A7 bootstrap por dias × sementes, binormal, Holm | `src/estatistica.py` | — |
| EviCon (§2, §9.C), ablações A1–A7 | `src/evicon.py`, `src/experimento_evicon.py` | 15 testes de propriedade passam |
| pares contrafactuais e intervenção (§3.1, §3.4) | `src/contrafactual.py` | — |
| dados em W com aparelho, lacuna e p95 | `src/dados_v2.py` | — |
| relatório (resumo, E1.1, pareado) | `src/relatorio_v2.py` | recalcula a decisão a partir das predições salvas |
| 11 casas REFIT novas no protocolo (E1.2) | `scripts/preparar_artigo.py --casas_refit ...` | rodou |

Não implementado (condicional ao portão G2, ou fora do orçamento desta rodada): A8 (TimesFM
como encoder do contexto), P3 inteiro (pré-treino auto-supervisionado, LoRA, Chronos/MOMENT/
Mantis), M3 MiniRocket, M4 filtro casado, UK-DALE/IDEAL como fonte, faixas B/C/D.

Disciplina com o teste: todo desenvolvimento usa o holdout temporal (4 últimas semanas de cada
casa fonte, embargo de 289 intervalos) e três casas REFIT fixas de **validação cross** (3, 7, 20),
nunca usadas em treino. As olhadas no mês de teste da casa 8 e no ECO ficam em
`resultados/v2/olhadas.csv`.

## 1. P0 — saneamento e diagnóstico

### E0.1 — auditoria de bordas (resolução nativa)

Média disparada em **inícios de sessão** (aparelho desligado em toda a meia-janela anterior; sem
isso, a 8 s a máquina de lavar cruza 20 W milhares de vezes dentro do ciclo e o "degrau" médio sai
negativo). Degraus = média dos lags 1–5 (~8–40 s) menos a linha de base.

| casa | aparelho | bordas | degrau plugue (W) | degrau agregado (W) | resíduo ± EP |
|---|---|---|---|---|---|
| REFIT 8 | freezer | 9 801 | +100 | +41 | −59 ± 8 |
| REFIT 8 | m. lavar | 402 | +107 | +429 | **+322 ± 112** |
| REFIT 8 | TV | 1 304 | +83 | +40 | −43 ± 18 |
| REFIT 3 | TV | 1 166 | +129 | +42 | −87 ± 25 |
| REFIT 4 | TV | 1 175 | +56 | **−17** | −74 ± 15 |
| REFIT 7 | freezer | 15 568 | +105 | +108 | +4 ± 4 |
| REFIT 7 | m. lavar | 700 | +65 | +365 | +300 ± 30 |
| ECO 2 | TV | 51 | +160 | +149 | −11 ± 26 |
| ECO 2 | freezer | 1 369 | +259 | +80 | −179 ± 6 |

Tabela completa (31 + 21 canais): `resultados/p0_auditoria*.csv`; curvas: `resultados/p0_curvas*.csv`.

Leitura, pelas regras do v2 §1.3a:

- **Máquina de lavar — confusão comportamental.** Na casa 8 o plugue sobe até ~1,3 kW em 2,5 min
  (o aquecedor aparece no plugue), mas o agregado sobe ~2,1 kW e o resíduo **cresce ao longo dos
  minutos** (+800 W no lag 20). É o padrão "resíduo espalhado": outra carga grande liga junto. Com
  97,8% do tempo ligado entre 0 h e 8 h [reg §6], a hipótese de tarifa noturna (aquecedor de água
  no mesmo horário) fica reforçada. Mesmo padrão nas casas 3, 7, 9, 13.
- **TV e freezer — o agregado registra ~40–60% do degrau do plugue**, e o déficit **persiste** por
  minutos (não é atraso: o agregado não "alcança" depois). Em aparelho regido por termostato (o
  freezer) coocorrência comportamental é implausível; a explicação mais provável é de medição
  (princípios diferentes entre o sensor do agregado e os plugues). Não foi resolvido.
- **O "agregado cai 75 W quando a TV liga" da casa 8 não se reproduz.** Na transição do rótulo a
  5 min, o agregado **sobe** +28,8 W; na resolução nativa, +40 W. O −75 W do registro [reg §10] vinha
  de outra estatística e não é um fato sobre as bordas. A queda real só aparece na **casa 4
  (−136 W a 5 min)** e na **casa 6 (−155 W)**.
- **Lag.** O critério "lag de maior |correlação| ≠ 0" é ruído para a TV: as correlações das
  diferenças primeiras ficam em |r| < 0,005 em todos os lags (o degrau é pequeno frente ao ruído do
  agregado a 8 s). Na m. lavar e no freezer o lag é 0 em todas as casas.

### E0.2 — lacunas

Hipótese **refutada**: no REFIT limpo, as lacunas aparecem como linhas ausentes, não como zeros; a
fração de intervalos de 5 min com zeros exatos ≥ 2 min é ≤ 0,01% em todas as casas, e ≤ 0,05% no
ECO. Refazer a média disparada sem as bordas perto de lacuna não muda nenhum número (coluna
`residuo_sem_lacuna`).

### Portão G0 — fontes aprovadas

A regra do v2 ("|resíduo| > 30% do degrau → canal fora") reprovaria a própria casa 8 em todos os
aparelhos, e ela é a casa do protocolo. Critério operacional adotado e declarado: sai da fonte o
canal com **degrau do agregado ≤ 0 (ou ~0) na transição do rótulo a 5 min** ou **degenerado**
(prevalência ≥ 70% ou 0%).

| aparelho | casas extras aprovadas | excluídas |
|---|---|---|
| TV | 1, 2, 5, 13, 15, 16, 19, 21 | 4 (−136 W), 6 (−155 W, 93% ligada), 9 (73%), 10 (100%), 12 (0%), 17 (81%, +2 W), 18 (100%) |
| m. lavar | 1, 2, 4, 5, 6, 9, 10, 11, 13, 15, 16, 17, 18, 19, 21 | — |
| freezer | 1, 4, 6, 13, 17, 18 | — (só 7 casas REFIT têm freezer) |

Validação cross fixa: casas 3, 7, 20. A TV da casa 7 tem degrau de +0,2 W no agregado: é uma casa
de validação em que a TV é praticamente invisível; mantida, porque a validação não se escolhe
pelo resultado.

### E0.3 — vazamento

`test_sem_vazamento` roda em toda execução (assert bloqueante) sobre as amostras brutas cobertas
pelas janelas de 24 h. Divisão atual (holdout temporal de 4 semanas + embargo de 289): 0 amostras
compartilhadas. Divisão aleatória por centro (teste sintético): falha, como deveria.

### E0.4 — inventário das células de teste (só rótulos)

| aparelho | célula | dias | prevalência | ativações | trivial | artigo |
|---|---|---|---|---|---|---|
| m. lavar | intra | 25 | 4,4% | **19** | 0,084 | 0,63 |
| m. lavar | cross (ECO 1) | 31 | 6,8% | 36 | 0,128 | 0,30 |
| freezer | intra | 25 | 33,5% | 487 | 0,501 | 0,46 |
| freezer | cross (ECO 1+2) | 62 | 65,3% | 2 432 | 0,790 | 0,78 |
| TV | intra | 25 | 24,3% | 56 | 0,391 | 0,68 |
| TV | cross (ECO 2) | 31 | 21,6% | 41 | 0,355 | 0,54 |

O freezer do artigo fica **abaixo do trivial** nas duas células. A m. lavar intra tem 19 ativações:
qualquer F1 ali tem IC de ±0,10 ou mais.

## 2. P1 — estágio de decisão (E1.1) e re-execuções (E1.3)

### Dois desvios do código de referência do v2, ambos declarados

1. **Contagem ajustada no limiar 0,5 (apêndice A3) quebra em classe rara.** A probabilidade
   calibrada de uma máquina de lavar quase nunca passa de 0,5: TPR = 0,06, a trava
   "TPR − FPR < 0,1" dispara com AUC 0,86 e a decisão vira "tudo positivo" (F1 0,12 contra 0,28
   da regra a). Corrigido para o limiar que maximiza TPR − FPR na calibração (método "Max" de
   Forman). `src/decisao.py::taxas_na_fonte`.
2. **A trava trata qualquer disparo como "classificador não informativo".** Com AUC 0,86 e o EM
   discordando da contagem ajustada, "tudo positivo" dá F1 0,06–0,15 contra 0,22–0,38 da regra
   (a). Regra **(e)**: tudo positivo só se AUC retida < 0,6 ou TPR − FPR < 0,1; se o que falha é
   só a estimativa de prevalência, cai para a regra indutiva (d). A regra (c) original continua
   reportada.

### E1.1 — regras de decisão nas casas de validação cross (calibração LOHO, 5 sementes)

| aparelho | modelo | AUC | oráculo | (a) origem | (b) EM + 0,5 | (c) v2 | (d) indutiva | trivial |
|---|---|---|---|---|---|---|---|---|
| m. lavar | A0 | 0,832 | 0,301 | 0,281 | 0,054 | 0,123 | 0,281 | 0,109 |
| m. lavar | A2 | 0,823 | 0,362 | 0,354 | 0,162 | 0,274 | 0,354 | 0,109 |
| TV | A0cf | 0,714 | 0,613 | 0,513 | 0,223 | 0,545 | 0,513 | 0,507 |
| freezer | A2 | 0,704 | 0,715 | 0,695 | 0,587 | 0,675 | 0,695 | 0,692 |

**H3 refutada.** O caminho transdutivo (EM + F1 esperado) perde para o limiar da origem na
m. lavar em todos os modelos, e (b) "EM + 0,5" é desastroso em todos os aparelhos. Entre casas não
muda só a prevalência: muda p(x | y), e o EM fica mal especificado — o cenário em que Ye et al.
[R6] acham o limiar empírico mais robusto. Pela regra do v2 (E1.1), adota-se **(d)**, que aqui
coincide com (a). A única célula em que (c) ajudou foi a TV com A0cf (0,545 contra 0,513).

### E1.3 — contexto e K rótulos com a divisão corrigida (CNN, faixa A, 5 sementes)

| aparelho | 2 h | 4 h | 8 h | 24 h | K = 8 | K = 32 |
|---|---|---|---|---|---|---|
| TV (AUC cv) | **0,653** | 0,632 | 0,635 | 0,491 | 0,489 | 0,491 |
| m. lavar (AUC cv) | **0,875** | **0,875** | 0,860 | 0,832 | 0,835 | 0,828 |
| m. lavar (F1 d, cv) | 0,339 | 0,340 | 0,310 | 0,281 | 0,276 | 0,266 |

A conclusão "contexto curto transfere melhor" do registro [reg §5.2] **sobrevive** à correção do
bug; K > 1 continua sem ganho. O v2 suspeitava que as duas fossem artefato (§1.2): não eram.
Contexto padrão da CNN: 4 h.

## 3. P2 — escada do EviCon (faixa A: só a casa 8 no treino)

Avaliação nas casas de validação cross 3/7/20, 5 sementes. F1 = regra (d) com calibração LOHO.
Δ_rem e Δ_add em logits (intervenção na casa 3).

| aparelho | modelo | AUC | AP | oráculo | F1 (d) | Δ_rem | Δ_add |
|---|---|---|---|---|---|---|---|
| m. lavar | A0 CNN 24 h | 0,832 | 0,200 | 0,301 | 0,281 | 1,67 | 2,51 |
| m. lavar | A0_4h | 0,875 | 0,246 | 0,346 | 0,340 | — | — |
| m. lavar | A1 sem acumulador | 0,682 | 0,186 | 0,292 | 0,285 | 0,11 | 0,33 |
| m. lavar | **A2 evidência** | 0,823 | 0,239 | **0,362** | **0,354** | 0,26 | 0,92 |
| m. lavar | A3 sem sinal | 0,778 | 0,244 | 0,362 | 0,352 | 0,43 | 1,22 |
| m. lavar | A4 EviCon | 0,704 | 0,159 | 0,229 | 0,206 | 0,13 | 0,53 |
| m. lavar | A5 contexto livre | 0,835 | 0,221 | 0,332 | 0,319 | 1,32 | 2,85 |
| TV | A0 CNN 24 h | 0,491 | 0,360 | 0,508 | 0,501 | 0,39 | 0,67 |
| TV | A0cf (+25% contrafactuais) | **0,714** | 0,516 | 0,613 | 0,513 | 0,32 | 0,62 |
| TV | **A0_4hcf** | 0,668 | 0,464 | 0,597 | **0,581** | — | — |
| TV | A1 | 0,557 | 0,380 | 0,513 | 0,500 | 0,05 | 0,28 |
| TV | A2 | 0,657 | 0,483 | 0,564 | 0,506 | 0,07 | 0,24 |
| TV | A3 | 0,589 | 0,448 | 0,520 | 0,474 | 0,04 | 0,14 |
| TV | A4 | 0,524 | 0,349 | 0,512 | 0,505 | 0,02 | 0,18 |
| TV | A4m (c = 0) | 0,539 | 0,359 | 0,513 | 0,503 | — | — |
| TV | A5 | 0,607 | 0,450 | 0,539 | 0,464 | 0,57 | 0,69 |
| freezer | A0 | 0,532 | 0,557 | 0,692 | 0,692 | −0,01 | 0,00 |
| freezer | A2 | 0,704 | 0,714 | 0,715 | 0,695 | 0,15 | 0,23 |
| freezer | A5 | 0,711 | 0,724 | 0,718 | 0,703 | 0,18 | 0,27 |

Trivial: m. lavar 0,109 · TV 0,507 · freezer 0,692. Tabela completa (14 modelos × 3 aparelhos):
`resultados/v2/escada_A*__recomputado.csv`.

### Hipóteses do EviCon

| id | hipótese | resultado |
|---|---|---|
| HE1 | a estrutura bloqueia o atalho da TV | **confirmada para o ramo de evidência**: AUC 0,491 → 0,657; bootstrap pareado A2 − A0 com IC excluindo zero nas casas 3 e 7 |
| HE2 | a restrição de sinal é decisiva | **confirmada na TV** (A3 0,589 < A2 0,657); na m. lavar e no freezer, A3 ≈ A2 |
| HE3 | o contexto só ajuda quando restrito | **refutada**: A4 < A2 nos três aparelhos, e A5 (livre) > A4 |
| HE4 | o acumulador importa | **confirmada** nos três (A2 > A1) |
| HE5 | o EviCon responde ao aparelho, não ao contexto | **refutada na forma prevista**: os Δ da CNN são *maiores* (as escalas de logit diferem; a evidência é pequena em valor absoluto) |

**Por que o contexto restrito falhou — uma falha da garantia G1 que o v2 não previa.** O teto ±c
protege o *limiar*: sem evidência, nenhuma janela passa de σ(b₀ + c). Mas não protege o *ranking*:
na grande maioria das janelas e ≈ 0, e a ordem entre elas passa a ser decidida inteiramente pelo
contexto — o atalho da casa 8 volta pela AUC. A variante A4m (c = 0, contexto só multiplicativo,
desenhada depois de ver a validação, não o teste) não resolve: AUC 0,539 na TV. O contexto
treinado numa casa só piorou a transferência em todas as formas testadas.

**Portão G2.** A2 ≥ A4 em todos os aparelhos → pela regra do v2, fica o **modelo só de
evidência**. Mas A2 só vence a CNN na m. lavar (F1) e no freezer (AUC); na TV, o que venceu foi a
**CNN com pares contrafactuais** — a infraestrutura da §3.1, não a arquitetura da §2.

**Pares contrafactuais.** Na CNN da TV, 25% de pares (fundo + ativação; fundo onde a TV estava
ligada) levam a AUC de 0,491 para 0,714 **sem mudar a arquitetura** — o maior ganho isolado da
escada. No EviCon (A7, A2cf) os pares não ajudam: a evidência já não usava o atalho que eles
desmontam.

**Freezer.** A evidência tira a AUC do acaso (0,53 → 0,70) com **uma casa só** — o registro
concluíra que isso só vinha com várias casas [reg §11]. O F1 fica colado no trivial porque o
oráculo (0,715) mal passa dele (0,692).

## 4. P4 — rodada final, faixa A (olhada única no teste)

Pré-registro `resultados/v2/preregistro_faixaA.json` (hash `5d82066dd7690030`), escrito antes da
rodada: modelo por aparelho escolhido pelo maior F1 (d) na validação cross — **m. lavar A2, TV
A0_4hcf, freezer A5** —, 10 sementes, regra primária d_A (calibrada no holdout da casa 8: faixa A
estrita e indutiva). d_L (calibrada nas casas REFIT 3/7/20) é secundária e usa rótulos de casas
extras. "Supera" = limite inferior do IC95% acima do número do artigo; "perde" = limite superior
abaixo.

| aparelho | célula | ativações | F1 d_A [IC95%] | artigo | trivial | vs artigo | vs trivial | AUC | oráculo |
|---|---|---|---|---|---|---|---|---|---|
| freezer | intra | 403 | **0,569** [0,542; 0,600] | 0,46 | 0,496 | **supera** | **vence** (+0,05..+0,10) | 0,713 | 0,572 |
| freezer | cross | 1 938 | **0,805** [0,796; 0,814] | 0,78 | 0,790 | **supera** | **vence** (+0,002..+0,027) | 0,792 | 0,829 |
| TV | intra | 44 | 0,616 [0,546; 0,681] | 0,68 | 0,386 | inconclusivo | vence | 0,821 | 0,627 |
| TV | cross | 39 | 0,277 [0,204; 0,352] | 0,54 | 0,346 | **perde** | **perde** | **0,430** | 0,365 |
| m. lavar | intra | 16 | 0,279 [0,196; 0,363] | 0,63 | 0,086 | **perde** | vence | 0,833 | 0,314 |
| m. lavar | cross | 29 | 0,171 [0,113; 0,232] | 0,30 | 0,137 | **perde** | inconclusivo | 0,786 | 0,313 |

As ativações aqui são menores que no inventário porque a janela de 24 h centrada consome 12 h em
cada ponta de cada trecho contínuo. Com d_L (secundária): freezer cross 0,826 [0,813; 0,838],
m. lavar cross 0,219 [0,155; 0,282], TV cross 0,348. Tudo em `resultados/v2/final_A__celulas.csv`.

Leitura:

- **Freezer: primeira vez no projeto que o freezer vence o trivial com IC** — nas duas células, e
  por consequência supera o artigo, que fica abaixo do trivial nas duas. O ganho é pequeno em
  valor absoluto (+0,015 no cross), mas é detecção real (AUC 0,71–0,79), vinda do ramo de evidência
  treinado numa casa só.
- **TV cross: a seleção pela validação britânica não transferiu para a Suíça.** A AUC no ECO 2 é
  0,43 — abaixo do acaso — com um modelo que tinha AUC 0,67 nas casas REFIT de validação. É o
  risco da §8 do v2 ("seleção por LOHO britânico não prevê o ECO"), agora medido.
- **M. lavar: o modelo só de evidência não serve no intra** — oráculo 0,314 com AUC 0,833, contra
  0,607 da CNN do registro. A evidência ranqueia bem mas é pouco precisa no topo. No cross, a AUC
  (0,786) é igual à do registro e o F1 continua abaixo do CO (0,30).

## 5. Faixa A+ — várias casas fonte (portão G0), mesma escada

Fonte = casa 8 + casas aprovadas (§1), passo de treino 8 (metade das janelas por casa, por custo),
5 sementes, mesma validação cross 3/7/20.

| aparelho | modelo | AUC | AP | oráculo | F1 (d) | vs faixa A (F1 d) |
|---|---|---|---|---|---|---|
| m. lavar (16 casas) | **A5 evidência + contexto livre** | **0,892** | 0,410 | 0,444 | **0,429** | 0,319 |
| m. lavar | A0_4h | 0,887 | 0,305 | 0,378 | 0,374 | 0,340 |
| m. lavar | A2 | 0,856 | 0,228 | 0,357 | 0,352 | 0,354 |
| m. lavar | A0_4hcf | 0,853 | 0,272 | 0,325 | 0,320 | 0,328 |
| TV (9 casas) | **A0_4h** | **0,757** | 0,590 | 0,640 | **0,637** | 0,478 |
| TV | A0cf | 0,755 | 0,574 | 0,643 | 0,634 | 0,513 |
| TV | A0 | 0,740 | 0,543 | 0,620 | 0,601 | 0,501 |
| TV | A0_4hcf | 0,682 | 0,492 | 0,582 | 0,568 | 0,581 |
| TV | A5 | 0,680 | 0,499 | 0,562 | 0,544 | 0,464 |
| TV | A2 | 0,645 | 0,471 | 0,556 | 0,516 | 0,506 |
| TV | A4m / A4mcf / A2cf | 0,60 | 0,40–0,42 | 0,53 | 0,50–0,52 | — |
| freezer (7 casas) | **A0_4hcf** | 0,726 | 0,718 | 0,746 | **0,732** | 0,692 |
| freezer | A2 | 0,728 | 0,724 | 0,740 | 0,726 | 0,695 |
| freezer | A0_4h | 0,706 | 0,693 | 0,735 | 0,723 | 0,692 |
| freezer | A5 | 0,699 | 0,710 | 0,726 | 0,695 | 0,703 |

Leitura:

- **O contexto só é perigoso com uma casa.** Com 16 casas, o contexto livre (A5) passa de pior a
  melhor modelo da m. lavar (F1 0,319 → 0,429; AP 0,221 → 0,410). É a mesma história do registro
  [reg §11], agora do lado do EviCon: a restrição arquitetural compensa falta de diversidade, e
  deixa de ser necessária quando a diversidade existe.
- **Na TV, diversidade de casas resolve o que a arquitetura não resolveu.** A CNN sai de AUC
  0,49–0,63 para 0,74–0,76; o ramo de evidência não se beneficia (0,657 → 0,645). Os pares
  contrafactuais, que eram o maior ganho com uma casa, deixam de somar com nove (A0cf ≈ A0_4h).
- **Freezer:** com várias casas a CNN alcança o ramo de evidência (AUC ~0,73), e o F1 passa do
  trivial por +0,04 na validação.

### E1.2 — curva de escala de fontes (AUC média nas casas de validação cross)

2 sorteios da ordem das casas extras, 3 sementes; nível 1 = faixa A, nível "todas" = tabela acima.

| aparelho | modelo | 1 casa | 2 (sorteio 1 / 2) | 4 | 8 | todas |
|---|---|---|---|---|---|---|
| TV | A0_4h | 0,632 | 0,756 / 0,613 | 0,767 / 0,756 | 0,728 / 0,728 | 0,757 (9) |
| TV | A2 | 0,657 | 0,650 / 0,590 | 0,595 / 0,573 | 0,638 / 0,641 | 0,645 (9) |
| m. lavar | A0_4h | 0,875 | 0,863 / 0,870 | 0,865 / 0,876 | 0,887 / 0,867 | 0,887 (16) |
| m. lavar | A2 | 0,823 | 0,837 / 0,844 | 0,840 / 0,852 | 0,862 / 0,850 | 0,856 (16) |

**H1 refutada na forma do v2** (crescimento log-linear sem platô): na TV a CNN satura em ~4 casas
(~0,76), e *quais* casas entram pesa tanto quanto *quantas* — com 2 casas a AUC é 0,756 ou 0,613
conforme o sorteio. É a explicação que faltava para a não-monotonicidade do registro (5 casas
melhor que 9 [reg §11]): variação de composição, não de quantidade. Na m. lavar a CNN já começa
alta e fica plana. Pela regra do portão G1 ("platô com ≤ 4 casas e TV abaixo de ~0,80 → atalho
ou representação"), o próximo investimento na TV é representação, não mais casas REFIT.

## 6. P4 — rodada final, faixa A+ (olhada única no teste)

Pré-registro `resultados/v2/preregistro_faixaAmais.json` (hash `53de97a2685bb7bb`): **m. lavar A5,
TV A0_4h, freezer A0_4hcf**, 10 sementes, regra primária d_L (calibrada nas casas 3/7/20).

| aparelho | célula | F1 d_L [IC95%] | artigo | trivial | vs artigo | vs trivial | AUC | oráculo |
|---|---|---|---|---|---|---|---|---|
| freezer | intra | 0,509 [0,491; 0,531] | 0,46 | 0,496 | **supera** | empata (−0,001..+0,028) | 0,594 | 0,520 |
| freezer | cross | **0,824** [0,816; 0,832] | 0,78 | 0,790 | **supera** | **vence** (+0,029..+0,039) | 0,735 | 0,831 |
| TV | intra | 0,500 [0,446; 0,549] | 0,68 | 0,386 | perde | vence | 0,752 | 0,548 |
| TV | cross | 0,476 [0,415; 0,534] | 0,54 | 0,346 | perde (por 0,006) | vence | **0,853** | 0,579 |
| m. lavar | intra | 0,398 [0,231; 0,537] | 0,63 | 0,086 | perde | vence | 0,913 | 0,428 |
| m. lavar | cross | 0,245 [0,157; 0,332] | 0,30 | 0,137 | inconclusivo | vence | 0,816 | 0,336 |

Tudo (inclusive d_A, a, c, e): `resultados/v2/final_Amais__celulas.csv`.

## 7. Placar final contra o artigo (números pré-registrados, olhada única)

| célula | artigo | trivial | faixa A (só casa 8) | faixa A+ (várias casas) |
|---|---|---|---|---|
| m. lavar intra | 0,63 | 0,086 | 0,279 — perde | 0,398 — perde |
| m. lavar cross | 0,30 | 0,137 | 0,171 — perde | 0,245 — inconclusivo |
| freezer intra | 0,46 | 0,496 | **0,569 — supera, vence o trivial** | 0,509 — supera, empata c/ trivial |
| freezer cross | 0,78 | 0,790 | **0,805 — supera, vence o trivial** | **0,824 — supera, vence o trivial** |
| TV intra | 0,68 | 0,386 | 0,616 — inconclusivo | 0,500 — perde |
| TV cross | 0,54 | 0,346 | 0,277 — perde | 0,476 — perde |

Pela meta operacional do v2 ("superar o artigo com IC acima do número deles nas células com
detecção real, e o freezer sempre contra o trivial"): **o freezer é a única célula ganha, nas duas
faixas, e é a primeira vez no projeto que ele vence o trivial com IC.** Máquina de lavar e TV não
superam o artigo em nenhuma faixa com a configuração pré-registrada.

Comparação honesta com o registro: os melhores números anteriores (TV cross 0,601 com 5 casas;
m. lavar intra 0,607) eram melhores que os pré-registrados aqui, mas foram escolhidos olhando o
teste entre várias configurações. Com seleção feita só na validação britânica, a TV cross cai para
0,476 — a diferença mede o otimismo da seleção pelo teste que o registro tinha.

## 8. O que o v2 acertou, o que errou, e o que fica

**Acertou:**
- os três diagnósticos de P0 mudaram a leitura do registro: o "agregado cai 75 W" da TV não
  existe nas bordas; lacunas zeradas não existem no REFIT; a m. lavar tem confusão com carga
  coocorrente (resíduo espalhado de +800 W);
- restrição de sinal e acumulador fazem o que prometiam (HE2, HE4), e o ramo de evidência bloqueia
  o atalho da TV com uma casa (HE1);
- pares contrafactuais: maior ganho isolado com uma casa (TV 0,49 → 0,71 de AUC);
- o protocolo estatístico: com ele, a maioria das diferenças do registro deixa de ser resolvível,
  e as que sobram (freezer vs trivial) são robustas.

**Errou:**
- contexto restrito (HE3): o teto ±c protege o limiar, não o ranking;
- estágio de decisão transdutivo (H3): o EM fica mal especificado entre casas; o código de
  referência (limiar 0,5 na contagem ajustada; trava como "tudo positivo") tinha dois defeitos;
- curva de escala log-linear (H1): platô em ~4 casas, dominado pela composição;
- a expectativa de que a validação britânica selecionasse bem para a Suíça: na TV (faixa A) o
  modelo escolhido tem AUC 0,43 no ECO.

**Não executado:** A8 (TimesFM no contexto), E2.4 completo (só Δ_add/Δ_rem na casa 3 e o corte
dia × madrugada da m. lavar, gravados nas colunas `delta_add_*`), P3 (pré-treino, LoRA, Chronos/
MOMENT/Mantis), M3/M4, UK-DALE/IDEAL como fonte, faixas B/C/D, Holm formal (só há duas
comparações primárias por célula).

**Onde está o F1 que falta:**
- **TV cross (faixa A+):** AUC 0,853 e oráculo 0,579 contra 0,476 obtidos — a lacuna agora é de
  **decisão**, não de ranking. É a célula em que um limiar melhor (sem olhar o ECO) viraria o
  resultado.
- **M. lavar cross:** oráculo 0,336 contra 0,245 — também de decisão, e o oráculo mal passa do CO.
- **M. lavar intra:** AUC 0,91 e oráculo 0,43 — com 16 ativações, o IC é de ±0,15; nada ali é
  resolvível com um mês.

