# Resultados do v3 — ReCo: implementação e execução do plano de `docs/v3.md`

Cada seção diz o que foi implementado, o que rodou e o que decidiu. [v3 §x] aponta para o plano;
"v2" para `docs/v2_resultados.md`. Toda seleção usou as casas REFIT 3/7/20 (validação cross, LOHO)
e o holdout temporal da casa 8; o ECO e o mês de teste da casa 8 só foram olhados em F4.

## 0. O que foi implementado

| peça do v3 | arquivo | estado |
|---|---|---|
| grade quase-nativa (30 s) com os 9 plugues e contagem de amostras | `scripts/preparar_v3.py` → `datasets/v3_fino/` | 20 casas REFIT; reamostragem reproduz a média do protocolo (diferença média 0,03 W, rótulos 100% iguais na casa 8) |
| gerador R0–R4 (+ Rf, o fundo sozinho), bancos por período | `src/reco.py` | — |
| ResNet 1D de 3 blocos, 4 canais de entrada, perda BCE + aux + L_inv + L_sens | `src/reco.py` | — |
| escada B1–B6 (+ B2r, Q9), variante intra (I<k>, C0), intervenção Δ_add/Δ_rem/Δ_fundo | `src/experimento_reco.py` | — |
| B0 e B7 a partir das predições salvas do v2; média de postos; bootstrap pareado | `src/relatorio_v3.py` | — |
| decisão por quantil, k̂, π̂ (fonte, Median Sweep, média) | `src/quantil.py`, `src/experimento_quantil.py` | — |
| D1, D2 | `src/diagnostico_v3.py` | — |
| discriminador real × recomposto (E2.0, teste 7) | `src/discriminador_v3.py` | — |
| seleção gulosa de casas (E2.3) | `src/selecao_casas_v3.py` | — |
| fechamento por pré-registro | `src/final_v3.py` | — |
| testes de unidade 1–6 de §8 (+ período, regime trivial) | `tests/test_v3.py` | 12 passam |

**Desvios declarados.**

- B0 (A0_4hcf) e o membro só-evidência do B7 (A2) vêm das predições salvas do v2, nas mesmas casas,
  alvos e sementes; não foram retreinados.
- Uma tentativa por kernel (o plano pede 3 com escolha pela validação). As 10 sementes do F4 dão a
  variância de treino.
- Épocas limitadas a 400 lotes (300 no intra) e paciência 5, por custo.
- D5 (UK-DALE/IDEAL) não rodou: não há leitor de HDF5 (h5py/pytables) instalado e o cache do
  UK-DALE não tem TV nem resolução nativa. Nenhuma casa fora do REFIT entrou.
- Fundos só de janelas "limpas" (alvo desligado na janela inteira), e fundo da mesma casa só de
  outro dia. B = A − (todos os plugues da categoria do alvo), para que uma segunda máquina de lavar
  (casa 4) não vire negativo.
- Distrator = ativação acima de max(limiar, mediana do plugue + 20 W), com o repouso subtraído. O
  computador da casa 8 fica em 18 W o tempo todo; sem isso, não haveria distrator na casa 8.
- Freezer: configuração do v2 congelada (§3.8); o ReCo não foi treinado para ele.

## 1. F0 — diagnósticos

### D1 — confundidor da máquina de lavar

Resíduo (agregado − plugue) nos 20 min após cada início de sessão, contra os 10 min anteriores
(grade de 30 s). Tabela completa em `resultados/v3/d1_*.csv`.

- **Casa 8:** dR = +192 W, quase todo de madrugada (+245 W antes das 8 h contra −40 W de dia; 81%
  das bordas são de madrugada). Os outros plugues explicam 47% (Washer Dryer 30%). O resto é
  carga não submedida noturna, compatível com tarifa (o metadado do REFIT não traz tarifa).
- **Outras casas:** a secadora explica a maior parte do resíduo nas casas 3 (99%), 5 e 7; a
  washer-dryer, na casa 9 (73%).

**Portão G0:** distratores e auxiliares da máquina de lavar = secadora (tumble/washer dryer) +
lava-louças. A troca de fundo (R3) cobre a carga noturna não submedida.

### D2 — teto do rótulo

Fração dos positivos com potência média do plugue abaixo de 50 W:

- máquina de lavar: mediana 6,6% (casa 8: 2,9%);
- TV: mediana 65%, porque muitas casas têm TVs de 15–40 W;
- freezer: 22%.

O corte de 30% da máquina de lavar não dispara, então não se rebaixa a expectativa.

## 2. F1 — decisão por quantil sobre os modelos do v2 (E1.1, E1.2, D3)

Escores finais do v2 nas casas 3/7/20, LOHO aninhado. Métrica: F1/oráculo por casa retida
(`resultados/v3/f1_quantil__*.csv`).

| modelo do v2 | (a) origem | qa: π fonte | qb: Median Sweep | qc: média |
|---|---|---|---|---|
| TV A0_4hcf (A, final) | 0,977 | 0,820 | 0,885 | 0,979 |
| TV A0_4h (A+, final) | 0,995 | 0,954 | 0,694 | 0,865 |
| m. lavar A2 (A, final) | 0,993 | 0,854 | 0,748 | 0,874 |
| m. lavar A5 (A+, final) | 0,995 | 0,903 | 0,629 | 0,924 |

**D3.** No LOHO britânico, o limiar da origem já chega a 0,97–0,99 do oráculo, mesmo prevendo
frações bem diferentes da do oráculo (r*/π = 1,4–3,2). Perto do ótimo, a curva de F1 é plana. A
lacuna de decisão da TV cross do v2 (0,476 contra oráculo 0,579) é um fenômeno do ECO que o LOHO
não reproduz.

O Median Sweep erra muito (π̂ = 0 ou 1 em várias casas): p(x | y) muda entre casas.

**Portão G1:** nenhuma regra de quantil tem mediana ≥ 0,95 e vence (a) em 2/3 das casas. A regra
indutiva **d** fica congelada como primária; o quantil (q) é reportado como secundário
(transdutivo, A-t/A+-t).

## 3. E2.0 — discriminador real × recomposto

AUC de uma ResNet treinada para separar recompostas de reais com o mesmo rótulo, avaliada no
holdout temporal (`resultados/v3/e20_*.csv`):

| | TV A | TV A+ | m. lavar A | m. lavar A+ |
|---|---|---|---|---|
| R1 injeção | 0,84 | 0,71 | 0,92 | 0,87 |
| R2 remoção | 0,80 | 0,68 | 0,68 | 0,88 |
| R3 troca de fundo | 0,74 | **0,56** | **0,57** | **0,52** |
| R4 distrator | 0,75 | 0,73 | 0,85 | 0,90 |
| Rf fundo, controle (s = 1, negativos limpos) | 0,62 | **0,53** | **0,50** | **0,50** |

**O critério < 0,6 falha para R1, R2 e R4.** O controle localiza o problema:

- **A subtração e a reamostragem não deixam marca.** Rf ≈ 0,5 fora da TV na faixa A, e a troca de
  fundo com rótulo exato (R3, o núcleo de [V2]) passa em A+.
- **O que separa R1/R2/R4 é o que elas mudam de propósito:**
  - a co-ocorrência (a carga de +800 W que acompanha a máquina de lavar some);
  - a raridade dos distratores;
  - na TV, a razão degrau do agregado / degrau do plugue, que vai de 0,33 a 2,3 entre casas
    (casa 8: 0,48; E0.1 do v2). Nenhum ganho único a corrige.

Na faixa A da TV, até R3 separa (0,74): é o limite do canal da TV da casa 8 previsto em §3.9,
declarado e não contornado. Como alternativa que respeita o critério, entraram na escada braços só
com R0 + R3 (B2r, Q9).

## 4. F2 — escada do ReCo (E2.1)

Casas 3/7/20, 5 sementes. AUC = média das 3 casas; F1 (d) = média por casa; Δ = bootstrap pareado
contra B0 (dias × sementes, 1000 réplicas; F1 das contagens agregadas das 3 casas). Tabelas
completas em `resultados/v3/escada_{A,Amais}__*.csv`.

### Faixa A (só a casa 8)

| modelo | TV AUC | TV F1 d | ΔAUC TV [IC95%] | m. lavar AUC | m. lavar F1 d | ΔF1 d m. lavar [IC95%] |
|---|---|---|---|---|---|---|
| B0 (v2) | 0,668 | 0,581 | — | 0,859 | 0,328 | — |
| B1 gerador | 0,689 | 0,563 | +0,013 [+0,010; +0,017] | 0,847 | 0,364 | +0,038 [+0,024; +0,051] |
| B2 + L_inv | 0,684 | 0,567 | +0,008 | 0,872 | 0,349 | +0,002 [−0,009; +0,014] |
| B3 + L_sens | 0,691 | 0,526 | +0,018 | 0,815 | 0,372 | +0,037 |
| B2r (só R0+R3) | 0,660 | 0,601 | −0,016 | 0,870 | 0,331 | −0,005 |
| B4 ResNet k = 9 | 0,744 | 0,548 | +0,076 [+0,060; +0,091] | 0,864 | 0,379 | +0,065 |
| **B5 = ReCo** | **0,770** | 0,583 | **+0,090 [+0,075; +0,105]** | **0,884** | **0,397** | **+0,084 [+0,068; +0,101]** |
| B6 + auxiliares | 0,772 | 0,570 | +0,092 | 0,877 | 0,391 | +0,080 |
| B7 + só-evidência | 0,779 | 0,590 | +0,099 | 0,889 | 0,397 | +0,085 |
| Q9 (ResNet, só R0+R3) | 0,746 | 0,565 | +0,073 | 0,679 | 0,260 | −0,035 |

### Faixa A+ (casa 8 + casas do portão G0 do v2)

| modelo | TV AUC | TV F1 d | ΔF1 d TV [IC95%] | m. lavar AUC | m. lavar F1 d | ΔF1 d m. lavar [IC95%] |
|---|---|---|---|---|---|---|
| B0 (v2) | 0,682 | 0,568 | — | 0,853 | 0,320 | — |
| B1 | 0,718 | 0,595 | +0,028 [+0,017; +0,038] | 0,891 | 0,420 | +0,116 [+0,100; +0,131] |
| B2 | 0,730 | 0,605 | +0,038 | 0,870 | 0,340 | +0,022 |
| B3 | 0,697 | 0,583 | +0,016 | 0,869 | 0,347 | +0,026 |
| B2r | 0,747 | 0,620 | +0,053 | 0,878 | 0,343 | +0,027 |
| B4 | 0,738 | 0,589 | +0,017 [−0,001; +0,033] | 0,900 | 0,495 | +0,175 |
| **B5 = ReCo** | **0,763** | 0,594 | +0,022 [+0,009; +0,036] | **0,907** | **0,505** | **+0,189 [+0,168; +0,209]** |
| B6 | 0,763 | 0,598 | +0,027 | 0,910 | 0,512 | +0,195 |
| B7 | — | — | — | 0,912 | 0,508 | +0,197 |
| Q9 | 0,771 | 0,619 | +0,052 | 0,915 | 0,475 | +0,151 |

(B7 da TV A+ ficou de fora do relatório; a correção que busca o A2 da TV na rodada `escada_Amais`
do v2 foi feita depois.)

### Hipóteses

| id | resultado |
|---|---|
| HR1 recomposição quebra atalhos | **confirmada em parte.** Com a arquitetura antiga, o gerador (B1) sobe a AUC em A+ nos dois aparelhos (+0,03/+0,04) e o F1 da m. lavar A+ em +0,12. Na faixa A, o ganho é pequeno (TV +0,013) ou negativo (m. lavar −0,011 de AUC). |
| HR2 consistência acrescenta | **refutada na forma prevista.** B2 ≈ B1; em A+ na m. lavar, B2 < B1. Δ_fundo não cai com L_inv de forma consistente. |
| HR3 distratores reduzem FP da m. lavar | não isolada: sem ablação só de R4. Os braços sem R1/R2/R4 (B2r, Q9) perdem na m. lavar, sobretudo na faixa A (Q9: 0,68 de AUC). |
| HR4 diversidade de kernels | **confirmada:** B5 > B4 nas 4 combinações (+0,007 a +0,026 de AUC). |
| HR5 quantil fecha a lacuna | **refutada no LOHO** (§2). No teste, porém, ajudou onde a calibração da faixa A falhou (§6). |
| HR6 intra precisa de seleção própria | **confirmada:** m. lavar intra 0,797 contra 0,398 do v2 no teste; TV intra 0,677 contra 0,616 do v2 (§6). |
| HR7 auxiliares ajudam na m. lavar | **nula:** B6 ≈ B5 (±0,007). |

**A leitura que muda o projeto.** Na tabela, o que mais pesa é o backbone. A ResNet com 4 canais,
com entrada em W/P_ref, mais o conjunto de kernels, vale mais que os termos de par. L_sens chega a
atrapalhar o F1 da TV. A recomposição ajuda sobretudo em A+ (mais fundos e ativações para
recombinar).

### E2.2 — mistura (R9, A+, 3 sementes)

Fração real 0,3/0,4/0,5/0,7: AUC da TV 0,729/0,742/0,741/0,740; m. lavar 0,902/0,899/0,901/0,902.
Plano; 0,4 mantido.

### E2.3 — seleção gulosa (R9, 2 sementes, épocas curtas)

- **TV:** casa 8 → + casa 15 (AUC 0,737 → 0,779); o passo seguinte não ganha. Com todas as 9
  casas, o R9 tem 0,738.
- **Máquina de lavar:** casa 8 → + casa 21 (0,813 → 0,896), e parou. Com todas as 16 casas, o R9
  tem 0,899.

A seleção foi feita nas mesmas casas de validação da escada, portanto é otimista. A TV adota
{8, 15}. Na máquina de lavar, o ganho guloso sobre todas as casas é nulo, e o desempate declarado
fica com as 16 casas (mais diversidade).

### Portão G2

O ReCo (B5) vence B0 com IC pareado excluindo zero na TV e na máquina de lavar, nas duas faixas,
sem perder no outro aparelho. B6/B7 não têm ganho resolvível sobre B5. **Adotado: B5 em todas as
células cross.**

## 5. F3 — variante intra (holdout: 15% finais do ano da casa 8)

3 sementes, I9 (ReCo intra: R0/R1/R2, hora do dia) contra C0 (a CNN do B0 só com janelas reais):

| | contexto | TV AUC / oráculo | m. lavar AUC / oráculo |
|---|---|---|---|
| A, C0 | 4 h | 0,881 / 0,715 | 0,957 / 0,505 |
| A, C0 | 12 h | **0,882 / 0,722** | 0,957 / 0,527 |
| A, I9 | 4 h | 0,844 / 0,669 | **0,988 / 0,765** |
| A, I9 | 12 h | 0,800 / 0,594 | 0,871 / 0,254 |
| A+, C0 | 4 h / 12 h | 0,759 / 0,833 | 0,938 / 0,920 |
| A+, I9 | 4 h / 12 h | 0,859 / 0,811 | 0,971 / 0,856 |

**Portão G3:**

- **TV intra = C0**, faixa A, 12 h: na mesma casa, a recomposição não ajuda.
- **M. lavar intra = conjunto I5–I25**, faixa A, 4 h.

## 6. F4 — fechamento pré-registrado (olhada única por faixa)

Pré-registros datados, gravados antes das rodadas: `resultados/v3/preregistro_A.json` (hash
`35e79f3c3bc98edc`) e `preregistro_Amais.json` (hash `250815e4228749e7`).

- **Modelos:** 10 sementes; ReCo = média de postos (q) ou média dos logits (d) de R5–R25.
- **Regras primárias:**
  - cross, faixa A: d_A (calibrada no holdout da casa 8);
  - cross, faixa A+: d_L (calibrada nas casas 3/7/20);
  - intra: d_A no holdout intra.
- **Olhadas:** 6 linhas novas em `resultados/v2/olhadas.csv`, uma por aparelho × rodada. O total
  do projeto é 13; o pré-registro registrava 7 antes do v3. A rodada da m. lavar foi dividida em
  processos no meio (`--continuacao`), sem nova olhada.
- **Família primária:** TV cross × artigo, uma comparação. As demais células ficam na família
  secundária, com Holm dentro de cada faixa.

| célula | faixa | regra | F1 [IC95%] | artigo | trivial | vs artigo | vs trivial | AUC | oráculo | ativ. |
|---|---|---|---|---|---|---|---|---|---|---|
| **TV cross** | **A+** | **d_L*** | **0,818 [0,761; 0,862]** | 0,54 | 0,346 | **supera** (p < 0,001) | vence | 0,980 | 0,867 | 39 |
| TV cross | A | d_A* | 0,424 [0,204; 0,632] | 0,54 | 0,346 | inconclusivo | inconclusivo | 0,951 | 0,776 | 39 |
| TV cross | A | d_L / q | 0,713 / 0,776 | 0,54 | | supera (secundárias) | vence | | | |
| **m. lavar cross** | **A+** | **d_L*** | **0,613 [0,489; 0,713]** | 0,30 | 0,137 | **supera** (p_Holm < 0,001) | vence | 0,924 | 0,635 | 29 |
| m. lavar cross | A | d_A* | 0,069 [0,025; 0,128] | 0,30 | 0,137 | **perde** | perde | 0,766 | 0,334 | 29 |
| m. lavar cross | A | d_L / q | 0,196 / 0,177 | 0,30 | | inconclusivo / perde | — | | | |
| **m. lavar intra** | A | d_A* | **0,797 [0,650; 0,890]** | 0,63 | 0,086 | IC acima (p = 0,036; **p_Holm = 0,072**) | vence | 0,992 | 0,860 | 16 |
| TV intra | A | d_A* | 0,677 [0,612; 0,735] | 0,68 | 0,386 | inconclusivo (empata) | vence | 0,862 | 0,682 | 44 |

\* regra primária pré-registrada. Tudo em `resultados/v3/final_{A,Amais}__celulas.csv` e
`final_holm.csv`.

Leitura:

- **TV cross, faixa A+: vitória na família primária.** F1 0,818, limite inferior 0,761 contra
  0,54 do artigo. A AUC no ECO 2 foi de 0,853 (v2) para 0,980. A expectativa declarada no plano
  era ~0,59.
  - O ECO 2 tem um canal de TV limpo: o agregado registra 149 dos 160 W do degrau (E0.1 do v2).
    Um modelo que aprende degraus em W/P_ref, sem o atalho de co-ocorrência, se beneficia disso.
  - É uma casa só (39 ativações); o número não deve ser lido como geral.
  - A escolha das fontes {8, 15} foi feita na validação britânica, não no ECO.
- **Máquina de lavar cross, faixa A+: vitória com IC** sobre o artigo e o trivial (0,613 contra
  0,30). É a primeira vez no projeto.
- **Faixa A cross: a calibração d_A (holdout da casa 8) não transfere.**
  - Na TV, a AUC é 0,951, mas a d_A prevê 8% de positivos para uma prevalência de 21%.
  - As regras que usam os rótulos das casas 3/7/20 (d_L) ou o quantil (q) chegam a 0,71–0,78,
    mas são secundárias e não mudam o placar pré-registrado.
  - É o caso que motivou a decisão por quantil (§3.6). O LOHO britânico não o reproduziu (§2),
    então o G1 não a adotou.
- **Máquina de lavar intra:** 0,797, com o IC inteiro acima de 0,63. Após Holm, p = 0,072: não
  é vitória formal na família secundária. São 16 ativações.
- **TV intra:** empata com o artigo (0,677 contra 0,68), acima do v2 (0,616).
- **Freezer:** não reavaliado; o placar do v2 vale (supera o artigo nas duas células).

## 7. Placar do projeto contra o artigo (números pré-registrados)

| célula | artigo | v2 melhor faixa | **v3** |
|---|---|---|---|
| TV cross | 0,54 | 0,476 — perde | **0,818 — supera (A+)** |
| m. lavar cross | 0,30 | 0,245 — inconclusivo | **0,613 — supera (A+)** |
| TV intra | 0,68 | 0,616 — inconclusivo | 0,677 — empata |
| m. lavar intra | 0,63 | 0,398 — perde | 0,797 — IC acima, p_Holm 0,07 |
| freezer intra | 0,46 | **0,569 — supera** | (v2 congelado) |
| freezer cross | 0,78 | **0,805/0,824 — supera** | (v2 congelado) |

## 8. Ressalvas e o que fica em aberto

- **Portão E2.0 reprovado ao pé da letra.** As operações que alteram co-ocorrência (R1/R2/R4)
  são separáveis das reais. O modelo que venceu foi treinado com elas. Q9 (só R0+R3) é a
  alternativa que respeita o portão: vence na TV A+, mas perde na m. lavar.
- **Seleção otimista.** E2.3, e a escolha entre B5/B6/B7 e entre I9/C0, foram feitas nas mesmas
  casas de validação que medem a escada.
- **Uma olhada por faixa, mas por aparelho.** Com 13 olhadas acumuladas no ECO/mês de teste, os
  números do v2 e do v3 não são independentes da história do projeto.
- **Faixa A cross.** A lacuna é de decisão, não de ranking (AUC 0,95 na TV, 0,77 na m. lavar).
  Uma regra de decisão pré-registrada que não dependa só do holdout da casa 8 é o próximo passo
  natural.
- **Fora desta rodada:** UK-DALE/IDEAL como fonte (D5), ReCo para o freezer, adaptação
  adversarial (faixa C), 3 tentativas por kernel.

## 9. v3.1 — melhorias pós-v3 (itens 1, 2 e 6)

Tudo decidido só na validação (REFIT 3/7/20); pré-registros `resultados/v3/preregistro_v31_A.json`
(hash `1339be0d2ebcf6be`) e `preregistro_v31_Amais.json` (hash `cb94a9dfde8e4cf3`), gravados antes
das rodadas; 4 olhadas novas em `resultados/v2/olhadas.csv` (total do projeto: 17).

**Item 6 — simplificação (CRAFT).** O ReCo sem L_inv, L_sens e cabeças auxiliares (conjunto G5–G25:
5 ResNets + gerador R0–R4, só BCE) empata com o B5 na validação: TV A AUC 0,769 × 0,770 e F1 d
0,606 × 0,583; m. lavar A F1 d agregado 0,460 × 0,460 (AUC 0,861 × 0,884); m. lavar A+ 0,531 × 0,534.
Os componentes descartados ficam no código só para reproduzir o v3 (`src/experimento_reco.py`).

**Item 1 — decisão calibrada em várias casas.** `src/experimento_decisao_v3.py`, LOHO nas casas
3/7/20 com as predições da escada (F1/oráculo, mediana das casas):

| | d_A (holdout da casa 8) | **d_L (outras casas empilhadas)** | d_M (mediana dos limiares) | q (quantil) |
|---|---|---|---|---|
| TV A | 0,63 | **0,88** | 0,86 | 0,96 |
| TV A+ | 0,70 | **0,95** | 0,96 | 0,94 |
| m. lavar A | 0,91 | **0,99** | 0,96 | 0,89 |
| m. lavar A+ | 1,00 | 0,98 | 0,98 | 0,87 |

d_L vira a primária em todas as células cross. **Declaração:** a falha da d_A na faixa A já era
visível neste LOHO antes do F4 do v3; ela foi pré-registrada por convenção do v2 sem essa checagem.

**Item 2 — UK-DALE** (`scripts/preparar_ukdale_v3.py`; `tables` instalado). Auditoria de bordas (D5):

| casa | aparelho | prevalência | degrau plugue / agregado (5 min) | papel |
|---|---|---|---|---|
| UK-DALE 1 | TV | 11,4% | +62 / +2 W (TV quase invisível no agregado) | **alvo novo** (jan–mar 2015) |
| UK-DALE 4 | TV + set-top | 100% | — | excluída (degenerada) |
| UK-DALE 5 | TV | 8,9% | +31 / +178 W | fonte da TV (A+) |
| UK-DALE 2 | m. lavar | 1,5% | +637 / +661 W | **alvo novo** (120 dias) |

Washer dryer (casas 1 e 5) = categoria "dryer", como no protocolo. Agregado do UK-DALE em potência
aparente, plugues em ativa (declarado, sem conversão). Na validação, TV A+ com {8, 15} × {8, 15,
UK-DALE 5}: AUC 0,770 × 0,762, F1 d 0,578 × 0,600 — empate; desempate por diversidade.

### Resultado (10 sementes, regra primária d_L)

| célula | faixa | F1 [IC95%] | artigo | trivial | v3 | AUC | oráculo | ativ. |
|---|---|---|---|---|---|---|---|---|
| **TV cross (ECO 2)** | A | **0,683 [0,604; 0,756] — supera** | 0,54 | 0,346 | 0,424 | 0,976 | 0,853 | 39 |
| **TV cross (ECO 2)** | A+ | **0,834 [0,784; 0,873] — supera** | 0,54 | 0,346 | 0,818 | 0,982 | 0,874 | 39 |
| m. lavar cross (ECO 1) | A | 0,262 [0,162; 0,363] — inconclusivo | 0,30 | 0,137 | 0,069 | 0,747 | 0,316 | 29 |
| **m. lavar cross (ECO 1)** | A+ | **0,626 [0,500; 0,735] — supera** | 0,30 | 0,137 | 0,613 | 0,931 | 0,639 | 29 |
| TV novo (UK-DALE 1) | A | 0,287 [0,257; 0,318] — vence trivial | — | 0,205 | — | 0,871 | 0,482 | 122 |
| TV novo (UK-DALE 1) | A+ | 0,279 [0,250; 0,306] — vence trivial | — | 0,205 | — | 0,892 | 0,515 | 122 |
| m. lavar novo (UK-DALE 2) | A | 0,194 [0,129; 0,259] — vence trivial | — | 0,029 | — | 0,914 | 0,220 | 58 |
| **m. lavar novo (UK-DALE 2)** | A+ | **0,541 [0,452; 0,626] — vence trivial** | — | 0,029 | — | 0,976 | 0,590 | 58 |

Tudo em `resultados/v3/final_v31_{A,Amais}__celulas.csv` (inclui d_A e q).

Leitura:

- **Item 1 funcionou onde devia:** TV cross faixa A 0,424 → 0,683 e m. lavar A 0,069 → 0,262. Com
  só a casa 8 no treino, a TV agora também supera o artigo.
- **Confirmação independente da m. lavar:** no UK-DALE 2, nunca visto, A+ dá 0,541 (oráculo 0,590,
  AUC 0,976) contra trivial 0,029. O resultado do ECO não era casual.
- **TV no UK-DALE 1: ranking bom, decisão ruim.** AUC 0,87–0,89, mas a d_L prevê 70% de positivos
  para 11% de prevalência (F1 0,28 contra oráculo 0,52); o quantil daria 0,44–0,47. É uma casa em
  que o agregado quase não registra a TV (+2 W): a calibração britânica não transfere, e é a
  mesma lacuna de decisão da faixa A do v3, agora em outra casa. A regra de decisão continua sendo
  o ponto fraco para a TV.
- **Ganho de A+ nos alvos novos:** m. lavar 0,194 → 0,541; TV sem mudança.

## 10. Pós-v3.1 — CRAFT + ramo de evidência, e CRAFT no freezer (só validação)

Rodadas de 27 e 29/09, todas **só na validação** (LOHO nas casas REFIT 3/7/20, `olhar_teste=false`,
5 sementes); nenhuma olhada nova em `resultados/v2/olhadas.csv` (total segue 17). Referência B0 =
A0_4hcf do v2; A2v2 = ramo só-evidência do v2; A5v2 = campeão do v2. Decisão `d` calibrada nas
outras casas de validação; IC pareado por bootstrap de dias contra B0 (`relatorio_v3.py`).

Três formas de juntar o CRAFT (conjunto G5–G25) com o ramo de evidência do v2:

- **CRAFT+E** — o A2v2 como 6º membro da média de postos (sem treino novo).
- **Empilhamento** (`src/empilhamento_v3.py`) — escore `w·E + (1−w)·C`, logits padronizados no
  holdout temporal das fontes, `w` (grade 0–1, passo 0,1) escolhido em LOHO aninhado pela AUC.
- **FUSÃO** (F5–F25, `ResNetEvid` em `src/reco.py`) — os 4 canais do ramo de evidência (a+, a−,
  estados para frente e para trás) entram na ResNet ao lado dos canais do CRAFT; perda BCE +
  0,3 × perda de transições do v2. Treino ponta a ponta.

### TV e máquina de lavar (média das 3 casas de validação)

| aparelho | faixa | modelo | AUC | F1 d | F1 q | oráculo |
|---|---|---|---|---|---|---|
| TV | A | CRAFT | 0,769 | 0,606 | 0,574 | 0,645 |
| | | CRAFT+E | **0,777** | **0,612** | 0,577 | 0,649 |
| | | FUSÃO | 0,761 | 0,594 | 0,575 | 0,639 |
| TV | A+ | CRAFT | 0,762 | 0,600 | 0,584 | 0,646 |
| | | CRAFT+E | 0,767 | **0,604** | 0,584 | 0,643 |
| | | FUSÃO | **0,773** | 0,589 | **0,596** | 0,656 |
| m. lavar | A | CRAFT | 0,861 | 0,420 | 0,373 | 0,427 |
| | | CRAFT+E | **0,872** | 0,423 | **0,386** | 0,438 |
| | | FUSÃO | 0,858 | **0,426** | 0,372 | 0,442 |
| m. lavar | A+ | CRAFT | 0,910 | 0,496 | 0,475 | 0,519 |
| | | CRAFT+E | **0,915** | **0,506** | **0,480** | 0,526 |
| | | FUSÃO | 0,912 | 0,499 | 0,473 | 0,525 |

Arquivos: `resultados/v3/fus_{A,Amais_tv,Amais_wm}__{resumo,pareado}.csv`.

Empilhamento (`resultados/v3/empilhamento__resumo.csv`; na faixa A+ o CRAFT de validação é
`craft_tv815u5` / `craft_Amais_wm`):

| aparelho | faixa | AUC CRAFT → empilhado | F1 d CRAFT → empilhado | w médio |
|---|---|---|---|---|
| TV | A | 0,777 → 0,782 | 0,606 → 0,616 | 0,27 |
| TV | A+ | 0,761 → 0,764 | 0,600 → 0,608 | 0,11 |
| m. lavar | A | 0,867 → 0,883 | 0,420 → 0,426 | 0,41 |
| m. lavar | A+ | 0,911 → 0,914 | 0,496 → 0,494 | 0,37 |

Leitura:

- **A fusão ponta a ponta não acrescenta nada.** Nas quatro células a FUSÃO fica dentro de ±0,01
  do CRAFT em AUC e F1 d. Os ICs pareados contra B0 se sobrepõem. Na TV A, o F1 d da FUSÃO não
  se separa de B0 (dF1 d IC [−0,018; 0,025]), enquanto o do CRAFT se separa ([0,001; 0,039]).
- **Juntar os escores depois do treino ajuda pouco, mas de forma consistente.** CRAFT+E e o
  empilhamento sobem a AUC em 0,003–0,016 em todas as células. O F1 d sobe até 0,01, exceto na m.
  lavar A+, onde o empilhamento fica em −0,002. É um ganho do tamanho do ruído entre sementes. Não
  justifica uma olhada nova no teste sozinho.
- **O ramo de evidência sozinho é mais fraco que o CRAFT na TV e na m. lavar.** Na TV A, o A2v2
  tem F1 d 0,506 contra 0,606 do CRAFT. Por isso o `w` escolhido fica baixo (0,1–0,4).

### Freezer

O CRAFT foi estendido ao freezer (`reco.py`: categoria `freezer`, limiar 10 W, proporções da TV,
escala 0,7–1,5; fonte A = casa 8, A+ = casas do portão do v2; validação 3/7/20). Duas versões:

- `craft_fz` — primeira tentativa, com a geladeira/combinado como distrator.
- `craft_fz2` — **sem distrator** ("a geladeira confunde mais do que ensina") e fundo = A − P em
  qualquer janela (`FUNDO_LIMPO = False`). Com um aparelho cíclico quase não há janela de 4 h com
  ele desligado, então exigir janela limpa deixava poucos fundos.

| faixa | modelo | AUC | F1 d | F1 q | oráculo |
|---|---|---|---|---|---|
| A | B0 | 0,494 | 0,692 | 0,588 | 0,692 |
| | CRAFT (fz, com distrator) | 0,552 | 0,692 | 0,619 | 0,692 |
| | CRAFT (fz2) | 0,671 | 0,700 | 0,667 | 0,717 |
| | CRAFT+E (fz2) | 0,689 | 0,701 | 0,667 | 0,725 |
| | FUSÃO | 0,669 | 0,701 | 0,659 | 0,716 |
| | A2v2 | 0,704 | 0,695 | 0,680 | 0,715 |
| | **A5v2** | **0,712** | **0,703** | **0,681** | **0,718** |
| A+ | B0 | 0,727 | **0,732** | 0,708 | **0,746** |
| | CRAFT (fz2) | 0,726 | 0,724 | 0,707 | 0,738 |
| | CRAFT+E (fz2) | **0,735** | 0,727 | 0,711 | 0,743 |
| | FUSÃO | 0,725 | 0,727 | 0,711 | 0,739 |
| | A2v2 | 0,728 | 0,726 | **0,716** | 0,740 |

Arquivos: `resultados/v3/craft_fz{,2}_{A,Amais}__{resumo,pareado}.csv`, `fus_{A,Amais}_fz__*.csv`.

Leitura:

- **Tirar o distrator e liberar o fundo foi decisivo para o CRAFT no freezer.** Na faixa A, a AUC
  sobe de 0,552 para 0,671, e o `delta_fundo` cai de ~20 para ~2,5.
- **Ainda assim, o CRAFT não supera o v2 no freezer.** Na faixa A, A2v2/A5v2 continuam à frente em
  AUC (0,70–0,71 × 0,67). Na A+, tudo empata com B0 dentro de ±0,01. Nenhuma variante (CRAFT,
  CRAFT+E, FUSÃO) passa o B0 de forma separável em F1 d na A+ (todos os ICs cruzam zero).
- **O F1 d do freezer quase não discrimina modelos.** Com prevalência de 0,49–0,59 nas casas de
  validação, o F1 fica colado no trivial (0,69–0,73) mesmo com AUC 0,49. A comparação útil aqui
  é a AUC.
- **Empilhamento no freezer:** feito com a versão antiga (`craft_fz`, AUC 0,55), então `w = 1` na
  faixa A (só evidência). Na A+, AUC 0,733 × 0,728 do A2v2 sozinho. Não foi refeito com `craft_fz2`.

### Decisão proposta (a confirmar)

- O **CRAFT continua sendo o modelo do v3.1** para TV e m. lavar. A fusão ponta a ponta é descartada
  (sem ganho). CRAFT+E/empilhamento ficam como opção barata, sem
  olhada no teste enquanto o ganho for do tamanho do ruído.
- O **freezer continua com o modelo do v2** (o único que já superava o artigo no v2).

> **Atualização (§11):** a decisão acima foi substituída pelo fechamento unificado. O CRAFT+E virou
> o modelo único para os três aparelhos.

## 11. Fechamento unificado — CRAFT+E em todas as células (sem UK-DALE)

Pedido: um modelo só para o artigo, sem UK-DALE por enquanto. Pré-registro
`resultados/v3/preregistro_craftE.json` (rascunho `6841e05124e48277`, aprovado antes de qualquer
rodada de teste: `2def81803f358a29`). Script: `src/final_craftE.py`. Saída:
`resultados/v3/final_craftE__celulas.csv`.

**Decidido só na validação** (`src/validacao_craftE.py`, `resultados/v3/validacao_craftE__resumo.csv`):

- **E como 6º membro:** CRAFT+E ≥ CRAFT em AUC e F1 d_L nas casas 3/7/20, nas seis combinações
  aparelho × faixa. No holdout da casa 8 (proxy do intra), empata.
- **Sem variante do intra:** na TV intra, 12 h e hora do dia já tinham sido reprovadas (F3). Um
  membro só com janelas reais não muda o holdout da casa 8.
- **Freezer:** configuração `craft_fz2` (sem distrator, fundo de qualquer janela).

**Modelo.** Média dos logits de 6 membros com peso igual: G5, G7, G9, G15, G25 (CRAFT) e o A2 do
v2 (ramo só-evidência). Um treino por aparelho e faixa serve o cross e o intra. Fontes só REFIT:

| aparelho | faixa A | faixa A+ |
|---|---|---|
| TV | 8 | 8, 15 |
| m. lavar | 8 | 8 + as 15 casas do portão G0 |
| freezer | 8 | 8, 1, 4, 6, 13, 17, 18 |

**Regras de decisão:** d_L no cross; d_A no intra.

**Rodadas:** 8 treinos novos em paralelo na GPU (29/09, 15:13–17:20). Foram reaproveitadas
`craftF_A` (TV e m. lavar A), `craftF_Amais` (m. lavar A+) e `final_A` (A2 da m. lavar A).
Olhadas: 8 automáticas + 1 da análise; total do projeto 26.

### Família primária (Holm, 6 comparações)

| célula | F1 [IC95%] | artigo | trivial | p_Holm | vs artigo | vs trivial | AUC | oráculo | ativ. |
|---|---|---|---|---|---|---|---|---|---|
| **TV cross A+** | **0,846 [0,793; 0,887]** | 0,54 | 0,346 | 0,002 | **supera** | vence | 0,984 | 0,890 | 39 |
| **m. lavar cross A+** | **0,620 [0,495; 0,729]** | 0,30 | 0,137 | 0,002 | **supera** | vence | 0,932 | 0,630 | 29 |
| freezer cross A+ | 0,791 [0,772; 0,807] | 0,78 | 0,790 | 0,53 | empata | empata | 0,710 | 0,821 | 1 938 |
| TV intra A | 0,697 [0,617; 0,773] | 0,68 | 0,386 | 0,65 | empata | vence | 0,896 | 0,712 | 44 |
| **m. lavar intra A** | **0,847 [0,734; 0,919]** | 0,63 | 0,086 | 0,015 | **supera** | vence | 0,995 | 0,876 | 16 |
| **freezer intra A** | **0,692 [0,652; 0,738]** | 0,46 | 0,496 | 0,002 | **supera** | vence | 0,858 | 0,696 | 403 |

### Família secundária: cross na faixa A (só a casa 8 no treino, como o artigo)

| célula | F1 d_L [IC95%] | artigo | trivial | p_Holm | vs artigo | vs trivial |
|---|---|---|---|---|---|---|
| TV | **0,693 [0,613; 0,766]** | 0,54 | 0,346 | 0,003 | **supera** | vence |
| m. lavar | 0,265 [0,164; 0,367] | 0,30 | 0,137 | 0,52 | inconclusivo | vence |
| freezer | 0,748 [0,727; 0,765] | 0,78 | 0,790 | 0,003 | **perde** | perde |

O CRAFT sem o E fica a ±0,01 do CRAFT+E em quase todas as células (linhas `CRAFT` no csv).

Leitura:

- **Quatro vitórias formais em seis na família primária.** TV cross 0,846 e m. lavar cross 0,620
  confirmam o v3.1, agora sem UK-DALE. A **m. lavar intra passa o Holm pela primeira vez** (0,797 →
  0,847; p_Holm 0,07 → 0,015). O **freezer intra sobe de 0,569 (v2) para 0,692**, e o limite
  inferior do IC fica 0,19 acima do artigo.
- **TV intra: empate, como previsto.** 0,697 contra 0,68, e o oráculo é 0,712. O limite é o ranking
  (AUC 0,90), não a decisão.
- **Freezer cross: o CRAFT+E perde para o v2.** Na faixa A+ ele empata com o artigo e com o trivial
  (0,791 × 0,790). O v2 (A0_4hcf) tinha dado 0,824 e vencido o trivial. Na faixa A, o CRAFT+E fica
  **abaixo do artigo** (0,748).
  - A AUC no ECO (0,63–0,71) é bem menor que na casa 8 (0,86): a recomposição aprende o freezer da
    casa 8 e das britânicas, mas não transfere para os freezers suíços.
  - Com prevalência de 65%, o F1 fica colado no trivial.
- **Placar do modelo único:**

  | família | supera | empata | inconclusivo | perde |
  |---|---|---|---|---|
  | primária (6) | 4 | 2 | — | 0 |
  | secundária, faixa A cross (3) | 1 | — | 1 | 1 (freezer) |

- **Declarações obrigatórias:** 26 olhadas acumuladas no teste; portão E2.0 reprovado ao pé da
  letra (§8); uma casa por célula cross.

## 12. CRAFT a 15 min — as células de 15 min do artigo (rodada pós-fechamento)

Pedido do usuário em 2026-10-01. **Rodada nova, depois do fechamento** (o pré-registro do §11 dizia que
não haveria outra); precisa ser declarada assim. Mesmas casas e meses de teste do §11, reamostrados a
15 min: **não é um teste independente**.

- Código: `src/taxa.py` (`NILM_TAXA=900`; sem a variável, tudo fica a 5 min, 27 testes passam).
  Tamanhos convertidos pela mesma duração (24 h → 96, 4 h → 16); passo 4 → 1 (A) e 8 → 3 (A+);
  arquitetura, kernels e treino inalterados. Dados: `scripts/preparar_artigo.py --taxas 15min` para as
  casas REFIT de validação e do POOL, mais as máscaras de lacuna de 15 min.
- Validação (só casas 3/7/20): contexto do CRAFT 4 h (AUC 0,762) × 12 h (0,765), empate → 4 h
  (`src/validacao_ctx15.py`, `validacao_ctx15__sementes.csv`).
- Pré-registro `resultados/v3/preregistro_craftE15.json` (hash `c6afccfc583788e6`, rascunho
  `4157a608dbce01ea`), aprovado antes do teste. 13 linhas novas em `olhadas.csv` (total 46).
- Resultado: `final_craftE15__celulas.csv`; comparação: `src/comparar_taxas.py` →
  `comparacao_5x15__celulas.csv`.

Família primária a 15 min (regra pré-registrada, Holm):

| célula | CRAFT 15 min | artigo 15 min | trivial | p_Holm | | CRAFT 5 min | artigo 5 min |
|---|---|---|---|---|---|---|---|
| TV cross (A+) | **0,905 [0,867; 0,936]** | 0,36 | 0,356 | 0,002 | supera | 0,846 | 0,54 |
| m. lavar cross (A+) | **0,450 [0,317; 0,560]** | 0,30 | 0,151 | 0,03 | supera | 0,620 | 0,30 |
| freezer cross (A+) | 0,899 [0,883; 0,914] | 0,84 | 0,897 | 0,002 | supera* | 0,791 | 0,78 |
| TV intra | **0,671 [0,596; 0,738]** | 0,53 | 0,399 | 0,002 | supera | 0,697 | 0,68 |
| m. lavar intra | **0,649 [0,430; 0,801]** | 0,35 | 0,092 | 0,03 | supera | 0,847 | 0,63 |
| freezer intra | 0,650 [0,628; 0,675] | 0,55 | 0,611 | 0,002 | supera* | 0,692 | 0,46 |

Secundária (faixa A cross): TV 0,760 supera (0,36); m. lavar 0,271 inconclusivo (0,30); freezer
0,897 "supera" (0,84), mas prevendo ligado em 100% do tempo.

Leitura:

- **6 de 6 formalmente, 4 de 6 com substância.** \* No freezer o artigo a 15 min fica **abaixo do
  classificador trivial** (cross 0,84 × 0,897; prevalência no ECO 81%). O CRAFT prevê "ligado" em 98%
  do tempo (A+) e 100% (A): F1 = trivial, AUC 0,68 (A+) e 0,49 (A). No intra, prevê ligado em 81% contra
  prevalência de 44%; vence o trivial por só 0,04. Essas duas vitórias vêm da regra de decisão
  colando no trivial, não de reconhecer o freezer.
- **TV e m. lavar resistem bem a 15 min; o artigo cai mais.** TV: AUC quase igual (cross 0,987 ×
  0,984; intra 0,866 × 0,896) e o F1 até sobe no cross. M. lavar perde F1 (cross 0,62 → 0,45, intra
  0,85 → 0,65), mas o artigo cai junto (intra 0,63 → 0,35) e a margem continua grande.
- **O freezer é o que mais sofre com a taxa:** AUC intra 0,86 → 0,70, cross A 0,63 → 0,49 (acaso).
- O rótulo muda com a taxa (média de 15 min ≥ limiar): prevalências e triviais diferem das de 5 min;
  F1 − trivial e AUC são a comparação justa entre taxas (`comparacao_5x15__celulas.csv`).

## 13. S6 — CRAFT treinado no UK-DALE 1, teste no REDD 1 (5 min)

Pré-registro `resultados/v3/preregistro_ukdale_redd.json` (hash `aa9a8e9641b1ce68`, rascunho
`c8b6bd41057f132d`), aprovado em 2026-10-01. S4/S5 (§ pré-registro `preregistro_ukdale_s.json`)
reaproveitados sem nova rodada. Casa do REDD escolhida sem rótulos: mais intervalos de 5 min com
agregado e os três plugues válidos (casa 1, 18,2 dias; `scripts/preparar_redd_cenario.py`). Lava-e-seca
do REDD = medidores 10 + 20 somados. Regra d_L (REFIT 3/7/20). 7 olhadas novas (total 53).
Resultado: `final_ukdale_r__celulas.csv` (`python src/final_ukdale.py --cenario s6`).

| aparelho | F1 [IC95%] | sempre ligado | prevalência | prevista | ativações |
|---|---|---|---|---|---|
| geladeira | **0,600** [0,554; 0,659] | 0,433 | 0,276 | 0,640 | 116 |
| lava-louças | 0,000 [0; 0] | 0,115 | 0,061 | 0,005 | 8 |
| lava-e-seca | 0,368 [0; 0,600] | 0,054 | 0,028 | 0,048 | 6 |

Diagnóstico (só sobre as predições já analisadas, sem nova olhada):

- **Geladeira:** ranking muito bom (AUC 0,95), mas o limiar calibrado nas REFIT prevê ligado em 64%
  contra 28% (revocação 0,99, precisão 0,43). Mesmo padrão do S5: problema de limiar, não de escore.
- **Lava-louças:** AUC 0,30, abaixo do acaso — o modelo ordena ao contrário. A lava-louças do REDD
  fica ligada ~100 min por ciclo (mediana 20 intervalos, ~390 W); a da UK-DALE 1 tem P_ref 121 W. Não
  transfere; F1 0 porque quase nada é previsto ligado.
- **Lava-e-seca:** AUC 0,83, F1 0,37 contra 0,05 do trivial, mas só 6 ativações (IC até zero).

## 14. Segundo cross-house no UK-DALE — treino casa 1, teste casa 5 (5 min)

Pré-registro `resultados/v3/preregistro_ukdale_casa5.json` (hash `3f10d3a40801daf4`, rascunho
`92ffa28624573a9d`), aprovado em 2026-10-02 com o pedido de comparar com o "sempre ligado". Casa 5:
132 dias, geladeira = fridge freezer, lava-loucas, lava-e-seca (mesmo tipo do alvo de treino). Antes, o
inventario (sem modelo) mostrou que a casa 2 e a de mais ciclos de lava-loucas (114 x 47). Regra d_L.
Olhadas: 1 do inventario + 7 da rodada (total 61). Resultado: `final_ukdale_c5__celulas.csv`.

| aparelho | F1 [IC95%] | sempre ligado | F1 - sempre ligado [IC95%] | prevalência | prevista | ativações |
|---|---|---|---|---|---|---|
| geladeira | **0,643** [0,623; 0,663] | 0,536 | +0,09 a +0,13 | 0,366 | 0,646 | 2715 |
| lava-loucas | **0,653** [0,576; 0,721] | 0,047 | +0,53 a +0,67 | 0,024 | 0,027 | 44 |
| lava-e-seca | **0,357** [0,299; 0,410] | 0,104 | +0,20 a +0,30 | 0,055 | 0,071 | 142 |

- Os tres vencem o "sempre ligado" com IC excluindo zero.
- Lava-loucas: 0,65 na casa 5 contra 0,40 na casa 2 — a da casa 5 se parece mais com a da casa 1.
- Lava-e-seca: 0,36 contra 0,14 da maquina de lavar da casa 2 — o mesmo tipo de aparelho transfere melhor,
  mas longe do intra (0,94).
- Geladeira: limiar de novo superestima "ligado" (65% x 37%; revocacao 0,89, precisao 0,51).

## 15. UK-DALE a 15 min — treino casa 1, intra-house (casa 1) e cross-house (casa 5)

Pré-registro `resultados/v3/preregistro_ukdale_15.json` (hash `c85a0262ce7efec3`, rascunho
`68434da8ad6e0527`), aprovado em 2026-10-02. Configuração de 15 min do artigo (contexto de 4 h, passo 1),
sem ajuste para o UK-DALE; dados de `scripts/preparar_ukdale_15min.py`. Mesmas casas e períodos já
olhados a 5 min. 7 olhadas (total 68). Resultado: `final_ukdale_15__celulas.csv`
(`NILM_TAXA=900 python src/final_ukdale.py --cenario 15`).

| aparelho | cenário | F1 5 min | F1 15 min [IC95%] | sempre ligado 15 min | F1 - sempre ligado 15 min [IC95%] |
|---|---|---|---|---|---|
| geladeira | intra (casa 1) | 0,830 | 0,747 [0,722; 0,771] | 0,578 | +0,12 a +0,20 |
| lava-louças | intra (casa 1) | 0,961 | 0,952 [0,899; 0,985] | 0,058 | +0,85 a +0,93 |
| lava-e-seca | intra (casa 1) | 0,936 | 0,896 [0,838; 0,946] | 0,143 | +0,69 a +0,82 |
| geladeira | cross (casa 5) | 0,643 | 0,599 [0,566; 0,626] | 0,550 | +0,02 a +0,08 |
| lava-louças | cross (casa 5) | 0,653 | 0,691 [0,629; 0,746] | 0,050 | +0,58 a +0,69 |
| lava-e-seca | cross (casa 5) | 0,357 | 0,388 [0,324; 0,444] | 0,120 | +0,22 a +0,31 |

- As seis células vencem o "sempre ligado" com IC excluindo zero; a geladeira cross por pouco.
- Lava-louças e lava-e-seca quase não perdem com a taxa (e no cross até sobem um pouco); a geladeira,
  cíclica, é a que mais perde (0,83 → 0,75 intra), como o freezer do REFIT.
- Geladeira cross: o limiar volta a superestimar "ligado" (65% x 38%).

## 16. Pós-quali — itens 2 e 3 (só validação, nenhuma olhada no teste)

Pedido do usuário em 2026-10-16: melhorar S1/S2 nas mesmas condições do artigo (faixa A, só a
casa 8 no treino). Código: `reco.py` (`Gerador(deform=..., trem=...)`, `injetar_deformado`,
`injetar_trem`, `periodicidade`, `ResNetGlob`), `experimento_reco.py` (`--deform`, `--trem`,
`--glob`, `--parada auc`, `--prop`), testes 8–11 em `tests/test_v3.py` (16 passam). Rodadas:
`logs/pos_quali/rodar.sh` (tags `q_*`, G5..G25, 5 sementes, config do `craft_A`/`craft_fz2_A`,
referência retreinada com o mesmo código). Avaliação: `src/validacao_posquali.py` →
`validacao_posquali__resumo.csv`. `olhadas.csv` continua com 68.

Conjunto ResNets+E, média de 5 sementes (cv = REFIT 3/7/20 em LOHO, proxy do S2; val = holdout
da casa 8, proxy do S1):

| aparelho | variante | cv AUC | cv oráculo | cv F1 d_L | val AUC | val oráculo |
|---|---|---|---|---|---|---|
| m. lavar | referência | 0,864 | 0,443 | 0,422 | 0,995 | 0,842 |
| m. lavar | R1 deformada | 0,887 | 0,452 | 0,389 | 0,993 | 0,769 |
| m. lavar | R1 deformada, 30% R1 | 0,894 | 0,459 | 0,405 | 0,994 | 0,798 |
| freezer | referência | 0,672 | 0,717 | 0,698 | 0,847 | 0,644 |
| freezer | trem de ciclos | 0,717 | 0,729 | 0,697 | 0,833 | 0,631 |
| freezer | periodicidade | 0,675 | 0,718 | 0,699 | 0,843 | 0,637 |
| freezer | trem + periodicidade | 0,719 | 0,728 | 0,696 | 0,835 | 0,635 |
| freezer | trem + period. + parada AUC | 0,717 | 0,728 | 0,698 | 0,835 | 0,633 |

Leitura:

- **Item 2 reprovado.** A deformação melhora a ordenação entre casas (AUC +0,02 a +0,03, 5/5
  sementes; oráculo +0,01 a +0,02), mas o F1 d_L cai (−0,02 a −0,03; 1–2/5 sementes acima),
  sobretudo na casa 20 (0,32 → 0,23–0,25), onde o limiar das outras duas casas passa a transferir
  pior. No holdout da casa 8 o oráculo cai 0,04–0,07 (0/5): o modelo perde o que é específico dos
  ciclos da própria casa. Ganho de escore pequeno demais para mover o S2 (oráculo do teste 0,32).
- **Item 3: o trem de ciclos melhora a ordenação, mas não o F1.** AUC cv +0,045 (5/5 sementes; casas
  3 e 7 sobem, a 20 cai um pouco), a maior AUC de freezer do projeto na faixa A. O F1 d_L não sai
  do nível do "sempre ligado" (F1 − trivial ≈ 0,005): com prevalência de 51–61%, AUC ~0,72 não
  basta. Holdout da casa 8: AUC −0,014 (0/5). Periodicidade e parada por AUC: efeito nulo.
- **Decisão:** nenhuma das variantes justifica uma rodada de teste no S1/S2. O trem fica como
  opção para o freezer se o critério de seleção passar a ser AUC (ex.: com regra de decisão
  adaptada à casa de destino, item 1).

## 17. Pós-quali — transfer learning para a casa de destino (só validação)

Pedido do usuário em 2026-10-17. Três formas, avaliadas nas casas REFIT 3/7/20 (faixa A, CRAFT =
G5..G25 + A2, 5 sementes): em cada casa, os primeiros k dias com dado adaptam e os dias depois do
14º avaliam (os mesmos para toda forma e todo k). Código: `experimento_reco.py --adaptar`
(`adabn`, `ajuste_fino`; ajuste fino de toda a rede com BN congelado, AdamW 1e-4, 100 passos,
lote 128, sem ajuste desses valores), `src/adaptacao_posquali.py`; rodadas `ta_tv`, `ta_wm`,
`ta_fz` (`logs/pos_quali/rodar_tl.sh`). `olhadas.csv` continua com 68.

F1 médio (3 casas × 5 sementes), mesmos dias de avaliação:

| aparelho | dL (atual) | limiar ideal | recal. melhor k | AdaBN | ajuste fino k=3 | k=7 | k=14 |
|---|---|---|---|---|---|---|---|
| TV | 0,610 | 0,659 | 0,615 (misto, 3) | 0,573 | 0,531 | 0,496 | 0,567 |
| m. lavar | 0,425 | 0,446 | 0,420 (misto, 3) | 0,380 | 0,262 | 0,296 | 0,364 |
| freezer | 0,706 | 0,720 | 0,718 (recal, 14) | 0,707 | 0,753 | 0,758 | **0,777** |

- **Forma 2 (recalibrar o limiar com k dias): sem ganho.** Nas casas britânicas a dL já fica a
  0,02–0,05 do limiar ideal, e o limiar de poucos dias traz mais ruído do que ganho. A lacuna da TV
  no ECO (0,69 × 0,86) não se reproduz na validação.
- **Forma 1 (AdaBN, sem rótulo): piora** a TV e a máquina de lavar (−0,04) e não muda o freezer. A
  normalização de entrada por casa já cobre parte disso.
- **Forma 3 (ajuste fino): só o freezer ganha.** k=14: +0,071 sobre a dL (12/15 pares positivos),
  acima do próprio limiar ideal do modelo base (0,72). Por casa: 20 0,70 → 0,85 e 7 0,76 → 0,82
  (o "sempre ligado" é 0,68 e 0,74); a casa 3 fica no trivial (0,66). Na TV e na máquina de lavar o
  ajuste fino piora (poucos acionamentos em k dias; com 1 dia, 1/3 dos casos de m. lavar não tem
  nenhum ciclo).
- **Leitura:** o freezer não transfere porque cada compressor tem a sua assinatura; 3–14 dias
  rotulados da casa de destino bastam para aprendê-la. TV e máquina de lavar já transferem o que
  dá com uma casa, e poucos dias de destino não acrescentam. Taxa e passos do ajuste fino não
  foram ajustados; o dano na TV/m. lavar pode diminuir com taxa menor ou só a última camada.
- **Teste (não feito):** um cenário S2-TL do freezer usaria k dias rotulados das casas ECO 1 e 2
  fora do mês de teste (a gravação vai de 01/06/2012 a 31/01/2013) e exigiria pré-registro.

## 18. Pós-quali — pós-processamento temporal das decisões (só validação)

Pedido do usuário em 2026-10-18. Sobre as predições das rodadas `ta_*` (CRAFT, 5 sementes):
suavização do logit por média móvel (largura w), decodificação por HMM de 2 estados (transições dos
rótulos de calibração × kappa) e filtro de duração mínima (preenche desligados ≤ g, apaga ligados
≤ r). Parâmetros e limiar escolhidos sem a casa avaliada: proxy do S2 = REFIT 3/7/20 em LOHO;
proxy do S1 = holdout da casa 8, 1ª metade calibra e escolhe, 2ª avalia. Código:
`src/posproc_posquali.py` → `posproc_posquali__resumo.csv`. `olhadas.csv` continua com 68.

| aparelho | cenário | base (regra d) | suavização | HMM | filtro | limiar ideal |
|---|---|---|---|---|---|---|
| TV | S2 proxy | 0,608 | 0,613 (13/15 melhoram) | 0,610 | 0,609 | 0,653 |
| TV | S1 proxy | 0,693 | 0,692 | 0,668 | 0,690 | 0,707 |
| m. lavar | S2 proxy | 0,421 | 0,420 | 0,408 | 0,418 | 0,441 |
| m. lavar | S1 proxy | 0,841 | 0,839 | 0,831 | 0,853 (2/5) | 0,883 |
| freezer | S2 proxy | 0,706 | 0,706 | 0,694 | 0,705 | 0,719 |
| freezer | S1 proxy | 0,651 | 0,651 | 0,649 | 0,656 (5/5) | 0,656 |

Leitura: **sem ganho útil.** O maior efeito é +0,005 (suavização na TV, S2) e +0,012 (filtro na
m. lavar, S1, em 2 de 5 sementes); o HMM piora quase sempre. A janela de 4 h já dá a cada decisão
o contexto vizinho: os erros que sobram não são oscilações curtas, e sim bordas de sessão e níveis
de escore. No freezer a regra escolhe w=1 e g=r=0 (nada a filtrar). Não justifica teste.

## 19. Pós-quali — saída de regressão da potência (só validação)

Pedido do usuário em 2026-10-19. Saída extra na ResNet (a última) que regride log1p(P/P_ref) no
intervalo central, perda BCE + 0,5·MSE (peso fixado antes de rodar); a decisão usa só o logit de
classificação. `experimento_reco.py --reg --lam_reg 0.5`; rodadas `q_tv_reg`, `q_wm_reg`,
`q_fz_reg` (`logs/pos_quali/rodar_reg.sh`), comparadas a `ta_tv`, `q_wm_base`, `q_fz_base` em
`src/validacao_posquali.py`. `olhadas.csv` continua com 68.

| aparelho | cv AUC ref → reg | cv F1 d_L ref → reg | val oráculo ref → reg |
|---|---|---|---|
| TV | 0,776 → 0,780 | 0,608 → 0,606 | 0,721 → 0,718 |
| m. lavar | 0,864 → 0,869 | 0,422 → 0,420 | 0,842 → 0,846 |
| freezer | 0,672 → 0,676 | 0,698 → 0,701 | 0,644 → 0,640 |

Intervenções (média do |Δlogit| nas casas 3/7/20, mesmos modelos G5/G7/G9 × 5 sementes; razão =
troca de fundo / remoção do alvo, menor = mais dependente da assinatura):

| aparelho | remoção ref → reg | troca de fundo ref → reg | razão ref → reg |
|---|---|---|---|
| TV | 0,91 → 1,24 | 2,58 → 3,67 | 2,84 → 2,96 |
| m. lavar | 2,29 → 2,41 | 2,57 → 2,76 | 1,12 → 1,15 |
| freezer | 0,95 → 0,89 | 2,92 → 2,83 | 3,07 → 3,18 |

Leitura: **efeito nulo.** F1 e AUC mudam ±0,005. A regressão só aumenta a escala dos logits na TV
(todas as intervenções crescem ~35%); a razão fundo/remoção não cai em nenhum aparelho. Ou seja,
a saída de regressão não desloca o modelo do contexto para a assinatura. As intervenções mostram
que TV e freezer respondem ~3× mais à troca de fundo que à remoção do próprio aparelho: é o alvo
a atacar (entrada só com diferenças; termo de invariância mais forte), medido por essa razão.

## 20. Pós-quali — entrada só com diferenças e invariância forte; resumo de §16–§20

Pedido do usuário em 2026-10-19 (classificador que dependa da assinatura, não do contexto).
`experimento_reco.py --entrada dif` (ResNet só com subidas e descidas em P_ref) e `--inv --perda
'{"lam_inv":λ,"m_inv":0,"g":0}'` (penaliza |z(R0) − z(R3)| em todo par, λ = 1 e 3, fixados antes).
Rodadas `q_{tv,wm,fz}_{dif,inv1,inv3}` (`logs/pos_quali/rodar_assinatura.sh`), 5 sementes.
Intervenções: `validacao_posquali__intervencoes.csv` (|Δlogit| nas casas 3/7/20, mesmos modelos da
referência; razão = troca de fundo / remoção do alvo). `olhadas.csv` continua com 68.

| aparelho | variante | cv F1 d_L (S2 proxy) | val oráculo (S1 proxy) | razão fundo/remoção |
|---|---|---|---|---|
| TV | referência | 0,608 | 0,721 | 2,83 |
| TV | só diferenças | 0,525 | 0,653 | 14,2 (remoção quase nula) |
| TV | invariância λ=1 / 3 | 0,510 / 0,507 | 0,455 / 0,452 | colapso (Δ ≈ 0) |
| m. lavar | referência | 0,422 | 0,842 | 1,10 |
| m. lavar | só diferenças | 0,415 | 0,825 | 1,06 |
| m. lavar | invariância λ=1 / 3 | 0,315 / 0,355 | 0,829 / 0,320 | 1,44 / colapso |
| freezer | referência | 0,698 | 0,644 | 2,77 |
| freezer | só diferenças | 0,682 | 0,606 | 3,56 |
| freezer | invariância λ=1 / 3 | 0,698 / 0,696 | 0,523 / 0,522 | colapso |

- **Invariância forte: reprovada.** Sem margem e sem porta, a penalidade é satisfeita por um logit
  quase constante (todos os Δ ≈ 0,001): o F1 cai ao nível do "sempre ligado" na TV e no freezer.
  Uma versão útil precisaria de uma âncora que impeça a solução constante (ex.: a sensibilidade
  L_sens junto, ou invariância na representação e não no logit).
- **Só diferenças: reprovada.** Sem o nível, a rede não sabe se a TV continua ligada quando a
  subida ficou fora da janela de 4 h (remoção do alvo quase não muda a resposta): TV −0,08.
  Máquina de lavar empata (AUC +0,02, F1 −0,007); freezer piora.

### Resumo de todos os experimentos pós-quali (§16–§20), só validação

| experimento | efeito no F1 (S2 / S1 proxy) | dependência da assinatura |
|---|---|---|
| R1 deformada (m. lavar) | ↓ / ↓ | razão 1,10 → ~1,0 |
| trem de ciclos (freezer) | = / ↓ leve (AUC cv +0,045) | **razão 2,77 → 1,21** |
| periodicidade, parada por AUC (freezer) | = | = |
| recalibrar limiar com k dias (TL forma 2) | = ou ↓ | — |
| AdaBN (TL forma 1) | ↓ TV e m. lavar | — |
| **ajuste fino com 14 dias rotulados (TL forma 3)** | **freezer +0,07**; TV e m. lavar ↓ | — |
| pós-processamento temporal | ≤ +0,01 | — |
| saída de regressão da potência | = | = |
| entrada só com diferenças | ↓ | piora (TV) |
| invariância forte | ↓ (colapso) | colapso |

Conclusão: **nas mesmas condições do artigo (uma casa, nenhum dado do destino), nenhuma mudança
melhorou o F1 na validação.** O que mexeu nos números foi dado novo: dias rotulados da casa de
destino (freezer, +0,07) e, antes, mais casas no treino (S3). A única mudança que tornou o modelo
claramente mais dependente da assinatura foi o trem de ciclos do freezer, sem efeito no F1 porque
a prevalência alta prende o F1 ao trivial.
