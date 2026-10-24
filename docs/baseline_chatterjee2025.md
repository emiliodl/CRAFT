# Linha de base: Chatterjee & Heer, *Energy & Buildings* 344 (2025) 116002

"Non-Intrusive Load Monitoring (NILM) with very low-frequency data from smart
meters in Switzerland". Open access, CC BY. Código:
https://github.com/Vincisteel/NILM_Experimentation.git

Protocolo a ser imitado. **Só a F1** é registrada aqui — MAE e NEP são métricas
de desagregação e estão fora do escopo. Só 5 e 15 min.

## Protocolo do artigo

| item | especificação |
|---|---|
| datasets | REFIT (20 casas, 8 s) e ECO (6 casas suíças, 1 s) |
| casas | **REFIT: casa 8**. **ECO: casas 1, 2 e 5** |
| aparelhos | freezer, kettle, computer, washing machine, microwave, television |
| reamostragem | 1, 5 e 15 min, pelas funções do nilmtk |
| intra-dataset | treino REFIT 07/03/2014–06/03/2015 (1 ano), teste 07/03/2015–06/04/2015 (1 mês) |
| cross-dataset | treino no mesmo intervalo do REFIT, teste em ECO 02/06/2012–01/07/2012 (1 mês) |
| algoritmos | FHMM, CO (clássicos); Seq2Point, SGN, DAE, BiLSTM (profundos) |
| métricas | MAE, F1, NEP |

Escolha das casas justificada no artigo pela "presença de um número maior de
aparelhos em comum".

## F1 — intra-dataset (REFIT casa 8 → REFIT casa 8)

### 5 min

| aparelho | FHMM | CO | Seq2Point | SGN | DAE | BiLSTM |
|---|---|---|---|---|---|---|
| kettle | 0,12 | 0,09 | 0,00 | 0,00 | 0,00 | 0,00 |
| freezer | 0,44 | 0,44 | **0,46** | 0,45 | 0,45 | 0,45 |
| washing machine | 0,21 | 0,06 | 0,36 | **0,63** | 0,18 | 0,10 |
| computer | **1,00** | 0,85 | **1,00** | **1,00** | **1,00** | **1,00** |
| microwave | 0,04 | 0,04 | 0,12 | **0,30** | 0,13 | 0,09 |
| television | 0,48 | 0,30 | 0,56 | **0,68** | 0,53 | 0,56 |

### 15 min

| aparelho | FHMM | CO | Seq2Point | SGN | DAE | BiLSTM |
|---|---|---|---|---|---|---|
| kettle | 0,11 | **0,16** | 0,00 | 0,00 | 0,00 | 0,00 |
| freezer | 0,39 | 0,51 | **0,55** | **0,55** | **0,55** | **0,55** |
| washing machine | 0,17 | 0,12 | 0,27 | **0,35** | 0,29 | 0,21 |
| computer | **1,00** | 0,82 | **1,00** | **1,00** | **1,00** | **1,00** |
| microwave | 0,07 | 0,07 | **0,28** | 0,27 | 0,15 | 0,15 |
| television | 0,47 | 0,40 | **0,53** | 0,42 | 0,47 | **0,53** |

## F1 — cross-dataset (treino REFIT casa 8 → teste ECO casas 1, 2, 5)

### 5 min

| aparelho | FHMM | CO | Seq2Point | SGN | DAE | BiLSTM |
|---|---|---|---|---|---|---|
| kettle | **0,10** | 0,08 | 0,00 | 0,00 | 0,00 | 0,00 |
| freezer | 0,76 | 0,56 | **0,78** | 0,76 | 0,76 | 0,76 |
| washing machine | 0,14 | **0,30** | 0,22 | 0,19 | 0,20 | 0,21 |
| computer | **0,49** | 0,45 | 0,44 | 0,44 | 0,44 | 0,44 |
| microwave | 0,01 | 0,01 | 0,07 | **0,09** | 0,04 | 0,02 |
| television | 0,38 | 0,35 | **0,54** | 0,52 | 0,35 | 0,35 |

### 15 min

| aparelho | FHMM | CO | Seq2Point | SGN | DAE | BiLSTM |
|---|---|---|---|---|---|---|
| kettle | **0,13** | 0,11 | 0,00 | 0,00 | 0,00 | 0,00 |
| freezer | 0,77 | 0,74 | **0,84** | **0,84** | **0,84** | **0,84** |
| washing machine | 0,17 | **0,30** | 0,17 | 0,21 | 0,22 | 0,25 |
| computer | **0,50** | 0,46 | 0,45 | 0,45 | 0,45 | 0,45 |
| microwave | 0,03 | 0,03 | **0,05** | 0,00 | 0,04 | 0,02 |
| television | 0,36 | 0,36 | 0,35 | 0,26 | 0,32 | **0,34** |

## Alvos a bater (melhor algoritmo de cada célula)

| aparelho | intra 5 min | intra 15 min | cross 5 min | cross 15 min |
|---|---|---|---|---|
| washing machine | 0,63 (SGN) | 0,35 (SGN) | 0,30 (CO) | 0,30 (CO) |
| freezer | 0,46 (Seq2Point) | 0,55 (vários) | 0,78 (Seq2Point) | 0,84 (vários) |
| television | 0,68 (SGN) | 0,53 (Seq2Point/BiLSTM) | 0,54 (Seq2Point) | 0,36 (FHMM/CO) |
| microwave | 0,30 (SGN) | 0,28 (Seq2Point) | 0,09 (SGN) | 0,05 (Seq2Point) |
| computer | 1,00 | 1,00 | 0,49 (FHMM) | 0,50 (FHMM) |
| kettle | 0,12 (FHMM) | 0,16 (CO) | 0,10 (FHMM) | 0,13 (FHMM) |

## Observações sobre a comparabilidade

**A F1 deles é subproduto de regressão.** Os seis algoritmos são
desagregadores: preveem potência, e a F1 sai de limiarizar essa potência
prevista (é assim que o nilmtk calcula). Um classificador treinado diretamente
para o rótulo otimiza o que é medido, e portanto **deveria** ganhar. A
comparação é justa como linha de base publicada, mas a vantagem metodológica
precisa ser declarada — não é o mesmo tipo de modelo resolvendo o mesmo
problema.

**A F1 do computador é 1,00 no intra-dataset.** Isso quase certamente indica um
aparelho ligado o tempo todo (prevalência ~100%), caso em que a F1 da classe
positiva é trivialmente 1. Não é evidência de nada. Serve de alerta: o artigo
não reporta prevalência, então várias dessas células são difíceis de
interpretar sem recalcular a taxa-base.

**O kettle dá F1 = 0,00 em todos os métodos profundos a 5 e 15 min.** Colapso
completo. A 5 min uma chaleira que roda 2 min desaparece na média.

**O limiar de ON não é reportado no artigo, mas está no código.** Em
`nilmtk/losses.py` do NeuralNILM_Pytorch (github.com/Ming-er/NeuralNILM_Pytorch):

    on_threhold = {'fridge':50, 'kettle':2000, 'dish washer':10,
                   'washing machine':20, 'drill':0}
    threshold = on_threhold.get(app_name, 10)

`freezer` não está na tabela, então usa o padrão de **10 W**. Máquina de lavar:
20 W. O mesmo limiar é aplicado à verdade e à predição, sobre a série **já
reamostrada**. O notebook deles confirma o resto da configuração: REFIT casa 8,
`2014-03-07`→`2015-03-06` para treino, `2015-03-07`→`2015-04-06` para teste,
`sample_rate: 900`, `pre_trained: False`, `n_epochs: 2`.

## Lacunas entre este protocolo e o que temos hoje

| | situação |
|---|---|
| REFIT casa 8, 8 s, 2013-11-01 a 2015-05-10 | **em disco**; cobre os dois intervalos do artigo (363 e 27 dias) |
| canais da casa 8 | Fridge, Freezer, Washer Dryer, Washing Machine, Toaster, Computer, Television Site, Microwave, Kettle — **tem os 6 do artigo**, e **não tem lava-louças** |
| ECO casa 2, 1 s | **em disco** (2,3 GB), 01/06/12 a 31/01/13, cobre o intervalo de teste |
| ECO casas 1 e 5 | **não estão em disco** — precisam ser baixadas |
| reamostragem a 15 min | **não implementada** (o pipeline atual só faz 5 min) |
| aparelhos | o pipeline atual só trata lava-louças, geladeira e máquina de lavar; **falta freezer, kettle, computer, microwave, television** |

Plugs da ECO casa 2: 01 Tablet, 02 Dishwasher, 03 Air exhaust, 04 Fridge,
05 Entertainment, 06 Freezer, 07 Kettle, 08 Lamp, 09 Laptops, 10 Stove,
11 TV, 12 Stereo. Ou seja, para a casa 2 dá para montar freezer, kettle,
computer (09 Laptops) e television (11 TV) — **mas não máquina de lavar**,
que a casa 2 não tem.

---

# Nossos números no mesmo protocolo (SUPERADO — limiar do freezer errado, 50 W em vez de 10 W; ver a seção final)

Classificação direta (`bruto`, 1 encoder, seq2seq de 1 dia), 3 sementes, treino
na casa 8 do REFIT com os últimos 15% de cada corrida como validação de origem.
Limiar e parada decididos só na validação de origem.
Código: [src/experimento_artigo.py](../src/experimento_artigo.py).

| célula | artigo (melhor) | nosso, rótulo `nativo` | nosso, rótulo `media` | piso (reg. log.) |
|---|---|---|---|---|
| freezer · intra · 5 min | 0,46 (Seq2Point) | **0,466** | 0,443 | 0,437 |
| freezer · intra · 15 min | **0,55** (vários) | 0,426 | 0,342 | 0,404 |
| freezer · cross · 5 min | **0,78** (Seq2Point) | 0,638 | 0,626 | 0,000 |
| freezer · cross · 15 min | **0,84** (vários) | 0,693 | 0,309 | 0,001 |
| máq. lavar · intra · 5 min | **0,63** (SGN) | 0,457 | 0,438 | 0,340 |
| máq. lavar · intra · 15 min | 0,35 (SGN) | **0,438** | 0,413 | 0,357 |
| máq. lavar · cross · 5 min | 0,30 (CO) | — | — | — |
| máq. lavar · cross · 15 min | 0,30 (CO) | — | — | — |

As células cruzadas da máquina de lavar dependem das casas 1 e 5 da ECO, que
ainda não estão em disco — a casa 2 não tem máquina de lavar.

Desvio entre sementes: 0,001 a 0,009 no freezer, 0,007 a 0,062 na máquina de
lavar (o 0,062 é a máquina de lavar a 15 min, onde o mês de teste tem poucos
ciclos).

## Leitura

**Ganhamos em uma célula, empatamos em uma, perdemos em quatro.** A vitória é a
máquina de lavar a 15 min (0,438 contra 0,35). O empate é o freezer intra a
5 min. As derrotas grandes são a máquina de lavar intra a 5 min (0,457 contra
0,63) e o freezer cross (0,64-0,69 contra 0,78-0,84).

Isso é um resultado desconfortável para a premissa do projeto: um classificador
treinado diretamente para o rótulo **deveria** ter vantagem sobre um
desagregador cuja F1 é subproduto de limiarizar potência prevista, e na maior
parte das células não tem.

**Mas há um confundidor grande e não resolvido: o limiar de ON.** O artigo não
reporta o dele. A evidência de que isso domina está na linha do freezer cross:
com o rótulo `media` (o mesmo procedimento do nilmtk) a nossa F1 desaba de
0,626 para 0,309 ao passar de 5 para 15 min, enquanto a deles **sobe** de 0,78
para 0,84. A causa é conhecida e medida: o freezer da ECO puxa 52 W contra o
nosso limiar de 50 W, e a média de 15 min o afunda abaixo do limiar, levando a
prevalência de 45% para 18%. Se o limiar deles fosse mais baixo, esse colapso
não aconteceria com eles — o que é consistente com o que se observa. Enquanto o
limiar deles não for conhecido, as células do freezer não são comparáveis.

Com o rótulo `nativo`, que não depende da taxa, o nosso freezer cross fica em
0,638 (5 min) e 0,693 (15 min): sobe com a janela maior, na mesma direção que a
deles.

**A regressão logística colapsa no cruzado (F1 ≈ 0,000).** Ela usa a potência do
instante, padronizada com as estatísticas do REFIT; a escala do agregado da ECO
é outra e o limiar calibrado na origem não transfere. A rede sobrevive à mesma
troca de domínio. É o contraste mais limpo entre os dois que temos até aqui, e
inverte o que víamos no UK-DALE, onde o piso ganhava.

## O que ainda falta para a comparação fechar

1. ECO casas 1 e 5 (bloqueia as duas células cruzadas da máquina de lavar).
2. O limiar de ON usado pelo artigo. Sem isso, as células do freezer carregam um
   confundidor do tamanho do efeito medido.
3. A prevalência de cada célula deles. A F1 do computador ser 1,00 sugere
   prevalência ~100% em pelo menos uma célula, o que torna a métrica vazia ali.


---

# Resultado parcial, com os limiares do artigo (sem ECO 1 e 5 — ver a seção final)

Freezer a 10 W, máquina de lavar a 20 W, rótulo `media` (o procedimento do
nilmtk). 3 sementes. `tudo_ligado` é o classificador trivial que responde
"ligado" sempre — com prevalência p sua F1 é 2p/(1+p).

| célula | prev. | artigo (melhor) | nossa rede | piso | **tudo_ligado** |
|---|---|---|---|---|---|
| freezer · intra · 5 min | 33,1% | 0,46 | **0,531** | 0,495 | 0,498 |
| freezer · intra · 15 min | 43,7% | 0,55 | 0,608 | 0,609 | **0,608** |
| freezer · cross · 5 min | 69,2% | 0,78 | 0,715 | 0,000 | **0,818** |
| freezer · cross · 15 min | 91,2% | 0,84 | 0,954 | 0,938 | **0,954** |
| máq. lavar · intra · 5 min | 4,1% | **0,63** | 0,428 | 0,340 | 0,078 |
| máq. lavar · intra · 15 min | 4,4% | 0,35 | **0,413** | 0,343 | 0,084 |

## O freezer não é uma tarefa de classificação a esta resolução

Das quatro células do freezer, **as quatro do artigo ficam abaixo do
classificador trivial**: 0,46 < 0,498; 0,55 < 0,608; 0,78 < 0,818; 0,84 < 0,954.
Nenhum dos seis algoritmos deles supera responder "ligado" para tudo.

E as nossas não são melhores por mérito: em duas células a rede dá **exatamente**
a F1 do trivial (0,608 e 0,954), o que significa que ela aprendeu a prever
sempre ligado — com 91% de positivos e BCE, é quase ótimo fazer isso.

A causa é física, não metodológica. Um freezer cicla continuamente; qualquer
janela de 5 ou 15 min contém parte de um ciclo, e a média fica acima de 10 W.
A 15 min na ECO, 91% das janelas são "ligado". A pergunta "o freezer está
ligado nesta janela?" quase não tem variância para explicar.

Com o rótulo `nativo` (limiar na taxa nativa, janela ligada se ≥50% das
amostras nativas estavam ligadas) a prevalência cai para 57% na ECO e 28% no
REFIT, mas a rede continua empatando ou perdendo do trivial no cruzado (0,655 e
0,715 contra 0,728). O problema não é a definição do rótulo, é o aparelho.

## A máquina de lavar é a tarefa real

Prevalência de 4%, trivial em 0,078-0,084, e tanto a nossa rede (0,413-0,428)
quanto a deles (0,35-0,63) ficam muito acima. Aqui a F1 mede modelo.

Resultado honesto: **perdemos a 5 min** (0,428 contra 0,63 do SGN) e **ganhamos
a 15 min** (0,413 contra 0,35). A derrota a 5 min é grande e real, e é o único
número desta tabela que vale como alvo a perseguir.

## Consequências para o que se pode escrever

1. Comparar-se com as células do freezer não significa nada. Se elas entrarem
   num texto, têm que vir com a linha do classificador trivial ao lado.
2. A métrica que o artigo coleta e não reporta — MCC, que está na lista de
   `metrics` do notebook deles — não teria esse problema. Nem a precisão e o
   recall separados, que também são calculados e não publicados.
3. O alvo real é a máquina de lavar a 5 min: 0,63.


---

# Resultado completo (ECO 1, 2 e 5 processadas)

ECO casa 5 não tem freezer nem máquina de lavar, então o cruzado da máquina de
lavar é a casa 1 apenas, e o do freezer são as casas 1 e 2. Rótulo `media`
(procedimento do nilmtk), limiares do artigo, 3 sementes.

| célula | prev. | artigo | nossa rede | oráculo | AUC | trivial |
|---|---|---|---|---|---|---|
| freezer · intra · 5 min | 33,1% | 0,46 | **0,531** | 0,537 | 0,651 | 0,498 |
| freezer · intra · 15 min | 43,7% | 0,55 | 0,608 | 0,608 | **0,497** | 0,608 |
| freezer · cross eco1 · 5 min | 61,2% | 0,78* | 0,664 | 0,767 | 0,654 | 0,760 |
| freezer · cross eco1 · 15 min | 72,1% | 0,84* | 0,838 | 0,838 | **0,516** | 0,838 |
| freezer · cross eco2 · 5 min | 69,2% | 0,78* | 0,715 | 0,821 | 0,701 | 0,818 |
| freezer · cross eco2 · 15 min | 91,2% | 0,84* | 0,954 | 0,953 | **0,527** | 0,954 |
| máq. lavar · intra · 5 min | 4,1% | **0,63** | 0,428 | 0,456 | 0,897 | 0,078 |
| máq. lavar · intra · 15 min | 4,4% | 0,35 | **0,413** | 0,444 | 0,920 | 0,084 |
| máq. lavar · cross eco1 · 5 min | 6,7% | **0,30** | 0,000 | 0,216 | 0,637 | 0,126 |
| máq. lavar · cross eco1 · 15 min | 7,5% | **0,30** | 0,009 | 0,227 | 0,623 | 0,140 |

\* o cruzado do artigo agrega ECO 1, 2 e 5; o nosso é por casa. Não é a mesma
célula, e a comparação direta é aproximada.

## 1. O freezer a 15 min tem AUC de 0,50

A coluna AUC fecha o caso que a taxa-base já sugeria. Nas três células de 15 min
o AUC é **0,497, 0,516 e 0,527** — indistinguível de sorteio. O modelo não
consegue separar freezer ligado de desligado, e a F1 de 0,838 ou 0,954 vem
inteiramente de 72% e 91% de positivos.

A F1 de oráculo confirma: ela é igual à F1 do limiar transferido *e* igual ao
trivial, porque o melhor limiar possível é o que marca tudo como ligado.

A 5 min ainda há sinal (AUC 0,65-0,70), e é a única faixa em que a tarefa do
freezer significa alguma coisa.

## 2. A máquina de lavar não transfere para a Suíça — e a falha é dupla

Intra: AUC 0,90-0,93, F1 0,41-0,43. Cross: **AUC cai para 0,62-0,64** e a F1 vai
a zero.

São duas falhas empilhadas:

- **ordenação**: o AUC de 0,63 diz que o modelo mal distingue as janelas. A F1
  de oráculo, que é o teto com limiar perfeito, é só 0,216-0,227.
- **calibração**: sobre isso, o limiar calibrado no REFIT não transfere, e a F1
  cai de 0,22 (oráculo) para 0,000.

A causa material está nos dados: a máquina de lavar da casa 8 do REFIT puxa
825 W médios quando ligada, a da ECO casa 1 puxa 306 W. É a mesma falha de
assinatura entre casas que já tínhamos medido no UK-DALE, agora entre países.

Comparando honestamente: nosso teto com oráculo (0,22) fica na faixa dos modelos
profundos do artigo (Seq2Point 0,22, BiLSTM 0,21, DAE 0,20, SGN 0,19), mas eles
chegam lá **sem** oráculo. E o melhor deles nessa célula é o CO, um método
clássico, com 0,30.

## 3. Placar honesto

Das dez células, as seis do freezer não medem modelo (quatro têm AUC ≈ 0,5 ou
ficam abaixo do trivial). Restam quatro células reais, todas de máquina de
lavar:

| | artigo | nós |
|---|---|---|
| intra 5 min | **0,63** | 0,428 |
| intra 15 min | 0,35 | **0,413** |
| cross 5 min | **0,30** | 0,000 |
| cross 15 min | **0,30** | 0,009 |

Ganhamos uma de quatro. As duas cruzadas são derrotas completas, não parciais.


---

# Normalização por casa: a correção que faltava

Diagnóstico da F1 = 0,000 (ver o corpo do relatório): a entrada é padronizada
com as estatísticas do REFIT (média 693 W, desvio 1003 W). Aplicada à ECO casa 1,
cujo agregado é ~2,4x menor, ela produz um sinal de média −0,43 e desvio 0,43 —
comprimido a 43% da variância de treino. A maior probabilidade que o modelo
atinge em 8 064 janelas da ECO é 0,271, contra um limiar de 0,2515. Três janelas
passam, nenhuma correta.

A correção é a vista `casa` (`x / p95 do próprio agregado`), que torna a entrada
adimensional antes da padronização. Mesmo protocolo, trocando só a vista.

| célula | vista | F1 | oráculo | AUC | trivial |
|---|---|---|---|---|---|
| máq. lavar · cross · 5 min | bruto | 0,000 | 0,216 | 0,637 | 0,126 |
| | **casa** | **0,168** | 0,280 | **0,794** | |
| máq. lavar · cross · 15 min | bruto | 0,009 | 0,227 | 0,623 | 0,140 |
| | **casa** | **0,232** | 0,295 | **0,766** | |
| máq. lavar · intra · 5 min | bruto | 0,428 | 0,456 | 0,897 | 0,078 |
| | casa | 0,429 | 0,457 | 0,897 | |
| máq. lavar · intra · 15 min | bruto | 0,413 | 0,444 | 0,920 | 0,084 |
| | casa | 0,427 | 0,442 | 0,915 | |
| freezer · cross eco1 · 5 min | bruto | 0,664 | 0,767 | 0,654 | 0,760 |
| | casa | 0,771 | 0,792 | 0,725 | |

**O ganho é de ordenação, não de calibração.** O AUC do cruzado sobe de 0,63 para
0,77-0,79. A F1 de oráculo — o teto com limiar perfeito — sobe de 0,22 para
0,28-0,30. Não é só o limiar que passou a transferir: o modelo passou a
distinguir as janelas.

**E é de graça no intra.** 0,428 -> 0,429 e 0,413 -> 0,427. Nenhum custo.

## O achado metodológico

Esta mesma vista **falhou** no leave-one-house-out com ~18 casas de treino
(`3v_novo` sempre abaixo de `3v_antigo` e de `bruto`). Aqui, com **uma** casa de
origem e outro país, ela é a diferença entre 0,000 e 0,232.

A leitura une os dois resultados: a normalização por casa e a variedade de casas
no treino resolvem o **mesmo** problema — ensinar a rede que a escala do
domicílio não é informação. Com 18 casas a rede aprende isso sozinha e impor a
normalização só gasta capacidade. Com uma casa não há como aprender, e a
normalização é indispensável.

Isso é testável e foi testado nos dois regimes, o que o torna uma afirmação
defensável e não uma racionalização pós-hoc.

## Placar atualizado contra o artigo

Só as quatro células de máquina de lavar, que são as que medem modelo:

| | artigo (melhor) | melhor do artigo entre os profundos | nós (`casa`) |
|---|---|---|---|
| intra 5 min | **0,63** (SGN) | 0,63 | 0,429 |
| intra 15 min | 0,35 (SGN) | 0,35 | **0,427** |
| cross 5 min | **0,30** (CO) | 0,22 (Seq2Point) | 0,168 |
| cross 15 min | **0,30** (CO) | 0,25 (BiLSTM) | 0,232 |

Continuamos perdendo em três de quatro. No cruzado a 15 min o nosso 0,232 fica
dentro da faixa dos modelos profundos deles (0,17-0,25), à frente de FHMM,
Seq2Point, SGN e DAE, atrás de BiLSTM e do CO. A derrota que continua grande e
sem explicação é o intra a 5 min: 0,429 contra 0,63 do SGN.

---

# Teste do tamanho de patch (hipótese refutada)

Hipótese: a cabeça produz `patch` rótulos a partir de um único token
([modelo.py:109](../src/modelo.py#L109)), então com `patch = 4` os 4 instantes
dentro do bloco não têm representação própria. Isso explicaria a derrota no
intra a 5 min (0,429 contra 0,63 do SGN), onde nem domínio nem calibração
explicam nada. Previsão: `patch = 1` deveria subir esse número.

Máquina de lavar, vista `casa`, rótulo `media`, 3 sementes.

| célula | patch | F1 | ± | oráculo | AUC |
|---|---|---|---|---|---|
| intra · 5 min | **4** | **0,429** | 0,040 | 0,457 | 0,897 |
| | 2 | 0,336 | 0,028 | 0,350 | 0,878 |
| | 1 | 0,312 | 0,005 | 0,325 | 0,861 |
| intra · 15 min | **4** | **0,427** | 0,019 | 0,442 | 0,915 |
| | 2 | 0,358 | 0,088 | 0,414 | 0,900 |
| | 1 | 0,330 | 0,015 | 0,352 | 0,867 |
| cross · 5 min | 4 | 0,168 | 0,011 | 0,280 | 0,794 |
| | 2 | 0,191 | 0,022 | 0,299 | 0,816 |
| | **1** | **0,239** | 0,037 | 0,294 | 0,807 |
| cross · 15 min | **4** | **0,232** | 0,033 | 0,295 | 0,766 |
| | 2 | 0,188 | 0,015 | 0,339 | 0,809 |
| | 1 | 0,193 | 0,049 | 0,329 | 0,812 |

**A hipótese está errada.** No intra, que era a célula que eu disse que ia
melhorar, `patch = 4` é o melhor nas duas taxas e a F1 cai monotonicamente ao
refinar o patch: 0,429 -> 0,336 -> 0,312 a 5 min. A AUC acompanha (0,897 ->
0,861). O gargalo do intra a 5 min **não** é granularidade da cabeça.

A explicação plausível é a oposta da que eu dei: o patch funciona como
suavização e regularização. Com `patch = 1` há 288 tokens de 1 amostra cada,
cada um com menos contexto próprio, e o modelo tem mais o que ajustar com a
mesma quantidade de rótulos.

**Efeito colateral real, em outra célula.** No cruzado a 5 min, `patch = 1` leva
a F1 de 0,168 para **0,239** — à frente de Seq2Point (0,22), SGN (0,19), DAE
(0,20), BiLSTM (0,21) e atrás só do CO (0,30). Patch grosso parece decorar a
assinatura temporal da casa de origem.

**Mas esse ganho não é selecionável de forma honesta.** O patch teria que ser
escolhido na validação de origem, que é REFIT, e lá `patch = 4` vence com folga.
Escolher `patch = 1` porque ele foi melhor na ECO é usar o alvo para decidir.
Fica registrado como observação de oráculo, não como resultado.

**Outra confirmação:** o fechamento morfológico rende mais quanto mais
fragmentada a saída — no intra a 15 min com `patch = 1` ele leva 0,330 para
0,402 (+0,072), contra +0,006 com `patch = 4`. Mesmo padrão já medido no
leave-one-house-out.

---

# Seq2point (a forma da tarefa era a peça que faltava)

Até aqui o projeto fazia **seq2seq**: 288 amostras entravam, 288 rótulos saíam,
e na inferência eu usava janelas sobrepostas com média das probabilidades. Eu
havia descrito isso como "equivalente ao seq2point a 1/72 do custo". **Não é.**
No seq2point o modelo é treinado para prever uma posição fixa e especializa a
representação para ela; mediar saídas de um seq2seq não reproduz isso.

Seq2Point e SGN — os dois métodos que venciam no artigo — são seq2point.

Implementado em [src/ponto.py](../src/ponto.py). Um exemplo, um rótulo, sempre
na mesma posição relativa. Dois formatos:

- `centrado`: janela `[X − 12 h, X + 12 h)`, alvo no meio (o Seq2Point original)
- `causal`: `[X − 23 h 55, X + 5 min]`, formato usável em tempo real

Máquina de lavar, vista `casa`, 3 sementes.

| célula | seq2seq | seq2point `centrado` | seq2point `causal` | artigo |
|---|---|---|---|---|
| intra · 5 min | 0,470 | **0,645** ±0,036 | 0,596 ±0,044 | 0,63 |
| intra · 15 min | 0,355 | 0,374 ±0,059 | **0,408** ±0,015 | 0,35 |
| cross · 5 min | **0,157** | 0,147 ±0,056 | 0,133 ±0,040 | 0,30 |
| cross · 15 min | **0,260** | 0,147 ±0,010 | 0,088 ±0,018 | 0,30 |

## O troco

**Dentro da casa, seq2point é muito melhor.** 0,470 -> 0,645 a 5 min, com AUC
de 0,911 -> 0,920 e F1 de oráculo praticamente igual à transferida (0,652
contra 0,645), ou seja o limiar também passou a transferir bem. **Supera o
artigo** nas duas células intra: 0,645 contra 0,63 a 5 min, e 0,408 (causal)
contra 0,35 a 15 min.

**Fora da casa, é pior.** A AUC no cruzado desaba de 0,77-0,79 (seq2seq) para
0,56-0,59. Especializar a representação para uma posição fixa faz o modelo
decorar a assinatura da casa de origem com muito mais força.

É um trade-off limpo e mensurado: **a mesma mudança que dá +0,175 no intra tira
-0,113 no cruzado a 15 min**. Não é ruído — a AUC cai junto, em todas as
sementes e nos dois formatos.

## Placar final, com seleção honesta

O formato tem que ser escolhido na validação de origem, que é REFIT. Ela
escolheria `centrado` a 5 min e `causal` a 15 min:

| célula | nós | artigo | |
|---|---|---|---|
| intra · 5 min | **0,645** | 0,63 | ganhamos |
| intra · 15 min | **0,408** | 0,35 | ganhamos |
| cross · 5 min | 0,147 | **0,30** | perdemos |
| cross · 15 min | 0,147 | **0,30** | perdemos |

De quatro células, duas vitórias e duas derrotas. As duas derrotas são as
cruzadas, e agora a causa está localizada: o seq2point sobreajusta o domínio.

## Ressalva de método

Três sementes, com desvios de até 0,059 (intra 15 min centrado) e 0,056
(cross 5 min centrado). Várias diferenças desta tabela não são resolvíveis com
essa amostra. Antes de qualquer afirmação publicável, rodar 10 sementes.

---

# Vision Transformer sobre transformadas imagem-de-série

Seq2point: a janela de contexto vira imagem (GASF, GADF, MTF, RP como canais) e
sai um rótulo. ViT de 4 camadas, d=192, patch 8, imagem 96×96 após PAA.
Implementado em [src/imagem.py](../src/imagem.py) — as transformadas rodam na
GPU dentro do laço de lote, nada é pré-computado.

Vista `casa`, formato centrado, 5 min, 3 sementes. A linha de base 1D sai da
mesma execução, com a mesma divisão e as mesmas sementes.

## Máquina de lavar

| | ViT (1 dia) | seq2point 1D | ViT (1+7 dias) |
|---|---|---|---|
| intra F1 | 0,498 ±0,027 | **0,639** ±0,035 | 0,323 ±0,065 |
| intra F1 oráculo | 0,516 | 0,654 | 0,368 |
| intra AUC | **0,939** | 0,920 | 0,871 |
| cross F1 | **0,209** ±0,015 | 0,144 ±0,051 | 0,073 ±0,038 |
| cross F1 oráculo | **0,269** | 0,228 | 0,161 |
| cross AUC | **0,745** | 0,590 | 0,549 |
| parâmetros | 1,86 M | 0,35 M | 1,91 M |

**A imagem troca desempenho no domínio por robustez entre domínios.** Dentro da
casa o 1D ganha por 0,14 de F1 — mas a AUC do ViT é *maior* (0,939 contra
0,920), ou seja ele ordena tão bem quanto e perde na forma da curva
precisão-recall, não na separabilidade. Fora da casa o ViT ganha, e ganha onde
importa: **AUC de 0,745 contra 0,590**, com um terço do desvio entre sementes.

Isso confirma o que a figura do GASF sugeria: GASF, GADF e MTF reescalam cada
janela pelo próprio min-max ou pelos próprios quantis, logo são invariantes à
escala da casa — e a escala da casa era a causa medida da falha REFIT → ECO
(agregado suíço 2,4× menor). O preço é descartar a amplitude absoluta, que é o
que distingue uma máquina de lavar de 2 kW dentro da casa de treino.

É o trade-off oposto ao do seq2point, que deu +0,175 no intra e −0,113 no
cruzado. Os dois efeitos existem e vão em direções contrárias.

## A janela de uma semana não ajudou

Com a mesma arquitetura, contexto de 1 dia dá 0,498 e contexto multiescala de
1+7 dias dá 0,323 no intra. **Não é a representação em imagem que atrapalha a
semana — é a semana.** Provável causa: 8 canais de 96×96 vindos de 2016
amostras, com ~25 mil exemplos de treino, é dimensão de entrada demais para a
quantidade de rótulos disponível.

## O freezer: o ViT colapsa no classificador trivial

| célula | prevalência | ViT F1 | 2p/(1+p) |
|---|---|---|---|
| intra | 0,330 | 0,495 | 0,496 |
| cross eco 1 | 0,612 | 0,759 | 0,759 |
| cross eco 2 | 0,692 | 0,817 | 0,818 |

A F1 do ViT é igual a 2p/(1+p) até a terceira casa, com desvio entre sementes
de 0,000–0,001, e a AUC fica entre 0,483 e 0,558. O modelo aprendeu a responder
"ligado" para tudo. É a confirmação mais limpa até agora de que **o freezer a
5 min não é uma tarefa de classificação** — nem o 1D, nem o ViT, nem nenhum dos
seis algoritmos do artigo superam responder sempre sim.

# Resultado final do seq2point, nas condições exatas do artigo

`src/experimento_ponto.py`, vista `casa`, formato centrado, 3 sementes, alvo
`cross: eco agregada` (as casas ECO empilhadas, que é como o artigo reporta um
número só para REFIT → ECO). Três colunas de limiar:

- `f1_meio` — limiar fixo em 0,5, sem calibração nenhuma
- `f1` — limiar escolhido **na validação da casa de origem** (o número honesto)
- `oráculo` — melhor limiar possível no alvo, só como teto

| aparelho | célula | f1_meio | f1 | ± | oráculo | AUC | prev | 2p/(1+p) | artigo |
|---|---|---|---|---|---|---|---|---|---|
| washing machine | intra 5 min | 0,636 | **0,645** | 0,036 | 0,652 | 0,920 | 0,045 | 0,086 | 0,63 (SGN) |
| washing machine | intra 15 min | 0,390 | **0,374** | 0,059 | 0,487 | 0,872 | 0,048 | 0,092 | 0,35 (SGN) |
| washing machine | cross 5 min | 0,140 | 0,147 | 0,056 | 0,234 | 0,590 | 0,074 | 0,137 | 0,30 (CO) |
| washing machine | cross 15 min | 0,156 | 0,147 | 0,010 | 0,233 | 0,594 | 0,082 | 0,151 | 0,30 (CO) |
| freezer | intra 5 min | 0,211 | 0,413 | 0,072 | 0,496 | 0,501 | 0,330 | **0,496** | 0,46 |
| freezer | intra 15 min | 0,459 | 0,518 | 0,014 | 0,612 | 0,506 | 0,440 | **0,611** | 0,55 |
| freezer | cross 5 min | 0,283 | 0,624 | 0,148 | 0,790 | 0,489 | 0,653 | **0,790** | 0,78 |
| freezer | cross 15 min | 0,611 | 0,708 | 0,045 | 0,896 | 0,498 | 0,813 | **0,897** | 0,84 |

## O freezer, encerrado

Nas quatro células do freezer, **a F1 no melhor limiar possível é igual a
2p/(1+p) até a terceira casa decimal** — 0,496/0,496, 0,612/0,611, 0,790/0,790,
0,896/0,897 — com AUC entre 0,489 e 0,507. O melhor limiar que existe é aquele
que responde "ligado" sempre. Isso não é uma propriedade do nosso modelo: é uma
propriedade da tarefa a esta resolução, e as quatro células do artigo (0,46 /
0,55 / 0,78 / 0,84) ficam **abaixo** desse mesmo teto trivial. O freezer não
deve ser reportado como resultado de classificação a 5 ou 15 min.

## A máquina de lavar, com o placar fechado

Intra batemos o artigo nas duas taxas (0,645 vs 0,63; 0,374 vs 0,35) e com folga
enorme sobre o trivial (0,086 e 0,092) — e a AUC de 0,92/0,87 confirma que há
sinal de verdade, não sorte de limiar. A calibração não está inflando nada:
`f1_meio` e `f1` diferem por 0,009–0,016.

Cross continua perdendo: 0,147 contra os 0,30 do Combinatorial Optimisation. Com
AUC de 0,59 o ranqueamento entre casas quase não sobrevive, e o teto do oráculo
(0,234) mostra que **nem limiar perfeito fecharia a conta** — o problema é a
representação, não a calibração. Vale notar que o CO não é um modelo treinado:
ele não tem o que transferir errado, o que explica boa parte da vantagem dele
justamente na célula cross.

# Fusão dos dois ramos (hipótese refutada)

`src/experimento_duplo.py`, máquina de lavar, 5 min, centrado, 3 sementes.
Quatro modelos da mesma divisão e da mesma inicialização:

| modelo | intra F1 | ± | intra AUC | cross F1 | ± | cross AUC |
|---|---|---|---|---|---|---|
| `1d` (seq2point na série) | **0,639** | 0,035 | 0,920 | 0,144 | 0,051 | 0,590 |
| `vit` (ViT nas imagens) | 0,459 | 0,011 | 0,920 | **0,226** | 0,059 | 0,738 |
| `duplo` (ramos concatenados, treinados juntos) | 0,616 | 0,032 | 0,934 | 0,125 | 0,039 | 0,619 |
| `conjunto` (média das probabilidades, sem treino) | **0,639** | 0,034 | **0,951** | 0,133 | 0,056 | 0,729 |

A fusão treinada perde para o `1d` no intra e para o `vit` no cross — ou seja,
piora cada ramo justamente onde ele era forte. E o controle que importa, a média
simples das probabilidades dos dois ramos treinados separadamente, empata com o
melhor ramo no intra (0,639) e leva a melhor AUC da tabela (0,951) sem treinar
nada. **Concatenar as representações não acrescentou informação**; só gastou
mais parâmetros e mais tempo para reproduzir, pior, o que mediar já dava.

Observação que sobrevive ao resultado negativo: os dois ramos são
complementares na AUC — 1D vence dentro da casa, ViT vence entre casas
(0,738 vs 0,590), e a média bate os dois (0,729 com F1 calibrado na origem, mas
0,274 de oráculo contra 0,228 do 1D). A complementaridade existe; o que não
funciona é capturá-la com uma cabeça conjunta treinada de ponta a ponta.

# Saída de K rótulos (hipótese refutada)

O SGN prevê 32 rótulos para 200 amostras de entrada; o nosso seq2point prevê 1
para 288. A hipótese era que a supervisão mais densa ajudasse — e que, deixando
subamostrar mais os alvos, saísse mais barata. `src/experimento_ponto.py
--saidas 1,8,16,32`, máquina de lavar, centrado, 3 sementes. K rótulos centrados
no alvo entram na perda; a **previsão continua sendo de um rótulo só** (`SoCentro`
corta a coluna central) e o limiar é recalibrado numa validação de um rótulo só,
para que só a supervisão mude.

| taxa | K | % da janela | F1 | ± | AUC | tempo | épocas | pareado vs K=1 |
|---|---|---|---|---|---|---|---|---|
| 5 min | 1 | 0,3% | **0,645** | 0,036 | 0,920 | 199 s | 118 | — |
| 5 min | 8 | 2,8% | 0,603 | 0,046 | 0,913 | 324 s | 190 | −0,042 (dp 0,082) |
| 5 min | 16 | 5,6% | 0,621 | 0,054 | 0,916 | 308 s | 179 | −0,024 (dp 0,030) |
| 5 min | 32 | 11,1% | 0,658 | 0,038 | 0,922 | 280 s | 161 | +0,013 (dp 0,020) |
| 15 min | 1 | 1,0% | **0,374** | 0,059 | 0,872 | 62 s | 131 | — |
| 15 min | 8 | 8,3% | 0,420 | 0,073 | 0,794 | 82 s | 171 | +0,046 (dp 0,090) |
| 15 min | 16 | 16,7% | 0,258 | 0,025 | 0,830 | 100 s | 207 | −0,116 (dp 0,054) |
| 15 min | 32 | 33,3% | 0,249 | 0,010 | 0,765 | 107 s | 220 | −0,124 (dp 0,050) |

**Não adotar.** A 5 min nenhum K se separa da linha de base (as diferenças
pareadas têm desvio maior ou comparável à média, e o efeito não é monótono). A
15 min, K ≥ 16 piora de forma consistente: as seis diferenças pareadas são todas
negativas e o desvio entre sementes do próprio braço é o menor da tabela
(0,010–0,025) — uniformemente ruim, não ruidosamente ruim.

E a premissa de custo caiu: em **todas** as oito células, K > 1 precisa de mais
épocas e mais tempo de parede. A supervisão mais densa não acelerou a
convergência; atrasou.

Uma leitura por "fração da janela ocupada pela saída" foi levantada e não se
sustenta: ela previa que K=32 a 15 min (33%) fosse pior que K=16 (17%), e os dois
empatam em −0,12; e não explica K=32 a 5 min (11%) indo bem enquanto K=8 a
15 min (8,3%) também vai. A explicação mais simples é o tamanho da amostra: a
15 min há ~5,8 mil alvos de treino contra 17 mil, e a tarefa já é mais difícil
(F1 0,374 contra 0,645) — acrescentar saídas e parâmetros machuca justamente
onde os dados são escassos.

Conclusão para o texto: a diferença de tamanho de saída entre o nosso seq2point
e o SGN **não explica** nenhuma diferença de desempenho. Já se sabia que ela não
era desvantagem de qualidade (o seq2point com 1 rótulo iguala ou supera o SGN no
intra); agora se sabe que aumentá-la também não traz vantagem.

# Não-estacionariedade e generalização de domínio

## O diagnóstico que motivou a linha

O agregado da REFIT casa 8 não é estacionário: a média varia 6,8× ao longo do
dia (315 W a 2 148 W) e o desvio padrão 8,25×; entre meses a média varia 1,68×.
Mas o ADF dá −20,78 (p ≈ 0), ou seja **não há raiz unitária** — a não-estaciona-
riedade é sazonal determinística, não tendência estocástica. Consequência
prática: diferenciar é o remédio errado (remove um problema inexistente e destrói
o nível, que é o que identifica o aparelho). A caracterização correta é
ciclo-estacionária (24 h, 7 dias, ano) e estacionária por trechos.

Isso também explica o freezer de forma rigorosa: é o componente mais próximo de
periódico da casa, e agregar em janelas de 5–15 min, comparáveis ao ciclo do
compressor, converge para o nível médio e destrói a fase. Daí AUC 0,50.

E revelou um viés forte na fonte:

| prevalência da máquina de lavar | 00–08 h | 09–23 h | pico |
|---|---|---|---|
| REFIT casa 8 (treino) | 0,089 | **0,002** | 06 h |
| ECO casa 1 (teste) | 0,092 | 0,053 | 05 h |

Na casa de treino, 97,8% do tempo ligado cai entre 00 h e 08 h e das 10 h às 23 h
a prevalência é **exatamente zero**. Na casa de teste a atividade está espalhada.
Hipótese: com janela de 24 h o modelo se localiza no dia pela curva de carga e
aprende "só de madrugada" — regra perfeita na fonte, errada no alvo. É uma
não-estacionariedade de **fase**, que a vista `casa` (nível) não corrige.

## A ablação de contexto refutou a hipótese

`--horas_ctx 24,8,4,2`, máquina de lavar, 5 min, centrado, 3 sementes.

| contexto | amostras | intra F1 | ± | cross F1 | ± | cross oráculo | cross AUC |
|---|---|---|---|---|---|---|---|
| 24 h | 288 | 0,645 | 0,036 | 0,147 | 0,056 | 0,234 | 0,590 |
| 8 h | 96 | 0,623 | 0,034 | 0,153 | 0,032 | 0,224 | 0,514 |
| **4 h** | **48** | **0,666** | 0,014 | **0,213** | 0,043 | **0,254** | 0,576 |
| 2 h | 24 | 0,637 | 0,026 | 0,173 | 0,015 | 0,242 | 0,543 |

Se a pista de hora do dia fosse o mecanismo, 8 h — que a apaga tanto quanto 4 h —
teria melhorado a transferência. Foi o **pior** dos quatro em AUC de cross
(0,514, nível de moeda). O viés diurno é real nos dados, mas o modelo não parece
estar explorando-o. **Hipótese refutada.**

O que sobra é um ótimo em U com pico em 4 h, melhor que as 24 h nas duas pontas —
porém **sem significância**: o melhor p pareado é 0,22. O que se pode afirmar é o
lado negativo, e ele é útil: **as 288 amostras não compram nada**. Um contexto
6× menor iguala ou melhora as duas métricas. Isso é o terceiro resultado
independente apontando capacidade excessiva para a quantidade de rótulos, junto
com a janela de uma semana no ViT (pior que a de um dia) e o sweep de K a 15 min.

## Ressalva de método

O indício preliminar que motivou otimismo — um modelo de 3 épocas a 4 h com cross
AUC 0,69 — era artefato de treino incompleto, não ausência de viés. É o
fenômeno já documentado em `src/treino.py`: a F1 na origem sobe de forma monótona
enquanto a do alvo tem pico cedo e cai. Não se pode escolher a época olhando o
alvo, mas o fato de existir, no meio do treino, um modelo que transfere bem
melhor que o final indica que a perda de transferência vem da **especialização
na fonte**, e não de falta de representação.

## Fourier e group-DRO (24 h de contexto)

`src/dominio.py`: `randomizar_fourier` perturba amplitude e fase das raias de
período ≥ 3 h (nível da casa + envoltória diurna), preservando a banda de 1–2 h
onde vive o degrau; `PesosDRO` faz group-DRO sobre estação × fim de semana.
3 sementes, máquina de lavar, 5 min.

| braço | intra F1 | ± | intra AUC | cross F1 | ± | cross oráculo | cross AUC | ΔAUC cross |
|---|---|---|---|---|---|---|---|---|
| base | **0,645** | 0,036 | 0,920 | 0,147 | 0,056 | 0,234 | 0,590 | — |
| Fourier | 0,543 | 0,072 | 0,938 | 0,126 | 0,041 | **0,249** | **0,665** | **+0,075 (p 0,07)** |
| DRO | 0,543 | 0,066 | 0,914 | 0,168 | 0,032 | 0,245 | 0,607 | +0,017 (p 0,65) |
| ambos | 0,567 | 0,083 | 0,940 | **0,171** | 0,020 | 0,239 | 0,648 | +0,059 (p 0,26) |

Os três cobram o mesmo preço no intra (≈ −0,10 de F1) e nenhum devolve ganho de
F1 no cross que passe do ruído (melhor p = 0,62). O único efeito que quase se
separa é a **AUC de transferência do Fourier**: +0,075, três sementes positivas,
p 0,07.

O group-DRO é dominado pelo Fourier: mesmo custo no intra, um quarto do ganho de
AUC. Faz sentido em retrospecto — os subdomínios de calendário (estação × fim de
semana) capturam variação *dentro* da casa 8, e o que separa a Suíça do Reino
Unido não é estação nem dia da semana, mas nível, hábito e composição de cargas.
O DRO estava robustecendo contra o eixo errado.

## O gargalo mudou de lugar: representação → calibração

O achado central desta rodada é a **não-conversão**: +0,075 de AUC virou −0,021
de F1. O oráculo de cross do Fourier sobe (0,234 → 0,249), ou seja *existe* um
limiar melhor depois do aumento, mas o limiar escolhido na validação da fonte não
o encontra — o aumento desloca exatamente essa calibração.

Enquanto o limiar sair da fonte, todo ganho de invariância fica invisível na
métrica. Daí a coluna `f1_quantil` (casamento de taxa): escolhe no alvo o corte
que produz a mesma fração de positivos que o modelo emite na origem. Usa só as
probabilidades do alvo, nunca os rótulos — honesto, e aplicável numa casa nova
sem medição por aparelho.

# Comparação final em F1, protocolo completo (5 min, ECO 1+2+5)

Com a casa 5 da ECO processada e os seis aparelhos do artigo mapeados
(`scripts/preparar_artigo.py`). Cobertura por aparelho:

| aparelho | REFIT 8 | ECO 1 | ECO 2 | ECO 5 |
|---|---|---|---|---|
| freezer | ✓ | ✓ | ✓ | — |
| washing machine | ✓ | ✓ | — | — |
| kettle | ✓ | ✓ | ✓ | ✓ |
| computer | ✓ | PC | Laptops | PC |
| microwave | ✓ | — | — | ✓ |
| television | ✓ | — | TV | — |

## Metade dos aparelhos não é tarefa de classificação a 5 min

Verificável **nos rótulos, sem treinar nada**:

| aparelho | % ligado (REFIT treino) | diagnóstico |
|---|---|---|
| kettle | **0,00%** | limiar de 2 000 W sobre a média de 5 min nunca é atingido: uma chaleira roda 2–3 min a ~2 160 W, média de 5 min ≈ 1 000 W. **Zero positivos.** |
| computer | **100%** | ligado o tempo todo, treino e teste. **Zero negativos.** |
| freezer | 34,8% | prevalência sadia, mas oráculo = trivial e AUC ≈ 0,50 |

O F1 = 0,00 do kettle e o F1 = 1,00 do computador reportados no artigo **não são
resultados de modelo**: são consequências diretas da definição do rótulo. Três
dos seis aparelhos não medem capacidade de classificação.

## F1 dos três aparelhos com tarefa real

Duas configurações: `base` (contexto 24 h, sem aumento) e `2h+Fourier`
(contexto 2 h com randomização de Fourier). 3 sementes.

| aparelho | célula | base | 2h+Fourier | nosso melhor | artigo | trivial |
|---|---|---|---|---|---|---|
| washing machine | intra | **0,645** | 0,512 | **0,645** | 0,63 (SGN) | 0,086 |
| washing machine | cross | 0,147 | **0,261** | **0,261** | 0,30 (CO) | 0,138 |
| freezer | intra | 0,413 | **0,510** | **0,510** | 0,46 | 0,496 |
| freezer | cross | 0,624 | **0,787** | **0,787** | 0,78 | 0,790 |
| television | intra | 0,666 | **0,695** | **0,695** | 0,68 (SGN) | 0,394 |
| television | cross | 0,105 | **0,211** | **0,211** | 0,54 (Seq2Point) | 0,355 |

**4 vitórias, 2 derrotas — mas de qualidades diferentes.**

*Vitórias com detecção real:* máquina de lavar intra (0,645 contra trivial de
0,086, AUC 0,920) e televisão intra (0,695 contra trivial de 0,394, AUC 0,867).

*Vitórias que são só calibração:* as duas do freezer. Nosso F1 (0,510 / 0,787)
iguala nosso próprio oráculo (0,511 / 0,792) e o trivial (0,496 / 0,790), com
AUC 0,55. **Não detectamos o freezer — acertamos o limiar de um problema
degenerado**, e ganhamos porque os métodos do artigo ficam abaixo do trivial.
Descrever isso como detecção seria falso.

*Derrotas:* máquina de lavar cross (0,261 vs 0,30 do CO, mas acima de todos os
métodos treinados do artigo, cujo melhor é 0,22) e televisão cross (0,211 vs
0,54). A segunda é franca: nosso oráculo ali é 0,355 = trivial e a AUC é **0,36**,
abaixo do acaso. Hipótese a investigar: o canal "Television Site" da REFIT
agrega vários equipamentos, enquanto o plug "TV" da ECO é um aparelho só.

## Observações de método

A configuração `2h+Fourier` melhora 5 das 6 células e reduz muito a variância
(o desvio do freezer cross cai de 0,148 para 0,001). A única piora é a máquina
de lavar intra. **Não existe configuração única que vença todas as células** —
o aumento troca ajuste na fonte por generalização, e isso precisa ser reportado
assim, e não escolhendo a melhor configuração por célula como se fosse um
modelo só.

O estimador EM de prevalência (`f1_em`) funcionou na máquina de lavar (0,255
contra 0,261 do limiar normal) e **falhou no freezer** (0,23–0,58 contra 0,79).
Ele pressupõe que só o prior muda e que o modelo ranqueia; com AUC ≈ 0,50 a
premissa quebra e a estimativa vai para o lado errado.
