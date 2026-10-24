# Registro de experimentos — classificação de estado de cargas a baixa taxa

Documento de trabalho. Registra **tudo** que foi testado, incluindo (e
especialmente) o que não funcionou, com os números que sustentam cada conclusão.

Escopo fixado pelo projeto:

- **Rotulação, não desagregação.** O modelo decide ligado/desligado; nunca prevê watts como entregável.
- **Seq2point.** Uma janela de contexto, **um** rótulo. Nunca 288 rótulos de uma vez.
- **Um modelo por aparelho.** Nada de multirrótulo.
- **5 min** (e 15 min numa fase anterior).
- Métrica de interesse: **F1 da classe positiva**.

---

## 1. Estado atual em uma tabela

Protocolo de Chatterjee & Heer (2025): treino REFIT casa 8 (07/03/2014–06/03/2015),
teste intra no mês seguinte da mesma casa, teste cross em ECO (02/06–01/07/2012).
Tudo a 5 min, 3 sementes. **Números pós-correção do bug de validação (seção 9).**

| aparelho | célula | nosso melhor | quem | artigo (melhor) | trivial 2p/(1+p) |
|---|---|---|---|---|---|
| máquina de lavar | intra | 0,607 | seq2point 24 h, 1 casa | **0,63** (SGN) | 0,086 |
| máquina de lavar | cross | 0,234 | seq2point, **9 casas** | **0,30** (CO) | 0,138 |
| freezer | intra | **0,521** | seq2point, **3 casas** | 0,46 | 0,496 |
| freezer | cross (agregada) | 0,744 | seq2point, **3 casas** | **0,78** | 0,790 |
| televisão | intra | **0,700** | conjunto (nosso+TimesFM) | 0,68 (SGN) | 0,394 |
| televisão | cross | **0,601** | seq2point, **5 casas** | 0,54 (Seq2Point) | 0,355 |

**3 vitórias, 3 derrotas.** Com qualificação:

- **Televisão cross (0,601 vs 0,54) é a única vitória com margem confortável e
  detecção real** (AUC 0,872). Veio de treinar em 5 casas fonte — ver seção 11.
- Televisão intra (0,700 vs 0,68) é uma vitória apertada, mas com sinal real
  (AUC 0,873).
- Freezer intra (0,521 vs 0,46) fica acima do artigo **e** do trivial (0,496),
  mas por pouco; a tarefa segue praticamente inútil (seção 3).

---

## 2. O protocolo e suas armadilhas

Verificado contra o código do artigo (`nilmtk/losses.py` do NeuralNILM_Pytorch e
o notebook deles), não só contra o texto:

| item | valor |
|---|---|
| casas | REFIT 8 (treino/intra), ECO 1, 2, 5 (cross) |
| limiar ON | freezer 10 W, máquina de lavar 20 W, kettle 2000 W, resto 10 W |
| aplicação do limiar | sobre a série **já reamostrada** |
| épocas deles | **2** (`n_epochs: 2` no notebook) |
| F1 deles | subproduto de regressão — limiarizam a **potência prevista** |

Duas consequências para a comparação, que precisam ser declaradas:

1. **Vantagem metodológica nossa.** Um classificador treinado direto para o
   rótulo otimiza o que é medido; um desagregador não. Não é o mesmo tipo de
   modelo resolvendo o mesmo problema.
2. **Desvantagem deles por sub-treino.** 2 épocas contra as nossas 20–250.
   Vencer um modelo treinado por 2 épocas não é forte; **perder** para ele é
   informativo.

### Metade dos aparelhos não é tarefa de classificação a 5 min

Verificável nos rótulos, sem treinar nada:

| aparelho | % ligado (REFIT treino) | diagnóstico |
|---|---|---|
| **kettle** | **0,00%** | limiar de 2 000 W sobre a média de 5 min nunca é atingido — uma chaleira roda 2–3 min a ~2 160 W, média ≈ 1 000 W. **Zero positivos.** |
| **computer** | **100%** | ligado o tempo todo, treino e teste. **Zero negativos.** |
| freezer | 34,8% | prevalência sadia, mas oráculo = trivial (seção 3) |

O F1 = 0,00 do kettle e o F1 = 1,00 do computador reportados no artigo **não são
resultados de modelo**: são consequências da definição do rótulo. Três dos seis
aparelhos não medem capacidade de classificação.

---

## 3. O freezer: sinal abaixo do ruído (com uma ressalva tardia)

Medido em REFIT casa 8, 5 min:

| aparelho | amplitude típica | salto médio do agregado ao ligar | em desvios do agregado |
|---|---|---|---|
| freezer | 75 W | **−5 W** | **−0,01** |
| máquina de lavar | 162 W | +1 413 W | +1,42 |

Quando o freezer liga, o agregado muda −5 W — ou seja, nada, e com o sinal
errado. A assinatura está uma ordem de grandeza abaixo da flutuação normal da
casa (desvio 994 W).

E o oráculo iguala o trivial nas quatro células:

| célula | F1 (nosso) | oráculo | prevalência | 2p/(1+p) | artigo |
|---|---|---|---|---|---|
| intra 5 min | 0,413 | 0,496 | 0,330 | **0,496** | 0,46 |
| intra 15 min | 0,518 | 0,612 | 0,440 | **0,611** | 0,55 |
| cross 5 min | 0,622 | 0,775 | 0,633 | **0,775** | 0,78 |
| cross 15 min | 0,698 | 0,867 | 0,767 | **0,868** | 0,84 |

O melhor limiar que existe é "responder ligado sempre", e **as quatro células do
artigo ficam abaixo desse teto trivial**.

> **Ressalva importante — esta conclusão foi parcialmente revertida (seção 11).**
> A afirmação "não há o que aprender" era forte demais. Com 3 casas fonte em vez
> de 1, a AUC do freezer vai de **0,500 para 0,652** (intra) e de 0,462 para
> 0,671 (cross). O argumento de razão sinal-ruído continua válido, e o F1 segue
> sem superar o trivial com folga — mas a informação **existe**; ela apenas não
> é extraível de uma casa só.

---

## 4. O que foi testado e não funcionou

Sete hipóteses refutadas. Cada uma fecha um caminho.

### 4.1 Arquitetura TF-C de três ramos (UK-DALE casa 1 → casa 2)

Um encoder Transformer por vista, fundidos. 12 braços.

| braço | F1 | AUC | parâmetros |
|---|---|---|---|
| `arcsinh` (vista única) | **0,530** | 0,949 | 365 604 |
| `3v_soma` (3 ramos) | 0,509 | 0,958 | 1 077 028 |
| `3v_concat` (3 ramos) | 0,485 | 0,963 | 1 095 844 |
| `bruto` (vista única) | 0,445 | 0,936 | 365 604 |
| `3x_bruto` (controle) | 0,336 | 0,929 | 1 105 063 |
| `piso` (reg. logística) | 0,250 | 0,964 | 5 |

**Três ramos custam 3× os parâmetros e perdem para uma vista única bem
escolhida.** O controle `3x_bruto` (mesma vista três vezes) confirma que o ganho
dos multi-ramos não vem da multiplicidade.

### 4.2 Estrutura do SGN (porta + duas sub-redes)

Máquina de lavar, protocolo do artigo. Quatro variantes:

| variante | intra | cross |
|---|---|---|
| `cabeca` | 0,416 | 0,205 |
| `completo` (porta) | — | 0,204 |
| `duplo_soma` | — | 0,187 |
| `linear` | — | 0,200 |

Todas dentro do ruído; `completo` é o pior ou empatado. **A estrutura com porta
do SGN não é o ingrediente que faltava.**

### 4.3 Tamanho de patch

| patch | intra | cross |
|---|---|---|
| 4 | **0,323** | 0,156 |
| 2 | 0,285 | 0,155 |
| 1 | 0,277 | 0,165 |

F1 **cai** monotonicamente com patch mais fino. A hipótese de que a cabeça de
patch 4 era o gargalo está refutada.

### 4.4 Saída de K rótulos (o meio-termo do SGN)

O SGN prevê 32 rótulos para 200 amostras; nós 1 para 288. Hipótese: supervisão
mais densa ajudaria e sairia mais barata. K rótulos centrados no alvo entram na
perda; a **previsão continua sendo de um só** (`SoCentro`).

| taxa | K | % da janela | F1 | ± | pareado vs K=1 | tempo |
|---|---|---|---|---|---|---|
| 5 min | 1 | 0,3% | **0,645** | 0,036 | — | 199 s |
| 5 min | 8 | 2,8% | 0,603 | 0,046 | −0,042 (dp 0,082) | 324 s |
| 5 min | 16 | 5,6% | 0,621 | 0,054 | −0,024 (dp 0,030) | 308 s |
| 5 min | 32 | 11,1% | 0,658 | 0,038 | +0,013 (dp 0,020) | 280 s |
| 15 min | 1 | 1,0% | **0,374** | 0,059 | — | 62 s |
| 15 min | 8 | 8,3% | 0,420 | 0,073 | +0,046 (dp 0,090) | 82 s |
| 15 min | 16 | 16,7% | 0,258 | 0,025 | −0,116 (dp 0,054) | 100 s |
| 15 min | 32 | 33,3% | 0,249 | 0,010 | −0,124 (dp 0,050) | 107 s |

A 5 min nada se separa. A 15 min, K ≥ 16 piora de forma consistente (seis
diferenças pareadas negativas, desvio do braço 0,010–0,025). **E a premissa de
custo caiu: em todas as oito células K > 1 precisa de mais épocas e mais tempo.**

### 4.5 Fusão treinada dos dois ramos (1D + ViT)

| modelo | intra F1 | intra AUC | cross F1 | cross AUC |
|---|---|---|---|---|
| `1d` | **0,639** | 0,920 | 0,144 | 0,590 |
| `vit` | 0,459 | 0,920 | **0,226** | 0,738 |
| `duplo` (concatenado, treinado junto) | 0,616 | 0,934 | 0,125 | 0,619 |
| `conjunto` (média, sem treino) | **0,639** | **0,951** | 0,133 | 0,729 |

A fusão treinada **piora cada ramo onde ele era forte**. A média simples — que
não treina nada — empata com o melhor ramo e leva a melhor AUC.

### 4.6 Group-DRO sobre subdomínios de calendário

| braço | intra F1 | cross F1 | cross AUC |
|---|---|---|---|
| base | **0,645** | 0,147 | 0,590 |
| Fourier | 0,543 | 0,126 | **0,665** |
| **DRO** | 0,543 | 0,168 | 0,607 |
| ambos | 0,567 | **0,171** | 0,648 |

O DRO paga o mesmo preço do Fourier no intra (−0,102) e compra um quarto do
ganho de AUC. Em retrospecto: estação × fim de semana captura variação **dentro**
da casa 8, e o que separa Suíça de Reino Unido é nível, hábito e composição de
cargas. **Robustecia contra o eixo errado.**

### 4.7 Casamento de taxa de positivos (calibração)

| alvo | F1 (limiar da origem) | F1 (casamento de taxa) | taxa emitida | prevalência real |
|---|---|---|---|---|
| intra | 0,645 | 0,641 | 3,44% | 4,51% |
| cross | **0,147** | **0,104** | 3,44% | **7,37%** |

Piorou. A taxa de positivos **não é propriedade transferível do modelo**: mistura
qualidade do classificador com prevalência do domínio, e a prevalência dobra
entre REFIT e ECO. O estimador EM de prior shift (Saerens et al.) que veio
depois funciona quando a AUC é alta (máquina de lavar) e falha quando é 0,50
(freezer: F1 cai para 0,020).

---

## 5. Hipóteses que se sustentaram parcialmente

### 5.1 Vista `casa` (normalização por residência)

Dividir por p95 do agregado da própria casa. Com **uma** casa fonte, corrigiu o
colapso total da máquina de lavar: cross F1 **0,000 → 0,168/0,232**, AUC
0,637 → 0,794. Com **18** casas fonte, tornou-se redundante — resultado coerente
de dois regimes, e prenúncio da seção 11.

### 5.2 Comprimento de contexto

| contexto | amostras | intra F1 | cross F1 | cross AUC |
|---|---|---|---|---|
| 24 h | 288 | 0,645 | 0,147 | 0,590 |
| 8 h | 96 | 0,623 | 0,153 | 0,514 |
| **4 h** | **48** | **0,666** | **0,213** | 0,576 |
| 2 h | 24 | 0,637 | 0,173 | 0,543 |

Ótimo em U com pico em 4 h, melhor nas duas pontas — **mas sem significância**
(melhor p pareado = 0,22). O que se afirma é o lado negativo: **as 288 amostras
não compram nada**; um contexto 6× menor iguala ou melhora. Terceiro resultado
independente apontando capacidade excessiva, junto com a janela de uma semana no
ViT e o sweep de K a 15 min.

### 5.3 Perdas alternativas

Máquina de lavar, LOHO em 8 casas. Média de F1 por perda: `bce` e `bce_soft_f1`
lideram em 6 das 8 casas; `bce_peso` é consistentemente a pior (0,179–0,552
contra 0,258–0,759 da BCE). **Reponderar por prevalência não ajuda.**

### 5.4 Pós-processamento com prior de duração

Máquina de lavar, 8 casas: `cru` 0,387 médio, `morf` 0,394, `viterbi` 0,392.
Ganhos de 0,00–0,02, dentro do ruído. Vale mais pelo oráculo (`morf_oraculo`
sobe ~0,03), o que indica de novo problema de limiar, não de sequência.

---

## 6. Não-estacionariedade: diagnóstico e hipótese refutada

Medido em REFIT casa 8:

- Média do agregado varia **6,8×** ao longo do dia (315 W → 2 148 W), desvio **8,25×**
- Entre meses, média varia 1,68×
- **ADF = −20,78 (p ≈ 0)** — sem raiz unitária

Ou seja: a não-estacionariedade é **sazonal determinística**, não tendência
estocástica. **Diferenciar é o remédio errado** — remove um problema inexistente
e destrói o nível, que é o que identifica o aparelho. A caracterização correta é
ciclo-estacionária (24 h, 7 dias, ano) e estacionária por trechos.

Isso também dá a explicação rigorosa do freezer: é o componente mais próximo de
periódico da casa, e agregar em janelas comparáveis ao ciclo do compressor
converge para o nível médio e destrói a fase.

### A hipótese de fase (refutada)

| prevalência da máquina de lavar | 00–08 h | 09–23 h |
|---|---|---|
| REFIT casa 8 (treino) | 0,089 | **0,002** |
| ECO casa 1 (teste) | 0,092 | 0,053 |

Na fonte, 97,8% do tempo ligado cai entre 00 h e 08 h e das 10 h às 23 h a
prevalência é **exatamente zero**. Hipótese: com janela de 24 h o modelo se
localiza no dia e aprende "só de madrugada".

**Refutada pela ablação de contexto:** 8 h apaga a pista tanto quanto 4 h e foi o
**pior** dos quatro em AUC de transferência (0,514). O viés diurno é real nos
dados, mas o modelo não parece explorá-lo.

---

## 7. Randomização de Fourier

Perturba amplitude e fase das raias de período ≥ 3 h (nível da casa +
envoltória diurna), preservando a banda de 1–2 h do ciclo de lavagem.

### Seletividade medida (60 janelas com ativação)

| corte | período | energia do agregado atingida | energia do aparelho atingida | razão |
|---|---|---|---|---|
| raia 4 | 6,0 h | 25,9% | 15,6% | **1,66** |
| raia 8 | 3,0 h | 35,5% | 29,2% | 1,22 |
| raia 16 | 1,5 h | 54,8% | 50,6% | 1,08 |
| raia 24 | 1,0 h | 65,5% | 64,7% | 1,01 |

### A varredura refutou a premissa de seletividade

A 4 h de contexto, 3 sementes, ordenado por cross F1:

| configuração | cross F1 | cross AUC | intra F1 |
|---|---|---|---|
| 2 h + Fourier | **0,261** | **0,788** | 0,512 |
| 4 h, corte 8 | 0,253 | 0,754 | 0,496 |
| 4 h, corte 24 | 0,249 | 0,769 | 0,511 |
| 4 h, força 1,0 | 0,241 | 0,779 | 0,459 |
| 4 h, corte 16 | 0,234 | 0,764 | 0,513 |
| 4 h sem aumento | 0,213 | 0,576 | **0,666** |
| 4 h, corte 4 | 0,211 | 0,673 | 0,479 |
| 4 h, corte 2 | 0,180 | 0,595 | 0,475 |
| 4 h, **jitter 1,0** | **0,121** | 0,697 | 0,321 |

- **O corte é monótono na direção contrária à premissa:** randomizar **mais**
  bandas é melhor, inclusive as que carregam a assinatura. A ideia de preservar
  a banda do ciclo estava errada, e a tabela de seletividade acima mede uma
  quantidade que **não prevê desempenho**.
- **Amplitude e fase não são simétricas:** força ajuda até 0,5–1,0; jitter de
  fase acima de 0,5 destrói (0,121, o pior do sweep). Faz sentido: embaralhar
  fase desloca o evento no tempo, e num seq2point o rótulo é do instante central.
- O mecanismo, portanto, é **regularização forte**, não remoção seletiva de estilo.

> **Este resultado encolheu muito após a correção do bug (seção 9).**

---

## 8. Vision Transformer sobre transformadas imagem-de-série

GASF/GADF/MTF/RP, ViT com patch 8, d=192, 4 camadas.

| aparelho | alvo | 1D F1 | 1D AUC | ViT F1 | ViT AUC |
|---|---|---|---|---|---|
| m. lavar | intra | **0,639** | 0,920 | 0,498 | **0,939** |
| m. lavar | cross | 0,144 | 0,590 | **0,209** | **0,745** |
| freezer | intra | 0,483 | 0,509 | 0,495 | 0,483 |
| freezer | cross (1) | 0,742 | 0,503 | 0,759 | 0,558 |

O ViT **perde no intra e ganha no cross** — com 1,86 M parâmetros contra 353 mil.
Complementaridade real (AUC cross 0,745 vs 0,590), mas a fusão treinada não a
capturou (seção 4.5).

### A semana inteira piorou

| contexto | m. lavar intra | m. lavar cross |
|---|---|---|
| 1 dia | 0,498 | 0,209 |
| 1+7 dias | **0,323** | **0,073** |

Não é a representação em imagem que atrapalha a semana — **é a semana**. 8 canais
de 96×96 vindos de 2016 amostras, com ~25 mil exemplos, é dimensão de entrada
demais para a quantidade de rótulos.

---

## 9. ⚠ Bug de validação — invalidou a seleção de modelo de toda a fase seq2point

`dividir()` separava corretamente 85/15 dos **centros**, mas `alvos_validos()`
reconstruía os alvos a partir de `s.t`, que é o **mesmo array** nas duas Series.
Nunca olhava `s.centros`.

```
centros:       tr=40555  va=10412     (divisão correta)
alvos_validos: tr=69172  va=69172     (idênticos)
interseção:    69172 de 69172 = 100%
```

**Treino e validação eram o mesmo conjunto.** Afeta tudo que usa `FontePonto`:
`experimento_ponto`, `experimento_vit`, `experimento_duplo`. Os experimentos de
LOHO, perda e pós-processamento usam `Fonte` (que respeita `centros`) e **não
estão afetados**.

**Não é vazamento de teste** — os alvos de teste vêm de arquivos separados. O
estrago é na **seleção**: parada por paciência e escolha de limiar feitas sobre
dados de treino, o que seleciona o checkpoint mais sobreajustado.

### Efeito da correção (máquina de lavar)

| | épocas | intra F1 | cross F1 | cross AUC |
|---|---|---|---|---|
| base, com bug | 100–250 | 0,645 | 0,147 | 0,590 |
| base, corrigido | **20–41** | 0,607 | 0,165 | **0,736** |
| Fourier, corrigido | 20–41 | 0,368 | 0,194 | 0,760 |

- A parada precoce passou a existir de fato (épocas caem 5×).
- **A AUC de transferência do braço base subiu 0,590 → 0,736 só com o conserto.**
- **A vantagem do Fourier encolheu de +0,114 para +0,029**, com desvio 0,106 —
  ou seja, dentro do ruído. O ganho que fora apresentado como significativo
  (p = 0,031) era, em boa parte, aumento de dados compensando a falta de
  validação.

---

## 10. TimesFM (Google) como backbone congelado

`google/timesfm-2.5-200m-pytorch`: decoder-only de **previsão**, 231 M
parâmetros, 20 camadas, d=1280, patch 32. Congelado; treina-se só uma sonda.

| | por aparelho | compartilhado | treinado nos nossos dados |
|---|---|---|---|
| nosso `ClassificadorPonto` | 352 802 | — | 352 802 |
| TimesFM + sonda | 126 914 | 231 000 000 (congelado) | **126 914** |

### O enquadramento importa mais que o modelo

| variante | intra F1 | cross F1 | cross AUC |
|---|---|---|---|
| centrado + atenção | 0,259 | 0,084 | 0,552 |
| **causal + último patch** | **0,606** | 0,148 | **0,787** |

A primeira tentativa foi mal desenhada: janela centrada e pooling por atenção num
modelo **causal**. O último patch é o único que viu tudo até o alvo.

### A camada importa, na direção prevista

Máquina de lavar: no intra a camada 5 vence (0,606 vs 0,545 da 20); no cross é o
inverso (0,148 da 20 vs 0,114 da 5). A última camada está especializada em
**previsão**; as rasas guardam mais do estado instantâneo.

A mistura escalar aprendida (estilo ELMo) sobre as camadas 5/10/15/20
**convergiu para quase uniforme** (pesos 0,23–0,32) e deu ganho marginal — a
hipótese de que resolveria o conflito intra/cross **não se confirmou**.

### Resultado final (config fixada a priori: causal, último patch, mistura)

| aparelho | célula | nosso | TimesFM | conjunto | artigo |
|---|---|---|---|---|---|
| televisão | cross | 0,095 | **0,492** | 0,193 | 0,54 |
| televisão | intra | 0,688 | 0,668 | **0,700** | 0,68 |
| m. lavar | cross | 0,165 | 0,128 | **0,177** | 0,30 |
| m. lavar | intra | **0,607** | 0,585 | 0,607 | 0,63 |
| freezer | cross (2) | 0,585 | **0,746** | 0,741 | 0,78 |
| freezer | intra | 0,494 | **0,497** | 0,495 | 0,46 |

- **A televisão cross saltou de 0,095 para 0,492** (AUC 0,235 → 0,775). Maior
  melhora estrutural do projeto, e vem de **representação**, não de calibração.
- **O TimesFM não vence em geral** — perde na máquina de lavar. A televisão é
  consumo sustentado com padrão diário (parecido com o pré-treino); a máquina de
  lavar é evento raro e abrupto, mais próximo de anomalia que de padrão previsível.
- **O conjunto é condicional:** vence onde ambos são decentes (televisão intra,
  m. lavar cross) e **colapsa onde um é anticorrelacionado** (televisão cross:
  0,193 contra 0,492 do TimesFM sozinho).

### A causa do colapso da televisão

| série | correlação (aparelho, agregado) | salto no agregado ao ligar |
|---|---|---|
| REFIT 8 "Television Site" | **−0,027** | **−75 W** |
| ECO 2 "TV" | +0,355 | **+282 W** |

Na casa de treino, ligar a TV faz o agregado **cair**. Na de teste, **subir**. O
modelo aprendeu a regra certa da casa errada — daí AUC 0,235, pior que o acaso.
O TimesFM vai bem ali porque, sendo congelado, **não pôde aprender a correlação
espúria**.

> Levanta dúvida sobre o dado: um canal de TV de ~133 W cuja ativação reduz o
> agregado sugere medição problemática na casa 8, ou que "Television Site"
> agrega equipamentos que se alternam. **A checar antes de reportar.**

---

## 11. Múltiplas casas fonte — o resultado mais forte do projeto

A hipótese central que emergiu de tudo acima: **o colapso em transferência é
falta de diversidade de fonte, não limitação de arquitetura.** Treino em REFIT
casa 8 + K casas extras (1, 3, 4, 7, 13, 17, 18, 20); testes **idênticos** aos
do artigo, então as colunas seguem comparáveis. Desvio consciente do protocolo:
deixa de valer a cláusula "treinado numa casa só".

### Televisão — de irrecuperável a melhor que o artigo

| casas fonte | cross F1 | ± | cross AUC | intra F1 | intra AUC |
|---|---|---|---|---|---|
| 1 | 0,095 | 0,035 | **0,235** | **0,688** | 0,860 |
| 3 | 0,333 | 0,101 | 0,690 | 0,662 | 0,838 |
| **5** | **0,601** | 0,080 | **0,872** | 0,660 | 0,835 |
| 9 | 0,390 | 0,014 | 0,586 | 0,618 | 0,818 |

De 0,095 para **0,601**, acima do artigo (0,54) e do TimesFM (0,492). A AUC sai
de 0,235 — **pior que o acaso** — para 0,872. É exatamente o mecanismo previsto:
a correlação invertida da casa 8 (TV liga, agregado **cai** 75 W) deixa de ser
regra quando outras casas a contradizem.

### Freezer — a AUC sai do acaso

| casas | intra F1 | intra AUC | cross agregada F1 | cross agregada AUC |
|---|---|---|---|---|
| 1 | 0,494 | **0,500** | 0,682 | 0,462 |
| 3 | **0,521** | **0,652** | **0,744** | 0,671 |
| 5 | 0,516 | 0,672 | 0,708 | 0,669 |
| 9 | 0,508 | 0,648 | 0,722 | 0,697 |

**Correção à seção 3:** a afirmação de que o freezer "não é uma tarefa de
classificação" e que "não há o que aprender" era forte demais. O argumento de
razão sinal-ruído (75 W dentro de 994 W de desvio) continua correto, mas a
informação **existe** — ela só não é extraível de uma casa só. Com 3 casas a AUC
vai de 0,500 a 0,652 e fica lá. O F1 ainda não supera o trivial com folga, então
a tarefa segue inútil na prática, mas por outro motivo que o alegado.

### Máquina de lavar — troca limpa entre intra e cross

| casas | intra F1 | cross F1 | cross AUC |
|---|---|---|---|
| 1 | **0,607** | 0,165 | 0,736 |
| 3 | 0,457 | 0,219 | 0,701 |
| 5 | 0,423 | 0,203 | 0,727 |
| 9 | 0,327 | **0,234** | **0,753** |

Monótona nas duas direções: cross sobe de 0,165 para 0,234 enquanto intra
despenca de 0,607 para 0,327. Coerente — a casa 8 passa de 100% do treino para
1/9, e o modelo deixa de se ajustar a ela. É a mesma tensão de sempre, agora
vinda da **composição dos dados** em vez de regularização.

### A não-monotonicidade é o ponto em aberto

Televisão e freezer **pioram** de 5 para 9 casas (televisão cross: 0,601 → 0,390).
Duas causas plausíveis, ambas testáveis:

1. **Casas ruins no fim da lista** — a ordem (1, 3, 4, 7, 13, 17, 18, 20) é
   arbitrária. Testável embaralhando a ordem e repetindo.
2. **Saturação de capacidade** — 353 mil parâmetros para 75 mil exemplos.
   Testável aumentando `d` ou o número de camadas.

Enquanto isso não for resolvido, "5 casas" não deve ser reportado como ótimo:
é o melhor ponto de uma grade de 4 valores com ordem fixa e 3 sementes.

### Consequência para o resto do documento

Boa parte do esforço das seções 6 e 7 (invariância, randomização de Fourier,
group-DRO) foi **paliativo para um problema de dados**. A comparação direta, na
televisão cross:

| abordagem | cross F1 | cross AUC |
|---|---|---|
| 1 casa + regularização (Fourier) | 0,159 | 0,315 |
| 1 casa + backbone pré-treinado (TimesFM) | 0,492 | 0,775 |
| **5 casas, modelo simples** | **0,601** | **0,872** |

Mais dados de domínios diferentes venceu tanto a regularização quanto o modelo
de fundação de 231 M parâmetros — com a arquitetura mais simples das três.

---

## 12. Lições de método

1. **Validar a divisão treino/validação antes de tudo.** O bug da seção 9 ficou
   ativo por toda a fase seq2point e moveu conclusões nas duas direções.
2. **Comparar sempre contra `2p/(1+p)`.** Sem isso, o freezer parecia resultado.
3. **O oráculo separa representação de calibração.** Quando F1 sobe junto com o
   oráculo, melhorou a representação; quando só a lacuna fecha, foi limiar.
4. **AUC alta com F1 baixo é o padrão dominante aqui** e aponta o gargalo para
   transferência de limiar, não para o modelo.
5. **Não confiar em indício de modelo sub-treinado.** Um modelo de 3 épocas
   transferia com AUC 0,70 contra 0,51 do treinado até o fim — era artefato de
   parada precoce quebrada, não ausência de viés.
6. **Escolher configuração olhando o teste é seleção por oráculo.** As tabelas de
   TimesFM usam configuração fixada **a priori** por isso.
7. **Com 3 sementes e desvio de 0,10, quase nada é significativo.** Antes de mais
   métodos, mais sementes nas células que vão para o texto.
8. **Diversidade de dados venceu arquitetura, regularização e modelo de
   fundação.** Na televisão cross: 0,159 com Fourier, 0,492 com TimesFM (231 M
   parâmetros), **0,601 com 5 casas e o modelo mais simples**. Antes de desenhar
   invariância, verificar se o problema é só falta de domínios no treino.
9. **Uma correlação espúria numa casa vira regra quando há uma casa só.** O canal
   de TV da REFIT 8 anticorrelaciona com o agregado; o modelo aprendeu isso
   corretamente e transferiu com AUC 0,235. Diagnosticável antes de treinar, com
   uma correlação simples entre aparelho e agregado por casa.
