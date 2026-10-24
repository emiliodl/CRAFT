# Resultados: multi-casa e normalização por casa

Duas medidas implementadas sobre a arquitetura de três ramos
([arquitetura_3ramos.md](arquitetura_3ramos.md)), em resposta ao diagnóstico de
que o modelo ia bem dentro da casa e colapsava ao atravessar para outra.

1. **Leave-one-house-out** — treinar em ~18 casas em vez de uma
   ([src/experimento_loho.py](../src/experimento_loho.py)).
2. **Vista `casa`** — `x / p95(agregado da própria casa)` no lugar do `z`, que
   dividia pelo MAD de janela móvel ([src/dados.py](../src/dados.py)).

Um modelo por aparelho. A casa-alvo não entra em treino, parada nem escolha de
limiar; o único uso dela antes da medição é o p95 do próprio agregado, que não
precisa de submedidor.

## Comparação limpa: mesmo alvo, 1 casa de origem contra 18

UK-DALE casa 2, F1 da classe positiva.

| | piso | bruto | 3 vistas |
|---|---|---|---|
| **geladeira** 1 casa | 0,783 | 0,823 | 0,851 |
| **geladeira** 18 casas | 0,783 | **0,878** | 0,873 |
| **lava-louças** 1 casa | 0,643 | 0,399 | 0,451 |
| **lava-louças** 18 casas | 0,620 | **0,811** | 0,720 |
| **máq. lavar** 1 casa | 0,147 | 0,192 | 0,191 |
| **máq. lavar** 18 casas | 0,178 | **0,261** | 0,189 |

O lava-louças é o caso extremo: 0,399 → 0,811 no braço `bruto`. Era o aparelho
em que a regressão logística de 5 parâmetros humilhava a rede (0,643 contra
0,451); com 18 casas de origem a rede ganha por 0,19.

## Leave-one-house-out completo, 8 folds por aparelho

| braço | geladeira | lava-louças | máq. lavar |
|---|---|---|---|
| piso | 0,522 ± 0,233 | 0,320 ± 0,061 | 0,260 ± 0,096 |
| **bruto** | **0,558** ± 0,247 | **0,584** ± 0,131 | **0,387** ± 0,164 |
| 3v_antigo (base+z+arcsinh) | 0,547 ± 0,243 | 0,496 ± 0,105 | 0,355 ± 0,138 |
| 3v_novo (base+casa+arcsinh) | 0,518 ± 0,249 | 0,482 ± 0,128 | 0,322 ± 0,090 |
| 4v_novo (+bruto) | 0,519 ± 0,253 | 0,429 ± 0,119 | 0,314 ± 0,082 |

## O que ficou estabelecido

**1. A medida 1 funcionou nos três aparelhos.** Com uma casa de origem o modelo
não consegue separar "o que é lava-louças" de "o que é o lava-louças desta
casa", porque não há variação de assinatura no treino. Com 18, consegue.

**2. A medida 2 não funcionou em nenhum.** `3v_novo` fica sempre abaixo de
`3v_antigo`. A intuição (tirar a escala do domicílio) estava certa; o ganho
esperado não apareceu porque o problema que ela resolvia — a rede só ter visto
uma escala — já é resolvido por ter 18 escalas no treino.

**3. Com muitas casas, o agregado cru sozinho vence todos os conjuntos de
vistas, nos três aparelhos.** As transformações não são inúteis: elas ganhavam
quando havia uma casa só (geladeira 0,851 contra 0,823 do bruto). Elas eram um
**substituto para a variedade de domínios**. Com a variedade presente, a rede
aprende sozinha o que elas codificavam à mão, e impor as três vistas passa a
custar capacidade em vez de render.

Isto é o oposto do que a arquitetura de três ramos propunha, e é o achado mais
sólido do conjunto — vale nos três aparelhos, em duas formas de avaliação
(alvo fixo e média sobre 8 folds) e com o controle de capacidade `3x_bruto` do
experimento anterior mostrando que o efeito não era número de parâmetros.

## Problemas conhecidos nestes números

**Rótulo quebrado na casa 8 do REFIT (geladeira).** 1,06% de janelas ligadas,
contra 12%–55% nas outras 24 séries. Uma geladeira cicla o dia inteiro: o
rótulo está errado, não o modelo. É o limiar fixo de 50 W contra uma geladeira
que puxa 66 W médios. Esse fold dá F1 ≈ 0,03 em todos os braços, derruba a média
da geladeira em ~0,07 e é o que produz o desvio de ±0,25. Sem ele as médias
ficam em torno de 0,63. **Não foi removido**: mudar o critério depois de ver o
resultado é como se fabrica um número bonito. A correção defensável é limiar por
casa, calibrado pela distribuição de potência do próprio aparelho, aplicado a
todas as séries antes de rodar de novo.

**Uma semente por fold.** A dispersão reportada é entre casas, não entre
inicializações. No experimento anterior a dispersão entre sementes chegou a
±0,078, então parte do ±0,13 do lava-louças é ruído de treino.

**Folds enviesados para casas grandes.** Os 8 alvos são as casas com mais dias.

**Prevalência.** A máquina de lavar continua baixa em F1 bruto (0,261 na casa 2)
porque a prevalência lá é 1,6%; a F1 padronizada a 5% dá 0,467. As duas estão
reportadas em todas as linhas do CSV.

## Arquivos

- `resultados/loho_{geladeira,lava_loucas,maq_lavar}.csv` — 8 folds × 5 braços
- `resultados/loho_alvo_casa2*.csv` — comparação de alvo fixo
- `resultados/ablacao_*.csv` — linha de base casa 1 → casa 2, 12 braços × 3 sementes
- `figuras/loho_antes_depois.png`

---

# Pós-processamento com prior de duração (máquina de lavar)

Hipótese: a rede detecta o pico de aquecimento (recall 0,39–0,42 acima de 300 W)
mas perde o miolo do ciclo (recall 0,22–0,30 abaixo de 300 W); usar a
continuidade do ciclo faria o pico puxar o resto. Implementado em
[src/decodificador.py](../src/decodificador.py), avaliado em
[src/experimento_pos.py](../src/experimento_pos.py) sobre o braço `bruto`,
8 folds. Todos os parâmetros vêm da validação das casas de origem.

| decodificador | F1 | ± | precisão | recall | F1 na origem |
|---|---|---|---|---|---|
| cru (limiar por instante) | 0,387 | 0,164 | 0,396 | 0,395 | 0,549 |
| morfológico | **0,394** | 0,162 | 0,404 | 0,401 | 0,553 |
| Viterbi (HMM) | 0,389 | 0,172 | 0,424 | 0,384 | 0,559 |
| *cru, limiar de oráculo* | *0,405* | | *0,348* | *0,500* | |
| *morfológico de oráculo* | *0,422* | | *0,377* | *0,505* | |
| *Viterbi de oráculo* | *0,418* | | *0,372* | *0,498* | |

**Funciona, e é pequeno.** O morfológico melhora em 7 dos 8 folds, mas só
+0,007 na média. O Viterbi melhora em 5 de 8, +0,002. Mesmo com o parâmetro
ajustado no próprio alvo (oráculo, que não é resultado legítimo), o teto do
método é +0,017.

## Por que é pequeno

A medida que decide:

| | |
|---|---|
| ciclos verdadeiros tocados (≥1 janela detectada) | **59%** |
| cobertura média dentro dos ciclos tocados | 60% |

**41% dos ciclos não produzem nenhuma janela detectada.** Prior de duração
propaga a partir de uma detecção; em 41% dos casos não há de onde propagar. O
teto de recall de qualquer suavização é 0,59, e o recall atual já é 0,395.

E o ganho depende exatamente disso: a correlação entre a fração de ciclos
tocados e o ganho do Viterbi é **0,617** entre os folds. Onde a detecção é boa
(uk_dale casa 1, 69% tocados) o Viterbi dá +0,025; onde é ruim (refit casa 8,
41%) tira 0,009 — suavizar amplifica o erro em vez de corrigi-lo.

| alvo | cru | Viterbi | ganho | ciclos tocados |
|---|---|---|---|---|
| uk_dale/casa 1 | 0,448 | 0,473 | +0,025 | 68,7% |
| refit/casa 9 | 0,333 | 0,336 | +0,002 | 67,4% |
| refit/casa 15 | 0,758 | 0,771 | +0,013 | 66,5% |
| refit/casa 10 | 0,410 | 0,424 | +0,014 | 58,5% |
| refit/casa 2 | 0,251 | 0,260 | +0,009 | 57,9% |
| refit/casa 16 | 0,295 | 0,281 | −0,013 | 57,5% |
| refit/casa 6 | 0,309 | 0,287 | −0,021 | 54,8% |
| refit/casa 8 | 0,291 | 0,282 | −0,009 | 40,9% |

## Consequência

O gargalo da máquina de lavar **não é** transformar probabilidades em rótulos —
é que 41% dos ciclos não produzem nenhuma janela **acima do limiar de decisão**.
Isso desloca o problema para detecção, não para suavização.

Atenção ao que esta frase não diz: os 41% não são necessariamente invisíveis no
agregado. O experimento de perdas abaixo mostra que uma perda mais agressiva
leva a fração tocada de 59% para 86% — parte deles está no sinal, só abaixo do
ponto de operação escolhido. O que ainda não foi medido é quanto disso é ganho
real e quanto é apenas o efeito de prever mais positivos.


---

# Perda alinhada com a F1

Todos os experimentos anteriores treinaram com BCE pura. Com 2-5% de janelas
positivas ela é dominada pelos negativos, e o modelo aprende a ser conservador.
Implementado em [treino.py](../src/treino.py) (`fazer_perda`), avaliado em
[src/experimento_perda.py](../src/experimento_perda.py): 5 perdas × 8 folds ×
3 aparelhos, trocando **só** a função objetivo.

F1 com limiar cru / com fechamento morfológico:

| perda | geladeira | lava-louças | máq. lavar |
|---|---|---|---|
| **bce** | 0,602 / 0,600 | 0,584 / **0,612** | **0,387 / 0,394** |
| bce_peso | 0,600 / 0,599 | 0,518 / 0,552 | 0,279 / 0,278 |
| focal | 0,595 / 0,594 | 0,550 / 0,585 | 0,327 / 0,321 |
| bce_soft_f1 | **0,609** / 0,609 | 0,465 / 0,546 | 0,381 / 0,390 |
| soft_f1 | 0,498 / 0,495 | 0,458 / 0,529 | 0,307 / 0,313 |

**Não funcionou.** A BCE pura ganha ou empata nos três aparelhos. O único
resultado acima dela, `bce_soft_f1` na geladeira, é +0,007 contra um desvio
entre folds de 0,18 — não é efeito.

## Mas o mecanismo é real e mensurável

A fração de ciclos verdadeiros tocados (≥1 janela detectada):

| perda | geladeira | lava-louças | máq. lavar |
|---|---|---|---|
| bce | 89,9% | 74,2% | 59,0% |
| soft_f1 | **98,6%** | **84,4%** | **85,6%** |

A `soft_f1` faz exatamente o que se pediu dela: detecta muito mais ciclos. O
recall sobe junto (máq. lavar 0,395 → 0,458; geladeira 0,724 → 0,851). Mas a
precisão desaba (0,396 → 0,298 e 0,581 → 0,367) e o saldo em F1 é negativo.

Duas leituras, e não dá para escolher entre elas com o que foi medido:

1. o sinal dos ciclos perdidos existe, e a F1 é que não recompensa recuperá-lo
   ao preço de precisão que ele custa;
2. a `soft_f1` simplesmente prevê mais positivos (a taxa de positivos previstos
   sobe ~54%) e toca mais ciclos por sorte.

Para separar as duas basta medir os ciclos tocados do modelo BCE **em função do
limiar**: se baixando o limiar ele também chega a 86%, é só ponto de operação.

## Achado colateral: o fechamento morfológico depende do aparelho

| | ganho do morf sobre o cru (perda bce) |
|---|---|
| lava-louças | **+0,028** |
| máq. lavar | +0,007 |
| geladeira | −0,002 |

Faz sentido pelo comprimento do ciclo: o lava-louças roda 50-90 min (10-18
janelas) e se beneficia de fechar buracos; a geladeira cicla a cada 15-25 min
(3-5 janelas) e o fechamento apaga estrutura real. O ganho é maior quanto mais
fragmentada a saída — na `soft_f1` do lava-louças são +0,071.

Com isso o melhor número do lava-louças passa de 0,584 para **0,612**.

## Nota sobre os números da geladeira

Estes folds já rodaram **sem** `refit__fridge__casa_8`, removida a pedido. A
média subiu de 0,558 para 0,602 e o conjunto de alvos mudou (entrou a casa 19 e
a casa 4). A casa 4, que eu havia sinalizado como tendo o mesmo defeito de
rótulo em grau menor (14,1% de prevalência, p95 de 52,4 W contra limiar de
50 W), é agora o pior fold: 0,302.
