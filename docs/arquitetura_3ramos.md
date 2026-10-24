# Três ramos no molde do TF-C, sem frequência

Adaptação do TF-C (Zhang et al., NeurIPS 2022) para classificação de estado de
carga em NILM de baixa taxa. O esqueleto é o mesmo do artigo; o que muda é o que
entra em cada ramo e quantos ramos existem.

## O que foi trocado em relação ao TF-C

| TF-C original | Aqui |
|---|---|
| 2 ramos: tempo e frequência | 3 ramos, todos temporais |
| ramo F = FFT da janela | ramos = transformações da série |
| `L_T`, `L_F` (NT-Xent por ramo) | `L_v` para cada uma das 3 vistas |
| `L_C` entre os 2 espaços projetados | `L_C` sobre os 3 pares de vistas |
| pré-treino em dataset A, transferência para B | pré-treino e ajuste na mesma casa |
| tarefa: classificação de janela | tarefa: seq2seq, 288 rótulos por janela |

A razão de tirar a frequência: a 5 min, o espectro de um dia tem Nyquist em
10 min. O ciclo da geladeira (20–40 min) cai a dois ou três bins, e o
lava-louças não tem periodicidade nenhuma. Sobra pouca coisa. As três
transformações abaixo carregam o mesmo tipo de informação (o que muda, em que
escala) sem passar pelo domínio da frequência.

## As três vistas

Todas derivadas apenas do **agregado** — a única entrada disponível em operação.
Calculadas dentro de cada corrida contígua, nunca atravessando um buraco de dias.

| vista | definição | o que resolve |
|---|---|---|
| `base` | `x − mediana móvel (2 h)` | a carga de base varia de 100 a 550 W ao longo do dia; a diferenciação mata a base mas também mata o patamar do evento. Isto mata só a base. |
| `z` | `(x − mediana) / (1,4826 · MAD)` | adimensional. É o que dá chance de o mesmo limiar funcionar em casas com potências e ruídos diferentes. |
| `arcsinh` | `arcsinh(x / 50)` | faixa dinâmica. Sem isso a geladeira de 50 W é esmagada contra o zero pelo lava-louças de 2 kW. Arcsinh e não log porque aceita zero e negativo. |

## Arquitetura

```
             ┌──────────┐  H_base ┐
   x ──base──┤ G_base   ├─────────┤
             └──────────┘         │
             ┌──────────┐  H_z    │   ┌───────┐   ┌────────┐
   x ──z ────┤ G_z      ├─────────┼──▶│ fusão │──▶│ cabeça │──▶ 288 logits
             └──────────┘         │   └───────┘   └────────┘
             ┌──────────┐  H_arc  │
   x ─arcsinh┤ G_arcsinh├─────────┘
             └──────────┘
                  │
                  └── h_v = média dos tokens ──▶ R_v ──▶ z_v ──▶ L_v , L_C
```

- **G_v** — Transformer encoder, 3 camadas, `d = 96`, 4 cabeças, patch de 4
  passos (20 min) → 72 tokens por dia. Um encoder independente por vista, com
  pesos próprios: é isso que faz cada ramo poder se especializar.
- **Fusão** — por porta: um MLP lê os três tokens alinhados e produz um softmax
  sobre as três vistas, token a token. Escolhida por ser a única legível — dá
  para reportar quanto cada vista pesou, por aparelho e por instante. As
  alternativas (concatenação e soma) entram na ablação.
- **Cabeça** — `LayerNorm → Linear → GELU → Linear(d, 4)`, um patch de 4 rótulos
  por token, remontados nos 288 rótulos de 5 min do dia.
- **R_v** — projetor de 2 camadas, saída de 32 dimensões normalizada, só usado
  pelas perdas contrastivas.

## Perdas

```
L = BCE(ŷ, y)                                        sempre
  + λ_v · Σ_v NT-Xent(z_v, z̃_v)  / 3                 pré-treino / conjunto
  + λ_c · Σ_(i<j) NT-Xent(z_i, z_j) / 3              pré-treino / conjunto
```

`L_v` é o `L_T`/`L_F` do artigo: cada vista contra sua própria versão aumentada.
`L_C` é a consistência, generalizada de dois para três ramos — para cada par de
vistas, o mesmo dia é positivo e os outros dias do lote são negativos.
Aumentos genéricos (jitter, escala, máscara temporal de 15%), nada específico de
NILM: a hipótese do TF-C é que a invariância útil vem da consistência entre
ramos, não de aumentos desenhados à mão.

## Protocolo

- **Um modelo por aparelho.** Três instâncias independentes, saída escalar por
  instante. Não há cabeça multirrótulo.
- **Treino:** UK-DALE casa 1. **Teste:** UK-DALE casa 2. A casa 2 nunca entra em
  nenhuma decisão — nem parada, nem limiar, nem escolha de braço.
- **Validação:** últimos 20% de *cada corrida contígua* da casa 1, com folga de
  uma janela na fronteira. Divisão por corrida, e não um corte único no fim da
  série, porque as corridas estão espalhadas por anos — um corte único deixaria
  a validação concentrada numa estação só.
- **Parada:** AP na validação de origem, paciência de 6.
- **Limiar:** o que maximiza F1 na validação de origem. Reporto também o F1 de
  oráculo (melhor limiar no próprio alvo) como teto.
- **Inferência:** janelas sobrepostas com passo de 72 (4 previsões por instante)
  e média das probabilidades, de modo que todo instante acaba com contexto dos
  dois lados — a condição do seq2point, a 1/72 do custo.

## Braços da ablação

| braço | pergunta que responde |
|---|---|
| `piso` | regressão logística nas 4 vistas do próprio instante, sem contexto e sem rede. O chão. |
| `bruto`, `base`, `z`, `arcsinh` | cada vista sozinha vale quanto? |
| `3x_bruto` | 3 encoders, todos no agregado cru. **Isola "3 vistas" de "3× parâmetros".** |
| `3v_concat`, `3v_soma`, `3v_porta` | a fusão importa? |
| `4v_porta` | as transformações substituem o bruto ou complementam? |
| `3v_porta_tfc` | o pré-treino contrastivo do TF-C ajuda? |
| `3v_porta_conj` | e se as perdas contrastivas ficarem ligadas durante todo o treino? |

Três sementes por braço.
