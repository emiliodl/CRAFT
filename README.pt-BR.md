[English](README.md) | **Português**

# CRAFT

**Detecção do estado ligado/desligado de aparelhos a partir de medidores inteligentes de 5 e 15 min, com treino em agregados recompostos.**

Código do manuscrito *NILM from Low-Rate Smart Meter Data by Aggregate Recomposition*
(Emiliano D. S. C. Lima e Ricardo A. S. Fernandes, EESC, Universidade de São Paulo).

## Visão geral

Medidores inteligentes em operação guardam uma média a cada 5 a 15 min. Nessas taxas, a
assinatura dos aparelhos se dilui: uma televisão de 70 W vira um degrau pequeno sobre um
agregado de centenas de watts, e um ciclo de máquina de lavar se reduz a poucas amostras. O
benchmark de Chatterjee e Heer (2025, *Energy and Buildings*,
doi:10.1016/j.enbuild.2025.116002) treinou seis algoritmos de NILM com um ano da casa 8 do
REFIT. No teste em casas suíças do ECO, o melhor F1 a 5 min foi de só 0,54 para a televisão
e 0,30 para a máquina de lavar, e menor ainda a 15 min. Um modelo treinado numa casa aprende
os hábitos dela (horários de uso, aparelhos que ligam juntos), e esses hábitos não se
repetem em outra casa.

O CRAFT é um classificador binário, com um modelo por aparelho. Recebe uma janela de 4 h do
agregado (48 intervalos a 5 min, 16 a 15 min), diz se o aparelho-alvo está ligado no
intervalo central e usa os mesmos dados de treino do benchmark (um ano da casa 8 do REFIT).
Tem três componentes:

1. **Recomposição do agregado com rótulos recalculados.** Os dados de treino ficam numa
   grade de 30 s que guarda a potência média e o número de amostras nativas de cada caixa.
   O gerador monta janelas novas somando e subtraindo sinais reais nessa grade: troca o
   fundo (plugue do alvo sobre o fundo de outro dia), injeta uma ativação real reescalada
   em fase aleatória, remove o alvo ou injeta um aparelho confundível (distrator). Depois
   reamostra para 5 ou 15 min, com média ponderada pelas contagens, e recalcula o rótulo
   com a regra do próprio benchmark (média do plugue ≥ 10 W para TV e freezer, ≥ 20 W para
   a máquina de lavar). Cada janela recomposta muda um fator e mantém os outros; assim o
   modelo não pode se apoiar em coocorrências e precisa usar a assinatura do aparelho.
   Cerca de 40% de cada lote são janelas reais.
2. **Conjunto de cinco ResNets 1D.** Três blocos residuais (32/64/64 canais); as redes
   diferem só na largura do primeiro kernel de cada bloco (k = 5, 7, 9, 15, 25). Quatro
   canais de entrada: o nível em asinh normalizado pela casa e o nível, as subidas e as
   descidas em unidades da potência típica do alvo, `P_ref`. Sem hora do dia; 0,79 M de
   parâmetros no total.
3. **Ramo de evidência como sexto membro.** Dois detectores de borda pequenos, com
   restrição de sinal, pontuam subidas e descidas; o de subida só enxerga diferenças
   positivas, então uma queda no agregado nunca vira "ligou". Um acumulador bidirecional
   com decaimento aprendido transforma as bordas em estado. Algumas centenas de parâmetros.

O escore é a média dos seis logits. A decisão usa Platt e o limiar que maximiza o F1,
ajustados em dados rotulados que não entram no treino nem no teste: regra `d_L` (células
entre datasets) nas casas 3, 7 e 20 do REFIT; regra `d_A` (células intra-dataset) nas
quatro últimas semanas do ano de treino.

![Pipeline do CRAFT](artigo/figuras/pipeline_en.png)

## Principais resultados

F1 da classe "ligado" no teste, média de 10 sementes, CRAFT treinado só na casa 8 do REFIT.
**Bench.** é o melhor dos seis métodos do benchmark na mesma taxa. **S1** (intra-dataset):
casa 8 do REFIT, no mês seguinte ao treino. **S2** (entre datasets): casas suíças do ECO,
junho de 2012.

| Cenário | Aparelho | 5 min Bench. | 5 min CRAFT | 15 min Bench. | 15 min CRAFT |
|---|---|---|---|---|---|
| S1 intra | TV | 0,68 | 0,70 (empate) | 0,53 | **0,67** |
| S1 intra | Máquina de lavar | 0,63 | **0,85** | 0,35 | **0,65** |
| S1 intra | Freezer | 0,46 | **0,69** | 0,55 | **0,65** |
| S2 entre datasets | TV | 0,54 | **0,69** | 0,36 | **0,76** |
| S2 entre datasets | Máquina de lavar | 0,30 | 0,27 (empate) | 0,30 | 0,27 (empate) |
| S2 entre datasets | Freezer | 0,78 | *0,75* (abaixo)† | 0,84 | **0,90**† |

Negrito: supera o benchmark (F1 maior, p corrigido por Holm < 0,05). Itálico: fica abaixo.
†: não passa do F1 de um classificador que sempre diz "ligado".

![F1 do CRAFT e do melhor método do benchmark a 5 e 15 min](artigo/figuras/placar_taxas_en.png)

Ressalvas declaradas no manuscrito:

- **Freezer entre datasets, 5 min:** 0,75, abaixo do benchmark (0,78) e do "sempre ligado"
  (0,79); o freezer da casa 8 não transfere para os freezers suíços.
- **Freezer a 15 min:** as duas vitórias não dizem nada sobre o freezer. Entre datasets, o
  CRAFT previu "ligado" em todos os intervalos, então seu F1 é o do "sempre ligado" (0,90),
  que já fica acima do benchmark (0,84). No intra, vence o "sempre ligado" por só 0,04
  (0,65 contra 0,61).
- **Máquina de lavar entre datasets:** inconclusivo nas duas taxas (0,27 contra 0,30); uma
  casa de treino oferece poucos ciclos (ligada 4% do tempo) e um só conjunto de programas.
- **Computador e chaleira:** células degeneradas do benchmark, excluídas. O computador da
  casa 8 está ligado em 100% dos intervalos; a chaleira nunca atinge o limiar de 2 kW numa
  média de 5 ou 15 min.
- **TV entre datasets:** uma única casa do ECO (39 ativações), cujo agregado registra quase
  todo o degrau da TV.
- **Reuso do teste:** a 5 min, o ECO e o mês de teste da casa 8 foram consultados em 26
  rodadas registradas durante o desenvolvimento; a 15 min, as casas e os meses são os
  mesmos, então não é um teste independente.

**UK-DALE (5 min, sem referência publicada),** treino na casa 1, de março de 2013 a março de 2014:

| Aparelho | Mesma casa (casa 1, mês seguinte) | Outra casa (casa 2) |
|---|---|---|
| Geladeira | 0,83 | 0,80 |
| Lava-louças | 0,96 | 0,40 |
| Máquina de lavar (lava e seca na casa 1) | 0,94 | 0,14 |

Os resultados finais por célula por trás dessas tabelas (F1, IC95%, valores p, AUC,
prevalência e fração prevista "ligado", para cada regra de decisão, com e sem o ramo de
evidência) estão em `resultados/v3/`: `final_craftE__celulas.csv` (5 min),
`final_craftE15__celulas.csv` (15 min), `comparacao_5x15__celulas.csv` e
`final_ukdale_s__celulas.csv`. O manuscrito relata a faixa `A` e a regra pré-registrada de
cada célula (ver o glossário abaixo).

## Protocolo de avaliação

- **Pré-registro.** Antes de cada rodada de teste, um arquivo fixou o modelo final, as
  regras de decisão, as comparações e o resultado esperado de cada célula; o hash foi
  registrado. A rodada de 15 min teve pré-registro próprio, escrito antes de abrir qualquer
  dado de teste de 15 min.
- **Escolhas só na validação.** Tudo foi decidido nas casas 3, 7 e 20 do REFIT (deixando uma
  de fora por vez) e nas quatro últimas semanas da casa 8; nenhum dado do ECO ou do mês de
  teste foi usado antes do teste.
- **Sementes e métrica.** 10 sementes por célula, F1 médio. O F1 vem de TP/FP/FN somados
  sobre todos os intervalos de teste da célula, ao lado da prevalência π e do F1 do "sempre
  ligado", `2π/(1+π)`.
- **Incerteza.** 2000 réplicas de bootstrap sorteando blocos de um dia (a partir do
  meio-dia) junto com as sementes. O p bilateral contra o benchmark é corrigido por Holm
  dentro de cada família pré-registrada; vitória ou derrota exige p corrigido < 0,05,
  senão é empate.

## Datasets

Os dados brutos não são distribuídos. Baixe-os e coloque-os em `datasets/` (ignorado pelo git).

| Dataset | Uso | Fonte |
|---|---|---|
| REFIT (limpo, `CLEAN_REFIT_081116`) | treino (casa 8), validação e calibração (casas 3, 7, 20), teste S1 | https://doi.org/10.15129/9ab14b0e-19ac-4279-938f-27f643078cec |
| ECO | teste S2 (casas 1, 2, 5; 2 jun a 1 jul 2012) | https://vs.inf.ethz.ch/res/show.html?what=eco-data |
| UK-DALE (`ukdale.h5`, HDF5 do NILMTK) | cenários UK-DALE (casas 1, 2, 5) | https://jack-kelly.com/data/ |
| REDD (`redd.h5`, convertido com o NILMTK) | cenário extra opcional, fora do manuscrito | http://redd.csail.mit.edu/ |

```
datasets/
├── refit/CLEAN_REFIT_081116/{CLEAN_House1.csv ... CLEAN_House21.csv, MetaData_Tables.xlsx}
├── ECO/
│   ├── 01_sm_csv/01/2012-06-02.csv ...          # medidor, um CSV por dia
│   ├── 01_plugs_csv/01/<plug>/2012-06-02.csv ... # plugues, p. ex. 01_plugs_csv/01/05/
│   └── 02_..., 05_... (mesma estrutura; 0X_doc.txt opcional)
├── ukdale.h5
└── redd.h5                                       # opcional
```

Os scripts de preparação geram `datasets/protocolo_artigo/` (arquivos do protocolo a 5/15
min) e `datasets/v3_fino/` (grades de 30 s). Sem o `0X_doc.txt` de uma casa do ECO, o mapa
de plugues usa o metadado do NILMTK copiado em `scripts/preparar_artigo.py`.

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
# instale antes o PyTorch para a sua versão de CUDA (https://pytorch.org); depois:
pip install -r requirements.txt
```

Rodadas relatadas: Python 3.11, PyTorch 2.13 (CUDA 12.6), NumPy 2.4, scikit-learn 1.7, uma
NVIDIA RTX 3080 Ti, de 2 a 8 min por membro e semente. O código usa CUDA se houver e, se
não, a CPU (`src/treino.py`); nenhuma rodada foi feita só em CPU.

## Como reproduzir

Rode a partir da raiz do repositório (Git Bash ou similar para `VAR=valor comando`; no
PowerShell use `$env:NILM_TAXA=900`). Os nomes no código estão em português:

| Código | Manuscrito |
|---|---|
| `CRAFT+E` | CRAFT (`CRAFT` nos arquivos de resultado = as cinco ResNets sem o ramo de evidência) |
| `G5` ... `G25` / `A2` | as cinco ResNets / o ramo de evidência |
| `--fonte A` (faixa A) | treino só na casa 8 do REFIT; a faixa `A+` acrescenta casas do REFIT e não está no manuscrito |
| `d_A`, `d_L`, `q` | regras de decisão |
| `intra` / `cross` | S1 / S2 |

**1. Preparar os dados.**

```bash
python scripts/preparar_artigo.py --casas_refit 8,3,7,20   # protocolo a 5/15 min, REFIT + ECO 1/2/5
python scripts/auditoria_p0.py --casas_refit 8,3,7,20      # máscaras de lacuna; reescreve resultados/p0_*.csv
python scripts/preparar_v3.py --casas 8,3,7,20             # grades de 30 s para o gerador de recomposição
```

**2. Treinar (5 min, faixa A, 10 sementes).** As tags abaixo são as das rodadas originais,
então o pré-registro distribuído as encontra. `--olhar_teste` também prevê as células de
teste e acrescenta uma linha por aparelho ao registro de consultas ao teste,
`resultados/v2/olhadas.csv`. As predições vão para `resultados/v3/pred/<tag>/` (ResNets) e
`resultados/v2/pred/<tag>/` (ramo de evidência).

```bash
python src/experimento_reco.py --aparelhos "television,washing machine" --modelos G5,G7,G9,G15,G25 \
  --fonte A --sementes 10 --lotes_epoca 400 --passo 4 --olhar_teste --tag craftF_A
python src/experimento_reco.py --aparelhos freezer --modelos G5,G7,G9,G15,G25 \
  --fonte A --sementes 10 --lotes_epoca 400 --passo 4 --olhar_teste --tag craftE_A_fz
python src/experimento_evicon.py --aparelhos "television,freezer" --modelos A2 --fonte A \
  --sementes 10 --epocas 40 --paciencia 6 --passo 4 --olhar_teste --tag craftE_A_ev
python src/experimento_evicon.py --aparelhos "washing machine" --modelos A2 --fonte A \
  --sementes 10 --epocas 40 --paciencia 6 --passo 4 --olhar_teste --tag final_A
```

**3. Analisar.** O `src/final_craftE.py` junta os seis membros de cada semente, aplica as
regras, faz o bootstrap e o Holm e grava `resultados/v3/<nome>__celulas.csv` (`<nome>` = nome
do arquivo de pré-registro com `preregistro_` trocado por `final_`).

```bash
python src/final_craftE.py --preregistro resultados/v3/preregistro_craftE.json
```

Os pré-registros distribuídos (`preregistro_craftE.json`, `preregistro_craftE15.json`) trazem
as sementes (`sementes`), a regra primária por tipo de célula (`regra_primaria`), a família
de Holm (`familia_primaria`), as tags das rodadas por faixa e aparelho (`execucoes`), os
comandos exatos de treino (`rodadas_novas`), os resultados esperados e o hash. Eles também
listam rodadas da faixa A+: no manuscrito, as células intra foram corrigidas por Holm numa
família de seis junto com três células da A+, então reproduzir esses valores p exatamente
exige as rodadas A+ listadas lá. Rodadas ausentes geram avisos, e as demais células são
analisadas normalmente. **Rodar de novo sobrescreve os `*__celulas.csv` distribuídos.**

**4. 15 min.** Mesma configuração com `NILM_TAXA=900`, contexto de 16 intervalos e passo 1
(lista completa, inclusive a faixa A+, em `rodadas_novas` de `preregistro_craftE15.json`):

```bash
for p in "television tv" "washing_machine wm" "freezer fz"; do set -- $p
  NILM_TAXA=900 python src/experimento_reco.py --aparelhos "${1//_/ }" --modelos G5,G7,G9,G15,G25 \
    --fonte A --sementes 10 --epocas 30 --paciencia 5 --lotes_epoca 400 --ctx 16 --passo 1 \
    --olhar_teste --tag c15_A_$2
done
NILM_TAXA=900 python src/experimento_evicon.py --aparelhos "television,washing machine,freezer" \
  --modelos A2 --fonte A --sementes 10 --epocas 40 --paciencia 6 --passo 1 --olhar_teste --tag c15_A_ev
NILM_TAXA=900 python src/final_craftE.py --preregistro resultados/v3/preregistro_craftE15.json
python src/comparar_taxas.py          # -> resultados/v3/comparacao_5x15__celulas.csv
```

**5. Cenários UK-DALE** (o S3 e o S4 do manuscrito são S4 e S5 no código; desenho em
`resultados/v3/preregistro_ukdale_s.json`):

```bash
python scripts/preparar_ukdale_cenarios.py      # precisa das grades de 30 s das casas 3, 7, 20 do REFIT
T="intra: ukdale casa 1@ukdale_casa_1__{ap}__5min__teste.npz@fonte;cross: ukdale casa 2@ukdale_casa_2__{ap}__5min__cross.npz@proprio"
for ap in fridge dishwasher "washing machine"; do
  python src/experimento_reco.py --aparelhos "$ap" --modelos G5,G7,G9,G15,G25 --fonte A \
    --fonte_fina 301 --sementes 10 --lotes_epoca 400 --passo 4 --olhar_teste --testes "$T" \
    --tag "ukdale_s_${ap// /_}"
done
python src/experimento_evicon.py --aparelhos "fridge,dishwasher,washing machine" --modelos A2 \
  --fonte A --fonte_ds ukdale --fonte_casa 1 --sementes 10 --epocas 40 --paciencia 6 \
  --passo 4 --olhar_teste --testes "$T" --tag ukdale_s_ev
python src/final_ukdale.py                      # -> resultados/v3/final_ukdale_s__celulas.csv
```

`final_ukdale.py --cenario casa5|15|s6` cobre cenários extras, fora do manuscrito (casa 5 do
UK-DALE, 15 min, REDD), com dados de `preparar_ukdale_cenarios.py --casa5`,
`preparar_ukdale_15min.py` e `preparar_redd_cenario.py`.

**6. Figuras.** `artigo/figuras/gerar_placar_taxas_en.py` (placar 5 x 15 min) e
`gerar_placar.py` / `gerar_placar_en.py` (5 min) regeneram as figuras de placar a partir dos
`final_craftE*__celulas.csv` distribuídos. As figuras do pipeline e do exemplo precisam
também dos dados preparados e das predições salvas.

**Etapa de validação.** `src/validacao_craftE.py` (ResNets sozinhas x ResNets + ramo de
evidência) e `src/validacao_ctx15.py` (contexto de 4 h x 12 h a 15 min) leem predições de
tags fixas listadas no topo de cada arquivo e nunca leem predições de teste.

## Estrutura do repositório

```
src/                 modelo, treino e análise
  reco.py            gerador de recomposição (R0-R4), reamostragem de 30 s, ResNet 1D, entradas
  evicon.py          ramo de evidência (detectores de borda com restrição de sinal + acumulador)
  experimento_reco.py, experimento_evicon.py   pontos de entrada do treino
  final_craftE.py, final_ukdale.py, comparar_taxas.py   análises finais
  validacao_craftE.py, validacao_ctx15.py      escolhas feitas só na validação
  decisao.py, estatistica.py, quantil.py       regras de decisão, bootstrap + Holm, regra por quantil
  dados*.py, ponto.py, saneamento.py, treino.py, taxa.py, contrafactual.py,
  experimento_quantil.py, final_v2.py          leitura dos dados, holdout, taxa, utilitários
  (demais arquivos)  etapas anteriores e exploratórias: ablações do v1/v2/v3, Transformer e
                     sondas do TimesFM, teste do discriminador, experimentos pós-fechamento
scripts/
  preparar_artigo.py, auditoria_p0.py, preparar_v3.py   dados do pipeline final
  preparar_ukdale_*.py, preparar_redd_cenario.py        entradas do UK-DALE e do REDD
  preparar_5min.py, limpar_5min.py, indexar_5min.py     base de 5 min da primeira etapa (lê
                     datasets/cache/, que nenhum script daqui gera)
  fila_*.sh, final_faixa*.sh, scripts de gráficos       rodadas antigas e gráficos exploratórios
tests/               testes de unidade com dados sintéticos
docs/                notas de trabalho; v3_resultados.md, seções 11-15, cobre as rodadas finais
figuras/             figuras exploratórias
artigo/figuras/      figuras do manuscrito (PNG) e seus geradores (gerar_*.py)
resultados/
  *.csv, *.json      resultados agregados das etapas anteriores (reprodução do benchmark, ablações, auditoria)
  v3/preregistro_craftE.json, preregistro_craftE15.json, preregistro_ukdale_s.json   pré-registros
  v3/final_craftE__celulas.csv, final_craftE15__celulas.csv, comparacao_5x15__celulas.csv,
     final_ukdale_s__celulas.csv   resultados finais por célula do manuscrito
```

Os documentos também citam `logs/` e outros arquivos de `resultados/v2/` e `resultados/v3/`;
são arquivos de trabalho locais, fora deste repositório (os scripts de treino recriam as pastas).

## Testes

```bash
python -m pytest tests -q
```

Casas sintéticas, sem dataset. Verificam que a recomposição preserva ou recalcula o rótulo,
que recompor e reamostrar comutam, que as casas retidas nunca entram nos bancos, que o ramo
de evidência não transforma queda em "ligado", que as regras de decisão se comportam como
especificado e que o holdout temporal não tem vazamento.

## Citação

```bibtex
@unpublished{lima2026craft,
  author = {Lima, Emiliano D. S. C. and Fernandes, Ricardo A. S.},
  title  = {{NILM} from Low-Rate Smart Meter Data by Aggregate Recomposition},
  note   = {Manuscript},
  year   = {2026}
}
```

## Autores

- Emiliano D. S. C. Lima ([@emiliodl](https://github.com/emiliodl))
- Ricardo A. S. Fernandes

Departamento de Engenharia Elétrica e Computação, Escola de Engenharia de São Carlos
(EESC), Universidade de São Paulo, São Carlos, Brasil. Dúvidas: abra uma issue no GitHub.
