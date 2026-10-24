#!/bin/sh
cd /d/Emiliano/low_rate_nilm
AP="freezer,washing machine,microwave,television"
python src/experimento_ponto.py --aparelhos "$AP" --taxas 5min --formatos centrado \
  --sementes 3 --horas_ctx 24 --saida seis_base.csv > logs/seis_base.log 2>&1
python src/experimento_ponto.py --aparelhos "$AP" --taxas 5min --formatos centrado \
  --sementes 3 --horas_ctx 2 --aumento fourier --saida seis_four.csv > logs/seis_four.log 2>&1
echo "SEIS APARELHOS CONCLUIDO"
