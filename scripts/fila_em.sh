#!/bin/sh
# Repete os dois bracos de 24 h para que as quatro celulas do 2x2 tenham a
# coluna f1_em (limiar pela prevalencia estimada por EM de prior shift).
cd /d/Emiliano/low_rate_nilm
until grep -q "FILA 2x2 CONCLUIDA" logs/fila_2x2.log; do sleep 30; done
python src/experimento_ponto.py --aparelhos "washing machine" --taxas 5min \
  --formatos centrado --horas_ctx 24 --sementes 3 \
  --saida q_base_24h.csv > logs/q_base_24h.log 2>&1
python src/experimento_ponto.py --aparelhos "washing machine" --taxas 5min \
  --formatos centrado --horas_ctx 24 --sementes 3 --aumento fourier \
  --saida q_four_24h.csv > logs/q_four_24h.log 2>&1
echo "FILA EM CONCLUIDA"
