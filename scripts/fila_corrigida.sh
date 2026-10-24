#!/bin/sh
cd /d/Emiliano/low_rate_nilm
AP="freezer,washing machine,television"
python src/experimento_ponto.py --aparelhos "$AP" --taxas 5min --formatos centrado \
  --sementes 3 --horas_ctx 24 --saida fix_base.csv > logs/fix_base.log 2>&1
python src/experimento_ponto.py --aparelhos "$AP" --taxas 5min --formatos centrado \
  --sementes 3 --horas_ctx 2 --aumento fourier --saida fix_four.csv > logs/fix_four.log 2>&1
python src/experimento_timesfm.py --aparelhos "$AP" --horas_ctx 24 --sementes 3 \
  --saida fix_timesfm.csv > logs/fix_timesfm.log 2>&1
echo "FILA CORRIGIDA CONCLUIDA"
