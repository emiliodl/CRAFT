#!/bin/sh
# Espera a fila anterior terminar e roda o fatorial 2x2: contexto x Fourier,
# agora com a coluna f1_quantil (limiar por casamento de taxa).
cd /d/Emiliano/low_rate_nilm
until grep -q "FILA DE DOMINIO CONCLUIDA" logs/fila_dominio.log; do sleep 30; done

for h in 24 4; do
  python src/experimento_ponto.py --aparelhos "washing machine" --taxas 5min \
    --formatos centrado --horas_ctx $h --sementes 3 \
    --saida q_base_${h}h.csv > logs/q_base_${h}h.log 2>&1
  python src/experimento_ponto.py --aparelhos "washing machine" --taxas 5min \
    --formatos centrado --horas_ctx $h --sementes 3 --aumento fourier \
    --saida q_four_${h}h.csv > logs/q_four_${h}h.log 2>&1
done
echo "FILA 2x2 CONCLUIDA"
