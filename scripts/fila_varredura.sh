#!/bin/sh
# Varredura em torno do otimo encontrado (4 h, corte 8, forca 0.5, jitter 0.5).
# O ponto veio de um grid grosso e esta numa borda do espaco, entao vale medir
# as tres direcoes separadamente em vez de assumir que o acidente foi otimo.
cd /d/Emiliano/low_rate_nilm
R() { python src/experimento_ponto.py --aparelhos "washing machine" --taxas 5min \
        --formatos centrado --sementes 3 --aumento fourier "$@" ; }

# A) corte (raias randomizadas), a 4 h.  48 amostras -> raia k = periodo 4h/k
for c in 2 4 16 24; do
  R --horas_ctx 4 --corte $c --saida v_corte_$c.csv > logs/v_corte_$c.log 2>&1
done
# B) forca da amplitude e jitter da fase, a 4 h com corte 8
for f in 0.25 1.0; do
  R --horas_ctx 4 --corte 8 --forca $f --saida v_forca_$f.csv > logs/v_forca_$f.log 2>&1
done
for j in 0.25 1.0; do
  R --horas_ctx 4 --corte 8 --jitter $j --saida v_jitter_$j.csv > logs/v_jitter_$j.log 2>&1
done
# C) contexto, agora COM Fourier (a varredura anterior foi sem)
for h in 2 8; do
  R --horas_ctx $h --corte 8 --saida v_ctx_${h}h.csv > logs/v_ctx_${h}h.log 2>&1
done
echo "VARREDURA CONCLUIDA"
