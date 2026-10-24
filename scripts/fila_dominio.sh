#!/bin/sh
# Espera o sweep de K terminar e roda os dois experimentos de generalizacao de dominio.
cd /d/Emiliano/low_rate_nilm
until grep -q "media sobre as sementes" logs/saida_k.log; do sleep 30; done

# 1) Ablacao de contexto: o teste barato da hipotese de fase.
#    Com 24 h o modelo pode se localizar no dia pela curva de carga; com 2-4 h nao.
python src/experimento_ponto.py --aparelhos "washing machine" --taxas 5min \
  --formatos centrado --horas_ctx 24,8,4,2 --sementes 3 \
  --saida contexto.csv > logs/contexto.log 2>&1

# 2) Fourier e group-DRO, a 24 h (onde o vies de fase e maximo).
#    A linha de base 24 h sai do experimento 1, entao aqui so os tres bracos novos.
python src/experimento_ponto.py --aparelhos "washing machine" --taxas 5min \
  --formatos centrado --horas_ctx 24 --sementes 3 --aumento fourier \
  --saida dg_fourier.csv > logs/dg_fourier.log 2>&1
python src/experimento_ponto.py --aparelhos "washing machine" --taxas 5min \
  --formatos centrado --horas_ctx 24 --sementes 3 --dro \
  --saida dg_dro.csv > logs/dg_dro.log 2>&1
python src/experimento_ponto.py --aparelhos "washing machine" --taxas 5min \
  --formatos centrado --horas_ctx 24 --sementes 3 --aumento fourier --dro \
  --saida dg_ambos.csv > logs/dg_ambos.log 2>&1
echo "FILA DE DOMINIO CONCLUIDA"
