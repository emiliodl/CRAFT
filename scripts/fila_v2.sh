# espera a TV da faixa A+ terminar, relanca m. lavar e freezer com o conjunto reduzido
until grep -q "=== washing" logs/v2_escada_Amais.log; do sleep 20; done
powershell -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*--tag escada_Amais' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
python -u src/experimento_evicon.py --fonte A+ --passo 8 --aparelhos television --modelos A0_4h,A0_4hcf,A5 --sementes 5 --tag escada_Amais2 > logs/v2_escada_Amais2_tv.log 2>&1
python -u src/experimento_evicon.py --fonte A+ --passo 8 --aparelhos "washing machine,freezer" --modelos A0_4h,A0_4hcf,A2,A5 --sementes 5 --tag escada_Amais2 > logs/v2_escada_Amais2.log 2>&1
# E1.2: curva de escala, 2 sorteios, 3 sementes
for s in 1 2; do for n in 1 3 7; do
python -u src/experimento_evicon.py --fonte A+ --passo 8 --aparelhos "television,washing machine" --modelos A0_4h,A2 --sementes 3 --n_extra $n --sorteio $s --tag e12_n${n}_s${s} > logs/v2_e12_n${n}_s${s}.log 2>&1
done; done
