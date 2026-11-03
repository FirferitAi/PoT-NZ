@echo off
rem Полный прогон: тесты -> серии экспериментов -> проверки упаковки -> отчёты
cd /d "%~dp0"
echo === тесты квантователя и упаковщика ===
.venv\Scripts\python.exe tests\test_quant.py || exit /b 1
.venv\Scripts\python.exe tests\test_pack.py || exit /b 1
echo === серия экспериментов (маленькая модель) ===
.venv\Scripts\python.exe src\run_experiments.py --config config.yaml --spec experiments.yaml || exit /b 1
echo === проверка упаковки (нулевые потери, реальный размер .bin) ===
.venv\Scripts\python.exe src\verify_pack.py runs\C_p4_skipzero || exit /b 1
echo === проверка разреженной упаковки (маска нулей) ===
.venv\Scripts\python.exe src\verify_sparse.py runs\B_p4_zero || exit /b 1
echo === сравнение с int8 ===
.venv\Scripts\python.exe src\int8_baseline.py --run runs\A_fp32_e60 || exit /b 1
echo === средняя модель (BPE 16k, ~7.3M) ===
if not exist runs_medium mkdir runs_medium
.venv\Scripts\python.exe src\run_experiments.py --config config_medium.yaml --spec experiments_medium.yaml || exit /b 1
echo === квантование эмбеддинга (p=7 и p=4) ===
.venv\Scripts\python.exe src\run_experiments.py --config config_medium.yaml --spec experiments_embed.yaml || exit /b 1
.venv\Scripts\python.exe src\check_embedding.py --config config_medium.yaml || exit /b 1
echo === L1-разреженность (char-модель, p=7 и p=4) ===
if not exist runs_l1 mkdir runs_l1
.venv\Scripts\python.exe src\run_experiments.py --config config_l1.yaml --spec experiments_l1_all.yaml || exit /b 1
.venv\Scripts\python.exe src\l1_report.py --config config_l1.yaml --runs A_fp32_init2 C_p4_init2 B_p4_l1_0 B_p4_l1_1e5 B_p4_l1_2e5 B_p4_l1_3e5 B_p4_l1_5e5 B_p7_l1_0 B_p7_l1_1e5 B_p7_l1_5e5 B_p7_l1_2e4 B_p7_l1_1e3 || exit /b 1
echo готово: runs\comparison.md, runs\report_final.md, runs\report_bits.md, runs\report_sparse.md, runs\report_medium.md
