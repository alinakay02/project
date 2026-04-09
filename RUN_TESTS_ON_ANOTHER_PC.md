# Запуск ML-тестов на другом компьютере

## Что скопировать на второй ноут

Скопируйте всю папку проекта, или минимальный набор:

```
project/
  predictor/          # модуль прогнозирования (обязательно)
  controller/         # модуль принятия решений (обязательно)
  tests/              # тесты и baseline-методы (обязательно)
  scripts/
    run_all_tests.py  # скрипт запуска
    generate_alibaba_synthetic.py
  data/
    alibaba_cluster_trace_2018.csv
    google_cluster_trace_2019.csv
    azure_vm_trace_2019.csv
  config.yaml
  requirements-tests.txt
```

## Установка (один раз)

```powershell
# 1. Python 3.11 должен быть установлен
python -m venv venv
venv\Scripts\activate

# 2. Базовые зависимости (без torch — torch ставится отдельно под GPU)
pip install -r requirements-tests.txt

# 3. ОБЯЗАТЕЛЬНО: GPU-сборка PyTorch
#    Сначала проверьте версию CUDA в системе:
nvidia-smi
#    Если CUDA 12.x:
pip install --force-reinstall torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
#    Если CUDA 11.8:
pip install --force-reinstall torch==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# 4. Проверка, что torch видит GPU
python -c "import torch; print('CUDA OK:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '—')"

# 5. (если датасета нет) — сгенерировать синтетический Alibaba
python scripts/generate_alibaba_synthetic.py
```

⚠️ **БЕЗ GPU тесты идут в десятки раз медленнее (часами)**.
Если `python -c "import torch; print(torch.cuda.is_available())"` печатает `False`,
тесты всё равно запустятся, но НЕ оставляйте их так — переустановите torch с CUDA.
В тестовом скрипте есть проверка, которая выведет предупреждение `[ENV][WARN] CUDA НЕ ДОСТУПНА...`.

## Запуск тестов

```powershell
venv\Scripts\activate

# Вариант 1: Последовательный запуск (как раньше)
python -u scripts/run_all_tests.py

# Вариант 2: Параллельный запуск в 2 subprocess'ах (рекомендуется)
#   Группа A: TestCompareAlibaba + TestHorizonDependence
#   Группа B: TestCompareAllDatasets + TestComputationTime
python -u scripts/run_all_tests.py --parallel

# Вариант 3: Только сравнительные тесты (Alibaba + остальные датасеты)
python -u scripts/run_all_tests.py --only-compare
```

Результаты пишутся в:
- `results/experiments_output.txt`         — общий лог
- `results/experiments_output_A.txt`       — группа A (только при `--parallel`)
- `results/experiments_output_B.txt`       — группа B (только при `--parallel`)

## Очистка кэша моделей

Если вы обновили код прогнозатора (изменился `predictor/`), удалите старые
кэшированные веса — они автоматически будут отброшены по версии модели, но
лучше удалить вручную:

```powershell
Remove-Item -Recurse -Force models\cache
```

## Забрать результаты обратно

Скопируйте с второго ноута на основной:
- `results/experiments_output.txt` (или `_A.txt` / `_B.txt` для парного режима)
- `models/cache/` — кэш обученных моделей (опционально, чтобы не переобучать)

Потом на основном ноуте запустите парсинг в БД:
```powershell
kubectl port-forward service/postgres 5433:5432
python scripts/save_results.py --parse-file results/experiments_output.txt
```
