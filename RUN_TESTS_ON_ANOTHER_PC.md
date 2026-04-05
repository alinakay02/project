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
    alibaba_cluster_trace_2018.csv  # датасет (обязательно)
  config.yaml         # конфигурация
  requirements-tests.txt  # зависимости
```

## Установка (один раз)

```powershell
# 1. Python 3.11+ должен быть установлен
python -m venv venv
venv\Scripts\activate

# 2. Установить зависимости
pip install -r requirements-tests.txt

# 3. Сгенерировать датасет (если нет)
python scripts/generate_alibaba_synthetic.py
```

## Запуск тестов

```powershell
venv\Scripts\activate
python -u scripts/run_all_tests.py
```

Результаты пишутся в `results/experiments_output.txt`.

## Забрать результаты обратно

Скопируйте с второго ноута на основной:
- `results/experiments_output.txt` — результаты тестов
- `models/cache/` — кэш обученных моделей (чтобы не переобучать)

Потом на основном ноуте запустите парсинг в БД:
```powershell
kubectl port-forward service/postgres 5433:5432
python scripts/save_results.py --parse-file results/experiments_output.txt
```
