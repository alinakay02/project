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

### Вариант А: обычная установка (если PyPI доступен)

```powershell
# 1. Python 3.11 (64-bit) должен быть установлен
python -m venv venv
venv\Scripts\activate

# 2. Базовые зависимости
pip install -r requirements-tests.txt

# 3. ОБЯЗАТЕЛЬНО: GPU-сборка PyTorch (см. шаг 3 ниже)
```

### Вариант Б: оффлайн-установка через packages (если PyPI недоступен / таймауты)

Если `pip install` падает с ошибкой `ReadTimeoutError` или
`ConnectionResetError (10054)`, PyPI недоступен с целевого компьютера.
Скачайте пакеты на компьютере, где интернет работает, и перенесите.

**На компьютере с интернетом:**

```powershell
# Скачать все wheel-файлы в папку packages/
pip download -r requirements-tests.txt -d ./packages --python-version 3.11 --platform win_amd64 --only-binary=:all:
```

Перенесите папку `packages/` на целевой компьютер (флешка, облако, архив).

**На целевом компьютере:**

```powershell
python -m venv venv
venv\Scripts\activate

# Установить из локальной папки без обращения к интернету
pip install --no-index --find-links=./packages -r requirements-tests.txt
```

### Шаг 3: GPU-сборка PyTorch (обязательно для обоих вариантов)

```powershell
# Проверьте версию CUDA:
nvidia-smi

# Если CUDA 12.x:
pip install --force-reinstall torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
# Если CUDA 11.8:
pip install --force-reinstall torch==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# При оффлайн-установке: скачайте torch wheel отдельно на компьютере с интернетом:
#   pip download "torch==2.1.2+cu121" -d ./packages --python-version 3.11 --platform win_amd64 --only-binary=:all: --index-url https://download.pytorch.org/whl/cu121
# И установите локально:
#   pip install --no-index --find-links=./packages torch
```

### Шаг 4: проверка

```powershell
# Проверка, что torch видит GPU
python -c "import torch; print('CUDA OK:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '—')"

# (если датасета нет) — сгенерировать синтетический Alibaba
python scripts/generate_alibaba_synthetic.py
```

⚠️ **БЕЗ GPU тесты идут в десятки раз медленнее (часами)**.
Если `python -c "import torch; print(torch.cuda.is_available())"` печатает `False`,
тесты всё равно запустятся, но НЕ оставляйте их так — переустановите torch с CUDA.
В тестовом скрипте есть проверка, которая выведет предупреждение `[ENV][WARN] CUDA НЕ ДОСТУПНА...`.

## Запуск тестов

```powershell
venv\Scripts\activate
```

### Все датасеты

```powershell
# Последовательный запуск (все 4 эксперимента по очереди)
python -u scripts/run_all_tests.py

# Параллельный запуск в 2 потока (рекомендуется — быстрее)
python -u scripts/run_all_tests.py --parallel

# Только сравнительные тесты (без horizon/timing)
python -u scripts/run_all_tests.py --only-compare
```

### Один конкретный датасет

```powershell
python -u scripts/run_all_tests.py --dataset alibaba      # Alibaba (Compare + Horizon + Timing)
python -u scripts/run_all_tests.py --dataset google        # Google Cluster Trace
python -u scripts/run_all_tests.py --dataset azure         # Azure VM Traces
python -u scripts/run_all_tests.py --dataset stationary    # Синтетический стационарный ряд
python -u scripts/run_all_tests.py --dataset trend         # Синтетический ряд с трендом
python -u scripts/run_all_tests.py --dataset spike         # Синтетический ряд со всплесками
python -u scripts/run_all_tests.py --dataset mixed         # Смешанный синтетический ряд
```

### Комбинации

```powershell
# Только сравнение методов на конкретном датасете (без horizon/timing)
python -u scripts/run_all_tests.py --dataset alibaba --only-compare

# Дописать результаты к существующему файлу (не перезаписывать)
python -u scripts/run_all_tests.py --dataset google --append

# Посмотреть список доступных датасетов
python -u scripts/run_all_tests.py --list-datasets
```

### Что запускает каждый режим

| Флаг | Что прогоняется |
|------|----------------|
| _(без флагов)_ | Все 4 эксперимента последовательно: Compare Alibaba → Compare All Datasets → Horizon → Timing |
| `--parallel` | То же самое, но в 2 потока: **A** = Alibaba + Google + Azure + Стационарный + Horizon; **B** = Трендовый + Всплесковый + Смешанный + Timing |
| `--only-compare` | Только сравнительные тесты (Compare Alibaba + Compare All Datasets) |
| `--dataset X` | Только выбранный датасет (для alibaba — включая Horizon и Timing) |
| `--dataset X --only-compare` | Только сравнение методов на выбранном датасете |

### Результаты

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
