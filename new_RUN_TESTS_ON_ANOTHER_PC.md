# Запуск ML-тестов на другом компьютере

## Что скопировать на второй ноут

Скопируйте всю папку проекта, или минимальный набор:

```
project/
  predictor/          # модуль прогнозирования (обязательно)
  controller/         # модуль принятия решений (обязательно)
  tests/              # тесты и baseline-методы (обязательно)
  scripts/
    run_all_tests.py        # полные тесты (все методы)
    run_proposed_only.py    # только разработанный метод
    run_hp_search.py        # поиск гиперпараметров
    generate_alibaba_synthetic.py
  data/
    alibaba_cluster_trace_2018.csv
    google_cluster_trace_2019.csv
    azure_vm_trace_2019.csv
  config.yaml               # ЕДИНЫЙ файл конфигурации
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
python -c "import torch; print('CUDA OK:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# (если датасета нет) — сгенерировать синтетический Alibaba
python scripts/generate_alibaba_synthetic.py
```

> **БЕЗ GPU тесты идут в десятки раз медленнее (часами)**.
> Если `torch.cuda.is_available()` печатает `False`, переустановите torch с CUDA.

---

## Конфигурация

Все гиперпараметры модели читаются из **единого файла** `config.yaml`.
После поиска лучших гиперпараметров (HP-search) обновите ТОЛЬКО этот файл —
все тесты и скрипты подхватят новые значения автоматически.

---

## Рекомендуемый порядок запуска

### Этап 1: Поиск гиперпараметров (~2-4 часа)

```powershell
venv\Scripts\activate

# Все 12 свипов в 2 потока
python -u scripts/run_hp_search.py --parallel

# Или конкретные свипы
python -u scripts/run_hp_search.py --only HiddenDim Dropout LearningRate

# На конкретном датасете (по умолчанию azure)
python -u scripts/run_hp_search.py --dataset alibaba --parallel

# Посмотреть список свипов
python -u scripts/run_hp_search.py --list
```

Результат: `results/hp_search.txt` — найдите строки `BEST` для каждого параметра.

### Этап 2: Обновить config.yaml

Откройте `config.yaml` и замените значения на лучшие из HP-search.
Например, если HP-search показал `BEST hidden_dim=128`:

```yaml
model:
  hidden_dim: 128   # было 96
```

### Этап 3: Очистить кэш моделей

```powershell
Remove-Item -Recurse -Force models\cache
```

**Важно:** после любого изменения `config.yaml` или кода модели
нужно удалить кэш, иначе будут использоваться старые веса.

### Этап 4: Тесты разработанного метода (~30 мин)

```powershell
# Все 7 датасетов
python -u scripts/run_proposed_only.py

# Один датасет
python -u scripts/run_proposed_only.py --dataset alibaba
python -u scripts/run_proposed_only.py --dataset google
python -u scripts/run_proposed_only.py --dataset trend
```

Результат: `results/proposed_output.txt`

### Этап 5: Полное сравнение со всеми baseline (~3-5 часов)

```powershell
# Все датасеты параллельно (рекомендуется)
python -u scripts/run_all_tests.py --parallel

# Или последовательно
python -u scripts/run_all_tests.py

# Один датасет
python -u scripts/run_all_tests.py --dataset alibaba
python -u scripts/run_all_tests.py --dataset google
```

Результат: `results/experiments_output.txt`

---

## Все доступные датасеты

| Ключ | Датасет | Описание |
|------|---------|----------|
| `alibaba` | Alibaba Cluster Trace 2018 | 2243 точки, ~7.8 суток |
| `google` | Google Cluster Trace 2019 | 8064 точки, ~28 суток |
| `azure` | Azure VM Traces 2019 | 8640 точек, ~30 суток |
| `stationary` | Синтетический стационарный | 4320 точек, 15 суток |
| `trend` | Синтетический трендовый | 4320 точек, 15 суток |
| `spike` | Синтетический всплесковый | 4320 точек, 15 суток |
| `mixed` | Синтетический смешанный | 4320 точек, 15 суток |

---

## Что запускает каждый режим run_all_tests.py

| Флаг | Что прогоняется |
|------|----------------|
| _(без флагов)_ | Все 4 эксперимента последовательно |
| `--parallel` | То же, но в 2 потока |
| `--only-compare` | Только сравнительные тесты (без horizon/timing) |
| `--dataset X` | Только выбранный датасет |
| `--dataset X --only-compare` | Только сравнение на выбранном датасете |

---

## Очистка кэша моделей

Если вы обновили код прогнозатора (`predictor/`), `config.yaml`,
или получили ошибку `version mismatch` — удалите кэш:

```powershell
Remove-Item -Recurse -Force models\cache
```

---

## Забрать результаты обратно

Скопируйте с второго ноута на основной:
- `results/` — все логи экспериментов
- `models/cache/` — кэш обученных моделей (опционально)
- `config.yaml` — если обновили после HP-search
