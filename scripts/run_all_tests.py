"""
scripts/run_all_tests.py — Запуск экспериментов главы 4.

Поддерживает три режима:

  python -u scripts/run_all_tests.py
        Последовательный запуск всех тестов в одном потоке.

  python -u scripts/run_all_tests.py --parallel
        Параллельный запуск в двух subprocess'ах. Тесты делятся примерно
        поровну по объёму работы (см. PARALLEL_GROUP_A / PARALLEL_GROUP_B
        ниже). Логи обоих процессов транслируются в консоль с префиксом
        [A] / [B], плюс пишутся в свои файлы.

  python -u scripts/run_all_tests.py --only-compare
        Только сравнительные тесты (Alibaba + все датасеты).

  python -u scripts/run_all_tests.py --dataset google
        Только тесты на выбранном датасете. Доступные значения:
          all        — все датасеты (поведение по умолчанию)
          alibaba    — TestCompareAlibaba + TestHorizonDependence + TestComputationTime
          google     — TestCompareAllDatasets[Google]
          azure      — TestCompareAllDatasets[Azure]
          stationary — TestCompareAllDatasets[Стационарный]
          trend      — TestCompareAllDatasets[Трендовый]
          spike      — TestCompareAllDatasets[Всплесковый]
          mixed      — TestCompareAllDatasets[Смешанный]

  python -u scripts/run_all_tests.py --dataset alibaba --only-compare
        Комбинирование: только Alibaba-сравнение (без horizon/timing).

Опция --append дописывает к существующему файлу результатов.

Логи:
  results/experiments_output.txt        — общий журнал
  results/experiments_output_A.txt      — только группа A (в --parallel)
  results/experiments_output_B.txt      — только группа B (в --parallel)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from typing import List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_MAIN = os.path.join(RESULTS_DIR, "experiments_output.txt")

ALL_TESTS = [
    "TestCompareAlibaba",       # 9 методов на Alibaba
    "TestCompareAllDatasets",    # 9 методов на Google, Azure, синтетических
    "TestHorizonDependence",    # MAE при h=1,2,3,4,6
    "TestComputationTime",       # время одной итерации
]
COMPARE_TESTS = [
    "TestCompareAlibaba",
    "TestCompareAllDatasets",
]

# Pytest-фильтры для двух параллельных групп.
# Каждая строка передаётся в `pytest -k <expr>` как ИЛИ-выражение.
# Балансировка: TestCompareAllDatasets — самый тяжёлый (6 датасетов × 9 методов
# × 3 seeds), поэтому делим его пополам по датасетам через `-k`.
PARALLEL_GROUP_A = (
    "TestCompareAlibaba "
    "or TestHorizonDependence "
    "or test_all_methods[Google "
    "or test_all_methods[Azure "
    "or test_all_methods[Стационарный"
)
PARALLEL_GROUP_B = (
    "TestComputationTime "
    "or test_all_methods[Трендовый "
    "or test_all_methods[Всплесковый "
    "or test_all_methods[Смешанный"
)

# ──────────────────────────────────────────────────────────────────────────────
# Выбор датасета через --dataset.
#
# Ключ → pytest -k выражение для полного прогона (все тесты датасета).
# ──────────────────────────────────────────────────────────────────────────────
DATASET_CHOICES = [
    "all", "alibaba", "google", "azure",
    "stationary", "trend", "spike", "mixed",
]

# Полный прогон для датасета (включая horizon и timing для Alibaba).
DATASET_K_FULL: dict = {
    "alibaba":    (
        "TestCompareAlibaba or TestHorizonDependence or TestComputationTime"
    ),
    "google":     "test_all_methods[Google",
    "azure":      "test_all_methods[Azure",
    "stationary": "test_all_methods[Стационарный",
    "trend":      "test_all_methods[Трендовый",
    "spike":      "test_all_methods[Всплесковый",
    "mixed":      "test_all_methods[Смешанный",
}

# Только сравнительные тесты (--only-compare совместно с --dataset).
DATASET_K_COMPARE: dict = {
    "alibaba":    "TestCompareAlibaba",
    "google":     "test_all_methods[Google",
    "azure":      "test_all_methods[Azure",
    "stationary": "test_all_methods[Стационарный",
    "trend":      "test_all_methods[Трендовый",
    "spike":      "test_all_methods[Всплесковый",
    "mixed":      "test_all_methods[Смешанный",
}


def _make_env(seed_offset: int = 0) -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _check_python_environment() -> None:
    """
    Проверяет, что pytest доступен в текущем sys.executable, иначе падает с
    понятным сообщением и подсказкой активировать venv.
    """
    try:
        import pytest  # noqa: F401
    except ImportError:
        py = sys.executable
        msg = (
            f"\n[FATAL] pytest не найден в Python: {py}\n"
            "        Скорее всего venv не активирован. Сделайте:\n\n"
            "        cd C:\\diplom\\project\n"
            "        venv\\Scripts\\activate\n"
            "        python -u scripts/run_all_tests.py --parallel\n\n"
            "        Проверка: python -c \"import sys; print(sys.executable)\"\n"
            "        должно показать ...\\venv\\Scripts\\python.exe\n"
        )
        print(msg, flush=True, file=sys.stderr)
        sys.exit(2)
    try:
        sys.path.insert(0, ROOT)
        import predictor.forecaster  # noqa: F401
        import tests.baselines  # noqa: F401
    except Exception as e:
        print(
            f"\n[FATAL] не могу импортировать project-модули: "
            f"{type(e).__name__}: {e}\n"
            f"        Проверьте зависимости: pip install -r requirements-tests.txt\n",
            flush=True, file=sys.stderr,
        )
        sys.exit(2)


def run_test(
    pytest_filter: str,
    label: str,
    outfile,
    env: dict,
    line_prefix: str = "",
) -> int:
    """
    Запускает pytest с фильтром `-k <pytest_filter>` и потоково перекидывает
    stdout в outfile + в консоль (с опциональным line_prefix, например "[A] ").
    """
    bar = "=" * 60
    header_lines = ["", bar, f"[{time.strftime('%H:%M:%S')}] Запуск: {label}", bar]
    for line in header_lines:
        print(line_prefix + line, flush=True)
        outfile.write(line + "\n")
    outfile.flush()

    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "pytest",
         "tests/test_experiments.py", "-v", "-s", "--capture=no",
         "-k", pytest_filter],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=1,
    )
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        print(line_prefix + line, flush=True)
        outfile.write(line + "\n")
        outfile.flush()
    proc.wait()
    elapsed = time.time() - t0
    summary = (
        f">>> {label} — завершён за {elapsed:.0f}с (exit={proc.returncode})"
    )
    print(line_prefix + summary, flush=True)
    outfile.write(summary + "\n")
    outfile.flush()
    return int(proc.returncode or 0)


def run_serial(
    tests: List[str],
    out_path: str,
    append: bool,
    line_prefix: str = "",
) -> int:
    """Запускает каждый класс тестов отдельным pytest-вызовом."""
    mode = "a" if append else "w"
    rc_total = 0
    with open(out_path, mode, encoding="utf-8") as f:
        f.write(f"=== Запуск тестов: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.flush()
        for test in tests:
            rc = run_test(test, test, f, _make_env(), line_prefix=line_prefix)
            rc_total |= rc
        f.write(f"\n{'='*40}\n")
        f.write(f"ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*40}\n")
    print(line_prefix + f"\nРезультаты записаны в {out_path}")
    return rc_total


def run_filtered(
    k_filter: str,
    label: str,
    out_path: str,
    append: bool,
    line_prefix: str = "",
) -> int:
    """Запускает один pytest с произвольным -k фильтром (для --dataset)."""
    mode = "a" if append else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        f.write(
            f"=== Запуск тестов: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            f"Dataset filter: {k_filter}\n\n"
        )
        f.flush()
        rc = run_test(k_filter, label, f, _make_env(), line_prefix=line_prefix)
        f.write(
            f"\n{'='*40}\n"
            f"ТЕСТЫ ЗАВЕРШЕНЫ: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'='*40}\n"
        )
    print(line_prefix + f"\nРезультаты записаны в {out_path}")
    return rc


def _stream_subprocess(
    cmd: List[str],
    env: dict,
    out_path: str,
    line_prefix: str,
    append: bool = False,
) -> int:
    """
    Запускает subprocess, читает stdout построчно, дублирует в файл и в консоль
    с префиксом. Возвращает exit code.
    """
    mode = "a" if append else "w"
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=1,
    )
    assert proc.stdout is not None
    with open(out_path, mode, encoding="utf-8") as f:
        f.write(f"=== {line_prefix}запуск: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.flush()
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            print(line_prefix + line, flush=True)
            f.write(line + "\n")
            f.flush()
    proc.wait()
    return int(proc.returncode or 0)


def run_parallel(append: bool) -> int:
    """
    Параллельный запуск в двух subprocess-pytest'ах.
    Каждая группа = один pytest-процесс с -k фильтром, чтобы не плодить
    лишние pytest-сессии и кэши.
    """
    out_a = os.path.join(RESULTS_DIR, "experiments_output_A.txt")
    out_b = os.path.join(RESULTS_DIR, "experiments_output_B.txt")
    if not append:
        for p in (out_a, out_b, OUT_MAIN):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

    env = _make_env()
    cmd_a = [
        sys.executable, "-u", "-m", "pytest",
        "tests/test_experiments.py", "-v", "-s", "--capture=no",
        "-k", PARALLEL_GROUP_A,
    ]
    cmd_b = [
        sys.executable, "-u", "-m", "pytest",
        "tests/test_experiments.py", "-v", "-s", "--capture=no",
        "-k", PARALLEL_GROUP_B,
    ]

    print(f"[{time.strftime('%H:%M:%S')}] [parallel] запускаю 2 воркера", flush=True)
    print(f"  [A] {PARALLEL_GROUP_A}", flush=True)
    print(f"  [B] {PARALLEL_GROUP_B}", flush=True)

    rc_holder = {"A": None, "B": None}

    def worker(name: str, cmd: List[str], out_path: str):
        rc_holder[name] = _stream_subprocess(
            cmd, env, out_path, line_prefix=f"[{name}] ", append=append,
        )

    t_a = threading.Thread(target=worker, args=("A", cmd_a, out_a), daemon=False)
    t_b = threading.Thread(target=worker, args=("B", cmd_b, out_b), daemon=False)
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()

    rc_a = rc_holder["A"] or 0
    rc_b = rc_holder["B"] or 0
    print(f"[{time.strftime('%H:%M:%S')}] [parallel] worker A exit={rc_a}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] [parallel] worker B exit={rc_b}", flush=True)

    # Объединяем результаты в общий файл
    with open(OUT_MAIN, "w", encoding="utf-8") as out:
        out.write(
            f"=== Параллельный запуск: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n"
        )
        for label, path in (("A", out_a), ("B", out_b)):
            out.write(f"\n{'#'*60}\n# Группа {label} — {path}\n{'#'*60}\n")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    out.write(f.read())
            except FileNotFoundError:
                out.write(f"<файл {path} не найден>\n")

    print(f"\nСводный лог: {OUT_MAIN}")
    return rc_a | rc_b


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Запуск экспериментов главы 4 (сравнение методов прогнозирования).",
    )
    parser.add_argument("--append", action="store_true",
                        help="Дописать к файлу результатов, не перезаписывать.")
    parser.add_argument("--only-compare", action="store_true",
                        help="Только сравнительные тесты (без horizon/timing).")
    parser.add_argument("--parallel", action="store_true",
                        help="Запустить тесты в двух параллельных subprocess'ах.")
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="all",
        metavar="DATASET",
        help=(
            "Выбрать датасет для прогона. "
            f"Доступные: {', '.join(DATASET_CHOICES)}. "
            "По умолчанию — all (все датасеты). "
            "При указании конкретного датасета --parallel игнорируется."
        ),
    )
    parser.add_argument(
        "--list-datasets", action="store_true",
        help="Показать список доступных датасетов и выйти.",
    )
    args = parser.parse_args()

    if args.list_datasets:
        print("Available datasets (--dataset):")
        descriptions = {
            "all":        "все датасеты (поведение по умолчанию)",
            "alibaba":    "Alibaba Cluster Trace 2018 (Compare + Horizon + Timing)",
            "google":     "Google Cluster Trace (TestCompareAllDatasets[Google])",
            "azure":      "Azure VM Traces (TestCompareAllDatasets[Azure])",
            "stationary": "Синтетический стационарный ряд",
            "trend":      "Синтетический ряд с трендом",
            "spike":      "Синтетический ряд со всплесками",
            "mixed":      "Смешанный синтетический ряд",
        }
        for ds in DATASET_CHOICES:
            print(f"  {ds:12s}  {descriptions[ds]}")
        return 0

    _check_python_environment()
    print(
        f"[ENV] Python:  {sys.executable}\n"
        f"[ENV] cwd:     {ROOT}\n"
        f"[ENV] dataset: {args.dataset}",
        flush=True,
    )

    # Прогон конкретного датасета — однопоточно, с -k фильтром.
    if args.dataset != "all":
        if args.parallel:
            print(
                "[WARN] --parallel игнорируется при --dataset "
                "(один датасет не делится на два воркера).",
                flush=True,
            )
        k_map = DATASET_K_COMPARE if args.only_compare else DATASET_K_FULL
        k_filter = k_map[args.dataset]
        label = f"dataset={args.dataset}"
        print(f"[INFO] pytest -k '{k_filter}'", flush=True)
        return run_filtered(k_filter, label, OUT_MAIN, args.append)

    # Обычный прогон всех датасетов.
    if args.parallel:
        return run_parallel(args.append)

    tests = COMPARE_TESTS if args.only_compare else ALL_TESTS
    return run_serial(tests, OUT_MAIN, args.append)


if __name__ == "__main__":
    sys.exit(main())
