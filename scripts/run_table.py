"""
scripts/run_table.py — Универсальный runner для таблиц 4.7-4.11.

Запускает соответствующий pytest-файл с потоковой записью лога
в results/table_X_Y.txt.

  python -u scripts/run_table.py 4.7
  python -u scripts/run_table.py 4.8
  python -u scripts/run_table.py 4.9
  python -u scripts/run_table.py 4.10
  python -u scripts/run_table.py 4.11

Опции:
  --append   дописать к существующему файлу результата
  --list     показать все доступные таблицы и выйти
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Таблица → (pytest-файл, выходной файл, описание)
TABLES = {
    "4.7": (
        "tests/test_table_4_7.py",
        "results/table_4_7.txt",
        "Зависимость MAE от горизонта (разработанный, GRU, SARIMA)",
    ),
    "4.8": (
        "tests/test_table_4_8.py",
        "results/table_4_8.txt",
        "Влияние дообучения (Azure, 8 суток тестирования)",
    ),
    "4.9": (
        "tests/test_table_4_9.py",
        "results/table_4_9.txt",
        "Влияние календарных признаков (Стационарный синт., d_step=8 vs 2)",
    ),
    "4.10": (
        "tests/test_table_4_10.py",
        "results/table_4_10.txt",
        "Анализ абляции компонентов (Alibaba, 5 конфигураций)",
    ),
    "4.11": (
        "tests/test_table_4_11.py",
        "results/table_4_11.txt",
        "Вычислительное время по модулям (Alibaba, 50 итераций)",
    ),
}


def _make_env() -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _check_python_environment() -> None:
    try:
        import pytest  # noqa: F401
    except ImportError:
        print(
            f"\n[FATAL] pytest не найден в Python: {sys.executable}\n"
            "        Активируйте venv: venv\\Scripts\\activate\n",
            flush=True, file=sys.stderr,
        )
        sys.exit(2)


def _run_one(table: str, test_file: str, out_file: str, desc: str,
             append: bool) -> int:
    mode = "a" if append else "w"
    out_path = os.path.join(ROOT, out_file)
    label = f"Таблица {table}"

    print(
        f"\n[{time.strftime('%H:%M:%S')}] [runner] Запуск: {label} — {desc}\n"
        f"[runner] pytest-файл: {test_file}\n"
        f"[runner] выходной файл: {out_file}",
        flush=True,
    )

    cmd = [
        sys.executable, "-u", "-m", "pytest",
        test_file, "-v", "-s", "--capture=no",
    ]
    env = _make_env()
    t0 = time.time()

    with open(out_path, mode, encoding="utf-8") as f:
        f.write(
            f"=== {label} — {desc} ===\n"
            f"=== Запуск: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n"
        )
        f.flush()

        proc = subprocess.Popen(
            cmd, cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, bufsize=1,
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            print(line, flush=True)
            f.write(line + "\n")
            f.flush()
        proc.wait()

        elapsed = time.time() - t0
        summary = (
            f"\n>>> {label} завершена за {elapsed:.0f}с "
            f"(exit={proc.returncode})\n"
        )
        print(summary, flush=True)
        f.write(summary)

    print(f"[runner] Результаты записаны в {out_file}", flush=True)
    return int(proc.returncode or 0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Запуск отдельных таблиц главы 4 (4.7-4.11).",
    )
    parser.add_argument(
        "table", nargs="?", choices=list(TABLES.keys()) + ["all"],
        help="Номер таблицы или 'all' для последовательного запуска всех.",
    )
    parser.add_argument("--append", action="store_true",
                        help="Дописать к существующему файлу.")
    parser.add_argument("--list", action="store_true",
                        help="Показать список доступных таблиц.")
    args = parser.parse_args()

    if args.list or not args.table:
        print("Доступные таблицы:")
        for k, (tf, of, desc) in TABLES.items():
            print(f"  {k:5s}  {desc}")
            print(f"         pytest: {tf}")
            print(f"         output: {of}")
        if args.list:
            return 0
        if not args.table:
            return 0

    _check_python_environment()
    print(
        f"[ENV] Python:  {sys.executable}\n"
        f"[ENV] cwd:     {ROOT}",
        flush=True,
    )

    if args.table == "all":
        rc = 0
        for k in TABLES.keys():
            test_file, out_file, desc = TABLES[k]
            rc_one = _run_one(k, test_file, out_file, desc, args.append)
            rc |= rc_one
        return rc

    test_file, out_file, desc = TABLES[args.table]
    return _run_one(args.table, test_file, out_file, desc, args.append)


if __name__ == "__main__":
    sys.exit(main())
