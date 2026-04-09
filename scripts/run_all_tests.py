"""
scripts/run_all_tests.py — Запуск экспериментов главы 4.

Поддерживает три режима:

  python -u scripts/run_all_tests.py
        Последовательный запуск всех тестов в одном потоке.

  python -u scripts/run_all_tests.py --parallel
        Параллельный запуск в двух subprocess'ах: разделение по группам.
        Группа A: TestCompareAlibaba + TestHorizonDependence
        Группа B: TestCompareAllDatasets + TestComputationTime

  python -u scripts/run_all_tests.py --only-compare
        Только сравнительные тесты (Alibaba + все датасеты).

Опция --append дописывает к существующему файлу результатов.

Каждый subprocess пишет своё собственное логи в результаты:
  results/experiments_output.txt        — общий журнал
  results/experiments_output_A.txt      — только группа A (в --parallel)
  results/experiments_output_B.txt      — только группа B (в --parallel)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import List

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


def _make_env(seed_offset: int = 0) -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_test(test_name: str, outfile, env: dict) -> int:
    """Запускает один pytest-подтест и потоково перекидывает stdout в outfile."""
    bar = "=" * 60
    msg_lines = ["", bar, f"Запуск: {test_name}", bar]
    for line in msg_lines:
        print(line, flush=True)
        outfile.write(line + "\n")
    outfile.flush()

    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "pytest",
         "tests/test_experiments.py", "-v", "-s", "--capture=no",
         "-k", test_name],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        print(line, flush=True)
        outfile.write(line + "\n")
        outfile.flush()
    proc.wait()
    elapsed = time.time() - t0
    summary = f">>> {test_name} — завершён за {elapsed:.0f}с (exit={proc.returncode})"
    print(summary, flush=True)
    outfile.write(summary + "\n")
    outfile.flush()
    return int(proc.returncode or 0)


def run_serial(tests: List[str], out_path: str, append: bool) -> int:
    mode = "a" if append else "w"
    rc_total = 0
    with open(out_path, mode, encoding="utf-8") as f:
        f.write(f"=== Запуск тестов: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.flush()
        for test in tests:
            rc = run_test(test, f, _make_env())
            rc_total |= rc
        f.write(f"\n{'='*40}\n")
        f.write(f"ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*40}\n")
    print(f"\nРезультаты записаны в {out_path}")
    return rc_total


def run_parallel(append: bool) -> int:
    """
    Параллельный запуск в двух subprocess-«воркерах».
    Каждый воркер — это новый запуск этого же скрипта в --serial-group режиме.
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
    # На GPU второй процесс вынужден делить CUDA — torch это поддерживает,
    # но если хочется развести, можно установить CUDA_VISIBLE_DEVICES.
    print("[parallel] launching 2 workers...", flush=True)
    proc_a = subprocess.Popen(
        [sys.executable, "-u", __file__, "--serial-group", "A"]
        + (["--append"] if append else []),
        cwd=ROOT, env=env,
    )
    proc_b = subprocess.Popen(
        [sys.executable, "-u", __file__, "--serial-group", "B"]
        + (["--append"] if append else []),
        cwd=ROOT, env=env,
    )

    rc_a = proc_a.wait()
    rc_b = proc_b.wait()
    print(f"[parallel] worker A exit={rc_a}", flush=True)
    print(f"[parallel] worker B exit={rc_b}", flush=True)

    # Объединяем результаты в общий файл
    with open(OUT_MAIN, "w", encoding="utf-8") as out:
        out.write(f"=== Параллельный запуск: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--append", action="store_true",
                        help="Дописать к файлу результатов, не перезаписывать.")
    parser.add_argument("--only-compare", action="store_true",
                        help="Только сравнительные тесты (без horizon/timing).")
    parser.add_argument("--parallel", action="store_true",
                        help="Запустить тесты в двух параллельных subprocess'ах.")
    parser.add_argument("--serial-group", choices=["A", "B"], default=None,
                        help="(internal) запуск одной группы для --parallel.")
    args = parser.parse_args()

    if args.serial_group is not None:
        # Внутренний воркер для --parallel
        if args.serial_group == "A":
            tests = ["TestCompareAlibaba", "TestHorizonDependence"]
            out_path = os.path.join(RESULTS_DIR, "experiments_output_A.txt")
        else:
            tests = ["TestCompareAllDatasets", "TestComputationTime"]
            out_path = os.path.join(RESULTS_DIR, "experiments_output_B.txt")
        return run_serial(tests, out_path, args.append)

    if args.only_compare:
        tests = COMPARE_TESTS
    else:
        tests = ALL_TESTS

    if args.parallel:
        return run_parallel(args.append)
    return run_serial(tests, OUT_MAIN, args.append)


if __name__ == "__main__":
    sys.exit(main())
