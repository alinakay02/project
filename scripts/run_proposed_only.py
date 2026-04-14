"""
scripts/run_proposed_only.py — Запуск тестов ТОЛЬКО для разработанного метода.

Прогоняет STL + GRU на выбранных датасетах (без baseline-методов).
Значительно быстрее полного run_all_tests.py.

  python -u scripts/run_proposed_only.py
        Все 7 датасетов последовательно.

  python -u scripts/run_proposed_only.py --dataset alibaba
        Только один датасет. Доступные:
          all, alibaba, google, azure, stationary, trend, spike, mixed

  python -u scripts/run_proposed_only.py --list-datasets
        Показать список датасетов.

Лог: results/proposed_output.txt
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_MAIN = os.path.join(RESULTS_DIR, "proposed_output.txt")

DATASET_CHOICES = [
    "all", "alibaba", "google", "azure",
    "stationary", "trend", "spike", "mixed",
]

DATASET_K = {
    "alibaba":    "test_proposed[Alibaba",
    "google":     "test_proposed[Google",
    "azure":      "test_proposed[Azure",
    "stationary": "test_proposed[Стационарный",
    "trend":      "test_proposed[Трендовый",
    "spike":      "test_proposed[Всплесковый",
    "mixed":      "test_proposed[Смешанный",
}

DATASET_DESC = {
    "all":        "все датасеты (поведение по умолчанию)",
    "alibaba":    "Alibaba Cluster Trace 2018",
    "google":     "Google Cluster Trace",
    "azure":      "Azure VM Traces",
    "stationary": "Синтетический стационарный ряд",
    "trend":      "Синтетический ряд с трендом",
    "spike":      "Синтетический ряд со всплесками",
    "mixed":      "Смешанный синтетический ряд",
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
            "        Активируйте venv:\n"
            "        venv\\Scripts\\activate\n",
            flush=True, file=sys.stderr,
        )
        sys.exit(2)


def _run_pytest(k_filter: Optional[str], label: str, out_path: str,
                append: bool) -> int:
    mode = "a" if append else "w"
    cmd = [
        sys.executable, "-u", "-m", "pytest",
        "tests/test_proposed_only.py", "-v", "-s", "--capture=no",
    ]
    if k_filter:
        cmd += ["-k", k_filter]

    env = _make_env()
    t0 = time.time()

    with open(out_path, mode, encoding="utf-8") as f:
        f.write(
            f"=== Разработанный метод: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            f"Filter: {k_filter or 'all'}\n\n"
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
            f"\n>>> {label} — завершён за {elapsed:.0f}с (exit={proc.returncode})\n"
        )
        print(summary, flush=True)
        f.write(summary)

    print(f"\nРезультаты записаны в {out_path}")
    return int(proc.returncode or 0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Запуск тестов ТОЛЬКО для разработанного метода (STL + GRU).",
    )
    parser.add_argument("--append", action="store_true",
                        help="Дописать к файлу результатов.")
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="all",
        metavar="DATASET",
        help=f"Датасет для прогона. Доступные: {', '.join(DATASET_CHOICES)}.",
    )
    parser.add_argument("--list-datasets", action="store_true",
                        help="Показать список датасетов и выйти.")
    args = parser.parse_args()

    if args.list_datasets:
        print("Available datasets (--dataset):")
        for ds in DATASET_CHOICES:
            print(f"  {ds:12s}  {DATASET_DESC[ds]}")
        return 0

    _check_python_environment()
    print(
        f"[ENV] Python:  {sys.executable}\n"
        f"[ENV] cwd:     {ROOT}\n"
        f"[ENV] dataset: {args.dataset}",
        flush=True,
    )

    if args.dataset == "all":
        return _run_pytest(None, "proposed-all", OUT_MAIN, args.append)

    k_filter = DATASET_K[args.dataset]
    label = f"proposed-{args.dataset}"
    return _run_pytest(k_filter, label, OUT_MAIN, args.append)


if __name__ == "__main__":
    sys.exit(main())
