"""
scripts/run_hp_search.py — Запуск гиперпараметрических свипов.

По умолчанию запускает все 12 HP-тестов из tests/test_hp_search.py
последовательно, по одному pytest-вызову на каждый. Каждый свип пишет
свой блок в общий лог.

  python -u scripts/run_hp_search.py
        Все 12 свипов последовательно.

  python -u scripts/run_hp_search.py --only HiddenDim Dropout
        Только указанные свипы (по короткому имени теста).

  python -u scripts/run_hp_search.py --parallel
        Два параллельных воркера, делят свипы пополам.

Опция --append дописывает к существующему файлу.

Лог: results/hp_search.txt
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
OUT_MAIN = os.path.join(RESULTS_DIR, "hp_search.txt")

# Имя класса → короткое имя для CLI
ALL_HP_TESTS = [
    ("HiddenDim",     "TestHP_HiddenDim"),
    ("Dropout",       "TestHP_Dropout"),
    ("WInput",        "TestHP_WInput"),
    ("LearningRate",  "TestHP_LearningRate"),
    ("BatchSize",     "TestHP_BatchSize"),
    ("MaxEpochs",     "TestHP_MaxEpochs"),
    ("Patience",      "TestHP_Patience"),
    ("GradClip",      "TestHP_GradClip"),
    ("NTrend",        "TestHP_NTrend"),
    ("WNorm",         "TestHP_WNorm"),
    ("IqrAlpha",      "TestHP_IqrAlpha"),
    ("WAnomaly",      "TestHP_WAnomaly"),
]


def _make_env(dataset: Optional[str] = None) -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if dataset:
        env["HP_DATASET"] = dataset
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
            "        python -u scripts/run_hp_search.py --parallel\n\n"
            "        Проверка: python -c \"import sys; print(sys.executable)\"\n"
            "        должно показать ...\\venv\\Scripts\\python.exe\n"
        )
        print(msg, flush=True, file=sys.stderr)
        sys.exit(2)
    # Также проверим, что наш проектный модуль импортируется — иначе
    # subprocess'ы упадут позже в неочевидном месте.
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


def _stream_pytest(
    pytest_filter: str,
    label: str,
    out_file,
    line_prefix: str = "",
    dataset: Optional[str] = None,
) -> int:
    """Запускает pytest -k <filter> и стримит stdout в файл и в консоль."""
    bar = "=" * 60
    for line in ("", bar, f"[{time.strftime('%H:%M:%S')}] {label}", bar):
        print(line_prefix + line, flush=True)
        out_file.write(line + "\n")
    out_file.flush()

    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "pytest",
         "tests/test_hp_search.py", "-v", "-s", "--capture=no",
         "-k", pytest_filter],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=_make_env(dataset=dataset),
        bufsize=1,
    )
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        print(line_prefix + line, flush=True)
        out_file.write(line + "\n")
        out_file.flush()
    proc.wait()
    elapsed = time.time() - t0
    summary = (
        f">>> {label} — завершён за {elapsed:.0f}с (exit={proc.returncode})"
    )
    print(line_prefix + summary, flush=True)
    out_file.write(summary + "\n")
    out_file.flush()
    return int(proc.returncode or 0)


def run_serial(tests: List[tuple], out_path: str, append: bool,
               line_prefix: str = "", dataset: Optional[str] = None) -> int:
    """Запускает каждый свип отдельным pytest-процессом."""
    mode = "a" if append else "w"
    rc_total = 0
    with open(out_path, mode, encoding="utf-8") as f:
        f.write(
            f"=== HP search: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            f"Dataset: {dataset or 'azure'} (default if None)\n"
            f"Тесты: {[short for short, _ in tests]}\n\n"
        )
        f.flush()
        for short, cls_name in tests:
            rc = _stream_pytest(
                cls_name, short, f, line_prefix=line_prefix, dataset=dataset,
            )
            rc_total |= rc
        f.write(
            f"\n{'='*40}\n"
            f"ВСЕ HP-СВИПЫ ЗАВЕРШЕНЫ: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'='*40}\n"
        )
    print(line_prefix + f"\nРезультаты записаны в {out_path}")
    return rc_total


def run_parallel(tests: List[tuple], append: bool,
                 dataset: Optional[str] = None) -> int:
    """Делит свипы пополам и запускает в 2 потока."""
    out_a = os.path.join(RESULTS_DIR, "hp_search_A.txt")
    out_b = os.path.join(RESULTS_DIR, "hp_search_B.txt")
    if not append:
        for p in (out_a, out_b, OUT_MAIN):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

    half = (len(tests) + 1) // 2
    group_a = tests[:half]
    group_b = tests[half:]

    print(
        f"[{time.strftime('%H:%M:%S')}] [parallel HP] запускаю 2 воркера "
        f"(dataset={dataset or 'azure'})",
        flush=True,
    )
    print(f"  [A] {[s for s, _ in group_a]}", flush=True)
    print(f"  [B] {[s for s, _ in group_b]}", flush=True)

    rc_holder = {"A": 0, "B": 0}

    def worker(name: str, tests_for_worker: List[tuple], out_path: str):
        rc_holder[name] = run_serial(
            tests_for_worker, out_path, append,
            line_prefix=f"[{name}] ", dataset=dataset,
        )

    t_a = threading.Thread(target=worker, args=("A", group_a, out_a), daemon=False)
    t_b = threading.Thread(target=worker, args=("B", group_b, out_b), daemon=False)
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()

    rc_a = rc_holder["A"]
    rc_b = rc_holder["B"]
    print(
        f"[{time.strftime('%H:%M:%S')}] [parallel HP] worker A exit={rc_a}, "
        f"worker B exit={rc_b}",
        flush=True,
    )

    # Объединяем
    with open(OUT_MAIN, "w", encoding="utf-8") as out:
        out.write(
            f"=== Параллельный HP search: "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--append", action="store_true",
                        help="Дописать к существующему лог-файлу.")
    parser.add_argument("--parallel", action="store_true",
                        help="Два параллельных воркера, делят свипы пополам.")
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="Запустить только указанные свипы по короткому имени, "
             "например: --only HiddenDim Dropout WInput",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Показать список доступных свипов и выйти.",
    )
    parser.add_argument(
        "--dataset", choices=["alibaba", "google", "azure"], default="azure",
        help="На каком датасете крутить HP-поиск. По умолчанию — azure "
             "(самый длинный, ≈30 суток, наиболее стабильный val-сигнал). "
             "alibaba — быстрее всех (~7.8 суток), но шумнее.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available HP sweeps:")
        for short, cls in ALL_HP_TESTS:
            print(f"  {short:15s}  -> {cls}")
        return 0

    _check_python_environment()
    print(
        f"[ENV] Python:  {sys.executable}\n"
        f"[ENV] cwd:     {ROOT}\n"
        f"[ENV] dataset: {args.dataset}",
        flush=True,
    )

    if args.only is not None:
        wanted = {n.lower() for n in args.only}
        tests = [t for t in ALL_HP_TESTS if t[0].lower() in wanted]
        unknown = wanted - {t[0].lower() for t in ALL_HP_TESTS}
        if unknown:
            print(f"[error] неизвестные свипы: {sorted(unknown)}")
            print("Доступные:")
            for short, _ in ALL_HP_TESTS:
                print(f"  {short}")
            return 2
        if not tests:
            print("[error] список свипов пуст после фильтра.")
            return 2
    else:
        tests = list(ALL_HP_TESTS)

    if args.parallel:
        return run_parallel(tests, args.append, dataset=args.dataset)
    return run_serial(tests, OUT_MAIN, args.append, dataset=args.dataset)


if __name__ == "__main__":
    sys.exit(main())
