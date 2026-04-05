"""
scripts/run_all_tests.py — Запускает тесты по одному, пишет результаты в файл.
Каждый тест запускается отдельным subprocess, вывод сразу записывается.

Запуск:
    python -u scripts/run_all_tests.py              # все тесты
    python -u scripts/run_all_tests.py --append     # дописать к существующему файлу
    python -u scripts/run_all_tests.py --only-compare  # только сравнительные тесты
"""
import subprocess, sys, os, time, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--append", action="store_true", help="Дописать к файлу, не перезаписывать")
parser.add_argument("--only-compare", action="store_true", help="Только сравнительные тесты (базовые методы)")
args = parser.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "experiments_output.txt")
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

TESTS = [
    "TestCompareAlibaba",       # 9 методов на Alibaba (основной)
    "TestCompareAllDatasets",    # 9 методов на Google, Azure, синтетических
    "TestHorizonDependence",    # MAE при h=1,2,3,4,6
    "TestComputationTime",      # Время итерации
]

env = dict(os.environ)
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUTF8"] = "1"
env["PYTHONUNBUFFERED"] = "1"

COMPARE_TESTS = [
    "TestCompareAlibaba",
    "TestCompareAllDatasets",
]

if args.only_compare:
    TESTS = COMPARE_TESTS

outfile = open(OUT, "a" if args.append else "w", encoding="utf-8")

def log(msg):
    print(msg, flush=True)
    outfile.write(msg + "\n")
    outfile.flush()

log(f"=== Запуск тестов: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

for i, test in enumerate(TESTS, 1):
    log(f"\n{'='*60}")
    log(f"[{i}/{len(TESTS)}] Запуск: {test}")
    log(f"{'='*60}")

    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "pytest",
         "tests/test_experiments.py", "-v", "-s", "--capture=no",
         "-k", test],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )

    # Читаем побайтово для немедленного вывода
    for raw_line in proc.stdout:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")
        print(line, flush=True)
        outfile.write(line + "\n")
        outfile.flush()

    proc.wait()
    elapsed = time.time() - t0
    log(f"\n>>> [{i}/{len(TESTS)}] {test} — завершён за {elapsed:.0f}с (exit={proc.returncode})")

log(f"\n{'='*40}")
log(f"ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"{'='*40}")
outfile.close()

print(f"\nРезультаты записаны в {OUT}")
