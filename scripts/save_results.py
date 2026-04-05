"""
scripts/save_results.py — Запуск pytest и сохранение результатов [METRIC] в БД.

Использование:
    python scripts/save_results.py                     # все тесты
    python scripts/save_results.py "TestAblation"      # конкретный класс (-k фильтр)
    python scripts/save_results.py --db-url postgresql://user:pass@host:port/db
"""

import sys
import os
import re
import json
import uuid
import subprocess
from datetime import datetime

# Путь к корню проекта
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base, ExperimentResult


def get_db_url(override=None):
    if override:
        return override
    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    return cfg["webapp"]["database_url"]


def parse_metric_lines(output: str):
    """Разбирает [METRIC] строки из вывода pytest."""
    results = []
    for line in output.splitlines():
        line = line.strip()
        if "[METRIC]" not in line:
            continue

        raw = line
        text = line.split("[METRIC]", 1)[1].strip()

        # ── 1. COMPARE: Dataset | Method | MAE | RMSE | MAPE | COVERAGE ──
        m = re.match(
            r"COMPARE:\s*(.+?)\s*\|\s*(.+?)\s*\|\s*"
            r"MAE=([\d.]+)±([\d.]+)\s*\|\s*"
            r"RMSE=([\d.]+)±([\d.]+)\s*\|\s*"
            r"MAPE=([\d.]+)±([\d.]+)%\s*\|\s*"
            r"COVERAGE=([\d.]+)±([\d.]+)%",
            text,
        )
        if m:
            results.append({
                "experiment": "compare",
                "dataset": m.group(1).strip(),
                "label": m.group(2).strip(),
                "metrics": {
                    "mae": float(m.group(3)), "mae_std": float(m.group(4)),
                    "rmse": float(m.group(5)), "rmse_std": float(m.group(6)),
                    "mape": float(m.group(7)), "mape_std": float(m.group(8)),
                    "coverage": float(m.group(9)), "coverage_std": float(m.group(10)),
                },
                "raw": raw,
            })
            continue

        # ── 2. HORIZON: h=N | MAE ──
        m = re.match(r"HORIZON:\s*h=(\d+)\s*\|\s*MAE=([\d.]+)±([\d.]+)", text)
        if m:
            results.append({
                "experiment": "horizon", "dataset": "Alibaba",
                "label": f"h={m.group(1)}",
                "metrics": {
                    "h": int(m.group(1)),
                    "mae": float(m.group(2)),
                    "mae_std": float(m.group(3)),
                },
                "raw": raw,
            })
            continue

        # ── 3. TIMING ──
        m = re.match(
            r"TIMING:\s*Среднее=([\d.]+)мс\s*±\s*([\d.]+)мс\s*\|\s*Доля от Δt=([\d.]+)%",
            text,
        )
        if m:
            results.append({
                "experiment": "timing", "dataset": None,
                "label": "Полный цикл",
                "metrics": {
                    "mean_ms": float(m.group(1)),
                    "std_ms": float(m.group(2)),
                    "pct_of_dt": float(m.group(3)),
                },
                "raw": raw,
            })
            continue

        # Неразобранная строка
        results.append({
            "experiment": "unknown", "dataset": None,
            "label": None, "metrics": {"raw_text": text}, "raw": raw,
        })

    return results


def run_and_save(test_filter="", db_url=None):
    """Запускает pytest, парсит [METRIC] строки, сохраняет в БД."""
    run_id = uuid.uuid4().hex[:12]
    db_url = get_db_url(db_url)

    # Создаём таблицу если не существует
    engine = create_engine(db_url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    # Формируем команду pytest
    cmd = [sys.executable, "-m", "pytest", "tests/test_experiments.py", "-v", "-s"]
    if test_filter:
        cmd += ["-k", test_filter]

    print(f"[save_results] run_id={run_id}")
    print(f"[save_results] Запуск: {' '.join(cmd)}")
    print(f"[save_results] DB: {db_url}")
    print("=" * 60)

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    # Стриминг вывода — видим прогресс в реальном времени
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    all_output = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        all_output.append(line)

    proc.wait()
    output_text = "\n".join(all_output)

    # Парсим [METRIC] строки
    parsed = parse_metric_lines(output_text)
    print(f"\n[save_results] Разобрано {len(parsed)} строк [METRIC]")

    # Сохраняем в БД
    session = Session()
    try:
        for p in parsed:
            row = ExperimentResult(
                run_id=run_id,
                created_at=datetime.utcnow(),
                experiment=p["experiment"],
                dataset=p.get("dataset"),
                label=p.get("label"),
                metrics_json=json.dumps(p["metrics"], ensure_ascii=False),
                raw_line=p.get("raw"),
            )
            session.add(row)
        session.commit()
        print(f"[save_results] Сохранено {len(parsed)} записей в БД (run_id={run_id})")
    except Exception as e:
        session.rollback()
        print(f"[save_results] ОШИБКА записи в БД: {e}")
        raise
    finally:
        session.close()

    return run_id, len(parsed), proc.returncode or 0


def parse_file_and_save(filepath, db_url=None):
    """Парсит [METRIC] строки из готового файла и сохраняет в БД."""
    run_id = uuid.uuid4().hex[:12]
    db_url = get_db_url(db_url)

    engine = create_engine(db_url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with open(filepath, encoding="utf-8", errors="replace") as f:
        text = f.read()

    parsed = parse_metric_lines(text)
    print(f"[save_results] Разобрано {len(parsed)} строк [METRIC] из {filepath}")

    session = Session()
    try:
        for p in parsed:
            row = ExperimentResult(
                run_id=run_id,
                created_at=datetime.utcnow(),
                experiment=p["experiment"],
                dataset=p.get("dataset"),
                label=p.get("label"),
                metrics_json=json.dumps(p["metrics"], ensure_ascii=False),
                raw_line=p.get("raw"),
            )
            session.add(row)
        session.commit()
        print(f"[save_results] Сохранено {len(parsed)} записей в БД (run_id={run_id})")
    except Exception as e:
        session.rollback()
        print(f"[save_results] ОШИБКА: {e}")
        raise
    finally:
        session.close()

    return run_id, len(parsed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filter", nargs="?", default="", help="pytest -k filter")
    parser.add_argument("--db-url", default=None, help="PostgreSQL URL")
    parser.add_argument("--parse-file", default=None, help="Парсить готовый файл вместо запуска pytest")
    args = parser.parse_args()

    if args.parse_file:
        run_id, count = parse_file_and_save(args.parse_file, args.db_url)
        print(f"\nГотово: run_id={run_id}, записей={count}")
    else:
        run_id, count, rc = run_and_save(args.filter, args.db_url)
        print(f"\nГотово: run_id={run_id}, записей={count}, pytest exit code={rc}")
    sys.exit(0 if count > 0 else 1)
