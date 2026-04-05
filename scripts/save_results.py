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

        # Убираем префикс [METRIC]
        raw = line
        text = line.split("[METRIC]", 1)[1].strip()

        # ── 1. Proposed MAE / AutoGRU MAE (Alibaba comparison) ──
        m = re.match(r"Proposed MAE:\s*([\d.]+)\s*±\s*([\d.]+)", text)
        if m:
            results.append({
                "experiment": "compare", "dataset": "Alibaba",
                "label": "Разработанный метод",
                "metrics": {"mae": float(m.group(1)), "mae_std": float(m.group(2))},
                "raw": raw,
            })
            continue

        m = re.match(r"AutoGRU MAE:\s*([\d.]+)\s*±\s*([\d.]+)", text)
        if m:
            results.append({
                "experiment": "compare", "dataset": "Alibaba",
                "label": "Автономная GRU",
                "metrics": {"mae": float(m.group(1)), "mae_std": float(m.group(2))},
                "raw": raw,
            })
            continue

        # ── 2. Coverage (Alibaba) ──
        m = re.match(r"ТЕСТ: Alibaba Coverage \| mean=([\d.]+)%", text)
        if m:
            results.append({
                "experiment": "compare", "dataset": "Alibaba",
                "label": "Разработанный метод",
                "metrics": {"coverage": float(m.group(1))},
                "raw": raw,
            })
            continue

        # ── 3. Dataset comparison: Proposed vs SARIMA ──
        m = re.match(
            r"ТЕСТ:\s*(.+?)\s*\|\s*Proposed MAE=([\d.]+)±([\d.]+)\s*\|\s*SARIMA MAE=([\d.]+)±([\d.]+)",
            text,
        )
        if m:
            ds = m.group(1).strip()
            results.append({
                "experiment": "compare", "dataset": ds,
                "label": "Разработанный метод",
                "metrics": {"mae": float(m.group(2)), "mae_std": float(m.group(3))},
                "raw": raw,
            })
            results.append({
                "experiment": "compare", "dataset": ds,
                "label": "SARIMA",
                "metrics": {"mae": float(m.group(4)), "mae_std": float(m.group(5))},
                "raw": raw,
            })
            continue

        # ── 4. MGMT test: SLA, utilization, scale_ops ──
        m = re.match(
            r"MGMT ТЕСТ:\s*(.+?)\s*\|\s*SLA_VIOLATIONS=([\d.]+)%\s*\|\s*AVG_UTIL=([\d.]+)%\s*\|\s*SCALE_OPS=(\d+)",
            text,
        )
        if m:
            results.append({
                "experiment": "compare", "dataset": m.group(1).strip(),
                "label": "Разработанный метод",
                "metrics": {
                    "sla_pct": float(m.group(2)),
                    "avg_util": float(m.group(3)),
                    "scale_ops": int(m.group(4)),
                },
                "raw": raw,
            })
            continue

        # ── 5. Utilization ──
        m = re.match(r"Utilization \|\s*(.+?):\s*([\d.]+)%", text)
        if m:
            results.append({
                "experiment": "compare", "dataset": m.group(1).strip(),
                "label": "Разработанный метод",
                "metrics": {"avg_util": float(m.group(2))},
                "raw": raw,
            })
            continue

        # ── 6. Scale ops ratio (proposed vs HPA) ──
        m = re.match(
            r"Scale ops ratio \|\s*(.+?):\s*([\d.]+)\s*\(proposed=(\d+),\s*hpa=(\d+)\)",
            text,
        )
        if m:
            ds = m.group(1).strip()
            results.append({
                "experiment": "compare", "dataset": ds,
                "label": "Реактивный HPA",
                "metrics": {"scale_ops": int(m.group(4))},
                "raw": raw,
            })
            continue

        # ── 7. Horizon ──
        m = re.match(r"ТЕСТ: Горизонт h=(\d+) \| MAE=([\d.]+)±([\d.]+)", text)
        if m:
            results.append({
                "experiment": "horizon", "dataset": "Смешанный",
                "label": f"h={m.group(1)}",
                "metrics": {
                    "h": int(m.group(1)),
                    "mae": float(m.group(2)),
                    "mae_std": float(m.group(3)),
                },
                "raw": raw,
            })
            continue

        # ── 8. Retrain ──
        m = re.match(
            r"ТЕСТ: Дообучение \| MAE_с=([\d.]+) \| MAE_без=([\d.]+) \| Улучшение=([\d.]+)%",
            text,
        )
        if m:
            results.append({
                "experiment": "retrain", "dataset": "Смешанный",
                "label": "Дообучение",
                "metrics": {
                    "mae_with": float(m.group(1)),
                    "mae_without": float(m.group(2)),
                    "improvement": float(m.group(3)),
                },
                "raw": raw,
            })
            continue

        # ── 9. Phi (mixed scenario) ──
        m = re.match(
            r"ТЕСТ: Смешанный сценарий φ_t \| MAE_с_phi=([\d.]+) \| MAE_без_phi=([\d.]+) \| Улучшение=([\d.]+)%",
            text,
        )
        if m:
            results.append({
                "experiment": "phi", "dataset": "Смешанный",
                "label": "φ_t",
                "metrics": {
                    "mae_with_phi": float(m.group(1)),
                    "mae_without_phi": float(m.group(2)),
                    "improvement": float(m.group(3)),
                },
                "raw": raw,
            })
            continue

        # ── 10. Ablation: without STL ──
        m = re.match(
            r"ТЕСТ: Аблация без STL \| Полный метод MAE=([\d.]+) \| Без STL MAE=([\d.]+) \| Ухудшение=([\d.]+)%",
            text,
        )
        if m:
            results.append({
                "experiment": "ablation", "dataset": "Смешанный",
                "label": "Без STL-декомпозиции",
                "metrics": {
                    "mae_full": float(m.group(1)),
                    "mae": float(m.group(2)),
                    "worsening_pct": float(m.group(3)),
                },
                "raw": raw,
            })
            # Также сохраняем результат полного метода для абляции
            results.append({
                "experiment": "ablation", "dataset": "Смешанный",
                "label": "Полный метод",
                "metrics": {"mae": float(m.group(1))},
                "raw": raw,
            })
            continue

        # ── 11. Ablation: without hysteresis ──
        m = re.match(
            r"ТЕСТ: Аблация без гистерезиса \| Ops_с=(\d+) \| Ops_без=(\d+)",
            text,
        )
        if m:
            results.append({
                "experiment": "ablation", "dataset": "Смешанный",
                "label": "Без механизма гистерезиса",
                "metrics": {
                    "scale_ops_with": int(m.group(1)),
                    "scale_ops_without": int(m.group(2)),
                },
                "raw": raw,
            })
            continue

        # ── 12. Ablation: without quantile ──
        m = re.match(
            r"ТЕСТ: Аблация без квантиля \| SLA_с=([\d.]+)% \| SLA_без=([\d.]+)%",
            text,
        )
        if m:
            results.append({
                "experiment": "ablation", "dataset": "Смешанный",
                "label": "Без квантильной оценки",
                "metrics": {
                    "sla_with": float(m.group(1)),
                    "sla_without": float(m.group(2)),
                },
                "raw": raw,
            })
            continue

        # ── 13. Ablation: fixed delta ──
        m = re.match(
            r"ТЕСТ: Аблация фиксированный δ \| Util_адаптивный=([\d.]+)% \| Util_фикс=([\d.]+)%",
            text,
        )
        if m:
            results.append({
                "experiment": "ablation", "dataset": "Смешанный",
                "label": "С фиксированным порогом δ=1",
                "metrics": {
                    "util_adaptive": float(m.group(1)),
                    "util_fixed": float(m.group(2)),
                },
                "raw": raw,
            })
            continue

        # ── 14. Computation time ──
        m = re.match(
            r"ТЕСТ: Вычислительное время \| Среднее=([\d.]+)мс ± ([\d.]+)мс \| Доля от Δt=([\d.]+)%",
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

        # ── 15. Spike resilience ──
        m = re.match(
            r"ТЕСТ: Всплески σ=([\d.]+) \| MAE=([\d.]+)±([\d.]+)",
            text,
        )
        if m:
            results.append({
                "experiment": "spike", "dataset": "Всплесковый",
                "label": f"σ={m.group(1)}",
                "metrics": {
                    "amplitude_sigma": float(m.group(1)),
                    "mae": float(m.group(2)),
                    "mae_std": float(m.group(3)),
                },
                "raw": raw,
            })
            continue

        # Неразобранная строка — сохраняем как есть
        results.append({
            "experiment": "unknown", "dataset": None,
            "label": None,
            "metrics": {"raw_text": text},
            "raw": raw,
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
