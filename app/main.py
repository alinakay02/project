"""
app/main.py — Монолитное веб-приложение (глава 4, параграф 4.1)

Девять маршрутов трёх функциональных классов:
  Класс 1 (compute): /compute/light, /compute/medium, /compute/heavy
  Класс 2 (db):      /db/read, /db/write, /db/aggregate
  Класс 3 (memory):  /memory/alloc, /memory/process, /memory/gc

Метка class_id в счётчике Prometheus обеспечивает формирование φ_t (форм. 3.2).
"""

import gc
import math
import time
import random
import threading
from typing import Any

import numpy as np
from flask import Flask, jsonify, request
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Float
import yaml

# ── Конфигурация ─────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

DB_URL = CFG["webapp"]["database_url"]
POOL_SIZE = CFG["webapp"]["sqlalchemy_pool_size"]
MAX_OVERFLOW = CFG["webapp"]["sqlalchemy_max_overflow"]

# ── Flask и метрики Prometheus ────────────────────────────────────────────────
app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Счётчик с меткой class_id для формирования вектора φ_t (формула 3.2)
request_counter = Counter(
    "http_requests_total",
    "Total HTTP requests by class",
    ["class_id", "endpoint", "status"],
)
request_latency = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["class_id", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# ── База данных ────────────────────────────────────────────────────────────────
engine = create_engine(
    DB_URL,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_pre_ping=True,
)
Session = sessionmaker(bind=engine)
Base = declarative_base()


class Record(Base):
    __tablename__ = "records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(64), index=True)
    value = Column(Float)
    category = Column(Integer)


def init_db():
    """Создаёт таблицу и заполняет 1 000 000 строк для /db/read и /db/aggregate."""
    Base.metadata.create_all(engine)
    with Session() as session:
        count = session.execute(text("SELECT COUNT(*) FROM records")).scalar()
        if count < 1_000_000:
            app.logger.info("Filling DB with 1M rows…")
            batch = 10_000
            for i in range(0, 1_000_000, batch):
                session.bulk_insert_mappings(Record, [
                    {"name": f"item_{i+j}", "value": random.random(), "category": (i+j) % 10}
                    for j in range(batch)
                ])
                session.commit()
            app.logger.info("DB ready.")


# ── Вспомогательный декоратор для инструментирования ─────────────────────────
def track(class_id: int, endpoint: str):
    """Декоратор: считает запросы и задержку по классу."""
    def decorator(f):
        def wrapper(*args, **kwargs):
            start = time.time()
            status = "200"
            try:
                result = f(*args, **kwargs)
                return result
            except Exception as e:
                status = "500"
                raise
            finally:
                elapsed = time.time() - start
                request_counter.labels(
                    class_id=str(class_id),
                    endpoint=endpoint,
                    status=status,
                ).inc()
                request_latency.labels(
                    class_id=str(class_id),
                    endpoint=endpoint,
                ).observe(elapsed)
        wrapper.__name__ = f.__name__
        return wrapper
    return decorator


# ══════════════════════════════════════════════════════════════════════════════
# КЛАСС 1: Вычислительные операции (cpu_t ≈ 0.65–0.85 при 10 одновременных)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/compute/light")
@track(1, "/compute/light")
def compute_light():
    """Решето Эратосфена до N=5000 (класс 1, лёгкий)."""
    N = 5_000
    sieve = [True] * (N + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, N + 1, i):
                sieve[j] = False
    primes = [i for i, v in enumerate(sieve) if v]
    return jsonify({"count": len(primes), "last": primes[-1]})


@app.route("/compute/medium")
@track(1, "/compute/medium")
def compute_medium():
    """Целочисленная факторизация N=10^6 методом пробных делений (класс 1, средний)."""
    n = 10 ** 6
    factors = []
    d = 2
    tmp = n
    while d * d <= tmp:
        while tmp % d == 0:
            factors.append(d)
            tmp //= d
        d += 1
    if tmp > 1:
        factors.append(tmp)
    return jsonify({"n": n, "factors": factors})


@app.route("/compute/heavy")
@track(1, "/compute/heavy")
def compute_heavy():
    """Число π методом Монте-Карло, 10^6 итераций (класс 1, тяжёлый)."""
    n = 1_000_000
    rng = np.random.default_rng()
    x = rng.random(n)
    y = rng.random(n)
    pi_est = 4.0 * float(np.sum(x ** 2 + y ** 2 <= 1.0)) / n
    return jsonify({"pi": pi_est, "iterations": n})


# ══════════════════════════════════════════════════════════════════════════════
# КЛАСС 2: Запросы к базе данных (cpu_t ≈ 0.20–0.35 при 10 одновременных)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/db/read")
@track(2, "/db/read")
def db_read():
    """Чтение одной строки по первичному ключу (класс 2)."""
    row_id = random.randint(1, 1_000_000)
    with Session() as session:
        row = session.get(Record, row_id)
        if row is None:
            return jsonify({"error": "not found"}), 404
        return jsonify({"id": row.id, "name": row.name, "value": row.value})


@app.route("/db/write", methods=["POST", "GET"])
@track(2, "/db/write")
def db_write():
    """Вставка одной строки с индексированием (класс 2)."""
    with Session() as session:
        rec = Record(
            name=f"write_{random.randint(0, 10**9)}",
            value=random.random(),
            category=random.randint(0, 9),
        )
        session.add(rec)
        session.commit()
        return jsonify({"inserted_id": rec.id})


@app.route("/db/aggregate")
@track(2, "/db/aggregate")
def db_aggregate():
    """Агрегация с оконной функцией по 10^5 строкам (класс 2)."""
    with Session() as session:
        result = session.execute(
            text(
                "SELECT category, AVG(value) as avg_val, "
                "ROW_NUMBER() OVER (ORDER BY AVG(value) DESC) as rn "
                "FROM (SELECT * FROM records LIMIT 100000) sub "
                "GROUP BY category ORDER BY rn"
            )
        ).fetchall()
        return jsonify([{"category": r[0], "avg_value": float(r[1]), "rank": r[2]}
                        for r in result])


# ══════════════════════════════════════════════════════════════════════════════
# КЛАСС 3: Интенсивное управление памятью (cpu_t ≈ 0.30–0.55)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/memory/alloc")
@track(3, "/memory/alloc")
def memory_alloc():
    """Выделение и немедленное освобождение NumPy-массива 50 МБ (класс 3)."""
    arr = np.zeros(50 * 1024 * 1024 // 8, dtype=np.float64)  # 50 МБ
    checksum = float(arr.sum())
    del arr
    return jsonify({"alloc_mb": 50, "checksum": checksum})


@app.route("/memory/process")
@track(3, "/memory/process")
def memory_process():
    """Потоковая обработка 10^5 объектов с накоплением в списке (класс 3)."""
    def gen():
        for i in range(100_000):
            yield {"id": i, "val": math.sqrt(i + 1)}

    results = [item for item in gen()]  # накопление промежуточных объектов
    total = sum(r["val"] for r in results)
    return jsonify({"processed": len(results), "sum": total})


@app.route("/memory/gc")
@track(3, "/memory/gc")
def memory_gc():
    """Создание 10^4 объектов с циклическими ссылками + явный gc.collect() (класс 3)."""
    # Создаём циклические ссылки
    nodes = []
    for i in range(10_000):
        node = {"id": i, "data": list(range(10)), "ref": None}
        nodes.append(node)
    for i in range(len(nodes) - 1):
        nodes[i]["ref"] = nodes[i + 1]
    nodes[-1]["ref"] = nodes[0]  # замкнутый цикл

    # Явный вызов сборщика мусора
    collected = gc.collect()
    return jsonify({"objects_created": len(nodes), "gc_collected": collected})


# ── Health-check ──────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/ready")
def ready():
    """Readiness probe — проверяет доступность БД."""
    try:
        with Session() as session:
            session.execute(text("SELECT 1"))
        return jsonify({"status": "ready"}), 200
    except Exception as e:
        return jsonify({"status": "not ready", "error": str(e)}), 503


# ── Запуск ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    app.run(
        host=CFG["webapp"]["host"],
        port=CFG["webapp"]["port"],
        threaded=True,
    )
