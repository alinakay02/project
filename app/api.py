"""
app/api.py — REST API для веб-интерфейса экспериментального стенда (параграф 4.1)

Эндпоинты:
  GET  /api/status          — текущее состояние системы (m_t, r_cur, прогноз)
  GET  /api/history         — история метрик за последние N точек
  POST /api/load            — управление нагрузкой Locust (запуск/остановка)
  GET  /api/training/status — статус обучения модели (потери по эпохам)
  POST /api/training/start  — запустить обучение
  GET  /api/compare         — сравнение всех методов на выбранном датасете
  GET  /api/datasets        — список доступных наборов данных
  GET  /api/config          — текущая конфигурация
  POST /api/config          — обновить параметры (tau, beta, cpu_target)
"""

import threading
import time
import math
import json
import logging
import numpy as np
import yaml
from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy import create_engine, func, desc
from sqlalchemy.orm import sessionmaker
from app.models import Base as ExpBase, ExperimentResult

logger = logging.getLogger(__name__)

# ── Создаём Flask-приложение для API ─────────────────────────────────────────
api_app = Flask(__name__)
CORS(api_app)   # разрешаем запросы с Vue.js dev-сервера

# ── Подключение к БД для результатов экспериментов ──────────────────────────
try:
    with open("config.yaml") as _f:
        _cfg = yaml.safe_load(_f)
    _db_url = _cfg["webapp"]["database_url"]
except Exception:
    _db_url = "postgresql://appuser:apppass@localhost:5432/appdb"

_engine = create_engine(_db_url, pool_pre_ping=True, pool_size=3)
ExpBase.metadata.create_all(_engine)
_Session = sessionmaker(bind=_engine)

# ── Глобальное хранилище состояния ───────────────────────────────────────────
# (в production — Redis или база данных)
_state = {
    "cpu_t": 0.0, "mem_t": 0.0, "rps_t": 0.0, "lat_t": 0.0, "err_t": 0.0,
    "phi": [0.33, 0.33, 0.34],
    "r_cur": 2,
    "forecast": {"cpu_hat": [], "q_lower": [], "q_upper": []},
    "action": "no_change",
    "saturation": False,
    "training": {"running": False, "epoch": 0, "train_loss": [], "val_loss": [], "best_val": None},
    "load": {"running": False, "users_class1": 0, "users_class2": 0, "users_class3": 0},
    "iterations": 0,
    "history": [],          # список точек {ts, cpu, mem, rps, lat, err, r_cur}
    "comparison": None,
}
_state_lock = threading.Lock()


def update_state(new_vals: dict):
    with _state_lock:
        _state.update(new_vals)


def get_state() -> dict:
    with _state_lock:
        return dict(_state)


# ── Имитационный генератор данных (для демонстрации без реального Prometheus) ─
def _demo_generator():
    """Генерирует реалистичный ряд cpu_t для демо-режима."""
    t = 0
    while True:
        # Суточная сезонность + тренд + шум
        seasonal = 0.20 * math.sin(2 * math.pi * t / 288)
        trend = 0.35
        noise = 0.02 * (2 * float(np.random.rand()) - 1)
        cpu = min(max(trend + seasonal + noise, 0.05), 0.95)

        # Состав классов — периодически меняем доминирующий
        phase = (t // 120) % 3
        phi = [0.15, 0.15, 0.15]
        phi[phase] = 0.70

        mem = 0.30 + 0.05 * math.sin(2 * math.pi * t / 144)
        rps = 30 + 15 * math.sin(2 * math.pi * t / 288) + float(np.random.randn()) * 2
        lat = 80 + 40 * cpu + float(np.random.randn()) * 5
        err = max(0, 0.001 + 0.01 * max(0, cpu - 0.8))

        point = {
            "cpu_t": round(cpu, 4),
            "mem_t": round(mem, 4),
            "rps_t": round(max(rps, 0), 1),
            "lat_t": round(lat, 1),
            "err_t": round(err, 5),
            "phi":   [round(p, 3) for p in phi],
            "r_cur": max(2, min(20, math.ceil(cpu / 0.70))),
            "timestamp": int(time.time()),
            "iterations": t,
        }
        # Простой «прогноз» для демо
        point["forecast"] = {
            "cpu_hat": [round(cpu + 0.02*k, 4) for k in range(1, 4)],
            "q_lower": [round(cpu + 0.02*k - 0.08, 4) for k in range(1, 4)],
            "q_upper": [round(cpu + 0.02*k + 0.08, 4) for k in range(1, 4)],
        }

        with _state_lock:
            _state.update(point)
            _state["history"].append({
                "ts":    point["timestamp"],
                "cpu":   point["cpu_t"],
                "mem":   point["mem_t"],
                "rps":   point["rps_t"],
                "lat":   point["lat_t"],
                "err":   point["err_t"],
                "r_cur": point["r_cur"],
                "phi":   phi,
            })
            if len(_state["history"]) > 500:
                _state["history"] = _state["history"][-500:]

        t += 1
        time.sleep(5)   # Δt = 5 мин в реальности, 5 сек в демо


# ── Запускаем демо-генератор в фоне ─────────────────────────────────────────
_demo_thread = threading.Thread(target=_demo_generator, daemon=True)
_demo_thread.start()


# ══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@api_app.route("/api/status")
def api_status():
    """Текущее состояние системы: метрики, прогноз, число реплик."""
    s = get_state()
    return jsonify({
        "metrics": {
            "cpu_t":  s.get("cpu_t", 0),
            "mem_t":  s.get("mem_t", 0),
            "rps_t":  s.get("rps_t", 0),
            "lat_t":  s.get("lat_t", 0),
            "err_t":  s.get("err_t", 0),
            "phi":    s.get("phi", [0.33, 0.33, 0.34]),
        },
        "replicas": {
            "current":   s.get("r_cur", 2),
            "r_min":     2,
            "r_max":     20,
            "action":    s.get("action", "no_change"),
            "saturation": s.get("saturation", False),
        },
        "forecast": s.get("forecast", {"cpu_hat": [], "q_lower": [], "q_upper": []}),
        "iterations": s.get("iterations", 0),
        "timestamp": s.get("timestamp", int(time.time())),
    })


@api_app.route("/api/history")
def api_history():
    """История метрик за последние N точек."""
    n = int(request.args.get("n", 100))
    with _state_lock:
        hist = _state["history"][-n:]
    return jsonify({"history": hist, "count": len(hist)})


@api_app.route("/api/load", methods=["GET"])
def api_load_get():
    """Текущий статус нагрузки."""
    with _state_lock:
        load = dict(_state["load"])
    return jsonify(load)


@api_app.route("/api/load", methods=["POST"])
def api_load_post():
    """
    Управление Locust.
    Тело запроса: {users_class1: int, users_class2: int, users_class3: int, running: bool}
    """
    data = request.get_json(force=True)
    with _state_lock:
        _state["load"].update({
            "running":       data.get("running", False),
            "users_class1":  data.get("users_class1", 0),
            "users_class2":  data.get("users_class2", 0),
            "users_class3":  data.get("users_class3", 0),
        })
    # В реальной реализации здесь вызывается Locust REST API
    logger.info(f"Load updated: {_state['load']}")
    return jsonify({"ok": True, "load": _state["load"]})


@api_app.route("/api/training/status")
def api_training_status():
    """Статус обучения: потери по эпохам."""
    with _state_lock:
        t = dict(_state["training"])
    return jsonify(t)


@api_app.route("/api/training/start", methods=["POST"])
def api_training_start():
    """Запустить обучение в фоновом потоке."""
    data = request.get_json(force=True) or {}
    dataset = data.get("dataset", "synthetic_mixed")

    def _train_demo():
        with _state_lock:
            _state["training"]["running"] = True
            _state["training"]["train_loss"] = []
            _state["training"]["val_loss"] = []
            _state["training"]["epoch"] = 0

        for epoch in range(1, 44):   # ~43 эпохи как в таблице 4.4
            time.sleep(0.3)           # имитация одной эпохи
            t_loss = 0.18 * math.exp(-0.08 * epoch) + 0.028 + float(np.random.randn()) * 0.002
            v_loss = 0.20 * math.exp(-0.07 * epoch) + 0.031 + float(np.random.randn()) * 0.003
            with _state_lock:
                _state["training"]["epoch"] = epoch
                _state["training"]["train_loss"].append(round(t_loss, 4))
                _state["training"]["val_loss"].append(round(v_loss, 4))
                _state["training"]["best_val"] = round(min(
                    v_loss, _state["training"].get("best_val") or v_loss), 4)

        with _state_lock:
            _state["training"]["running"] = False

    thread = threading.Thread(target=_train_demo, daemon=True)
    thread.start()
    return jsonify({"ok": True, "dataset": dataset})


def _latest_run_id(session, experiment, dataset=None):
    """Находит run_id последнего прогона для указанного типа эксперимента."""
    q = session.query(ExperimentResult.run_id)\
        .filter(ExperimentResult.experiment == experiment)
    if dataset:
        q = q.filter(ExperimentResult.dataset == dataset)
    row = q.order_by(desc(ExperimentResult.created_at)).first()
    return row[0] if row else None


def _merge_metrics(rows):
    """Объединяет метрики из нескольких строк с одинаковым label."""
    by_label = {}
    for r in rows:
        m = json.loads(r.metrics_json or "{}")
        key = r.label or "unknown"
        if key not in by_label:
            by_label[key] = {}
        by_label[key].update(m)
    return by_label


@api_app.route("/api/compare")
def api_compare():
    """Сравнение методов — данные из реальных экспериментов pytest."""
    dataset_filter = request.args.get("dataset", "")
    session = _Session()
    try:
        # Берём последний run_id, содержащий compare-результаты
        run_id = _latest_run_id(session, "compare")
        if not run_id:
            return jsonify({"dataset": dataset_filter, "results": [], "source": "empty"})

        # Загружаем все compare-строки этого прогона
        q = session.query(ExperimentResult)\
            .filter_by(experiment="compare", run_id=run_id)
        if dataset_filter:
            # Маппинг id → русское название
            ds_map = {
                "alibaba": "Alibaba", "stationary": "Стационарный",
                "trend": "Трендовый", "spike": "Всплесковый", "mixed": "Смешанный",
            }
            ds_name = ds_map.get(dataset_filter, dataset_filter)
            q = q.filter(ExperimentResult.dataset == ds_name)

        rows = q.all()
        merged = _merge_metrics(rows)

        # Собираем ответ: сначала Разработанный метод, потом остальные
        results = []
        order = ["Разработанный метод", "Автономная GRU", "LSTM", "Prophet", "SARIMA", "Реактивный HPA"]
        seen = set()
        for method in order:
            if method in merged:
                m = merged[method]
                results.append({
                    "method": method,
                    "mae": m.get("mae"),
                    "rmse": m.get("rmse"),
                    "mape": m.get("mape"),
                    "coverage": m.get("coverage"),
                    "sla_pct": m.get("sla_pct"),
                    "avg_util": m.get("avg_util"),
                    "scale_ops": m.get("scale_ops"),
                })
                seen.add(method)
        # Добавляем методы, которых нет в order
        for method, m in merged.items():
            if method not in seen:
                results.append({
                    "method": method,
                    "mae": m.get("mae"), "rmse": m.get("rmse"),
                    "mape": m.get("mape"), "coverage": m.get("coverage"),
                    "sla_pct": m.get("sla_pct"), "avg_util": m.get("avg_util"),
                    "scale_ops": m.get("scale_ops"),
                })

        return jsonify({"dataset": dataset_filter, "results": results, "run_id": run_id, "source": "db"})
    finally:
        session.close()


@api_app.route("/api/datasets")
def api_datasets():
    """Список доступных наборов данных."""
    import os
    alibaba_available = os.path.exists("data/alibaba_cluster_trace_2018.csv")
    return jsonify({
        "datasets": [
            {"id": "alibaba",    "name": "Alibaba Cluster Trace 2018",
             "type": "real",     "available": alibaba_available, "n_obs": 2304},
            {"id": "stationary", "name": "Стационарный (синтетический)",
             "type": "synthetic","available": True, "n_obs": 4320},
            {"id": "trend",      "name": "Трендовый (синтетический)",
             "type": "synthetic","available": True, "n_obs": 4320},
            {"id": "spike",      "name": "Всплесковый (синтетический)",
             "type": "synthetic","available": True, "n_obs": 4320},
            {"id": "mixed",      "name": "Смешанный (синтетический)",
             "type": "synthetic","available": True, "n_obs": 4320},
        ]
    })


@api_app.route("/api/config", methods=["GET"])
def api_config_get():
    """Текущая конфигурация модуля принятия решений."""
    return jsonify({
        "cpu_target":   0.70,
        "epsilon":      0.05,
        "r_min":        2,
        "r_max_cluster": 20,
        "tau":          4,
        "beta":         0.3,
        "horizon_h":    3,
        "dt_minutes":   5,
    })


@api_app.route("/api/config", methods=["POST"])
def api_config_post():
    """Обновить параметры (tau, beta, cpu_target)."""
    data = request.get_json(force=True)
    allowed = {"tau", "beta", "cpu_target", "horizon_h"}
    updated = {k: v for k, v in data.items() if k in allowed}
    logger.info(f"Config updated: {updated}")
    return jsonify({"ok": True, "updated": updated})


@api_app.route("/api/ablation")
def api_ablation():
    """Результаты анализа абляции из реальных экспериментов."""
    session = _Session()
    try:
        run_id = _latest_run_id(session, "ablation")
        if not run_id:
            return jsonify({"results": [], "source": "empty"})

        rows = session.query(ExperimentResult)\
            .filter_by(experiment="ablation", run_id=run_id).all()

        results = []
        for r in rows:
            m = json.loads(r.metrics_json or "{}")
            results.append({
                "config": r.label,
                "mae": m.get("mae"),
                "sla_pct": m.get("sla_without") or m.get("sla_with"),
                "util_pct": m.get("util_adaptive") or m.get("util_fixed"),
                "scale_ops": m.get("scale_ops_without") or m.get("scale_ops_with"),
                "worsening_pct": m.get("worsening_pct"),
                "metrics": m,
            })

        # Полный метод первым
        results.sort(key=lambda x: 0 if x["config"] == "Полный метод" else 1)
        return jsonify({"results": results, "run_id": run_id, "source": "db"})
    finally:
        session.close()


@api_app.route("/api/horizon")
def api_horizon():
    """Зависимость MAE от горизонта h — из реальных экспериментов."""
    session = _Session()
    try:
        run_id = _latest_run_id(session, "horizon")
        if not run_id:
            return jsonify({"results": [], "source": "empty"})

        rows = session.query(ExperimentResult)\
            .filter_by(experiment="horizon", run_id=run_id).all()

        results = []
        for r in rows:
            m = json.loads(r.metrics_json or "{}")
            results.append({
                "h": m.get("h"),
                "mae": m.get("mae"),
                "mae_std": m.get("mae_std"),
            })
        results.sort(key=lambda x: x.get("h", 0))
        return jsonify({"results": results, "run_id": run_id, "source": "db"})
    finally:
        session.close()


@api_app.route("/api/phi")
def api_phi():
    """Эффект признаков φ_t — из реальных экспериментов."""
    session = _Session()
    try:
        run_id = _latest_run_id(session, "phi")
        if not run_id:
            return jsonify({"results": [], "source": "empty"})

        rows = session.query(ExperimentResult)\
            .filter_by(experiment="phi", run_id=run_id).all()

        results = []
        for r in rows:
            m = json.loads(r.metrics_json or "{}")
            results.append({
                "mae_with_phi": m.get("mae_with_phi"),
                "mae_without_phi": m.get("mae_without_phi"),
                "improvement": m.get("improvement"),
            })
        return jsonify({"results": results, "run_id": run_id, "source": "db"})
    finally:
        session.close()


@api_app.route("/api/timing")
def api_timing():
    """Вычислительное время — из реальных экспериментов."""
    session = _Session()
    try:
        run_id = _latest_run_id(session, "timing")
        if not run_id:
            return jsonify({"results": [], "source": "empty"})

        rows = session.query(ExperimentResult)\
            .filter_by(experiment="timing", run_id=run_id).all()

        results = []
        for r in rows:
            m = json.loads(r.metrics_json or "{}")
            results.append({
                "module": r.label,
                "mean_ms": m.get("mean_ms"),
                "std_ms": m.get("std_ms"),
                "pct": f"{m.get('pct_of_dt', 0):.3f}%",
            })
        return jsonify({"results": results, "run_id": run_id, "source": "db"})
    finally:
        session.close()


@api_app.route("/api/spike")
def api_spike():
    """Устойчивость к всплескам — из реальных экспериментов."""
    session = _Session()
    try:
        run_id = _latest_run_id(session, "spike")
        if not run_id:
            return jsonify({"results": [], "source": "empty"})

        rows = session.query(ExperimentResult)\
            .filter_by(experiment="spike", run_id=run_id).all()

        results = []
        for r in rows:
            m = json.loads(r.metrics_json or "{}")
            results.append({
                "amplitude_sigma": m.get("amplitude_sigma"),
                "mae": m.get("mae"),
                "mae_std": m.get("mae_std"),
            })
        results.sort(key=lambda x: x.get("amplitude_sigma", 0))
        return jsonify({"results": results, "run_id": run_id, "source": "db"})
    finally:
        session.close()


@api_app.route("/api/retrain")
def api_retrain():
    """Влияние дообучения — из реальных экспериментов."""
    session = _Session()
    try:
        run_id = _latest_run_id(session, "retrain")
        if not run_id:
            return jsonify({"results": [], "source": "empty"})

        rows = session.query(ExperimentResult)\
            .filter_by(experiment="retrain", run_id=run_id).all()

        results = []
        for r in rows:
            m = json.loads(r.metrics_json or "{}")
            results.append(m)
        return jsonify({"results": results, "run_id": run_id, "source": "db"})
    finally:
        session.close()


# ── Запуск / статус экспериментов ───────────────────────────────────────────

_experiment_status = {"running": False, "run_id": None, "started_at": None, "filter": ""}


@api_app.route("/api/experiments/run", methods=["POST"])
def api_experiments_run():
    """Запустить pytest-эксперименты в фоне и сохранить результаты в БД."""
    if _experiment_status["running"]:
        return jsonify({"ok": False, "error": "Тесты уже выполняются",
                        "run_id": _experiment_status["run_id"]}), 409

    data = request.get_json(force=True) or {}
    test_filter = data.get("filter", "")

    def _run():
        import subprocess, sys, os
        _experiment_status["running"] = True
        try:
            from scripts.save_results import run_and_save
            run_id, count, rc = run_and_save(test_filter, _db_url)
            _experiment_status["run_id"] = run_id
            _experiment_status["count"] = count
            _experiment_status["exit_code"] = rc
        except Exception as e:
            logger.error(f"Experiment run failed: {e}")
            _experiment_status["error"] = str(e)
        finally:
            _experiment_status["running"] = False

    _experiment_status["running"] = True
    _experiment_status["filter"] = test_filter
    _experiment_status["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return jsonify({"ok": True, "message": "Тесты запущены", "filter": test_filter})


@api_app.route("/api/experiments/status")
def api_experiments_status():
    """Статус текущего / последнего прогона экспериментов."""
    return jsonify(_experiment_status)


@api_app.route("/api/experiments/runs")
def api_experiments_runs():
    """Список последних прогонов экспериментов."""
    session = _Session()
    try:
        runs = session.query(
            ExperimentResult.run_id,
            func.min(ExperimentResult.created_at).label("started"),
            func.count(ExperimentResult.id).label("count"),
        ).group_by(ExperimentResult.run_id)\
         .order_by(desc("started"))\
         .limit(20).all()

        return jsonify({
            "runs": [{"run_id": r[0], "created_at": r[1].isoformat(), "count": r[2]}
                     for r in runs]
        })
    finally:
        session.close()


if __name__ == "__main__":
    api_app.run(host="0.0.0.0", port=5001, debug=False)
