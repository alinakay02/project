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
import os
import numpy as np
import yaml
from scipy.stats import linregress
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
    _cfg = {}
    _db_url = "postgresql://appuser:apppass@localhost:5432/appdb"

# ── Параметры из config.yaml, обновляемые через POST /api/config ───────────
_runtime_config = {
    "cpu_target":     _cfg.get("decision", {}).get("cpu_target", 0.70),
    "epsilon":        _cfg.get("decision", {}).get("epsilon", 0.05),
    "r_min":          _cfg.get("decision", {}).get("r_min", 2),
    "r_max_cluster":  _cfg.get("decision", {}).get("r_max_cluster", 20),
    "tau":            _cfg.get("decision", {}).get("tau", 4),
    "beta":           _cfg.get("decision", {}).get("beta", 0.3),
    "horizon_h":      _cfg.get("timeseries", {}).get("horizon_h", 3),
    "dt_minutes":     _cfg.get("timeseries", {}).get("dt_minutes", 5),
    "max_epochs":     _cfg.get("model", {}).get("max_epochs", 100),
    "patience":       _cfg.get("model", {}).get("patience", 10),
    "db_max_conn":    _cfg.get("decision", {}).get("db", {}).get("max_conn", 300),
    "db_conn_reserve": _cfg.get("decision", {}).get("db", {}).get("conn_reserve", 20),
    "db_pool_size":   _cfg.get("decision", {}).get("db", {}).get("pool_size", 5),
}
_config_lock = threading.Lock()


def _real_forecast(history_values, horizon_h, period=288, n_T=60, n_cycles=7):
    """
    Прогноз по реальному алгоритму проекта (параграф 3.2.2):
      - Тренд: линейная экстраполяция МНК (формула 3.11)
      - Сезонность: усреднение по N_c предыдущим суточным циклам (формула 3.12)
      - ДИ: на основе дисперсии остатка, расширяется с горизонтом
    """
    arr = np.array(history_values, dtype=float)
    n = len(arr)

    if n < 3:
        v = float(arr[-1]) if n > 0 else 0.0
        return [v] * horizon_h, [v] * horizon_h, [v] * horizon_h

    # ── Тренд: скользящее среднее ─────────────────────────────────────
    win = min(max(n // 4, 3), 30)
    if win > n:
        win = n
    if win % 2 == 0:
        win = max(win - 1, 1)
    kernel = np.ones(win) / win
    # mode='valid' + padding даёт точный размер n
    pad = win // 2
    padded = np.pad(arr, pad, mode='edge')
    trend = np.convolve(padded, kernel, mode='valid')[:n]

    # ── Сезонная компонента (отклонения от тренда) ─────────────────────
    detrended = arr - trend

    # ── Прогноз тренда: линейная экстраполяция (формула 3.11) ──────────
    trend_tail = trend[-min(n_T, n):]
    idx = np.arange(len(trend_tail))
    slope, intercept, _, _, _ = linregress(idx, trend_tail)

    # ── Стандартное отклонение остатка для ДИ ──────────────────────────
    resid_std = float(np.std(detrended[-min(60, n):]))
    if resid_std < 1e-8:
        resid_std = 0.01

    hat, lower, upper = [], [], []
    for k in range(1, horizon_h + 1):
        # T̂_{t+k}: экстраполяция линейного тренда
        t_hat = slope * (len(trend_tail) + k - 1) + intercept

        # Ŝ_{t+k}: среднее из предыдущих суточных циклов (формула 3.12)
        s_hat = 0.0
        season_vals = []
        for j in range(1, n_cycles + 1):
            sidx = n - period * j + k - 1
            if 0 <= sidx < n:
                season_vals.append(detrended[sidx])
        if season_vals:
            s_hat = float(np.mean(season_vals))

        forecast_val = t_hat + s_hat
        ci = resid_std * 1.96 * math.sqrt(k)

        hat.append(round(float(forecast_val), 4))
        lower.append(round(float(max(forecast_val - ci, 0)), 4))
        upper.append(round(float(forecast_val + ci), 4))

    return hat, lower, upper

# ── Глобальная обученная модель (подгружается после обучения) ───────────────
# Параметры по умолчанию (совпадают с config.yaml / test_experiments.py)
CFG_PREPROCESSOR = dict(w_a=48, iqr_alpha=1.5, w_norm=60, period=288, robust=True)
CFG_FORECASTER   = dict(n_T=60, n_cycles=7, period=288, horizon_h=3, w_input=60,
                        quantiles=[0.025, 0.5, 0.975], hidden_dim=64, dropout=0.10,
                        lr=0.001, lr_decay=0.99, grad_clip=1.0,
                        max_epochs=100, patience=0, batch_size=32)

_trained_forecaster = None
_forecaster_lock = threading.Lock()


def _load_latest_model():
    """Ищет последнюю обученную модель в models/ и загружает."""
    global _trained_forecaster
    import glob
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    files = glob.glob(os.path.join(model_dir, "gru_*.pt"))
    if not files:
        return
    latest = max(files, key=os.path.getmtime)
    try:
        from predictor.preprocessor import Preprocessor
        from predictor.forecaster import HybridForecaster
        pp = Preprocessor(**{k: _cfg.get("preprocessing", {}).get(k, v)
                             for k, v in CFG_PREPROCESSOR.items()})
        fc = HybridForecaster(preprocessor=pp, **{
            k: _cfg.get("model", {}).get(k,
               _cfg.get("timeseries", {}).get(k,
               _cfg.get("preprocessing", {}).get(k, v)))
            for k, v in CFG_FORECASTER.items()})
        fc.load_model(latest)
        with _forecaster_lock:
            _trained_forecaster = fc
        logger.info(f"Loaded trained model from {latest}")
    except Exception as e:
        logger.warning(f"Could not load model from {latest}: {e}")


# Дефолтные параметры моделей
_load_latest_model()

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
    "hourly_stats": {},     # ключ "HH" → {cpu_sum, cpu_max, rps_sum, rps_max, count}
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
    """Генерирует реалистичный ряд cpu_t для демо-режима.

    Реагирует на состояние нагрузки (_state["load"]):
      - когда нагрузка запущена — CPU, RPS, память растут пропорционально
        числу пользователей, φ отражает реальное распределение классов;
      - когда остановлена — фоновая синусоида с вращающимся доминирующим классом.
    Конфигурация берётся из _runtime_config (cpu_target, r_min, r_max_cluster).
    """
    t = 0
    prev_cpu_hat = None   # 1-step-ahead CPU прогноз с предыдущей итерации
    prev_rps_hat = None   # 1-step-ahead RPS прогноз с предыдущей итерации
    while True:
        # ── Считываем текущее состояние нагрузки ──────────────────────
        with _state_lock:
            load = dict(_state["load"])

        load_running = load.get("running", False)
        u1 = load.get("users_class1", 0)
        u2 = load.get("users_class2", 0)
        u3 = load.get("users_class3", 0)
        total_users = u1 + u2 + u3

        # ── Считываем конфигурацию ────────────────────────────────────
        with _config_lock:
            cpu_target = _runtime_config.get("cpu_target", 0.70)
            r_min = _runtime_config.get("r_min", 2)
            r_max = _runtime_config.get("r_max_cluster", 20)
            horizon_h = _runtime_config.get("horizon_h", 3)

        # ── Базовый сезонный сигнал ───────────────────────────────────
        seasonal = 0.20 * math.sin(2 * math.pi * t / 288)
        base_cpu = 0.35 + seasonal
        noise = 0.02 * (2 * float(np.random.rand()) - 1)

        if load_running and total_users > 0:
            # Нагрузка активна — CPU растёт пропорционально пользователям
            # При 3000 суммарных пользователей добавляется ~0.45 к CPU
            load_factor = min(total_users / 3000.0, 1.0)
            cpu = base_cpu + load_factor * 0.45 + noise
            # Φ — реальное соотношение классов
            phi = [u1 / total_users, u2 / total_users, u3 / total_users]
            # RPS: ~1 запрос/сек на пользователя (Locust wait_time 0.5–1.5с)
            rps = total_users * 1.0 + float(np.random.randn()) * total_users * 0.02
            # Память тоже растёт под нагрузкой
            mem = 0.30 + 0.05 * math.sin(2 * math.pi * t / 144) + load_factor * 0.25
        else:
            # Фоновый режим — плавная синусоида
            cpu = base_cpu + noise
            phase = (t // 120) % 3
            phi = [0.15, 0.15, 0.15]
            phi[phase] = 0.70
            rps = 30 + 15 * math.sin(2 * math.pi * t / 288) + float(np.random.randn()) * 2
            mem = 0.30 + 0.05 * math.sin(2 * math.pi * t / 144)

        cpu = min(max(cpu, 0.05), 0.95)
        mem = min(max(mem, 0.05), 0.95)
        lat = 80 + 40 * cpu + float(np.random.randn()) * 5
        err = max(0, 0.001 + 0.01 * max(0, cpu - 0.8))

        r_cur = max(r_min, min(r_max, math.ceil(cpu / cpu_target)))

        # Определяем действие масштабирования
        with _state_lock:
            prev_r = _state.get("r_cur", r_min)
        if r_cur > prev_r:
            action = "scale_up"
        elif r_cur < prev_r:
            action = "scale_down"
        else:
            action = "no_change"

        point = {
            "cpu_t": round(cpu, 4),
            "mem_t": round(mem, 4),
            "rps_t": round(max(rps, 0), 1),
            "lat_t": round(lat, 1),
            "err_t": round(err, 5),
            "phi":   [round(p, 3) for p in phi],
            "r_cur": r_cur,
            "action": action,
            "saturation": r_cur >= r_max,
            "timestamp": int(time.time()),
            "iterations": t,
        }
        # ── Прогноз ───────────────────────────────────────────────────────
        with _state_lock:
            hist = list(_state["history"])
        cpu_hist = [h["cpu"] for h in hist] + [round(cpu, 4)]
        rps_hist = [h["rps"] for h in hist] + [round(max(rps, 0), 1)]

        cpu_hat, cpu_lower, cpu_upper = None, None, None

        # Пытаемся использовать обученную GRU-модель
        with _forecaster_lock:
            fc_model = _trained_forecaster
        if fc_model is not None and len(cpu_hist) >= fc_model.w_input + 10:
            try:
                ts_arr = np.array([h["ts"] for h in hist] + [int(time.time())])
                phi_arr = np.array([h.get("phi", [0.33, 0.33, 0.34]) for h in hist] + [phi]).T
                cpu_arr = np.array(cpu_hist)
                c_hat, c_lo, c_hi = fc_model.predict(cpu_arr, ts_arr, phi_arr)
                cpu_hat = [round(float(v), 4) for v in c_hat]
                cpu_lower = [round(float(v), 4) for v in c_lo]
                cpu_upper = [round(float(v), 4) for v in c_hi]
            except Exception as e:
                logger.debug(f"GRU forecast failed, using fallback: {e}")

        # Fallback — линейная экстраполяция
        if cpu_hat is None:
            cpu_hat, cpu_lower, cpu_upper = _real_forecast(cpu_hist, horizon_h)

        rps_hat, rps_lower, rps_upper = _real_forecast(rps_hist, horizon_h)

        point["forecast"] = {
            "cpu_hat": cpu_hat, "q_lower": cpu_lower, "q_upper": cpu_upper,
            "rps_hat": rps_hat, "rps_lower": rps_lower, "rps_upper": rps_upper,
        }

        with _state_lock:
            _state.update(point)
            _state["history"].append({
                "ts":    point["timestamp"],
                "cpu":   point["cpu_t"],
                "cpu_pred": prev_cpu_hat,       # что модель предсказывала для этой точки
                "mem":   point["mem_t"],
                "rps":   point["rps_t"],
                "rps_pred": prev_rps_hat,       # что модель предсказывала для этой точки
                "lat":   point["lat_t"],
                "err":   point["err_t"],
                "r_cur": point["r_cur"],
                "phi":   phi,
            })
            if len(_state["history"]) > 500:
                _state["history"] = _state["history"][-500:]

            # ── Почасовая аккумуляция для суточного графика ───────────
            hour_key = time.strftime("%H")
            hs = _state["hourly_stats"]
            if hour_key not in hs:
                hs[hour_key] = {"cpu_sum": 0, "cpu_max": 0,
                                "rps_sum": 0, "rps_max": 0,
                                "cpu_pred_sum": 0, "rps_pred_sum": 0,
                                "lat_sum": 0, "err_sum": 0, "count": 0}
            bucket = hs[hour_key]
            bucket["cpu_sum"] += cpu
            bucket["cpu_max"] = max(bucket["cpu_max"], cpu)
            bucket["rps_sum"] += max(rps, 0)
            bucket["rps_max"] = max(bucket["rps_max"], max(rps, 0))
            bucket["lat_sum"] += lat
            bucket["err_sum"] += err
            # Прогнозные значения (1-step-ahead с прошлой итерации)
            if prev_cpu_hat is not None:
                bucket["cpu_pred_sum"] = bucket.get("cpu_pred_sum", 0) + prev_cpu_hat
            if prev_rps_hat is not None:
                bucket["rps_pred_sum"] = bucket.get("rps_pred_sum", 0) + prev_rps_hat
            bucket["count"] += 1

        # Сохраняем 1-step-ahead прогноз для следующей итерации
        fc = point["forecast"]
        prev_cpu_hat = fc["cpu_hat"][0] if fc["cpu_hat"] else None
        prev_rps_hat = fc["rps_hat"][0] if fc["rps_hat"] else None

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
            "current":   s.get("r_cur", _runtime_config["r_min"]),
            "r_min":     _runtime_config["r_min"],
            "r_max":     _runtime_config["r_max_cluster"],
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


@api_app.route("/api/history/daily")
def api_history_daily():
    """Почасовая статистика за последние 24 часа."""
    with _state_lock:
        hs = dict(_state["hourly_stats"])

    # Формируем 24 слота (00..23), заполняем имеющимися данными
    result = []
    for h_int in range(24):
        key = f"{h_int:02d}"
        label = f"{key}:00"
        if key in hs:
            s = hs[key]
            c = s["count"] or 1
            result.append({
                "hour": label,
                "cpu_avg": round(s["cpu_sum"] / c, 4),
                "cpu_max": round(s["cpu_max"], 4),
                "cpu_pred_avg": round(s.get("cpu_pred_sum", 0) / c, 4),
                "rps_avg": round(s["rps_sum"] / c, 1),
                "rps_max": round(s["rps_max"], 1),
                "rps_pred_avg": round(s.get("rps_pred_sum", 0) / c, 1),
                "lat_avg": round(s["lat_sum"] / c, 1),
                "count": s["count"],
            })
        else:
            result.append({
                "hour": label,
                "cpu_avg": None, "cpu_max": None, "cpu_pred_avg": None,
                "rps_avg": None, "rps_max": None, "rps_pred_avg": None,
                "lat_avg": None, "count": 0,
            })
    return jsonify({"hourly": result})


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


@api_app.route("/api/datasets/preview")
def api_dataset_preview():
    """Возвращает первые N точек датасета для визуализации."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from tests.data_generators import (
        generate_stationary, generate_trend, generate_spike,
        generate_mixed, load_alibaba_trace, load_google_trace, load_azure_trace,
    )
    dataset_id = request.args.get("dataset", "mixed")
    n_points = min(int(request.args.get("n", 500)), 2000)

    loaders = {
        "alibaba":    load_alibaba_trace,
        "google":     load_google_trace,
        "azure":      load_azure_trace,
        "stationary": generate_stationary,
        "trend":      generate_trend,
        "spike":      generate_spike,
        "mixed":      generate_mixed,
    }
    try:
        loader_fn = loaders.get(dataset_id, generate_mixed)
        cpu, ts, phi = loader_fn()
        step = max(1, len(cpu) // n_points)
        sampled_cpu = cpu[::step][:n_points]
        sampled_ts = ts[::step][:n_points]
        return jsonify({
            "dataset": dataset_id,
            "total_points": len(cpu),
            "cpu": [round(float(v), 4) for v in sampled_cpu],
            "timestamps": [int(v) for v in sampled_ts],
        })
    except Exception as e:
        return jsonify({"dataset": dataset_id, "error": str(e), "cpu": [], "timestamps": []})


@api_app.route("/api/training/status")
def api_training_status():
    """Статус обучения: потери по эпохам."""
    with _state_lock:
        t = dict(_state["training"])
    t["max_epochs"] = _runtime_config["max_epochs"]
    t["patience"] = _runtime_config["patience"]
    return jsonify(t)


@api_app.route("/api/training/start", methods=["POST"])
def api_training_start():
    """Запустить реальное обучение HybridForecaster в фоновом потоке."""
    data = request.get_json(force=True) or {}
    dataset_id = data.get("dataset", "mixed")

    with _state_lock:
        if _state["training"]["running"]:
            return jsonify({"ok": False, "error": "Обучение уже запущено"}), 409

    def _train_real():
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from predictor.preprocessor import Preprocessor
        from predictor.forecaster import HybridForecaster
        from tests.data_generators import (
            generate_stationary, generate_trend, generate_spike,
            generate_mixed, load_alibaba_trace,
        )

        with _state_lock:
            _state["training"].update({
                "running": True, "epoch": 0,
                "train_loss": [], "val_loss": [],
                "best_val": None, "stopped_early": False,
                "dataset": dataset_id, "error": None,
            })

        try:
            # ── 1. Загрузка датасета ─────────────────────────────────────
            loaders = {
                "alibaba":    load_alibaba_trace,
                "stationary": generate_stationary,
                "trend":      generate_trend,
                "spike":      generate_spike,
                "mixed":      generate_mixed,
            }
            loader_fn = loaders.get(dataset_id, generate_mixed)
            cpu_series, timestamps, phi = loader_fn()
            logger.info(f"Training on dataset '{dataset_id}': {len(cpu_series)} observations")

            # ── 2. Создание моделей ──────────────────────────────────────
            pp = Preprocessor(**{k: _cfg.get("preprocessing", {}).get(k, v)
                                 for k, v in CFG_PREPROCESSOR.items()})
            fc_cfg = {k: _cfg.get("model", {}).get(k,
                        _cfg.get("timeseries", {}).get(k,
                        _cfg.get("preprocessing", {}).get(k, v)))
                      for k, v in CFG_FORECASTER.items()}
            forecaster = HybridForecaster(preprocessor=pp, **fc_cfg)

            # ── 3. Callback для обновления прогресса по эпохам ────────
            def on_epoch(epoch, train_loss, val_loss, best_val):
                with _state_lock:
                    _state["training"]["epoch"] = epoch
                    _state["training"]["train_loss"].append(round(train_loss, 4))
                    _state["training"]["val_loss"].append(round(val_loss, 4))
                    _state["training"]["best_val"] = round(best_val, 4)

            # ── 4. Обучение ──────────────────────────────────────────────
            n_train = int(len(cpu_series) * 0.85)
            forecaster.fit(
                cpu_series[:n_train],
                timestamps[:n_train],
                phi[:, :n_train],
                val_split=0.15,
                epoch_callback=on_epoch,
            )

            with _state_lock:
                _state["training"]["stopped_early"] = forecaster.trainer.stopped_epoch > 0

            # ── 5. Сохранение модели и подключение к мониторингу ────────
            global _trained_forecaster
            model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"gru_{dataset_id}.pt")
            forecaster.save_model(model_path)
            with _forecaster_lock:
                _trained_forecaster = forecaster
            logger.info(f"Model saved to {model_path} and connected to monitoring")

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            with _state_lock:
                _state["training"]["error"] = str(e)
        finally:
            with _state_lock:
                _state["training"]["running"] = False

    # Фоновый поток для обучения, чтобы не блокировать API
    thread = threading.Thread(target=_train_real, daemon=True)
    thread.start()
    return jsonify({"ok": True, "dataset": dataset_id})


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
    return jsonify({
        "datasets": [
            {"id": "alibaba",    "name": "Alibaba Cluster Trace 2018",
             "type": "real",     "available": os.path.exists("data/alibaba_cluster_trace_2018.csv"), "n_obs": 2243},
            {"id": "google",     "name": "Google Cluster Trace 2019",
             "type": "real",     "available": os.path.exists("data/google_cluster_trace_2019.csv"), "n_obs": 8064},
            {"id": "azure",      "name": "Azure VM Trace 2019",
             "type": "real",     "available": os.path.exists("data/azure_vm_trace_2019.csv"), "n_obs": 8640},
            {"id": "stationary", "name": "Стационарный",
             "type": "synthetic","available": True, "n_obs": 4320},
            {"id": "trend",      "name": "Трендовый",
             "type": "synthetic","available": True, "n_obs": 4320},
            {"id": "spike",      "name": "Всплесковый",
             "type": "synthetic","available": True, "n_obs": 4320},
            {"id": "mixed",      "name": "Смешанный",
             "type": "synthetic","available": True, "n_obs": 4320},
        ]
    })


@api_app.route("/api/config", methods=["GET"])
def api_config_get():
    """Текущая конфигурация — из config.yaml + runtime-обновления."""
    with _config_lock:
        return jsonify(dict(_runtime_config))


@api_app.route("/api/config", methods=["POST"])
def api_config_post():
    """Обновить параметры (tau, beta, cpu_target и др.)."""
    data = request.get_json(force=True)
    allowed = {"tau", "beta", "cpu_target", "horizon_h", "r_min", "r_max_cluster", "epsilon"}
    updated = {}
    with _config_lock:
        for k, v in data.items():
            if k in allowed:
                _runtime_config[k] = v
                updated[k] = v
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
