"""
app/api.py — REST API для веб-интерфейса экспериментального стенда (параграф 4.1)

Режимы работы:
  • real — подключение к Prometheus и Kubernetes API. Поллер каждые 5 секунд
    забирает метрики из Prometheus (cpu, mem, rps, latency, errors, phi),
    читает текущее число реплик из Kubernetes API, строит прогноз обученной
    GRU-моделью (если доступна) и обновляет состояние. Используется при
    запуске стенда вместе с кластером kind.
  • demo — синтетический генератор с реалистичным сезонным паттерном.
    Реагирует на POST /api/load. Используется, если Prometheus недоступен
    или включён флаг DEMO_MODE=1.

Выбор режима:
  • Если переменная окружения DEMO_MODE=1 — принудительно demo.
  • Иначе при старте пробуется соединение с Prometheus по URL из config.yaml
    (prometheus.url). Если ответ 200 — включается real; иначе — demo.
  • Текущий режим экспонируется в GET /api/status → "mode": "real"|"demo".

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
import re
import numpy as np
import yaml
from scipy.stats import linregress
from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS
from sqlalchemy import create_engine, func, desc
from sqlalchemy.orm import sessionmaker
from app.models import Base as ExpBase, ExperimentResult
from controller.prometheus_collector import PrometheusCollector

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
from predictor.config import CFG_PREPROCESSOR, CFG_FORECASTER

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
        pp = Preprocessor(**CFG_PREPROCESSOR)
        fc = HybridForecaster(preprocessor=pp, **CFG_FORECASTER)
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

# ── Определение режима работы (real vs demo) ────────────────────────────────
# DEMO_MODE=1 в окружении → форсированный demo. Иначе пробуем Prometheus.

_FORCE_DEMO = os.environ.get("DEMO_MODE", "").strip() in ("1", "true", "True", "yes")
_PROMETHEUS_URL = _cfg.get("prometheus", {}).get("url", "http://localhost:9090")
_PROMQL_QUERIES = _cfg.get("prometheus", {}).get("queries", {})
_K8S_NAMESPACE = _cfg.get("kubernetes", {}).get("namespace", "default")
_K8S_DEPLOYMENT = _cfg.get("kubernetes", {}).get("deployment_name", "webapp")


def _probe_prometheus(url: str, timeout: float = 3.0) -> bool:
    """Пытается обратиться к /-/ready Prometheus. True — доступен."""
    try:
        import requests
        r = requests.get(f"{url.rstrip('/')}/-/ready", timeout=timeout)
        return r.status_code == 200
    except Exception as e:
        logger.info("Prometheus probe failed: %s (url=%s)", e, url)
        return False


def _prometheus_range_query(query, start_ts, end_ts, step_sec=15):
    """
    Выполняет PromQL query_range и возвращает [(timestamp, value), ...].

    Используется для построения истории за произвольный период напрямую из
    Prometheus — это надёжный источник, переживающий рестарты Flask и
    одинаковый для всех реплик webapp.

    Аргументы:
      query     : PromQL-запрос (агрегированный, чтобы вернулась одна серия)
      start_ts  : UNIX timestamp начала окна
      end_ts    : UNIX timestamp конца окна
      step_sec  : шаг между точками в секундах

    Возвращает пустой список при любых ошибках.
    """
    if _prometheus_collector is None:
        return []
    try:
        import requests
        url = f"{_PROMETHEUS_URL.rstrip('/')}/api/v1/query_range"
        r = requests.get(
            url,
            params={
                "query": query,
                "start": start_ts,
                "end":   end_ts,
                "step":  step_sec,
            },
            timeout=15,
        )
        r.raise_for_status()
        results = r.json().get("data", {}).get("result", [])
        if not results:
            return []
        # Агрегированный запрос возвращает одну серию; берём её values
        values = results[0].get("values", [])
        out = []
        for ts, v in values:
            if v in ("NaN", None):
                continue
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(val):
                continue
            out.append((float(ts), val))
        return out
    except Exception as e:
        logger.warning("Prometheus range query failed: %s | query=%s", e, query[:80])
        return []


def _parse_range(range_str, default_seconds):
    """Парсит строку вроде '60m', '6h', '24h', '7d' → секунды. Fallback при ошибке."""
    if not range_str:
        return default_seconds
    m = re.match(r"^(\d+)([smhd])$", range_str.strip().lower())
    if not m:
        return default_seconds
    num = int(m.group(1))
    unit = m.group(2)
    return num * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


def _build_k8s_apps_client():
    """
    Возвращает AppsV1Api или None. Пробует сначала in-cluster config, потом
    ~/.kube/config (локальный kubeconfig, используется когда api.py запущен
    вне кластера, например через port-forward).
    """
    try:
        from kubernetes import client as k8s_client, config as k8s_config
    except ImportError:
        logger.info("kubernetes package not available; replicas will be estimated.")
        return None
    try:
        k8s_config.load_incluster_config()
        logger.info("Kubernetes in-cluster config loaded.")
        return k8s_client.AppsV1Api()
    except Exception:
        pass
    try:
        k8s_config.load_kube_config()
        logger.info("Kubernetes kubeconfig loaded (local mode).")
        return k8s_client.AppsV1Api()
    except Exception as e:
        logger.info("Kubernetes config unavailable: %s", e)
        return None


# Попытка инициализации real-mode коннекторов
_prometheus_collector = None
_k8s_apps = None
_mode = "demo"

if not _FORCE_DEMO and _PROMQL_QUERIES and _probe_prometheus(_PROMETHEUS_URL):
    try:
        _prometheus_collector = PrometheusCollector(
            prometheus_url=_PROMETHEUS_URL,
            queries=_PROMQL_QUERIES,
        )
        _k8s_apps = _build_k8s_apps_client()
        _mode = "real"
        logger.info(
            "API started in REAL mode: Prometheus=%s, K8s=%s",
            _PROMETHEUS_URL, "available" if _k8s_apps else "unavailable",
        )
    except Exception as e:
        logger.warning("Failed to init real-mode collectors: %s. Falling back to demo.", e)
        _prometheus_collector = None
        _k8s_apps = None
        _mode = "demo"
else:
    if _FORCE_DEMO:
        logger.info("API started in DEMO mode (DEMO_MODE=1).")
    else:
        logger.info(
            "Prometheus unreachable at %s; API started in DEMO mode.",
            _PROMETHEUS_URL,
        )


def _get_current_replicas(default_val: int = 2) -> int:
    """Читает spec.replicas из Kubernetes Deployment, fallback — default_val."""
    if _k8s_apps is None:
        return default_val
    try:
        dep = _k8s_apps.read_namespaced_deployment(
            name=_K8S_DEPLOYMENT, namespace=_K8S_NAMESPACE,
        )
        return int(dep.spec.replicas or default_val)
    except Exception as e:
        logger.debug("K8s read replicas failed: %s", e)
        return default_val


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
    "mode": _mode,          # "real" | "demo" — видно фронтенду через /api/status
}
_state_lock = threading.Lock()


def update_state(new_vals: dict):
    with _state_lock:
        _state.update(new_vals)


def get_state() -> dict:
    with _state_lock:
        return dict(_state)


# ── Общая пост-обработка одного сэмпла (используется и real-, и demo-поллерами) ─
def _apply_sample(
    cpu: float, mem: float, rps: float, lat: float, err: float,
    phi,
    r_cur: int,
    action: str,
    saturation: bool,
    t_iter: int,
    prev_cpu_hat,
    prev_rps_hat,
) -> tuple:
    """
    Общая post-обработка измерения: строит прогноз по обученной GRU (или fallback),
    обновляет глобальное состояние _state (включая history и hourly_stats).
    Возвращает (new_prev_cpu_hat, new_prev_rps_hat) для следующей итерации.
    """
    with _config_lock:
        horizon_h = _runtime_config.get("horizon_h", 3)

    point = {
        "cpu_t": round(float(cpu), 4),
        "mem_t": round(float(mem), 4),
        "rps_t": round(max(float(rps), 0.0), 1),
        "lat_t": round(float(lat), 1),
        "err_t": round(float(err), 5),
        "phi":   [round(float(p), 3) for p in phi],
        "r_cur": int(r_cur),
        "action": action,
        "saturation": bool(saturation),
        "timestamp": int(time.time()),
        "iterations": int(t_iter),
    }

    # ── Прогноз ───────────────────────────────────────────────────────────
    with _state_lock:
        hist = list(_state["history"])
    cpu_hist = [h["cpu"] for h in hist] + [point["cpu_t"]]
    rps_hist = [h["rps"] for h in hist] + [point["rps_t"]]

    cpu_hat, cpu_lower, cpu_upper = None, None, None

    with _forecaster_lock:
        fc_model = _trained_forecaster
    if fc_model is not None and len(cpu_hist) >= fc_model.w_input + 10:
        try:
            ts_arr = np.array([h["ts"] for h in hist] + [point["timestamp"]])
            phi_arr = np.array(
                [h.get("phi", [0.33, 0.33, 0.34]) for h in hist] + [list(phi)]
            ).T
            cpu_arr = np.array(cpu_hist)
            c_hat, c_lo, c_hi = fc_model.predict(cpu_arr, ts_arr, phi_arr)
            cpu_hat = [round(float(v), 4) for v in c_hat]
            cpu_lower = [round(float(v), 4) for v in c_lo]
            cpu_upper = [round(float(v), 4) for v in c_hi]
        except Exception as e:
            logger.debug("GRU forecast failed, using fallback: %s", e)

    if cpu_hat is None:
        cpu_hat, cpu_lower, cpu_upper = _real_forecast(cpu_hist, horizon_h)

    rps_hat, rps_lower, rps_upper = _real_forecast(rps_hist, horizon_h)

    point["forecast"] = {
        "cpu_hat": cpu_hat, "q_lower": cpu_lower, "q_upper": cpu_upper,
        "rps_hat": rps_hat, "rps_lower": rps_lower, "rps_upper": rps_upper,
    }

    # ── Обновляем глобальное состояние ─────────────────────────────────────
    with _state_lock:
        _state.update(point)
        _state["history"].append({
            "ts":    point["timestamp"],
            "cpu":   point["cpu_t"],
            "cpu_pred": prev_cpu_hat,
            "mem":   point["mem_t"],
            "rps":   point["rps_t"],
            "rps_pred": prev_rps_hat,
            "lat":   point["lat_t"],
            "err":   point["err_t"],
            "r_cur": point["r_cur"],
            "phi":   list(point["phi"]),
        })
        if len(_state["history"]) > 500:
            _state["history"] = _state["history"][-500:]

        hour_key = time.strftime("%H")
        hs = _state["hourly_stats"]
        if hour_key not in hs:
            hs[hour_key] = {"cpu_sum": 0, "cpu_max": 0,
                            "rps_sum": 0, "rps_max": 0,
                            "cpu_pred_sum": 0, "rps_pred_sum": 0,
                            "lat_sum": 0, "err_sum": 0, "count": 0}
        bucket = hs[hour_key]
        bucket["cpu_sum"] += point["cpu_t"]
        bucket["cpu_max"] = max(bucket["cpu_max"], point["cpu_t"])
        bucket["rps_sum"] += point["rps_t"]
        bucket["rps_max"] = max(bucket["rps_max"], point["rps_t"])
        bucket["lat_sum"] += point["lat_t"]
        bucket["err_sum"] += point["err_t"]
        if prev_cpu_hat is not None:
            bucket["cpu_pred_sum"] = bucket.get("cpu_pred_sum", 0) + float(prev_cpu_hat)
        if prev_rps_hat is not None:
            bucket["rps_pred_sum"] = bucket.get("rps_pred_sum", 0) + float(prev_rps_hat)
        bucket["count"] += 1

    new_prev_cpu = cpu_hat[0] if cpu_hat else None
    new_prev_rps = rps_hat[0] if rps_hat else None
    return new_prev_cpu, new_prev_rps


# ── Real-поллер (Prometheus + Kubernetes) ───────────────────────────────────
def _real_poller():
    """
    Опрашивает Prometheus каждые 5 секунд, берёт текущее число реплик из
    Kubernetes API и обновляет глобальное состояние. Используется, когда
    api.py подключён к реальному кластеру.
    """
    logger.info("Real-mode poller started (Prometheus + K8s).")
    t = 0
    prev_cpu_hat = None
    prev_rps_hat = None
    prev_replicas = _get_current_replicas(
        default_val=_runtime_config.get("r_min", 2)
    )

    while True:
        try:
            m_t, phi_t = _prometheus_collector.collect()
            cpu = float(np.clip(m_t[0], 0.0, 1.0))
            mem = float(np.clip(m_t[1], 0.0, 1.0))
            rps = float(max(m_t[2], 0.0))
            lat = float(max(m_t[3], 0.0))
            err = float(np.clip(m_t[4], 0.0, 1.0))
            phi = [float(p) for p in phi_t.tolist()]

            # Текущее число реплик читаем из Kubernetes
            r_cur = _get_current_replicas(default_val=prev_replicas)

            if r_cur > prev_replicas:
                action = "scale_up"
            elif r_cur < prev_replicas:
                action = "scale_down"
            else:
                action = "no_change"

            r_max = _runtime_config.get("r_max_cluster", 20)
            saturation = r_cur >= r_max

            prev_cpu_hat, prev_rps_hat = _apply_sample(
                cpu, mem, rps, lat, err, phi,
                r_cur=r_cur, action=action, saturation=saturation,
                t_iter=t,
                prev_cpu_hat=prev_cpu_hat, prev_rps_hat=prev_rps_hat,
            )
            prev_replicas = r_cur
        except Exception as e:
            logger.warning("Real poller iteration failed: %s", e)

        t += 1
        time.sleep(5)


# ── Имитационный генератор данных (для демонстрации без Prometheus) ─────────
def _demo_generator():
    """Генерирует реалистичный ряд cpu_t для демо-режима.

    Реагирует на состояние нагрузки (_state["load"]):
      - когда нагрузка запущена — CPU, RPS, память растут пропорционально
        числу пользователей, φ отражает реальное распределение классов;
      - когда остановлена — фоновая синусоида с вращающимся доминирующим классом.
    Конфигурация берётся из _runtime_config (cpu_target, r_min, r_max_cluster).
    """
    t = 0
    prev_cpu_hat = None
    prev_rps_hat = None
    while True:
        with _state_lock:
            load = dict(_state["load"])

        load_running = load.get("running", False)
        u1 = load.get("users_class1", 0)
        u2 = load.get("users_class2", 0)
        u3 = load.get("users_class3", 0)
        total_users = u1 + u2 + u3

        with _config_lock:
            cpu_target = _runtime_config.get("cpu_target", 0.70)
            r_min = _runtime_config.get("r_min", 2)
            r_max = _runtime_config.get("r_max_cluster", 20)

        seasonal = 0.20 * math.sin(2 * math.pi * t / 288)
        base_cpu = 0.35 + seasonal
        noise = 0.02 * (2 * float(np.random.rand()) - 1)

        if load_running and total_users > 0:
            load_factor = min(total_users / 3000.0, 1.0)
            cpu = base_cpu + load_factor * 0.45 + noise
            phi = [u1 / total_users, u2 / total_users, u3 / total_users]
            rps = total_users * 1.0 + float(np.random.randn()) * total_users * 0.02
            mem = 0.30 + 0.05 * math.sin(2 * math.pi * t / 144) + load_factor * 0.25
        else:
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

        with _state_lock:
            prev_r = _state.get("r_cur", r_min)
        if r_cur > prev_r:
            action = "scale_up"
        elif r_cur < prev_r:
            action = "scale_down"
        else:
            action = "no_change"

        prev_cpu_hat, prev_rps_hat = _apply_sample(
            cpu, mem, rps, lat, err, phi,
            r_cur=r_cur, action=action, saturation=(r_cur >= r_max),
            t_iter=t,
            prev_cpu_hat=prev_cpu_hat, prev_rps_hat=prev_rps_hat,
        )

        t += 1
        time.sleep(5)


# ── Выбор и запуск поллера в зависимости от режима ──────────────────────────
if _mode == "real" and _prometheus_collector is not None:
    _poller_thread = threading.Thread(
        target=_real_poller, daemon=True, name="api-real-poller",
    )
else:
    _poller_thread = threading.Thread(
        target=_demo_generator, daemon=True, name="api-demo-generator",
    )
_poller_thread.start()


# ══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

def _status_snapshot() -> dict:
    """Формирует снапшот состояния системы — используется и /api/status, и /api/stream."""
    s = get_state()
    return {
        "mode": s.get("mode", _mode),
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
    }


@api_app.route("/api/status")
def api_status():
    """Текущее состояние системы: метрики, прогноз, число реплик. Точечный запрос."""
    return jsonify(_status_snapshot())


@api_app.route("/api/stream")
def api_stream():
    """
    Push-поток обновлений состояния через Server-Sent Events (SSE).

    Клиент (Vue/JS):
        const es = new EventSource('/api/stream');
        es.addEventListener('status', (e) => { const d = JSON.parse(e.data); ... });

    Сервер отправляет событие `status` каждый раз, когда счётчик iterations
    в глобальном состоянии увеличился (т.е. появился новый сэмпл от поллера),
    а также служебный comment-ping раз в 15 секунд для keep-alive против
    прокси/балансировщиков.
    """
    def gen():
        last_iterations = -1
        last_keepalive = time.time()
        # Сразу шлём текущее состояние, чтобы клиент не ждал следующего тика
        snap = _status_snapshot()
        last_iterations = int(snap.get("iterations", 0) or 0)
        yield f"event: status\ndata: {json.dumps(snap)}\n\n"

        while True:
            snap = _status_snapshot()
            it = int(snap.get("iterations", 0) or 0)
            if it != last_iterations:
                last_iterations = it
                yield f"event: status\ndata: {json.dumps(snap)}\n\n"
                last_keepalive = time.time()
            elif time.time() - last_keepalive >= 15:
                # ping-комментарий: браузер его игнорирует, но HTTP-соединение остаётся живым
                yield ": keep-alive\n\n"
                last_keepalive = time.time()
            time.sleep(0.5)

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",   # nginx: не буферизовать SSE
        "Connection": "keep-alive",
    }
    return Response(
        stream_with_context(gen()),
        mimetype="text/event-stream",
        headers=headers,
    )


@api_app.route("/api/history")
def api_history():
    """
    История метрик за заданный период.

    Источники:
      • real-режим — Prometheus query_range (постоянный, общий для всех реплик).
      • demo / Prometheus недоступен — in-memory _state["history"] (поллер).

    Параметры query string:
      n     — число точек (default: 80). При отсутствии range используется как
              ориентир: окно = n × step.
      range — окно времени: "60m", "6h", "24h", "7d". Имеет приоритет над n.
      step  — шаг в секундах (default: 15).

    Поля cpu_pred / rps_pred остаются null при источнике prometheus, т.к.
    исторические предсказания модели не сохраняются в TSDB.
    """
    n = int(request.args.get("n", 80))
    step = max(1, int(request.args.get("step", 15)))
    range_str = request.args.get("range", "")

    # ── Fallback на in-memory: demo-режим или Prometheus недоступен ──────
    if _mode != "real" or _prometheus_collector is None:
        with _state_lock:
            hist = _state["history"][-n:]
        return jsonify({"history": hist, "count": len(hist), "source": "memory"})

    # ── Real-режим: Prometheus query_range ────────────────────────────────
    end_ts = int(time.time())
    window_sec = _parse_range(range_str, n * step)
    start_ts = end_ts - window_sec

    cpu_series = _prometheus_range_query(_PROMQL_QUERIES.get("cpu", ""), start_ts, end_ts, step)
    mem_series = _prometheus_range_query(_PROMQL_QUERIES.get("mem", ""), start_ts, end_ts, step)
    rps_series = _prometheus_range_query(_PROMQL_QUERIES.get("rps", ""), start_ts, end_ts, step)
    lat_series = _prometheus_range_query(_PROMQL_QUERIES.get("lat_p95", ""), start_ts, end_ts, step)
    err_series = _prometheus_range_query(_PROMQL_QUERIES.get("err_rate", ""), start_ts, end_ts, step)
    rep_query = (
        f'kube_deployment_spec_replicas{{namespace="{_K8S_NAMESPACE}",'
        f'deployment="{_K8S_DEPLOYMENT}"}}'
    )
    rep_series = _prometheus_range_query(rep_query, start_ts, end_ts, step)

    # Прогнозы CPU из метрик контроллера (controller_forecast_*).
    # Агрегируем через max() для устойчивости к перезапускам controller pod'а:
    # при rolling update в TSDB появляются несколько временных серий с разными
    # лейблами pod/instance — каждая со своим набором timestamps. avg/max/min
    # сольёт их в одну серию, чтобы _prometheus_range_query видел все точки.
    # max используется потому что в один момент времени активен ровно один
    # под controller (replicas=1).
    cpu_pred_series = _prometheus_range_query(
        'max(controller_forecast_cpu{horizon="1"})', start_ts, end_ts, step,
    )
    cpu_pred_lower = _prometheus_range_query(
        'max(controller_forecast_cpu_lower{horizon="1"})', start_ts, end_ts, step,
    )
    cpu_pred_upper = _prometheus_range_query(
        'max(controller_forecast_cpu_upper{horizon="1"})', start_ts, end_ts, step,
    )

    by_ts = {}
    for ts, v in cpu_series:      by_ts.setdefault(int(ts), {})["cpu"]      = round(v, 4)
    for ts, v in mem_series:      by_ts.setdefault(int(ts), {})["mem"]      = round(v, 4)
    for ts, v in rps_series:      by_ts.setdefault(int(ts), {})["rps"]      = round(v, 1)
    for ts, v in lat_series:      by_ts.setdefault(int(ts), {})["lat"]      = round(v, 1)
    for ts, v in err_series:      by_ts.setdefault(int(ts), {})["err"]      = round(v, 5)
    for ts, v in rep_series:      by_ts.setdefault(int(ts), {})["r_cur"]    = int(v)
    for ts, v in cpu_pred_series: by_ts.setdefault(int(ts), {})["cpu_pred"] = round(v, 4)
    for ts, v in cpu_pred_lower:  by_ts.setdefault(int(ts), {})["cpu_lo"]   = round(v, 4)
    for ts, v in cpu_pred_upper:  by_ts.setdefault(int(ts), {})["cpu_hi"]   = round(v, 4)

    history = []
    last_r = _runtime_config.get("r_min", 2)
    for ts in sorted(by_ts.keys()):
        row = by_ts[ts]
        if "r_cur" in row:
            last_r = row["r_cur"]
        history.append({
            "ts":       ts,
            "cpu":      row.get("cpu", 0.0),
            "cpu_pred": row.get("cpu_pred"),    # из controller_forecast_cpu
            "cpu_lo":   row.get("cpu_lo"),      # нижний квантиль прогноза
            "cpu_hi":   row.get("cpu_hi"),      # верхний квантиль прогноза
            "mem":      row.get("mem", 0.0),
            "rps":      row.get("rps", 0.0),
            "rps_pred": None,                   # RPS-прогноз пока не публикуется
            "lat":      row.get("lat", 0.0),
            "err":      row.get("err", 0.0),
            "r_cur":    last_r,
            "phi":      [0.333, 0.333, 0.334],
        })

    # Возвращаем не больше n при заданном n без range (для backward compat).
    if not range_str and len(history) > n:
        history = history[-n:]

    return jsonify({"history": history, "count": len(history), "source": "prometheus"})


@api_app.route("/api/history/daily")
def api_history_daily():
    """
    Почасовая статистика за последние 24 часа.

    Источник:
      • real-режим — Prometheus query_range за 24ч с шагом 1 мин, агрегация
        в Python по локальному часу суток.
      • demo / Prometheus недоступен — in-memory _state["hourly_stats"].

    cpu_pred_avg / rps_pred_avg всегда null в режиме prometheus.
    """
    # ── Fallback на in-memory ────────────────────────────────────────────
    if _mode != "real" or _prometheus_collector is None:
        with _state_lock:
            hs = dict(_state["hourly_stats"])
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
        return jsonify({"hourly": result, "source": "memory"})

    # ── Real-режим: Prometheus query_range за 24 часа ────────────────────
    end_ts = int(time.time())
    start_ts = end_ts - 24 * 3600
    step = 60  # 1 минута — достаточно для агрегирования по часам

    cpu_series      = _prometheus_range_query(_PROMQL_QUERIES.get("cpu", ""), start_ts, end_ts, step)
    rps_series      = _prometheus_range_query(_PROMQL_QUERIES.get("rps", ""), start_ts, end_ts, step)
    lat_series      = _prometheus_range_query(_PROMQL_QUERIES.get("lat_p95", ""), start_ts, end_ts, step)
    cpu_pred_series = _prometheus_range_query(
        'max(controller_forecast_cpu{horizon="1"})', start_ts, end_ts, step,
    )

    # Бакеты "00".."23" — час локального времени
    hourly = {f"{h:02d}": {
        "cpu_sum": 0.0, "cpu_max": 0.0,
        "rps_sum": 0.0, "rps_max": 0.0,
        "lat_sum": 0.0,
        "cpu_pred_sum": 0.0, "cpu_pred_count": 0,
        "count":   0,
    } for h in range(24)}

    for ts, v in cpu_series:
        h = time.strftime("%H", time.localtime(ts))
        b = hourly[h]
        b["cpu_sum"] += v
        b["cpu_max"] = max(b["cpu_max"], v)
        b["count"]  += 1
    for ts, v in rps_series:
        h = time.strftime("%H", time.localtime(ts))
        b = hourly[h]
        b["rps_sum"] += v
        b["rps_max"] = max(b["rps_max"], v)
    for ts, v in lat_series:
        h = time.strftime("%H", time.localtime(ts))
        b = hourly[h]
        b["lat_sum"] += v
    for ts, v in cpu_pred_series:
        h = time.strftime("%H", time.localtime(ts))
        b = hourly[h]
        b["cpu_pred_sum"]   += v
        b["cpu_pred_count"] += 1

    result = []
    for h_int in range(24):
        key = f"{h_int:02d}"
        b = hourly[key]
        if b["count"] > 0:
            c = b["count"]
            pc = b["cpu_pred_count"]
            result.append({
                "hour":         f"{key}:00",
                "cpu_avg":      round(b["cpu_sum"] / c, 4),
                "cpu_max":      round(b["cpu_max"], 4),
                "cpu_pred_avg": round(b["cpu_pred_sum"] / pc, 4) if pc > 0 else None,
                "rps_avg":      round(b["rps_sum"] / c, 1),
                "rps_max":      round(b["rps_max"], 1),
                "rps_pred_avg": None,
                "lat_avg":      round(b["lat_sum"] / c, 1),
                "count":        c,
            })
        else:
            result.append({
                "hour":         f"{key}:00",
                "cpu_avg":      None, "cpu_max":      None, "cpu_pred_avg": None,
                "rps_avg":      None, "rps_max":      None, "rps_pred_avg": None,
                "lat_avg":      None, "count":        0,
            })
    return jsonify({"hourly": result, "source": "prometheus"})


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
            pp = Preprocessor(**CFG_PREPROCESSOR)
            forecaster = HybridForecaster(preprocessor=pp, **CFG_FORECASTER)

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
