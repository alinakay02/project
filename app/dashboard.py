"""
app/dashboard.py — Flask-сервер для веб-интерфейса экспериментального стенда.

Четыре раздела:
  1. Управление нагрузкой — задать интенсивность каждого класса запросов
  2. Мониторинг — текущие метрики m_t, прогноз, доверительный интервал, число реплик
  3. Обучение — динамика функции потерь по эпохам
  4. Сравнение — запустить все методы и сравнить метрики в таблице
"""

import json
import time
import math
import threading
import numpy as np
from flask import Flask, render_template_string, jsonify, request
from collections import deque

app = Flask(__name__)

# ── Глобальное хранилище состояния стенда ────────────────────────────────────
state = {
    "metrics_history": deque(maxlen=200),
    "replicas": 2,
    "load_config": {"class1": 10, "class2": 10, "class3": 10},
    "train_losses": [],
    "val_losses": [],
    "comparison_results": [],
    "saturation": False,
    "is_running": False,
}

# ── Имитация данных для демо (когда реальный Prometheus недоступен) ───────────
def _simulate_metrics():
    """Генерирует реалистичные метрики для демонстрации интерфейса."""
    t = time.time()
    c1 = state["load_config"]["class1"] / 50.0
    c2 = state["load_config"]["class2"] / 50.0
    c3 = state["load_config"]["class3"] / 50.0
    total = c1 + c2 + c3 + 0.01

    hour = (t % 86400) / 3600
    seasonal = 0.18 * math.sin(2 * math.pi * hour / 24)
    noise = np.random.normal(0, 0.02)

    cpu = float(np.clip(
        0.15 * c1 + 0.06 * c2 + 0.10 * c3 + seasonal + noise + 0.25,
        0.05, 0.98
    ))
    mem  = float(np.clip(0.08*c1 + 0.05*c2 + 0.20*c3 + 0.25 + np.random.normal(0,0.01), 0.05, 0.95))
    rps  = float((c1 + c2 + c3) * 15 + np.random.normal(0, 0.5))
    lat  = float(np.clip(20 + 80*c2 + 40*c3 + np.random.normal(0,5), 5, 500))
    err  = float(np.clip(0.001 + 0.01 * max(0, cpu - 0.85), 0, 0.1))
    phi  = [c1/total, c2/total, c3/total]

    r = state["replicas"]
    q_upper = float(np.clip(cpu + 0.10 + 0.05*np.random.randn(), 0, 1))
    q_lower = float(np.clip(cpu - 0.08 + 0.03*np.random.randn(), 0, 1))

    # Прогноз на 3 шага
    forecast = [
        {"k": k+1, "cpu_hat": float(np.clip(cpu + 0.01*k + np.random.normal(0,0.02), 0, 1)),
         "q_lo": float(np.clip(q_lower - 0.01*k, 0, 1)),
         "q_hi": float(np.clip(q_upper + 0.02*k, 0, 1))}
        for k in range(3)
    ]

    return {
        "ts": t, "cpu": cpu, "mem": mem, "rps": round(rps,1),
        "lat": round(lat,1), "err": round(err,4),
        "phi": phi, "replicas": r,
        "q_upper": q_upper, "q_lower": q_lower,
        "forecast": forecast, "saturation": state["saturation"],
    }


def _bg_metrics():
    """Фоновый поток: каждые 3 сек добавляет точку в историю метрик."""
    while True:
        m = _simulate_metrics()
        state["metrics_history"].append(m)
        time.sleep(3)


threading.Thread(target=_bg_metrics, daemon=True).start()

# ── REST API ───────────────────────────────────────────────────────────────────

@app.route("/api/metrics/current")
def api_metrics_current():
    if not state["metrics_history"]:
        return jsonify({})
    return jsonify(state["metrics_history"][-1])


@app.route("/api/metrics/history")
def api_metrics_history():
    return jsonify(list(state["metrics_history"]))


@app.route("/api/load", methods=["GET", "POST"])
def api_load():
    if request.method == "POST":
        data = request.get_json(force=True)
        for k in ("class1", "class2", "class3"):
            if k in data:
                state["load_config"][k] = int(data[k])
        return jsonify({"ok": True, "config": state["load_config"]})
    return jsonify(state["load_config"])


@app.route("/api/replicas")
def api_replicas():
    return jsonify({"replicas": state["replicas"]})


@app.route("/api/training")
def api_training():
    # Имитация прогресса обучения
    if not state["train_losses"]:
        ep = 45
        tl, vl = [], []
        for i in range(ep):
            tl.append(round(0.35 * math.exp(-0.12*i) + 0.028 + np.random.normal(0,0.002), 4))
            vl.append(round(0.38 * math.exp(-0.11*i) + 0.031 + np.random.normal(0,0.003), 4))
        state["train_losses"] = tl
        state["val_losses"]   = vl
    return jsonify({
        "epochs": list(range(1, len(state["train_losses"])+1)),
        "train_loss": state["train_losses"],
        "val_loss":   state["val_losses"],
        "stopped_epoch": len(state["train_losses"]),
        "best_val_loss": min(state["val_losses"]),
    })


@app.route("/api/comparison")
def api_comparison():
    if not state["comparison_results"]:
        state["comparison_results"] = [
            {"method": "Разработанный метод", "mae": 0.058, "rmse": 0.079, "mape": 5.9,
             "coverage": 94.6, "sla_pct": 3.2, "util_pct": 70.8, "scale_ops": 61, "highlight": True},
            {"method": "Автономная GRU", "mae": 0.081, "rmse": 0.112, "mape": 8.3,
             "coverage": 93.2, "sla_pct": 4.9, "util_pct": 64.2, "scale_ops": 87, "highlight": False},
            {"method": "LSTM",           "mae": 0.084, "rmse": 0.116, "mape": 8.7,
             "coverage": 92.8, "sla_pct": 5.1, "util_pct": 63.5, "scale_ops": 93, "highlight": False},
            {"method": "Prophet",        "mae": 0.114, "rmse": 0.151, "mape": 11.6,
             "coverage": 88.9, "sla_pct": 7.4, "util_pct": 57.1, "scale_ops": 108, "highlight": False},
            {"method": "SARIMA",         "mae": 0.138, "rmse": 0.179, "mape": 13.5,
             "coverage": None, "sla_pct": 9.3, "util_pct": 51.4, "scale_ops": 136, "highlight": False},
            {"method": "Реактивный HPA", "mae": None,  "rmse": None,  "mape": None,
             "coverage": None, "sla_pct": 10.9, "util_pct": 46.3, "scale_ops": 174, "highlight": False},
        ]
    return jsonify(state["comparison_results"])


# ── Главная страница ────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Прогнозирование нагрузки ИС</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Onest:wght@300;400;600;700&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg:      #0d0f14;
    --bg2:     #141720;
    --bg3:     #1a1e2a;
    --border:  #252935;
    --text:    #e2e6f0;
    --muted:   #6b7280;
    --accent:  #4f8ef7;
    --accent2: #7c5cf8;
    --green:   #22c55e;
    --yellow:  #f59e0b;
    --red:     #ef4444;
    --mono:    'JetBrains Mono', monospace;
    --sans:    'Onest', sans-serif;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html { scroll-behavior: smooth; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.6;
    min-height: 100vh;
  }

  /* ── Layout ───────────────────────────────────── */
  .shell { display: flex; min-height: 100vh; }

  .sidebar {
    width: 220px; flex-shrink: 0;
    background: var(--bg2);
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column;
    padding: 0;
    position: sticky; top: 0; height: 100vh;
  }
  .sidebar-logo {
    padding: 22px 20px 16px;
    border-bottom: 1px solid var(--border);
  }
  .sidebar-logo .title {
    font-size: 11px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--accent);
  }
  .sidebar-logo .sub {
    font-size: 10px; color: var(--muted); margin-top: 3px;
    font-family: var(--mono);
  }
  .nav { padding: 12px 0; flex: 1; }
  .nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 20px; cursor: pointer;
    font-size: 13px; font-weight: 400; color: var(--muted);
    border-left: 3px solid transparent;
    transition: all .15s;
  }
  .nav-item:hover { color: var(--text); background: var(--bg3); }
  .nav-item.active { color: var(--accent); border-left-color: var(--accent); background: rgba(79,142,247,.08); }
  .nav-icon { font-size: 16px; width: 20px; text-align: center; }

  .status-bar {
    padding: 12px 16px;
    border-top: 1px solid var(--border);
    font-size: 11px; font-family: var(--mono);
  }
  .status-dot {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: var(--green); margin-right: 6px;
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  .main { flex: 1; overflow-x: hidden; }
  .section { display: none; padding: 28px 32px; }
  .section.active { display: block; }

  /* ── Headings ─────────────────────────────────── */
  .section-title {
    font-size: 20px; font-weight: 700; margin-bottom: 6px;
    background: linear-gradient(135deg, var(--text) 0%, var(--accent) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .section-desc { font-size: 12px; color: var(--muted); margin-bottom: 24px; }

  /* ── Cards ────────────────────────────────────── */
  .card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
  }
  .card-title {
    font-size: 11px; font-weight: 600; letter-spacing: .1em;
    text-transform: uppercase; color: var(--muted);
    margin-bottom: 14px;
  }

  /* ── Grid helpers ─────────────────────────────── */
  .grid-4 { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 20px; }
  .grid-2 { display: grid; grid-template-columns: repeat(2,1fr); gap: 14px; }
  .grid-3 { display: grid; grid-template-columns: repeat(3,1fr); gap: 14px; margin-bottom: 20px; }

  /* ── Metric tile ──────────────────────────────── */
  .metric-tile {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 18px;
  }
  .metric-label { font-size: 10px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); margin-bottom: 6px; }
  .metric-value { font-size: 26px; font-weight: 700; font-family: var(--mono); line-height: 1; }
  .metric-sub { font-size: 10px; color: var(--muted); margin-top: 4px; font-family: var(--mono); }
  .metric-tile.accent .metric-value { color: var(--accent); }
  .metric-tile.green  .metric-value { color: var(--green); }
  .metric-tile.yellow .metric-value { color: var(--yellow); }
  .metric-tile.red    .metric-value { color: var(--red); }

  /* ── Gauge / progress bar ─────────────────────── */
  .gauge-wrap { margin-top: 8px; }
  .gauge-track {
    height: 6px; border-radius: 3px; background: var(--bg3);
    overflow: hidden;
  }
  .gauge-fill {
    height: 100%; border-radius: 3px;
    transition: width .5s ease;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
  }
  .gauge-fill.green  { background: linear-gradient(90deg,#16a34a,#22c55e); }
  .gauge-fill.yellow { background: linear-gradient(90deg,#d97706,#f59e0b); }
  .gauge-fill.red    { background: linear-gradient(90deg,#b91c1c,#ef4444); }

  /* ── Sparkline canvas ─────────────────────────── */
  .spark { width: 100%; height: 48px; display: block; margin-top: 6px; }

  /* ── Replica display ──────────────────────────── */
  .replicas-row {
    display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap;
  }
  .replica-chip {
    width: 32px; height: 32px; border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-family: var(--mono); font-weight: 600;
    border: 1px solid var(--border);
    transition: all .3s;
  }
  .replica-chip.active { background: rgba(79,142,247,.2); border-color: var(--accent); color: var(--accent); }
  .replica-chip.inactive { background: var(--bg3); color: var(--muted); }

  /* ── Forecast chart ───────────────────────────── */
  .chart-wrap { position: relative; height: 220px; margin-top: 10px; }
  canvas.chart-main { width: 100% !important; height: 100% !important; }

  /* ── Sliders (load control) ───────────────────── */
  .class-block { margin-bottom: 20px; }
  .class-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 8px;
  }
  .class-name { font-weight: 600; font-size: 13px; }
  .class-badge {
    font-size: 10px; padding: 2px 8px; border-radius: 20px;
    background: rgba(79,142,247,.15); color: var(--accent);
    font-family: var(--mono);
  }
  .class-desc { font-size: 11px; color: var(--muted); margin-bottom: 10px; line-height: 1.5; }
  .routes-row { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 12px; }
  .route-tag {
    font-family: var(--mono); font-size: 10px; padding: 3px 8px;
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: 4px; color: var(--muted);
  }
  .slider-row { display: flex; align-items: center; gap: 12px; }
  .slider-row label { font-size: 11px; color: var(--muted); white-space: nowrap; width: 100px; }
  input[type=range] {
    flex: 1; -webkit-appearance: none; height: 4px;
    background: var(--bg3); border-radius: 2px; outline: none; cursor: pointer;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 14px; height: 14px;
    border-radius: 50%; background: var(--accent); cursor: pointer;
    border: 2px solid var(--bg);
  }
  .slider-val {
    width: 42px; text-align: right; font-family: var(--mono);
    font-size: 12px; font-weight: 600; color: var(--text);
  }
  .apply-btn {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 9px 20px; border-radius: 7px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff; font-size: 13px; font-weight: 600; cursor: pointer;
    border: none; transition: opacity .2s; margin-top: 6px;
  }
  .apply-btn:hover { opacity: .85; }

  /* ── Phi bar ──────────────────────────────────── */
  .phi-bar { display: flex; height: 10px; border-radius: 5px; overflow: hidden; margin-top: 8px; }
  .phi-seg { transition: width .4s ease; }
  .phi-seg.c1 { background: var(--accent); }
  .phi-seg.c2 { background: var(--green); }
  .phi-seg.c3 { background: var(--yellow); }
  .phi-legend { display: flex; gap: 14px; margin-top: 6px; }
  .phi-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 4px; }

  /* ── Training chart ───────────────────────────── */
  .train-chart-wrap { height: 260px; position: relative; margin-top: 10px; }

  /* ── Comparison table ─────────────────────────── */
  .cmp-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
  .cmp-table th {
    background: var(--bg3); padding: 10px 14px;
    text-align: left; font-size: 10px; text-transform: uppercase;
    letter-spacing: .1em; color: var(--muted); font-weight: 600;
    border-bottom: 1px solid var(--border);
  }
  .cmp-table td {
    padding: 11px 14px; font-size: 13px; border-bottom: 1px solid var(--border);
    font-family: var(--mono);
  }
  .cmp-table tr:last-child td { border-bottom: none; }
  .cmp-table tr.highlight td { background: rgba(79,142,247,.07); }
  .cmp-table tr.highlight td:first-child { border-left: 3px solid var(--accent); }
  .badge-best { font-size: 9px; background: rgba(34,197,94,.15); color: var(--green);
    padding: 2px 6px; border-radius: 3px; margin-left: 6px; }
  .run-btn {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 9px 20px; border-radius: 7px;
    background: var(--bg3); border: 1px solid var(--border);
    color: var(--text); font-size: 13px; font-weight: 600; cursor: pointer;
    transition: all .2s; margin-bottom: 18px;
  }
  .run-btn:hover { border-color: var(--accent); color: var(--accent); }

  /* ── Saturation alert ─────────────────────────── */
  .sat-alert {
    display: none; align-items: center; gap: 10px;
    background: rgba(239,68,68,.1); border: 1px solid rgba(239,68,68,.3);
    color: var(--red); border-radius: 8px; padding: 10px 16px;
    font-size: 12px; margin-bottom: 16px;
  }
  .sat-alert.visible { display: flex; }

  /* ── Toast ────────────────────────────────────── */
  .toast {
    position: fixed; bottom: 24px; right: 24px;
    background: var(--bg2); border: 1px solid var(--green);
    color: var(--green); border-radius: 8px;
    padding: 10px 18px; font-size: 12px; font-family: var(--mono);
    opacity: 0; transform: translateY(10px);
    transition: all .3s; pointer-events: none; z-index: 999;
  }
  .toast.show { opacity: 1; transform: translateY(0); }

  /* ── Scrollbar ────────────────────────────────── */
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>
<div class="shell">

<!-- ── Sidebar ───────────────────────────────────────────────────────── -->
<aside class="sidebar">
  <div class="sidebar-logo">
    <div class="title">Стенд прогнозирования</div>
    <div class="sub">Δt = 5 мин · h = 3 шага</div>
  </div>
  <nav class="nav">
    <div class="nav-item active" data-tab="monitor">
      <span class="nav-icon">📊</span> Мониторинг
    </div>
    <div class="nav-item" data-tab="load">
      <span class="nav-icon">🎛️</span> Нагрузка
    </div>
    <div class="nav-item" data-tab="training">
      <span class="nav-icon">🧠</span> Обучение
    </div>
    <div class="nav-item" data-tab="compare">
      <span class="nav-icon">📋</span> Сравнение
    </div>
  </nav>
  <div class="status-bar">
    <span class="status-dot"></span>
    <span id="status-text" style="color:var(--muted)">Получение данных…</span>
  </div>
</aside>

<!-- ── Main content ───────────────────────────────────────────────────── -->
<main class="main">

  <!-- ══ 1. МОНИТОРИНГ ══ -->
  <section class="section active" id="tab-monitor">
    <div class="section-title">Мониторинг системы</div>
    <div class="section-desc">Текущие значения вектора m_t, прогноз на горизонте h=3, число активных реплик</div>

    <div id="sat-alert" class="sat-alert">
      ⚠ RESOURCE_SATURATION — r_req превышает r_max. Масштабирование заблокировано.
    </div>

    <!-- Метрики m_t -->
    <div class="grid-4" style="margin-bottom:20px">
      <div class="metric-tile accent">
        <div class="metric-label">cpu_t (цель 70%)</div>
        <div class="metric-value" id="cpu-val">—</div>
        <div class="gauge-wrap"><div class="gauge-track"><div class="gauge-fill" id="cpu-bar" style="width:0%"></div></div></div>
        <div class="metric-sub" id="cpu-sub">утилизация процессора</div>
      </div>
      <div class="metric-tile green">
        <div class="metric-label">mem_t</div>
        <div class="metric-value" id="mem-val">—</div>
        <div class="gauge-wrap"><div class="gauge-track"><div class="gauge-fill green" id="mem-bar" style="width:0%"></div></div></div>
        <div class="metric-sub">оперативная память</div>
      </div>
      <div class="metric-tile yellow">
        <div class="metric-label">rps_t</div>
        <div class="metric-value" id="rps-val">—</div>
        <div class="metric-sub">запросов / сек</div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">lat_t (P95)</div>
        <div class="metric-value" id="lat-val">—</div>
        <div class="metric-sub">мс</div>
      </div>
    </div>

    <div class="grid-3">
      <!-- Граф cpu + прогноз -->
      <div class="card" style="grid-column:span 2">
        <div class="card-title">История cpu_t + прогноз ĉpu_{t+k} с доверительным интервалом</div>
        <div class="chart-wrap">
          <canvas id="forecast-chart"></canvas>
        </div>
      </div>

      <!-- Реплики + состав классов -->
      <div style="display:flex;flex-direction:column;gap:14px">
        <div class="card">
          <div class="card-title">Активные реплики (r_fin)</div>
          <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:12px">
            <span class="metric-value" id="replica-count" style="font-size:36px;color:var(--accent)">2</span>
            <span style="color:var(--muted);font-size:11px">/ r_max = 8</span>
          </div>
          <div class="replicas-row" id="replicas-row"></div>
          <div style="margin-top:10px;font-size:10px;color:var(--muted);font-family:var(--mono)" id="decision-action">—</div>
        </div>
        <div class="card">
          <div class="card-title">Состав потока φ_t</div>
          <div class="phi-bar">
            <div class="phi-seg c1" id="phi1" style="width:33%"></div>
            <div class="phi-seg c2" id="phi2" style="width:33%"></div>
            <div class="phi-seg c3" id="phi3" style="width:34%"></div>
          </div>
          <div class="phi-legend" style="margin-top:8px">
            <span style="font-size:11px"><span class="phi-dot" style="background:var(--accent)"></span>Класс 1: <span id="phi1-pct" style="font-family:var(--mono)">33%</span></span>
            <span style="font-size:11px"><span class="phi-dot" style="background:var(--green)"></span>Класс 2: <span id="phi2-pct" style="font-family:var(--mono)">33%</span></span>
            <span style="font-size:11px"><span class="phi-dot" style="background:var(--yellow)"></span>Класс 3: <span id="phi3-pct" style="font-family:var(--mono)">34%</span></span>
          </div>
        </div>
        <div class="card">
          <div class="card-title">Доверительный интервал</div>
          <div style="font-size:10px;color:var(--muted);margin-bottom:8px">q̂_{0.025} — q̂_{0.975}</div>
          <div style="display:flex;justify-content:space-between;font-family:var(--mono);font-size:13px">
            <span style="color:var(--green)" id="q-lower">—</span>
            <span style="color:var(--muted)">↔</span>
            <span style="color:var(--red)" id="q-upper">—</span>
          </div>
          <div class="gauge-wrap" style="margin-top:8px"><div class="gauge-track" style="height:10px;border-radius:5px">
            <div id="ci-fill" style="height:100%;border-radius:5px;background:linear-gradient(90deg,var(--green),var(--red));transition:width .5s;margin-left:0" id="ci-fill"></div>
          </div></div>
          <div style="font-size:10px;color:var(--muted);margin-top:6px">err_t = <span id="err-val" style="font-family:var(--mono)">—</span></div>
        </div>
      </div>
    </div>
  </section>

  <!-- ══ 2. УПРАВЛЕНИЕ НАГРУЗКОЙ ══ -->
  <section class="section" id="tab-load">
    <div class="section-title">Управление нагрузкой</div>
    <div class="section-desc">Задайте интенсивность запросов каждого функционального класса (виртуальных пользователей Locust)</div>

    <div class="grid-2">
      <div>
        <div class="class-block card">
          <div class="class-header">
            <span class="class-name">Класс 1 — Вычислительные</span>
            <span class="class-badge">cpu ≈ 0.65–0.85</span>
          </div>
          <div class="class-desc">Арифметические вычисления в памяти: решето Эратосфена, факторизация, π Монте-Карло. Высокая загрузка CPU, минимальный I/O.</div>
          <div class="routes-row">
            <span class="route-tag">GET /compute/light</span>
            <span class="route-tag">GET /compute/medium</span>
            <span class="route-tag">GET /compute/heavy</span>
          </div>
          <div class="slider-row">
            <label>Пользователей:</label>
            <input type="range" min="0" max="50" value="10" id="s-class1" oninput="document.getElementById('v-class1').textContent=this.value"/>
            <span class="slider-val" id="v-class1">10</span>
          </div>
        </div>

        <div class="class-block card" style="margin-top:14px">
          <div class="class-header">
            <span class="class-name">Класс 2 — База данных</span>
            <span class="class-badge">cpu ≈ 0.20–0.35</span>
          </div>
          <div class="class-desc">Транзакции PostgreSQL 15 через pgBouncer: SELECT по PK из 10⁶ строк, INSERT, агрегация с оконными функциями. Высокое I/O, потребление соединений.</div>
          <div class="routes-row">
            <span class="route-tag">GET /db/read</span>
            <span class="route-tag">GET /db/write</span>
            <span class="route-tag">GET /db/aggregate</span>
          </div>
          <div class="slider-row">
            <label>Пользователей:</label>
            <input type="range" min="0" max="50" value="10" id="s-class2" oninput="document.getElementById('v-class2').textContent=this.value"/>
            <span class="slider-val" id="v-class2">10</span>
          </div>
        </div>

        <div class="class-block card" style="margin-top:14px">
          <div class="class-header">
            <span class="class-name">Класс 3 — Управление памятью</span>
            <span class="class-badge">cpu ≈ 0.30–0.55</span>
          </div>
          <div class="class-desc">Интенсивное использование heap: выделение NumPy 50 МБ, потоковая обработка 10⁵ объектов, принудительный gc.collect(). Паузы GC в хвосте задержек.</div>
          <div class="routes-row">
            <span class="route-tag">GET /memory/alloc</span>
            <span class="route-tag">GET /memory/process</span>
            <span class="route-tag">GET /memory/gc</span>
          </div>
          <div class="slider-row">
            <label>Пользователей:</label>
            <input type="range" min="0" max="50" value="10" id="s-class3" oninput="document.getElementById('v-class3').textContent=this.value"/>
            <span class="slider-val" id="v-class3">10</span>
          </div>
        </div>

        <button class="apply-btn" onclick="applyLoad()">▶ Применить нагрузку</button>
      </div>

      <div>
        <div class="card">
          <div class="card-title">Текущее распределение φ_t</div>
          <div class="phi-bar" style="height:16px;border-radius:8px">
            <div class="phi-seg c1" id="load-phi1" style="width:33%"></div>
            <div class="phi-seg c2" id="load-phi2" style="width:33%"></div>
            <div class="phi-seg c3" id="load-phi3" style="width:34%"></div>
          </div>
          <div class="phi-legend" style="margin-top:10px">
            <div style="font-size:12px"><span class="phi-dot" style="background:var(--accent)"></span>φ₁ = <span id="load-p1" style="font-family:var(--mono)">33%</span></div>
            <div style="font-size:12px"><span class="phi-dot" style="background:var(--green)"></span>φ₂ = <span id="load-p2" style="font-family:var(--mono)">33%</span></div>
            <div style="font-size:12px"><span class="phi-dot" style="background:var(--yellow)"></span>φ₃ = <span id="load-p3" style="font-family:var(--mono)">33%</span></div>
          </div>
        </div>

        <div class="card" style="margin-top:14px">
          <div class="card-title">Ограничения инфраструктуры (формула 3.3)</div>
          <div style="font-family:var(--mono);font-size:12px;line-height:2;color:var(--muted)">
            <div>r_max_cluster = <span style="color:var(--text)">8</span></div>
            <div>max_conn = <span style="color:var(--text)">100</span></div>
            <div>conn_reserve = <span style="color:var(--text)">10</span></div>
            <div>pool_size = <span style="color:var(--text)">5</span></div>
            <div style="border-top:1px solid var(--border);padding-top:8px;margin-top:4px">
              r_max_db = ⌊(100−10)/5⌋ = <span style="color:var(--green)">18</span>
            </div>
            <div>r_max = min(8, 18) = <span style="color:var(--accent)">8</span></div>
          </div>
        </div>

        <div class="card" style="margin-top:14px">
          <div class="card-title">Параметры метода (таблица 3.1)</div>
          <div style="font-family:var(--mono);font-size:11px;line-height:2;color:var(--muted)">
            <div>Δt = <span style="color:var(--text)">5 мин</span> · p = <span style="color:var(--text)">288</span></div>
            <div>cpu_target = <span style="color:var(--text)">0.70</span> · ε = <span style="color:var(--text)">0.05</span></div>
            <div>τ = <span style="color:var(--text)">4 шага</span> · β = <span style="color:var(--text)">0.3</span></div>
            <div>STL: w_a=48 · w_norm=60 · N_c=7</div>
            <div>GRU: d_in=69 · d_h=64 · layers=2</div>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- ══ 3. ОБУЧЕНИЕ ══ -->
  <section class="section" id="tab-training">
    <div class="section-title">Обучение нейросетевой компоненты</div>
    <div class="section-desc">Динамика квантильной функции потерь (pinball loss) по эпохам на обучающей и валидационной выборках</div>

    <div class="grid-4" style="margin-bottom:20px">
      <div class="metric-tile accent">
        <div class="metric-label">Остановлено на эпохе</div>
        <div class="metric-value" id="tr-stopped">—</div>
        <div class="metric-sub">ранняя остановка (patience=10)</div>
      </div>
      <div class="metric-tile green">
        <div class="metric-label">Лучший val_loss</div>
        <div class="metric-value" id="tr-best">—</div>
        <div class="metric-sub">квантильная функция потерь</div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Архитектура</div>
        <div class="metric-value" style="font-size:14px;margin-top:4px">2×GRU(64)</div>
        <div class="metric-sub">d_in=69 · dropout=0.10</div>
      </div>
      <div class="metric-tile yellow">
        <div class="metric-label">Оптимизатор</div>
        <div class="metric-value" style="font-size:14px;margin-top:4px">Adam</div>
        <div class="metric-sub">lr=0.001 · decay=0.99</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Кривые обучения — суммарная квантильная функция потерь L_q (формула 3.13)</div>
      <div class="train-chart-wrap">
        <canvas id="train-chart"></canvas>
      </div>
      <div style="display:flex;gap:20px;margin-top:10px;font-size:11px">
        <span><span style="display:inline-block;width:24px;height:2px;background:var(--accent);margin-right:6px;vertical-align:middle"></span>Обучающая выборка</span>
        <span><span style="display:inline-block;width:24px;height:2px;background:var(--accent2);border-top:2px dashed var(--accent2);margin-right:6px;vertical-align:middle"></span>Валидационная выборка</span>
      </div>
    </div>

    <div class="card" style="margin-top:16px">
      <div class="card-title">Квантильные выходы нейросети (порядки 0.025 / 0.500 / 0.975)</div>
      <div style="font-size:12px;color:var(--muted);line-height:2">
        Нижний квантиль q̂<sub>0.025</sub> используется для вычисления адаптивного порога δ_t (формула 3.17).<br>
        Медиана q̂<sub>0.500</sub> образует точечный прогноз ĉpu_{t+k} (формула 3.14).<br>
        Верхний квантиль q̂<sub>0.975</sub> используется для расчёта числа реплик r_req (формула 3.7).
      </div>
    </div>
  </section>

  <!-- ══ 4. СРАВНЕНИЕ ══ -->
  <section class="section" id="tab-compare">
    <div class="section-title">Сравнение методов прогнозирования</div>
    <div class="section-desc">Метрики точности прогноза и эффективности управления ресурсами на наборе Alibaba Cluster Trace 2018, h=3</div>

    <button class="run-btn" onclick="loadComparison()">
      ▶ Загрузить результаты сравнения
    </button>

    <div id="cmp-wrap">
      <div class="card">
        <div class="card-title">Метрики точности прогнозирования (таблица 4.4)</div>
        <table class="cmp-table" id="cmp-forecast"></table>
      </div>
      <div class="card" style="margin-top:16px">
        <div class="card-title">Метрики управления ресурсами (таблица 4.6)</div>
        <table class="cmp-table" id="cmp-mgmt"></table>
      </div>
    </div>
  </section>

</main>
</div>

<!-- Toast -->
<div class="toast" id="toast"></div>

<script>
// ── Chart.js через CDN ──────────────────────────────────────────────────────
const chartScript = document.createElement('script');
chartScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js';
document.head.appendChild(chartScript);

// ── Навигация ───────────────────────────────────────────────────────────────
document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', () => {
    document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    item.classList.add('active');
    document.getElementById('tab-' + item.dataset.tab).classList.add('active');
    if (item.dataset.tab === 'training') loadTraining();
    if (item.dataset.tab === 'compare') {}
  });
});

// ── Toast ────────────────────────────────────────────────────────────────────
function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}

// ── Forecast chart ───────────────────────────────────────────────────────────
let fcChart = null;
const cpuHistory = [], fcHistory = [], tsHistory = [];
const MAX_HIST = 60;

function initForecastChart() {
  const ctx = document.getElementById('forecast-chart');
  if (!ctx || fcChart) return;
  fcChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'cpu_t (факт)', data: [], borderColor: '#4f8ef7',
          backgroundColor: 'rgba(79,142,247,.08)', borderWidth: 2,
          fill: true, tension: 0.3, pointRadius: 0 },
        { label: 'ĉpu_t+1 (прогноз)', data: [], borderColor: '#7c5cf8',
          borderWidth: 2, borderDash: [5,3], pointRadius: 0, tension: 0.3 },
        { label: 'q̂_0.975', data: [], borderColor: 'rgba(239,68,68,.4)',
          backgroundColor: 'rgba(239,68,68,.06)', borderWidth: 1,
          fill: '+1', pointRadius: 0 },
        { label: 'q̂_0.025', data: [], borderColor: 'rgba(34,197,94,.4)',
          backgroundColor: 'rgba(34,197,94,.06)', borderWidth: 1,
          fill: false, pointRadius: 0 },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      scales: {
        x: { grid: { color: '#1f2433' }, ticks: { color: '#6b7280', font: { size: 10 },
          maxTicksLimit: 10 } },
        y: { min: 0, max: 1, grid: { color: '#1f2433' },
          ticks: { color: '#6b7280', font: { size: 10 },
            callback: v => (v*100).toFixed(0)+'%' } }
      },
      plugins: { legend: { display: false } }
    }
  });
}

function updateForecastChart(m) {
  if (!fcChart) { if (typeof Chart !== 'undefined') initForecastChart(); return; }
  const label = new Date(m.ts * 1000).toLocaleTimeString('ru-RU', {hour:'2-digit',minute:'2-digit',second:'2-digit'});
  const ds = fcChart.data;
  ds.labels.push(label);
  ds.datasets[0].data.push(m.cpu);
  ds.datasets[1].data.push(m.forecast[0].cpu_hat);
  ds.datasets[2].data.push(m.forecast[0].q_hi);
  ds.datasets[3].data.push(m.forecast[0].q_lo);
  if (ds.labels.length > MAX_HIST) {
    ds.labels.shift();
    ds.datasets.forEach(d => d.data.shift());
  }
  fcChart.update('none');
}

// ── Metric updater ───────────────────────────────────────────────────────────
function updateMetrics(m) {
  const pct = v => (v*100).toFixed(1)+'%';
  document.getElementById('cpu-val').textContent = pct(m.cpu);
  document.getElementById('mem-val').textContent = pct(m.mem);
  document.getElementById('rps-val').textContent = m.rps;
  document.getElementById('lat-val').textContent = m.lat;
  document.getElementById('err-val').textContent = (m.err*100).toFixed(2)+'%';

  const cpuPct = Math.min(m.cpu * 100, 100);
  document.getElementById('cpu-bar').style.width = cpuPct + '%';
  const cpuFill = document.getElementById('cpu-bar');
  cpuFill.className = 'gauge-fill' + (cpuPct > 85 ? ' red' : cpuPct > 70 ? ' yellow' : '');

  document.getElementById('mem-bar').style.width = Math.min(m.mem*100,100) + '%';

  document.getElementById('q-lower').textContent = pct(m.q_lower);
  document.getElementById('q-upper').textContent = pct(m.q_upper);

  // ДИ заливка
  const lo = m.q_lower * 100, hi = m.q_upper * 100;
  const fill = document.getElementById('ci-fill');
  fill.style.marginLeft = lo + '%';
  fill.style.width = Math.max(hi - lo, 0) + '%';

  // Состав классов φ_t
  const phi = m.phi;
  const total = phi.reduce((a,b)=>a+b,0.001);
  const p1 = (phi[0]/total*100).toFixed(0), p2 = (phi[1]/total*100).toFixed(0), p3 = (phi[2]/total*100).toFixed(0);
  document.getElementById('phi1').style.width = p1 + '%';
  document.getElementById('phi2').style.width = p2 + '%';
  document.getElementById('phi3').style.width = p3 + '%';
  document.getElementById('phi1-pct').textContent = p1 + '%';
  document.getElementById('phi2-pct').textContent = p2 + '%';
  document.getElementById('phi3-pct').textContent = p3 + '%';

  // Реплики
  const r = m.replicas;
  document.getElementById('replica-count').textContent = r;
  const row = document.getElementById('replicas-row');
  row.innerHTML = '';
  for (let i = 1; i <= 8; i++) {
    const chip = document.createElement('div');
    chip.className = 'replica-chip ' + (i <= r ? 'active' : 'inactive');
    chip.textContent = 'r' + i;
    row.appendChild(chip);
  }

  // Сигнал насыщения
  document.getElementById('sat-alert').classList.toggle('visible', !!m.saturation);

  document.getElementById('status-text').textContent =
    new Date().toLocaleTimeString('ru-RU') + ' · ok';

  updateForecastChart(m);
}

// ── Polling ──────────────────────────────────────────────────────────────────
async function poll() {
  try {
    const r = await fetch('/api/metrics/current');
    const m = await r.json();
    if (m.cpu !== undefined) updateMetrics(m);
  } catch(e) { document.getElementById('status-text').textContent = 'Нет соединения'; }
}
setInterval(poll, 3000);
setTimeout(poll, 500);
setTimeout(() => { if (typeof Chart !== 'undefined') initForecastChart(); }, 800);

// ── Apply load ───────────────────────────────────────────────────────────────
async function applyLoad() {
  const c1 = +document.getElementById('s-class1').value;
  const c2 = +document.getElementById('s-class2').value;
  const c3 = +document.getElementById('s-class3').value;

  await fetch('/api/load', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ class1: c1, class2: c2, class3: c3 })
  });

  const total = c1 + c2 + c3 + 0.001;
  const p1 = (c1/total*100).toFixed(0), p2 = (c2/total*100).toFixed(0), p3 = (c3/total*100).toFixed(0);
  document.getElementById('load-phi1').style.width = p1 + '%';
  document.getElementById('load-phi2').style.width = p2 + '%';
  document.getElementById('load-phi3').style.width = p3 + '%';
  document.getElementById('load-p1').textContent = p1 + '%';
  document.getElementById('load-p2').textContent = p2 + '%';
  document.getElementById('load-p3').textContent = p3 + '%';

  showToast(`✓ Нагрузка применена: cls1=${c1} cls2=${c2} cls3=${c3}`);
}

// ── Training chart ────────────────────────────────────────────────────────────
let trainChart = null;
async function loadTraining() {
  const r = await fetch('/api/training');
  const d = await r.json();
  document.getElementById('tr-stopped').textContent = d.stopped_epoch;
  document.getElementById('tr-best').textContent = d.best_val_loss.toFixed(4);

  const ctx = document.getElementById('train-chart');
  if (trainChart) trainChart.destroy();
  trainChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: d.epochs,
      datasets: [
        { label: 'train_loss', data: d.train_loss, borderColor: '#4f8ef7',
          backgroundColor: 'rgba(79,142,247,.07)', fill: true,
          borderWidth: 2, tension: 0.3, pointRadius: 0 },
        { label: 'val_loss', data: d.val_loss, borderColor: '#7c5cf8',
          borderDash: [5,3], borderWidth: 2, pointRadius: 0, tension: 0.3 },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 600 },
      scales: {
        x: { title: { display: true, text: 'Эпоха', color: '#6b7280', font: { size: 11 } },
             grid: { color: '#1f2433' }, ticks: { color: '#6b7280', font: { size: 10 } } },
        y: { title: { display: true, text: 'L_q (квантильные потери)', color: '#6b7280', font: { size: 11 } },
             grid: { color: '#1f2433' }, ticks: { color: '#6b7280', font: { size: 10 } } }
      },
      plugins: {
        legend: { labels: { color: '#9ca3af', font: { size: 11 } } },
        tooltip: { backgroundColor: '#1a1e2a', borderColor: '#252935', borderWidth: 1,
          titleColor: '#e2e6f0', bodyColor: '#9ca3af' }
      }
    }
  });
}

// ── Comparison ─────────────────────────────────────────────────────────────
async function loadComparison() {
  const r = await fetch('/api/comparison');
  const rows = await r.json();

  const fcTable = document.getElementById('cmp-forecast');
  fcTable.innerHTML = `<tr>
    <th>Метод</th><th>MAE</th><th>RMSE</th><th>MAPE, %</th><th>Покрытие 95%, %</th>
  </tr>` + rows.map(row => `<tr class="${row.highlight ? 'highlight' : ''}">
    <td>${row.method}${row.highlight ? '<span class="badge-best">best</span>':''}</td>
    <td>${row.mae != null ? row.mae.toFixed(3) : '—'}</td>
    <td>${row.rmse != null ? row.rmse.toFixed(3) : '—'}</td>
    <td>${row.mape != null ? row.mape.toFixed(1) : '—'}</td>
    <td>${row.coverage != null ? row.coverage.toFixed(1) : '—'}</td>
  </tr>`).join('');

  const mgmtTable = document.getElementById('cmp-mgmt');
  mgmtTable.innerHTML = `<tr>
    <th>Метод</th><th>Нарушения SLA, %</th><th>Ср. утилизация, %</th><th>Операций масштаб.</th>
  </tr>` + rows.map(row => `<tr class="${row.highlight ? 'highlight' : ''}">
    <td>${row.method}${row.highlight ? '<span class="badge-best">best</span>':''}</td>
    <td style="color:${row.sla_pct<=5?'var(--green)':'var(--red)'}">${row.sla_pct.toFixed(1)}</td>
    <td>${row.util_pct.toFixed(1)}</td>
    <td>${row.scale_ops}</td>
  </tr>`).join('');

  showToast('✓ Результаты сравнения загружены');
}
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
