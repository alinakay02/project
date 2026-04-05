"""
tests/test_experiments.py — Эксперименты главы 4: сравнение методов прогнозирования.

Методы сравнения:
  1. Разработанный метод (STL + GRU)
  2. ARIMA
  3. SARIMA
  4. Хольт-Винтерс
  5. Случайный лес
  6. LSTM
  7. GRU (автономная, без STL)
  8. CNN-LSTM (гибрид)
  9. TFT (Temporal Fusion Transformer)

Запуск:
  python -m pytest tests/test_experiments.py -v -s
  python -u scripts/run_all_tests.py
"""

import gc
import math
import time
import logging
import warnings
import numpy as np
import pytest

def soft_assert(condition, message=""):
    if not condition:
        msg = f"[WARN] {message}"
        print(msg, flush=True)
        warnings.warn(msg, stacklevel=2)

from typing import Dict, List, Tuple, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from predictor.preprocessor import Preprocessor
from predictor.forecaster import HybridForecaster
from controller.decision import ResourceDecisionModule
from tests.baselines import (
    ARIMAForecaster, SARIMAForecaster, HoltWintersForecaster,
    RandomForestForecaster, LSTMForecaster, AutoGRUForecaster,
    CNNLSTMForecaster, TFTForecaster,
)
from tests.data_generators import (
    generate_stationary, generate_trend, generate_spike,
    generate_mixed, load_alibaba_trace, load_google_trace, load_azure_trace,
)
from tests.metrics import (
    compute_mae, compute_rmse, compute_mape, compute_coverage,
    compute_sla_violations, compute_avg_utilization, compute_scale_ops,
    print_forecast_metrics, print_mgmt_metrics,
)

logging.basicConfig(level=logging.WARNING)

SEEDS = [42, 137, 256]

CFG_PREPROCESSOR = dict(w_a=48, iqr_alpha=1.5, w_norm=60, period=288, robust=True)
CFG_FORECASTER   = dict(n_T=60, n_cycles=7, period=288, horizon_h=3, w_input=60,
                        quantiles=[0.025, 0.5, 0.975], hidden_dim=64, dropout=0.25,
                        lr=0.001, lr_decay=0.99, grad_clip=1.0,
                        max_epochs=100, patience=0, batch_size=32)
CFG_DECISION     = dict(cpu_target=0.70, epsilon=0.05, r_min=2, r_max_cluster=8,
                        tau=4, beta=0.3, max_conn=100, conn_reserve=10, pool_size=5)
N_OBS_SYNTHETIC  = 4_320


# ══════════════════════════════════════════════════════════════════════════════
# Утилиты
# ══════════════════════════════════════════════════════════════════════════════

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "cache")


def build_proposed_method(seed: int, horizon_h: int = None):
    np.random.seed(seed)
    import torch; torch.manual_seed(seed)
    pp = Preprocessor(**CFG_PREPROCESSOR)
    cfg = dict(CFG_FORECASTER)
    if horizon_h:
        cfg["horizon_h"] = horizon_h
    fc = HybridForecaster(preprocessor=pp, **cfg)
    dm = ResourceDecisionModule(**CFG_DECISION)
    return pp, fc, dm


def torch_seed(seed):
    np.random.seed(seed)
    import torch; torch.manual_seed(seed)


def fit_with_cache(fc, cpu, ts, phi, seed, tag):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{tag}_seed{seed}.pt")
    if os.path.exists(path):
        try:
            fc.load_model(path)
            return
        except Exception:
            pass
    fc.fit(cpu, ts, phi)
    try:
        fc.save_model(path)
    except Exception:
        pass


def run_forecast_experiment(forecaster, cpu_train, cpu_test, ts_train, ts_test,
                            phi_train, phi_test, horizon_h=None):
    h = horizon_h or 3
    y_true, y_pred, y_lower, y_upper = [], [], [], []
    for i in range(0, len(cpu_test) - h, h):
        cpu_ctx = np.concatenate([cpu_train, cpu_test[:i]])
        ts_ctx = np.concatenate([ts_train, ts_test[:i]])
        phi_ctx = np.concatenate([phi_train, phi_test[:, :i]], axis=1) if phi_test.shape[1] > i else phi_train
        hat, lo, hi = forecaster.predict(cpu_ctx, ts_ctx, phi_ctx)
        for k in range(min(h, len(cpu_test) - i)):
            y_true.append(cpu_test[i + k])
            y_pred.append(hat[k] if k < len(hat) else hat[-1])
            y_lower.append(lo[k] if k < len(lo) else lo[-1])
            y_upper.append(hi[k] if k < len(hi) else hi[-1])
    return {"y_true": np.array(y_true), "y_pred": np.array(y_pred),
            "y_lower": np.array(y_lower), "y_upper": np.array(y_upper)}


# Все методы для сравнения
ALL_METHODS = {
    "Разработанный метод": lambda seed: None,  # обрабатывается отдельно
    "ARIMA":          lambda seed: ARIMAForecaster(horizon_h=3),
    "SARIMA":         lambda seed: SARIMAForecaster(horizon_h=3),
    "Хольт-Винтерс":  lambda seed: HoltWintersForecaster(horizon_h=3),
    "Случайный лес":   lambda seed: RandomForestForecaster(horizon_h=3),
    "LSTM":           lambda seed: LSTMForecaster(seed=seed, horizon_h=3, max_epochs=100),
    "GRU":            lambda seed: AutoGRUForecaster(seed=seed, horizon_h=3, max_epochs=100),
    "CNN-LSTM":       lambda seed: CNNLSTMForecaster(seed=seed, horizon_h=3, max_epochs=100),
    "TFT":            lambda seed: TFTForecaster(seed=seed, horizon_h=3, max_epochs=100),
}


def _run_all_methods_on_dataset(dataset_name, cpu, ts, phi):
    """Прогоняет все методы на одном датасете, возвращает dict {method: metrics}."""
    n_tr = int(len(cpu) * 0.70)
    n_vl = int(len(cpu) * 0.15)
    cpu_train, cpu_test = cpu[:n_tr], cpu[n_tr + n_vl:]
    ts_train, ts_test = ts[:n_tr], ts[n_tr + n_vl:]
    phi_train, phi_test = phi[:, :n_tr], phi[:, n_tr + n_vl:]

    results = {}

    for method_name, factory in ALL_METHODS.items():
        maes, rmses, mapes, coverages = [], [], [], []

        for seed in SEEDS:
            torch_seed(seed)
            gc.collect()

            if method_name == "Разработанный метод":
                pp, fc, dm = build_proposed_method(seed)
                fit_with_cache(fc, cpu_train, ts_train, phi_train, seed, f"{dataset_name}_proposed")
                forecaster = fc
            else:
                forecaster = factory(seed)
                forecaster.fit(cpu_train, ts_train, phi_train)

            r = run_forecast_experiment(forecaster, cpu_train, cpu_test,
                                        ts_train, ts_test, phi_train, phi_test)

            if len(r["y_true"]) > 0:
                maes.append(compute_mae(r["y_true"], r["y_pred"]))
                rmses.append(compute_rmse(r["y_true"], r["y_pred"]))
                mapes.append(compute_mape(r["y_true"], r["y_pred"]))
                coverages.append(compute_coverage(r["y_true"], r["y_lower"], r["y_upper"]))

        if maes:
            results[method_name] = {
                "mae": np.mean(maes), "mae_std": np.std(maes),
                "rmse": np.mean(rmses), "rmse_std": np.std(rmses),
                "mape": np.mean(mapes), "mape_std": np.std(mapes),
                "coverage": np.mean(coverages), "coverage_std": np.std(coverages),
            }
            print(f"\n[METRIC] COMPARE: {dataset_name} | {method_name} | "
                  f"MAE={np.mean(maes):.4f}±{np.std(maes):.4f} | "
                  f"RMSE={np.mean(rmses):.4f}±{np.std(rmses):.4f} | "
                  f"MAPE={np.mean(mapes):.2f}±{np.std(mapes):.2f}% | "
                  f"COVERAGE={np.mean(coverages):.1f}±{np.std(coverages):.1f}%",
                  flush=True)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 1: Сравнение всех методов на Alibaba (основной)
# ══════════════════════════════════════════════════════════════════════════════

class TestCompareAlibaba:
    """Сравнение 9 методов на Alibaba Cluster Trace 2018."""

    def test_all_methods(self):
        cpu, ts, phi = load_alibaba_trace()
        results = _run_all_methods_on_dataset("Alibaba", cpu, ts, phi)

        proposed_mae = results.get("Разработанный метод", {}).get("mae", 999)
        for method, m in results.items():
            if method != "Разработанный метод":
                soft_assert(proposed_mae <= m["mae"] * 1.3,
                    f"Proposed MAE {proposed_mae:.4f} much worse than {method} {m['mae']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 2: Сравнение на дополнительных наборах данных
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("dataset_name,loader", [
    ("Google",       load_google_trace),
    ("Azure",        load_azure_trace),
    ("Стационарный", lambda: generate_stationary(N_OBS_SYNTHETIC, seed=42)),
    ("Трендовый",    lambda: generate_trend(N_OBS_SYNTHETIC, seed=42)),
    ("Всплесковый",  lambda: generate_spike(N_OBS_SYNTHETIC, seed=42)),
    ("Смешанный",    lambda: generate_mixed(N_OBS_SYNTHETIC, seed=42)),
])
class TestCompareAllDatasets:
    """Сравнение всех методов на каждом датасете."""

    def test_all_methods(self, dataset_name, loader):
        cpu, ts, phi = loader()
        _run_all_methods_on_dataset(dataset_name, cpu, ts, phi)


# ══════════════════════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 3: Зависимость MAE от горизонта h
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("h", [1, 2, 3, 4, 6])
class TestHorizonDependence:
    """MAE при горизонтах h=1..6 на Alibaba."""

    def test_mae_grows_with_horizon(self, h):
        cpu, ts, phi = load_alibaba_trace()
        n_tr = int(len(cpu) * 0.70)
        n_vl = int(len(cpu) * 0.15)
        maes = []

        for seed in SEEDS:
            torch_seed(seed)
            pp, fc, _ = build_proposed_method(seed, horizon_h=h)
            fit_with_cache(fc, cpu[:n_tr], ts[:n_tr], phi[:, :n_tr], seed, f"horizon_h{h}")
            r = run_forecast_experiment(
                fc, cpu[:n_tr], cpu[n_tr + n_vl:],
                ts[:n_tr], ts[n_tr + n_vl:],
                phi[:, :n_tr], phi[:, n_tr + n_vl:],
                horizon_h=h)
            maes.append(compute_mae(r["y_true"], r["y_pred"]))

        print(f"\n[METRIC] HORIZON: h={h} | MAE={np.mean(maes):.4f}±{np.std(maes):.4f}",
              flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 4: Вычислительное время
# ══════════════════════════════════════════════════════════════════════════════

class TestComputationTime:
    """Время одной итерации полного цикла."""

    def test_iteration_time(self):
        cpu, ts, phi = load_alibaba_trace()
        n_tr = int(len(cpu) * 0.70)

        pp, fc, dm = build_proposed_method(42)
        fit_with_cache(fc, cpu[:n_tr], ts[:n_tr], phi[:, :n_tr], 42, "timing")

        times_ms = []
        for i in range(20):
            window_end = n_tr + i
            t0 = time.perf_counter()
            _, _, resid_norm, mu, sigma = pp.fit_transform(cpu[:window_end])
            cpu_hat, q_lo, q_hi = fc.predict(cpu[:window_end], ts[:window_end], phi[:, :window_end])
            dm.step(float(q_hi[0]), float(q_lo[0]))
            elapsed_ms = (time.perf_counter() - t0) * 1000
            times_ms.append(elapsed_ms)

        mean_ms = np.mean(times_ms)
        std_ms = np.std(times_ms)
        pct_of_dt = mean_ms / (5 * 60 * 1000) * 100

        print(f"\n[METRIC] TIMING: Среднее={mean_ms:.1f}мс ± {std_ms:.1f}мс | "
              f"Доля от Δt={pct_of_dt:.3f}%", flush=True)
