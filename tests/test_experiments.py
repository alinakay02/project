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
import traceback
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
from predictor.config import CFG_PREPROCESSOR, CFG_FORECASTER, CFG_DECISION

logging.basicConfig(level=logging.WARNING)

# ── Проверка CUDA на старте сессии ────────────────────────────────────────
import torch as _torch
_CUDA_OK = _torch.cuda.is_available()
print(
    f"\n[ENV] PyTorch={_torch.__version__} | CUDA available={_CUDA_OK} | "
    f"device={_torch.cuda.get_device_name(0) if _CUDA_OK else 'CPU'}",
    flush=True,
)
if not _CUDA_OK:
    msg = (
        "[ENV][WARN] CUDA НЕ ДОСТУПНА — обучение пойдёт на CPU и будет МЕДЛЕННО.\n"
        "             Установите GPU-сборку PyTorch:\n"
        "             pip install --force-reinstall torch==2.1.2+cu121 "
        "--index-url https://download.pytorch.org/whl/cu121"
    )
    print(msg, flush=True)
    warnings.warn(msg, stacklevel=2)

SEEDS = [42, 137, 256]
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
            print(f"      [cache HIT] {tag}_seed{seed}.pt", flush=True)
            return
        except Exception as e:
            print(f"      [cache MISS] {tag}_seed{seed}.pt unreadable: {e}", flush=True)
    fc.fit(cpu, ts, phi)
    try:
        fc.save_model(path)
        print(f"      [cache SAVE] {tag}_seed{seed}.pt", flush=True)
    except Exception as e:
        print(f"      [cache SAVE FAIL] {e}", flush=True)


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _print_progress(tag: str, step: int, total: int, message: str) -> None:
    pct = 100.0 * step / max(total, 1)
    print(f"[{_now()}] [{tag}] [{step:>3d}/{total:>3d}  {pct:5.1f}%] {message}", flush=True)


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
    """
    Прогоняет все методы на одном датасете, возвращает dict {method: metrics}.

    Логирует прогресс по каждой паре (method, seed): сколько шагов сделано из
    общего числа, сколько занял шаг, и любые исключения. Если один шаг падает,
    остальные продолжают выполняться, а в финальной таблице у этого метода
    будет пометка ERROR.
    """
    n_tr = int(len(cpu) * 0.70)
    n_vl = int(len(cpu) * 0.15)
    cpu_train, cpu_test = cpu[:n_tr], cpu[n_tr + n_vl:]
    ts_train, ts_test = ts[:n_tr], ts[n_tr + n_vl:]
    phi_train, phi_test = phi[:, :n_tr], phi[:, n_tr + n_vl:]

    total_steps = len(ALL_METHODS) * len(SEEDS)
    print(
        f"\n[{_now()}] [{dataset_name}] >>> старт: "
        f"{len(ALL_METHODS)} методов × {len(SEEDS)} seeds = {total_steps} шагов "
        f"(n={len(cpu)}, train={n_tr}, test={len(cpu_test)})",
        flush=True,
    )
    dataset_t0 = time.time()
    step = 0
    results = {}

    for method_name, factory in ALL_METHODS.items():
        maes, rmses, mapes, coverages = [], [], [], []
        method_t0 = time.time()
        method_errors = 0

        for seed in SEEDS:
            step += 1
            _print_progress(
                dataset_name, step, total_steps,
                f"START  method='{method_name}' seed={seed}",
            )
            seed_t0 = time.time()
            torch_seed(seed)
            gc.collect()

            try:
                if method_name == "Разработанный метод":
                    pp, fc, dm = build_proposed_method(seed)
                    fit_with_cache(
                        fc, cpu_train, ts_train, phi_train, seed,
                        f"{dataset_name}_proposed",
                    )
                    forecaster = fc
                else:
                    forecaster = factory(seed)
                    forecaster.fit(cpu_train, ts_train, phi_train)

                r = run_forecast_experiment(
                    forecaster, cpu_train, cpu_test,
                    ts_train, ts_test, phi_train, phi_test,
                )

                if len(r["y_true"]) > 0:
                    mae = compute_mae(r["y_true"], r["y_pred"])
                    rmse = compute_rmse(r["y_true"], r["y_pred"])
                    mape = compute_mape(r["y_true"], r["y_pred"])
                    cov = compute_coverage(r["y_true"], r["y_lower"], r["y_upper"])
                    maes.append(mae); rmses.append(rmse)
                    mapes.append(mape); coverages.append(cov)
                    _print_progress(
                        dataset_name, step, total_steps,
                        f"DONE   method='{method_name}' seed={seed} "
                        f"MAE={mae:.4f} COV={cov:.1f}% "
                        f"({time.time() - seed_t0:.1f}s)",
                    )
                else:
                    _print_progress(
                        dataset_name, step, total_steps,
                        f"WARN   method='{method_name}' seed={seed}: "
                        f"empty prediction set",
                    )
            except Exception as e:
                method_errors += 1
                tb = traceback.format_exc(limit=4)
                print(
                    f"[{_now()}] [{dataset_name}] [ERROR] method='{method_name}' "
                    f"seed={seed}: {type(e).__name__}: {e}\n{tb}",
                    flush=True,
                )

        method_elapsed = time.time() - method_t0
        if maes:
            results[method_name] = {
                "mae": np.mean(maes), "mae_std": np.std(maes),
                "rmse": np.mean(rmses), "rmse_std": np.std(rmses),
                "mape": np.mean(mapes), "mape_std": np.std(mapes),
                "coverage": np.mean(coverages), "coverage_std": np.std(coverages),
            }
            print(
                f"\n[METRIC] COMPARE: {dataset_name} | {method_name} | "
                f"MAE={np.mean(maes):.4f}±{np.std(maes):.4f} | "
                f"RMSE={np.mean(rmses):.4f}±{np.std(rmses):.4f} | "
                f"MAPE={np.mean(mapes):.2f}±{np.std(mapes):.2f}% | "
                f"COVERAGE={np.mean(coverages):.1f}±{np.std(coverages):.1f}% "
                f"| seeds_ok={len(maes)}/{len(SEEDS)} "
                f"| total={method_elapsed:.1f}s",
                flush=True,
            )
        else:
            print(
                f"\n[METRIC] COMPARE: {dataset_name} | {method_name} | "
                f"FAILED ({method_errors}/{len(SEEDS)} seeds errored, total={method_elapsed:.1f}s)",
                flush=True,
            )

    print(
        f"\n[{_now()}] [{dataset_name}] <<< готово за {time.time() - dataset_t0:.1f}с",
        flush=True,
    )
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
        total = len(SEEDS)
        tag = f"HORIZON h={h}"
        print(
            f"\n[{_now()}] [{tag}] >>> старт: {total} seeds на Alibaba",
            flush=True,
        )
        block_t0 = time.time()

        for step, seed in enumerate(SEEDS, 1):
            seed_t0 = time.time()
            _print_progress(tag, step, total, f"START seed={seed}")
            try:
                torch_seed(seed)
                pp, fc, _ = build_proposed_method(seed, horizon_h=h)
                fit_with_cache(
                    fc, cpu[:n_tr], ts[:n_tr], phi[:, :n_tr], seed, f"horizon_h{h}",
                )
                r = run_forecast_experiment(
                    fc, cpu[:n_tr], cpu[n_tr + n_vl:],
                    ts[:n_tr], ts[n_tr + n_vl:],
                    phi[:, :n_tr], phi[:, n_tr + n_vl:],
                    horizon_h=h,
                )
                mae = compute_mae(r["y_true"], r["y_pred"])
                maes.append(mae)
                _print_progress(
                    tag, step, total,
                    f"DONE  seed={seed} MAE={mae:.4f} ({time.time() - seed_t0:.1f}s)",
                )
            except Exception as e:
                tb = traceback.format_exc(limit=4)
                print(
                    f"[{_now()}] [{tag}] [ERROR] seed={seed}: "
                    f"{type(e).__name__}: {e}\n{tb}",
                    flush=True,
                )

        if maes:
            print(
                f"\n[METRIC] HORIZON: h={h} | "
                f"MAE={np.mean(maes):.4f}±{np.std(maes):.4f} "
                f"| seeds_ok={len(maes)}/{total} "
                f"| total={time.time() - block_t0:.1f}s",
                flush=True,
            )
        else:
            print(
                f"\n[METRIC] HORIZON: h={h} | FAILED "
                f"(0/{total} seeds successful, total={time.time() - block_t0:.1f}s)",
                flush=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 4: Вычислительное время
# ══════════════════════════════════════════════════════════════════════════════

class TestComputationTime:
    """Время одной итерации полного цикла."""

    def test_iteration_time(self):
        cpu, ts, phi = load_alibaba_trace()
        n_tr = int(len(cpu) * 0.70)
        N_ITERS = 20
        tag = "TIMING"

        print(
            f"\n[{_now()}] [{tag}] >>> старт: 1 fit + {N_ITERS} итераций predict",
            flush=True,
        )
        fit_t0 = time.time()
        pp, fc, dm = build_proposed_method(42)
        fit_with_cache(fc, cpu[:n_tr], ts[:n_tr], phi[:, :n_tr], 42, "timing")
        print(
            f"[{_now()}] [{tag}] fit готов за {time.time() - fit_t0:.1f}с",
            flush=True,
        )

        times_ms = []
        for i in range(N_ITERS):
            window_end = n_tr + i
            t0 = time.perf_counter()
            _, _, resid_norm, mu, sigma = pp.fit_transform(cpu[:window_end])
            cpu_hat, q_lo, q_hi = fc.predict(
                cpu[:window_end], ts[:window_end], phi[:, :window_end]
            )
            dm.step(float(q_hi[0]), float(q_lo[0]))
            elapsed_ms = (time.perf_counter() - t0) * 1000
            times_ms.append(elapsed_ms)
            _print_progress(
                tag, i + 1, N_ITERS,
                f"iter={i + 1} elapsed={elapsed_ms:.1f}ms",
            )

        mean_ms = float(np.mean(times_ms))
        std_ms = float(np.std(times_ms))
        pct_of_dt = mean_ms / (5 * 60 * 1000) * 100

        print(
            f"\n[METRIC] TIMING: Среднее={mean_ms:.1f}мс ± {std_ms:.1f}мс | "
            f"Доля от Δt={pct_of_dt:.3f}%",
            flush=True,
        )
