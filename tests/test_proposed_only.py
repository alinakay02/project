"""
tests/test_proposed_only.py — Тесты ТОЛЬКО для разработанного метода (STL + GRU).

Прогоняет разработанный метод на всех 7 датасетах (3 реальных + 4 синтетических)
с 3 seeds, без запуска baseline-методов.

Запуск:
  python -m pytest tests/test_proposed_only.py -v -s
  python -u scripts/run_proposed_only.py
  python -u scripts/run_proposed_only.py --dataset alibaba
"""

from __future__ import annotations

import gc
import os
import sys
import time
import traceback
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from predictor.preprocessor import Preprocessor
from predictor.forecaster import HybridForecaster
from tests.baselines import (
    load_alibaba_trace, load_google_trace, load_azure_trace,
)
from tests.baselines import (
    generate_stationary, generate_trend, generate_spike, generate_mixed,
)
from tests.metrics import (
    compute_mae, compute_rmse, compute_mape, compute_coverage,
)

from predictor.config import CFG_PREPROCESSOR, CFG_FORECASTER

warnings.filterwarnings("ignore")

# ── Конфигурация из config.yaml ──────────────────────────────────────────

SEEDS = [42, 137, 256]
N_OBS_SYNTHETIC = 4_320

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "cache")


# ── Утилиты ──────────────────────────────────────────────────────────────

def _now() -> str:
    return time.strftime("%H:%M:%S")


def _print_progress(tag: str, step: int, total: int, message: str) -> None:
    pct = 100.0 * step / max(total, 1)
    print(
        f"[{_now()}] [{tag}] [{step:>3d}/{total:>3d}  {pct:5.1f}%] {message}",
        flush=True,
    )


def _build_proposed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    pp = Preprocessor(**CFG_PREPROCESSOR)
    fc = HybridForecaster(preprocessor=pp, **CFG_FORECASTER)
    return pp, fc


def _fit_with_cache(fc, cpu, ts, phi, seed, tag):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{tag}_seed{seed}.pt")
    if os.path.exists(path):
        try:
            fc.load_model(path)
            print(f"      [cache HIT] {tag}_seed{seed}.pt", flush=True)
            return
        except Exception as e:
            print(f"      [cache MISS] {tag}_seed{seed}.pt: {e}", flush=True)
    fc.fit(cpu, ts, phi)
    try:
        fc.save_model(path)
        print(f"      [cache SAVE] {tag}_seed{seed}.pt", flush=True)
    except Exception as e:
        print(f"      [cache SAVE FAIL] {e}", flush=True)


def _run_forecast(forecaster, cpu_train, cpu_test, ts_train, ts_test,
                  phi_train, phi_test):
    h = forecaster.horizon_h
    y_true, y_pred, y_lower, y_upper = [], [], [], []
    for i in range(0, len(cpu_test) - h, h):
        cpu_ctx = np.concatenate([cpu_train, cpu_test[:i]])
        ts_ctx = np.concatenate([ts_train, ts_test[:i]])
        phi_ctx = (
            np.concatenate([phi_train, phi_test[:, :i]], axis=1)
            if phi_test.shape[1] > i else phi_train
        )
        hat, lo, hi = forecaster.predict(cpu_ctx, ts_ctx, phi_ctx)
        for k in range(min(h, len(cpu_test) - i)):
            y_true.append(cpu_test[i + k])
            y_pred.append(hat[k] if k < len(hat) else hat[-1])
            y_lower.append(lo[k] if k < len(lo) else lo[-1])
            y_upper.append(hi[k] if k < len(hi) else hi[-1])
    return {
        "y_true": np.array(y_true), "y_pred": np.array(y_pred),
        "y_lower": np.array(y_lower), "y_upper": np.array(y_upper),
    }


# ── Тесты ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dataset_name,loader", [
    ("Alibaba",      load_alibaba_trace),
    ("Google",       load_google_trace),
    ("Azure",        load_azure_trace),
    ("Стационарный", lambda: generate_stationary(N_OBS_SYNTHETIC, seed=42)),
    ("Трендовый",    lambda: generate_trend(N_OBS_SYNTHETIC, seed=42)),
    ("Всплесковый",  lambda: generate_spike(N_OBS_SYNTHETIC, seed=42)),
    ("Смешанный",    lambda: generate_mixed(N_OBS_SYNTHETIC, seed=42)),
])
class TestProposedMethod:
    """Разработанный метод (STL + GRU) на каждом датасете, 3 seeds."""

    def test_proposed(self, dataset_name, loader):
        cpu, ts, phi = loader()
        n_tr = int(len(cpu) * 0.70)
        n_vl = int(len(cpu) * 0.15)
        cpu_train, cpu_test = cpu[:n_tr], cpu[n_tr + n_vl:]
        ts_train, ts_test = ts[:n_tr], ts[n_tr + n_vl:]
        phi_train, phi_test = phi[:, :n_tr], phi[:, n_tr + n_vl:]

        maes, rmses, mapes, coverages = [], [], [], []
        total = len(SEEDS)

        print(
            f"\n[{_now()}] [{dataset_name}] >>> старт: {total} seeds "
            f"(n={len(cpu)}, train={n_tr}, test={len(cpu_test)})",
            flush=True,
        )
        dataset_t0 = time.time()

        for step, seed in enumerate(SEEDS, 1):
            _print_progress(dataset_name, step, total, f"START seed={seed}")
            t0 = time.time()
            gc.collect()

            try:
                np.random.seed(seed)
                torch.manual_seed(seed)
                pp, fc = _build_proposed(seed)
                _fit_with_cache(
                    fc, cpu_train, ts_train, phi_train,
                    seed, f"{dataset_name}_proposed",
                )

                r = _run_forecast(
                    fc, cpu_train, cpu_test,
                    ts_train, ts_test, phi_train, phi_test,
                )

                mae = compute_mae(r["y_true"], r["y_pred"])
                rmse = compute_rmse(r["y_true"], r["y_pred"])
                mape = compute_mape(r["y_true"], r["y_pred"])
                cov = compute_coverage(r["y_true"], r["y_lower"], r["y_upper"])
                maes.append(mae)
                rmses.append(rmse)
                mapes.append(mape)
                coverages.append(cov)

                _print_progress(
                    dataset_name, step, total,
                    f"DONE  seed={seed} MAE={mae:.4f} COV={cov:.1f}% "
                    f"({time.time() - t0:.1f}s)",
                )
            except Exception as e:
                tb = traceback.format_exc(limit=4)
                print(
                    f"[{_now()}] [{dataset_name}] [ERROR] seed={seed}: "
                    f"{type(e).__name__}: {e}\n{tb}",
                    flush=True,
                )

        elapsed = time.time() - dataset_t0
        if maes:
            print(
                f"\n[METRIC] PROPOSED: {dataset_name} | "
                f"MAE={np.mean(maes):.4f}\u00b1{np.std(maes):.4f} | "
                f"RMSE={np.mean(rmses):.4f}\u00b1{np.std(rmses):.4f} | "
                f"MAPE={np.mean(mapes):.2f}\u00b1{np.std(mapes):.2f}% | "
                f"COVERAGE={np.mean(coverages):.1f}\u00b1{np.std(coverages):.1f}% "
                f"| seeds_ok={len(maes)}/{total} | total={elapsed:.1f}s",
                flush=True,
            )
        else:
            print(
                f"\n[METRIC] PROPOSED: {dataset_name} | FAILED "
                f"(0/{total} seeds, total={elapsed:.1f}s)",
                flush=True,
            )
