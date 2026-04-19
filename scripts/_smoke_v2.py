"""
scripts/_smoke_v2.py — Быстрый прогон разработанного метода на всех датасетах
с одним seed. Используется для валидации изменений forecaster/preprocessor
без полной 7x3 матрицы тестов.

Запуск:
  python -u scripts/_smoke_v2.py
"""
from __future__ import annotations

import gc
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")

from predictor.preprocessor import Preprocessor
from predictor.forecaster import HybridForecaster
from predictor.config import CFG_PREPROCESSOR, CFG_FORECASTER
from tests.baselines import (
    load_alibaba_trace, load_google_trace, load_azure_trace,
    generate_stationary, generate_trend, generate_spike, generate_mixed,
)
from tests.metrics import compute_mae, compute_rmse, compute_coverage


DATASETS = [
    ("Alibaba",      load_alibaba_trace,                                          0.0419),
    ("Google",       load_google_trace,                                           0.0149),
    ("Azure",        load_azure_trace,                                            0.0277),
    ("Стационарный", lambda: generate_stationary(4320, seed=42),                  0.0308),
    ("Трендовый",    lambda: generate_trend(4320, seed=42),                       0.0242),
    ("Всплесковый",  lambda: generate_spike(4320, seed=42),                       0.0446),
    ("Смешанный",    lambda: generate_mixed(4320, seed=42),                       0.0429),
]


def run_one(name, loader, target, seed: int = 42):
    cpu, ts, phi = loader()
    n_tr = int(len(cpu) * 0.70)
    n_vl = int(len(cpu) * 0.15)
    cpu_train, cpu_test = cpu[:n_tr], cpu[n_tr + n_vl:]
    ts_train, ts_test = ts[:n_tr], ts[n_tr + n_vl:]

    np.random.seed(seed)
    torch.manual_seed(seed)
    pp = Preprocessor(**CFG_PREPROCESSOR)
    fc = HybridForecaster(preprocessor=pp, **CFG_FORECASTER)

    t0 = time.time()
    fc.fit(cpu_train, ts_train, None)
    fit_t = time.time() - t0

    h = fc.horizon_h
    y_true, y_pred, y_lo, y_hi = [], [], [], []
    t0 = time.time()
    for i in range(0, len(cpu_test) - h, h):
        cpu_ctx = np.concatenate([cpu_train, cpu_test[:i]])
        ts_ctx = np.concatenate([ts_train, ts_test[:i]])
        hat, lo, hi = fc.predict(cpu_ctx, ts_ctx, None)
        for k in range(min(h, len(cpu_test) - i)):
            y_true.append(cpu_test[i + k])
            y_pred.append(hat[k])
            y_lo.append(lo[k])
            y_hi.append(hi[k])
    pred_t = time.time() - t0

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_lo = np.array(y_lo)
    y_hi = np.array(y_hi)
    mae = compute_mae(y_true, y_pred)
    cov = compute_coverage(y_true, y_lo, y_hi)
    mark = "WIN " if mae <= target else "MISS"
    print(
        f"{mark} {name:14s}  MAE={mae:.4f}  COV={cov:4.1f}%  "
        f"alpha={fc.blend_alpha:.3f}  target<={target:.4f}  "
        f"(fit={fit_t:.0f}s pred={pred_t:.0f}s)",
        flush=True,
    )
    return mae, cov


def main():
    print("=" * 80, flush=True)
    print("SMOKE v2 — все датасеты, seed=42", flush=True)
    print("Target = лучший baseline MAE из summary_results.md", flush=True)
    print("=" * 80, flush=True)
    for name, loader, target in DATASETS:
        try:
            run_one(name, loader, target)
        except Exception as e:
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
        gc.collect()


if __name__ == "__main__":
    main()
