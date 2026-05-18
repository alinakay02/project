"""
tests/test_table_4_7.py — Зависимость MAE от горизонта
прогнозирования на наборе Alibaba Cluster Trace 2018.

Прогоняет 3 метода (Разработанный, автономная GRU, SARIMA) на горизонтах
h = 1, 2, 3, 4, 6 шагов. Каждая пара (метод, h) — 3 seed'а, метрики
усредняются.

Запуск:
  python -m pytest tests/test_table_4_7.py -v -s
  python -u scripts/run_table_4_7.py

Результаты в: results/table_4_7.txt
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
from predictor.config import CFG_PREPROCESSOR, CFG_FORECASTER
from tests.baselines import (
    AutoGRUForecaster, SARIMAForecaster, load_alibaba_trace,
)
from tests.metrics import compute_mae

warnings.filterwarnings("ignore")

# ── Проверка CUDA ────────────────────────────────────────────────────────
_CUDA_OK = torch.cuda.is_available()
print(
    f"\n[T4.7][ENV] PyTorch={torch.__version__} | CUDA available={_CUDA_OK} | "
    f"device={torch.cuda.get_device_name(0) if _CUDA_OK else 'CPU'}",
    flush=True,
)

SEEDS = [42, 137, 256]
HORIZONS = [1, 2, 3, 4, 6]


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _run_forecast(forecaster, cpu_train, cpu_test, ts_train, ts_test,
                  phi_train, phi_test, h):
    """Rolling-window evaluation: возвращает только y_true и y_pred."""
    y_true, y_pred = [], []
    for i in range(0, len(cpu_test) - h, h):
        cpu_ctx = np.concatenate([cpu_train, cpu_test[:i]])
        ts_ctx = np.concatenate([ts_train, ts_test[:i]])
        phi_ctx = (
            np.concatenate([phi_train, phi_test[:, :i]], axis=1)
            if phi_test.shape[1] > i else phi_train
        )
        hat, _, _ = forecaster.predict(cpu_ctx, ts_ctx, phi_ctx)
        for k in range(min(h, len(cpu_test) - i)):
            y_true.append(cpu_test[i + k])
            y_pred.append(hat[k] if k < len(hat) else hat[-1])
    return np.array(y_true), np.array(y_pred)


def _make_forecaster(method_name: str, seed: int, horizon_h: int):
    """Создаёт прогнозатор одного из трёх типов."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if method_name == "proposed":
        pp = Preprocessor(**CFG_PREPROCESSOR)
        cfg = dict(CFG_FORECASTER)
        cfg["horizon_h"] = horizon_h
        return HybridForecaster(preprocessor=pp, **cfg)
    if method_name == "gru":
        return AutoGRUForecaster(seed=seed, horizon_h=horizon_h, max_epochs=100)
    if method_name == "sarima":
        return SARIMAForecaster(horizon_h=horizon_h)
    raise ValueError(f"Unknown method: {method_name}")


@pytest.mark.parametrize("method_name", ["proposed", "gru", "sarima"])
@pytest.mark.parametrize("h", HORIZONS)
class TestTable47:
    """MAE × (метод, горизонт)."""

    def test_horizon(self, method_name, h):
        cpu, ts, phi = load_alibaba_trace()
        n_tr = int(len(cpu) * 0.70)
        n_vl = int(len(cpu) * 0.15)
        cpu_train, cpu_test = cpu[:n_tr], cpu[n_tr + n_vl:]
        ts_train, ts_test = ts[:n_tr], ts[n_tr + n_vl:]
        phi_train, phi_test = phi[:, :n_tr], phi[:, n_tr + n_vl:]

        print(
            f"\n[{_now()}] [T4.7][{method_name}][h={h}] >>> старт: {len(SEEDS)} seeds "
            f"(train={n_tr}, test={len(cpu_test)})",
            flush=True,
        )

        maes = []
        t0 = time.time()

        for step, seed in enumerate(SEEDS, 1):
            seed_t0 = time.time()
            print(
                f"[{_now()}] [T4.7][{method_name}][h={h}] "
                f"[{step}/{len(SEEDS)}] START seed={seed}",
                flush=True,
            )
            gc.collect()

            try:
                fc = _make_forecaster(method_name, seed, h)
                fc.fit(cpu_train, ts_train, phi_train)
                y_true, y_pred = _run_forecast(
                    fc, cpu_train, cpu_test,
                    ts_train, ts_test, phi_train, phi_test, h,
                )
                mae = compute_mae(y_true, y_pred) if len(y_true) else float("nan")
                maes.append(mae)
                print(
                    f"[{_now()}] [T4.7][{method_name}][h={h}] "
                    f"[{step}/{len(SEEDS)}] DONE  seed={seed} MAE={mae:.4f} "
                    f"({time.time() - seed_t0:.1f}s)",
                    flush=True,
                )
            except Exception as e:
                tb = traceback.format_exc(limit=4)
                print(
                    f"[{_now()}] [T4.7][{method_name}][h={h}] [ERROR] "
                    f"seed={seed}: {type(e).__name__}: {e}\n{tb}",
                    flush=True,
                )

        elapsed = time.time() - t0
        if maes:
            mae_mean = float(np.mean(maes))
            mae_std = float(np.std(maes))
            print(
                f"\n[METRIC] TABLE_4_7: method={method_name} | h={h} | "
                f"MAE={mae_mean:.4f}\u00b1{mae_std:.4f} | "
                f"seeds_ok={len(maes)}/{len(SEEDS)} | total={elapsed:.1f}s",
                flush=True,
            )
        else:
            print(
                f"\n[METRIC] TABLE_4_7: method={method_name} | h={h} | "
                f"FAILED (0/{len(SEEDS)} seeds, total={elapsed:.1f}s)",
                flush=True,
            )
