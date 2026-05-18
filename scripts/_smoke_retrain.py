"""scripts/_smoke_retrain.py - проверка нового retrain на Azure.

Сравниваем:
  A) retrain (новый, с сохранением preprocessor) на 7-дневном окне
  B) без retrain (initial fit держится весь тест)
  C) полный fit() на 7-дневном окне (старое поведение)

Прогноз на сутках 3-4 после initial fit на 22 сутках.
"""
from __future__ import annotations
import sys, time, copy
import numpy as np
import torch

sys.path.insert(0, ".")

from predictor.preprocessor import Preprocessor
from predictor.forecaster import HybridForecaster
from predictor.config import CFG_PREPROCESSOR, CFG_FORECASTER
from tests.baselines import load_azure_trace
from tests.metrics import compute_mae


POINTS_PER_DAY = 288
RETRAIN_WINDOW = 2016  # 7 дней
SEG_SIZE = POINTS_PER_DAY * 2


def eval_segment(fc, cpu, ts, seg_start, seg_end):
    h = fc.horizon_h
    y_true, y_pred = [], []
    for i in range(seg_start, seg_end - h, h):
        cpu_ctx = cpu[:i]
        ts_ctx = ts[:i]
        hat, _, _ = fc.predict(cpu_ctx, ts_ctx, None)
        for k in range(min(h, seg_end - i)):
            y_true.append(cpu[i + k])
            y_pred.append(hat[k])
    return compute_mae(np.array(y_true), np.array(y_pred))


def main():
    cpu, ts, phi = load_azure_trace()
    n = len(cpu)
    # initial fit на первых 22 сутках (6336 точек)
    initial_end = 22 * POINTS_PER_DAY
    # Тестовый сегмент сутки 3-4 (после initial fit)
    seg_start = initial_end + SEG_SIZE   # сутки 3-4 = позиции [6912, 7488)
    seg_end = seg_start + SEG_SIZE

    retrain_window_start = seg_start - RETRAIN_WINDOW

    print(f"Azure: n={n}, initial_end={initial_end}, seg [{seg_start}:{seg_end}]", flush=True)

    # ── A) НОВЫЙ retrain (preprocessor preserved) ──────────────────────
    print("\n[A] NEW retrain (preprocessor preserved)", flush=True)
    np.random.seed(42); torch.manual_seed(42)
    pp_a = Preprocessor(**CFG_PREPROCESSOR)
    fc_a = HybridForecaster(preprocessor=pp_a, **CFG_FORECASTER)
    t0 = time.time()
    fc_a.fit(cpu[:initial_end], ts[:initial_end])
    print(f"  initial fit: {time.time()-t0:.0f}s, alpha={fc_a.blend_alpha:.3f}", flush=True)
    mu_before = pp_a.mu_cpu; sig_before = pp_a.sigma_cpu
    s_before = pp_a.seasonal_by_phase.copy() if pp_a.seasonal_by_phase is not None else None

    t0 = time.time()
    fc_a.retrain(cpu[retrain_window_start:seg_start], ts[retrain_window_start:seg_start])
    print(f"  retrain: {time.time()-t0:.0f}s, alpha={fc_a.blend_alpha:.3f}", flush=True)
    print(f"  mu_cpu unchanged: {np.isclose(pp_a.mu_cpu, mu_before)} "
          f"sigma_cpu unchanged: {np.isclose(pp_a.sigma_cpu, sig_before)}", flush=True)
    if s_before is not None:
        print(f"  seasonal_profile unchanged: {np.allclose(pp_a.seasonal_by_phase, s_before)}",
              flush=True)

    mae_a = eval_segment(fc_a, cpu, ts, seg_start, seg_end)
    print(f"  MAE seg 3-4 = {mae_a:.4f}", flush=True)

    # ── B) БЕЗ retrain (initial fit only) ──────────────────────────────
    print("\n[B] NO retrain (initial fit only)", flush=True)
    np.random.seed(42); torch.manual_seed(42)
    pp_b = Preprocessor(**CFG_PREPROCESSOR)
    fc_b = HybridForecaster(preprocessor=pp_b, **CFG_FORECASTER)
    t0 = time.time()
    fc_b.fit(cpu[:initial_end], ts[:initial_end])
    print(f"  initial fit: {time.time()-t0:.0f}s, alpha={fc_b.blend_alpha:.3f}", flush=True)
    mae_b = eval_segment(fc_b, cpu, ts, seg_start, seg_end)
    print(f"  MAE seg 3-4 = {mae_b:.4f}", flush=True)

    # ── C) СТАРЫЙ retrain (полный fit на 7-дневном окне) ───────────────
    print("\n[C] OLD retrain (full fit on 7-day window)", flush=True)
    np.random.seed(42); torch.manual_seed(42)
    pp_c = Preprocessor(**CFG_PREPROCESSOR)
    fc_c = HybridForecaster(preprocessor=pp_c, **CFG_FORECASTER)
    t0 = time.time()
    fc_c.fit(cpu[:initial_end], ts[:initial_end])
    print(f"  initial fit: {time.time()-t0:.0f}s, alpha={fc_c.blend_alpha:.3f}", flush=True)
    # Имитируем старый retrain: полный fit на окне
    pp_c._fitted = False  # заставляем refit
    t0 = time.time()
    fc_c.fit(cpu[retrain_window_start:seg_start], ts[retrain_window_start:seg_start])
    print(f"  old retrain: {time.time()-t0:.0f}s, alpha={fc_c.blend_alpha:.3f}", flush=True)
    mae_c = eval_segment(fc_c, cpu, ts, seg_start, seg_end)
    print(f"  MAE seg 3-4 = {mae_c:.4f}", flush=True)

    # ── Сводка ─────────────────────────────────────────────────────────
    print("\n=== SUMMARY ===", flush=True)
    print(f"  [A] NEW retrain (preserved preprocessor): MAE={mae_a:.4f}", flush=True)
    print(f"  [B] NO retrain:                            MAE={mae_b:.4f}", flush=True)
    print(f"  [C] OLD retrain (full refit):              MAE={mae_c:.4f}", flush=True)
    print(f"\n  Reference from table_4_8.txt (seed=42 retrain=True seg2): MAE=0.0307", flush=True)
    print(f"  Reference from table_4_8.txt (seed=42 no_retrain seg2):   MAE=0.0297", flush=True)


if __name__ == "__main__":
    main()
