"""scripts/_smoke_alibaba_3seeds.py - 3-seed Alibaba smoke (ASCII-only output)."""
from __future__ import annotations
import sys, time
import numpy as np
import torch

sys.path.insert(0, ".")

from predictor.preprocessor import Preprocessor
from predictor.forecaster import HybridForecaster
from predictor.config import CFG_PREPROCESSOR, CFG_FORECASTER
from tests.baselines import load_alibaba_trace
from tests.metrics import compute_mae, compute_rmse, compute_coverage


def run_seed(seed, cpu, ts):
    n_tr = int(len(cpu) * 0.70); n_vl = int(len(cpu) * 0.15)
    cpu_train, cpu_test = cpu[:n_tr], cpu[n_tr + n_vl:]
    ts_train, ts_test = ts[:n_tr], ts[n_tr + n_vl:]

    np.random.seed(seed); torch.manual_seed(seed)
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
            y_true.append(cpu_test[i + k]); y_pred.append(hat[k])
            y_lo.append(lo[k]); y_hi.append(hi[k])
    pred_t = time.time() - t0

    y_true=np.array(y_true); y_pred=np.array(y_pred)
    y_lo=np.array(y_lo); y_hi=np.array(y_hi)
    return compute_mae(y_true, y_pred), compute_rmse(y_true, y_pred), compute_coverage(y_true, y_lo, y_hi), fc.blend_alpha, fit_t, pred_t


def main():
    cpu, ts, phi = load_alibaba_trace()
    print(f"Alibaba: n={len(cpu)}  config: w_input={CFG_FORECASTER['w_input']}  hidden={CFG_FORECASTER['hidden_dim']}", flush=True)

    maes, rmses, covs, alphas = [], [], [], []
    for seed in [42, 137, 256]:
        t0 = time.time()
        mae, rmse, cov, alpha, fit_t, pred_t = run_seed(seed, cpu, ts)
        total_t = time.time() - t0
        print(f"  seed={seed}: MAE={mae:.4f} RMSE={rmse:.4f} COV={cov:.1f}% alpha={alpha:.3f} (fit={fit_t:.0f}s pred={pred_t:.0f}s total={total_t:.0f}s)", flush=True)
        maes.append(mae); rmses.append(rmse); covs.append(cov); alphas.append(alpha)

    mae_mean = np.mean(maes); mae_std = np.std(maes)
    print(f"AVERAGE: MAE={mae_mean:.4f}+-{mae_std:.4f}  RMSE={np.mean(rmses):.4f}  COV={np.mean(covs):.1f}%  alpha_mean={np.mean(alphas):.3f}", flush=True)
    baseline = 0.0448
    delta_pct = (mae_mean - baseline) / baseline * 100
    print(f"BASELINE comparison (w=96, h=96, 3 seeds): MAE=0.0448", flush=True)
    print(f"DELTA: {delta_pct:+.2f}%  {'(better!)' if delta_pct < 0 else '(worse)'}", flush=True)


if __name__ == "__main__":
    main()
