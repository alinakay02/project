"""scripts/_smoke_alibaba.py - Quick Alibaba-only smoke test (ASCII-only output)."""
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


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    cpu, ts, phi = load_alibaba_trace()
    n_tr = int(len(cpu) * 0.70); n_vl = int(len(cpu) * 0.15)
    cpu_train, cpu_test = cpu[:n_tr], cpu[n_tr + n_vl:]
    ts_train, ts_test = ts[:n_tr], ts[n_tr + n_vl:]
    print(f"Alibaba: n={len(cpu)} train={n_tr} test={len(cpu_test)}", flush=True)

    pp = Preprocessor(**CFG_PREPROCESSOR)
    fc = HybridForecaster(preprocessor=pp, **CFG_FORECASTER)
    t0 = time.time()
    fc.fit(cpu_train, ts_train, None)
    print(f"Fit: {time.time()-t0:.1f}s  blend_alpha={fc.blend_alpha:.3f}", flush=True)

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
    y_true=np.array(y_true); y_pred=np.array(y_pred)
    y_lo=np.array(y_lo); y_hi=np.array(y_hi)
    mae = compute_mae(y_true, y_pred)
    cov = compute_coverage(y_true, y_lo, y_hi)
    print(f"Pred: {time.time()-t0:.1f}s", flush=True)
    print(f"RESULT Alibaba MAE={mae:.4f} RMSE={compute_rmse(y_true, y_pred):.4f} COV={cov:.1f}% target_ARIMA=0.0419 target_SARIMA=0.0429", flush=True)
    verdict = "BEATS ARIMA" if mae <= 0.0419 else ("BEATS SARIMA" if mae <= 0.0429 else ("BEATS OLD" if mae <= 0.0452 else "WORSE"))
    print(f"VERDICT {verdict}", flush=True)


if __name__ == "__main__":
    main()
