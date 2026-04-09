"""Compact end-to-end smoke test (one seed, all 9 methods on Alibaba)."""
import warnings; warnings.filterwarnings("ignore")
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch

from tests.test_experiments import build_proposed_method, run_forecast_experiment, fit_with_cache, ALL_METHODS
from tests.baselines import load_alibaba_trace
from tests.metrics import compute_mae, compute_rmse, compute_coverage

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "_smoke.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def log(msg):
    print(msg, flush=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

with open(OUT, "w", encoding="utf-8") as f:
    f.write(f"=== smoke run started: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    f.write(f"CUDA available: {torch.cuda.is_available()}\n")

cpu, ts, phi = load_alibaba_trace()
n_tr = int(len(cpu) * 0.70); n_vl = int(len(cpu) * 0.15)
cpu_train, cpu_test = cpu[:n_tr], cpu[n_tr+n_vl:]
ts_train, ts_test = ts[:n_tr], ts[n_tr+n_vl:]
phi_train, phi_test = phi[:, :n_tr], phi[:, n_tr+n_vl:]

log("Smoke: Alibaba single-seed, all 9 methods")
log("=" * 60)

seed = 42
for method_name, factory in ALL_METHODS.items():
    np.random.seed(seed); torch.manual_seed(seed)
    t0 = time.time()
    if method_name == "Разработанный метод":
        pp, fc, dm = build_proposed_method(seed)
        fit_with_cache(fc, cpu_train, ts_train, phi_train, seed, "smoke_alibaba_proposed")
        forecaster = fc
    else:
        forecaster = factory(seed)
        forecaster.fit(cpu_train, ts_train, phi_train)
    t_fit = time.time() - t0
    t1 = time.time()
    r = run_forecast_experiment(forecaster, cpu_train, cpu_test, ts_train, ts_test, phi_train, phi_test)
    t_pred = time.time() - t1
    mae = compute_mae(r["y_true"], r["y_pred"])
    rmse = compute_rmse(r["y_true"], r["y_pred"])
    cov = compute_coverage(r["y_true"], r["y_lower"], r["y_upper"])
    log("%-22s MAE=%.4f RMSE=%.4f COV=%5.1f%% fit=%5.1fs pred=%5.1fs" %
        (method_name[:22], mae, rmse, cov, t_fit, t_pred))

log("=" * 60)
log("done.")
