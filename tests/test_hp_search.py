"""
tests/test_hp_search.py — Подбор гиперпараметров предложенного метода.

Каждый класс TestHP* — это отдельный тест на ОДИН гиперпараметр.
Внутри теста: фиксированная базовая конфигурация + перебор значений
по этому гиперпараметру + полное обучение и оценка на одном датасете.

Запуск всех:
    python -u scripts/run_hp_search.py

Запуск одного:
    pytest tests/test_hp_search.py::TestHP_HiddenDim -v -s

Запуск группы:
    pytest tests/test_hp_search.py -v -s -k "HiddenDim or Dropout"

Каждый прогон печатает строку:
    [METRIC] HPSEARCH: dataset=Alibaba | param=hidden_dim | value=64 |
        MAE=0.0445 | RMSE=0.0593 | MAPE=10.4% | COVERAGE=96.7%
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from predictor.preprocessor import Preprocessor  # noqa: E402
from predictor.forecaster import HybridForecaster  # noqa: E402
from tests.baselines import (  # noqa: E402
    load_alibaba_trace, load_google_trace, load_azure_trace,
)
from tests.metrics import (  # noqa: E402
    compute_mae, compute_rmse, compute_mape, compute_coverage,
)
from predictor.config import (  # noqa: E402
    CFG_PREPROCESSOR as _CFG_PP, CFG_FORECASTER as _CFG_FC,
)

warnings.filterwarnings("ignore")

# ── Проверка CUDA на старте ──────────────────────────────────────────────
_CUDA_OK = torch.cuda.is_available()
print(
    f"\n[HP][ENV] PyTorch={torch.__version__} | CUDA available={_CUDA_OK} | "
    f"device={torch.cuda.get_device_name(0) if _CUDA_OK else 'CPU'}",
    flush=True,
)
if not _CUDA_OK:
    print(
        "[HP][ENV][WARN] CUDA НЕ ДОСТУПНА — обучение GRU пойдёт на CPU.\n"
        "                Установите GPU-сборку PyTorch:\n"
        "                pip install --force-reinstall torch==2.1.2+cu121 "
        "--index-url https://download.pytorch.org/whl/cu121",
        flush=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Выбор датасета для HP-поиска.
#
# По умолчанию — Azure (самый длинный, ≈30 суток, наиболее стабильный val-сигнал).
# Можно переопределить через переменную окружения HP_DATASET={alibaba,google,azure}
# или через CLI-флаг --dataset в scripts/run_hp_search.py.
# ─────────────────────────────────────────────────────────────────────────────

_DATASET_LOADERS = {
    "alibaba": ("Alibaba", load_alibaba_trace),
    "google":  ("Google",  load_google_trace),
    "azure":   ("Azure",   load_azure_trace),
}

_DATASET_KEY = os.environ.get("HP_DATASET", "azure").strip().lower()
if _DATASET_KEY not in _DATASET_LOADERS:
    print(
        f"[HP][WARN] Unknown HP_DATASET={_DATASET_KEY!r}, "
        f"falling back to 'azure'. Allowed: {sorted(_DATASET_LOADERS)}",
        flush=True,
    )
    _DATASET_KEY = "azure"

DATASET_NAME, _DATASET_LOADER = _DATASET_LOADERS[_DATASET_KEY]


# ─────────────────────────────────────────────────────────────────────────────
# Базовая конфигурация — точка отсчёта для всех HP-сравнений.
# Если какой-то параметр в свипе совпадает с базовым — он показывает текущий
# baseline-результат, относительно которого можно судить улучшения.
# ─────────────────────────────────────────────────────────────────────────────

BASE_PREPROCESSOR: Dict[str, Any] = dict(_CFG_PP)

BASE_FORECASTER: Dict[str, Any] = dict(_CFG_FC)
# HP-поиск: меньше эпох и patience для скорости свипов
BASE_FORECASTER["max_epochs"] = 40
BASE_FORECASTER["patience"] = 6

# Один seed для скорости — HP-поиск не требует усреднения по seed'ам.
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return time.strftime("%H:%M:%S")


def _print(msg: str) -> None:
    print(msg, flush=True)


_DATA_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Кэшированная загрузка датасета — каждый класс тестов вызывает один раз."""
    if _DATASET_KEY not in _DATA_CACHE:
        _DATA_CACHE[_DATASET_KEY] = _DATASET_LOADER()
        cpu_arr, _, _ = _DATA_CACHE[_DATASET_KEY]
        print(
            f"[HP] Loaded dataset '{DATASET_NAME}' "
            f"({len(cpu_arr)} points)",
            flush=True,
        )
    return _DATA_CACHE[_DATASET_KEY]


def _eval_one_config(
    pp_cfg: Dict[str, Any], fc_cfg: Dict[str, Any]
) -> Dict[str, float]:
    """
    Полный цикл fit + predict на test-выборке Alibaba.
    Возвращает dict с метриками + временами.
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    cpu, ts, phi = _load_dataset()
    n_tr = int(len(cpu) * 0.70)
    n_vl = int(len(cpu) * 0.15)
    cpu_train, cpu_test = cpu[:n_tr], cpu[n_tr + n_vl:]
    ts_train, ts_test = ts[:n_tr], ts[n_tr + n_vl:]
    phi_train = phi[:, :n_tr]

    pp = Preprocessor(**pp_cfg)
    fc = HybridForecaster(preprocessor=pp, **fc_cfg)

    t_fit0 = time.time()
    fc.fit(cpu_train, ts_train, phi_train)
    fit_time = time.time() - t_fit0

    h = fc_cfg["horizon_h"]
    y_t: List[float] = []
    y_p: List[float] = []
    y_lo: List[float] = []
    y_hi: List[float] = []
    t_pred0 = time.time()
    for i in range(0, len(cpu_test) - h, h):
        cpu_ctx = np.concatenate([cpu_train, cpu_test[:i]])
        ts_ctx = np.concatenate([ts_train, ts_test[:i]])
        hat, lo, hi = fc.predict(cpu_ctx, ts_ctx, None)
        for k in range(min(h, len(cpu_test) - i)):
            y_t.append(float(cpu_test[i + k]))
            y_p.append(float(hat[k]))
            y_lo.append(float(lo[k]))
            y_hi.append(float(hi[k]))
    pred_time = time.time() - t_pred0

    y_t_arr = np.array(y_t)
    y_p_arr = np.array(y_p)
    y_lo_arr = np.array(y_lo)
    y_hi_arr = np.array(y_hi)
    return {
        "mae": compute_mae(y_t_arr, y_p_arr),
        "rmse": compute_rmse(y_t_arr, y_p_arr),
        "mape": compute_mape(y_t_arr, y_p_arr),
        "coverage": compute_coverage(y_t_arr, y_lo_arr, y_hi_arr),
        "blend_alpha": float(getattr(fc, "blend_alpha", 1.0)),
        "fit_time": fit_time,
        "pred_time": pred_time,
    }


def _run_sweep(
    param_name: str,
    values: List[Any],
    fc_override: Optional[Callable[[Any], Dict[str, Any]]] = None,
    pp_override: Optional[Callable[[Any], Dict[str, Any]]] = None,
) -> None:
    """
    Полный свип одного гиперпараметра.

    fc_override(value) → dict patches для BASE_FORECASTER
    pp_override(value) → dict patches для BASE_PREPROCESSOR
    """
    n = len(values)
    base_value = (
        (fc_override and fc_override(BASE_FORECASTER.get(param_name))
         and BASE_FORECASTER.get(param_name))
        or BASE_PREPROCESSOR.get(param_name)
        or BASE_FORECASTER.get(param_name)
    )
    _print(f"\n[{_now()}] [HP {param_name}] >>> старт: {n} значений {values}")
    _print(f"[{_now()}] [HP {param_name}] base value = {base_value}")
    sweep_t0 = time.time()
    results: List[Tuple[Any, Dict[str, float]]] = []

    for i, v in enumerate(values, 1):
        fc_cfg = dict(BASE_FORECASTER)
        pp_cfg = dict(BASE_PREPROCESSOR)
        if fc_override is not None:
            fc_cfg.update(fc_override(v))
        if pp_override is not None:
            pp_cfg.update(pp_override(v))

        _print(
            f"\n[{_now()}] [HP {param_name}] [{i:>2d}/{n}] START {param_name}={v}"
        )
        try:
            m = _eval_one_config(pp_cfg, fc_cfg)
            _print(
                f"[{_now()}] [HP {param_name}] [{i:>2d}/{n}] DONE  {param_name}={v}  "
                f"MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} "
                f"MAPE={m['mape']:.2f}% COV={m['coverage']:.1f}% "
                f"alpha={m['blend_alpha']:.2f} "
                f"(fit={m['fit_time']:.0f}s pred={m['pred_time']:.1f}s)"
            )
            _print(
                f"[METRIC] HPSEARCH: dataset={DATASET_NAME} | "
                f"param={param_name} | value={v} | "
                f"MAE={m['mae']:.4f} | RMSE={m['rmse']:.4f} | "
                f"MAPE={m['mape']:.2f}% | COVERAGE={m['coverage']:.1f}%"
            )
            results.append((v, m))
        except Exception as e:
            tb = traceback.format_exc(limit=4)
            _print(
                f"[{_now()}] [HP {param_name}] [{i:>2d}/{n}] ERROR {param_name}={v}: "
                f"{type(e).__name__}: {e}\n{tb}"
            )

    # Итоговый отчёт по свипу
    elapsed = time.time() - sweep_t0
    _print(
        f"\n[{_now()}] [HP {param_name}] === Сводка свипа ({elapsed:.0f}с) ==="
    )
    if not results:
        _print(f"[{_now()}] [HP {param_name}] Все запуски упали с ошибкой.")
        return

    # Сортируем по MAE
    results_sorted = sorted(results, key=lambda r: r[1]["mae"])
    best_v, best_m = results_sorted[0]
    worst_v, worst_m = results_sorted[-1]
    _print(
        f"[{_now()}] [HP {param_name}] BEST  {param_name}={best_v}: "
        f"MAE={best_m['mae']:.4f} COV={best_m['coverage']:.1f}%"
    )
    _print(
        f"[{_now()}] [HP {param_name}] WORST {param_name}={worst_v}: "
        f"MAE={worst_m['mae']:.4f} COV={worst_m['coverage']:.1f}%"
    )
    _print(f"[{_now()}] [HP {param_name}] полная таблица (по MAE):")
    for v, m in results_sorted:
        marker = "  <-- BEST" if v == best_v else ""
        _print(
            f"    {param_name}={v!r:>8}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
            f"MAPE={m['mape']:.2f}%  COV={m['coverage']:.1f}%{marker}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Один класс на гиперпараметр.
# Каждый класс — независимый pytest-тест, запускается отдельно через -k.
# ═══════════════════════════════════════════════════════════════════════════

class TestHP_HiddenDim:
    """Свип размерности скрытого состояния GRU."""

    def test_sweep(self):
        _run_sweep(
            "hidden_dim",
            [32, 64, 96, 128, 192],
            fc_override=lambda v: {"hidden_dim": v},
        )


class TestHP_Dropout:
    """Свип dropout в GRU и в голове."""

    def test_sweep(self):
        _run_sweep(
            "dropout",
            [0.00, 0.10, 0.15, 0.20, 0.30, 0.40],
            fc_override=lambda v: {"dropout": v},
        )


class TestHP_WInput:
    """Свип длины входного окна GRU (history length)."""

    def test_sweep(self):
        _run_sweep(
            "w_input",
            [30, 45, 60, 90, 120, 180],
            fc_override=lambda v: {"w_input": v},
        )


class TestHP_LearningRate:
    """Свип скорости обучения Adam."""

    def test_sweep(self):
        _run_sweep(
            "lr",
            [3e-4, 5e-4, 1e-3, 2e-3, 5e-3],
            fc_override=lambda v: {"lr": v},
        )


class TestHP_BatchSize:
    """Свип размера мини-батча."""

    def test_sweep(self):
        _run_sweep(
            "batch_size",
            [16, 32, 64, 128, 256],
            fc_override=lambda v: {"batch_size": v},
        )


class TestHP_MaxEpochs:
    """Свип максимального числа эпох (с включённой ранней остановкой)."""

    def test_sweep(self):
        _run_sweep(
            "max_epochs",
            [20, 40, 60, 100],
            fc_override=lambda v: {"max_epochs": v},
        )


class TestHP_Patience:
    """Свип терпения ранней остановки (0 = выключена)."""

    def test_sweep(self):
        _run_sweep(
            "patience",
            [0, 4, 6, 10, 15],
            fc_override=lambda v: {"patience": v},
        )


class TestHP_GradClip:
    """Свип нормы обрезки градиента."""

    def test_sweep(self):
        _run_sweep(
            "grad_clip",
            [0.5, 1.0, 2.0, 5.0],
            fc_override=lambda v: {"grad_clip": v},
        )


class TestHP_NTrend:
    """Свип глубины окна для линейной экстраполяции тренда."""

    def test_sweep(self):
        _run_sweep(
            "n_T",
            [20, 40, 60, 90, 120],
            fc_override=lambda v: {"n_T": v},
        )


class TestHP_WNorm:
    """Свип ширины окна сезонного сглаживателя STL (seasonal smoother window)."""

    def test_sweep(self):
        _run_sweep(
            "w_norm",
            [7, 13, 21, 41, 61, 121],
            pp_override=lambda v: {"w_norm": v},
        )


class TestHP_IqrAlpha:
    """Свип коэффициента α метода Тьюки для обнаружения аномалий."""

    def test_sweep(self):
        _run_sweep(
            "iqr_alpha",
            [1.0, 1.5, 2.0, 2.5, 3.0],
            pp_override=lambda v: {"iqr_alpha": v},
        )


class TestHP_WAnomaly:
    """Свип ширины окна для обнаружения аномалий (IQR)."""

    def test_sweep(self):
        _run_sweep(
            "w_a",
            [24, 48, 96, 144],
            pp_override=lambda v: {"w_a": v},
        )
