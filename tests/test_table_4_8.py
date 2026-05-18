"""
tests/test_table_4_8.py — Влияние дообучения на точность
прогноза во времени на наборе Azure VM Trace 2019 (30 суток).

Сравнивает две конфигурации:
  - С дообучением: перед каждым 2-суточным сегментом тестовой выборки
    модель дообучается на скользящем окне последних 2016 наблюдений (7 суток).
  - Без дообучения: модель обучается один раз на начальном участке и
    предсказывает все 4 сегмента без обновления.

Используется Azure (не Alibaba), так как Alibaba слишком короткий (~7.8 суток)
для 8-суточного тестового периода.

Запуск:
  python -m pytest tests/test_table_4_8.py -v -s
  python -u scripts/run_table_4_8.py

Результаты в: results/table_4_8.txt
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
from tests.baselines import load_azure_trace
from tests.metrics import compute_mae

warnings.filterwarnings("ignore")

_CUDA_OK = torch.cuda.is_available()
print(
    f"\n[T4.8][ENV] PyTorch={torch.__version__} | CUDA available={_CUDA_OK} | "
    f"device={torch.cuda.get_device_name(0) if _CUDA_OK else 'CPU'}",
    flush=True,
)

# ── Параметры сегментации ────────────────────────────────────────────────
# Azure: 8640 точек при dt=5 мин = 30 суток (288 точек/сутки)
POINTS_PER_DAY = 288
SEGMENT_DAYS = 2              # Каждый сегмент = 2 суток
N_SEGMENTS = 4                # Итого 8 суток тестирования
SEGMENT_SIZE = POINTS_PER_DAY * SEGMENT_DAYS  # 576 точек на сегмент
RETRAIN_WINDOW = 2016          # 7 суток — окно дообучения

SEEDS = [42, 137]  # 2 seeds (этот тест медленный из-за 4 дообучений на seed)


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _make_fc(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    pp = Preprocessor(**CFG_PREPROCESSOR)
    return HybridForecaster(preprocessor=pp, **CFG_FORECASTER)


def _rolling_predict(fc, cpu_full, ts_full, phi_full,
                     start_idx: int, end_idx: int):
    """Прогноз с расширяющимся окном в диапазоне [start_idx, end_idx)."""
    h = fc.horizon_h
    y_true, y_pred = [], []
    for i in range(start_idx, end_idx - h, h):
        cpu_ctx = cpu_full[:i]
        ts_ctx = ts_full[:i]
        hat, _, _ = fc.predict(cpu_ctx, ts_ctx, None)
        for k in range(min(h, end_idx - i)):
            y_true.append(cpu_full[i + k])
            y_pred.append(hat[k] if k < len(hat) else hat[-1])
    return np.array(y_true), np.array(y_pred)


def _run_one_config(with_retrain: bool, seed: int, cpu, ts, phi):
    """
    Возвращает список MAE по сегментам.

    Конфигурация сегментов:
      - Начальное обучение: первые (N - 8 суток) наблюдений
      - Сегмент k (k=1..4): 2 суток, начиная с (начало_теста + (k-1)*SEGMENT_SIZE)
      - В режиме with_retrain перед каждым сегментом модель переобучается
        на скользящем окне последних RETRAIN_WINDOW точек.
    """
    n = len(cpu)
    test_start = n - N_SEGMENTS * SEGMENT_SIZE
    if test_start <= 0:
        raise ValueError(f"Dataset too short: n={n}")

    tag = f"retrain={with_retrain}"

    # Начальное обучение
    print(
        f"[{_now()}] [T4.8][{tag}][seed={seed}] initial fit: "
        f"train на {test_start} точках (до дня {test_start / POINTS_PER_DAY:.1f})",
        flush=True,
    )
    fc = _make_fc(seed)
    t0 = time.time()
    fc.fit(cpu[:test_start], ts[:test_start], phi[:, :test_start])
    print(
        f"[{_now()}] [T4.8][{tag}][seed={seed}] initial fit готов за {time.time() - t0:.1f}s",
        flush=True,
    )

    maes_per_segment = []
    for seg_idx in range(N_SEGMENTS):
        seg_start = test_start + seg_idx * SEGMENT_SIZE
        seg_end = seg_start + SEGMENT_SIZE

        if with_retrain and seg_idx > 0:
            # Дообучение на скользящем окне из последних RETRAIN_WINDOW точек
            retrain_start = max(0, seg_start - RETRAIN_WINDOW)
            print(
                f"[{_now()}] [T4.8][{tag}][seed={seed}][seg{seg_idx + 1}] "
                f"retrain на окне [{retrain_start}:{seg_start}] "
                f"({seg_start - retrain_start} точек)",
                flush=True,
            )
            rt0 = time.time()
            # Пересоздаём fresh модель (самый честный способ) — fit фиксирует μ/σ заново
            fc = _make_fc(seed)
            fc.fit(cpu[retrain_start:seg_start], ts[retrain_start:seg_start],
                   phi[:, retrain_start:seg_start])
            print(
                f"[{_now()}] [T4.8][{tag}][seed={seed}][seg{seg_idx + 1}] "
                f"retrain готов за {time.time() - rt0:.1f}s",
                flush=True,
            )

        pt0 = time.time()
        y_true, y_pred = _rolling_predict(fc, cpu, ts, phi, seg_start, seg_end)
        mae = compute_mae(y_true, y_pred) if len(y_true) else float("nan")
        maes_per_segment.append(mae)
        print(
            f"[{_now()}] [T4.8][{tag}][seed={seed}][seg{seg_idx + 1}] "
            f"Сутки {seg_idx * 2 + 1}-{seg_idx * 2 + 2} | MAE={mae:.4f} "
            f"({time.time() - pt0:.1f}s)",
            flush=True,
        )

    return maes_per_segment


class TestTable48:
    """Дообучение vs без дообучения (Azure, 8 суток теста)."""

    def test_retrain_effect(self):
        cpu, ts, phi = load_azure_trace()

        print(
            f"\n[{_now()}] [T4.8] >>> старт: Azure n={len(cpu)}, "
            f"{N_SEGMENTS} сегментов × {SEGMENT_DAYS} суток, "
            f"окно retrain={RETRAIN_WINDOW} ({RETRAIN_WINDOW / POINTS_PER_DAY:.0f} сут)",
            flush=True,
        )
        t_total = time.time()

        # Сбор по 2 seeds обеих конфигураций
        results = {True: [], False: []}  # {with_retrain: [per-seed-list-of-MAEs]}

        for with_retrain in [True, False]:
            for seed in SEEDS:
                gc.collect()
                try:
                    maes = _run_one_config(with_retrain, seed, cpu, ts, phi)
                    results[with_retrain].append(maes)
                except Exception as e:
                    tb = traceback.format_exc(limit=4)
                    print(
                        f"[{_now()}] [T4.8] [ERROR] retrain={with_retrain} "
                        f"seed={seed}: {type(e).__name__}: {e}\n{tb}",
                        flush=True,
                    )

        # Усреднение по seeds
        def _avg(configs):
            if not configs:
                return [float("nan")] * N_SEGMENTS, [float("nan")] * N_SEGMENTS
            arr = np.array(configs)  # (n_seeds, N_SEGMENTS)
            return arr.mean(axis=0).tolist(), arr.std(axis=0).tolist()

        mean_w, std_w = _avg(results[True])
        mean_wo, std_wo = _avg(results[False])

        print(f"\n[{_now()}] [T4.8] === Сводные результаты ===", flush=True)
        for seg_idx in range(N_SEGMENTS):
            day1 = seg_idx * 2 + 1
            day2 = seg_idx * 2 + 2
            mw = mean_w[seg_idx]
            mwo = mean_wo[seg_idx]
            diff_pct = (
                100.0 * (mwo - mw) / mw if (not np.isnan(mw) and mw > 0) else float("nan")
            )
            print(
                f"[METRIC] TABLE_4_8: period=Сутки_{day1}-{day2} | "
                f"MAE_with_retrain={mw:.4f}\u00b1{std_w[seg_idx]:.4f} | "
                f"MAE_no_retrain={mwo:.4f}\u00b1{std_wo[seg_idx]:.4f} | "
                f"diff_pct={diff_pct:+.2f}%",
                flush=True,
            )

        print(
            f"\n[{_now()}] [T4.8] <<< завершено за {time.time() - t_total:.0f}с",
            flush=True,
        )
