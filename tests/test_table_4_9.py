"""
tests/test_table_4_9.py — Влияние календарных признаков
на точность прогноза на синтетическом «Стационарном» наборе данных.

Сравнивает две конфигурации:
  - С календарными признаками: d_step = 8 (cpu_norm + stl_resid + 6 time)
  - Без календарных признаков: d_step = 2 (cpu_norm + stl_resid только)

ВЫБОР ДАТАСЕТА: использован синтетический «Стационарный» (4320 точек,
15 суток с чёткой суточной сезонностью периодом 288 и умеренным шумом).

Почему синтетический, а не реальный:
  • На Google CPU стабильная и низкая (0.15-0.25) → GRU сводится к
    persistence, календарные признаки игнорируются.
  • На Alibaba (2243 точки ≈ 7.8 суток) недельный паттерн не успевает
    проявиться, а автокорреляция ряда затеняет вклад временных признаков.
  • Стационарный синтетический набор по дизайну содержит суточную
    сезонность как основной сигнал — там вклад календарных признаков
    будет наиболее чистым и хорошо измеримым.

Переопределяет TimeSeriesDataset для обучения без временных признаков
и _stack_features для прогноза без них — чтобы обучение и предсказание
использовали одинаковую размерность входа.

Запуск:
  python -m pytest tests/test_table_4_9.py -v -s
  python -u scripts/run_table.py 4.9

Результаты в: results/table_4_9.txt
"""

from __future__ import annotations

import gc
import os
import sys
import time
import traceback
import warnings
from typing import Optional, Tuple

import numpy as np
import pytest
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from predictor.preprocessor import Preprocessor
from predictor.forecaster import HybridForecaster
from predictor.model import GRUQuantileNet, GRUTrainer
from predictor.config import CFG_PREPROCESSOR, CFG_FORECASTER
from tests.baselines import generate_stationary
from tests.metrics import compute_mae

warnings.filterwarnings("ignore")

_CUDA_OK = torch.cuda.is_available()
print(
    f"\n[T4.9][ENV] PyTorch={torch.__version__} | CUDA available={_CUDA_OK} | "
    f"device={torch.cuda.get_device_name(0) if _CUDA_OK else 'CPU'}",
    flush=True,
)

SEEDS = [42, 137, 256]
N_OBS_SYNTHETIC = 4_320  # 15 суток при dt=5 мин


def _now() -> str:
    return time.strftime("%H:%M:%S")


# ── Датасет без временных признаков (d_step=2) ───────────────────────────

class TimeSeriesDatasetNoCalendar(Dataset):
    """Датасет для обучения без временных признаков. Возвращает только (cpu_norm, extra)."""

    def __init__(self, target, timestamps, w_input=60, horizon_h=3,
                 extra_channel=None, phi=None):
        self.target = np.asarray(target, dtype=np.float32)
        self.timestamps = np.asarray(timestamps, dtype=np.int64)
        self.w = int(w_input)
        self.h = int(horizon_h)
        self.n = max(0, len(self.target) - self.w - self.h + 1)
        if extra_channel is None:
            raise ValueError("extra_channel is required for d_step=2 dataset")
        extra = np.asarray(extra_channel, dtype=np.float32)
        if len(extra) != len(self.target):
            raise ValueError(
                f"extra_channel length {len(extra)} != target length {len(self.target)}"
            )
        self.extra = extra

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx):
        w, h = self.w, self.h
        x = self.target[idx: idx + w]
        e = self.extra[idx: idx + w]
        y = self.target[idx + w: idx + w + h]
        X = np.stack([x, e], axis=1).astype(np.float32)
        return torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))


# ── Прогнозатор без календарных признаков ────────────────────────────────

class HybridForecasterNoCalendar(HybridForecaster):
    """
    Вариант HybridForecaster с d_step=2.

    Переопределяет:
      - архитектуру сети (GRUQuantileNet с d_step=2)
      - _stack_features для прогноза (возвращает только main + extra)
      - fit для использования TimeSeriesDatasetNoCalendar
    """

    def __init__(self, preprocessor, **kwargs):
        super().__init__(preprocessor=preprocessor, **kwargs)
        # Пересоздаём сеть с d_step=2
        self.net = GRUQuantileNet(
            d_step=2,
            hidden_dim=self.hidden_dim,
            n_layers=2,
            dropout=self.dropout,
            n_quantiles=len(self.quantiles),
            horizon_h=self.horizon_h,
            residual_idx=0,
        )
        old_trainer = self.trainer
        self.trainer = GRUTrainer(
            self.net,
            quantiles=self.quantiles,
            lr=old_trainer.lr,
            lr_decay=old_trainer.lr_decay,
            grad_clip=old_trainer.grad_clip,
            max_epochs=old_trainer.max_epochs,
            patience=old_trainer.patience,
            batch_size=old_trainer.batch_size,
        )

    def _stack_features(self, window_main, window_extra, window_ts):
        """Возвращает (w, 2) — только main + extra, без временных признаков."""
        w = self.w_input
        main_w = np.asarray(window_main, dtype=np.float32)
        extra_w = np.asarray(window_extra, dtype=np.float32)
        if len(main_w) < w:
            pad = w - len(main_w)
            main_w = np.concatenate([np.zeros(pad, dtype=np.float32), main_w])
            extra_w = np.concatenate([np.zeros(pad, dtype=np.float32), extra_w])
        return np.stack([main_w, extra_w], axis=1)

    def fit(self, cpu_series, timestamps, phi=None,
            val_split: float = 0.15, epoch_callback=None):
        """
        Копия основной fit(), но с TimeSeriesDatasetNoCalendar вместо
        стандартного датасета, чтобы обучение шло с d_step=2.
        """
        import logging
        logger = logging.getLogger(__name__)

        cpu_series = np.asarray(cpu_series, dtype=np.float32)
        timestamps = np.asarray(timestamps, dtype=np.int64)
        n = len(cpu_series)
        if n < self.w_input + self.horizon_h + 8:
            raise ValueError(
                f"Series too short ({n}); need at least w_input+horizon_h+8 points."
            )

        # 1. Предобработка
        self.preprocessor._fitted = False
        trend, _, resid_norm, _, _ = self.preprocessor.fit_transform(cpu_series)

        mu_cpu = float(self.preprocessor.mu_cpu)
        sigma_cpu = float(self.preprocessor.sigma_cpu)
        # Главный канал — z-score сырого ряда (а не очищенного): это
        # согласовано с новой редакцией HybridForecaster.fit().
        cpu_norm = ((cpu_series - mu_cpu) / sigma_cpu).astype(np.float32)
        # Нормализованный in-sample наивный прогноз (для _calibrate_conformal).
        naive_in = self.preprocessor.naive_in_sample(n, offset=0)
        naive_norm = ((naive_in - mu_cpu) / sigma_cpu).astype(np.float32)

        self._cpu_train = cpu_series.copy()
        self._timestamps_train = timestamps.copy()

        # 2. Хронологическое разбиение
        n_val = max(int(n * val_split), self.w_input + self.horizon_h + 4)
        n_val = min(n_val, n - self.w_input - self.horizon_h - 4)
        n_tr = n - n_val
        if n_tr <= self.w_input + self.horizon_h:
            raise ValueError(f"Train portion too small: n_tr={n_tr}")

        # 3. Датасеты БЕЗ временных признаков (d_step=2). Согласовано с новой
        # редакцией HybridForecaster.fit(): цель = cpu_norm, extra = resid_norm.
        train_ds = TimeSeriesDatasetNoCalendar(
            target=cpu_norm[:n_tr],
            timestamps=timestamps[:n_tr],
            w_input=self.w_input,
            horizon_h=self.horizon_h,
            extra_channel=resid_norm[:n_tr],
        )
        val_start = n_tr - self.w_input
        val_ds = TimeSeriesDatasetNoCalendar(
            target=cpu_norm[val_start:],
            timestamps=timestamps[val_start:],
            w_input=self.w_input,
            horizon_h=self.horizon_h,
            extra_channel=resid_norm[val_start:],
        )

        # 4. Обучение
        self.trainer.fit(train_ds, val_ds, epoch_callback=epoch_callback)
        self._is_trained = True

        # 5. Конформная калибровка (использует _stack_features — уже 2 признака)
        self._calibrate_conformal(
            cpu_series, timestamps, cpu_norm, resid_norm, n_tr,
        )


# ── Оценка ───────────────────────────────────────────────────────────────

def _run_forecast(forecaster, cpu_train, cpu_test, ts_train, ts_test,
                  phi_train, phi_test):
    h = forecaster.horizon_h
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


def _eval_config(with_calendar: bool, seed: int, cpu, ts, phi):
    n_tr = int(len(cpu) * 0.70)
    n_vl = int(len(cpu) * 0.15)
    cpu_train, cpu_test = cpu[:n_tr], cpu[n_tr + n_vl:]
    ts_train, ts_test = ts[:n_tr], ts[n_tr + n_vl:]
    phi_train, phi_test = phi[:, :n_tr], phi[:, n_tr + n_vl:]

    np.random.seed(seed)
    torch.manual_seed(seed)
    pp = Preprocessor(**CFG_PREPROCESSOR)

    if with_calendar:
        fc = HybridForecaster(preprocessor=pp, **CFG_FORECASTER)
    else:
        fc = HybridForecasterNoCalendar(preprocessor=pp, **CFG_FORECASTER)

    fc.fit(cpu_train, ts_train, phi_train)
    d_step_actual = fc.net.gru.input_size
    blend = float(fc.blend_alpha)
    y_true, y_pred = _run_forecast(
        fc, cpu_train, cpu_test, ts_train, ts_test, phi_train, phi_test,
    )
    mae = compute_mae(y_true, y_pred) if len(y_true) else float("nan")
    return mae, d_step_actual, blend


class TestTable49:
    """С календарными признаками vs без них (Стационарный синт.)."""

    def test_calendar_features(self):
        cpu, ts, phi = generate_stationary(N_OBS_SYNTHETIC, seed=42)
        print(
            f"\n[{_now()}] [T4.9] >>> старт: Стационарный n={len(cpu)} "
            f"(15 суток, суточная сезонность p=288), "
            f"{len(SEEDS)} seeds x 2 конфигурации",
            flush=True,
        )
        t_total = time.time()

        results = {True: [], False: []}

        for with_calendar in [True, False]:
            tag = "with_cal" if with_calendar else "no_cal"
            for step, seed in enumerate(SEEDS, 1):
                gc.collect()
                print(
                    f"[{_now()}] [T4.9][{tag}] [{step}/{len(SEEDS)}] "
                    f"START seed={seed}",
                    flush=True,
                )
                t0 = time.time()
                try:
                    mae, d_step, blend = _eval_config(with_calendar, seed, cpu, ts, phi)
                    results[with_calendar].append(mae)
                    print(
                        f"[{_now()}] [T4.9][{tag}] [{step}/{len(SEEDS)}] "
                        f"DONE  seed={seed} MAE={mae:.4f} "
                        f"d_step={d_step} blend_alpha={blend:.2f} "
                        f"({time.time() - t0:.1f}s)",
                        flush=True,
                    )
                except Exception as e:
                    tb = traceback.format_exc(limit=4)
                    print(
                        f"[{_now()}] [T4.9][{tag}] [ERROR] seed={seed}: "
                        f"{type(e).__name__}: {e}\n{tb}",
                        flush=True,
                    )

        mae_with = np.mean(results[True]) if results[True] else float("nan")
        std_with = np.std(results[True]) if results[True] else float("nan")
        mae_without = np.mean(results[False]) if results[False] else float("nan")
        std_without = np.std(results[False]) if results[False] else float("nan")
        if np.isnan(mae_with) or mae_with <= 0:
            improvement_pct = float("nan")
        else:
            improvement_pct = 100.0 * (mae_without - mae_with) / mae_with

        print(f"\n[{_now()}] [T4.9] === Сводные результаты ===", flush=True)
        print(
            f"[METRIC] TABLE_4_9: dataset=Stationary | config=with_calendar | "
            f"MAE={mae_with:.4f}\u00b1{std_with:.4f} | "
            f"seeds_ok={len(results[True])}/{len(SEEDS)}",
            flush=True,
        )
        print(
            f"[METRIC] TABLE_4_9: dataset=Stationary | config=no_calendar | "
            f"MAE={mae_without:.4f}\u00b1{std_without:.4f} | "
            f"seeds_ok={len(results[False])}/{len(SEEDS)}",
            flush=True,
        )
        print(
            f"[METRIC] TABLE_4_9: improvement_pct={improvement_pct:+.2f}%",
            flush=True,
        )
        print(
            f"\n[{_now()}] [T4.9] <<< завершено за {time.time() - t_total:.0f}с",
            flush=True,
        )
