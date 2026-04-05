"""
predictor/forecaster.py — Модуль гибридного прогнозирования (параграф 3.2.2)

Реализует:
  - Прогноз тренда: линейная экстраполяция МНК (формула 3.11)
  - Прогноз сезонности: усреднение по N_c=7 суточным циклам (формула 3.12)
  - Прогноз остатка: GRU с квантильными выходами (формулы 3.13–3.14)
  - Суммирование с обратным преобразованием (формулы 3.14, 3.15)
"""

import math
import os
import logging
import pickle
import numpy as np
import torch
from scipy.stats import linregress
from typing import Tuple, List, Optional

from predictor.model import GRUQuantileNet, GRUTrainer, TimeSeriesDataset
from predictor.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class HybridForecaster:
    """
    Гибридная модель прогнозирования нагрузки.

    Параметры (из config.yaml):
      n_T           : n_T = 60 — окно для линейной экстраполяции тренда
      n_cycles      : N_c = 7 — число суточных циклов для сезонности
      period        : p = 288
      horizon_h     : h = 3 шага
      w_input       : w = 60 — скользящее окно для GRU
      quantiles     : [0.025, 0.5, 0.975]
    """

    def __init__(
        self,
        preprocessor: Preprocessor,
        n_T: int = 60,
        n_cycles: int = 7,
        period: int = 288,
        horizon_h: int = 3,
        w_input: int = 60,
        quantiles: List[float] = (0.025, 0.5, 0.975),
        hidden_dim: int = 64,
        dropout: float = 0.10,
        lr: float = 0.001,
        lr_decay: float = 0.99,
        grad_clip: float = 1.0,
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 32,
    ):
        self.preprocessor = preprocessor
        self.n_T = n_T
        self.n_cycles = n_cycles
        self.period = period
        self.horizon_h = horizon_h
        self.w_input = w_input
        self.quantiles = list(quantiles)

        # Нейросетевая компонента
        self.net = GRUQuantileNet(
            d_in=69,
            hidden_dim=hidden_dim,
            n_layers=2,
            dropout=dropout,
            n_quantiles=len(quantiles),
        )
        self.trainer = GRUTrainer(
            self.net,
            quantiles=quantiles,
            lr=lr,
            lr_decay=lr_decay,
            grad_clip=grad_clip,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
        )
        self._is_trained = False

        # Сохраняем компоненты STL для инкрементального использования
        self._trend: Optional[np.ndarray] = None
        self._seasonal: Optional[np.ndarray] = None
        self._residual_norm: Optional[np.ndarray] = None
        self._timestamps: Optional[np.ndarray] = None
        self._phi: Optional[np.ndarray] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Обучение
    # ─────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        cpu_series: np.ndarray,     # полный обучающий ряд cpu_t
        timestamps: np.ndarray,     # unix-метки времени
        phi: np.ndarray,            # (3, n) — состав классов
        val_split: float = 0.15,
        epoch_callback=None,
    ) -> None:
        """Полный цикл обучения: предобработка → создание датасета → GRU.fit()."""
        logger.info(f"Starting fit on {len(cpu_series)} observations.")

        # Предобработка
        trend, seasonal, resid_norm, mu, sigma = self.preprocessor.fit_transform(cpu_series)
        self._trend = trend
        self._seasonal = seasonal
        self._residual_norm = resid_norm
        self._timestamps = timestamps
        self._phi = phi

        # Хронологическое разделение
        n = len(resid_norm)
        n_val = max(int(n * val_split), self.w_input + 1)
        n_train = n - n_val

        train_ds = TimeSeriesDataset(
            resid_norm[:n_train],
            timestamps[:n_train],
            phi[:, :n_train],
            w_input=self.w_input,
        )
        val_ds = TimeSeriesDataset(
            resid_norm[n_train - self.w_input:],
            timestamps[n_train - self.w_input:],
            phi[:, n_train - self.w_input:],
            w_input=self.w_input,
        )

        self.trainer.fit(train_ds, val_ds, epoch_callback=epoch_callback)
        self._is_trained = True
        logger.info(
            f"Fit complete. Stopped at epoch {self.trainer.stopped_epoch or 'max'}. "
            f"Best val_loss={self.trainer.best_val_loss:.4f}"
        )

    def retrain(
        self,
        cpu_series: np.ndarray,
        timestamps: np.ndarray,
        phi: np.ndarray,
    ) -> None:
        """
        Дообучение на скользящем окне (2016 наблюдений = 7 суток).
        Вызывается каждые 60 итераций основного цикла (параграф 3.2.2).
        """
        logger.info(f"Retraining on window of {len(cpu_series)} obs.")
        # Сброс счётчика ранней остановки для дообучения
        self.trainer.epochs_no_improve = 0
        self.trainer.best_val_loss = float("inf")
        self.fit(cpu_series, timestamps, phi)

    # ─────────────────────────────────────────────────────────────────────────
    # Сохранение / загрузка обученной модели
    # ─────────────────────────────────────────────────────────────────────────

    def save_model(self, path: str) -> None:
        """Сохраняет обученную модель (веса GRU + состояние) на диск."""
        if not self._is_trained:
            raise RuntimeError("Model is not trained. Call fit() first.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "net_state_dict": self.net.state_dict(),
            "best_val_loss": self.trainer.best_val_loss,
            "stopped_epoch": self.trainer.stopped_epoch,
            "train_losses": self.trainer.train_losses,
            "val_losses": self.trainer.val_losses,
            "preprocessor_mu": self.preprocessor.mu_w,
            "preprocessor_sigma": self.preprocessor.sigma_w,
            "trend": self._trend,
            "seasonal": self._seasonal,
            "residual_norm": self._residual_norm,
            "timestamps": self._timestamps,
            "phi": self._phi,
        }
        torch.save(state, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Загружает обученную модель из файла."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.net.load_state_dict(state["net_state_dict"])
        self.trainer.best_val_loss = state["best_val_loss"]
        self.trainer.stopped_epoch = state.get("stopped_epoch")
        self.trainer.train_losses = state.get("train_losses", [])
        self.trainer.val_losses = state.get("val_losses", [])
        self.preprocessor.mu_w = state.get("preprocessor_mu", 0.0)
        self.preprocessor.sigma_w = state.get("preprocessor_sigma", 1.0)
        self._trend = state.get("trend")
        self._seasonal = state.get("seasonal")
        self._residual_norm = state.get("residual_norm")
        self._timestamps = state.get("timestamps")
        self._phi = state.get("phi")
        self._is_trained = True
        logger.info(f"Model loaded from {path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Прогноз
    # ─────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        cpu_series: np.ndarray,
        timestamps: np.ndarray,
        phi: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Формирует прогноз на горизонте h шагов.

        Возвращает:
          cpu_hat     : ĉpu_{t+k} — точечный прогноз (медиана), k=1..h
          q_lower     : q̂_ε(t+k) — нижний квантиль
          q_upper     : q̂_{1-ε}(t+k) — верхний квантиль (используется в формуле 3.7)
        """
        if not self._is_trained:
            raise RuntimeError("Model is not trained. Call fit() first.")

        # Обновляем компоненты декомпозиции
        trend, seasonal, resid_norm, mu, sigma = self.preprocessor.fit_transform(cpu_series)

        cpu_hat = np.zeros(self.horizon_h)
        q_lower = np.zeros(self.horizon_h)
        q_upper = np.zeros(self.horizon_h)

        for k in range(1, self.horizon_h + 1):
            # ── 1. Прогноз тренда T̂_{t+k} (формула 3.11) ─────────────────
            trend_window = trend[-self.n_T:]
            t_idx = np.arange(len(trend_window))
            slope, intercept, _, _, _ = linregress(t_idx, trend_window)
            t_next = len(trend_window) + k - 1
            trend_hat = slope * t_next + intercept

            # ── 2. Прогноз сезонности Ŝ_{t+k} (формула 3.12) ─────────────
            seasonal_hat = self._forecast_seasonal(seasonal, k)

            # ── 3. Прогноз остатка R̂_{t+k} через GRU ─────────────────────
            X = self._build_input(resid_norm, timestamps, phi)
            gru_out = self.trainer.predict(X)   # [q_025, q_50, q_975]
            r_lower_norm, r_median_norm, r_upper_norm = gru_out[0], gru_out[1], gru_out[2]

            # ── 4. Суммирование + обратное преобразование (форм. 3.14, 3.15) ──
            # cpu_hat  = T̂ + Ŝ + σ_w * R̂_0.5 + μ_w
            # q_upper  = T̂ + Ŝ + σ_w * R̂_0.975 + μ_w
            cpu_hat[k - 1] = trend_hat + seasonal_hat + sigma * r_median_norm + mu
            q_lower[k - 1] = trend_hat + seasonal_hat + sigma * r_lower_norm + mu
            q_upper[k - 1] = trend_hat + seasonal_hat + sigma * r_upper_norm + mu

        # Ограничиваем в допустимый диапазон [0, 1]
        cpu_hat = np.clip(cpu_hat, 0.0, 1.0)
        q_lower = np.clip(q_lower, 0.0, 1.0)
        q_upper = np.clip(q_upper, 0.0, 1.0)

        return cpu_hat, q_lower, q_upper

    # ─────────────────────────────────────────────────────────────────────────
    # Вспомогательные методы
    # ─────────────────────────────────────────────────────────────────────────

    def _forecast_seasonal(self, seasonal: np.ndarray, k: int) -> float:
        """
        Ŝ_{t+k} = среднее S_{t+k-j*p} по j=1..N_c (формула 3.12).
        Берём значение из предыдущих суточных циклов.
        """
        values = []
        for j in range(1, self.n_cycles + 1):
            idx = len(seasonal) - self.period * j + k - 1
            if 0 <= idx < len(seasonal):
                values.append(seasonal[idx])
        return float(np.mean(values)) if values else 0.0

    def _build_input(
        self,
        resid_norm: np.ndarray,
        timestamps: np.ndarray,
        phi: np.ndarray,
    ) -> np.ndarray:
        """
        Формирует входную последовательность (w_input, 7) для GRU:
          Каждый шаг: [R̃_t, sin_h, cos_h, sin_d, cos_d, sin_m, cos_m]
        """
        w = self.w_input
        r_window = resid_norm[-w:]
        ts_window = timestamps[-w:]

        if len(r_window) < w:
            pad_n = w - len(r_window)
            r_window = np.concatenate([np.zeros(pad_n), r_window])
            ts_window = np.concatenate([np.full(pad_n, ts_window[0]), ts_window])

        hours = (ts_window % 86400) / 3600.0
        dows = ((ts_window // 86400) % 7).astype(float)
        minutes = ((ts_window % 3600) / 60.0)

        X = np.stack([
            r_window,
            np.sin(2 * math.pi * hours / 24),
            np.cos(2 * math.pi * hours / 24),
            np.sin(2 * math.pi * dows / 7),
            np.cos(2 * math.pi * dows / 7),
            np.sin(2 * math.pi * minutes / 60),
            np.cos(2 * math.pi * minutes / 60),
        ], axis=1)  # (w, 7)

        return X.reshape(1, w, 7)
