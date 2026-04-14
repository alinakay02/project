"""
predictor/preprocessor.py — Сбор и предобработка временного ряда (параграф 3.2.1).

Реализованы три этапа:
  1. Робастное обнаружение аномалий по скользящему МКР Тьюки (формула 3.8).
  2. STL-декомпозиция: cpu_t = T_t + S_t + R_t  (формула 3.9).
  3. Глобальная (по обучающему ряду) z-score нормализация остатка
     R̃_t = (R_t − μ_R) / σ_R  (формула 3.10).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Предобработка временного ряда утилизации процессора.

    Параметры (config.yaml → preprocessing):
      w_a       : окно МКР для обнаружения аномалий
      iqr_alpha : коэффициент α метода Тьюки (формула 3.8)
      w_norm    : ширина окна сезонного сглаживания STL (нечётное ≥ 7)
      period    : суточный период (288 = 24ч / 5мин)
      robust    : использовать робастный STL (устойчивость к выбросам)
    """

    def __init__(
        self,
        w_a: int = 48,
        iqr_alpha: float = 1.5,
        w_norm: int = 60,
        period: int = 288,
        robust: bool = True,
    ):
        self.w_a = int(w_a)
        self.iqr_alpha = float(iqr_alpha)
        self.w_norm = int(w_norm)
        self.period = int(period)
        self.robust = bool(robust)

        # Зафиксированные параметры нормализации остатка (после fit_transform)
        self.mu_w: float = 0.0
        self.sigma_w: float = 1.0
        # Зафиксированные параметры нормализации raw cpu (z-score по обучающему ряду)
        self.mu_cpu: float = 0.0
        self.sigma_cpu: float = 1.0
        self._fitted: bool = False

        # Последние компоненты декомпозиции (диагностика)
        self.trend_: Optional[np.ndarray] = None
        self.seasonal_: Optional[np.ndarray] = None
        self.residual_: Optional[np.ndarray] = None
        self.clean_series_: Optional[np.ndarray] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Главный публичный метод
    # ─────────────────────────────────────────────────────────────────────────

    def fit_transform(
        self, series: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Полный цикл предобработки.

        Возвращает:
            trend, seasonal, residual_norm, mu_R, sigma_R
        """
        clean = self._remove_anomalies(series)
        trend, seasonal, residual = self._stl_decompose(clean)

        # Глобальные параметры нормализации фиксируем только при первом fit().
        if not self._fitted:
            self.mu_w = float(np.mean(residual))
            sigma_r = float(np.std(residual))
            self.sigma_w = sigma_r if sigma_r > 1e-8 else 1e-8
            self.mu_cpu = float(np.mean(clean))
            sigma_c = float(np.std(clean))
            self.sigma_cpu = sigma_c if sigma_c > 1e-8 else 1e-8
            self._fitted = True

        residual_norm = ((residual - self.mu_w) / self.sigma_w).astype(np.float32)

        self.clean_series_ = clean.astype(np.float32)
        self.trend_ = trend
        self.seasonal_ = seasonal
        self.residual_ = residual

        return trend, seasonal, residual_norm, self.mu_w, self.sigma_w

    def transform(
        self, series: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Применение предобработки без изменения зафиксированных μ_R / σ_R.
        Используется в HybridForecaster.predict().
        """
        clean = self._remove_anomalies(series)
        trend, seasonal, residual = self._stl_decompose(clean)
        residual_norm = ((residual - self.mu_w) / self.sigma_w).astype(np.float32)

        self.clean_series_ = clean.astype(np.float32)
        self.trend_ = trend
        self.seasonal_ = seasonal
        self.residual_ = residual
        return trend, seasonal, residual_norm, self.mu_w, self.sigma_w

    def inverse_transform_residual(self, residual_norm: np.ndarray) -> np.ndarray:
        """R_t = σ_R · R̃_t + μ_R."""
        return self.sigma_w * residual_norm + self.mu_w

    # ─────────────────────────────────────────────────────────────────────────
    # Шаг 1: робастное обнаружение аномалий (формула 3.8)
    # ─────────────────────────────────────────────────────────────────────────

    def _remove_anomalies(self, series: np.ndarray) -> np.ndarray:
        """
        Векторизованное скользящее обнаружение и интерполяция выбросов.
        Граница окна: только прошлые данные (предотвращение утечки).
        """
        s = np.asarray(series, dtype=np.float64).copy()
        n = len(s)
        if n == 0:
            return s

        ser = pd.Series(s)
        roll = ser.rolling(window=self.w_a, min_periods=8)
        q1 = roll.quantile(0.25).shift(1).to_numpy()
        q3 = roll.quantile(0.75).shift(1).to_numpy()
        iqr = q3 - q1
        lower = q1 - self.iqr_alpha * iqr
        upper = q3 + self.iqr_alpha * iqr
        mask = (~np.isnan(lower)) & ((s < lower) | (s > upper))
        if mask.any():
            idx = np.arange(n)
            valid = ~mask
            s[mask] = np.interp(idx[mask], idx[valid], s[valid])
        return s

    # ─────────────────────────────────────────────────────────────────────────
    # Шаг 2: STL-декомпозиция (формула 3.9)
    # ─────────────────────────────────────────────────────────────────────────

    def _stl_decompose(
        self, clean: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        STL-декомпозиция: cpu_t = T_t + S_t + R_t.

        Параметры сезонного сглаживателя берутся из self.w_norm
        (должно быть нечётное ≥ 7; корректируется автоматически).
        """
        n = len(clean)
        # Если данных меньше 2 полных периодов — fallback на скользящее среднее
        if n < 2 * self.period:
            trend = (
                pd.Series(clean)
                .rolling(window=min(n, self.period), center=True, min_periods=1)
                .mean()
                .to_numpy()
            )
            seasonal = np.zeros_like(trend, dtype=np.float32)
            residual = (clean - trend).astype(np.float32)
            return trend.astype(np.float32), seasonal, residual

        from statsmodels.tsa.seasonal import STL

        seasonal_window = max(self.w_norm, 7)
        if seasonal_window % 2 == 0:
            seasonal_window += 1

        result = STL(
            clean,
            period=self.period,
            seasonal=seasonal_window,
            robust=self.robust,
        ).fit()

        return (
            result.trend.astype(np.float32),
            result.seasonal.astype(np.float32),
            result.resid.astype(np.float32),
        )
