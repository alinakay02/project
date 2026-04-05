"""
predictor/preprocessor.py — Модуль сбора и предобработки данных (параграф 3.2.1)

Реализует три этапа:
  1. Обнаружение аномалий методом скользящего МКР (формула 3.8)
  2. Декомпозиция STL: cpu_t = T_t + S_t + R_t  (формула 3.9)
  3. Нормализация остатка R̃_t = (R_t - μ_w) / σ_w  (формула 3.10)
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Предобработка временного ряда утилизации процессора.

    Параметры (из config.yaml → preprocessing):
        w_a     : окно обнаружения аномалий (48 наблюдений = 4 часа при Δt=5 мин)
        iqr_alpha: коэффициент α = 1.5 (метод Тьюки, формула 3.8)
        w_norm  : окно нормализации (60 наблюдений = 5 часов)
        period  : p = 288 (суточный период при Δt=5 мин)
        robust  : True (STL с перевзвешиванием выбросов)
    """

    def __init__(
        self,
        w_a: int = 48,
        iqr_alpha: float = 1.5,
        w_norm: int = 60,
        period: int = 288,
        robust: bool = True,
    ):
        self.w_a = w_a
        self.iqr_alpha = iqr_alpha
        self.w_norm = w_norm
        self.period = period
        self.robust = robust

        # Параметры нормализации — сохраняются для обратного преобразования
        self.mu_w: float = 0.0
        self.sigma_w: float = 1.0

        # Компоненты декомпозиции последнего вызова fit_transform
        self.trend_: Optional[np.ndarray] = None
        self.seasonal_: Optional[np.ndarray] = None
        self.residual_: Optional[np.ndarray] = None
        self.clean_series_: Optional[np.ndarray] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Публичный интерфейс
    # ─────────────────────────────────────────────────────────────────────────

    def fit_transform(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Полный цикл предобработки.

        Аргументы:
            series: одномерный массив значений cpu_t длиной >= period

        Возвращает:
            trend     : T_t — трендовая компонента
            seasonal  : S_t — сезонная компонента (период p)
            residual_norm: R̃_t — нормализованный остаток
            mu_w      : μ_w(t) — скользящее среднее остатка
            sigma_w   : σ_w(t) — скользящее стандартное отклонение
        """
        # Шаг 1: обнаружение и замена аномалий (формула 3.8)
        clean = self._remove_anomalies(series)
        self.clean_series_ = clean

        # Шаг 2: STL-декомпозиция (формула 3.9)
        trend, seasonal, residual = self._stl_decompose(clean)
        self.trend_ = trend
        self.seasonal_ = seasonal
        self.residual_ = residual

        # Шаг 3: нормализация остатка (формула 3.10)
        residual_norm, mu, sigma = self._normalize(residual)
        self.mu_w = mu
        self.sigma_w = sigma

        return trend, seasonal, residual_norm, mu, sigma

    def inverse_transform_residual(self, residual_norm: np.ndarray) -> np.ndarray:
        """
        Обратное преобразование нормализованного остатка:
        R_t = σ_w(t) * R̃_t + μ_w(t)
        Параметры μ_w, σ_w строго берутся по данным до момента t (нет утечки).
        """
        return self.sigma_w * residual_norm + self.mu_w

    # ─────────────────────────────────────────────────────────────────────────
    # Шаг 1: Обнаружение аномалий
    # ─────────────────────────────────────────────────────────────────────────

    def _remove_anomalies(self, series: np.ndarray) -> np.ndarray:
        """
        Метод скользящего МКР (Тьюки, формула 3.8).
        Границы: [Q1 - α*IQR, Q3 + α*IQR], окно w_a.
        Аномальные значения заменяются линейной интерполяцией.
        """
        s = series.copy().astype(float)
        n = len(s)
        anomaly_mask = np.zeros(n, dtype=bool)

        for i in range(n):
            # Скользящее окно — только прошлые данные (избегаем утечки)
            lo = max(0, i - self.w_a)
            window = s[lo:i] if i > 0 else s[:1]
            if len(window) < 4:
                continue
            q1, q3 = np.percentile(window, [25, 75])
            iqr = q3 - q1
            lower = q1 - self.iqr_alpha * iqr
            upper = q3 + self.iqr_alpha * iqr
            if s[i] < lower or s[i] > upper:
                anomaly_mask[i] = True

        # Линейная интерполяция аномальных значений
        if anomaly_mask.any():
            n_anomalies = int(anomaly_mask.sum())
            logger.debug(f"Detected {n_anomalies} anomalies, applying linear interpolation")
            idx = np.arange(n)
            valid_idx = idx[~anomaly_mask]
            valid_vals = s[~anomaly_mask]
            if len(valid_idx) > 1:
                s[anomaly_mask] = np.interp(
                    idx[anomaly_mask], valid_idx, valid_vals
                )

        return s

    # ─────────────────────────────────────────────────────────────────────────
    # Шаг 2: STL-декомпозиция
    # ─────────────────────────────────────────────────────────────────────────

    def _stl_decompose(
        self, series: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        STL-декомпозиция: cpu_t = T_t + S_t + R_t (формула 3.9).
        Параметры: period=288, robust=True.
        Минимальная длина ряда: 2*period.
        """
        if len(series) < 2 * self.period:
            logger.warning(
                f"Series too short for STL ({len(series)} < {2*self.period}). "
                "Using fallback (zero trend and seasonal)."
            )
            return np.zeros_like(series), np.zeros_like(series), series.copy()

        stl = STL(series, period=self.period, robust=self.robust)
        result = stl.fit()
        return (
            np.array(result.trend),
            np.array(result.seasonal),
            np.array(result.resid),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Шаг 3: Нормализация
    # ─────────────────────────────────────────────────────────────────────────

    def _normalize(
        self, residual: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        Адаптивная нормализация по скользящему окну w_norm (формула 3.10):
        R̃_t = (R_t - μ_w(t)) / σ_w(t)
        μ_w и σ_w вычисляются по последним w_norm значениям.
        """
        window = residual[-self.w_norm:] if len(residual) >= self.w_norm else residual
        mu = float(np.mean(window))
        sigma = float(np.std(window))
        if sigma < 1e-8:
            sigma = 1e-8  # защита от деления на ноль
        residual_norm = (residual - mu) / sigma
        return residual_norm, mu, sigma
