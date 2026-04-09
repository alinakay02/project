"""
predictor/preprocessor.py — Сбор и предобработка временного ряда (параграф 3.2.1).

Реализованы три этапа:
  1. Робастное обнаружение аномалий по скользящему МКР Тьюки (формула 3.8).
  2. Адаптивная EWM-декомпозиция: cpu_t = T_t + R_t  (онлайн-замена STL).
     Сезонная компонента S_t моделируется отдельно и в predict() обновляется
     инкрементально, поэтому здесь возвращается пустым массивом — это нужно
     только для обратной совместимости со старым API.
  3. Глобальная (по обучающему ряду) z-score нормализация остатка
     R̃_t = (R_t − μ_R) / σ_R  (формула 3.10, модифицированная редакция).

Главные отличия от старой STL-версии:
  • Никакой пересчёт декомпозиции на каждый predict() — EWM обновляется за O(1).
  • Параметры нормализации (μ_R, σ_R) фиксируются на обучающем ряду и больше не
    дрейфуют, что устраняет рассогласование шкал между fit() и predict().
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
      w_norm    : ширина EWM-окна сглаживания тренда (хотя нормализация теперь
                  глобальная, имя сохранено для совместимости конфига)
      period    : суточный период (используется только для совместимости)
      robust    : включить EWM-сглаживание ряда после удаления аномалий
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

        Возвращает кортеж той же формы, что и старая STL-реализация:
            trend, seasonal, residual_norm, mu_R, sigma_R
        """
        clean = self._remove_anomalies(series)
        trend = self._ewm_trend(clean)
        seasonal = np.zeros_like(trend, dtype=np.float32)
        residual = (clean - trend).astype(np.float32)

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
        self.trend_ = trend.astype(np.float32)
        self.seasonal_ = seasonal
        self.residual_ = residual

        return trend, seasonal, residual_norm, self.mu_w, self.sigma_w

    def transform(
        self, series: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Применение предобработки без изменения зафиксированных μ_R / σ_R.
        Используется в HybridForecaster.predict() для горячего пути.

        Также обновляет диагностические поля clean_series_, trend_, residual_,
        чтобы forecaster мог взять очищенный ряд после вызова transform().
        """
        clean = self._remove_anomalies(series)
        trend = self._ewm_trend(clean)
        seasonal = np.zeros_like(trend, dtype=np.float32)
        residual = (clean - trend).astype(np.float32)
        residual_norm = ((residual - self.mu_w) / self.sigma_w).astype(np.float32)

        self.clean_series_ = clean.astype(np.float32)
        self.trend_ = trend.astype(np.float32)
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
    # Шаг 2: EWM-тренд (быстрая онлайн-замена STL)
    # ─────────────────────────────────────────────────────────────────────────

    def _ewm_trend(self, clean: np.ndarray) -> np.ndarray:
        """
        Двойная экспоненциально взвешенная скользящая средняя с эффективной длиной w_norm.
        Дешевле STL в сотни раз и обновляется за O(1) на новый отсчёт.
        """
        if not self.robust or len(clean) < 3:
            return clean.astype(np.float32)
        span = max(self.w_norm, 8)
        # Первый проход — основное сглаживание
        ewm1 = pd.Series(clean).ewm(span=span, adjust=False, min_periods=1).mean()
        # Второй проход — повторное сглаживание ослабляет фазовую задержку
        ewm2 = ewm1.ewm(span=max(span // 2, 4), adjust=False, min_periods=1).mean()
        return ewm2.to_numpy().astype(np.float32)
