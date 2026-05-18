"""
predictor/preprocessor.py — Сбор и предобработка временного ряда.

Реализованы этапы:
  1. Робастное обнаружение аномалий по скользящему МКР Тьюки.
  2. STL-декомпозиция: cpu_t = T_t + S_t + R_t.
  3. Глобальная (по обучающему ряду) z-score нормализация остатка
     R̃_t = (R_t − μ_R) / σ_R.
  4. Хранение фазового сезонного профиля и тренда для использования в
     качестве аналитического «наивного» прогноза (анкер для GRU-коррекции).
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
      iqr_alpha : коэффициент α метода Тьюки
      w_norm    : ширина окна сезонного сглаживания STL (нечётное ≥ 7).
                  Автоматически ограничивается доступным числом циклов.
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

        # Параметры нормализации остатка STL (после fit_transform)
        self.mu_w: float = 0.0
        self.sigma_w: float = 1.0
        # Параметры нормализации raw cpu (z-score по обучающему ряду, на СЫРЫХ данных)
        self.mu_cpu: float = 0.0
        self.sigma_cpu: float = 1.0
        self._fitted: bool = False

        # Сезонный профиль по фазам суток (period элементов) — усреднённый по
        # всем доступным циклам. Используется как аналитический наивный прогноз
        # сезонной компоненты в predict(): S̄[phase] → Ŝ(t+k).
        self.seasonal_by_phase: Optional[np.ndarray] = None
        # Линейная экстраполяция тренда: (slope, intercept) по последним n_T
        # точкам train. Фиксируется в fit_transform и используется в predict.
        self.trend_level: float = 0.0
        self.trend_slope: float = 0.0
        self.mu_raw: float = 0.0  # среднее исходного ряда (центр тренда)

        # Последние компоненты декомпозиции (диагностика / последнее окно)
        self.trend_: Optional[np.ndarray] = None
        self.seasonal_: Optional[np.ndarray] = None
        self.residual_: Optional[np.ndarray] = None
        self.clean_series_: Optional[np.ndarray] = None
        # Сырой ряд (без удаления аномалий) — для целевой нормализации
        self.raw_series_: Optional[np.ndarray] = None

    def fit_transform(
        self, series: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Полный цикл предобработки.

        Возвращает:
            trend, seasonal, residual_norm, mu_R, sigma_R
        """
        raw = np.asarray(series, dtype=np.float64)
        clean = self._remove_anomalies(raw)
        trend, seasonal, residual = self._stl_decompose(clean)

        # Нормализация фиксируется только при первом fit().
        if not self._fitted:
            self.mu_w = float(np.mean(residual))
            sigma_r = float(np.std(residual))
            self.sigma_w = sigma_r if sigma_r > 1e-8 else 1e-8
            # Статистики по СЫРОМУ ряду — модель выдаёт прогноз в шкале сырых
            # измерений и сравнивается с ними же на тесте.
            self.mu_cpu = float(np.mean(raw))
            sigma_c = float(np.std(raw))
            self.sigma_cpu = sigma_c if sigma_c > 1e-8 else 1e-8
            self.mu_raw = self.mu_cpu

            self._fit_naive_anchors(trend, seasonal)
            self._fitted = True

        residual_norm = ((residual - self.mu_w) / self.sigma_w).astype(np.float32)

        self.clean_series_ = clean.astype(np.float32)
        self.raw_series_ = raw.astype(np.float32)
        self.trend_ = trend
        self.seasonal_ = seasonal
        self.residual_ = residual

        return trend, seasonal, residual_norm, self.mu_w, self.sigma_w

    def transform(
        self, series: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Применение предобработки без изменения зафиксированных параметров.
        Используется в HybridForecaster.predict().
        """
        raw = np.asarray(series, dtype=np.float64)
        clean = self._remove_anomalies(raw)
        trend, seasonal, residual = self._stl_decompose(clean)
        residual_norm = ((residual - self.mu_w) / self.sigma_w).astype(np.float32)

        self.clean_series_ = clean.astype(np.float32)
        self.raw_series_ = raw.astype(np.float32)
        self.trend_ = trend
        self.seasonal_ = seasonal
        self.residual_ = residual
        return trend, seasonal, residual_norm, self.mu_w, self.sigma_w

    def inverse_transform_residual(self, residual_norm: np.ndarray) -> np.ndarray:
        """R_t = σ_R · R̃_t + μ_R."""
        return self.sigma_w * residual_norm + self.mu_w

    # ─────────────────────────────────────────────────────────────────────────
    # Наивный прогноз: T̂(k) + Ŝ(phase) — аналитический анкер для GRU.
    # ─────────────────────────────────────────────────────────────────────────

    def _fit_naive_anchors(
        self, trend: np.ndarray, seasonal: np.ndarray
    ) -> None:
        """
        Фиксирует компоненты наивного прогноза один раз на обучающем ряде:

          • сезонный профиль seasonal_by_phase[phase] — усреднённая сезонная
            компонента по фазам суток (period элементов). В predict()
            используется как Ŝ(t+k) = seasonal_by_phase[phase(t+k)].
          • линейная экстраполяция тренда по последним min(period, len) точкам:
            T̂(t+k) = trend_level + trend_slope * (k+1).

        Хранение этих величин отдельно от STL-декомпозиции позволяет выдавать
        наивный прогноз в predict() без повторного запуска STL, и превращает
        STL из простой «фичи» в полноценный анкер модели.
        """
        n = len(trend)
        p = self.period
        if n < 2 or p <= 0:
            self.seasonal_by_phase = np.zeros(max(p, 1), dtype=np.float64)
            self.trend_level = float(trend[-1]) if n else 0.0
            self.trend_slope = 0.0
            return

        # 1. Сезонный профиль: среднее seasonal[j] по каждой фазе phase(j)=j%period.
        #    Индексирование по линейной позиции — устойчиво к tz, НЕ зависит от ts.
        prof = np.zeros(p, dtype=np.float64)
        counts = np.zeros(p, dtype=np.int64)
        idx = np.arange(n) % p
        np.add.at(prof, idx, seasonal.astype(np.float64))
        np.add.at(counts, idx, 1)
        counts = np.where(counts == 0, 1, counts)
        prof = prof / counts
        # Центрируем (STL сезонная по построению имеет нулевое среднее, но
        # фаз может быть неполное число → вычитаем остаточный дрифт).
        prof = prof - float(np.mean(prof))
        self.seasonal_by_phase = prof

        # 2. Линейная экстраполяция тренда по последним min(p, n) точкам.
        tail = min(p, n)
        seg = trend[-tail:].astype(np.float64)
        x = np.arange(tail, dtype=np.float64)
        x_mean = x.mean()
        y_mean = seg.mean()
        denom = float(np.sum((x - x_mean) ** 2))
        slope = 0.0 if denom < 1e-12 else float(
            np.sum((x - x_mean) * (seg - y_mean)) / denom
        )
        # level = значение тренда в последней точке обучающего ряда.
        self.trend_level = float(seg[-1])
        self.trend_slope = slope

    def naive_forecast(
        self, h: int, phase_start: int, last_value: Optional[float] = None,
    ) -> np.ndarray:
        """
        Наивный прогноз на h шагов вперёд.

        Стратегия — «сезонная коррекция персистентности»:
            ĥ(k) = last_value + (S[phase_start+k] − S[phase_start−1])
        где last_value — последнее наблюдение контекстного ряда.
        Такой прогноз:
          • на сильно сезонных данных приближает истинный ход S-составляющей,
          • на стационарных данных сводится к персистентности (robust fallback),
          • не страдает от устаревания trend_level в rolling-evaluation.

        Если last_value не задан, используется trend_level (фиксированный на fit)
        плюс линейная экстраполяция тренда — это режим in-sample (например,
        для формирования анкеров на обучающей выборке).
        """
        p = self.period
        if self.seasonal_by_phase is None or p <= 0:
            base = last_value if last_value is not None else self.trend_level
            return np.full(h, base, dtype=np.float32)
        prof = self.seasonal_by_phase
        if last_value is not None:
            s_prev = float(prof[(phase_start - 1) % p])
            out = np.empty(h, dtype=np.float32)
            for k in range(h):
                s_k = float(prof[(phase_start + k) % p])
                out[k] = float(last_value) + (s_k - s_prev)
            return out
        # In-sample режим (last_value не задан): линейная экстраполяция тренда.
        out = np.empty(h, dtype=np.float32)
        for k in range(h):
            t_k = self.trend_level + self.trend_slope * (k + 1)
            s_k = float(prof[(phase_start + k) % p])
            out[k] = t_k + s_k
        return out

    def naive_in_sample(
        self, n: int, offset: int = 0
    ) -> np.ndarray:
        """
        In-sample «наивный» ряд длины n, начиная с фазы `offset`:
          ĥ(j) = trend[j] (восстановленный) + seasonal_by_phase[(offset+j) % period].

        Используется при формировании целевой переменной обучения
        (target = raw - naive) так, чтобы сеть изучала именно остаточную
        структуру. На обучении trend в in-sample известен точно (STL-fit),
        поэтому используется self.trend_ (полный STL-тренд) + усреднённый
        сезонный профиль.
        """
        p = self.period
        if self.trend_ is None or self.seasonal_by_phase is None or p <= 0:
            return np.zeros(n, dtype=np.float32)
        trend = self.trend_.astype(np.float32)
        if len(trend) < n:
            n = len(trend)
        prof = self.seasonal_by_phase.astype(np.float32)
        idx = (np.arange(n, dtype=np.int64) + int(offset)) % p
        return (trend[:n] + prof[idx]).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Шаг 1: робастное обнаружение аномалий
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
    # Шаг 2: STL-декомпозиция
    # ─────────────────────────────────────────────────────────────────────────

    def _stl_decompose(
        self, clean: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        STL-декомпозиция: cpu_t = T_t + S_t + R_t.

        Адаптивный выбор окна сезонного сглаживания:
        при коротких рядах (<6 периодов) сезонное сглаживание LOESS с большим
        окном вырождается и сваливает всю вариацию в тренд+остаток. Здесь мы
        уменьшаем seasonal-окно, если доступных циклов меньше self.w_norm.
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

        n_cycles = n // max(self.period, 1)
        # Адаптивный seasonal-смуфер: не больше 2*циклов+1, но и не больше w_norm.
        # При малом числе циклов это ограничивает «размазывание» сезонности по
        # дням и позволяет сезонной компоненте нести реальную вариацию.
        adaptive = max(7, min(self.w_norm, 2 * max(n_cycles, 2) + 1))
        if adaptive % 2 == 0:
            adaptive += 1

        result = STL(
            clean,
            period=self.period,
            seasonal=adaptive,
            robust=self.robust,
        ).fit()

        return (
            result.trend.astype(np.float32),
            result.seasonal.astype(np.float32),
            result.resid.astype(np.float32),
        )
