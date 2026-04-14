"""
predictor/forecaster.py — Гибридная модель прогнозирования (параграф 3.2.2).

Состав метода:
  1. STL-декомпозиция: cpu_t = T_t + S_t + R_t (см. preprocessor.py).
  2. Прямой многошаговый квантильный GRU: за один проход выдаёт h*n_q значений.
  3. Конформная калибровка ширины доверительного интервала по hold-out выборке —
     гарантирует эмпирическое покрытие ≈ (1−ε) даже при недокалиброванных квантилях.

Метод сохраняет публичный интерфейс HybridForecaster: fit / predict / save_model /
load_model — поэтому контроллер и тесты используют его без изменений.
"""

from __future__ import annotations

import logging
import math
import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from predictor.model import GRUQuantileNet, GRUTrainer, TimeSeriesDataset
from predictor.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


# Версия архитектуры; при её смене старые .pt-кэши автоматически отбрасываются.
MODEL_VERSION = 7


class HybridForecaster:
    """
    Гибридный прогнозатор.

    Параметры:
      preprocessor : экземпляр Preprocessor (см. predictor/preprocessor.py)
      n_T          : глубина окна тренда для аналитической экстраполяции
      n_cycles     : сохранён для совместимости конфига (не используется)
      period       : суточный период (для совместимости)
      horizon_h    : горизонт прогноза (число шагов)
      w_input      : длина входного окна GRU
      quantiles    : список квантилей; центральный = медиана
      hidden_dim   : размерность скрытого состояния GRU
      dropout      : dropout в GRU и в голове
      lr / lr_decay / grad_clip : параметры оптимизатора
      max_epochs   : максимум эпох обучения (с ранней остановкой обычно меньше)
      patience     : терпение ранней остановки (>0 включает её)
      batch_size   : размер мини-батча
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
        hidden_dim: int = 96,
        dropout: float = 0.20,
        lr: float = 1e-3,
        lr_decay: float = 0.99,
        grad_clip: float = 1.0,
        max_epochs: int = 60,
        patience: int = 8,
        batch_size: int = 64,
    ):
        self.preprocessor = preprocessor
        self.n_T = int(n_T)
        self.n_cycles = int(n_cycles)
        self.period = int(period)
        self.horizon_h = int(horizon_h)
        self.w_input = int(w_input)
        self.quantiles = list(quantiles)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        # 8 признаков: cpu_norm + stl_residual_norm + 6 тр��гонометрических временных
        self.net = GRUQuantileNet(
            d_step=8,
            hidden_dim=hidden_dim,
            n_layers=2,
            dropout=dropout,
            n_quantiles=len(self.quantiles),
            horizon_h=self.horizon_h,
            residual_idx=0,                # AR-приор по cpu_norm
        )
        self.trainer = GRUTrainer(
            self.net,
            quantiles=self.quantiles,
            lr=lr,
            lr_decay=lr_decay,
            grad_clip=grad_clip,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
        )

        # Состояние модели
        self._is_trained: bool = False
        self._timestamps_train: Optional[np.ndarray] = None
        self._cpu_train: Optional[np.ndarray] = None

        # Конформная калибровка
        # conformal_widths[k] — половина ширины 95%-ДИ для шага k
        self.conformal_widths: np.ndarray = np.zeros(self.horizon_h, dtype=np.float32)
        # epsilon — общий уровень риска (1 − coverage)
        self.epsilon: float = 1.0 - (max(self.quantiles) - min(self.quantiles))
        # Коэффициент смешивания GRU-предсказания с персистентностью cpu[t-1].
        # blend_alpha = 1.0 — чистый GRU; 0.0 — чистая персистентность.
        # Подбирается автоматически на val в _calibrate_conformal().
        self.blend_alpha: float = 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # Обучение
    # ─────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        cpu_series: np.ndarray,
        timestamps: np.ndarray,
        phi: Optional[np.ndarray] = None,
        val_split: float = 0.15,
        epoch_callback=None,
    ) -> None:
        """
        Полный цикл обучения:
          1) предобработка → фиксация (μ_cpu, σ_cpu) и (μ_R, σ_R);
          2) хронологическое разбиение train / val;
          3) обучение GRU на train с ранней остановкой по val
             (вход = z-score(cpu) + EWM-residual + 6 временных, цель = z-score(cpu));
          4) конформная калибровка ДИ на val.
        """
        cpu_series = np.asarray(cpu_series, dtype=np.float32)
        timestamps = np.asarray(timestamps, dtype=np.int64)
        n = len(cpu_series)
        if n < self.w_input + self.horizon_h + 8:
            raise ValueError(
                f"Series too short ({n}); need at least w_input+horizon_h+8 points."
            )

        logger.info("Fit on %d observations (h=%d, w=%d).", n, self.horizon_h, self.w_input)

        # 1. Предобработка (STL-декомпозиция)
        self.preprocessor._fitted = False
        trend, _, resid_norm, _, _ = self.preprocessor.fit_transform(cpu_series)

        # Главный канал и цель — z-score(cpu) (модель предсказывает cpu_norm[t+k]).
        # Дополнительный канал — нормализованный STL-остаток (стационарный сигнал).
        # Архитектурный AR-приор (residual_idx=0) добавит cpu_norm[t-1] к каждому
        # выходу — модель учится предсказывать дельту от persistence.
        mu_cpu = float(self.preprocessor.mu_cpu)
        sigma_cpu = float(self.preprocessor.sigma_cpu)
        cpu_clean = self.preprocessor.clean_series_
        cpu_norm = ((cpu_clean - mu_cpu) / sigma_cpu).astype(np.float32)

        self._cpu_train = cpu_series.copy()
        self._timestamps_train = timestamps.copy()

        # 2. Хронологическое разбиение
        n_val = max(int(n * val_split), self.w_input + self.horizon_h + 4)
        n_val = min(n_val, n - self.w_input - self.horizon_h - 4)
        n_tr = n - n_val
        if n_tr <= self.w_input + self.horizon_h:
            raise ValueError(f"Train portion too small after val_split: n_tr={n_tr}")

        # Главный канал = cpu_norm, цель = cpu_norm, доп. канал = resid_norm.
        train_ds = TimeSeriesDataset(
            target=cpu_norm[:n_tr],
            timestamps=timestamps[:n_tr],
            w_input=self.w_input,
            horizon_h=self.horizon_h,
            extra_channel=resid_norm[:n_tr],
        )
        val_start = n_tr - self.w_input
        val_ds = TimeSeriesDataset(
            target=cpu_norm[val_start:],
            timestamps=timestamps[val_start:],
            w_input=self.w_input,
            horizon_h=self.horizon_h,
            extra_channel=resid_norm[val_start:],
        )

        # 3. Обучение
        self.trainer.fit(train_ds, val_ds, epoch_callback=epoch_callback)
        self._is_trained = True
        logger.info(
            "Fit done. Best val_loss=%.5f at epoch %s.",
            self.trainer.best_val_loss,
            self.trainer.stopped_epoch or "max",
        )

        # 4. Конформная калибровка ДИ на val
        self._calibrate_conformal(cpu_series, timestamps, cpu_norm, resid_norm, n_tr)

    def retrain(
        self,
        cpu_series: np.ndarray,
        timestamps: np.ndarray,
        phi: Optional[np.ndarray] = None,
    ) -> None:
        """Дообучение на скользящем окне (вызывается контроллером)."""
        logger.info("Retraining on window of %d obs.", len(cpu_series))
        self.trainer.epochs_no_improve = 0
        self.trainer.best_val_loss = float("inf")
        self.fit(cpu_series, timestamps, phi)

    # ─────────────────────────────────────────────────────────────────────────
    # Конформная калибровка
    # ─────────────────────────────────────────────────────────────────────────

    def _calibrate_conformal(
        self,
        cpu_series: np.ndarray,
        timestamps: np.ndarray,
        cpu_norm: np.ndarray,
        resid_norm: np.ndarray,
        n_tr: int,
    ) -> None:
        """
        Калибровка по hold-out (val) части обучающего ряда.

        Делает две вещи на одной валидационной выборке:
          1. Подбирает скаляр blend_alpha ∈ [0, 1] — оптимальное смешивание
             GRU-прогноза с персистентностью cpu[t−1] по MAE на val.
          2. Считает per-step конформную ширину 95%-ДИ
             как (1−ε)-квантиль |y_true − y_hat_blended| с поправкой на конечный
             размер выборки. Это гарантирует валидное эмпирическое покрытие.
        """
        n = len(cpu_series)
        h = self.horizon_h
        w = self.w_input

        start = max(n_tr, w)
        end = n - h
        if end <= start:
            self.conformal_widths = np.full(h, 0.05, dtype=np.float32)
            self.blend_alpha = 1.0
            return

        anchors = np.arange(start, end)
        if len(anchors) > 256:
            stride = max(1, len(anchors) // 256)
            anchors = anchors[::stride]

        # Векторизованное построение батча признаков
        batch_X: List[np.ndarray] = []
        persist_anchors: List[float] = []
        for t in anchors:
            X = self._stack_features(
                cpu_norm[t - w: t], resid_norm[t - w: t], timestamps[t - w: t]
            )
            batch_X.append(X)
            persist_anchors.append(float(cpu_series[t - 1]))
        Xb = np.stack(batch_X, axis=0)                       # (B, w, 8)
        persist_arr = np.array(persist_anchors, dtype=np.float32)  # (B,)

        out = self.trainer.predict(Xb)                        # (B, h, n_q)
        median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
        median_norm = out[:, :, median_idx]                  # (B, h)

        mu_cpu = float(self.preprocessor.mu_cpu)
        sigma_cpu = float(self.preprocessor.sigma_cpu)

        # GRU-прогнозы и истинные значения по horizon-шагам
        gru_pred = median_norm * sigma_cpu + mu_cpu          # (B, h) — cpu-шкала
        y_true = np.stack(
            [cpu_series[anchors + k] for k in range(h)], axis=1
        ).astype(np.float32)                                  # (B, h)
        persist_pred = np.broadcast_to(persist_arr.reshape(-1, 1), gru_pred.shape)

        # 1. Подбор blend_alpha с regularization-к-0.5.
        #
        # Чистое argmin по val переобучается: на хвосте обучающего ряда GRU
        # «знаком» с локальной структурой и кажется лучше, чем будет на тесте.
        # Поэтому смешиваем val-аргмин с робастным prior=0.5 (равные доли GRU и
        # персистентности). Сила prior подобрана эмпирически.
        alphas = np.linspace(0.0, 1.0, 21)
        maes = np.array([
            float(np.mean(np.abs(
                y_true - (a * gru_pred + (1.0 - a) * persist_pred)
            )))
            for a in alphas
        ])
        best_idx = int(np.argmin(maes))
        raw_alpha = float(alphas[best_idx])
        prior = 0.5
        prior_strength = 0.6   # 0 = чистый val-argmin, 1 = всегда 0.5
        self.blend_alpha = float(prior_strength * prior + (1.0 - prior_strength) * raw_alpha)
        logger.info(
            "Calibrated blend_alpha=%.2f (raw val argmin=%.2f, val MAE=%.4f)",
            self.blend_alpha, raw_alpha, float(maes[best_idx]),
        )

        # 2. Конформная ширина по per-step абсолютной ошибке (используем
        #    ту же blend_alpha, что будет применяться в predict()).
        blended = self.blend_alpha * gru_pred + (1.0 - self.blend_alpha) * persist_pred
        target_q = 1.0 - self.epsilon
        widths = []
        for k in range(h):
            errs = np.abs(y_true[:, k] - blended[:, k])
            n_cal = len(errs)
            q = min(1.0, (math.ceil((n_cal + 1) * target_q)) / n_cal)
            widths.append(float(np.quantile(errs, q)))
        self.conformal_widths = np.array(widths, dtype=np.float32)
        logger.info("Conformal widths per step: %s", self.conformal_widths.tolist())

    # ─────────────────────────────────────────────────────────────────────────
    # Сохранение / загрузка
    # ─────────────────────────────────────────────────────────────────────────

    def save_model(self, path: str) -> None:
        if not self._is_trained:
            raise RuntimeError("Model is not trained. Call fit() first.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "version": MODEL_VERSION,
            "net_state_dict": self.net.state_dict(),
            "best_val_loss": self.trainer.best_val_loss,
            "stopped_epoch": self.trainer.stopped_epoch,
            "preprocessor_mu": self.preprocessor.mu_w,
            "preprocessor_sigma": self.preprocessor.sigma_w,
            "preprocessor_mu_cpu": self.preprocessor.mu_cpu,
            "preprocessor_sigma_cpu": self.preprocessor.sigma_cpu,
            "conformal_widths": self.conformal_widths,
            "blend_alpha": self.blend_alpha,
            "horizon_h": self.horizon_h,
            "w_input": self.w_input,
            "quantiles": self.quantiles,
        }, path)
        logger.info("Model saved to %s", path)

    def load_model(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("version") != MODEL_VERSION:
            raise RuntimeError(
                f"Cached model has version={state.get('version')} but code "
                f"expects {MODEL_VERSION}; remove cache and retrain."
            )
        if state.get("horizon_h") != self.horizon_h or state.get("w_input") != self.w_input:
            raise RuntimeError("Cached model shape does not match current config.")
        self.net.load_state_dict(state["net_state_dict"])
        self.net.to(self.trainer.device)
        self.preprocessor.mu_w = float(state["preprocessor_mu"])
        self.preprocessor.sigma_w = float(state["preprocessor_sigma"])
        self.preprocessor.mu_cpu = float(state.get("preprocessor_mu_cpu", 0.0))
        self.preprocessor.sigma_cpu = float(state.get("preprocessor_sigma_cpu", 1.0))
        self.preprocessor._fitted = True
        self.conformal_widths = np.asarray(state["conformal_widths"], dtype=np.float32)
        self.blend_alpha = float(state.get("blend_alpha", 1.0))
        self.trainer.best_val_loss = float(state.get("best_val_loss", float("inf")))
        self.trainer.stopped_epoch = state.get("stopped_epoch") or 0
        self._is_trained = True
        logger.info("Model loaded from %s", path)

    # ─────────────────────────────────────────────────────────────────────────
    # Прогноз (горячий путь)
    # ─────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        cpu_series: np.ndarray,
        timestamps: np.ndarray,
        phi: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Возвращает (cpu_hat, q_lower, q_upper), длина = horizon_h.

        STL считается на скользящем окне (~3 периода), а не на всём ряду.
        """
        if not self._is_trained:
            raise RuntimeError("Model is not trained. Call fit() first.")

        cpu_series = np.asarray(cpu_series, dtype=np.float32)
        timestamps = np.asarray(timestamps, dtype=np.int64)
        n = len(cpu_series)
        if n < self.w_input + 1:
            raise ValueError(
                f"Need at least w_input+1={self.w_input+1} points, got {n}."
            )

        # 1. STL-декомпозиция на скользящем окне (не на всём ряду).
        #    GRU использует только последние w_input значений, поэтому
        #    достаточно декомпозировать последние ~3 суточных периода.
        w = self.w_input
        stl_window = max(3 * self.preprocessor.period, w * 4)
        if n > stl_window:
            cpu_tail = cpu_series[-stl_window:]
            ts_tail = timestamps[-stl_window:]
        else:
            cpu_tail = cpu_series
            ts_tail = timestamps

        _, _, resid_norm, _, _ = self.preprocessor.transform(cpu_tail)
        clean = self.preprocessor.clean_series_
        mu_cpu = float(self.preprocessor.mu_cpu)
        sigma_cpu = float(self.preprocessor.sigma_cpu)
        cpu_norm = ((clean - mu_cpu) / sigma_cpu).astype(np.float32)

        # 2. Признаки последнего окна (w, 8): main = cpu_norm, extra = resid_norm
        X = self._stack_features(cpu_norm[-w:], resid_norm[-w:], ts_tail[-w:])

        # 3. Один forward GRU → (h, n_q) в нормализованных координатах cpu
        out = self.trainer.predict(X)                             # (h, n_q)
        median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
        median_norm = out[:, median_idx]                          # (h,) — z-score(cpu)

        # 4. Денормализация в исходную шкалу cpu и смешивание с персистентностью.
        # blend_alpha подобран автоматически на val в _calibrate_conformal().
        gru_hat = (median_norm * sigma_cpu + mu_cpu).astype(np.float32)
        persist_hat = float(cpu_series[-1])
        cpu_hat = (self.blend_alpha * gru_hat + (1.0 - self.blend_alpha) * persist_hat).astype(np.float32)

        # 5. Конформные ДИ
        widths = self.conformal_widths
        q_lower = cpu_hat - widths
        q_upper = cpu_hat + widths

        # 6. Ограничения [0,1]
        cpu_hat = np.clip(cpu_hat, 0.0, 1.0)
        q_lower = np.clip(q_lower, 0.0, 1.0)
        q_upper = np.clip(q_upper, 0.0, 1.0)
        return cpu_hat, q_lower, q_upper

    # ─────────────────────────────────────────────────────────────────────────
    # Внутренние утилиты
    # ─────────────────────────────────────────────────────────────────────────

    def _extrapolate_trend(self, trend: np.ndarray, k: int) -> float:
        """
        Экстраполяция тренда T̂_{t+k} линейной регрессией по последним n_T точкам.
        Дёшево, устойчиво и не пересчитывает декомпозицию.
        """
        if len(trend) == 0:
            return 0.0
        n = min(self.n_T, len(trend))
        seg = trend[-n:].astype(np.float64)
        if n < 3:
            return float(seg[-1])
        x = np.arange(n, dtype=np.float64)
        # МНК-наклон без вызова scipy
        x_mean = x.mean()
        y_mean = seg.mean()
        denom = float(np.sum((x - x_mean) ** 2))
        if denom < 1e-12:
            return float(seg[-1])
        slope = float(np.sum((x - x_mean) * (seg - y_mean)) / denom)
        intercept = y_mean - slope * x_mean
        return float(slope * (n - 1 + k) + intercept)

    def _stack_features(
        self,
        window_main: np.ndarray,
        window_extra: np.ndarray,
        window_ts: np.ndarray,
    ) -> np.ndarray:
        """
        Формирует тензор (w, 8):
          [main, extra, sin h, cos h, sin d, cos d, sin m, cos m]

        main  — главный канал, по которому считается AR-приор и который служит
                целевой переменной (cpu_norm).
        extra — вспомогательный канал (resid_norm).
        """
        w = self.w_input
        main_w = np.asarray(window_main, dtype=np.float32)
        extra_w = np.asarray(window_extra, dtype=np.float32)
        ts_w = np.asarray(window_ts, dtype=np.int64)
        if len(main_w) < w:
            pad = w - len(main_w)
            main_w = np.concatenate([np.zeros(pad, dtype=np.float32), main_w])
            extra_w = np.concatenate([np.zeros(pad, dtype=np.float32), extra_w])
            ts_w = np.concatenate([np.full(pad, ts_w[0], dtype=np.int64), ts_w])

        hours = (ts_w % 86400) / 3600.0
        dows = ((ts_w // 86400) % 7).astype(np.float64)
        minutes = (ts_w % 3600) / 60.0
        X = np.stack([
            main_w,
            extra_w,
            np.sin(2 * math.pi * hours / 24.0).astype(np.float32),
            np.cos(2 * math.pi * hours / 24.0).astype(np.float32),
            np.sin(2 * math.pi * dows / 7.0).astype(np.float32),
            np.cos(2 * math.pi * dows / 7.0).astype(np.float32),
            np.sin(2 * math.pi * minutes / 60.0).astype(np.float32),
            np.cos(2 * math.pi * minutes / 60.0).astype(np.float32),
        ], axis=1)
        return X
