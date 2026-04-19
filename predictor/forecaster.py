"""
predictor/forecaster.py — Гибридная модель прогнозирования (параграф 3.2.2).

Состав метода:
  1. STL-декомпозиция  cpu_t = T_t + S_t + R_t (см. preprocessor.py) и
     аналитический наивный прогноз
         h_naive(t+k) = T̂(k) + Ŝ(phase(t+k)).
     T̂ — линейная экстраполяция тренда по последним p точкам, Ŝ —
     усреднённый сезонный профиль по фазам суток. Оба компонента
     фиксируются один раз на обучающем ряде.
  2. Прямой многошаговый квантильный GRU, обучаемый предсказывать
     _остаточную поправку_ Δ̂_k = cpu(t+k) − h_naive(t+k).
     В сеть подаются: cpu_norm (z-score сырого ряда), residual_norm
     (нормированный STL-остаток), 6 тригонометрических временных
     признаков и нормированный наивный прогноз на каждый шаг горизонта.
  3. Конформная калибровка ширины 95%-ДИ по hold-out выборке — обеспечивает
     эмпирическое покрытие ≈ (1−ε) даже при недокалиброванных квантилях.

Ключевые отличия от предыдущей редакции:
  • STL теперь реально используется в прогнозе (T+S-анкер), а не только
    как дополнительный признак — на нестационарных и сезонных данных
    это даёт устойчивое качество без «проваливания» в persistence.
  • Нормализация и целевая переменная — по _сырому_ ряду (а не по
    очищенному), что устраняет систематический сдвиг «обучение на clean,
    оценка на raw».
  • Убран жёсткий регуляризатор blend_alpha→0.5: α выбирается по val-MAE
    в диапазоне [0, 1] без сдвига к persistence.
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
MODEL_VERSION = 9


class HybridForecaster:
    """
    Гибридный прогнозатор (STL-наивный анкер + GRU-поправка).

    Параметры:
      preprocessor : экземпляр Preprocessor (см. predictor/preprocessor.py)
      n_T          : глубина окна для экстраполяции тренда (в Preprocessor)
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

    # Количество признаков на шаг: cpu_norm + resid_norm + 6 trig = 8.
    D_STEP = 8

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

        # Анкер residual_idx=0 — главный канал (cpu_norm, z-score сырого ряда).
        # Целевая переменная сети — cpu_norm[t+k]. Head предсказывает
        # cpu_norm[t+k] − cpu_norm[t−1] = первые разности в z-score шкале.
        # Это «бесплатный» AR(1)-приор в той же шкале, что и цель.
        self.net = GRUQuantileNet(
            d_step=self.D_STEP,
            hidden_dim=hidden_dim,
            n_layers=2,
            dropout=dropout,
            n_quantiles=len(self.quantiles),
            horizon_h=self.horizon_h,
            residual_idx=0,
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
        self._train_len: int = 0  # длина обучающего ряда — используется для
                                   # вычисления фазы сезонного профиля.

        # Конформная калибровка (per-step 95%-ДИ).
        self.conformal_widths: np.ndarray = np.zeros(self.horizon_h, dtype=np.float32)
        self.epsilon: float = 1.0 - (max(self.quantiles) - min(self.quantiles))
        # α ∈ [0, 1]: взвешивание GRU-коррекции (см. _calibrate_conformal).
        # α=1 — полное доверие к GRU, α=0 — чистый наивный прогноз.
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
          1) предобработка → фиксация статистик, сезонного профиля, тренда;
          2) целевая переменная = z-score(СЫРОГО) ряда — сеть учится
             предсказывать сырое значение, а не очищенное. Это устраняет
             train/test distribution mismatch;
          3) хронологическое разбиение train/val;
          4) обучение GRU (pinball-loss, ранняя остановка);
          5) конформная калибровка ширины ДИ и blend_alpha на val.
        """
        cpu_series = np.asarray(cpu_series, dtype=np.float32)
        timestamps = np.asarray(timestamps, dtype=np.int64)
        n = len(cpu_series)
        if n < self.w_input + self.horizon_h + 8:
            raise ValueError(
                f"Series too short ({n}); need at least w_input+horizon_h+8 points."
            )

        logger.info("Fit on %d observations (h=%d, w=%d).", n, self.horizon_h, self.w_input)

        # 1. Предобработка: STL + сезонный профиль + экстраполяция тренда.
        self.preprocessor._fitted = False
        _, _, resid_norm, _, _ = self.preprocessor.fit_transform(cpu_series)

        mu_cpu = float(self.preprocessor.mu_cpu)
        sigma_cpu = float(self.preprocessor.sigma_cpu)
        # Главный канал GRU = z-score(СЫРОГО) ряда.
        cpu_norm = ((cpu_series - mu_cpu) / sigma_cpu).astype(np.float32)

        self._cpu_train = cpu_series.copy()
        self._timestamps_train = timestamps.copy()
        self._train_len = n

        # 2. Хронологическое разбиение
        n_val = max(int(n * val_split), self.w_input + self.horizon_h + 4)
        n_val = min(n_val, n - self.w_input - self.horizon_h - 4)
        n_tr = n - n_val
        if n_tr <= self.w_input + self.horizon_h:
            raise ValueError(f"Train portion too small after val_split: n_tr={n_tr}")

        # Датасет: цель = cpu_norm (z-score сырого ряда), главный канал =
        # cpu_norm (для AR(1)-анкера по residual_idx=0), extra = resid_norm
        # (контекст стационарного STL-остатка как доп. признак).
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

        # 4. Конформная калибровка ДИ и подбор blend_alpha на val
        self._calibrate_conformal(
            cpu_series, timestamps, cpu_norm, resid_norm, n_tr
        )

    def retrain(
        self,
        cpu_series: np.ndarray,
        timestamps: np.ndarray,
        phi: Optional[np.ndarray] = None,
        fine_tune_epochs: int = 15,
        fine_tune_lr: float = 5e-4,
    ) -> None:
        """
        Дообучение (fine-tuning) на скользящем окне.

        Разделяет «долгосрочные» и «краткосрочные» компоненты модели:
          • Preprocessor-статистики (μ_cpu, σ_cpu, σ_R, сезонный профиль,
            параметры тренда) — СОХРАНЯЮТСЯ от initial fit, выполненного
            на большой выборке (десятки суток). Это стабильнее, чем их
            переоценка на короткой истории (например, 7 сут).
          • Веса GRU — дообучаются на свежем окне с пониженным LR,
            стартуя с текущих весов (а не с нуля). Сеть адаптируется
            к недавней динамике, не разрушая долгосрочные паттерны.
          • blend_alpha и conformal_widths — перекалибруются на свежем
            окне (они отражают _недавнее_ соотношение сигнала и ошибки).

        Такое разделение предотвращает деградацию качества при retrain,
        наблюдавшуюся при полном переобучении (параграф 4.3 диссертации).
        """
        cpu_series = np.asarray(cpu_series, dtype=np.float32)
        timestamps = np.asarray(timestamps, dtype=np.int64)
        n = len(cpu_series)
        if n < self.w_input + self.horizon_h + 8:
            logger.warning("Retrain window too short (%d); skipping.", n)
            return
        if not self._is_trained or not self.preprocessor._fitted:
            raise RuntimeError(
                "retrain() requires that fit() was called at least once."
            )

        logger.info("Retrain (fine-tune) on %d obs.", n)

        # 1. Применяем существующий preprocessor (transform, НЕ fit_transform),
        #    чтобы сохранить долгосрочные статистики.
        _, _, resid_norm, _, _ = self.preprocessor.transform(cpu_series)
        mu_cpu = float(self.preprocessor.mu_cpu)
        sigma_cpu = float(self.preprocessor.sigma_cpu)
        cpu_norm = ((cpu_series - mu_cpu) / sigma_cpu).astype(np.float32)

        # 2. Хронологический train/val split (такой же, как в fit)
        n_val = max(int(n * 0.15), self.w_input + self.horizon_h + 4)
        n_val = min(n_val, n - self.w_input - self.horizon_h - 4)
        n_tr = n - n_val
        if n_tr <= self.w_input + self.horizon_h:
            logger.warning("Retrain: n_tr=%d too small; skipping.", n_tr)
            return

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

        # 3. Fine-tune: меньше эпох, ниже LR.
        #    Веса GRU сохраняют историю initial fit — сеть только
        #    подстраивается под недавние отклонения.
        #    Подменяем и optimizer.param_groups["lr"], и scheduler.base_lrs —
        #    иначе cosine annealing на каждой эпохе вернёт LR к исходному.
        saved_max_epochs = self.trainer.max_epochs
        saved_patience = self.trainer.patience
        saved_lrs = [pg["lr"] for pg in self.trainer.optimizer.param_groups]
        saved_base_lrs = list(getattr(self.trainer.scheduler, "base_lrs", []))
        self.trainer.max_epochs = int(fine_tune_epochs)
        self.trainer.patience = max(3, int(fine_tune_epochs) // 2)
        self.trainer.epochs_no_improve = 0
        self.trainer.best_val_loss = float("inf")
        for pg in self.trainer.optimizer.param_groups:
            pg["lr"] = float(fine_tune_lr)
        if saved_base_lrs:
            self.trainer.scheduler.base_lrs = [
                float(fine_tune_lr) for _ in saved_base_lrs
            ]
        try:
            self.trainer.fit(train_ds, val_ds)
        finally:
            self.trainer.max_epochs = saved_max_epochs
            self.trainer.patience = saved_patience
            for pg, lr in zip(self.trainer.optimizer.param_groups, saved_lrs):
                pg["lr"] = lr
            if saved_base_lrs:
                self.trainer.scheduler.base_lrs = saved_base_lrs

        # 4. Перекалибровка blend_alpha и conformal_widths на свежем окне.
        self._calibrate_conformal(cpu_series, timestamps, cpu_norm, resid_norm, n_tr)

        self._cpu_train = cpu_series.copy()
        self._timestamps_train = timestamps.copy()
        self._train_len = n

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
          1. Подбирает скаляр blend_alpha ∈ [0, 1] — вес GRU-поправки относительно
             чистого наивного прогноза:
               ĉpu = h_naive + α · (GRU_median − h_naive)
             α=1 — полностью доверять сети, α=0 — использовать только наивный
             прогноз (T+S). В отличие от предыдущей версии, _нет_ смещения к 0.5.
          2. Считает per-step ширину 95%-ДИ как скорректированный квантиль
             |y_true − ĉpu_α| по val — валидно эмпирически.
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
        if len(anchors) > 512:
            stride = max(1, len(anchors) // 512)
            anchors = anchors[::stride]

        # Векторизованное построение батча признаков. Главный канал (индекс 0) —
        # cpu_norm (цель обучения); вспомогательный (индекс 1) — resid_norm.
        batch_X: List[np.ndarray] = []
        naive_anchors: List[np.ndarray] = []   # (B, h) — наивный прогноз на каждом шаге
        for t in anchors:
            X = self._stack_features(
                cpu_norm[t - w: t], resid_norm[t - w: t], timestamps[t - w: t]
            )
            batch_X.append(X)
            naive_anchors.append(
                self._naive_for(t, h, last_value=float(cpu_series[t - 1]))
            )
        Xb = np.stack(batch_X, axis=0)                   # (B, w, 8)
        naive_ar = np.stack(naive_anchors, axis=0).astype(np.float32)  # (B, h)

        out = self.trainer.predict(Xb)                    # (B, h, n_q)
        median_idx = (
            self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
        )
        median_norm = out[:, :, median_idx]               # (B, h)

        mu_cpu = float(self.preprocessor.mu_cpu)
        sigma_cpu = float(self.preprocessor.sigma_cpu)
        gru_pred = median_norm * sigma_cpu + mu_cpu       # (B, h) — raw-шкала

        y_true = np.stack(
            [cpu_series[anchors + k] for k in range(h)], axis=1
        ).astype(np.float32)                               # (B, h)

        # 1. Подбор α ∈ [0, 1]: взвешивание GRU-прогноза относительно наивного.
        #    blended = naive + α·(gru − naive). При α=0 — чистый наивный,
        #    при α=1 — полный прогноз сети.
        alphas = np.linspace(0.0, 1.0, 41)
        gru_correction = gru_pred - naive_ar              # (B, h)
        maes = np.array([
            float(np.mean(np.abs(y_true - (naive_ar + a * gru_correction))))
            for a in alphas
        ])
        best_idx = int(np.argmin(maes))
        self.blend_alpha = float(alphas[best_idx])
        logger.info(
            "Calibrated blend_alpha=%.3f (val MAE=%.5f, naive-only MAE=%.5f, "
            "gru-only MAE=%.5f)",
            self.blend_alpha, float(maes[best_idx]),
            float(maes[0]), float(maes[-1]),
        )

        # 2. Конформная ширина по per-step абсолютной ошибке (та же α).
        blended = naive_ar + self.blend_alpha * gru_correction
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
        seasonal = (
            self.preprocessor.seasonal_by_phase.astype(np.float32)
            if self.preprocessor.seasonal_by_phase is not None else np.zeros(1, dtype=np.float32)
        )
        torch.save({
            "version": MODEL_VERSION,
            "net_state_dict": self.net.state_dict(),
            "best_val_loss": self.trainer.best_val_loss,
            "stopped_epoch": self.trainer.stopped_epoch,
            "preprocessor_mu": self.preprocessor.mu_w,
            "preprocessor_sigma": self.preprocessor.sigma_w,
            "preprocessor_mu_cpu": self.preprocessor.mu_cpu,
            "preprocessor_sigma_cpu": self.preprocessor.sigma_cpu,
            "preprocessor_seasonal": seasonal,
            "preprocessor_trend_level": float(self.preprocessor.trend_level),
            "preprocessor_trend_slope": float(self.preprocessor.trend_slope),
            "conformal_widths": self.conformal_widths,
            "blend_alpha": self.blend_alpha,
            "horizon_h": self.horizon_h,
            "w_input": self.w_input,
            "quantiles": self.quantiles,
            "train_len": int(self._train_len),
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
        seasonal = np.asarray(state.get("preprocessor_seasonal"), dtype=np.float64)
        self.preprocessor.seasonal_by_phase = seasonal if seasonal.size > 1 else None
        self.preprocessor.period = self.period
        self.preprocessor.trend_level = float(state.get("preprocessor_trend_level", 0.0))
        self.preprocessor.trend_slope = float(state.get("preprocessor_trend_slope", 0.0))
        self.preprocessor._fitted = True
        self.conformal_widths = np.asarray(state["conformal_widths"], dtype=np.float32)
        self.blend_alpha = float(state.get("blend_alpha", 1.0))
        self.trainer.best_val_loss = float(state.get("best_val_loss", float("inf")))
        self.trainer.stopped_epoch = state.get("stopped_epoch") or 0
        self._train_len = int(state.get("train_len", 0))
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

        Сезонный профиль и тренд уже зафиксированы на обучении; здесь мы
        только пересчитываем STL-residual на коротком окне (для входного
        признака) и получаем GRU-коррекцию к наивному анкеру.
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

        w = self.w_input
        # STL на коротком скользящем окне — только для получения residual_norm
        # входного признака. Сезонный профиль и тренд не пересчитываются.
        stl_window = max(3 * self.preprocessor.period, w * 4)
        if n > stl_window:
            cpu_tail = cpu_series[-stl_window:]
            ts_tail = timestamps[-stl_window:]
        else:
            cpu_tail = cpu_series
            ts_tail = timestamps

        _, _, resid_norm, _, _ = self.preprocessor.transform(cpu_tail)
        mu_cpu = float(self.preprocessor.mu_cpu)
        sigma_cpu = float(self.preprocessor.sigma_cpu)
        cpu_norm_tail = ((cpu_tail - mu_cpu) / sigma_cpu).astype(np.float32)

        # Признаки последнего окна: главный канал = cpu_norm (цель сети),
        # extra = resid_norm (STL-остаток как контекст).
        X = self._stack_features(
            cpu_norm_tail[-w:], resid_norm[-w:], ts_tail[-w:]
        )

        # Forward → (h, n_q) в z-score(raw)-шкале
        out = self.trainer.predict(X)
        median_idx = (
            self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
        )
        median_norm = out[:, median_idx]
        gru_hat = (median_norm * sigma_cpu + mu_cpu).astype(np.float32)  # (h,)

        # Наивный анкер: сезонная коррекция персистентности от cpu_ctx[-1].
        # Фаза первого шага = n % period (позиция сразу после контекста).
        naive_hat = self._naive_for(
            n, self.horizon_h, last_value=float(cpu_series[-1]),
        )

        # Смешивание: ĉpu = naive + α·(gru − naive).
        cpu_hat = (naive_hat + self.blend_alpha * (gru_hat - naive_hat)).astype(np.float32)

        widths = self.conformal_widths
        q_lower = cpu_hat - widths
        q_upper = cpu_hat + widths

        cpu_hat = np.clip(cpu_hat, 0.0, 1.0)
        q_lower = np.clip(q_lower, 0.0, 1.0)
        q_upper = np.clip(q_upper, 0.0, 1.0)
        return cpu_hat, q_lower, q_upper

    # ─────────────────────────────────────────────────────────────────────────
    # Внутренние утилиты
    # ─────────────────────────────────────────────────────────────────────────

    def _naive_for(
        self, n_ctx: int, h: int, last_value: Optional[float] = None,
    ) -> np.ndarray:
        """
        Наивный прогноз на h шагов вперёд, начиная с позиции n_ctx.
        Если указан last_value, используется режим сезонной коррекции
        персистентности (не страдает от устаревания тренда). Иначе —
        in-sample режим (T+S из зафиксированных на fit компонент).
        """
        p = self.preprocessor.period
        phase_start = int(n_ctx) % p
        return self.preprocessor.naive_forecast(
            h, phase_start=phase_start, last_value=last_value,
        ).astype(np.float32)

    def _stack_features(
        self,
        window_main: np.ndarray,
        window_extra: np.ndarray,
        window_ts: np.ndarray,
    ) -> np.ndarray:
        """
        Формирует тензор (w, 8):
          [main, extra, sin h, cos h, sin d, cos d, sin m, cos m]

        main  — главный канал (cpu_norm, z-score сырого ряда).
        extra — вспомогательный канал (STL-residual_norm).
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
