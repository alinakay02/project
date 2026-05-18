"""
tests/baselines.py — Базовые методы сравнения.
tests/data_generators.py — Генераторы синтетических данных.
tests/metrics.py — Метрики оценки.
"""

# ════════════════════════════════════════════════════════════════════════════
# baselines.py
# ════════════════════════════════════════════════════════════════════════════

import math
import logging
import numpy as np
from typing import Tuple, List, Optional
from scipy.stats import linregress

logger = logging.getLogger(__name__)


class BaseForecaster:
    """Общий интерфейс для всех методов прогнозирования."""
    def fit(self, cpu, ts, phi): pass
    def predict(self, cpu, ts, phi) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


class SARIMAForecaster(BaseForecaster):
    """SARIMA через statsmodels.SARIMAX."""

    _PERIOD = 24  # сезонный период (288 → 24 для скорости)

    def __init__(self, horizon_h=3):
        self.horizon_h = horizon_h
        self._model_fit = None

    def fit(self, cpu, ts, phi):
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            model = SARIMAX(cpu, order=(1,1,1),
                            seasonal_order=(0,1,1, self._PERIOD),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            self._model_fit = model.fit(disp=False, maxiter=50)
        except Exception as e:
            logger.warning(f"SARIMA fit failed: {e}")
            self._model_fit = None

    def predict(self, cpu, ts, phi):
        if self._model_fit is None:
            return np.full(self.horizon_h, np.mean(cpu[-60:])), \
                   np.zeros(self.horizon_h), np.ones(self.horizon_h)
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            model = SARIMAX(cpu, order=(1,1,1),
                            seasonal_order=(0,1,1, self._PERIOD),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            fit = model.filter(self._model_fit.params)
            fc = fit.forecast(self.horizon_h)
            fc = np.clip(fc, 0, 1)
            std_est = float(np.std(cpu[-288:]) * 1.5)
            return fc, fc - 1.96*std_est, fc + 1.96*std_est
        except Exception:
            fallback = np.full(self.horizon_h, float(np.mean(cpu[-10:])))
            return fallback, fallback*0.8, fallback*1.2


class ARIMAForecaster(BaseForecaster):
    """ARIMA(p,d,q) без сезонной компоненты."""

    def __init__(self, horizon_h=3, **kwargs):
        self.horizon_h = horizon_h
        self._model_fit = None

    def fit(self, cpu, ts, phi):
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(cpu, order=(2, 1, 2))
            self._model_fit = model.fit(method_kwargs={"maxiter": 200})
        except Exception as e:
            logger.warning(f"ARIMA fit failed: {e}")
            self._model_fit = None

    def predict(self, cpu, ts, phi):
        if self._model_fit is None:
            fb = np.full(self.horizon_h, float(np.mean(cpu[-60:])))
            return fb, fb * 0.8, fb * 1.2
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(cpu, order=(2, 1, 2))
            fit = model.filter(self._model_fit.params)
            fc = fit.forecast(self.horizon_h)
            fc = np.clip(fc, 0, 1)
            std_est = float(np.std(cpu[-288:])) * 1.5
            return fc, np.clip(fc - 1.96 * std_est, 0, 1), np.clip(fc + 1.96 * std_est, 0, 1)
        except Exception:
            fb = np.full(self.horizon_h, float(np.mean(cpu[-10:])))
            return fb, fb * 0.8, fb * 1.2


class HoltWintersForecaster(BaseForecaster):
    """Экспоненциальное сглаживание Хольта-Винтерса с сезонностью."""

    def __init__(self, horizon_h=3, period=24, **kwargs):
        self.horizon_h = horizon_h
        self.period = period
        self._model_fit = None
        self._alpha = None
        self._beta = None
        self._gamma = None

    def fit(self, cpu, ts, phi):
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            model = ExponentialSmoothing(
                cpu, trend="add", seasonal="add",
                seasonal_periods=self.period,
                initialization_method="estimated",
            )
            self._model_fit = model.fit(optimized=True, use_brute=False)
            self._alpha = self._model_fit.params['smoothing_level']
            self._beta = self._model_fit.params['smoothing_trend']
            self._gamma = self._model_fit.params['smoothing_seasonal']
        except Exception as e:
            logger.warning(f"Holt-Winters fit failed: {e}")
            self._model_fit = None

    def predict(self, cpu, ts, phi):
        if self._model_fit is None:
            fb = np.full(self.horizon_h, float(np.mean(cpu[-60:])))
            return fb, fb * 0.8, fb * 1.2
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            model = ExponentialSmoothing(
                cpu, trend="add", seasonal="add",
                seasonal_periods=self.period,
                initialization_method="estimated",
            )
            fit = model.fit(
                smoothing_level=self._alpha,
                smoothing_trend=self._beta,
                smoothing_seasonal=self._gamma,
                optimized=False,
                use_brute=False,
            )
            fc = fit.forecast(self.horizon_h)
            fc = np.clip(fc, 0, 1)
            std_est = float(np.std(cpu[-288:])) * 1.5
            return fc, np.clip(fc - 1.96 * std_est, 0, 1), np.clip(fc + 1.96 * std_est, 0, 1)
        except Exception:
            fb = np.full(self.horizon_h, float(np.mean(cpu[-10:])))
            return fb, fb * 0.8, fb * 1.2


class RandomForestForecaster(BaseForecaster):
    """Случайный лес: вход — окно из w последних значений + календарные признаки."""

    def __init__(self, horizon_h=3, w_input=60, n_estimators=100, **kwargs):
        self.horizon_h = horizon_h
        self.w = w_input
        self.n_estimators = n_estimators
        self._model = None

    def _build_features(self, cpu, ts, start, end):
        """Строит матрицу признаков для обучения/предсказания."""
        X, y = [], []
        for i in range(start, end):
            window = cpu[i - self.w:i]
            t = ts[i]
            hour = (t % 86400) / 3600.0
            dow = (t // 86400) % 7
            feats = np.concatenate([window, [
                np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            ]])
            X.append(feats)
            y.append(cpu[i])
        return np.array(X), np.array(y)

    def fit(self, cpu, ts, phi):
        try:
            from sklearn.ensemble import RandomForestRegressor
            X, y = self._build_features(cpu, ts, self.w, len(cpu))
            self._model = RandomForestRegressor(
                n_estimators=self.n_estimators, max_depth=12, random_state=42, n_jobs=-1)
            self._model.fit(X, y)
            self._cpu_train = cpu
            self._ts_train = ts
        except Exception as e:
            logger.warning(f"RandomForest fit failed: {e}")

    def predict(self, cpu, ts, phi):
        if self._model is None:
            fb = np.full(self.horizon_h, float(np.mean(cpu[-60:])))
            return fb, fb * 0.8, fb * 1.2
        preds = []
        current = cpu.copy()
        for k in range(self.horizon_h):
            window = current[-self.w:]
            t = ts[-1] + (k + 1) * 300
            hour = (t % 86400) / 3600.0
            dow = (t // 86400) % 7
            feats = np.concatenate([window, [
                np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            ]]).reshape(1, -1)
            p = float(self._model.predict(feats)[0])
            preds.append(np.clip(p, 0, 1))
            current = np.append(current, p)
        preds = np.array(preds)
        std_est = float(np.std(cpu[-288:])) * 1.5
        return preds, np.clip(preds - 1.96 * std_est, 0, 1), np.clip(preds + 1.96 * std_est, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Общая инфраструктура нейросетевых базовых методов
# ─────────────────────────────────────────────────────────────────────────────

def _build_features_window(cpu_norm, ts, w):
    """(w, 7) — нормализованный сигнал + 6 тригонометрических признаков времени."""
    r_window = cpu_norm[-w:]
    ts_window = ts[-w:]
    if len(r_window) < w:
        pad_n = w - len(r_window)
        r_window = np.concatenate([np.zeros(pad_n, dtype=np.float32), r_window])
        ts_window = np.concatenate([np.full(pad_n, ts_window[0]), ts_window])
    hours = (ts_window % 86400) / 3600.0
    dows = ((ts_window // 86400) % 7).astype(float)
    minutes = (ts_window % 3600) / 60.0
    X = np.stack([
        r_window.astype(np.float32),
        np.sin(2 * math.pi * hours / 24).astype(np.float32),
        np.cos(2 * math.pi * hours / 24).astype(np.float32),
        np.sin(2 * math.pi * dows / 7).astype(np.float32),
        np.cos(2 * math.pi * dows / 7).astype(np.float32),
        np.sin(2 * math.pi * minutes / 60).astype(np.float32),
        np.cos(2 * math.pi * minutes / 60).astype(np.float32),
    ], axis=1)
    return X


class _NeuralBaseline(BaseForecaster):
    """
    Общий базовый класс для нейросетевых аналогов прогнозатора.
    Каждый наследник реализует _build_net() — модель, выдающую (B, h, n_q).
    Подкласс получает прямой многошаговый квантильный прогноз через тот же
    GRUTrainer, что и предложенный метод, что обеспечивает честное сравнение.
    """

    def __init__(self, seed=42, horizon_h=3, w_input=60,
                 quantiles=(0.025, 0.5, 0.975), hidden_dim=64, dropout=0.25,
                 lr=1e-3, lr_decay=0.99, grad_clip=1.0,
                 max_epochs=40, patience=6, batch_size=64, **kwargs):
        self.seed = int(seed)
        self.horizon_h = int(horizon_h)
        self.w_input = int(w_input)
        self.quantiles = list(quantiles)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.lr_decay = float(lr_decay)
        self.grad_clip = float(grad_clip)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self._trainer = None
        self._mu = 0.0
        self._sigma = 1.0

    def _build_net(self):
        raise NotImplementedError

    def fit(self, cpu, ts, phi):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import torch
        from predictor.model import GRUTrainer, TimeSeriesDataset

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        cpu = np.asarray(cpu, dtype=np.float32)
        ts = np.asarray(ts, dtype=np.int64)

        self._mu = float(np.mean(cpu))
        self._sigma = max(float(np.std(cpu)), 1e-8)
        cpu_norm = (cpu - self._mu) / self._sigma

        n_val = max(int(len(cpu_norm) * 0.15), self.w_input + self.horizon_h + 4)
        n_val = min(n_val, len(cpu_norm) - self.w_input - self.horizon_h - 4)
        n_tr = len(cpu_norm) - n_val
        if n_tr <= self.w_input + self.horizon_h:
            return  # слишком короткий ряд — оставим self._trainer = None

        train_ds = TimeSeriesDataset(
            target=cpu_norm[:n_tr], timestamps=ts[:n_tr],
            w_input=self.w_input, horizon_h=self.horizon_h,
        )
        val_start = n_tr - self.w_input
        val_ds = TimeSeriesDataset(
            target=cpu_norm[val_start:], timestamps=ts[val_start:],
            w_input=self.w_input, horizon_h=self.horizon_h,
        )

        net = self._build_net()
        self._trainer = GRUTrainer(
            net, quantiles=self.quantiles, lr=self.lr, lr_decay=self.lr_decay,
            grad_clip=self.grad_clip, max_epochs=self.max_epochs,
            patience=self.patience, batch_size=self.batch_size,
        )
        self._trainer.fit(train_ds, val_ds)

    def predict(self, cpu, ts, phi):
        if self._trainer is None:
            fb = np.full(self.horizon_h, float(np.mean(cpu[-10:])))
            return fb, fb * 0.8, fb * 1.2
        cpu = np.asarray(cpu, dtype=np.float32)
        ts = np.asarray(ts, dtype=np.int64)
        cpu_norm = (cpu - self._mu) / self._sigma
        X = _build_features_window(cpu_norm, ts, self.w_input)            # (w, 7)
        out = self._trainer.predict(X)                                     # (h, n_q)
        # Денормализация
        preds = out * self._sigma + self._mu                                # (h, n_q)
        median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
        cpu_hat = np.clip(preds[:, median_idx], 0.0, 1.0)
        q_lower = np.clip(preds[:, 0], 0.0, 1.0)
        q_upper = np.clip(preds[:, -1], 0.0, 1.0)
        return cpu_hat, q_lower, q_upper


class AutoGRUForecaster(_NeuralBaseline):
    """Автономная GRU без декомпозиции (аблация: тот же GRU, но без EWM-тренда)."""

    def _build_net(self):
        from predictor.model import GRUQuantileNet
        return GRUQuantileNet(
            d_step=7, hidden_dim=self.hidden_dim, n_layers=2,
            dropout=self.dropout, n_quantiles=len(self.quantiles),
            horizon_h=self.horizon_h,
        )


class LSTMForecaster(_NeuralBaseline):
    """LSTM с теми же гиперпараметрами, прямой многошаговый квантильный выход."""

    def _build_net(self):
        import torch.nn as nn
        h, q, w_dropout = self.horizon_h, len(self.quantiles), self.dropout
        hidden = self.hidden_dim

        class LSTMQuantileNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(7, hidden, num_layers=2, batch_first=True,
                                    dropout=w_dropout)
                self.head = nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(w_dropout),
                    nn.Linear(64, q * h),
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                flat = self.head(out[:, -1, :])
                return flat.view(-1, h, q)

        return LSTMQuantileNet()


class CNNLSTMForecaster(_NeuralBaseline):
    """CNN-LSTM гибрид: Conv1D для локальных паттернов + LSTM, прямой многошаговый выход."""

    def _build_net(self):
        import torch.nn as nn
        h, q, w_dropout = self.horizon_h, len(self.quantiles), self.dropout
        hidden = self.hidden_dim

        class CNNLSTMNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(7, 32, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
                )
                self.lstm = nn.LSTM(32, hidden, num_layers=2, batch_first=True,
                                    dropout=w_dropout)
                self.head = nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(w_dropout),
                    nn.Linear(64, q * h),
                )

            def forward(self, x):
                c = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, seq, 32)
                out, _ = self.lstm(c)
                flat = self.head(out[:, -1, :])
                return flat.view(-1, h, q)

        return CNNLSTMNet()


class TFTForecaster(_NeuralBaseline):
    """Упрощённый Temporal Fusion Transformer, прямой многошаговый квантильный выход."""

    def __init__(self, seed=42, horizon_h=3, w_input=60,
                 d_model=32, n_heads=4, dropout=0.25,
                 max_epochs=40, patience=6, batch_size=64, **kwargs):
        super().__init__(
            seed=seed, horizon_h=horizon_h, w_input=w_input,
            hidden_dim=d_model, dropout=dropout,
            max_epochs=max_epochs, patience=patience, batch_size=batch_size,
            **kwargs,
        )
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)

    def _build_net(self):
        import torch.nn as nn
        d_model, n_heads = self.d_model, self.n_heads
        h, q, w_dropout = self.horizon_h, len(self.quantiles), self.dropout

        class TFTNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Linear(7, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                    dropout=w_dropout, batch_first=True, activation="gelu",
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
                self.head = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(w_dropout),
                    nn.Linear(64, q * h),
                )

            def forward(self, x):
                z = self.input_proj(x)
                z = self.transformer(z)
                flat = self.head(z[:, -1, :])
                return flat.view(-1, h, q)

        return TFTNet()


class ReactiveHPA(BaseForecaster):
    """
    Реактивный HPA: r = ceil(r_cur * cpu_actual / cpu_target).
    Использует текущее значение cpu_t, не прогноз.
    """
    def __init__(self, cpu_target=0.70, r_min=2, r_max_cluster=8, **kwargs):
        self.cpu_target = cpu_target
        self.r_min = r_min
        self.r_max = r_max_cluster
        self._r_cur = r_min

    def fit(self, cpu, ts, phi): pass

    def predict(self, cpu, ts, phi):
        """Возвращает 'прогноз' = текущее значение cpu (реактивный подход)."""
        cpu_now = float(cpu[-1])
        return np.array([cpu_now]*3), np.array([cpu_now*0.9]*3), np.array([cpu_now*1.1]*3)

    def step(self, q_upper, q_lower):
        """Совместимость с run_management_simulation."""
        from controller.decision import DecisionResult
        cpu_est = q_upper * self.cpu_target  # «разворачиваем» квантиль
        r_new = math.ceil(self._r_cur * cpu_est / self.cpu_target)
        r_new = max(self.r_min, min(self.r_max, r_new))
        action = "scale_up" if r_new > self._r_cur else (
            "scale_down" if r_new < self._r_cur else "no_change")
        self._r_cur = r_new
        return DecisionResult(r_req=r_new, r_bounded=r_new, r_fin=r_new,
                               action=action, delta_t=0.0, saturation=False, confirm_counter=0)

    def reset(self): self._r_cur = self.r_min

    @property
    def cpu_target(self): return self._cpu_target
    @cpu_target.setter
    def cpu_target(self, v): self._cpu_target = v
    @property
    def r_min(self): return self._r_min
    @r_min.setter
    def r_min(self, v): self._r_min = v
    @property
    def r_max(self): return self._r_max
    @r_max.setter
    def r_max(self, v): self._r_max = v


# ════════════════════════════════════════════════════════════════════════════
# data_generators.py
# ════════════════════════════════════════════════════════════════════════════

def _make_timestamps(n, start_ts=1_704_067_200, dt_sec=300):
    """Генерирует массив unix-меток времени c шагом dt_sec=5 мин."""
    return np.arange(start_ts, start_ts + n * dt_sec, dt_sec, dtype=np.int64)


def _make_phi(n, dominant_class=None, phase_changes=False):
    """
    Генерирует вектор состава классов φ_t (3, n).
    phase_changes=True — смена доминирующего класса каждые n//4 шагов.
    """
    phi = np.zeros((3, n), dtype=np.float32)
    if phase_changes:
        segment = n // 4
        for i in range(n):
            seg_idx = min(i // segment, 2)
            phi[seg_idx, i] = 0.70
            for j in range(3):
                if j != seg_idx:
                    phi[j, i] = 0.15
    elif dominant_class is not None:
        phi[dominant_class - 1, :] = 0.70
        for j in range(3):
            if j != dominant_class - 1:
                phi[j, :] = 0.15
    else:
        phi[:, :] = 1/3
    return phi


def generate_stationary(n=4320, seed=42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Стационарный набор: стабильный сервис с предсказуемой суточной нагрузкой.
    Высокий базовый уровень (~60%), чёткая дневная/ночная разница.
    """
    np.random.seed(seed)
    t = np.arange(n, dtype=float)
    # «Рабочий день» — утро-вечер высокая нагрузка, ночью низкая
    hour_frac = (t % 288) / 288  # 0..1 в течение суток
    daily = 0.25 * np.exp(-((hour_frac - 0.45) ** 2) / (2 * 0.04))  # пик в ~11:00
    daily += 0.15 * np.exp(-((hour_frac - 0.65) ** 2) / (2 * 0.03))  # второй пик ~16:00
    # AR(1) шум
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.75 * noise[i-1] + np.random.randn() * 0.03
    cpu = np.clip(0.35 + daily + noise, 0.05, 0.95)
    ts  = _make_timestamps(n)
    phi = _make_phi(n)
    return cpu.astype(np.float32), ts, phi


def generate_trend(n=4320, seed=42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Трендовый набор: нагрузка растёт по мере увеличения аудитории.
    Начинается с 15%, заканчивается ~75%. Есть 2-3 резких скачка (деплои).
    Слабая сезонность на фоне сильного тренда.
    """
    np.random.seed(seed)
    t = np.arange(n, dtype=float)
    # Логистический рост (S-кривая) — имитация роста аудитории
    midpoint = n * 0.5
    steepness = 8.0 / n
    trend = 0.15 + 0.55 / (1 + np.exp(-steepness * (t - midpoint)))
    # 3 резких скачка (деплои, маркетинговые кампании)
    jumps = sorted(np.random.choice(range(n // 5, 4 * n // 5), 3, replace=False))
    for j in jumps:
        trend[j:] += np.random.uniform(0.03, 0.08)
    # Слабая сезонность
    seasonal = 0.06 * np.sin(2 * np.pi * t / 288)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.5 * noise[i-1] + np.random.randn() * 0.025
    cpu = np.clip(trend + seasonal + noise, 0.05, 0.95)
    ts  = _make_timestamps(n)
    phi = _make_phi(n)
    return cpu.astype(np.float32), ts, phi


def generate_spike(n=4320, seed=42, amplitude=3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Всплесковый набор: спокойный фон с агрессивными аномалиями.
    Низкий базовый уровень (~25%), но регулярные всплески до 70-95%.
    Имитирует DDoS, вирусные посты, batch-обработку.
    """
    np.random.seed(seed)
    t = np.arange(n, dtype=float)
    # Низкий спокойный фон
    seasonal = 0.08 * np.sin(2 * np.pi * t / 288)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.6 * noise[i-1] + np.random.randn() * 0.02
    cpu = 0.25 + seasonal + noise

    # Мощные всплески разного характера
    i = 0
    while i < n:
        gap = np.random.poisson(60)
        i += gap
        if i >= n:
            break
        spike_type = np.random.choice(['sharp', 'plateau', 'ramp'])
        amp = np.random.uniform(0.3, 0.65)
        if spike_type == 'sharp':
            # Мгновенный пик 1-2 точки
            dur = np.random.choice([1, 2])
            for d in range(min(dur, n - i)):
                cpu[i + d] += amp
        elif spike_type == 'plateau':
            # Плато на 5-15 точек (batch job)
            dur = np.random.randint(5, 16)
            for d in range(min(dur, n - i)):
                cpu[i + d] += amp * 0.8
        else:
            # Нарастание и спад (вирусный контент)
            dur = np.random.randint(10, 30)
            for d in range(min(dur, n - i)):
                phase = d / dur
                cpu[i + d] += amp * np.sin(np.pi * phase)

    cpu = np.clip(cpu, 0.05, 0.99)
    ts  = _make_timestamps(n)
    phi = _make_phi(n)
    return cpu.astype(np.float32), ts, phi


def generate_mixed(n=4320, seed=42, phase_changes=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Смешанный набор: реалистичный production-трафик.
    Тренд + суточная + недельная сезонность + всплески + смена режимов.
    Средний уровень ~45%, амплитуда суток ~25%, есть «выходные».
    """
    np.random.seed(seed)
    t = np.arange(n, dtype=float)
    # Слабый тренд
    trend = 0.30 + 0.002 * (t / 288)
    # Суточная: пик днём, провал ночью (асимметричная форма)
    hour_frac = (t % 288) / 288
    daily = 0.20 * np.exp(-((hour_frac - 0.42) ** 2) / (2 * 0.035))
    daily += 0.12 * np.exp(-((hour_frac - 0.62) ** 2) / (2 * 0.025))
    # Недельная: выходные на 30% ниже
    day_of_week = ((t // 288) % 7).astype(float)
    weekend = np.where((day_of_week >= 5), -0.12, 0.0)
    # Нестационарная дисперсия
    sigma_vary = 0.035 * (1 + 0.6 * np.maximum(daily, 0))
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i-1] + np.random.randn() * sigma_vary[i]

    if phase_changes:
        segment = n // 4
        level_shift = np.zeros(n)
        level_shift[segment:2*segment]   = +0.25
        level_shift[2*segment:3*segment] = -0.10
        level_shift[3*segment:]          = +0.15
        trend = trend + level_shift

    # Редкие всплески
    for _ in range(n // 200):
        idx = np.random.randint(n)
        dur = np.random.randint(3, 10)
        for d in range(min(dur, n - idx)):
            trend[idx + d] += np.random.uniform(0.10, 0.25) * np.sin(np.pi * d / dur)

    cpu = np.clip(trend + daily + weekend + noise, 0.05, 0.99)
    ts  = _make_timestamps(n)
    phi = _make_phi(n, phase_changes=phase_changes)
    return cpu.astype(np.float32), ts, phi


def _load_trace_csv(path: str, max_n: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Универсальный загрузчик реальных датасетов.
    Формат CSV: timestamp,cpu_avg,rps,mem_avg,lat_p95,err_rate,phi1,phi2,phi3
    """
    import pandas as pd, os
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at '{path}'.")
    df = pd.read_csv(path, parse_dates=False)
    df = df.sort_values("timestamp").dropna(subset=["cpu_avg"])
    n = min(len(df), max_n) if max_n else len(df)
    cpu = np.clip(df["cpu_avg"].values[:n].astype(np.float32), 0.01, 0.99)
    ts  = df["timestamp"].values[:n].astype(np.int64)
    if ts.max() < 1e9:
        ts = _make_timestamps(n)
    if "phi1" in df.columns:
        phi = np.stack([
            df["phi1"].values[:n],
            df["phi2"].values[:n],
            df["phi3"].values[:n],
        ]).astype(np.float32)
    else:
        phi = _make_phi(n)
    return cpu, ts, phi


def load_alibaba_trace(
    path: str = "data/alibaba_cluster_trace_2018.csv"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alibaba Cluster Trace 2018 — 2243 точки, 5 реальных метрик, dt=300s."""
    return _load_trace_csv(path)


def load_google_trace(
    path: str = "data/google_cluster_trace_2019.csv"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Google Cluster Trace 2019 — 8064 точки (~28 дней), dt=300s."""
    return _load_trace_csv(path)


def load_azure_trace(
    path: str = "data/azure_vm_trace_2019.csv"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Azure VM Trace 2019 — 8640 точек (~30 дней), dt=300s."""
    return _load_trace_csv(path)


# ════════════════════════════════════════════════════════════════════════════
# metrics.py
# ════════════════════════════════════════════════════════════════════════════

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, eps=1e-8) -> float:
    return float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)) * 100)


def compute_coverage(
    y_true: np.ndarray, q_lower: np.ndarray, q_upper: np.ndarray
) -> float:
    """Покрытие 95%-ДИ."""
    covered = np.sum((y_true >= q_lower) & (y_true <= q_upper))
    return 100.0 * covered / max(len(y_true), 1)


def compute_sla_violations(
    cpu_actual: np.ndarray, r_history: np.ndarray, cpu_target: float = 0.70
) -> float:
    """Доля нарушений SLA."""
    capacity = cpu_target * r_history
    violations = np.sum(cpu_actual > capacity)
    return 100.0 * violations / max(len(cpu_actual), 1)


def compute_avg_utilization(
    cpu_actual: np.ndarray, r_history: np.ndarray, cpu_target: float = 0.70
) -> float:
    """Средняя утилизация."""
    capacity = cpu_target * r_history + 1e-9
    return 100.0 * float(np.mean(cpu_actual / capacity))


def compute_scale_ops(r_history: np.ndarray) -> int:
    return int(np.sum(np.diff(r_history) != 0))


def print_forecast_metrics(
    test_name: str,
    maes: list, rmses: list, mapes: list, coverages: list
):
    print(
        f"\n[METRIC] ТЕСТ: {test_name} | "
        f"MAE={np.mean(maes):.4f}±{np.std(maes):.4f} | "
        f"RMSE={np.mean(rmses):.4f}±{np.std(rmses):.4f} | "
        f"MAPE={np.mean(mapes):.2f}±{np.std(mapes):.2f}% | "
        f"COVERAGE={np.mean(coverages):.1f}±{np.std(coverages):.1f}%"
    )


def print_mgmt_metrics(test_name: str, sla: list, util: list, ops: list):
    print(
        f"\n[METRIC] MGMT ТЕСТ: {test_name} | "
        f"SLA_VIOLATIONS={np.mean(sla):.2f}%±{np.std(sla):.2f}% | "
        f"AVG_UTIL={np.mean(util):.1f}%±{np.std(util):.1f}% | "
        f"SCALE_OPS={np.mean(ops):.0f}±{np.std(ops):.0f}"
    )
