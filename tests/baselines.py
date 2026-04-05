"""
tests/baselines.py — Базовые методы сравнения (параграф 4.1)
tests/data_generators.py — Генераторы синтетических данных (параграф 4.2)
tests/metrics.py — Метрики оценки (параграф 4.2)
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
    """SARIMA через statsmodels.SARIMAX с авто-подбором порядков по AIC."""

    def __init__(self, horizon_h=3):
        self.horizon_h = horizon_h
        self._model_fit = None

    def fit(self, cpu, ts, phi):
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            # Подбираем простой SARIMA(1,1,1)(0,1,1,288)
            model = SARIMAX(cpu, order=(1,1,1), seasonal_order=(0,1,1,288),
                            enforce_stationarity=False, enforce_invertibility=False)
            self._model_fit = model.fit(disp=False, maxiter=100)
        except Exception as e:
            logger.warning(f"SARIMA fit failed: {e}")
            self._model_fit = None

    def predict(self, cpu, ts, phi):
        if self._model_fit is None:
            return np.full(self.horizon_h, np.mean(cpu[-60:])), \
                   np.zeros(self.horizon_h), np.ones(self.horizon_h)
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            model = SARIMAX(cpu, order=(1,1,1), seasonal_order=(0,1,1,288),
                            enforce_stationarity=False, enforce_invertibility=False)
            fit = model.filter(self._model_fit.params)
            fc = fit.forecast(self.horizon_h)
            fc = np.clip(fc, 0, 1)
            std_est = float(np.std(cpu[-288:]) * 1.5)
            return fc, fc - 1.96*std_est, fc + 1.96*std_est
        except Exception:
            fallback = np.full(self.horizon_h, float(np.mean(cpu[-10:])))
            return fallback, fallback*0.8, fallback*1.2


class ProphetForecaster(BaseForecaster):
    """Prophet с параметрами по умолчанию для суточных рядов."""

    def __init__(self, horizon_h=3, dt_minutes=5):
        self.horizon_h = horizon_h
        self.dt_minutes = dt_minutes
        self._model = None

    def fit(self, cpu, ts, phi):
        try:
            from prophet import Prophet
            import pandas as pd
            df = pd.DataFrame({"ds": pd.to_datetime(ts, unit="s"), "y": cpu})
            self._model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05,
            )
            self._model.fit(df)
            self._ts_last = int(ts[-1])
        except Exception as e:
            logger.warning(f"Prophet fit failed: {e}")
            self._model = None

    def predict(self, cpu, ts, phi):
        if self._model is None:
            fallback = np.full(self.horizon_h, float(np.mean(cpu[-10:])))
            return fallback, fallback*0.8, fallback*1.2
        try:
            import pandas as pd
            last_ts = int(ts[-1])
            future_ts = [last_ts + (k+1)*self.dt_minutes*60 for k in range(self.horizon_h)]
            df_future = pd.DataFrame({"ds": pd.to_datetime(future_ts, unit="s")})
            fc_df = self._model.predict(df_future)
            y = fc_df["yhat"].values
            lo = fc_df["yhat_lower"].values
            hi = fc_df["yhat_upper"].values
            return np.clip(y, 0, 1), np.clip(lo, 0, 1), np.clip(hi, 0, 1)
        except Exception:
            fallback = np.full(self.horizon_h, float(np.mean(cpu[-10:])))
            return fallback, fallback*0.8, fallback*1.2


class AutoGRUForecaster(BaseForecaster):
    """
    Автономная GRU без STL-декомпозиции (аблация и базовый метод).
    Та же архитектура, что и в HybridForecaster, но прогнозирует cpu_t напрямую.
    """

    def __init__(self, seed=42, horizon_h=3, w_input=60, quantiles=(0.025, 0.5, 0.975),
                 hidden_dim=64, dropout=0.10, lr=0.001, lr_decay=0.99,
                 grad_clip=1.0, max_epochs=100, patience=10, batch_size=32, **kwargs):
        self.horizon_h = horizon_h
        self.w_input   = w_input
        self.quantiles = list(quantiles)
        self.seed      = seed
        self.hidden_dim = hidden_dim
        self.dropout    = dropout
        self.lr         = lr
        self.lr_decay   = lr_decay
        self.grad_clip  = grad_clip
        self.max_epochs = max_epochs
        self.patience   = patience
        self.batch_size = batch_size
        self._trainer = None
        self._mu = 0.0
        self._sigma = 1.0

    def fit(self, cpu, ts, phi):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from predictor.model import GRUQuantileNet, GRUTrainer, TimeSeriesDataset

        np.random.seed(self.seed)
        # Нормализуем входной ряд
        self._mu = float(np.mean(cpu[-60:]))
        self._sigma = max(float(np.std(cpu[-60:])), 1e-8)
        cpu_norm = (cpu - self._mu) / self._sigma

        n_val = max(int(len(cpu_norm) * 0.15), self.w_input + 1)
        n_tr  = len(cpu_norm) - n_val

        # Создаём фиктивный phi (нули)
        phi_zeros = np.zeros((3, len(cpu_norm)), dtype=np.float32)

        train_ds = TimeSeriesDataset(cpu_norm[:n_tr], ts[:n_tr],
                                      phi_zeros[:, :n_tr], self.w_input)
        val_ds   = TimeSeriesDataset(cpu_norm[n_tr-self.w_input:],
                                      ts[n_tr-self.w_input:],
                                      phi_zeros[:, n_tr-self.w_input:], self.w_input)

        net = GRUQuantileNet(d_in=69, hidden_dim=self.hidden_dim,
                              dropout=self.dropout, n_quantiles=3)
        self._trainer = GRUTrainer(net, quantiles=self.quantiles, lr=self.lr,
                                    lr_decay=self.lr_decay, grad_clip=self.grad_clip,
                                    max_epochs=self.max_epochs, patience=self.patience,
                                    batch_size=self.batch_size)
        self._trainer.fit(train_ds, val_ds)

    def predict(self, cpu, ts, phi):
        if self._trainer is None:
            fallback = np.full(self.horizon_h, float(np.mean(cpu[-10:])))
            return fallback, fallback*0.8, fallback*1.2

        from predictor.forecaster import HybridForecaster
        import math

        cpu_norm = (cpu - self._mu) / self._sigma
        # Строим входной вектор так же, как HybridForecaster
        r_window = cpu_norm[-self.w_input:]
        if len(r_window) < self.w_input:
            r_window = np.pad(r_window, (self.w_input - len(r_window), 0))

        ts_last = ts[-1]
        hour   = (ts_last % 86400) / 3600.0
        dow    = (ts_last // 86400) % 7
        minute = (ts_last % 3600) / 60.0
        time_feats = np.array([
            math.sin(2*math.pi*hour/24), math.cos(2*math.pi*hour/24),
            math.sin(2*math.pi*dow/7),   math.cos(2*math.pi*dow/7),
            math.sin(2*math.pi*minute/60),math.cos(2*math.pi*minute/60),
        ])
        phi_last = phi[:, -1] if phi.shape[1] > 0 else np.array([1/3, 1/3, 1/3])
        X = np.concatenate([r_window, time_feats, phi_last]).reshape(1, -1)

        out = self._trainer.predict(X)
        # Обратное преобразование
        preds = out * self._sigma + self._mu
        preds = np.clip(preds, 0, 1)
        return (np.full(self.horizon_h, preds[1]),
                np.full(self.horizon_h, preds[0]),
                np.full(self.horizon_h, preds[2]))


class LSTMForecaster(AutoGRUForecaster):
    """
    LSTM с теми же гиперпараметрами, что и GRU (параграф 4.1).
    Переопределяет только тип рекуррентного блока.
    """
    def fit(self, cpu, ts, phi):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import torch.nn as nn
        from predictor.model import GRUQuantileNet, GRUTrainer, TimeSeriesDataset

        np.random.seed(self.seed)
        self._mu = float(np.mean(cpu[-60:]))
        self._sigma = max(float(np.std(cpu[-60:])), 1e-8)
        cpu_norm = (cpu - self._mu) / self._sigma

        n_val = max(int(len(cpu_norm) * 0.15), self.w_input + 1)
        n_tr  = len(cpu_norm) - n_val
        phi_zeros = np.zeros((3, len(cpu_norm)), dtype=np.float32)

        train_ds = TimeSeriesDataset(cpu_norm[:n_tr], ts[:n_tr],
                                      phi_zeros[:, :n_tr], self.w_input)
        val_ds   = TimeSeriesDataset(cpu_norm[n_tr-self.w_input:],
                                      ts[n_tr-self.w_input:],
                                      phi_zeros[:, n_tr-self.w_input:], self.w_input)

        # Используем LSTM вместо GRU (другой класс)
        class LSTMQuantileNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm1 = nn.LSTM(69, 64, num_layers=1, batch_first=True)
                self.drop  = nn.Dropout(0.10)
                self.lstm2 = nn.LSTM(64, 64, num_layers=1, batch_first=True)
                self.fc    = nn.Linear(64, 3)
            def forward(self, x):
                o1,_ = self.lstm1(x); o1=self.drop(o1)
                o2,_ = self.lstm2(o1)
                return self.fc(o2[:,-1,:])

        net = LSTMQuantileNet()
        self._trainer = GRUTrainer(net, quantiles=self.quantiles, lr=self.lr,
                                    lr_decay=self.lr_decay, grad_clip=self.grad_clip,
                                    max_epochs=self.max_epochs, patience=self.patience,
                                    batch_size=self.batch_size)
        self._trainer.fit(train_ds, val_ds)


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
    Стационарный набор: суточная сезонность p=288, без тренда, умеренный шум.
    Параметры: параграф 4.2.
    """
    np.random.seed(seed)
    t = np.arange(n)
    seasonal = 0.20 * np.sin(2 * np.pi * t / 288)
    base     = 0.35
    noise    = 0.02 * np.random.randn(n)
    cpu      = np.clip(base + seasonal + noise, 0.05, 0.95)
    ts       = _make_timestamps(n)
    phi      = _make_phi(n)
    return cpu.astype(np.float32), ts, phi


def generate_trend(n=4320, seed=42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Трендовый набор: линейный рост ~0.002/наблюдение + суточная сезонность.
    """
    np.random.seed(seed)
    t        = np.arange(n, dtype=float)
    trend    = 0.20 + 0.002 * (t / 288)    # ~0.002 * сутки
    seasonal = 0.15 * np.sin(2 * np.pi * t / 288)
    noise    = 0.02 * np.random.randn(n)
    cpu      = np.clip(trend + seasonal + noise, 0.05, 0.95)
    ts       = _make_timestamps(n)
    phi      = _make_phi(n)
    return cpu.astype(np.float32), ts, phi


def generate_spike(n=4320, seed=42, amplitude=3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Всплесковый набор: стационарный фон + пики амплитудой amplitude*σ.
    Межсобытийные интервалы: пуассоновские со средним 50 наблюдений.
    """
    np.random.seed(seed)
    t        = np.arange(n)
    seasonal = 0.15 * np.sin(2 * np.pi * t / 288)
    base     = 0.35
    noise    = 0.02 * np.random.randn(n)
    cpu      = base + seasonal + noise

    # Добавляем всплески
    sigma_bg = float(np.std(noise))
    spike_idx = []
    i = 0
    while i < n:
        gap = np.random.poisson(50)
        i += gap
        if i < n:
            spike_idx.append(i)
    for idx in spike_idx:
        cpu[idx] += amplitude * sigma_bg

    cpu = np.clip(cpu, 0.05, 0.99)
    ts  = _make_timestamps(n)
    phi = _make_phi(n)
    return cpu.astype(np.float32), ts, phi


def generate_mixed(n=4320, seed=42, phase_changes=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Смешанный набор: слабый тренд + сезонность + всплески 2-4σ + нестационарная дисперсия.
    Опционально — смена доминирующего класса (для теста φ_t).
    """
    np.random.seed(seed)
    t        = np.arange(n, dtype=float)
    trend    = 0.25 + 0.001 * (t / 288)
    seasonal = 0.18 * np.sin(2 * np.pi * t / 288) + 0.05 * np.sin(4 * np.pi * t / 288)
    sigma_base = 0.02
    sigma_vary = sigma_base * (1 + 0.5 * np.abs(np.sin(2 * np.pi * t / (288 * 7))))
    noise    = sigma_vary * np.random.randn(n)

    # Если смена классов активна — добавляем скачок уровня
    if phase_changes:
        segment = n // 4
        level_shift = np.zeros(n)
        level_shift[segment:2*segment]   = +0.35   # класс 1 доминирует → высокое CPU
        level_shift[2*segment:3*segment] = -0.15   # класс 2 → низкое CPU
        level_shift[3*segment:]          = +0.15   # класс 3 → среднее CPU
        trend = trend + level_shift

    # Всплески 2-4σ
    for _ in range(n // 100):
        idx = np.random.randint(n)
        trend[idx] += np.random.uniform(2, 4) * sigma_base

    cpu = np.clip(trend + seasonal + noise, 0.05, 0.99)
    ts  = _make_timestamps(n)
    phi = _make_phi(n, phase_changes=phase_changes)
    return cpu.astype(np.float32), ts, phi


def load_alibaba_trace(
    path: str = "data/alibaba_cluster_trace_2018.csv"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Загружает Alibaba Cluster Trace 2018 (параграф 4.2).
    Ожидаемый формат CSV: timestamp,cpu_avg
    Источник: https://github.com/alibaba/clusterdata
    """
    import pandas as pd, os
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Alibaba dataset not found at '{path}'. "
            "Run scripts/download_data.sh to download and preprocess."
        )
    df = pd.read_csv(path, parse_dates=False)
    df = df.sort_values("timestamp").dropna()
    cpu = df["cpu_avg"].values[:2304].astype(np.float32)
    cpu = np.clip(cpu, 0.01, 0.99)
    ts  = df["timestamp"].values[:2304].astype(np.int64)
    # Если timestamps не unix — создаём синтетические
    if ts.max() < 1e9:
        ts = _make_timestamps(len(cpu))
    phi = _make_phi(len(cpu))
    return cpu, ts, phi


# ════════════════════════════════════════════════════════════════════════════
# metrics.py
# ════════════════════════════════════════════════════════════════════════════

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE (формула 4.1)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE (формула 4.2)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, eps=1e-8) -> float:
    """MAPE (формула 4.3)."""
    return float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)) * 100)


def compute_coverage(
    y_true: np.ndarray, q_lower: np.ndarray, q_upper: np.ndarray
) -> float:
    """Покрытие ДИ 95% (параграф 4.2)."""
    covered = np.sum((y_true >= q_lower) & (y_true <= q_upper))
    return 100.0 * covered / max(len(y_true), 1)


def compute_sla_violations(
    cpu_actual: np.ndarray, r_history: np.ndarray, cpu_target: float = 0.70
) -> float:
    """Доля нарушений SLA (параграф 4.2)."""
    capacity = cpu_target * r_history
    violations = np.sum(cpu_actual > capacity)
    return 100.0 * violations / max(len(cpu_actual), 1)


def compute_avg_utilization(
    cpu_actual: np.ndarray, r_history: np.ndarray, cpu_target: float = 0.70
) -> float:
    """Средняя утилизация (параграф 4.2)."""
    capacity = cpu_target * r_history + 1e-9
    return 100.0 * float(np.mean(cpu_actual / capacity))


def compute_scale_ops(r_history: np.ndarray) -> int:
    """Число операций масштабирования."""
    return int(np.sum(np.diff(r_history) != 0))


def print_forecast_metrics(
    test_name: str,
    maes: list, rmses: list, mapes: list, coverages: list
):
    """Выводит строку метрик для копирования в таблицу."""
    print(
        f"\n[METRIC] ТЕСТ: {test_name} | "
        f"MAE={np.mean(maes):.4f}±{np.std(maes):.4f} | "
        f"RMSE={np.mean(rmses):.4f}±{np.std(rmses):.4f} | "
        f"MAPE={np.mean(mapes):.2f}±{np.std(mapes):.2f}% | "
        f"COVERAGE={np.mean(coverages):.1f}±{np.std(coverages):.1f}%"
    )


def print_mgmt_metrics(test_name: str, sla: list, util: list, ops: list):
    """Выводит строку метрик управления для таблицы."""
    print(
        f"\n[METRIC] MGMT ТЕСТ: {test_name} | "
        f"SLA_VIOLATIONS={np.mean(sla):.2f}%±{np.std(sla):.2f}% | "
        f"AVG_UTIL={np.mean(util):.1f}%±{np.std(util):.1f}% | "
        f"SCALE_OPS={np.mean(ops):.0f}±{np.std(ops):.0f}"
    )
