"""
app/data_generators.py — Самодостаточные генераторы и загрузчики наборов данных.

Используется веб-приложением (app/api.py) для:
  • эндпоинта /api/datasets/preview — предпросмотр CSV-датасета
  • эндпоинта /api/training/start  — обучение модели на выбранном наборе

Этот модуль не зависит от tests/ — поэтому работает внутри webapp-контейнера,
где каталог tests/ не копируется (см. Dockerfile.webapp).
"""

import os
from typing import Tuple
import numpy as np


def _make_timestamps(n, start_ts=1_704_067_200, dt_sec=300):
    """Unix-метки с шагом 5 мин, начиная с 2024-01-01."""
    return np.arange(start_ts, start_ts + n * dt_sec, dt_sec, dtype=np.int64)


def _make_phi(n, dominant_class=None, phase_changes=False):
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
        phi[:, :] = 1 / 3
    return phi


def generate_stationary(n=4320, seed=42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Стационарный набор: стабильная суточная нагрузка ~60% базовый, дневной пик ~85%."""
    np.random.seed(seed)
    t = np.arange(n, dtype=float)
    hour_frac = (t % 288) / 288
    daily = 0.25 * np.exp(-((hour_frac - 0.45) ** 2) / (2 * 0.04))
    daily += 0.15 * np.exp(-((hour_frac - 0.65) ** 2) / (2 * 0.03))
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.75 * noise[i - 1] + np.random.randn() * 0.03
    cpu = np.clip(0.35 + daily + noise, 0.05, 0.95)
    return cpu.astype(np.float32), _make_timestamps(n), _make_phi(n)


def generate_trend(n=4320, seed=42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Трендовый набор: рост 15% → 75% с тремя скачками-деплоями."""
    np.random.seed(seed)
    t = np.arange(n, dtype=float)
    midpoint = n * 0.5
    steepness = 8.0 / n
    trend = 0.15 + 0.55 / (1 + np.exp(-steepness * (t - midpoint)))
    jumps = sorted(np.random.choice(range(n // 5, 4 * n // 5), 3, replace=False))
    for j in jumps:
        trend[j:] += np.random.uniform(0.03, 0.08)
    seasonal = 0.06 * np.sin(2 * np.pi * t / 288)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.5 * noise[i - 1] + np.random.randn() * 0.025
    cpu = np.clip(trend + seasonal + noise, 0.05, 0.95)
    return cpu.astype(np.float32), _make_timestamps(n), _make_phi(n)


def generate_spike(n=4320, seed=42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Всплесковый набор: фон ~25%, регулярные всплески до 90%."""
    np.random.seed(seed)
    t = np.arange(n, dtype=float)
    seasonal = 0.08 * np.sin(2 * np.pi * t / 288)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.6 * noise[i - 1] + np.random.randn() * 0.02
    cpu = 0.25 + seasonal + noise

    i = 0
    while i < n:
        gap = np.random.poisson(60)
        i += gap
        if i >= n:
            break
        spike_type = np.random.choice(['sharp', 'plateau', 'ramp'])
        amp = np.random.uniform(0.3, 0.65)
        if spike_type == 'sharp':
            dur = int(np.random.choice([1, 2]))
            for d in range(min(dur, n - i)):
                cpu[i + d] += amp
        elif spike_type == 'plateau':
            dur = int(np.random.randint(5, 16))
            for d in range(min(dur, n - i)):
                cpu[i + d] += amp * 0.8
        else:
            dur = int(np.random.randint(10, 30))
            for d in range(min(dur, n - i)):
                phase = d / dur
                cpu[i + d] += amp * np.sin(np.pi * phase)

    cpu = np.clip(cpu, 0.05, 0.99)
    return cpu.astype(np.float32), _make_timestamps(n), _make_phi(n)


def generate_mixed(n=4320, seed=42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Смешанный реалистичный набор: тренд + сутки + неделя + выходные + редкие всплески."""
    np.random.seed(seed)
    t = np.arange(n, dtype=float)
    trend = 0.30 + 0.002 * (t / 288)
    hour_frac = (t % 288) / 288
    daily = 0.20 * np.exp(-((hour_frac - 0.42) ** 2) / (2 * 0.035))
    daily += 0.12 * np.exp(-((hour_frac - 0.62) ** 2) / (2 * 0.025))
    day_of_week = ((t // 288) % 7).astype(float)
    weekend = np.where((day_of_week >= 5), -0.12, 0.0)

    sigma_vary = 0.035 * (1 + 0.6 * np.maximum(daily, 0))
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i - 1] + np.random.randn() * sigma_vary[i]

    for _ in range(n // 200):
        idx = int(np.random.randint(n))
        dur = int(np.random.randint(3, 10))
        for d in range(min(dur, n - idx)):
            trend[idx + d] += np.random.uniform(0.10, 0.25) * np.sin(np.pi * d / dur)

    cpu = np.clip(trend + daily + weekend + noise, 0.05, 0.99)
    return cpu.astype(np.float32), _make_timestamps(n), _make_phi(n)


def _load_trace_csv(path: str, max_n: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Универсальный загрузчик CSV-датасетов."""
    import pandas as pd
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at '{path}'.")
    df = pd.read_csv(path, parse_dates=False)
    df = df.sort_values("timestamp").dropna(subset=["cpu_avg"])
    n = min(len(df), max_n) if max_n else len(df)
    cpu = np.clip(df["cpu_avg"].values[:n].astype(np.float32), 0.01, 0.99)
    ts = df["timestamp"].values[:n].astype(np.int64)
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


def load_alibaba_trace(path: str = "data/alibaba_cluster_trace_2018.csv"):
    return _load_trace_csv(path)


def load_google_trace(path: str = "data/google_cluster_trace_2019.csv"):
    return _load_trace_csv(path)


def load_azure_trace(path: str = "data/azure_vm_trace_2019.csv"):
    return _load_trace_csv(path)


# Удобный диспетчер
LOADERS = {
    "alibaba":    load_alibaba_trace,
    "google":     load_google_trace,
    "azure":      load_azure_trace,
    "stationary": generate_stationary,
    "trend":      generate_trend,
    "spike":      generate_spike,
    "mixed":      generate_mixed,
}
