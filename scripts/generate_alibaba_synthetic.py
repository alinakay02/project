"""
scripts/generate_alibaba_synthetic.py
Генерирует синтетический датасет, имитирующий характеристики Alibaba Cluster Trace 2018.
Статистические свойства: mean≈0.35, std≈0.18, суточная сезонность.

Запуск: python scripts/generate_alibaba_synthetic.py
"""
import os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "alibaba_cluster_trace_2018.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

N = 2304  # 8 суток * 288 наблюдений/сутки
DT = 300  # 5 минут в секундах

np.random.seed(2018)

# Суточная сезонность (пик днём, минимум ночью)
t = np.arange(N)
seasonal = 0.15 * np.sin(2 * np.pi * t / 288 - np.pi / 3)

# Медленный тренд (рост утилизации к концу недели)
trend = 0.02 * np.sin(2 * np.pi * t / (288 * 7))

# Базовый уровень
base = 0.35

# Шум с тяжёлыми хвостами (характерно для реальных трейсов)
noise = np.random.normal(0, 0.08, N)
noise += np.random.exponential(0.03, N) * np.random.choice([-1, 1], N)

# Всплески (5-10% наблюдений — резкие скачки, как в реальном Alibaba)
spike_mask = np.random.random(N) < 0.07
spikes = spike_mask * np.random.uniform(0.15, 0.35, N) * np.random.choice([1, -0.5], N)

cpu = base + seasonal + trend + noise + spikes
cpu = np.clip(cpu, 0.01, 0.99).astype(np.float32)

# Timestamps
ts = np.arange(N) * DT + 1527379200  # начало: 2018-05-27 00:00:00 UTC

df = pd.DataFrame({"timestamp": ts, "cpu_avg": cpu})
df.to_csv(OUT, index=False)

print(f"Сгенерировано {N} наблюдений -> {OUT}")
print(f"cpu_avg: min={cpu.min():.3f}, max={cpu.max():.3f}, mean={cpu.mean():.3f}, std={cpu.std():.3f}")
