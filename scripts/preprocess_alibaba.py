"""
scripts/preprocess_alibaba.py
Агрегирует Alibaba Cluster Trace 2018 до 5-минутных интервалов.
Вход:  data/container_usage.csv
Выход: data/alibaba_cluster_trace_2018.csv  (2304 строки)
"""
import os, sys
import pandas as pd
import numpy as np

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "container_usage.csv")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "alibaba_cluster_trace_2018.csv")

if not os.path.exists(RAW):
    print(f"ERROR: {RAW} not found.")
    print("Download from https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018")
    sys.exit(1)

print(f"Reading {RAW}...")
df = pd.read_csv(RAW, header=None,
                 names=["timestamp", "container_id", "machine_id",
                        "cpu_avg", "cpu_max", "mem_avg", "mem_max"],
                 dtype={"timestamp": np.int64, "cpu_avg": np.float32})

print(f"Raw rows: {len(df)}")

# Берём контейнер с наибольшим числом наблюдений (наиболее полный ряд)
top_container = df["container_id"].value_counts().index[0]
print(f"Using container: {top_container}")
df = df[df["container_id"] == top_container].copy()
df = df.sort_values("timestamp").reset_index(drop=True)

# Агрегация до 5-минутных интервалов (Δt = 300 сек)
df["bucket"] = (df["timestamp"] // 300) * 300
agg = df.groupby("bucket").agg(
    cpu_avg=("cpu_avg", "mean"),
    cpu_max=("cpu_max", "max"),
).reset_index()
agg.columns = ["timestamp", "cpu_avg", "cpu_max"]
agg = agg.sort_values("timestamp").reset_index(drop=True)

# Нормализуем: cpu_avg ∈ [0, 1]
if agg["cpu_avg"].max() > 1.5:
    # Если значения > 1.5, это абсолютное использование (нужно нормировать)
    agg["cpu_avg"] = agg["cpu_avg"] / 100.0
agg["cpu_avg"] = agg["cpu_avg"].clip(0.01, 0.99)

# Оставляем 2304 наблюдения = 8 суток
agg = agg.head(2304)

agg[["timestamp", "cpu_avg"]].to_csv(OUT, index=False)
print(f"Saved {len(agg)} rows to {OUT}")
print(f"cpu_avg: min={agg['cpu_avg'].min():.3f}, max={agg['cpu_avg'].max():.3f}, "
      f"mean={agg['cpu_avg'].mean():.3f}")
