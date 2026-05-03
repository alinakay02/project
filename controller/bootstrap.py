"""
controller/bootstrap.py — точка входа контроллера с pre-train на трейсе Alibaba.

Загружает исторический датасет Alibaba Cluster Trace 2018, выполняет
начальное обучение HybridForecaster, прогревает скользящее окно
наблюдений, после чего запускает основной управляющий цикл. Это
позволяет модели работать с первой же итерации — без ожидания 48 часов
накопления реальных данных из Prometheus.

Запуск:  python -m controller.bootstrap
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd

from controller.control_loop import ControlLoop


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("controller.bootstrap")


DEFAULT_DATASET = "/app/data/alibaba_cluster_trace_2018.csv"


def load_alibaba(path: str, n_points: int) -> dict:
    """
    Читает CSV (timestamp, cpu_avg, ..., phi1, phi2, phi3) и возвращает
    последние n_points наблюдений с временными метками, сдвинутыми так,
    что последняя точка соответствует now - dt.
    """
    df = pd.read_csv(path)
    required = {"cpu_avg", "phi1", "phi2", "phi3"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}. Got: {list(df.columns)}")

    df = df.tail(n_points).reset_index(drop=True)

    dt = 300  # 5 минут
    now = int(time.time())
    end_ts = (now // dt) * dt - dt
    n = len(df)
    new_ts = np.array(
        [end_ts - (n - 1 - i) * dt for i in range(n)],
        dtype=np.int64,
    )

    cpu = df["cpu_avg"].to_numpy(dtype=np.float64)
    phi = np.column_stack(
        [df["phi1"].to_numpy(), df["phi2"].to_numpy(), df["phi3"].to_numpy()]
    ).T  # shape (3, n)

    return {"cpu": cpu, "timestamps": new_ts, "phi": phi}


def main() -> None:
    dataset_path = os.environ.get("ALIBABA_DATASET", DEFAULT_DATASET)
    if not os.path.exists(dataset_path):
        logger.error(
            f"Dataset not found at {dataset_path}. "
            f"Falling back to ControlLoop().run() without pre-training."
        )
        ControlLoop().run()
        return

    logger.info(f"Loading dataset: {dataset_path}")
    cl = ControlLoop()

    n_initial = int(cl.cfg["model"]["retrain_window"])  # 2016 точек = 7 суток
    data = load_alibaba(dataset_path, n_initial)
    logger.info(
        f"Loaded {len(data['cpu'])} points "
        f"(cpu range: {data['cpu'].min():.3f} - {data['cpu'].max():.3f})"
    )

    cl._initial_fit(data)
    logger.info("Initial fit done. Pre-filling buffers...")

    # Прогреваем буферы хвостом датасета. Минимум — 2*period для STL,
    # сверху берём запас, чтобы forecaster.predict() сразу имел полное окно.
    n_warmup = max(int(cl.cfg["model"]["w_input"]) * 2, 2 * cl.preprocessor.period)
    n_warmup = min(n_warmup, len(data["cpu"]))
    start = len(data["cpu"]) - n_warmup
    for i in range(start, len(data["cpu"])):
        cl.cpu_buffer.append(float(data["cpu"][i]))
        cl.ts_buffer.append(int(data["timestamps"][i]))
        cl.phi_buffer.append(np.asarray(data["phi"][:, i], dtype=np.float64))
    logger.info(f"Pre-filled buffers with {n_warmup} points")

    # Запускаем основной цикл без initial_data — обучение и прогрев уже сделаны.
    cl.run()


if __name__ == "__main__":
    main()
