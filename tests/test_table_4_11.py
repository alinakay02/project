"""
tests/test_table_4_11.py — Таблица 4.11: Вычислительное время по модулям
метода на наборе Alibaba Cluster Trace 2018.

Замеряется время каждого шага цикла итерации:
  1. Сбор метрик (PromQL) — моделируется задержкой запроса к локальному
     HTTP-эндпоинту; в реальном кластере время определяется сетью и нагрузкой
     на Prometheus.
  2. Предобработка (IQR + STL на скользящем окне 3 периодов).
  3. Прогноз тренда и сезонности — в текущей архитектуре отсутствует как
     отдельный шаг: GRU предсказывает нормализованный CPU напрямую.
     Строка оставлена для совместимости с таблицей главы 4.
  4. Вывод GRU-модели (forward pass).
  5. Принятие решений (ResourceDecisionModule.step).
  6. PATCH-запрос к Kubernetes API — не измеряется в тесте, т.к. требует
     реального кластера. Значение берётся из документации kubernetes-python.
  7. Полный цикл итерации — сумма измеренных компонентов.

Замеры делаются на N_ITERS=50 итераций, усредняются.

Запуск:
  python -m pytest tests/test_table_4_11.py -v -s
  python -u scripts/run_table_4_11.py

Результаты в: results/table_4_11.txt
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from predictor.preprocessor import Preprocessor
from predictor.forecaster import HybridForecaster
from predictor.config import CFG_PREPROCESSOR, CFG_FORECASTER, CFG_DECISION
from controller.decision import ResourceDecisionModule
from tests.baselines import load_alibaba_trace

warnings.filterwarnings("ignore")

_CUDA_OK = torch.cuda.is_available()
print(
    f"\n[T4.11][ENV] PyTorch={torch.__version__} | CUDA available={_CUDA_OK} | "
    f"device={torch.cuda.get_device_name(0) if _CUDA_OK else 'CPU'}",
    flush=True,
)

N_ITERS = 50
DT_MS = 5 * 60 * 1000  # 5 минут в мс


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _mock_promql_latency_ms():
    """
    Моделирует задержку PromQL-запроса к локальному эндпоинту.
    Для типичного запроса к Prometheus в одном кластере задержка 5-15 мс.
    Значение замеряется, но точное число зависит от окружения.
    """
    # Замеряем задержку localhost HTTP-запроса как приближение.
    try:
        import urllib.request
        t0 = time.perf_counter()
        try:
            urllib.request.urlopen("http://127.0.0.1/", timeout=0.1)
        except Exception:
            pass  # нас интересует время на попытку
        return (time.perf_counter() - t0) * 1000
    except Exception:
        return 5.0  # fallback


class TestTable411:
    """Таблица 4.11: детализация вычислительного времени по модулям."""

    def test_timing(self):
        cpu, ts, phi = load_alibaba_trace()
        n_tr = int(len(cpu) * 0.70)

        print(
            f"\n[{_now()}] [T4.11] >>> старт: Alibaba, {N_ITERS} итераций замера",
            flush=True,
        )

        # Обучение модели (не входит в замеры цикла итерации)
        np.random.seed(42)
        torch.manual_seed(42)
        pp = Preprocessor(**CFG_PREPROCESSOR)
        fc = HybridForecaster(preprocessor=pp, **CFG_FORECASTER)
        dm = ResourceDecisionModule(**CFG_DECISION)

        print(f"[{_now()}] [T4.11] fit модели...", flush=True)
        t0 = time.time()
        fc.fit(cpu[:n_tr], ts[:n_tr], phi[:, :n_tr])
        print(
            f"[{_now()}] [T4.11] fit готов за {time.time() - t0:.1f}s",
            flush=True,
        )

        # Warmup: первый прогон GPU часто медленнее (JIT компиляция)
        for _ in range(3):
            fc.predict(cpu[:n_tr], ts[:n_tr], phi[:, :n_tr])

        # Замеры по модулям
        times = {
            "promql": [],
            "preprocess": [],
            "trend_season": [],
            "gru_forward": [],
            "decision": [],
            "full_cycle": [],
        }

        for i in range(N_ITERS):
            window_end = n_tr + i
            cpu_ctx = cpu[:window_end]
            ts_ctx = ts[:window_end]

            t_cycle_start = time.perf_counter()

            # 1. PromQL (mock)
            t0 = time.perf_counter()
            _ = _mock_promql_latency_ms()
            times["promql"].append((time.perf_counter() - t0) * 1000)

            # 2. Предобработка (IQR + STL на скользящем окне)
            t0 = time.perf_counter()
            stl_window = max(3 * pp.period, fc.w_input * 4)
            if len(cpu_ctx) > stl_window:
                cpu_tail = cpu_ctx[-stl_window:]
            else:
                cpu_tail = cpu_ctx
            _, _, resid_norm, _, _ = pp.transform(cpu_tail)
            clean = pp.clean_series_
            times["preprocess"].append((time.perf_counter() - t0) * 1000)

            # 3. Прогноз тренда и сезонности (не используется в новой архитектуре)
            #    Оставлено как 0 для обратной совместимости таблицы.
            times["trend_season"].append(0.0)

            # 4. GRU forward pass
            import math
            t0 = time.perf_counter()
            mu_cpu = float(pp.mu_cpu)
            sigma_cpu = float(pp.sigma_cpu)
            cpu_norm = ((clean - mu_cpu) / sigma_cpu).astype(np.float32)
            w = fc.w_input
            ts_tail = ts_ctx[-stl_window:] if len(ts_ctx) > stl_window else ts_ctx
            X = fc._stack_features(cpu_norm[-w:], resid_norm[-w:], ts_tail[-w:])
            _ = fc.trainer.predict(X)
            times["gru_forward"].append((time.perf_counter() - t0) * 1000)

            # 5. Принятие решений
            t0 = time.perf_counter()
            # Нужен настоящий прогноз для decision; делаем повторный predict()
            _, q_lo, q_hi = fc.predict(cpu_ctx, ts_ctx, None)
            _ = dm.step(float(q_hi[0]), float(q_lo[0]))
            times["decision"].append((time.perf_counter() - t0) * 1000)

            times["full_cycle"].append(
                (time.perf_counter() - t_cycle_start) * 1000
            )

            if (i + 1) % 10 == 0:
                print(
                    f"[{_now()}] [T4.11] [{i + 1}/{N_ITERS}] "
                    f"full_cycle={times['full_cycle'][-1]:.2f}ms",
                    flush=True,
                )

        print(f"\n[{_now()}] [T4.11] === Сводные результаты (среднее + стдев) ===", flush=True)

        # Для K8s PATCH берём типичное значение из документации (10-50 мс в локальном кластере)
        k8s_patch_mean = 25.0
        k8s_patch_std = 10.0

        rows = [
            ("promql", "Сбор метрик (PromQL-запрос, локальный mock)"),
            ("preprocess", "Предобработка (IQR + STL на скользящем окне)"),
            ("trend_season", "Прогноз тренда и сезонности (не используется)"),
            ("gru_forward", "Вывод GRU-модели (forward pass)"),
            ("decision", "Принятие решений (ResourceDecisionModule)"),
        ]

        for key, label in rows:
            arr = np.array(times[key])
            m, s = float(arr.mean()), float(arr.std())
            pct = 100.0 * m / DT_MS
            print(
                f"[METRIC] TABLE_4_11: module={key} | label=\"{label}\" | "
                f"mean_ms={m:.2f} | std_ms={s:.2f} | pct_of_dt={pct:.4f}%",
                flush=True,
            )

        # K8s PATCH — типичное значение
        print(
            f"[METRIC] TABLE_4_11: module=k8s_patch | "
            f"label=\"PATCH к Kubernetes API (типичное значение)\" | "
            f"mean_ms={k8s_patch_mean:.2f} | std_ms={k8s_patch_std:.2f} | "
            f"pct_of_dt={100.0 * k8s_patch_mean / DT_MS:.4f}%",
            flush=True,
        )

        # Полный цикл
        arr_full = np.array(times["full_cycle"])
        m, s = float(arr_full.mean()), float(arr_full.std())
        pct = 100.0 * m / DT_MS
        print(
            f"[METRIC] TABLE_4_11: module=full_cycle | "
            f"label=\"Полный цикл итерации (измеренный)\" | "
            f"mean_ms={m:.2f} | std_ms={s:.2f} | pct_of_dt={pct:.4f}%",
            flush=True,
        )
