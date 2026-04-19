"""
tests/test_table_4_10.py — Таблица 4.10: Анализ вклада компонентов метода
на наборе Alibaba Cluster Trace 2018 при горизонте h = 3.

Alibaba выбран потому, что его утилизация CPU находится в рабочем
диапазоне 0.4-0.9 — это единственный реальный датасет, на котором
симуляция цикла управления создаёт активное масштабирование. На Google
CPU слишком низкая (0.15-0.25), из-за чего r_req всегда ограничивается
r_min=2 и решения о масштабировании не принимаются.

ВАЖНО: прогноз forecaster'а q_upper выдаётся в долях [0,1]. Формула
decision module r_req = ceil(q_upper / cpu_target) рассчитана на q_upper
как «предсказанную нагрузку в единицах реплик» (т.е. уже умноженную
на текущее число реплик). В симуляции ниже мы явно масштабируем прогноз:
  load_in_replicas = q_upper * r_cur
и подаём это в decision.step(). Так правильно учитывается, что чем
больше текущих реплик — тем больше суммарная ёмкость.

Пять конфигураций:
  1. Полный метод                — HybridForecaster + стандартный DecisionModule
  2. Без STL-декомпозиции         — AutoGRU (без preprocessor) + стандартный DM
  3. Без квантильной оценки       — HybridForecaster, но прогноз трактуется как
                                    точечный с фиксированным буфером 20%
                                    (q_up = median * 1.2, q_lo = median * 0.8)
  4. Без механизма гистерезиса    — HybridForecaster + DM с tau=1
                                    (масштабирование вниз без подтверждения)
  5. С фиксированным δ=1          — HybridForecaster + DM, в котором
                                    adaptive δ_t заменён на константу 1.0

Для каждой конфигурации симулируется цикл управления на тестовой выборке:
  - на каждом шаге вызывается predict()
  - q_upper, q_lower подаются в DecisionModule (масштабируются на r_cur)
  - ведётся история r_fin
По истории вычисляются: MAE, SLA violations, avg utilization, scale ops.

ОЖИДАЕМОЕ поведение (часть сути абляции):
  - no_stl: MAE ухудшается → подтверждает вклад STL-декомпозиции в прогноз
  - no_quantile: MAE одинаков с full (медиана та же), но SLA растут
    → квантильный прогноз важен для УПРАВЛЕНИЯ, а не точности
  - no_hysteresis: MAE тот же, SLA почти не меняется, но OPS резко растёт
    → гистерезис снижает flap масштабирования без ущерба SLA
  - fixed_delta: MAE тот же, SLA растут, UTIL падает
    → адаптивный порог лучше фиксированного

То есть одинаковый MAE у последних трёх конфигураций — это не баг, а
дизайн эксперимента: они изменяют только модуль решений, не модель.

Запуск:
  python -m pytest tests/test_table_4_10.py -v -s
  python -u scripts/run_table_4_10.py

Результаты в: results/table_4_10.txt
"""

from __future__ import annotations

import gc
import os
import sys
import time
import traceback
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from predictor.preprocessor import Preprocessor
from predictor.forecaster import HybridForecaster
from predictor.config import CFG_PREPROCESSOR, CFG_FORECASTER, CFG_DECISION
from controller.decision import ResourceDecisionModule, DecisionResult
from tests.baselines import AutoGRUForecaster, load_alibaba_trace
from tests.metrics import (
    compute_mae, compute_sla_violations, compute_avg_utilization,
    compute_scale_ops,
)

warnings.filterwarnings("ignore")

_CUDA_OK = torch.cuda.is_available()
print(
    f"\n[T4.10][ENV] PyTorch={torch.__version__} | CUDA available={_CUDA_OK} | "
    f"device={torch.cuda.get_device_name(0) if _CUDA_OK else 'CPU'}",
    flush=True,
)

SEEDS = [42, 137, 256]


def _now() -> str:
    return time.strftime("%H:%M:%S")


# ── Модифицированный DecisionModule с фиксированным δ=1 ──────────────────

class FixedDeltaDecisionModule(ResourceDecisionModule):
    """Вариант модуля решений с жёстко зафиксированным delta_t = 1.0."""

    def step(self, q_upper, q_lower):
        import math
        r_req = math.ceil(q_upper / self.cpu_target)
        r_req = max(r_req, self.r_min)
        saturation = False
        r_bounded = min(r_req, self.r_max)
        if r_bounded < r_req:
            saturation = True
        delta_t = 1.0  # ФИКСИРОВАННОЕ
        action = "no_change"
        if r_bounded > self._r_cur:
            self._r_cur = r_bounded
            self._confirm_counter = 0
            action = "scale_up"
        elif r_bounded < self._r_cur - delta_t:
            self._confirm_counter += 1
            if self._confirm_counter >= self.tau:
                self._r_cur = max(self.r_min, r_bounded)
                self._confirm_counter = 0
                action = "scale_down"
        else:
            self._confirm_counter = 0
        r_fin = max(self.r_min, self._r_cur)
        if saturation:
            action = "saturated"
        result = DecisionResult(
            r_req=r_req, r_bounded=r_bounded, r_fin=r_fin, action=action,
            delta_t=delta_t, saturation=saturation,
            confirm_counter=self._confirm_counter,
        )
        self.history.append(result)
        return result


# ── Симуляция цикла управления ───────────────────────────────────────────

def _simulate_control_loop(forecaster, decision_module, cpu_train, cpu_test,
                            ts_train, ts_test, phi_train, phi_test,
                            point_forecast_buffer=None):
    """
    Симулирует цикл управления на тестовой выборке.

    Прогноз forecaster'а даётся в долях cpu∈[0,1]. Для decision module
    прогноз масштабируется на текущее число реплик:
      load_in_replicas = q_upper * r_cur
    т.к. формула r_req = ceil(q_upper_scaled / cpu_target) рассчитывает
    требуемое число реплик исходя из «предсказанной суммарной нагрузки».

    Возвращает:
      y_true, y_pred, r_history

    Если point_forecast_buffer задан (например, 0.2 = 20%), то q_upper
    и q_lower вычисляются как median*(1+buffer) и median*(1-buffer)
    вместо реальных квантилей GRU.
    """
    h = forecaster.horizon_h
    y_true, y_pred = [], []
    r_history = []

    for i in range(0, len(cpu_test) - h, h):
        cpu_ctx = np.concatenate([cpu_train, cpu_test[:i]])
        ts_ctx = np.concatenate([ts_train, ts_test[:i]])
        phi_ctx = (
            np.concatenate([phi_train, phi_test[:, :i]], axis=1)
            if phi_test.shape[1] > i else phi_train
        )
        hat, lo, hi = forecaster.predict(cpu_ctx, ts_ctx, phi_ctx)

        for k in range(min(h, len(cpu_test) - i)):
            y_true.append(cpu_test[i + k])
            y_pred.append(hat[k] if k < len(hat) else hat[-1])

        # Принятие решения по прогнозу первого шага
        med = float(hat[0])
        if point_forecast_buffer is not None:
            q_up_raw = min(1.0, med * (1.0 + point_forecast_buffer))
            q_lo_raw = max(0.0, med * (1.0 - point_forecast_buffer))
        else:
            q_up_raw = float(hi[0])
            q_lo_raw = float(lo[0])

        # Масштабирование на текущее число реплик
        r_cur = decision_module.current_replicas
        q_up_scaled = q_up_raw * r_cur
        q_lo_scaled = q_lo_raw * r_cur

        res = decision_module.step(q_up_scaled, q_lo_scaled)
        # r_fin распространяется на h шагов (до следующего вызова step)
        for _ in range(min(h, len(cpu_test) - i)):
            r_history.append(res.r_fin)

    return np.array(y_true), np.array(y_pred), np.array(r_history)


def _eval_config(config_name: str, seed: int, cpu, ts, phi):
    """Оценивает одну конфигурацию: обучает модель + прогон с симуляцией."""
    n_tr = int(len(cpu) * 0.70)
    n_vl = int(len(cpu) * 0.15)
    cpu_train, cpu_test = cpu[:n_tr], cpu[n_tr + n_vl:]
    ts_train, ts_test = ts[:n_tr], ts[n_tr + n_vl:]
    phi_train, phi_test = phi[:, :n_tr], phi[:, n_tr + n_vl:]

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Настройка прогнозатора и модуля решений в зависимости от конфигурации
    point_buffer = None
    if config_name == "full":
        pp = Preprocessor(**CFG_PREPROCESSOR)
        fc = HybridForecaster(preprocessor=pp, **CFG_FORECASTER)
        dm = ResourceDecisionModule(**CFG_DECISION)
    elif config_name == "no_stl":
        fc = AutoGRUForecaster(seed=seed, horizon_h=3, max_epochs=100)
        dm = ResourceDecisionModule(**CFG_DECISION)
    elif config_name == "no_quantile":
        pp = Preprocessor(**CFG_PREPROCESSOR)
        fc = HybridForecaster(preprocessor=pp, **CFG_FORECASTER)
        dm = ResourceDecisionModule(**CFG_DECISION)
        point_buffer = 0.20  # 20% буфер
    elif config_name == "no_hysteresis":
        pp = Preprocessor(**CFG_PREPROCESSOR)
        fc = HybridForecaster(preprocessor=pp, **CFG_FORECASTER)
        cfg_dm = dict(CFG_DECISION)
        cfg_dm["tau"] = 1
        dm = ResourceDecisionModule(**cfg_dm)
    elif config_name == "fixed_delta":
        pp = Preprocessor(**CFG_PREPROCESSOR)
        fc = HybridForecaster(preprocessor=pp, **CFG_FORECASTER)
        dm = FixedDeltaDecisionModule(**CFG_DECISION)
    else:
        raise ValueError(f"Unknown config: {config_name}")

    fc.fit(cpu_train, ts_train, phi_train)
    y_true, y_pred, r_hist = _simulate_control_loop(
        fc, dm, cpu_train, cpu_test, ts_train, ts_test,
        phi_train, phi_test, point_forecast_buffer=point_buffer,
    )

    # Выравниваем длины
    min_len = min(len(y_true), len(r_hist))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    r_hist = r_hist[:min_len]

    mae = compute_mae(y_true, y_pred) if min_len else float("nan")
    sla = compute_sla_violations(y_true, r_hist, CFG_DECISION["cpu_target"])
    util = compute_avg_utilization(y_true, r_hist, CFG_DECISION["cpu_target"])
    ops = compute_scale_ops(r_hist)
    return mae, sla, util, ops


CONFIGS = [
    ("full", "Полный метод"),
    ("no_stl", "Без STL-декомпозиции"),
    ("no_quantile", "Без квантильной оценки"),
    ("no_hysteresis", "Без механизма гистерезиса"),
    ("fixed_delta", "С фиксированным \u03b4=1"),
]


class TestTable410:
    """Таблица 4.10: абляция на Alibaba."""

    def test_ablation(self):
        cpu, ts, phi = load_alibaba_trace()
        print(
            f"\n[{_now()}] [T4.10] >>> старт: Alibaba n={len(cpu)}, "
            f"{len(CONFIGS)} конфигураций x {len(SEEDS)} seeds",
            flush=True,
        )
        t_total = time.time()

        all_results = {}  # {config_name: [(mae, sla, util, ops), ...]}

        for cfg_key, cfg_label in CONFIGS:
            all_results[cfg_key] = []
            print(
                f"\n[{_now()}] [T4.10] === Конфигурация: {cfg_label} ({cfg_key}) ===",
                flush=True,
            )
            for step, seed in enumerate(SEEDS, 1):
                gc.collect()
                print(
                    f"[{_now()}] [T4.10][{cfg_key}] [{step}/{len(SEEDS)}] "
                    f"START seed={seed}",
                    flush=True,
                )
                t0 = time.time()
                try:
                    mae, sla, util, ops = _eval_config(cfg_key, seed, cpu, ts, phi)
                    all_results[cfg_key].append((mae, sla, util, ops))
                    print(
                        f"[{_now()}] [T4.10][{cfg_key}] [{step}/{len(SEEDS)}] "
                        f"DONE  seed={seed} MAE={mae:.4f} SLA={sla:.2f}% "
                        f"UTIL={util:.1f}% OPS={ops} ({time.time() - t0:.1f}s)",
                        flush=True,
                    )
                except Exception as e:
                    tb = traceback.format_exc(limit=4)
                    print(
                        f"[{_now()}] [T4.10][{cfg_key}] [ERROR] seed={seed}: "
                        f"{type(e).__name__}: {e}\n{tb}",
                        flush=True,
                    )

        print(f"\n[{_now()}] [T4.10] === Сводные результаты ===", flush=True)
        for cfg_key, cfg_label in CONFIGS:
            runs = all_results[cfg_key]
            if not runs:
                print(
                    f"[METRIC] TABLE_4_10: dataset=Alibaba | config={cfg_key} | FAILED",
                    flush=True,
                )
                continue
            arr = np.array(runs)
            mae_m, mae_s = arr[:, 0].mean(), arr[:, 0].std()
            sla_m, sla_s = arr[:, 1].mean(), arr[:, 1].std()
            util_m, util_s = arr[:, 2].mean(), arr[:, 2].std()
            ops_m, ops_s = arr[:, 3].mean(), arr[:, 3].std()
            print(
                f"[METRIC] TABLE_4_10: dataset=Alibaba | config={cfg_key} | "
                f"label=\"{cfg_label}\" | "
                f"MAE={mae_m:.4f}\u00b1{mae_s:.4f} | "
                f"SLA={sla_m:.2f}\u00b1{sla_s:.2f}% | "
                f"UTIL={util_m:.1f}\u00b1{util_s:.1f}% | "
                f"OPS={ops_m:.0f}\u00b1{ops_s:.0f}",
                flush=True,
            )

        print(
            f"\n[{_now()}] [T4.10] <<< завершено за {time.time() - t_total:.0f}с",
            flush=True,
        )
