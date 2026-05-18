"""
controller/control_loop.py — Основной управляющий цикл.

Итерация каждые Δt=5 мин:
  1. Сбор m_t, φ_t из Prometheus
  2. Предобработка + декомпозиция STL
  3. Гибридное прогнозирование (тренд + сезонность + GRU)
  4. Принятие решений о масштабировании
  5. PATCH к Kubernetes API при необходимости
  6. Дообучение GRU каждые 60 итераций (retrain_every)
"""

import math
import time
import logging
import numpy as np
from collections import deque
from typing import Optional, List, Dict, Any

import yaml

from predictor.preprocessor import Preprocessor
from predictor.forecaster import HybridForecaster
from controller.decision import ResourceDecisionModule, DecisionResult
from controller.prometheus_collector import PrometheusCollector

logger = logging.getLogger(__name__)

# ── Prometheus-метрики прогнозов и решений ─────────────────────────────────
# Публикуются на /metrics (HTTP server поднимается в bootstrap.py), Prometheus
# скрейпит их раз в 5 секунд. Тем самым прогнозы за прошедшее время сохраняются
# в TSDB и доступны для отображения на графиках истории.
from prometheus_client import Gauge

FORECAST_CPU = Gauge(
    "controller_forecast_cpu",
    "Прогноз медианы утилизации CPU (доля от лимита) на k шагов вперёд",
    ["horizon"],
)
FORECAST_CPU_LOWER = Gauge(
    "controller_forecast_cpu_lower",
    "Нижний квантиль ε/2 прогноза CPU",
    ["horizon"],
)
FORECAST_CPU_UPPER = Gauge(
    "controller_forecast_cpu_upper",
    "Верхний квантиль 1-ε/2 прогноза CPU (используется для масштабирования)",
    ["horizon"],
)
DECISION_R_REQ = Gauge(
    "controller_decision_r_req",
    "Требуемое число реплик (до применения r_max)",
)
DECISION_R_FIN = Gauge(
    "controller_decision_r_fin",
    "Итоговое решение о числе реплик (после гистерезиса)",
)
DECISION_DELTA_T = Gauge(
    "controller_decision_delta_t",
    "Адаптивный порог гистерезиса δ_t",
)
DECISION_SATURATION = Gauge(
    "controller_decision_saturation",
    "1 если r_req превышает r_max (RESOURCE_SATURATION), иначе 0",
)
DECISION_CONFIRM_COUNTER = Gauge(
    "controller_decision_confirm_counter",
    "Счётчик подтверждения масштабирования вниз (растёт до tau шагов)",
)
DECISION_ITERATION = Gauge(
    "controller_iteration",
    "Текущая итерация управляющего цикла",
)


class ControlLoop:
    """
    Замкнутый цикл управления нагрузкой.

      - Буфер наблюдений: скользящее окно retrain_window=2016 точек
      - Дообучение: каждые retrain_every=60 итераций
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        ts_cfg  = self.cfg["timeseries"]
        pp_cfg  = self.cfg["preprocessing"]
        mdl_cfg = self.cfg["model"]
        dec_cfg = self.cfg["decision"]
        k8s_cfg = self.cfg["kubernetes"]
        prom_cfg = self.cfg["prometheus"]

        # ── Модуль 1: Предобработка ───────────────────────────────────────
        self.preprocessor = Preprocessor(
            w_a=pp_cfg["w_a"],
            iqr_alpha=pp_cfg["iqr_alpha"],
            w_norm=pp_cfg["w_norm"],
            period=ts_cfg["period_p"],
            robust=pp_cfg["stl_robust"],
        )

        # ── Модуль 2: Гибридное прогнозирование ──────────────────────────
        self.forecaster = HybridForecaster(
            preprocessor=self.preprocessor,
            n_T=pp_cfg["n_trend"],
            n_cycles=pp_cfg["n_cycles_seasonal"],
            period=ts_cfg["period_p"],
            horizon_h=ts_cfg["horizon_h"],
            w_input=mdl_cfg["w_input"],
            quantiles=mdl_cfg["quantiles"],
            hidden_dim=mdl_cfg["hidden_dim"],
            dropout=mdl_cfg["dropout"],
            lr=mdl_cfg["lr"],
            lr_decay=mdl_cfg["lr_decay"],
            grad_clip=mdl_cfg["grad_clip"],
            max_epochs=mdl_cfg["max_epochs"],
            patience=mdl_cfg["patience"],
            batch_size=mdl_cfg["batch_size"],
        )

        # ── Модуль 3: Принятие решений ────────────────────────────────────
        db_cfg = dec_cfg["db"]
        # Kubernetes-клиент подключается только если in_cluster=True
        k8s_client = None
        if k8s_cfg.get("in_cluster", False):
            try:
                from kubernetes import config as k8s_config
                k8s_config.load_incluster_config()
                k8s_client = True
                logger.info("Kubernetes in-cluster config loaded.")
            except Exception as e:
                logger.warning(f"K8s in-cluster config failed: {e}")

        self.decision_module = ResourceDecisionModule(
            cpu_target=dec_cfg["cpu_target"],
            epsilon=dec_cfg["epsilon"],
            r_min=dec_cfg["r_min"],
            r_max_cluster=dec_cfg["r_max_cluster"],
            tau=dec_cfg["tau"],
            beta=dec_cfg["beta"],
            max_conn=db_cfg["max_conn"],
            conn_reserve=db_cfg["conn_reserve"],
            pool_size=db_cfg["pool_size"],
            kubernetes_client=k8s_client,
            namespace=k8s_cfg["namespace"],
            deployment_name=k8s_cfg["deployment_name"],
        )

        # Sync _r_cur with the actual cluster state — иначе контроллер
        # стартует с r_min=2, а в манифесте deployment'а уже стоит
        # replicas=4, и первое решение "не нужно масштабировать" принимается
        # на основе неверного значения.
        if k8s_client is not None and self.decision_module._k8s is not None:
            try:
                from kubernetes import client as kc
                scale = kc.AppsV1Api().read_namespaced_deployment_scale(
                    name=k8s_cfg["deployment_name"],
                    namespace=k8s_cfg["namespace"],
                )
                cur = int(scale.spec.replicas)
                self.decision_module._r_cur = cur
                logger.info(f"Synced _r_cur with cluster: {cur} replicas")
            except Exception as e:
                logger.warning(f"Could not sync replicas with cluster: {e}")

        # ── Сборщик метрик ────────────────────────────────────────────────
        self.collector = PrometheusCollector(
            prometheus_url=prom_cfg["url"],
            queries=prom_cfg["queries"],
        )

        # ── Буферы наблюдений ─────────────────────────────────────────────
        retrain_window = mdl_cfg["retrain_window"]   # 2016
        self.cpu_buffer: deque = deque(maxlen=retrain_window)
        self.ts_buffer: deque  = deque(maxlen=retrain_window)
        self.phi_buffer: deque = deque(maxlen=retrain_window)   # каждый элемент (3,)

        # ── Счётчики ──────────────────────────────────────────────────────
        self.iteration: int = 0
        self.retrain_every: int = mdl_cfg["retrain_every"]   # 60
        self.dt_seconds: int = ts_cfg["dt_minutes"] * 60     # 300

        # История метрик для API стенда
        self.metrics_history: List[Dict[str, Any]] = []
        self._is_initial_training_done: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Запуск
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, initial_data: Optional[Dict] = None) -> None:
        """
        Бесконечный управляющий цикл.
        initial_data: можно передать исторические данные для начального обучения.
        """
        logger.info("Control loop starting.")

        # Если переданы исторические данные — обучаем немедленно
        if initial_data is not None:
            self._initial_fit(initial_data)

        while True:
            iter_start = time.time()
            self.iteration += 1

            try:
                self._step()
            except Exception as e:
                logger.error(f"Iteration {self.iteration} failed: {e}", exc_info=True)

            # Ждём до следующего интервала Δt=5 мин
            elapsed = time.time() - iter_start
            sleep_time = max(0.0, self.dt_seconds - elapsed)
            logger.debug(f"Iteration {self.iteration} took {elapsed:.1f}s, sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)

    def _step(self) -> Optional[DecisionResult]:
        """Одна итерация управляющего цикла."""
        now_ts = int(time.time())

        # ── 1. Сбор метрик ───────────────────────────────────────────────
        m_t, phi_t = self.collector.collect()
        cpu_t = float(m_t[0])

        # Forward fill при пропуске
        if cpu_t == 0.0 and len(self.cpu_buffer) > 0:
            cpu_t = self.cpu_buffer[-1]
            logger.debug("Forward fill applied (cpu=0 from Prometheus)")

        # Добавляем в буфер
        self.cpu_buffer.append(cpu_t)
        self.ts_buffer.append(now_ts)
        self.phi_buffer.append(phi_t)

        # ── 2. Прогнозирование ───────────────────────────────────────────
        result = None
        if self._is_initial_training_done:
            cpu_arr = np.array(self.cpu_buffer)
            ts_arr  = np.array(self.ts_buffer)
            phi_arr = np.column_stack(list(self.phi_buffer)).T  # (3, n)

            try:
                cpu_hat, q_lower, q_upper = self.forecaster.predict(
                    cpu_arr, ts_arr, phi_arr.T if phi_arr.ndim == 2 else phi_arr
                )
                # Публикуем прогнозы по всем горизонтам в Prometheus
                for k in range(len(cpu_hat)):
                    h_label = str(k + 1)
                    FORECAST_CPU.labels(horizon=h_label).set(float(cpu_hat[k]))
                    FORECAST_CPU_LOWER.labels(horizon=h_label).set(float(q_lower[k]))
                    FORECAST_CPU_UPPER.labels(horizon=h_label).set(float(q_upper[k]))

                # Используем h=1 (первый шаг) для текущего управляющего воздействия
                result = self.decision_module.step(
                    q_upper=float(q_upper[0]),
                    q_lower=float(q_lower[0]),
                )
            except Exception as e:
                logger.error(f"Forecasting/decision error: {e}", exc_info=True)
                # Fallback: реактивный HPA
                result = self._reactive_hpa(cpu_t)
        else:
            logger.info(
                f"Accumulating data: {len(self.cpu_buffer)} / "
                f"{2 * self.preprocessor.period} observations needed for STL"
            )
            # Переключаемся на реактивный HPA пока не накопили данные
            result = self._reactive_hpa(cpu_t)

        # ── 3. Дообучение каждые retrain_every итераций ──────────────────
        min_obs = 2 * self.preprocessor.period  # минимум для STL
        if (
            len(self.cpu_buffer) >= min_obs
            and self.iteration % self.retrain_every == 0
        ):
            logger.info(f"Retraining at iteration {self.iteration}.")
            self._retrain()
            self._is_initial_training_done = True

        # ── 4. Публикация решения в Prometheus ────────────────────────────
        DECISION_ITERATION.set(float(self.iteration))
        if result is not None:
            DECISION_R_REQ.set(float(result.r_req))
            DECISION_R_FIN.set(float(result.r_fin))
            DECISION_DELTA_T.set(float(result.delta_t))
            DECISION_SATURATION.set(1.0 if result.saturation else 0.0)
            DECISION_CONFIRM_COUNTER.set(float(result.confirm_counter))

        # ── 5. Логирование ────────────────────────────────────────────────
        record = {
            "iteration": self.iteration,
            "timestamp": now_ts,
            "cpu_t": cpu_t,
            "phi_t": phi_t.tolist(),
            "r_cur": self.decision_module.current_replicas,
            "result": result.__dict__ if result else None,
        }
        self.metrics_history.append(record)
        if len(self.metrics_history) > 10_000:
            self.metrics_history = self.metrics_history[-10_000:]

        return result

    def _initial_fit(self, data: Dict) -> None:
        """Начальное обучение на исторических данных."""
        cpu_arr = np.array(data["cpu"])
        ts_arr  = np.array(data["timestamps"])
        phi_arr = np.array(data["phi"])  # (3, n)
        self.forecaster.fit(cpu_arr, ts_arr, phi_arr)
        self._is_initial_training_done = True
        logger.info("Initial fit complete.")

    def _retrain(self) -> None:
        """Дообучение на текущем буфере."""
        cpu_arr = np.array(self.cpu_buffer)
        ts_arr  = np.array(self.ts_buffer)
        phi_arr = np.column_stack(list(self.phi_buffer))  # (3, n)
        self.forecaster.retrain(cpu_arr, ts_arr, phi_arr)

    def _reactive_hpa(self, cpu_t: float) -> DecisionResult:
        """
        Fallback: реактивный HPA по стандартной схеме Kubernetes.
        r = ceil(r_cur * cpu_t / cpu_target)
        """
        r_cur = self.decision_module.current_replicas
        r_new = math.ceil(r_cur * cpu_t / self.decision_module.cpu_target)
        r_new = max(self.decision_module.r_min,
                    min(self.decision_module.r_max, r_new))
        if r_new != self.decision_module._r_cur:
            old = self.decision_module._r_cur
            self.decision_module._r_cur = r_new
            if self.decision_module._k8s is not None:
                self.decision_module._apply_k8s(r_new)
            logger.info(f"Reactive HPA: {old} -> {r_new} (cpu_t={cpu_t:.2f})")
        return DecisionResult(
            r_req=r_new, r_bounded=r_new, r_fin=r_new,
            action="hpa_reactive", delta_t=0.0,
            saturation=False, confirm_counter=0,
        )
