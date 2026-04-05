"""
controller/decision.py — Модуль принятия решений об управлении ресурсами (параграф 3.2.3)

Реализует:
  - Вычисление r_req по формуле (3.7)
  - Проверку ограничений инфраструктуры (формула 3.6, условие 3.3)
  - Механизм гистерезиса с адаптивным порогом δ_t (формулы 3.16, 3.17)
  - Итоговое воздействие r_fin (формула 3.18)
  - PATCH-запрос к Kubernetes API
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional
import datetime

logger = logging.getLogger(__name__)


@dataclass
class DecisionResult:
    """Результат одного шага принятия решений."""
    r_req: int              # требуемое число реплик по формуле (3.7)
    r_bounded: int          # после применения ограничений r_max
    r_fin: int              # итоговое воздействие (формула 3.18)
    action: str             # "scale_up" | "scale_down" | "no_change" | "saturated"
    delta_t: float          # адаптивный порог δ_t (формула 3.17)
    saturation: bool        # True если r_req > r_max (RESOURCE_SATURATION)
    confirm_counter: int    # текущий счётчик подтверждения масштабирования вниз
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())


class ResourceDecisionModule:
    """
    Модуль принятия решений о масштабировании вычислительных ресурсов.

    Параметры (из config.yaml → decision):
      cpu_target   : 0.70
      epsilon      : 0.05
      r_min        : 2
      r_max_cluster: 8
      tau          : 4  (шага подтверждения)
      beta         : 0.3
      db.max_conn  : 100
      db.conn_reserve: 10
      db.pool_size : 5
    """

    def __init__(
        self,
        cpu_target: float = 0.70,
        epsilon: float = 0.05,
        r_min: int = 2,
        r_max_cluster: int = 8,
        tau: int = 4,
        beta: float = 0.3,
        max_conn: int = 100,
        conn_reserve: int = 10,
        pool_size: int = 5,
        kubernetes_client=None,   # передаётся при реальном k8s
        namespace: str = "default",
        deployment_name: str = "webapp",
    ):
        self.cpu_target = cpu_target
        self.epsilon = epsilon
        self.r_min = r_min
        self.tau = tau
        self.beta = beta
        self.namespace = namespace
        self.deployment_name = deployment_name
        self._k8s = kubernetes_client

        # Ограничение СУБД (формула 3.3): r_max_db = ⌊(max_conn - conn_reserve) / pool_size⌋
        r_max_db = math.floor((max_conn - conn_reserve) / pool_size)
        self.r_max = min(r_max_cluster, r_max_db)
        logger.info(
            f"r_max_cluster={r_max_cluster}, r_max_db={r_max_db} → r_max={self.r_max}"
        )

        # Состояние гистерезиса
        self._r_cur: int = r_min
        self._confirm_counter: int = 0   # счётчик подтверждения для масштабирования вниз

        # История решений для логирования
        self.history: list = []

    @property
    def current_replicas(self) -> int:
        return self._r_cur

    # ─────────────────────────────────────────────────────────────────────────
    # Основной метод (вызывается на каждой итерации цикла управления)
    # ─────────────────────────────────────────────────────────────────────────

    def step(
        self,
        q_upper: float,   # q̂_{1-ε}(t+1) — верхний квантиль прогноза
        q_lower: float,   # q̂_{ε/2}(t+1)  — нижний квантиль прогноза
    ) -> DecisionResult:
        """
        Один шаг модуля принятия решений.

        Аргументы:
          q_upper : верхний квантиль прогноза утилизации (порядок 1-ε = 0.975)
          q_lower : нижний квантиль (порядок ε/2 = 0.025) — для вычисления δ_t

        Возвращает:
          DecisionResult с итоговым r_fin и описанием действия.
        """
        # ── Шаг 1: r_req (формула 3.7) ────────────────────────────────────
        r_req = math.ceil(q_upper / self.cpu_target)
        r_req = max(r_req, self.r_min)

        # ── Шаг 2: Проверка ограничений инфраструктуры ────────────────────
        saturation = False
        r_bounded = min(r_req, self.r_max)
        if r_bounded < r_req:
            saturation = True
            logger.warning(
                f"RESOURCE_SATURATION: r_req={r_req} > r_max={self.r_max}. "
                f"Setting r_bounded={r_bounded}."
            )

        # ── Шаг 3: Адаптивный порог δ_t (формула 3.17) ───────────────────
        ci_width = max(q_upper - q_lower, 0.0)
        delta_t = self.beta * ci_width / self.cpu_target

        # ── Шаг 4: Механизм гистерезиса (формулы 3.16, 3.18) ─────────────
        action = "no_change"

        if r_bounded > self._r_cur:
            # Масштабирование ВВЕРХ — немедленно
            self._r_cur = r_bounded
            self._confirm_counter = 0
            action = "scale_up"
            logger.info(f"Scale UP → {self._r_cur} replicas (r_req={r_req})")

        elif r_bounded < self._r_cur - delta_t:
            # Условие для масштабирования вниз — проверяем τ шагов
            self._confirm_counter += 1
            if self._confirm_counter >= self.tau:
                # Масштабирование ВНИЗ подтверждено
                self._r_cur = max(self.r_min, r_bounded)
                self._confirm_counter = 0
                action = "scale_down"
                logger.info(
                    f"Scale DOWN → {self._r_cur} replicas "
                    f"(confirmed {self.tau} steps, δ_t={delta_t:.3f})"
                )
            else:
                action = "no_change"
                logger.debug(
                    f"Scale down candidate, confirm_counter={self._confirm_counter}/{self.tau}"
                )
        else:
            # Нет оснований для масштабирования
            self._confirm_counter = 0
            action = "no_change"

        # ── Шаг 5: r_fin (формула 3.18): max(r_min, r_dec) ───────────────
        r_fin = max(self.r_min, self._r_cur)
        if saturation:
            action = "saturated"

        result = DecisionResult(
            r_req=r_req,
            r_bounded=r_bounded,
            r_fin=r_fin,
            action=action,
            delta_t=delta_t,
            saturation=saturation,
            confirm_counter=self._confirm_counter,
        )
        self.history.append(result)

        # ── Шаг 6: PATCH к Kubernetes API (при наличии клиента) ──────────
        if self._k8s is not None and action in ("scale_up", "scale_down"):
            self._apply_k8s(r_fin)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Kubernetes PATCH
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_k8s(self, replicas: int) -> None:
        """
        PATCH /apis/apps/v1/namespaces/{ns}/deployments/{name}/scale
        spec.replicas = replicas
        """
        try:
            from kubernetes import client as k8s_client
            apps_v1 = k8s_client.AppsV1Api()
            body = {"spec": {"replicas": replicas}}
            apps_v1.patch_namespaced_deployment_scale(
                name=self.deployment_name,
                namespace=self.namespace,
                body=body,
            )
            logger.info(f"K8s PATCH applied: {self.deployment_name} → {replicas} replicas")
        except Exception as e:
            logger.error(f"K8s PATCH failed: {e}")

    def reset(self, initial_replicas: Optional[int] = None) -> None:
        """Сброс состояния (используется в тестах и при перезапуске)."""
        self._r_cur = initial_replicas if initial_replicas is not None else self.r_min
        self._confirm_counter = 0
        self.history.clear()
