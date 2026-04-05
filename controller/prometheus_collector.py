"""
controller/prometheus_collector.py — Сбор метрик из Prometheus (параграф 3.2.1)

Формирует вектор m_t = (cpu_t, mem_t, rps_t, lat_t, err_t) и вектор φ_t
посредством PromQL-запросов, описанных в config.yaml.
"""

import time
import logging
import numpy as np
import requests
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)


class PrometheusCollector:
    """
    Клиент Prometheus: выполняет PromQL-запросы и формирует m_t, φ_t.

    Агрегирование до Δt=5 мин выполняется на стороне PromQL (rate(...[5m])).
    Скрапинг cAdvisor и /metrics каждые 5 секунд (config.yaml).
    """

    def __init__(self, prometheus_url: str, queries: Dict[str, str]):
        self.url = prometheus_url.rstrip("/")
        self.queries = queries
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    def _query(self, promql: str) -> Optional[float]:
        """Выполняет instant query, возвращает первое скалярное значение."""
        try:
            resp = self._session.get(
                f"{self.url}/api/v1/query",
                params={"query": promql},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("data", {}).get("result", [])
            if results:
                return float(results[0]["value"][1])
            return None
        except Exception as e:
            logger.warning(f"Prometheus query failed: {e} | query: {promql[:60]}")
            return None

    def _query_vector(self, promql: str) -> Dict[str, float]:
        """Выполняет query, возвращает dict {label_value: value}."""
        try:
            resp = self._session.get(
                f"{self.url}/api/v1/query",
                params={"query": promql},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("data", {}).get("result", [])
            out = {}
            for r in results:
                label = r["metric"].get("class_id", "unknown")
                out[label] = float(r["value"][1])
            return out
        except Exception as e:
            logger.warning(f"Prometheus vector query failed: {e}")
            return {}

    def collect(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Формирует m_t (5 метрик) и φ_t (3 компоненты).

        Возвращает:
          m_t  : np.ndarray shape (5,) — [cpu, mem, rps, lat_p95_ms, err_rate]
          phi_t: np.ndarray shape (3,) — доли классов 1, 2, 3
        """
        cpu    = self._query(self.queries["cpu"]) or 0.0
        mem    = self._query(self.queries["mem"]) or 0.0
        rps    = self._query(self.queries["rps"]) or 0.0
        lat    = self._query(self.queries["lat_p95"]) or 0.0
        err    = self._query(self.queries["err_rate"]) or 0.0

        # Ограничиваем cpu и mem в [0, 1]
        cpu = float(np.clip(cpu, 0.0, 2.0))   # может временно превышать 1.0 при throttling
        mem = float(np.clip(mem, 0.0, 1.5))
        err = float(np.clip(err, 0.0, 1.0))

        m_t = np.array([cpu, mem, rps, lat, err], dtype=np.float32)

        # Состав классов (формула 3.2)
        class_rates = self._query_vector(self.queries["class_counts"])
        c1 = class_rates.get("1", 0.0)
        c2 = class_rates.get("2", 0.0)
        c3 = class_rates.get("3", 0.0)
        total = c1 + c2 + c3
        if total > 0:
            phi_t = np.array([c1 / total, c2 / total, c3 / total], dtype=np.float32)
        else:
            phi_t = np.array([1/3, 1/3, 1/3], dtype=np.float32)

        logger.debug(
            f"m_t=[cpu={cpu:.3f} mem={mem:.3f} rps={rps:.1f} "
            f"lat={lat:.1f}ms err={err:.4f}] φ={phi_t}"
        )
        return m_t, phi_t
