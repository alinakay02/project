"""
locust/locustfile.py — Сценарии генерации нагрузки (параграф 4.1, инструмент Locust 2.20)

Три класса пользователей соответствуют трём функциональным классам запросов.

ПОСТОЯННАЯ НАГРУЗКА:
  locust -f locustfile.py --headless -u 30 -r 5 --run-time 5m --host http://localhost:8080

ПЕРЕМЕННАЯ НАГРУЗКА (суточный профиль, см. DailyDiurnalShape ниже):
  # 24 часа (естественная развёртка):
  locust -f locustfile.py --headless --host http://localhost:8080

  # Сжатый профиль (24ч → 2ч для отладки):
  $env:COMPRESS_FACTOR="12"; locust -f locustfile.py --headless --host http://localhost:8080

  # С веб-интерфейсом (UI на http://localhost:8089):
  locust -f locustfile.py --host http://localhost:8080
"""

import math
import os
import random
from locust import HttpUser, LoadTestShape, task, between, constant_pacing

# ─── Класс 1: Вычислительные запросы (cpu_t ≈ 0.65–0.85) ───────────────────
class ComputeUser(HttpUser):
    """Генерирует запросы класса 1: /compute/light, /compute/medium, /compute/heavy."""
    weight = 1
    wait_time = between(0.1, 0.5)

    @task(3)
    def compute_light(self):
        self.client.get("/compute/light", name="/compute/light [cls1]")

    @task(2)
    def compute_medium(self):
        self.client.get("/compute/medium", name="/compute/medium [cls1]")

    @task(1)
    def compute_heavy(self):
        self.client.get("/compute/heavy", name="/compute/heavy [cls1]")


# ─── Класс 2: Запросы к базе данных (cpu_t ≈ 0.20–0.35) ─────────────────────
class DBUser(HttpUser):
    """Генерирует запросы класса 2: /db/read, /db/write, /db/aggregate."""
    weight = 1
    wait_time = between(0.2, 1.0)

    @task(5)
    def db_read(self):
        self.client.get("/db/read", name="/db/read [cls2]")

    @task(2)
    def db_write(self):
        self.client.get("/db/write", name="/db/write [cls2]")

    @task(1)
    def db_aggregate(self):
        self.client.get("/db/aggregate", name="/db/aggregate [cls2]")


# ─── Класс 3: Управление памятью (cpu_t ≈ 0.30–0.55) ───────────────────────
class MemoryUser(HttpUser):
    """Генерирует запросы класса 3: /memory/alloc, /memory/process, /memory/gc."""
    weight = 1
    wait_time = between(0.3, 1.5)

    @task(3)
    def memory_alloc(self):
        self.client.get("/memory/alloc", name="/memory/alloc [cls3]")

    @task(2)
    def memory_process(self):
        self.client.get("/memory/process", name="/memory/process [cls3]")

    @task(1)
    def memory_gc(self):
        self.client.get("/memory/gc", name="/memory/gc [cls3]")


# ─── Суточный профиль нагрузки (LoadTestShape) ─────────────────────────────
#
# Имитирует реальную суточную нагрузку веб-сервиса:
#   00:00–06:00 — ночь, минимальная активность (~BASE_USERS)
#   06:00–10:00 — утренний рост
#   10:00–14:00 — рабочий пик
#   14:00–18:00 — небольшой спад (обеденное время, потом возврат)
#   18:00–22:00 — вечерний пик (максимум, ~PEAK_USERS)
#   22:00–24:00 — ночной спад
#
# Параметры через переменные окружения:
#   DURATION_HOURS    — длительность теста в виртуальных часах (default: 24)
#   COMPRESS_FACTOR   — во сколько раз сжать виртуальное время (default: 1 → 24ч).
#                       Например, 12 → весь суточный цикл за 2 реальных часа.
#   BASE_USERS        — минимальное число пользователей (ночь). Default: 100
#   PEAK_USERS        — максимальное число пользователей (вечерний пик). Default: 1500
#   NOISE_AMPLITUDE   — амплитуда случайного шума (default: 0.10 = ±10%)

DURATION_HOURS  = float(os.getenv("DURATION_HOURS",  "24"))
COMPRESS_FACTOR = float(os.getenv("COMPRESS_FACTOR", "1"))
BASE_USERS      = int(  os.getenv("BASE_USERS",      "100"))
PEAK_USERS      = int(  os.getenv("PEAK_USERS",      "1500"))
NOISE_AMPLITUDE = float(os.getenv("NOISE_AMPLITUDE", "0.10"))


class DailyDiurnalShape(LoadTestShape):
    """
    Суточный профиль с двумя гауссовыми пиками (рабочий + вечерний).

    Форма (доля от PEAK_USERS):
      f(h) = max(
        0.65 * exp(-((h-12)^2) / 10),   # рабочий пик в 12:00, ширина σ≈3.2ч
        1.00 * exp(-((h-20)^2) / 6),    # вечерний пик в 20:00, ширина σ≈2.4ч
      ) + случайный шум

    Базовый уровень BASE_USERS подмешивается всегда.
    """

    total_duration_sec = int(DURATION_HOURS * 3600 / COMPRESS_FACTOR)

    def tick(self):
        run_time = self.get_run_time()
        if run_time >= self.total_duration_sec:
            return None  # тест завершён

        # Виртуальный час суток в [0, 24)
        virtual_hour = (run_time / self.total_duration_sec) * 24.0

        # Два пика (рабочий и вечерний)
        morning = 0.65 * math.exp(-((virtual_hour - 12) ** 2) / 10.0)
        evening = 1.00 * math.exp(-((virtual_hour - 20) ** 2) / 6.0)
        shape   = max(morning, evening)

        # Случайный шум ±NOISE_AMPLITUDE
        noise = 1.0 + random.uniform(-NOISE_AMPLITUDE, NOISE_AMPLITUDE)
        shape *= noise

        users = int(BASE_USERS + (PEAK_USERS - BASE_USERS) * max(0.0, shape))

        # spawn_rate: 5% от текущего числа в секунду (но не меньше 5)
        spawn_rate = max(5, users // 20)

        return (users, spawn_rate)
