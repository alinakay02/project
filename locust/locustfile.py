"""
locust/locustfile.py — Сценарии генерации нагрузки (параграф 4.1, инструмент Locust 2.20)

Три класса пользователей соответствуют трём функциональным классам запросов.
Запуск: locust -f locustfile.py --headless -u 30 -r 5 --run-time 5m --host http://localhost:5000
"""

import random
from locust import HttpUser, task, between, constant_pacing

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
