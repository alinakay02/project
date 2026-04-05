"""
app/models.py — Модель для хранения результатов экспериментов в PostgreSQL.

Таблица experiment_results хранит разобранные [METRIC] строки из pytest.
Каждая строка — одно наблюдение (метод, датасет, набор метрик).
"""

import json
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Index
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ExperimentResult(Base):
    __tablename__ = "experiment_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(32), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Тип эксперимента: compare, ablation, horizon, phi, timing, spike, retrain
    experiment = Column(String(64), nullable=False, index=True)

    # Датасет (Alibaba, Стационарный, Трендовый, Всплесковый, Смешанный)
    dataset = Column(String(64))

    # Название метода или конфигурации
    label = Column(String(128))

    # JSON с разобранными числовыми метриками
    metrics_json = Column(Text, nullable=False, default="{}")

    # Исходная [METRIC] строка для отладки
    raw_line = Column(Text)

    __table_args__ = (
        Index("ix_exp_run_experiment", "run_id", "experiment"),
    )

    @property
    def metrics(self):
        return json.loads(self.metrics_json or "{}")

    def __repr__(self):
        return f"<ExperimentResult {self.experiment}/{self.label} run={self.run_id}>"
