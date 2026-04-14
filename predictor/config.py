"""
predictor/config.py — Единая загрузка конфигурации из config.yaml.

Все тесты, скрипты и продакшн-код импортируют параметры отсюда,
чтобы менять гиперпараметры в одном месте.

    from predictor.config import CFG_PREPROCESSOR, CFG_FORECASTER
"""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CONFIG_PATH = os.path.join(_ROOT, "config.yaml")


def load_raw(path: str = _CONFIG_PATH) -> Dict[str, Any]:
    """Загружает и возвращает сырой dict из config.yaml."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_configs(raw: Dict[str, Any]):
    ts = raw["timeseries"]
    pp = raw["preprocessing"]
    m = raw["model"]

    cfg_preprocessor = dict(
        w_a=pp["w_a"],
        iqr_alpha=pp["iqr_alpha"],
        w_norm=pp["w_norm"],
        period=ts["period_p"],
        robust=pp["stl_robust"],
    )

    cfg_forecaster = dict(
        n_T=pp["n_trend"],
        n_cycles=pp["n_cycles_seasonal"],
        period=ts["period_p"],
        horizon_h=ts["horizon_h"],
        w_input=m["w_input"],
        quantiles=m["quantiles"],
        hidden_dim=m["hidden_dim"],
        dropout=m["dropout"],
        lr=m["lr"],
        lr_decay=m["lr_decay"],
        grad_clip=m["grad_clip"],
        max_epochs=m["max_epochs"],
        patience=m["patience"],
        batch_size=m["batch_size"],
    )

    d = raw.get("decision", {})
    db = d.get("db", {})
    cfg_decision = dict(
        cpu_target=d.get("cpu_target", 0.70),
        epsilon=d.get("epsilon", 0.05),
        r_min=d.get("r_min", 2),
        r_max_cluster=d.get("r_max_cluster", 8),
        tau=d.get("tau", 4),
        beta=d.get("beta", 0.3),
        max_conn=db.get("max_conn", 100),
        conn_reserve=db.get("conn_reserve", 10),
        pool_size=db.get("pool_size", 5),
    )

    return cfg_preprocessor, cfg_forecaster, cfg_decision


_raw = load_raw()
CFG_PREPROCESSOR, CFG_FORECASTER, CFG_DECISION = _build_configs(_raw)
