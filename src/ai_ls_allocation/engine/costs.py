from __future__ import annotations
import yaml
import pandas as pd

def load_costs(path: str = "config/costs.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def trade_cost_bps_total(cfg) -> float:
    return float(cfg["commission_bps"] + cfg["half_spread_bps"] + cfg["slippage_extra_bps"])

def borrow_daily_rate(cfg) -> float:
    return float(cfg["short_borrow_annual_bps"]) / 10000.0 / 252.0

def trade_costs(delta_w: pd.DataFrame, cfg) -> pd.Series:
    # Kosten als Return-Abzug (bps * Turnover)
    bps = trade_cost_bps_total(cfg) / 10000.0
    turn = delta_w.abs().sum(axis=1)
    return bps * turn

def borrow_costs(weights: pd.DataFrame, cfg) -> pd.Series:
    # Borrow nur auf Shorts
    rate = borrow_daily_rate(cfg)
    shorts = weights.clip(upper=0).abs().sum(axis=1)
    return rate * shorts
