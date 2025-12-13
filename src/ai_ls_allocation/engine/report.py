from __future__ import annotations
from pathlib import Path
import pandas as pd
from .metrics import sharpe, calmar, max_drawdown, cagr

def main():
    p = Path("reports/bt_dummy_timeseries.csv")
    ts = pd.read_csv(p, parse_dates=["date"]).set_index("date")
    net = ts["net_ret"].fillna(0.0)
    eq = (1 + net).cumprod()
    print("[REPORT]")
    print(f"CAGR  : {cagr(net):.4f}")
    print(f"Sharpe: {sharpe(net):.2f}")
    print(f"Calmar: {calmar(net):.2f}")
    print(f"MaxDD : {max_drawdown(eq):.2%}")
