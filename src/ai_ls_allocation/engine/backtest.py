from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Wir benutzen die Universe-Definition und Pfade aus data.clean
from ai_ls_allocation.data.clean import load_universe, CLEAN_DIR

REPORTS_DIR = Path("reports")


def load_prices() -> Tuple[pd.DataFrame, List[str]]:
    """
    Lädt die bereinigten Close-Preise aus data/clean/*.parquet und baut
    ein Price-DataFrame px mit Spalten = Symbole.
    """
    symbols, start, end = load_universe()

    cols = []
    for s in symbols:
        df = pd.read_parquet(CLEAN_DIR / f"{s}.parquet")
        ser = df["close"].copy()
        ser.name = s
        cols.append(ser)

    px = pd.concat(cols, axis=1)

    # Index sicher als tz-naive Datetimes & sortiert
    idx = pd.to_datetime(px.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    px.index = idx
    px = px.sort_index()

    # Auf Zeitraum aus dem Universe beschränken (sollte eh schon passen)
    px = px.loc[start:end]

    return px, symbols


def build_dummy_weights(r: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """
    Sehr einfache Dummy-Allokation: jeden Tag gleichgewichtet über alle Symbole.
    """
    n = len(symbols)
    w = pd.DataFrame(index=r.index, columns=symbols, dtype=float)

    if n > 0:
        w.iloc[:, :] = 1.0 / n
    else:
        w.iloc[:, :] = 0.0

    return w


def run_backtest() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Preise & Returns genau wie im Test berechnen
    px, symbols = load_prices()
    r = px.pct_change(fill_method=None).fillna(0.0)

    # 2) Dummy-Gewichte bauen
    w = build_dummy_weights(r, symbols)

    # 3) Portfoliorendite mit gelaggten Gewichten:
    #    port_ret_t = w_{t-1} · r_t
    w_lag = w.shift(1).fillna(0.0)
    port_ret = (w_lag * r).sum(axis=1)
    port_ret.name = "port_ret"

    # 4) Equity-Kurve & Turnover (nur für die Summary, Tests achten nur auf port_ret)
    eq = (1.0 + port_ret).cumprod()
    eq.name = "equity"

    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    turnover.name = "turnover"

    ts = pd.concat([port_ret, eq, turnover], axis=1)

    # 5) CSVs im erwarteten Format schreiben
    w.to_csv(REPORTS_DIR / "bt_dummy_weights.csv", index_label="date")
    ts.to_csv(REPORTS_DIR / "bt_dummy_timeseries.csv", index_label="date")

    # 6) Summary ausgeben (Werte sind egal für Tests, aber nice to have)
    days = len(port_ret)
    if days > 1:
        cagr = eq.iloc[-1] ** (252.0 / days) - 1.0
        vol = port_ret.std(ddof=0)
        sharpe = (np.sqrt(252.0) * port_ret.mean() / vol) if vol > 0 else np.nan
        running_max = eq.cummax()
        maxdd = (eq / running_max - 1.0).min()
        avg_turnover = turnover.mean()
    else:
        cagr = sharpe = maxdd = avg_turnover = np.nan

    print("[SUMMARY]")
    print(f"days: {days}")
    print(f"CAGR: {cagr:.6f}")
    print(f"Sharpe: {sharpe:.6f}")
    print(f"MaxDD: {maxdd:.6f}")
    print(f"AvgDailyTurnover: {avg_turnover:.6f}")
    # Wir modellieren hier keine expliziten Kosten → 0.0
    print("CostShare: 0.000000")
    print("[done] wrote reports/bt_dummy_timeseries.csv & bt_dummy_weights.csv")


if __name__ == "__main__":
    run_backtest()
