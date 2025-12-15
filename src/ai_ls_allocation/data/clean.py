from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml
import pandas_market_calendars as mcal

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
CONFIG_DIR = Path("config")


def load_universe() -> Tuple[List[str], pd.Timestamp, pd.Timestamp]:
    """
    Lies die ETF-Universen und den Zeitraum aus config/universes.yaml.

    etf:
      symbols: [SPY, QQQ, ...]
      start: 2007-01-01
      end: today
    """
    with open(CONFIG_DIR / "universes.yaml") as f:
        cfg = yaml.safe_load(f)

    etf_cfg = cfg["etf"]
    symbols = etf_cfg["symbols"]
    start = pd.Timestamp(etf_cfg.get("start", "2007-01-01"))

    end_raw = etf_cfg.get("end", "today")
    if isinstance(end_raw, str) and end_raw.lower() == "today":
        end = pd.Timestamp.today().normalize()
    else:
        end = pd.Timestamp(end_raw)

    return symbols, start, end


def nyse_calendar_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """
    Erzeuge einen einheitlichen NYSE-Handelstage-Index (zeit­zonenfrei, sortiert).
    """
    nyse = mcal.get_calendar("XNYS")
    sched = nyse.schedule(start_date=start, end_date=end)
    idx = sched.index

    # ggf. Zeitzone entfernen
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)

    idx = pd.DatetimeIndex(idx).sort_values()
    return idx


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Falls die Spalten MultiIndex oder Tupel sind (yfinance kann das tun),
    auf einfache Strings herunterbrechen.
    """
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) for c in df.columns]
    else:
        new_cols = []
        for c in df.columns:
            if isinstance(c, tuple):
                new_cols.append(str(c[0]))
            else:
                new_cols.append(str(c))
        df.columns = new_cols

    return df


def normalize_close(df: pd.DataFrame) -> pd.Series:
    """
    Extrahiere eine saubere Close-Serie als float mit DatetimeIndex.
    Bevorzugt 'Adj Close', sonst 'Close'.
    """
    df = _flatten_columns(df)

    # Index als Datetime ohne TZ
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    df = df.copy()
    df.index = idx

    cols_lower = {str(c).lower(): c for c in df.columns}

    close_col = None
    for cand in ["adj close", "adj_close", "close"]:
        if cand in cols_lower:
            close_col = cols_lower[cand]
            break

    if close_col is None:
        raise KeyError(f"no close/adj close column found in {list(df.columns)}")

    s = df[close_col]
    s = pd.to_numeric(s, errors="coerce").astype(float)
    s.index.name = "Date"
    return s


def clean_symbol(symbol: str, cal_index: pd.DatetimeIndex) -> None:
    """
    Ein Symbol:
    - raw/{symbol}.parquet laden
    - Close normalisieren
    - auf NYSE-Kalender reindizieren + ffill
    - nach data/clean/{symbol}.parquet schreiben
    """
    fp_raw = RAW_DIR / f"{symbol}.parquet"
    df_raw = pd.read_parquet(fp_raw)

    close = normalize_close(df_raw)

    # Auf Kalender reindexen und Lücken nur wegen Nicht-Handelstagen per ffill füllen
    s = close.reindex(cal_index).ffill()

    out = pd.DataFrame({"close": s})
    out.index.name = "Date"

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    fp_clean = CLEAN_DIR / f"{symbol}.parquet"
    out.to_parquet(fp_clean)
    print(f"[clean] {symbol} -> {fp_clean}")


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    symbols, start, end = load_universe()
    print(f"[clean] universe: {symbols}, {start.date()} -> {end.date()}")

    cal_idx = nyse_calendar_index(start, end)

    for s in symbols:
        clean_symbol(s, cal_idx)

    print("[done] saved to data/clean/*.parquet")


if __name__ == "__main__":
    main()
