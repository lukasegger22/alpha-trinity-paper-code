from __future__ import annotations
import pandas as pd, pandas_market_calendars as mcal

def nyse_trading_days(start: str, end: str) -> pd.DatetimeIndex:
    cal = mcal.get_calendar("NYSE")
    sched = cal.schedule(start_date=start, end_date=end)
    idx = mcal.date_range(sched, frequency="1D")
    return pd.DatetimeIndex(idx.tz_localize(None))
