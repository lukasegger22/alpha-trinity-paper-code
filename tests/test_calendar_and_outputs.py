import pandas as pd, os
from pathlib import Path

def test_clean_index_monotonic():
    p = Path("data/clean/SPY.parquet")
    assert p.exists()
    df = pd.read_parquet(p)
    assert df.index.is_monotonic_increasing
    assert df.index.tz is None

def test_reports_exist_after_bt_dummy():
    # Der Backtest wurde bereits gelaufen; prüfe Outputs
    times = Path("reports/bt_dummy_timeseries.csv")
    wts   = Path("reports/bt_dummy_weights.csv")
    assert times.exists() and wts.exists()
