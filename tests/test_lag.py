import yaml, pandas as pd
from pathlib import Path
from pandas.testing import assert_series_equal

def test_port_ret_uses_shifted_weights():
    with open("config/universes.yaml") as f:
        symbols = yaml.safe_load(f)["etf"]["symbols"]

    # Preise → Returns (Series sauber benennen, ohne rename(s))
    cols = []
    for s in symbols:
        df = pd.read_parquet(Path("data/clean")/f"{s}.parquet")
        ser = df["close"].copy()
        ser.name = s
        cols.append(ser)
    px = pd.concat(cols, axis=1)
    r = px.pct_change(fill_method=None).fillna(0.0)

    # Weights & Timeseries laden
    w  = pd.read_csv("reports/bt_dummy_weights.csv", parse_dates=["date"]).set_index("date").reindex(r.index).fillna(0.0)
    ts = pd.read_csv("reports/bt_dummy_timeseries.csv", parse_dates=["date"]).set_index("date").reindex(r.index)

    # Recompute: port_ret = w_{t-1} · r_t
    port_ret_recalc = (w.shift(1).fillna(0.0) * r).sum(axis=1)
    port_ret_recalc.name = "port_ret"

    # Vergleich mit Toleranz (floating)
    assert_series_equal(
        port_ret_recalc.fillna(0).round(12),
        ts["port_ret"].fillna(0).round(12),
        check_names=False,
        check_dtype=False,
        atol=0, rtol=0
    )
