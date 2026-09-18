from pathlib import Path

import pandas as pd
import yaml


def _load_universe() -> list[str]:
    with open("config/universes.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    return list(cfg.get("universe") or cfg.get("etf", {}).get("symbols", []))


def _load_panel() -> pd.DataFrame:
    panel = pd.read_parquet(Path("assessment/inputs/panel.parquet")).reset_index()
    if "date" in panel.columns:
        panel = panel.rename(columns={"date": "Date"})
    if "Symbol" in panel.columns and "symbol" not in panel.columns:
        panel = panel.rename(columns={"Symbol": "symbol"})
    panel["Date"] = pd.to_datetime(panel["Date"]).dt.tz_localize(None)
    return panel.sort_values(["symbol", "Date"])


def test_feature_panel_calendar_is_shared_and_sane():
    symbols = _load_universe()
    panel = _load_panel()

    assert symbols
    assert set(symbols).issubset(set(panel["symbol"].unique()))

    reference_dates = None
    for symbol in symbols:
        dates = panel.loc[panel["symbol"] == symbol, "Date"].reset_index(drop=True)
        assert dates.is_monotonic_increasing
        assert not dates.duplicated().any()
        assert dates.dt.tz is None

        if reference_dates is None:
            reference_dates = dates
        else:
            pd.testing.assert_series_equal(dates, reference_dates, check_names=False)
