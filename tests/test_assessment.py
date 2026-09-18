import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_trinity_assessment import first_three as core
from alpha_trinity_assessment.robustness_repro_audit import feature_df_with_forward_horizon


def frames(n=6):
    dates = pd.bdate_range("2022-01-03", periods=n)
    returns = pd.DataFrame(0.0, index=dates, columns=["SPY", "GLD", "UUP"])
    return returns, returns.copy()


def test_signal_fills_next_close_not_same_close():
    returns, targets = frames()
    returns.iloc[1, 0] = 0.20
    returns.iloc[2, 0] = 0.10
    targets["SPY"] = 1.0
    result = core.evaluate_weights("test", targets, returns, transaction_cost=0)
    assert result.gross_returns.iloc[1] == 0
    assert result.gross_returns.iloc[2] == pytest.approx(0.10)
    assert result.turnover.iloc[0] == 0
    assert result.turnover.iloc[1] == 1


def test_daily_rebalancing_charges_drift_and_buy_hold_does_not():
    returns, targets = frames()
    targets[["SPY", "GLD"]] = 0.5
    returns.iloc[2, 0] = 0.20
    daily = core.evaluate_weights("mix", targets, returns, transaction_cost=0)
    hold = core.evaluate_weights("hold", targets, returns, transaction_cost=0, rebalance="buy_and_hold")
    assert daily.turnover.iloc[2] == pytest.approx(0.10)  # start-of-session NAV units
    assert hold.turnover.iloc[2] == 0
    assert hold.effective_weights.iloc[3, 0] == pytest.approx(0.6 / 1.1)
    assert daily.effective_weights.iloc[3, 0] == 0.5


def test_fees_are_self_financing_and_at_fill():
    returns, targets = frames()
    targets["SPY"] = 1
    result = core.evaluate_weights("test", targets, returns)
    assert result.trading_cost.iloc[1] == pytest.approx(0.001 / 1.001)
    assert result.net_returns.iloc[1] == pytest.approx(-result.trading_cost.iloc[1])
    np.testing.assert_allclose(result.trading_cost, result.turnover * core.TRANSACTION_COST)
    np.testing.assert_allclose(result.net_returns, result.gross_returns - result.trading_cost - result.carry_cost)
    assert (result.turnover.iloc[2:] < 1e-12).all()


@pytest.mark.parametrize("horizon", [10, 20, 40])
def test_horizon_targets_and_end_dates_are_purged_together(horizon):
    dates = pd.bdate_range("2021-10-01", periods=110)
    df = pd.DataFrame({"Date": dates, "symbol": "SPY", "proxy_price": np.arange(110) + 100.0})
    updated = feature_df_with_forward_horizon(df, horizon)
    cutoff = pd.Timestamp("2021-12-31")
    train = updated[(updated.Date < cutoff) & (updated.target_20d_end_date <= cutoff)]
    for i in train.index:
        assert dates[i + horizon] <= cutoff
        assert updated.loc[i, "target_20d_forward"] == pytest.approx((100 + i + horizon) / (100 + i) - 1)
    assert updated.tail(horizon).target_20d_forward.isna().all()
    assert updated.tail(horizon).target_20d_end_date.isna().all()


def test_prediction_only_has_no_vix_exposure_control():
    returns, signal = frames(80)
    signal["SPY"] = 1
    kwargs = dict(name="prediction", returns=returns, ml_signal=signal,
                  crisis_scores=pd.Series(0.9, index=returns.index), use_ml=True, use_crisis=False)
    low = core.build_engine_weights(**kwargs, daily_vix=pd.Series(12.0, index=returns.index))
    high = core.build_engine_weights(**kwargs, daily_vix=pd.Series(35.0, index=returns.index))
    pd.testing.assert_frame_equal(low, high)


def test_panic_boundary_bypasses_gate_and_holding_guard():
    returns, signal = frames(70)
    returns["SPY"] = 0.01
    returns[["GLD", "UUP"]] = -0.001
    signal["SPY"] = 1
    scores = pd.Series(0.0, index=returns.index)
    scores.iloc[-1] = 0.65
    weights = core.build_engine_weights("full", returns, signal, scores,
                                        pd.Series(12.0, index=returns.index), True, True)
    assert weights.iloc[-1]["SPY"] == 0
    assert weights.iloc[-1]["UUP"] == pytest.approx(0.5)
    assert weights.iloc[-1]["GLD"] == pytest.approx(0.3)
    assert weights.abs().sum(axis=1).max() <= 1.30 + 1e-12


def test_post_sizing_caps_do_not_reexpand_clipped_positions():
    w = pd.Series({"SPY": 0.99, "QQQ": 0.01})
    assert core.constrain_weights(w, 1.3).max() == 0.30
    assert core.constrain_weights(w, 0.2).abs().sum() <= 0.2 + 1e-12


def test_holding_guard_counts_sessions_and_blocks_sign_reversal():
    calendar = pd.bdate_range("2022-01-07", periods=7)
    current = pd.Series({"SPY": 0.2})
    target = pd.Series({"SPY": -0.2})
    entry = {"SPY": calendar[0]}
    early = core.enforce_minimum_hold(target, current, calendar[3], entry, 5, calendar)
    later = core.enforce_minimum_hold(target, current, calendar[5], entry, 5, calendar)
    assert early.SPY == 0.2
    assert later.SPY == -0.2


def test_future_changes_do_not_change_past_features_or_crisis():
    panel = core.load_panel(Path.cwd())
    cutoff = pd.Timestamp("2023-06-01")
    before = core.add_assessment_features(panel)
    changed = panel.copy()
    future = changed.Date > cutoff
    changed.loc[future, ["returns_1d", "Volume", "VIX"]] *= 1.5
    after = core.add_assessment_features(changed)
    pd.testing.assert_frame_equal(before.loc[before.Date <= cutoff, core.FEATURES],
                                  after.loc[after.Date <= cutoff, core.FEATURES])
    dates = pd.DatetimeIndex([cutoff])
    pd.testing.assert_series_equal(core.calculate_crisis_scores(before, dates), core.calculate_crisis_scores(after, dates))


def test_snapshot_checksum_and_fixed_dates():
    path = Path("assessment/inputs/panel.parquet")
    provenance = json.loads(path.with_name("provenance.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["sha256"]
    dates = core.evaluation_returns(core.load_panel(Path.cwd())).index
    assert dates.min() == pd.Timestamp("2022-01-03")
    assert dates.max() == core.TEST_END_DATE


def test_submission_includes_every_manuscript_figure():
    import re
    from alpha_trinity_assessment.submission import PAPER_FIGURES

    paper = Path("assessment_run/writing/abgegebenes_PAPER.tex").read_text()
    referenced = set(re.findall(r"\\resultfigure\{([^}]+)\}", paper))
    assert referenced == set(PAPER_FIGURES)
    for name in referenced:
        assert (Path("assessment/outputs/figures") / name).is_file()
        assert (Path("assessment_run/writing/figures") / name).is_file()


def test_training_scaler_and_labels_exclude_future_rows(monkeypatch):
    panel = core.add_assessment_features(core.load_panel(Path.cwd()))
    eligible = panel[(panel.Date < core.TRAIN_END_DATE)
                     & (panel.target_20d_end_date <= core.TRAIN_END_DATE)
                     & panel.target_20d_forward.notna()]
    records = []

    class RecordingModel:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            records.append((x.copy(), y.copy()))
            return self

        def predict(self, x):
            return np.zeros(len(x))

    monkeypatch.setattr(core, "MLPRegressor", RecordingModel)
    monkeypatch.setattr(core, "RandomForestRegressor", RecordingModel)
    panel.loc[panel.Date >= pd.Timestamp("2022-01-01"), core.FEATURES] = 1e6
    core.train_ml_signal(panel)
    for x, y in records:
        assert len(x) == len(eligible)
        np.testing.assert_allclose(x.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(y, eligible.target_20d_forward)


def test_actual_twenty_day_target_matches_same_asset_close():
    df = core.add_assessment_features(core.load_panel(Path.cwd()))
    expected = df.groupby("symbol").Close.shift(-20) / df.Close - 1
    np.testing.assert_allclose(df.target_20d_forward, expected, atol=1e-12, equal_nan=True)


def test_sortino_uses_rms_shortfall_to_same_reference():
    returns, targets = frames()
    targets["SPY"] = 1
    returns.iloc[2:, 0] = [-0.10, 0.20, -0.05, 0.01]
    result = core.evaluate_weights("risk", targets, returns, transaction_cost=0)
    row = core.metric_row(result, "test")
    shortfall = np.minimum(result.net_returns - 0.04 / 252, 0)
    downside = np.sqrt(np.mean(shortfall ** 2) * 252)
    assert row["downside_deviation_4rf"] == pytest.approx(downside)
    assert row["sortino_4rf"] == pytest.approx((result.net_returns.mean() * 252 - 0.04) / downside)


def test_supplemental_shy_checksum_and_calendar():
    path = Path("assessment/inputs/shy_adjusted_close.csv")
    provenance = json.loads(path.with_name("shy_provenance.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["sha256"]
    close = pd.read_csv(path, index_col=0, parse_dates=True)["SHY"]
    dates = core.evaluation_returns(core.load_panel(Path.cwd())).index
    assert not close.pct_change(fill_method=None).reindex(dates).isna().any()


def test_effective_holding_episodes_split_sign_changes():
    from alpha_trinity_assessment.completion import holding_period_runs
    weights = pd.DataFrame({"SPY": [0, .2, .2, -.2, -.2, 0]}, index=pd.bdate_range("2022-01-03", periods=6))
    runs = holding_period_runs(weights)
    assert runs.holding_days.tolist() == [2, 2]
    assert not runs.right_censored.any()
