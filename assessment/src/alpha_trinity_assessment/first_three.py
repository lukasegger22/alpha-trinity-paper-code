from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "alpha_trinity_mpl"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


PROTOCOL = json.loads((Path(__file__).resolve().parents[2] / "config/protocol.json").read_text())
TRAIN_END_DATE = pd.Timestamp(PROTOCOL["train_end"])
TEST_START_DATE = pd.Timestamp(PROTOCOL["test_start"])
TEST_END_DATE = pd.Timestamp(PROTOCOL["test_end"])
TRANSACTION_COST = PROTOCOL["transaction_cost"]
COST_OF_CARRY = PROTOCOL["annual_gross_carry"]
PERIODS_PER_YEAR = PROTOCOL["sessions_per_year"]
TOP_K = PROTOCOL["top_k"]
MOMENTUM_LOOKBACK = PROTOCOL["momentum_lookback"]
TARGET_VOL = PROTOCOL["volatility_target"]
MAX_VOL_TARGET_EXPOSURE = PROTOCOL["leverage_cap"]
MIN_POSITION_SIZE = PROTOCOL["intermediate_position_filter"]
MIN_HOLD_DAYS = PROTOCOL["exit_guard_sessions"]
ENSEMBLE_SIZE = PROTOCOL["ensemble_size"]
REBALANCE_SPEED = PROTOCOL["growth_adjustment_speed"]

FEATURES = [
    "returns_1d",
    "returns_5d",
    "volatility_60d",
    "VIX",
    "obv_trend",
    "dist_vwap",
    "mfi",
    "bb_pos",
]

DEFENSIVE_ASSETS = ["UUP", "GLD", "AGG", "TLT", "SHY"]
GROWTH_AI_ASSETS = ["NVDA", "MSFT", "META", "ASML"]


@dataclass(frozen=True)
class BacktestResult:
    name: str
    net_returns: pd.Series
    gross_returns: pd.Series
    trading_cost: pd.Series
    carry_cost: pd.Series
    turnover: pd.Series
    effective_weights: pd.DataFrame
    signal_weights: pd.DataFrame
    trades: pd.DataFrame | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the first three assessment upgrades: research question, baselines, and ablations."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("assessment"))
    return parser.parse_args()


def ensure_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    return pd.to_datetime(index).tz_localize(None) if getattr(index, "tz", None) else pd.to_datetime(index)


def load_panel(project_root: Path) -> pd.DataFrame:
    panel_path = project_root / "assessment" / "inputs" / "panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing feature panel: {panel_path}")

    panel = pd.read_parquet(panel_path).reset_index()
    if "date" in panel.columns:
        panel = panel.rename(columns={"date": "Date"})
    if "index" in panel.columns and "Date" not in panel.columns:
        panel = panel.rename(columns={"index": "Date"})
    if "Symbol" in panel.columns and "symbol" not in panel.columns:
        panel = panel.rename(columns={"Symbol": "symbol"})
    if "Date" not in panel.columns or "symbol" not in panel.columns:
        raise ValueError(f"Panel must contain Date/date and symbol columns. Columns: {panel.columns.tolist()}")

    panel["Date"] = pd.to_datetime(panel["Date"]).dt.tz_localize(None)
    panel = panel[panel["Date"] <= TEST_END_DATE].copy()
    if "vix_close" in panel.columns and ("VIX" not in panel.columns or panel["VIX"].nunique(dropna=True) <= 1):
        panel["VIX"] = panel["vix_close"]
    elif "VIX" not in panel.columns:
        panel["VIX"] = 15.0
    panel = panel.drop_duplicates(subset=["Date", "symbol"], keep="last")
    return panel.sort_values(["symbol", "Date"]).reset_index(drop=True)


def load_saved_trinity_weights(project_root: Path) -> pd.DataFrame:
    weights_path = project_root / "data" / "features" / "trinity_history.parquet"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing saved Trinity weights: {weights_path}")

    weights = pd.read_parquet(weights_path)
    weights.index = ensure_datetime_index(weights.index)
    weights = weights.sort_index()
    weights.columns = weights.columns.astype(str)
    return weights


def pivot_returns(panel: pd.DataFrame) -> pd.DataFrame:
    returns = panel.pivot(index="Date", columns="symbol", values="returns_1d").sort_index()
    returns.index = ensure_datetime_index(returns.index)
    returns.columns = returns.columns.astype(str)
    if returns.isna().any().any():
        raise ValueError("The frozen panel must have a complete shared return calendar.")
    return returns


def evaluation_returns(panel: pd.DataFrame) -> pd.DataFrame:
    returns = pivot_returns(panel)
    return returns.loc[TEST_START_DATE:TEST_END_DATE]


def calculate_efficiency_ratio(series: pd.Series, window: int = 20) -> pd.Series:
    direction = series.diff(window).abs()
    volatility = series.diff().abs().rolling(window).sum().replace(0, 0.001)
    return direction / volatility


def calculate_bb_position(series: pd.Series, window: int = 20) -> pd.Series:
    moving_avg = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = moving_avg + (2 * std)
    lower = moving_avg - (2 * std)
    bandwidth = (upper - lower).replace(0, 0.001)
    return (series - lower) / bandwidth


def add_assessment_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy().sort_values(["symbol", "Date"]).reset_index(drop=True)
    if "Volume" not in df.columns:
        df["Volume"] = 1.0

    df["returns_1d"] = df["returns_1d"].fillna(0.0)
    df["proxy_price"] = df.groupby("symbol")["returns_1d"].transform(lambda x: (1.0 + x).cumprod())
    df["efficiency"] = (
        df.groupby("symbol")["proxy_price"].transform(lambda x: calculate_efficiency_ratio(x)).fillna(0.5)
    )

    direction = np.sign(df["returns_1d"].fillna(0.0))
    df["obv"] = (df["Volume"].fillna(1.0) * direction).groupby(df["symbol"]).cumsum()
    df["obv_trend"] = df.groupby("symbol")["obv"].pct_change(20).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["bb_pos"] = df.groupby("symbol")["proxy_price"].transform(lambda x: calculate_bb_position(x)).fillna(0.5)

    mfi_values = []
    vwap_values = []
    for _, group in df.groupby("symbol", sort=False):
        typical_price = group["proxy_price"]
        volume = group["Volume"].fillna(1.0)
        money_flow = typical_price * volume
        delta = typical_price.diff()
        pos_flow = pd.Series(np.where(delta > 0, money_flow, 0.0), index=group.index)
        neg_flow = pd.Series(np.where(delta < 0, money_flow, 0.0), index=group.index)
        rolling_pos = pos_flow.rolling(14).sum()
        rolling_neg = neg_flow.rolling(14).sum().replace(0, 0.001)
        mfi_ratio = rolling_pos / rolling_neg
        mfi_values.append((100.0 - (100.0 / (1.0 + mfi_ratio))).fillna(50.0) / 100.0)

        vwap = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        vwap_values.append(vwap)

    df["mfi"] = pd.concat(mfi_values).sort_index()
    df["vwap_20"] = pd.concat(vwap_values).sort_index()
    df["dist_vwap"] = ((df["proxy_price"] / df["vwap_20"]) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    df["target_20d_forward"] = df.groupby("symbol")["returns_20d"].shift(-20)
    df["target_20d_end_date"] = df.groupby("symbol")["Date"].shift(-20)
    df = df.replace([np.inf, -np.inf], np.nan)
    df[FEATURES] = df[FEATURES].fillna(0.0)
    return df


def train_ml_signal(feature_df: pd.DataFrame, train_end: pd.Timestamp = TRAIN_END_DATE) -> pd.DataFrame:
    train_df = feature_df[
        (feature_df["Date"] < train_end)
        & (feature_df["target_20d_end_date"] <= train_end)
        & feature_df["target_20d_forward"].notna()
    ].copy()
    if train_df.empty:
        raise ValueError("Training set is empty. Check train_end and panel date range.")

    x_train = train_df[FEATURES].to_numpy()
    y_train = train_df["target_20d_forward"].to_numpy()

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_all_scaled = scaler.transform(feature_df[FEATURES].to_numpy())

    predictions = np.zeros((len(feature_df), ENSEMBLE_SIZE))
    for i in range(ENSEMBLE_SIZE):
        if i < ENSEMBLE_SIZE // 2:
            model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42 + i)
        else:
            model = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42 + i, n_jobs=-1)
        model.fit(x_train_scaled, y_train)
        predictions[:, i] = model.predict(x_all_scaled)

    scored = feature_df[["Date", "symbol"]].copy()
    scored["raw_signal"] = predictions.mean(axis=1)
    scored["ensemble_signal"] = (
        scored.groupby("symbol")["raw_signal"].transform(lambda x: x.ewm(span=5, adjust=False).mean()) * 50.0
    )
    signal = scored.pivot(index="Date", columns="symbol", values="ensemble_signal").sort_index()
    signal.index = ensure_datetime_index(signal.index)
    signal.columns = signal.columns.astype(str)
    return signal.fillna(0.0)


def calculate_crisis_scores(feature_df: pd.DataFrame, dates: pd.Index) -> pd.Series:
    spy = feature_df[feature_df["symbol"] == "SPY"].copy()
    if spy.empty:
        spy = feature_df[feature_df["symbol"] == "QQQ"].copy()
    spy = spy.set_index("Date").sort_index()
    vix = feature_df.groupby("Date")["VIX"].mean().sort_index()
    scores = {}

    for current_date in dates:
        if current_date not in spy.index:
            scores[current_date] = 0.0
            continue
        spy_current = spy.loc[:current_date]
        if len(spy_current) < 200:
            scores[current_date] = 0.0
            continue

        price = spy_current["proxy_price"].iloc[-1]
        sma_50 = spy_current["proxy_price"].rolling(50).mean().iloc[-1]
        sma_200 = spy_current["proxy_price"].rolling(200).mean().iloc[-1]

        trend_signal = 0.0
        if price < sma_50:
            trend_signal += 0.5
        if sma_50 < sma_200:
            trend_signal = 1.0

        curr_vix = vix.asof(current_date)
        if pd.isna(curr_vix):
            curr_vix = 20.0
        if curr_vix > 30:
            vix_signal = 1.0
        elif curr_vix > 20:
            vix_signal = 0.5
        else:
            vix_signal = 0.0

        if len(spy_current) > 252:
            peak = spy_current["proxy_price"].rolling(252).max().iloc[-1]
            drawdown_signal = min((peak - price) / peak / 0.15, 1.0)
        else:
            drawdown_signal = 0.0

        vix_series = vix.loc[:current_date].tail(6)
        if len(vix_series) >= 6:
            vix_change = (vix_series.iloc[-1] / vix_series.iloc[0]) - 1.0
            vix_acceleration = min(max(vix_change * 2.0, 0.0), 1.0)
        else:
            vix_acceleration = 0.0

        if len(spy_current) > 60:
            mom_20 = (spy_current["proxy_price"].iloc[-1] / spy_current["proxy_price"].iloc[-21]) - 1.0
            mom_60 = (spy_current["proxy_price"].iloc[-1] / spy_current["proxy_price"].iloc[-61]) - 1.0
            momentum_collapse = 1.0 if mom_20 < mom_60 * 0.5 else 0.0
        else:
            momentum_collapse = 0.0

        score = (
            trend_signal * 0.25
            + vix_signal * 0.20
            + drawdown_signal * 0.20
            + vix_acceleration * 0.15
            + momentum_collapse * 0.10
        )
        scores[current_date] = min(float(score), 1.0)

    return pd.Series(scores, name="crisis_score")


def calculate_momentum_score(returns_series: pd.Series, lookback: int = MOMENTUM_LOOKBACK) -> float:
    values = pd.Series(returns_series).dropna().tail(lookback)
    if len(values) < max(10, lookback // 3):
        return 0.0
    return 1.0 if (1.0 + values).prod() - 1.0 > 0.0 else 0.0


def calculate_kelly_positions(signals: pd.Series, past_returns: pd.DataFrame, max_kelly: float = 0.25) -> pd.Series:
    kelly_weights = pd.Series(0.0, index=signals.index)
    for symbol in signals.index:
        if symbol not in past_returns.columns:
            continue
        symbol_rets = past_returns[symbol].dropna().tail(PERIODS_PER_YEAR)
        if len(symbol_rets) < 60:
            continue
        wins = symbol_rets[symbol_rets > 0]
        losses = symbol_rets[symbol_rets < 0]
        if len(wins) == 0 or len(losses) == 0:
            continue
        win_rate = len(wins) / len(symbol_rets)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        if avg_loss == 0:
            continue
        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = (win_rate * win_loss_ratio - (1.0 - win_rate)) / win_loss_ratio
        kelly_fraction = max(0.0, min(kelly_fraction * 0.5, max_kelly))
        direction = 1.0 if signals[symbol] > 0.0 else -1.0
        kelly_weights[symbol] = kelly_fraction * direction * abs(signals[symbol])
    return kelly_weights


def enforce_minimum_hold(
    target_weights: pd.Series,
    current_weights: pd.Series,
    current_date: pd.Timestamp,
    entry_dates: dict[str, pd.Timestamp],
    min_hold_days: int = MIN_HOLD_DAYS,
    calendar: pd.Index | None = None,
) -> pd.Series:
    adjusted = target_weights.copy()
    for symbol in current_weights.index:
        closing = abs(target_weights.get(symbol, 0.0)) < 0.01
        reversing = current_weights[symbol] * target_weights.get(symbol, 0.0) < 0
        if abs(current_weights[symbol]) > 0.01 and (closing or reversing):
            if symbol in entry_dates:
                if calendar is None:
                    raise ValueError("A trading calendar is required for the holding guard.")
                days_held = calendar.get_loc(current_date) - calendar.get_loc(entry_dates[symbol])
                if days_held < min_hold_days:
                    adjusted[symbol] = current_weights[symbol]

    return adjusted


def apply_transaction_cost_gate(
    current_weights: pd.Series,
    target_weights: pd.Series,
    expected_returns_proxy: pd.Series,
    transaction_cost: float = TRANSACTION_COST,
) -> pd.Series:
    turnover = (target_weights - current_weights).abs().sum()
    cost = turnover * transaction_cost
    expected_benefit = 0.0
    for symbol in target_weights.index:
        weight_diff = target_weights[symbol] - current_weights.get(symbol, 0.0)
        expected_benefit += weight_diff * expected_returns_proxy.get(symbol, 0.0)
    return current_weights if expected_benefit < cost * 2.0 else target_weights


def growth_base_leverage(vix_value: float, leverage_cap: float = MAX_VOL_TARGET_EXPOSURE) -> float:
    if pd.isna(vix_value):
        vix_value = 20.0
    if vix_value < 15:
        return min(1.30, leverage_cap)
    if vix_value < 20:
        return min(1.10, leverage_cap)
    if vix_value < 25:
        return min(1.00, leverage_cap)
    return min(0.70, leverage_cap)


def normalize_gross(weights: pd.Series) -> pd.Series:
    gross = weights.abs().sum()
    return weights / gross if gross > 0 else weights


def constrain_weights(weights: pd.Series, gross_cap: float, position_cap: float = 0.30) -> pd.Series:
    """Hard post-sizing bounds; clipped capital stays in cash, never re-normalized."""
    bounded = weights.clip(-position_cap, position_cap)
    gross = bounded.abs().sum()
    return bounded * min(1.0, gross_cap / gross) if gross else bounded


def threshold_asof(threshold: float | pd.Series, current_date: pd.Timestamp, fallback: float) -> float:
    if isinstance(threshold, pd.Series):
        value = threshold.asof(current_date)
        return fallback if pd.isna(value) else float(value)
    return float(threshold)


def build_engine_weights(
    name: str,
    returns: pd.DataFrame,
    ml_signal: pd.DataFrame,
    crisis_scores: pd.Series,
    daily_vix: pd.Series,
    use_ml: bool,
    use_crisis: bool,
    caution_threshold: float | pd.Series = PROTOCOL["caution_threshold"],
    panic_threshold: float | pd.Series = PROTOCOL["panic_threshold"],
    min_hold_days: int = MIN_HOLD_DAYS,
    leverage_cap: float = MAX_VOL_TARGET_EXPOSURE,
    transaction_cost: float = TRANSACTION_COST,
    panic_assets: dict[str, float] | None = None,
    start_date: pd.Timestamp = TEST_START_DATE,
) -> pd.DataFrame:
    dates = returns.index[(returns.index >= start_date) & (returns.index <= TEST_END_DATE)]
    columns = returns.columns
    ml_signal = ml_signal.reindex(index=returns.index, columns=columns).fillna(0.0)

    weights_history = []
    current_weights = pd.Series(0.0, index=columns)
    entry_dates: dict[str, pd.Timestamp] = {}
    risk_assets = [c for c in columns if c not in DEFENSIVE_ASSETS]

    for current_date in dates:
        idx = returns.index.get_loc(current_date)
        past_returns = returns.iloc[:idx + 1]
        current_signal = ml_signal.loc[current_date].copy()
        current_vix = daily_vix.asof(current_date)
        crisis_score = float(crisis_scores.asof(current_date)) if use_crisis else 0.0
        caution_value = threshold_asof(caution_threshold, current_date, 0.35)
        panic_value = threshold_asof(panic_threshold, current_date, 0.65)
        panic_value = max(panic_value, caution_value + 0.05)

        panic = use_crisis and crisis_score >= panic_value - 1e-12
        caution = use_crisis and crisis_score >= caution_value - 1e-12
        if panic:
            target_weights = pd.Series(0.0, index=columns)
            active_panic_assets = {"UUP": 0.50, "GLD": 0.30} if panic_assets is None else panic_assets
            for symbol, weight in active_panic_assets.items():
                if symbol in target_weights.index:
                    target_weights[symbol] = weight
            used_speed = 1.0
            base_leverage = min(0.80, leverage_cap)

        elif caution:
            target_weights = pd.Series(0.0, index=columns)
            if use_ml:
                selected = current_signal[current_signal > 0.0].nlargest(TOP_K)
                if selected.sum() > 0.0:
                    target_weights[selected.index] = selected / selected.sum() * 0.40
            else:
                momentum = (1.0 + past_returns[risk_assets].tail(MOMENTUM_LOOKBACK)).prod() - 1.0
                selected = momentum.nlargest(TOP_K)
                if len(selected) > 0:
                    target_weights[selected.index] = 0.40 / len(selected)
            if "UUP" in target_weights.index:
                target_weights["UUP"] += 0.25
            used_speed = 0.30
            base_leverage = min(0.85, leverage_cap)

        else:
            target_weights = pd.Series(0.0, index=columns)
            if use_ml:
                for symbol in current_signal.index:
                    momentum_ok = calculate_momentum_score(
                        past_returns[symbol] if symbol in past_returns.columns else pd.Series(dtype=float)
                    )
                    if momentum_ok > 0 and current_signal[symbol] > 0.0:
                        current_signal[symbol] *= 1.2

                target_weights = calculate_kelly_positions(current_signal, past_returns)
                if target_weights.abs().sum() == 0.0:
                    selected = current_signal.nlargest(TOP_K)
                    selected = selected[selected > 0.0]
                    if selected.sum() > 0.0:
                        target_weights[selected.index] = selected / selected.sum()
            else:
                if risk_assets:
                    target_weights[risk_assets] = 1.0 / len(risk_assets)

            target_weights = target_weights.clip(lower=-0.30, upper=0.30)
            target_weights = normalize_gross(target_weights)
            used_speed = REBALANCE_SPEED
            base_leverage = growth_base_leverage(current_vix, leverage_cap=leverage_cap) if use_crisis else min(1.0, leverage_cap)

        target_weights = target_weights.where(target_weights.abs() >= MIN_POSITION_SIZE, 0.0)
        if panic or caution:
            target_weights = normalize_gross(target_weights)
        target_weights = target_weights * base_leverage
        if not panic:
            target_weights = enforce_minimum_hold(
                target_weights, current_weights, current_date, entry_dates,
                min_hold_days=min_hold_days, calendar=returns.index,
            )
            target_weights = apply_transaction_cost_gate(
                current_weights, target_weights, past_returns.tail(20).mean().fillna(0.0),
                transaction_cost=transaction_cost,
            )
        final_weights = current_weights * (1.0 - used_speed) + target_weights * used_speed
        # Defensive sleeves may concentrate, but the overall exposure cap is always hard.
        final_weights = constrain_weights(final_weights, min(base_leverage, leverage_cap),
                                          PROTOCOL["defensive_asset_cap"] if panic or caution else PROTOCOL["growth_asset_cap"])
        for symbol in columns:
            if abs(final_weights[symbol]) > 0.01 and (abs(current_weights[symbol]) <= 0.01 or final_weights[symbol] * current_weights[symbol] < 0):
                entry_dates[symbol] = current_date
        current_weights = final_weights.copy()
        final_weights.name = current_date
        weights_history.append(final_weights)

    weights = pd.DataFrame(weights_history)
    weights.name = name
    return weights.fillna(0.0)


def static_weights(index: pd.Index, columns: pd.Index, allocations: dict[str, float]) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=index, columns=columns)
    for symbol, weight in allocations.items():
        if symbol in weights.columns:
            weights[symbol] = weight
    return weights


def equal_weight_weights(index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=index, columns=columns)
    if len(columns) > 0:
        weights.loc[:, :] = 1.0 / len(columns)
    return weights


def momentum_top_k_weights(returns: pd.DataFrame, lookback: int = MOMENTUM_LOOKBACK, top_k: int = TOP_K) -> pd.DataFrame:
    momentum = (1.0 + returns).rolling(lookback).apply(np.prod, raw=True) - 1.0
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    for current_date, row in momentum.iterrows():
        selected = row.dropna().nlargest(top_k)
        if len(selected) > 0:
            weights.loc[current_date, selected.index] = 1.0 / len(selected)
    return weights.fillna(0.0)


def volatility_targeting_spy_weights(returns: pd.DataFrame) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    if "SPY" not in returns.columns:
        return weights
    realized_vol = returns["SPY"].rolling(MOMENTUM_LOOKBACK).std() * np.sqrt(PERIODS_PER_YEAR)
    exposure = (TARGET_VOL / realized_vol).clip(lower=0.0, upper=MAX_VOL_TARGET_EXPOSURE).fillna(1.0)
    weights["SPY"] = exposure
    return weights


def risk_parity_weights(returns: pd.DataFrame) -> pd.DataFrame:
    realized_vol = returns.rolling(MOMENTUM_LOOKBACK).std().replace(0.0, np.nan)
    inv_vol = 1.0 / realized_vol
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    fallback = 1.0 / len(returns.columns)
    return weights.fillna(fallback)


def align_weights(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    common_dates = weights.index.intersection(returns.index)
    common_cols = weights.columns.intersection(returns.columns)
    return weights.loc[common_dates, common_cols].reindex(columns=returns.columns, fill_value=0.0)


def evaluate_weights(
    name: str,
    signal_weights: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_cost: float = TRANSACTION_COST,
    cost_of_carry: float = COST_OF_CARRY,
    rebalance: str = "daily",
) -> BacktestResult:
    signal_weights = align_weights(signal_weights.sort_index(), returns.sort_index()).fillna(0.0)
    returns = returns.loc[signal_weights.index, signal_weights.columns].fillna(0.0)

    if rebalance not in {"daily", "buy_and_hold"}:
        raise ValueError("Unknown rebalance schedule")
    values = returns.to_numpy()
    targets = signal_weights.to_numpy()
    n, m = values.shape
    held = np.zeros(m)
    effective = np.zeros((n, m))
    trades = np.zeros((n, m))
    gross_values, carry_values, fee_values, net_values = (np.zeros(n) for _ in range(4))
    for i in range(n):
        effective[i] = held
        gross_values[i] = held @ values[i]
        carry_values[i] = max(np.abs(held).sum() - 1.0, 0.0) * cost_of_carry / PERIODS_PER_YEAR
        pre_nav = 1.0 + gross_values[i] - carry_values[i]
        assets = held * (1.0 + values[i])
        if pre_nav <= 0:
            raise ValueError("Portfolio insolvent; cannot normalize holdings")
        # Signal at close i-1 fills at close i, after today's existing holdings earn returns.
        if i >= 1 and (rebalance == "daily" or i == 1):
            target = targets[i - 1]
            fee = 0.0
            for _ in range(30):
                new_fee = transaction_cost * np.abs(target * (pre_nav - fee) - assets).sum()
                if abs(new_fee - fee) < 1e-14:
                    fee = new_fee
                    break
                fee = new_fee
            post_nav = pre_nav - fee
            if post_nav <= 0:
                raise ValueError("Transaction costs exhaust portfolio")
            trades[i] = target * post_nav - assets
            fee_values[i] = transaction_cost * np.abs(trades[i]).sum()
            held = target.copy()
        else:
            held = assets / pre_nav
        net_values[i] = pre_nav - fee_values[i] - 1.0
    effective_weights = pd.DataFrame(effective, index=returns.index, columns=returns.columns)
    trade_frame = pd.DataFrame(trades, index=returns.index, columns=returns.columns)
    gross_returns = pd.Series(gross_values, index=returns.index)
    turnover = trade_frame.abs().sum(axis=1)
    trading_cost = pd.Series(fee_values, index=returns.index)
    carry_cost = pd.Series(carry_values, index=returns.index)
    net_returns = pd.Series(net_values, index=returns.index)

    return BacktestResult(
        name=name,
        net_returns=net_returns,
        gross_returns=gross_returns,
        trading_cost=trading_cost,
        carry_cost=carry_cost,
        turnover=turnover,
        effective_weights=effective_weights,
        signal_weights=signal_weights,
        trades=trade_frame,
    )


def scale_to_no_leverage(weights: pd.DataFrame) -> pd.DataFrame:
    gross = weights.abs().sum(axis=1)
    scale = np.where(gross > 1.0, 1.0 / gross, 1.0)
    return weights.mul(scale, axis=0)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    return float((equity / equity.cummax().clip(lower=1.0) - 1.0).min())


def cagr(net_returns: pd.Series) -> float:
    clean = pd.Series(net_returns).dropna()
    if clean.empty:
        return float("nan")
    equity = (1.0 + clean).cumprod()
    years = len(clean) / PERIODS_PER_YEAR
    return float(equity.iloc[-1] ** (1.0 / years) - 1.0)


def metric_row(result: BacktestResult, group: str) -> dict[str, object]:
    net = result.net_returns.fillna(0.0)
    gross = result.gross_returns.fillna(0.0)
    net_equity = (1.0 + net).cumprod()
    gross_equity = (1.0 + gross).cumprod()
    ann_return = net.mean() * PERIODS_PER_YEAR
    ann_vol = net.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
    downside = np.sqrt(np.mean(np.minimum(net, 0.0) ** 2) * PERIODS_PER_YEAR)
    downside_4rf = np.sqrt(np.mean(np.minimum(net - 0.04 / PERIODS_PER_YEAR, 0.0) ** 2) * PERIODS_PER_YEAR)
    sharpe_0rf = ann_return / ann_vol if ann_vol > 0 else np.nan
    sharpe_4rf = (ann_return - 0.04) / ann_vol if ann_vol > 0 else np.nan
    sortino_0rf = ann_return / downside if downside > 0 else np.nan
    sortino_4rf = (ann_return - 0.04) / downside_4rf if downside_4rf > 0 else np.nan
    max_dd = max_drawdown(net_equity)
    total_return = float(net_equity.iloc[-1] - 1.0) if not net_equity.empty else np.nan
    gross_return = float(gross_equity.iloc[-1] - 1.0) if not gross_equity.empty else np.nan
    total_cost = float((result.trading_cost + result.carry_cost).sum())
    trade_days = int((result.turnover > 1e-8).sum())
    weight_changes = result.trades.abs() if result.trades is not None else result.signal_weights.diff().abs().fillna(result.signal_weights.abs())
    trade_events = int((weight_changes > 1e-6).sum().sum())
    gross_exposure = result.effective_weights.abs().sum(axis=1)
    net_exposure = result.effective_weights.sum(axis=1)
    avg_positions = float((result.effective_weights.abs() > 0.01).sum(axis=1).mean())

    return {
        "group": group,
        "strategy": result.name,
        "start": net.index.min().date().isoformat(),
        "end": net.index.max().date().isoformat(),
        "trading_days": int(len(net)),
        "gross_return": gross_return,
        "net_return": total_return,
        "cost_drag_compounded": gross_return - total_return,
        "total_cost_sum": total_cost,
        "cagr": cagr(net),
        "annualized_return_arithmetic": float(ann_return),
        "annualized_volatility": float(ann_vol),
        "downside_deviation_0rf": float(downside),
        "downside_deviation_4rf": float(downside_4rf),
        "avg_monthly_turnover": float(result.turnover.resample("ME").sum().mean()),
        "annualized_cost_sum": total_cost * PERIODS_PER_YEAR / len(net),
        "sharpe_0rf": float(sharpe_0rf),
        "sharpe_4rf": float(sharpe_4rf),
        "sortino_0rf": float(sortino_0rf),
        "sortino_4rf": float(sortino_4rf),
        "calmar": float(cagr(net) / abs(max_dd)) if max_dd < 0 else np.nan,
        "max_drawdown": max_dd,
        "total_turnover": float(result.turnover.sum()),
        "avg_daily_turnover": float(result.turnover.mean()),
        "annualized_turnover": float(result.turnover.mean() * PERIODS_PER_YEAR),
        "trade_days": trade_days,
        "trade_events": trade_events,
        "avg_gross_exposure": float(gross_exposure.mean()),
        "max_gross_exposure": float(gross_exposure.max()),
        "avg_net_exposure": float(net_exposure.mean()),
        "avg_positions_over_1pct": avg_positions,
    }


def combine_curves(results: list[BacktestResult], attr: str) -> pd.DataFrame:
    curves = {}
    for result in results:
        returns = getattr(result, attr).fillna(0.0)
        curves[result.name] = (1.0 + returns).cumprod()
    return pd.DataFrame(curves)


def plot_curves(curves: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for column in curves.columns:
        width = 2.4 if "Alpha Trinity" in column or "Full" in column else 1.5
        ax.plot(curves.index, curves[column], label=column, linewidth=width)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.30)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_crisis_scores(crisis_scores: pd.Series, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(crisis_scores.index, crisis_scores, color="#d62728", linewidth=1.6)
    ax.axhline(0.35, color="#ffbf00", linestyle="--", linewidth=1.0, label="Caution threshold")
    ax.axhline(0.65, color="#b00020", linestyle="--", linewidth=1.0, label="Panic threshold")
    ax.fill_between(crisis_scores.index, 0.65, 1.0, color="#b00020", alpha=0.08)
    ax.set_title("Crisis Score Used for Assessment Ablations")
    ax.set_ylabel("Crisis score")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.30)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def fmt_pct(value: float) -> str:
    return "nan" if pd.isna(value) else f"{value:.2%}"


def fmt_num(value: float) -> str:
    return "nan" if pd.isna(value) else f"{value:.2f}"


def write_research_question(output_dir: Path) -> None:
    text = """# Research Question

## Final Main Question

Does an interpretable deterministic crisis-control layer improve a cost-aware MLP/Random-Forest multi-asset allocation strategy under non-stationary market regimes?

## Testable Version

Compared with a prediction-only MLP/RF allocation and simple multi-asset benchmarks, does adding the deterministic crisis-control layer improve out-of-sample risk-adjusted performance after realistic trading frictions?

## Hypotheses

H1: The full Alpha Trinity system achieves a higher out-of-sample Sharpe ratio than the prediction-only MLP/RF ablation after transaction and leverage costs.

H2: The full Alpha Trinity system reduces drawdown depth or drawdown recovery burden relative to prediction-only allocation and broad passive benchmarks.

H3: The full Alpha Trinity system remains economically meaningful after a fixed 10 bps turnover cost, 6 percent annual leverage carry, and a one-trading-day execution lag.

## Scope

The project is not claiming a universally robust trading system. The defensible claim is narrower: this is a transparent proof-of-concept for combining return prediction, crisis detection, and cost-aware execution in one reproducible out-of-sample protocol.

## Primary Comparisons

The main comparison is Full Alpha Trinity versus Prediction-only. The secondary comparisons are Full Alpha Trinity versus Crisis-only, no-cost, no-leverage, SPY, 60/40, equal-weight universe, momentum Top-K, volatility targeting, and risk parity.

## Evaluation Criteria

The main criteria are net return, Sharpe ratio, maximum drawdown, Calmar ratio, turnover, total cost drag, gross exposure, and trade frequency. Portfolio metrics are reported separately from model-design claims so the report does not confuse forecasting skill with portfolio construction effects.
"""
    (output_dir / "research_question.md").write_text(text, encoding="utf-8")


def write_summary(
    output_dir: Path,
    baseline_table: pd.DataFrame,
    ablation_table: pd.DataFrame,
    assumptions: dict[str, object],
) -> None:
    baseline_view = baseline_table[
        ["strategy", "net_return", "sharpe_4rf", "max_drawdown", "total_turnover", "avg_gross_exposure"]
    ].copy()
    ablation_view = ablation_table[
        [
            "strategy",
            "net_return",
            "sharpe_4rf",
            "max_drawdown",
            "cost_drag_compounded",
            "total_turnover",
            "avg_gross_exposure",
        ]
    ].copy()

    def baseline_markdown_table(df: pd.DataFrame) -> str:
        lines = [
            "| Strategy | Net Return | Sharpe 4% RF | Max Drawdown | Total Turnover | Avg Gross Exposure |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, row in df.iterrows():
            lines.append(
                f"| {row['strategy']} | {fmt_pct(row['net_return'])} | {fmt_num(row['sharpe_4rf'])} | "
                f"{fmt_pct(row['max_drawdown'])} | {fmt_num(row['total_turnover'])} | "
                f"{fmt_num(row['avg_gross_exposure'])} |"
            )
        return "\n".join(lines)

    def ablation_markdown_table(df: pd.DataFrame) -> str:
        lines = [
            "| Strategy | Net Return | Sharpe 4% RF | Max Drawdown | Cost Drag | Total Turnover | Avg Gross Exposure |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, row in df.iterrows():
            lines.append(
                f"| {row['strategy']} | {fmt_pct(row['net_return'])} | {fmt_num(row['sharpe_4rf'])} | "
                f"{fmt_pct(row['max_drawdown'])} | {fmt_pct(row['cost_drag_compounded'])} | "
                f"{fmt_num(row['total_turnover'])} | {fmt_num(row['avg_gross_exposure'])} |"
            )
        return "\n".join(lines)

    summary = f"""# Assessment Step 1-3 Summary

Generated: {datetime.now().isoformat(timespec="seconds")}

## 1. Sharpened Research Question

Does an interpretable deterministic crisis-control layer improve a cost-aware MLP/Random-Forest multi-asset allocation strategy under non-stationary market regimes?

The testable version compares Full Alpha Trinity with prediction-only, crisis-only, and simple passive/tactical baselines after the same execution lag and cost assumptions.

## 2. Baseline Comparison

{baseline_markdown_table(baseline_view)}

## 3. Ablation Study

{ablation_markdown_table(ablation_view)}

## Protocol Assumptions

- Data source in this run: `{assumptions['panel_path']}`.
- Saved Trinity weights: `{assumptions['weights_path']}`.
- Evaluation period: {assumptions['start']} to {assumptions['end']}.
- Signal/execution timing: signal close t; fill close t+1; first return ends t+2.
- Transaction cost: {TRANSACTION_COST:.4f} per one-way turnover dollar.
- Leverage carry: {COST_OF_CARRY:.2%} per year on gross exposure above 1.0.
- ML training: rows before {TRAIN_END_DATE.date()} only; fixed seeds 42-51; MLP/RF ensemble size {ENSEMBLE_SIZE}.
- Prediction target in the assessment script: group-wise 20-day forward return from each asset's own history.
- Crisis thresholds: Growth below 0.35, Caution from 0.35 to 0.65, Panic above 0.65.

## Important Implementation Note

The current feature panel contains both `VIX` and `vix_close`. The production engine and this assessment script map `vix_close` into `VIX` when the stored `VIX` column is missing or constant, so the assessment path uses dynamic VIX information. In this run `VIX` has {assumptions['vix_unique_values']} unique value(s), while `vix_close` has {assumptions['vix_close_unique_values']} unique value(s).
"""
    (output_dir / "first_three_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    assessment_dir = (project_root / args.output_dir).resolve()
    docs_dir = assessment_dir / "docs"
    tables_dir = assessment_dir / "outputs" / "tables"
    figures_dir = assessment_dir / "outputs" / "figures"
    manifests_dir = assessment_dir / "outputs" / "manifests"
    docs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    print("Loading frozen assessment panel...")
    panel = load_panel(project_root)
    returns = pivot_returns(panel)
    returns_eval = evaluation_returns(panel)
    common_index = returns_eval.index

    print("Engineering assessment features and training MLP/RF signal...")
    feature_df = add_assessment_features(panel)
    ml_signal = train_ml_signal(feature_df)
    daily_vix = feature_df.groupby("Date")["VIX"].mean().sort_index()
    crisis_scores = calculate_crisis_scores(feature_df, returns.index)
    crisis_scores = crisis_scores.loc[common_index]

    print("Building baseline portfolios...")
    baseline_weights = {
        "SPY buy-and-hold": static_weights(common_index, returns_eval.columns, {"SPY": 1.0}),
        "60/40 SPY/AGG": static_weights(common_index, returns_eval.columns, {"SPY": 0.60, "AGG": 0.40}),
        "Equal-weight universe": equal_weight_weights(common_index, returns_eval.columns),
        "Momentum Top-5": momentum_top_k_weights(returns).loc[common_index],
        "Volatility-targeted SPY": volatility_targeting_spy_weights(returns).loc[common_index],
        "Risk parity inverse-vol": risk_parity_weights(returns).loc[common_index],
        "Defensive basket buy-and-hold": static_weights(
            common_index,
            returns_eval.columns,
            {"UUP": 0.25, "GLD": 0.25, "AGG": 0.25, "TLT": 0.25},
        ),
    }

    baseline_results = [
        evaluate_weights(name, weights, returns_eval, rebalance="buy_and_hold" if "buy-and-hold" in name else "daily") for name, weights in baseline_weights.items()
    ]

    print("Building ablation portfolios...")
    rebuilt_full_weights = build_engine_weights(
        "Full Alpha Trinity rebuilt", returns, ml_signal, crisis_scores, daily_vix, use_ml=True, use_crisis=True
    )
    prediction_only_weights = build_engine_weights(
        "Prediction-only MLP/RF", returns, ml_signal, crisis_scores, daily_vix, use_ml=True, use_crisis=False
    )
    crisis_only_weights = build_engine_weights(
        "Crisis-only deterministic", returns, ml_signal, crisis_scores, daily_vix, use_ml=False, use_crisis=True
    )
    no_leverage_weights = scale_to_no_leverage(rebuilt_full_weights)

    ablation_results = [
        evaluate_weights("Full Alpha Trinity rebuilt", rebuilt_full_weights, returns_eval),
        evaluate_weights("Prediction-only MLP/RF", prediction_only_weights, returns_eval),
        evaluate_weights("Crisis-only deterministic", crisis_only_weights, returns_eval),
        evaluate_weights(
            "Full Alpha Trinity rebuilt no-cost",
            rebuilt_full_weights,
            returns_eval,
            transaction_cost=0.0,
            cost_of_carry=0.0,
        ),
        evaluate_weights("Full Alpha Trinity no-leverage", no_leverage_weights, returns_eval),
    ]

    print("Writing tables, curves, and plots...")
    baseline_table = pd.DataFrame([metric_row(result, "baseline") for result in baseline_results])
    ablation_table = pd.DataFrame([metric_row(result, "ablation") for result in ablation_results])
    all_results = baseline_results + ablation_results

    baseline_table.to_csv(tables_dir / "baseline_comparison.csv", index=False)
    ablation_table.to_csv(tables_dir / "ablation_study.csv", index=False)
    benchmark_curve_results = [result for result in ablation_results if result.name == "Full Alpha Trinity rebuilt"] + baseline_results
    combine_curves(benchmark_curve_results, "net_returns").to_csv(tables_dir / "baseline_equity_curves.csv")
    combine_curves(ablation_results, "net_returns").to_csv(tables_dir / "ablation_equity_curves.csv")
    pd.DataFrame({result.name: result.net_returns for result in all_results}).to_csv(tables_dir / "daily_net_returns.csv")
    pd.DataFrame({result.name: result.turnover for result in all_results}).to_csv(tables_dir / "daily_turnover.csv")
    rebuilt_full_weights.to_csv(tables_dir / "full_signal_weights.csv")
    ablation_results[0].effective_weights.to_csv(tables_dir / "full_effective_weights.csv")
    ablation_results[0].trades.to_csv(tables_dir / "full_executed_trades.csv")
    crisis_scores.to_csv(tables_dir / "crisis_scores.csv", header=True)

    plot_curves(
        combine_curves(benchmark_curve_results, "net_returns"),
        "Baseline Comparison: Net Equity Curves",
        figures_dir / "baseline_equity_curves.png",
    )
    plot_curves(
        combine_curves(ablation_results, "net_returns"),
        "Ablation Study: Net Equity Curves",
        figures_dir / "ablation_equity_curves.png",
    )
    plot_crisis_scores(crisis_scores, figures_dir / "crisis_score.png")

    assumptions = {
        "project_root": str(project_root),
        "panel_path": "assessment/inputs/panel.parquet",
        "weights_path": "Not used: all weights rebuilt from the frozen panel",
        "start": common_index.min().date().isoformat(),
        "end": common_index.max().date().isoformat(),
        "trading_days": int(len(common_index)),
        "symbols": list(returns_eval.columns),
        "transaction_cost": TRANSACTION_COST,
        "cost_of_carry": COST_OF_CARRY,
        "signal_lag_days": 1,
        "execution": "signal close t; fill close t+1; first return close t+1 to t+2; costs at fill",
        "train_end_date": TRAIN_END_DATE.date().isoformat(),
        "test_start_date": TEST_START_DATE.date().isoformat(),
        "features": FEATURES,
        "defensive_assets": DEFENSIVE_ASSETS,
        "growth_ai_assets": GROWTH_AI_ASSETS,
        "vix_column_used": "VIX",
        "vix_unique_values": int(feature_df["VIX"].nunique()),
        "vix_close_unique_values": int(feature_df["vix_close"].nunique()) if "vix_close" in feature_df.columns else None,
    }
    (manifests_dir / "protocol_assumptions.json").write_text(json.dumps(assumptions, indent=2), encoding="utf-8")
    write_research_question(docs_dir)
    write_summary(docs_dir, baseline_table, ablation_table, assumptions)

    print(f"Done. Outputs written to {assessment_dir}")


if __name__ == "__main__":
    main()
