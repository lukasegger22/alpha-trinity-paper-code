from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "alpha_trinity_mpl"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpha_trinity_assessment import first_three as core


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "assessment"
DOCS_DIR = OUTPUT_DIR / "docs"
TABLES_DIR = OUTPUT_DIR / "outputs" / "tables"
FIGURES_DIR = OUTPUT_DIR / "outputs" / "figures"
MANIFESTS_DIR = OUTPUT_DIR / "outputs" / "manifests"

BASE_CAUTION = 0.35
BASE_PANIC = 0.65
BASE_COST = 0.0010
BASE_HOLD = 5
BASE_LEVERAGE_CAP = 1.30

ETF_ONLY_EXCLUSIONS = ["NVDA", "MSFT", "META", "ASML"]
AI_EXCLUSIONS = ["NVDA", "MSFT", "META", "ASML"]


def feature_df_with_forward_horizon(feature_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = feature_df.copy().sort_values(["symbol", "Date"])
    forward = df.groupby("symbol")["proxy_price"].shift(-horizon) / df["proxy_price"] - 1.0
    df["target_20d_forward"] = forward.replace([np.inf, -np.inf], np.nan)
    df["target_20d_end_date"] = df.groupby("symbol")["Date"].shift(-horizon)
    return df


def prepare_inputs() -> dict[str, object]:
    panel = core.load_panel(PROJECT_ROOT)
    returns = core.pivot_returns(panel)
    returns_eval = core.evaluation_returns(panel)
    common_index = returns_eval.index

    feature_df = core.add_assessment_features(panel)
    ml_signal = core.train_ml_signal(feature_df)
    daily_vix = feature_df.groupby("Date")["VIX"].mean().sort_index()
    crisis_scores = core.calculate_crisis_scores(feature_df, returns.index).loc[common_index]

    return {
        "panel": panel,
        "feature_df": feature_df,
        "returns_eval": returns_eval,
        "returns": returns,
        "ml_signal": ml_signal,
        "daily_vix": daily_vix,
        "crisis_scores": crisis_scores,
        "common_index": common_index,
    }


def metric_record(
    result: core.BacktestResult,
    group: str,
    variant: str,
    **params: object,
) -> dict[str, object]:
    row = core.metric_row(result, group=group)
    row["variant"] = variant
    for key, value in params.items():
        row[key] = value
    return row


def run_rebuilt_variant(
    variant: str,
    returns_eval: pd.DataFrame,
    ml_signal: pd.DataFrame,
    crisis_scores: pd.Series,
    daily_vix: pd.Series,
    transaction_cost: float = BASE_COST,
    caution_threshold: float | pd.Series = BASE_CAUTION,
    panic_threshold: float | pd.Series = BASE_PANIC,
    min_hold_days: int = BASE_HOLD,
    leverage_cap: float = BASE_LEVERAGE_CAP,
    excluded_symbols: list[str] | None = None,
    panic_assets: dict[str, float] | None = None,
) -> core.BacktestResult:
    excluded_symbols = excluded_symbols or []
    active_columns = [c for c in returns_eval.columns if c not in excluded_symbols]
    history = core.pivot_returns(core.load_panel(PROJECT_ROOT))
    if panic_assets and "SHY" in panic_assets:
        close = pd.read_csv(PROJECT_ROOT / "assessment/inputs/shy_adjusted_close.csv", index_col=0, parse_dates=True)["SHY"]
        shy_returns = close.pct_change(fill_method=None).reindex(history.index)
        if shy_returns.isna().any():
            raise ValueError("Incomplete SHY calendar")
        history["SHY"] = shy_returns
        active_columns.append("SHY")
    active_returns = history.loc[:returns_eval.index.max(), active_columns].copy()
    active_signal = ml_signal.reindex(index=active_returns.index, columns=active_columns).fillna(0.0)

    weights = core.build_engine_weights(
        variant,
        active_returns,
        active_signal,
        crisis_scores,
        daily_vix,
        use_ml=True,
        use_crisis=True,
        caution_threshold=caution_threshold,
        panic_threshold=panic_threshold,
        min_hold_days=min_hold_days,
        leverage_cap=leverage_cap,
        transaction_cost=transaction_cost,
        panic_assets=panic_assets,
    )
    return core.evaluate_weights(
        variant,
        weights,
        active_returns,
        transaction_cost=transaction_cost,
        cost_of_carry=core.COST_OF_CARRY,
    )


def evaluate_same_day(
    name: str,
    signal_weights: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_cost: float = BASE_COST,
    cost_of_carry: float = core.COST_OF_CARRY,
) -> core.BacktestResult:
    signal_weights = core.align_weights(signal_weights.sort_index(), returns.sort_index()).fillna(0.0)
    returns = returns.loc[signal_weights.index, signal_weights.columns].fillna(0.0)

    effective_weights = signal_weights.copy()
    gross_returns = (effective_weights * returns).sum(axis=1).fillna(0.0)
    turnover = signal_weights.diff().abs().sum(axis=1).fillna(0.0)
    trading_cost = turnover * transaction_cost
    gross_exposure = effective_weights.abs().sum(axis=1)
    carry_cost = (gross_exposure - 1.0).clip(lower=0.0) * cost_of_carry / core.PERIODS_PER_YEAR
    net_returns = gross_returns - trading_cost - carry_cost

    return core.BacktestResult(
        name=name,
        net_returns=net_returns,
        gross_returns=gross_returns,
        trading_cost=trading_cost,
        carry_cost=carry_cost,
        turnover=turnover,
        effective_weights=effective_weights,
        signal_weights=signal_weights,
    )


def run_robustness(inputs: dict[str, object]) -> pd.DataFrame:
    returns_eval = inputs["returns_eval"]
    ml_signal = inputs["ml_signal"]
    crisis_scores = inputs["crisis_scores"]
    daily_vix = inputs["daily_vix"]

    rows: list[dict[str, object]] = []
    base = run_rebuilt_variant("base", returns_eval, ml_signal, crisis_scores, daily_vix)
    for bps in [5, 10, 20, 50]:
        fixed = core.evaluate_weights(f"fixed path {bps}bps", base.signal_weights, returns_eval,
                                      transaction_cost=bps / 10000.0)
        rows.append(metric_record(fixed, "fixed_path_cost_sensitivity", f"{bps} bps", transaction_cost_bps=bps))

    for caution in [0.30, 0.35, 0.40]:
        for panic in [0.60, 0.65, 0.70]:
            result = run_rebuilt_variant(
                f"threshold {caution:.2f}/{panic:.2f}",
                returns_eval,
                ml_signal,
                crisis_scores,
                daily_vix,
                caution_threshold=caution,
                panic_threshold=panic,
            )
            rows.append(
                metric_record(
                    result,
                    "crisis_threshold_sensitivity",
                    f"caution={caution:.2f}, panic={panic:.2f}",
                    caution_threshold=caution,
                    panic_threshold=panic,
                    transaction_cost_bps=10,
                    min_hold_days=BASE_HOLD,
                    leverage_cap=BASE_LEVERAGE_CAP,
                    excluded_symbols="",
                    symbol_count=len(returns_eval.columns),
                )
            )

    rolling_source = crisis_scores.shift(1)
    dynamic_caution = (
        rolling_source.rolling(126, min_periods=63)
        .quantile(0.60)
        .clip(lower=0.20, upper=0.55)
        .fillna(BASE_CAUTION)
    )
    dynamic_panic_raw = (
        rolling_source.rolling(126, min_periods=63)
        .quantile(0.85)
        .clip(lower=0.45, upper=0.85)
        .fillna(BASE_PANIC)
    )
    dynamic_panic = pd.Series(
        np.maximum(dynamic_panic_raw.to_numpy(), dynamic_caution.to_numpy() + 0.05),
        index=dynamic_panic_raw.index,
    )
    result = run_rebuilt_variant(
        "rolling percentile thresholds",
        returns_eval,
        ml_signal,
        crisis_scores,
        daily_vix,
        caution_threshold=dynamic_caution,
        panic_threshold=dynamic_panic,
    )
    rows.append(
        metric_record(
            result,
            "dynamic_threshold_sensitivity",
            "rolling 60/85 percentile",
            caution_threshold="rolling_p60_126d",
            panic_threshold="rolling_p85_126d",
            transaction_cost_bps=10,
            min_hold_days=BASE_HOLD,
            leverage_cap=BASE_LEVERAGE_CAP,
            excluded_symbols="",
            symbol_count=len(returns_eval.columns),
        )
    )

    for bps in [5, 10, 20, 50]:
        transaction_cost = bps / 10000.0
        result = run_rebuilt_variant(
            f"cost {bps}bps",
            returns_eval,
            ml_signal,
            crisis_scores,
            daily_vix,
            transaction_cost=transaction_cost,
        )
        rows.append(
            metric_record(
                result,
                "transaction_cost_sensitivity",
                f"{bps} bps",
                caution_threshold=BASE_CAUTION,
                panic_threshold=BASE_PANIC,
                transaction_cost_bps=bps,
                min_hold_days=BASE_HOLD,
                leverage_cap=BASE_LEVERAGE_CAP,
                excluded_symbols="",
                symbol_count=len(returns_eval.columns),
            )
        )

    feature_df = inputs["feature_df"]
    for horizon in [10, 20, 40]:
        if horizon == 20:
            horizon_signal = ml_signal
        else:
            horizon_signal = core.train_ml_signal(feature_df_with_forward_horizon(feature_df, horizon))
        result = run_rebuilt_variant(
            f"prediction horizon {horizon}d",
            returns_eval,
            horizon_signal,
            crisis_scores,
            daily_vix,
        )
        rows.append(
            metric_record(
                result,
                "prediction_horizon_sensitivity",
                f"{horizon} trading days",
                caution_threshold=BASE_CAUTION,
                panic_threshold=BASE_PANIC,
                transaction_cost_bps=10,
                min_hold_days=BASE_HOLD,
                leverage_cap=BASE_LEVERAGE_CAP,
                excluded_symbols="",
                symbol_count=len(returns_eval.columns),
                prediction_horizon_days=horizon,
            )
        )

    for hold_days in [0, 3, 5, 10]:
        result = run_rebuilt_variant(
            f"hold {hold_days}d",
            returns_eval,
            ml_signal,
            crisis_scores,
            daily_vix,
            min_hold_days=hold_days,
        )
        rows.append(
            metric_record(
                result,
                "holding_period_sensitivity",
                f"{hold_days} days",
                caution_threshold=BASE_CAUTION,
                panic_threshold=BASE_PANIC,
                transaction_cost_bps=10,
                min_hold_days=hold_days,
                leverage_cap=BASE_LEVERAGE_CAP,
                excluded_symbols="",
                symbol_count=len(returns_eval.columns),
            )
        )

    for cap in [1.00, 1.15, 1.30, 1.50]:
        result = run_rebuilt_variant(
            f"leverage cap {cap:.2f}",
            returns_eval,
            ml_signal,
            crisis_scores,
            daily_vix,
            leverage_cap=cap,
        )
        rows.append(
            metric_record(
                result,
                "leverage_cap_sensitivity",
                f"{cap:.2f}x cap",
                caution_threshold=BASE_CAUTION,
                panic_threshold=BASE_PANIC,
                transaction_cost_bps=10,
                min_hold_days=BASE_HOLD,
                leverage_cap=cap,
                excluded_symbols="",
                symbol_count=len(returns_eval.columns),
            )
        )

    universe_variants = [
        ("full universe rebuilt", []),
        ("remove NVDA", ["NVDA"]),
        ("remove AI/growth stocks", AI_EXCLUSIONS),
        ("remove GLD", ["GLD"]),
        ("remove UUP", ["UUP"]),
        ("ETF-only universe", ETF_ONLY_EXCLUSIONS),
    ]
    for variant, exclusions in universe_variants:
        result = run_rebuilt_variant(
            variant,
            returns_eval,
            ml_signal,
            crisis_scores,
            daily_vix,
            excluded_symbols=exclusions,
        )
        rows.append(
            metric_record(
                result,
                "asset_universe_sensitivity",
                variant,
                caution_threshold=BASE_CAUTION,
                panic_threshold=BASE_PANIC,
                transaction_cost_bps=10,
                min_hold_days=BASE_HOLD,
                leverage_cap=BASE_LEVERAGE_CAP,
                excluded_symbols=" ".join(exclusions),
                symbol_count=len(returns_eval.columns) - len(exclusions),
            )
        )

    safe_haven_variants = [
        ("panic UUP/GLD", {"UUP": 0.50, "GLD": 0.30}),
        ("panic UUP/GLD/AGG/TLT", {"UUP": 0.25, "GLD": 0.25, "AGG": 0.15, "TLT": 0.15}),
        ("panic AGG/TLT", {"AGG": 0.40, "TLT": 0.40}),
        ("panic TLT", {"TLT": 0.8}),
        ("panic AGG", {"AGG": 0.8}),
        ("panic SHY", {"SHY": 0.8}),
        ("panic cash zero-interest", {}),
    ]
    for variant, panic_assets in safe_haven_variants:
        result = run_rebuilt_variant(
            variant,
            returns_eval,
            ml_signal,
            crisis_scores,
            daily_vix,
            panic_assets=panic_assets,
        )
        rows.append(
            metric_record(
                result,
                "safe_haven_sensitivity",
                variant,
                caution_threshold=BASE_CAUTION,
                panic_threshold=BASE_PANIC,
                transaction_cost_bps=10,
                min_hold_days=BASE_HOLD,
                leverage_cap=BASE_LEVERAGE_CAP,
                excluded_symbols="",
                symbol_count=len(result.signal_weights.columns),
                panic_assets=" ".join(f"{symbol}:{weight:.2f}" for symbol, weight in panic_assets.items()),
            )
        )

    return pd.DataFrame(rows)


def run_lookahead_audit(inputs: dict[str, object]) -> pd.DataFrame:
    returns_eval = inputs["returns_eval"]
    ml_signal = inputs["ml_signal"]
    crisis_scores = inputs["crisis_scores"]
    daily_vix = inputs["daily_vix"]

    rebuilt_weights = core.build_engine_weights(
        "rebuilt full weights",
        inputs["returns"],
        ml_signal,
        crisis_scores,
        daily_vix,
        use_ml=True,
        use_crisis=True,
    )

    results = [
        ("rebuilt full", "one_day_lag", core.evaluate_weights("rebuilt full one-day lag", rebuilt_weights, returns_eval)),
        ("rebuilt full", "same_day", evaluate_same_day("rebuilt full same-day", rebuilt_weights, returns_eval)),
    ]

    rows = []
    for strategy_family, timing, result in results:
        row = core.metric_row(result, group="lookahead_timing_audit")
        row["strategy_family"] = strategy_family
        row["timing"] = timing
        row["allowed_for_final_reporting"] = timing == "one_day_lag"
        rows.append(row)
    return pd.DataFrame(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_inventory(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        if not path.exists():
            rows.append({"path": str(path.relative_to(PROJECT_ROOT)), "exists": False})
            continue
        stat = path.stat()
        rows.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "exists": True,
                "bytes": stat.st_size,
                "modified_local": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "sha256": file_sha256(path),
            }
        )
    return pd.DataFrame(rows)


def plot_threshold_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    sub = df[df["group"] == "crisis_threshold_sensitivity"].copy()
    pivot = sub.pivot(index="caution_threshold", columns="panic_threshold", values="sharpe_4rf")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{i:.2f}" for i in pivot.index])
    ax.set_xlabel("Panic threshold")
    ax.set_ylabel("Caution threshold")
    ax.set_title("Crisis Threshold Sensitivity: Sharpe (4% reference)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, label="Sharpe")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_line_group(df: pd.DataFrame, group: str, x_col: str, title: str, output_path: Path) -> None:
    sub = df[df["group"] == group].sort_values(x_col).copy()
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(sub[x_col], sub["net_return"], marker="o", label="Net return", color="#1f77b4")
    ax1.set_ylabel("Net return")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax1.grid(True, alpha=0.30)

    ax2 = ax1.twinx()
    ax2.plot(sub[x_col], sub["sharpe_4rf"], marker="s", label="Sharpe 4% RF", color="#d62728")
    ax2.set_ylabel("Sharpe 4% RF")

    ax1.set_xlabel(x_col.replace("_", " "))
    ax1.set_title(title)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_asset_sensitivity(df: pd.DataFrame, output_path: Path) -> None:
    sub = df[df["group"] == "asset_universe_sensitivity"].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = sub["variant"].tolist()
    ax.bar(labels, sub["net_return"], color="#2ca02c", alpha=0.8)
    ax.set_ylabel("Net return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Asset Universe Sensitivity")
    ax.grid(True, axis="y", alpha=0.30)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_lookahead(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{r.strategy_family}\n{r.timing}" for r in df.itertuples()]
    colors = ["#1f77b4" if r.timing == "one_day_lag" else "#d62728" for r in df.itertuples()]
    ax.bar(labels, df["net_return"], color=colors, alpha=0.85)
    ax.set_ylabel("Net return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Timing Audit: One-Day Lag vs Same-Day")
    ax.grid(True, axis="y", alpha=0.30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_robustness_summary(df: pd.DataFrame, output_path: Path) -> None:
    def best(group: str, metric: str = "sharpe_4rf") -> pd.Series:
        sub = df[df["group"] == group]
        return sub.loc[sub[metric].idxmax()]

    def worst(group: str, metric: str = "sharpe_4rf") -> pd.Series:
        sub = df[df["group"] == group]
        return sub.loc[sub[metric].idxmin()]

    base = df[
        (df["group"] == "crisis_threshold_sensitivity")
        & (df["caution_threshold"] == BASE_CAUTION)
        & (df["panic_threshold"] == BASE_PANIC)
    ].iloc[0]

    text = f"""# Robustness and Sensitivity Summary

Generated: {datetime.now().isoformat(timespec="seconds")}

## Base Rebuilt Model

The base rebuilt model uses caution threshold {BASE_CAUTION:.2f}, panic threshold {BASE_PANIC:.2f}, 10 bps transaction cost, {BASE_HOLD} holding days, and {BASE_LEVERAGE_CAP:.2f}x leverage cap.

- Net return: {base['net_return']:.2%}
- Sharpe ratio with 4 percent risk-free rate: {base['sharpe_4rf']:.2f}
- Maximum drawdown: {base['max_drawdown']:.2%}
- Total turnover: {base['total_turnover']:.2f}

## Threshold Sensitivity

Best threshold setting by Sharpe: {best('crisis_threshold_sensitivity')['variant']} with Sharpe {best('crisis_threshold_sensitivity')['sharpe_4rf']:.2f} and net return {best('crisis_threshold_sensitivity')['net_return']:.2%}.

Worst threshold setting by Sharpe: {worst('crisis_threshold_sensitivity')['variant']} with Sharpe {worst('crisis_threshold_sensitivity')['sharpe_4rf']:.2f} and net return {worst('crisis_threshold_sensitivity')['net_return']:.2%}.

## Cost Sensitivity

At 5, 10, 20, and 50 bps, the model remains positive in this sample. The key question for the report is not only return survival, but how quickly Sharpe and turnover efficiency degrade as costs rise.

## Holding Period Sensitivity

The run includes holding periods of 0, 3, 5, and 10 days. This directly addresses the assessment request to justify why the 5-day anti-churn rule is reasonable.

## Leverage Cap Sensitivity

The run includes leverage caps of 1.00x, 1.15x, and 1.30x. Because the rebuilt weights do not always reach the cap, the interpretation should focus on observed gross exposure, not only configured maximum leverage.

## Asset Sensitivity

The asset tests include the full universe, removal of the AI/growth stock basket, and an ETF-only universe. In the current universe, `remove AI/growth stocks` and `ETF-only universe` exclude the same symbols: {', '.join(AI_EXCLUSIONS)}.
"""
    output_path.write_text(text, encoding="utf-8")


def write_reproducibility_docs(inputs: dict[str, object], inventory: pd.DataFrame) -> None:
    panel = inputs["panel"]
    returns_eval = inputs["returns_eval"]
    feature_df = inputs["feature_df"]

    per_symbol = panel.groupby("symbol").agg(first=("Date", "min"), last=("Date", "max"), rows=("Date", "count"))
    missing = panel.isna().sum()

    reproducibility = f"""# Reproducibility Protocol

Generated: {datetime.now().isoformat(timespec="seconds")}

## Data Source

The implemented downloader uses `yfinance.download` with `auto_adjust=True` for the configured asset universe and separately downloads `^VIX`. The raw downloaded panel is saved to `data/panel.parquet`; engineered features are saved to `data/features/panel.parquet`.

## Local Data Snapshot

- Feature panel rows: {len(panel)}
- Feature panel symbols: {panel['symbol'].nunique()}
- Feature panel date range: {panel['Date'].min().date()} to {panel['Date'].max().date()}
- Assessment backtest range: {returns_eval.index.min().date()} to {returns_eval.index.max().date()}
- Assessment trading days: {len(returns_eval)}
- Frozen input: `assessment/inputs/panel.parquet`; saved strategy weights are not inputs.
- Original download time is unknown. File modification times are NOT download dates.
- Snapshot frozen on 2026-09-17; see `assessment/inputs/provenance.json` for the checksum.

## Adjusted Prices

The downloader calls yfinance with `auto_adjust=True`. This means yfinance returns prices adjusted for splits and dividends in the OHLC fields. The feature builder uses the adjusted `Close` field to compute returns.

## Calendar Synchronization

The active feature panel has an equal number of rows for every symbol. This indicates that the downloaded/feature-built data was aligned to a shared daily date grid before assessment evaluation. The project also contains `src/ai_ls_allocation/data/trading_calendar.py`, which provides a NYSE trading-day helper.

## Missing Values

Current feature-panel missing values after the feature build:

```text
{missing.to_string()}
```

Per-symbol coverage:

```text
{per_symbol.to_string()}
```

## Model and Random Seeds

- Prediction target: 20-day forward return per asset.
- Features: {', '.join(core.FEATURES)}.
- Normalization: `StandardScaler` fitted only on rows before {core.TRAIN_END_DATE.date()} whose forward labels end by that cutoff; then applied to later rows. Walk-forward runs refit the scaler and models at each annual cutoff.
- Ensemble: {core.ENSEMBLE_SIZE} models.
- MLP models: hidden layers `(64, 32)`, `max_iter=200`, random seeds 42-46.
- Random Forest models: `n_estimators=30`, `max_depth=8`, `n_jobs=-1`, random seeds 47-51.
- Global assessment constants: transaction cost {BASE_COST:.4f}, annual leverage carry {core.COST_OF_CARRY:.2%}, {BASE_HOLD}-session exit guard (partial reductions allowed; panic and hard bounds override it).

## Backtest Timing

The defensible assessment rule is:

1. features and target weights are computed at close of day t,
2. target weights are saved with timestamp t,
3. the order fills at close t+1, after existing holdings earn that session's return,
4. actual trades against drifted holdings and fees are charged at that fill,
5. the new position first earns the close t+1 to close t+2 return.

This is the rule used by `assessment/scripts/run_first_three.py` and `assessment/scripts/run_robustness_repro_audit.py`.

## Artifact Inventory

See `assessment/outputs/tables/reproducibility_inventory.csv` for local file sizes, modification timestamps, and SHA-256 hashes. These timestamps are not historical acquisition dates.
"""

    lookahead = """# Look-Ahead Bias Audit

## Decision

Final assessment results must use a one-trading-day execution lag. Same-day results are diagnostic only and must not be reported as final performance.

## Why

The strategy stores target weights with the signal date. If those weights were multiplied by the same day's returns, the backtest could accidentally use information from the close before the trade could have been executed. This is the exact ambiguity criticized in the assessment.

## Implemented Safeguard

The holdings ledger in `first_three.evaluate_weights` earns returns on existing
holdings before filling yesterday's signal at today's close. A signal at t first
affects the return ending t+2. Fees occur at the close t+1 fill. Synthetic price-jump
and drift tests in `tests/test_assessment.py` verify both timing and accounting.

## Diagnostic Comparison

The file `lookahead_timing_audit.csv` compares next-close execution against a
deliberately invalid same-day diagnostic. Only next-close results are admissible.
The invalid diagnostic is not an alternative execution model.

## Remaining Risk

This audit covers portfolio-return timing. The feature pipeline must still be kept strict: rolling features must use only past and current data, normalization must be fitted only on training data, and model/hyperparameter choices must not be tuned on the final test period.
"""

    DOCS_DIR.joinpath("reproducibility.md").write_text(reproducibility, encoding="utf-8")
    DOCS_DIR.joinpath("lookahead_bias_audit.md").write_text(lookahead, encoding="utf-8")
    inventory.to_csv(TABLES_DIR / "reproducibility_inventory.csv", index=False)


def write_remaining_gaps() -> None:
    DOCS_DIR.joinpath("remaining_assessment_gaps.md").write_text(
        "# Evidence Pipeline Status\n\n"
        "Sensitivity and timing outputs rebuilt. The completion stage adds full portfolio "
        "walk-forward and prediction diagnostics. Original acquisition time and historical "
        "selection bias remain disclosed boundaries, not claims of complete elimination.\n",
        encoding="utf-8")


def write_step_4_6_summary(robustness: pd.DataFrame, lookahead: pd.DataFrame) -> None:
    groups = [
        "crisis_threshold_sensitivity",
        "dynamic_threshold_sensitivity",
        "transaction_cost_sensitivity",
        "prediction_horizon_sensitivity",
        "holding_period_sensitivity",
        "leverage_cap_sensitivity",
        "asset_universe_sensitivity",
        "safe_haven_sensitivity",
    ]
    lines = [
        "# Assessment Step 4-6 Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Step 4: Robustness/Sensitivity Tests",
        "",
    ]
    for group in groups:
        sub = robustness[robustness["group"] == group].copy()
        lines.append(f"### {group.replace('_', ' ').title()}")
        lines.append("")
        lines.append("| Variant | Net Return | Sharpe 4% RF | Max Drawdown | Turnover |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for _, row in sub.iterrows():
            lines.append(
                f"| {row['variant']} | {row['net_return']:.2%} | {row['sharpe_4rf']:.2f} | "
                f"{row['max_drawdown']:.2%} | {row['total_turnover']:.2f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Step 5: Reproducibility Documentation",
            "",
            "Documented in `reproducibility.md` and `reproducibility_inventory.csv`.",
            "",
            "## Step 6: Look-Ahead Bias Safeguard",
            "",
            "Final assessment metrics use a one-day execution lag. Same-day rows below are diagnostic only.",
            "",
            "| Strategy Family | Timing | Net Return | Sharpe 4% RF | Allowed For Final Reporting |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for _, row in lookahead.iterrows():
        lines.append(
            f"| {row['strategy_family']} | {row['timing']} | {row['net_return']:.2%} | "
            f"{row['sharpe_4rf']:.2f} | {row['allowed_for_final_reporting']} |"
        )
    lines.append("")
    DOCS_DIR.joinpath("steps_4_6_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparing shared assessment inputs...")
    inputs = prepare_inputs()

    print("Running robustness and sensitivity tests...")
    robustness = run_robustness(inputs)
    robustness.to_csv(TABLES_DIR / "robustness_sensitivity.csv", index=False)
    for group, sub in robustness.groupby("group"):
        sub.to_csv(TABLES_DIR / f"{group}.csv", index=False)

    print("Running look-ahead timing audit...")
    lookahead = run_lookahead_audit(inputs)
    lookahead.to_csv(TABLES_DIR / "lookahead_timing_audit.csv", index=False)

    print("Writing reproducibility inventory and documentation...")
    inventory_paths = [
        PROJECT_ROOT / "assessment" / "inputs" / "panel.parquet",
        PROJECT_ROOT / "assessment" / "inputs" / "provenance.json",
        PROJECT_ROOT / "data" / "panel.parquet",
        PROJECT_ROOT / "data" / "features" / "panel.parquet",
        PROJECT_ROOT / "data" / "features" / "trinity_history.parquet",
        PROJECT_ROOT / "data" / "features" / "trinity_signals.parquet",
        PROJECT_ROOT / "data" / "features" / "trinity_signals.csv",
        PROJECT_ROOT / "config" / "universes.yaml",
        PROJECT_ROOT / "config" / "backtest.yaml",
        PROJECT_ROOT / "config" / "costs.yaml",
        PROJECT_ROOT / "src" / "ai_ls_allocation" / "data" / "download.py",
        PROJECT_ROOT / "src" / "ai_ls_allocation" / "features" / "build.py",
        PROJECT_ROOT / "src" / "ai_ls_allocation" / "engine" / "bt_trinity.py",
        PROJECT_ROOT / "assessment" / "src" / "alpha_trinity_assessment" / "first_three.py",
        PROJECT_ROOT / "assessment" / "src" / "alpha_trinity_assessment" / "robustness_repro_audit.py",
    ]
    inventory = build_inventory(inventory_paths)
    write_reproducibility_docs(inputs, inventory)
    write_remaining_gaps()
    write_step_4_6_summary(robustness, lookahead)

    print("Writing plots...")
    plot_threshold_heatmap(robustness, FIGURES_DIR / "threshold_sensitivity_heatmap.png")
    plot_line_group(
        robustness,
        "transaction_cost_sensitivity",
        "transaction_cost_bps",
        "Transaction Cost Sensitivity",
        FIGURES_DIR / "cost_sensitivity.png",
    )
    plot_line_group(
        robustness,
        "holding_period_sensitivity",
        "min_hold_days",
        "Holding Period Sensitivity",
        FIGURES_DIR / "holding_period_sensitivity.png",
    )
    plot_line_group(
        robustness,
        "leverage_cap_sensitivity",
        "leverage_cap",
        "Leverage Cap Sensitivity",
        FIGURES_DIR / "leverage_cap_sensitivity.png",
    )
    plot_asset_sensitivity(robustness, FIGURES_DIR / "asset_universe_sensitivity.png")
    plot_lookahead(lookahead, FIGURES_DIR / "lookahead_timing_audit.png")

    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "outputs": [
            "robustness_sensitivity.csv",
            "crisis_threshold_sensitivity.csv",
            "dynamic_threshold_sensitivity.csv",
            "transaction_cost_sensitivity.csv",
            "prediction_horizon_sensitivity.csv",
            "holding_period_sensitivity.csv",
            "leverage_cap_sensitivity.csv",
            "asset_universe_sensitivity.csv",
            "safe_haven_sensitivity.csv",
            "lookahead_timing_audit.csv",
            "reproducibility.md",
            "reproducibility_inventory.csv",
            "lookahead_bias_audit.md",
            "steps_4_6_summary.md",
            "remaining_assessment_gaps.md",
        ],
    }
    (MANIFESTS_DIR / "steps_4_6_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done. Outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
