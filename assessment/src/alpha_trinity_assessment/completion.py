from __future__ import annotations

import json
import os
import tempfile
import warnings
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from alpha_trinity_assessment import first_three as core


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ASSESSMENT_DIR = PROJECT_ROOT / "assessment"
DOCS_DIR = ASSESSMENT_DIR / "docs"
TABLES_DIR = ASSESSMENT_DIR / "outputs" / "tables"
FIGURES_DIR = ASSESSMENT_DIR / "outputs" / "figures"
MANIFESTS_DIR = ASSESSMENT_DIR / "outputs" / "manifests"
PRIMARY_STRATEGY = "Full Alpha Trinity rebuilt dynamic VIX"


FEATURE_CATALOG = [
    {
        "feature": "returns_1d",
        "group": "momentum",
        "formula": "Close_t / Close_{t-1} - 1",
        "window": "1 trading day",
        "purpose": "Short-term price movement.",
        "leakage_guard": "Uses current and past adjusted closes only; evaluated with one-day execution lag.",
    },
    {
        "feature": "returns_5d",
        "group": "momentum",
        "formula": "Close_t / Close_{t-5} - 1",
        "window": "5 trading days",
        "purpose": "Weekly momentum.",
        "leakage_guard": "Rolling past window only.",
    },
    {
        "feature": "volatility_60d",
        "group": "risk",
        "formula": "std(returns_1d over 60 days) * sqrt(252)",
        "window": "60 trading days",
        "purpose": "Realized volatility and risk regime.",
        "leakage_guard": "Trailing volatility only.",
    },
    {
        "feature": "VIX",
        "group": "macro stress",
        "formula": "VIX close from yfinance ^VIX, mapped from vix_close when needed",
        "window": "same market date",
        "purpose": "Market fear and volatility stress proxy.",
        "leakage_guard": "Signal at close t fills at close t+1; first return ends t+2.",
    },
    {
        "feature": "obv_trend",
        "group": "volume",
        "formula": "20-day pct_change of On-Balance Volume proxy",
        "window": "20 trading days",
        "purpose": "Volume participation and accumulation/distribution trend.",
        "leakage_guard": "Cumulative volume sign uses realized return through t only.",
    },
    {
        "feature": "dist_vwap",
        "group": "mean reversion",
        "formula": "proxy_price_t / rolling_vwap_20_t - 1",
        "window": "20 trading days",
        "purpose": "Distance from volume-weighted mean.",
        "leakage_guard": "Trailing price-volume window only.",
    },
    {
        "feature": "mfi",
        "group": "volume",
        "formula": "Money Flow Index proxy scaled to [0, 1]",
        "window": "14 trading days",
        "purpose": "Buying/selling pressure.",
        "leakage_guard": "Trailing positive and negative money flow only.",
    },
    {
        "feature": "bb_pos",
        "group": "mean reversion",
        "formula": "(proxy_price_t - lower_band_20_t) / (upper_band_20_t - lower_band_20_t)",
        "window": "20 trading days",
        "purpose": "Position inside Bollinger channel.",
        "leakage_guard": "Trailing moving average and standard deviation only.",
    },
]

FEATURE_GROUPS = {
    "momentum": ["returns_1d", "returns_5d"],
    "risk": ["volatility_60d", "VIX"],
    "volume": ["obv_trend", "mfi"],
    "mean_reversion": ["dist_vwap", "bb_pos"],
}


def ensure_dirs() -> None:
    for path in [DOCS_DIR, TABLES_DIR, FIGURES_DIR, MANIFESTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def prepare_inputs() -> dict[str, object]:
    panel = core.load_panel(PROJECT_ROOT)
    returns = core.pivot_returns(panel)
    returns_eval = core.evaluation_returns(panel)
    common_index = returns_eval.index
    feature_df = core.add_assessment_features(panel)
    crisis_scores = core.calculate_crisis_scores(feature_df, returns.index).loc[common_index]
    daily_vix = feature_df.groupby("Date")["VIX"].mean().sort_index()
    return {
        "panel": panel,
        "feature_df": feature_df,
        "returns": returns,
        "returns_eval": returns_eval,
        "common_index": common_index,
        "crisis_scores": crisis_scores,
        "daily_vix": daily_vix,
    }


def actual_forward_return(feature_df: pd.DataFrame) -> pd.Series:
    ordered = feature_df.sort_values(["symbol", "Date"]).copy()
    forward = ordered.groupby("symbol")["Close"].shift(-20) / ordered["Close"] - 1.0
    return forward.reindex(feature_df.index)


def actual_forward_end_date(feature_df: pd.DataFrame) -> pd.Series:
    ordered = feature_df.sort_values(["symbol", "Date"]).copy()
    target_end = ordered.groupby("symbol")["Date"].shift(-20)
    return target_end.reindex(feature_df.index)


def fit_prediction_models(
    feature_df: pd.DataFrame,
    train_end: pd.Timestamp,
    mlp_count: int = 5,
    rf_count: int = 5,
    features: list[str] | None = None,
    train_start: pd.Timestamp | None = None,
    permutation: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = list(features or core.FEATURES)
    df = feature_df.copy()
    df["actual_forward_20d"] = actual_forward_return(df)
    df["actual_forward_20d_end_date"] = actual_forward_end_date(df)
    train = df[
        (df["Date"] < train_end)
        & (df["actual_forward_20d_end_date"] <= train_end)
        & df["actual_forward_20d"].notna()
    ].copy()
    if train_start is not None:
        train = train[train["Date"] >= train_start]
    if train.empty:
        raise ValueError(f"No training rows before {train_end.date()}")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[features].to_numpy())
    y_train = train["actual_forward_20d"].to_numpy()
    x_all = scaler.transform(df[features].to_numpy())

    mlp_predictions = []
    rf_predictions = []
    rf_importances = []
    permutation_scores = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for i in range(mlp_count):
            model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42 + i)
            model.fit(x_train, y_train)
            mlp_predictions.append(model.predict(x_all))

        for i in range(rf_count):
            seed = 42 + mlp_count + i
            model = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=seed, n_jobs=-1)
            model.fit(x_train, y_train)
            rf_predictions.append(model.predict(x_all))
            rf_importances.append(model.feature_importances_)
            if permutation:
                evaluation = (df["Date"] >= core.TEST_START_DATE) & df["actual_forward_20d"].notna()
                scored = permutation_importance(model, x_all[evaluation], df.loc[evaluation, "actual_forward_20d"],
                                               n_repeats=5, random_state=47 + i, scoring="neg_mean_squared_error")
                permutation_scores.append(scored.importances_mean)

    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train, y_train)
    pred_ridge = ridge.predict(x_all)
    pred_mlp = np.mean(mlp_predictions, axis=0) if mlp_predictions else np.zeros(len(df))
    pred_rf = np.mean(rf_predictions, axis=0) if rf_predictions else np.zeros(len(df))
    pred_ensemble = np.mean(np.vstack([pred_mlp, pred_rf]), axis=0)

    diagnostics = df[["Date", "symbol", "actual_forward_20d"]].copy()
    diagnostics["pred_mlp"] = pred_mlp
    diagnostics["pred_rf"] = pred_rf
    diagnostics["pred_ridge"] = pred_ridge
    diagnostics["pred_ensemble"] = pred_ensemble
    diagnostics["model_disagreement_abs"] = np.abs(diagnostics["pred_mlp"] - diagnostics["pred_rf"])

    if rf_importances:
        importance = pd.DataFrame(rf_importances, columns=features).mean().reset_index()
        importance.columns = ["feature", "rf_importance"]
        total = importance["rf_importance"].sum()
        if total > 0:
            importance["rf_importance"] = importance["rf_importance"] / total
        importance = importance.sort_values("rf_importance", ascending=False)
        if permutation_scores:
            permutation_map = dict(zip(features, np.mean(permutation_scores, axis=0)))
            importance["permutation_mse_increase"] = importance["feature"].map(permutation_map)
    else:
        importance = pd.DataFrame({"feature": features, "rf_importance": np.nan})

    return diagnostics, importance


def signal_from_prediction_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    scored = diagnostics[["Date", "symbol", "pred_ensemble"]].copy()
    scored["ensemble_signal"] = (
        scored.groupby("symbol")["pred_ensemble"].transform(lambda x: x.ewm(span=5, adjust=False).mean()) * 50.0
    )
    signal = scored.pivot(index="Date", columns="symbol", values="ensemble_signal").sort_index()
    signal.index = core.ensure_datetime_index(signal.index)
    signal.columns = signal.columns.astype(str)
    return signal.fillna(0.0)


def summarize_prediction_quality(
    diagnostics: pd.DataFrame,
    eval_index: pd.Index,
    pred_col: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    eval_df = diagnostics[
        diagnostics["Date"].isin(eval_index)
        & diagnostics["actual_forward_20d"].notna()
        & diagnostics[pred_col].notna()
    ].copy()
    actual = eval_df["actual_forward_20d"]
    pred = eval_df[pred_col]
    error = pred - actual

    daily_rows = []
    for date, group in eval_df.groupby("Date"):
        if len(group) < 2:
            continue
        top = group.nlargest(core.TOP_K, pred_col)
        bottom = group.nsmallest(core.TOP_K, pred_col)
        daily_rows.append(
            {
                "Date": date,
                "prediction_column": pred_col,
                "rank_correlation": group[pred_col].corr(group["actual_forward_20d"], method="spearman"),
                "top_k_hit_rate": float((top["actual_forward_20d"] > 0).mean()),
                "all_asset_positive_rate": float((group["actual_forward_20d"] > 0).mean()),
                "top_k_avg_realized_20d": float(top["actual_forward_20d"].mean()),
                "bottom_k_avg_realized_20d": float(bottom["actual_forward_20d"].mean()),
                "top_bottom_spread_20d": float(top["actual_forward_20d"].mean() - bottom["actual_forward_20d"].mean()),
            }
        )
    daily = pd.DataFrame(daily_rows)

    row = {
        "prediction_column": pred_col,
        "rows": int(len(eval_df)),
        "start": eval_df["Date"].min().date().isoformat(),
        "end": eval_df["Date"].max().date().isoformat(),
        "pearson_correlation": float(pred.corr(actual, method="pearson")),
        "spearman_correlation": float(pred.corr(actual, method="spearman")),
        "directional_hit_rate": float((np.sign(pred) == np.sign(actual)).mean()),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "mean_daily_rank_correlation": float(daily["rank_correlation"].mean()),
        "median_daily_rank_correlation": float(daily["rank_correlation"].median()),
        "top_k_hit_rate": float(daily["top_k_hit_rate"].mean()),
        "top_k_avg_realized_20d": float(daily["top_k_avg_realized_20d"].mean()),
        "top_bottom_spread_20d": float(daily["top_bottom_spread_20d"].mean()),
    }
    return row, daily


def write_prediction_outputs(inputs: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    diagnostics, importance = fit_prediction_models(inputs["feature_df"], core.TRAIN_END_DATE, permutation=True)
    diagnostics.to_csv(TABLES_DIR / "prediction_diagnostics.csv", index=False)

    metric_rows = []
    daily_rank_tables = []
    for pred_col in ["pred_ensemble", "pred_mlp", "pred_rf", "pred_ridge"]:
        row, daily = summarize_prediction_quality(diagnostics, inputs["common_index"], pred_col)
        metric_rows.append(row)
        daily_rank_tables.append(daily)
    metrics = pd.DataFrame(metric_rows)
    daily_rank = pd.concat(daily_rank_tables, ignore_index=True)

    metrics.to_csv(TABLES_DIR / "prediction_metrics.csv", index=False)
    daily_rank.to_csv(TABLES_DIR / "prediction_daily_rank_metrics.csv", index=False)
    importance.to_csv(TABLES_DIR / "feature_importance.csv", index=False)

    plot_feature_importance(importance, FIGURES_DIR / "feature_importance.png")
    return diagnostics, metrics, importance


def plot_feature_importance(importance: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = importance.sort_values("rf_importance")
    ax.barh(ordered["feature"], ordered["rf_importance"], color="#1f77b4", alpha=0.85)
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Normalized importance")
    ax.grid(True, axis="x", alpha=0.30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_feature_outputs(inputs: dict[str, object], importance: pd.DataFrame) -> None:
    feature_catalog = pd.DataFrame(FEATURE_CATALOG)
    feature_catalog = feature_catalog.merge(importance, on="feature", how="left")
    feature_catalog.to_csv(TABLES_DIR / "feature_catalog.csv", index=False)

    train_features = inputs["feature_df"][inputs["feature_df"]["Date"] < core.TRAIN_END_DATE][core.FEATURES]
    corr = train_features.corr()
    corr.to_csv(TABLES_DIR / "feature_correlation_matrix.csv")

    high_pairs = []
    for i, left in enumerate(core.FEATURES):
        for right in core.FEATURES[i + 1 :]:
            value = corr.loc[left, right]
            if abs(value) >= 0.70:
                high_pairs.append({"feature_a": left, "feature_b": right, "correlation": value})
    high_corr = pd.DataFrame(high_pairs)
    if not high_corr.empty:
        high_corr = high_corr.sort_values("correlation", key=lambda s: s.abs(), ascending=False)
    high_corr.to_csv(TABLES_DIR / "feature_high_correlation_pairs.csv", index=False)
    plot_feature_correlation(corr, FIGURES_DIR / "feature_correlation_heatmap.png")

    doc = f"""# Feature Documentation

Generated: {datetime.now().isoformat(timespec="seconds")}

The active assessment feature set contains {len(core.FEATURES)} model inputs. All rolling features are trailing-window calculations, and final portfolio evaluation uses a one-trading-day execution lag.

## Feature Catalog

See `outputs/tables/feature_catalog.csv` for formulas, window lengths, purposes, leakage guards, and Random Forest feature importance.

## Redundancy Check

The training-period correlation matrix is stored in `outputs/tables/feature_correlation_matrix.csv`. Pairs with absolute correlation above 0.70 are stored in `outputs/tables/feature_high_correlation_pairs.csv`.

High-correlation pairs found: {len(high_corr)}.

## VIX Handling

The assessment loader maps `vix_close` into `VIX` when the existing `VIX` column is missing or constant. This fixes the earlier constant-VIX issue for assessment reruns while keeping a clear audit trail in the reproducibility files.
"""
    (DOCS_DIR / "feature_documentation.md").write_text(doc, encoding="utf-8")


def plot_feature_correlation(corr: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    ax.set_title("Training Feature Correlation")
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_feature_group_ablation_outputs(inputs: dict[str, object]) -> pd.DataFrame:
    rows = []
    specs = [("all_features_diagnostic", [], core.FEATURES)]
    for group, removed in FEATURE_GROUPS.items():
        remaining = [feature for feature in core.FEATURES if feature not in removed]
        specs.append((f"drop_{group}", removed, remaining))

    for label, removed_features, selected_features in specs:
        diagnostics, _ = fit_prediction_models(
            inputs["feature_df"],
            core.TRAIN_END_DATE,
            features=selected_features,
        )
        ml_signal = signal_from_prediction_diagnostics(diagnostics)
        weights = core.build_engine_weights(
            f"feature ablation {label}",
            inputs["returns"],
            ml_signal,
            inputs["crisis_scores"],
            inputs["daily_vix"],
            use_ml=True,
            use_crisis=True,
        )
        result = core.evaluate_weights(f"feature ablation {label}", weights, inputs["returns_eval"])
        row = core.metric_row(result, "feature_group_ablation")
        row.update(
            {
                "variant": label,
                "removed_features": " ".join(removed_features),
                "remaining_features": " ".join(selected_features),
                "remaining_feature_count": len(selected_features),
                "diagnostic_model_count": 10,
            }
        )
        rows.append(row)

    output = pd.DataFrame(rows)
    output.to_csv(TABLES_DIR / "feature_group_ablation.csv", index=False)
    plot_feature_group_ablation(output, FIGURES_DIR / "feature_group_ablation.png")
    return output


def plot_feature_group_ablation(table: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = table.sort_values("net_return", ascending=True)
    ax.barh(ordered["variant"], ordered["net_return"], color="#2ca02c", alpha=0.82)
    ax.set_title("Feature Group Ablation Diagnostic")
    ax.set_xlabel("Net return")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, axis="x", alpha=0.30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def classify_regime(score: float) -> str:
    if pd.isna(score):
        return "unassigned"
    if score >= 0.65 - 1e-12:
        return "panic"
    if score >= 0.35 - 1e-12:
        return "caution"
    return "growth"


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    return equity / equity.cummax().clip(lower=1.0) - 1.0


def write_regime_outputs(
    inputs: dict[str, object],
    diagnostics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, core.BacktestResult]]:
    returns_eval = inputs["returns_eval"]
    crisis_scores = inputs["crisis_scores"]
    daily_vix = inputs["daily_vix"]
    ml_signal = signal_from_prediction_diagnostics(diagnostics)

    rebuilt_weights = core.build_engine_weights(
        "Full Alpha Trinity rebuilt dynamic VIX",
        inputs["returns"],
        ml_signal,
        crisis_scores,
        daily_vix,
        use_ml=True,
        use_crisis=True,
    )
    prediction_only_weights = core.build_engine_weights(
        "Prediction-only MLP/RF dynamic VIX",
        inputs["returns"],
        ml_signal,
        crisis_scores,
        daily_vix,
        use_ml=True,
        use_crisis=False,
    )

    results = {
        "Full Alpha Trinity rebuilt dynamic VIX": core.evaluate_weights(
            "Full Alpha Trinity rebuilt dynamic VIX", rebuilt_weights, returns_eval
        ),
        "Prediction-only MLP/RF dynamic VIX": core.evaluate_weights(
            "Prediction-only MLP/RF dynamic VIX", prediction_only_weights, returns_eval
        ),
        "SPY buy-and-hold": core.evaluate_weights(
            "SPY buy-and-hold",
            core.static_weights(returns_eval.index, returns_eval.columns, {"SPY": 1.0}),
            returns_eval,
            rebalance="buy_and_hold",
        ),
        "60/40 SPY/AGG": core.evaluate_weights(
            "60/40 SPY/AGG",
            core.static_weights(returns_eval.index, returns_eval.columns, {"SPY": 0.60, "AGG": 0.40}),
            returns_eval,
        ),
        "Equal-weight universe": core.evaluate_weights(
            "Equal-weight universe", core.equal_weight_weights(returns_eval.index, returns_eval.columns), returns_eval
        ),
        "Defensive basket buy-and-hold": core.evaluate_weights(
            "Defensive basket buy-and-hold",
            core.static_weights(
                returns_eval.index,
                returns_eval.columns,
                {"UUP": 0.25, "GLD": 0.25, "AGG": 0.25, "TLT": 0.25},
            ),
            returns_eval,
            rebalance="buy_and_hold",
        ),
    }

    primary = results[PRIMARY_STRATEGY]
    signal_regime = crisis_scores.map(classify_regime)
    return_regime = crisis_scores.shift(2).map(classify_regime).reindex(primary.net_returns.index).fillna("unassigned")
    regime_daily = pd.DataFrame(
        {
            "Date": primary.net_returns.index,
            "signal_crisis_score": crisis_scores.reindex(primary.net_returns.index).values,
            "signal_regime": signal_regime.reindex(primary.net_returns.index).values,
            "return_regime_from_prior_signal": return_regime.values,
            "net_return_primary_full": primary.net_returns.values,
            "gross_exposure_primary_full": primary.effective_weights.abs().sum(axis=1).values,
            "turnover_primary_full": primary.turnover.values,
        }
    )
    regime_daily.to_csv(TABLES_DIR / "regime_daily.csv", index=False)

    summary_rows = []
    for regime in ["growth", "caution", "panic", "unassigned"]:
        signal_count = int((signal_regime == regime).sum())
        mask = return_regime == regime
        ret = primary.net_returns[mask]
        dd = drawdown_series(ret)
        ann_vol = ret.std(ddof=1) * np.sqrt(core.PERIODS_PER_YEAR)
        ann_return = ret.mean() * core.PERIODS_PER_YEAR
        summary_rows.append(
            {
                "regime": regime,
                "signal_days": signal_count,
                "signal_day_share": signal_count / len(signal_regime),
                "return_days": int(mask.sum()),
                "net_return_primary_full": float((1.0 + ret.fillna(0.0)).prod() - 1.0),
                "annualized_return_arithmetic": float(ann_return) if len(ret) else np.nan,
                "annualized_volatility": float(ann_vol) if len(ret) else np.nan,
                "sharpe_0rf": float(ann_return / ann_vol) if ann_vol > 0 else np.nan,
                "max_drawdown_within_regime": float(dd.min()) if len(dd) else np.nan,
                "avg_gross_exposure": float(primary.effective_weights.abs().sum(axis=1)[mask].mean()) if mask.any() else np.nan,
                "avg_turnover": float(primary.turnover[mask].mean()) if mask.any() else np.nan,
                "cost_sum_on_return_days": float((primary.trading_cost + primary.carry_cost)[mask].sum()),
            }
        )
    regime_summary = pd.DataFrame(summary_rows)
    regime_summary.to_csv(TABLES_DIR / "regime_analytics.csv", index=False)

    snapshots = []
    for regime in ["growth", "caution", "panic"]:
        dates = signal_regime[signal_regime == regime].index
        if len(dates) == 0:
            continue
        if regime == "panic":
            date = crisis_scores.loc[dates].idxmax()
        elif regime == "caution":
            date = dates[len(dates) // 2]
        else:
            date = crisis_scores.loc[dates].idxmin()
        if date not in rebuilt_weights.index:
            continue
        weights = rebuilt_weights.loc[date].sort_values(key=lambda s: s.abs(), ascending=False).head(10)
        for symbol, weight in weights.items():
            snapshots.append(
                {
                    "regime": regime,
                    "date": date.date().isoformat(),
                    "crisis_score": float(crisis_scores.loc[date]),
                    "symbol": symbol,
                    "weight": float(weight),
                }
            )
    pd.DataFrame(snapshots).to_csv(TABLES_DIR / "regime_allocation_snapshots.csv", index=False)
    plot_regime_score(crisis_scores, FIGURES_DIR / "regime_score_dynamic_vix.png")
    return regime_summary, regime_daily, results


def plot_regime_score(crisis_scores: pd.Series, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(crisis_scores.index, crisis_scores, color="#d62728", linewidth=1.5)
    ax.axhline(0.35, color="#ffbf00", linestyle="--", linewidth=1)
    ax.axhline(0.65, color="#8b0000", linestyle="--", linewidth=1)
    ax.fill_between(crisis_scores.index, 0.65, 1.0, color="#8b0000", alpha=0.08)
    ax.set_title("Dynamic VIX Crisis Score")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def period_returns(results: dict[str, core.BacktestResult], freq: str) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        compounded = (1.0 + result.net_returns.fillna(0.0)).resample(freq).prod() - 1.0
        for date, value in compounded.items():
            rows.append({"period": date.date().isoformat(), "strategy": name, "net_return": float(value)})
    return pd.DataFrame(rows)


def drawdown_recovery_rows(results: dict[str, core.BacktestResult]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        ret = result.net_returns.fillna(0.0)
        equity = (1.0 + ret).cumprod()
        dd = equity / equity.cummax().clip(lower=1.0) - 1.0
        trough_date = dd.idxmin()
        peak_date = equity.loc[:trough_date].idxmax()
        peak_value = equity.loc[peak_date]
        after_trough = equity.loc[trough_date:]
        recovered = after_trough[after_trough >= peak_value]
        recovery_date = recovered.index[0] if not recovered.empty else pd.NaT
        rows.append(
            {
                "strategy": name,
                "max_drawdown": float(dd.loc[trough_date]),
                "peak_date": peak_date.date().isoformat(),
                "trough_date": trough_date.date().isoformat(),
                "recovery_date": None if pd.isna(recovery_date) else recovery_date.date().isoformat(),
                "days_peak_to_trough": int((trough_date - peak_date).days),
                "days_trough_to_recovery": None if pd.isna(recovery_date) else int((recovery_date - trough_date).days),
                "recovered_by_end": bool(not recovered.empty),
                "ending_drawdown": float(dd.iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def scenario_stress_rows(results: dict[str, core.BacktestResult]) -> pd.DataFrame:
    scenarios = [
        ("2022 inflation shock", "2022-01-03", "2022-10-12"),
        ("2023-2024 AI rally", "2023-01-03", "2024-12-31"),
        ("2025 correction", "2025-02-14", "2025-04-08"),
        ("2023 bond selloff", "2023-07-03", "2023-10-31"),
        ("2022 commodity shock", "2022-02-01", "2022-06-30"),
        ("August 2024 volatility spike", "2024-07-31", "2024-08-09"),
        ("early 2026 volatility", "2026-01-02", "2026-04-10"),
    ]
    selected = [
        PRIMARY_STRATEGY,
        "Full Alpha Trinity saved",
        "SPY buy-and-hold",
        "60/40 SPY/AGG",
        "Defensive basket buy-and-hold",
        "Prediction-only MLP/RF dynamic VIX",
    ]
    rows = []
    for scenario, start, end in scenarios:
        for strategy in selected:
            if strategy not in results:
                continue
            ret = results[strategy].net_returns.loc[start:end].fillna(0.0)
            rows.append(
                {
                    "scenario": scenario,
                    "start": start,
                    "end": end,
                    "strategy": strategy,
                    "net_return": float((1.0 + ret).prod() - 1.0),
                }
            )
    return pd.DataFrame(rows)


def write_return_and_drawdown_outputs(results: dict[str, core.BacktestResult]) -> pd.DataFrame:
    monthly = period_returns(results, "ME")
    yearly = period_returns(results, "YE")
    monthly.to_csv(TABLES_DIR / "monthly_returns.csv", index=False)
    yearly.to_csv(TABLES_DIR / "yearly_returns.csv", index=False)

    recovery = drawdown_recovery_rows(results)
    recovery.to_csv(TABLES_DIR / "drawdown_recovery.csv", index=False)
    scenario_stress_rows(results).to_csv(TABLES_DIR / "scenario_stress_tests.csv", index=False)
    plot_yearly_returns(yearly, FIGURES_DIR / "yearly_returns.png")
    return recovery


def holding_period_runs(weights: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    rows = []
    for symbol in weights:
        state = np.sign(weights[symbol]).where(weights[symbol].abs() > threshold, 0)
        groups = state.ne(state.shift()).cumsum()
        for _, episode in state.groupby(groups):
            if episode.iloc[0] == 0:
                continue
            rows.append({"symbol": symbol, "start_date": episode.index[0].date().isoformat(),
                         "end_date": episode.index[-1].date().isoformat(),
                         "holding_days": len(episode), "start_weight": float(weights.loc[episode.index[0], symbol]),
                         "right_censored": bool(episode.index[-1] == weights.index[-1])})
    return pd.DataFrame(rows)


def write_portfolio_diagnostics(results: dict[str, core.BacktestResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    concentration_rows = []
    holding_summary_rows = []
    holding_detail_frames = []
    top_holding_frames = []

    for name, result in results.items():
        weights = result.effective_weights.fillna(0.0)
        abs_weights = weights.abs()
        gross = abs_weights.sum(axis=1)
        gross_nonzero = gross.replace(0.0, np.nan)
        sorted_abs = np.sort(abs_weights.to_numpy(), axis=1)[:, ::-1]
        top1 = pd.Series(sorted_abs[:, 0], index=weights.index)
        top3 = pd.Series(sorted_abs[:, :3].sum(axis=1), index=weights.index)
        normalized_abs = abs_weights.div(gross_nonzero, axis=0).fillna(0.0)
        hhi = (normalized_abs**2).sum(axis=1)
        position_count = (abs_weights > 0.01).sum(axis=1)

        concentration_rows.append(
            {
                "strategy": name,
                "avg_gross_exposure": float(gross.mean()),
                "max_gross_exposure": float(gross.max()),
                "avg_position_count_over_1pct": float(position_count.mean()),
                "avg_top1_abs_weight": float(top1.mean()),
                "max_top1_abs_weight": float(top1.max()),
                "avg_top3_abs_weight": float(top3.mean()),
                "avg_top1_share_of_gross": float((top1 / gross_nonzero).mean()),
                "avg_top3_share_of_gross": float((top3 / gross_nonzero).mean()),
                "avg_hhi_abs_weight_share": float(hhi.mean()),
                "max_hhi_abs_weight_share": float(hhi.max()),
            }
        )

        runs = holding_period_runs(result.effective_weights)
        if not runs.empty:
            runs.insert(0, "strategy", name)
            holding_detail_frames.append(runs)
            holding_summary_rows.append(
                {
                    "strategy": name,
                    "position_runs": int(len(runs)),
                    "avg_holding_days": float(runs["holding_days"].mean()),
                    "median_holding_days": float(runs["holding_days"].median()),
                    "min_holding_days": int(runs["holding_days"].min()),
                    "max_holding_days": int(runs["holding_days"].max()),
                }
            )
        else:
            holding_summary_rows.append(
                {
                    "strategy": name,
                    "position_runs": 0,
                    "avg_holding_days": np.nan,
                    "median_holding_days": np.nan,
                    "min_holding_days": np.nan,
                    "max_holding_days": np.nan,
                }
            )

        top_holdings = pd.DataFrame(
            {
                "symbol": abs_weights.columns,
                "mean_abs_weight": abs_weights.mean(axis=0).to_numpy(),
                "max_abs_weight": abs_weights.max(axis=0).to_numpy(),
                "active_day_share_over_1pct": (abs_weights > 0.01).mean(axis=0).to_numpy(),
            }
        )
        top_holdings.insert(0, "strategy", name)
        top_holding_frames.append(top_holdings.sort_values("mean_abs_weight", ascending=False))

    concentration = pd.DataFrame(concentration_rows)
    holding_summary = pd.DataFrame(holding_summary_rows)
    holding_details = (
        pd.concat(holding_detail_frames, ignore_index=True)
        if holding_detail_frames
        else pd.DataFrame(columns=["strategy", "symbol", "start_date", "end_date", "holding_days", "start_weight"])
    )
    top_holdings = pd.concat(top_holding_frames, ignore_index=True)

    concentration.to_csv(TABLES_DIR / "portfolio_concentration.csv", index=False)
    holding_summary.to_csv(TABLES_DIR / "holding_period_summary.csv", index=False)
    holding_details.to_csv(TABLES_DIR / "holding_periods.csv", index=False)
    top_holdings.to_csv(TABLES_DIR / "top_holdings_concentration.csv", index=False)
    plot_top_holdings(top_holdings, FIGURES_DIR / "top_holdings_concentration.png")
    return concentration, holding_summary


def plot_top_holdings(top_holdings: pd.DataFrame, output_path: Path) -> None:
    primary = top_holdings[top_holdings["strategy"] == PRIMARY_STRATEGY]
    if primary.empty:
        primary = top_holdings[top_holdings["strategy"] == "Full Alpha Trinity saved"]
    primary = primary.head(12).sort_values("mean_abs_weight")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(primary["symbol"], primary["mean_abs_weight"], color="#9467bd", alpha=0.82)
    ax.set_title("Primary Full Model: Average Absolute Top Holdings")
    ax.set_xlabel("Average absolute weight")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, axis="x", alpha=0.30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_yearly_returns(yearly: pd.DataFrame, output_path: Path) -> None:
    pivot = yearly.pivot(index="period", columns="strategy", values="net_return")
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Yearly Net Returns")
    ax.set_ylabel("Net return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, axis="y", alpha=0.30)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_system_architecture(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis("off")
    boxes = [
        (0.03, 0.62, 0.18, 0.18, "Adjusted OHLCV\n+ VIX data"),
        (0.28, 0.62, 0.18, 0.18, "Feature builder\ntrailing windows"),
        (0.53, 0.72, 0.18, 0.16, "MLP / RF\nprediction engine"),
        (0.53, 0.43, 0.18, 0.16, "Deterministic\ncrisis engine"),
        (0.77, 0.58, 0.18, 0.20, "Portfolio weights\n+ regime exposure"),
        (0.77, 0.25, 0.18, 0.16, "Execution audit\nlag + costs"),
    ]
    for x, y, width, height, label in boxes:
        rect = plt.Rectangle((x, y), width, height, fc="#e8f1f8", ec="#1f4e79", lw=1.8)
        ax.add_patch(rect)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=11, weight="bold")

    arrows = [
        ((0.21, 0.71), (0.28, 0.71)),
        ((0.46, 0.71), (0.53, 0.80)),
        ((0.46, 0.71), (0.53, 0.51)),
        ((0.71, 0.80), (0.77, 0.68)),
        ((0.71, 0.51), (0.77, 0.63)),
        ((0.86, 0.58), (0.86, 0.41)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#333333"})
    ax.text(0.50, 0.12, "Signal: close t. Fill: close t+1. First holding return: ending t+2.", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_turnover_costs(results: dict[str, core.BacktestResult], output_path: Path) -> None:
    primary = results[PRIMARY_STRATEGY]
    turnover = primary.turnover.fillna(0.0)
    gross_exposure = primary.effective_weights.abs().sum(axis=1).reindex(turnover.index).fillna(0.0)
    trading_cost = turnover * core.TRANSACTION_COST
    carry_cost = (gross_exposure - 1.0).clip(lower=0.0) * core.COST_OF_CARRY / 252.0
    cumulative_cost = (trading_cost + carry_cost).cumsum()

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.fill_between(turnover.index, 0, turnover.values, color="#9ecae1", alpha=0.65, label="Daily turnover")
    ax1.set_ylabel("Daily turnover")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(cumulative_cost.index, cumulative_cost.values * 100.0, color="#b22222", lw=2.0, label="Cumulative modeled cost")
    ax2.set_ylabel("Cumulative modeled cost (%)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    ax1.set_title("Turnover and Modeled Trading Costs")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_return_by_regime(regime_summary: pd.DataFrame, output_path: Path) -> None:
    table = regime_summary[regime_summary["regime"].isin(["growth", "caution", "panic"])].copy()
    order = ["growth", "caution", "panic"]
    table["regime"] = pd.Categorical(table["regime"], order, ordered=True)
    table = table.sort_values("regime")

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.bar(table["regime"].astype(str), table["net_return_primary_full"], color=["#2ca02c", "#ffbf00", "#d62728"], alpha=0.82)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax1.set_ylabel("Net return in regime")
    ax1.grid(True, axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(table["regime"].astype(str), table["avg_gross_exposure"], color="#1f4e79", marker="o", lw=2.0)
    ax2.set_ylabel("Average gross exposure")
    ax1.set_title("Return and Exposure by Regime")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_gross_exposure(results: dict[str, core.BacktestResult], output_path: Path) -> None:
    primary = results[PRIMARY_STRATEGY]
    equity = (1.0 + primary.net_returns.fillna(0.0)).cumprod()
    gross_exposure = primary.effective_weights.abs().sum(axis=1).reindex(equity.index).fillna(0.0)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(equity.index, equity.values, color="#1f4e79", lw=2.0)
    axes[0].set_ylabel("Net equity")
    axes[0].grid(True, alpha=0.25)
    axes[0].set_title("Primary Full Model: Equity and Gross Exposure")

    axes[1].plot(gross_exposure.index, gross_exposure.values, color="#b22222", lw=1.8)
    axes[1].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Gross exposure")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_drawdown_chart(results: dict[str, core.BacktestResult], output_path: Path) -> None:
    selected = [PRIMARY_STRATEGY, "SPY buy-and-hold", "60/40 SPY/AGG"]
    fig, ax = plt.subplots(figsize=(11, 5))
    for name in selected:
        if name not in results:
            continue
        dd = drawdown_series(results[name].net_returns).fillna(0.0)
        ax.plot(dd.index, dd.values, lw=1.8, label=name)
    ax.set_title("Underwater Plot: Primary Strategy vs Benchmarks")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_rolling_risk_chart(results: dict[str, core.BacktestResult], output_path: Path) -> None:
    selected = [PRIMARY_STRATEGY, "SPY buy-and-hold", "60/40 SPY/AGG"]
    fig, ax = plt.subplots(figsize=(11, 5))
    for name in selected:
        if name not in results:
            continue
        rolling_vol = results[name].net_returns.rolling(60).std() * np.sqrt(core.PERIODS_PER_YEAR)
        ax.plot(rolling_vol.index, rolling_vol.values, lw=1.8, label=name)
    ax.set_title("60-Day Rolling Annualized Volatility")
    ax.set_ylabel("Annualized volatility")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_report_figures(regime_summary: pd.DataFrame, results: dict[str, core.BacktestResult]) -> None:
    plot_system_architecture(FIGURES_DIR / "system_architecture.png")
    plot_turnover_costs(results, FIGURES_DIR / "turnover_costs.png")
    plot_return_by_regime(regime_summary, FIGURES_DIR / "return_by_regime.png")
    plot_gross_exposure(results, FIGURES_DIR / "gross_exposure_over_time.png")
    plot_drawdown_chart(results, FIGURES_DIR / "drawdown_chart.png")
    plot_rolling_risk_chart(results, FIGURES_DIR / "rolling_risk_chart.png")


def bootstrap_path(values: np.ndarray, rng: np.random.Generator, block_len: int, horizon: int) -> np.ndarray:
    starts = np.arange(0, max(1, len(values) - block_len + 1))
    chunks = []
    while sum(len(chunk) for chunk in chunks) < horizon:
        start = int(rng.choice(starts))
        chunks.append(values[start : start + block_len])
    return np.concatenate(chunks)[:horizon]


def block_bootstrap(results: dict[str, core.BacktestResult], samples: int = 1000, block_len: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    selected = [PRIMARY_STRATEGY, "Prediction-only MLP/RF dynamic VIX", "SPY buy-and-hold"]
    for name in selected:
        values = results[name].net_returns.fillna(0.0).to_numpy()
        total_returns = []
        max_drawdowns = []
        sharpes = []
        for _ in range(samples):
            path = bootstrap_path(values, rng, block_len=block_len, horizon=len(values))
            equity = np.cumprod(1.0 + path)
            total_returns.append(equity[-1] - 1.0)
            max_drawdowns.append(np.min(equity / np.maximum(np.maximum.accumulate(equity), 1.0) - 1.0))
            vol = np.std(path, ddof=1) * np.sqrt(core.PERIODS_PER_YEAR)
            ann = np.mean(path) * core.PERIODS_PER_YEAR
            sharpes.append((ann - 0.04) / vol if vol > 0 else np.nan)
        rows.append(
            {
                "strategy": name,
                "samples": samples,
                "block_length_days": block_len,
                "total_return_p05": float(np.quantile(total_returns, 0.05)),
                "total_return_p50": float(np.quantile(total_returns, 0.50)),
                "total_return_p95": float(np.quantile(total_returns, 0.95)),
                "max_drawdown_p05": float(np.quantile(max_drawdowns, 0.05)),
                "max_drawdown_p50": float(np.quantile(max_drawdowns, 0.50)),
                "max_drawdown_p95": float(np.quantile(max_drawdowns, 0.95)),
                "sharpe_4rf_p05": float(np.nanquantile(sharpes, 0.05)),
                "sharpe_4rf_p50": float(np.nanquantile(sharpes, 0.50)),
                "sharpe_4rf_p95": float(np.nanquantile(sharpes, 0.95)),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES_DIR / "block_bootstrap_summary.csv", index=False)
    plot_bootstrap_summary(summary, FIGURES_DIR / "block_bootstrap_summary.png")
    return summary


def plot_bootstrap_summary(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary))
    med = summary["total_return_p50"].to_numpy()
    low = med - summary["total_return_p05"].to_numpy()
    high = summary["total_return_p95"].to_numpy() - med
    ax.bar(x, med, yerr=[low, high], color="#1f77b4", alpha=0.8, capsize=5)
    ax.set_xticks(x, summary["strategy"], rotation=20, ha="right")
    ax.set_ylabel("Bootstrapped total return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Block Bootstrap Total Return Range")
    ax.grid(True, axis="y", alpha=0.30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def walk_forward_prediction_validation(feature_df: pd.DataFrame, eval_index: pd.Index) -> pd.DataFrame:
    rows = []
    portfolio_rows = []
    daily_returns = {}
    returns = core.pivot_returns(feature_df)
    scores = core.calculate_crisis_scores(feature_df, returns.index)
    vix = feature_df.groupby("Date")["VIX"].mean().sort_index()
    for mode in ["expanding", "rolling_2y"]:
        signal_parts = []
        for year in sorted(eval_index.year.unique()):
            print(f"Walk-forward {mode}: {year}", flush=True)
            fold_start = pd.Timestamp(f"{year}-01-01")
            fold_dates = eval_index[eval_index.year == year]
            cutoff = fold_start - pd.Timedelta(days=1)
            train_start = pd.Timestamp(f"{year - 2}-01-01") if mode == "rolling_2y" else None
            diagnostics, _ = fit_prediction_models(feature_df, train_end=cutoff, train_start=train_start)
            row, _ = summarize_prediction_quality(diagnostics, fold_dates, "pred_ensemble")
            row.update(mode=mode, fold_year=int(year), train_end=cutoff.date().isoformat(),
                       train_start=str(train_start), model_count=10,
                       fold_start=fold_dates.min().date().isoformat(), fold_end=fold_dates.max().date().isoformat())
            rows.append(row)
            signal_parts.append(signal_from_prediction_diagnostics(diagnostics).loc[fold_dates])
        signal = pd.concat(signal_parts).sort_index()
        for use_crisis in [True, False]:
            name = f"{mode} {'full' if use_crisis else 'prediction-only'}"
            weights = core.build_engine_weights(name, returns, signal, scores, vix, True, use_crisis)
            result = core.evaluate_weights(name, weights, returns)
            portfolio_rows.append(core.metric_row(result, "walk_forward"))
            daily_returns[name] = result.net_returns
    # Alternative initial training split, evaluated against the original model on identical dates.
    common = eval_index[eval_index >= pd.Timestamp("2023-01-01")]
    for cutoff in [core.TRAIN_END_DATE, pd.Timestamp("2022-12-31")]:
        diagnostics, _ = fit_prediction_models(feature_df, train_end=cutoff)
        signal = signal_from_prediction_diagnostics(diagnostics)
        weights = core.build_engine_weights("split", returns, signal, scores, vix, True, True,
                                             start_date=common.min())
        result = core.evaluate_weights(f"Train through {cutoff.year}; test 2023+", weights, returns.loc[common])
        portfolio_rows.append(core.metric_row(result, "initial_split"))
    output = pd.DataFrame(rows)
    output.to_csv(TABLES_DIR / "walk_forward_prediction_validation.csv", index=False)
    pd.DataFrame(portfolio_rows).to_csv(TABLES_DIR / "walk_forward_portfolio.csv", index=False)
    pd.DataFrame(daily_returns).to_csv(TABLES_DIR / "walk_forward_daily_returns.csv")
    return output


def advanced_variants(inputs: dict[str, object], diagnostics: pd.DataFrame) -> None:
    returns = inputs["returns"]
    signal = signal_from_prediction_diagnostics(diagnostics)
    full = core.build_engine_weights("confidence", returns, signal, inputs["crisis_scores"], inputs["daily_vix"], True, True)
    disagreement = diagnostics.pivot(index="Date", columns="symbol", values="model_disagreement_abs")
    confidence = 1.0 / (1.0 + disagreement / 0.05)
    weighted = full * confidence.reindex_like(full).fillna(1.0)
    simple_assets = ["SPY", "EFA", "AGG", "GLD", "UUP"]
    simple = core.momentum_top_k_weights(returns[simple_assets], top_k=2).loc[inputs["common_index"]]
    panic = inputs["crisis_scores"] >= 0.65 - 1e-12
    simple.loc[panic] = 0.0
    simple.loc[panic, ["GLD", "UUP"]] = 0.4
    top_k = pd.DataFrame(0.0, index=inputs["common_index"], columns=returns.columns)
    for date in top_k.index:
        chosen = signal.loc[date].nlargest(core.TOP_K).index
        top_k.loc[date, chosen] = 1 / len(chosen)
    results = [core.evaluate_weights("Model-confidence weighting", weighted, returns),
               core.evaluate_weights("Simple five-ETF momentum/crisis", simple, returns),
               core.evaluate_weights("Prediction long-only Top-5", top_k, returns)]
    pd.DataFrame([core.metric_row(r, "advanced") for r in results]).to_csv(TABLES_DIR / "advanced_variants.csv", index=False)


def write_final_docs(
    prediction_metrics: pd.DataFrame,
    feature_group_ablation: pd.DataFrame,
    regime_summary: pd.DataFrame,
    recovery: pd.DataFrame,
    bootstrap: pd.DataFrame,
    walk_forward: pd.DataFrame,
    concentration: pd.DataFrame,
    holding_summary: pd.DataFrame,
) -> None:
    text = """# Corrected Evidence Status

The canonical pipeline rebuilds all portfolio weights from the delivered snapshot,
uses next-close execution and self-financing drift-aware costs, and purges labels
for the actual prediction horizon. No saved legacy weight path is a research input.

Evidence includes seven benchmarks, component and feature-group ablations,
all sensitivity families, prediction and permutation diagnostics, regime analytics,
calendar returns, recovery and holding episodes, concentration, block bootstrap,
full ten-model annual expanding/rolling portfolio validation, confidence sizing
and a simpler five-ETF rule. See final_assessment_checklist.md for the mapping.

Results do not establish universal robustness. Walk-forward results weaken the
fixed-split conclusion; original data acquisition time, point-in-time survivorship,
historical strategy selection, real execution and multiple-testing uncertainty
remain disclosed limitations. They cannot be repaired by relabeling this history
as pristine out-of-sample. No formal White reality check is claimed.

Build the paper with make paper after make assessment. A successful pipeline alone
is not a substitute for reviewing the manuscript and compiled PDF.
"""
    (DOCS_DIR / "assessment_completion_status.md").write_text(text, encoding="utf-8")
    (DOCS_DIR / "remaining_assessment_gaps.md").write_text(
        "# Remaining Scientific Boundaries\n\n"
        "Original acquisition time is unknown; exact snapshot reproduction is supplied.\n"
        "No pristine prospective holdout, point-in-time delisted universe, broker execution validation "
        "or formal multiple-testing correction is claimed. These limitations are explicit in the paper.\n"
        "The requested retrospective analyses have executable evidence; negative outcomes are retained.\n",
        encoding="utf-8")


def write_manifest() -> None:
    outputs = sorted(str(path.relative_to(ASSESSMENT_DIR)) for path in ASSESSMENT_DIR.rglob("*") if path.is_file())
    manifest = {"generated": datetime.now().isoformat(timespec="seconds"), "outputs": outputs}
    (MANIFESTS_DIR / "completion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    print("Preparing assessment inputs with dynamic VIX mapping...")
    inputs = prepare_inputs()

    print("Running prediction diagnostics and feature importance...")
    diagnostics, prediction_metrics, importance = write_prediction_outputs(inputs)
    advanced_variants(inputs, diagnostics)
    write_feature_outputs(inputs, importance)
    feature_group_ablation = write_feature_group_ablation_outputs(inputs)

    print("Running regime, return, and drawdown analytics...")
    regime_summary, _, results = write_regime_outputs(inputs, diagnostics)
    recovery = write_return_and_drawdown_outputs(results)
    concentration, holding_summary = write_portfolio_diagnostics(results)
    write_report_figures(regime_summary, results)

    print("Running block bootstrap and walk-forward diagnostics...")
    bootstrap = block_bootstrap(results)
    walk_forward = walk_forward_prediction_validation(inputs["feature_df"], inputs["common_index"])

    write_final_docs(
        prediction_metrics,
        feature_group_ablation,
        regime_summary,
        recovery,
        bootstrap,
        walk_forward,
        concentration,
        holding_summary,
    )
    write_manifest()
    print(f"Done. Completion outputs written to {ASSESSMENT_DIR}")


if __name__ == "__main__":
    main()
