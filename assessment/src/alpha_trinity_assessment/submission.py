"""Build manuscript tables and a consistent, self-contained submission from CSV evidence."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ASSESSMENT = ROOT / "assessment"
TABLES = ASSESSMENT / "outputs/tables"
WRITING = ROOT / "assessment_run/writing"
GENERATED = WRITING / "generated"
PAPER_FIGURES = (
    "system_architecture.png", "crisis_score.png", "baseline_equity_curves.png",
    "drawdown_chart.png", "gross_exposure_over_time.png", "turnover_costs.png",
    "return_by_regime.png", "threshold_sensitivity_heatmap.png",
)
FULL = "Full Alpha Trinity rebuilt dynamic VIX"
PRED = "Prediction-only MLP/RF dynamic VIX"
NAMES = {
    FULL: "Full Alpha Trinity", "Full Alpha Trinity rebuilt": "Full Alpha Trinity",
    PRED: "Prediction-only", "Prediction-only MLP/RF": "Prediction-only",
    "Crisis-only deterministic": "Crisis-only",
    "Full Alpha Trinity rebuilt no-cost": "No-cost (fixed signals)",
    "Full Alpha Trinity no-leverage": "No-leverage (fixed signals)",
    "Risk parity inverse-vol": "Inverse-volatility proxy",
    "Defensive basket buy-and-hold": "Defensive buy-and-hold",
    "all_features_diagnostic": "All features", "drop_momentum": "Without returns",
    "drop_risk": "Without volatility/VIX", "drop_volume": "Without volume",
    "drop_mean_reversion": "Without channels",
    "Train through 2021; test 2023+": "Train to 2021 / test 2023+",
    "Train through 2022; test 2023+": "Train to 2022 / test 2023+",
}


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv")


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def num(value: float) -> str:
    return f"{value:.2f}"


def table(name: str, frame: pd.DataFrame, caption: str) -> None:
    text = frame.to_latex(index=False, escape=True, caption=caption,
                         label=f"tab:{name}", position="htbp")
    text = text.replace("\\centering", "\\centering\n\\small")
    (GENERATED / f"{name}.tex").write_text(text, encoding="utf-8")


def performance(frame: pd.DataFrame, name: str, caption: str, key: str = "strategy") -> None:
    frame = frame.copy()
    result = pd.DataFrame({"Strategy / variant": frame[key].replace(NAMES),
                           "Net return": frame.net_return.map(pct),
                           "Sharpe": frame.sharpe_4rf.map(num),
                           "Max DD": frame.max_drawdown.map(pct),
                           "Turnover": frame.total_turnover.map(num)})
    table(name, result, caption)


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    baseline, ablation = read("baseline_comparison"), read("ablation_study")
    full = ablation.set_index("strategy").loc["Full Alpha Trinity rebuilt"]
    pred = ablation.set_index("strategy").loc["Prediction-only MLP/RF"]
    spy = baseline.set_index("strategy").loc["SPY buy-and-hold"]
    # Both analysis paths must agree before writing any paper numbers.
    yearly = read("yearly_returns")
    for name, row in [(FULL, full), (PRED, pred), ("SPY buy-and-hold", spy)]:
        value = np.prod(1 + yearly.loc[yearly.strategy == name, "net_return"]) - 1
        if not np.isclose(value, row.net_return, atol=1e-10):
            raise ValueError(f"Inconsistent primary/completion results: {name}")
    macros = {}
    for prefix, row in [("Full", full), ("Pred", pred), ("Spy", spy)]:
        for key, column in [("Return", "net_return"), ("Drawdown", "max_drawdown"),
                            ("Cagr", "cagr"), ("Volatility", "annualized_volatility")]:
            macros[prefix + key] = pct(row[column]).replace("%", r"\%")
        macros[prefix + "Sharpe"] = num(row.sharpe_4rf)
    macros["CostDrag"] = num(full.cost_drag_compounded * 100)
    macros["FullTurnover"] = num(full.total_turnover)
    macros["FullExposure"] = num(full.avg_gross_exposure)
    (GENERATED / "results.tex").write_text("\n".join(
        rf"\newcommand{{\{key}}}{{{value}}}" for key, value in macros.items()) + "\n", encoding="utf-8")

    fields = [("Net return", "net_return", pct), ("CAGR", "cagr", pct),
              ("Annualized arithmetic return", "annualized_return_arithmetic", pct),
              ("Annualized volatility", "annualized_volatility", pct),
              ("Downside deviation, MAR 4%", "downside_deviation_4rf", pct),
              ("Sharpe, 4% reference", "sharpe_4rf", num), ("Sortino, MAR 4%", "sortino_4rf", num),
              ("Calmar", "calmar", num), ("Maximum drawdown", "max_drawdown", pct),
              ("Cost drag (percentage points)", "cost_drag_compounded", lambda x: num(100*x)),
              ("Annualized turnover", "annualized_turnover", num),
              ("Average monthly turnover", "avg_monthly_turnover", num),
              ("Annualized additive costs", "annualized_cost_sum", pct),
              ("Executed asset trades (>1e-6 NAV)", "trade_events", lambda x: str(int(x))),
              ("Average gross exposure", "avg_gross_exposure", num)]
    table("metrics", pd.DataFrame([(label, fmt(spy[col]), fmt(full[col])) for label, col, fmt in fields],
                                   columns=["Metric", "SPY", "Full Alpha Trinity"]),
          "Core metrics, 3 January 2022 to 8 May 2026. Costs are modeled, not observed fills.")
    performance(pd.concat([ablation.iloc[[0]], baseline], ignore_index=True), "baselines", "Fair benchmark comparison with common dates, next-close execution, and costs.")
    performance(ablation, "ablations", "Component ablations. No-cost and no-leverage hold the full model's signals fixed.")
    robustness = read("robustness_sensitivity")
    summary = robustness.groupby("group").agg(runs=("net_return", "size"), minimum=("net_return", "min"), maximum=("net_return", "max"), worst_dd=("max_drawdown", "min")).reset_index()
    summary["group"] = summary.group.str.replace("_sensitivity", "").str.replace("_", " ")
    for col in ["minimum", "maximum", "worst_dd"]:
        summary[col] = summary[col].map(pct)
    summary.columns = ["Test family", "Runs", "Min return", "Max return", "Worst DD"]
    table("robustness", summary, "All sensitivity families. Ranges are descriptive, not a parameter-selection exercise.")
    selected = robustness[robustness.group.isin(["asset_universe_sensitivity", "safe_haven_sensitivity", "fixed_path_cost_sensitivity"])]
    performance(selected, "sensitivity-detail", "Asset removal, defensive alternatives and fixed-signal cost sensitivity.", "variant")
    forecast = read("prediction_metrics")
    table("prediction", pd.DataFrame({"Model": forecast.prediction_column.replace({
          "pred_ensemble": "Ensemble", "pred_mlp": "MLP", "pred_rf": "RF", "pred_ridge": "Ridge"}),
          "Daily rank IC": forecast.mean_daily_rank_correlation.map(lambda x: f"{x:.3f}"),
          "Top-5 hit": forecast.top_k_hit_rate.map(pct), "MAE": forecast.mae.map(pct), "RMSE": forecast.rmse.map(pct)}),
          "Prediction quality for realized 20-session labels; overlapping targets are not independent trials.")
    performance(read("feature_group_ablation"), "features-ablation", "Ten-model feature-group ablations with unchanged portfolio rules.", "variant")
    performance(read("walk_forward_portfolio"), "walk-forward", "Annual expanding and rolling-two-year portfolio validation; alternative initial splits use the same 2023--2026 test dates.")
    performance(read("advanced_variants"), "advanced", "Exploratory confidence sizing, simpler ETF strategy and prediction-only long-only Top-5.")

    year = yearly[yearly.strategy.isin([FULL, PRED, "SPY buy-and-hold", "60/40 SPY/AGG"])].pivot(index="period", columns="strategy", values="net_return")
    year.index = pd.to_datetime(year.index).year.astype(str)
    year.index = year.index.str.replace("2026", "2026 to May 8")
    year = year[[FULL, PRED, "SPY buy-and-hold", "60/40 SPY/AGG"]].map(pct).rename(columns=NAMES).reset_index()
    year.columns = ["Year", "Full", "Prediction-only", "SPY", "60/40"]
    table("yearly", year, "Calendar-year net returns; the last year is partial.")
    monthly = read("monthly_returns")
    monthly = monthly[monthly.strategy == FULL].set_index("period").net_return
    holding = read("holding_period_summary").set_index("strategy").loc[FULL]
    conc = read("portfolio_concentration").set_index("strategy").loc[FULL]
    monthly_rows = [("Mean monthly return", pct(monthly.mean())), ("Monthly standard deviation", pct(monthly.std())),
                    ("10th / 90th monthly percentiles", pct(monthly.quantile(.1)) + " / " + pct(monthly.quantile(.9))),
                    ("Best month", monthly.idxmax()[:7] + ": " + pct(monthly.max())),
                    ("Worst month", monthly.idxmin()[:7] + ": " + pct(monthly.min())),
                    ("Positive month share", pct((monthly > 0).mean())),
                    ("Mean / median holding sessions", num(holding.avg_holding_days) + " / " + num(holding.median_holding_days)),
                    ("Average top-1 / top-3 absolute weights", pct(conc.avg_top1_abs_weight) + " / " + pct(conc.avg_top3_abs_weight)),
                    ("Maximum absolute asset weight", pct(conc.max_top1_abs_weight))]
    table("monthly", pd.DataFrame(monthly_rows, columns=["Diagnostic", "Full Alpha Trinity"]),
          "Monthly and concentration diagnostics. First/last months and unfinished holding episodes are retained.")
    regime = read("regime_analytics").query("regime != 'unassigned'")
    table("regime", pd.DataFrame({"Regime": regime.regime, "Signal share": regime.signal_day_share.map(pct),
                                   "Conditional return": regime.net_return_primary_full.map(pct),
                                   "Gross exposure": regime.avg_gross_exposure.map(num),
                                   "Costs (sum)": regime.cost_sum_on_return_days.map(pct)}),
          "Conditional regime analytics. Returns use the regime of the signal two sessions earlier; cost attribution uses those same return dates.")
    recovery = read("drawdown_recovery")
    recovery = recovery[recovery.strategy.isin([FULL, PRED, "SPY buy-and-hold"])].copy()
    table("recovery", pd.DataFrame({"Strategy": recovery.strategy.replace(NAMES), "Peak": recovery.peak_date,
                                     "Trough": recovery.trough_date, "Recovery": recovery.recovery_date.fillna("Not recovered"),
                                     "Calendar days": recovery.days_trough_to_recovery.map(lambda x: "Censored" if pd.isna(x) else str(int(x)))}),
          "Maximum-drawdown episodes. Different peak dates prevent a like-for-like causal claim about recovery speed.")
    scenarios = read("scenario_stress_tests")
    scenarios = scenarios[scenarios.strategy.isin([FULL, PRED, "SPY buy-and-hold"])].pivot(index="scenario", columns="strategy", values="net_return")
    scenarios = scenarios[[FULL, PRED, "SPY buy-and-hold"]].map(pct).reset_index()
    scenarios.columns = ["Historical window", "Full", "Prediction-only", "SPY"]
    table("scenarios", scenarios, "Retrospectively named historical windows; dates are specified in the accompanying scenario CSV.")
    boot = read("block_bootstrap_summary")
    table("bootstrap", pd.DataFrame({"Strategy": boot.strategy.replace(NAMES), "Return p05": boot.total_return_p05.map(pct),
                                      "Return p95": boot.total_return_p95.map(pct), "Sharpe p05": boot.sharpe_4rf_p05.map(num),
                                      "Sharpe p95": boot.sharpe_4rf_p95.map(num)}),
          "Marginal 90-percent block-bootstrap ranges: 1000 paths, 20-session blocks, seed 42. No correction for strategy selection.")

    (WRITING / "figures").mkdir(parents=True, exist_ok=True)
    for name in PAPER_FIGURES:
        shutil.copyfile(ASSESSMENT / "outputs/figures" / name, WRITING / "figures" / name)
    manifest = {"python": platform.python_version(), "platform": platform.platform(),
                "inputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ASSESSMENT / "inputs").glob("*")) if p.is_file()},
                "paper_source": "assessment_run/writing/abgegebenes_PAPER.tex",
                "status": "Generated from corrected evidence; PDF must be built and checked separately."}
    manifest["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ASSESSMENT / "src").rglob("*.py"))}
    manifest["protocol"] = json.loads((ASSESSMENT / "config/protocol.json").read_text())
    (ASSESSMENT / "outputs/manifests/submission_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Manuscript tables generated and evidence synchronized.")


if __name__ == "__main__":
    main()
