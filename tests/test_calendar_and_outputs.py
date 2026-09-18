from pathlib import Path

import pandas as pd


def test_core_feature_and_signal_outputs_exist():
    required_paths = [
        Path("assessment/inputs/panel.parquet"),
        Path("assessment/inputs/provenance.json"),
    ]

    for path in required_paths:
        assert path.exists(), f"missing required pipeline output: {path}"

    panel = pd.read_parquet(required_paths[0]).reset_index()

    if "date" in panel.columns:
        panel = panel.rename(columns={"date": "Date"})
    panel["Date"] = pd.to_datetime(panel["Date"]).dt.tz_localize(None)

    assert not panel.empty
    assert panel["Date"].max() == pd.Timestamp("2026-05-08")


def test_assessment_outputs_cover_required_evidence():
    required_paths = [
        Path("README.md"),
        Path("assessment/config/protocol.json"),
        Path("assessment/docs/reproducibility.md"),
        Path("assessment/docs/feature_documentation.md"),
        Path("assessment_run/writing/abgegebenes_PAPER.tex"),
        Path("assessment_run/writing/abgegebenes_PAPER.pdf"),
        Path("assessment/outputs/tables/baseline_comparison.csv"),
        Path("assessment/outputs/tables/ablation_study.csv"),
        Path("assessment/outputs/tables/robustness_sensitivity.csv"),
        Path("assessment/outputs/tables/prediction_metrics.csv"),
        Path("assessment/outputs/tables/walk_forward_prediction_validation.csv"),
        Path("assessment/outputs/tables/walk_forward_portfolio.csv"),
    ]

    for path in required_paths:
        assert path.exists(), f"missing required assessment evidence: {path}"
        assert path.stat().st_size > 0

    paper = Path("assessment_run/writing/abgegebenes_PAPER.tex").read_text()
    assert "The research question is:" in paper
    assert r"\subsection{Execution and Self-Financing Costs}" in paper
    assert r"\section{Limitations}" in paper
