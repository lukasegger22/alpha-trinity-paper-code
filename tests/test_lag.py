import sys
from pathlib import Path

import pandas as pd


ASSESSMENT_SRC = Path(__file__).resolve().parents[1] / "assessment" / "src"
sys.path.insert(0, str(ASSESSMENT_SRC))

from alpha_trinity_assessment.first_three import evaluate_weights


def test_assessment_evaluator_executes_weights_one_day_late():
    dates = pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"])
    returns = pd.DataFrame(
        {
            "A": [0.10, 0.20, -0.10],
            "B": [0.00, 0.00, 0.00],
        },
        index=dates,
    )
    signal_weights = pd.DataFrame(
        {
            "A": [1.00, 0.00, 0.00],
            "B": [0.00, 0.00, 0.00],
        },
        index=dates,
    )

    result = evaluate_weights(
        "lag test",
        signal_weights,
        returns,
        transaction_cost=0.0,
        cost_of_carry=0.0,
    )

    expected_returns = pd.Series([0.0, 0.0, -0.10], index=dates)
    pd.testing.assert_series_equal(
        result.net_returns,
        expected_returns,
        check_names=False,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        result.effective_weights,
        signal_weights.shift(2).fillna(0.0),
        check_dtype=False,
    )
