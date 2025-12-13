from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml

from ai_ls_allocation.models.dummy_quantile import DummyQuantileModel

PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT / "data"
FEAT_DIR = DATA_DIR / "features"
CONFIG_DIR = PROJECT_ROOT / "config"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_universe(kind: str = "etf") -> List[str]:
    """Liest die Symbol-Liste aus config/universes.yaml."""
    with open(CONFIG_DIR / "universes.yaml") as f:
        cfg = yaml.safe_load(f)
    u = cfg[kind]
    return u["symbols"]


def pinball_loss(y: pd.Series, q: pd.Series, alpha: float) -> float:
    """
    Pinball-Loss für ein gegebenes Quantil alpha.
    Standard-Definition: immer >= 0.
    """
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    diff = y - q
    loss = np.where(diff < 0, (alpha - 1.0) * diff, alpha * diff)
    return float(np.mean(loss))


def evaluate_symbol(symbol: str, window: int = 60) -> pd.DataFrame:
    """
    Lädt Features für ein Symbol, berechnet Dummy-Quantile und misst
    Coverage & Pinball-Loss für Q10/Q50/Q90.
    """
    fp = FEAT_DIR / f"{symbol}.parquet"
    if not fp.exists():
        print(f"[calibration] skip {symbol}: no features at {fp}")
        return pd.DataFrame()

    df = pd.read_parquet(fp)

    if "r1" not in df.columns:
        print(f"[calibration] skip {symbol}: no 'r1' column in {fp}")
        return pd.DataFrame()

    df = df.copy()
    df["r1"] = pd.to_numeric(df["r1"], errors="coerce").astype(float)

    model = DummyQuantileModel(window=window)
    q = model.predict(df)

    eval_df = pd.concat([df["r1"].rename("r1"), q], axis=1).dropna()

    if eval_df.empty:
        print(f"[calibration] {symbol}: no rows after dropna (window too short?)")
        return pd.DataFrame()

    rows = []
    for alpha, col in zip((0.1, 0.5, 0.9), ("q10", "q50", "q90")):
        if col not in eval_df.columns:
            continue
        y = eval_df["r1"]
        qhat = eval_df[col]

        coverage = float((y <= qhat).mean())
        pl = pinball_loss(y, qhat, alpha)

        rows.append(
            dict(
                symbol=symbol,
                quantile=alpha,
                coverage=coverage,
                target=alpha,
                pinball=pl,
                n=len(eval_df),
            )
        )

    return pd.DataFrame(rows)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    symbols = load_universe("etf")
    all_rows = []

    for s in symbols:
        print(f"[calibration] evaluating {s}")
        df_sym = evaluate_symbol(s)
        if not df_sym.empty:
            all_rows.append(df_sym)

    if not all_rows:
        print("[calibration] no data to aggregate, nothing to write.")
        return

    df_all = pd.concat(all_rows, ignore_index=True)

    out_path = REPORTS_DIR / "calibration_summary.csv"
    df_all.to_csv(out_path, index=False)
    print(f"[calibration] wrote {out_path}")

    agg = (
        df_all.groupby("quantile")
        .agg(
            coverage=("coverage", "mean"),
            target=("target", "mean"),
            pinball=("pinball", "mean"),
            n=("n", "sum"),
        )
        .reset_index()
    )

    out_global = REPORTS_DIR / "calibration_summary_global.csv"
    agg.to_csv(out_global, index=False)
    print(f"[calibration] wrote {out_global}")

    try:
        import matplotlib.pyplot as plt

        if not agg.empty:
            fig, ax = plt.subplots()
            ax.bar(agg["quantile"].astype(str), agg["coverage"], label="empirical")
            for alpha in agg["quantile"]:
                ax.axhline(y=alpha, linestyle="--", linewidth=1)

            ax.set_xlabel("Quantile level")
            ax.set_ylabel("Coverage")
            ax.set_title("Global empirical coverage vs target")
            fig.tight_layout()
            fig.savefig(REPORTS_DIR / "calibration_coverage_global.png")
            plt.close(fig)
            print("[calibration] wrote calibration_coverage_global.png")
        else:
            print("[calibration] no aggregated data for coverage plot, skipping.")
    except Exception as e:
        print(f"[calibration] plot failed: {e}")


if __name__ == "__main__":
    main()
