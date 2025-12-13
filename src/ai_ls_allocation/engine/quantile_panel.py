from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_DIR = Path("data")
FEAT_FP = DATA_DIR / "features" / "panel.parquet"
REPORTS_DIR = Path("reports")

# Bis hierhin trainieren, danach Test
TRAIN_END = pd.Timestamp("2017-12-31")

QUANTILES = [0.1, 0.5, 0.9]

FEAT_COLS = [
    "vol20",
    "vol60",
    "ma20",
    "ma60",
    "trend_ma_diff",
    "macd",
    "macd_signal",
    "macd_hist",
    "spread_hyg_ief",
]


def pinball_loss(y: np.ndarray, y_hat: np.ndarray, q: float) -> float:
    """
    Pinball-Loss für Quantil q.
    y, y_hat: 1D-Arrays gleicher Länge.
    """
    diff = y - y_hat
    # Wenn y >= y_hat: q * (y - y_hat)
    # Sonst: (1-q) * (y_hat - y)
    return np.mean(np.where(diff >= 0, q * diff, (1.0 - q) * -diff))


def main():
    REPORTS_DIR.mkdir(exist_ok=True)

    print(f"[load] {FEAT_FP}")
    panel = pd.read_parquet(FEAT_FP).reset_index()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "symbol"])

    # Nur Zeilen, wo Target & Features alle definiert sind
    cols_needed = ["r1"] + FEAT_COLS
    panel = panel.dropna(subset=cols_needed)

    # Train/Test-Split nach Datum
    train = panel[panel["date"] <= TRAIN_END].copy()
    test = panel[panel["date"] > TRAIN_END].copy()

    print(f"[split] train rows: {len(train)}, test rows: {len(test)}")

    X_train = train[FEAT_COLS]
    y_train = train["r1"]

    # Konstante für lineares Modell
    X_train = sm.add_constant(X_train)

    models = {}
    for q in QUANTILES:
        print(f"[fit] Quantile {q}")
        mod = sm.QuantReg(y_train, X_train)
        res = mod.fit(q=q)
        models[q] = res
        print(f"  -> done, nobs={int(res.nobs)}")

    # Vorhersage auf Test
    X_test = sm.add_constant(test[FEAT_COLS], has_constant="add")

    for q, res in models.items():
        col = f"q_{int(q * 100):02d}"
        test[col] = res.predict(X_test)

    # Evaluation global über alle Symbole
    rows = []
    y_true = test["r1"].values

    for q in QUANTILES:
        col = f"q_{int(q * 100):02d}"
        y_hat = test[col].values

        coverage = float((y_true <= y_hat).mean())  # Anteil y <= q_hat
        pb = float(pinball_loss(y_true, y_hat, q))

        rows.append(
            dict(
                quantile=q,
                coverage=coverage,
                target=q,
                pinball=pb,
                n=len(y_true),
            )
        )

    eval_df = pd.DataFrame(rows)
    out_fp = REPORTS_DIR / "quantile_panel_eval.csv"
    eval_df.to_csv(out_fp, index=False)
    print(f"[done] wrote {out_fp}")
    print(eval_df)


if __name__ == "__main__":
    main()
