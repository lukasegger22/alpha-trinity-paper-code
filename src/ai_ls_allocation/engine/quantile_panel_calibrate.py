"""
Quantile panel calibration:
- loads panel features (data/features/panel.parquet)
- splits by date into train/valid/test (60/20/20)
- fits QuantReg for q in {0.1, 0.5, 0.9} on TRAIN
- predicts RAW on VALID/TEST
- builds SHIFT calibration (q-quantile of residuals on VALID)
- builds LINEAR calibration (QuantReg of r1 ~ const + yhat_raw on VALID, applied to TEST)
- evaluates coverage & pinball loss on TEST for RAW/CAL/LIN
- writes reports/quantile_panel_eval.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---- paths / constants -------------------------------------------------------
DATA_DIR = Path("data")
FEAT_DIR = DATA_DIR / "features"
PANEL_FP = FEAT_DIR / "panel.parquet"

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

QS: List[float] = [0.1, 0.5, 0.9]

# must match your feature builder
FEATS: List[str] = [
    "vol20", "vol60",
    "ma20", "ma60", "trend_ma_diff",
    "macd", "macd_signal", "macd_hist",
    "spread_hyg_ief",
]


# ---- helpers ----------------------------------------------------------------
def _clean_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Ensure index/dtypes are sane and rows sorted."""
    if isinstance(panel.index, pd.MultiIndex):
        # expect names ('date', 'symbol')
        if panel.index.names != ["date", "symbol"]:
            panel.index.set_names(["date", "symbol"], inplace=True)
    else:
        # try to coerce: assume there are columns 'date' & 'symbol'
        if {"date", "symbol"}.issubset(panel.columns):
            panel = panel.set_index(["date", "symbol"])
        else:
            raise ValueError("panel must be MultiIndex (date, symbol) or have 'date'/'symbol' columns")

    # enforce datetime index at level 0
    idx_df = panel.index.to_frame(index=False)
    idx_df["date"] = pd.to_datetime(idx_df["date"])
    panel.index = pd.MultiIndex.from_frame(idx_df, names=["date", "symbol"])

    # sort
    panel = panel.sort_index()
    return panel


def _date_splits(dates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """60/20/20 split by unique dates."""
    n = len(dates)
    i1 = int(0.60 * n)
    i2 = int(0.80 * n)
    train_dates = dates[:i1]
    valid_dates = dates[i1:i2]
    test_dates  = dates[i2:]
    return train_dates, valid_dates, test_dates


def _fit_qreg(X: pd.DataFrame, y: pd.Series, q: float):
    """Fit statsmodels QuantReg with safe numeric casting & finite filtering."""
    Xf = X.astype(float)
    yf = y.astype(float)
    df = Xf.join(yf.to_frame("r1"), how="inner")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    Xf = df[Xf.columns]
    yf = df["r1"]
    # add constant
    Xf = sm.add_constant(Xf, has_constant="add")
    model = sm.QuantReg(yf, Xf)
    return model.fit(q=q)


def _predict_qreg(res, X: pd.DataFrame) -> pd.Series:
    Xp = X.astype(float)
    Xp = Xp.replace([np.inf, -np.inf], np.nan).dropna()
    Xp = sm.add_constant(Xp, has_constant="add")
    pred = res.predict(Xp)
    pred.name = "pred"
    return pred


def _pinball_loss(y_true: pd.Series, y_pred: pd.Series, q: float) -> float:
    """Mean pinball (check) loss for quantile q."""
    e = (y_true - y_pred)
    return (np.maximum(q * e, (q - 1) * e)).mean()


def _lin_calib(valid: pd.DataFrame, test: pd.DataFrame, q: float, colname_raw: str) -> pd.Series:
    """
    Linear calibration: fit QuantReg on VALID: r1 ~ const + yhat_raw (quantile q),
    then predict calibrated values on TEST.
    """
    Xv = sm.add_constant(valid[[colname_raw]].astype(float), has_constant="add")
    yv = valid["r1"].astype(float)
    res = sm.QuantReg(yv, Xv).fit(q=q)

    Xt = sm.add_constant(test[[colname_raw]].astype(float), has_constant="add")
    out = res.predict(Xt)
    out.index = test.index
    return out


# ---- main -------------------------------------------------------------------
def main():
    print("[load]", PANEL_FP)
    panel = pd.read_parquet(PANEL_FP)
    panel = _clean_panel(panel)

    # ensure required columns
    needed = set(FEATS + ["r1"])
    missing = needed - set(panel.columns)
    if missing:
        raise ValueError(f"panel parquet missing columns: {sorted(missing)}")

    # split by date level (unique, sorted)
    all_dates = panel.index.get_level_values("date").unique().sort_values()
    train_d, valid_d, test_d = _date_splits(all_dates.to_numpy())

    # cut and copy to avoid SettingWithCopy
    train = panel.loc[pd.IndexSlice[train_d, :], :].copy()
    valid = panel.loc[pd.IndexSlice[valid_d, :], :].copy()
    test  = panel.loc[pd.IndexSlice[test_d,  :], :].copy()

    # feature matrices / targets
    X_tr, y_tr = train[FEATS], train["r1"]
    X_va, y_va = valid[FEATS], valid["r1"]
    X_te, y_te = test[FEATS],  test["r1"]

    # fit models for each quantile on TRAIN
    print(f"[split] train rows: {len(train):5d}, valid rows: {len(valid):5d}, test rows: {len(test):5d}")
    res_q: Dict[float, any] = {q: _fit_qreg(X_tr, y_tr, q=q) for q in QS}
    for q in QS:
        nobs = int(res_q[q].nobs) if hasattr(res_q[q], "nobs") else -1
        print(f"[fit] Quantile {q:0.1f}\n  -> done, nobs={nobs:6d}")

    # write RAW predictions for valid/test using .loc (avoid warnings)
    for q in QS:
        col = f"q{int(q*100)}_raw"
        # VALID
        yhat_va = _predict_qreg(res_q[q], X_va)
        valid.loc[yhat_va.index, col] = yhat_va.reindex(valid.index)
        # TEST
        yhat_te = _predict_qreg(res_q[q], X_te)
        test.loc[yhat_te.index, col] = yhat_te.reindex(test.index)

    # compute SHIFTs from VALID residuals (q-quantile of residuals)
    shifts: Dict[float, float] = {}
    for q in QS:
        col = f"q{int(q*100)}_raw"
        resid = (valid["r1"] - valid[col]).astype(float)
        resid = resid.replace([np.inf, -np.inf], np.nan).dropna()
        if len(resid) == 0:
            s = 0.0
        else:
            s = float(resid.quantile(q))
        shifts[q] = s

    # apply SHIFT to TEST; build LINEAR calibration too
    for q in QS:
        col_raw = f"q{int(q*100)}_raw"
        col_cal = f"q{int(q*100)}_cal"
        test.loc[:, col_cal] = test[col_raw].astype(float) + shifts[q]

        col_lin = f"q{int(q*100)}_lin"
        test.loc[:, col_lin] = _lin_calib(valid, test, q=q, colname_raw=col_raw)

    # evaluate on TEST
    rows = []
    for kind, suffix in [("raw", "raw"), ("cal", "cal"), ("lin", "lin")]:
        for q in QS:
            col = f"q{int(q*100)}_{suffix}"
            # align finite
            df_eval = pd.DataFrame({"y": y_te, "yhat": test[col]}).replace([np.inf, -np.inf], np.nan).dropna()
            if len(df_eval) == 0:
                cov, pin, n = np.nan, np.nan, 0
            else:
                cov = (df_eval["y"] <= df_eval["yhat"]).mean()
                pin = _pinball_loss(df_eval["y"], df_eval["yhat"], q)
                n = len(df_eval)
            rows.append({"kind": kind, "quantile": q, "coverage": cov, "target": q, "pinball": pin, "n": n})

    out = pd.DataFrame(rows)
    print("\n", out)
    out.to_csv(REPORTS_DIR / "quantile_panel_eval.csv", index=False)

    print("\nShifts:", shifts)


if __name__ == "__main__":
    main()
