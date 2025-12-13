from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

# ---- config -----------------------------------------------------------------
DATA_DIR = Path("data")
FEAT_DIR = DATA_DIR / "features"
PANEL_FP = FEAT_DIR / "panel.parquet"
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

QS: List[float] = [0.1, 0.5, 0.9]
FEATS: List[str] = [
    "vol20", "vol60",
    "ma20", "ma60", "trend_ma_diff",
    "macd", "macd_signal", "macd_hist",
    "spread_hyg_ief",
    "r1" # r1 (Return t) ist auch ein Feature für t+1!
]
TARGET_COL = "target_r1"  # Wir sagen t+1 vorher

PRED_KIND = os.getenv("QPNL_PRED_KIND", "cal")
COL_MAP = {"raw": "q50_raw", "cal": "q50_cal", "lin": "q50_lin"}


# ---- helpers ----------------------------------------------------------------
def _clean_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if isinstance(panel.index, pd.MultiIndex):
        if panel.index.names != ["date", "symbol"]:
            panel.index.set_names(["date", "symbol"], inplace=True)
    else:
        if {"date", "symbol"}.issubset(panel.columns):
            panel = panel.set_index(["date", "symbol"])
        else:
            raise ValueError("panel must be MultiIndex")

    idx_df = panel.index.to_frame(index=False)
    idx_df["date"] = pd.to_datetime(idx_df["date"])
    panel.index = pd.MultiIndex.from_frame(idx_df, names=["date", "symbol"])
    return panel.sort_index()


def _date_splits(dates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(dates)
    i1 = int(0.60 * n)
    i2 = int(0.80 * n)
    return dates[:i1], dates[i1:i2], dates[i2:]


def _fit_qreg(X: pd.DataFrame, y: pd.Series, q: float):
    # Robustes Fitting: Entferne NaNs und Infs
    data = pd.concat([X, y.rename("y")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        raise ValueError(f"No valid data to fit quantile {q}")
    
    X_clean = sm.add_constant(data[X.columns], has_constant="add")
    return sm.QuantReg(data["y"], X_clean).fit(q=q)


def _predict_qreg(res, X: pd.DataFrame) -> pd.Series:
    Xp = X.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xp = sm.add_constant(Xp, has_constant="add")
    pred = res.predict(Xp)
    return pd.Series(pred, index=X.index)


def _weights_from_signal(df_cs: pd.DataFrame, col_signal: str) -> pd.Series:
    """
    Echtes Long/Short: Gewichtet proportional zur Signalstärke.
    Summe der Absolutwerte (Gross Exposure) = 1.0
    """
    s = df_cs[col_signal].astype(float).fillna(0.0)
    
    gross = s.abs().sum()
    if gross < 1e-9:
        return pd.Series(0.0, index=df_cs.index)
        
    return s / gross


# ---- main -------------------------------------------------------------------
def main():
    print("[load]", PANEL_FP)
    panel = pd.read_parquet(PANEL_FP)
    panel = _clean_panel(panel)

    # Check Columns
    needed = set(FEATS + [TARGET_COL])
    missing = needed - set(panel.columns)
    if missing:
        # Fallback: Falls target_r1 fehlt, User warnen
        raise ValueError(f"Panel missing columns: {missing}. Did you run corrected build.py?")

    all_dates = panel.index.get_level_values("date").unique().sort_values()
    train_d, valid_d, test_d = _date_splits(all_dates.to_numpy())

    train = panel.loc[pd.IndexSlice[train_d, :], :].copy()
    valid = panel.loc[pd.IndexSlice[valid_d, :], :].copy()
    test  = panel.loc[pd.IndexSlice[test_d,  :], :].copy()

    X_tr, y_tr = train[FEATS], train[TARGET_COL]
    X_va = valid[FEATS]
    X_te = test[FEATS]

    print(f"[fit] Training on {len(train)} rows...")
    res_q = {q: _fit_qreg(X_tr, y_tr, q=q) for q in QS}

    # Predict & Calibrate
    for q in QS:
        col_raw = f"q{int(q*100)}_raw"
        
        # Valid Predictions (für Calibration)
        valid[col_raw] = _predict_qreg(res_q[q], X_va)
        
        # Test Predictions
        test[col_raw] = _predict_qreg(res_q[q], X_te)
        
        # Simple Shift Calibration
        resid = (valid[TARGET_COL] - valid[col_raw]).dropna()
        shift = resid.quantile(q) if not resid.empty else 0.0
        
        test[f"q{int(q*100)}_cal"] = test[col_raw] + shift

    # --- Backtest Execution ---
    pred_col = COL_MAP.get(PRED_KIND, "q50_cal")
    print(f"[exec] Using signal column: {pred_col}")

    # Gewichte berechnen
    dates = test.index.get_level_values("date").unique()
    W_list = []
    
    for d in dates:
        # Cross-Section für diesen Tag
        try:
            df_cs = test.loc[d] 
        except KeyError:
            continue
            
        w = _weights_from_signal(df_cs, pred_col)
        w.name = d
        W_list.append(w)

    W = pd.DataFrame(W_list).sort_index().fillna(0.0)

    # Returns berechnen
    # ACHTUNG: Hier nutzen wir r1 (Returns von HEUTE), nicht target_r1!
    # Wir traden basierend auf Signal gestern für Return heute.
    # W enthält Signal von heute für morgen? 
    # NEIN: test enthält Vorhersagen für target_r1 (morgen).
    # Das heißt, an Tag t wissen wir pred(t+1). Wir setzen Position für t+1.
    # Return an Tag t+1 ist r1 von t+1.
    
    r_realized = test["r1"].unstack("symbol").reindex(W.index).fillna(0.0)
    
    # Da W[t] auf target_r1[t] (also r[t+1]) zielt, 
    # müssen wir W[t] mit r_realized[t+1] multiplizieren.
    # In Pandas bedeutet das: Shift der Gewichte ist hier NICHT nötig, 
    # WENN W[t] bereits "Position für Morgen" bedeutet.
    # ABER: Um Konfusion zu vermeiden, folgen wir dem Standard:
    # Signal t -> Trade zum Close t -> Position t+1 -> Return t+1.
    # Also: port_ret[t+1] = W[t] * r[t+1]
    
    port_ret = (W.shift(1) * r_realized).sum(axis=1)
    
    # Equity
    equity = (1.0 + port_ret).cumprod()
    turnover = W.diff().abs().sum(axis=1) / 2.0

    # Save
    ts = pd.DataFrame({"port_ret": port_ret, "equity": equity, "turnover": turnover})
    ts.to_csv(REPORTS_DIR / "bt_qpanel_timeseries.csv")
    W.to_csv(REPORTS_DIR / "bt_qpanel_weights.csv")
    
    print("[done] wrote reports/bt_qpanel_timeseries.csv")
    print("Final Equity:", equity.iloc[-1] if not equity.empty else "N/A")

if __name__ == "__main__":
    main()