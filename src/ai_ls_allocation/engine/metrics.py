from __future__ import annotations

import numpy as np
import pandas as pd


# ---- Klassische Performance-Metriken --------------------------------------------

def sharpe(ret: pd.Series, periods_per_year: int = 252) -> float:
    ret = pd.Series(ret).dropna()
    if ret.empty:
        return float("nan")
    mu = ret.mean()
    sig = ret.std(ddof=1)
    if sig == 0:
        return float("nan")
    return float((mu / sig) * np.sqrt(periods_per_year))


def cagr(ret: pd.Series, periods_per_year: int = 252) -> float:
    ret = pd.Series(ret).dropna()
    if ret.empty:
        return 0.0
    eq = (1.0 + ret).cumprod()
    t = len(eq) / periods_per_year
    if t <= 0:
        return 0.0
    return float(eq.iloc[-1] ** (1.0 / t) - 1.0)


def max_drawdown(eq: pd.Series) -> float:
    eq = pd.Series(eq).dropna()
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def calmar(ret: pd.Series, periods_per_year: int = 252) -> float:
    ret = pd.Series(ret).dropna()
    if ret.empty:
        return float("nan")
    eq = (1.0 + ret).cumprod()
    dd = max_drawdown(eq)
    if dd == 0:
        return float("nan")
    return float(cagr(ret, periods_per_year) / abs(dd))


# ---- Turnover & Kosten ----------------------------------------------------------

def avg_turnover(turnover: pd.Series) -> float:
    turnover = pd.Series(turnover).dropna()
    if turnover.empty:
        return 0.0
    return float(turnover.mean())


def cost_share(costs: pd.Series, gross_pnl: pd.Series) -> float:
    costs = pd.Series(costs).fillna(0.0)
    gross_pnl = pd.Series(gross_pnl).fillna(0.0)
    denom = gross_pnl.abs().sum()
    if denom == 0:
        return 0.0
    return float(costs.sum() / denom)


# ---- Quantil- / Kalibrierungsmetriken -------------------------------------------

def pinball_loss_series(y: pd.Series, q: pd.Series, alpha: float) -> pd.Series:
    """
    Elementweise Pinball-Loss für Quantil alpha.
    Standardform: alpha * max(y-q,0) + (1-alpha) * max(q-y,0)
    """
    y = pd.Series(y, index=q.index)
    q = pd.Series(q, index=y.index)
    diff = y - q
    loss = alpha * diff.clip(lower=0.0) + (1.0 - alpha) * (-diff.clip(upper=0.0))
    return loss


def pinball_loss_mean(y: pd.Series, q: pd.Series, alpha: float) -> float:
    loss = pinball_loss_series(y, q, alpha)
    return float(loss.mean())


def quantile_coverage(y: pd.Series, q: pd.Series, alpha: float) -> float:
    """
    Empirische Coverage P(Y <= q_alpha).
    Für ein gut kalibriertes Modell ~ alpha.
    """
    y = pd.Series(y, index=q.index)
    q = pd.Series(q, index=y.index)
    valid = y.notna() & q.notna()
    if not valid.any():
        return float("nan")
    cov = (y[valid] <= q[valid]).mean()
    return float(cov)
