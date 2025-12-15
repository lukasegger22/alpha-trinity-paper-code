import pandas as pd
import numpy as np
from pathlib import Path

# --- KONFIGURATION ---
# 📅 WELCHES JAHR WILLST DU PRÜFEN?
TARGET_YEAR = 2020  # Ändere das auf 2020, 2021, 2022, 2023...

# Pfade
FEATURE_DIR = Path("data/features")
HISTORY_PATH = FEATURE_DIR / "trinity_history.parquet"
PANEL_PATH = FEATURE_DIR / "panel.parquet"

# Kosten
COST_OF_CARRY_RATE = 0.06 
TRANSACTION_COST = 0.0010 

def check_period():
    print(f"\n--- 🔎 YEARLY PERFORMANCE CHECK: {TARGET_YEAR} ---")

    if not HISTORY_PATH.exists():
        print("❌ Noch keine History gefunden.")
        return

    # 1. Daten laden
    weights = pd.read_parquet(HISTORY_PATH)
    panel = pd.read_parquet(PANEL_PATH)
    
    panel = panel.reset_index()
    if 'Date' not in panel.columns:
        if 'index' in panel.columns: panel = panel.rename(columns={'index': 'Date'})
        else: panel = panel.rename(columns={panel.columns[0]: 'Date'})
    panel['Date'] = pd.to_datetime(panel['Date'])
    
    # 2. Returns pivotisieren
    try:
        returns = panel.pivot(index='Date', columns='symbol', values='returns_1d').fillna(0)
    except KeyError:
        print("❌ Fehler beim Laden der Returns.")
        return

    # 3. Zeitraum filtern (Januar bis Dezember) 📆
    start_date = pd.Timestamp(f"{TARGET_YEAR}-01-01")
    end_date = pd.Timestamp(f"{TARGET_YEAR}-12-31")
    
    # Filter
    mask = (weights.index >= start_date) & (weights.index <= end_date)
    zoom_weights = weights[mask].copy()
    
    # Returns auch filtern
    common_dates = zoom_weights.index.intersection(returns.index)
    zoom_weights = zoom_weights.loc[common_dates]
    zoom_returns = returns.loc[common_dates]
    
    if len(zoom_weights) == 0:
        print(f"⚠️ Keine Daten für das Jahr {TARGET_YEAR} gefunden!")
        return

    # 4. Performance berechnen
    daily_gross = (zoom_weights * zoom_returns).sum(axis=1)
    
    leverage = zoom_weights.sum(axis=1)
    borrowed = (leverage - 1.0).clip(lower=0)
    interest_cost = (borrowed * COST_OF_CARRY_RATE) / 252 
    
    turnover = zoom_weights.diff().abs().sum(axis=1).fillna(0) 
    trading_cost = turnover * TRANSACTION_COST
    
    daily_net = daily_gross - interest_cost - trading_cost
    
    # Kumulativ für das Jahr
    cum_ret = (1 + daily_net).cumprod()
    total_return = cum_ret.iloc[-1] - 1
    
    # Drawdown nur in diesem Jahr
    running_max = cum_ret.cummax()
    drawdown = (cum_ret / running_max) - 1
    max_dd = drawdown.min()
    
    # Sharpe für das Jahr
    mean_ret = daily_net.mean() * 252
    volatility = daily_net.std() * np.sqrt(252)
    sharpe = mean_ret / volatility if volatility > 0 else 0

    # 5. Ergebnis
    print("-" * 40)
    print(f"💰 Performance im Jahr {TARGET_YEAR}:")
    print(f"📈 Net Return:     {total_return:.2%}")
    print(f"📉 Max Drawdown:   {max_dd:.2%}")
    print(f"⚡ Sharpe Ratio:   {sharpe:.2f}")
    print("-" * 40)
    
    # Benchmark Vergleich (S&P 500 grob)
    spy_ret = zoom_returns['SPY'].sum() if 'SPY' in zoom_returns else 0
    print(f"📊 vs. SPY (Buy&Hold): ca. {spy_ret:.2%}")
    print("-" * 40)

if __name__ == "__main__":
    check_period()