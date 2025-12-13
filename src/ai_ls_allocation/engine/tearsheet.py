import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- PFADE ---
FEATURE_DIR = Path("data/features")
REPORT_DIR = Path("reports")
HISTORY_PATH = FEATURE_DIR / "trinity_history.parquet"
PRICES_PATH = FEATURE_DIR / "panel.parquet"

def calculate_max_drawdown(cum_returns):
    """Berechnet den maximalen Verlust von der Spitze (High Water Mark)."""
    high_water_mark = cum_returns.cummax()
    drawdown = (cum_returns - high_water_mark) / high_water_mark
    return drawdown.min(), drawdown

def generate_tearsheet():
    print("--- 📑 Generating Institutional Tearsheet ---")
    
    # 1. Daten laden
    if not HISTORY_PATH.exists():
        raise FileNotFoundError("Run bt_trinity.py first!")
    
    weights = pd.read_parquet(HISTORY_PATH)
    weights.index = pd.to_datetime(weights.index)
    
    panel = pd.read_parquet(PRICES_PATH)
    if 'Date' in panel.columns:
        panel['Date'] = pd.to_datetime(panel['Date']).dt.normalize()
    
    market_returns = panel.pivot(index='Date', columns='symbol', values='returns_1d')
    
    # Sync
    common_dates = weights.index.intersection(market_returns.index)
    weights = weights.loc[common_dates]
    market_returns = market_returns.loc[common_dates]
    
    # 2. Daily PnL berechnen (Shifted Weights * Returns)
    # Kosten abziehen (0.10% = 0.0010)
    shifted_weights = weights.shift(1).fillna(0)
    turnover = (weights - shifted_weights).abs().sum(axis=1)
    costs = turnover * 0.0010
    
    daily_rets = (shifted_weights * market_returns).sum(axis=1) - costs
    cum_rets = (1 + daily_rets).cumprod()
    
    # 3. KENNZAHLEN (The Metrics that matter)
    
    # A. Total Return & CAGR
    total_ret = (cum_rets.iloc[-1] - 1)
    days = len(daily_rets)
    cagr = (1 + total_ret) ** (252 / days) - 1 # Annualized
    
    # B. Risiko (Volatilität & Downside)
    volatility = daily_rets.std() * np.sqrt(252)
    sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252)
    
    # Sortino Ratio (Nur negative Volatilität zählt)
    neg_rets = daily_rets[daily_rets < 0]
    sortino = (daily_rets.mean() / neg_rets.std()) * np.sqrt(252)
    
    # C. Drawdown
    max_dd, dd_series = calculate_max_drawdown(cum_rets)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0 # Return pro Drawdown-Einheit
    
    # D. Win Rate
    wins = len(daily_rets[daily_rets > 0])
    losses = len(daily_rets[daily_rets < 0])
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    
    # 4. ATTRIBUTION (Woher kam das Geld?)
    # Wir schauen, welches Asset den meisten Profit gebracht hat
    asset_pnl = (shifted_weights * market_returns).sum()
    top_winner = asset_pnl.idxmax()
    top_loser = asset_pnl.idxmin()

    # --- OUTPUT ---
    print("\n" + "="*40)
    print(f"   TRINITY STRATEGY TEARSHEET")
    print("="*40)
    print(f"Performance:")
    print(f"  Total Return:    {total_ret*100:.2f}%")
    print(f"  CAGR (Yearly):   {cagr*100:.2f}%")
    print(f"  Win Rate:        {win_rate*100:.1f}%  (>50% ist gut)")
    print("-" * 40)
    print(f"Risk Management:")
    print(f"  Sharpe Ratio:    {sharpe:.2f}      (Industry Standard)")
    print(f"  Sortino Ratio:   {sortino:.2f}     (Nur Downside Risk)")
    print(f"  Max Drawdown:    {max_dd*100:.2f}%  (Schlimmster Verlust)")
    print(f"  Calmar Ratio:    {calmar:.2f}      (Return / Drawdown)")
    print("-" * 40)
    print(f"Attribution:")
    print(f"  Best Asset:      {top_winner} (+{asset_pnl[top_winner]*100:.1f}%)")
    print(f"  Worst Asset:     {top_loser} ({asset_pnl[top_loser]*100:.1f}%)")
    print("="*40 + "\n")
    
    # Plot Drawdown
    plt.figure(figsize=(10, 4))
    plt.fill_between(dd_series.index, dd_series, 0, color='red', alpha=0.3)
    plt.plot(dd_series.index, dd_series, color='red', linewidth=1)
    plt.title('Underwater Plot (Drawdown)')
    plt.ylabel('Loss from Peak')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "drawdown_chart.png")
    print(f"📉 Drawdown chart saved to reports/drawdown_chart.png")

if __name__ == "__main__":
    generate_tearsheet()