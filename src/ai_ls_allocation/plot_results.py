import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- PFADE ---
FEATURE_DIR = Path("data/features")
REPORT_DIR = Path("reports")
HISTORY_PATH = FEATURE_DIR / "trinity_history.parquet"
PRICES_PATH = FEATURE_DIR / "panel.parquet"

def plot_dashboard():
    print("--- 🎨 Generating Strategy Dashboard ---")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Gewichte (Positionen) laden
    if not HISTORY_PATH.exists():
        raise FileNotFoundError("Run bt_trinity.py first!")
    
    weights = pd.read_parquet(HISTORY_PATH)
    weights.index = pd.to_datetime(weights.index)
    
    # 2. Performance berechnen (gleiche Logik wie Simulation)
    panel = pd.read_parquet(PRICES_PATH)
    if 'Date' in panel.columns:
        panel['Date'] = pd.to_datetime(panel['Date']).dt.normalize()
    
    market_returns = panel.pivot(index='Date', columns='symbol', values='returns_1d')
    
    # Sync
    common_dates = weights.index.intersection(market_returns.index)
    weights = weights.loc[common_dates]
    market_returns = market_returns.loc[common_dates]
    
    # Shift weights (Trade based on yesterday's decision)
    shifted_weights = weights.shift(1).fillna(0)
    
    # Strategy Returns
    strat_ret = (shifted_weights * market_returns).sum(axis=1)
    # Benchmark (Equal Weight)
    bench_ret = market_returns.mean(axis=1)
    
    cum_strat = (1 + strat_ret).cumprod()
    cum_bench = (1 + bench_ret).cumprod()

    # --- PLOT 1: Performance ---
    plt.figure(figsize=(14, 7))
    plt.plot(cum_strat, label='Trinity AI', color='#1f77b4', linewidth=2.5)
    plt.plot(cum_bench, label='Benchmark (Equal Weight)', color='gray', linestyle='--', alpha=0.6)
    plt.title('Trinity Strategy vs. Benchmark', fontsize=16)
    plt.ylabel('Growth of $1')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.savefig(REPORT_DIR / "m1_dashboard.png")
    print("✅ Dashboard saved to reports/m1_dashboard.png")

    # --- PLOT 2: Positions Heatmap (Das "Röntgenbild") ---
    # Wir zeigen nur die letzten 50 Tage, damit es lesbar bleibt
    recent_weights = weights.iloc[-50:].T
    
    plt.figure(figsize=(16, 10))
    sns.heatmap(recent_weights, cmap="RdBu", center=0, annot=False, cbar_kws={'label': 'Position Size'})
    plt.title('Trinity Positions (Last 50 Days) - Blue=Long, Red=Short', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Asset')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "positions_heatmap.png")
    print("✅ Heatmap saved to reports/positions_heatmap.png")

if __name__ == "__main__":
    plot_dashboard()