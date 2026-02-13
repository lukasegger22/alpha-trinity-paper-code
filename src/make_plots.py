import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Setup
HISTORY_PATH = Path("data/features/trinity_history.parquet")
PANEL_PATH = Path("data/features/panel.parquet")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

def generate_plots():
    print("🎨 Generating Academic Plots (Robust Version)...")
    
    # 1. Lade die History (Gewichte)
    if not HISTORY_PATH.exists():
        print(f"❌ Error: History file not found at {HISTORY_PATH}")
        return
    
    # Weights haben normalerweise das Datum als Index
    weights = pd.read_parquet(HISTORY_PATH)
    print(f"   ✅ Loaded Weights: {weights.shape} (from {weights.index[0].date()} to {weights.index[-1].date()})")
    
    # 2. Lade Panel für Returns (ROBUST)
    if not PANEL_PATH.exists():
        print(f"❌ Error: Panel file not found at {PANEL_PATH}")
        return
        
    panel = pd.read_parquet(PANEL_PATH)
    
    # --- FIX FÜR DEN KEYERROR ---
    # Wir holen den Index zurück in die Spalten
    panel = panel.reset_index()
    
    # Intelligente Spalten-Suche (wie im Haupt-Bot)
    # Wir suchen nach der Datums-Spalte
    date_col = None
    for col in ['Date', 'date', 'index', 'time', 'timestamp']:
        if col in panel.columns:
            date_col = col
            break
            
    # Wir suchen nach der Symbol-Spalte
    sym_col = None
    for col in ['symbol', 'Symbol', 'ticker', 'Ticker']:
        if col in panel.columns:
            sym_col = col
            break
            
    if not date_col or not sym_col:
        print(f"❌ CRITICAL: Could not identify Date or Symbol columns. Found: {panel.columns.tolist()}")
        return
        
    print(f"   ℹ️  Identified columns: Date='{date_col}', Symbol='{sym_col}'")
    
    # Pivot erstellen (Index=Date, Columns=Symbol, Values=returns_1d)
    # returns_1d muss existieren
    if 'returns_1d' not in panel.columns:
        print("❌ Error: 'returns_1d' column missing in panel data.")
        return

    returns = panel.pivot(index=date_col, columns=sym_col, values='returns_1d').fillna(0)
    
    # 3. Align & Calculate
    common_dates = weights.index.intersection(returns.index)
    weights = weights.loc[common_dates]
    returns = returns.loc[common_dates]
    
    # Portfolio Return berechnen
    # Shift weights by 1 day because weights determined at Close T are for T+1
    port_ret = (weights.shift(1).fillna(0) * returns).sum(axis=1)
    
    # Kosten abziehen (Simulation)
    cost_per_day = 0.0005 
    net_ret = port_ret - cost_per_day
    
    # Equity Curve
    equity_curve = (1 + net_ret).cumprod()
    
    # Benchmark (SPY) laden falls vorhanden, sonst Dummy
    if 'SPY' in returns.columns:
        spy_ret = returns['SPY']
    else:
        # Fallback auf erstes Asset falls SPY fehlt (sollte aber da sein)
        spy_ret = returns.iloc[:, 0]
        
    spy_curve = (1 + spy_ret).cumprod()
    
    # --- PLOT 1: Performance Comparison ---
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve.index, equity_curve, label='Alpha Trinity (AI)', color='#1f77b4', linewidth=2)
    plt.plot(spy_curve.index, spy_curve, label='S&P 500 (Benchmark)', color='#7f7f7f', linestyle='--', alpha=0.7)
    
    plt.title('Out-of-Sample Performance (Net of Costs)', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Return (1.0 = Start)', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "performance_chart.png", dpi=300)
    print("✅ Saved performance_chart.png")
    
    # --- PLOT 2: Drawdown (Underwater) ---
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1
    
    plt.figure(figsize=(10, 4))
    plt.fill_between(drawdown.index, drawdown, 0, color='#d62728', alpha=0.3)
    plt.plot(drawdown.index, drawdown, color='#d62728', linewidth=1)
    
    plt.title('Drawdown Profile (Risk Management)', fontsize=12, fontweight='bold')
    plt.ylabel('Drawdown (%)', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "drawdown_chart.png", dpi=300)
    print("✅ Saved drawdown_chart.png")

if __name__ == "__main__":
    generate_plots()