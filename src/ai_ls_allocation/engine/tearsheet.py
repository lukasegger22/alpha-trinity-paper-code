import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- KONFIGURATION ---
FEATURE_DIR = Path("data/features")
HISTORY_PATH = FEATURE_DIR / "trinity_history.parquet"
PANEL_PATH = FEATURE_DIR / "panel.parquet"
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Kosten (Konservativ & Realistisch)
COST_OF_CARRY_RATE = 0.06  # 6% p.a. für Leverage
TRANSACTION_COST = 0.0010  # 10bps Slippage pro Trade

def standardize_date_col(df):
    """
    MENTOR TOOL: Macht aus jedem DataFrame einen sauberen DataFrame
    mit einer Spalte 'Date' (datetime) und 'symbol' (falls vorhanden).
    Verhindert Index-Chaos.
    """
    # 1. Alles in den Speicher holen (Reset Index)
    df = df.reset_index()
    
    # 2. Spalten normalisieren (alles lowercase machen für den Check)
    col_map = {c: c for c in df.columns} # Original behalten
    
    # Suche nach Datum-Spalte
    date_col = None
    for c in df.columns:
        if c.lower() in ['date', 'time', 'timestamp', 'index']:
            date_col = c
            break
            
    if date_col:
        df.rename(columns={date_col: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        # Letzter Versuch: Index selbst
        if isinstance(df.index, pd.DatetimeIndex):
            df['Date'] = df.index
            df = df.reset_index(drop=True)
            
    return df

def generate_tearsheet():
    print("\n--- 📑 Generating Realistic Tearsheet (With Costs) ---")
    
    if not HISTORY_PATH.exists():
        print(f"❌ No history found at {HISTORY_PATH}")
        return

    # 1. Daten laden
    weights = pd.read_parquet(HISTORY_PATH)
    panel = pd.read_parquet(PANEL_PATH)
    
    # --- MENTOR FIX: Robuste Bereinigung ---
    panel = standardize_date_col(panel)
    
    # Gewichte haben das Datum oft im Index, also auch hier bereinigen
    # Aber Achtung: weights hat Symbole als Spalten, Datum als Index
    weights.index.name = 'Date'
    weights = weights.reset_index()
    weights['Date'] = pd.to_datetime(weights['Date'])
    weights = weights.set_index('Date').sort_index()

    # Check ob 'Date' im Panel jetzt da ist
    if 'Date' not in panel.columns:
        print("❌ CRITICAL: Could not find 'Date' column in panel data even after fixing.")
        print(f"Columns: {panel.columns.tolist()}")
        return

    # Returns pivotisieren
    # Wir brauchen: Index=Date, Cols=Symbol, Values=returns_1d
    try:
        returns = panel.pivot(index='Date', columns='symbol', values='returns_1d').fillna(0)
    except KeyError as e:
        print(f"❌ DATA STRUCTURE ERROR: {e}")
        # Fallback falls 'symbol' anders heißt
        print("Available columns:", panel.columns)
        return
    
    # Indizes abgleichen (Intersection)
    common_dates = weights.index.intersection(returns.index)
    
    if len(common_dates) == 0:
        print("❌ No overlapping dates found between Weights and Returns!")
        print(f"Weights: {weights.index.min()} -> {weights.index.max()}")
        print(f"Returns: {returns.index.min()} -> {returns.index.max()}")
        return
        
    weights = weights.loc[common_dates]
    # Nur die Assets nehmen, die wir auch handeln (Spalten-Schnittmenge)
    valid_cols = weights.columns.intersection(returns.columns)
    weights = weights[valid_cols]
    returns = returns.loc[common_dates, valid_cols]
    
    # 2. Portfolio Returns berechnen
    # Shift(1): Die Gewichte von Gestern bestimmen den Return von Heute
    port_returns = (weights.shift(1) * returns).sum(axis=1)
    
    # 3. KOSTEN ABZIEHEN 🔨
    
    # A. Zinskosten
    leverage = weights.sum(axis=1)
    borrowed_amount = (leverage - 1.0).clip(lower=0) 
    interest_cost = (borrowed_amount * COST_OF_CARRY_RATE) / 252 
    
    # B. Transaktionskosten
    turnover = weights.diff().abs().sum(axis=1) 
    trading_cost = turnover * TRANSACTION_COST
    
    # Netto Return
    net_returns = port_returns - interest_cost - trading_cost
    net_returns = net_returns.fillna(0)
    
    # 4. Metriken
    cum_ret = (1 + net_returns).cumprod()
    total_ret = cum_ret.iloc[-1] - 1 if len(cum_ret) > 0 else 0
    
    # Sharpe
    mean_ret = net_returns.mean() * 252
    volatility = net_returns.std() * np.sqrt(252)
    sharpe = (mean_ret - 0.04) / volatility if volatility > 0 else 0 # 4% Risk Free assumption
    
    # Drawdown
    running_max = cum_ret.cummax()
    drawdown = (cum_ret / running_max) - 1
    max_dd = drawdown.min()
    
    print("-" * 40)
    print("🏆 TRINITY REALITY CHECK (Out-of-Sample)")
    print("-" * 40)
    print(f"📅 Period:       {weights.index[0].date()} to {weights.index[-1].date()}")
    print(f"💰 Gross Return: {((1+port_returns).cumprod().iloc[-1]-1):.2%}")
    print(f"💸 Costs (Est.): -{(((1+port_returns).cumprod().iloc[-1]-1) - total_ret):.2%} (Interest + Slippage)")
    print(f"💎 NET RETURN:   {total_ret:.2%} (Nach allen Kosten)")
    print(f"⚡ Sharpe Ratio: {sharpe:.2f}")
    print(f"📉 Max Drawdown: {max_dd:.2%}")
    print("-" * 40)
    
    # Plotting (Mit Benchmark-Safety)
    plt.figure(figsize=(12, 6))
    plt.plot(cum_ret, label='Trinity Net Return', color='#2962FF', linewidth=2)
    
    # Benchmark Logic
    if 'SPY' in returns.columns:
        bench_ret = (1 + returns['SPY']).cumprod()
        bench_label = 'S&P 500 (SPY)'
    else:
        # Fallback: Equal Weight Index aller Assets
        bench_ret = (1 + returns.mean(axis=1)).cumprod()
        bench_label = 'Market Average (Eq. Weight)'
        
    # Benchmark normieren auf Startdatum
    if not bench_ret.empty:
        bench_ret = bench_ret / bench_ret.iloc[0] * cum_ret.iloc[0]
        plt.plot(bench_ret, label=bench_label, color='gray', alpha=0.6, linestyle='--')
    
    plt.fill_between(cum_ret.index, cum_ret, 1, where=(cum_ret < 1), alpha=0.1, color='red')
    plt.fill_between(cum_ret.index, cum_ret, 1, where=(cum_ret >= 1), alpha=0.1, color='green')
    
    plt.title(f"Trinity Strategy Performance (Sharpe: {sharpe:.2f})")
    plt.ylabel("Equity Curve (Start = 1.0)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = REPORT_DIR / "performance_chart.png"
    plt.savefig(out_path)
    print(f"📈 Chart saved to {out_path}")

if __name__ == "__main__":
    generate_tearsheet()