import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- KONFIGURATION ---
FEATURE_DIR = Path("data/features")
HISTORY_PATH = FEATURE_DIR / "trinity_history.parquet"
PANEL_PATH = FEATURE_DIR / "panel.parquet"
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

# Kosten (Müssen realistisch sein)
COST_OF_CARRY_RATE = 0.06 # 6% p.a. für Leverage
TRANSACTION_COST = 0.0010 # 10bps Slippage pro Trade

def generate_tearsheet():
    print("\n--- 📑 Generating Realistic Tearsheet (With Costs) ---")
    
    if not HISTORY_PATH.exists():
        print("❌ No history found.")
        return

    # 1. Daten laden
    weights = pd.read_parquet(HISTORY_PATH)
    panel = pd.read_parquet(PANEL_PATH)
    
    # --- BUG FIX: Index in Spalten umwandeln ---
    panel = panel.reset_index()
    
    # Sicherstellen, dass 'Date' existiert
    if 'Date' not in panel.columns:
        if 'index' in panel.columns:
            panel = panel.rename(columns={'index': 'Date'})
        elif panel.index.name == 'Date':
             # Falls reset_index nicht gereicht hat (selten)
             panel['Date'] = panel.index
    
    # Falls 'Date' immer noch fehlt, nehmen wir die erste Spalte als Datum an (Fallback)
    if 'Date' not in panel.columns:
         # Versuch, MultiIndex zu fixen
         print("⚠️ Warning: 'Date' column missing. Trying to infer...")
         # Wenn MultiIndex (Date, symbol), dann sind die jetzt Spalten durch reset_index
         # Wir schauen, ob eine Spalte wie ein Datum aussieht
         pass 

    # Returns pivotisieren (Jetzt ist Date sicher eine Spalte)
    # Wir brauchen 'Date', 'symbol' und 'returns_1d'
    try:
        returns = panel.pivot(index='Date', columns='symbol', values='returns_1d').fillna(0)
    except KeyError as e:
        print(f"❌ DATA STRUCTURE ERROR: {e}")
        print("Available columns:", panel.columns)
        return
    
    # Indizes abgleichen (Schnittmenge aus Trade-Tagen und Markt-Daten)
    common_dates = weights.index.intersection(returns.index)
    weights = weights.loc[common_dates]
    returns = returns.loc[common_dates]
    
    if len(weights) == 0:
        print("❌ No overlapping dates found between Weights and Returns!")
        return
    
    # 2. Portfolio Returns berechnen
    # Portfolio Return = Summe(Gewicht * Return)
    port_returns = (weights * returns).sum(axis=1)
    
    # 3. KOSTEN ABZIEHEN (Der Realitäts-Hammer) 🔨
    
    # A. Zinskosten (Cost of Carry)
    # Wenn Exposure > 1.0 (Leverage), zahlen wir Zinsen auf den geliehenen Teil
    leverage = weights.sum(axis=1)
    borrowed_amount = (leverage - 1.0).clip(lower=0) # Alles über 100% ist geliehen
    interest_cost = (borrowed_amount * COST_OF_CARRY_RATE) / 252 # Täglicher Zins
    
    # B. Transaktionskosten (Turnover)
    turnover = weights.diff().abs().sum(axis=1) # Wie viel haben wir umgeschichtet?
    trading_cost = turnover * TRANSACTION_COST
    
    # Netto Return (Gewinn nach allen Kosten)
    net_returns = port_returns - interest_cost - trading_cost
    
    # 4. Metriken berechnen
    cum_ret = (1 + net_returns).cumprod()
    if len(cum_ret) > 0:
        total_ret = cum_ret.iloc[-1] - 1
    else:
        total_ret = 0
    
    # Sharpe (Annualisiert)
    mean_ret = net_returns.mean() * 252
    volatility = net_returns.std() * np.sqrt(252)
    sharpe = mean_ret / volatility if volatility > 0 else 0
    
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
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(cum_ret, label='Trinity Net Return (Real)', color='blue', linewidth=1.5)
    plt.plot((1+returns['SPY']).cumprod(), label='S&P 500 (Benchmark)', color='gray', alpha=0.5, linestyle='--')
    
    # Drawdown Plot (Rot unten)
    plt.fill_between(cum_ret.index, cum_ret, 1, alpha=0.05, color='blue')
    
    plt.title(f"Trinity Walk-Forward Performance (Sharpe: {sharpe:.2f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(REPORT_DIR / "performance_chart.png")
    print(f"📈 Chart saved to {REPORT_DIR}/performance_chart.png")

if __name__ == "__main__":
    generate_tearsheet()