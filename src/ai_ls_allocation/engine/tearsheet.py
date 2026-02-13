import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

# Style-Update für "Paper-Ready" Charts
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'trinity': '#2962FF', 'spy': '#757575', 'dd': '#D32F2F', 'lev': '#FF9800'}

def standardize_date_col(df):
    """Bereinigt DataFrames auf einheitliches Datumsformat."""
    df = df.reset_index()
    col_map = {c: c for c in df.columns}
    date_col = None
    for c in df.columns:
        if c.lower() in ['date', 'time', 'timestamp', 'index']:
            date_col = c
            break
            
    if date_col:
        df.rename(columns={date_col: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        if isinstance(df.index, pd.DatetimeIndex):
            df['Date'] = df.index
            df = df.reset_index(drop=True)
    return df

def generate_tearsheet():
    print("\n--- 📑 Generating Institutional Tearsheet (4 Plots) ---")
    
    if not HISTORY_PATH.exists():
        print(f"❌ No history found at {HISTORY_PATH}")
        return

    # 1. Daten laden & Bereinigen
    weights = pd.read_parquet(HISTORY_PATH)
    panel = pd.read_parquet(PANEL_PATH)
    panel = standardize_date_col(panel)
    
    weights.index.name = 'Date'
    weights = weights.reset_index()
    weights['Date'] = pd.to_datetime(weights['Date'])
    weights = weights.set_index('Date').sort_index()

    if 'Date' not in panel.columns:
        print("❌ CRITICAL: Could not find 'Date' column in panel.")
        return

    # Returns pivotisieren
    try:
        returns = panel.pivot(index='Date', columns='symbol', values='returns_1d').fillna(0)
    except KeyError as e:
        print(f"❌ DATA STRUCTURE ERROR: {e}")
        return
    
    # Schnittmenge finden
    common_dates = weights.index.intersection(returns.index)
    if len(common_dates) == 0:
        print("❌ No overlapping dates found!")
        return
        
    weights = weights.loc[common_dates]
    valid_cols = weights.columns.intersection(returns.columns)
    weights = weights[valid_cols]
    returns = returns.loc[common_dates, valid_cols]
    
    # 2. Portfolio Returns berechnen
    # Shift(1): Die Gewichte von Gestern bestimmen den Return von Heute
    raw_port_returns = (weights.shift(1) * returns).sum(axis=1)
    
    # 3. KOSTEN ABZIEHEN
    leverage = weights.abs().sum(axis=1) # WICHTIG: abs() für Short-Positionen
    borrowed_amount = (leverage - 1.0).clip(lower=0) 
    interest_cost = (borrowed_amount * COST_OF_CARRY_RATE) / 252 
    turnover = weights.diff().abs().sum(axis=1) 
    trading_cost = turnover * TRANSACTION_COST
    
    net_returns = raw_port_returns - interest_cost - trading_cost
    net_returns = net_returns.fillna(0)
    
    # 4. Metriken Berechnung
    cum_ret = (1 + net_returns).cumprod()
    total_ret = cum_ret.iloc[-1] - 1 if len(cum_ret) > 0 else 0
    
    # Benchmark (SPY)
    if 'SPY' in returns.columns:
        bench_ret_daily = returns['SPY']
    else:
        bench_ret_daily = returns.mean(axis=1)
    
    bench_cum = (1 + bench_ret_daily).cumprod()
    # Benchmark normalisieren auf Trinity Startwert
    bench_cum = bench_cum / bench_cum.iloc[0] * cum_ret.iloc[0]

    # Drawdown
    running_max = cum_ret.cummax()
    drawdown = (cum_ret / running_max) - 1
    max_dd = drawdown.min()
    
    # Sharpe
    mean_ret = net_returns.mean() * 252
    volatility = net_returns.std() * np.sqrt(252)
    sharpe = (mean_ret - 0.04) / volatility if volatility > 0 else 0

    # Console Output
    print("-" * 40)
    print("🏆 TRINITY REALITY CHECK (Out-of-Sample)")
    print("-" * 40)
    print(f"📅 Period:       {weights.index[0].date()} to {weights.index[-1].date()}")
    print(f"💰 Gross Return: {((1+raw_port_returns).cumprod().iloc[-1]-1):.2%}")
    print(f"💸 Costs (Est.): -{(((1+raw_port_returns).cumprod().iloc[-1]-1) - total_ret):.2%}")
    print(f"💎 NET RETURN:   {total_ret:.2%}")
    print(f"⚡ Sharpe Ratio: {sharpe:.2f}")
    print(f"📉 Max Drawdown: {max_dd:.2%}")
    print("-" * 40)
    
    # ==============================================================================
    # 📊 PLOT 1: EQUITY CURVE (Der Klassiker)
    # ==============================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(cum_ret.index, cum_ret, label='Alpha Trinity (Net)', color=COLORS['trinity'], linewidth=2)
    ax.plot(bench_cum.index, bench_cum, label='S&P 500 (Benchmark)', color=COLORS['spy'], alpha=0.6, linestyle='--')
    
    # Fill areas for profit/loss relative to start
    ax.fill_between(cum_ret.index, cum_ret, 1.0, where=(cum_ret >= 1), color='green', alpha=0.05)
    ax.fill_between(cum_ret.index, cum_ret, 1.0, where=(cum_ret < 1), color='red', alpha=0.05)
    
    ax.set_title(f"Cumulative Net Performance (Sharpe: {sharpe:.2f})", fontsize=14, fontweight='bold')
    ax.set_ylabel("Equity Growth ($1 Invested)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    out_path = REPORT_DIR / "performance_chart.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"📈 Chart 1 saved: {out_path}")
    plt.close(fig)

    # ==============================================================================
    # 📊 PLOT 2: UNDERWATER PLOT (Robustheit)
    # ==============================================================================
    # Benchmark Drawdown berechnen
    bench_max = bench_cum.cummax()
    bench_dd = (bench_cum / bench_max) - 1

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(drawdown.index, drawdown, label='Alpha Trinity', color=COLORS['trinity'], linewidth=1.5)
    ax.plot(bench_dd.index, bench_dd, label='S&P 500', color=COLORS['spy'], alpha=0.4, linestyle='--')
    ax.fill_between(drawdown.index, drawdown, 0, color=COLORS['trinity'], alpha=0.15)
    
    ax.set_title("Drawdown Profile (Underwater Plot)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    out_path = REPORT_DIR / "drawdown_chart.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"📈 Chart 2 saved: {out_path}")
    plt.close(fig)

    # ==============================================================================
    # 📊 PLOT 3: REGIME & LEVERAGE (Crisis Engine Proof)
    # ==============================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Oben: Kursverlauf
    ax1.plot(cum_ret.index, cum_ret, label='Trinity Equity', color=COLORS['trinity'])
    ax1.set_title("Crisis Mechanism Verification: Performance vs. Exposure", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Equity")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Unten: Leverage Area
    ax2.fill_between(leverage.index, 0, leverage, color=COLORS['lev'], alpha=0.4, label='Portfolio Exposure (Leverage)')
    ax2.set_ylabel("Exposure (0 = Cash, >1 = Lev)")
    ax2.set_ylim(0, 1.8)
    
    # Markiere Krisen-Phasen (Wo wir Cash waren)
    crisis_dates = leverage[leverage < 0.5].index
    if len(crisis_dates) > 0:
        # Trick für saubere Legende: Nur einmal labeln
        ax1.axvline(crisis_dates[0], color='red', alpha=0.0, label='Defensive Regime') 
        for date in crisis_dates:
            ax1.axvline(date, color='red', alpha=0.05)
            ax2.axvline(date, color='red', alpha=0.05)
            
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    out_path = REPORT_DIR / "regime_chart.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"📈 Chart 3 saved: {out_path}")
    plt.close(fig)

    # ==============================================================================
    # 📊 PLOT 4: ROLLING VOLATILITY (Stabilität)
    # ==============================================================================
    roll_vol_trinity = net_returns.rolling(60).std() * np.sqrt(252)
    roll_vol_spy = bench_ret_daily.rolling(60).std() * np.sqrt(252)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(roll_vol_trinity.index, roll_vol_trinity, label='Trinity Volatility (60d)', color=COLORS['trinity'])
    ax.plot(roll_vol_spy.index, roll_vol_spy, label='S&P 500 Volatility', color=COLORS['spy'], linestyle='--')
    
    ax.set_title("Risk Stability Analysis (Rolling Volatility)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Annualized Volatility")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    out_path = REPORT_DIR / "rolling_risk_chart.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"📈 Chart 4 saved: {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    generate_tearsheet()