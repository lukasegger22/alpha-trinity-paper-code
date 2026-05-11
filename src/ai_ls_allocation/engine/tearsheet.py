import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
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

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "trinity": "#2962FF",
    "spy": "#6E6E6E",
    "drawdown": "#D32F2F",
    "exposure": "#F57C00",
}

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

def format_year_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

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

    if 'SPY' in returns.columns:
        bench_ret_daily = returns['SPY'].fillna(0)
        bench_label = 'S&P 500 (SPY)'
    else:
        bench_ret_daily = returns.mean(axis=1).fillna(0)
        bench_label = 'Market Average'

    bench_cum = (1 + bench_ret_daily).cumprod()
    if not bench_cum.empty:
        bench_cum = bench_cum / bench_cum.iloc[0] * cum_ret.iloc[0]
    
    # Sharpe
    mean_ret = net_returns.mean() * 252
    volatility = net_returns.std() * np.sqrt(252)
    sharpe = (mean_ret - 0.04) / volatility if volatility > 0 else 0 # 4% Risk Free assumption
    
    # Drawdown
    running_max = cum_ret.cummax()
    drawdown = (cum_ret / running_max) - 1
    max_dd = drawdown.min()

    bench_drawdown = (bench_cum / bench_cum.cummax()) - 1
    
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
    
    # Plot 1: Performance
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(cum_ret.index, cum_ret, label='Alpha Trinity (Net)', color=COLORS["trinity"], linewidth=2.2)
    ax.plot(bench_cum.index, bench_cum, label=bench_label, color=COLORS["spy"], alpha=0.75, linestyle='--', linewidth=1.7)
    ax.fill_between(cum_ret.index, cum_ret, 1, where=(cum_ret < 1), alpha=0.08, color='red')
    ax.fill_between(cum_ret.index, cum_ret, 1, where=(cum_ret >= 1), alpha=0.06, color='green')
    ax.set_title(f"Cumulative Net Performance (Sharpe: {sharpe:.2f})", fontsize=14, fontweight='bold')
    ax.set_ylabel("Equity Growth ($1 Invested)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    format_year_axis(ax)
    fig.tight_layout()
    out_path = REPORT_DIR / "performance_chart.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"📈 Chart saved to {out_path}")

    # Plot 2: Drawdown with S&P 500 benchmark
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(drawdown.index, drawdown, label='Alpha Trinity', color=COLORS["trinity"], linewidth=1.8)
    ax.plot(bench_drawdown.index, bench_drawdown, label=bench_label, color=COLORS["spy"], linestyle='--', alpha=0.75, linewidth=1.5)
    ax.fill_between(drawdown.index, drawdown, 0, color=COLORS["trinity"], alpha=0.14)
    ax.fill_between(bench_drawdown.index, bench_drawdown, 0, color=COLORS["spy"], alpha=0.07)
    ax.set_title("Drawdown Profile: Alpha Trinity vs. S&P 500", fontsize=14, fontweight='bold')
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    format_year_axis(ax)
    fig.tight_layout()
    out_path = REPORT_DIR / "drawdown_chart.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"📉 Chart saved to {out_path}")

    # Plot 3: Regime and exposure verification
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]}
    )
    ax_top.plot(cum_ret.index, cum_ret, label='Alpha Trinity Equity', color=COLORS["trinity"], linewidth=2.0)
    ax_top.plot(bench_cum.index, bench_cum, label=bench_label, color=COLORS["spy"], linestyle='--', alpha=0.65, linewidth=1.4)
    ax_top.set_title("Crisis Mechanism Verification: Performance vs. Exposure", fontsize=14, fontweight='bold')
    ax_top.set_ylabel("Equity")
    ax_top.legend(loc="upper left")
    ax_top.grid(True, alpha=0.3)

    ax_bottom.fill_between(leverage.index, 0, leverage, color=COLORS["exposure"], alpha=0.35, label='Portfolio Exposure')
    ax_bottom.plot(leverage.index, leverage, color=COLORS["exposure"], linewidth=1.1)
    ax_bottom.axhline(1.0, color="#444444", linestyle=":", linewidth=1, alpha=0.7)
    ax_bottom.set_ylabel("Exposure")
    ax_bottom.set_ylim(0, max(1.45, float(leverage.max()) * 1.10))
    ax_bottom.legend(loc="upper left")
    ax_bottom.grid(True, alpha=0.3)
    format_year_axis(ax_bottom)
    fig.tight_layout()
    out_path = REPORT_DIR / "regime_chart.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"🛡️ Chart saved to {out_path}")

    # Plot 4: Rolling volatility stability
    roll_vol_trinity = net_returns.rolling(60).std() * np.sqrt(252)
    roll_vol_spy = bench_ret_daily.rolling(60).std() * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(roll_vol_trinity.index, roll_vol_trinity, label='Alpha Trinity 60d Volatility', color=COLORS["trinity"], linewidth=1.9)
    ax.plot(roll_vol_spy.index, roll_vol_spy, label='S&P 500 60d Volatility', color=COLORS["spy"], linestyle='--', alpha=0.75, linewidth=1.5)
    ax.set_title("Risk Stability Analysis: Rolling 60-Day Volatility", fontsize=14, fontweight='bold')
    ax.set_ylabel("Annualized Volatility")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    format_year_axis(ax)
    fig.tight_layout()
    out_path = REPORT_DIR / "rolling_risk_chart.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Chart saved to {out_path}")

if __name__ == "__main__":
    generate_tearsheet()
