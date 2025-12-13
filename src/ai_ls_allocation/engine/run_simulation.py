import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- PFADE ---
FEATURE_DIR = Path("data/features")
REPORT_DIR = Path("reports")
SIGNALS_PATH = FEATURE_DIR / "trinity_history.parquet"
PRICES_PATH = FEATURE_DIR / "panel.parquet"

def run_simulation():
    print("--- 📉 Running Performance Simulation ---")
    
    # 1. Signale laden (Das, was Trinity berechnet hat)
    if not SIGNALS_PATH.exists():
        raise FileNotFoundError("Signals not found. Run bt_trinity.py first!")
    
    signals = pd.read_parquet(SIGNALS_PATH)
    # Entferne Zeitstempel aus dem Index, falls vorhanden, nur Datum behalten
    signals.index = pd.to_datetime(signals.index).normalize()
    
    # 2. Echte Returns laden (Was der Markt wirklich gemacht hat)
    panel = pd.read_parquet(PRICES_PATH)
    if 'Date' in panel.columns:
        panel['Date'] = pd.to_datetime(panel['Date']).dt.normalize()
    
    # Pivot: Index=Date, Columns=Symbol, Values=Returns_1d
    market_returns = panel.pivot(index='Date', columns='symbol', values='returns_1d')
    
    # 3. Daten synchronisieren
    # Wir nehmen nur Tage, wo wir BEIDES haben (Signal & Markt-Daten)
    common_dates = signals.index.intersection(market_returns.index)
    signals = signals.loc[common_dates]
    market_returns = market_returns.loc[common_dates]
    
    print(f"Simulating {len(common_dates)} trading days...")

    # 4. Die PnL Berechnung (Profit and Loss)
    # WICHTIG: Wir handeln basierend auf dem Signal von GESTERN zum Preis von HEUTE.
    # Deshalb müssen wir die Signale um 1 Tag shiften (laggen).
    # Signal(t-1) * Return(t) = Profit(t)
    
    shifted_signals = signals.shift(1)
    
    # Element-weise Multiplikation: Gewicht * Return
    # Beispiel: 0.2 (Long AAPL) * -0.01 (AAPL fällt 1%) = -0.002 (Verlust)
    # Beispiel: -0.2 (Short NVDA) * -0.03 (NVDA fällt 3%) = +0.006 (Gewinn!)
    strategy_daily_returns = (shifted_signals * market_returns).sum(axis=1)
    
    # Transaktionskosten simulieren (vereinfacht 5 Basispunkte pro Trade)
    # Wir berechnen die Veränderung der Gewichte (Turnover)
    turnover = (signals - shifted_signals).abs().sum(axis=1)
    costs = turnover * 0.0005 
    
    strategy_net_returns = strategy_daily_returns - costs
    
    # 5. Benchmark (Was wäre passiert, wenn wir einfach SPY gehalten hätten?)
    if 'SPY' in market_returns.columns:
        benchmark_returns = market_returns['SPY']
    else:
        # Fallback: Gleichgewichtetes Portfolio aller Aktien
        benchmark_returns = market_returns.mean(axis=1)

    # 6. Kumulierte Performance (Equity Curve)
    # Wir starten bei 1.0 (100%)
    equity_curve = (1 + strategy_net_returns).cumprod()
    benchmark_curve = (1 + benchmark_returns).cumprod()
    
    # 7. Kennzahlen berechnen
    total_return = (equity_curve.iloc[-1] - 1) * 100
    sharpe_ratio = (strategy_net_returns.mean() / strategy_net_returns.std()) * np.sqrt(252) # Jahres-Sharpe
    
    print(f"\n📊 RESULTS:")
    print(f"   Total Return:      {total_return:.2f}%")
    print(f"   Sharpe Ratio:      {sharpe_ratio:.2f} (Ziel > 1.0)")
    print(f"   Final Capital:     {equity_curve.iloc[-1]:.4f}x")

    # 8. Plotten
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve.index, equity_curve, label=f'Trinity AI (Sharpe: {sharpe_ratio:.2f})', color='blue', linewidth=2)
    plt.plot(benchmark_curve.index, benchmark_curve, label='Benchmark (SPY Buy&Hold)', color='gray', linestyle='--', alpha=0.6)
    
    plt.title('Trinity Strategy vs. Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Growth of $1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_img = REPORT_DIR / "performance_chart.png"
    plt.savefig(output_img)
    print(f"\n✅ Chart saved to {output_img}")
    
    # CSV speichern für Excel-Analyse
    equity_curve.to_csv(REPORT_DIR / "equity_curve.csv")

if __name__ == "__main__":
    run_simulation()