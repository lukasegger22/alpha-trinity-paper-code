import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- PFADE ---
FEATURE_DIR = Path("data/features")
REPORT_DIR = Path("reports")
HISTORY_PATH = FEATURE_DIR / "trinity_history.parquet"
PRICES_PATH = FEATURE_DIR / "panel.parquet"

def run_stress_test(num_simulations=1000, days=252):
    print(f"--- 🌪️ RUNNING MONTE CARLO STRESS TEST ({num_simulations} Simulations) ---")
    
    # 1. Daten laden & Returns berechnen
    if not HISTORY_PATH.exists():
        raise FileNotFoundError("Run bt_trinity.py first!")
    
    weights = pd.read_parquet(HISTORY_PATH)
    panel = pd.read_parquet(PRICES_PATH)
    if 'Date' in panel.columns:
        panel['Date'] = pd.to_datetime(panel['Date']).dt.normalize()
    
    market_returns = panel.pivot(index='Date', columns='symbol', values='returns_1d')
    common_dates = weights.index.intersection(market_returns.index)
    
    # Strategie-Returns (Netto nach Kosten)
    weights = weights.loc[common_dates]
    market_returns = market_returns.loc[common_dates]
    shifted_weights = weights.shift(1).fillna(0)
    turnover = (weights - shifted_weights).abs().sum(axis=1)
    costs = turnover * 0.0010 # 10bps Kosten
    
    strategy_returns = (shifted_weights * market_returns).sum(axis=1) - costs
    strategy_returns = strategy_returns.dropna()
    
    # 2. Monte Carlo Simulation
    # Wir nehmen die echten täglichen Returns deiner Strategie und mischen sie zufällig neu.
    # Das zerstört die zeitliche Reihenfolge, zeigt aber, was statistisch möglich ist.
    
    simulation_results = []
    
    np.random.seed(42) # Für Reproduzierbarkeit
    
    for i in range(num_simulations):
        # Zufällige Auswahl von Tagen aus deiner Historie (mit Zurücklegen)
        random_returns = np.random.choice(strategy_returns, size=days, replace=True)
        
        # Kumulierte Kurve berechnen
        cum_path = (1 + random_returns).cumprod()
        simulation_results.append(cum_path)

    # 3. Auswertung
    sim_df = pd.DataFrame(simulation_results).T
    
    # Quantile berechnen (Best Case, Worst Case, Median)
    worst_case = sim_df.iloc[-1].quantile(0.05) # Die unteren 5% (Pech)
    median_case = sim_df.iloc[-1].quantile(0.50) # Der Normalfall
    best_case = sim_df.iloc[-1].quantile(0.95) # Die oberen 5% (Glück)
    
    # Value at Risk (VaR) - Historisch auf Tagesbasis
    var_95 = strategy_returns.quantile(0.05)
    var_99 = strategy_returns.quantile(0.01)

    print("\n" + "="*40)
    print("   STRESS TEST RESULTS (1 Year Forecast)")
    print("="*40)
    print(f"Based on historical strategy returns:")
    print(f"  Expected Return (Median):  {(median_case - 1)*100:.2f}%")
    print(f"  Worst Case (5% Chance):    {(worst_case - 1)*100:.2f}%  <-- WICHTIG")
    print(f"  Best Case (5% Chance):     {(best_case - 1)*100:.2f}%")
    print("-" * 40)
    print("RISK METRICS (Daily):")
    print(f"  VaR (95% Confidence):      {var_95*100:.2f}%")
    print(f"  VaR (99% Confidence):      {var_99*100:.2f}%")
    print(f"  (Bedeutet: Mit 99% Wahrscheinlichkeit verlierst du")
    print(f"   morgen NICHT mehr als {abs(var_99)*100:.2f}%)")
    print("="*40)

    # 4. Plotten ("Spaghetti Plot")
    plt.figure(figsize=(12, 6))
    
    # Zeichne die ersten 100 Simulationen in grau
    plt.plot(sim_df.iloc[:, :100], color='gray', alpha=0.1, linewidth=0.5)
    
    # Zeichne die Quantile
    plt.plot(sim_df.mean(axis=1), color='blue', linewidth=2, label='Average Path')
    plt.plot(sim_df.quantile(0.05, axis=1), color='red', linestyle='--', linewidth=2, label='Worst Case (95%)')
    
    plt.title(f'Monte Carlo Simulation (1000 Scenarios for next {days} days)')
    plt.ylabel('Growth of $1')
    plt.xlabel('Days into Future')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = REPORT_DIR / "monte_carlo_stress.png"
    plt.savefig(output_path)
    print(f"\n📉 Monte Carlo chart saved to {output_path}")

if __name__ == "__main__":
    run_stress_test()