import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- KONFIGURATION ---
FEATURE_DIR = Path("data/features")
HISTORY_PATH = FEATURE_DIR / "trinity_history.parquet"
PANEL_PATH = FEATURE_DIR / "panel.parquet"

# Harte Realitätseinstellungen
BASE_COST_OF_CARRY = 0.06 
BASE_TRANSACTION_COST = 0.0010
SIMULATIONS = 1000  # Wie oft würfeln wir?

def run_stress_test():
    print(f"\n--- 🌪️ TRINITY STRESS TEST (Monte Carlo & Worst Case) ---")
    
    if not HISTORY_PATH.exists():
        print("❌ Keine History gefunden.")
        return

    # 1. Daten laden
    weights = pd.read_parquet(HISTORY_PATH)
    panel = pd.read_parquet(PANEL_PATH)
    
    panel = panel.reset_index()
    if 'Date' not in panel.columns:
        if 'index' in panel.columns: panel = panel.rename(columns={'index': 'Date'})
        else: panel = panel.rename(columns={panel.columns[0]: 'Date'})
    panel['Date'] = pd.to_datetime(panel['Date'])
    
    returns = panel.pivot(index='Date', columns='symbol', values='returns_1d').fillna(0)
    
    # Schnittmenge
    common_dates = weights.index.intersection(returns.index)
    weights = weights.loc[common_dates]
    returns = returns.loc[common_dates]

    # 2. Basis-Performance berechnen (Die "Echte" Kurve)
    daily_gross = (weights * returns).sum(axis=1)
    
    # Kosten abziehen
    leverage = weights.sum(axis=1)
    interest_cost = ((leverage - 1.0).clip(lower=0) * BASE_COST_OF_CARRY) / 252
    turnover = weights.diff().abs().sum(axis=1).fillna(0) 
    trading_cost = turnover * BASE_TRANSACTION_COST
    
    real_net_returns = daily_gross - interest_cost - trading_cost
    
    print(f"📊 Datenpunkte: {len(real_net_returns)} Handelstage")
    print("-" * 40)

    # --- TEST 1: DER KOSTEN-SCHOCK (Slippage x2) ---
    print("Test 1: Doppelte Gebühren & Zinsen...")
    stress_net = daily_gross - (interest_cost * 1.5) - (trading_cost * 2.0)
    stress_total = (1 + stress_net).cumprod().iloc[-1] - 1
    print(f"   -> Rendite bei doppelten Kosten: {stress_total:.2%}")
    if stress_total > 0: print("   ✅ BESTANDEN (System ist profitabel auch bei hohen Kosten)")
    else: print("   ❌ DURCHGEFALLEN (System ist zu sensibel auf Gebühren)")
    
    # --- TEST 2: MONTE CARLO SIMULATION (Glück vs. Können) ---
    print("-" * 40)
    print(f"Test 2: Monte Carlo Simulation ({SIMULATIONS} Runs)...")
    print("   (Wir mischen die Tagesrenditen zufällig, um Glück auszuschließen)")
    
    final_values = []
    max_drawdowns = []
    
    # Die Renditen als Array für Speed
    ret_array = real_net_returns.values
    
    for i in range(SIMULATIONS):
        # Zufälliges Mischen der Tage (Bootstrap)
        # Wir simulieren: Was wäre, wenn die Markttage in anderer Reihenfolge kamen?
        np.random.shuffle(ret_array) 
        
        # Equity Curve berechnen
        equity_curve = np.cumprod(1 + ret_array)
        final_values.append(equity_curve[-1])
        
        # Max Drawdown dieses Runs
        running_max = np.maximum.accumulate(equity_curve)
        dd = (equity_curve / running_max) - 1
        max_drawdowns.append(dd.min())

    # Statistiken
    median_ret = np.median(final_values) - 1
    worst_case_ret = np.percentile(final_values, 5) - 1 # 5% Quantil (Worst Case)
    prob_loss = np.sum(np.array(final_values) < 1.0) / SIMULATIONS
    avg_dd = np.mean(max_drawdowns)
    
    print(f"\n🎲 ERGEBNISSE DER SIMULATION:")
    print(f"   🔹 Median Return (Erwartungswert): {median_ret:.2%}")
    print(f"   🔹 Worst Case (95% Sicherheit):    {worst_case_ret:.2%}")
    print(f"   🔹 Ø Max Drawdown:                 {avg_dd:.2%}")
    print(f"   🔹 Wahrscheinlichkeit für Verlust: {prob_loss:.1%}")
    
    print("-" * 40)
    if prob_loss < 0.05:
        print("🏆 URTEIL: INSTITUTIONAL GRADE. (Verlustwahrscheinlichkeit < 5%)")
    elif prob_loss < 0.20:
        print("✅ URTEIL: SOLIDE. (Aber Risiko vorhanden)")
    else:
        print("⚠️ URTEIL: GLÜCKSSPIEL. (Zu hohe Wahrscheinlichkeit für Verlust)")
    print("-" * 40)

if __name__ == "__main__":
    run_stress_test()