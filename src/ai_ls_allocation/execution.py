import pandas as pd
import numpy as np
from pathlib import Path

# --- KONFIGURATION ---
CAPITAL = 10000.0  # Dein Startkapital in Dollar/Euro
FEATURE_DIR = Path("data/features")
SIGNALS_PATH = FEATURE_DIR / "trinity_signals.parquet"
PRICES_PATH = FEATURE_DIR / "panel.parquet"

def generate_orders():
    print(f"--- 🛒 TRINITY EXECUTION SYSTEM (Capital: ${CAPITAL:,.2f}) ---")
    
    # 1. Neueste Signale laden (Was will der Bot?)
    if not SIGNALS_PATH.exists():
        raise FileNotFoundError("Keine Signale gefunden! Lass erst bt_trinity.py laufen.")
    
    signals_df = pd.read_parquet(SIGNALS_PATH)
    # Wir nehmen die allerletzte Zeile (Das Signal für MORGEN)
    target_weights = signals_df.iloc[-1]
    target_date = signals_df.index[-1]
    
    # Filtern: Nur Assets mit Gewicht != 0
    active_positions = target_weights[target_weights != 0].sort_values(ascending=False)
    
    if active_positions.empty:
        print(f"[{target_date}] 😴 Bot is 100% Cash. No trades needed.")
        return

    # 2. Neueste Preise laden (Was kostet das Zeug?)
    panel = pd.read_parquet(PRICES_PATH)
    # Wir brauchen den 'Close' Preis des letzten verfügbaren Tages
    latest_prices = panel.groupby('symbol')['Close'].last()
    
    print(f"\n📅 PLAN FOR: {target_date} (Based on Signal)")
    print("-" * 65)
    print(f"{'ACTION':<6} | {'SYMBOL':<8} | {'WEIGHT':<8} | {'VALUE ($)':<10} | {'PRICE ($)':<10} | {'SHARES':<8}")
    print("-" * 65)
    
    total_invested = 0
    
    for symbol, weight in active_positions.items():
        if symbol not in latest_prices:
            print(f"⚠️  WARNING: No price found for {symbol}. Skipping.")
            continue
            
        price = latest_prices[symbol]
        
        # Wie viel Geld in diese Aktie?
        # Gewicht * Kapital (z.B. 0.35 * 10.000 = 3.500$)
        position_value = weight * CAPITAL
        
        # Wie viele Aktien? (Gerundet auf ganze Zahlen)
        # Bei Short-Positionen ist der Value negativ, Shares also auch negativ
        shares = int(position_value / price)
        
        # Action bestimmen
        action = "BUY" if shares > 0 else "SELL" # (SELL SHORT)
        
        print(f"{action:<6} | {symbol:<8} | {weight*100:>6.1f}% | {position_value:>10.2f} | {price:>10.2f} | {shares:>8}")
        
        total_invested += abs(position_value)

    print("-" * 65)
    cash = CAPITAL - total_invested
    # Hinweis: Bei Short-Selling ist Cash-Rechnung komplexer (Margin), 
    # aber hier vereinfacht als "Nicht investiertes Kapital".
    print(f"Gross Exposure: ${total_invested:,.2f} ({total_invested/CAPITAL*100:.1f}%)")
    print("=" * 65)
    print("🚀 Ready to execute at market open!")

if __name__ == "__main__":
    generate_orders()