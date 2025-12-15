import pandas as pd
import numpy as np
import os # NEU: Für Environment Variablen (API Keys)
import time # NEU: Für Wartezeiten
from pathlib import Path
from alpaca_trade_api.rest import REST, TimeFrame # NEU: Alpaca API

# --- KONFIGURATION ---
CAPITAL = 100000.0
FEATURE_DIR = Path("data/features")
SIGNALS_PATH = FEATURE_DIR / "trinity_signals.parquet"
PRICES_PATH = FEATURE_DIR / "panel.parquet"

# --- 🎯 NEUE FUNKTION: ALPINE EXECUTION CORE ---
def execute_orders(orders_list):
    """
    Stellt die Verbindung zu Alpaca her, liquidiert alte Positionen
    und sendet die neuen Orders basierend auf der berechneten Liste.
    """
    
    # 1. API-KEYS aus Umgebungsvariablen laden (von GitHub Secrets!)
    API_KEY = os.getenv("ALPACA_API_KEY")
    SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    ENDPOINT = os.getenv("ALPACA_ENDPOINT")

    if not all([API_KEY, SECRET_KEY, ENDPOINT]):
        print("\n❌ CRITICAL: Alpaca API Keys not found in environment variables!")
        print("   Orders were NOT placed. Please check GitHub Secrets.")
        return False
        
    try:
        # Initialisierung der Alpaca REST API
        api = REST(API_KEY, SECRET_KEY, ENDPOINT)
        account = api.get_account()
        
        print(f"\n--- 🌐 ALPACA CONNECTION SUCCESS ---")
        print(f"   Account Status: {account.status}")
        print(f"   Buying Power:   ${float(account.buying_power):,.2f}")
        print("-" * 35)

        # 2. Alte Positionen schließen (Strategie: Täglich neu aufbauen)
        print("   2.1. Liquidating all previous positions...")
        api.close_all_positions()
        time.sleep(2) # Kurze Pause, damit Alpaca die Liquidation verarbeiten kann
        print("   ✅ Liquidation request sent.")
        
        # 3. Neue Orders senden
        print("   2.2. Sending new market orders...")
        
        # Sicherheits-Check: Order nur senden, wenn der Markt offen ist oder kurz davor
        # Für Paper Trading vereinfachen wir das, aber in der Live-Version WICHTIG!
        
        successful_orders = 0
        
        for order in orders_list:
            symbol = order['SYMBOL']
            shares = abs(order['SHARES'])
            side = 'buy' if order['ACTION'] == 'BUY' else 'sell'
            
            # Alpaca akzeptiert keine Orders mit 0 Shares!
            if shares == 0:
                print(f"      - Skipping {order['SYMBOL']} (0 Shares).")
                continue
                
            try:
                api.submit_order(
                    symbol=symbol,
                    qty=shares,
                    side=side,
                    type='market',
                    time_in_force='day'
                )
                print(f"      - ✅ ORDER SENT: {side.upper()} {shares} shares of {symbol}")
                successful_orders += 1
            except Exception as e:
                print(f"      - ❌ ORDER FAILED for {symbol}: {e}")
                
        print("-" * 35)
        print(f"   🎉 Execution finished. Total {successful_orders} orders sent.")
        return True
        
    except Exception as e:
        print(f"\n❌ CRITICAL CONNECTION ERROR: {e}")
        print("   Orders were NOT placed.")
        return False

# --- BESTEHENDE LOGIK (JETZT MIT RÜCKGABEWERT) ---
def generate_orders():
    # ... (Dein gesamter Code von Zeile 10 bis 50 bleibt fast gleich) ...
    # HINWEIS: Ich habe den Code gekürzt, um die neue Funktion zu zeigen.
    # Du musst nur diese Funktion in deiner execution.py ersetzen, 
    # und generate_orders so anpassen, dass es eine Liste zurückgibt.
    
    # NEU: Liste für die Orders erstellen
    orders_to_execute = [] 
    
    # ... (Deine Lade- und Berechnungslogik) ...

    # ... (Innerhalb der Schleife, wo du printest) ...
    for symbol, weight in active_positions.items():
        # ... (Deine Berechnung von price, position_value, shares, action) ...
        
        print(f"{action:<6} | {symbol:<8} | {weight*100:>6.1f}% | {position_value:>10.2f} | {price:>10.2f} | {shares:>8}")
        
        # NEU: Füge die Order zur Liste hinzu
        orders_to_execute.append({
            'SYMBOL': symbol,
            'SHARES': shares,
            'ACTION': action
        })
        
        total_invested += abs(position_value)

    print("-" * 65)
    print(f"Gross Exposure: ${total_invested:,.2f} ({total_invested/CAPITAL*100:.1f}%)")
    print("=" * 65)
    print("🚀 Ready to execute at market open!")
    
    # NEU: Gib die Liste zurück
    return orders_to_execute

# --- HAUPTFUNKTION (JETZT MIT EXECUTION) ---
if __name__ == "__main__":
    orders = generate_orders()
    
    # Führe nur aus, wenn Orders berechnet wurden
    if orders:
        execute_orders(orders)