import pandas as pd
import numpy as np
import os
import time
from pathlib import Path

# --- NEUE ALPAC API IMPORTS ---
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# --- KONFIGURATION ---
# WICHTIG: Setze CAPITAL auf 10000.0, damit die Orders im Paper-Konto passen.
CAPITAL = 10000.0
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
    
    # ENDPOINT wird bei der neuen SDK automatisch durch den 'paper' Flag gesteuert
    if not all([API_KEY, SECRET_KEY]):
        print("\n❌ CRITICAL: Alpaca API Keys not found in environment variables!")
        print("   Orders wurden NICHT platziert. Bitte GitHub Secrets prüfen.")
        return False
        
    try:
        # 1. Initialisierung der Alpaca Trading API (Neue SDK-Struktur)
        # Wir setzen 'paper=True' da wir davon ausgehen, dass der ENDPOINT der Paper-API ist.
        trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        account = trading_client.get_account()
        
        print(f"\n--- 🌐 ALPACA CONNECTION SUCCESS ---")
        print(f"   Account Status: {account.status}")
        print(f"   Buying Power:   ${float(account.buying_power):,.2f}")
        print("-" * 35)

        # 2. Alte Positionen schließen (Liquidieren)
        print("   2.1. Liquidating all previous positions and cancelling orders...")
        # Die neue Methode schließt alle Positionen und storniert alle offenen Orders in einem Aufruf
        trading_client.close_all_positions(cancel_orders=True)
        print("   ✅ Liquidation request sent.")
        
        # 3. Neue Orders senden
        print("   2.2. Sending new market orders...")
        
        successful_orders = 0
        
        for order in orders_list:
            symbol = order['SYMBOL']
            qty = abs(order['SHARES'])
            side = OrderSide.BUY if order['ACTION'] == 'BUY' else OrderSide.SELL

            if qty == 0:
                print(f"      - Skipping {order['SYMBOL']} (0 Shares).")
                continue
                
            try:
                # Neue SDK-Syntax für eine Market Order
                market_order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
                
                trading_client.submit_order(order_data=market_order_data)
                
                print(f"      - ✅ ORDER SENT: {side.value.upper()} {qty} shares of {symbol}")
                successful_orders += 1
            except Exception as e:
                print(f"      - ❌ ORDER FAILED for {symbol}: {e}")
                
        print("-" * 35)
        print(f"   🎉 Execution finished. Total {successful_orders} orders sent.")
        return True
        
    except Exception as e:
        print(f"\n❌ CRITICAL CONNECTION ERROR: {e}")
        print("   Orders wurden NICHT platziert.")
        return False

# --- BESTEHENDE LOGIK (JETZT MIT KORREKTEM RÜCKGABEWERT) ---
def generate_orders():
    print(f"--- 🛒 TRINITY EXECUTION SYSTEM (Capital: ${CAPITAL:,.2f}) ---")
    
    # 1. Neueste Signale laden (Was will der Bot?)
    if not SIGNALS_PATH.exists():
        # Wir heben den Fehler auf, damit der Runner abstürzt, falls kein Signal da ist
        raise FileNotFoundError("Keine Signale gefunden! Lass erst bt_trinity.py laufen.")
    
    signals_df = pd.read_parquet(SIGNALS_PATH)
    target_weights = signals_df.iloc[-1]
    target_date = signals_df.index[-1]
    
    active_positions = target_weights[target_weights != 0].sort_values(ascending=False)
    
    if active_positions.empty:
        print(f"[{target_date}] 😴 Bot is 100% Cash. No trades needed.")
        return []

    # 2. Neueste Preise laden (Was kostet das Zeug?)
    panel = pd.read_parquet(PRICES_PATH)
    latest_prices = panel.groupby('symbol')['Close'].last()
    
    print(f"\n📅 PLAN FOR: {target_date} (Based on Signal)")
    print("-" * 65)
    print(f"{'ACTION':<6} | {'SYMBOL':<8} | {'WEIGHT':<8} | {'VALUE ($)':<10} | {'PRICE ($)':<10} | {'SHARES':<8}")
    print("-" * 65)
    
    total_invested = 0
    orders_to_execute = [] 
    
    for symbol, weight in active_positions.items():
        if symbol not in latest_prices:
            print(f"⚠️  WARNING: No price found for {symbol}. Skipping.")
            continue
            
        price = latest_prices[symbol]
        position_value = weight * CAPITAL
        shares = int(position_value / price)
        action = "BUY" if shares > 0 else "SELL"
        
        print(f"{action:<6} | {symbol:<8} | {weight*100:>6.1f}% | {position_value:>10.2f} | {price:>10.2f} | {shares:>8}")
        
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
    
    return orders_to_execute

# --- HAUPTFUNKTION (JETZT MIT EXECUTION) ---
if __name__ == "__main__":
    orders = generate_orders()
    
    # Führe nur aus, wenn Orders berechnet wurden
    if orders:
        execute_orders(orders)