import pandas as pd
import numpy as np
import os
import sys
import time
from pathlib import Path

# --- ALPAC API IMPORTS ---
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest

# --- KONFIGURATION ---
FEATURE_DIR = Path("data/features")
SIGNALS_PATH = FEATURE_DIR / "trinity_signals.parquet"
PRICES_PATH = FEATURE_DIR / "panel.parquet"

# --- 🧠 SMART REBALANCING LOGIC ---
def rebalance_portfolio():
    """
    Vergleicht das aktuelle Portfolio mit den KI-Signalen und handelt nur die DIFFERENZ.
    Spart Gebühren und lässt Gewinne laufen.
    """
    
    # 1. API VERBINDUNG
    API_KEY = os.getenv("ALPACA_API_KEY")
    SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    
    if not all([API_KEY, SECRET_KEY]):
        print("❌ CRITICAL: Keine Alpaca Keys gefunden.")
        sys.exit(2)

    try:
        trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
        account = trading_client.get_account()
        
        # WICHTIG: Wir nutzen jetzt das ECHTE Geld im Account (inkl. Gewinne)
        total_equity = float(account.portfolio_value)
        buying_power = float(account.buying_power)
        
        print(f"\n--- ⚖️ SMART REBALANCING ENGINE ---")
        print(f"   Total Equity:   ${total_equity:,.2f}")
        print(f"   Buying Power:   ${buying_power:,.2f}")
        print("-" * 40)

        # 2. IST-ZUSTAND: Was besitzen wir gerade?
        # Wir laden alle offenen Positionen von Alpaca
        current_positions = {}
        try:
            positions = trading_client.get_all_positions()
            for p in positions:
                current_positions[p.symbol] = int(p.qty)
        except Exception as e:
            print(f"   ⚠️ Konnte Positionen nicht laden: {e}")
            return

        print(f"   Aktuelle Positionen: {current_positions}")

        # 3. SOLL-ZUSTAND: Was will die KI haben?
        if not SIGNALS_PATH.exists():
            print("❌ Keine Signale gefunden.")
            return

        signals_df = pd.read_parquet(SIGNALS_PATH)
        # Die allerneuesten Signale (letzte Zeile)
        target_weights = signals_df.iloc[-1]
        target_weights = target_weights[target_weights > 0.001] # Nur was > 0.1% Gewicht hat

        # 4. PREISE HOLEN (Für Berechnung der Stückzahlen)
        # Wir nutzen das lokale Panel für eine Schätzung, holen aber LIVE Preise für Limits
        panel = pd.read_parquet(PRICES_PATH)
        # Fallback Preise von gestern (falls Live gleich fehlschlägt)
        fallback_prices = panel.groupby('symbol')['Close'].last()

        # 5. DIE DIFFERENZ BERECHNEN (Delta)
        # Wir gehen durch ALLE Symbole (die wir haben + die wir wollen)
        all_symbols = set(current_positions.keys()) | set(target_weights.index)
        
        orders_to_send = []

        for symbol in all_symbols:
            # A. Aktueller Preis holen (Live Check)
            try:
                req = StockLatestTradeRequest(symbol_or_symbols=symbol)
                res = data_client.get_stock_latest_trade(req)
                price = res[symbol].price
            except:
                # Falls Live fehlschlägt, nimm Gestern oder überspringe
                price = fallback_prices.get(symbol, None)
                if price is None:
                    print(f"   ⚠️ Kein Preis für {symbol}. Überspringe.")
                    continue

            # B. Ziel-Menge berechnen
            weight = target_weights.get(symbol, 0.0) # Wenn KI es nicht will, ist Gewicht 0
            target_value = total_equity * weight
            target_shares = int(target_value / price)
            
            # C. Aktuelle Menge
            current_shares = current_positions.get(symbol, 0)
            
            # D. Das Delta (Die Differenz)
            diff = target_shares - current_shares
            
            # E. Filter: Lohnt sich der Trade? (Rauschen filtern)
            # Wir handeln nur, wenn wir mindestens 1 Aktie bewegen müssen
            if diff == 0:
                print(f"   ✅ {symbol}: Halte {current_shares} Shares (Perfekt gewichtet).")
                continue
            
            action = OrderSide.BUY if diff > 0 else OrderSide.SELL
            qty = abs(diff)
            
            # Speichern für Execution
            orders_to_send.append({
                'symbol': symbol,
                'qty': qty,
                'side': action,
                'price': price # Merk dir den Preis für das Limit
            })

        # 6. ORDERS AUSFÜHREN (Verkäufe zuerst, um Cash freizumachen!)
        # Wir sortieren: Erst SELLS, dann BUYS
        orders_to_send.sort(key=lambda x: x['side'] == OrderSide.BUY) # False (Sell) kommt vor True (Buy)

        print("-" * 40)
        
        if not orders_to_send:
            print("   💤 Keine Änderungen nötig. Portfolio ist optimal.")
            return

        for order in orders_to_send:
            symbol = order['symbol']
            qty = order['qty']
            side = order['side']
            base_price = order['price']
            
            # Limit Preis berechnen (0.5% Puffer)
            limit_buffer = 0.005
            if side == OrderSide.BUY:
                limit_price = round(base_price * (1 + limit_buffer), 2)
            else:
                limit_price = round(base_price * (1 - limit_buffer), 2)

            try:
                # Alte offene Orders für dieses Symbol löschen (damit nichts doppelt läuft)
                # trading_client.cancel_orders(symbols=[symbol]) # Optional, falls unterstützt
                
                req = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type='limit',
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price
                )
                
                trading_client.submit_order(order_data=req)
                
                arrow = "🟢 KAUFE" if side == OrderSide.BUY else "🔴 VERKAUFE"
                print(f"   {arrow} {qty} x {symbol} @ Limit {limit_price} (Aktuell: {current_positions.get(symbol,0)} -> Ziel: {current_positions.get(symbol,0) + (qty if side==OrderSide.BUY else -qty)})")
                
            except Exception as e:
                print(f"   ❌ Fehler bei {symbol}: {e}")

    except Exception as e:
        print(f"\n❌ CRITICAL SYSTEM ERROR: {e}")

if __name__ == "__main__":
    rebalance_portfolio()
