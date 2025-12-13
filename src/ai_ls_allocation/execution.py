import os
import pandas as pd
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
from pathlib import Path

# 1. Passwörter laden (.env Datei muss im Hauptordner liegen!)
load_dotenv()
# Wir nutzen .strip(), um versehentliche Leerzeichen oder Enter-Tasten zu entfernen
KEY = os.getenv("ALPACA_KEY", "").strip()
SECRET = os.getenv("ALPACA_SECRET", "").strip()
URL = os.getenv("ALPACA_ENDPOINT", "").strip()

# Debugging: Prüfen, ob Keys da sind (nur die ersten 4 Zeichen zeigen)
print(f"DEBUG: Key: {KEY[:4] if KEY else 'None'}...")
print(f"DEBUG: Endpoint: {URL}")

# Pfad zu deinen KI-Signalen
REPORTS_DIR = Path("reports")
WEIGHTS_FILE = REPORTS_DIR / "bt_neural_weights.csv"

def main():
    print("--- 🤖 AI Execution Engine Starting ---")
    
    # Check: Sind Keys da?
    if not KEY or not SECRET:
        print("❌ FEHLER: Keine Keys in .env gefunden!")
        return

    # 2. Verbindung zu Alpaca herstellen
    try:
        api = tradeapi.REST(KEY, SECRET, base_url=URL, api_version='v2')
        account = api.get_account()
    except Exception as e:
        print(f"❌ Verbindungsfehler: {e}")
        print("Tipp: Prüfe, ob in .env hinter der URL '/v2' steht (das muss weg!)")
        return

    if account.trading_blocked:
        print("❌ Account is blocked!")
        return
        
    equity = float(account.equity)
    print(f"💰 Portfolio Value (Paper): ${equity:,.2f}")

    # 3. KI-Signale laden
    if not WEIGHTS_FILE.exists():
        print("❌ Keine Weights-Datei gefunden. Lass erst bt_neural.py laufen!")
        return

    # CSV laden, letzte Zeile nehmen (= Signale für heute/morgen)
    df = pd.read_csv(WEIGHTS_FILE, index_col=0, parse_dates=True)
    latest_weights = df.iloc[-1]
    last_date = df.index[-1]
    
    print(f"📅 Trading based on AI Signals from: {last_date}")
    
    # Filter: Wir ignorieren Mini-Positionen (< 1%)
    targets = latest_weights[latest_weights.abs() > 0.01].to_dict()
    
    # 4. Aktuelle Positionen abfragen
    current_positions = {p.symbol: float(p.market_value) for p in api.list_positions()}
    
    # 5. Orders berechnen & senden
    for symbol, target_pct in targets.items():
        alpaca_symbol = symbol
        if "BTC" in symbol:
            print(f"⚠️ Skipping Crypto {symbol}")
            continue

        # Ziel-Wert und aktueller Wert
        target_value = equity * target_pct
        current_value = current_positions.get(alpaca_symbol, 0.0)
        diff_value = target_value - current_value
        
        # Filter: Nichts tun bei kleinen Änderungen (< $50)
        if abs(diff_value) < 50: 
            continue
            
        side = 'buy' if diff_value > 0 else 'sell'
        qty_value = round(abs(diff_value), 2) # Dollar-Betrag
        
        print(f"🚀 ORDER PLAN: {side.upper()} {alpaca_symbol} for ~${qty_value} (Target: {target_pct:.1%})")

        try:
            # UNTERSCHEIDUNG: LONG VS SHORT
            if side == 'buy':
                # Long: Wir nutzen 'notional' (Dollar), Fractional Shares erlaubt
                api.submit_order(
                    symbol=alpaca_symbol,
                    notional=qty_value,
                    side=side,
                    type='market',
                    time_in_force='day'
                )
                print(f"   -> ✅ BUY Order sent (${qty_value})")
                
            else:
                # Short: Wir MÜSSEN ganze Stückzahlen nutzen!
                # 1. Aktuellen Preis holen
                quote = api.get_latest_trade(alpaca_symbol)
                price = float(quote.price)
                
                # 2. Stückzahl berechnen und abrunden (floor)
                qty_shares = int(qty_value / price)
                
                if qty_shares < 1:
                    print("   -> ⚠️ Amount too small for 1 whole share, skipping short.")
                    continue
                
                # 3. Order senden mit 'qty' (Stückzahl) statt 'notional'
                api.submit_order(
                    symbol=alpaca_symbol,
                    qty=qty_shares,  # Ganze Zahl!
                    side=side,
                    type='market',
                    time_in_force='day'
                )
                print(f"   -> ✅ SELL SHORT Order sent ({qty_shares} shares @ ~${price})")

        except Exception as e:
            print(f"   -> ❌ Error sending order: {e}")

    # ... (Dein bisheriger Code, der Orders sendet) ...

    # ---------------------------------------------------------
    # 6. AUFRÄUMEN: Positionen schließen, die die KI nicht mehr will
    # ---------------------------------------------------------
    print("\n--- 🧹 Cleaning up Legacy Positions ---")
    
    # Wir holen nochmal frisch die offenen Positionen
    open_positions = api.list_positions()
    
    for p in open_positions:
        symbol_held = p.symbol
        
        # Mapping prüfen (Falls Alpaca andere Namen nutzt)
        # Wenn wir "BTC/USD" halten, aber unser Target "BTC-USD" heißt
        
        # Einfacher Check: Ist das Symbol in unseren KI-Zielen?
        if symbol_held not in targets:
            # Ausnahme: Wir wollen vielleicht manuelle Positionen behalten? 
            # Hier gehen wir davon aus: Der Bot kontrolliert ALLES.
            
            print(f"📉 CLOSING {symbol_held}: AI target is 0%, but we hold ${float(p.market_value):.2f}")
            
            try:
                api.close_position(symbol_held)
                print(f"   -> ✅ Position closed.")
            except Exception as e:
                print(f"   -> ❌ Error closing position: {e}")
        else:
            # Wenn es im Target ist, wurde es oben in der Schleife schon angepasst (Rebalancing).
            pass

    print("--- Execution Finished ---")

if __name__ == "__main__":
    main()
