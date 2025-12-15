import os
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

# --- MAC FIX ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# --- KONFIGURATION ---
FEATURE_DIR = Path("data/features")
FUNDAMENTALS_PATH = FEATURE_DIR / "fundamental_quality.parquet"

# Unsere Watchlist
SYMBOLS = [
    "SPY", "QQQ", "IWM", "EEM", "VGK", 
    "NVDA", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "PLTR", 
    "JPM", "JNJ", "XOM", "BRK-B", "LMT", 
    "GLD", "SLV", "USO", "TLT", 
    "BTC-USD", "ETH-USD"
]

def fetch_fundamentals():
    print("\n--- 💎 STARTING FUNDAMENTAL QUALITY SCAN ---")
    
    quality_data = []
    
    for symbol in SYMBOLS:
        print(f"   >>> 🔍 Auditing {symbol}...", end=" ")
        
        # ETFs und Krypto haben keine klassischen Bilanzen wie Firmen
        # Wir geben ihnen einen neutralen/guten Score, da sie Diversifikation bieten
        if symbol in ["BTC-USD", "ETH-USD", "GLD", "SLV", "USO", "TLT", "SPY", "QQQ", "IWM", "EEM", "VGK"]:
            print("Asset Class (ETF/Crypto) -> Auto-Score: 70")
            quality_data.append({
                'symbol': symbol,
                'Quality_Score': 0.70, # Neutral-Gut
                'PE_Ratio': 0,
                'Profit_Margin': 0
            })
            continue
            
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # 1. Profitabilität (Gewinnmarge)
            margin = info.get('profitMargins', 0)
            if margin is None: margin = 0
            
            # 2. Bewertung (KGV / PE Ratio)
            pe = info.get('trailingPE', 0)
            if pe is None: pe = 50 # Fallback wenn Verlustschreiber
            
            # 3. Schulden (Debt to Equity)
            debt_eq = info.get('debtToEquity', 0)
            if debt_eq is None: debt_eq = 100
            
            # --- SCORING LOGIK (0 bis 100) ---
            score = 50 # Startwert
            
            # Profit Bonus
            if margin > 0.20: score += 20 # Cash Cow (z.B. Nvidia, Google)
            elif margin > 0.10: score += 10
            elif margin < 0: score -= 20 # Geldverbrenner
            
            # PE Malus (Zu teuer?)
            if pe > 100: score -= 10 # Blase?
            if pe < 25 and pe > 0: score += 10 # Value Schnäppchen
            
            # Schulden Malus
            if debt_eq > 200: score -= 10 # Überschuldet
            
            # Limits
            score = max(10, min(100, score))
            normalized_score = score / 100.0
            
            print(f"Score: {score}/100 (Margin: {margin:.1%}, PE: {pe:.1f})")
            
            quality_data.append({
                'symbol': symbol,
                'Quality_Score': normalized_score,
                'PE_Ratio': pe,
                'Profit_Margin': margin
            })
            
        except Exception as e:
            print(f"⚠️ Error: {e}")
            # Fallback bei Fehler
            quality_data.append({'symbol': symbol, 'Quality_Score': 0.5, 'PE_Ratio': 0, 'Profit_Margin': 0})

    # Speichern
    df = pd.DataFrame(quality_data)
    # Datum hinzufügen (Fundamentaldaten ändern sich langsam, wir setzen das heutige Datum)
    # Damit wir es später mergen können
    df['Date'] = pd.Timestamp.now().normalize()
    
    print("-" * 40)
    print(df.sort_values('Quality_Score', ascending=False)[['symbol', 'Quality_Score', 'Profit_Margin']])
    
    df.to_parquet(FUNDAMENTALS_PATH)
    print(f"\n   >>> ✅ Quality Database Updated: {FUNDAMENTALS_PATH}")

if __name__ == "__main__":
    fetch_fundamentals()