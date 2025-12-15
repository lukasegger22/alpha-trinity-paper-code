import os
import time

# --- MAC FIX ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import google.generativeai as genai

# --- KONFIGURATION ---
FEATURE_DIR = Path("data/features")
SENTIMENT_PATH = FEATURE_DIR / "sentiment_signals.parquet"

# !!! HIER DEINEN API KEY EINFÜGEN !!!
GEMINI_API_KEY = "AIzaSyBDugTdEvFLt31k245kzxMnWnzNRhburM0" 

# Wir testen wieder mit den wichtigsten
# Die komplette Trinity-Watchlist
SYMBOLS = [
    "SPY", "QQQ", "IWM", "EEM", "VGK", 
    "NVDA", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "PLTR", 
    "JPM", "JNJ", "XOM", "BRK-B", "LMT", 
    "GLD", "SLV", "USO", "TLT", 
    "BTC-USD", "ETH-USD"
]

def analyze_sentiment_with_gemini(symbol, headlines):
    genai.configure(api_key=GEMINI_API_KEY)
    
    # FIX: Wir nehmen den Alias, der IMMER funktioniert
    model = genai.GenerativeModel('gemini-flash-latest')
    
    prompt = f"""
    Role: Senior Hedge Fund Analyst.
    Task: Rate the sentiment for the asset '{symbol}' based on these headlines.
    Scale: -1.0 (Very Negative) to 1.0 (Very Positive). 0.0 is Neutral.
    
    Headlines:
    {headlines}
    
    Instructions:
    - Analyze the *impact* on the stock price.
    - Output ONLY the floating point number (e.g. 0.45). No text.
    """
    
    try:
        response = model.generate_content(prompt)
        text_val = response.text.strip()
        
        # Zahl extrahieren
        import re
        match = re.search(r"[-+]?\d*\.\d+|\d+", text_val)
        if match:
            return float(match.group())
        return 0.0
    except Exception as e:
        print(f"       ⚠️ Gemini API Error: {e}")
        # Wenn wir ins Limit laufen (429), warten wir kurz und geben 0 zurück
        if "429" in str(e):
            print("       ⏳ Quota Hit - Cooling down...")
            time.sleep(5)
        return 0.0

def build_sentiment_engine():
    print("\n--- 🗞️ STARTING SENTIMENT ENGINE (Gemini Flash Stable) ---")
    
    if "DEIN_GEMINI" in GEMINI_API_KEY:
        print("❌ ERROR: API Key fehlt im Skript!")
        return

    # --- 1. DER HARDWARE TEST (Fake News) ---
    print("   >>> 🧪 Running Hardware Test (Fake News)...")
    test_symbol = "TEST-COIN"
    test_news = "TEST-COIN profits are up 500%. CEO is very happy."
    
    score = analyze_sentiment_with_gemini(test_symbol, test_news)
    print(f"       -> TEST-COIN Score: {score}")
    
    if score == 0.0:
        print("   ❌ Test fehlgeschlagen oder neutral. Warte kurz...")
        time.sleep(2)
    else:
        print("   ✅ Test erfolgreich! Gemini arbeitet.")

    # --- 2. ECHTER SCAN ---
    all_sentiments = []
    
    print("   >>> 📡 Scanning Real Assets...")
    
    for symbol in SYMBOLS:
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            headlines_text = ""
            count = 0
            
            if news:
                for item in news:
                    title = item.get('title', '')
                    if len(title) > 5:
                        headlines_text += f"- {title}\n"
                        count += 1
            
            if count == 0:
                continue

            # Wir sind besonders vorsichtig mit dem Rate Limit
            time.sleep(4.0) 
            
            ai_score = analyze_sentiment_with_gemini(symbol, headlines_text)
            
            emoji = "😐"
            if ai_score > 0.2: emoji = "🚀"
            if ai_score < -0.2: emoji = "🐻"
            
            print(f"       -> {symbol}: {count} headlines -> Gemini Score: {ai_score:.2f} {emoji}")
            
            all_sentiments.append({
                'symbol': symbol,
                'Date': pd.Timestamp.now().normalize(),
                'sentiment_score': ai_score,
                'news_count': count
            })
            
        except Exception as e:
            print(f"       ⚠️ Error {symbol}: {e}")

    # Speichern
    if all_sentiments:
        df_sent = pd.DataFrame(all_sentiments)
        if SENTIMENT_PATH.exists():
            try:
                old_df = pd.read_parquet(SENTIMENT_PATH)
                combined = pd.concat([old_df, df_sent]).drop_duplicates(subset=['Date', 'symbol'], keep='last')
                combined.to_parquet(SENTIMENT_PATH)
                print(f"   >>> ✅ Saved {len(all_sentiments)} signals.")
            except:
                df_sent.to_parquet(SENTIMENT_PATH)
        else:
            df_sent.to_parquet(SENTIMENT_PATH)
            print(f"   >>> ✅ Created database.")
    else:
        print("   >>> ⚠️ No news found via Yahoo (Weekend?), but connection works.")

if __name__ == "__main__":
    build_sentiment_engine()