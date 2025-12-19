import pandas as pd
import pandas_ta as ta
import numpy as np
from pathlib import Path

# --- KONFIGURATION ---
DATA_DIR = Path("data")
INPUT_PATH = DATA_DIR / "panel.parquet"       # Das kommt vom Downloader
OUTPUT_DIR = DATA_DIR / "features"            # Da soll es hin
OUTPUT_PATH = OUTPUT_DIR / "panel.parquet"    # Das braucht das Training

def build_features():
    print("--- 3. Building Features (Smart Engine) ---")
    
    # 1. Daten laden (die wir gerade mit VIX erstellt haben)
    if not INPUT_PATH.exists():
        print(f"❌ Critical Error: Input file {INPUT_PATH} not found!")
        print("   Did you run download.py?")
        return

    df = pd.read_parquet(INPUT_PATH)
    print(f"   Daten geladen. Shape: {df.shape}")
    
    # Check ob VIX da ist
    if 'vix_close' in df.columns:
        print("   ✅ VIX Feature gefunden. Wird durchgeschleift.")
    else:
        print("   ⚠️ WARNUNG: VIX fehlt! Training wird schlechter sein.")

    # 2. Technische Indikatoren berechnen (RSI, SMA, Volatilität)
    # Wir gruppieren nach Symbol, damit Indikatoren pro Aktie berechnet werden
    # und nicht "über" die Aktien hinweg vermischt werden.
    
    # Liste für Ergebnisse
    processed_dfs = []
    
    for symbol, group in df.groupby(level='symbol'):
        group = group.sort_index() # Sicherstellen, dass Zeit stimmt
        
        # Kopie um Warnungen zu vermeiden
        g = group.copy()
        
        # --- FEATURE ENGINEERING ---
        
        # RSI (Relative Strength Index)
        g['rsi'] = ta.rsi(g['close'], length=14)
        
        # SMA (Simple Moving Average) - Trend
        g['sma_50'] = ta.sma(g['close'], length=50)
        g['sma_200'] = ta.sma(g['close'], length=200)
        
        # Abstand zum SMA (Trend-Stärke)
        g['dist_sma200'] = (g['close'] - g['sma_200']) / g['sma_200']
        
        # Volatilität (Rolling Std Dev)
        g['volatility_20'] = g['close'].pct_change().rolling(20).std()
        
        # Returns (für Zielvariable beim Training wichtig)
        g['return_1d'] = g['close'].pct_change()
        g['return_5d'] = g['close'].pct_change(5)
        
        # VIX lassen wir so wie er ist (er ist ja schon gemerged)
        
        processed_dfs.append(g)

    # 3. Wieder zusammenfügen
    df_features = pd.concat(processed_dfs)
    
    # NaN Werte entfernen (die ersten 200 Tage fehlen wegen SMA200)
    original_len = len(df_features)
    df_features.dropna(inplace=True)
    print(f"   NaNs entfernt: {original_len} -> {len(df_features)} Zeilen übrig.")

    # 4. Speichern
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(OUTPUT_PATH)
    
    print(f"   💾 Features gespeichert nach: {OUTPUT_PATH}")
    print("--- Feature Engineering Complete ---")

if __name__ == "__main__":
    build_features()