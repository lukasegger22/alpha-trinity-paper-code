import pandas as pd
import numpy as np
from pathlib import Path

# --- KONFIGURATION ---
DATA_DIR = Path("data")
INPUT_PATH = DATA_DIR / "panel.parquet"
OUTPUT_DIR = DATA_DIR / "features"
OUTPUT_PATH = OUTPUT_DIR / "panel.parquet"

def calculate_rsi(series, period=14):
    """Berechnet den RSI manuell mit Pandas (ohne externe Lib)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    # Wilder's Smoothing (Standard RSI)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def build_features():
    print("--- 3. Building Features (Zero-Dependency Engine) ---")
    
    if not INPUT_PATH.exists():
        print(f"❌ Critical Error: Input file {INPUT_PATH} not found!")
        return

    # Daten laden
    df = pd.read_parquet(INPUT_PATH).reset_index()
    
    # Spalten klein schreiben
    df.columns = [c.lower() for c in df.columns]
    
    if 'close' not in df.columns:
        print("❌ Error: Keine 'close' Spalte!")
        return

    print("   Berechne Indikatoren (Native Pandas)...")
    processed_dfs = []
    
    # Index auf Date setzen für Berechnungen
    df.set_index('date', inplace=True)
    
    for symbol, group in df.groupby('symbol'):
        group = group.sort_index()
        g = group.copy()
        
        # 1. Returns
        g['returns_1d'] = g['close'].pct_change(1)
        g['returns_5d'] = g['close'].pct_change(5)
        g['returns_20d'] = g['close'].pct_change(20)
        
        # 2. Volatilität
        g['volatility_60d'] = g['returns_1d'].rolling(60).std() * np.sqrt(252)
        
        # 3. SMA (Simple Moving Average) - Einfach mit .rolling().mean()
        g['sma_200'] = g['close'].rolling(window=200).mean()
        
        # 4. RSI (Manuell berechnet)
        g['rsi'] = calculate_rsi(g['close'], period=14)
        
        # 5. Abstand zum SMA
        g['dist_sma200'] = (g['close'] - g['sma_200']) / g['sma_200']
        
        processed_dfs.append(g)

    if not processed_dfs:
        print("❌ Keine Daten verarbeitet.")
        return

    df_features = pd.concat(processed_dfs)
    
    # NaNs entfernen
    orig_len = len(df_features)
    df_features.dropna(inplace=True)
    print(f"   NaNs entfernt: {orig_len} -> {len(df_features)} Zeilen.")

    # Speichern
    df_features.reset_index(inplace=True)
    df_features.set_index(['date', 'symbol'], inplace=True)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(OUTPUT_PATH)
    
    print(f"   💾 Features gespeichert: {OUTPUT_PATH}")

if __name__ == "__main__":
    build_features()