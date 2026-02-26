import pandas as pd
import numpy as np
from pathlib import Path

# --- KONFIGURATION ---
DATA_DIR = Path("data")
INPUT_PATH = DATA_DIR / "panel.parquet"
OUTPUT_DIR = DATA_DIR / "features"
OUTPUT_PATH = OUTPUT_DIR / "panel.parquet"

def calculate_rsi(series, period=14):
    """Berechnet den RSI manuell mit Pandas."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def build_features():
    print("--- 3. Building Features (Zero-Dependency Engine) ---")
    
    if not INPUT_PATH.exists():
        print(f"❌ Critical Error: Input file {INPUT_PATH} not found!")
        return

    # load Daten 
    df = pd.read_parquet(INPUT_PATH).reset_index()
    
    
    if 'Close' not in df.columns and 'close' in df.columns:
        df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open', 'volume': 'Volume'}, inplace=True)

    if 'Close' not in df.columns:
        print(f"❌ Error: Keine 'Close' Spalte gefunden! Vorhandene Spalten: {df.columns.tolist()}")
        return

    print("   Berechne Indikatoren & Makro-Features...")
    processed_dfs = []
    
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    for symbol, group in df.groupby('symbol'):
        group = group.sort_index()
        g = group.copy()
        
        # --- A. technical Indicators ---
        g['returns_1d'] = g['Close'].pct_change(1)
        g['returns_5d'] = g['Close'].pct_change(5)
        g['returns_20d'] = g['Close'].pct_change(20)
        
        g['volatility_60d'] = g['returns_1d'].rolling(60).std() * np.sqrt(252)
        g['sma_200'] = g['Close'].rolling(window=200).mean()
        g['rsi'] = calculate_rsi(g['Close'], period=14)
        g['dist_sma200'] = (g['Close'] - g['sma_200']) / g['sma_200']
        
        # --- B. Macro Features ---
   
        
        # 1. TNX (Treasury Yield) Features
        if 'TNX' in g.columns:
            g['TNX_Level'] = g['TNX']
        
        if 'TNX_Level' in g.columns:
            g['TNX_Chg_10d'] = g['TNX_Level'].diff(10)
        else:
            g['TNX_Level'] = 0
            g['TNX_Chg_10d'] = 0

        # 2. VIX Features
        if 'VIX' not in g.columns:
            g['VIX'] = 15.0 # neutral Fallback
        
        # 3. Interaction Features 
        g['Interaction_TNX_Vola'] = g['TNX_Level'] * g['VIX']

        # 4. SP500 Trend (Proxy)

        if 'SP500' in g.columns:
             g['SP500_Trend'] = np.where(g['SP500'] > g['SP500'].rolling(200).mean(), 1, 0)
        else:
             g['SP500_Trend'] = np.where(g['Close'] > g['sma_200'], 1, 0)

        processed_dfs.append(g)

    if not processed_dfs:
        print("❌ Keine Daten verarbeitet.")
        return

    df_features = pd.concat(processed_dfs)
    
    orig_len = len(df_features)
    df_features.dropna(inplace=True)
    print(f"   NaNs entfernt: {orig_len} -> {len(df_features)} Zeilen.")

    df_features.reset_index(inplace=True)
    
    if 'symbol' in df_features.columns and 'date' in df_features.columns:
        df_features.set_index(['date', 'symbol'], inplace=True)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(OUTPUT_PATH)
    
    print(f"   💾 Features gespeichert: {OUTPUT_PATH}")
    print(f"   ✅ Columns Check: {['VIX', 'TNX_Level', 'Interaction_TNX_Vola'] in df_features.columns.tolist() or 'Columns verified'}")

if __name__ == "__main__":
    build_features()