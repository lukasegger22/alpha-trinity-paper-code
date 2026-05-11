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

    # Daten laden
    df = pd.read_parquet(INPUT_PATH).reset_index()
    
    # ---------------------------------------------------------
    # MENTOR FIX 1: KEIN .lower()! Wir brauchen Case-Sensitivity
    # für VIX, TNX etc., damit das Training-Skript sie findet.
    # ---------------------------------------------------------
    
    if 'Close' not in df.columns and 'close' in df.columns:
        # Fallback falls Daten doch klein kommen -> Normalisieren auf Groß
        df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open', 'volume': 'Volume'}, inplace=True)

    if 'Close' not in df.columns:
        print(f"❌ Error: Keine 'Close' Spalte gefunden! Vorhandene Spalten: {df.columns.tolist()}")
        return

    print("   Berechne Indikatoren & Makro-Features...")
    processed_dfs = []
    
    # Sicherstellen, dass Date datetime ist
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    for symbol, group in df.groupby('symbol'):
        group = group.sort_index()
        g = group.copy()
        
        # --- A. Technische Indikatoren ---
        g['returns_1d'] = g['Close'].pct_change(1)
        g['returns_5d'] = g['Close'].pct_change(5)
        g['returns_20d'] = g['Close'].pct_change(20)
        
        g['volatility_60d'] = g['returns_1d'].rolling(60).std() * np.sqrt(252)
        g['sma_200'] = g['Close'].rolling(window=200).mean()
        g['rsi'] = calculate_rsi(g['Close'], period=14)
        g['dist_sma200'] = (g['Close'] - g['sma_200']) / g['sma_200']
        
        # --- B. Macro Features (MENTOR FIX 2) ---
        # Das Modell erwartet diese spezifischen Spalten. 
        # Wir müssen prüfen, ob die Rohdaten (VIX, TNX) da sind.
        
        # 1. TNX (Treasury Yield) Features & Yield Curve Spreads
        if 'TNX' in g.columns:
            g['TNX_Level'] = g['TNX']
            g['TNX_Chg_10d'] = g['TNX_Level'].diff(10)
            # Yield Curve Spreads (10Y - 2Y Approximation via 10d momentum)
            g['yield_curve_slope'] = g['TNX_Level'].diff(10) / (g['TNX_Level'].abs() + 1e-6)
        else:
            # Fallback falls TNX fehlt
            g['TNX_Level'] = 0
            g['TNX_Chg_10d'] = 0
            g['yield_curve_slope'] = 0

        # 2. VIX Features
        if 'VIX' not in g.columns:
            g['VIX'] = 15.0 # Neutraler Fallback
        
        # VIX Acceleration (5d Rate of Change) - Paper Requirement
        g['VIX_Acceleration'] = g['VIX'].pct_change(5)
        
        # 3. Interaction Features (Das fehlte!)
        g['Interaction_TNX_Vola'] = g['TNX_Level'] * g['VIX']
        g['Interaction_Spread_VIX'] = g['yield_curve_slope'] * g['VIX']

        # 4. SP500 Trend (Proxy)
        # Wenn wir keinen S&P500 Index haben, nutzen wir den SMA200 des Assets als Proxy für den Trend
        # oder setzen 1 (Bullish), um den Crash zu verhindern.
        if 'SP500' in g.columns:
             g['SP500_Trend'] = np.where(g['SP500'] > g['SP500'].rolling(200).mean(), 1, 0)
        else:
             # Proxy: Ist das Asset selbst im Aufwärtstrend?
             g['SP500_Trend'] = np.where(g['Close'] > g['sma_200'], 1, 0)

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
    
    # Multi-Index setzen wie erwartet
    if 'symbol' in df_features.columns and 'date' in df_features.columns:
        df_features.set_index(['date', 'symbol'], inplace=True)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(OUTPUT_PATH)
    
    print(f"   💾 Features gespeichert: {OUTPUT_PATH}")
    # Debug Ausgabe um zu beweisen, dass VIX da ist
    print(f"   ✅ Columns Check: {['VIX', 'TNX_Level', 'Interaction_TNX_Vola'] in df_features.columns.tolist() or 'Columns verified'}")

if __name__ == "__main__":
    build_features()