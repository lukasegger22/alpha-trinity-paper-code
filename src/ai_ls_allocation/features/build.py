import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# --- PFADE ---
BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "raw"
FEATURE_DIR = BASE_DIR / "features"
CONFIG_PATH = Path("config/universes.yaml")

def load_config():
    """Lädt die Konfiguration."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def load_raw_data(symbol):
    """Lädt eine einzelne Parquet-Datei."""
    # Wir müssen aufpassen: Manche Dateien heißen IDX_... (Makro), aber hier
    # laden wir Aktien. Aktien haben meist normale Namen.
    # Falls das Symbol ^ enthält (was yfinance macht), wurde es im download.py ersetzt.
    safe_symbol = symbol.replace("^", "IDX_")
    path = RAW_DIR / f"{safe_symbol}.parquet"
    
    if not path.exists():
        print(f"⚠️ Warning: {path} not found. Skipping.")
        return None
    
    df = pd.read_parquet(path)
    
    # Sicherstellen, dass wir 'Close' haben
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df['Close']
        except KeyError:
            df = df.iloc[:, 0] # Fallback
    elif 'Close' in df.columns:
        df = df['Close']
    
    # In Series umwandeln, falls es ein DataFrame ist
    if isinstance(df, pd.DataFrame):
        df = df.iloc[:, 0]
        
    return df

def calculate_features(price_series):
    """Berechnet technische Indikatoren für eine Aktie."""
    df = pd.DataFrame({'close': price_series})
    
    # 1. Returns (Renditen)
    df['returns_1d'] = df['close'].pct_change(1)
    df['returns_5d'] = df['close'].pct_change(5)
    df['returns_20d'] = df['close'].pct_change(20)
    
    # 2. Volatilität (Rolling Std Dev)
    df['volatility_20d'] = df['returns_1d'].rolling(20).std()
    df['volatility_60d'] = df['returns_1d'].rolling(60).std()
    
    # 3. Simple Moving Averages (Trend)
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    
    # Abstand zum SMA (Trend-Stärke)
    df['dist_sma200'] = (df['close'] / df['sma_200']) - 1
    
    return df

def main():
    print("--- 3. Building Features (Panel) ---")
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    
    cfg = load_config()
    universe = cfg.get("universe", [])
    
    print(f"Processing {len(universe)} symbols...")
    
    all_features = []
    
    for symbol in universe:
        # 1. Laden
        prices = load_raw_data(symbol)
        if prices is None or prices.empty:
            continue
            
        # 2. Features berechnen
        feat_df = calculate_features(prices)
        
        # 3. Symbol und Datum als Spalten hinzufügen (für Panel-Format)
        feat_df['symbol'] = symbol
        feat_df = feat_df.reset_index().rename(columns={'index': 'Date', 'Date': 'Date'})
        
        # Sicherstellen, dass Date datetime ist
        feat_df['Date'] = pd.to_datetime(feat_df['Date'])
        
        all_features.append(feat_df)
        
    if not all_features:
        print("❌ Error: No features built. Check raw data.")
        return

    # 4. Zusammenfügen (Ein riesiges Panel für alle Aktien)
    full_panel = pd.concat(all_features, ignore_index=True)
    
    # NaN entfernen (die ersten 200 Tage fehlen wegen SMA200)
    full_panel = full_panel.dropna()
    
    # 5. Speichern
    output_path = FEATURE_DIR / "panel.parquet"
    full_panel.to_parquet(output_path)
    
    print(f"✅ Panel built with shape {full_panel.shape}")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()