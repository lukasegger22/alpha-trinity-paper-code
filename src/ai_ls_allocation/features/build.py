import pandas as pd
import numpy as np
from pathlib import Path

# --- KONFIGURATION ---
RAW_DIR = Path("data/raw")
FEATURE_DIR = Path("data/features")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

def find_file(keyword):
    """
    Sucht eine Datei, die das Keyword (z.B. 'VIX') im Namen hat.
    Das löst das Problem mit ^VIX vs VIX ein für alle Mal.
    """
    files = list(RAW_DIR.glob(f"*{keyword}*.parquet"))
    if not files:
        raise FileNotFoundError(f"❌ Critical: No file found containing '{keyword}' in {RAW_DIR}")
    
    # Nimm die erste, die passt (z.B. ^VIX.parquet)
    path = files[0]
    print(f"   -> Found {keyword} data: {path.name}")
    
    df = pd.read_parquet(path)
    col = 'Close' if 'Close' in df.columns else 'Adj Close'
    return df[col]

def build_features():
    print("--- 3. Building Features (Panel) ---")
    
    # 1. Macro Daten laden (Intelligente Suche)
    try:
        vix = find_file("VIX").rename("VIX")
        tnx = find_file("TNX").rename("TNX")
    except FileNotFoundError as e:
        print(e)
        return

    # 2. Equity Universe laden
    universe_files = list(RAW_DIR.glob("*.parquet"))
    # Filtere Macro-Files raus
    macro_keywords = ["VIX", "TNX", "GSPC"]
    equity_files = [f for f in universe_files if not any(k in f.name for k in macro_keywords)]
    
    print(f"Processing {len(equity_files)} symbols...")
    
    all_dfs = []
    
    for file_path in equity_files:
        symbol = file_path.stem.replace("^", "") # Bereinigen
        
        try:
            df = pd.read_parquet(file_path)
            
            # Preis-Spalte finden
            if 'Close' in df.columns:
                price_col = 'Close'
            elif 'Adj Close' in df.columns:
                price_col = 'Adj Close'
            elif 'close' in df.columns:
                price_col = 'close'
            else:
                continue # Überspringen wenn kein Preis
            
            # WICHTIG: 'Close' explizit setzen für Execution System
            df['Close'] = df[price_col]
            df = df[['Close']].copy()
            df['symbol'] = symbol
            
            # --- FEATURE ENGINEERING ---
            df['returns_1d'] = df['Close'].pct_change(1)
            df['returns_5d'] = df['Close'].pct_change(5)
            df['returns_20d'] = df['Close'].pct_change(20)
            df['volatility_60d'] = df['returns_1d'].rolling(60).std() * np.sqrt(252)
            
            # Macro Join
            df = df.join(vix).join(tnx)
            df['VIX'] = df['VIX'].ffill()
            df['TNX_Level'] = df['TNX'].ffill()
            
            df['TNX_Chg_10d'] = df['TNX_Level'].diff(10)
            df['Interaction_TNX_Vola'] = df['TNX_Level'] * df['volatility_60d']
            df['SP500_Trend'] = np.where(df['Close'] > df['Close'].rolling(200).mean(), 1, 0)

            df = df.dropna()
            
            cols_to_keep = [
                'symbol', 'Close', 
                'returns_1d', 'returns_5d', 'returns_20d', 
                'volatility_60d', 
                'VIX', 'TNX_Level', 'TNX_Chg_10d', 'Interaction_TNX_Vola', 'SP500_Trend'
            ]
            
            existing_cols = [c for c in cols_to_keep if c in df.columns]
            all_dfs.append(df[existing_cols])
            
        except Exception as e:
            print(f"⚠️ Error processing {symbol}: {e}")
            continue

    if not all_dfs:
        print("❌ No data processed!")
        return

    full_panel = pd.concat(all_dfs)
    output_path = FEATURE_DIR / "panel.parquet"
    full_panel.to_parquet(output_path)
    print(f"✅ Panel built with shape {full_panel.shape}")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    build_features()