import pandas as pd
from pathlib import Path

# Pfade
RAW_DIR = Path("data/raw")
FEATURE_DIR = Path("data/features")

def load_raw(symbol):
    """Lädt eine Parquet-Datei, behandelt das IDX_ Prefix."""
    safe_symbol = symbol.replace("^", "IDX_")
    path = RAW_DIR / f"{safe_symbol}.parquet"
    
    if not path.exists():
        print(f"⚠️ Warning: {path} not found. Skipping.")
        return None
        
    df = pd.read_parquet(path)
    
    # Sicherstellen, dass wir nur 'Close' Preise haben und keine MultiIndex Spalten
    if isinstance(df.columns, pd.MultiIndex):
        # Nimm nur 'Close', falls vorhanden, sonst die erste Spalte
        try:
            df = df['Close']
        except KeyError:
            df = df.iloc[:, 0]
    elif 'Close' in df.columns:
        df = df['Close']
        
    return df

def process_macro():
    print("--- 🧠 Building Macro Brain (Indicators) ---")
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Daten laden
    vix = load_raw("^VIX")
    tnx = load_raw("^TNX")
    gspc = load_raw("^GSPC")

    if vix is None or tnx is None or gspc is None:
        print("❌ Error: Missing macro raw data. Run download.py first!")
        return

    # 2. Features berechnen
    
    # --- A. Market Regime (S&P 500 Trend) ---
    # Wir berechnen den 200-Tage-Durchschnitt (SMA 200)
    # Wenn Preis > SMA 200 -> Bullenmarkt (1)
    # Wenn Preis < SMA 200 -> Bärenmarkt (0)
    gspc_ma200 = gspc.rolling(window=200).mean()
    market_trend = (gspc > gspc_ma200).astype(int)
    
    # --- B. Zins-Schock (TNX) ---
    # Wir schauen uns die Veränderung der Zinsen über 10 Tage an.
    # Steigende Zinsen sind oft Gift für Tech-Aktien.
    tnx_change_10d = tnx.diff(10)

    # 3. Alles in einen DataFrame packen
    macro_df = pd.DataFrame({
        'VIX': vix,                 # Absolute Angst
        'TNX_Level': tnx,           # Absolute Zinsen
        'TNX_Chg_10d': tnx_change_10d, # Zins-Dynamik
        'SP500_Trend': market_trend # 1=Bull, 0=Bear
    })

    # Vorwärts füllen (falls Feiertage fehlen) und NaN droppen
    macro_df = macro_df.ffill().dropna()

    # 4. Speichern
    output_path = FEATURE_DIR / "macro_features.parquet"
    macro_df.to_parquet(output_path)
    
    print(f"✅ Macro features calculated and saved to {output_path}")
    print(macro_df.tail())

if __name__ == "__main__":
    process_macro()