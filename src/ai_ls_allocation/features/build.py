import pandas as pd
import pandas_ta as ta
import numpy as np
from pathlib import Path

# --- KONFIGURATION ---
DATA_DIR = Path("data")
INPUT_PATH = DATA_DIR / "panel.parquet"       # Hier liegt deine neue, große Datei
OUTPUT_DIR = DATA_DIR / "features"            # Hier muss das Ergebnis hin
OUTPUT_PATH = OUTPUT_DIR / "panel.parquet"    # Das finale Futter für die KI

def build_features():
    print("--- 3. Building Features (Smart Engine) ---")
    
    # 1. Daten laden (die wir gerade mit download.py erstellt haben)
    if not INPUT_PATH.exists():
        print(f"❌ Critical Error: Input file {INPUT_PATH} not found!")
        print("   Did you run download.py?")
        return

    # Wir laden die Datei. Da wir im Downloader 'date' und 'symbol' als Index gesetzt haben,
    # resetten wir hier kurz, damit wir sauber damit arbeiten können.
    df = pd.read_parquet(INPUT_PATH).reset_index()
    
    print(f"   Daten geladen. Shape: {df.shape}")
    print(f"   Spalten: {list(df.columns)}")
    
    # Spaltennamen normalisieren (alles kleinschreiben zur Sicherheit)
    df.columns = [c.lower() for c in df.columns]
    
    # Sicherstellen, dass 'close' existiert
    if 'close' not in df.columns:
        print("❌ Error: Keine 'close' Spalte gefunden!")
        return

    # Check ob VIX da ist (sollte im Downloader passiert sein)
    if 'vix_close' in df.columns:
        print("   ✅ VIX Feature gefunden. Wird genutzt.")
    else:
        print("   ⚠️ WARNUNG: VIX fehlt! Wurde download.py ausgeführt?")

    # 2. Features berechnen
    # Wir müssen pro Symbol gruppieren, damit der RSI nicht von Apple auf Bitcoin "überschwappt".
    
    print("   Berechne Indikatoren (RSI, SMA, Returns)...")
    processed_dfs = []
    
    # Wir setzen den Index für die Berechnung wieder auf Datum
    df.set_index('date', inplace=True)
    
    for symbol, group in df.groupby('symbol'):
        # Sortieren ist wichtig für Rolling Windows
        group = group.sort_index()
        
        # Kopie erstellen
        g = group.copy()
        
        # --- DEINE ALTEN FEATURES (Portiert) ---
        
        # 1. Returns (Wachstum)
        g['returns_1d'] = g['close'].pct_change(1)
        g['returns_5d'] = g['close'].pct_change(5)
        g['returns_20d'] = g['close'].pct_change(20)
        
        # 2. Volatilität (60 Tage)
        g['volatility_60d'] = g['returns_1d'].rolling(60).std() * np.sqrt(252)
        
        # --- NEUE FEATURES (für die KI) ---
        
        # 3. RSI (Relative Strength Index) - Klassiker
        g['rsi'] = ta.rsi(g['close'], length=14)
        
        # 4. SMA (Gleitende Durchschnitte) - Trend
        g['sma_200'] = ta.sma(g['close'], length=200)
        
        # 5. Abstand zum SMA (Wie weit sind wir weg?)
        # Wenn Preis > SMA200 -> Aufwärtstrend -> Wert positiv
        g['dist_sma200'] = (g['close'] - g['sma_200']) / g['sma_200']
        
        # VIX ist schon als Spalte 'vix_close' da, müssen wir nicht neu berechnen
        
        processed_dfs.append(g)

    # 3. Alles wieder zusammenkleben
    if not processed_dfs:
        print("❌ Keine Daten verarbeitet.")
        return

    df_features = pd.concat(processed_dfs)
    
    # 4. Bereinigen (NaNs entfernen)
    # Durch den SMA_200 fehlen die ersten 200 Tage jeder Aktie. Das ist normal.
    original_len = len(df_features)
    df_features.dropna(inplace=True)
    print(f"   NaNs entfernt (wegen SMA200): {original_len} -> {len(df_features)} Zeilen übrig.")
    
    # Index wieder sauber setzen für Parquet (Date + Symbol)
    # 'symbol' ist aktuell eine Spalte, 'date' ist der Index
    df_features.reset_index(inplace=True)
    df_features.set_index(['date', 'symbol'], inplace=True)

    # 5. Speichern
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(OUTPUT_PATH)
    
    print(f"   💾 Features gespeichert nach: {OUTPUT_PATH}")
    print("--- Feature Engineering Complete ---")

if __name__ == "__main__":
    build_features()