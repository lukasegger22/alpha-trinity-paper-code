import os
import yaml
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime

# Wir definieren die Pfade relativ zum Projekt-Root
# (Geht davon aus, dass wir das Skript vom Root aus starten)
BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "raw"
CONFIG_PATH = Path("config/universes.yaml")

# --- NEU: Liste der Makro-Indikatoren ---
# ^VIX = Volatilitäts-Index (Angst-Barometer)
# ^TNX = 10 Year Treasury Yield (US-Zinsen)
# ^GSPC = S&P 500 Index (Gesamtmarkt-Referenz)
MACRO_SYMBOLS = ["^VIX", "^TNX", "^GSPC"]

def load_universe():
    """Lädt die Liste der Aktien aus der YAML-Config."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found at {CONFIG_PATH}")
    
    with open(CONFIG_PATH, "r") as f:
        conf = yaml.safe_load(f)
    
    # Wir nehmen an, dass im YAML eine Liste unter 'universe' steht
    return conf.get("universe", [])

def download_data(symbol, start_date="2010-01-01"):
    """Lädt Daten für ein Symbol und speichert sie als Parquet."""
    print(f"[download] {symbol} {start_date}->today")
    
    # Daten laden
    df = yf.download(symbol, start=start_date, progress=False)
    
    if df.empty:
        print(f"⚠️ Warning: No data for {symbol}")
        return

    # MultiIndex Problematik bei neuem yfinance beheben
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Speichern
    # Wir ersetzen das ^ Zeichen im Dateinamen, weil das Probleme machen kann
    safe_symbol = symbol.replace("^", "IDX_") 
    file_path = RAW_DIR / f"{safe_symbol}.parquet"
    
    df.to_parquet(file_path)

def main():
    # 1. Ordner erstellen, falls nicht existent
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Aktien-Universum laden
    universe = load_universe()
    print(f"Loading Equity Universe: {len(universe)} symbols")
    
    # 3. Aktien herunterladen
    for symbol in universe:
        try:
            download_data(symbol)
        except Exception as e:
            print(f"❌ Error downloading {symbol}: {e}")

    print("-" * 30)
    print(f"Loading Macro Indicators: {MACRO_SYMBOLS}")
    
    # 4. NEU: Makro-Daten herunterladen
    for symbol in MACRO_SYMBOLS:
        try:
            download_data(symbol)
        except Exception as e:
            print(f"❌ Error downloading MACRO {symbol}: {e}")

    print("[done] All data saved to data/raw/")

if __name__ == "__main__":
    main()