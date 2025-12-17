import yfinance as yf
import pandas as pd
import yaml
from pathlib import Path

# --- KONFIGURATION ---
CONFIG_PATH = Path("config/universes.yaml")
DATA_DIR = Path("data")
PANEL_PATH = DATA_DIR / "panel.parquet" 

def load_universe():
    """Lädt die Ticker aus der YAML oder nutzt deine Gewinner-Liste."""
    tickers = []
    
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                conf = yaml.safe_load(f)
                tickers = conf.get("universe", [])
        except Exception as e:
            print(f"⚠️ Fehler beim Lesen der Config: {e}")
    
    if not tickers:
        print("⚠️ Config leer/nicht gefunden. Nutze Fallback-Liste.")
        tickers = [
            "BTC-USD", "ETH-USD", "NVDA", "TSLA", "MSFT", 
            "AAPL", "GOOGL", "AMD", "COIN", "META", "AMZN",
            "SOL-USD", "BNB-USD"
        ]
        
    return list(set(tickers))

def download_data():
    print("--- 📥 TRINITY DATA DOWNLOADER (FINAL FIX) ---")
    
    tickers = load_universe()
    print(f"   Lade Daten für {len(tickers)} Assets...")

    # 1. Assets herunterladen
    print("   1. Downloading Asset Prices...")
    try:
        df = yf.download(tickers, start="2020-01-01", group_by='ticker', auto_adjust=True, progress=True)
    except Exception as e:
        print(f"❌ Critical Download Error: {e}")
        return

    # Struktur reparieren
    df = df.stack(level=0) 
    df.index.names = ['Date', 'symbol']
    df = df.reset_index()
    
    # 2. VIX laden
    print("   2. Downloading VIX (Market Fear Feature)...")
    vix_df = yf.download("^VIX", start="2020-01-01", auto_adjust=True, progress=False)
    
    # Sicherstellen, dass wir keine Multi-Index Spalten haben
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.get_level_values(0)
    
    vix_clean = vix_df[['Close']].copy()
    vix_clean.columns = ['vix_close']
    vix_clean.index.name = 'Date'

    # --- TIMEZONE FIX (DIE KORREKTUR) ---
    print("   3. Fixing Timezones...")
    
    # A) Haupt-Daten (Date ist eine Spalte -> .dt accessor nötig)
    df['Date'] = pd.to_datetime(df['Date'])
    if df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_localize(None)

    # B) VIX Daten (Date ist ein Index -> KEIN .dt accessor!)
    vix_clean.index = pd.to_datetime(vix_clean.index)
    if vix_clean.index.tz is not None:
        vix_clean.index = vix_clean.index.tz_localize(None)

    # 4. Zusammenfügen
    print("   4. Merging VIX into dataset...")
    df_final = pd.merge(df, vix_clean, on='Date', how='left')
    
    # Lücken füllen
    df_final['vix_close'] = df_final['vix_close'].ffill()

    # 5. Aufräumen & Speichern
    df_final.columns = [c.lower() for c in df_final.columns]
    df_final.set_index(['date', 'symbol'], inplace=True)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"   💾 Speichere Panel Data nach: {PANEL_PATH}")
    
    if 'vix_close' in df_final.columns:
        print("      ✅ SUCCESS: VIX Feature ist drin!")
    else:
        print("      ❌ ERROR: VIX fehlt.")

    df_final.to_parquet(PANEL_PATH)
    print("--- Download Complete ---")

if __name__ == "__main__":
    download_data()