import os
import pandas as pd
import numpy as np
from pathlib import Path
from fredapi import Fred

# --- MAC FIX ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# --- KONFIGURATION ---
FEATURE_DIR = Path("data/features")
MACRO_DATA_PATH = FEATURE_DIR / "fred_macro_economic.parquet"

# !!! HIER DEINEN FRED API KEY EINFÜGEN !!!
FRED_API_KEY = "02324d960aeb860d822c70ada2110861"

# Die Indikatoren-Liste (Das Weltklasse-Setup)
INDICATORS = {
    'T10Y2Y': 'Yield_Curve',      # 10Y minus 2Y Treasury (Rezessions-Warnung)
    'M2SL': 'Money_Supply_M2',    # Wie viel Geld ist im System? (Treibstoff für Crypto/Tech)
    'CPIAUCSL': 'Inflation_CPI',  # Inflation (Zins-Angst)
    'UNRATE': 'Unemployment',     # Arbeitslosigkeit (Wirtschafts-Gesundheit)
    'DGS10': '10Y_Treasury',      # Der wichtigste Zins der Welt
    'DTWEXBGS': 'Dollar_Index'    # Trade Weighted Dollar Index (Starker Dollar = Schlecht für Assets)
}

def fetch_fred_data():
    print("\n--- 🏛️  STARTING FEDERAL RESERVE DATA INGESTION ---")
    
    if "DEIN_FRED" in FRED_API_KEY:
        print("❌ CRITICAL: FRED API Key fehlt! Bitte in src/ai_ls_allocation/data/fred_macro.py einfügen.")
        return

    try:
        fred = Fred(api_key=FRED_API_KEY)
        
        all_series = []
        
        for series_id, name in INDICATORS.items():
            print(f"   >>> 📥 Downloading {name} ({series_id})...")
            try:
                # Wir laden die Daten ab 2018
                data = fred.get_series(series_id, observation_start='2018-01-01')
                df_series = pd.DataFrame(data, columns=[name])
                
                # FRED Daten haben oft unterschiedliche Frequenzen (Monatlich vs Täglich)
                # Wir müssen alles auf tägliche Basis bringen
                df_series = df_series.resample('D').ffill() # Forward Fill: Wert bleibt gleich bis neuer kommt
                
                all_series.append(df_series)
            except Exception as e:
                print(f"       ⚠️ Error downloading {name}: {e}")

        if not all_series:
            print("❌ No data downloaded.")
            return

        # Alles zusammenfügen
        print("   >>> 🔄 Merging Economic Indicators...")
        macro_df = pd.concat(all_series, axis=1)
        
        # Datums-Format cleanen
        macro_df.index.name = 'Date'
        macro_df = macro_df.sort_index()
        
        # Feature Engineering: Wir berechnen Veränderungen (Trends)
        # Ist die Inflation gestiegen oder gefallen?
        macro_df['Inflation_Change'] = macro_df['Inflation_CPI'].pct_change(30) # 30-Tage Trend
        macro_df['Liquidity_Trend'] = macro_df['Money_Supply_M2'].pct_change(30) # Druckt die Fed Geld?
        
        # Zinskurve Inversion Flag (Wenn < 0, dann Rezessionsgefahr)
        macro_df['Recession_Signal'] = np.where(macro_df['Yield_Curve'] < 0, 1, 0)
        
        # NaN Werte am Anfang entfernen (durch pct_change entstanden)
        macro_df = macro_df.fillna(method='ffill').fillna(0)

        # Speichern
        macro_df.to_parquet(MACRO_DATA_PATH)
        print(f"   >>> ✅ Saved World Economic Data: {macro_df.shape}")
        print(f"          Path: {MACRO_DATA_PATH}")
        
        # Kleiner Einblick
        print("\n   📊 Latest Economic Readings:")
        print(macro_df.tail(1)[['Yield_Curve', 'Liquidity_Trend', 'Inflation_Change']].T)
        
    except Exception as e:
        print(f"❌ FATAL FRED ERROR: {e}")

if __name__ == "__main__":
    fetch_fred_data()