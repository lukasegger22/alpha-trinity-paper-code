import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error

# --- KONFIGURATION ---
DATA_DIR = Path("data")
# Input: Die Datei aus build.py
FEATURES_PATH = DATA_DIR / "features" / "panel.parquet"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xgb_model.json"
# Output: WICHTIG für bt_trinity.py!
MACRO_SIGNALS_PATH = DATA_DIR / "features" / "macro_signals.parquet"

# --- FEATURES DEFINIEREN ---
# Das sind die stabilen Features aus deinem neuen build.py
FEATURES = [
    'rsi', 
    'dist_sma200', 
    'volatility_60d', 
    'vix_close',
    'returns_5d' 
]

def train_macro_brain():
    print("--- Output from MACRO BRAIN TRAINING ---")
    
    # 1. Daten laden
    print("[1] Loading Data...")
    if not FEATURES_PATH.exists():
        print(f"❌ Error: File not found {FEATURES_PATH}")
        return

    df = pd.read_parquet(FEATURES_PATH)
    
    # Spalten klein schreiben zur Sicherheit
    df.columns = [c.lower() for c in df.columns]

    # 2. Target erstellen (Regression: Wir wollen die Rendite vorhersagen)
    print("[2] Engineering Target...")
    # Wir wollen wissen: Wie hoch ist der Return in 20 Tagen?
    df['target'] = df.groupby('symbol')['close'].pct_change(20).shift(-20)
    
    # NaNs in Features oder Target entfernen für das Training
    train_df = df.dropna(subset=FEATURES + ['target'])
    
    if len(train_df) == 0:
        print("❌ CRITICAL: No data left for training!")
        return

    print(f"   Training Data Shape: {train_df.shape}")

    # 3. Training (Regressor)
    # Wir nehmen die letzten 20% als Test-Set (Zeitbasiert)
    dates = train_df.index.get_level_values('date').unique().sort_values()
    split_idx = int(len(dates) * 0.8)
    split_date = dates[split_idx]
    
    print(f"   Splitting data at {split_date}...")
    
    train_data = train_df[train_df.index.get_level_values('date') < split_date]
    test_data = train_df[train_df.index.get_level_values('date') >= split_date]

    X_train = train_data[FEATURES]
    y_train = train_data['target']
    X_test = test_data[FEATURES]
    y_test = test_data['target']

    print(f"[3] Training XGBoost Regressor on {len(X_train)} rows...")
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        n_jobs=1, # Stabilisiert GitHub Actions
        objective='reg:squarederror'
    )
    
    model.fit(X_train, y_train)

    # Evaluation
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"   ✅ Test MSE: {mse:.6f}")

    # 4. Speichern des Modells
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"   💾 Model saved to {MODEL_PATH}")

    # --- 5. PRE-CALCULATION (WICHTIG!) ---
    print("[5] Pre-calculating Macro Signals for Trinity...")
    
    # Wir machen Vorhersagen für ALLE Daten (auch die, wo wir das Target noch nicht kennen)
    # Das ist wichtig für die Live-Vorhersage von "heute"
    inference_data = df[FEATURES].dropna()
    
    if inference_data.empty:
        print("❌ Error: No inference data available.")
        return

    # Die KI sagt voraus: "Wie viel % Gewinn erwarte ich?"
    macro_preds = model.predict(inference_data)
    
    # Speichern als DataFrame
    signal_df = pd.DataFrame(macro_preds, index=inference_data.index, columns=['pred_macro'])
    
    # WICHTIG: Das Skript bt_trinity erwartet die Datei genau hier
    signal_df.to_parquet(MACRO_SIGNALS_PATH)
    print(f"   ✅ Macro Signals pre-calculated and saved to {MACRO_SIGNALS_PATH}")

if __name__ == "__main__":
    train_macro_brain()