import pandas as pd
import sys
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
current_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(current_dir))

# --- KONFIGURATION ---
FEATURE_DIR = Path("data/features")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PANEL_PATH = FEATURE_DIR / "panel.parquet"
MODEL_PATH = MODEL_DIR / "xgboost_macro_v1.joblib"
# NEU: Wir speichern die Vorhersagen hier
MACRO_SIGNALS_PATH = FEATURE_DIR / "macro_signals.parquet"

def train_macro_brain():
    print("--- Output from MACRO BRAIN TRAINING ---")
    
    # 1. Daten laden
    print("[1] Loading Data...")
    if not PANEL_PATH.exists():
        print(f"❌ Error: File not found {PANEL_PATH}")
        return

    df = pd.read_parquet(PANEL_PATH)
    
    # Fix Spaltennamen
    if 'close' in df.columns:
        df = df.rename(columns={'close': 'Close'})

    # 2. Features
    print("[2] Engineering Features...")
    df['target'] = df['Close'].pct_change(20).shift(-20)
    
    features = [
        'returns_1d', 'returns_5d', 'returns_20d', 
        'volatility_60d', 'VIX', 'TNX_Level', 
        'TNX_Chg_10d', 'SP500_Trend', 'Interaction_TNX_Vola'
    ]
    
    data = df.dropna()
    X = data[features]
    y = data['target']
    
    # 3. Training
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    print(f"[3] Training XGBoost on {len(X_train)} rows...")
    
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        n_jobs=1 # WICHTIG: Single Threaded um Konflikte zu minimieren
    )
    
    model.fit(X_train, y_train)
    
    # Evaluation
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"    Test MSE: {mse:.6f}")
    
    # 4. Speichern des Modells
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

    # --- 5. NEU: PRE-CALCULATION (Der Trick) ---
    print("[5] Pre-calculating Macro Signals for Trinity...")
    # Wir machen Vorhersagen für den GANZEN Datensatz (df)
    # Zuerst droppen wir NaNs in den Features, damit wir predicten können
    inference_data = df[features].dropna()
    
    # Predict
    macro_preds = model.predict(inference_data)
    
    # Wir erstellen einen DataFrame nur mit den Signalen
    # Index (Datum, Symbol) bleibt erhalten, damit wir es später mergen können
    signal_df = pd.DataFrame(macro_preds, index=inference_data.index, columns=['pred_macro'])
    
    signal_df.to_parquet(MACRO_SIGNALS_PATH)
    print(f"✅ Macro Signals pre-calculated and saved to {MACRO_SIGNALS_PATH}")

if __name__ == "__main__":
    train_macro_brain()