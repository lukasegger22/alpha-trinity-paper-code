import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- PFADE ---
BASE_DIR = Path("data")
FEATURE_DIR = BASE_DIR / "features"
MODEL_DIR = Path("models")

def load_data():
    """Lädt Aktien-Panel und Makro-Daten und verschmilzt sie."""
    print("[1] Loading Data...")
    
    # 1. Aktien-Panel laden (das haben wir schon aus V1)
    panel_path = FEATURE_DIR / "panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError("Panel data not found. Run 'features/build.py' first!")
    
    df = pd.read_parquet(panel_path)
    
    # Sicherstellen, dass Date ein echtes Datum ist (manchmal Index)
    if 'Date' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    
    # 2. Makro-Daten laden (die wir gerade gebaut haben)
    macro_path = FEATURE_DIR / "macro_features.parquet"
    if not macro_path.exists():
        raise FileNotFoundError("Macro data not found. Run 'data/process_macro.py' first!")
    
    macro_df = pd.read_parquet(macro_path)
    # Reset index, damit 'Date' eine Spalte wird für den Merge
    macro_df = macro_df.reset_index().rename(columns={'index': 'Date'})
    
    # 3. Merge: Wir kleben an jede Aktie die Makro-Daten des jeweiligen Tages
    # Wir nutzen 'left' merge, wir behalten alle Aktien-Zeilen
    full_df = pd.merge(df, macro_df, on='Date', how='left')
    
    # Vorwärts füllen, falls Makro-Daten an einem Tag fehlen
    full_df = full_df.ffill()
    
    return full_df

def prepare_features(df):
    """Erstellt das Ziel (Target) und wählt Features aus."""
    print("[2] Engineering Features...")
    
    df = df.copy()
    
    # --- TARGET: Rendite von MORGEN (Shift -1) ---
    # Wir wollen vorhersagen, wie sich der Preis zum nächsten Tag ändert
    # Dafür gruppieren wir nach Symbol, damit wir nicht AAPL Daten in MSFT schieben
    df['Target_Return'] = df.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1)
    
    # --- FEATURES ---
    # Wir nutzen Returns, Vola und die neuen MAKRO-Daten
    # (Wir nehmen an, dass 'returns' und 'volatility' schon im Panel sind)
    
    # Wir erstellen Interaktionen: Zins-Änderung * Volatilität der Aktie
    # Idee: Hohe Zinsen tun besonders weh, wenn die Aktie eh schon wackelt
    df['Interaction_TNX_Vola'] = df['TNX_Chg_10d'] * df['volatility_60d']
    
    feature_cols = [
        'returns_1d', 'returns_5d', 'returns_20d', 'volatility_60d',  # Aktien-Daten
        'VIX', 'TNX_Level', 'TNX_Chg_10d', 'SP500_Trend',             # Makro-Daten
        'Interaction_TNX_Vola'                                        # Smart Feature
    ]
    
    # NaN entfernen (die entstehen durch Shift und Diff)
    df_clean = df.dropna(subset=feature_cols + ['Target_Return'])
    
    return df_clean, feature_cols

def train_model():
    # Ordner erstellen
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Daten laden & vorbereiten
    df = load_data()
    df_model, features = prepare_features(df)
    
    print(f"    Data shape: {df_model.shape}")
    print(f"    Features used: {features}")
    
    # 2. Split (Train / Test)
    # WICHTIG: Im Trading dürfen wir nicht zufällig splitten (Shuffle=False)!
    # Wir müssen die VERGANGENHEIT nutzen, um die ZUKUNFT zu testen.
    # Wir nehmen die letzten 20% als Test.
    
    # Wir sortieren nach Zeit
    df_model = df_model.sort_values('Date')
    
    X = df_model[features]
    y = df_model['Target_Return']
    
    # Time-Series Split
    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    print(f"[3] Training XGBoost on {len(X_train)} rows...")
    
    # 3. Modell konfigurieren (Konservative Einstellungen gegen Overfitting)
    model = xgb.XGBRegressor(
        n_estimators=100,       # Anzahl der Bäume
        max_depth=4,            # Nicht zu tief (verhindert Auswendiglernen)
        learning_rate=0.05,     # Langsam lernen
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 4. Evaluieren
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"    Test MSE: {mse:.6f} (kleiner ist besser)")
    
    # WICHTIG: Feature Importance anzeigen
    # Das zeigt uns, ob die Makro-Daten wirklich genutzt werden!
    print("\n--- 📊 Feature Importance (Was dem Bot wichtig ist) ---")
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(importance)
    
    # 5. Speichern
    model_path = MODEL_DIR / "xgboost_macro_v1.joblib"
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved to {model_path}")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Tip: Did you run 'src/ai_ls_allocation/features/build.py' first?")