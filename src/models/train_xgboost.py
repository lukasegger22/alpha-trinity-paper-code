import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import joblib
from pathlib import Path
import warnings

# Unterdrücke Warnungen für saubereren Output
warnings.filterwarnings('ignore')

# --- KONFIGURATION ---
DATA_DIR = Path("data")
FEATURES_PATH = DATA_DIR / "features" / "panel.parquet"
MODEL_DIR = Path("models")
MACRO_SIGNALS_PATH = DATA_DIR / "features" / "macro_signals.parquet"

# Feature-Gruppen Konfiguration
LOOKBACK_WINDOWS = [1, 5, 20, 60]

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def engineer_features(df):
    """
    Erstellt professionelle Features für das ML-Modell.
    Gruppen: Momentum, Volatilität, Trend, Markt-Regime
    """
    print("   ...engineering advanced features")
    df = df.copy()
    
    # Sicherstellen, dass nach Symbol und Datum sortiert ist
    df = df.sort_values(['symbol', 'date'])
    
    # 1. MOMENTUM FEATURES (Die Basis)
    for lag in LOOKBACK_WINDOWS:
        # Returns über verschiedene Horizonte
        df[f'ret_{lag}d'] = df.groupby('symbol')['close'].pct_change(lag)
    
    # Acceleration: Beschleunigt sich der Trend? (5d vs 20d)
    df['momentum_accel'] = df['ret_5d'] - df['ret_20d']

    # 2. VOLATILITY FEATURES (Risk)
    for window in [20, 60]:
        # Rollierende Standardabweichung
        df[f'vol_{window}d'] = df.groupby('symbol')['ret_1d'].transform(lambda x: x.rolling(window).std())
    
    # Volatility Regime: Ist die aktuelle Vola höher als der Durchschnitt?
    df['vol_regime'] = df['vol_20d'] / df['vol_60d']

    # 3. TREND FEATURES (Direction)
    # Distanz zu SMAs
    for window in [20, 50, 200]:
        sma = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window).mean())
        df[f'dist_sma{window}'] = (df['close'] / sma) - 1.0
    
    # Trend Strength: Sind wir über SMA200?
    df['trend_strength'] = np.where(df['dist_sma200'] > 0, 1.0, -1.0)

    # 4. MARKET REGIME (Context)
    # VIX (Angst-Index) - falls vorhanden, sonst 0
    if 'vix' in df.columns:
        df['vix_level'] = df['vix']
        # VIX Change (steigende Angst?)
        df['vix_change_5d'] = df['vix'].diff(5)
    else:
        df['vix_level'] = 20.0
        df['vix_change_5d'] = 0.0

    # 5. CROSS-SECTIONAL RANK (Relativer Vergleich)
    # Wie gut ist der Stock im Vergleich zu anderen am selben Tag?
    # (Wir nehmen ret_20d als Proxy für relative Stärke)
    df['rank_ret_20d'] = df.groupby('date')['ret_20d'].rank(pct=True)

    # Clean up NaNs (entstanden durch Rolling Windows)
    # Wir droppen am Anfang, damit das Training sauber ist
    df = df.dropna()
    
    return df

def train_ensemble_model(X, y):
    """
    Trainiert ein Ensemble aus XGBoost (nicht-linear) und Ridge (linear).
    """
    # 1. XGBoost (Der "Scharfschütze" für komplexe Muster)
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1, # Stabil für GitHub Actions
        objective='reg:squarederror'
    )
    xgb_model.fit(X, y)
    
    # 2. Ridge Regression (Der "Anker" für Stabilität)
    # Ridge braucht skalierte Daten
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_scaled, y)
    
    return xgb_model, ridge_model, scaler

def predict_ensemble(xgb_model, ridge_model, scaler, X):
    """Kombiniert die Vorhersagen."""
    pred_xgb = xgb_model.predict(X)
    pred_ridge = ridge_model.predict(scaler.transform(X))
    
    # Gewichtung: 70% XGBoost, 30% Ridge (Linearer Anker)
    return (pred_xgb * 0.7) + (pred_ridge * 0.3)

def run_training_pipeline():
    print("--- 🧠 STARTING ADVANCED BRAIN TRAINING ---")
    
    # 1. Load Data
    print("[1] Loading Feature Data...")
    if not FEATURES_PATH.exists():
        print(f"❌ Error: {FEATURES_PATH} not found.")
        return
    
    df = pd.read_parquet(FEATURES_PATH)
    # Lowercase columns
    df.columns = [c.lower() for c in df.columns]
    
    # 2. Feature Engineering
    print("[2] Generating Advanced Features...")
    df_engineered = engineer_features(df)
    
    # Feature List definieren (alle numerischen Spalten außer Targets/Meta)
    exclude_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'target_1d', 'target_5d', 'target_20d']
    feature_cols = [c for c in df_engineered.columns if c not in exclude_cols]
    print(f"   Using {len(feature_cols)} Features: {feature_cols}")

    # 3. Target Creation (Multi-Horizon)
    print("[3] Creating Multi-Horizon Targets...")
    # Wir wollen 1 Tag, 5 Tage und 20 Tage vorhersagen
    targets = {
        '1d': 1,
        '5d': 5,
        '20d': 20
    }
    
    models = {} # Speichert die trainierten Modelle pro Horizont
    
    for horizon_name, horizon_days in targets.items():
        print(f"\n   👉 Training for Horizon: {horizon_name} (Shift -{horizon_days})")
        
        # Target erstellen
        target_col = f'target_{horizon_name}'
        df_model = df_engineered.copy()
        df_model[target_col] = df_model.groupby('symbol')['close'].pct_change(horizon_days).shift(-horizon_days)
        
        # NaNs im Target entfernen
        df_model = df_model.dropna(subset=[target_col])
        
        X = df_model[feature_cols]
        y = df_model[target_col]
        
        # Walk-Forward Validation (TimeSeriesSplit)
        tscv = TimeSeriesSplit(n_splits=3)
        fold = 1
        scores = []
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Train Ensemble
            xgb_m, ridge_m, scl = train_ensemble_model(X_train, y_train)
            
            # Evaluate
            preds = predict_ensemble(xgb_m, ridge_m, scl, X_test)
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            print(f"      Fold {fold}: RMSE = {rmse:.5f}")
            scores.append(rmse)
            fold += 1
            
        print(f"      ✅ Average RMSE ({horizon_name}): {np.mean(scores):.5f}")
        
        # Finales Training auf ALLEN Daten für diesen Horizont
        print(f"      Training Final Model ({horizon_name})...")
        final_xgb, final_ridge, final_scaler = train_ensemble_model(X, y)
        models[horizon_name] = (final_xgb, final_ridge, final_scaler)

    # 4. Final Inference (Prediction für Trinity)
    print("\n[4] Generating Signals for Trinity...")
    
    # Wir nutzen die engineered features von HEUTE (ohne Shift)
    X_latest = df_engineered[feature_cols]
    
    # Vorhersagen für alle Horizonte
    pred_1d = predict_ensemble(*models['1d'], X_latest)
    pred_5d = predict_ensemble(*models['5d'], X_latest)
    pred_20d = predict_ensemble(*models['20d'], X_latest)
    
    # 5. Signal Combination (Weighted Average)
    # Hier passiert die Magie: Wir mischen die Zeithorizonte
    # 50% kurzfristig (Entry), 30% Swing, 20% Trend
    final_signal = (pred_1d * 0.5) + (pred_5d * 0.3) + (pred_20d * 0.2)
    
    # Ergebnis speichern
    signal_df = pd.DataFrame(index=X_latest.index)
    signal_df['pred_macro'] = final_signal
    
    # Skalierung des Signals für Trinity (damit es z.B. zwischen -1 und 1 liegt)
    # Wir nutzen tanh für Soft-Clipping
    signal_df['pred_macro'] = np.tanh(signal_df['pred_macro'] * 10) # Faktor 10 verstärkt kleine Returns
    
    # Speichern
    MACRO_SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    signal_df.to_parquet(MACRO_SIGNALS_PATH)
    
    print(f"   ✅ Saved combined signals to {MACRO_SIGNALS_PATH}")
    print(f"   Shape: {signal_df.shape}")
    print("   Signal Stats:")
    print(signal_df['pred_macro'].describe())

    # Optional: Modelle speichern
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Joblib könnte hier genutzt werden, um das Dictionary 'models' zu dumpen
    joblib.dump(models, MODEL_DIR / "ensemble_models.pkl")
    print(f"   💾 Models saved to {MODEL_DIR}")

if __name__ == "__main__":
    run_training_pipeline()