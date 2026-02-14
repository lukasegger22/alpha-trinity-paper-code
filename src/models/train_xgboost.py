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

def engineer_features(df):
    """
    Erstellt professionelle Features für das ML-Modell.
    Gruppen: Momentum, Volatilität, Trend, Markt-Regime
    """
    print("   ...engineering advanced features")
    df = df.copy()
    
    # Sortieren ist essenziell für Rolling Windows
    if 'symbol' in df.columns:
        df = df.sort_values(['symbol', 'date'])
    else:
        df = df.sort_index()

    # 1. MOMENTUM FEATURES
    for lag in LOOKBACK_WINDOWS:
        df[f'ret_{lag}d'] = df.groupby('symbol')['close'].pct_change(lag)
    
    df['momentum_accel'] = df['ret_5d'] - df['ret_20d']

    # 2. VOLATILITY FEATURES
    for window in [20, 60]:
        df[f'vol_{window}d'] = df.groupby('symbol')['ret_1d'].transform(lambda x: x.rolling(window).std())
    
    # Fillna für Volatility Ratio um Division durch 0 zu verhindern
    df['vol_60d'] = df['vol_60d'].replace(0, np.nan) 
    df['vol_regime'] = df['vol_20d'] / df['vol_60d']

    # 3. TREND FEATURES
    for window in [20, 50, 200]:
        sma = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window).mean())
        df[f'dist_sma{window}'] = (df['close'] / sma) - 1.0
    
    df['trend_strength'] = np.where(df['dist_sma200'] > 0, 1.0, -1.0)

    # 4. MARKET REGIME (VIX Handle)
    if 'vix' in df.columns:
        df['vix_level'] = df['vix'].fillna(20.0)
        df['vix_change_5d'] = df['vix'].diff(5).fillna(0.0)
    else:
        df['vix_level'] = 20.0
        df['vix_change_5d'] = 0.0

    # 5. CROSS-SECTIONAL RANK
    # Gruppieren nach Datum, um Ranks pro Tag zu bekommen
    df['rank_ret_20d'] = df.groupby('date')['ret_20d'].rank(pct=True)

    # WICHTIG: Wir droppen hier noch NICHTS, um die Inference-Daten (heute) nicht zu verlieren!
    # Wir füllen NaNs, die unkritisch sind, mit 0
    df = df.fillna(0)
    
    return df

def train_ensemble_model(X, y):
    """
    Trainiert ein Ensemble aus XGBoost (nicht-linear) und Ridge (linear).
    """
    # Check ob genug Daten da sind
    if len(X) < 10:
        print("      ⚠️ Warning: Not enough data to train model.")
        return None, None, None

    # 1. XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4, # Reduziert gegen Overfitting
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
        objective='reg:squarederror'
    )
    xgb_model.fit(X, y)
    
    # 2. Ridge Regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_scaled, y)
    
    return xgb_model, ridge_model, scaler

def predict_ensemble(xgb_model, ridge_model, scaler, X):
    """Kombiniert die Vorhersagen."""
    if xgb_model is None:
        return np.zeros(len(X))
        
    pred_xgb = xgb_model.predict(X)
    pred_ridge = ridge_model.predict(scaler.transform(X))
    
    # Gewichtung: 70% XGBoost, 30% Ridge
    return (pred_xgb * 0.7) + (pred_ridge * 0.3)

def run_training_pipeline():
    print("--- 🧠 STARTING ADVANCED BRAIN TRAINING ---")
    
    # 1. Load Data
    print("[1] Loading Feature Data...")
    if not FEATURES_PATH.exists():
        print(f"❌ Error: {FEATURES_PATH} not found.")
        return
    
    df = pd.read_parquet(FEATURES_PATH)
    df.columns = [c.lower() for c in df.columns]
    
    # Reset Index falls nötig, damit 'date' eine Spalte ist
    if 'date' not in df.columns:
        df = df.reset_index()
    
    # 2. Feature Engineering
    print("[2] Generating Advanced Features...")
    df_engineered = engineer_features(df)
    
    # Feature List definieren
    exclude_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
    # Alle Spalten die mit 'target_' anfangen auch ausschließen
    feature_cols = [c for c in df_engineered.columns if c not in exclude_cols and not c.startswith('target_')]
    
    print(f"   Using {len(feature_cols)} Features.")

    # 3. Target Creation & Training
    print("[3] Creating Multi-Horizon Targets & Training...")
    targets = {'1d': 1, '5d': 5, '20d': 20}
    models = {} 
    
    for horizon_name, horizon_days in targets.items():
        print(f"\n   👉 Training for Horizon: {horizon_name}")
        
        # Target erstellen (Shifted Returns)
        target_col = f'target_{horizon_name}'
        df_train = df_engineered.copy()
        
        # Wir berechnen den Future Return
        df_train[target_col] = df_train.groupby('symbol')['close'].pct_change(horizon_days).shift(-horizon_days)
        
        # JETZT erst droppen wir NaNs, aber NUR für das Training!
        # Wir brauchen ein sauberes Dataset wo X und y existieren
        df_train_clean = df_train.dropna(subset=[target_col] + feature_cols)
        
        # Safety Check
        if len(df_train_clean) < 100:
            print(f"      ❌ Not enough data for horizon {horizon_name}. Skipping.")
            continue

        X = df_train_clean[feature_cols]
        y = df_train_clean[target_col]
        
        # Walk-Forward Validation
        try:
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                xgb_m, ridge_m, scl = train_ensemble_model(X_train, y_train)
                if xgb_m is not None:
                    preds = predict_ensemble(xgb_m, ridge_m, scl, X_test)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    scores.append(rmse)
            
            if scores:
                print(f"      ✅ Avg RMSE: {np.mean(scores):.5f}")
        except Exception as e:
            print(f"      ⚠️ CV skipped due to data size: {e}")

        # Finales Training auf ALLEN verfügbaren Daten
        final_xgb, final_ridge, final_scaler = train_ensemble_model(X, y)
        models[horizon_name] = (final_xgb, final_ridge, final_scaler)

    # 4. Final Inference (Prediction für HEUTE)
    print("\n[4] Generating Signals for Trinity...")
    
    # Wir nehmen die aktuellsten Daten (die wir für Training evtl. gedroppt haben, weil Target fehlte)
    X_latest = df_engineered[feature_cols].fillna(0)
    
    # Sicherstellen, dass wir Modelle haben
    if not models:
        print("❌ CRITICAL: No models trained. Check data history length.")
        return

    # Predictions
    # Wenn ein Modell fehlt (z.B. 20d wegen zu wenig Daten), füllen wir mit 0
    pred_1d = predict_ensemble(*models.get('1d', (None, None, None)), X_latest)
    pred_5d = predict_ensemble(*models.get('5d', (None, None, None)), X_latest)
    pred_20d = predict_ensemble(*models.get('20d', (None, None, None)), X_latest)
    
    # 5. Signal Combination
    final_signal = (pred_1d * 0.5) + (pred_5d * 0.3) + (pred_20d * 0.2)
    
    # DataFrame bauen (Index muss erhalten bleiben)
    signal_df = pd.DataFrame(index=X_latest.index)
    
    # Tanh Scaling für das Signal (-1 bis 1)
    signal_df['pred_macro'] = np.tanh(final_signal * 10)
    
    # Speichern
    # Wir müssen sicherstellen, dass wir das Datum und Symbol wieder haben für den Merge später
    if 'symbol' in df_engineered.columns:
        signal_df['symbol'] = df_engineered['symbol']
    if 'date' in df_engineered.columns:
        signal_df['date'] = df_engineered['date']
        
    MACRO_SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    signal_df.to_parquet(MACRO_SIGNALS_PATH)
    
    print(f"   ✅ Saved combined signals to {MACRO_SIGNALS_PATH}")
    print(signal_df['pred_macro'].describe())
    
    # Models speichern
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, MODEL_DIR / "ensemble_models.pkl")

if __name__ == "__main__":
    run_training_pipeline()