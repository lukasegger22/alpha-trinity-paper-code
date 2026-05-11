import pandas as pd
import sys
import numpy as np
import joblib
import pickle
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

current_dir = Path(__file__).resolve().parent.parent.parent 
sys.path.append(str(current_dir))

# --- KONFIGURATION ---
FEATURE_DIR = Path("data/features")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PANEL_PATH = FEATURE_DIR / "panel.parquet"

# Modelle speichern
MODEL_XGBOOST_1D = MODEL_DIR / "xgboost_1d.joblib"
MODEL_XGBOOST_5D = MODEL_DIR / "xgboost_5d.joblib"
MODEL_XGBOOST_20D = MODEL_DIR / "xgboost_20d.joblib"
MODEL_RIDGE = MODEL_DIR / "ridge_meta_learner.joblib"
SCALER_PATH = MODEL_DIR / "feature_scaler.joblib"

# Vorhersagen speichern
MACRO_SIGNALS_PATH = FEATURE_DIR / "macro_signals.parquet"

def train_hybrid_xgboost_ridge():
    print("--- 🧠 MACRO BRAIN TRAINING: XGBoost + Ridge Stacking (Paper-Exact) ---")
    
    # 1. Daten laden
    print("[1] Loading Data...")
    if not PANEL_PATH.exists():
        print(f"❌ Error: File not found {PANEL_PATH}")
        return

    df = pd.read_parquet(PANEL_PATH).reset_index()
    
    # Fix Spaltennamen
    if 'close' in df.columns:
        df = df.rename(columns={'close': 'Close'})

    # 2. Features Engineering
    print("[2] Engineering Features...")
    
    # WICHTIG: Z-Score Normalization ZUERST
    features = [
        'returns_1d', 'returns_5d', 'returns_20d', 
        'volatility_60d', 'VIX', 'TNX_Level', 
        'TNX_Chg_10d', 'SP500_Trend', 'Interaction_TNX_Vola'
    ]
    
    # Z-Score Normalization
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features].fillna(0))
    joblib.dump(scaler, SCALER_PATH)
    print(f"   ✅ Scaler fitted and saved")
    
    # 3. Multi-Horizon Targets erstellen
    print("[3] Creating Multi-Horizon Targets...")
    data = df_scaled.dropna()
    
    # Target für verschiedene Horizonte (20-Tage-Vorhersagen)
    data['target_1d'] = data['returns_1d'].shift(-1)
    data['target_5d'] = data['returns_5d'].shift(-5)
    data['target_20d'] = data['returns_20d'].shift(-20)
    
    data = data.dropna(subset=['target_1d', 'target_5d', 'target_20d'])
    
    X = data[features].values
    y_1d = data['target_1d'].values
    y_5d = data['target_5d'].values
    y_20d = data['target_20d'].values
    
    # 4. Train/Test Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_1d_train, y_1d_test = y_1d[:split], y_1d[split:]
    y_5d_train, y_5d_test = y_5d[:split], y_5d[split:]
    y_20d_train, y_20d_test = y_20d[:split], y_20d[split:]
    
    # 5. Training XGBoost für 3 Horizonte
    print("[4] Training XGBoost Models (1d, 5d, 20d)...")
    
    xgb_params = {
        'n_estimators': 100,
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': 1
    }
    
    # 1d Horizon
    model_1d = XGBRegressor(**xgb_params)
    model_1d.fit(X_train, y_1d_train)
    pred_1d_test = model_1d.predict(X_test)
    r2_1d = r2_score(y_1d_test, pred_1d_test)
    print(f"   ✅ XGBoost 1d: R² = {r2_1d:.4f}")
    
    # 5d Horizon
    model_5d = XGBRegressor(**xgb_params)
    model_5d.fit(X_train, y_5d_train)
    pred_5d_test = model_5d.predict(X_test)
    r2_5d = r2_score(y_5d_test, pred_5d_test)
    print(f"   ✅ XGBoost 5d: R² = {r2_5d:.4f}")
    
    # 20d Horizon
    model_20d = XGBRegressor(**xgb_params)
    model_20d.fit(X_train, y_20d_train)
    pred_20d_test = model_20d.predict(X_test)
    r2_20d = r2_score(y_20d_test, pred_20d_test)
    print(f"   ✅ XGBoost 20d: R² = {r2_20d:.4f}")
    
    # 6. Ridge Meta-Learner (Stacking)
    # Ridge kombiniert die 3 XGBoost-Vorhersagen mit optimalen Gewichten
    print("[5] Training Ridge Meta-Learner (Stacking)...")
    
    # Stack Training Data: Predictions von 3 XGBoost Models
    X_meta_train = np.column_stack([
        model_1d.predict(X_train),
        model_5d.predict(X_train),
        model_20d.predict(X_train)
    ])
    
    # Ridge lernt, wie man die 3 Vorhersagen kombiniert
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_meta_train, y_20d_train)  # Target ist 20d
    
    # Evaluation
    X_meta_test = np.column_stack([pred_1d_test, pred_5d_test, pred_20d_test])
    ridge_preds = ridge_model.predict(X_meta_test)
    r2_ridge = r2_score(y_20d_test, ridge_preds)
    ridge_coef = ridge_model.coef_
    print(f"   ✅ Ridge Meta: R² = {r2_ridge:.4f}")
    print(f"   ✅ Ridge Weights: 1d={ridge_coef[0]:.3f}, 5d={ridge_coef[1]:.3f}, 20d={ridge_coef[2]:.3f}")
    
    # 7. Speichern der Modelle
    print("[6] Saving Models...")
    joblib.dump(model_1d, MODEL_XGBOOST_1D)
    joblib.dump(model_5d, MODEL_XGBOOST_5D)
    joblib.dump(model_20d, MODEL_XGBOOST_20D)
    joblib.dump(ridge_model, MODEL_RIDGE)
    print(f"   ✅ All models saved")
    
    # 8. PRE-CALCULATION für Trinity Engine
    print("[7] Pre-calculating Multi-Horizon Signals...")
    
    # Wir machen Vorhersagen für den GANZEN Datensatz
    X_all = df_scaled[features].fillna(0).values
    
    # XGBoost Predictions
    pred_all_1d = model_1d.predict(X_all)
    pred_all_5d = model_5d.predict(X_all)
    pred_all_20d = model_20d.predict(X_all)
    
    # Ridge Meta-Prediction
    X_meta_all = np.column_stack([pred_all_1d, pred_all_5d, pred_all_20d])
    pred_all_ridge = ridge_model.predict(X_meta_all)
    
    # Multi-Horizon Voting (Paper Formula)
    # S_t = 0.5 * pred_1d + 0.3 * pred_5d + 0.2 * pred_20d
    voting_signal = 0.5 * pred_all_1d + 0.3 * pred_all_5d + 0.2 * pred_all_20d
    
    # Speichern
    signal_df = pd.DataFrame({
        'pred_1d': pred_all_1d,
        'pred_5d': pred_all_5d,
        'pred_20d': pred_all_20d,
        'pred_ridge': pred_all_ridge,  # Ridge Meta-Learner
        'voting_signal': voting_signal,  # Multi-Horizon Voting
    }, index=df.index)
    
    signal_df.to_parquet(MACRO_SIGNALS_PATH)
    print(f"   ✅ Signals saved to {MACRO_SIGNALS_PATH}")
    print(f"\n🎉 TRAINING COMPLETE - Ready for Trinity Engine!")

if __name__ == "__main__":
    train_hybrid_xgboost_ridge()