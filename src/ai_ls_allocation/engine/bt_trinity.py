# --- DEBUG PRINT GANZ OBEN ---
print("--- 🟢 Script bt_trinity.py is initializing... ---", flush=True)

import os
import sys
import pandas as pd
import numpy as np
import random
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

# Pfad-Hack, damit Imports funktionieren
current_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(current_dir))
try:
    from ai_ls_allocation.engine.optimizer import MarkowitzOptimizer
except ImportError:
    # Fallback falls Optimizer nicht gefunden wird (einfacher Equal-Weight Dummy für Notfälle)
    print("⚠️ Warning: MarkowitzOptimizer not found. Using simple fallback logic if needed.")
    class MarkowitzOptimizer:
        def __init__(self, **kwargs): pass
        def optimize(self, signals, returns): return signals / signals.sum()

# --- ENV VARS ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

# --- KONFIGURATION ---
FEATURE_DIR = Path("data/features")
MACRO_SIGNALS_PATH = FEATURE_DIR / "macro_signals.parquet"
SENTIMENT_PATH = FEATURE_DIR / "sentiment_signals.parquet"
FRED_PATH = FEATURE_DIR / "fred_macro_economic.parquet"
FUNDAMENTALS_PATH = FEATURE_DIR / "fundamental_quality.parquet"
PANEL_PATH = FEATURE_DIR / "panel.parquet"
HISTORY_PATH = FEATURE_DIR / "trinity_history.parquet"
SIGNALS_PATH = FEATURE_DIR / "trinity_signals.parquet"
LIVE_SIGNALS_PATH = FEATURE_DIR / "trinity_signals.csv"

# --- REALITY SETTINGS ---
TRAIN_END_DATE = "2024-01-01" 
COST_OF_CARRY_RATE = 0.06     
REBALANCE_SPEED = 0.20   
MIN_POSITION_SIZE = 0.03 
ENSEMBLE_SIZE = 10 

# --- GLOBAL SEEDING ---
np.random.seed(42)
random.seed(42)

# --- INDICATORS ---

def calculate_efficiency_ratio(series, window=20):
    direction = series.diff(window).abs()
    volatility = series.diff().abs().rolling(window).sum().replace(0, 0.001)
    return direction / volatility

def calculate_obv(df):
    vol = df['Volume'] if 'Volume' in df.columns else pd.Series(1, index=df.index)
    change = df['returns_1d']
    direction = np.where(change > 0, 1, -1)
    direction = np.where(change == 0, 0, direction)
    return (vol * direction).cumsum()

def calculate_mfi(df, window=14):
    typical_price = df['proxy_price'] 
    money_flow = typical_price * df['Volume']
    delta = typical_price.diff()
    pos_flow = pd.Series(np.where(delta > 0, money_flow, 0), index=df.index)
    neg_flow = pd.Series(np.where(delta < 0, money_flow, 0), index=df.index)
    rolling_pos = pos_flow.rolling(window).sum()
    rolling_neg = neg_flow.rolling(window).sum().replace(0, 0.001)
    mfi_ratio = rolling_pos / rolling_neg
    return 100 - (100 / (1 + mfi_ratio))

def calculate_bb_position(series, window=20):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + (2 * std)
    lower = ma - (2 * std)
    bandwidth = (upper - lower).replace(0, 0.001)
    pct_b = (series - lower) / bandwidth
    return pct_b

def load_data():
    if not PANEL_PATH.exists(): raise FileNotFoundError("Panel data not found.")
    df = pd.read_parquet(PANEL_PATH)
    return df

def run_trinity_engine():
    print(f"\n--- 🚀 STARTING TRINITY ENGINE (Restored Champion: Efficiency + Hybrid Ensemble) ---", flush=True)
    print(f"   📅 Training Limit: {TRAIN_END_DATE}")

    # 1. Daten laden & Index bereinigen (MENTOR FIX)
    full_df = load_data()
    full_df = full_df.reset_index()
    
    # Intelligentes Umbenennen statt blindes Raten
    if 'date' in full_df.columns:
        full_df.rename(columns={'date': 'Date'}, inplace=True)
    if 'index' in full_df.columns and 'Date' not in full_df.columns:
        full_df.rename(columns={'index': 'Date'}, inplace=True)
        
    if 'Date' not in full_df.columns:
        raise ValueError(f"❌ CRITICAL: Could not find 'Date' column. Available: {full_df.columns.tolist()}")
    
    # Sicherstellen dass Symbol existiert
    if 'Symbol' in full_df.columns:
        full_df.rename(columns={'Symbol': 'symbol'}, inplace=True)
        
    full_df['Date'] = pd.to_datetime(full_df['Date'])
    
    if 'Volume' not in full_df.columns:
        print("⚠️ Warning: No Volume data found! Using dummy volume.")
        full_df['Volume'] = 1.0

    # Duplikate entfernen
    full_df = full_df.drop_duplicates(subset=['Date', 'symbol'], keep='last')
    full_df = full_df.sort_values(by=['symbol', 'Date'])
    
    start_history = pd.Timestamp(TRAIN_END_DATE) - pd.DateOffset(years=10) 
    full_df = full_df[full_df['Date'] >= start_history].copy()

    # --- FEATURE ENGINEERING ---
    print("   >>> 🛠️  Engineering Features (Volume, Traps, Macro)...")
    
    full_df['proxy_price'] = (1 + full_df['returns_1d'])
    full_df['proxy_price'] = full_df.groupby('symbol')['proxy_price'].cumprod()
    
    # BACK TO EFFICIENCY
    full_df['efficiency'] = full_df.groupby('symbol')['proxy_price'].transform(lambda x: calculate_efficiency_ratio(x, window=20)).fillna(0.5)
    
    # OBV mit Warnungs-Unterdrückung
    full_df['obv'] = full_df.groupby('symbol', group_keys=False).apply(calculate_obv).reset_index(level=0, drop=True)
    full_df['obv_trend'] = full_df.groupby('symbol')['obv'].pct_change(20).fillna(0)
    
    def rolling_vwap(x_vol, x_price, w=20):
        pv = x_price * x_vol
        return pv.rolling(w).sum() / x_vol.rolling(w).sum()
    
    full_df['vwap_20'] = full_df.groupby('symbol', group_keys=False).apply(lambda x: rolling_vwap(x['Volume'], x['proxy_price']))
    full_df['dist_vwap'] = (full_df['proxy_price'] / full_df['vwap_20']) - 1.0
    full_df['dist_vwap'] = full_df['dist_vwap'].fillna(0)

    full_df['mfi'] = full_df.groupby('symbol', group_keys=False).apply(lambda x: calculate_mfi(x))
    full_df['mfi'] = full_df['mfi'].fillna(50) / 100.0

    full_df['bb_pos'] = full_df.groupby('symbol')['proxy_price'].transform(lambda x: calculate_bb_position(x)).fillna(0.5)

    # --- MERGING (FIXED V2: NO SYMBOL CONFLICT) ---
    print("   >>> 🔗 Merging Data Sources...")
    
    # 1. MACRO SIGNALS
    if MACRO_SIGNALS_PATH.exists():
        try:
            macro = pd.read_parquet(MACRO_SIGNALS_PATH)
            
            # Reset Index um sicherzugehen
            macro = macro.reset_index()
            
            # Datum finden & standardisieren
            date_col = None
            for c in macro.columns:
                if c.lower() in ['date', 'index', 'time']:
                    date_col = c
                    break
            
            if date_col:
                macro.rename(columns={date_col: 'Date'}, inplace=True)
                macro['Date'] = pd.to_datetime(macro['Date'])
                
                # WICHTIG: Wir behalten NUR Date und pred_macro
                # Das verhindert, dass eine 'symbol'-Spalte im Macro-File
                # unsere Haupt-Daten zerschießt (KeyError Fix).
                if 'pred_macro' in macro.columns:
                    macro = macro[['Date', 'pred_macro']].copy()
                    
                    # Duplikate entfernen und mergen
                    macro = macro.drop_duplicates(subset=['Date'], keep='last')
                    full_df = pd.merge(full_df, macro, on='Date', how='left')
                    full_df['pred_macro'] = full_df['pred_macro'].ffill().fillna(0)
                    print("       ✅ Macro Signals merged successfully. (Safety Net ACTIVE)")
                else:
                    print("       ⚠️ Warning: Macro file has no 'pred_macro' column.")
                    full_df['pred_macro'] = 0
            else:
                print("       ⚠️ Warning: No Date column in Macro file.")
                full_df['pred_macro'] = 0
                
        except Exception as e:
            print(f"       ⚠️ Macro Merge Failed: {e}")
            full_df['pred_macro'] = 0
    else:
        full_df['pred_macro'] = 0

    if FRED_PATH.exists():
        fred = pd.read_parquet(FRED_PATH).reset_index()
        if 'index' in fred.columns: fred = fred.rename(columns={'index': 'Date'})
        full_df = pd.merge(full_df, fred, on='Date', how='left')
        for c in ['Yield_Curve', 'Money_Supply_M2', 'Inflation_Change', 'Recession_Signal']:
            if c in full_df.columns: full_df[c] = full_df[c].ffill().fillna(0)

    if SENTIMENT_PATH.exists():
        sent = pd.read_parquet(SENTIMENT_PATH)
        sent['Date'] = pd.to_datetime(sent['Date']).dt.normalize()
        full_df['Date_Norm'] = full_df['Date'].dt.normalize()
        sent_agg = sent.groupby(['Date', 'symbol'])['sentiment_score'].mean().reset_index()
        full_df = pd.merge(full_df, sent_agg, left_on=['Date_Norm', 'symbol'], right_on=['Date', 'symbol'], how='left', suffixes=('', '_y'))
        if 'sentiment_score_y' in full_df.columns: full_df = full_df.rename(columns={'sentiment_score_y': 'sentiment_score'})
        full_df['sentiment_score'] = full_df['sentiment_score'].fillna(0.0)
        full_df = full_df.drop(columns=['Date_Norm', 'Date_y'], errors='ignore')
    else: full_df['sentiment_score'] = 0.0

    if FUNDAMENTALS_PATH.exists():
        fund = pd.read_parquet(FUNDAMENTALS_PATH)[['symbol', 'Quality_Score']].drop_duplicates()
        full_df = pd.merge(full_df, fund, on='symbol', how='left')
        full_df['Quality_Score'] = full_df['Quality_Score'].fillna(0.5)
    else: full_df['Quality_Score'] = 0.7

    # --- TRAINING SPLIT ---
    print(f"   >>> ✂️  Splitting Data at {TRAIN_END_DATE}...")
    
    features = [
        'returns_1d', 'returns_5d', 'volatility_60d', 'VIX', 
        'obv_trend', 'dist_vwap', 'mfi', 'bb_pos'
    ]
    
    full_df = full_df.drop_duplicates(subset=['Date', 'symbol'], keep='last')
    full_df = full_df.replace([np.inf, -np.inf], np.nan)
    
    train_mask = full_df['Date'] < TRAIN_END_DATE
    train_df = full_df[train_mask].dropna(subset=features + ['returns_20d'])
    
    if len(train_df) > 0:
        X_train = train_df[features].values
        y_train = train_df['returns_20d'].shift(-20).fillna(0).values 
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_all = full_df[features].fillna(0).values
        X_all_scaled = scaler.transform(X_all)
        
        # --- HYBRID ENSEMBLE ---
        print(f"   >>> 🧠 Training Hybrid Ensemble (5x NN + 5x RF)...")
        ensemble_predictions = np.zeros((len(full_df), ENSEMBLE_SIZE))
        split_point = ENSEMBLE_SIZE // 2 
        
        for i in range(ENSEMBLE_SIZE):
            seed = 42 + i
            if i < split_point:
                # NN
                model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=250, random_state=seed, early_stopping=True)
            else:
                # RF
                model = RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_leaf=20, random_state=seed, n_jobs=-1)
            
            model.fit(X_train_scaled, y_train)
            ensemble_predictions[:, i] = model.predict(X_all_scaled)
            
        full_df['pred_neural'] = np.mean(ensemble_predictions, axis=1)
        print("       ✅ Hybrid Consensus Calculated.")

    else:
        print("❌ Not enough training data!")
        return

    # --- SIGNAL GENERATION ---
    print("   >>> ⚖️  Calculating Signals...")
    full_df['VIX'] = full_df['VIX'].fillna(20.0)
    sentiment_boost = 1.0 + (full_df['sentiment_score'] * 0.2)
    quality_factor = 0.6 + (full_df['Quality_Score'] * 0.6)
    
    is_calm = full_df['VIX'] <= 20.0
    full_df['raw_signal'] = np.where(is_calm,
        (full_df['pred_neural'] * 0.7) + (full_df['pred_macro'] * 0.3), 
        (full_df['pred_neural'] * 0.2) + (full_df['pred_macro'] * 0.8)
    )
    
    full_df['raw_signal'] = full_df['raw_signal'] * sentiment_boost * quality_factor
    
    # FILTER
    full_df['regime_filter'] = np.where(full_df['efficiency'] < 0.15, 0.0, 1.0)
    full_df['raw_signal'] = full_df['raw_signal'] * full_df['regime_filter']
    
    scaled_signal = full_df['raw_signal'] * 50.0 
    full_df['ensemble_signal'] = full_df.groupby('symbol')['raw_signal'].transform(
        lambda x: (scaled_signal.loc[x.index]).ewm(span=10).mean()
    )

    # --- BACKTESTING ---
    print(f"   >>> 📐 Backtesting Simulation starting 2024...")
    
    full_df = full_df.drop_duplicates(subset=['Date', 'symbol'], keep='last')
    pivot_signals = full_df.pivot(index='Date', columns='symbol', values='ensemble_signal').fillna(0)
    pivot_returns = full_df.pivot(index='Date', columns='symbol', values='returns_1d').fillna(0)
    daily_vix = full_df.groupby('Date')['VIX'].mean()
    
    if 'Yield_Curve' in full_df.columns:
        daily_yield_curve = full_df.groupby('Date')['Yield_Curve'].mean()
    else: daily_yield_curve = pd.Series(1.0, index=pivot_signals.index)

    test_dates = pivot_signals.index[pivot_signals.index >= TRAIN_END_DATE]
    
    optimizer = MarkowitzOptimizer(window_size=60, risk_aversion=0.5, target_vol=0.15)
    weights_history = []
    current_weights = pd.Series(0.0, index=pivot_signals.columns)
    
    SHORT_CONFIDENCE_THRESHOLD = -0.05

    for current_date in test_dates:
        idx = pivot_signals.index.get_loc(current_date)
        
        # 1. Das rohe Signal holen (Kopie erstellen!)
        current_sig = pivot_signals.iloc[idx].copy()
        past_ret = pivot_returns.iloc[:idx]
        
        # --- 🛡️ MENTOR LOGIC: THE SNIPER FILTER ---
        # Wir manipulieren das Signal, BEVOR es in den Optimizer geht.
        
        # A) Finde alle "schlechten" Signale, die aber nicht "schlecht genug" sind.
        #    Das ist die "Dead Zone" (z.B. zwischen -0.08 und 0.0).
        #    Hier würden wir normalerweise shorten, aber wir gehen lieber in Cash.
        weak_short_mask = (current_sig < 0.0) & (current_sig > SHORT_CONFIDENCE_THRESHOLD)
        
        # B) Setze diese wackeligen Kandidaten auf 0.0
        #    Der Optimizer denkt jetzt: "Aha, keine Meinung dazu -> ich kaufe/shorte es nicht."
        current_sig[weak_short_mask] = 0.0
        
        # C) Jetzt erst Optimieren (Mit erlaubten Shorts im Optimizer!)
        try:
            target_weights = optimizer.optimize(current_sig, past_ret)
        except:
            target_weights = pd.Series(0, index=current_sig.index)

        # ------------------------------------------

        # Ab hier fast wie vorher, aber wir müssen aufpassen beim Normalisieren
        # Wenn wir Short sind, können Gewichte negativ sein!
        
        # Filter: Kleinstbeträge löschen
        target_weights[target_weights.abs() < MIN_POSITION_SIZE] = 0.0 
        
        # Capping (jetzt in beide Richtungen!)
        target_weights = target_weights.clip(lower=-0.25, upper=0.25)
        
        # Normalisierung (Komplexer bei Long/Short! Wir nutzen Gross Exposure)
        gross_exposure = target_weights.abs().sum()
        if gross_exposure > 0: 
            target_weights = target_weights / gross_exposure
            
        # Glättung (Rebalance Speed)
        current_weights = (current_weights * (1 - REBALANCE_SPEED)) + (target_weights * REBALANCE_SPEED)
        
        # Macro Brain Logik (Bleibt gleich, steuert den Gesamt-Hebel)
        curr_vix = daily_vix.asof(current_date)
        if pd.isna(curr_vix): curr_vix = 20.0
        
        # Hier steuern wir, wie viel GAS wir geben
        if curr_vix < 15.0: 
            base_lev = 1.60 
        elif curr_vix < 26.0: 
            base_lev = 1.00 
        elif curr_vix < 30.0: 
            base_lev = 0.60 
        else: 
            base_lev = 0.0
            
        final_weights = current_weights * base_lev
        final_weights.name = current_date
        weights_history.append(final_weights)
        
    weights_df = pd.DataFrame(weights_history)
    
    if not weights_df.empty:
        weights_df.to_parquet(HISTORY_PATH)
        latest_signal = weights_df.iloc[-1]
        latest_signal.to_frame().T.to_csv(LIVE_SIGNALS_PATH)
        weights_df.to_parquet(SIGNALS_PATH)
        
        print("\n" + "="*40)
        print(f"🏁 LIVE PORTFOLIO (Final Restored Champion)")
        print("="*40)
        print(f"📅 Date: {weights_df.index[-1].date()}")
        exposure = latest_signal.sum()
        print(f"🛡️ Exposure: {exposure:.1%}")
        print("-" * 40)
        print(latest_signal[latest_signal > 0.001].sort_values(ascending=False))
        print("="*40 + "\n")

if __name__ == "__main__":
    try:
        run_trinity_engine()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        # Traceback für besseres Debugging
        import traceback
        traceback.print_exc()
        sys.exit(1)