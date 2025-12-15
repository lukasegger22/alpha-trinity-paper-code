# --- DEBUG PRINT GANZ OBEN ---
print("--- 🟢 Script bt_trinity.py is initializing... ---", flush=True)

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from ai_ls_allocation.engine.optimizer import MarkowitzOptimizer

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

# --- ADVANCED INDICATORS (THE TRAP DETECTORS) 🕵️‍♂️ ---

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

# NEU: Money Flow Index (Volume-weighted RSI)
def calculate_mfi(df, window=14):
    # Da wir oft nur Close haben, approximieren wir Typical Price mit Close
    typical_price = df['proxy_price'] 
    money_flow = typical_price * df['Volume']
    
    # Positive/Negative Flow
    delta = typical_price.diff()
    pos_flow = pd.Series(np.where(delta > 0, money_flow, 0), index=df.index)
    neg_flow = pd.Series(np.where(delta < 0, money_flow, 0), index=df.index)
    
    rolling_pos = pos_flow.rolling(window).sum()
    rolling_neg = neg_flow.rolling(window).sum().replace(0, 0.001)
    
    mfi_ratio = rolling_pos / rolling_neg
    return 100 - (100 / (1 + mfi_ratio))

# NEU: Bollinger Band Position (%B) - Der Gummiband-Effekt
def calculate_bb_position(series, window=20):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + (2 * std)
    lower = ma - (2 * std)
    
    # Wo sind wir? (1.0 = Oben, 0.0 = Unten, >1.0 = Überkauft/Ausbruch)
    bandwidth = (upper - lower).replace(0, 0.001)
    pct_b = (series - lower) / bandwidth
    return pct_b

def load_data():
    if not PANEL_PATH.exists(): raise FileNotFoundError("Panel data not found.")
    df = pd.read_parquet(PANEL_PATH)
    return df

def run_trinity_engine():
    print(f"\n--- 🚀 STARTING TRINITY ENGINE (Ultimate: MFI + Bollinger + Volume) ---", flush=True)
    print(f"   📅 Training Limit: {TRAIN_END_DATE}")

    # 1. Daten laden
    full_df = load_data()
    full_df = full_df.reset_index()
    if 'Date' not in full_df.columns:
        if 'index' in full_df.columns: full_df = full_df.rename(columns={'index': 'Date'})
        else: full_df = full_df.rename(columns={full_df.columns[0]: 'Date'})
    full_df['Date'] = pd.to_datetime(full_df['Date'])
    
    if 'Volume' not in full_df.columns:
        print("⚠️ Warning: No Volume data found! Using dummy volume.")
        full_df['Volume'] = 1.0

    full_df = full_df.drop_duplicates(subset=['Date', 'symbol'], keep='last')
    full_df = full_df.sort_values(by=['symbol', 'Date'])
    
    start_history = pd.Timestamp(TRAIN_END_DATE) - pd.DateOffset(years=4)
    full_df = full_df[full_df['Date'] >= start_history].copy()

    # --- FEATURE ENGINEERING ---
    print("   >>> 🛠️  Engineering Trap-Detection Features (MFI, Bollinger)...")
    
    full_df['proxy_price'] = (1 + full_df['returns_1d'])
    full_df['proxy_price'] = full_df.groupby('symbol')['proxy_price'].cumprod()
    
    # 1. Efficiency
    full_df['efficiency'] = full_df.groupby('symbol')['proxy_price'].transform(lambda x: calculate_efficiency_ratio(x, window=20)).fillna(0.5)
    
    # 2. OBV Trend
    full_df['obv'] = full_df.groupby('symbol').apply(calculate_obv).reset_index(level=0, drop=True)
    full_df['obv_trend'] = full_df.groupby('symbol')['obv'].pct_change(20).fillna(0)
    
    # 3. VWAP Distance
    def rolling_vwap(x_vol, x_price, w=20):
        pv = x_price * x_vol
        return pv.rolling(w).sum() / x_vol.rolling(w).sum()
    full_df['vwap_20'] = full_df.groupby('symbol', group_keys=False).apply(lambda x: rolling_vwap(x['Volume'], x['proxy_price']))
    full_df['dist_vwap'] = (full_df['proxy_price'] / full_df['vwap_20']) - 1.0
    full_df['dist_vwap'] = full_df['dist_vwap'].fillna(0)

    # 4. MFI (The Exhaustion Trap Detector) 🆕
    full_df['mfi'] = full_df.groupby('symbol', group_keys=False).apply(lambda x: calculate_mfi(x))
    full_df['mfi'] = full_df['mfi'].fillna(50) / 100.0 # Skalieren auf 0-1

    # 5. Bollinger Position (The Extension Trap Detector) 🆕
    full_df['bb_pos'] = full_df.groupby('symbol')['proxy_price'].transform(lambda x: calculate_bb_position(x)).fillna(0.5)

    # --- MERGING ---
    print("   >>> 🔗 Merging Data Sources...")
    if MACRO_SIGNALS_PATH.exists():
        macro = pd.read_parquet(MACRO_SIGNALS_PATH)
        macro = macro[~macro.index.duplicated(keep='last')]
        full_df = pd.merge(full_df, macro, on='Date', how='left')
        full_df['pred_macro'] = full_df['pred_macro'].ffill().fillna(0)

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
    
    # DIE LISTE DER WAHRHEIT (8 Features)
    features = [
        'returns_1d', 'returns_5d', 'volatility_60d', 'VIX', # Basic
        'obv_trend', 'dist_vwap',                            # Volume
        'mfi', 'bb_pos'                                      # Traps
    ]
    
    full_df = full_df.drop_duplicates(subset=['Date', 'symbol'], keep='last')
    full_df = full_df.replace([np.inf, -np.inf], np.nan)
    
    train_mask = full_df['Date'] < TRAIN_END_DATE
    train_df = full_df[train_mask].dropna(subset=features + ['returns_20d'])
    
    print(f"   >>> 🧠 Training Neural Network with Features: {len(features)}")
    scaler = StandardScaler()
    
    if len(train_df) > 0:
        X_train = train_df[features].values
        y_train = train_df['returns_20d'].shift(-20).fillna(0).values 
        
        X_train_scaled = scaler.fit_transform(X_train)
        # Netz größer machen für mehr Features (128 Neuronen)
        model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        X_all = full_df[features].fillna(0).values
        X_all_scaled = scaler.transform(X_all)
        full_df['pred_neural'] = model.predict(X_all_scaled)
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
    
    # Noise Filter (Efficiency < 0.15)
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
    
    for current_date in test_dates:
        idx = pivot_signals.index.get_loc(current_date)
        current_sig = pivot_signals.iloc[idx]
        past_ret = pivot_returns.iloc[:idx]
        
        target_weights = optimizer.optimize(current_sig, past_ret)
        target_weights[target_weights < MIN_POSITION_SIZE] = 0.0 
        target_weights = target_weights.clip(upper=0.25)
        if target_weights.sum() > 0: target_weights = target_weights / target_weights.sum()
            
        current_weights = (current_weights * (1 - REBALANCE_SPEED)) + (target_weights * REBALANCE_SPEED)
        current_weights[current_weights < MIN_POSITION_SIZE] = 0.0
        if current_weights.sum() > 0: current_weights = current_weights / current_weights.sum()
        
        curr_vix = daily_vix.asof(current_date)
        if pd.isna(curr_vix): curr_vix = 20.0
        yield_curve = daily_yield_curve.asof(current_date)
        if pd.isna(yield_curve): yield_curve = 0.5
        
        if curr_vix < 16.0: base_lev = 1.3
        elif curr_vix < 24.0: base_lev = 1.0
        elif curr_vix < 30.0: base_lev = 0.6
        else: base_lev = 0.0
        
        if yield_curve < -0.1 and base_lev > 1.0: base_lev = 1.0
            
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
        print("🏁 LIVE PORTFOLIO (Final Master Model)")
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
        sys.exit(1)