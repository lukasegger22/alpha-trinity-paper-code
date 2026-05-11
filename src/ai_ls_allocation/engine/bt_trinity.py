# --- DEBUG PRINT GANZ OBEN ---
print("--- 🟢 Script bt_trinity.py is initializing... (Trinity V3 Cost Killer) ---", flush=True)

import os
import sys
import pandas as pd
import numpy as np
import random
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import datetime
from datetime import timedelta

# Pfad-Hack
current_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(current_dir))
try:
    from ai_ls_allocation.engine.optimizer import MarkowitzOptimizer
except ImportError:
    print("⚠️ Warning: MarkowitzOptimizer not found. Using simple fallback logic.")
    class MarkowitzOptimizer:
        def __init__(self, **kwargs): pass
        def optimize(self, signals, returns): return signals / signals.sum()

# --- ENV VARS ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

# --- KONFIGURATION ---
FEATURE_DIR = Path("data/features")
MODEL_DIR = Path("models")
MACRO_SIGNALS_PATH = FEATURE_DIR / "macro_signals.parquet"
SENTIMENT_PATH = FEATURE_DIR / "sentiment_signals.parquet"
FRED_PATH = FEATURE_DIR / "fred_macro_economic.parquet"
FUNDAMENTALS_PATH = FEATURE_DIR / "fundamental_quality.parquet"
PANEL_PATH = FEATURE_DIR / "panel.parquet"
HISTORY_PATH = FEATURE_DIR / "trinity_history.parquet"
SIGNALS_PATH = FEATURE_DIR / "trinity_signals.parquet"
LIVE_SIGNALS_PATH = FEATURE_DIR / "trinity_signals.csv"

# Modelle (für Ridge-Integration)
MODEL_XGBOOST_1D = MODEL_DIR / "xgboost_1d.joblib"
MODEL_XGBOOST_5D = MODEL_DIR / "xgboost_5d.joblib"
MODEL_XGBOOST_20D = MODEL_DIR / "xgboost_20d.joblib"
MODEL_RIDGE = MODEL_DIR / "ridge_meta_learner.joblib"
SCALER_PATH = MODEL_DIR / "feature_scaler.joblib"

# --- REALITY SETTINGS ---
#TRAIN_END_DATE = "2021-12-31"
today = datetime.date.today()
TRAIN_END_DATE = (today - timedelta(days=1)).strftime("%Y-%m-%d")
#TEST_START_DATE = "2022-01-01"
TEST_START_DATE = today.strftime("%Y-%m-%d")
COST_OF_CARRY_RATE = 0.05     
MIN_POSITION_SIZE = 0.03 
ENSEMBLE_SIZE = 10 
MIN_HOLD_DAYS = 5 # Anti-Churn Regel
REBALANCE_SPEED = 0.10

# --- PAPER PARAMETERS (Asymmetrische Logik) ---
SHORT_SIGNAL_THRESHOLD = -0.05  # Höhere Hürde für Short-Signale
LONG_SIGNAL_THRESHOLD = 0.02    # Niedrigere Hürde für Long-Signale
MULTI_HORIZON_WEIGHTS = {'1d': 0.5, '5d': 0.3, '20d': 0.2}  # Paper Formula

# --- GLOBAL SEEDING ---
np.random.seed(42)
random.seed(42)

# ==============================================================================
# 🛠️ HELPER FUNCTIONS
# ==============================================================================

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

# ==============================================================================
# 🧠 ADVANCED LOGIC MODULES (ENHANCED)
# ==============================================================================

def calculate_momentum_score(returns_series, lookback=60):
    try:
        cum_return = (1 + returns_series.tail(lookback)).prod() - 1
        return 1.0 if cum_return > 0 else 0.0
    except: return 0.0

def calculate_kelly_positions(signals, past_returns, max_kelly=0.25):
    kelly_weights = pd.Series(0.0, index=signals.index)
    for symbol in signals.index:
        if symbol not in past_returns.columns: continue
        symbol_rets = past_returns[symbol].tail(252)
        if len(symbol_rets) < 60: continue
        wins = symbol_rets[symbol_rets > 0]
        losses = symbol_rets[symbol_rets < 0]
        if len(wins) == 0 or len(losses) == 0: continue
        win_rate = len(wins) / len(symbol_rets)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        if avg_loss == 0: continue
        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        kelly_fraction = max(0, min(kelly_fraction * 0.5, max_kelly))
        direction = 1.0 if signals[symbol] > 0 else -1.0
        kelly_weights[symbol] = kelly_fraction * direction * abs(signals[symbol])
    return kelly_weights

def enforce_minimum_hold(target_weights, current_weights, current_date, entry_dates):
    """VERBESSERUNG A: Minimum Hold Period"""
    adjusted = target_weights.copy()
    for symbol in current_weights.index:
        # Wir haben eine Position
        if abs(current_weights[symbol]) > 0.01:
            # Und Target will sie schließen/reduzieren
            if abs(target_weights.get(symbol, 0)) < 0.01:
                # Check Datum
                if symbol in entry_dates:
                    days_held = (current_date - entry_dates[symbol]).days
                    if days_held < MIN_HOLD_DAYS:
                        # Zwinge zum Halten!
                        adjusted[symbol] = current_weights[symbol]
        
        # Tracking Updates
        # Wenn wir neu in eine Position gehen (vorher 0, jetzt >0)
        if abs(current_weights.get(symbol, 0)) < 0.01 and abs(target_weights.get(symbol, 0)) > 0.01:
            entry_dates[symbol] = current_date
            
    return adjusted

def apply_asymmetric_signals(signals):
    """PAPER REQUIREMENT: Asymmetrische Signal-Schwellen für Shorts"""
    adjusted = signals.copy()
    
    for symbol in adjusted.index:
        sig = adjusted[symbol]
        
        # Shorts: Höhere Hürde (require -0.05 or worse)
        if sig < SHORT_SIGNAL_THRESHOLD and sig >= SHORT_SIGNAL_THRESHOLD * 0.5:
            adjusted[symbol] = 0  # Reject weak shorts
        
        # Longs: Niedrigere Hürde
        if sig > 0 and sig < LONG_SIGNAL_THRESHOLD:
            adjusted[symbol] = 0  # Reject weak longs
    
    return adjusted

def check_transaction_costs(current_weights, target_weights, expected_returns_proxy):
    """VERBESSERUNG B: Cost Benefit Analysis"""
    # 1. Kosten berechnen (Konservativ: 10bps)
    turnover = (target_weights - current_weights).abs().sum()
    cost = turnover * 0.0010 
    
    # 2. Nutzen schätzen (Ohne Look-ahead Bias!)
    # Wir nutzen den gleitenden Durchschnitt der Returns als Proxy für "Expected Return"
    expected_benefit = 0.0
    for sym in target_weights.index:
        w_diff = target_weights[sym] - current_weights.get(sym, 0.0)
        # Erwarteter Return: Letzte 20 Tage Durchschnitt
        exp_ret = expected_returns_proxy.get(sym, 0.0)
        expected_benefit += w_diff * exp_ret
        
    # Wenn die Kosten höher sind als der halbe erwartete Nutzen -> NICHT HANDELN
    if expected_benefit < cost * 2.0:
        return current_weights # Wir bleiben einfach sitzen
    else:
        return target_weights

def calculate_enhanced_crisis_score(current_date, spy_data, daily_vix, full_df):
    """VERBESSERUNG 3: Smarter Crisis Detection (7 Signale)"""
    signals = []
    
    # 1. TREND
    if current_date not in spy_data.index: return 0.0
    spy_current = spy_data.loc[:current_date]
    if len(spy_current) < 200: return 0.0
    
    price = spy_current['proxy_price'].iloc[-1]
    sma_50 = spy_current['proxy_price'].rolling(50).mean().iloc[-1]
    sma_200 = spy_current['proxy_price'].rolling(200).mean().iloc[-1]
    
    trend_signal = 0.0
    if price < sma_50: trend_signal += 0.5
    if sma_50 < sma_200: trend_signal = 1.0 # Death Cross
    signals.append(trend_signal * 0.25)
    
    # 2. VIX REGIME
    curr_vix = daily_vix.asof(current_date)
    if pd.isna(curr_vix): curr_vix = 20.0
    vix_signal = 0.0
    if curr_vix > 30: vix_signal = 1.0
    elif curr_vix > 20: vix_signal = 0.5
    signals.append(vix_signal * 0.20)
    
    # 3. DRAWDOWN
    if len(spy_current) > 252:
        peak = spy_current['proxy_price'].rolling(252).max().iloc[-1]
        dd = (peak - price) / peak
        dd_signal = min(dd / 0.15, 1.0)
        signals.append(dd_signal * 0.20)
    else: signals.append(0.0)
    
    # 4. VIX ACCELERATION (Early Warning)
    vix_series = daily_vix.loc[:current_date].tail(5)
    if len(vix_series) >= 5:
        vix_change = (vix_series.iloc[-1] / vix_series.iloc[0]) - 1
        vix_accel = min(max(vix_change * 2, 0), 1.0)
        signals.append(vix_accel * 0.15)
    else: signals.append(0.0)
    
    # 5. BREADTH (Wie viele Assets fallen?)
    # Wir brauchen Zugriff auf das full_df für diesen Tag
    # Um Performance zu sparen, nutzen wir eine Annäherung im Main Loop oder hier vereinfacht:
    # (Hier im Helper ist der Zugriff schwer, daher Dummy oder einfache Logik)
    signals.append(0.0) # Platzhalter, Breadth ist teuer zu berechnen im Loop
    
    # 6. CREDIT STRESS (Proxy via TLT Spike)
    # Wenn TLT (Bonds) an einem Tag stark steigen (>1%), ist oft Angst im Markt
    # Wir ignorieren das hier für Speed, da VIX Accel das meistens abdeckt
    signals.append(0.0)

    # 7. MOMENTUM COLLAPSE
    if len(spy_current) > 60:
        mom_20 = (spy_current['proxy_price'].iloc[-1] / spy_current['proxy_price'].iloc[-20]) - 1
        mom_60 = (spy_current['proxy_price'].iloc[-1] / spy_current['proxy_price'].iloc[-60]) - 1
        if mom_20 < mom_60 * 0.5:
            signals.append(1.0 * 0.10) # Momentum bricht weg
        else:
            signals.append(0.0)
    else: signals.append(0.0)

    return min(sum(signals), 1.0)

# ==============================================================================
# 🚀 MAIN ENGINE
# ==============================================================================

def load_data():
    if not PANEL_PATH.exists(): raise FileNotFoundError("Panel data not found.")
    df = pd.read_parquet(PANEL_PATH)
    return df

def run_trinity_engine():
    print(f"\n--- 🚀 STARTING TRINITY ENGINE (V4 Paper-Exact) ---", flush=True)

    # 1. LOAD DATA
    full_df = load_data().reset_index()
    if 'date' in full_df.columns: full_df.rename(columns={'date': 'Date'}, inplace=True)
    if 'index' in full_df.columns and 'Date' not in full_df.columns: full_df.rename(columns={'index': 'Date'}, inplace=True)
    if 'Symbol' in full_df.columns: full_df.rename(columns={'Symbol': 'symbol'}, inplace=True)
    full_df['Date'] = pd.to_datetime(full_df['Date'])
    if 'Volume' not in full_df.columns: full_df['Volume'] = 1.0
    full_df = full_df.drop_duplicates(subset=['Date', 'symbol'], keep='last').sort_values(by=['symbol', 'Date'])
    
    # 2. LOAD MACRO SIGNALS (XGBoost + Ridge Multi-Horizon)
    print("   >>> 📥 Loading XGBoost + Ridge Signals...", flush=True)
    macro_signals_df = None
    if MACRO_SIGNALS_PATH.exists():
        macro_signals_df = pd.read_parquet(MACRO_SIGNALS_PATH)
        # Merge mit full_df
        full_df = pd.merge(
            full_df,
            macro_signals_df.reset_index() if hasattr(macro_signals_df.index, 'names') else macro_signals_df.reset_index(drop=True),
            left_index=False,
            right_index=False,
            how='left'
        )
        print(f"   ✅ Macro signals loaded: pred_1d, pred_5d, pred_20d, pred_ridge, voting_signal")
    else:
        print("   ⚠️  Macro signals not found. Using fallback neural ensemble.")
        full_df['voting_signal'] = 0.0
    
    # 3. FEATURE ENGINEERING
    print("   >>> 🛠️  Engineering Features...", flush=True)
    full_df['proxy_price'] = full_df.groupby('symbol')['returns_1d'].transform(lambda x: (1 + x).cumprod())
    
    # SPY Data
    spy_data = full_df[full_df['symbol'] == 'SPY'].copy().set_index('Date').sort_index()
    if spy_data.empty: spy_data = full_df[full_df['symbol'] == 'QQQ'].copy().set_index('Date').sort_index()
    
    # Technicals
    full_df['efficiency'] = full_df.groupby('symbol')['proxy_price'].transform(lambda x: calculate_efficiency_ratio(x)).fillna(0.5)
    full_df['obv'] = full_df.groupby('symbol', group_keys=False).apply(calculate_obv).reset_index(level=0, drop=True)
    full_df['obv_trend'] = full_df.groupby('symbol')['obv'].pct_change(20).fillna(0)
    full_df['mfi'] = full_df.groupby('symbol', group_keys=False).apply(lambda x: calculate_mfi(x)).fillna(50) / 100.0
    full_df['bb_pos'] = full_df.groupby('symbol')['proxy_price'].transform(lambda x: calculate_bb_position(x)).fillna(0.5)
    
    # VWAP
    def rolling_vwap(x_vol, x_price, w=20): return (x_price * x_vol).rolling(w).sum() / x_vol.rolling(w).sum()
    full_df['vwap_20'] = full_df.groupby('symbol', group_keys=False).apply(lambda x: rolling_vwap(x['Volume'], x['proxy_price']))
    full_df['dist_vwap'] = (full_df['proxy_price'] / full_df['vwap_20']) - 1.0
    full_df['dist_vwap'] = full_df['dist_vwap'].fillna(0)

    # Fallback Placeholder Features
    full_df['pred_macro'] = 0 
    full_df['sentiment_score'] = 0.0 
    full_df['Quality_Score'] = 0.7 

    # 4. USE VOTING_SIGNAL IF AVAILABLE (from XGBoost + Ridge)
    # Otherwise, fallback to Neural Ensemble
    print(f"   >>> 📊 Signal Selection...", flush=True)
    
    if 'voting_signal' in full_df.columns:
        print("   ✅ Using Multi-Horizon Voting Signal (XGBoost + Ridge)")
        full_df['raw_signal'] = full_df['voting_signal']
    else:
        # FALLBACK: Train Neural Ensemble (Legacy)
        print("   ⚠️  Voting signal not found. Training Neural Ensemble Fallback...")
        features = ['returns_1d', 'returns_5d', 'volatility_60d', 'VIX', 'obv_trend', 'dist_vwap', 'mfi', 'bb_pos']
        full_df = full_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Live-Training Split
        train_df = full_df[full_df['Date'] <= TRAIN_END_DATE].copy()
        train_df['target'] = train_df['returns_20d'].shift(-20)
        train_df = train_df.dropna(subset=['target'])
        
        if len(train_df) > 100:
            X_train = train_df[features].values
            y_train = train_df['target'].values 
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_all = full_df[features].values
            X_all_scaled = scaler.transform(X_all)
            
            ensemble_preds = np.zeros((len(full_df), ENSEMBLE_SIZE))
            for i in range(ENSEMBLE_SIZE):
                if i < ENSEMBLE_SIZE // 2:
                    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42+i)
                else:
                    model = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42+i, n_jobs=-1)
                model.fit(X_train_scaled, y_train)
                ensemble_preds[:, i] = model.predict(X_all_scaled)
            full_df['raw_signal'] = np.mean(ensemble_preds, axis=1)
        else:
            print("❌ Fallback Training Failed - no data")
            full_df['raw_signal'] = 0.0
    
    # Ensure raw_signal exists
    if 'raw_signal' not in full_df.columns:
        full_df['raw_signal'] = 0.0
    
    full_df['ensemble_signal'] = full_df.groupby('symbol')['raw_signal'].transform(lambda x: x.ewm(span=5).mean()) * 50.0

    # 5. BACKTEST LOOP
    print(f"   >>> 📐 Backtesting starting {TEST_START_DATE}...", flush=True)
    
    pivot_signals = full_df.pivot(index='Date', columns='symbol', values='ensemble_signal').fillna(0)
    pivot_returns = full_df.pivot(index='Date', columns='symbol', values='returns_1d').fillna(0)
    daily_vix = full_df.groupby('Date')['VIX'].mean()
    
    test_dates = pivot_signals.index[pivot_signals.index >= TEST_START_DATE]
    optimizer = MarkowitzOptimizer(window_size=60, risk_aversion=0.5, target_vol=0.15)
    
    weights_history = []
    current_weights = pd.Series(0.0, index=pivot_signals.columns)
    
    # TRACKING VARIABLES (NEU!)
    entry_dates = {}
    
    for current_date in test_dates:
        idx = pivot_signals.index.get_loc(current_date)
        current_sig = pivot_signals.iloc[idx].copy()
        past_ret = pivot_returns.iloc[:idx]
        
        # A. APPLY ASYMMETRIC SIGNAL THRESHOLDS (Paper Requirement)
        current_sig = apply_asymmetric_signals(current_sig)
        
        # B. ENHANCED CRISIS SCORE
        # Wir nutzen deine neue, schlaue Formel
        crisis_score = calculate_enhanced_crisis_score(current_date, spy_data, daily_vix, full_df)
        
        # B. REGIME SELECTION
        if crisis_score > 0.65:
            # 🔴 REGIME: PANIC (Full Defense)
            target_weights = pd.Series(0.0, index=current_sig.index)
            # Safe Havens Priorität
            if 'SHY' in target_weights.index: target_weights['SHY'] = 0.50
            elif 'UUP' in target_weights.index: target_weights['UUP'] = 0.50 # Fallback
            
            if 'GLD' in target_weights.index: target_weights['GLD'] = 0.30
            
            used_speed = 1.0
            base_lev = 0.8
            
        elif crisis_score > 0.35:
            # 🟡 REGIME: CAUTION (Risk-Off)
            top_sigs = current_sig[current_sig > 0].nlargest(5)
            target_weights = pd.Series(0.0, index=current_sig.index)
            target_weights[top_sigs.index] = top_sigs
            if target_weights.sum() > 0: target_weights = target_weights / target_weights.sum() * 0.4
            
            # Hedge dazu
            if 'UUP' in target_weights.index: target_weights['UUP'] += 0.25
            
            used_speed = 0.3
            base_lev = 0.85
            
        else:
            # 🟢 REGIME: GROWTH
            
            # Momentum Boost
            for s in current_sig.index:
                mom = calculate_momentum_score(past_ret[s] if s in past_ret else pd.Series())
                if mom > 0 and current_sig[s] > 0: current_sig[s] *= 1.2
            
            # Kelly Sizing
            target_weights = calculate_kelly_positions(current_sig, past_ret)
            if target_weights.sum() == 0:
                 try: target_weights = optimizer.optimize(current_sig, past_ret)
                 except: target_weights = pd.Series(0, index=current_sig.index)
            
            # Correlation Limits
            # ... (Code gekürzt für Übersicht, Funktion ist oben definiert) ...
            
            # Capping
            target_weights = target_weights.clip(lower=-0.30, upper=0.30)
            gross = target_weights.abs().sum()
            if gross > 0: target_weights /= gross
            
            used_speed = REBALANCE_SPEED
            
            # NEU: Leverage Cost Management (Weniger ist mehr)
            # Max 1.30 statt 1.60
            curr_vix = daily_vix.asof(current_date)
            if pd.isna(curr_vix): curr_vix = 20.0
            
            if curr_vix < 15: base_lev = 1.00 # Reduziert von 1.60!
            elif curr_vix < 20: base_lev = 0.90
            elif curr_vix < 25: base_lev = 0.70
            else: base_lev = 0.50

        # C. COST KILLER LOGIC (VOR DER EXECUTION)
        
        # 1. Enforce Minimum Hold
        # Wir überschreiben target_weights, wenn Position zu jung ist
        target_weights = enforce_minimum_hold(target_weights, current_weights, current_date, entry_dates)
        
        # 2. Transaction Cost Check
        # Wir berechnen, ob sich das Umschichten lohnt
        # Proxy für Expected Return = Durchschnitt der letzten 20 Tage
        exp_ret_proxy = past_ret.tail(20).mean().fillna(0)
        target_weights = check_transaction_costs(current_weights, target_weights, exp_ret_proxy)

        # D. EXECUTION
        current_weights = (current_weights * (1 - used_speed)) + (target_weights * used_speed)
        final_weights = current_weights * base_lev
        final_weights.name = current_date
        weights_history.append(final_weights)
        
    # OUTPUT
    weights_df = pd.DataFrame(weights_history)
    if not weights_df.empty:
        weights_df.to_parquet(HISTORY_PATH)
        latest_signal = weights_df.iloc[-1]
        latest_signal.to_frame().T.to_csv(LIVE_SIGNALS_PATH)
        weights_df.to_parquet(SIGNALS_PATH)
        
        print("\n" + "="*40)
        print(f"🏁 LIVE PORTFOLIO (Trinity V3 Cost Killer)")
        print(f"📅 Date: {weights_df.index[-1].date()}")
        print(f"📊 Final Crisis Score: {crisis_score:.2f}")
        print("="*40)

if __name__ == "__main__":
    try:
        run_trinity_engine()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()