import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# Wir importieren deine existierenden Module
from ai_ls_allocation.features.build import load_config
from ai_ls_allocation.models.net import GRUModel
from ai_ls_allocation.engine.optimizer import MarkowitzOptimizer
from ai_ls_allocation.engine.backtest import run_backtest
from ai_ls_allocation.engine.report import save_signals

from sklearn.preprocessing import StandardScaler

# --- KONFIGURATION ---
FEATURE_DIR = Path("data/features")
MODEL_DIR = Path("models")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    """Lädt das Panel und die Makro-Daten."""
    # 1. Panel laden
    df = pd.read_parquet(FEATURE_DIR / "panel.parquet")
    
    # 2. Makro-Features laden (für XGBoost)
    macro_df = pd.read_parquet(FEATURE_DIR / "macro_features.parquet")
    macro_df = macro_df.reset_index().rename(columns={'index': 'Date'})
    
    # 3. Mergen (damit wir alles in einem DF haben)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    full_df = pd.merge(df, macro_df, on='Date', how='left').ffill()
    
    return full_df

def get_xgboost_prediction(df):
    """Lädt das trainierte XGBoost Modell und macht Vorhersagen."""
    print("   >>> 🌳 Asking XGBoost (Macro Brain)...")
    model_path = MODEL_DIR / "xgboost_macro_v1.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError("XGBoost model not found! Run train_xgboost.py first.")
        
    model = joblib.load(model_path)
    
    # Features müssen EXAKT so sein wie beim Training
    features = [
        'returns_1d', 'returns_5d', 'returns_20d', 'volatility_60d',
        'VIX', 'TNX_Level', 'TNX_Chg_10d', 'SP500_Trend',
        'Interaction_TNX_Vola'
    ]
    
    # Feature Engineering (Interaction neu berechnen)
    df['Interaction_TNX_Vola'] = df['TNX_Chg_10d'] * df['volatility_60d']
    
    # Vorhersage
    X = df[features]
    preds = model.predict(X)
    
    return preds

def train_and_predict_neural(df):
    """Trainiert das GRU Netz frisch und gibt Vorhersagen zurück."""
    print("   >>> 🧠 Training Neural Network (Chart Brain with Scaling)...")
    
    # 1. Daten vorbereiten
    pivot_df = df.pivot(index='Date', columns='symbol', values='returns_1d').fillna(0)
    
    # SCALING (Das ist neu und extrem wichtig!)
    # Wir skalieren die Daten, damit das Netz nicht verwirrt wird
    scaler = StandardScaler()
    data_values = scaler.fit_transform(pivot_df.values)
    
    num_symbols = pivot_df.shape[1]
    
    # Tensor erstellen
    data_tensor = torch.tensor(data_values, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 2. Modell
    model = GRUModel(input_dim=num_symbols, hidden_dim=64, num_layers=1, output_dim=num_symbols).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    
    # 3. Training (Wir erhöhen auf 20 Epochen für besseres Lernen)
    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        inp = data_tensor[:, :-1, :]
        target = data_tensor[:, 1:, :]
        out = model(inp)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        
    print(f"       Neural Training done. Final Loss: {loss.item():.6f}")
    
    # 4. Vorhersage
    model.eval()
    with torch.no_grad():
        full_preds = model(data_tensor)
        final_preds_np = full_preds.squeeze(0).cpu().numpy()
        
        # WICHTIG: Scaling Rückgängig machen (Inverse Transform)
        # Damit wir wieder echte Returns haben
        final_preds_np = scaler.inverse_transform(final_preds_np)
        
    preds_df = pd.DataFrame(final_preds_np, index=pivot_df.index, columns=pivot_df.columns)
    preds_melted = preds_df.melt(ignore_index=False, var_name='symbol', value_name='pred_neural').reset_index()
    
    return preds_melted

def run_trinity_engine():
    print("\n--- 🚀 STARTING TRINITY ENGINE (Ensemble) ---")
    
    # 1. Daten laden
    full_df = load_data()
    print(f"Data loaded: {full_df.shape}")
    
    # 2. XGBoost Meinung holen (Makro)
    full_df['pred_macro'] = get_xgboost_prediction(full_df)
    
    # 3. Neural Net Meinung holen (Chart)
    # (Da Training lange dauert, nutzen wir hier einen Trick: 
    # Wir nehmen an, das Neural Net liefert ähnliche Ergebnisse wie XGBoost aber geglättet,
    # Um das Skript jetzt lauffähig zu halten, nutzen wir hier 'pred_macro' mit Noise 
    # als Platzhalter ODER trainieren richtig.
    # Lass uns richtig trainieren:)
    neural_preds = train_and_predict_neural(full_df)
    
    # Zusammenfügen
    full_df = pd.merge(full_df, neural_preds, on=['Date', 'symbol'], how='left')
    
    # 4. DAS ENSEMBLE (Die Fusion)
    # Wir gewichten beide Meinungen.
    # 50% Makro, 50% Chart
    print("   >>> ⚖️  Fusing Signals (50% Macro / 50% Neural)...")
    
    # 1. Roh-Signal
    raw_signal = (full_df['pred_macro'] * 0.5) + (full_df['pred_neural'] * 0.5)
    
    # 2. Alpha-Scaling
    scaled_signal = raw_signal * 100.0
    
    # --- NEU: VIX REGIME FILTER (Die Notbremse) ---
    # Wir holen uns die VIX Spalte aus dem DataFrame
    # (Wir haben VIX ja im Panel als Feature, aber hier müssen wir aufpassen:
    # Die Features im 'full_df' heißen 'VIX'. Wir nutzen das Level.)
    
    # Logik: 
    # VIX < 20: Alles normal (Faktor 1.0)
    # VIX 20-28: Vorsicht (Faktor 0.5 - halbe Positionen)
    # VIX > 28: PANIK (Faktor 0.0 - alles verkaufen/Cash)
    
    def apply_vix_filter(row):
        vix = row['VIX']
        if vix > 28.0:
            return 0.0  # Markt ist zu gefährlich -> Cash
        elif vix > 20.0:
            return 0.5  # Markt ist nervös -> Halbes Risiko
        else:
            return 1.0  # Feuer frei
            
    # Wir wenden den Filter an (Vektorisiert ist schneller, aber apply ist lesbarer hier)
    # Da apply langsam sein kann bei 80k Zeilen, nutzen wir numpy where für Speed
    import numpy as np
    
    # Standard: Volle Power
    vix_multiplier = np.ones(len(full_df))
    
    # Vorsicht bei VIX > 20
    vix_multiplier = np.where(full_df['VIX'] > 20, 0.5, vix_multiplier)
    
    # Stopp bei VIX > 28
    vix_multiplier = np.where(full_df['VIX'] > 28, 0.0, vix_multiplier)
    
    # Signal dämpfen
    filtered_signal = scaled_signal * vix_multiplier
    
    # 3. Smoothing (wie vorher, aber auf dem gefilterten Signal)
    full_df['ensemble_signal'] = full_df.groupby('symbol')['pred_neural'].transform(
        lambda x: (filtered_signal.loc[x.index]).ewm(span=2).mean()
    )
    # Shift zurücknehmen (Vorhersage war für t+1)
    # Wir wollen heute handeln basierend auf der Vorhersage für morgen.
    
    # 5. Optimierung (Markowitz mit Ensemble-Signal)
    print("   >>> 📐 Optimizing Portfolio (Markowitz)...")
    
    # Wir müssen die Signale in das Format bringen, das der Optimizer will:
    # Index=Date, Columns=Symbol, Values=Signal
    signals_pivot = full_df.pivot(index='Date', columns='symbol', values='ensemble_signal')
    
    # Optimizer initialisieren
    opt = MarkowitzOptimizer(window_size=60, risk_aversion=2.0)
    
    # Returns für Kovarianz-Matrix
    returns_pivot = full_df.pivot(index='Date', columns='symbol', values='returns_1d').fillna(0)
    
    # Backtest Loop
    weights_history = []
    
    # Wir simulieren die letzten 200 Tage
    test_dates = signals_pivot.index[-200:]
    
    for i, date in enumerate(test_dates):
        if i % 50 == 0:
            print(f"       Optimizing {date.date()}...")
            
        # Aktuelle Vorhersagen
        try:
            current_signals = signals_pivot.loc[date]
            
            # Kovarianz auf vergangene Daten
            past_returns = returns_pivot.loc[:date].iloc[-60:]
            
            # Optimieren
            w = opt.optimize(current_signals, past_returns)
            w.name = date
            weights_history.append(w)
        except Exception as e:
            continue
            
    weights_df = pd.concat(weights_history, axis=1).T
    
    # 6. Speichern für Execution
    print(f"\n✅ Trinity Optimization Complete!")
    print(weights_df.tail())
    
    # Live Signal für "Morgen" speichern
    # --- NEU: Historie für Backtest speichern ---
    history_path = FEATURE_DIR / "trinity_history.parquet"
    weights_df.to_parquet(history_path)
    print(f"💾 History saved to {history_path}")

    # Live Signal für "Morgen" speichern (für den Trader)
    last_weights = weights_df.iloc[-1]
    save_signals(last_weights, FEATURE_DIR / "trinity_signals.parquet")
    print(f"💾 Live Signals saved to data/features/trinity_signals.parquet")

if __name__ == "__main__":
    run_trinity_engine()