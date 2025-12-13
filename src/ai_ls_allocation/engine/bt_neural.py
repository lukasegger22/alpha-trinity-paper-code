from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from ai_ls_allocation.engine.optimizer import get_portfolio_weights

# Importiere unser neues Netz
from ai_ls_allocation.models.net import QuantileGRU, quantile_loss

# ---- Config ----
DATA_DIR = Path("data")
PANEL_FP = DATA_DIR / "features" / "panel.parquet"
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Modell-Parameter
SEQ_LEN = 20        # Wir schauen 20 Tage zurück
HIDDEN_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 15
LR = 0.001

# In bt_neural.py
FEATS = [
    "vol20", "vol60", 
    "dist_ma20", "dist_ma60", "trend_strength",  # Neue Namen
    "macd", "macd_signal", "macd_hist",          # Namen bleiben gleich (aber Inhalt ist normalisiert)
    "spread_hyg_ief", "r1"
]
TARGET_COL = "target_r1" # Wir sagen t+1 vorher

def load_and_scale_data():
    print(f"[neural] Loading {PANEL_FP}")
    df = pd.read_parquet(PANEL_FP).reset_index()
    
    # 1. Datum global sortieren für korrekten Split!
    # Wir brauchen alle einzigartigen Tage in richtiger Reihenfolge
    all_dates = np.sort(df["date"].unique())
    
    split_idx1 = int(len(all_dates) * 0.6)
    split_idx2 = int(len(all_dates) * 0.8)
    
    train_date_end = all_dates[split_idx1]
    valid_date_end = all_dates[split_idx2]
    
    print(f"[neural] Split Dates: Train end {train_date_end}, Valid end {valid_date_end}")

    # Drop rows where target is NaN
    df = df.dropna(subset=FEATS + [TARGET_COL])
    
    # Scaler fitten (nur auf Train!)
    train_mask = df["date"] <= train_date_end
    
    # Sicherheitscheck: Haben wir Daten im Training?
    if not train_mask.any():
        raise ValueError("No training data found! Check date format.")

    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, FEATS])
    
    X_scaled = scaler.transform(df[FEATS])
    df_scaled = df.copy()
    df_scaled[FEATS] = X_scaled
    
    # Nach Symbol sortieren für Sequenz-Bau
    df_scaled = df_scaled.sort_values(["symbol", "date"])
    
    return df_scaled, train_date_end, valid_date_end

def create_sequences(df, seq_len):
    """
    Erstellt (Batch, Seq_Len, Feats) Arrays für jedes Symbol.
    Das ist tricky: Wir dürfen nicht über Symbol-Grenzen hinwegfenstern.
    """
    all_X, all_y, all_dates, all_syms = [], [], [], []
    
    for sym, group in df.groupby("symbol"):
        # Arrays extrahieren
        data_x = group[FEATS].values
        data_y = group[TARGET_COL].values
        dates = group["date"].values
        
        if len(data_x) <= seq_len:
            continue
            
        # Sliding Window Trick mit numpy strides (schnell)
        # Aber hier loop ist sicherer für Verständnis
        for i in range(len(data_x) - seq_len):
            # Input: Tag i bis i+seq_len-1
            seq_x = data_x[i : i+seq_len]
            # Target: Tag i+seq_len-1 (der korrespondiert zu target_r1 an diesem Tag)
            # Da target_r1[t] = return[t+1], passt das: 
            # Wir nutzen Daten bis t, um Return t+1 vorherzusagen.
            target = data_y[i + seq_len - 1] 
            
            date_of_pred = dates[i + seq_len - 1]
            
            all_X.append(seq_x)
            all_y.append(target)
            all_dates.append(date_of_pred)
            all_syms.append(sym)
            
    return np.array(all_X), np.array(all_y), np.array(all_dates), np.array(all_syms)

def train_model(X_train, y_train, X_val, y_val, input_dim):
    """PyTorch Training Loop"""
    # Tensors
    X_tr_t = torch.FloatTensor(X_train)
    y_tr_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_va_t = torch.FloatTensor(X_val)
    y_va_t = torch.FloatTensor(y_val).unsqueeze(1)
    
    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = QuantileGRU(input_size=input_dim, hidden_size=HIDDEN_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"[neural] Start training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = quantile_loss(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_va_t)
            val_loss = quantile_loss(val_preds, y_va_t).item()
            
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss/len(loader):.4f} | Val Loss: {val_loss:.4f}")
            
    return model

# In src/ai_ls_allocation/engine/bt_neural.py

def main():
    # 1. Daten laden & skalieren
    df, t_end, v_end = load_and_scale_data()
    
    # --- LIVE DATA SICHERN ---
    df_live = df[df[TARGET_COL].isna()].copy()
    df_backtest = df.dropna(subset=[TARGET_COL]).copy()
    
    # 2. Sequenzen bauen (Nur Backtest-Daten für Training)
    print("[neural] Building sequences for training...")
    X, y, dates, syms = create_sequences(df_backtest, SEQ_LEN)
    
    # 3. Split
    mask_train = dates <= t_end
    mask_val = (dates > t_end) & (dates <= v_end)
    mask_test = dates > v_end
    
    X_train, y_train = X[mask_train], y[mask_train]
    X_val, y_val = X[mask_val], y[mask_val]
    X_test, y_test = X[mask_test], y[mask_test]
    
    dates_test = dates[mask_test]
    syms_test = syms[mask_test]
    
    print(f"[neural] Splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # 4. Training
    model = train_model(X_train, y_train, X_val, y_val, input_dim=len(FEATS))
    
    # 5. Backtest (Prediction)
    print("[neural] Running Backtest on Test Set...")
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test)
        preds = model.predict_sorted(X_test_t).numpy()
    
    q10_pred = preds[:, 0]
    q50_pred = preds[:, 1]
    q90_pred = preds[:, 2]
    
    res_df = pd.DataFrame({
        "date": dates_test, "symbol": syms_test,
        "q10": q10_pred, "q50": q50_pred, "q90": q90_pred,
        "r1_realized": y_test 
    })
    
    # 6. Backtest Logik (Pivot & Sizing)
    df_q10 = res_df.pivot(index="date", columns="symbol", values="q10").fillna(0.0)
    df_q50 = res_df.pivot(index="date", columns="symbol", values="q50").fillna(0.0)
    df_q90 = res_df.pivot(index="date", columns="symbol", values="q90").fillna(0.0)
    r_next = res_df.pivot(index="date", columns="symbol", values="r1_realized").fillna(0.0)
    
    # -- Fast Optimization für Backtest (Simulation) --
    iqr = (df_q90 - df_q10).replace(0, 1e-6)
    
    # Alpha Signal
    raw_signal = (df_q50 / iqr) * 2.0
    
    # Filter: Rauschen entfernen
    smart_signal = raw_signal.map(lambda x: x if abs(x) > 0.02 else 0.0)
    
    # Normalisieren
    weights = smart_signal.apply(lambda row: row / max(1.0, row.abs().sum()), axis=1)
    
    # PROFIT-UPGRADE: Diversifikations-Zwang im Backtest
    # Wir cappen jede Position auf max 25% Portfolio-Größe.
    # Das verhindert, dass der Bot "All-In" auf eine Aktie geht (Klumpenrisiko).
    weights = weights.clip(-0.25, 0.25)
    
    # Nach dem Capping müssen wir neu normalisieren, damit wir nicht >100% investiert sind
    weights = weights.apply(lambda row: row / max(1.0, row.abs().sum()), axis=1)
    
    # Reporting & Speichern
    COST_BPS = 0.0010 
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost_drag = turnover * COST_BPS
    gross_ret = (weights * r_next).sum(axis=1)
    net_ret = gross_ret - cost_drag
    equity = (1.0 + net_ret).cumprod()
    
    ts = pd.DataFrame({"port_ret": net_ret, "equity": equity, "turnover": turnover, "cost": cost_drag})
    ts.to_csv(REPORTS_DIR / "bt_neural_timeseries.csv")
    
    # -------------------------------------------------------------------------
    # TEIL 2: LIVE INFERENCE (HIER KOMMT DER PROFI-OPTIMIERER)
    # -------------------------------------------------------------------------
    print("\n--- 🟢 LIVE PRO-OPTIMIZATION FOR TOMORROW ---")
    
    # A) Live Predictions sammeln
    live_preds_list = []
    
    for sym, group in df.groupby("symbol"):
        if len(group) < SEQ_LEN: continue
        # Letzte Sequenz (inkl. heute)
        last_seq = group[FEATS].iloc[-SEQ_LEN:].values
        
        with torch.no_grad():
            seq_t = torch.FloatTensor(last_seq).unsqueeze(0)
            pred = model.predict_sorted(seq_t).numpy()[0]
        
        q10, q50, q90 = pred
        iqr_val = q90 - q10 if (q90 - q10) > 0 else 1e-6
        
        # Das ist unser "Expected Return" Signal
        sig = (q50 / iqr_val) * 2.0
        
        # Filter
        if abs(sig) < 0.02: sig = 0.0
            
        live_preds_list.append({"symbol": sym, "signal": sig})
        
    # B) Vorbereiten für Optimizer
    # Forecasts (Alpha)
    forecasts = pd.DataFrame(live_preds_list).set_index("symbol")["signal"]
    
    # Historische Returns (für Risiko-Matrix)
    # Wir holen die letzten 60 Tage Returns aus den Rohdaten
    returns_history = df.pivot(index="date", columns="symbol", values="r1").iloc[-60:].fillna(0.0)
    
    # C) ECHTE OPTIMIERUNG LAUFEN LASSEN
    try:
        # Hier passiert die Magie: Korrelationen werden geprüft!
        optimal_weights = get_portfolio_weights(
            forecasts=forecasts,
            returns_history=returns_history,
            target_vol=0.15,  # Ziel-Volatilität
            max_weight=0.25   # Max 25% in einer Aktie (Sicherheitsnetz)
        )
        
        print("✅ Optimized Weights (Risk Adjusted & Diversified):")
        print(optimal_weights[optimal_weights.abs() > 0.01].sort_values(ascending=False))
        
        # Anhängen an CSV für Execution
        next_date = df["date"].max()
        new_row = optimal_weights.to_frame().T
        new_row.index = [next_date]
        
        weights_combined = pd.concat([weights, new_row])
        weights_combined = weights_combined[~weights_combined.index.duplicated(keep='last')]
        weights_combined.to_csv(REPORTS_DIR / "bt_neural_weights.csv")
        print(f"[live] Appended OPTIMIZED signals for {next_date}")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        # Fallback: Falls Solver crasht, nehmen wir die rohen Signale
        # (Sollte nicht passieren, aber Safety First)

if __name__ == "__main__":
    main()