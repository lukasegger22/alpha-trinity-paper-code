import pandas as pd
import numpy as np
from pathlib import Path
import base64

# Config
REPORTS_DIR = Path("reports")
FILE_TS = REPORTS_DIR / "bt_neural_timeseries.csv"
FILE_W = REPORTS_DIR / "bt_neural_weights.csv"
IMG_PATH = REPORTS_DIR / "m1_dashboard.png"
OUT_HTML = REPORTS_DIR / "strategy_report.html"

def calculate_metrics(df):
    """Berechnet CAGR, Sharpe, MaxDD"""
    days = len(df)
    years = days / 252
    
    # CAGR
    total_ret = df["equity"].iloc[-1]
    cagr = (total_ret ** (1/years)) - 1
    
    # Sharpe (tägliche Returns)
    daily_ret = df["port_ret"]
    vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() * 252) / vol if vol > 0 else 0
    
    # Max Drawdown
    peak = df["equity"].cummax()
    dd = (df["equity"] / peak) - 1
    max_dd = dd.min()
    
    return cagr, sharpe, max_dd, vol

def get_image_string(path):
    """Wandelt Bild in Base64 String um für HTML Einbettung"""
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{encoded}"

def main():
    print("[report] Generating HTML report...")
    
    # 1. Daten laden
    if not FILE_TS.exists() or not FILE_W.exists():
        print("Error: Backtest files not found. Run bt_neural.py first.")
        return

    # Wir nutzen index_col=0 (Erste Spalte), egal wie sie heißt
    ts = pd.read_csv(FILE_TS, index_col=0, parse_dates=True)
    weights = pd.read_csv(FILE_W, index_col=0, parse_dates=True)

    # 2. KPIs berechnen
    cagr, sharpe, max_dd, vol = calculate_metrics(ts)
    
    # 3. Aktuelle Signale (Die letzte Zeile!)
    last_date = weights.index[-1]
    last_positions = weights.iloc[-1]
    
    # Nur Positionen anzeigen, die nicht 0 sind (> 1%)
    active_positions = last_positions[last_positions.abs() > 0.01].sort_values(ascending=False)
    
    # Cash Position berechnen
    invested_pct = active_positions.abs().sum()
    cash_pct = 1.0 - invested_pct

    # 4. HTML bauen
    img_b64 = get_image_string(IMG_PATH)
    
    html = f"""
    <html>
    <head>
        <title>AI L/S Allocation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f4f4f9; color: #333; }}
            .container {{ max_width: 1000px; margin: 0 auto; background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
            h2 {{ color: #555; margin-top: 30px; }}
            .kpi-box {{ display: flex; justify-content: space-between; margin: 20px 0; }}
            .metric {{ background: #eef; padding: 15px; border-radius: 5px; text-align: center; width: 22%; }}
            .metric h3 {{ margin: 0; font-size: 24px; color: #2a5caa; }}
            .metric p {{ margin: 5px 0 0; color: #666; font-size: 14px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f8f8f8; }}
            .pos-long {{ color: green; font-weight: bold; }}
            .pos-short {{ color: red; font-weight: bold; }}
            .img-container {{ text-align: center; margin-top: 20px; }}
            img {{ max-width: 100%; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 AI Long/Short Allocation Report</h1>
            <p><strong>Date:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
            <p><strong>Strategy Logic:</strong> GRU Neural Network with Quantile Regression & Uncertainty-Aware Sizing (IQR).</p>
            
            <div class="kpi-box">
                <div class="metric"><h3>{cagr:.1%}</h3><p>CAGR (Ann. Return)</p></div>
                <div class="metric"><h3>{sharpe:.2f}</h3><p>Sharpe Ratio</p></div>
                <div class="metric"><h3>{max_dd:.1%}</h3><p>Max Drawdown</p></div>
                <div class="metric"><h3>{vol:.1%}</h3><p>Ann. Volatility</p></div>
            </div>

            <h2>🚀 Current Signals (For: {last_date.date()})</h2>
            <p>Based on the latest available data close. Positive = LONG, Negative = SHORT.</p>
            <table>
                <tr><th>Asset</th><th>Allocation %</th><th>Action</th></tr>
    """
    
    # Tabelle füllen
    for sym, w in active_positions.items():
        color = "pos-long" if w > 0 else "pos-short"
        action = "BUY / HOLD" if w > 0 else "SELL / SHORT"
        html += f"<tr><td>{sym}</td><td class='{color}'>{w:.1%}</td><td>{action}</td></tr>"
    
    html += f"""
                <tr><td><strong>CASH / IDLE</strong></td><td><strong>{cash_pct:.1%}</strong></td><td>WAIT</td></tr>
            </table>

            <h2>📈 Performance Dashboard</h2>
            <div class="img-container">
                <img src="{img_b64}" alt="Performance Chart">
            </div>
        </div>
    </body>
    </html>
    """

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
        
    print(f"[report] Saved report to {OUT_HTML}")
    # Versuchen, es im Browser zu öffnen (Mac)
    try:
        import subprocess
        subprocess.run(["open", str(OUT_HTML)])
    except:
        pass

if __name__ == "__main__":
    main()