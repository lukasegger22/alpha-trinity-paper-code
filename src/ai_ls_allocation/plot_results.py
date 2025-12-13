import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

# Config
REPORTS_DIR = Path("reports")
FILE_TS = REPORTS_DIR / "bt_neural_timeseries.csv"
FILE_W = REPORTS_DIR / "bt_neural_weights.csv"

def main():
    if not FILE_TS.exists():
        print(f"File not found: {FILE_TS}")
        return

    # Daten laden
    df = pd.read_csv(FILE_TS, parse_dates=["date"]).set_index("date")
    weights = pd.read_csv(FILE_W, parse_dates=["date"]).set_index("date")

    # Exposure berechnen (Wie viel % waren wir investiert?)
    # Summe der absoluten Gewichte
    exposure = weights.abs().sum(axis=1)

    # Setup Plot (3 untereinander)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1.5]})
    
    # --- 1. Equity Curve ---
    ax1.plot(df.index, df["equity"], label="Neural Strategy (Net)", color="#1f77b4", linewidth=2)
    ax1.set_title("Milestone 1: Neural GRU Strategy (Signal-to-Noise Sizing)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Equity ($1 start)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    
    # Benchmark Vergleich (optional, falls SPY Daten da sind)
    # ax1.plot(spy_data, label="SPY Benchmark", color="gray", alpha=0.5, linestyle="--")

    # --- 2. Drawdown ---
    # Drawdown berechnen: (Equity / Peak) - 1
    dd = (df["equity"] / df["equity"].cummax()) - 1
    ax2.fill_between(dd.index, dd, 0, color="red", alpha=0.3, label="Drawdown")
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left")

    # --- 3. Exposure (Der Beweis für Smart Sizing) ---
    ax3.fill_between(exposure.index, exposure, 0, color="green", alpha=0.2, label="Gross Exposure (Invested %)")
    ax3.plot(exposure.index, exposure, color="green", linewidth=1)
    
    # 100% Linie einzeichnen
    ax3.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    
    ax3.set_ylabel("Exposure")
    ax3.set_xlabel("Date")
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax3.set_ylim(0, 1.1) # Max 110% anzeigen
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left")
    ax3.text(df.index[0], 0.1, "  Low Exposure = High Uncertainty (IQR)", fontsize=9, style='italic')

    plt.tight_layout()
    
    out_path = REPORTS_DIR / "m1_dashboard.png"
    plt.savefig(out_path, dpi=300)
    print(f"[plot] Saved dashboard to {out_path}")
    # plt.show() # Optional, falls du es direkt sehen willst

if __name__ == "__main__":
    main()