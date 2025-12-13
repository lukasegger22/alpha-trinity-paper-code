import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Config
REPORTS_DIR = Path("reports")
FILE_W = REPORTS_DIR / "bt_neural_weights.csv"
OUT_IMG = REPORTS_DIR / "positions_heatmap.png"

def main():
    if not FILE_W.exists():
        print("Weights file not found.")
        return

    # Daten laden
    # Robustes Laden: Nimm immer die erste Spalte als Index (egal wie sie heißt)
    w = pd.read_csv(FILE_W, index_col=0, parse_dates=True)
    
    # Nur die relevanten Assets filtern (die, die jemals > 5% Position hatten)
    # Das macht den Chart lesbar
    relevant = w.columns[w.abs().max() > 0.05]
    w_clean = w[relevant]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Wir nutzen einen "Stackplot" für Long-Positionen, um die Allokation zu zeigen
    # (Shorts sind im Stackplot schwer darstellbar, daher plotten wir hier die Netto-Positionen als Linien oder Area)
    
    # Besser: Eine Heatmap oder "Area Chart" der Top 5
    # Wir nehmen die Top 7 Assets nach absoluter Summe der Gewichte
    top_assets = w.abs().sum().sort_values(ascending=False).head(7).index
    w_top = w[top_assets]

    # Plotting Lines mit Fill
    for col in w_top.columns:
        ax.plot(w_top.index, w_top[col], label=col, linewidth=1.5, alpha=0.9)
        # Optional: Fill
        # ax.fill_between(w_top.index, w_top[col], alpha=0.1)

    ax.set_title("Active Positioning: Top 7 Assets over Time", fontsize=16, fontweight="bold")
    ax.set_ylabel("Weight (-1.0 = Short, +1.0 = Long)")
    ax.set_xlabel("Date")
    
    # Nulllinie
    ax.axhline(0, color="black", linewidth=1, linestyle="-")
    
    # Gitter und Legende
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    
    # Formatter für Datum
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(OUT_IMG, dpi=300)
    print(f"[plot] Saved positions chart to {OUT_IMG}")

if __name__ == "__main__":
    main()