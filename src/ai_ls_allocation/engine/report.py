import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def save_signals(weights_series, output_path):
    """
    Speichert die finalen Gewichte für den Execution-Bot.
    :param weights_series: Pandas Series mit Symbol als Index und Gewicht als Value.
    :param output_path: Pfad, wo die Datei hin soll.
    """
    # Sicherstellen, dass der Ordner existiert
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # In DataFrame wandeln
    df = weights_series.to_frame(name='weight')
    
    # Datum hinzufügen (heute)
    df['created_at'] = pd.Timestamp.now()
    
    # Speichern (Parquet ist schneller und sicherer als CSV)
    df.to_parquet(output_path)
    
    # Optional: Auch als CSV für Menschen lesbar speichern
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path)
    
    print(f"    [Report] Signals saved to {csv_path}")

def plot_performance(portfolio_cum_returns, benchmark_cum_returns=None, save_path=None):
    """
    Erstellt eine schnelle Grafik der Performance.
    """
    plt.figure(figsize=(12, 6))
    portfolio_cum_returns.plot(label='Trinity Strategy', color='blue', linewidth=2)
    
    if benchmark_cum_returns is not None:
        benchmark_cum_returns.plot(label='Benchmark', color='gray', linestyle='--', alpha=0.7)
        
    plt.title('Trinity Strategy Performance')
    plt.ylabel('Cumulative Return')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"    [Report] Chart saved to {save_path}")
    else:
        plt.show()
    plt.close()