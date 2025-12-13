import numpy as np
import pandas as pd
from scipy.optimize import minimize

def get_portfolio_weights(
    forecasts: pd.Series,       # Deine KI-Signale (z.B. MSFT: 0.05, GOOG: -0.01)
    returns_history: pd.DataFrame, # Die echten Returns der letzten 60 Tage (für Risiko-Berechnung)
    target_vol: float = 0.15,   # Wir zielen auf 15% Vola (typisch Aktienmarkt)
    max_weight: float = 0.20    # Keine Position darf größer als 20% sein (Diversifikations-Zwang)
) -> pd.Series:
    """
    Berechnet die optimalen Gewichte basierend auf Mean-Variance-Optimization.
    Ziel: Maximiere (Signal * Gewicht) - (Risiko-Penalty).
    """
    assets = forecasts.index
    n = len(assets)
    
    # 1. Kovarianz-Matrix berechnen (Wie hängen die Aktien zusammen?)
    # Wir nehmen nur die Historie der relevanten Assets
    recent_returns = returns_history[assets].iloc[-60:] # Letzte 3 Monate reichen für aktuelle Korrelation
    cov_matrix = recent_returns.cov().values
    
    # Die Signale sind unsere "Expected Returns" (Alphas)
    alphas = forecasts.values
    
    # Start-Gewichte (gleichverteilt)
    w0 = np.ones(n) / n
    
    # --- Die Mathematik (Solver) ---
    
    # Zielfunktion: Maximiere Portfolio-Score (Alpha - Risiko)
    # Da der Solver "minimiert", drehen wir das Vorzeichen von Alpha um.
    # Risk Aversion Lambda: Wie stark bestrafen wir Risiko?
    risk_aversion = 2.0 
    
    def objective(w):
        port_alpha = np.dot(w, alphas) # Wie viel KI-Signal fangen wir ein?
        port_risk = np.dot(w.T, np.dot(cov_matrix, w)) # Portfolio Varianz
        # Wir wollen Alpha maximieren (also -Alpha minimieren) und Risiko minimieren
        return -port_alpha + (risk_aversion * port_risk)
    
    # Constraints (Regeln für den Solver)
    cons = [
        {'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1.0} # Summe der Absolutwerte = 1 (100% Investiert)
    ]
    
    # Bounds (Grenzen pro Aktie)
    # Long/Short erlaubt: von -max_weight bis +max_weight
    bounds = [(-max_weight, max_weight) for _ in range(n)]
    
    # Optimierung starten
    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons)
    
    # Ergebnis zurückgeben
    return pd.Series(res.x, index=assets)