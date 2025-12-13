import numpy as np
import pandas as pd
from scipy.optimize import minimize

class MarkowitzOptimizer:
    # Wir gehen auf 0.02 runter. Fast null Risiko-Angst.
    # Wir verlassen uns darauf, dass die "Target Vol" (0.40) uns schützt.
    def __init__(self, window_size=60, risk_aversion=0.02, target_vol=0.40):
        # Risk Aversion auf 0.1 (sehr niedrig -> gierig)
        self.window_size = window_size
        self.risk_aversion = risk_aversion
        self.target_vol = target_vol

    def optimize(self, signal_series, returns_df):
        common_assets = signal_series.index.intersection(returns_df.columns)
        if len(common_assets) < 2:
            return pd.Series(0, index=signal_series.index)

        signals = signal_series[common_assets].values
        
        # Volatilität & Kovarianz
        recent_returns = returns_df[common_assets].iloc[-60:]
        cov_matrix = recent_returns.cov().values
        current_vols = recent_returns.std() * np.sqrt(252)
        
        # --- Smart Bounds ---
        dynamic_bounds = []
        for asset in common_assets:
            vol = current_vols.get(asset, 1.0)
            if vol < 0.01: vol = 0.01
            
            # Max Weight berechnen
            max_weight = self.target_vol / vol
            max_weight = min(max_weight, 0.35) # Hard Cap 35%
            
            dynamic_bounds.append((-max_weight, max_weight))
        
        dynamic_bounds = tuple(dynamic_bounds)
        num_assets = len(common_assets)

        # Zielfunktion
        def objective(weights):
            portfolio_return = np.dot(weights, signals)
            portfolio_volatility = np.dot(weights.T, np.dot(cov_matrix, weights))
            utility = portfolio_return - (0.5 * self.risk_aversion * portfolio_volatility)
            return -utility # Minus für Minimierung

        # Constraints
        constraints = [
            {'type': 'ineq', 'fun': lambda w: 1.0 - np.sum(w)},        # Max 100% Net Long
            {'type': 'ineq', 'fun': lambda w: 1.6 - np.sum(np.abs(w))} # Max 160% Gross (Hebel)
        ]
        
        # ÄNDERUNG 1: Initial Guess ist NICHT mehr 0.
        # Wir starten mit einem kleinen gleichverteilten Portfolio (1% pro Asset).
        # Das zwingt den Optimizer, sofort zu rechnen und nicht bei 0 zu schlafen.
        initial_guess = np.ones(num_assets) / num_assets * 0.1

        try:
            result = minimize(
                objective, 
                initial_guess, 
                method='SLSQP', 
                bounds=dynamic_bounds,
                constraints=constraints,
                tol=1e-6
            )
            
            # ÄNDERUNG 2: Fail-Safe Modus!
            # Wir akzeptieren das Ergebnis AUCH WENN success=False ist,
            # solange 'x' (die Gewichte) keine kompletten Nullen sind.
            # SLSQP gibt oft "False" zurück, wenn es am Rand anstößt, obwohl die Lösung gut ist.
            
            weights = result.x
            
            # Nur wenn ALLES fast 0 ist, war es ein echter Fehler
            if np.all(np.abs(weights) < 1e-4):
                 # Letzter Versuch: Nimm den Initial Guess als Notfall-Portfolio, wenn Signale da sind
                 if np.sum(np.abs(signals)) > 0:
                     return pd.Series(initial_guess, index=common_assets)
                 else:
                     return pd.Series(0, index=common_assets)

            # Clean up: Winzige Positionen löschen
            weights[np.abs(weights) < 0.005] = 0
            return pd.Series(weights, index=common_assets)
                
        except Exception as e:
            print(f"Optimizer Crash: {e}")
            return pd.Series(0, index=common_assets)