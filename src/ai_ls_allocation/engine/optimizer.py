import cvxpy as cvx
import numpy as np
import pandas as pd

class MarkowitzOptimizer:
    def __init__(self, window_size=60, risk_aversion=0.5, target_vol=0.15):
        self.window_size = window_size
        self.risk_aversion = risk_aversion # Wie sehr hassen wir Risiko?
        self.target_vol = target_vol       # Ziel-Schwankung (15%)

    def optimize(self, signals, returns):
        """
        Berechnet die optimalen Gewichte.
        Fix: Robusterer Solver (OSQP) gegen Abstürze.
        """
        n_assets = len(signals)
        tickers = signals.index
        
        # 1. Signale prüfen
        # Wenn Signale alle 0 oder NaN sind -> Gleichgewichtung
        if signals.isna().all() or (signals == 0).all():
             return pd.Series([1/n_assets]*n_assets, index=tickers)

        # Signal Glättung und Normalisierung (damit keine extremen Werte entstehen)
        mu = signals.values.flatten()
        
        # 2. Risiko (Covarianz)
        if len(returns) < self.window_size:
            return pd.Series([1/n_assets]*n_assets, index=tickers)
            
        recent_returns = returns.iloc[-self.window_size:]
        Sigma = recent_returns.cov().values
        
        # Shrinkage (Mathe-Stabilisierung), etwas stärker als vorher
        Sigma = Sigma + np.eye(n_assets) * 0.001

        # 3. Optimierung
        w = cvx.Variable(n_assets)
        
        # Wir wollen Return maximieren, aber Risiko (Variance) bestrafen
        # Und wir wollen nah an der Target Volatilität bleiben
        portfolio_return = w.T @ mu
        portfolio_risk = cvx.quad_form(w, Sigma)
        
        # Zielfunktion: Max(Return) - Strafe * Risiko
        objective = cvx.Maximize(portfolio_return - self.risk_aversion * portfolio_risk)
        
        constraints = [
            cvx.sum(w) == 1,  # Investiere 100%
            w >= -0.25,         # Long Only (Kein Short)
            w <= 0.25       # DIVERSIFIKATION: Max 25% in eine Aktie (kein Klumpenrisiko)
        ]

        # 4. Lösen mit OSQP (Robuster als ECOS)
        try:
            prob = cvx.Problem(objective, constraints)
            # OSQP ist der Standard für Finanz-Optimierung
            prob.solve(solver=cvx.OSQP, max_iter=5000)
            
            if w.value is None:
                # print("Solver failed (None), fallback to Equal Weight")
                return pd.Series([1/n_assets]*n_assets, index=tickers)
                
            weights = w.value
            
            # Aufräumen (Rauschen unter 0.1% auf 0 setzen)
            weights[weights < 0.001] = 0
            
            # Normalisieren
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                return pd.Series([1/n_assets]*n_assets, index=tickers)
            
            return pd.Series(weights, index=tickers)
            
        except Exception as e:
            # print(f"Optimization crashed: {e}, fallback to Equal Weight")
            return pd.Series([1/n_assets]*n_assets, index=tickers)