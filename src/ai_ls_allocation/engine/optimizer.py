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
        
        # 1. check Signals 

        if signals.isna().all() or (signals == 0).all():
             return pd.Series([1/n_assets]*n_assets, index=tickers)

        mu = signals.values.flatten()
        
        # 2. Risiko )
        if len(returns) < self.window_size:
            return pd.Series([1/n_assets]*n_assets, index=tickers)
            
        recent_returns = returns.iloc[-self.window_size:]
        Sigma = recent_returns.cov().values
        
        # Shrinkage 
        Sigma = Sigma + np.eye(n_assets) * 0.001

        # 3. Optimisation
        w = cvx.Variable(n_assets)
        

        portfolio_return = w.T @ mu
        portfolio_risk = cvx.quad_form(w, Sigma)

        objective = cvx.Maximize(portfolio_return - self.risk_aversion * portfolio_risk)
        
        constraints = [
            cvx.sum(w) == 1,  
            w >= -0.25,         # Long Only (no Short)
            w <= 0.25       
        ]

        # 4. solve OSQP 
        try:
            prob = cvx.Problem(objective, constraints)
            prob.solve(solver=cvx.OSQP, max_iter=5000)
            
            if w.value is None:
                return pd.Series([1/n_assets]*n_assets, index=tickers)
                
            weights = w.value
            
            weights[weights < 0.001] = 0
            
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                return pd.Series([1/n_assets]*n_assets, index=tickers)
            
            return pd.Series(weights, index=tickers)
            
        except Exception as e:
            return pd.Series([1/n_assets]*n_assets, index=tickers)