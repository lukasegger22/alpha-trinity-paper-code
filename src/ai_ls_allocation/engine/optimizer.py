import numpy as np
import pandas as pd
from scipy.optimize import minimize

class MarkowitzOptimizer:
    def __init__(self, window_size=60, risk_aversion=2.0):
        """
        Initialisiert den Optimizer.
        :param window_size: Wie viele Tage zurück schauen wir für die Kovarianz?
        :param risk_aversion: Wie stark bestrafen wir Risiko? (Höher = Vorsichtiger)
        """
        self.window_size = window_size
        self.risk_aversion = risk_aversion

    def optimize(self, signal_series, returns_df):
        """
        Berechnet die optimalen Gewichte.
        :param signal_series: Die Vorhersagen für morgen (Pandas Series)
        :param returns_df: Die historischen Returns für die Kovarianz (Pandas DataFrame)
        :return: Pandas Series mit Gewichten (z.B. AAPL: 0.05, MSFT: -0.05)
        """
        # 1. Datenvorbereitung
        # Wir nehmen nur Assets, für die wir beides haben (Signal UND Returns)
        common_assets = signal_series.index.intersection(returns_df.columns)
        
        if len(common_assets) < 2:
            # Falls wir zu wenig Assets haben, machen wir nichts (0 Gewichte)
            return pd.Series(0, index=signal_series.index)

        signals = signal_series[common_assets].values
        # Kovarianzmatrix (Risiko-Landkarte) berechnen
        cov_matrix = returns_df[common_assets].cov().values

        num_assets = len(common_assets)
        
        # 2. Die Zielfunktion (Was wollen wir?)
        # Wir wollen: (Rendite * Signal) maximieren UND (Risiko * Aversion) minimieren.
        # Da der Computer nur minimieren kann, drehen wir das Vorzeichen bei Rendite um.
        def objective(weights):
            portfolio_return = np.dot(weights, signals)
            portfolio_volatility = np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Utility = Return - (0.5 * Risk_Aversion * Variance)
            utility = portfolio_return - (0.5 * self.risk_aversion * portfolio_volatility)
            return -utility # Minus, weil wir minimieren

        # 3. Nebenbedingungen (Regeln)
        # ... (innerhalb von optimize Methode)

        # 4. Grenzen (Bounds) lockern!
        # Wir erlauben bis zu 40% in einem Asset (statt 20%).
        # Das gibt ihm die Chance, starke Trends (wie NVDA) auch zu reiten.
        bounds = tuple((-0.4, 0.4) for _ in range(num_assets))
        
        # 3. Nebenbedingungen
        constraints = [
            # WICHTIG: Wir erlauben ihm, NICHT voll investiert zu sein.
            # Statt "Summe MUSS 1 sein", sagen wir: "Summe <= 1".
            # Der Rest ist automatisch Cash (Risikofrei).
            {'type': 'ineq', 'fun': lambda w: 1.0 - np.sum(w)},  # Summe <= 1.0
            
            # Wir wollen aber mindestens 50% investiert sein (damit er nicht nur schläft)
            {'type': 'ineq', 'fun': lambda w: np.sum(w) - 0.5},  # Summe >= 0.5
            
            # Gross Exposure Beschränkung (kein extremer Hebel)
            {'type': 'ineq', 'fun': lambda w: 1.6 - np.sum(np.abs(w))}
        ]

        # 4. Grenzen (Bounds)
        # Kein Asset darf mehr als 20% des Portfolios ausmachen (Diversifikation!)
        bounds = tuple((-0.2, 0.2) for _ in range(num_assets))
        
        # 5. Startwert (Wir starten mit Gleichverteilung)
        initial_guess = np.zeros(num_assets)

        # 6. Optimierung starten (SLSQP ist ein Standard-Solver für solche Probleme)
        try:
            result = minimize(
                objective, 
                initial_guess, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                tol=1e-6
            )
            
            if result.success:
                return pd.Series(result.x, index=common_assets)
            else:
                # Fallback: Wenn Mathe versagt, alles 0
                return pd.Series(0, index=common_assets)
                
        except Exception as e:
            # Notfall-Catch
            return pd.Series(0, index=common_assets)