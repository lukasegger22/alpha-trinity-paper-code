from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd

QuantileLevels = Tuple[float, float, float]


@dataclass
class DummyQuantileModel:
    """
    Rolling-quantile baseline auf 1-Tages-Logreturns.

    Für jedes Datum t benutzt das Modell die vorherigen `window` Tage von r1,
    um Q10/Q50/Q90 für den *nächsten* Return zu schätzen.
    """
    window: int = 60
    quantiles: QuantileLevels = (0.1, 0.5, 0.9)
    target_col: str = "r1"

    def fit(self, feats: pd.DataFrame) -> "DummyQuantileModel":
        # Dummy: nichts zu fitten, wir behalten nur das API-Schema
        if self.target_col not in feats.columns:
            raise KeyError(f"Expected column '{self.target_col}' in features")
        return self

    def predict(self, feats: pd.DataFrame) -> pd.DataFrame:
        """
        Nimmt ein Feature-DataFrame mit Spalte `target_col` = r1 und gibt
        ein DataFrame mit ['q10','q50','q90'] zurück.

        Wert an Tag t basiert nur auf Daten bis t-1.
        """
        if self.target_col not in feats.columns:
            raise KeyError(f"Expected column '{self.target_col}' in features")
        r1 = feats[self.target_col].astype(float)

        # Shift um 1: Fenster an Tag t sieht nur r1 bis t-1
        hist = r1.shift(1)

        q10, q50, q90 = self.quantiles

        q10_series = hist.rolling(self.window).quantile(q10)
        q50_series = hist.rolling(self.window).quantile(q50)
        q90_series = hist.rolling(self.window).quantile(q90)

        out = pd.concat(
            [
                q10_series.rename("q10"),
                q50_series.rename("q50"),
                q90_series.rename("q90"),
            ],
            axis=1,
        )
        return out


def rolling_quantile_baseline(
    r1: pd.Series,
    window: int = 60,
    quantiles: QuantileLevels = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    """
    Convenience-Funktion: nimmt eine r1-Series und gibt Rolling-Q10/Q50/Q90
    (nur basierend auf Vergangenheitsdaten) zurück.
    """
    r1 = pd.Series(r1).astype(float)
    feats = pd.DataFrame({"r1": r1})
    model = DummyQuantileModel(window=window, quantiles=quantiles)
    model.fit(feats)
    return model.predict(feats)


def predict_quantiles(feats: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper, damit bestehender Code weiter funktioniert:
    nutzt intern DummyQuantileModel mit Standardparametern.
    """
    model = DummyQuantileModel()
    return model.predict(feats)
