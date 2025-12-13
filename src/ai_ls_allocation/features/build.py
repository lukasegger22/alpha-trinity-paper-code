from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml

DATA_DIR = Path("data")
CLEAN_DIR = DATA_DIR / "clean"
FEAT_DIR = DATA_DIR / "features"
CONFIG_DIR = Path("config")


def load_universe() -> Tuple[List[str], pd.Timestamp, pd.Timestamp]:
    with open(CONFIG_DIR / "universes.yaml") as f:
        cfg = yaml.safe_load(f)

    etf_cfg = cfg["etf"]
    symbols = etf_cfg["symbols"]
    start = pd.Timestamp(etf_cfg.get("start", "2007-01-01"))
    end_raw = etf_cfg.get("end", "today")
    
    if isinstance(end_raw, str) and end_raw.lower() == "today":
        end = pd.Timestamp.today().normalize()
    else:
        end = pd.Timestamp(end_raw)

    return symbols, start, end


def load_close(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    fp = CLEAN_DIR / f"{symbol}.parquet"
    # Fallback, falls Datei leer oder nicht existent
    if not fp.exists():
        return pd.Series(dtype=float)
        
    df = pd.read_parquet(fp)
    
    if "close" in df.columns:
        s = df["close"]
    else:
        s = df.squeeze()

    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]

    s = pd.to_numeric(s, errors="coerce").astype(float)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    return s.loc[start:end]


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def build_features_for_symbol(
    symbol: str,
    close: pd.Series,
    spread_hyg_ief: Optional[pd.Series] = None,
) -> pd.DataFrame:
    close = close.astype(float)
    
    # 1. Returns (Stationär: OK)
    r1 = np.log(close).diff().rename("r1")
    
    # 2. Vola (Stationär: OK)
    vol20 = r1.rolling(20).std().rename("vol20")
    vol60 = r1.rolling(60).std().rename("vol60")
    
    # 3. Trends (ABSOLUTE PREISE: BÖSE!) -> Wir machen relative Abstände daraus
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    
    # Neu: Distanz zum MA in Prozent (Log-Space)
    # Wenn dist > 0, ist Preis über MA.
    dist_ma20 = np.log(close / ma20).rename("dist_ma20")
    dist_ma60 = np.log(close / ma60).rename("dist_ma60")
    
    # Neu: Trend-Stärke relativ zum Preis
    # (ma20 - ma60) / close
    trend_strength = ((ma20 - ma60) / close).rename("trend_strength")
    
    # 4. MACD (Ist oft absoluter Preis-Differenz, besser normalisieren)
    # Wir teilen durch Close, um es prozentual zu machen
    macd_line, macd_signal, macd_hist = macd(close)
    macd_norm = (macd_line / close).rename("macd")
    macd_sig_norm = (macd_signal / close).rename("macd_signal")
    macd_hist_norm = (macd_hist / close).rename("macd_hist")

    # Zusammenbauen
    feats = pd.concat(
        [r1, vol20, vol60, dist_ma20, dist_ma60, trend_strength, macd_norm, macd_sig_norm, macd_hist_norm],
        axis=1,
    )

    # Spread Feature joinen
    if spread_hyg_ief is not None:
        feats["spread_hyg_ief"] = spread_hyg_ief.reindex(feats.index).fillna(0.0)
    else:
        feats["spread_hyg_ief"] = 0.0

    # Target Shift (t+1)
    feats["target_r1"] = r1.shift(-1)

    return feats


def main():
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    symbols, start, end = load_universe()
    print(f"[features] universe: {symbols}")

    # HYG-IEF-Spread global berechnen
    spread_hyg_ief = None
    if "HYG" in symbols and "IEF" in symbols:
        try:
            hyg_close = load_close("HYG", start, end)
            ief_close = load_close("IEF", start, end)
            # Nur berechnen wenn Datenlänge ausreicht
            if len(hyg_close) > 10 and len(ief_close) > 10:
                hyg_r1 = np.log(hyg_close).diff()
                ief_r1 = np.log(ief_close).diff()
                spread_hyg_ief = (hyg_r1 - ief_r1).rename("spread_hyg_ief")
                print("[features] Calculated HYG-IEF spread successfully.")
        except Exception as e:
            print(f"[features] Warning: Could not calc spread: {e}")

    frames = []
    for s in symbols:
        close = load_close(s, start, end)
        if close.empty:
            print(f"[features] SKIP {s} (no data)")
            continue
            
        feats = build_features_for_symbol(s, close, spread_hyg_ief)
        feats.to_parquet(FEAT_DIR / f"{s}.parquet")
        
        # Für Panel vorbereiten
        df = feats.copy()
        idx_name = df.index.name or "date"
        df = df.reset_index().rename(columns={idx_name: "date"})
        df["symbol"] = s
        frames.append(df)

    if not frames:
        print("[features] No data frames built!")
        return

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.set_index(["date", "symbol"]).sort_index()

    panel_fp = FEAT_DIR / "panel.parquet"
    panel.to_parquet(panel_fp)
    print(f"[features] panel -> {panel_fp}")

if __name__ == "__main__":
    main()