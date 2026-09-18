# Reproducibility Protocol

Generated: 2026-09-17T14:56:05

## Data Source

The implemented downloader uses `yfinance.download` with `auto_adjust=True` for the configured asset universe and separately downloads `^VIX`. The raw downloaded panel is saved to `data/panel.parquet`; engineered features are saved to `data/features/panel.parquet`.

## Local Data Snapshot

- Feature panel rows: 29337
- Feature panel symbols: 21
- Feature panel date range: 2020-10-15 to 2026-05-08
- Assessment backtest range: 2022-01-03 to 2026-05-08
- Assessment trading days: 1091
- Frozen input: `assessment/inputs/panel.parquet`; saved strategy weights are not inputs.
- Original download time is unknown. File modification times are NOT download dates.
- Snapshot frozen on 2026-09-17; see `assessment/inputs/provenance.json` for the checksum.

## Adjusted Prices

The downloader calls yfinance with `auto_adjust=True`. This means yfinance returns prices adjusted for splits and dividends in the OHLC fields. The feature builder uses the adjusted `Close` field to compute returns.

## Calendar Synchronization

The active feature panel has an equal number of rows for every symbol. This indicates that the downloaded/feature-built data was aligned to a shared daily date grid before assessment evaluation. The project also contains `src/ai_ls_allocation/data/trading_calendar.py`, which provides a NYSE trading-day helper.

## Missing Values

Current feature-panel missing values after the feature build:

```text
Date                    0
symbol                  0
Open                    0
High                    0
Low                     0
Close                   0
Volume                  0
vix_close               0
returns_1d              0
returns_5d              0
returns_20d             0
volatility_60d          0
sma_200                 0
rsi                     0
dist_sma200             0
TNX_Level               0
TNX_Chg_10d             0
VIX                     0
Interaction_TNX_Vola    0
SP500_Trend             0
```

Per-symbol coverage:

```text
            first       last  rows
symbol                            
AGG    2020-10-15 2026-05-08  1397
ASML   2020-10-15 2026-05-08  1397
DBA    2020-10-15 2026-05-08  1397
EFA    2020-10-15 2026-05-08  1397
GLD    2020-10-15 2026-05-08  1397
IWM    2020-10-15 2026-05-08  1397
META   2020-10-15 2026-05-08  1397
MSFT   2020-10-15 2026-05-08  1397
NVDA   2020-10-15 2026-05-08  1397
QQQ    2020-10-15 2026-05-08  1397
SPY    2020-10-15 2026-05-08  1397
TLT    2020-10-15 2026-05-08  1397
UUP    2020-10-15 2026-05-08  1397
VNQ    2020-10-15 2026-05-08  1397
VWO    2020-10-15 2026-05-08  1397
XLE    2020-10-15 2026-05-08  1397
XLF    2020-10-15 2026-05-08  1397
XLI    2020-10-15 2026-05-08  1397
XLP    2020-10-15 2026-05-08  1397
XLU    2020-10-15 2026-05-08  1397
XLV    2020-10-15 2026-05-08  1397
```

## Model and Random Seeds

- Prediction target: 20-day forward return per asset.
- Features: returns_1d, returns_5d, volatility_60d, VIX, obv_trend, dist_vwap, mfi, bb_pos.
- Normalization: `StandardScaler` fitted only on rows before 2021-12-31 whose forward labels end by that cutoff; then applied to later rows. Walk-forward runs refit the scaler and models at each annual cutoff.
- Ensemble: 10 models.
- MLP models: hidden layers `(64, 32)`, `max_iter=200`, random seeds 42-46.
- Random Forest models: `n_estimators=30`, `max_depth=8`, `n_jobs=-1`, random seeds 47-51.
- Global assessment constants: transaction cost 0.0010, annual leverage carry 6.00%, five-session exit guard (partial reductions allowed; panic and hard bounds override it).

## Backtest Timing

The defensible assessment rule is:

1. features and target weights are computed at close of day t,
2. target weights are saved with timestamp t,
3. the order fills at close t+1, after existing holdings earn that session's return,
4. actual trades against drifted holdings and fees are charged at that fill,
5. the new position first earns the close t+1 to close t+2 return.

This is the rule used by `assessment/scripts/run_first_three.py` and `assessment/scripts/run_robustness_repro_audit.py`.

## Artifact Inventory

See `assessment/outputs/tables/reproducibility_inventory.csv` for local file sizes, modification timestamps, and SHA-256 hashes. These timestamps are not historical acquisition dates.
