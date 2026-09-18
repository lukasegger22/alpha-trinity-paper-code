# Feature Documentation

Generated: 2026-09-17T14:56:23

The active assessment feature set contains 8 model inputs. All rolling features are trailing-window calculations, and final portfolio evaluation uses a one-trading-day execution lag.

## Feature Catalog

See `outputs/tables/feature_catalog.csv` for formulas, window lengths, purposes, leakage guards, and Random Forest feature importance.

## Redundancy Check

The training-period correlation matrix is stored in `outputs/tables/feature_correlation_matrix.csv`. Pairs with absolute correlation above 0.70 are stored in `outputs/tables/feature_high_correlation_pairs.csv`.

High-correlation pairs found: 2.

## VIX Handling

The assessment loader maps `vix_close` into `VIX` when the existing `VIX` column is missing or constant. This fixes the earlier constant-VIX issue for assessment reruns while keeping a clear audit trail in the reproducibility files.
