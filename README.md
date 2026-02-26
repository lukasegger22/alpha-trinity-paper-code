# 🏦 Alpha Trinity: Robust Multi-Asset Allocation under Non-Stationary Regimes

**Status:** Submitted to *Procedia Computer Science* (Elsevier)

This repository contains the codebase and supplementary material for the paper *"Robust Multi-Asset Allocation under Non-Stationary Regimes: A Regime-Aware Ensemble Approach with Dynamic Crisis Control."*

## 📖 Overview

Financial markets exhibit non-stationary behavior, often rendering static trading strategies and pure "black-box" Deep Learning models ineffective during structural breaks (e.g., the 2022 inflation shock). 

**Alpha Trinity** is a decoupled algorithmic trading architecture designed to improve out-of-sample robustness and capital preservation. It separates signal generation from risk management by integrating two distinct layers:
1. **The Hybrid Ensemble:** A predictive model combining Gradient Boosting (XGBoost) and Ridge Regression to capture nonlinear and linear price trends across multiple time horizons.
2. **The Macro-Crisis Engine:** A deterministic regime filter that dynamically switches the portfolio between "Risk-On" and "Defense" modes based on VIX acceleration and trend breakdowns.



## 📊 Out-of-Sample Performance (Jan 2022 – Feb 2026)

The strategy was tested using a strict Expanding Window Walk-Forward Validation, entirely out-of-sample, including realistic transaction costs (10bps slippage) and dynamic leverage holding costs.

| Metric | S&P 500 (SPY) | Naive ML (No Rules) | Alpha Trinity (Final) |
| :--- | :--- | :--- | :--- |
| **Total Net Return** | +51.2% | +98.4% | **+169.3%** |
| **Sharpe Ratio** | 0.54 | 0.72 | **1.03** |
| **Max Drawdown** | -33.9% | -41.2% | **-22.5%** |
| **Cost Drag** | 0.0% | -18.4% | **-6.4%** |

*Note the "Cost Killer" effect: The execution logic enforces minimum holding periods and volatility-adjusted rebalancing, reducing estimated cost drag from >18% down to 6.4%.*



## 🌍 Asset Universe

The system trades a diversified universe of highly liquid ETFs and high-conviction mega-caps, explicitly excluding highly volatile cryptocurrencies to optimize risk-adjusted returns:
* **Equities & Beta:** SPY, QQQ, IWM
* **Sector Rotation:** XLF, XLI, XLV, XLP, XLU
* **International:** EFA, VWO, FXI
* **Growth / Alpha:** NVDA, MSFT, META, ASML
* **Real Assets:** GLD, XLE, URA, DBA, VNQ
* **Safe Havens / Hedges:** SHY, VCSH, GOVT, UUP, AGG, TLT, SVXY

## 🚀 Quickstart & Reproducibility

To run the pipeline locally and reproduce the data ingestion, feature engineering, and backtesting:

### 1. Installation
Ensure you have Python 3.11+ installed. Clone the repository and set up the environment:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -U pip
pip install -e .