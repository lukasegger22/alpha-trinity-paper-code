# AI/ML Long/Short Allocation (MVP → Robustness)

Minimal reproducible scaffold for Lukas Egger's thesis project.
- Universe: liquid ETFs (SPY, QQQ, TLT, IEF, HYG, GLD, FXE, VNQ)
- Calendar: NYSE, daily close-to-close
- Costs & +1d lag defined in config
- Tests ensure no look-ahead and calendar sanity (to be added in M0)

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pytest


## 4) `pyproject.toml`
```bash
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "ai_ls_allocation"
version = "0.1.0"
description = "Long/Short allocation on ETFs using probabilistic ML. MVP scaffold."
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "pandas>=2.2",
  "numpy>=1.26",
  "scipy>=1.11",
  "pyyaml>=6.0",
  "matplotlib>=3.8",
  "statsmodels>=0.14",
  "pandas-market-calendars>=4.3",
  "yfinance>=0.2.40",
  "pytest>=8.0",
]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.pytest.ini_options]
addopts = "-q"
pythonpath = ["src"]


## Quickstart (M0)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
make all
```
Outputs: `reports/bt_dummy_timeseries.csv`, `reports/bt_dummy_weights.csv`.
