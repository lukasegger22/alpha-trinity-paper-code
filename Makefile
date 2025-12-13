.PHONY: setup test data clean_data features

setup:
	python3 -m venv .venv
	. .venv/bin/activate && python -m pip install -U pip && pip install -e .

test:
	. .venv/bin/activate && pytest

data:
	. .venv/bin/activate && python -m ai_ls_allocation.data.download

clean_data:
	. .venv/bin/activate && python -m ai_ls_allocation.data.clean

features:
	. .venv/bin/activate && python -m ai_ls_allocation.features.build


bt-dummy:
	. .venv/bin/activate && python -m ai_ls_allocation.engine.backtest


all:
	. .venv/bin/activate && python -m ai_ls_allocation.data.download
	. .venv/bin/activate && python -m ai_ls_allocation.data.clean
	. .venv/bin/activate && python -m ai_ls_allocation.features.build
	. .venv/bin/activate && python -m ai_ls_allocation.engine.backtest


report:
	. .venv/bin/activate && python -m ai_ls_allocation.engine.report

calib:
	. .venv/bin/activate && python -m ai_ls_allocation.engine.calibration

