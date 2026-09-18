.PHONY: setup test data clean_data features trinity simulate assessment assessment-check all legacy-all paper report calib

PYTHON ?= .venv/bin/python

setup:
	python3.14 -m venv .venv
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r assessment/requirements-lock.txt
	$(PYTHON) -m pip install -e . --no-deps

test:
	$(PYTHON) -m pytest

data:
	$(PYTHON) -m ai_ls_allocation.data.download

clean_data:
	$(PYTHON) -m ai_ls_allocation.data.clean

features:
	$(PYTHON) -m ai_ls_allocation.features.build

trinity:
	$(PYTHON) -m ai_ls_allocation.engine.bt_trinity

simulate:
	$(PYTHON) -m ai_ls_allocation.engine.run_simulation

assessment:
	$(MAKE) -C assessment all

assessment-check:
	$(MAKE) -C assessment check

legacy-all:
	$(PYTHON) -m ai_ls_allocation.data.download
	$(PYTHON) -m ai_ls_allocation.data.clean
	$(PYTHON) -m ai_ls_allocation.features.build
	$(PYTHON) -m ai_ls_allocation.engine.bt_trinity
	$(PYTHON) -m ai_ls_allocation.engine.run_simulation

all: test assessment paper

paper:
	$(PYTHON) assessment/scripts/build_submission.py
	cd assessment_run/writing && latexmk -pdf -interaction=nonstopmode -halt-on-error abgegebenes_PAPER.tex
	cp assessment_run/writing/abgegebenes_PAPER.pdf paper_text.pdf


report:
	$(PYTHON) -m ai_ls_allocation.engine.report

calib:
	$(PYTHON) -m ai_ls_allocation.engine.calibration
