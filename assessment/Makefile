.PHONY: all first-three robustness completion submission check

PYTHON ?= ../.venv/bin/python

all: first-three robustness completion submission

submission:
	$(PYTHON) scripts/build_submission.py

first-three:
	$(PYTHON) scripts/run_first_three.py --project-root .. --output-dir assessment

robustness:
	$(PYTHON) scripts/run_robustness_repro_audit.py

completion:
	$(PYTHON) scripts/run_completion.py

check:
	$(PYTHON) -m py_compile \
		scripts/run_first_three.py \
		scripts/run_robustness_repro_audit.py \
		scripts/run_completion.py \
		src/alpha_trinity_assessment/first_three.py \
		src/alpha_trinity_assessment/robustness_repro_audit.py \
		src/alpha_trinity_assessment/completion.py
