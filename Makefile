# Convenience targets. Use the project's Python (override with PY=...).
PY ?= python

.PHONY: help setup data test lint fmt bench benchmarks clean

help:
	@echo "setup       install dependencies"
	@echo "data        download the open datasets into ./data"
	@echo "test        run the unit test suite (no downloads)"
	@echo "lint        ruff check + folderinfo lint"
	@echo "fmt         ruff format + import sort"
	@echo "bench       run the core baselines (classification + quantification)"
	@echo "benchmarks  run every benchmark incl. domain-shift / open-set / transfer"

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/download_data.py

test:
	$(PY) -m pytest -q

lint:
	-ruff check src scripts tests
	bash scripts/folderinfo.sh

fmt:
	ruff format src scripts tests
	ruff check --select I --fix src scripts tests

bench:
	$(PY) scripts/run_classification.py
	$(PY) scripts/run_quantification.py

benchmarks: bench
	$(PY) scripts/run_domain_shift.py
	$(PY) scripts/run_sota_classification.py
	$(PY) scripts/run_openset.py
	$(PY) scripts/run_calibration_transfer.py
	$(PY) scripts/run_pca_explore.py
	$(PY) scripts/run_interpretability.py
	$(PY) scripts/run_dimreduction.py

.PHONY: figures
figures:
	$(PY) scripts/infographic.py
	$(PY) scripts/plot_comparison.py

clean:
	$(PY) -c "import shutil,glob,os; [shutil.rmtree(p,ignore_errors=True) for p in glob.glob('**/__pycache__',recursive=True)+['.pytest_cache']]"
