PYTHON ?= python

.PHONY: setup download smoke features benchmark final cnn ablation plot web test lint

setup:
	$(PYTHON) -m pip install -e ".[dev]"

download:
	$(PYTHON) -m fpd.cli download

smoke:
	$(PYTHON) -m fpd.cli smoke --config configs/smoke.yaml

features:
	$(PYTHON) -m fpd.cli features --config configs/stage_a.yaml

benchmark:
	$(PYTHON) -m fpd.cli benchmark --config configs/stage_a.yaml
	$(PYTHON) -m fpd.cli benchmark --config configs/stage_b.yaml

final:
	$(PYTHON) -m fpd.cli final --config configs/final.yaml

cnn:
	$(PYTHON) -m fpd.cli cnn --config configs/cnn.yaml

ablation:
	$(PYTHON) -m fpd.cli ablation --config configs/ablation.yaml

plot:
	$(PYTHON) -m fpd.cli plot --config configs/final.yaml

web:
	$(PYTHON) -m http.server 8799 --directory web

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests
