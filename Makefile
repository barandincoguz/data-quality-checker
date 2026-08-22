.PHONY: help install install-dev test test-cov lint format sample-data serve-demo clean build

PYTHON ?= ./.venv/bin/python

help:
	@echo "data-quality-checker development targets:"
	@echo "  install        Install core package in editable mode"
	@echo "  install-dev    Install package with dev & test dependencies"
	@echo "  test           Run all unit and integration tests"
	@echo "  test-cov       Run tests with coverage reporting"
	@echo "  lint           Run linter (ruff check)"
	@echo "  format         Format code with ruff"
	@echo "  sample-data    Generate sample mock annotation & document pool ZIPs"
	@echo "  serve-demo     Run HITL Web UI with sample demo batch"
	@echo "  clean          Remove build, cache, and temporary test artifacts"
	@echo "  build          Build sdist and wheel packages"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,test,compute]"

test:
	$(PYTHON) -m pytest -q

test-cov:
	$(PYTHON) -m pytest --cov=data_quality_checker --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check src/ tests/

format:
	$(PYTHON) -m ruff format src/ tests/

sample-data:
	$(PYTHON) sample_data/generate_sample_zips.py

serve-demo: sample-data
	@echo "Preparing sample batch..."
	$(PYTHON) -m data_quality_checker --config configs/presets/sample_data.json prepare \
		--annotation-zip sample_data/mock_annotations.zip \
		--document-pool-zip sample_data/mock_documents.zip \
		--batch-id demo_batch_001 \
		--hmac-key-file sample_data/sample_hmac.key
	@echo "Running fake-backend processing..."
	$(PYTHON) -m data_quality_checker --config configs/presets/sample_data.json process \
		--prepared-batch demo_batch_001 --fake-backend
	@echo "Starting HITL web interface on http://127.0.0.1:5055..."
	$(PYTHON) -m data_quality_checker --config configs/presets/sample_data.json serve \
		--batch-id demo_batch_001 --port 5055

clean:
	rm -rf build/ dist/ src/*.egg-info .pytest_cache/ .ruff_cache/ .coverage htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	$(PYTHON) -m build
