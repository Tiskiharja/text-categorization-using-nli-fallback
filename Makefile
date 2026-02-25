.PHONY: help install install-dev test download-data train train-quick export-nli-onnx eval api classify health categories demo clean

UV ?= uv
PYTHON ?= python3
HOST ?= 127.0.0.1
PORT ?= 8000
BASE_URL ?= http://$(HOST):$(PORT)
CONFIG_FILE ?= config.yaml
REUTERS_DIR ?= reuters+21578+text+categorization+collection
REUTERS_ARCHIVE ?= $(REUTERS_DIR)/reuters21578.tar.gz
REUTERS_URL ?= https://archive.ics.uci.edu/static/public/137/reuters+21578+text+categorization+collection.zip

help:
	@echo "Available targets:"
	@echo "  make install      - install project dependencies with uv"
	@echo "  make install-dev  - install project + dev dependencies (tests)"
	@echo "  make test         - run pytest with coverage"
	@echo "  make download-data - download Reuters-21578 archive to expected project path"
	@echo "  make train        - train DistilBERT and export ONNX"
	@echo "  make train-quick  - fast training run for local smoke checks"
	@echo "  make export-nli-onnx - export NLI fallback model to ONNX"
	@echo "  make eval         - run hybrid threshold/weight evaluation"
	@echo "  make api          - start FastAPI server (loads APP_CONFIG_FILE, default: config.yaml)"
	@echo "  make health       - call /v1/health"
	@echo "  make categories   - call /v1/categories"
	@echo "  make classify     - POST examples/request.json to /v1/classify"
	@echo "  make demo         - run health, categories, classify in sequence"
	@echo "  make clean        - remove generated model/output artifacts"

install:
	$(UV) sync

install-dev:
	$(UV) sync --group dev

test:
	$(UV) run --group dev pytest -q --cov=. --cov-report=term-missing --cov-fail-under=35

download-data:
	mkdir -p "$(REUTERS_DIR)"
	curl -fL "$(REUTERS_URL)" -o "$(REUTERS_DIR)/reuters21578.zip"
	unzip -o "$(REUTERS_DIR)/reuters21578.zip" -d "$(REUTERS_DIR)"
	test -f "$(REUTERS_ARCHIVE)"
	@echo "Reuters archive ready at $(REUTERS_ARCHIVE)"

train:
	$(UV) run $(PYTHON) train.py --export-onnx --export-nli-onnx

train-quick:
	$(UV) run $(PYTHON) train.py --max-steps 50 --export-onnx --export-nli-onnx

export-nli-onnx:
	$(UV) run $(PYTHON) train.py --export-nli-onnx-only

eval:
	$(UV) run $(PYTHON) evaluate_hybrid.py --max-docs 800 --holdout-k 8 --min-support 20

api:
	APP_CONFIG_FILE="$(CONFIG_FILE)" $(UV) run uvicorn api:app --host $(HOST) --port $(PORT) --reload

health:
	curl -sS $(BASE_URL)/v1/health | $(PYTHON) -m json.tool

categories:
	curl -sS $(BASE_URL)/v1/categories | $(PYTHON) -m json.tool

classify:
	curl -sS -X POST "$(BASE_URL)/v1/classify" \
		-H "Content-Type: application/json" \
		--data @examples/request.json | $(PYTHON) -m json.tool

demo: health categories classify

clean:
	rm -rf onnx_model onnx_nli_model output __pycache__ .pytest_cache
