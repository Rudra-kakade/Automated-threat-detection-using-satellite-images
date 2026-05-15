# ═══════════════════════════════════════════════════════════════════════════════
# Makefile — Satellite Defence Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
# Usage: make <target>
#   make pipeline       — Run full pipeline
#   make test           — Run unit tests
#   make train          — Training only
#   make evaluate       — Evaluation only

PYTHON   ?= python
CONFIG   ?= pipeline_config.yaml
VENV_DIR ?= venv

.PHONY: help install test lint pipeline ingest augment train evaluate infer visualize export dry-run docker-build docker-run dvc-init dvc-push dvc-pull clean

help:  ## Show this help
	@echo ═══════════════════════════════════════════
	@echo   Satellite Defence Pipeline — Targets
	@echo ═══════════════════════════════════════════
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

# ─── Setup ───────────────────────────────────────────────────────────────────

install:  ## Install dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# ─── Testing & Linting ──────────────────────────────────────────────────────

test:  ## Run unit tests
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:  ## Validate Python imports and syntax
	$(PYTHON) -c "import src.data.tile_prep; import src.data.augmentation; import src.train.train; import src.eval.evaluate; import src.eval.dashboard; import src.eval.report; print('All imports OK')"
	$(PYTHON) -m py_compile pipeline.py

# ─── Pipeline Stages ────────────────────────────────────────────────────────

pipeline:  ## Run full pipeline (all stages)
	$(PYTHON) pipeline.py --stage all --config $(CONFIG)

dry-run:  ## Validate pipeline without GPU
	$(PYTHON) pipeline.py --stage all --config $(CONFIG) --dry-run

ingest:  ## Data ingestion — tile satellite images
	$(PYTHON) pipeline.py --stage ingest --config $(CONFIG)

augment:  ## Data augmentation
	$(PYTHON) pipeline.py --stage augment --config $(CONFIG)

train:  ## Model training
	$(PYTHON) pipeline.py --stage train --config $(CONFIG)

evaluate:  ## Model evaluation + quality gate
	$(PYTHON) pipeline.py --stage evaluate --config $(CONFIG)

infer:  ## SAHI inference on test images
	$(PYTHON) pipeline.py --stage infer --config $(CONFIG)

visualize:  ## Generate dashboard + overlays
	$(PYTHON) pipeline.py --stage visualize --config $(CONFIG)

export:  ## Export model to ONNX
	$(PYTHON) pipeline.py --stage export --config $(CONFIG)

# ─── Docker ──────────────────────────────────────────────────────────────────

docker-build:  ## Build Docker image
	docker build -t satellite-defence:latest .

docker-run:  ## Run pipeline in Docker
	docker run --gpus all -v $(PWD)/data:/app/data -v $(PWD)/runs:/app/runs -v $(PWD)/eval_results:/app/eval_results satellite-defence:latest --stage all

# ─── DVC ─────────────────────────────────────────────────────────────────────

dvc-init:  ## Initialize DVC
	dvc init
	dvc remote add -d local_store .dvc_storage
	@echo "DVC initialized. Configure remote storage as needed."

dvc-push:  ## Push data/models to DVC remote
	dvc push

dvc-pull:  ## Pull data/models from DVC remote
	dvc pull

# ─── Cleanup ─────────────────────────────────────────────────────────────────

clean:  ## Remove generated artifacts
	rm -rf __pycache__ .pytest_cache
	rm -rf eval_results/pipeline_run_*.json eval_results/pipeline_report*.md
	rm -rf inference_results/ vis_output/
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
