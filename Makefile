# Common commands for my tea harvest MLOps project (run from the repo root).
.PHONY: help lint format test doctor dvc-auth experiments repro serve docker-build docker-run app app-docker notebook clean

DAGSHUB_REPO ?= $(or $(DAGSHUB_REPO_ENV),tea-harvest-mlops)
IMAGE ?= tea-harvest:local

help:
	@echo "make lint | format | test | doctor | dvc-auth | experiments | repro | serve | docker-build | docker-run | app | notebook"

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest -q tests

doctor:
	@echo "=== Checking tools ==="
	@uv --version || echo "uv missing"
	@uv run python --version
	@uv run dvc --version || echo "dvc missing"
	@docker --version || echo "docker missing (only needed for docker-build)"
	@az version --query '"azure-cli"' -o tsv 2>/dev/null || echo "azure-cli not installed (only needed for Azure)"
	@echo "=== Checking DagsHub auth ==="
	@test -n "$$DAGSHUB_OWNER" || (echo "DAGSHUB_OWNER is not set" && exit 1)
	@test -n "$$DAGSHUB_TOKEN" || (echo "DAGSHUB_TOKEN is not set" && exit 1)
	@uv run python -c "from src import tracking; import mlflow; tracking.configure(); mlflow.search_experiments(max_results=1); print('dagshub OK')"
	@echo "=== All checks passed ==="

# Stores my DagsHub login in .dvc/config.local (never committed) so dvc push/pull work.
dvc-auth:
	uv run dvc remote modify origin --local url "https://dagshub.com/$$DAGSHUB_OWNER/$${DAGSHUB_REPO:-tea-harvest-mlops}.dvc"
	uv run dvc remote modify origin --local auth basic
	uv run dvc remote modify origin --local user "$$DAGSHUB_OWNER"
	uv run dvc remote modify origin --local password "$$DAGSHUB_TOKEN"

experiments:
	uv run python -m src.experiments

repro:
	uv run dvc repro

serve:
	MODEL_VERSION=dev uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 -e MODEL_VERSION=local $(IMAGE)

app:
	uv run streamlit run frontend/streamlit_app.py

app-docker:
	docker build -f Dockerfile.frontend -t tea-harvest-frontend:local . && \
	docker run --rm -p 8501:8501 tea-harvest-frontend:local

notebook:
	cd notebooks && uv run jupyter nbconvert --to notebook --execute --inplace \
	  --ExecutePreprocessor.timeout=3600 Tea_Harvest_Readiness_EBM.ipynb

clean:
	rm -rf .pytest_cache .ruff_cache mlruns
