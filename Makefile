.PHONY: setup lab test lint format train api mlflow clean

setup:            ## one-shot environment setup (macOS)
	./setup.sh

lab:              ## open JupyterLab
	uv run jupyter lab --notebook-dir=notebooks

test:             ## offline tests on synthetic data
	MOTOR_PRICING_SYNTHETIC=1 uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

train:            ## train + save models/pricing_artifact.joblib (needs data download on first run)
	uv run python -m motor_pricing.train

api:              ## serve the pricing API (run `make train` or notebook 03 first)
	uv run uvicorn motor_pricing.api:app --reload

mlflow:           ## experiment tracking UI
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:
	rm -rf .pytest_cache .ruff_cache mlruns
	find . -name __pycache__ -prune -exec rm -rf {} +
