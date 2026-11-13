"""Project-wide paths and constants. Everything can be overridden with env vars."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("MOTOR_PRICING_DATA_DIR", ROOT / "data"))
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(os.environ.get("MOTOR_PRICING_MODELS_DIR", ROOT / "models"))
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT = "motor-pricing"

SEED = 42
N_JOBS = 4  # sensible for an 8-core M2 that must also run a browser + IDE
