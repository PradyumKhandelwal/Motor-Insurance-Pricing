#!/usr/bin/env bash
# One-shot setup for macOS on Apple Silicon (tested layout: M-series, 8 GB RAM).
# Usage:  chmod +x setup.sh && ./setup.sh
set -euo pipefail

say() { printf "\n\033[1;34m▶ %s\033[0m\n" "$*"; }
die() { printf "\033[1;31m✖ %s\033[0m\n" "$*" >&2; exit 1; }

cd "$(dirname "$0")"

say "Checking platform"
[[ "$(uname -s)" == "Darwin" ]] || echo "  (not macOS – continuing, but libomp/Homebrew steps are skipped)"
[[ "$(uname -m)" == "arm64" ]] && echo "  Apple Silicon detected ✔"

if [[ "$(uname -s)" == "Darwin" ]]; then
  say "Homebrew"
  command -v brew >/dev/null 2>&1 || die "Homebrew not found. Install it from https://brew.sh then re-run ./setup.sh"

  say "OpenMP runtime (required by LightGBM on macOS)"
  brew list libomp >/dev/null 2>&1 && echo "  libomp already installed ✔" || brew install libomp

  say "uv (fast Python package/venv manager)"
  command -v uv >/dev/null 2>&1 && echo "  uv already installed ✔" || brew install uv
else
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
fi

say "Python 3.12 virtual environment + dependencies (this can take a few minutes)"
uv python install 3.12
uv sync --extra dev

say "Registering Jupyter kernel 'motor-pricing'"
uv run python -m ipykernel install --user --name motor-pricing --display-name "Python (motor-pricing)"

say "Creating working folders"
mkdir -p data/processed models

if [[ -d .git ]]; then
  say "Installing git pre-commit hook"
  uv run pre-commit install
fi

say "Running the test-suite (offline, synthetic data)"
MOTOR_PRICING_SYNTHETIC=1 uv run pytest

cat <<'MSG'

✅ Setup complete.

Next steps
  1. make lab      # opens JupyterLab – run notebooks/01 → 04 in order
                   #   (first run downloads the freMTPL2 data from OpenML, needs internet)
  2. make train    # (optional) train the production models without notebooks
  3. make api      # serve quotes at http://127.0.0.1:8000/docs
  4. make mlflow   # browse experiment runs at http://127.0.0.1:5000

Tip for 8 GB RAM: close other heavy apps while the GLM / Optuna cells run.
MSG
