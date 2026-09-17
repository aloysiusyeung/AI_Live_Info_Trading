#!/usr/bin/env bash
# One-time setup: virtualenv, dependencies, .env scaffold, database.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"

echo "==> Creating virtualenv in .venv"
"$PYTHON" -m venv .venv
./.venv/bin/pip install --quiet --upgrade pip

echo "==> Installing dependencies"
./.venv/bin/pip install --quiet -r requirements.txt

if [[ ! -f .env ]]; then
  echo "==> Creating .env from .env.example"
  cp .env.example .env
  echo "    Edit .env and add your Alpaca PAPER keys before running."
else
  echo "==> .env already exists; leaving it alone"
fi

echo "==> Running tests"
./.venv/bin/python -m pytest -q

echo
echo "Setup complete."
echo "  1. Put your Alpaca paper keys in .env (or export them)"
echo "  2. ./scripts/check_connection.sh    # read-only, submits no orders"
echo "  3. ./scripts/bootstrap_data.sh      # download history and train"
echo "  4. ./scripts/run_dashboard.sh       # Streamlit UI"
echo "  5. ./scripts/run_scheduler.sh       # 10-minute update loop"
