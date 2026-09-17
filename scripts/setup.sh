#!/usr/bin/env bash
# One-time setup: virtualenv, dependencies, .env scaffold, database.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"

# Fail early and clearly on an unsupported interpreter. macOS ships Python
# 3.9, which cannot install the pinned pandas/numpy/scikit-learn wheels, and
# the resulting pip error is far from obvious.
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  CURRENT="$("$PYTHON" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || echo unknown)"
  cat >&2 <<MSG
ERROR: Python 3.10 or newer is required; found $CURRENT at $(command -v "$PYTHON").

On macOS the system python3 is too old. Install a newer one:

    brew install python@3.12
    PYTHON=\$(brew --prefix)/bin/python3.12 ./scripts/setup.sh

On Linux, install python3.12 (or 3.11) from your package manager and pass it
the same way.
MSG
  exit 1
fi

echo "==> Using $("$PYTHON" --version) at $(command -v "$PYTHON")"
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
