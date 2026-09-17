#!/usr/bin/env bash
# Launch the Streamlit dashboard. Read-only apart from the emergency stop.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${STREAMLIT_PORT:-8501}"
exec ./.venv/bin/streamlit run dashboard/app.py \
  --server.port "$PORT" \
  --server.address "${STREAMLIT_ADDRESS:-0.0.0.0}" \
  --server.headless true
