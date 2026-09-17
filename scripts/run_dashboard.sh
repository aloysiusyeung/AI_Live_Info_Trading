#!/usr/bin/env bash
# Launch the Streamlit dashboard. Read-only apart from the emergency stop.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${STREAMLIT_PORT:-8501}"

# Bind to loopback by default. The dashboard shows your positions and carries
# the emergency stop, so it is not something to serve to whatever network the
# machine happens to be on. Set STREAMLIT_ADDRESS=0.0.0.0 deliberately, and
# only behind a reverse proxy with authentication.
ADDRESS="${STREAMLIT_ADDRESS:-127.0.0.1}"
if [[ "$ADDRESS" != "127.0.0.1" && "$ADDRESS" != "localhost" ]]; then
  echo "WARNING: binding the dashboard to $ADDRESS exposes positions and the" >&2
  echo "         emergency stop to that network. Put it behind authentication." >&2
fi

exec ./.venv/bin/streamlit run dashboard/app.py \
  --server.port "$PORT" \
  --server.address "$ADDRESS" \
  --server.headless true
