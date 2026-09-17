#!/usr/bin/env bash
# Run the 10-minute analysis loop in the foreground.
# Use systemd, supervisor or Docker (see README) to keep it running.
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./.venv/bin/python -m stockbot.cli run --train-on-start
