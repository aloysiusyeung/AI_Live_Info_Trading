#!/usr/bin/env bash
# Read-only Alpaca paper connection test. Submits no orders.
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./.venv/bin/python -m stockbot.cli check
