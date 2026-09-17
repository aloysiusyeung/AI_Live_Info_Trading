#!/usr/bin/env bash
# Download bar history, then walk-forward validate and select models.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "==> Backfilling market-wide news (all US symbols)"
./.venv/bin/python -m stockbot.cli news --backfill

echo "==> Backfilling bars"
./.venv/bin/python -m stockbot.cli backfill
echo "==> Training and selecting models"
./.venv/bin/python -m stockbot.cli train --force
