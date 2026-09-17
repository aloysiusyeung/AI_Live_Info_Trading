#!/usr/bin/env bash
# Download bar history, then walk-forward validate and select models.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "==> Backfilling bars"
./.venv/bin/python -m stockbot.cli backfill
echo "==> Training and selecting models"
./.venv/bin/python -m stockbot.cli train --force
