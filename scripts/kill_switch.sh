#!/usr/bin/env bash
# Emergency stop. Usage: ./scripts/kill_switch.sh engage|release
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./.venv/bin/python -m stockbot.cli kill-switch "${1:?usage: kill_switch.sh engage|release}"
