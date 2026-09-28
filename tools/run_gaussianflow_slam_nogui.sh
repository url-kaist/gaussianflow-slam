#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export USE_GUI=0
exec "$SCRIPT_DIR/run_gaussianflow_modes.sh" slam "$@"
