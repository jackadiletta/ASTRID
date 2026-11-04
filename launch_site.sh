#!/usr/bin/env bash
set -euo pipefail

SITE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_1_site"
exec conda run --no-capture-output -n ASTRID_ENV python3 -u "$SITE_DIR/site_control.py" launch
