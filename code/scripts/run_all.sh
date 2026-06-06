#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/lorenz63.sh"
bash "${SCRIPT_DIR}/rossler.sh"
bash "${SCRIPT_DIR}/chua.sh"
bash "${SCRIPT_DIR}/lorenz96.sh"
