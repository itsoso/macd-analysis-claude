#!/usr/bin/env bash
# 兼容入口: 在项目根目录调用，转发到 scripts/cron_h800_nightly.sh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
exec bash scripts/cron_h800_nightly.sh "$@"
