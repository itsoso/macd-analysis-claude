#!/usr/bin/env bash
# H800 夜间自动训练入口（cron 调用）
#
# 执行内容:
# 1) 调用训练计划执行器 scripts/run_h800_training_plan.sh
# 2) 生成统一汇总 scripts/build_h800_summary.py (json+md)
#
# 建议 cron:
#   20 2 * * * /opt/macd-analysis/scripts/cron_h800_nightly.sh >> /opt/macd-analysis/logs/retrain/cron_h800_nightly.log 2>&1

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

STAGE="${H800_STAGE:-all}"
SYMBOL="${H800_SYMBOL:-ETHUSDT}"
TIMEFRAMES="${H800_TIMEFRAMES:-1h,4h,24h}"
ALIAS_TF="${STACKING_ALIAS_TF:-${ML_STACKING_TIMEFRAME:-1h}}"
MIN_STACKING_SAMPLES="${MIN_STACKING_SAMPLES:-20000}"

# cron 场景默认不重复安装依赖/验数，可通过环境变量覆盖
H800_NO_INSTALL="${H800_NO_INSTALL:-1}"
H800_NO_VERIFY_DATA="${H800_NO_VERIFY_DATA:-1}"

LOG_DIR="${ROOT_DIR}/logs/retrain"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/nightly_${TS}.log"
LOCK_FILE="${LOG_DIR}/nightly.lock"
LOCK_DIR="${LOG_DIR}/nightly.lockdir"

exec > >(tee -a "${RUN_LOG}") 2>&1

echo "=================================================="
echo "H800 Nightly Train Job"
echo "=================================================="
echo "[INFO] root=${ROOT_DIR}"
echo "[INFO] stage=${STAGE} symbol=${SYMBOL} tfs=${TIMEFRAMES} alias_tf=${ALIAS_TF}"
echo "[INFO] log=${RUN_LOG}"

# 避免并发重入
exec 9>"${LOCK_FILE}"
if command -v flock >/dev/null 2>&1; then
  if ! flock -n 9; then
    echo "[WARN] another nightly job is running; exit"
    exit 0
  fi
else
  if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "[WARN] another nightly job is running (lockdir exists); exit"
    exit 0
  fi
  trap 'rmdir "${LOCK_DIR}" >/dev/null 2>&1 || true' EXIT
  echo "[WARN] flock not found; use lockdir fallback"
fi

cd "${ROOT_DIR}"

TRAIN_CMD=(
  bash scripts/run_h800_training_plan.sh
  --stage "${STAGE}"
  --tf "${TIMEFRAMES}"
  --symbol "${SYMBOL}"
  --alias-tf "${ALIAS_TF}"
  --min-stacking-samples "${MIN_STACKING_SAMPLES}"
)
if [[ "${H800_NO_INSTALL}" == "1" ]]; then
  TRAIN_CMD+=(--no-install)
fi
if [[ "${H800_NO_VERIFY_DATA}" == "1" ]]; then
  TRAIN_CMD+=(--no-verify-data)
fi

echo "[INFO] train_cmd: ${TRAIN_CMD[*]}"
train_rc=0
if ! "${TRAIN_CMD[@]}"; then
  train_rc=$?
  echo "[WARN] training pipeline failed rc=${train_rc}"
fi

echo "[INFO] build nightly summary"
"${PYTHON_BIN}" scripts/build_h800_summary.py --output-dir "${LOG_DIR}" --target-tf "${ALIAS_TF}" || true

echo "=================================================="
echo "Nightly job done: rc=${train_rc}"
echo "=================================================="

exit "${train_rc}"
