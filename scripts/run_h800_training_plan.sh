#!/usr/bin/env bash
# H800 训练计划执行器（分阶段）
# 目标：把“可优化方向”落成可重复执行的流水线
#
# 阶段:
#   base      -> LGB + LSTM(MH) + TFT + Cross-Asset
#   stacking  -> Stacking 训练 + 默认别名同步
#   onnx      -> ONNX 导出
#   report    -> 汇总关键指标与门禁判断
#   all       -> 依次执行全部阶段（默认）
#
# 用法:
#   bash scripts/run_h800_training_plan.sh
#   bash scripts/run_h800_training_plan.sh --stage base
#   bash scripts/run_h800_training_plan.sh --tf 1h,4h --symbol ETHUSDT
#   bash scripts/run_h800_training_plan.sh --no-install --no-verify-data

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SYMBOL="${SYMBOL:-ETHUSDT}"
TIMEFRAMES="${TIMEFRAMES:-1h,4h,24h}"
STAGE="${STAGE:-all}"
ALIAS_TF="${STACKING_ALIAS_TF:-${ML_STACKING_TIMEFRAME:-1h}}"

MIN_STACKING_SAMPLES="${MIN_STACKING_SAMPLES:-20000}"
STACKING_MIN_VAL_AUC="${STACKING_MIN_VAL_AUC:-0.53}"
STACKING_MIN_TEST_AUC="${STACKING_MIN_TEST_AUC:-0.52}"
STACKING_MIN_OOF_AUC="${STACKING_MIN_OOF_AUC:-0.53}"
STACKING_MAX_OOF_TEST_GAP="${STACKING_MAX_OOF_TEST_GAP:-0.10}"

INSTALL_DEPS=1
VERIFY_DATA=1

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_h800_training_plan.sh [options]

Options:
  --stage <name>         all|base|stacking|onnx|report (default: all)
  --tf <list>            Timeframes, e.g. 1h,4h,24h
  --symbol <name>        Symbol, default ETHUSDT
  --python <bin>         Python binary, default python3
  --alias-tf <tf>        Stacking 默认别名周期（默认取环境变量或 1h）
  --min-stacking-samples <n>
  --no-install           Skip dependency installation
  --no-verify-data       Skip verify_data.py
  -h, --help             Show help

Env overrides:
  PYTHON_BIN, SYMBOL, TIMEFRAMES, STAGE, STACKING_ALIAS_TF
  MIN_STACKING_SAMPLES, STACKING_MIN_VAL_AUC, STACKING_MIN_TEST_AUC,
  STACKING_MIN_OOF_AUC, STACKING_MAX_OOF_TEST_GAP
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --tf)
      TIMEFRAMES="$2"
      shift 2
      ;;
    --symbol)
      SYMBOL="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --alias-tf)
      ALIAS_TF="$2"
      shift 2
      ;;
    --min-stacking-samples)
      MIN_STACKING_SAMPLES="$2"
      shift 2
      ;;
    --no-install)
      INSTALL_DEPS=0
      shift
      ;;
    --no-verify-data)
      VERIFY_DATA=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

cd "${ROOT_DIR}"

if [[ ! -f "${ROOT_DIR}/train_gpu.py" ]]; then
  echo "[ERROR] train_gpu.py not found under ${ROOT_DIR}" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/h800_plan_${TS}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=================================================="
echo "H800 训练计划执行器"
echo "=================================================="
echo "[INFO] root=${ROOT_DIR}"
echo "[INFO] python=${PYTHON_BIN}"
echo "[INFO] stage=${STAGE}"
echo "[INFO] symbol=${SYMBOL}"
echo "[INFO] timeframes=${TIMEFRAMES}"
echo "[INFO] alias_tf=${ALIAS_TF}"
echo "[INFO] min_stacking_samples=${MIN_STACKING_SAMPLES}"
echo "[INFO] log=${LOG_FILE}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] GPU:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  echo "[WARN] nvidia-smi not found"
fi

if [[ "${INSTALL_DEPS}" -eq 1 ]]; then
  echo ""
  echo "[SETUP] install deps"
  "${PYTHON_BIN}" -m pip install -r requirements.txt
  "${PYTHON_BIN}" -m pip install -r requirements-gpu.txt
else
  echo ""
  echo "[SETUP] skip deps install (--no-install)"
fi

if [[ "${VERIFY_DATA}" -eq 1 ]]; then
  echo ""
  echo "[SETUP] verify data"
  "${PYTHON_BIN}" verify_data.py
else
  echo ""
  echo "[SETUP] skip data verify (--no-verify-data)"
fi

run_base() {
  echo ""
  echo "[BASE] LightGBM"
  "${PYTHON_BIN}" train_gpu.py --mode lgb --tf "${TIMEFRAMES}" --symbol "${SYMBOL}"

  echo ""
  echo "[BASE] LSTM (multi-horizon=1)"
  "${PYTHON_BIN}" train_gpu.py --mode lstm --tf "${TIMEFRAMES}" --symbol "${SYMBOL}" --multi-horizon 1

  echo ""
  echo "[BASE] TFT"
  "${PYTHON_BIN}" train_gpu.py --mode tft --tf "${TIMEFRAMES}" --symbol "${SYMBOL}"

  echo ""
  echo "[BASE] Cross-Asset"
  "${PYTHON_BIN}" train_gpu.py --mode cross_asset --tf "${TIMEFRAMES}" --symbol "${SYMBOL}"
}

run_stacking() {
  echo ""
  echo "[STACKING] train"
  "${PYTHON_BIN}" train_gpu.py \
    --mode stacking \
    --tf "${TIMEFRAMES}" \
    --symbol "${SYMBOL}" \
    --min-stacking-samples "${MIN_STACKING_SAMPLES}"

  echo ""
  echo "[STACKING] sync alias -> ${ALIAS_TF}"
  "${PYTHON_BIN}" scripts/sync_stacking_alias.py --tf "${ALIAS_TF}"
}

run_onnx() {
  echo ""
  echo "[ONNX] export"
  "${PYTHON_BIN}" train_gpu.py --mode onnx --symbol "${SYMBOL}"
}

run_report() {
  echo ""
  echo "[REPORT] summarize model metrics and gates"
  STACKING_MIN_VAL_AUC="${STACKING_MIN_VAL_AUC}" \
  STACKING_MIN_TEST_AUC="${STACKING_MIN_TEST_AUC}" \
  STACKING_MIN_OOF_AUC="${STACKING_MIN_OOF_AUC}" \
  STACKING_MAX_OOF_TEST_GAP="${STACKING_MAX_OOF_TEST_GAP}" \
  MIN_STACKING_SAMPLES="${MIN_STACKING_SAMPLES}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

model_dir = Path("data/ml_models")
if not model_dir.exists():
    print("[FAIL] data/ml_models not found")
    raise SystemExit(1)

def read_json(p):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

print("[INFO] Stacking gates:")
print(
    "  min_val_auc={:.4f} min_test_auc={:.4f} min_oof_auc={:.4f} max_gap={:.4f} min_samples={}".format(
        float(os.environ.get("STACKING_MIN_VAL_AUC", "0.53")),
        float(os.environ.get("STACKING_MIN_TEST_AUC", "0.52")),
        float(os.environ.get("STACKING_MIN_OOF_AUC", "0.53")),
        float(os.environ.get("STACKING_MAX_OOF_TEST_GAP", "0.10")),
        int(os.environ.get("MIN_STACKING_SAMPLES", "20000")),
    )
)

for p in sorted(model_dir.glob("stacking_meta*.json")):
    d = read_json(p)
    tf = d.get("timeframe")
    va = d.get("val_auc")
    ta = d.get("test_auc")
    oa = d.get("oof_meta_auc")
    ns = d.get("n_oof_samples")
    gap = None
    try:
        gap = float(oa) - float(ta)
    except Exception:
        gap = None
    reasons = []
    try:
        if float(va) < float(os.environ.get("STACKING_MIN_VAL_AUC", "0.53")):
            reasons.append("val_auc")
    except Exception:
        reasons.append("val_auc_missing")
    try:
        if float(ta) < float(os.environ.get("STACKING_MIN_TEST_AUC", "0.52")):
            reasons.append("test_auc")
    except Exception:
        reasons.append("test_auc_missing")
    try:
        if float(oa) < float(os.environ.get("STACKING_MIN_OOF_AUC", "0.53")):
            reasons.append("oof_auc")
    except Exception:
        reasons.append("oof_auc_missing")
    try:
        if int(ns) < int(os.environ.get("MIN_STACKING_SAMPLES", "20000")):
            reasons.append("n_oof_samples")
    except Exception:
        reasons.append("n_oof_samples_missing")
    try:
        if gap is not None and gap > float(os.environ.get("STACKING_MAX_OOF_TEST_GAP", "0.10")):
            reasons.append("overfit_gap")
    except Exception:
        pass
    status = "PASS" if not reasons else "BLOCKED"
    print(
        f"[STACKING] {p.name} tf={tf} val={va} test={ta} oof={oa} "
        f"n_oof={ns} gap={gap} => {status}"
        + (f" reasons={','.join(reasons)}" if reasons else "")
    )

for fn in ("lstm_1h_meta.json", "tft_1h.meta.json"):
    p = model_dir / fn
    if p.exists():
        d = read_json(p)
        keys = ["multi_horizon", "best_head", "seq_len", "input_dim", "val_auc_5h", "val_auc_12h", "val_auc_24h"]
        vals = {k: d.get(k) for k in keys if k in d}
        print(f"[MODEL] {fn}: {vals}")

alias = model_dir / "stacking_meta.json"
if alias.exists():
    d = read_json(alias)
    print(f"[ALIAS] stacking_meta.json -> tf={d.get('timeframe')} alias_of={d.get('alias_of', '-')}")
PY
}

case "${STAGE}" in
  all)
    run_base
    run_stacking
    run_onnx
    run_report
    ;;
  base)
    run_base
    ;;
  stacking)
    run_stacking
    ;;
  onnx)
    run_onnx
    ;;
  report)
    run_report
    ;;
  *)
    echo "[ERROR] invalid stage: ${STAGE}" >&2
    exit 1
    ;;
esac

echo ""
echo "=================================================="
echo "Done"
echo "=================================================="
