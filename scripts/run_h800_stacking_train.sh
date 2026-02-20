#!/usr/bin/env bash
# H800 一键训练脚本（stacking 多周期）
# 功能:
# 1) 安装训练依赖（可跳过）
# 2) 校验关键 Python 包
# 3) 校验数据完整性
# 4) 训练 stacking (1h,4h,24h 默认)
# 5) 校验模型产物
# 6) 打包模型与结果

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TIMEFRAMES="${TIMEFRAMES:-1h,4h,24h}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
ALIAS_TF="${STACKING_ALIAS_TF:-${ML_STACKING_TIMEFRAME:-1h}}"

INSTALL_DEPS=1
VERIFY_DATA=1
RUN_BACKGROUND=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_h800_stacking_train.sh [options] [-- <extra train_gpu args>]

Options:
  --tf <list>            Timeframes, e.g. 1h,4h,24h (default: 1h,4h,24h)
  --python <bin>         Python binary (default: python3)
  --no-install           Skip dependency installation
  --no-verify-data       Skip verify_data.py
  --background           Run training with nohup in background
  -h, --help             Show this help

Environment overrides:
  TIMEFRAMES, PYTHON_BIN, TORCH_INDEX_URL

Examples:
  bash scripts/run_h800_stacking_train.sh
  bash scripts/run_h800_stacking_train.sh --tf 1h,4h --no-install
  bash scripts/run_h800_stacking_train.sh --background -- --symbol ETHUSDT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tf)
      TIMEFRAMES="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
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
    --background)
      RUN_BACKGROUND=1
      shift
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
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

if [[ ! -f "${ROOT_DIR}/train_gpu.py" ]]; then
  echo "[ERROR] train_gpu.py not found under ${ROOT_DIR}" >&2
  exit 1
fi

cd "${ROOT_DIR}"

echo "=========================================="
echo "H800 Stacking 一键训练"
echo "=========================================="
echo "[INFO] Root: ${ROOT_DIR}"
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Timeframes: ${TIMEFRAMES}"
echo "[INFO] Stacking alias timeframe: ${ALIAS_TF}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] GPU:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  echo "[WARN] nvidia-smi not found; GPU may be unavailable"
fi

if [[ "${INSTALL_DEPS}" -eq 1 ]]; then
  echo ""
  echo "[1/6] 安装依赖"
  "${PYTHON_BIN}" -m pip install -r requirements.txt
  "${PYTHON_BIN}" -m pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
  "${PYTHON_BIN}" -m pip install -r requirements-gpu.txt
else
  echo ""
  echo "[1/6] 跳过依赖安装 (--no-install)"
fi

echo ""
echo "[2/6] 校验关键依赖"
"${PYTHON_BIN}" - <<'PY'
import importlib
mods = ["torch", "lightgbm", "xgboost", "sklearn", "pandas", "numpy"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "unknown")
        print(f"[OK] {m} {ver}")
    except Exception as e:
        print(f"[FAIL] {m}: {e}")
        raise
import torch
print(f"[INFO] torch.cuda.is_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] CUDA device count={torch.cuda.device_count()}")
PY

echo ""
if [[ "${VERIFY_DATA}" -eq 1 ]]; then
  echo "[3/6] 校验训练数据"
  "${PYTHON_BIN}" verify_data.py
else
  echo "[3/6] 跳过数据校验 (--no-verify-data)"
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/h800_stacking_${TS}.log"
TRAIN_CMD=( "${PYTHON_BIN}" train_gpu.py --mode stacking --tf "${TIMEFRAMES}" )
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  TRAIN_CMD+=( "${EXTRA_ARGS[@]}" )
fi

echo ""
echo "[4/6] 启动训练"
echo "[INFO] Log: ${LOG_FILE}"
echo "[INFO] Command: ${TRAIN_CMD[*]}"

if [[ "${RUN_BACKGROUND}" -eq 1 ]]; then
  nohup "${TRAIN_CMD[@]}" > "${LOG_FILE}" 2>&1 &
  PID=$!
  echo "[OK] 后台训练已启动, PID=${PID}"
  echo "[TIP] 追踪日志: tail -f ${LOG_FILE}"
  exit 0
else
  "${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
fi

echo ""
echo "[5/6] 校验 stacking 产物"
IFS=',' read -r -a tf_arr <<< "${TIMEFRAMES}"
for tf in "${tf_arr[@]}"; do
  tf="$(echo "${tf}" | xargs)"
  j="data/ml_models/stacking_meta_${tf}.json"
  p="data/ml_models/stacking_meta_${tf}.pkl"
  [[ -f "${j}" ]] || { echo "[ERROR] Missing ${j}" >&2; exit 1; }
  [[ -f "${p}" ]] || { echo "[ERROR] Missing ${p}" >&2; exit 1; }
  echo "[OK] ${j}"
  echo "[OK] ${p}"
done
[[ -f data/ml_models/stacking_meta.json ]] || { echo "[ERROR] Missing data/ml_models/stacking_meta.json" >&2; exit 1; }
[[ -f data/ml_models/stacking_meta.pkl ]] || { echo "[ERROR] Missing data/ml_models/stacking_meta.pkl" >&2; exit 1; }
echo "[OK] 默认别名 data/ml_models/stacking_meta.json/.pkl"

echo "[INFO] 同步默认别名到目标周期: ${ALIAS_TF}"
"${PYTHON_BIN}" scripts/sync_stacking_alias.py --tf "${ALIAS_TF}"

"${PYTHON_BIN}" - <<'PY'
import glob, json
for p in sorted(glob.glob("data/ml_models/stacking_meta*.json")):
    d = json.load(open(p))
    print(
        f"[META] {p}: tf={d.get('timeframe')}, "
        f"val_auc={d.get('val_auc')}, test_auc={d.get('test_auc')}, "
        f"oof_auc={d.get('oof_meta_auc')}, n_oof={d.get('n_oof_samples')}"
    )
PY

echo ""
echo "[6/6] 打包模型"
PKG="macd_models_h800_stacking_${TS}.tar.gz"
tar -czf "${PKG}" data/ml_models/ data/gpu_results/
echo "[OK] ${PKG}"
ls -lh "${PKG}"

echo ""
echo "=========================================="
echo "训练完成"
echo "=========================================="
echo "下一步回传:"
echo "scp -J <jump_user>@<jump_host> ${PKG} <dev_user>@<dev_host>:~/macd-analysis/"
