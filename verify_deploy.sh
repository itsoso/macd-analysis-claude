#!/bin/bash
# ============================================================
# 部署验证脚本 — 在生产机上执行
# 用法: bash verify_deploy.sh
# ============================================================

cd /opt/macd-analysis

echo "=========================================="
echo "  部署验证"
echo "=========================================="

PASS=0
FAIL=0

check() {
    if [ $1 -eq 0 ]; then
        echo "  [OK] $2"
        PASS=$((PASS+1))
    else
        echo "  [FAIL] $2"
        FAIL=$((FAIL+1))
    fi
}

# 1. Git 版本
echo ""
echo "[1/6] Git 版本..."
LATEST=$(git log --oneline -1)
echo "  $LATEST"
git log --oneline -1 | grep -q "ml_models\|ML\|ml" 2>/dev/null && RET=0 || RET=1
check $RET "最新 commit 包含 ML 相关变更"

# 2. 模型文件
echo ""
echo "[2/6] 模型文件..."
MODEL_DIR="data/ml_models"
EXPECTED_FILES=(
    "lgb_direction_model.txt"
    "lgb_direction_model.txt.meta.json"
    "lstm_1h.pt"
    "ensemble_config.json"
    "training_meta.json"
    "trend_regime_model.txt"
    "vol_regime_model.txt"
    "quantile_config.json"
)
for f in "${EXPECTED_FILES[@]}"; do
    if [ -f "$MODEL_DIR/$f" ]; then
        SIZE=$(du -h "$MODEL_DIR/$f" | cut -f1)
        echo "  [OK] $f ($SIZE)"
        PASS=$((PASS+1))
    else
        echo "  [FAIL] $f 缺失"
        FAIL=$((FAIL+1))
    fi
done
TOTAL_FILES=$(ls "$MODEL_DIR" | wc -l)
echo "  模型文件总数: $TOTAL_FILES"

# 3. 关键代码文件
echo ""
echo "[3/6] 关键代码文件..."
CODE_FILES=(
    "ml_live_integration.py"
    "train_gpu.py"
    "train_production_model.py"
    "ml_features.py"
    "ml_predictor.py"
)
for f in "${CODE_FILES[@]}"; do
    test -f "$f" && RET=0 || RET=1
    check $RET "$f 存在"
done

# 4. ml_live_integration v5
echo ""
echo "[4/6] ML 集成版本..."
grep -q "v5" ml_live_integration.py 2>/dev/null && RET=0 || RET=1
check $RET "ml_live_integration.py 已升级到 v5"
grep -q "direction_model" ml_live_integration.py 2>/dev/null && RET=0 || RET=1
check $RET "方向预测模型已集成"

# 5. Web 服务
echo ""
echo "[5/6] Web 服务..."
WEB_STATUS=$(systemctl is-active macd-analysis 2>/dev/null || echo "inactive")
echo "  服务状态: $WEB_STATUS"
[ "$WEB_STATUS" = "active" ] && RET=0 || RET=1
check $RET "macd-analysis 服务运行中"

HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://localhost:5100/ 2>/dev/null || echo "000")
echo "  HTTP 状态码: $HTTP_CODE"
[ "$HTTP_CODE" = "200" ] && RET=0 || RET=1
check $RET "Web 响应正常 (200)"

# 6. Python 导入测试
echo ""
echo "[6/6] Python 模块导入..."
cd /opt/macd-analysis
IMPORT_RESULT=$(python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from ml_live_integration import MLSignalEnhancer
    e = MLSignalEnhancer()
    loaded = e.load_model()
    parts = []
    if e._direction_model: parts.append('LGB')
    if e._lstm_meta: parts.append('LSTM')
    if e._regime_model: parts.append('Regime')
    if e._quantile_model: parts.append('Quantile')
    print(f'OK|{len(parts)}|{\",\".join(parts)}')
except Exception as ex:
    print(f'FAIL|0|{ex}')
" 2>/dev/null || echo "FAIL|0|Python error")

IFS='|' read -r STATUS COUNT COMPONENTS <<< "$IMPORT_RESULT"
echo "  加载状态: $STATUS"
echo "  模型组件: $COMPONENTS ($COUNT 个)"
[ "$STATUS" = "OK" ] && [ "$COUNT" -ge 2 ] && RET=0 || RET=1
check $RET "MLSignalEnhancer 加载成功 (>=2 组件)"

# 汇总
echo ""
echo "=========================================="
echo "  验证结果: $PASS 通过, $FAIL 失败"
echo "=========================================="
if [ $FAIL -eq 0 ]; then
    echo "  ALL PASSED - 部署成功!"
else
    echo "  有 $FAIL 项失败，请检查"
fi
