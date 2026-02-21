#!/bin/bash
# ============================================================
# 本机部署脚本 — 在阿里云生产机上直接执行
# 用法: bash deploy_local.sh
# ============================================================

cd /opt/macd-analysis

ML_ENV_OVERRIDE_CONTENT='[Service]
Environment=ML_ENABLE_STACKING=0
Environment=ML_STACKING_TIMEFRAME=1h
Environment=ML_STACKING_MIN_OOF_SAMPLES=20000
Environment=ML_STACKING_MIN_VAL_AUC=0.53
Environment=ML_STACKING_MIN_TEST_AUC=0.52
Environment=ML_STACKING_MIN_OOF_AUC=0.53
Environment=ML_STACKING_MAX_OOF_TEST_GAP=0.10
Environment=ML_STACKING_MIN_FEATURE_COVERAGE_73=0.90
Environment=ML_STACKING_MIN_FEATURE_COVERAGE_94=0.78
Environment=ML_CROSS_ASSET_MIN_FEATURE_COVERAGE=0.80'

echo "=========================================="
echo "  本机部署 ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "=========================================="

# 1. 拉取最新代码
echo ""
echo "[1/5] 拉取最新代码..."
git fetch origin main
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)
if [ "$LOCAL" = "$REMOTE" ]; then
    echo "  已是最新版本: $(git log --oneline -1)"
else
    echo "  本地:  $(git log --oneline -1)"
    git pull origin main
    echo "  更新到: $(git log --oneline -1)"
    echo "  新增 commits:"
    git log --oneline $LOCAL..HEAD | sed 's/^/    /'
fi

# 2. 安装依赖 (如有变化)
echo ""
echo "[2/5] 检查依赖..."
if [ "$LOCAL" != "$REMOTE" ] && git diff $LOCAL..HEAD --name-only | grep -q "requirements"; then
    echo "  requirements 有变更，安装依赖..."
    /opt/macd-analysis/venv/bin/pip install -r requirements.txt -q
    echo "  依赖已更新"
else
    echo "  依赖无变化，跳过"
fi

# 3. 部署前 ML 健康检查
echo ""
echo "[3/5] 部署前 ML 健康检查..."
/opt/macd-analysis/venv/bin/python3 check_ml_health.py --skip-live-check --timeframe 1h --fix-stacking-alias

# 4. 更新 systemd ML 环境覆盖
echo ""
echo "[4/5] 更新 systemd ML 环境覆盖..."
mkdir -p /etc/systemd/system/macd-analysis.service.d /etc/systemd/system/macd-engine.service.d
cat > /etc/systemd/system/macd-analysis.service.d/20-ml-stacking.conf <<EOF
$ML_ENV_OVERRIDE_CONTENT
EOF
cat > /etc/systemd/system/macd-engine.service.d/20-ml-stacking.conf <<EOF
$ML_ENV_OVERRIDE_CONTENT
EOF
systemctl daemon-reload
echo "  已写入: /etc/systemd/system/*/20-ml-stacking.conf"

# 5. 重启服务
echo ""
echo "[5/5] 重启服务..."
systemctl restart macd-analysis
sleep 2
WEB=$(systemctl is-active macd-analysis)
echo "  Web 服务: $WEB"

if systemctl is-active macd-engine >/dev/null 2>&1; then
    systemctl restart macd-engine
    sleep 2
    ENGINE=$(systemctl is-active macd-engine)
    echo "  交易引擎: $ENGINE"
else
    echo "  交易引擎: 未配置，跳过"
fi

# 健康检查
echo ""
echo "[检查] 健康检查..."
HTTP=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://localhost:5100/ 2>/dev/null)
echo "  HTTP: $HTTP"

if [ "$WEB" = "active" ] && { [ "$HTTP" = "200" ] || [ "$HTTP" = "302" ]; }; then
    echo ""
    echo "=========================================="
    echo "  部署成功!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "  部署异常，请检查日志: journalctl -u macd-analysis -n 30"
    echo "=========================================="
    exit 1
fi
