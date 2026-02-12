#!/bin/bash
# ============================================================
# 部署脚本 — 将代码部署到远程服务器
# 服务器: ssh -p 22222 root@47.237.191.17
# ============================================================

set -e

SERVER="root@47.237.191.17"
PORT=22222
REMOTE_DIR="/opt/macd-analysis"
SSH_CMD="ssh -p $PORT $SERVER"

echo "=========================================="
echo "  部署到服务器 $SERVER:$PORT"
echo "=========================================="

# 1. 拉取最新代码
echo ""
echo "[1/3] 拉取最新代码..."
$SSH_CMD "cd $REMOTE_DIR && git reset --hard HEAD && git pull origin main"

# 2. 重启 Web 服务 (macd-analysis.service / gunicorn)
echo ""
echo "[2/2] 重启 Web 服务..."
$SSH_CMD "systemctl restart macd-analysis && echo '  ✅ macd-analysis 服务已重启' || echo '  ❌ 重启失败'"

echo ""
echo "=========================================="
echo "  部署完成!"
echo "=========================================="
