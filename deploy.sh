#!/bin/bash
# ============================================================
# 部署脚本 — 将代码部署到远程服务器
# 服务器: ssh -p 22222 root@47.237.191.17
# ============================================================

set -e

SERVER="root@47.237.191.17"
PORT=22222
REMOTE_DIR="/root/macd-analysis-claude"
SSH_CMD="ssh -p $PORT $SERVER"

echo "=========================================="
echo "  部署到服务器 $SERVER:$PORT"
echo "=========================================="

# 1. 拉取最新代码
echo ""
echo "[1/3] 拉取最新代码..."
$SSH_CMD "cd $REMOTE_DIR && git reset --hard HEAD && git pull origin main"

# 2. 重启 Web 服务
echo ""
echo "[2/3] 重启 Web 服务..."
$SSH_CMD "systemctl restart macd-web 2>/dev/null || echo '  macd-web 服务不存在，跳过'"

# 3. 重启交易引擎（如果在运行的话）
echo ""
echo "[3/3] 重启交易引擎..."
$SSH_CMD "systemctl restart macd-trading 2>/dev/null || echo '  macd-trading 服务不存在，跳过'"

echo ""
echo "=========================================="
echo "  部署完成!"
echo "=========================================="
