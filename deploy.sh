#!/bin/bash
# ============================================================
# 部署脚本 — 将代码部署到远程服务器
# 服务器: ssh -p 22222 root@47.237.191.17
# 项目路径: /opt/macd-analysis
# Web服务: macd-analysis.service (gunicorn)
# 交易引擎: macd-engine.service (live_runner.py --phase paper)
# ============================================================

set -e
cd "$(dirname "$0")"

SERVER="root@47.237.191.17"
PORT=22222
REMOTE_DIR="/opt/macd-analysis"
SSH_CMD="ssh -p $PORT $SERVER"

echo "=========================================="
echo "  部署到服务器 $SERVER:$PORT"
echo "=========================================="

# 1. 检查交易引擎状态 (通过 systemd)
echo ""
echo "[1/5] 检查服务状态..."
ENGINE_ACTIVE=$($SSH_CMD "systemctl is-active macd-engine 2>/dev/null || echo inactive")
WEB_ACTIVE=$($SSH_CMD "systemctl is-active macd-analysis 2>/dev/null || echo inactive")
echo "  Web 服务:    $WEB_ACTIVE"
echo "  交易引擎:    $ENGINE_ACTIVE"
if [ "$ENGINE_ACTIVE" = "active" ]; then
    echo "  ⚠️  交易引擎正在运行，部署后将自动重启以加载新代码。"
fi

# 2. 拉取最新代码
echo ""
echo "[2/5] 拉取最新代码..."
DIRTY=$($SSH_CMD "cd $REMOTE_DIR && git status --porcelain" 2>/dev/null || true)
if [ -n "$DIRTY" ] && [ "$1" != "--force" ]; then
    echo "  ⚠️  服务器存在未提交修改，直接 reset 将永久丢失："
    echo "$DIRTY" | head -5
    echo "  请先在服务器上 git stash 或提交后再部署，或执行: bash deploy.sh --force"
    exit 1
fi
$SSH_CMD "cd $REMOTE_DIR && git reset --hard HEAD && git pull origin main"

# 3. 同步回测数据
echo ""
echo "[3/5] 同步回测数据..."
$SSH_CMD "mkdir -p $REMOTE_DIR/data/backtests"
SYNC_FILES=()
[ -f data/backtests/multi_tf_daily_backtest.db ] && SYNC_FILES+=(data/backtests/multi_tf_daily_backtest.db)
[ -f data/backtests/multi_tf_date_range_reports.db ] && SYNC_FILES+=(data/backtests/multi_tf_date_range_reports.db)
[ -f data/backtests/naked_kline_backtest.db ] && SYNC_FILES+=(data/backtests/naked_kline_backtest.db)
if [ ${#SYNC_FILES[@]} -gt 0 ]; then
  rsync -avz -e "ssh -p $PORT" "${SYNC_FILES[@]}" $SERVER:$REMOTE_DIR/data/backtests/
  echo "  ✅ 已同步 ${#SYNC_FILES[@]} 个回测 DB"
else
  echo "  ⚠️  本地无 data/backtests/*.db，跳过"
fi

# 4. 重启 Web 服务
echo ""
echo "[4/5] 重启 Web 服务..."
$SSH_CMD "systemctl restart macd-analysis && sleep 2 && systemctl is-active macd-analysis >/dev/null && echo '  ✅ Web 服务已重启' || echo '  ❌ Web 服务重启失败!'"

# 5. 重启交易引擎 (加载新代码)
echo ""
echo "[5/5] 重启交易引擎..."
$SSH_CMD "systemctl restart macd-engine && sleep 3 && systemctl is-active macd-engine >/dev/null && echo '  ✅ 交易引擎已重启 (paper mode)' || echo '  ❌ 交易引擎重启失败!'"

# 最终确认
echo ""
echo "[确认] 服务状态..."
$SSH_CMD "echo '  Web:    '\$(systemctl is-active macd-analysis); echo '  Engine: '\$(systemctl is-active macd-engine); echo '  PID:    '\$(pgrep -f 'live_runner.py' 2>/dev/null || echo 'N/A')"

echo ""
echo "=========================================="
echo "  部署完成!"
echo "=========================================="
