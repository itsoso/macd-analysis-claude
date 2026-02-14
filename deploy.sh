#!/bin/bash
# ============================================================
# 部署脚本 — 将代码部署到远程服务器
# 服务器: ssh -p 22222 root@47.237.191.17
# 项目路径: /opt/macd-analysis
# Web服务: macd-analysis.service (gunicorn)
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

# 1. 检查交易引擎是否在运行
echo ""
echo "[1/4] 检查交易引擎状态..."
ENGINE_STATUS=$($SSH_CMD "pgrep -af live_runner.py 2>/dev/null || true")
if [ -n "$ENGINE_STATUS" ]; then
    echo "  ⚠️  交易引擎正在运行:"
    echo "  $ENGINE_STATUS"
    echo "  引擎使用 start_new_session 独立运行，Web 重载不会影响持仓。"
else
    echo "  交易引擎未运行。"
fi

# 2. 拉取最新代码
echo ""
echo "[2/4] 拉取最新代码..."
$SSH_CMD "cd $REMOTE_DIR && git reset --hard HEAD && git pull origin main"

# 3. 同步回测数据（本地 data/backtests/*.db → 服务器，页面展示用）
echo ""
echo "[3/4] 同步回测数据..."
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

# 4. 重启 Web 服务 (preload_app=True 时 reload 不会加载新代码, 必须 restart)
echo ""
echo "[4/4] 重启 Web 服务..."
$SSH_CMD "systemctl restart macd-analysis && sleep 2 && systemctl is-active macd-analysis >/dev/null && echo '  ✅ Web 服务已重启 (restart)' || echo '  ❌ 服务重启失败!'"

# 4. 再次确认引擎状态
if [ -n "$ENGINE_STATUS" ]; then
    echo ""
    echo "[确认] 检查交易引擎是否仍在运行..."
    POST_STATUS=$($SSH_CMD "pgrep -af live_runner.py 2>/dev/null || true")
    if [ -n "$POST_STATUS" ]; then
        echo "  ✅ 交易引擎仍在运行，持仓安全。"
    else
        echo "  ❌ 警告：交易引擎已停止！请手动检查。"
    fi
fi

echo ""
echo "=========================================="
echo "  部署完成!"
echo "=========================================="
