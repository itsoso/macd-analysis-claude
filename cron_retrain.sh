#!/bin/bash
# 每日自动重训脚本
# Cron: 0 4 * * * /workspace/macd-analysis-claude/cron_retrain.sh

set -e

# 配置
PROJECT_DIR="/workspace/macd-analysis-claude"
PYTHON="/root/anaconda3/bin/python"
LOG_DIR="$PROJECT_DIR/logs/retrain"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建日志目录
mkdir -p "$LOG_DIR"

# 日志文件
LOG_FILE="$LOG_DIR/retrain_$DATE.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "自动重训开始: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

cd "$PROJECT_DIR"

# 执行重训
$PYTHON train_gpu.py --mode retrain --tf 1h 2>&1 | tee -a "$LOG_FILE"

# 检查退出状态
if [ $? -eq 0 ]; then
    echo "✓ 重训成功" | tee -a "$LOG_FILE"

    # 可选: 部署到生产服务器
    # ./deploy.sh update
else
    echo "✗ 重训失败" | tee -a "$LOG_FILE"
    exit 1
fi

echo "========================================" | tee -a "$LOG_FILE"
echo "自动重训完成: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# 清理旧日志 (保留最近 30 天)
find "$LOG_DIR" -name "retrain_*.log" -mtime +30 -delete

# 可选: 发送通知
# curl -X POST "https://api.telegram.org/bot<TOKEN>/sendMessage" \
#   -d "chat_id=<CHAT_ID>" \
#   -d "text=模型重训完成: AUC $(grep 'AUC' $LOG_FILE | tail -1)"
