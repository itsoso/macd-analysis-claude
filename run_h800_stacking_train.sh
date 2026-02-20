#!/bin/bash
# H800 Stacking Ensemble 训练脚本
# 用途: 在 H800 GPU 上训练多周期 Stacking 模型
# 作者: Claude Code
# 日期: 2026-02-20

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  H800 Stacking Ensemble 训练${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查 GPU
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}错误: 未检测到 NVIDIA GPU${NC}"
    exit 1
fi

# 检查 Anaconda Python
PYTHON_CMD="/root/anaconda3/bin/python"
if [ ! -f "$PYTHON_CMD" ]; then
    echo -e "${YELLOW}警告: Anaconda Python 不存在，使用系统 Python${NC}"
    PYTHON_CMD="python3"
fi

# 检查训练脚本
if [ ! -f "train_gpu.py" ]; then
    echo -e "${RED}错误: train_gpu.py 不存在${NC}"
    exit 1
fi

# 默认训练周期
TIMEFRAMES="${1:-1h,4h,24h}"

echo -e "${GREEN}[1/4] 环境检查${NC}"
echo "  Python: $PYTHON_CMD"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  训练周期: $TIMEFRAMES"
echo ""

echo -e "${GREEN}[2/4] 检查现有模型${NC}"
if ls data/ml_models/stacking_*.pt &> /dev/null; then
    echo -e "${YELLOW}  发现现有 Stacking 模型:${NC}"
    ls -lh data/ml_models/stacking_*.pt | awk '{print "    " $9 " (" $5 ")"}'
    echo ""
    read -p "  是否覆盖现有模型? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}  训练已取消${NC}"
        exit 0
    fi
fi

echo -e "${GREEN}[3/4] 开始训练${NC}"
echo "  命令: $PYTHON_CMD train_gpu.py --mode stacking --tf $TIMEFRAMES"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 执行训练
$PYTHON_CMD train_gpu.py --mode stacking --tf "$TIMEFRAMES" 2>&1 | tee /tmp/stacking_training_$(date +%Y%m%d_%H%M%S).log

# 检查训练是否成功
if [ $? -eq 0 ]; then
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo ""
    echo -e "${GREEN}[4/4] 训练完成 ✓${NC}"
    echo "  耗时: ${ELAPSED}s ($((ELAPSED / 60))分钟)"
    echo ""

    # 显示训练结果
    echo -e "${GREEN}训练产出:${NC}"
    ls -lh data/ml_models/stacking_*.{txt,pt,json,pkl} 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    echo ""

    # 显示最新结果文件
    LATEST_RESULT=$(ls -t data/gpu_results/stacking_ensemble_*.json 2>/dev/null | head -1)
    if [ -f "$LATEST_RESULT" ]; then
        echo -e "${GREEN}训练结果:${NC}"
        cat "$LATEST_RESULT" | python3 -m json.tool 2>/dev/null || cat "$LATEST_RESULT"
    fi
else
    echo ""
    echo -e "${RED}[4/4] 训练失败 ✗${NC}"
    echo "  请检查日志: /tmp/stacking_training_*.log"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  下一步操作${NC}"
echo -e "${GREEN}========================================${NC}"
echo "1. 提交模型到 Git:"
echo "   git add data/ml_models/stacking_*"
echo "   git commit -m 'feat: 更新 Stacking 模型'"
echo ""
echo "2. 部署到生产服务器:"
echo "   ./deploy.sh update"
echo ""
echo "3. 测试实盘信号:"
echo "   python live_runner.py --test-signal-multi --tfs 1h,4h"
echo ""
