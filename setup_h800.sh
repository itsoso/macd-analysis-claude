#!/bin/bash
# H800 环境设置脚本
# 用途: 在 H800 GPU 训练机上一键配置环境

set -e

echo "=========================================="
echo "H800 GPU 训练环境设置"
echo "=========================================="

# 检查 GPU
echo ""
echo "1. 检查 GPU 状态..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "  ✅ GPU 可用"
else
    echo "  ⚠️  nvidia-smi 未找到，可能无 GPU 或驱动未安装"
fi

# 检查 Python
echo ""
echo "2. 检查 Python 环境..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "  ✅ $PYTHON_VERSION"
else
    echo "  ❌ Python3 未安装"
    exit 1
fi

# 检查 conda
echo ""
echo "3. 检查 Conda 环境..."
if command -v conda &> /dev/null; then
    CONDA_VERSION=$(conda --version)
    echo "  ✅ $CONDA_VERSION"

    # 创建或激活环境
    if conda env list | grep -q "macd-train"; then
        echo "  ✅ macd-train 环境已存在"
    else
        echo "  创建 macd-train 环境..."
        conda create -n macd-train python=3.10 -y
        echo "  ✅ macd-train 环境已创建"
    fi

    echo ""
    echo "  激活环境: conda activate macd-train"
else
    echo "  ⚠️  Conda 未安装，使用系统 Python"
fi

# 安装依赖
echo ""
echo "4. 安装 Python 依赖..."

if [ -f "requirements-gpu.txt" ]; then
    echo "  安装 GPU 训练依赖..."
    pip install -r requirements-gpu.txt -q
    echo "  ✅ GPU 依赖已安装"
fi

if [ -f "requirements.txt" ]; then
    echo "  安装基础依赖..."
    pip install -r requirements.txt -q
    echo "  ✅ 基础依赖已安装"
fi

# 检查关键库
echo ""
echo "5. 验证关键库..."
python3 -c "import torch; print(f'  ✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')" 2>/dev/null || echo "  ⚠️  PyTorch 未安装"
python3 -c "import lightgbm; print(f'  ✅ LightGBM {lightgbm.__version__}')" 2>/dev/null || echo "  ⚠️  LightGBM 未安装"
python3 -c "import pandas; print(f'  ✅ Pandas {pandas.__version__}')" 2>/dev/null || echo "  ⚠️  Pandas 未安装"
python3 -c "import numpy; print(f'  ✅ NumPy {numpy.__version__}')" 2>/dev/null || echo "  ⚠️  NumPy 未安装"

# 创建输出目录
echo ""
echo "6. 创建输出目录..."
mkdir -p data/ml_models
mkdir -p data/gpu_results
echo "  ✅ 输出目录已创建"

echo ""
echo "=========================================="
echo "✅ 环境设置完成！"
echo "=========================================="
echo ""
echo "下一步操作:"
echo ""
echo "1. 验证数据完整性:"
echo "   python3 verify_data.py"
echo ""
echo "2. 开始训练 (LGB 方向模型):"
echo "   python3 train_gpu.py --mode lgb --tf 1h"
echo ""
echo "3. 查看训练结果:"
echo "   ls -lh data/ml_models/"
echo "   cat data/gpu_results/lgb_training.json"
echo ""
