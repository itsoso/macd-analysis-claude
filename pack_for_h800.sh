#!/bin/bash
# H800 训练数据打包脚本
# 用途: 将代码和数据打包传输到 H800 GPU 训练机

set -e

echo "=========================================="
echo "H800 训练数据打包工具"
echo "=========================================="

# 检查必需数据
echo ""
echo "1. 检查数据完整性..."

REQUIRED_DIRS=(
    "data/klines/ETHUSDT"
    "data/klines/BTCUSDT"
    "data/klines/SOLUSDT"
    "data/klines/BNBUSDT"
)

MISSING=0
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "  ❌ 缺失: $dir"
        MISSING=1
    else
        count=$(ls -1 "$dir"/*.parquet 2>/dev/null | wc -l)
        echo "  ✅ $dir ($count 个文件)"
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "⚠️  数据不完整，请先运行: python3 fetch_5year_data.py"
    exit 1
fi

# 创建打包目录
echo ""
echo "2. 创建打包目录..."
PACK_DIR="h800_training_pack"
rm -rf "$PACK_DIR"
mkdir -p "$PACK_DIR"

# 复制代码文件
echo ""
echo "3. 复制代码文件..."
CODE_FILES=(
    "train_gpu.py"
    "verify_data.py"
    "ml_features.py"
    "ml_predictor.py"
    "ml_regime.py"
    "ml_quantile.py"
    "ml_tabnet.py"
    "ml_catboost.py"
    "indicators.py"
    "ma_indicators.py"
    "config.py"
    "binance_fetcher.py"
    "requirements-gpu.txt"
    "requirements.txt"
)

for file in "${CODE_FILES[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$PACK_DIR/"
        echo "  ✅ $file"
    else
        echo "  ⚠️  跳过: $file (不存在)"
    fi
done

# 复制数据目录
echo ""
echo "4. 复制数据目录..."
cp -r data "$PACK_DIR/"
echo "  ✅ data/ ($(du -sh data | cut -f1))"

# 创建打包文件
echo ""
echo "5. 创建压缩包..."
tar -czf macd_train_data.tar.gz -C "$PACK_DIR" .
SIZE=$(du -sh macd_train_data.tar.gz | cut -f1)
echo "  ✅ macd_train_data.tar.gz ($SIZE)"

# 清理临时目录
rm -rf "$PACK_DIR"

# 生成传输命令
echo ""
echo "=========================================="
echo "✅ 打包完成！"
echo "=========================================="
echo ""
echo "下一步操作:"
echo ""
echo "1. 传输到 H800 (通过跳板机):"
echo "   scp -J user@jumphost macd_train_data.tar.gz user@h800:~/work/"
echo ""
echo "2. 在 H800 上解压:"
echo "   ssh -J user@jumphost user@h800"
echo "   cd ~/work"
echo "   tar -xzf macd_train_data.tar.gz"
echo ""
echo "3. 设置环境 (首次):"
echo "   ./setup_h800.sh"
echo ""
echo "4. 验证数据:"
echo "   python3 verify_data.py"
echo ""
echo "5. 开始训练:"
echo "   python3 train_gpu.py --mode lgb --tf 1h"
echo ""
echo "6. 训练完成后，打包模型回传:"
echo "   tar -czf macd_models.tar.gz data/ml_models/ data/gpu_results/"
echo "   scp -J jumphost macd_models.tar.gz user@dev:~/macd-analysis/"
echo ""
