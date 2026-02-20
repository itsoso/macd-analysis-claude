# H800 部署清单 - Multi-Horizon LSTM

## 文件清单

需要传输到 H800 的文件:

```
train_gpu.py                    # 已更新 (Multi-Horizon LSTM 实现)
H800_MULTI_HORIZON_LSTM.md      # 实现说明文档
```

## 部署步骤

### 1. 本机打包

```bash
# 创建补丁包 (只包含修改的文件)
tar -czf h800_multi_horizon_patch.tar.gz \
    train_gpu.py \
    H800_MULTI_HORIZON_LSTM.md

# 验证打包
tar -tzf h800_multi_horizon_patch.tar.gz
```

### 2. 传输到 H800

```bash
# 通过跳板机传输
scp -J user@jumphost h800_multi_horizon_patch.tar.gz user@h800:~/work/macd-analysis/
```

### 3. H800 解压并验证

```bash
# SSH 到 H800
ssh -J user@jumphost user@h800

# 进入项目目录
cd ~/work/macd-analysis

# 备份原文件
cp train_gpu.py train_gpu.py.backup_$(date +%Y%m%d_%H%M%S)

# 解压补丁
tar -xzf h800_multi_horizon_patch.tar.gz

# 验证语法
python3 -c "import train_gpu; print('✓ Syntax OK')"
```

### 4. 执行训练

```bash
# 激活环境
conda activate macd

# 训练 1h Multi-Horizon LSTM
python3 train_gpu.py --mode lstm --tf 1h

# 预计耗时: 15-25 分钟 (取决于数据量和 GPU 负载)
```

### 5. 验证输出

训练完成后检查:

```bash
# 检查模型文件
ls -lh data/ml_models/lstm_1h.pt
# 预期: ~5.3MB

# 检查元数据
cat data/ml_models/lstm_1h_meta.json
# 预期包含: multi_horizon: true, best_head, val_auc_5h/12h/24h

# 检查训练日志
tail -50 data/gpu_results/train_lstm_*.log
# 预期看到: 3 个头的 AUC 值
```

### 6. 回传模型

```bash
# 打包新模型
tar -czf macd_multi_horizon_models.tar.gz \
    data/ml_models/lstm_1h.pt \
    data/ml_models/lstm_1h_meta.json \
    data/gpu_results/train_lstm_*.log

# 传回本机
scp -J jumphost macd_multi_horizon_models.tar.gz user@dev:~/macd-analysis/
```

## 成功标准

- [ ] 训练无错误完成
- [ ] 至少一个头的验证 AUC ≥ 0.57
- [ ] 测试 AUC 与验证 AUC 差距 < 0.02
- [ ] `lstm_1h_meta.json` 正确生成
- [ ] 日志显示 3 个头的独立 AUC

## 故障排查

### 问题 1: 内存不足 (OOM)

**症状**: `RuntimeError: CUDA out of memory`

**解决**:
```python
# 修改 train_gpu.py:288
BATCH_SIZE = 128  # 从 256 降低到 128
```

### 问题 2: 标签缺失

**症状**: `KeyError: 'profitable_long_12'` 或 `'profitable_long_24'`

**原因**: 数据不足，无法生成 12h/24h 标签

**解决**: 确保数据至少有 24 个周期的前瞻窗口

### 问题 3: 所有头 AUC 相近

**症状**: 3 个头的 AUC 差距 < 0.01

**分析**: 可能是:
- 数据不足以区分不同时间跨度
- 需要增加头间正则化

**解决**: 先完成训练，后续可以添加 diversity loss

## 回退方案

如果 Multi-Horizon 效果不佳:

```python
# 修改 train_gpu.py:358
use_multi_horizon = False  # 改为 False

# 重新训练
python3 train_gpu.py --mode lstm --tf 1h
```

## 下一步任务

Multi-Horizon LSTM 完成后，继续执行:

- **H800-New-2**: 24h Regime 分类器
- **H800-New-3**: 15m LSTM
- **H800-Fix-2**: TFT 输出 0.0 问题诊断
- **H800-Fix-3**: Stacking 激活路径诊断

---

**创建时间**: 2026-02-20
**优先级**: P1 (最高 ROI)
**预期收益**: AUC 0.54 → 0.57+
