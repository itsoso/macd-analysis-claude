# H800-New-1: Multi-Horizon LSTM 实现说明

## 概述

实现了 Multi-Horizon LSTM 模型，通过 3 个独立的分类头预测不同时间跨度的收益，以提高预测准确性。

**目标**: 将 LSTM AUC 从 0.54 提升至 0.57+

## 技术实现

### 1. 模型架构

新增 `LSTMMultiHorizon` 类 (train_gpu.py:312-352):

```
共享 LSTM 编码器 (Bidirectional, hidden_dim=192)
    ↓
Attention 机制
    ↓
3 个独立分类头:
  ├─ head_5h  → 预测 5 小时收益 (短期)
  ├─ head_12h → 预测 12 小时收益 (中期)
  └─ head_24h → 预测 24 小时收益 (长期)
```

**参数量**: 1,383,172 (vs 单头 1,316,932)

### 2. 训练策略

- **多标签训练**: 同时训练 3 个预测头，使用 `profitable_long_5`, `profitable_long_12`, `profitable_long_24` 标签
- **联合损失**: `loss = (loss_5h + loss_12h + loss_24h) / 3.0`
- **最佳头选择**: 验证时选择 AUC 最高的头作为最终预测
- **元数据保存**: 保存 `lstm_{tf}_meta.json` 记录最佳预测头信息

### 3. 代码修改

| 位置 | 修改内容 |
|------|---------|
| 292-352 | 新增 `LSTMMultiHorizon` 类 |
| 362-372 | 多标签数据准备 (label_5h, label_12h, label_24h) |
| 381-405 | `make_sequences()` 支持多标签 |
| 418-424 | 模型实例化 (根据 `use_multi_horizon` 标志) |
| 430-439 | DataLoader 支持多标签 |
| 446-485 | 训练循环支持多头预测和联合损失 |
| 490-535 | 验证循环计算 3 个头的 AUC，选择最佳 |
| 539-546 | 日志输出 3 个头的 AUC |
| 552-580 | 测试集评估 3 个头 |
| 585-607 | 返回结果包含多头指标 |

## 使用方法

### 在 H800 上训练

```bash
# 1h 周期 Multi-Horizon LSTM
python3 train_gpu.py --mode lstm --tf 1h

# 多周期训练
python3 train_gpu.py --mode lstm --tf 1h,4h
```

### 输出文件

训练完成后会生成:

1. **模型文件**: `data/ml_models/lstm_1h.pt`
2. **元数据**: `data/ml_models/lstm_1h_meta.json`
   ```json
   {
     "multi_horizon": true,
     "best_head": "12h",
     "val_auc_5h": 0.5423,
     "val_auc_12h": 0.5712,
     "val_auc_24h": 0.5589
   }
   ```

### 预期结果

训练日志示例:

```
[Multi-Horizon] 使用 3 个预测头: 5h/12h/24h
序列数据: train=25000, val=4000, test=12000

Epoch   0: loss=0.6821 val_AUC: 5h=0.5234 12h=0.5401 24h=0.5312 best=0.5401 (12h)
Epoch   5: loss=0.6543 val_AUC: 5h=0.5389 12h=0.5623 24h=0.5498 best=0.5623 (12h)
Epoch  10: loss=0.6421 val_AUC: 5h=0.5423 12h=0.5712 24h=0.5589 best=0.5712 (12h)
...
Early stopping at epoch 23

[Multi-Horizon] 最佳预测头: 12h
  验证 AUC: 5h=0.5423, 12h=0.5712, 24h=0.5589
  测试 AUC: 5h=0.5401, 12h=0.5698, 24h=0.5567
```

## 推理侧集成

推理时需要:

1. 读取 `lstm_{tf}_meta.json` 获取 `best_head` 信息
2. 加载模型时使用 `LSTMMultiHorizon` 类
3. 调用 `model(x, return_all=True)` 获取 3 个头的输出
4. 根据 `best_head` 选择对应的预测值

示例代码:

```python
import json
import torch

# 加载元数据
with open('data/ml_models/lstm_1h_meta.json') as f:
    meta = json.load(f)

# 加载模型
model = LSTMMultiHorizon(...)
model.load_state_dict(torch.load('data/ml_models/lstm_1h.pt'))
model.eval()

# 推理
with torch.no_grad():
    pred_5h, pred_12h, pred_24h = model(x, return_all=True)
    pred_5h = torch.sigmoid(pred_5h)
    pred_12h = torch.sigmoid(pred_12h)
    pred_24h = torch.sigmoid(pred_24h)

# 选择最佳头
best_head = meta['best_head']
if best_head == '5h':
    final_pred = pred_5h
elif best_head == '12h':
    final_pred = pred_12h
else:
    final_pred = pred_24h
```

## 验证清单

训练完成后检查:

- [ ] `lstm_1h.pt` 文件大小约 5.3MB (vs 单头 5.0MB)
- [ ] `lstm_1h_meta.json` 存在且包含 `multi_horizon: true`
- [ ] 验证 AUC 至少有一个头 ≥ 0.57
- [ ] 测试 AUC 与验证 AUC 差距 < 0.02 (无过拟合)
- [ ] 日志显示 3 个头的 AUC 值

## 回退方案

如果 Multi-Horizon 效果不佳，可以通过修改 `train_gpu.py:363` 回退到单头模式:

```python
use_multi_horizon = False  # 改为 False
```

## 后续优化方向

1. **加权集成**: 不只选最佳头，而是加权平均 3 个头的预测
2. **动态头选择**: 根据市场 Regime 动态选择预测头
3. **更多时间跨度**: 增加 8h, 16h 等中间跨度
4. **头间正则化**: 添加 diversity loss 鼓励头间差异

---

**实现时间**: 2026-02-20
**优先级**: P1 (最高 ROI)
**预期收益**: AUC 0.54 → 0.57+
