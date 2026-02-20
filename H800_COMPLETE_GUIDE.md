# H800 完整训练执行指南 v2.0
## Multi-Horizon LSTM + Stacking + 健康检查

**更新时间**: 2026-02-20
**适用版本**: 包含 Multi-Horizon LSTM, Shadow 日志诊断, Stacking 门禁

---

## 概述

本指南涵盖从 H800 训练到生产部署的完整流程：

1. **H800 训练**: Multi-Horizon LSTM + ONNX 导出 + Stacking
2. **本机验证**: 健康检查 + Shadow 日志分析
3. **生产部署**: 模型部署 + 观察验证
4. **正式开启**: Stacking 激活

---

## 第一部分: H800 训练 (2-3 小时)

### 前置检查

```bash
# 1. 确认数据完整性
cd ~/work/macd-analysis
python3 verify_data.py

# 2. 确认 GPU 可用
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 3. 确认依赖
python3 -c "import torch, lightgbm, sklearn, onnxruntime; print('✓ All packages OK')"
```

### 任务 1: Multi-Horizon LSTM (20-30 分钟)

```bash
python3 train_gpu.py --mode lstm --tf 1h
```

**预期输出**:
```
[Multi-Horizon] 使用 3 个预测头: 5h/12h/24h
序列数据: train=25000, val=4000, test=12000
Epoch   0: loss=0.6821 val_AUC: 5h=0.5234 12h=0.5401 24h=0.5312 best=0.5401 (12h)
...
[Multi-Horizon] 最佳预测头: 12h
  验证 AUC: 5h=0.5423, 12h=0.5712, 24h=0.5589
  测试 AUC: 5h=0.5401, 12h=0.5698, 24h=0.5567
```

**成功标准**:
- ✓ 至少一个头 Val AUC ≥ 0.57
- ✓ Test AUC 与 Val AUC 差距 < 0.02
- ✓ 生成 `lstm_1h.pt` (~5.3MB) + `lstm_1h_meta.json`

### 任务 2: ONNX 导出 (5 分钟)

```bash
python3 train_gpu.py --mode onnx
```

**验证 ONNX 输出**:
```bash
python3 -c "
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession('data/ml_models/lstm_1h.onnx')
x = np.random.randn(1, 48, 80).astype(np.float32)
out = sess.run(None, {'input': x})[0]
print(f'ONNX output range: [{out.min():.4f}, {out.max():.4f}]')
assert 0.0 <= out.min() <= 1.0 and 0.0 <= out.max() <= 1.0, 'ONNX output abnormal'
print('✓ ONNX export successful')
"
```

### 任务 3: Stacking Ensemble (30-40 分钟)

```bash
python3 train_gpu.py --mode stacking --tf 1h
```

**检查 Stacking 权重**:
```bash
python3 -c "
import json
with open('data/ml_models/stacking_meta_1h.json') as f:
    meta = json.load(f)
print('Meta coefficients:')
for model, coef in meta['meta_coefficients'].items():
    print(f'  {model:20s}: {coef:7.4f}')
print(f'\\nOOF samples: {meta[\"n_oof_samples\"]}')
print(f'Meta OOF AUC: {meta[\"oof_meta_auc\"]}')
"
```

**成功标准**:
- ✓ Meta OOF AUC ≥ 0.58
- ✓ n_oof_samples ≥ 10000
- ✓ 生成 `stacking_meta_1h.pkl` + `stacking_meta_1h.json`

### 任务 4: 打包回传

```bash
tar -czf macd_models_$(date +%Y%m%d_%H%M%S).tar.gz \
    data/ml_models/lstm_1h.* \
    data/ml_models/tft_1h.* \
    data/ml_models/stacking_* \
    data/gpu_results/train_*.log

ls -lh macd_models_*.tar.gz
# 传回本机: scp -J jumphost macd_models_*.tar.gz user@dev:~/
```

---

## 第二部分: 本机验证 (30 分钟)

### 步骤 1: 解压模型

```bash
cd ~/macd-analysis
tar -xzf macd_models_*.tar.gz
ls -lh data/ml_models/lstm_1h.* data/ml_models/stacking_meta_1h.*
```

### 步骤 2: 安装依赖 (如需要)

```bash
pip install lightgbm scikit-learn torch onnxruntime
```

### 步骤 3: 健康检查

```bash
python check_ml_health.py --skip-live-check
```

**预期输出**:
```
✓ LightGBM 方向预测模型加载成功
✓ LSTM 模型加载成功 (Multi-Horizon: best_head=12h)
✓ TFT 模型加载成功
✓ Stacking 元模型加载成功 (n_oof_samples=24492)
✓ 所有模型健康
```

---

## 第三部分: 生产部署 (1 小时)

### 步骤 1: 部署模型

```bash
# 方案 1: 自动部署 (推荐)
./deploy.sh update

# 方案 2: 手动部署
scp -P 22222 -r data/ml_models/* root@47.237.191.17:/opt/macd-analysis/data/ml_models/
ssh -p 22222 root@47.237.191.17 "systemctl restart macd-analysis"
```

### 步骤 2: 观察日志 (1-2 小时)

```bash
# SSH 到服务器
ssh -p 22222 root@47.237.191.17

# 实时查看日志
tail -f /opt/macd-analysis/data/live/trades_$(date +%Y%m%d).jsonl | grep -E "ml_|stacking"
```

**观察要点**:
- ✓ 是否有 `ml_stacking_mode` 字段
- ✓ 是否有 ML 错误 (ml_error)
- ✓ Stacking 是否激活

### 步骤 3: Shadow 日志分析

```bash
# 拉取日志到本机
scp -P 22222 root@47.237.191.17:/opt/macd-analysis/data/live/trades_$(date +%Y%m%d).jsonl data/live/

# 分析
python analyze_shadow_logs.py --days 2 --min-bars 20
```

**预期输出**:
```
=== Shadow 模式诊断报告 ===
总信号数: 45
ML 覆盖率: 38/45 (84.4%)

[Stacking 激活情况]
✓ Stacking 已激活: 35/38 (92.1%)
✗ Stacking 未激活: 3/38 (7.9%)
  - 样本不足: 2
  - 基模型缺失: 1

[TFT 输出诊断]
✓ TFT 输出正常: 32/35 (91.4%)

[LSTM 输出诊断]
✓ LSTM 输出正常: 35/35 (100%)
```

---

## 第四部分: 正式开启 (确认稳定后)

### 开启条件检查

- [ ] ML 覆盖率 > 80%
- [ ] Stacking 激活率 > 90%
- [ ] TFT 输出正常率 > 80%
- [ ] LSTM 输出正常率 > 95%
- [ ] 无严重 ML 错误

### 开启 Stacking

```bash
ssh -p 22222 root@47.237.191.17
echo "ML_ENABLE_STACKING=1" >> /opt/macd-analysis/.env
systemctl restart macd-analysis

# 验证
systemctl show macd-analysis | grep ML_ENABLE_STACKING
```

### 持续监控 (24 小时)

```bash
# 每 2 小时检查一次
python analyze_shadow_logs.py --days 1 --min-bars 20
```

---

## 故障排查

### 问题 1: LSTM 训练 OOM

**解决**: 修改 `train_gpu.py:288` → `BATCH_SIZE = 128`

### 问题 2: Stacking 样本不足

**解决**: `export ML_STACKING_MIN_SAMPLES=5000`

### 问题 3: TFT 输出恒为 0

**诊断**:
```bash
python3 -c "
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession('data/ml_models/tft_1h.onnx')
out = sess.run(None, {'input': np.random.randn(1, 48, 80).astype(np.float32)})[0]
print(f'TFT range: [{out.min():.6f}, {out.max():.6f}]')
"
```

### 问题 4: ML 覆盖率为 0

**检查**:
```bash
# 模型文件
ls -lh /opt/macd-analysis/data/ml_models/

# 依赖
python3 -c "import lightgbm, sklearn, torch, onnxruntime"

# 日志错误
journalctl -u macd-analysis -n 100 | grep -i "error"
```

---

## 完整检查清单

### H800 训练
- [ ] Multi-Horizon LSTM (Val AUC ≥ 0.57)
- [ ] ONNX 导出 (推理测试通过)
- [ ] Stacking (Meta OOF AUC ≥ 0.58)
- [ ] 模型打包回传

### 本机验证
- [ ] 模型解压完整
- [ ] 健康检查通过
- [ ] 依赖已安装

### 生产部署
- [ ] 模型已部署
- [ ] 服务已重启
- [ ] 日志有 ML 输出

### Shadow 验证 (1-2 小时)
- [ ] ML 覆盖率 > 80%
- [ ] Stacking 激活率 > 90%
- [ ] TFT 正常率 > 80%
- [ ] LSTM 正常率 > 95%

### 正式开启
- [ ] ML_ENABLE_STACKING=1
- [ ] 持续监控 24 小时

---

**执行时间**: 2-3 小时 (H800) + 1-2 小时 (部署验证)
**风险等级**: 低 (Shadow 模式，不影响实盘)
