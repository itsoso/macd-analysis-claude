# H800 改进任务执行清单

## 第一阶段: 紧急修复 (1-2天)

### ✅ H800-New-1: Multi-Horizon LSTM (已完成)
- [x] 实现 `LSTMMultiHorizon` 类
- [x] 多标签训练循环
- [x] 元数据保存
- [ ] **H800 训练验证** ← 下一步
- [ ] 回传模型到本机
- [ ] 推理侧集成

---

### 🔴 H800-Fix-2: TFT 输出 0.0 诊断 (优先级最高)

**问题**: TFT 在 Stacking 中权重为负 (-0.124)，可能输出异常

**执行步骤**:

1. **检查训练日志** (10分钟)
```bash
# H800 上查看 TFT 训练日志
grep -A 20 "TFT" data/gpu_results/train_*.log | tail -50

# 检查关键指标:
# - 训练 loss 是否下降
# - 验证 AUC 是否 > 0.5
# - 是否有 NaN/Inf
```

2. **对比 PyTorch vs ONNX 输出** (30分钟)
```python
# 在 train_gpu.py 添加诊断代码
def diagnose_tft_output(model_path, onnx_path, X_test):
    # PyTorch 推理
    model = TFTModel(...)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        pytorch_out = torch.sigmoid(model(X_test)).numpy()

    # ONNX 推理
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    onnx_out = sess.run(None, {'input': X_test.numpy()})[0]

    # 对比
    print(f"PyTorch output range: [{pytorch_out.min():.4f}, {pytorch_out.max():.4f}]")
    print(f"ONNX output range: [{onnx_out.min():.4f}, {onnx_out.max():.4f}]")
    print(f"Mean absolute diff: {np.abs(pytorch_out - onnx_out).mean():.6f}")
```

3. **检查输入归一化** (20分钟)
```python
# 检查 ml_live_integration.py 中 TFT 的预处理
# 确保与训练时一致
```

4. **修复方案** (1-2小时)
- 如果 ONNX 有问题: 重新导出或使用 PyTorch 推理
- 如果归一化有问题: 修正预处理代码
- 如果训练有问题: 调整超参数重新训练

**预期结果**: TFT 输出范围 [0.3, 0.7]，Stacking 权重变正

---

### 🟡 H800-Fix-3: Stacking 激活路径诊断 (中优先级)

**问题**: OOF 0.5883 → Test 0.5429 (泛化差)

**执行步骤**:

1. **检查元学习器激活** (15分钟)
```python
# 在 train_gpu.py Stacking 训练部分添加
from sklearn.linear_model import LogisticRegression

meta_model = LogisticRegression(max_iter=1000, random_state=42)
meta_model.fit(oof_filled, y_oof)

# 检查: LogisticRegression 内置 sigmoid，不需要手动应用
print(f"Meta model coefficients: {meta_model.coef_}")
print(f"Meta model intercept: {meta_model.intercept_}")

# 测试预测
test_pred = meta_model.predict_proba(test_base_preds)[:, 1]
print(f"Test pred range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
```

2. **检查推理侧激活** (15分钟)
```python
# 在 ml_predictor.py → StackingPredictor.predict() 中
# 确认使用 predict_proba() 而非 predict()
```

3. **添加单元测试** (30分钟)
```python
# test_stacking_consistency.py
def test_stacking_activation():
    # 训练简单 Stacking
    # 验证训练和推理输出一致
    pass
```

**预期结果**: Test AUC 接近 OOF AUC (差距 < 0.02)

---

### 🟢 H800-Fix-4: 过拟合缓解 (高收益)

**方案 1: 数据增强** (2小时实现 + 2小时训练)

```python
# 在 train_gpu.py 添加数据增强函数
def augment_time_series(X, y, augment_ratio=0.5):
    """时间序列数据增强"""
    n_aug = int(len(X) * augment_ratio)
    X_aug, y_aug = [], []

    for _ in range(n_aug):
        idx = np.random.randint(len(X))
        x, label = X[idx], y[idx]

        # 1. 高斯噪声 (50% 概率)
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.01, x.shape)
            x = x + noise

        # 2. 特征 Dropout (30% 概率)
        if np.random.rand() < 0.3:
            mask = np.random.rand(*x.shape) > 0.1
            x = x * mask

        # 3. 时间扭曲 (20% 概率)
        if np.random.rand() < 0.2:
            # 随机拉伸/压缩时间轴
            indices = np.sort(np.random.choice(len(x), len(x), replace=True))
            x = x[indices]

        X_aug.append(x)
        y_aug.append(label)

    return np.vstack([X, X_aug]), np.hstack([y, y_aug])

# 在训练前应用
X_train_aug, y_train_aug = augment_time_series(X_train, y_train, augment_ratio=0.3)
```

**方案 2: 正则化增强** (30分钟)

```python
# LightGBM
lgb_params = {
    'lambda_l1': 0.05,  # 从 0.025 增加
    'lambda_l2': 1.0,   # 从 0.66 增加
    'min_child_samples': 100,  # 从 52 增加
    'feature_fraction': 0.6,  # 从 0.67 降低
}

# LSTM
DROPOUT = 0.4  # 从 0.3 增加
weight_decay = 5e-4  # 从 1e-4 增加
```

**方案 3: Early Stopping 严格化** (15分钟)

```python
# 使用 holdout set 而非 validation set
patience = 5  # 从 10 降低
```

**执行顺序**:
1. 先实现数据增强 (最高收益)
2. 训练 LGB + LSTM 验证效果
3. 如果效果好，应用到所有模型

**预期结果**:
- LGB: Holdout AUC 0.5533 → 0.58
- LSTM: Val AUC 0.5454 → 0.56

---

## 第二阶段: 模型增强 (3-5天)

### 🟡 H800-New-1-v2: Multi-Horizon LSTM 优化

**前置条件**: H800-New-1 训练完成并验证

**优化 1: Focal Loss** (1小时)

```python
# 在 train_gpu.py 添加
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

# 替换 criterion
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

**优化 2: 加权损失** (30分钟)

```python
# 根据时间跨度重要性加权
loss = (focal_loss(pred_5h, yb_5h) * 0.4 +
        focal_loss(pred_12h, yb_12h) * 0.35 +
        focal_loss(pred_24h, yb_24h) * 0.25)
```

**优化 3: 头间多样性正则化** (1小时)

```python
def diversity_loss(pred_5h, pred_12h, pred_24h):
    # 计算预测间的相关性
    preds = torch.stack([pred_5h, pred_12h, pred_24h])
    corr_matrix = torch.corrcoef(preds)

    # 惩罚高相关性 (鼓励多样性)
    off_diag = corr_matrix[torch.triu(torch.ones_like(corr_matrix), diagonal=1) == 1]
    return off_diag.abs().mean()

# 添加到总损失
total_loss = task_loss + 0.1 * diversity_loss(pred_5h, pred_12h, pred_24h)
```

**优化 4: 集成推理** (30分钟)

```python
# 在推理时加权平均 3 个头
def forward(self, x, return_all=False):
    out_5h = self.head_5h(context).squeeze(-1)
    out_12h = self.head_12h(context).squeeze(-1)
    out_24h = self.head_24h(context).squeeze(-1)

    if return_all:
        return out_5h, out_12h, out_24h

    # 加权集成 (根据验证 AUC)
    return 0.4 * out_5h + 0.35 * out_12h + 0.25 * out_24h
```

**执行**: 在 H800 上重新训练

**预期结果**: Val AUC 0.57 → 0.60+

---

### 🟢 H800-New-2: 24h Regime 分类器 (高价值)

**实现** (4小时)

```python
# 在 train_gpu.py 添加新模式
def train_regime_classifier(timeframe='1h'):
    """训练市场 Regime 分类器"""

    # 1. 准备数据
    features, _ = prepare_features(SYMBOL, timeframe)

    # 2. 生成 Regime 标签
    df = load_klines_local(SYMBOL, timeframe)

    # 未来 24h 波动率
    fwd_vol_24h = df['close'].pct_change().rolling(24).std().shift(-24)

    # 三分位数分类
    labels = pd.qcut(fwd_vol_24h, q=3, labels=[0, 1, 2])
    # 0: 低波, 1: 中波, 2: 高波

    # 3. 模型
    class RegimeClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim=128):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, 2,
                                batch_first=True, bidirectional=True)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(64, 3),  # 3 classes
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.classifier(lstm_out[:, -1, :])

    # 4. 训练
    model = RegimeClassifier(input_dim, hidden_dim=128).to(device)
    criterion = nn.CrossEntropyLoss()
    # ... 训练循环 ...

    # 5. 保存
    torch.save(model.state_dict(), 'data/ml_models/regime_classifier_24h.pt')

# 在 main() 中添加
elif args.mode == 'regime':
    result = train_regime_classifier(args.tf)
```

**训练** (2小时)

```bash
python3 train_gpu.py --mode regime --tf 1h
```

**集成到实盘** (2小时)

```python
# 在 live_config.py 中根据 Regime 调整参数
regime = regime_classifier.predict(current_features)

if regime == 0:  # 低波
    leverage = 10
    stop_loss_pct = 0.02
elif regime == 1:  # 中波
    leverage = 5
    stop_loss_pct = 0.03
else:  # 高波
    leverage = 2
    stop_loss_pct = 0.05
```

**预期结果**:
- 分类准确率 > 60%
- 夏普比率 +20%

---

### 🟡 H800-New-3: 15m LSTM

**实现** (2小时)

```python
# 修改 train_lstm() 支持 15m
# 主要调整:
SEQ_LEN = 192  # 48h 历史 (192 * 15m)
HIDDEN_DIM = 128  # 比 1h 小
label = 'profitable_long_3'  # 3 个 15m = 45 分钟
```

**训练** (2小时)

```bash
python3 train_gpu.py --mode lstm --tf 15m
```

**预期结果**: Val AUC 0.52-0.54

---

### 🟢 H800-New-4: 损失函数优化

**已在 H800-New-1-v2 中实现 Focal Loss**

**额外优化: AUC Loss** (2小时)

```python
# 需要安装 libauc
# pip install libauc

from libauc.losses import AUCMLoss
from libauc.optimizers import PESG

criterion = AUCMLoss()
optimizer = PESG(model.parameters(), lr=0.1, momentum=0.9)
```

**预期结果**: AUC +0.01-0.02

---

## 第三阶段: 架构升级 (5-7天)

### 🔵 H800-Arch-1: Transformer 替代 LSTM (高风险高收益)

**实现** (8小时)

```python
class TransformerPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, 200, hidden_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Multi-Horizon 输出
        self.head_5h = nn.Linear(hidden_dim, 1)
        self.head_12h = nn.Linear(hidden_dim, 1)
        self.head_24h = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x[:, -1, :]  # 最后一个时间步

        return self.head_5h(x), self.head_12h(x), self.head_24h(x)
```

**训练** (4小时)

```bash
python3 train_gpu.py --mode transformer --tf 1h
```

**风险**: 训练不稳定，可能需要调参

**预期结果**: Val AUC 0.60+

---

### 🔵 H800-Arch-3: 知识蒸馏 (中风险中收益)

**实现** (6小时)

```python
def train_with_distillation(student_model, teacher_model, X_train, y_train):
    """知识蒸馏训练"""

    # Teacher 预测 (软标签)
    teacher_model.eval()
    with torch.no_grad():
        teacher_probs = torch.sigmoid(teacher_model(X_train))

    # Student 训练
    for epoch in range(EPOCHS):
        student_model.train()
        for xb, yb in train_loader:
            # 获取 teacher 软标签
            with torch.no_grad():
                teacher_soft = torch.sigmoid(teacher_model(xb))

            # Student 预测
            student_logits = student_model(xb)

            # 蒸馏损失
            loss = distillation_loss(
                student_logits,
                teacher_soft,
                yb,
                alpha=0.5,  # 软硬标签权重
                T=2.0       # 温度
            )

            loss.backward()
            optimizer.step()
```

**预期结果**: LSTM AUC 0.5454 → 0.57+

---

## 执行时间表

| 日期 | 任务 | 预计耗时 | 负责人 |
|------|------|---------|--------|
| Day 1 | H800-Fix-2 (TFT 诊断) | 2h | H800 |
| Day 1 | H800-Fix-3 (Stacking 诊断) | 1h | H800 |
| Day 1 | H800-Fix-4 (数据增强实现) | 2h | H800 |
| Day 1 | H800-New-1 训练验证 | 2h | H800 |
| Day 2 | H800-Fix-4 (重新训练 LGB/LSTM) | 4h | H800 |
| Day 2 | H800-New-1-v2 (Focal Loss) | 2h | H800 |
| Day 2 | H800-New-1-v2 训练 | 2h | H800 |
| Day 3 | H800-New-2 (Regime 实现) | 4h | H800 |
| Day 3 | H800-New-2 训练 | 2h | H800 |
| Day 3 | H800-New-3 (15m LSTM) | 2h | H800 |
| Day 4 | H800-New-3 训练 | 2h | H800 |
| Day 4 | H800-Arch-1 (Transformer 实现) | 6h | H800 |
| Day 5 | H800-Arch-1 训练调试 | 8h | H800 |
| Day 6 | H800-Arch-3 (知识蒸馏) | 6h | H800 |
| Day 7 | 模型回传 + 推理集成 | 4h | 本机 |

---

## 检查点

### Day 1 结束
- [ ] TFT 问题已诊断
- [ ] Stacking 激活路径已验证
- [ ] 数据增强代码已实现
- [ ] Multi-Horizon LSTM 训练完成

### Day 3 结束
- [ ] 过拟合问题已缓解 (Holdout AUC > 0.58)
- [ ] Multi-Horizon LSTM 优化完成 (Val AUC > 0.57)
- [ ] Regime 分类器训练完成

### Day 5 结束
- [ ] 15m LSTM 训练完成
- [ ] Transformer 模型训练完成

### Day 7 结束
- [ ] 所有模型已回传
- [ ] 推理侧已集成
- [ ] 回测验证通过

---

**创建时间**: 2026-02-20
**预计完成**: 2026-02-27
