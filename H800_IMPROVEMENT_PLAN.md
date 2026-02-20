# H800 训练系统改进计划

## 当前系统诊断

### 性能现状

| 模型 | 验证 AUC | 测试/Holdout AUC | 过拟合程度 | 状态 |
|------|---------|-----------------|-----------|------|
| Optuna LGB | 0.6055 | 0.5533 | -0.052 ⚠️ | 严重过拟合 |
| Stacking 1h | 0.5577 | 0.5429 | -0.015 | 轻度过拟合 |
| LSTM 1h | 0.5454 | ? | ? | 性能不佳 |
| TFT 1h | ? | ? | ? | 🔴 输出 0.0 (失效) |
| Multi-Horizon LSTM | 待训练 | 待训练 | ? | ✅ 刚实现 |

### 关键问题

1. **过拟合严重** (P0)
   - Optuna LGB: Val 0.6055 → Holdout 0.5533 (下降 8.6%)
   - Stacking: OOF 0.5883 → Test 0.5429 (下降 7.7%)
   - 原因: 数据泄露、特征过拟合、模型复杂度过高

2. **LSTM 性能不佳** (P1)
   - Val AUC 0.5454，远低于 LGB 0.6055
   - 增加 hidden_dim 192 后反而过拟合
   - 原因: 序列建模能力不足、标签不匹配

3. **TFT 完全失效** (P0)
   - 输出恒为 0.0
   - 原因: ONNX 转换问题 / 输入归一化问题

4. **样本不足** (P2)
   - 4h: ~6000 样本
   - 24h: ~1000 样本
   - 影响长周期模型训练

---

## 改进计划

### P0 - 紧急修复 (1-2天)

#### H800-Fix-2: TFT 输出 0.0 诊断

**问题**: TFT 模型输出恒为 0.0，在 Stacking 中权重为负 (-0.124)

**诊断步骤**:
1. 检查 TFT 训练日志，确认训练时 loss 是否下降
2. 检查 ONNX 转换前后输出是否一致
3. 检查输入特征归一化是否正确
4. 对比 PyTorch 原生推理 vs ONNX 推理

**修复方案**:
- 如果是 ONNX 问题: 重新导出或使用 PyTorch 推理
- 如果是归一化问题: 修正 `ml_live_integration.py` 的预处理
- 如果是训练问题: 调整 TFT 超参数 (learning_rate, hidden_size)

**预期收益**: Stacking AUC +0.01~0.02

---

#### H800-Fix-3: Stacking 激活路径诊断

**问题**: Stacking 泛化差 (OOF 0.5883 → Test 0.5429)

**诊断步骤**:
1. 检查元学习器是否使用了 sigmoid 激活 (LogisticRegression 内置)
2. 检查推理时是否正确应用 sigmoid
3. 对比训练时和推理时的激活路径

**代码位置**:
- 训练: `train_gpu.py` Stacking 模式
- 推理: `ml_predictor.py` → `StackingPredictor`

**修复方案**:
- 确保训练和推理使用相同的激活函数
- 添加单元测试验证一致性

**预期收益**: 修复后 Test AUC 应接近 OOF AUC

---

#### H800-Fix-4: 过拟合缓解

**问题**: 所有模型都存在过拟合

**方案 1: 数据增强** (优先)
```python
# 时间序列数据增强
def augment_sequences(X, y):
    # 1. 添加高斯噪声
    X_noise = X + np.random.normal(0, 0.01, X.shape)

    # 2. 时间扭曲 (time warping)
    X_warp = time_warp(X, sigma=0.2)

    # 3. 特征 dropout (随机遮蔽部分特征)
    X_dropout = feature_dropout(X, p=0.1)

    return np.vstack([X, X_noise, X_warp, X_dropout]), np.tile(y, 4)
```

**方案 2: 正则化增强**
- LightGBM: 增加 `lambda_l1`, `lambda_l2`
- LSTM: 增加 Dropout (0.3 → 0.4), Weight Decay (1e-4 → 5e-4)
- Stacking: 使用 Ridge/Lasso 替代 LogisticRegression

**方案 3: Early Stopping 严格化**
- 当前: patience=10
- 优化: patience=5, 使用 holdout set 而非 validation set

**预期收益**: 过拟合降低 3-5%

---

### P1 - 模型增强 (3-5天)

#### H800-New-1-v2: Multi-Horizon LSTM 优化

**当前实现问题**:
1. 简单平均损失: `loss = (loss_5h + loss_12h + loss_24h) / 3.0`
2. 没有头间多样性正则化
3. 推理时只用单个最佳头

**优化方案**:

**1. 加权损失** (根据样本正负比)
```python
# 计算每个头的正样本比例
pos_ratio_5h = y_train_5h.mean()
pos_ratio_12h = y_train_12h.mean()
pos_ratio_24h = y_train_24h.mean()

# 使用 Focal Loss 处理类别不平衡
from torch.nn import functional as F

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()

loss = (focal_loss(pred_5h, yb_5h) * 0.4 +
        focal_loss(pred_12h, yb_12h) * 0.35 +
        focal_loss(pred_24h, yb_24h) * 0.25)
```

**2. 头间多样性正则化**
```python
# 鼓励不同头学习不同模式
def diversity_loss(pred_5h, pred_12h, pred_24h):
    # 计算头间相关性
    corr_5_12 = torch.corrcoef(torch.stack([pred_5h, pred_12h]))[0, 1]
    corr_5_24 = torch.corrcoef(torch.stack([pred_5h, pred_24h]))[0, 1]
    corr_12_24 = torch.corrcoef(torch.stack([pred_12h, pred_24h]))[0, 1]

    # 惩罚高相关性
    return (corr_5_12.abs() + corr_5_24.abs() + corr_12_24.abs()) / 3.0

total_loss = task_loss + 0.1 * diversity_loss(pred_5h, pred_12h, pred_24h)
```

**3. 集成推理** (替代单头选择)
```python
# 加权平均 3 个头的预测
final_pred = (0.4 * pred_5h + 0.35 * pred_12h + 0.25 * pred_24h)
```

**预期收益**: AUC 0.54 → 0.58+

---

#### H800-New-2: 24h Regime 分类器

**目标**: 预测未来 24h 的市场状态 (低波/中波/高波)

**架构**:
```python
class RegimeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 2,
                            batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),  # 3 classes: low/medium/high volatility
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.classifier(lstm_out[:, -1, :])
```

**标签定义**:
```python
# 未来 24h 波动率
fwd_vol_24h = df['close'].pct_change().rolling(24).std().shift(-24)

# 三分位数分类
labels = pd.qcut(fwd_vol_24h, q=3, labels=[0, 1, 2])
# 0: 低波 (< 33%)
# 1: 中波 (33%-66%)
# 2: 高波 (> 66%)
```

**用途**:
- 实盘根据 Regime 动态调整仓位
- 低波: 高杠杆 (10x)
- 中波: 中杠杆 (5x)
- 高波: 低杠杆 (2x) 或观望

**预期收益**: 夏普比率 +20%

---

#### H800-New-3: 15m LSTM

**目标**: 捕捉短期价格动量

**优势**:
- 15m 数据量充足 (~100K 样本)
- 适合日内交易
- 可与 1h/4h 形成多周期共识

**配置**:
```python
SEQ_LEN = 192  # 48h 历史 (192 * 15m = 48h)
HIDDEN_DIM = 128  # 比 1h 小 (数据噪声大)
EPOCHS = 30
```

**标签**: `profitable_long_3` (3 个 15m = 45 分钟持仓)

**预期收益**: AUC 0.52-0.54 (短周期难度大)

---

#### H800-New-4: 损失函数优化

**当前问题**: 使用标准 BCEWithLogitsLoss，未考虑类别不平衡

**优化方案**:

**1. Focal Loss** (处理类别不平衡)
```python
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
```

**2. AUC Loss** (直接优化 AUC)
```python
# 使用 LibAUC 库
from libauc.losses import AUCMLoss

criterion = AUCMLoss()
```

**3. 分位数损失** (用于分位数回归模型)
```python
def quantile_loss(pred, target, quantile=0.5):
    error = target - pred
    return torch.max(quantile * error, (quantile - 1) * error).mean()
```

**预期收益**: AUC +0.01-0.02

---

### P2 - 架构升级 (5-7天)

#### H800-Arch-1: Transformer 替代 LSTM

**动机**: LSTM 性能不佳 (0.5454)，Transformer 可能更适合金融时序

**架构**: Temporal Fusion Transformer (TFT) 增强版

```python
class EnhancedTFT(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        # 1. 输入嵌入
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 2. 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 4. Multi-Horizon 输出头
        self.head_5h = nn.Linear(hidden_dim, 1)
        self.head_12h = nn.Linear(hidden_dim, 1)
        self.head_24h = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)

        # 使用最后一个时间步
        x = x[:, -1, :]

        return self.head_5h(x), self.head_12h(x), self.head_24h(x)
```

**预期收益**: AUC 0.54 → 0.60+

---

#### H800-Arch-2: 对比学习

**动机**: 学习更鲁棒的特征表示

**方法**: SimCLR for Time Series

```python
class ContrastiveLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=192):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x1, x2):
        # x1, x2: 同一序列的两个增强版本
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        return z1, z2

# NT-Xent Loss (对比损失)
def nt_xent_loss(z1, z2, temperature=0.5):
    z = torch.cat([z1, z2], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim = sim / temperature

    # 正样本: (z1[i], z2[i])
    # 负样本: 其他所有
    labels = torch.arange(len(z1)).to(z.device)
    labels = torch.cat([labels + len(z1), labels])

    loss = F.cross_entropy(sim, labels)
    return loss
```

**训练流程**:
1. 预训练: 对比学习 (无监督)
2. 微调: 分类任务 (有监督)

**预期收益**: 泛化能力 +10%

---

#### H800-Arch-3: 知识蒸馏

**动机**: 将大模型 (Optuna LGB 0.6055) 的知识迁移到小模型 (LSTM)

**方法**:
```python
# Teacher: Optuna LGB (AUC 0.6055)
# Student: LSTM (AUC 0.5454)

def distillation_loss(student_logits, teacher_probs, labels, alpha=0.5, T=2.0):
    # 软标签损失 (从 teacher 学习)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_probs / T, dim=1),
        reduction='batchmean'
    ) * (T ** 2)

    # 硬标签损失 (从真实标签学习)
    hard_loss = F.binary_cross_entropy_with_logits(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss
```

**预期收益**: LSTM AUC 0.5454 → 0.57+

---

#### H800-Arch-4: 集成策略优化

**当前问题**: Stacking 简单线性组合，未充分利用模型多样性

**优化方案**:

**1. 动态加权** (根据市场状态)
```python
class DynamicEnsemble(nn.Module):
    def __init__(self, n_models=5):
        super().__init__()
        # 根据市场特征动态生成权重
        self.weight_net = nn.Sequential(
            nn.Linear(10, 32),  # 10: 市场特征 (波动率, 趋势强度等)
            nn.ReLU(),
            nn.Linear(32, n_models),
            nn.Softmax(dim=1)
        )

    def forward(self, base_preds, market_features):
        weights = self.weight_net(market_features)
        return (base_preds * weights).sum(dim=1)
```

**2. Boosting 替代 Stacking**
```python
# 使用 AdaBoost 思想
# 每个模型关注前一个模型的错误样本
```

**预期收益**: Stacking AUC 0.5577 → 0.60+

---

### P3 - 数据优化 (并行进行)

#### H800-Data-1: 样本扩充

**方案 1: 扩展历史数据**
- 当前: 5 年
- 目标: 7-10 年 (如果 Binance 有)

**方案 2: 多交易对训练**
```python
# 联合训练 ETH/BTC/SOL/BNB
# 共享 LSTM 编码器，独立输出头
```

**方案 3: 滑动窗口增强**
```python
# 当前: 固定 SEQ_LEN=48
# 优化: 随机 SEQ_LEN ∈ [36, 60]
```

**预期收益**: 4h/24h 样本 +50%

---

#### H800-Data-2: 特征工程

**新增特征类别**:

1. **订单簿特征** (如果可获取)
   - Bid-Ask Spread
   - Order Book Imbalance
   - Depth at different levels

2. **链上特征** (ETH)
   - Gas Price
   - Active Addresses
   - Exchange Inflow/Outflow

3. **情绪特征**
   - Twitter Sentiment (需要 API)
   - Fear & Greed Index

4. **宏观特征**
   - DXY (美元指数)
   - Gold Price
   - US10Y (美债收益率)

**预期收益**: AUC +0.02-0.03

---

## 执行优先级

### 第一阶段 (1-2天): 紧急修复
```
H800-Fix-2: TFT 输出 0.0 诊断          [4h]
H800-Fix-3: Stacking 激活路径诊断      [2h]
H800-Fix-4: 过拟合缓解 (数据增强)      [6h]
H800-New-1: Multi-Horizon LSTM 训练    [2h]
```

### 第二阶段 (3-5天): 模型增强
```
H800-New-1-v2: Multi-Horizon 优化      [8h]
H800-New-2: 24h Regime 分类器          [6h]
H800-New-3: 15m LSTM                   [4h]
H800-New-4: 损失函数优化 (Focal Loss)  [4h]
```

### 第三阶段 (5-7天): 架构升级
```
H800-Arch-1: Transformer 替代 LSTM     [12h]
H800-Arch-3: 知识蒸馏                  [8h]
H800-Arch-4: 集成策略优化              [8h]
```

### 第四阶段 (并行): 数据优化
```
H800-Data-1: 样本扩充                  [4h]
H800-Data-2: 特征工程                  [8h]
```

---

## 成功指标

| 指标 | 当前 | 目标 | 改进 |
|------|------|------|------|
| LGB Val AUC | 0.6055 | 0.60 | 保持 (降低过拟合) |
| LGB Holdout AUC | 0.5533 | 0.58 | +0.047 |
| LSTM Val AUC | 0.5454 | 0.58 | +0.035 |
| Stacking Val AUC | 0.5577 | 0.62 | +0.062 |
| Stacking Test AUC | 0.5429 | 0.60 | +0.057 |
| 过拟合程度 | -8.6% | -3% | 改善 5.6% |

---

## 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| Transformer 训练不稳定 | 中 | 高 | 使用预训练权重, 降低学习率 |
| 数据增强引入噪声 | 中 | 中 | A/B 测试, 逐步增加增强强度 |
| 新模型过拟合更严重 | 高 | 高 | 严格 Early Stopping, 使用 Holdout |
| H800 GPU 资源不足 | 低 | 高 | 优先训练小模型, 使用混合精度 |

---

**创建时间**: 2026-02-20
**预计完成**: 2026-02-27 (7天)
**负责人**: H800 训练团队
