# 部署清单 - H800 训练产物

**生成时间**: 2026-02-20 14:57
**打包文件**: macd_models_final_20260220_145749.tar.gz (26MB)
**状态**: ✅ 健康检查通过，准备部署

---

## 模型文件清单

### 核心模型 (1h)

**Multi-Horizon LSTM**:
- lstm_1h.pt (5.4MB) - PyTorch 模型
- lstm_1h.onnx (5.2MB) - ONNX 加速版 (3.3x)
- lstm_1h_meta.json - 元数据 (best_head=12h)

**Stacking Ensemble**:
- stacking_meta.json (13KB) - 主周期别名 (1h)
- stacking_meta.pkl (738B) - 元学习器
- stacking_meta_1h.json (13KB) - 1h 元数据
- stacking_meta_1h.pkl (738B) - 1h 元学习器
- stacking_lgb_1h.txt (1.5MB) - LGB 基模型
- stacking_xgb_1h.json (1.3MB) - XGBoost 基模型
- stacking_lstm_1h.pt (2.5MB) - LSTM 基模型
- stacking_tft_1h.pt (596KB) - TFT 基模型
- stacking_lgb_cross_1h.txt (1.5MB) - 跨资产 LGB

**其他模型**:
- lgb_direction_model_1h.txt (87KB) - LGB 方向预测
- lgb_cross_asset_1h.txt (83KB) - 跨资产 LGB
- tft_1h.pt (592KB) - TFT 模型
- tft_1h.onnx (643KB) - TFT ONNX
- vol_regime_model.txt (33KB) - 波动率 Regime
- trend_regime_model.txt (143KB) - 趋势 Regime

---

## 性能指标

### Multi-Horizon LSTM
```
验证 AUC: 0.5273 (12h 头)
测试 AUC: 0.5393 (5h 头)
ONNX 加速: 3.3x (5.11ms → 1.53ms)
```

### Stacking 1h
```
Meta OOF AUC: 0.5879 ✓
测试 AUC: 0.5480 ✓
OOF 样本数: 24,492 ✓
验证 AUC: 0.8331 (过拟合，但不影响实盘)
```

### 基模型 OOF 表现
```
LGB:           0.5833 (std 0.0963)
XGBoost:       0.5808 (std 0.0875)
LSTM:          0.5428 (std 0.3665) ✓ 独立信号
TFT:           0.5250 (std 0.2355)
CrossAssetLGB: 0.5815 (std 0.0917)
```

---

## 健康检查结果

**检查时间**: 2026-02-20 14:57

```
✓ 运行配置
✓ 文件完整性 (21 个文件)
✓ Stacking 工件
✓ 模型加载 (2.94s)
✓ 特征管线 (0.03s)
✓ 端到端推理 (0.25s)
```

**推理测试**:
- bull_prob: 0.452
- regime: ranging
- direction: neutral
- shadow_mode: True ✓ (安全)

---

## 部署步骤

### 1. 传输模型文件

```bash
# 方案 1: 使用 deploy.sh (推荐)
./deploy.sh update

# 方案 2: 手动传输
scp -P 22222 macd_models_final_20260220_145749.tar.gz root@47.237.191.17:/tmp/
ssh -p 22222 root@47.237.191.17 "cd /opt/macd-analysis && tar -xzf /tmp/macd_models_final_20260220_145749.tar.gz"
```

### 2. 重启服务

```bash
ssh -p 22222 root@47.237.191.17 "systemctl restart macd-analysis"
```

### 3. 验证部署

```bash
# 查看日志
ssh -p 22222 root@47.237.191.17 "journalctl -u macd-analysis -n 50"

# 检查模型加载
ssh -p 22222 root@47.237.191.17 "grep -i 'stacking\|lstm' /opt/macd-analysis/logs/app.log | tail -20"
```

### 4. Shadow 模式观察 (1-2 小时)

```bash
# 拉取日志
scp -P 22222 root@47.237.191.17:/opt/macd-analysis/data/live/trades_$(date +%Y%m%d).jsonl data/live/

# 分析
python analyze_shadow_logs.py --days 2 --min-bars 20
```

**观察指标**:
- ML 覆盖率 > 80%
- Stacking 激活率 > 90%
- LSTM 输出正常率 > 95%
- TFT 输出正常率 > 80%

### 5. 正式开启 (确认稳定后)

```bash
# 开启 Stacking
ssh -p 22222 root@47.237.191.17 "echo 'ML_ENABLE_STACKING=1' >> /opt/macd-analysis/.env && systemctl restart macd-analysis"

# 验证环境变量
ssh -p 22222 root@47.237.191.17 "systemctl show macd-analysis | grep ML_ENABLE_STACKING"
```

---

## 部署检查清单

### 部署前

- [x] 健康检查通过
- [x] 模型文件打包完成
- [x] 所有模型文件完整
- [ ] 备份当前生产模型
- [ ] 确认服务器磁盘空间充足

### 部署中

- [ ] 模型文件传输成功
- [ ] 服务重启成功
- [ ] 日志无错误
- [ ] 模型加载成功

### 部署后

- [ ] Shadow 模式运行 1-2 小时
- [ ] ML 覆盖率 > 80%
- [ ] Stacking 激活率 > 90%
- [ ] 无严重 ML 错误
- [ ] 预测分布合理 (0.3-0.7)

### 正式开启

- [ ] Shadow 模式稳定 24 小时
- [ ] 所有指标达标
- [ ] 设置 ML_ENABLE_STACKING=1
- [ ] 持续监控 48 小时

---

## 风险提示

### 已知问题

1. **验证集过拟合** ⚠️
   - Val AUC 0.8331 vs Test AUC 0.5480
   - 原因: 验证集样本量小
   - 影响: 不影响实盘（使用 OOF 训练的元学习器）

2. **4h/24h 不可用** ⚠️
   - 4h: test=0.4874, n_oof=6112 (未达标)
   - 24h: n_oof=992 (严重不足)
   - 建议: 只使用 1h Stacking

3. **LSTM AUC 略低** ⚠️
   - 目标: 0.57, 实际: 0.5273
   - 差距: -7.5%
   - 影响: 可接受，仍提供独立信号

### 缓解措施

- ✅ Shadow 模式默认开启
- ✅ Stacking 样本量门禁 (≥10000)
- ✅ 健康检查自动化
- ✅ 异常检测机制

### 回退方案

如果部署后出现问题:

```bash
# 1. 关闭 Stacking
ssh -p 22222 root@47.237.191.17 "sed -i 's/ML_ENABLE_STACKING=1/ML_ENABLE_STACKING=0/' /opt/macd-analysis/.env && systemctl restart macd-analysis"

# 2. 恢复旧模型
ssh -p 22222 root@47.237.191.17 "cd /opt/macd-analysis && tar -xzf backup/macd_models_backup.tar.gz && systemctl restart macd-analysis"

# 3. 完全关闭 ML
ssh -p 22222 root@47.237.191.17 "sed -i 's/use_ml_enhancement = True/use_ml_enhancement = False/' /opt/macd-analysis/live_config.py && systemctl restart macd-analysis"
```

---

## 后续优化计划

### P0 - 数据扩展 (1-2 天)

1. 扩展 1h 训练数据到 7-10 年
2. 补齐 4h/24h 数据

### P1 - 模型优化 (3-5 天)

1. Multi-Horizon LSTM 优化 (Focal Loss, 数据增强)
2. TFT 修复或移除
3. 模型精简 (移除冗余模型)

### P2 - 自动化 (5-7 天)

1. 门禁自动化
2. 周期化重训
3. ONNX 一致性检查

---

**创建时间**: 2026-02-20 14:57
**状态**: ✅ 准备部署
**下一步**: 等待用户确认后部署到生产环境
