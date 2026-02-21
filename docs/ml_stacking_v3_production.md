# Stacking v4 生产部署文档

## 概述

Stacking Ensemble v4 是当前生产环境中的 ML 方向预测核心模块。3 个异构基模型的 OOF 预测通过 LogisticRegression 元学习器融合，输出 bull_prob（看多概率）。

## 架构

```
ml_features.py (94 维特征)
    ↓
┌─ 基模型 1: LGB Direction    (73维, stacking_lgb_1h.txt)
├─ 基模型 2: LSTM+Attention    (73维, stacking_lstm_1h.pt, 延迟加载)
├─ 基模型 3: 跨资产 LGB        (94维, stacking_lgb_cross_1h.txt)
└─ 附加特征: hvol_20           (z-score 标准化)
    ↓
元学习器: LogisticRegression(C=0.5, class_weight=None, L2)
    → stacking_meta_1h.pkl
    ↓
bull_prob → MLSignalEnhancer → SS/BS 调整
```

## 版本变更历史

### v3 → v4

| 变更 | 说明 |
|------|------|
| 移除 XGBoost | 与 LGB 高度相关，系数 -0.04，移除后减少多重共线性 |
| 移除 TFT | 负系数 -0.10，多重共线性导致，移除后简化架构 |
| 折内 label_smooth | 0 → 0.10，与全量重训对齐，消除 OOF/推理概率分布不一致 |
| 全量 label_smooth | 0.05 → 0.10，统一平滑系数 |
| 基模型数 | 5 → 3 (LGB + LSTM + CrossAsset LGB) |

### v2 → v3

| 参数 | v2 | v3 | 说明 |
|------|----|----|------|
| class_weight | `'balanced'` | `None` | 去除 balanced 修正 +13.4% 看多偏差 |
| C | 0.3 | 0.5 | 适度 L2 正则化 |
| Platt Scaling | 无 | 有 (A≈1, B≈0) | 校准已内置，当前 enabled=false (无需) |

## v3 性能指标 (基准)

| 指标 | v2 | v3 |
|------|----|----|
| OOF Meta AUC | 0.5880 | **0.5893** |
| Test AUC | 0.5466 | 0.5437 |
| 样本数 | 24,492 | 24,492 |
| 训练时间 | ~2min | ~2min |

> v4 预期: OOF AUC 可能略降 (移除 2 个模型)，但 OOF-Test 差距应缩小 (减少过拟合)，推理稳定性提升 (消除 lstm=- 问题)。

## 环境变量（systemd override）

```ini
ML_ENABLE_STACKING=1              # 总开关
ML_STACKING_TIMEFRAME=1h          # 目标周期
ML_STACKING_MIN_OOF_SAMPLES=20000 # 最小 OOF 样本数
ML_STACKING_MIN_VAL_AUC=0.53      # 最小验证 AUC
ML_STACKING_MIN_TEST_AUC=0.52     # 最小测试 AUC
ML_STACKING_MIN_OOF_AUC=0.53      # 最小 OOF AUC
ML_STACKING_MAX_OOF_TEST_GAP=0.10 # OOF/Test 最大差距
ML_STACKING_MIN_FEATURE_COVERAGE_73=0.90  # 73维覆盖率门槛
ML_STACKING_MIN_FEATURE_COVERAGE_94=0.78  # 94维覆盖率门槛
```

## 推理流程（ml_live_integration.py）

1. 质量门控 (`_stacking_quality_gate`): 检查 AUC、样本数、OOF-Test 差距
2. 特征对齐: 73 维基础特征 + 94 维跨资产特征
3. 3 基模型预测: LGB 即时推理, LSTM 延迟加载, CrossAsset LGB 即时推理
4. 附加特征: hvol_20 z-score 标准化 (使用训练时的 mean/std)
5. 元学习器: `predict_proba(meta_X)[0, 1]` → bull_prob
6. 方向判定: bull_prob > 0.58 → 看多，< 0.42 → 看空

## 已知问题

| 问题 | 状态 | 影响 |
|------|------|------|
| LSTM 极端概率被过滤 (lstm=-) | v4 修复中 | label_smooth 0.10 应缓解极端 logit |
| Platt 推理端未接入 | 待实现 | 当前 enabled=false，暂无影响 |

## 重训流程

```bash
# H800 上 (推荐通过 GitHub 同步):
git pull origin main
python3 train_gpu.py --mode stacking --tf 1h
# 验证: cat data/gpu_results/stacking_ensemble_*.json | tail -1
git add data/ml_models/stacking_* data/gpu_results/stacking_*
git commit -m "feat: Stacking v4 retrain"
git push origin main

# 本机:
git pull origin main
./deploy.sh
```

## 部署检查清单

- [ ] `stacking_meta_1h.json` version = `stacking_v4`
- [ ] `stacking_meta_1h.json` base_models 仅含 `['lgb', 'lstm', 'cross_asset_lgb']`
- [ ] `check_ml_health.py --timeframe 1h` 全部通过
- [ ] `systemctl show macd-engine --property=Environment` 含 `ML_ENABLE_STACKING=1`
- [ ] 引擎日志出现 `(stk: lgb=... lstm=... cross=...)`
- [ ] `git_dirty: false` 在 RUNTIME MANIFEST 中
