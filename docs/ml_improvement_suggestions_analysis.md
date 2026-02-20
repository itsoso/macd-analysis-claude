# ML 后续改进建议 — 分析与落地对照

基于 3 轮迭代的改进路线图与当前代码库的对照分析，便于按优先级落地。

---

## 一、短期改进 (1–2 周)

### 1. 扩大数据集 ⭐⭐⭐⭐⭐

| 建议 | 当前状态 | 落地要点 |
|------|----------|----------|
| 180 天 → 1–2 年，目标 10000+ 样本 | **部分满足** | 见下 |

- **ml/config.py**：`FETCH_DAYS = 730`（2 年），仅用于 `ml/` 子模块（prepare_data、download_data）。
- **train_gpu.py**：`prepare_features()` 使用 **本地 Parquet** `data/klines/{symbol}/{interval}.parquet`，**不写死天数**，样本量 = 当前缓存长度。
- **结论**：若本地 Parquet 只含约 180 天，需先扩充缓存到 1–2 年再训练。

**建议操作**：

- 用 `ml/download_data.py` 或 `binance_fetcher` 拉取 **730 天**（或 365）写入 `data/klines/ETHUSDT/1h.parquet` 等，再跑 `train_gpu.py`。
- 或在 `train_gpu.py` 中增加“无本地时自动按 FETCH_DAYS 拉取”的路径（可选），统一数据口径。
- 1h 周期 730 天 ≈ 17520 条，去 NaN 后通常 >10000，满足“10000+ 样本”目标。

---

### 2. 特征选择优化

| 建议 | 当前状态 | 落地要点 |
|------|----------|----------|
| SHAP 分析特征重要性 | **未使用** | 新增 SHAP 分析脚本或步骤 |
| 相关性 >0.85 去冗余 | **已实现** | 保持 |
| 保留 Top 30–40 | **已实现** | 可微调 |

- **ml_features.py**：`select_features()` 已实现  
  - 相关性过滤：`corr_threshold=0.85`，保留与标签相关更高的特征；  
  - 重要性：用 **LightGBM gain** 排序，`importance_top_n=35`。
- **ml_predictor.py**：`feature_corr_threshold=0.85`，`feature_top_n=30`。

**建议操作**：

- 增加 **SHAP**（TreeExplainer）分析脚本：在 1h 全量特征上跑一次，输出特征重要性排序与可选 Top 30–40 列表，与现有 LGB importance 对比；数据量 >1 万时可用子采样以控制耗时。
- 保持现有 0.85 与 Top 30–40 逻辑，仅将“重要性来源”从纯 LGB 扩展为“LGB + SHAP 可选”。

---

### 3. 正则化增强

| 建议 | 当前状态 | 落地要点 |
|------|----------|----------|
| L1/L2 增强 | **已有** | 生产用 Optuna 结果，可做 A/B |
| num_leaves 20→15 | **部分** | WF 用 20，可试 15 |
| min_child_samples 50→100 | **部分** | 可试 50–100 |

- **ml_predictor.py (WF/实盘)**：`num_leaves=20`，`min_child_samples=30`，`lambda_l1=0.5`，`lambda_l2=1.0`（已偏强）。
- **train_gpu / train_production_model.py（生产/Stacking）**：Optuna 最优 `num_leaves=34`，`min_child_samples=56`，`lambda_l1=0.0114`，`lambda_l2=0.2146`。
- **ml_features.select_features 内部 LGB**：`num_leaves=15`，`min_child_samples=50`。

**建议操作**：

- 短期：在 **ml_predictor** 或单独回测脚本中增加一组“更强正则”配置（如 num_leaves=15，min_child_samples=50 或 100），与当前配置做 OOS 对比，再决定是否切到生产。
- 生产/Stacking 参数建议保留 Optuna 结果为主，仅在明确过拟合时再收紧（如 num_leaves 上限 24、min_child_samples 下限 80）。

---

## 二、中期改进 (1–2 月)

### 4. 多交易对数据融合

| 建议 | 当前状态 | 落地要点 |
|------|----------|----------|
| BTC/SOL/BNB 完整数据 + 跨资产特征 | **已实现** | 保证数据与训练入口一致 |

- **train_gpu.py**：`_add_cross_asset_features()` 已为 ETH 主特征增加 btc/sol/bnb 的 ret、corr、rel_strength、vol_ratio 等；`train_cross_asset()`、`train_tft()` 已使用。
- **数据**：跨资产 K 线需存在于 `data/klines/BTCUSDT/`、`SOLUSDT/`、`BNBUSDT/` 等（或统一从某处加载）。

**建议操作**：确保各交易对 Parquet 与 ETH 时间范围对齐（如均为 730 天），再跑 `train_cross_asset` / `train_tft`；可选增加更多交叉特征（如 cross_vol 等）做一次消融。

---

### 5. 多周期数据融合

| 建议 | 当前状态 | 落地要点 |
|------|----------|----------|
| 15m/1h/4h/24h 联合建模 | **已有** | 扩展特征与评估 |

- **train_gpu.py**：`train_mtf_fusion()` 已做多周期（各 TF 的 ss/bs/net/max + large_small_agree 等）融合训练。
- **multi_tf_consensus.py**：实盘多周期权重（24h=28, 15m=3 等）为规则融合，非同一套“多周期联合模型”。

**建议操作**：在现有 `train_mtf_fusion` 上增加 15m（若数据已有），或做 1h/4h/24h 联合特征对齐与评估；中长期可考虑“多周期联合特征 + 单模型”与当前“多 TF 规则共识”做 A/B。

---

### 6. 在线学习 (Incremental Learning)

| 建议 | 当前状态 | 落地要点 |
|------|----------|----------|
| 增量更新、滚动窗口 | **已实现** | 运维化与监控 |

- **train_gpu.py**：`train_incremental_wf()` 滚动窗口 Walk-Forward；`train_online_retrain()` 设计为每日 cron 全量/增量重训并写 `retrain_log.jsonl`。

**建议操作**：将 `train_online_retrain` 挂到 cron（如 0 4 * * *），并加简单监控（如 retrain_log 最后一条的 AUC/action）；可选增加“仅用最近 N 天”的增量模式以节省时间。

---

### 7. 集成学习改进版

| 建议 | 当前状态 | 落地要点 |
|------|----------|----------|
| 数据量扩大后重试 Stacking / Bagging | **已有 Stacking** | 数据扩到 1 万+ 后重跑并对比 |

- **train_gpu.py**：`train_stacking_ensemble()` 已实现 LGB + XGB + Ridge 等 OOF + 元学习器；当前生产也使用 Stacking 系模型。

**建议操作**：在“短期 1”完成数据扩展到 1–2 年后，用同一份数据重新跑 `train_stacking_ensemble`，并可选增加纯 Bagging（同模型多子样）对比，记录 OOS AUC/夏普等，再决定是否切换生产权重。

---

## 三、长期改进 (3–6 月)

### 8. 深度学习模型 (50000+ 样本)

| 建议 | 当前状态 | 落地要点 |
|------|----------|----------|
| Transformer-XL / Informer / N-BEATS | **已有 LSTM、TFT** | 样本足够再上大模型 |

- **train_gpu.py**：已有 LSTM+Attention、TFT 等序列模型；输入 lookback=96，多尺度标签。
- **ml/config.py**：FETCH_DAYS=730，LOOKBACK=96 等已就绪。

**建议操作**：样本量达到 50000+ 后，再引入 Transformer-XL/Informer/N-BEATS 等；当前可先做 LSTM/TFT 与 LGB 的融合权重优化。

---

### 9. 强化学习

| 建议 | 当前状态 | 落地要点 |
|------|----------|----------|
| PPO 仓位 / 动态止损止盈 | **已有** | 与规则策略对比 |

- **train_gpu.py**：`train_ppo_position()` 已实现 PPO 仓位优化。

**建议操作**：在回测/模拟盘里系统对比“PPO 仓位 + 现有信号”与“固定仓位 + 现有信号”的夏普与回撤，再决定是否上线。

---

### 10. 多模态融合

| 建议 | 当前状态 | 落地要点 |
|------|----------|----------|
| 价格 + 情绪/链上/订单簿 | **未实现** | 需数据管道与标注 |

- 当前仅价格/量/资金费率/OI 等，无 Twitter、链上、订单簿深度。

**建议操作**：作为独立项目阶段，先选一种数据源（如订单簿深度），做小规模特征工程与单模型 AUC 提升验证，再考虑与价格模型融合。

---

## 四、创新方向（建议延后）

| 建议 | 当前状态 | 说明 |
|------|----------|------|
| 因果推断 (Granger 等) | 未实现 | 可作为特征筛选的补充，在 SHAP 之后做 |
| 对抗训练 | 未实现 | 需先有稳定基线再考虑鲁棒性 |
| 元学习 / Few-shot | 未实现 | 研究性质强，适合有专门人力时再开 |

---

## 五、优先级与依赖（与建议一致）

- **立即执行**  
  1. **扩大数据集**：保证 `data/klines` 含 1–2 年（如 730 天），再跑所有训练脚本。  
  2. **特征选择**：加 SHAP 分析，与现有 0.85 + Top 30–40 并存或替代。  
  3. **正则化**：在 WF/回测中 A/B 测试 num_leaves=15、min_child_samples=50–100。

- **数据量 10000+ 后**  
  4. 多交易对：确认 BTC/SOL/BNB 数据与 `train_cross_asset`/TFT 对齐。  
  5. 多周期：扩展 `train_mtf_fusion` 的 TF 与特征。  
  6. 集成：**Stacking 建议在样本 20000+ 后再重训**（三轮迭代中 4241 样本下 Stacking 相对基线 -13.35%，元学习器过拟合）；届时再重跑 Stacking/ Bagging 并做 OOS 对比。详见 `docs/ml_three_round_iteration_conclusion.md`。

- **数据量 50000+ 后**  
  7. 深度学习：引入 Transformer-XL/Informer/N-BEATS。  
  8. 多模态：选一种非价格数据做试点。

---

## 六、预期收益路线图（保留原表，与实现对应）

| 阶段 | 改进内容 | 累计 AUC 提升 | 时间 |
|------|----------|---------------|------|
| 当前 | 高频微结构特征 | +3.52% | 已完成 |
| 短期 | 数据扩展 + 特征优化 + 正则 A/B | +5~7% | 1–2 周 |
| 中期 | 多资产 + 多周期 + 集成 | +8~12% | 1–2 月 |
| 长期 | 深度学习 + 多模态 | +15~20% | 3–6 月 |

**结论**：建议先完成短期三项（数据扩展、SHAP/特征、正则 A/B），验证 OOS 后再投入中期多资产/多周期/集成与长期深度学习/多模态。
