# ETH/USDT 六书融合量化交易策略 v10.2+ML v3.2 — 完整技术规格书（2026-02-18 更新）

**交易标的**: ETH/USDT 永续合约 (Binance Futures)
**主时间框架**: 1h K线
**决策时间框架**: 15m, 1h, 4h, 24h (多周期联合决策)
**回测区间 (IS)**: 2025-01 ~ 2026-01 (12个月)
**OOS验证区间**: 2024-01 ~ 2024-12 (12个月)
**Walk-Forward验证**: 6窗口滚动 (2024Q1 ~ 2025Q4)
**初始资金**: $100,000 USDT
**策略版本**: v10.2+ML v3.2 — 六书融合 + ML 预测子系统 + H800 离线训练
**生产配置版本**: v6（`STRATEGY_VERSION` 默认 `v6`，可环境变量切换）
**GPU训练**: H800 离线训练（本地 Parquet 数据管线，LSTM/TFT/PPO/TabNet/Stacking 等模式）
**ML 模式**: 正式增强（shadow_mode=False, 2026-02-20 起）

> **最新架构与代码结构摘要** (2026-02-18):
> 1. **信号计算单源**：`signal_core.py` 是回测与实盘共用核心，统一计算六书评分与融合分数，避免信号漂移。  
> 2. **实盘链路模块化**：`live_runner.py`（入口）→ `live_trading_engine.py`（编排）→ `live_signal_generator.py`（信号）→ `order_manager.py`/`risk_manager.py`（执行与风控）。  
> 3. **ML 正式增强**：`ml_live_integration.py` 已关闭 shadow 模式（2026-02-20），ML 实际修改 SS/BS 信号；支持方向预测 (Stacking 优先)、Regime、分位数与可选远程 GPU 推理 API 回退。  
> 4. **Stacking 上线门控**：Stacking 由环境变量与质量门槛联合控制（val/test/oof AUC、过拟合 gap、特征覆盖率）；不满足时自动回退加权 LGB+LSTM+TFT+跨资产 LGB。当前 1h Stacking OOF AUC=0.5883, 24492 样本，已通过门控。  
> 5. **H800 训练入口统一**：`train_gpu.py` 含样本量门控（Stacking≥20k, TabNet≥10k），Multi-Horizon LSTM，ONNX 导出，12 种训练模式。  
> 6. **文档渲染链路**：`/strategy/tech-doc` 页面直接渲染 `docs/strategy_tech_doc.md`，是线上技术文档单一来源。

**代码规模统计**:
- Python 文件: **170 个**, 总行数 ~76,600
- Top 5 大文件: `optimize_six_book.py` (5,488), `train_gpu.py` (3,798), `app.py` (2,180), `ml_live_integration.py` (1,782), `candlestick_patterns.py` (1,657)
- 模板文件: 48 个 HTML 模板
- 测试文件: 7 个 (tests/ + 根目录 test_*.py)

**最新代码结构（顶层）**:
- `app.py`: Flask 主应用与文档/API 页面入口 (2,180 行)
- `web_routes/`: 页面与结果 API 子路由
- `templates/`: 控制台/文档/回测页面模板 (48 个)
- `signal_core.py`: 六书信号与融合核心 (1,105 行)
- `optimize_six_book.py`: 回测优化与多周期策略运行 (5,488 行, **待拆分**)
- `strategy_futures.py`: 合约回测执行引擎 (1,378 行)
- `live_*.py`: 实盘入口、信号生成、交易编排
- `ml_live_integration.py`: ML 信号增强器 (1,782 行, 含 Stacking 门控 + 远程推理)
- `train_gpu.py`: H800 离线训练总入口 (3,798 行, **待拆分**)
- `data/`: klines、ml_models、backtests、live 状态
- `tests/` + `test_*.py`: 单元/集成/策略一致性测试

**近期改动摘要（架构/运维）**:
- **ML 正式上线**: shadow_mode=False (2026-02-20)，ML 实际增强信号分数，Stacking 作为方向预测优先路径。
- Stacking 1h 重训完成 (H800): OOF AUC=0.5883, 24492 样本; 元学习器系数分析：LGB(1.24) 和跨资产LGB(1.47) 贡献最大，XGBoost(-0.02) 和 TFT(-0.12) 贡献极低。
- 新增 `scripts/sync_stacking_alias.py` 保证默认别名指向指定 TF 的 Stacking 模型。
- `check_ml_health.py` 修复 shadow_mode 误报，优先读取 runtime config。
- `train_gpu.py` 新增样本量门控（Stacking≥20k, TabNet≥10k），避免小样本浪费 GPU。
- Multi-Horizon LSTM 训练与推理支持（best_head 自动选择）。
- ONNX 模型重新导出（LSTM/TFT/MTF）。
- 新增独立推理服务 `ml_inference_server.py`，支持 `/health`、`/predict`，ECS 侧可远程调用并失败回退本地推理。
- 实控面板多周期检测增强：自动检测恢复补跑、请求超时控制、检测健康状态指示。
- 部署脚本 `deploy.sh` / `deploy_local.sh` 维持双服务重启与健康检查流程。

>
> **v10.2 Phase 2 改造** (2026-02-15):
> 1. **MAE 追踪**: trade record 新增 min_pnl_r/max_pnl_r 字段，收集仓位生命周期最大逆向偏移数据
> 2. **Leg Budget 5×2**: 差异化 regime×direction 仓位预算 (high_vol_choppy 做空=0.20, trend 做多=1.20)
> 3. **Regime Sigmoid**: SL/杠杆/阈值 3 处 sigmoid 连续过渡，消除 regime 切换硬跳变
> 4. **TP 禁用**: trend/low_vol_trend 禁用 TP1/TP2, 仅用 P13 连续追踪让利润奔跑
>
> **v10.1 Phase 1 改造** (保留): ATR-SL + P21 绑定 + Funding Z-Score + OI 审计
> **v10.0 改造** (保留): Soft Veto + Leg Risk Budget + Funding-in-PnL
>
> **回测验证**: OOS Ret +8.7%→**+16.0%**, OOS PF 1.04→**1.31**, OOS Calmar 0.43→**0.82**, 尾部风险 (W5) 减少 **55%**
> **DIV 0.50 实验失败**: OOS 从 +8.7%→-3.1%, 保留 DIV=0.70

---

<h2 id="section-1">一、策略概述与核心指标</h2>

### 核心性能指标

**样本内 (IS: 2025-01 ~ 2026-01, 保守风控口径)**

| 指标 | v6.0 | v7.0 (B3) | v8.0 (B3+P13) |
|------|------|-----------|---------------|
| 总收益 | +206.3% | ~+208% | IS保守风控 |
| 胜率 | 63.1% (135胜/79负) | ~62.3% | **53.8%** (91笔) |
| 合约 Profit Factor | 1.53 | ~1.56 | **1.08** (+0.17 vs B3) |
| 组合 Profit Factor | 2.76 | ~2.80 | — |
| 最大回撤 | -14.0% | ≤-14.0% | — |
| Alpha (vs ETH持有) | +233.4% | ~+235% | — |

> IS PF 0.91→1.08：P13连续追踪止盈让IS从亏损翻正，91笔交易结构更精简。

**样本外 (OOS: 2024-01 ~ 2024-12, 保守风控口径)**

| 指标 | v6.0 | v7.0 (B3) | v8.0 (B3+P13) |
|------|------|-----------|---------------|
| 总收益 | +35.95% | +41.6% | **+41.4%** (-0.2%) |
| 胜率 | 60.7% | 61.0% | **64.9%** (+3.9%) |
| Profit Factor | 1.67 | 2.33 | **2.51** (+0.18) |
| 交易笔数 | — | — | 77笔 |
| 最大回撤 | -16.84% | 改善 | — |

**v9.0 Round 3 A/B 实验 (P17 统一口径)**

| 变体 | IS Ret | IS PF | OOS Ret | OOS PF | Calmar | Worst-5 |
|------|--------|-------|---------|--------|--------|---------|
| E0 baseline (bugfix) | +49.1% | 0.93 | -3.3% | 1.70 | — | -$15k |
| E1 P23 weighted | +44.2% | 0.88 | -8.1% | 1.52 | — | -$12k |
| E2 P21 risk-R | +32.5% | 0.81 | -5.7% | 1.55 | — | -$9k |
| **E4 P18lite+P23** | +38.7% | 0.85 | **+28.9%** | **1.95** | — | -$10k |
| **E5 full v9** | +35.1% | 0.82 | +18.2% | 1.78 | — | **-$7k** |

> **关键发现**: E4 (P18-lite + P23 加权确认) OOS 收益从 -3.3% 飙升至 +28.9%，说明 regime-adaptive 融合权重与加权确认的组合能显著提升 OOS 泛化能力。E5 (全 v9 候选) 大幅降低尾部风险（Worst-5 从 -$15k 降至 -$7k）。

**v9.0 完整回测: v5 生产配置验证 (2026-02-15)**

| 变体 | IS Ret | IS WR | IS PF | IS MDD | IS Calmar | IS Trades | OOS Ret | OOS WR | OOS PF | OOS MDD | OOS Calmar | OOS Trades |
|------|--------|-------|-------|--------|-----------|-----------|---------|--------|--------|---------|------------|------------|
| **E0** v8基线 | **+67.3%** | 56.9% | **1.18** | -11.9% | **5.66** | 153 | -3.8% | 65.1% | 1.59 | -26.7% | -0.14 | 43 |
| **E1** v5生产 | +46.6% | 56.4% | **0.90** ⚠️ | -14.9% | 3.12 | 110 | -3.8% | 65.1% | 1.59 | -26.7% | -0.14 | 43 |
| E2 v5+P18 | -14.4% | 53.8% | 0.44 | -19.6% | -0.74 | 26 | **+41.1%** | 68.4% | **2.27** | -16.9% | 2.43 | 76 |
| E3 v5+P18+P23 | +5.0% | 59.2% | 0.71 | -17.0% | 0.30 | 76 | +37.8% | 69.3% | 2.13 | -16.5% | 2.28 | 75 |

> **⚠️ 关键发现 (v5 生产配置)**:
> 1. **E1 IS PF=0.90 跌破 1.0**: v5 生产配置在 IS 中从盈亏比角度净亏损。Return +46.6% 由少数大赢单支撑，非系统性盈利。
> 2. **B1b 误杀盈利信号**: E0 中 neutral short 34笔 WR=68% PnL=+$985 (盈利)，被 B1b 全部禁止。IS 数据不支持 "neutral short 无 alpha" 假设。
> 3. **P24 止损过度截断**: E0 trend short PnL=+$9,121 → E1 仅 +$374 (WR 64%→77% 但利润蒸发 96%)。-15% 止损收紧导致大盈利交易提前出局。
> 4. **OOS 无差异**: 2024 年无 short 交易，E0/E1 完全相同 → v9 新特性 (B1b/P24/Anti-Squeeze) 在 OOS 中零验证。
> 5. **IS/OOS 分裂持续**: E2/E3 OOS 大幅正收益但 IS 严重亏损，regime shift 敏感度高。

**v10.0 改造回测验证 (2026-02-15)**

| 变体 | IS Ret | IS WR | IS PF | IS MDD | IS Calmar | N | OOS Ret | OOS WR | OOS PF | OOS MDD | OOS Calmar | N |
|------|--------|-------|-------|--------|-----------|---|---------|--------|--------|---------|------------|---|
| E0 v8基线 | +67.3% | 56.9% | 1.19 | -11.9% | 5.66 | 153 | -3.8% | 65.1% | 1.56 | -26.7% | -0.14 | 43 |
| E1 v9生产 | +46.6% | 56.4% | **0.90** | -14.9% | 3.12 | 110 | -3.8% | 65.1% | 1.56 | -26.7% | -0.14 | 43 |
| E2 +soft veto | +51.1% | 58.9% | 1.03 | -11.0% | 4.64 | 141 | **+7.0%** | 56.2% | 0.92 | -20.3% | 0.35 | 73 |
| **E4 v10 full** | **+49.1%** | **61.4%** | **1.31** | -12.1% | 4.07 | 140 | **+7.9%** | 56.2% | 1.02 | -20.3% | 0.39 | 73 |
| E5 v10+SL | +46.8% | 58.7% | 1.14 | -11.7% | 4.00 | 143 | **+8.4%** | 58.1% | 1.09 | -20.3% | 0.41 | 74 |

> **v10 核心改善** (E4 vs E1): IS PF 0.90→**1.31** (+0.41)，IS WR +5.0%，OOS 从 -3.8% 翻正至 **+7.9%**。
> Soft veto 放行边界信号在 OOS 中产生 short 交易 (从 0 笔增至 3 笔)，OOS MDD 从 -26.7% 改善至 -20.3%。

**v10.1 Phase 1 回测验证 (2026-02-15)**

| 变体 | IS Ret | IS WR | IS PF | IS MDD | W5 | N | OOS Ret | OOS WR | OOS PF | OOS MDD | OOS Calmar | N |
|------|--------|-------|-------|--------|-----|---|---------|--------|--------|---------|------------|---|
| E0 v8基线 | +68.9% | 56.9% | 1.19 | -11.9% | -$15,026 | 153 | -4.1% | 65.1% | 1.51 | -26.9% | -0.15 | 43 |
| E1 v10生产 | +49.3% | 61.4% | 1.31 | -12.0% | -$15,557 | 140 | +8.7% | 56.2% | 1.04 | -20.4% | 0.43 | 73 |
| E2 +ATR-SL | +36.3% | 49.3% | 1.05 | -11.6% | -$6,181 | 138 | +9.2% | 50.7% | 1.06 | -20.4% | 0.45 | 73 |
| **E3 +ATR+P21** | **+42.1%** | 51.5% | 1.13 | **-11.5%** | **-$7,042** | 134 | **+16.0%** | 50.7% | **1.31** | **-19.6%** | **0.82** | 73 |
| E4 +DIV50 | +38.4% | 59.7% | 1.59 | -13.0% | -$12,343 | 124 | -3.1% | 48.8% | 1.12 | -16.5% | -0.19 | 41 |
| E5 全量 | +28.4% | 51.9% | 1.42 | -13.2% | -$5,286 | 129 | -1.6% | 47.6% | 1.12 | -16.9% | -0.09 | 42 |

> **v10.1 核心改善** (E3 vs E1): **OOS Ret +8.7%→+16.0%** (+7.3%), **OOS PF 1.04→1.31** (+0.27), **OOS Calmar 0.43→0.82** (1.9x)。
> 尾部风险控制: W5 从 -$15,557 降至 **-$7,042** (减少 55%)。ATR-SL + P21 绑定使仓位与止损联动。
> **DIV 0.50 回测失败**: OOS 从 +8.7%→-3.1%, 交易数从 73→41, DIV=0.70 保留。

**v10.2 Phase 2 回测验证 (2026-02-15)**

| 变体 | IS Ret | IS WR | IS PF | IS MDD | IS W5 | N | OOS Ret | OOS WR | OOS PF | OOS MDD | OOS Calmar | N |
|------|--------|-------|-------|--------|-------|---|---------|--------|--------|---------|------------|---|
| E0 v10.1基线 | +40.1% | 51.5% | 1.26 | -11.4% | -$8,084 | 134 | +14.4% | 50.7% | 1.38 | -19.6% | 0.74 | 73 |
| E1 +Leg5×2 | +34.7% | 51.5% | 1.29 | -11.7% | -$5,496 | 134 | +9.8% | 50.7% | **1.50** | -20.7% | 0.47 | 73 |
| **E2 +Sigmoid** | **+40.7%** | 51.1% | 1.28 | -11.7% | -$7,854 | 133 | **+15.0%** | **52.8%** | **1.42** | **-19.6%** | **0.77** | 72 |
| E3 +TP禁用 | +40.1% | 51.5% | 1.26 | -11.4% | -$8,084 | 134 | +14.4% | 50.7% | 1.38 | -19.6% | 0.74 | 73 |
| E4 全量 | +34.1% | 51.1% | 1.27 | -11.9% | -$5,496 | 133 | +9.9% | 52.8% | 1.51 | -20.7% | 0.48 | 72 |
| E5 保守 | +37.0% | 51.1% | 1.34 | -11.9% | -$6,304 | 133 | +11.9% | 52.8% | 1.49 | -20.4% | 0.59 | 72 |

> **v10.2 Phase 2 结论**: **Regime Sigmoid (E2) 是明确的正面改造** — OOS 全面微提升: Ret +0.6%, PF +0.04, WR +2.1%, Calmar +0.03。
> Leg Budget 5×2 (E1) 降低收益但提升 PF(+0.12) 和 W5(+$474), 属于风险质量优化。
> TP 禁用 (E3) 当前无效 — 趋势 regime 交易太少, 但保留不会有害。
> **MAE 追踪**: IS 平均 MAE=-4.2%, P10=-8.3%, P25=-6.6% — 可用于后续 ATR mult 校准。
> **生产部署**: Sigmoid + TP禁用 + 保守版 Leg Budget。

**v5 生产配置 Regime × Direction 明细**

| 变体 | Period | short\|neutral | short\|trend | short\|high_vol | long\|neutral | long\|trend | long\|high_vol |
|------|--------|----------------|--------------|-----------------|---------------|-------------|----------------|
| E0 | IS | n=34 WR=68% +$985 | n=14 WR=64% +$9,121 | n=1 WR=100% +$35 | n=27 WR=48% -$374 | n=6 WR=83% +$6,855 | n=2 WR=0% -$1,219 |
| E1 | IS | **禁止 (B1b)** | n=13 WR=77% **+$374** | — | n=28 WR=57% +$3,589 | n=6 WR=100% +$4,423 | n=2 WR=0% -$1,219 |
| E0/E1 | OOS | — | — | — | n=29 WR=66% +$6,280 | n=6 WR=83% +$3,179 | n=1 WR=0% -$247 |

**Walk-Forward 验证 (6窗口滚动)**

| 指标 | 值 |
|------|------|
| 盈利窗口 | 3/6 (50%) |
| 平均季度收益 | +9.6%/Q |
| 平均胜率 | 始终 >54% |
| 平均PF | 1.77 |

### Regime 分布表现 (IS run#499)

| Regime | 笔数 | 净PnL | PF |
|--------|------|-------|-----|
| neutral | 277 | +$127,258 | 2.08 |
| trend | 89 | +$74,430 | 6.35 |
| low_vol_trend | 40 | +$35,330 | 5.50 |
| high_vol | 21 | +$17,343 | 4.73 |
| high_vol_choppy | 7 | +$1,405 | 2.16 |

### OOS Regime × 方向 表现 (2024)

| Regime × 方向 | 笔数 | WR | PF | 净PnL |
|---------------|------|-----|-----|-------|
| neutral × long | 20 | **85.0%** | **17.67** | +$7,175 |
| neutral × short | 33 | 54.5% | 1.07 | +$169 |
| trend × short | 15 | 46.7% | 0.84 | -$606 |
| high_vol × short | 6 | 33.3% | 0.47 | -$1,050 |

> **v9.0 架构决策依据 (已修正)**: neutral long 是核心利润来源 (WR=85%, PF=17.67)。neutral short OOS 2024 PF=1.07 微利。四大 LLM 共识建议禁止 (B1b)。
> **v10.0 纠偏**: v9.0.1 IS 回测发现 neutral short WR=68% PnL=+$985 (盈利), B1b 为误杀。v10 改为 Leg Budget (10% 仓位) + Soft Veto (sigmoid), IS PF 0.90→1.31。

---

<h2 id="section-2">二、系统架构</h2>

### 2.1 六层架构 (v10.2+ML)

```
┌─────────────────────────────────────────────────────────┐
│  Layer 0: 数据层 (binance_fetcher.py)                    │
│  K线数据 + Mark Price + Funding Rate + OI + Taker Volume │
│  merge_perp_data_into_klines() 合并衍生品数据              │
└──────────────┬──────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 1: 信号生成 (signal_core.py)                      │
│  六维独立评分: DIV / MA / CS / BB / VP / KDJ              │
│  每本书独立输出 sell_score 和 buy_score (0-100)            │
│  向量化批量计算 + 多进程并行 (P0/P1 优化)                  │
└──────────────┬──────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 1.5: ML 增强 (ml_live_integration.py) ← NEW      │
│  ml_features.py → 94 维特征工程 (73 基础 + 21 跨资产)     │
│  ml_predictor.py → LightGBM/XGBoost 方向预测             │
│  ml_regime.py → 波动率 regime + 趋势质量分类              │
│  ml_quantile.py → 收益分位数预测 (q05~q95)               │
│  → MLSignalEnhancer: 增强/抑制六书信号                    │
└──────────────┬──────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 2: 多周期融合 (multi_tf_consensus.py)              │
│  4个TF的6书分数 → 加权共识 SS/BS                          │
│  链式一致性检测 + 大小周期冲突检查                          │
│  book_features_weighted 传递给后续层                       │
└──────────────┬──────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 3: 决策引擎 (optimize_six_book.py)                │
│  Regime检测 → P24 Regime-Adaptive SL                     │
│  → v10 Leg Risk Budget (neutral short 10%)               │
│  → Anti-Squeeze Filter (OI+Funding+Taker组合)            │
│  → 微结构叠加 → 入场条件判断 → 仓位计算                    │
│  → P13 连续追踪止盈 + P20 空头追踪收紧                    │
└──────────────┬──────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 4: 执行引擎 (FuturesEngine / live_runner.py)      │
│  延迟执行(T+1 open) → 滑点/手续费建模                     │
│  分段止盈(TP1@12%,TP2@25%) → P13连续追踪止盈              │
│  Mark Price 强平检测                                     │
└──────────────┬──────────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 5: 风控保护                                       │
│  硬断路器(-28%) → 日亏/周亏/连亏熔断                       │
│  全局回撤停机(15%) → 现货趋势底仓保护                      │
└─────────────────────────────────────────────────────────┘
```

### 2.2 数据流水线

```
  实盘 (live_signal_generator.py)           回测 (optimize_six_book.py)
  ┌─────────────────────────┐              ┌─────────────────────────┐
  │ fetch_binance_klines()  │              │ 从本地 DB 加载 K线       │
  │ fetch_mark_price_klines()│←── 共享 ──→ │ merge_perp_data()       │
  │ fetch_funding_rate_history()│           │ (mark/funding/OI 合并)  │
  │ fetch_open_interest_history()│          │                         │
  │ merge_perp_data_into_klines()│         │ _compute_microstructure()│
  └──────────┬──────────────┘              └──────────┬──────────────┘
             │                                        │
             ▼                                        ▼
    compute_signals_six()                    _vectorized_fuse_scores()
    calc_fusion_score_six()                  _apply_microstructure_overlay()
             │                                └─ Anti-Squeeze Filter
             ▼
    ml_live_integration.py (可选)   GPU 离线训练 (train_gpu.py)
    ┌─────────────────────────┐    ┌──────────────────────────┐
    │ ml_features → 94 维     │    │ load_klines_local()      │
    │ ml_predictor → 方向预测 │    │ → add_all_indicators()   │
    │ ml_regime → regime 分类 │    │ → compute_ml_features()  │
    │ MLSignalEnhancer        │    │ → train LGB/LSTM/Optuna  │
    │ → 增强/抑制六书信号     │    │ → data/ml_models/*.txt   │
    └─────────────────────────┘    └──────────────────────────┘
```

### 2.3 训练后推理架构示意（本地 / 远程 GPU）

```text
                  ┌────────────────────────────────────────────┐
                  │         H800 离线训练 (train_gpu.py)       │
                  │  lgb/lstm/tft/stacking/tabnet/optuna 等    │
                  └─────────────────┬──────────────────────────┘
                                    │ 产出模型文件
                                    ▼
                      data/ml_models/*.txt|*.pt|*.json|*.pkl
                                    │
               ┌────────────────────┴────────────────────┐
               │                                         │
               ▼                                         ▼
┌──────────────────────────────────┐      ┌──────────────────────────────────┐
│ 路径A: 本地推理 (默认)            │      │ 路径B: 远程推理 API (可选)       │
│ ml_live_integration.py           │      │ ml_inference_server.py           │
│ MLSignalEnhancer.load_model()    │      │ POST /predict                    │
│ - 方向预测(LGB/LSTM/TFT/CA)      │      │ - 接收 features/sell_score/buy   │
│ - Stacking(质量门控通过才启用)   │      │ - 调 MLSignalEnhancer 推理        │
│ - Regime + Quantile              │      │ - 返回 bull_prob + 细项           │
└─────────────────┬────────────────┘      └─────────────────┬────────────────┘
                  │                                          │
                  │                                          │
                  └──────────────┬───────────────────────────┘
                                 ▼
                   live_signal_generator.py / live_runner.py
                                 ▼
                         live_trading_engine.py
                                 ▼
                          order_manager.py
```

**推理分流规则（当前实现）**:
- 若配置 `ML_GPU_INFERENCE_URL`（或 `strategy.ml_gpu_inference_url`），`enhance_signal()` 优先走远程 `/predict`；失败超时回退本地。
- 本地与远程都复用 `predict_direction_from_features()`，保证方向预测逻辑一致。
- Stacking 受环境变量与质量门槛控制（`ML_ENABLE_STACKING`、AUC/gap/覆盖率门控）；不满足自动回退加权融合。
- 推理设备可由 `ML_INFERENCE_DEVICE` 指定（`cpu`/`cuda`），否则自动探测 CUDA。

### 2.4 核心代码文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `optimize_six_book.py` | ~5000 | 回测优化与多周期策略运行核心（Regime、仓位、Anti-Squeeze） |
| `live_config.py` | ~1180 | 策略参数、版本管理（v1-v6）、风控与阶段配置 |
| `signal_core.py` | ~940 | 六书信号计算、融合评分（单bar+批量向量化） |
| `live_signal_generator.py` | ~840 | 实盘信号: 数据获取(含衍生品)、信号计算、共识融合 |
| `binance_fetcher.py` | ~690 | Binance API: K线+Mark/Funding/OI 获取与本地缓存 |
| `multi_tf_consensus.py` | ~330 | 多周期共识、TF权重、链式检测 |
| `live_runner.py` | — | 实盘运行入口、systemd 管理 |
| `app.py` | — | Flask Web 应用、监控面板 |
| **ML/GPU 文件** | | |
| `ml_features.py` | — | ML 特征工程（含高频微结构特征） |
| `ml_predictor.py` | — | LightGBM/XGBoost Walk-Forward 预测 |
| `ml_regime.py` | — | 波动率 regime + 趋势质量分类 |
| `ml_quantile.py` | — | 收益分位数预测 (q05~q95) |
| `ml_live_integration.py` | — | ML 信号增强器 (含 Stacking 门控 + 远程推理回退) |
| `ml_inference_server.py` | — | 独立推理服务（/health, /predict） |
| `train_gpu.py` | — | H800 GPU 离线训练入口（多模式） |
| `fetch_5year_data.py` | — | 批量下载 5 年训练数据 (4对×4周期) |

---

<h2 id="section-3">三、信号生成 — 六书融合</h2>

**六本书（Six Books）** 是六个独立的技术指标维度，每本书在每根K线上输出 `sell_score` 和 `buy_score`（0-100分）:

| 书名 | 缩写 | 核心指标 | 默认融合权重 |
|------|------|---------|-------------|
| Divergence | DIV | MACD柱/DIF与价格背离 | 55% (div_weight) |
| Moving Average | MA | 均线排列/交叉/斜率 | 30% (基数中隐含) |
| Candlestick | CS | K线形态识别(锤子/吞没等) | 6% (cs_bonus) |
| Bollinger Band | BB | 布林带突破/挤压/回归 | 10% (bb_bonus) |
| Volume-Price | VP | 量价关系/量能异常 | 8% (vp_bonus) |
| KDJ | KDJ | 超买超卖+趋势确认 | 9% (kdj_bonus) |

### 融合模式: `c6_veto_4` (生产默认)

```python
# 融合公式:
base_ss = (div_sell * 0.70 + ma_sell * 0.30) * ma_arrangement_bonus
# 四书加成: BB/VP/KDJ (>=15分) + CS (>=25分) → 乘法加成
```

### v10.0 Soft Veto (替代硬二元门控)

```python
# v9 硬 veto (已废弃): ≥2 本反向书 ≥ 25 → score × 0.30 (硬开关)
# v10 soft veto: 连续 sigmoid 衰减, 消除边界敏感性

# 1. 计算反向强度 (0 ~ 4 之间连续值)
opp_strength = Σ max(0, book_buy - veto_threshold) / (100 - veto_threshold)
# 当没有反向信号时 opp=0, 当 4 本书都极强反向时 opp→4

# 2. Sigmoid 衰减 (连续可微)
penalty = veto_dampen + (1 - veto_dampen) / (1 + exp(k × (opp - midpoint)))
# k=3.0 (steepness), midpoint=1.0 (约等于原 "2 本书@阈值" 水平)
# opp << 1 → penalty ≈ 1.0 (无惩罚)
# opp ≈ 1 → penalty ≈ 0.65 (中等衰减)
# opp >> 2 → penalty → 0.30 (接近原硬 veto)

# 3. 同时应用加成和衰减 (不再互斥)
sell_score = base_sell × (1 + bonus) × penalty
```

> **v10 改造理由**: v9.0.1 回测实证硬 veto 的蝴蝶效应 — B1b (`neutral:999`) 是硬门控的极端形式, 导致 IS PF 跌破 1.0。GPT Pro 建议"用连续罚函数替代硬开关"。soft veto 使 IS PF 从 0.90 回升至 1.03, 叠加 leg budget 后达 1.31。

### P18: Regime-Adaptive 融合 (已实现，当前未在 v5 中启用)

```python
# 每个 regime 有独立的权重矩阵:
# neutral:  DIV 25% + MA 75% (DIV d=-0.64 反向, 大幅降权)
# trend:    DIV 60% + MA 40% (背离在趋势末端有效)
# high_vol: DIV 45% + MA 55% + VP bonus 12% (量价在高波中更有效)
# 实现路径: 主循环中逐 bar 从 book_features 重新融合 (L2174-2221)
# 批量路径: regime_adaptive 回退到 c6_veto_4, 主循环覆盖分数
```

### P4 信号判别力诊断（Cohen's d）

| 书 | Cohen's d (neutral short) | 判别等级 | 说明 |
|----|--------------------------|---------|------|
| DIV_sell | **-0.64** | **反向** | 赢单DIV分数反而更低 |
| CS_sell | +0.11 | 无效 | d < 0.20 |
| KDJ_sell | +0.08 | 无效 | d < 0.20 |
| BB_sell | +0.05 | 无效 | d < 0.20 |
| VP_sell | +0.03 | 无效 | d < 0.20 |
| MA_sell | -0.02 | 无效 | d < 0.20 |

> **v9.0 决策 (已废弃)**: v9 通过 B1b 直接禁止 neutral short，但 v9.0.1 回测实证 IS neutral short WR=68% PnL=+$985 (盈利)，B1b 为误杀。
> **v10.0 决策**: 恢复 neutral short (阈值=60), 通过 Leg Risk Budget (仓位×0.10) 和 Soft Veto (sigmoid) 实现风险控制而非完全禁止。

### P16 六书全 regime 信号判别力

| 书 | 全 regime Cohen's d (空仓) | Alpha等级 |
|------|--------------------------|----------|
| CS | **+0.215** | 正向 (最强) |
| MA | **+0.215** | 正向 (最强) |
| VP | +0.178 | 正向 |
| KDJ | +0.135 | 正向 |
| BB | 负向 | 无正alpha |
| DIV | 负向 | 无正alpha |

---

<h2 id="section-4">四、多周期共识</h2>

### 4.1 权重分配

```python
_MTF_WEIGHT = {
    '15m': 5, '30m': 8, '1h': 15, '2h': 12,
    '4h': 20, '8h': 18, '12h': 15, '24h': 25,
}
# v9.0: 决策TF = ['15m', '1h', '4h', '24h'] (P17统一口径)
```

### 4.2 共识融合流程

1. 收集各TF的 (ss, bs) 分数
2. 加权平均得到 weighted_ss, weighted_bs
3. 链式一致性检测: 相邻TF方向一致 → boost +8%/TF
4. 覆盖率门控: 有效TF数 / 总TF数 >= 0.5 才出信号
5. 方向决策: dominance_ratio=1.3, 即 SS > BS*1.3 才判定为"卖"
6. 输出: (consensus_ss, consensus_bs, meta_dict)
7. **book_features_weighted**: 12维特征加权汇总，供 P18/结构折扣使用

---

<h2 id="section-5">五、Regime 检测</h2>

基于 **波动率 + 趋势强度 + ATR** 三维判定，输出5种市场状态：

```python
# 输入指标（仅使用 idx-1 及之前数据，避免前视偏差）:
vol  = 48bar 收益率标准差
trend = (MA12 - MA48) / price
atr_pct = 14bar ATR / price

# 判定逻辑:
if vol >= 0.020 or atr_pct >= 0.018:
    regime = 'high_vol' / 'high_vol_choppy'
elif vol <= 0.007 and abs(trend) >= 0.015:
    regime = 'low_vol_trend'
elif abs(trend) >= 0.015:
    regime = 'trend'
else:
    regime = 'neutral'
```

### v10.0 Regime 对策略行为的影响

| Regime | 空单决策 (v10.0) | 止损 (P24) | 门槛乘数 | 杠杆乘数 |
|--------|----------------|-----------|---------|---------|
| **neutral** | **允许, 仓位×0.10 (Leg Budget)** | -12% | 1.00 | 1.00 |
| trend | 允许, Anti-Squeeze过滤 | **-15%** | 0.98 | 1.00 |
| low_vol_trend | 允许, +35门槛 | -18% | 0.95 | 1.00 |
| high_vol | 允许, Anti-Squeeze过滤 | **-12%** | 1.12 | 0.75 |
| high_vol_choppy | 允许, Anti-Squeeze过滤 | -12% | 1.22 | 0.58 |

---

<h2 id="section-6">六、v10.0 开仓逻辑</h2>

### 6.1 信号延迟执行 (T+1)

```python
# 在 bar[i] 收盘时计算信号 → 存入 pending_ss/pending_bs
# 在 bar[i+1] 的 open 价格执行交易 → 消除 same-bar bias
exec_price = open_prices[idx]
ss, bs = pending_ss, pending_bs
```

### 6.2 v10.0 开空条件 (做空)

```python
_short_candidate = (
    short_cd == 0                        # 冷却期结束 (cooldown=6)
    and ss >= effective_short_threshold   # SS >= 动态阈值
    and not eng.futures_short             # 无存量空仓
    and not eng.futures_long              # 无存量多仓
    and sell_dom                          # SS > BS * 1.5
    and not in_conflict                   # 非冲突区间
    and can_open_risk                     # 风控允许
    and not micro_block_short             # 微结构不阻止 ← 含Anti-Squeeze
    and _bk_short_ok                      # 六书共识门控通过
    # v10: regime_short_threshold = "neutral:60" (恢复 B3 水平)
    # → neutral 中 SS 须 >= 60 才开空 (非完全禁止)
)
# v10: 仓位计算时应用 leg budget
margin *= _leg_budget_mult(regime, 'short')
# neutral short: margin × 0.10 (仅 10% 仓位, 软降权替代硬禁止)
# trend/high_vol short: margin × 1.0 (正常)
```

### 6.3 v9.0 Anti-Squeeze Filter (新增)

```python
# 位于 _apply_microstructure_overlay() 中
# 三条件组合 (全部满足才触发):

# 多头拥挤 → 禁止开空:
if (funding_rate >= 0.08%        # 多头付费, 拥挤
    and oi_z_score >= 1.0        # OI 上升, 杠杆堆积
    and taker_imbalance >= 0.12): # 主动买盘强
    block_short = True
    margin_mult *= 0.60          # 已有空仓降风险

# 空头拥挤 → 禁止开多 (镜像):
if (funding_rate <= -0.08%
    and oi_z_score >= 1.0
    and taker_imbalance <= -0.12):
    block_long = True
    margin_mult *= 0.60
```

### 6.4 仓位计算 (v10.0)

```python
margin = available_margin * cur_margin_use  # 基础仓位

# v10 Leg Risk Budget: regime × direction 风险预算
margin *= _leg_budget_mult(regime, direction)
# neutral_short → 0.10 (10%), 其他 → 1.0

# 结构折扣 (neutral long/short 有效)
if struct_confirms == 0: margin *= max(0.0, soft_struct_min_mult)  # v10: 0→0.02
elif struct_confirms == 1: margin *= 0.05 # 1本→5%
elif struct_confirms == 2: margin *= 0.15 # 2本→15%
elif struct_confirms == 3: margin *= 0.50 # 3本→50%

# 冲突折扣: trend/high_vol/neutral (v10 恢复 B3 含 neutral)
if regime in ('trend', 'high_vol', 'neutral'):
    if div_buy >= 50 and ma_sell >= 12:
        margin *= 0.60

# 杠杆动态调整
actual_lev = 5 if ss >= 50 else 3 if ss >= 35 else 2
```

---

<h2 id="section-7">七、持仓管理与退出</h2>

### 7.1 退出优先级 (空仓)

| 优先级 | 退出类型 | 条件 | 备注 |
|--------|---------|------|------|
| 1 | 强平检测 | Mark Price HIGH 穿越强平价 | v9.0: 使用 mark_high/mark_low |
| 2 | 硬断路器 | PnL < -28% | 安全网 |
| 3 | **P24 Regime-Adaptive SL** | trend: -15%, high_vol: -12% | **v9.0 新增** |
| 4 | 分段止盈TP1 | PnL >= +12% → 平仓30% | |
| 5 | 分段止盈TP2 | PnL >= +25% → 再平仓30% | |
| 6 | 完全止盈 | PnL >= +60% → 全部平仓 | |
| 7 | **P13连续追踪止盈** | 利润≥5%起追踪, 回撤容忍60%→30% | |
| 8 | **P20空头追踪收紧** | 空头回撤容忍40% (vs多头60%) | **v8.1** |
| 9 | 反向信号平仓 | BS >= 40 且 SS < BS*0.7 且持仓>=8bars | |
| 10 | 超时平仓 | 持仓 >= 48 bars | |

### 7.2 P13 连续追踪止盈 (v8.0 核心)

```python
continuous_trail_start_pnl = 0.05   # 利润达5%即开始追踪
continuous_trail_max_pb    = 0.60   # 低利润: 宽容忍度 60%
continuous_trail_min_pb    = 0.30   # 高利润: 紧容忍度 30%
# v9.0 P20: 空头专用
continuous_trail_max_pb_short = 0.40  # 空头更紧: 40%

# 动态回撤计算
progress = min(max_pnl / TP, 1.0)
effective_pullback = max_pb - (max_pb - min_pb) * progress
if pnl_r < max_pnl * (1 - effective_pullback):
    close_position()
```

### 7.3 P24 Regime-Adaptive Stop-Loss (v9.0 新增, v10 沿用)

```python
# 按 regime 差异化止损 (替代全局 -20%):
regime_short_sl_map = {
    'neutral':        -0.12,  # v10: Leg Budget 仓位仅 10%, SL 仍适用
    'trend':          -0.15,  # 收紧: 趋势中错误应快速认输
    'low_vol_trend':  -0.18,  # 略收紧
    'high_vol':       -0.12,  # 最紧: 挤压风险最高
    'high_vol_choppy':-0.12,
}
# 实际止损 = regime_short_sl_map[current_regime]
```

### 7.4 冷却机制

| 事件 | 冷却时长 |
|------|---------|
| 正常开仓 | 6 bars (6h) |
| 止盈后 | 12 bars |
| 反向平仓后 | 18 bars |
| 止损后 | 24 bars |
| 连续止损 | 48 bars |
| 硬止损后 | 30 bars |
| 强平后 | 24 bars + 跨方向12 bars |

### 7.5 Funding-in-PnL (v10.0 新增)

```python
# v9 及之前: trade PnL = close_price - entry_price (不含 funding)
# v10: trade PnL 包含持仓期累积 funding cost

# 每 8h funding 结算时:
if is_funding_settlement:
    if holding_long:
        position.accumulated_funding += funding_cost   # 多头付 funding
    if holding_short:
        position.accumulated_funding -= funding_income  # 空头收 funding

# 平仓时:
net_pnl = gross_pnl - accumulated_funding
# PF/WR 基于 net_pnl 计算 → 更真实的策略评估
```

> **影响**: 由于 ETH 多数时间 funding > 0 (多头付费), 纳入 funding 后多头实际收益略降、空头略增。IS PF 从 1.03 (E3 仅 soft) 提升至 1.31 (E4 含 funding), 说明之前的 PF 低估了空头的真实贡献。

---

<h2 id="section-8">八、衍生品数据集成 (v9.0/v10.0)</h2>

### 8.1 数据源

| 数据 | API 端点 | 用途 |
|------|---------|------|
| Mark Price K线 | `/fapi/v1/markPriceKlines` | 强平检测、真实 PnL |
| Funding Rate 历史 | `/fapi/v1/fundingRate` | 资金费率成本、拥挤度信号 |
| Open Interest 历史 | `/futures/data/openInterestHist` | 杠杆堆积检测 |
| Taker Buy/Sell Volume | K线内置字段 | 主动买卖力量对比 |

### 8.2 数据合并流程 (merge_perp_data_into_klines)

```python
# 合并到主 K线 DataFrame:
# 1. Mark Price: mark_high, mark_low, mark_close (前向填充)
# 2. Funding Rate: funding_rate 列 (前向填充到每根 bar)
# 3. Open Interest: open_interest, open_interest_value (前向填充)
# 4. Taker Volume: taker_buy_base, taker_buy_quote (K线内置)
```

### 8.3 微结构特征计算 (_compute_microstructure_features)

```python
# 从合并后的 DataFrame 计算:
taker_imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol)
oi_z = z_score(open_interest, lookback=48)
basis_z = z_score((mark_price - index_price) / index_price, lookback=48)
funding_rate = 直接使用 funding_rate 列 (若有真实数据)
participation = volume_ratio / historical_mean
```

### 8.4 实盘数据获取 (v9.0 修复)

```python
# live_signal_generator.py refresh_data():
# 1. 获取主 K线数据
# 2. 并行获取 Mark Price / Funding Rate / OI (非阻塞, 失败不影响主信号)
# 3. merge_perp_data_into_klines() 合并
# 4. 计算指标 → 六维信号
```

> **v9.0 之前的问题**: `live_signal_generator.py` 完全没有获取衍生品数据，导致 Anti-Squeeze Filter 等依赖 OI/Funding 的功能在实盘中无法工作。v9.0 已修复。

---

<h2 id="section-9">九、v10.0 完整参数集 (v5 配置)</h2>

### 入场参数

| 参数 | 值 | 说明 |
|------|------|------|
| short_threshold | 40 | 空单基础阈值 |
| long_threshold | 25 | 多单门槛 |
| **regime_short_threshold** | **neutral:60** | **v10: 恢复 B3 水平 (v9 的 999 已废弃)** |
| close_short_bs | 40 | 反向平空BS阈值 |
| close_long_ss | 40 | 反向平多SS阈值 |
| cooldown | 6 | 基础冷却期 |

### 止损止盈

| 参数 | 值 | 说明 |
|------|------|------|
| short_sl | -0.20 | 全局默认 (被 P24 覆盖) |
| **use_regime_adaptive_sl** | **True** | **v9.0 P24** |
| **regime_trend_short_sl** | **-0.15** | trend 收紧 |
| **regime_high_vol_short_sl** | **-0.12** | high_vol 最紧 |
| long_sl | -0.10 | 多单止损 |
| short_tp | 0.60 | 空单止盈 |
| long_tp | 0.40 | 多单止盈 |
| hard_stop_loss | -0.28 | 硬断路器 |

### P13 连续追踪止盈

| 参数 | 值 | 说明 |
|------|------|------|
| use_continuous_trail | True | 启用 |
| continuous_trail_start_pnl | 0.05 | 利润5%开始追踪 |
| continuous_trail_max_pb | 0.60 | 多头宽容忍度 |
| continuous_trail_min_pb | 0.30 | 高利润紧容忍度 |
| **continuous_trail_max_pb_short** | **0.40** | **v8.1 P20: 空头收紧** |

### Anti-Squeeze Filter (v9.0 新增)

| 参数 | 值 | 说明 |
|------|------|------|
| anti_squeeze_fr_threshold | 0.0008 | funding rate 阈值 (0.08%) |
| anti_squeeze_oi_z_threshold | 1.0 | OI z-score 阈值 |
| anti_squeeze_taker_imb_threshold | 0.12 | taker imbalance 阈值 |

### v10.0 Soft Veto 参数 (新增)

| 参数 | 值 | 说明 |
|------|------|------|
| **use_soft_veto** | **True** | **v10: 启用 sigmoid 连续衰减** |
| soft_veto_steepness | 3.0 | sigmoid 陡度 (越大越接近硬开关) |
| soft_veto_midpoint | 1.0 | 惩罚中点 (1.0≈原2本书@阈值) |
| **soft_struct_min_mult** | **0.02** | **v10: confirms=0 → 2% 仓位 (非 0%)** |

### v10.0 Leg Risk Budget 参数 (新增)

| 参数 | 值 | 说明 |
|------|------|------|
| **use_leg_risk_budget** | **True** | **v10: 启用 regime×direction 风险预算** |
| **risk_budget_neutral_short** | **0.10** | **neutral short 仅 10% 仓位 (替代 B1b 硬禁止)** |
| risk_budget_*_* | 1.0 | 其他 regime×direction 不变 |

### 冲突折扣 (v10.0 恢复 B3)

| 参数 | v8.0/B3 | v9.0 | **v10.0** | 说明 |
|------|---------|------|-----------|------|
| short_conflict_regimes | trend,high_vol,neutral | trend,high_vol | **trend,high_vol,neutral** | v10 恢复含 neutral |
| short_conflict_discount_mult | 0.60 | 0.60 | 0.60 | 不变 |

---

<h2 id="section-10">十、历史实验总结</h2>

### 10.1 P0-P4 诊断实验 (驱动 v7.0)

| 实验 | 核心发现 |
|------|---------|
| P0 OOS验证 | WR 衰减仅 3.8%, PF 反而 +9.2%, 未严重过拟合 |
| P1 Monte Carlo | 50组±10%扰动, 参数整体稳定 |
| P2 short_trail | OOS 完全不敏感 |
| P3 退出消融 | 反向平仓(-8.2%)和超时(-5.7%)是核心退出支柱 |
| P4 信号判别力 | neutral short 六书 Cohen's d 全部无效, DIV=-0.64反向 |

### 10.2 B0-B3 A/B 实验 (驱动 v7.0 参数)

| 变体 | OOS Ret | OOS PF | 说明 |
|------|---------|--------|------|
| B0 基线 (v6.0) | +35.95% | 1.67 | |
| **B1b 禁止neutral空** | **+40.78%** | **2.09** | v9 采用, **v10 废弃** (Leg Budget 替代) |
| B2 强门控(60) | +41.46% | 2.25 | |
| **B3 B2+长冷却** | **+41.59%** | **2.33** | v7.0/v8.0 采用 |

### 10.3 P6-P16 全量诊断 (驱动 v8.0)

| 实验 | 描述 | 结论 |
|------|------|------|
| P6 Ghost CD | 过滤信号触发冷却 | 无效果(B3 CD=6已充分) |
| P7 24h门控 | 大周期方向过滤 | 无效果 |
| P9 SS衰减 | neutral SS*0.85 | IS好但OOS差 |
| **P13 连续追踪** | **替代离散门槛** | **部署: IS PF+0.17, OOS PF+0.18** |
| P16 信号分析 | CS/MA d=0.215最强, DIV/BB负向 | 驱动v9.0权重调整 |

### 10.4 v9.0 实验 (Round 1-3)

**Round 1 (P17 口径)**:
- P18 Regime-Adaptive: OOS +24% 但 IS -63%（分裂严重，暂缓）
- **P20 空头追踪收紧**: 唯一 IS 改善 + OOS 无害 → 部署为 v8.1

**Round 3 (Bug fixes + P21/P23)**:
- E4 P18-lite+P23: **OOS +28.9%**（vs baseline -3.3%）→ OOS 泛化最佳
- E5 full v9: Worst-5 从 -$15k → **-$7k** → 尾部风险大幅降低

### 10.5 v9.0 完整回测 — v5 生产配置验证 (2026-02-15)

**目标**: 验证 v5 精确生产配置 (B1b + P24 + Anti-Squeeze，不含 P18/P21/P23) 的 IS/OOS 表现。

**结果摘要**:

| 对比项 | E0 (v8基线) | E1 (v5生产) | 变化 | 评估 |
|--------|------------|------------|------|------|
| IS Return | +67.3% | +46.6% | **-20.7%** | ⚠️ 显著下降 |
| IS PF | 1.18 | **0.90** | **-0.28** | ❌ 跌破1.0 |
| IS MDD | -11.9% | -14.9% | -3.0% | ⚠️ 恶化 |
| IS Calmar | 5.66 | 3.12 | -2.54 | ⚠️ 恶化 |
| IS Trades | 153 | 110 | -43 | B1b 移除 neutral short |
| OOS Return | -3.8% | -3.8% | 0.0% | 无变化 (无short交易) |
| OOS PF | 1.59 | 1.59 | 0.0 | 无变化 |

**根因分析**:

1. **B1b 负面影响**: IS 中 neutral short 实际是盈利信号 (34笔, WR=68%, PnL=+$985)。LLM 共识基于 OOS 2024 数据 (neutral short PF=1.07, 微利)，但 IS 2025 数据中该信号表现改善。B1b 可能是基于过时判断的过度简化。

2. **P24 止损截断利润**: trend short 从 PnL=$9,121 骤降至 $374 (降幅 96%)，尽管 WR 从 64%→77%。-15% 止损 vs 原 -20% 止损多截断了 5% 回撤空间，导致大趋势交易无法充分发展。

3. **OOS 验证空白**: 2024 年 OOS 期间策略仅触发 long 交易 (43笔)，v9 的三项核心变更 (B1b/P24/Anti-Squeeze) 全部针对 short，因此 OOS 完全无法验证。

**建议行动**: 见第十三章优化方向更新。

---

<h2 id="section-11">十一、v9.0 架构简化决策</h2>

### 11.1 核心哲学转变

```
v7.0/v8.0 ("修补派"):
  neutral short 有问题 → 加门槛60 → 加结构折扣 → 加冲突折扣
  → 加冷却延长 → 加共识门控 → 加极端背离否决 → ...
  问题: 复杂度指数增长, 但天花板不高

v9.0 ("减法派"):
  neutral short 有问题 → 直接禁止 (B1b neutral:999)
  剩余 short → 差异化止损 (P24) + 反挤压过滤 (Anti-Squeeze)
  问题: 简单直接, 但 IS 回测证明 B1b 误杀盈利信号 (PF<1.0)

v10.0 ("连续化改造"):
  v9 硬门控导致蝴蝶效应 → 用连续函数替代二元开关
  Soft Veto (sigmoid) + Leg Budget (仓位10%) + Funding-in-PnL
  结果: IS PF 0.90→1.31, OOS -3.8%→+7.9%, 兼顾风控与信号保留
```

### 11.2 v10 恢复/重新激活的机制 (v9 中被 B1b 冻结)

| 机制 | v9.0状态 | v10.0状态 | 变更原因 |
|------|---------|----------|---------|
| neutral_struct_discount (short) | 不可达 (B1b) | **重新激活** + soft_struct_min_mult=0.02 | confirms=0 → 2% 仓位 (非 0%) |
| short_conflict_regimes 含 'neutral' | 已移除 | **恢复** (trend,high_vol,neutral) | neutral short 再次允许 |
| Leg Risk Budget (neutral short) | 不存在 | **新增** risk_budget=0.10 | 替代 B1b 硬禁止 |
| use_neutral_book_consensus | 已关 | 冗余 | 不变 |
| use_extreme_divergence_short_veto | 已关 | 冗余 | 不变 |
| use_neutral_short_structure_gate | 已关 | 冗余 | 不变 |
| P9 use_regime_adaptive_reweight | 已关 | 被 P18 取代 | 不变 |

### 11.3 四大 LLM 审计共识 (Claude/GPT Pro/Gemini/Grok) — v10 执行状态

| 建议 | v9.0 状态 | **v10.0 状态** |
|------|----------|---------------|
| Regime-Adaptive 权重 | 代码完整, 待验证 | 不变, 待 Walk-Forward |
| 禁止 neutral short (B1b) | 已部署 (neutral:999) | **已废弃** → Leg Budget 0.10 替代 |
| Perp 专属数据 (OI/Funding) | Anti-Squeeze 已实现 | **Funding-in-PnL 新增** (准确性提升) |
| 空头不对称防御 | P24+P20 已部署 | **不变** + Soft Veto 增强信号层风控 |
| 用 R (风险单位) 定义仓位 | P21 太保守 | 不变, 需调参 |
| 加权结构确认 (替代计数) | P23 独立效果差 | 不变, 需与 P18 绑定 |
| 架构简化 ("多做减法") | v5 配置已简化 | **v10 连续化**: 减少硬开关, 增加连续函数 |
| 用连续罚函数替代硬门控 (GPT Pro) | 不适用 | **v10 核心**: Soft Veto + Soft Struct |

---

<h2 id="section-12">十二、当前已知问题</h2>

### 12.1 高优先级

| # | 问题 | 影响 | 状态 |
|---|------|------|------|
| ~~1~~ | ~~v5 生产配置 IS PF < 1.0~~ | ~~v5 IS PF=0.90~~ | **✅ v10 已修复**: IS PF 1.31 |
| ~~2~~ | ~~B1b 误杀 IS 盈利信号~~ | ~~neutral short WR=68% 被禁止~~ | **✅ v10 已修复**: Leg Budget 0.10 替代 |
| 3 | **P24 止损仍可能截断利润** | v10 trend short 未调 SL | 待评估: -15% → ATR-based |
| 4 | **OOS 验证仍有限** | 2024 OOS short 交易稀少 | 需 Walk-Forward 验证 |
| 5 | **P18 IS/OOS 分裂** | regime_adaptive 融合 OOS 优秀但 IS 回退 | 需 Walk-Forward 验证稳定性后再启用 |
| 6 | **P21 risk_per_trade 参数过保守** | 1.5% R% 使仓位过小 | 调参至 2.5-3.5%, 重跑 A/B |
| 7 | **P23 独立效果差** | 加权确认独立使用 OOS -8.1% | 与 P18 绑定使用, 不独立启用 |
| 8 | **trend/high_vol short 胜率低** | trend WR=46.7% PF=0.84 | Anti-Squeeze 应可缓解, 需回测验证 |

### 12.2 中优先级

| # | 问题 | 建议 |
|---|------|------|
| 5 | 回测未计入 funding 现金流 | 长期偏多策略在正 funding 环境被侵蚀, 需纳入 |
| 6 | CS-KDJ 相关性高 (0.474) | 去相关处理, 避免"虚假共识" |
| 7 | 静态 Regime 阈值 | vol=0.020, trend=0.015 为绝对值, 未适应市场周期 |
| 8 | Walk-Forward 仅 3/6 窗口盈利 | 亏损来自少数大亏单, P24/Anti-Squeeze 应改善 |

### 12.3 低优先级 / 已关闭

| 功能 | 测试结果 |
|------|---------|
| ATR自适应止损 | 对ETH无效 |
| NoTP提前退出 | WR 62%→55%, 有害 |
| 保本止损 | 压制利润 |
| 棘轮追踪 | 过度截断利润 |
| 二元门控 | 有蝴蝶效应 |
| neutral 分层 SPOT_SELL | 主样本负贡献 |
| 停滞再入场 | 触发稀少且负贡献 |

---

<h2 id="section-13">十三、ML 预测子系统</h2>

### 13.1 ML 模型矩阵 (H800 训练产出)

| 模型 | 文件 | 架构 | Test AUC | 用途 |
|------|------|------|----------|------|
| **Stacking Ensemble** | `stacking_meta.pkl` + `stacking_meta.json` | 4 基模型 OOF → LogisticRegression 元学习器 | Val/Test 见 meta | **方向预测优先路径** (有则覆盖加权集成) |
| **LGB 方向预测** | `lgb_direction_model.txt` | LightGBM (Optuna) | **0.5537** | 涨跌概率 → bull_prob (Stacking 基模型之一 / fallback) |
| **LSTM+Attention** | `lstm_1h.pt` + ONNX | 双向LSTM+Attention, BF16 | **0.5366** | 序列模式方向预测 (Stacking 基模型之一 / fallback) |
| **跨资产 LGB** | `lgb_cross_asset_1h.txt` | LightGBM + BTC/SOL/BNB 特征 | **0.5485** | 跨市场联动预测 (fallback 加权) |
| **TFT** | `tft_1h.pt` + ONNX | Temporal Fusion Transformer | — | 可解释时序预测 (Stacking 基模型之一 / fallback) |
| **MTF 融合 MLP** | `mtf_fusion_mlp.pt` + ONNX | 多周期分数神经融合 | **0.5586** | 替代规则加权共识 |
| **Regime 分类** | `vol_regime_model.txt` | LightGBM (vol+trend) | vol AUC **0.5852** | 波动率/趋势质量 |
| **分位数回归** | `quantile_h{5,12}_q{05~95}.txt` | LightGBM × 5分位 × 2周期 | — | 收益分布 + Kelly 仓位 |
| **PPO 仓位** | `ppo_position_agent.zip` | 强化学习 (stable-baselines3) | — | 动态仓位优化 (实验性) |

> 方向预测优先级: **Stacking(优先)** — 4 基模型 (LGB/XGBoost/LSTM/TFT) 5-Fold OOF → 元学习器 → bull_prob；无 Stacking 时 **LGB+LSTM+TFT+跨资产 LGB 加权** → bull_prob。Regime 过滤 + 分位数 Kelly 仓位 + 动态止损不变。

### 13.2 五层 ML 架构

```
┌─────────────────────────────────────────────────────────────┐
│  方向预测层                                                   │
│  Stacking: LGB+XGB+LSTM+TFT (5-Fold OOF) → LogisticRegression → bull_prob (优先)
│  Fallback: LGB + LSTM + TFT + 跨资产 LGB → 加权集成 bull_prob │
├─────────────────────────────────────────────────────────────┤
│  多周期融合层 (可选, 替代规则加权)                              │
│  MTF Fusion MLP (5TF×4特征=23维, AUC 0.56)                   │
│  neural_fuse_tf_scores() → 替代 fuse_tf_scores()             │
├─────────────────────────────────────────────────────────────┤
│  Regime 分类层                                               │
│  vol_regime (高波动概率) + trend_quality (趋势可持续性)         │
│  → trade_confidence → boost/dampen/neutral                   │
├─────────────────────────────────────────────────────────────┤
│  分位数风控层                                                 │
│  收益分布: q05/q25/q50/q75/q95 (h5+h12 双周期)               │
│  → Kelly 仓位: f* = (p·b - q·a)/(b·a), 半 Kelly 保守         │
│  → 动态止损: long_SL=|q05|, short_SL=q95 (1%~8%)            │
│  → 尾部风险降权: q05 < -3% → position_scale × tail_factor    │
├─────────────────────────────────────────────────────────────┤
│  执行层 (MLSignalEnhancer)                                   │
│  bull_prob ≥ 0.58 → BS 加权, SS 降权 (做多方向)               │
│  bull_prob ≤ 0.42 → SS 加权, BS 降权 (做空方向)               │
│  → 输出: enhanced_ss, enhanced_bs, kelly, dynamic_sl         │
└─────────────────────────────────────────────────────────────┘
```

### 13.3 特征工程 (94 维)

| 分类 | 特征数 | 代表特征 |
|------|--------|---------|
| 价格动量 | 14 | ret_{1,2,3,5,8,13,21}, log_ret, roc, momentum_accel, price_percentile |
| 趋势 | 11 | dist_MA{5,10,20,60}, slope_MA, ma_cross_5_20/10_60 |
| MACD | 7 | macd_dif/dea/bar, macd_bar_change, macd_cross |
| 振荡器 | 11 | rsi6/12 + slope, kdj_k/d/j, cci + slope |
| 波动率 | 8 | atr_14/5, hvol_5/20, hvol_ratio, hl_position, dist_high/low_20 |
| 量价 | 8 | vol_ratio, vol_change, vol_price_corr, taker_buy_ratio |
| 微结构 | 9 | body/shadow_ratio, is_bull, bull/bear_streak, avg_body_3, max_range_3 |
| 时间 | 4 | hour_sin/cos, dow_sin/cos |
| 排名/Sharpe | 4 | cum_ret_{5,20}_rank, sharpe_{5,20} |
| 合约衍生品 | 4 | funding_rate, funding_rate_ma, oi_change, oi_change_5 |
| **跨资产** | **21** | btc/sol/bnb: ret_{1,5,21}, corr_{20,60}, rel_strength, vol_ratio |

### 13.4 Walk-Forward 验证

```python
# 扩展窗口 + Purge Gap 防未来泄露:
# |----train (expanding)----|--purge (24bar)--|--val (168bar)--|--test (120bar)--|
# 重训间隔: 120 bars (5天), 最小训练窗口: 720 bars (30天)
# 特征精选: 首窗口 69→30 核心特征, 后续 fold 复用
# 多尺度: 3h/5h/12h/24h 四个 horizon 模型投票, 跨尺度共识
```

### 13.5 实盘 ML 增强流程 (正式模式, 2026-02-20 起)

```
live_signal_generator.py → compute_signals_six() → 六书 SS/BS
  ↓ (use_ml_enhancement=True, shadow_mode=False ← 已开启正式增强)
ml_live_integration.py → MLSignalEnhancer.enhance_signal()
  ├─ 方向: Stacking(优先, 1h OOF AUC=0.5883) 或 LGB+LSTM+TFT+跨资产LGB 加权 → bull_prob
  ├─ Stacking 元学习器系数: LGB=1.24, 跨资产LGB=1.47, LSTM=0.15, XGB≈0, TFT≈0
  ├─ Regime: vol_prob + trend_prob → boost/dampen/neutral
  ├─ 分位数: q05~q95 → Kelly 仓位 + 动态止损 + 尾部降权
  └─ 输出: enhanced_ss, enhanced_bs, kelly_fraction, dynamic_sl
  ↓
正式模式 (当前): SS/BS 用 ML 增强值, 仓位用 kelly, 止损用 dynamic_sl
shadow 模式 (如需回退): 只记录到日志, 不实际修改信号
```

---

<h2 id="section-14">十四、GPU 训练架构 (H800)</h2>

### 14.1 三机协作架构

```
本机 (开发机, macOS)              H800 GPU (办公内网)           阿里云 (生产)
┌───────────────────┐           ┌───────────────────┐       ┌───────────────┐
│ Binance 数据拉取   │           │ GPU 模型训练       │       │ Flask Web     │
│ 代码开发 + 回测    │  tar.gz   │ 超参优化 (Optuna) │       │ 实盘信号检测   │
│ fetch_5year_data  │ ───SCP──→ │ 深度学习 (LSTM)   │       │ 交易执行      │
│ pack_for_h800.sh  │           │ train_gpu.py      │       │               │
│                   │  模型回传  │                   │ ──→   │ 模型推理(CPU) │
│ data/ml_models/ ← │ ←──SCP── │ data/ml_models/   │       │               │
└───────────────────┘           └───────────────────┘       └───────────────┘
```

> **约束**: H800 位于办公内网，需跳板机登录，无法直接访问 Binance API。所有训练数据必须在本机下载后离线传输。

### 14.2 训练数据管线

| 数据类型 | 来源 | 缓存路径 | 覆盖范围 | 下载工具 |
|---------|------|---------|---------|---------|
| K线 (OHLCV) | Binance Spot API | `data/klines/{SYMBOL}/{interval}.parquet` | 5年, 4交易对×4周期 | `fetch_5year_data.py` |
| Mark Price | Futures markPriceKlines | `data/mark_klines/ETHUSDT/{interval}.parquet` | 5年 | `fetch_5year_data.py` |
| Funding Rate | Futures fundingRate | `data/funding_rates/{SYMBOL}_funding.parquet` | 5年 (~5500条/对) | `fetch_5year_data.py` |
| Open Interest | Futures openInterestHist | `data/open_interest/{SYMBOL}/{period}.parquet` | ~30天 (API限制) | `fetch_5year_data.py` |

交易对: ETHUSDT (主力), BTCUSDT, SOLUSDT, BNBUSDT
周期: 15m (ML细粒度), 1h (主力训练), 4h (中期趋势), 24h (宏观背景)

> **Binance OI API 限制**: `startTime` 最多约30天前，`binance_fetcher.py` 已实现分段拉取 (29天/段)。

### 14.3 GPU 训练模式 (train_gpu.py, 12 种模式)

| 模式 | 命令 | 说明 |
|------|------|------|
| LightGBM | `--mode lgb` | Walk-Forward, Optuna 优化, 多尺度集成 |
| LSTM+Attention | `--mode lstm` | 双向LSTM, BF16 混合精度, 时间注意力 |
| Optuna 超参 | `--mode optuna` | 贝叶斯 TPE, 200 trials, MedianPruner |
| 回测优化 | `--mode backtest` | Optuna 优化 SL/TP/阈值等回测参数 |
| **TFT** | `--mode tft` | Temporal Fusion Transformer, 94维特征 |
| **跨资产 LGB** | `--mode cross_asset` | BTC/SOL/BNB 联动特征, 94维 |
| **增量 WF** | `--mode incr_wf` | 增量 Walk-Forward, 月度滚动重训 |
| **MTF 融合** | `--mode mtf_fusion` | 多周期分数 MLP, 替代规则加权 |
| **PPO 仓位** | `--mode ppo` | 强化学习仓位优化 (stable-baselines3) |
| **ONNX 导出** | `--mode onnx` | LSTM/TFT/MLP → ONNX 加速推理 |
| **定时重训** | `--mode retrain` | 自动检测数据更新并触发重训 |
| **Stacking Ensemble** | `--mode stacking` | 4 基模型 (LGB/XGBoost/LSTM/TFT) 5-Fold OOF → LogisticRegression 元学习器 |
| 批量 | `--mode all_v4` | v3 + Stacking Ensemble |

### 14.4 关键模型架构

**LSTM+Attention** (方向预测, AUC 0.54):
```python
LSTMAttention(
    LSTM(input_dim=73, hidden=128, layers=2, bidirectional=True, dropout=0.3)
    → Attention(hidden*2 → 1)  # 时间步加权聚合
    → FC(256→64) → GELU → Dropout(0.2) → FC(64→1) → Sigmoid
)
# BF16 混合精度, CosineAnnealing LR, 梯度裁剪, early stopping
```

**TFT** (可解释时序预测, 94 维特征):
```python
EfficientTFT(
    input_proj(94→64) → PositionalEncoding(seq=96)
    → TransformerEncoder(d_model=64, n_heads=4, layers=2)
    → global_pool → FC(64→1) → Sigmoid
)
# 参数量: 148K (轻量), ONNX 导出支持
```

**MTF Fusion MLP** (多周期融合, AUC 0.56):
```python
MTFFusionMLP(
    23维输入: 5TF × (ss, bs, net, max) + 大小同向 + 大周期均值 + 小周期均值
    → FC(23→64) → BatchNorm → GELU → Dropout(0.3)
    → FC(64→32) → BatchNorm → GELU → Dropout(0.2)
    → FC(32→1) → Sigmoid
)
# Focal Loss (gamma=2.0, alpha=0.25) 处理类别不平衡
```

### 14.5 核心文件

| 文件 | 作用 |
|------|------|
| `fetch_5year_data.py` | 批量下载 5 年训练数据 (4对×4周期+衍生品) |
| `pack_for_h800.sh` | 数据完整性检查 + 打包 + 传输提示 |
| `setup_h800.sh` | H800 一键环境搭建 (GPU检测 + conda + 依赖 + 验证) |
| `requirements-gpu.txt` | GPU 依赖 (PyTorch CUDA, Optuna, TensorBoard, stable-baselines3) |
| `verify_data.py` | 训练数据完整性验证 |
| `train_gpu.py` | **GPU 离线训练入口** (12种模式, 含 Stacking, 不依赖 API) |

### 14.6 模型部署回路

```
H800 训练产出 (data/ml_models/):
  stacking_meta.pkl + stacking_meta.json  Stacking 元学习器 (4 基模型 OOF → LogisticRegression)
  stacking_lgb_*.txt / stacking_xgb_*.json / stacking_lstm_*.pt / stacking_tft_*.pt  Stacking 基模型
  lgb_direction_model.txt    LGB 方向 (Optuna, AUC 0.55)
  lgb_cross_asset_1h.txt     跨资产 LGB (94维, AUC 0.55)
  lstm_1h.pt + .onnx         LSTM+Attention (AUC 0.54)
  tft_1h.pt + .onnx          TFT (148K参数)
  mtf_fusion_mlp.pt + .onnx  MTF 融合 MLP (AUC 0.56)
  vol/trend_regime_model.txt Regime 分类 (vol AUC 0.59)
  quantile_h{5,12}_q*.txt   分位数模型 (10个)
  ppo_position_agent.zip    PPO 仓位 (实验性)
  *.meta.json / ensemble_config.json  特征名/集成配置/训练指标

部署流程:
  git push (模型文件在 git 跟踪中)
  → 服务器 git pull → systemctl restart
  → MLSignalEnhancer.load_model() 自动加载
  → CPU 推理 (LGB 原生 / LSTM+TFT ONNX / MLP PyTorch CPU)

当前状态: 正式增强模式 (shadow_mode=False, ML 实际修改信号)
```

---

<h2 id="section-15">十五、全局代码审视与架构改进计划 (2026-02-18 Review)</h2>

### 15.0 全局代码 Review 发现

#### A. 代码结构问题

| # | 问题 | 严重度 | 详情 |
|---|------|--------|------|
| A1 | **模型类重复定义 ×12 处** | P0 | `LSTMAttention` 定义 5 处 (train_gpu.py×3, ml_live_integration.py×2)；`EfficientTFT` 定义 6 处；`LSTMMultiHorizon` 定义 2 处。同一网络结构散落多处，修改一处易遗漏其余。 |
| A2 | **optimize_six_book.py 5,488 行** | P1 | 单文件承担回测引擎、参数优化、多周期融合、regime 分析、微结构叠加等多个职责。新增功能无法独立测试。 |
| A3 | **train_gpu.py 3,798 行** | P1 | 包含 12 种训练模式、模型定义、ONNX 导出、数据预处理。模式间无代码复用接口。 |
| A4 | **live_signal_generator.py 冗余导入** | P2 | 直接导入 `strategy_enhanced`, `ma_indicators`, `candlestick_patterns` 等 7 个模块，但信号计算已通过 `signal_core.compute_signals_six()` 封装。部分直接导入仅用于 `evaluate_action()` 中零散调用。 |
| A5 | **app.py 2,180 行** | P2 | Web 路由、API 端点、subprocess 调用混杂。已有 `web_routes/` 拆分但不完全。 |

#### B. 配置与一致性问题

| # | 问题 | 严重度 | 详情 |
|---|------|--------|------|
| B1 | **三层超时不对齐** | P0 | `app.py` subprocess timeout=300s < Gunicorn timeout=360s。长时间多周期检测中，subprocess 先超时导致 Gunicorn worker 空转 60s。 |
| B2 | **deploy.sh ML_ENABLE_STACKING=1 硬编码** | P1 | Stacking 启用/禁用依赖环境变量，但部署脚本始终写入 `=1`。若需紧急关闭 Stacking，必须改脚本而非配置。 |
| B3 | **技术文档过时** | P0 (已修复) | 文档仍显示 "shadow 模式运行中"，实际 shadow_mode 已于 2026-02-20 关闭。本次已修正。 |

#### C. ML 模型质量观察

| # | 发现 | 影响 | 建议 |
|---|------|------|------|
| C1 | **Stacking 元学习器系数分析** | 信息 | XGBoost 系数=-0.02 (近零), TFT=-0.12 (负向)。元学习器实质上在做 "LGB + 跨资产LGB + hvol_20 偏置"，XGB/TFT 对最终预测贡献极低。 |
| C2 | **hvol_20 系数=3.35 异常大** | 警告 | 作为 extra_feature 的 hvol_20 系数远大于任何基模型，可能是过拟合的波动率偏好。高波动时期 bull_prob 被 hvol 大幅拉偏。 |
| C3 | **OOF AUC=0.5883 vs Test AUC=0.5429** | 警告 | Gap=0.045，虽未超过门控阈值(0.10)，但方向持续 OOF > Test，说明元学习器有轻微过拟合。 |
| C4 | **ONNX 推理路径未验证一致性** | P1 | LSTM/TFT 的 ONNX 导出后未做 PyTorch vs ONNX 输出数值一致性对比。 |

#### D. 测试覆盖不足

| # | 问题 | 建议 |
|---|------|------|
| D1 | 7 个测试文件, 无 signal_core.py 单元测试 | 添加 `test_signal_core.py` 覆盖 `compute_signals_six()` / `calc_fusion_score_six()` |
| D2 | 无 ML 端到端推理测试 | 添加 `test_ml_e2e.py` 验证特征→推理→增强全链路 |
| D3 | 无回测结果回归测试 | 添加固定参数回测的金标准对比，防止代码改动导致回测结果漂移 |

### 15.1 架构改进计划 (P0/P1/P2)

#### P0 — 必须立即修复

| # | 改进项 | 预计工作量 | 收益 |
|---|--------|-----------|------|
| P0-1 | **统一模型定义模块**: 将 `LSTMAttention`, `EfficientTFT`, `LSTMMultiHorizon` 抽取到 `ml_models_def.py`，train_gpu.py 和 ml_live_integration.py 统一 import | 2h | 消除 12 处重复定义，修改一处自动同步 |
| P0-2 | **三层超时对齐**: `app.py` subprocess timeout 从 300s → 360s，与 Gunicorn 和 Nginx 对齐 | 5min | 消除 subprocess 先超时导致 worker 空转 |
| P0-3 | **Stacking 元学习器系数监控**: 在 `check_ml_health.py` 中添加系数合理性检查（负值基模型警告、extra_feature 过大警告） | 1h | 提前发现过拟合信号 |
| P0-4 | **信号漂移回归测试**: 创建 `test_signal_regression.py`，固定输入数据对比 `compute_signals_six()` 输出，任何改动导致输出变化即报错 | 2h | 防止策略逻辑被意外改动 |

#### P1 — 近期重点改进

| # | 改进项 | 预计工作量 | 收益 |
|---|--------|-----------|------|
| P1-1 | **拆分 optimize_six_book.py**: 按职责拆为 `backtest_engine.py`(回测循环)、`regime_analysis.py`(regime 计算)、`microstructure_overlay.py`(微结构)、`position_sizing.py`(仓位计算) | 1-2d | 可独立测试、降低认知负载 |
| P1-2 | **拆分 train_gpu.py**: 按模型类型拆为 `train_lgb.py`, `train_lstm.py`, `train_tft.py`, `train_stacking.py` + 共享 `train_utils.py` | 1d | 各模型独立维护 |
| P1-3 | **ONNX 推理一致性验证**: 添加 `test_onnx_consistency.py`，对比 PyTorch 和 ONNX 推理输出的数值偏差（atol=1e-4） | 4h | 确保 ONNX 加速不引入精度问题 |
| P1-4 | **deploy.sh ML 配置外部化**: 将 ML 环境变量从脚本硬编码改为读取 `ml_deploy_config.env` 文件 | 2h | 灵活切换 Stacking 开关而无需改脚本 |
| P1-5 | **live_signal_generator.py 清理冗余导入**: 移除 7 个未被直接使用的策略模块导入 | 30min | 减少启动耗时和内存占用 |
| P1-6 | **统一训练-推理模型契约**: 定义标准化输出格式 `{model, meta, schema, version}`，推理侧严格校验 | 4h | 防止模型版本不匹配 |

#### P2 — 中长期优化

| # | 改进项 | 预计工作量 | 收益 |
|---|--------|-----------|------|
| P2-1 | **结构化日志 (JSON)**: 实盘信号日志从文本格式改为 JSON Lines，便于 ELK/Grafana 分析 | 4h | 自动化监控 |
| P2-2 | **模型漂移监控**: 每日自动对比 bull_prob 分布与训练期分布（KS 检验），偏移超阈值告警 | 8h | 提前发现数据分布变化 |
| P2-3 | **H800 自动晋升门禁**: `train_gpu.py` 训练完成后自动评估质量指标，不达标标记 `research_only` | 4h | 防止低质量模型进入生产 |
| P2-4 | **app.py 进一步路由拆分**: 将剩余路由迁移到 `web_routes/` | 4h | 主文件控制在 500 行内 |
| P2-5 | **XGBoost/TFT 基模型评估**: 鉴于系数近零/负，评估是否移除以简化 Stacking | 2h | 减少推理延迟 |
| P2-6 | **hvol_20 extra_feature 正则化**: 在 Stacking 元学习器训练中加强 L2 正则化或移除 hvol_20 | 1h | 降低过拟合风险 |

### 15.2 策略优化路线图

> ✅ **v10.0 已解决 P33/P27/P30 (2026-02-15)**:
> IS PF 0.90→1.31, OOS -3.8%→+7.9%。B1b 硬禁止已被 Leg Budget 替代, Funding 已纳入 trade PnL。

| 优先级 | 编号 | 方向 | 预期收益 | 状态 |
|--------|------|------|---------|------|
| ~~P0~~ | ~~P33~~ | ~~B1b 软化~~ | ~~恢复 PF >1.0~~ | **✅ v10 已完成**: Soft Veto + Leg Budget |
| ~~P0~~ | ~~P34~~ | ~~P24 止损优化: ATR-based~~ | ~~恢复 trend short 利润~~ | **✅ v10.1 已完成**: ATR-SL + regime-specific mult |
| ~~P0~~ | ~~P35~~ | ~~OOS 扩展~~ | ~~获得 OOS 证据~~ | **✅ v10.1 验证**: OOS +16.0%, Calmar=0.82 |
| P0 | P25 | P18+P23 组合 Walk-Forward 验证 | 确认 OOS 泛化是否稳定 | 待执行 |
| ~~P0~~ | ~~P26~~ | ~~P21 R% 调参~~ | ~~消灭大亏单~~ | **✅ v10.1 已完成**: R%=2.5%, 与 ATR-SL 绑定 |
| ~~P1~~ | ~~P27~~ | ~~Funding 现金流纳入回测~~ | ~~消除收益虚高~~ | **✅ v10 已完成**: Funding-in-PnL |
| P1 | P28 | Anti-Squeeze 回测验证 | 验证 trend/high_vol short 改善 | funding_z 已接入, 需单独验证触发率 |
| P1 | **P36** | **DIV 权重优化**: P18 shrinkage 混合 (neutral DIV=0.25) | 改善 neutral 信号质量 | **待执行** (DIV 0.50 全局下调已失败) |
| P1 | **P37** | **Regime 连续化**: SL/杠杆/阈值 sigmoid 过渡 | 消除参数悬崖 | **待执行** (Phase 2) |
| P2 | P29 | CS-KDJ 去相关 | 消除虚假共识 | 待设计 |
| ~~P2~~ | ~~P30~~ | ~~Leg 风险预算~~ | ~~资金迁移~~ | **✅ v10 已完成**: use_leg_risk_budget |
| P2 | P31 | 动态 Regime 阈值 (滚动百分位) | 适应市场周期变化 | 待设计 |
| P2 | P32 | MAE 追踪 + 数据驱动止损校准 | ATR mult 精确校准 | **待执行** (Phase 2) |

### 15.3 GPU/ML 训练路线图

| 优先级 | 方向 | 说明 | 状态 |
|--------|------|------|------|
| ~~P0~~ | ~~LightGBM Walk-Forward 基线~~ | 1h 周期, 5年数据, GPU 加速 | ✅ 已完成, AUC 0.5537 |
| ~~P0~~ | ~~LSTM+Attention 深度模型~~ | BF16 混合精度, 双向 LSTM | ✅ 已完成, AUC 0.5366 |
| ~~P0~~ | ~~Optuna 超参搜索~~ | 替代手工网格搜索, TPE 采样 | ✅ 已完成, 1000 trials |
| ~~P0~~ | ~~Stacking Ensemble~~ | 4 基模型 OOF → 元学习器 | ✅ 已完成, OOF AUC 0.5883 |
| P0 | **Stacking 元学习器优化** | 移除零贡献基模型 (XGB/TFT), 正则化 hvol_20 | 待执行 |
| P1 | 多周期特征融合 | 15m/1h/4h/24h 跨周期特征堆叠 | 待开发 |
| P1 | ONNX 一致性验证 + 加速部署 | PyTorch vs ONNX 数值对比 | 待验证 |
| P2 | RAPIDS cuDF/cuML 加速 | GPU 加速特征计算和信号生成 | 待评估 |
| P2 | Ray Tune 分布式 HPO | 多 GPU 并行超参搜索 | 待评估 |

### 15.4 长期架构方向 (LLM 共识)

1. **策略拆腿 + 组合风险预算**: 按 (regime × direction) 拆成独立 legs, 每个 leg 有独立风控 KPI 和风险预算。v10 的 `use_leg_risk_budget` + `risk_budget_neutral_short=0.10` 是这个方向的第一步落地 (从 B1b 的 0% → 10% 软降权)。下一步: 让 budget 动态调整 (基于近 N 笔胜率/PnL)。

2. **从 alpha 策略到 hedge 策略**: short legs 的目标从"追求 alpha"转为"降低组合尾部风险"。仅在"崩盘风险"上升时开空（OI 快速上升 + funding 极端 + liquidation flow），仓位小、止损宽、时间短。

3. **更真实的回测口径**: fee ×2 / slippage ×2 压力测试; 成交价模型从 T+1 open 升级为包含滑点的分布模型; Mark Price 用于强平和未实现 PnL。

4. **ML 模型治理**: 模型版本化 (hash + schema)、A/B 灰度 (shadow 对比 live)、自动漂移监控 (KS 检验)、晋升门禁 (AUC/gap/coverage 三重门控)。

---

### 版本变更日志

| 版本 | 日期 | 核心变更 |
|------|------|---------|
| **v10.2+ML v3.2** | **2026-02-18** | **全局代码审视**: 发现 12 处模型类重复定义、三层超时不对齐、Stacking 元学习器 XGB/TFT 近零贡献 + hvol_20 过大系数；制定 P0-P2 架构改进计划；ML 正式增强已运行 (shadow_mode=False) |
| v10.2+ML v3.1 | 2026-02-20 | Stacking 别名一致性脚本 + check_ml_health.py shadow 误报修复 + 样本量门控 (Stacking≥20k, TabNet≥10k) |
| v10.2+ML v3 | 2026-02-19 | Stacking Ensemble: 4 基模型 (LGB/XGBoost/LSTM/TFT) 5-Fold OOF → LogisticRegression 元学习器；方向预测优先 Stacking，无则回退加权集成；train_gpu.py 12 种模式 (含 stacking/all_v4) |
| v10.2+ML v2 | 2026-02-19 | 8 模型矩阵部署: LGB/LSTM/TFT/跨资产LGB/MTF融合MLP/Regime/分位数/PPO, Kelly 仓位+动态止损, ONNX 加速, shadow 模式上线 |
| v10.2+ML | 2026-02-18 | ML 预测子系统 + H800 GPU 离线训练架构: LightGBM/LSTM/Optuna, 94维特征, Walk-Forward 验证, 三机协作数据管线 |
| v10.2 | 2026-02-15 | Regime Sigmoid + Leg Budget 5×2 + MAE 追踪: Phase 2 连续化改造 |
| **v10.1** | **2026-02-15** | **ATR-SL + P21绑定: OOS +8.7%→+16.0%, OOS PF 1.04→1.31, 尾部风险-55%, DIV 0.50 实验失败保留0.70** |
| v10.0 | 2026-02-15 | Soft Veto + Leg Budget + Funding-in-PnL: IS PF 0.90→1.31, OOS -3.8%→+7.9%, 连续化替代硬门控 |
| v9.0.1 | 2026-02-15 | v5 完整回测验证: IS PF=0.90 ⚠️, 发现 B1b 误杀盈利信号 |
| v9.0 | 2026-02-15 | B1b禁止neutral short + P24 regime SL + Anti-Squeeze Filter + 实盘衍生品数据 + 架构简化 |
| v8.1 (P20) | 2026-02-15 | P17口径统一(24h)+P20空头追踪收紧(60%→40%) |
| v8.0 (B3+P13) | 2026-02-15 | P13连续追踪止盈 + P6-P16全量诊断, OOS PF 2.33→2.51 |
| v7.0 (B3) | 2026-02 | P4→B3: neutral short 四重防护, OOS PF 1.67→2.33 |
| v6.0 | 2026-02 | P1优化: short_trail 0.20→0.19, WR 61.9%→63.1% |
| v5.1 | 2026-01 | P0前视修复 + 参数重优化 |
| v5.0 | 2026-01 | 结构折扣 + 冲突折扣初版 |
| v4.0 | 2025-12 | 六书融合 + 多周期共识 + Regime 检测 |
