# ETH/USDT 六书融合量化交易策略 v10.1 — 完整技术规格书

**交易标的**: ETH/USDT 永续合约 (Binance Futures)
**主时间框架**: 1h K线
**决策时间框架**: 15m, 1h, 4h, 24h (多周期联合决策)
**回测区间 (IS)**: 2025-01 ~ 2026-01 (12个月)
**OOS验证区间**: 2024-01 ~ 2024-12 (12个月)
**Walk-Forward验证**: 6窗口滚动 (2024Q1 ~ 2025Q4)
**初始资金**: $100,000 USDT
**策略版本**: v10.1 — ATR-SL + P21 绑定版 (波动率驱动止损 + 仓位联动)
**生产配置版本**: v5 (`STRATEGY_VERSION=v5`)

> **v10.1 Phase 1 改造摘要** (2026-02-15):
> 基于跨模型共识建议 (Claude/GPT Pro/Gemini/Grok)，执行 Phase 1 改造:
> 1. **ATR-SL**: 波动率驱动止损替代 P24 固定百分比，regime-specific ATR 乘数 (trend=3.5x, neutral=2.0x)
> 2. **P21 Risk-per-trade 绑定**: 仓位与止损联动 (size = risk_budget / stop_distance)，止损放宽不增大亏损
> 3. **Funding Z-Score**: Anti-Squeeze 改用 funding_z 替代原始 funding_rate
> 4. **OI 数据审计**: merge_perp_data 后自动统计 coverage/staleness
>
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

### 2.1 五层架构 (v10.0)

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

### 2.2 数据流水线 (v9.0 新增)

```
  实盘 (live_signal_generator.py)           回测 (optimize_six_book.py)
  ┌─────────────────────────┐              ┌─────────────────────────┐
  │ fetch_binance_klines()  │              │ 从本地 DB 加载 K线       │
  │ fetch_mark_price_klines()│←── 新增 ──→ │ merge_perp_data()       │
  │ fetch_funding_rate_history()│           │ (mark/funding/OI 合并)  │
  │ fetch_open_interest_history()│          │                         │
  │ merge_perp_data_into_klines()│         │ _compute_microstructure()│
  └──────────┬──────────────┘              └──────────┬──────────────┘
             │                                        │
             ▼                                        ▼
    compute_signals_six()                    _vectorized_fuse_scores()
    calc_fusion_score_six()                  _apply_microstructure_overlay()
                                             └─ Anti-Squeeze Filter
```

### 2.3 核心代码文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `optimize_six_book.py` | ~4900 | 回测引擎核心: Regime检测、持仓管理、Anti-Squeeze |
| `live_config.py` | ~977 | 策略参数、版本管理 (v1-v5)、风控配置 |
| `signal_core.py` | ~978 | 六书信号计算、融合评分（单bar+批量向量化） |
| `live_signal_generator.py` | ~675 | 实盘信号: 数据获取(含衍生品)、信号计算、交易决策 |
| `binance_fetcher.py` | ~690 | Binance API: K线+Mark Price+Funding+OI 获取/缓存 |
| `multi_tf_consensus.py` | ~641 | 多周期共识、TF权重、链式检测 |
| `live_runner.py` | — | 实盘运行入口、systemd 管理 |
| `app.py` | — | Flask Web 应用、监控面板 |

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

<h2 id="section-13">十三、未来优化方向</h2>

### 13.1 优先级路线图

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

### 13.2 长期架构方向 (LLM 共识)

1. **策略拆腿 + 组合风险预算**: 按 (regime × direction) 拆成独立 legs, 每个 leg 有独立风控 KPI 和风险预算。v10 的 `use_leg_risk_budget` + `risk_budget_neutral_short=0.10` 是这个方向的第一步落地 (从 B1b 的 0% → 10% 软降权)。下一步: 让 budget 动态调整 (基于近 N 笔胜率/PnL)。

2. **从 alpha 策略到 hedge 策略**: short legs 的目标从"追求 alpha"转为"降低组合尾部风险"。仅在"崩盘风险"上升时开空（OI 快速上升 + funding 极端 + liquidation flow），仓位小、止损宽、时间短。

3. **更真实的回测口径**: fee ×2 / slippage ×2 压力测试; 成交价模型从 T+1 open 升级为包含滑点的分布模型; Mark Price 用于强平和未实现 PnL。

---

### 版本变更日志

| 版本 | 日期 | 核心变更 |
|------|------|---------|
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
