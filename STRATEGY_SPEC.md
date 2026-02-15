# MACD Analysis 六书融合策略 — 完整技术规格

> 版本: v7.0 B3 (2026-02-15, run#499 baseline)
> 标的: ETH/USDT 永续合约 + 现货
> 主周期: 1h | 决策周期: 15m + 1h + 4h + 24h

---

## 一、策略架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        多周期联合决策层                               │
│    15m(w=3) + 1h(w=8) + 4h(w=15) + 24h(w=28)                      │
│    → 加权共识 → 共振链检测 → 大周期定调 → 决策矩阵                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │ weighted_ss, weighted_bs
┌────────────────────────────▼────────────────────────────────────────┐
│                     六书信号融合层 (每个 TF 独立计算)                 │
│                                                                     │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────┐ ┌─────┐│
│  │背离(DIV)│ │均线(MA) │ │蜡烛图(CS)│ │布林带(BB)│ │量价(VP)│ │KDJ ││
│  │div 55%  │ │ kdj 15% │ │ bonus%  │ │ bonus%  │ │bonus%│ │bonus││
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └──┬───┘ └──┬──┘│
│       └──────┬────┘           └──────┬─────┘         └────┬───┘   │
│         base_score            veto/bonus 层           veto/bonus   │
│   (div*0.55+kdj*0.15)   2+本反向≥25 → dampen×0.30    KDJ timing  │
│   +bonus(BB/VP/CS)       同向≥15 → +bonus              adjustment │
└────────────────────────────┬────────────────────────────────────────┘
                             │ sell_score(SS), buy_score(BS)
┌────────────────────────────▼────────────────────────────────────────┐
│                      微结构叠加 + 双引擎切换                         │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────┐ │
│  │ 微结构(基差/OI/  │  │ 双引擎选择:      │  │ Vol-Targeting:     │ │
│  │ 资金费率代理)    │  │ trend: 顺势入场  │  │ margin *= target/  │ │
│  │ → SS/BS ±8~10% │  │ reversion: 反转  │  │ realized_vol       │ │
│  │ → 开仓限制      │  │ → 阈值/杠杆调整  │  │ (0.45x ~ 1.35x)   │ │
│  └─────────────────┘  └─────────────────┘  └────────────────────┘ │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                          门控过滤层                                  │
│                                                                     │
│  ┌──────────────┐ ┌───────────────┐ ┌────────────────────────────┐ │
│  │ Regime 动态   │ │ Neutral 结构  │ │ 趋势/冲突 做空抑制          │ │
│  │ 阈值调整      │ │ 质量折扣      │ │ (regime_short_gate,        │ │
│  │ 5种市场状态   │ │ 0确认→禁止   │ │  short_conflict_discount,  │ │
│  │              │ │ 1确认→0.05x  │ │  neutral:SS≥60)            │ │
│  │              │ │ 2确认→0.15x  │ │                            │ │
│  │              │ │ 3确认→0.50x  │ │                            │ │
│  │              │ │ 4+确认→1.0x  │ │                            │ │
│  └──────────────┘ └───────────────┘ └────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                      执行引擎 (FuturesEngine)                       │
│                                                                     │
│  信号在 bar[i-1] close 计算 → bar[i] open 执行 (延迟一根)           │
│  费用: Taker 0.05% + Slippage 0.1% + Funding ±0.01%/8h            │
│  保证金: 逐仓隔离, 最大5x杠杆, 可用=usdt-frozen                    │
│  止损止盈: SL/TP/Trail/PartialTP1(+12%)/PartialTP2(+25%)          │
│  风控: 反手最少8bar, 冷却6bar, 强平6x冷却                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、六维信号计算

每个时间框架独立计算 6 维信号，产出 `sell_score` 和 `buy_score`（0~100 范围）。

### 2.1 背离信号 (Divergence) — div_weight=0.55

基于 MACD 的顶底背离检测，输入主周期 + 8h 辅助周期。

```python
# 信号计算入口 (signal_core.py:59-64)
lookback = max(60, min(200, len(df) // 3))
div_signals = analyze_signals_enhanced(df, lookback)  # 返回 {timestamp: signal_dict}
```

**背离类型与分值**:
| 类型 | 分值 | 条件 |
|------|------|------|
| 双隔堆顶背离 | +30 | sep_divs_top >= 2 |
| 单隔堆顶背离 | +18 | sep_divs_top >= 1 或 separated_top >= 1 |
| 面积顶背离 | +12 | area_top_div >= 1 |
| DIF 顶背离 | +8 | dif_top_div >= 1 |
| 空头力竭 | +18 | exhaust_sell = True |
| 零轴回归顶 | +10 | zero_returns_top >= 1 |
| 趋势加成 | ×1.3 | 下跌趋势中 (c5 < c20 × 0.99) |

底背离类似但方向相反，趋势加成 ×1.2。

**8h 辅助背离合并** (signal_core.py:295-302):
```python
merged["top"] = main_top * 1.0 + sig_8h_top * 0.5   # 加权合并
merged["bottom"] = main_bottom * 1.0 + sig_8h_bottom * 0.5
# 其他字段取两者最大值
```

### 2.2 均线信号 (MA)

基于多条移动平均线的排列、交叉、斜率综合评分。
- 多头排列(MA5>MA10>MA20>MA48): +10
- 金叉(短MA上穿长MA): +8~12
- 斜率加速: 额外加分
- 排列加成: 空头排列时 sell_score ×1.10，多头排列时 buy_score ×1.10

### 2.3 蜡烛图 (Candlestick) — cs_bonus=0.06

检测经典K线形态：吊人线、倒锤线、吞没、十字星等。
- 同向确认(cs_sell ≥ 25): +cs_bonus(6%)
- 反向 veto: 计入否决票

### 2.4 布林带 (Bollinger) — bb_bonus=0.10

基于价格与布林带上下轨的相对位置。
- 触及上轨 + 缩口: sell_score 升高
- 触及下轨 + 扩口: buy_score 升高
- 同向确认(bb_sell ≥ 15): +bb_bonus(10%)

### 2.5 量价 (Volume-Price) — vp_bonus=0.08

基于成交量异常检测和量价背离。
- 放量下跌(量价确认): sell_score 升高
- 缩量上涨(量价背离): sell 信号增强
- 同向确认(vp_sell ≥ 15): +vp_bonus(8%)

### 2.6 KDJ — kdj_bonus=0.09, kdj_weight=0.15

基于 KDJ 指标的超买超卖和交叉。
- KDJ 高位死叉: kdj_sell 升高
- KDJ 低位金叉: kdj_buy 升高
- 同向确认(kdj_sell ≥ 15): +kdj_bonus(9%)
- **KDJ timing**: 同向 ×1.25, 正常确认 ×1.12, 反向 ×0.70

### 2.7 融合模式: c6_veto_4

当前部署的融合模式 (signal_core.py:388-420):

```python
# 基础分数 = 55%背离 + 15%KDJ (div_weight=0.55, kdj_weight=0.15)
base_sell = div_sell * 0.55 + kdj_sell * 0.15
base_buy  = div_buy  * 0.55 + kdj_buy  * 0.15
# 均线排列加成
base_sell *= ma_arr_bonus_sell  # 空头排列 ×1.10
base_buy  *= ma_arr_bonus_buy   # 多头排列 ×1.10

# 否决计数: 4本书(BB/VP/CS/KDJ)中,反方向≥veto_threshold(25)的数量
sell_vetoes = sum(1 for s in [bb_buy, vp_buy, cs_buy, kdj_buy] if s >= 25)
buy_vetoes  = sum(1 for s in [bb_sell, vp_sell, cs_sell, kdj_sell] if s >= 25)

if sell_vetoes >= 2:
    sell_score = base_sell * 0.30   # 否决: 大幅削弱
else:
    bonus = 0
    if bb_sell >= 15: bonus += 0.10   # bb_bonus
    if vp_sell >= 15: bonus += 0.08   # vp_bonus
    if cs_sell >= 25: bonus += 0.06   # cs_bonus
    if kdj_sell >= 15: bonus += 0.09  # kdj_bonus
    sell_score = base_sell * (1 + bonus)  # 加成
```

> **已知问题**: DIV 在 neutral regime 中 Cohen's d ≈ -0.04（几乎无判别力），但占权重 55%。
> CS(d=0.40) 和 KDJ(d=0.42) 才是真正有效的确认书。
> 现通过结构折扣缓解，但从源头修正（regime-adaptive 融合权重）是下一步改进方向。

---

## 三、多周期联合决策

### 3.1 权重分配

```python
TF_WEIGHT = {
    '15m': 3,   # 微观入场计时
    '1h':  8,   # 核心信号源 (primary_tf)
    '4h':  15,  # 中期趋势确认 (大周期门槛)
    '24h': 28,  # 宏观趋势锚定
}
# 总权重 = 54, 大周期(≥4h) 占比 79.6%
```

### 3.2 Regime 驱动的动态权重 (multi_tf_consensus.py:393-409)

```python
if regime_label in ('high_vol_choppy', 'high_vol'):
    # 高波动: 降低小周期权重(噪声大), 抬高大周期
    if tf_min < 60:   base_w *= 0.6     # 15m 权重降40%
    if tf_min >= 240:  base_w *= 1.15   # 4h/24h 权重升15%
elif regime_label in ('trend', 'low_vol_trend'):
    # 趋势市: 大周期更重要
    if tf_min >= 240:  base_w *= 1.25   # 4h/24h 权重升25%
    if tf_min < 60:    base_w *= 0.8    # 15m 权重降20%
```

### 3.3 加权融合公式

```python
weighted_ss = Σ(tf_scores[tf].ss × effective_weight[tf]) / Σ(effective_weight)
weighted_bs = Σ(tf_scores[tf].bs × effective_weight[tf]) / Σ(effective_weight)
```

### 3.4 共振链检测

扫描相邻 TF 连续同向的链条（允许 1 个 hold 间隔）：
```python
# 如 15m→1h→4h 均看空 → 链长3, 含4h+ → 强共振
if chain_len >= 3 and has_4h_plus:
    boost = 1.0 + chain_len * 0.08   # 每级 +8%
    weighted_ss *= boost  # (或 weighted_bs)
elif chain_len >= 2:
    boost = 1.0 + chain_len * 0.04   # 弱共振每级 +4%
```

### 3.5 大小周期反向衰减

```python
# 大周期偏空 + 小周期偏多 → 衰减买入信号
if large_ss > large_bs * 1.3 and small_bs > small_ss * 1.3:
    weighted_bs *= 0.5    # 买入信号打5折
    weighted_ss *= 1.15   # 卖出信号增强
```

### 3.6 决策矩阵

| 优先级 | 条件 | 结果 | strength |
|--------|------|------|----------|
| A | 大小同向 + 3+级共振链(含4h+) + net>15 | 强入场 | 70-100 |
| B | 大周期明确 + 小周期同向(无共振链) | 中等入场 | 50-70 |
| C | 大小周期反向 | 不做 | 0-15 |
| D | 多空同时存在 | 观望 | 0-15 |
| E | 仅小周期信号 + 大周期中性 | 弱信号(非actionable) | 10-40 |
| F | 完全中性 | 无信号 | 0 |

---

## 四、Regime 分类与动态阈值

### 4.1 五种市场状态 (optimize_six_book.py:267-386)

基于 3 个维度实时分类：

```python
# 输入 (均使用 idx-1 的已完成bar数据)
vol = ret.rolling(48).std()                     # 48bar 收益率标准差
trend = (MA12 - MA48) / price                   # 均线价差/价格
atr_pct = ATR(14) / price                       # 14bar真实波幅/价格

# 固定阈值分类
vol_high  = 0.020    # 高波动
vol_low   = 0.007    # 低波动
trend_strong = 0.015 # 强趋势
trend_weak   = 0.006 # 弱趋势
atr_high  = 0.018    # ATR高波动
```

| Regime | 条件 | 入场门槛调整 | 杠杆/仓位 |
|--------|------|-------------|----------|
| `high_vol_choppy` | vol≥0.020 且 trend≤0.006 | ST/LT × 1.22 | 杠杆 × 0.58 |
| `high_vol` | vol≥0.020 或 ATR≥0.018 | ST/LT × 1.12 | 杠杆 × 0.75 |
| `low_vol_trend` | vol≤0.007 且 trend≥0.015 | ST/LT × 0.95 | 正常 |
| `trend` | trend≥0.015 | ST/LT × 0.98 | 正常 |
| `neutral` | 以上均不满足 | 基线不变 | 正常 |

### 4.2 趋势方向倾斜 (optimize_six_book.py:382-386)

```python
if trend >= 0.006:     # 上升趋势
    long_threshold  *= 0.88  # 做多更容易
    short_threshold *= 1.10  # 做空更难
elif trend <= -0.006:  # 下降趋势
    short_threshold *= 0.88  # 做空更容易
    long_threshold  *= 1.10  # 做多更难
```

---

## 五、微结构叠加与双引擎 (v7.0 新增)

### 5.1 微结构信号 (optimize_six_book.py:425-567)

基于 K 线内可得字段构建的代理指标，用于增强/抑制入场信号：

```python
# 输入指标 (均使用 idx-1 bar, 无前视)
basis_z     # 基差 z-score (close vs SMA 偏离度)
oi_trend_z  # 持仓量趋势代理 z-score (volume-based)
funding_z   # 资金费率代理 z-score (basis × 0.35)
imbalance   # 主动买卖失衡 (taker_buy_ratio - 0.5)

# 微结构对 SS/BS 的影响
if micro_long_score > threshold:
    bs *= (1 + micro_score_boost)   # +8%
    ss *= (1 - micro_score_dampen)  # -10%
# 极端拥挤时直接阻止开仓
if basis_z > micro_basis_crowded_z:  # z > 2.2
    block_long = True  # 多头极度拥挤
```

### 5.2 双引擎切换 (optimize_six_book.py:611-616, 1881-1899)

根据 regime + 微结构状态自动选择执行引擎：

| 引擎模式 | 触发条件 | 入场门槛 | 退出门槛 | 持仓上限 | 杠杆 |
|---------|---------|---------|---------|---------|------|
| **trend** | trend/low_vol_trend regime | ×0.95 (更宽松) | ×1.05 (更耐心) | ×1.35 | ×1.10 |
| **reversion** | high_vol/high_vol_choppy | ×1.12 (更严格) | ×0.90 (更快退) | ×0.70 | ×0.75 |
| **single** | neutral 或未启用 | ×1.0 | ×1.0 | ×1.0 | ×1.0 |

### 5.3 Vol-Targeting 仓位缩放 (optimize_six_book.py:626-656)

```python
# 动态仓位缩放: 高波动自动降风险, 低波动允许恢复
target_vol_annual = 0.85                    # 目标年化波动率
realized_vol = ret.rolling(48).std() * sqrt(bars_per_year)
scale = clip(target_vol / realized_vol, 0.45, 1.35)
margin_use *= scale
```

---

## 六、门控过滤层（入场前多层过滤）

### 6.1 Regime 做空门控

**S1-a: neutral 空单门槛提高** (v7.0 B3):
```python
# live_config.py:427
regime_short_threshold = 'neutral:60'
# → 在 neutral regime 中，SS 必须 ≥ 60 才允许开空
# v7.0 B3: 从 45 提高到 60 (P4实证: neutral short 六书均无判别力)
```

**S1-b: low_vol_trend 做空门控**:
```python
# 在 low_vol_trend 中:
# 做空门槛 += 35 → 需要 SS ≥ 75 才能开空
if regime == 'low_vol_trend':
    short_threshold += 35
```

### 6.2 Neutral 结构质量渐进折扣 (核心创新, v7.0 B3 加强)

**核心思路**: 在 neutral regime 中，divergence 判别力近乎为零 (Cohen's d = -0.04)。检查 5 本结构书 (CS/KDJ/MA/BB/VP) 的独立确认数量，渐进折扣仓位：

```python
# 5 本结构书的活跃判定 (activity_threshold = 10)
structural_keys = ['ma_sell', 'cs_sell', 'bb_sell', 'vp_sell', 'kdj_sell']
confirms = sum(1 for k in structural_keys if book_features[k] > 10.0)

# v7.0 B3 折扣表 (比 v6.0 更激进):
discount_map = {
    0: 0.00,   # 仅div驱动 → 直接禁止 (v6: 0.10)
    1: 0.05,   # 微弱支撑 → 5%仓位 (v6: 0.20)
    2: 0.15,   # 尚可 → 15%仓位 (v6: 1.00)
    3: 0.50,   # 强共识 → 50%仓位 (v6: 1.00)
    4: 1.00,   # 极强 → 全额
    5: 1.00,   # 满书确认 → 全额
}
margin *= discount_map[confirms]
```

> **v6→v7 变化**: 0 本确认从 0.10 降到 **0.00（完全禁止）**，2 本确认从 1.00 降到 **0.15**，
> 新增 3 本确认梯度 0.50。这使得 neutral 空单需要更多结构书支持才能获得有意义的仓位。

**零蝴蝶效应**: 只调节仓位大小，不改变入场时序，所以不会连锁影响后续所有信号。

### 6.3 空单冲突软折扣 (v7.0 B3: 扩展到 neutral)

```python
# v7.0 B3: 生效范围从 trend,high_vol 扩展到 trend,high_vol,neutral
if regime in ('trend', 'high_vol', 'neutral'):
    if book_features['div_buy'] >= 50.0 and book_features['ma_sell'] >= 12.0:
        margin *= 0.60  # 缩仓 0.60×
```

### 6.4 趋势保护 (现货底仓)

```python
# EMA10 > EMA30 × 1.005 → 趋势激活 (滞后退出: EMA10 < EMA30 × 0.98)
if trend_up:
    # ETH 仓位 ≤ 50% 时完全禁止卖出
    # 超出部分最多卖 10%, 且卖出阈值提高到 SS≥55
    # 现货冷却期拉长到 48 bar
    # 做空门槛提高到 max(base, 55), dominance 要求 > 2.5x
```

---

## 七、执行引擎参数 (v7.0 B3)

### 7.1 信号阈值

| 参数 | 值 | 含义 |
|------|-----|------|
| sell_threshold | 18 | 现货卖出门槛 (SS≥18) |
| buy_threshold | 25 | 现货买入门槛 (BS≥25) |
| short_threshold | **40** | 合约做空门槛 (SS≥40) |
| long_threshold | **25** | 合约做多门槛 (BS≥25) |
| close_short_bs | 40 | 反向平空门槛 |
| close_long_ss | 40 | 反向平多门槛 |

### 7.2 止损止盈

| 参数 | 空头 | 多头 | 说明 |
|------|------|------|------|
| SL (止损) | **-20%** | **-10%** | 放宽呼吸空间 |
| TP (止盈) | +60% | +40% | |
| Trail (追踪) | **19%** | 12% | ⚠️ cliff effect: 0.01差异=$15K |
| trail_pullback | 50% | 50% | 最高点回撤比例触发追踪止盈 |
| PartialTP1 (v3早锁利) | **+12%** | +12% | 平仓 30% 仓位 |
| PartialTP2 (v3早锁利) | **+25%** | +25% | 平仓 30% 仓位 |
| hard_stop_loss | -28% | -28% | 硬断路器 |
| max_hold | 48 bar | 72 bar | 最大持仓K线数 |

### 7.3 仓位管理

```python
leverage = 5                    # 基础杠杆
max_leverage = 5                # 上限
margin_use = 0.70               # 可用保证金使用比例
single_pct = 0.20               # 单仓最大占比
total_pct = 0.50                # 总仓位最大占比

# 动态杠杆(按信号强度)
if SS >= 50:  actual_lev = min(5, max_lev)
elif SS >= 35: actual_lev = min(3, max_lev)
else:          actual_lev = 2

# Vol-Targeting: margin_use *= clip(0.85 / realized_vol, 0.45, 1.35)
# Regime: high_vol_choppy → risk_mult=0.58, high_vol → risk_mult=0.75
```

### 7.4 冷却机制

| 事件 | 冷却 |
|------|------|
| 正常开仓后 | **cooldown = 6 bar** (v7.0 B3: 4→6) |
| 现货操作后 | spot_cooldown = 12 bar |
| 止损后 | cooldown × 4 (空/多各自) |
| 止损后(跨方向) | cooldown × 3 |
| 强平后 | cooldown × 6 |
| 强平后(跨方向) | cooldown × 3 |
| 反手最少持仓 | 8 bar (多/空均是) |

### 7.5 费用模型

```python
TAKER_FEE    = 0.0005   # 0.05% (开/平仓均收)
SLIPPAGE     = 0.001    # 0.1%  (嵌入执行价)
FUNDING_RATE = 0.0001   # ±0.01% / 8h
LIQUIDATION_FEE = 0.005 # 0.5%
MAINTENANCE_RATE = 0.05 # 5% 维持保证金率
```

---

## 八、信号时序与执行流程

### 8.1 延迟执行 (消除 same-bar bias)

```
bar[i-1] close:  计算 score_provider(idx=i-1) → pending_ss, pending_bs
bar[i]   open:   使用 pending_ss/pending_bs 做决策
                 exec_price = open_prices[idx=i]  # 用当前 bar 的开盘价执行
bar[i]   close:  计算 score_provider(idx=i) → 存为下一根 bar 的 pending
```

### 8.2 每根 bar 的处理顺序

```
1. 读取 exec_price = open[idx]
2. 月度保证金额度重置
3. 强平检测 (用 bar 内 high/low 极值)
4. 资金费率计算 (每 bars_per_8h 根 bar)
5. 冷却倒计时
6. 风控检查 (日内亏损预算、全局回撤停机)
7. Regime检测 + 双引擎切换 + 微结构叠加
8. 使用 pending 信号做决策:
   a. 卖出现货 (带确认过滤)
   b. 反向平仓 (先平再开)
   c. 开空 (多层门控过滤)
   d. 开多 (门控过滤)
   e. 买入现货
9. 止损/止盈/追踪/分段止盈检查
10. 记录历史
11. 计算当前 bar 信号 → 存为 pending (供下根 bar 使用)
```

---

## 九、回测验证结果

### 9.1 核心性能指标 (run#499, 2025-01~2026-01)

| 指标 | 值 |
|------|------|
| 总收益 | +206.3% |
| 胜率 | 63.1% (135胜/79负) |
| 合约 Profit Factor | 1.53 |
| 组合 Profit Factor | 2.76 |
| 最大回撤 | -14.0% |
| Alpha (vs ETH持有) | +233.4% |
| 总费用 | $14,312 |

### 9.2 Walk-forward 验证 (90d训练/7d测试, 4窗口)

| 排名 | 方案 | 决策TFs | avg_alpha | win_rate | robust_score |
|------|------|---------|-----------|----------|-------------|
| **1** | **均衡搭配@1h** | **15m,1h,4h,12h** | **10.69%** | **100%** | **7.94** |
| 2 | 均衡搭配@2h | 15m,1h,4h,12h | 9.60% | 100% | 4.53 |
| 3 | 均衡搭配@4h | 15m,1h,4h,12h | 8.89% | 100% | 4.37 |

> Walk-forward alpha 衰减到 10.69%（vs 训练集 40.62%），**衰减率 ~74%**。
> 部署时 12h → 24h 以获得更强宏观锚定。
> 214 笔交易的胜率 63.1% 的 95% Wilson 区间: **56.4%~69.3%**（样本量有限）。

### 9.3 v7.0 B3 消融实验 (OOS 验证)

| 指标 | v6.0 baseline | v7.0 B3 | 变化 |
|------|--------------|---------|------|
| OOS 收益 | +36% | +41.6% | +5.6pp |
| OOS PF | 1.67 | 2.33 | +0.66 |
| OOS WR | 60.7% | 61.0% | +0.3pp |

B3 变更项: cooldown 4→6, neutral:45→60, 冲突折扣+neutral, 折扣梯度加强

---

## 十、已验证无效的实验功能 (默认关闭)

| 功能 | 代码开关 | A/B 结论 |
|------|---------|---------|
| 保本止损 (TP1后移SL到入场价) | use_breakeven_after_tp1=False | 压制利润 |
| 棘轮追踪止损 | use_ratchet_trail=False | 叠加后利润大幅下降 |
| 信号质量止损 (弱信号用紧SL) | use_ss_quality_sl=False | 收益导向下无效 |
| ATR自适应止损 | use_atr_sl=False | 对ETH无效 |
| Neutral 质量门控 (二元) | use_neutral_quality_gate=False | OOS退化 |
| 六书共识门控 (二元) | use_neutral_book_consensus=False | 被渐进折扣替代 |
| Neutral 空头结构确认器 | use_neutral_short_structure_gate=False | 过度拦截 |
| 空单逆势防守退出 | use_short_adverse_exit=False | 未充分验证 |
| 极端背离空否决 | use_extreme_divergence_short_veto=False | 未充分验证 |
| 信号置信度学习层 | use_confidence_learning=False | 先用于复盘 |
| Neutral 分层 SPOT_SELL | use_neutral_spot_sell_layer=False | run#496~499 主样本负贡献 |
| 停滞再入场 | use_stagnation_reentry=False | 触发少且主样本负贡献 |
| 多头冲突软折扣 | use_long_conflict_soft_discount=False | 仅实验 |
| Long 高置信门控 A/B | use_long_high_conf_gate_a/b=False | OOS正向但主区间回撤 |

---

## 十一、已知弱点与改进方向

### 11.1 过拟合敏感点 (P0 — cliff effect)

- **short_trail 0.19 vs 0.20 差 $15K**: MFE 分布在 19%~20% 密集堆积，0.01 的差异改变大量交易的退出路径。在 live 中会被滑点、延迟、funding 轻易击穿。
- **改进**: 追踪止盈连续化（始终追踪，pullback 随 max_pnl 连续变化），消除硬开关。

### 11.2 trend/high_vol 的 short 负期望 (P0)

- trend 空单 52.3% WR，high_vol 空单 46.2% WR，2~11 bars 快速止损。
- 说明**入场边际在这些 regime 上就是错的**，退出优化救不了。
- 冲突软折扣只是减小伤害，不能改变负期望。
- **改进**: 对 trend/high_vol 的 short 上"24h 方向门控 + 结构确认≥3 硬门槛"（仅限 short，不影响 long）。配合 ghost cooldown（被 gate 过滤的交易仍触发冷却期，减少蝴蝶效应）。

### 11.3 DIV 权重-有效性错配 (P1)

- 64% 交易在 neutral，DIV 判别力近乎为零 (d≈-0.04) 却占 55% 权重。
- 结构折扣是"仓位补丁"——仍在付手续费、占冷却期、引入路径依赖。
- **改进**: Regime-adaptive 融合权重。neutral 时 DIV 降到 0.30，CS/KDJ 升到 0.25。用连续插值（trend_strength → 权重）而非离散 regime 切换，避免边界抖动。

### 11.4 参数稳定性未验证 (P1)

- 需对 short_trail(0.16~0.24) × div_weight(0.20~0.55) × conflict_threshold(45~60) 做网格扫描。
- 观察收益是否形成"平台"（鲁棒）还是"尖峰"（过拟合）。

### 11.5 Regime 阈值固化 (P2)

- 使用固定阈值 (vol_high=0.020)，不适应 ETH 波动率长期漂移。
- **改进**: 改为 rolling percentile 或 vol-targeting 自适应。

### 11.6 训练集-OOS alpha 衰减 74%

- 训练集 40.62% → WF平均 10.69%
- **可能原因**: 参数过拟合、regime shift、策略容量限制
- **需要**: OOS 补 2024 数据 + 参数扰动蒙特卡洛 + 压力测试（2×费用/2×滑点/真实 funding）

---

## 十二、下一步改进路线图 (多模型共识)

### Phase 0: 诊断基线 (零风险)

- [ ] 按 side×regime 输出交易明细（WR/PF/avg_pnl/MFE/MAE/持仓bars/退出原因）
- [ ] short_trail 参数面扫描 0.14~0.26
- [ ] neutral confirm_count 分桶 expectancy（含手续费）
- [ ] regime 边界切换频率统计（判断是否需要滞回）

### Phase 1: 结构修正 (~36行代码)

| 步骤 | 改动 | 代码量 |
|------|------|--------|
| 1a | 追踪止盈连续化（pullback 随 max_pnl 递减） | ~10行 |
| 1b | Ghost cooldown（gate 过滤后仍触发冷却期） | ~3行 |
| 1c | trend/high_vol short 24h 方向门控 | ~5行 |
| 1d | trend/high_vol short confirms≥3 硬门槛 | ~8行 |
| 1e | Vol-targeting 替代硬编码 regime 乘数 | ~5行 |

### Phase 2: 信号层修正 (需参数面验证)

- [ ] Neutral DIV 权重连续插值（单自由度 `t`）
- [ ] 观察结构折扣是否可放宽
- [ ] 压力测试：2×手续费 + 2×滑点 + 真实 funding

---

## 附录 A: 完整参数清单 (v7.0 B3 部署值)

```json
{
  "symbol": "ETHUSDT",
  "timeframe": "1h",
  "sell_threshold": 18,
  "buy_threshold": 25,
  "short_threshold": 40,
  "long_threshold": 25,
  "close_short_bs": 40,
  "close_long_ss": 40,
  "sell_pct": 0.55,
  "short_sl": -0.20,
  "short_tp": 0.60,
  "long_sl": -0.10,
  "long_tp": 0.40,
  "short_trail": 0.19,
  "long_trail": 0.12,
  "trail_pullback": 0.50,
  "use_partial_tp": true,
  "partial_tp_1": 0.15,
  "partial_tp_1_pct": 0.30,
  "use_partial_tp_2": true,
  "partial_tp_2": 0.50,
  "partial_tp_2_pct": 0.30,
  "use_partial_tp_v3": true,
  "partial_tp_1_early": 0.12,
  "partial_tp_2_early": 0.25,
  "hard_stop_loss": -0.28,
  "leverage": 5,
  "max_lev": 5,
  "margin_use": 0.70,
  "single_pct": 0.20,
  "total_pct": 0.50,
  "cooldown": 6,
  "spot_cooldown": 12,
  "short_max_hold": 48,
  "long_max_hold": 72,
  "reverse_min_hold_short": 8,
  "reverse_min_hold_long": 8,
  "fusion_mode": "c6_veto_4",
  "veto_threshold": 25,
  "div_weight": 0.55,
  "kdj_weight": 0.15,
  "kdj_bonus": 0.09,
  "kdj_strong_mult": 1.25,
  "kdj_normal_mult": 1.12,
  "kdj_reverse_mult": 0.70,
  "kdj_gate_threshold": 10,
  "bb_bonus": 0.10,
  "vp_bonus": 0.08,
  "cs_bonus": 0.06,
  "veto_dampen": 0.30,
  "decision_timeframes": ["15m", "1h", "4h", "24h"],
  "consensus_min_strength": 40,
  "coverage_min": 0.5,
  "use_regime_aware": true,
  "regime_lookback_bars": 48,
  "regime_vol_high": 0.020,
  "regime_vol_low": 0.007,
  "regime_trend_strong": 0.015,
  "regime_trend_weak": 0.006,
  "regime_atr_high": 0.018,
  "regime_short_threshold": "neutral:60",
  "use_regime_short_gate": true,
  "regime_short_gate_add": 35,
  "regime_short_gate_regimes": "low_vol_trend",
  "use_neutral_structural_discount": true,
  "neutral_struct_activity_thr": 10.0,
  "neutral_struct_discount_0": 0.00,
  "neutral_struct_discount_1": 0.05,
  "neutral_struct_discount_2": 0.15,
  "neutral_struct_discount_3": 0.50,
  "neutral_struct_discount_4plus": 1.00,
  "structural_discount_short_regimes": "neutral",
  "structural_discount_long_regimes": "neutral",
  "use_short_conflict_soft_discount": true,
  "short_conflict_regimes": "trend,high_vol,neutral",
  "short_conflict_div_buy_min": 50.0,
  "short_conflict_ma_sell_min": 12.0,
  "short_conflict_discount_mult": 0.60,
  "use_dual_engine": true,
  "entry_dominance_ratio": 1.5,
  "trend_engine_entry_mult": 0.95,
  "trend_engine_exit_mult": 1.05,
  "trend_engine_hold_mult": 1.35,
  "trend_engine_risk_mult": 1.10,
  "reversion_engine_entry_mult": 1.12,
  "reversion_engine_exit_mult": 0.90,
  "reversion_engine_hold_mult": 0.70,
  "reversion_engine_risk_mult": 0.75,
  "use_microstructure": true,
  "micro_lookback_bars": 48,
  "micro_imbalance_threshold": 0.08,
  "micro_basis_crowded_z": 2.2,
  "micro_funding_proxy_mult": 0.35,
  "micro_score_boost": 0.08,
  "micro_score_dampen": 0.10,
  "use_vol_target": true,
  "vol_target_annual": 0.85,
  "vol_target_lookback_bars": 48,
  "vol_target_min_scale": 0.45,
  "vol_target_max_scale": 1.35,
  "use_trend_enhance": true,
  "trend_floor_ratio": 0.50,
  "use_spot_sell_confirm": true,
  "spot_sell_confirm_ss": 35,
  "spot_sell_confirm_min": 3,
  "spot_sell_regime_block": "high_vol,trend",
  "use_neutral_spot_sell_layer": false,
  "use_stagnation_reentry": false
}
```

## 附录 B: 关键源码文件索引

| 文件 | 行数 | 职责 |
|------|------|------|
| optimize_six_book.py | ~4200 | 回测引擎核心: `_run_strategy_core()` 主循环 |
| signal_core.py | 919 | 信号计算: `compute_signals_six()`, `calc_fusion_score_six_batch()` |
| multi_tf_consensus.py | 641 | 多周期共识: `fuse_tf_scores()`, `compute_weighted_consensus()` |
| strategy_futures.py | 1365 | 合约引擎: `FuturesEngine`, `FuturesPosition` |
| live_config.py | 770 | 配置: `StrategyConfig` (80+ 参数, 含 v1-v4 版本) |
| backtest_multi_tf_daily.py | ~1200 | 日级回测框架 |
| backtest_multi_tf_30d_7d.py | ~1350 | Walk-forward 验证框架 |

## 附录 C: 回测准确性审查结论

| 检查项 | 结论 |
|--------|------|
| 信号延迟执行 (无 same-bar bias) | ✅ signal at bar[i-1] close → execute at bar[i] open |
| Regime 无前视 | ✅ 传 idx-1, rolling 数据只到 idx-1 |
| 滑点无重复扣除 | ✅ slippage_cost 仅追踪不重复扣 |
| Profit Factor 公式 | ✅ losses 过滤确保 pnl ≤ 0 |
| 保证金会计 | ✅ margin 在 usdt 内冻结, 平仓只加 PnL-fee |
| 强平检测 | ✅ 用 bar 内 high/low 极值 |
| 双向持仓互斥 | ✅ 有测试保证 |
| 资金费率计算 | ✅ 基于当前价格 × qty (符合交易所规则) |
| 微量精度损失 | ⚠️ 强平 PnL 高估 ~$12.5/笔 (maintenance margin) |

## 附录 D: v6.0 → v7.0 B3 变更摘要

| 参数 | v6.0 | v7.0 B3 | 变更原因 |
|------|------|---------|---------|
| cooldown | 4 | **6** | 近似 ghost cooldown, OOS +5.6% |
| regime_short_threshold | neutral:45 | **neutral:60** | P4实证 neutral short 无 alpha |
| short_conflict_regimes | trend,high_vol | **trend,high_vol,neutral** | 扩展到 neutral |
| neutral_struct_discount_0 | 0.10 | **0.00** | 0本确认直接禁止 |
| neutral_struct_discount_1 | 0.20 | **0.05** | 更激进折扣 |
| neutral_struct_discount_2 | 1.00 | **0.15** | 新增梯度 |
| neutral_struct_discount_3 | (无) | **0.50** | 新增梯度 |
