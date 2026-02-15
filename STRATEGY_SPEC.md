# MACD Analysis 六书融合策略 — 完整技术规格

> 版本: v4 (2026-02-15, run#499 baseline)
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
│  │背离(div)│ │均线(MA) │ │蜡烛图(CS)│ │布林带(BB)│ │量价(VP)│ │KDJ ││
│  │ 权重70% │ │ 权重30% │ │ bonus%  │ │ bonus%  │ │bonus%│ │bonus││
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └──┬───┘ └──┬──┘│
│       └──────┬────┘           └──────┬─────┘         └────┬───┘   │
│              │                       │                    │        │
│         base_score            veto/bonus 层           veto/bonus   │
│     (div*0.7+MA*0.3)    2+本反向≥25 → dampen×0.30    KDJ timing  │
│                         同向≥15 → +bonus              adjustment  │
└────────────────────────────┬────────────────────────────────────────┘
                             │ sell_score(SS), buy_score(BS)
┌────────────────────────────▼────────────────────────────────────────┐
│                          门控过滤层                                  │
│                                                                     │
│  ┌──────────────┐ ┌───────────────┐ ┌────────────────────────────┐ │
│  │ Regime 动态   │ │ Neutral 结构  │ │ 趋势/冲突 做空抑制          │ │
│  │ 阈值调整      │ │ 质量折扣      │ │ (regime_short_gate,        │ │
│  │ 5种市场状态   │ │ 0确认→0.1x   │ │  short_conflict_discount)  │ │
│  │              │ │ 1确认→0.2x   │ │                            │ │
│  │              │ │ 2+确认→1.0x  │ │                            │ │
│  └──────────────┘ └───────────────┘ └────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                      执行引擎 (FuturesEngine)                       │
│                                                                     │
│  信号在 bar[i-1] close 计算 → bar[i] open 执行 (延迟一根)           │
│  费用: Taker 0.05% + Slippage 0.1% + Funding ±0.01%/8h            │
│  保证金: 逐仓隔离, 最大5x杠杆, 可用=usdt-frozen                    │
│  止损止盈: SL/TP/Trail/PartialTP1(+15%)/PartialTP2(+50%)          │
│  风控: 反手最少8bar, 冷却4bar, 强平6x冷却                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、六维信号计算

每个时间框架独立计算 6 维信号，产出 `sell_score` 和 `buy_score`（0~100 范围）。

### 2.1 背离信号 (Divergence) — 权重 70%

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

### 2.2 均线信号 (MA) — 权重 30%

基于多条移动平均线的排列、交叉、斜率综合评分。
- 多头排列(MA5>MA10>MA20>MA48): +10
- 金叉(短MA上穿长MA): +8~12
- 斜率加速: 额外加分
- 排列加成: 空头排列时 sell_score ×1.10，多头排列时 buy_score ×1.10

### 2.3 蜡烛图 (Candlestick) — bonus/veto

检测经典K线形态：吊人线、倒锤线、吞没、十字星等。
- 同向确认(cs_sell ≥ 25): +cs_bonus(6%)
- 反向 veto: 计入否决票

### 2.4 布林带 (Bollinger) — bonus/veto

基于价格与布林带上下轨的相对位置。
- 触及上轨 + 缩口: sell_score 升高
- 触及下轨 + 扩口: buy_score 升高
- 同向确认(bb_sell ≥ 15): +bb_bonus(10%)

### 2.5 量价 (Volume-Price) — bonus/veto

基于成交量异常检测和量价背离。
- 放量下跌(量价确认): sell_score 升高
- 缩量上涨(量价背离): sell 信号增强
- 同向确认(vp_sell ≥ 15): +vp_bonus(8%)

### 2.6 KDJ — bonus/veto (c6_veto_4 模式)

基于 KDJ 指标的超买超卖和交叉。
- KDJ 高位死叉: kdj_sell 升高
- KDJ 低位金叉: kdj_buy 升高
- 同向确认(kdj_sell ≥ 15): +kdj_bonus(9%)

### 2.7 融合模式: c6_veto_4

当前部署的融合模式 (signal_core.py:388-420):

```python
# 基础分数 = 70%背离 + 30%均线
base_sell = (div_sell * 0.7 + ma_sell * 0.3) * ma_arr_bonus_sell
base_buy  = (div_buy  * 0.7 + ma_buy  * 0.3) * ma_arr_bonus_buy

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

## 五、门控过滤层（入场前多层过滤）

### 5.1 Regime 做空门控 (S1: 最有效的单一改进)

```python
# live_config.py:398
regime_short_threshold = 'neutral:45'
# → 在 neutral regime 中，SS 必须 ≥ 45 才允许开空 (默认 short_threshold=40)
```

**数据支撑**: neutral 中做空贡献 -$47K/88笔，抬高门槛到 45 过滤掉最弱的 ~30% 空单。

### 5.2 Neutral 结构质量渐进折扣 (核心创新)

**核心思路**: 在 neutral regime 中，divergence 判别力近乎为零 (Cohen's d = -0.04)。仅靠 divergence 驱动的信号大概率是假信号。检查 5 本结构书 (CS/KDJ/MA/BB/VP) 的独立确认数量，折扣仓位：

```python
# 5 本结构书的活跃判定 (activity_threshold = 10)
structural_keys = ['ma_sell', 'cs_sell', 'bb_sell', 'vp_sell', 'kdj_sell']
confirms = sum(1 for k in structural_keys if book_features[k] > 10.0)

# 按确认数折扣保证金 (不阻止交易,仅缩小仓位)
discount_map = {
    0: 0.10,   # 仅div驱动, WR=33% → 极小仓位
    1: 0.20,   # 微弱支撑, WR=50% → 小仓位
    2: 1.00,   # 尚可, WR=62.5% → 全额
    3: 1.00,   # 强共识 → 全额
    4: 1.00,   # 极强 → 全额
    5: 1.00,   # 满书确认 → 全额
}
margin *= discount_map[confirms]
```

**零蝴蝶效应**: 只调节仓位大小，不改变入场时序，所以不会连锁影响后续所有信号。

### 5.3 空单冲突软折扣

```python
# 在 trend/high_vol 中:
# 如果买方 divergence ≥ 50 且卖方 MA ≥ 12 → 缩仓 0.60×
if regime in ('trend', 'high_vol'):
    if book_features['div_buy'] >= 50.0 and book_features['ma_sell'] >= 12.0:
        margin *= 0.60
```

### 5.4 Regime 做空门控 (low_vol_trend)

```python
# 在 low_vol_trend 中:
# 做空门槛 += 35 → 需要 SS ≥ 75 才能开空
if regime == 'low_vol_trend':
    short_threshold += 35
```

### 5.5 趋势保护 (现货底仓)

```python
# EMA10 > EMA30 × 1.005 → 趋势激活 (滞后退出: EMA10 < EMA30 × 0.98)
if trend_up:
    # ETH 仓位 ≤ 50% 时完全禁止卖出
    # 超出部分最多卖 10%, 且卖出阈值提高到 SS≥55
    # 现货冷却期拉长到 48 bar
```

---

## 六、执行引擎参数 (v4)

### 6.1 信号阈值

| 参数 | 值 | 含义 |
|------|-----|------|
| sell_threshold | 18 | 现货卖出门槛 (SS≥18) |
| buy_threshold | 25 | 现货买入门槛 (BS≥25) |
| short_threshold | **40** | 合约做空门槛 (SS≥40, v4提高) |
| long_threshold | **25** | 合约做多门槛 (BS≥25, v4降低) |
| close_short_bs | 40 | 反向平空门槛 |
| close_long_ss | 40 | 反向平多门槛 |

### 6.2 止损止盈

| 参数 | 空头 | 多头 | 说明 |
|------|------|------|------|
| SL (止损) | **-20%** | **-10%** | v4放宽呼吸空间 |
| TP (止盈) | +60% | +40% | |
| Trail (追踪) | 19% | 12% | 回撤触发比例 |
| trail_pullback | 50% | 50% | 最高点回撤比例触发追踪止盈 |
| PartialTP1 | +12% (v3早锁利) | +12% | 平仓 30% 仓位 |
| PartialTP2 | +25% (v3早锁利) | +25% | 平仓 30% 仓位 |
| hard_stop_loss | -28% | -28% | 硬断路器 |
| max_hold | 48 bar | 72 bar | 最大持仓K线数 |

### 6.3 仓位管理

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
```

### 6.4 冷却机制

| 事件 | 冷却 |
|------|------|
| 正常开仓后 | cooldown = 4 bar |
| 现货操作后 | spot_cooldown = 12 bar |
| 止损后 | cooldown × 4 (空/多各自) |
| 止损后(跨方向) | cooldown × 3 |
| 强平后 | cooldown × 6 |
| 强平后(跨方向) | cooldown × 3 |
| 反手最少持仓 | 8 bar (多/空均是) |

### 6.5 费用模型

```python
TAKER_FEE    = 0.0005   # 0.05% (开/平仓均收)
SLIPPAGE     = 0.001    # 0.1%  (嵌入执行价)
FUNDING_RATE = 0.0001   # ±0.01% / 8h
LIQUIDATION_FEE = 0.005 # 0.5%
MAINTENANCE_RATE = 0.05 # 5% 维持保证金率
```

---

## 七、信号时序与执行流程

### 7.1 延迟执行 (消除 same-bar bias)

```
bar[i-1] close:  计算 score_provider(idx=i-1) → pending_ss, pending_bs
bar[i]   open:   使用 pending_ss/pending_bs 做决策
                 exec_price = open_prices[idx=i]  # 用当前 bar 的开盘价执行
bar[i]   close:  计算 score_provider(idx=i) → 存为下一根 bar 的 pending
```

### 7.2 每根 bar 的处理顺序

```
1. 读取 exec_price = open[idx]
2. 月度保证金额度重置
3. 强平检测 (用 bar 内 high/low 极值)
4. 资金费率计算 (每 bars_per_8h 根 bar)
5. 冷却倒计时
6. 风控检查 (日内亏损预算、全局回撤停机)
7. 使用 pending 信号做决策:
   a. 卖出现货 (带确认过滤)
   b. 反向平仓 (先平再开)
   c. 开空 (多层门控过滤)
   d. 开多 (门控过滤)
   e. 买入现货
8. 止损/止盈/追踪/分段止盈检查
9. 记录历史
10. 计算当前 bar 信号 → 存为 pending (供下根 bar 使用)
```

---

## 八、回测验证结果

### 8.1 单周期优化器结果 (30天, 1473 变体)

| 排名 | TF | 配置 | Alpha | MDD | 交易数 |
|------|-----|------|-------|-----|--------|
| 1 | 1h | 组合B_TOP1+最佳分段 | 149.89% | -4.05% | 138 |
| 2 | 1h | 模式_c6_veto | 107.46% | -4.33% | 136 |
| 3 | 2h | 择时_强1.2 | 82.40% | -5.98% | 97 |

> 注意: 单周期 Top1 的 alpha=149.89% 存在严重过拟合（1473 变体中挑选最优 + 仅 30 天窗口）

### 8.2 Walk-forward 验证 (90d训练/7d测试, 4窗口)

| 排名 | 方案 | 决策TFs | avg_alpha | win_rate | robust_score |
|------|------|---------|-----------|----------|-------------|
| **1** | **均衡搭配@1h** | **15m,1h,4h,12h** | **10.69%** | **100%** | **7.94** |
| 2 | 均衡搭配@2h | 15m,1h,4h,12h | 9.60% | 100% | 4.53 |
| 3 | 均衡搭配@4h | 15m,1h,4h,12h | 8.89% | 100% | 4.37 |
| 4 | 全周期@1h | 15m~24h (8TF) | 10.26% | 100% | 3.76 |

> walk-forward alpha 衰减到 10.69%（vs 训练集 40.62%），这是更诚实的估计。
> 部署时 12h → 24h 以获得更强宏观锚定和 kline_store 优化。

### 8.3 费用模型验证

| 组件 | 占总费用比例 | 数值 (典型) |
|------|------------|-----------|
| Taker Fee | ~40% | $6,000-16,000 |
| Slippage | ~35% | 嵌入执行价 |
| Funding Rate | ~25% | 每 8h 计算 |

### 8.4 最新 2x2 消融（2025-01~2026-01）

| 组合 | run_id | 收益 | Alpha | MaxDD | 组合PF |
|------|--------|------|-------|-------|--------|
| layer=ON, reentry=ON | 496 | 200.57% | 227.68% | -13.89% | 2.71 |
| layer=OFF, reentry=ON | 497 | 200.38% | 227.48% | -13.89% | 2.71 |
| layer=ON, reentry=OFF | 498 | 199.64% | 226.75% | -13.88% | 2.70 |
| **layer=OFF, reentry=OFF** | **499** | **206.33%** | **233.44%** | **-13.98%** | **2.76** |

> 结论：`use_neutral_spot_sell_layer=False`、`use_stagnation_reentry=False` 为当前主样本最优基线。

---

## 九、已验证无效的实验功能 (默认关闭)

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

---

## 十、已知弱点与改进方向

### 10.1 Regime 阈值固化
- 当前使用固定阈值 (vol_high=0.020)，ETH 近期波动率落在灰色地带，策略长期停留在 neutral
- **改进**: 改为 rolling percentile（如 75th/25th）自适应

### 10.2 多空不对称保护
- 做空有 4 层独立门控（regime_short_gate, short_conflict_discount, structural_discount, neutral:45）
- 做多仅有 1 层（long_high_conf_gate，且默认关闭）
- **改进**: 镜像做多保护

### 10.3 融合权重固化
- div 70% + MA 30% 在所有 regime 下不变
- 实际上趋势市 MA 更可靠，震荡市 KDJ/BB 更有价值
- **改进**: 按 regime 动态调整融合权重

### 10.4 训练集-OOS alpha 衰减 83%
- 训练集 40.62% → WF平均 10.69% → 最新7d 6.89%
- **可能原因**: 参数过拟合、regime shift、策略容量限制

### 10.5 配置管理
- StrategyConfig (505行)、_build_default_config (293行)、_build_strategy_snapshot (295行) 三处近乎相同的参数列表需手动同步

---

## 附录 A: 完整参数清单 (v4 部署值)

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
  "cooldown": 4,
  "spot_cooldown": 12,
  "short_max_hold": 48,
  "long_max_hold": 72,
  "reverse_min_hold_short": 8,
  "reverse_min_hold_long": 8,
  "fusion_mode": "c6_veto_4",
  "veto_threshold": 25,
  "div_weight": 0.55,
  "kdj_bonus": 0.09,
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
  "regime_short_threshold": "neutral:45",
  "use_regime_short_gate": true,
  "regime_short_gate_add": 35,
  "regime_short_gate_regimes": "low_vol_trend",
  "use_neutral_structural_discount": true,
  "neutral_struct_activity_thr": 10.0,
  "neutral_struct_discount_0": 0.10,
  "neutral_struct_discount_1": 0.20,
  "neutral_struct_discount_2": 1.00,
  "use_short_conflict_soft_discount": true,
  "short_conflict_regimes": "trend,high_vol",
  "short_conflict_div_buy_min": 50.0,
  "short_conflict_discount_mult": 0.60,
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
| optimize_six_book.py | ~2800 | 回测引擎核心: `_run_strategy_core()` 主循环 |
| signal_core.py | 919 | 信号计算: `compute_signals_six()`, `calc_fusion_score_six_batch()` |
| multi_tf_consensus.py | 641 | 多周期共识: `fuse_tf_scores()`, `compute_weighted_consensus()` |
| strategy_futures.py | 1365 | 合约引擎: `FuturesEngine`, `FuturesPosition` |
| live_config.py | 709 | 配置: `StrategyConfig` (60+ 参数) |
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
