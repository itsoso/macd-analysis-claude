# ETH/USDT 六书融合量化交易策略 v6.0 — 完整技术文档

**交易标的**: ETH/USDT 永续合约 (Binance)
**主时间框架**: 1h K线
**决策时间框架**: 15m, 1h, 4h, 24h (多周期联合决策)
**回测区间**: 2025-01 ~ 2026-01 (12个月)
**初始资金**: $100,000 USDT

<h2 id="section-1">一、策略概述与核心指标</h2>

### 核心性能指标 (run#488)

| 指标 | 值 |
|------|------|
| 总收益 | +206.3% |
| 胜率 | 63.1% (135胜/79负) |
| 合约 Profit Factor | 1.53 |
| 组合 Profit Factor | 2.76 |
| 最大回撤 | -14.0% |
| Alpha (vs ETH持有) | +233.4% |
| 总费用 | $14,312 |

### Regime 分布表现

| Regime | 笔数 | 净PnL | PF |
|--------|------|-------|-----|
| neutral | 277 | +$127,258 | 2.08 |
| trend | 89 | +$74,430 | 6.35 |
| low_vol_trend | 40 | +$35,330 | 5.50 |
| high_vol | 21 | +$17,343 | 4.73 |
| high_vol_choppy | 7 | +$1,405 | 2.16 |

所有 5 个 regime 均为正 PnL，策略在各市场状态下均有效。

---

<h2 id="section-2">二、架构分层</h2>

策略采用 **信号层 → 融合层 → 决策层 → 执行层 → 风控层** 五层架构：

```
┌─────────────────────────────────────────────┐
│  Layer 1: 信号生成 (signal_core.py)          │
│  六维独立评分: DIV / MA / CS / BB / VP / KDJ  │
│  每本书独立输出 sell_score 和 buy_score (0-100) │
└──────────────┬──────────────────────────────┘
               ▼
┌─────────────────────────────────────────────┐
│  Layer 2: 多周期融合 (multi_tf_consensus.py)  │
│  4个TF的6书分数 → 加权共识 SS/BS              │
│  链式一致性检测 + 大小周期冲突检查              │
└──────────────┬──────────────────────────────┘
               ▼
┌─────────────────────────────────────────────┐
│  Layer 3: 决策引擎 (optimize_six_book.py)    │
│  Regime检测 → 双引擎切换 → 微结构叠加          │
│  → 入场条件判断 → 仓位计算 → 结构折扣/冲突折扣  │
└──────────────┬──────────────────────────────┘
               ▼
┌─────────────────────────────────────────────┐
│  Layer 4: 执行引擎 (FuturesEngine)           │
│  延迟执行(T+1 open) → 滑点/手续费建模         │
│  分段止盈(TP1@12%,TP2@25%) → 追踪止盈        │
└──────────────┬──────────────────────────────┘
               ▼
┌─────────────────────────────────────────────┐
│  Layer 5: 风控保护                            │
│  硬断路器(-28%) → 日亏/周亏/连亏熔断           │
│  全局回撤停机(15%) → 现货趋势底仓保护          │
└─────────────────────────────────────────────┘
```

---

<h2 id="section-3">三、信号生成 — 六书融合</h2>

**六本书（Six Books）** 是六个独立的技术指标维度，每本书在每根K线上输出一个 `sell_score` 和 `buy_score`（0-100分）:

| 书名 | 缩写 | 核心指标 | 融合权重 |
|------|------|---------|---------|
| Divergence | DIV | MACD柱/DIF与价格背离 | 55% (div_weight=0.55) |
| Moving Average | MA | 均线排列/交叉/斜率 | 隐含在veto中 |
| Candlestick | CS | K线形态识别(锤子/吞没等) | 6% (cs_bonus=0.06) |
| Bollinger Band | BB | 布林带突破/挤压/回归 | 10% (bb_bonus=0.10) |
| Volume-Price | VP | 量价关系/量能异常 | 8% (vp_bonus=0.08) |
| KDJ | KDJ | 超买超卖+趋势确认 | 15% (kdj_weight=0.15) |

### 融合模式: `c6_veto_4`

```python
# 融合公式 (简化):
raw_ss = div_sell * 0.55 + kdj_sell * 0.15 + bb_bonus + vp_bonus + cs_bonus
# Veto 机制: 任一本书的反向分数 > veto_threshold(25) → 融合分数打折
if any_book_buy > 25:
    raw_ss *= veto_dampen(0.30)
# KDJ 强化: KDJ 方向一致时乘以 1.25, 反向时乘以 0.70
```

**关键发现**: Divergence 在 neutral regime 中 Cohen's d ≈ -0.04（几乎无判别力），而 CS(d=0.40) 和 KDJ(d=0.42) 才是真正有效的确认书。这是结构折扣机制的理论基础。

---

<h2 id="section-4">四、多周期共识</h2>

### 4.1 权重分配

```python
_MTF_WEIGHT = {
    '15m': 5, '30m': 8, '1h': 15, '2h': 12,
    '4h': 20, '8h': 18, '12h': 15, '24h': 25,
}
```

24h 权重最高(25)，作为趋势锚定；1h 权重15作为主执行框架。

### 4.2 共识融合流程

1. 收集各TF的 (ss, bs) 分数
2. 加权平均得到 weighted_ss, weighted_bs
3. 链式一致性检测: 相邻TF方向一致 → boost +8%/TF
4. 覆盖率门控: 有效TF数 / 总TF数 >= 0.5 才出信号
5. 方向决策: dominance_ratio=1.3, 即 SS > BS*1.3 才判定为"卖"
6. 输出: (consensus_ss, consensus_bs, meta_dict)

### 4.3 六书特征加权聚合

每个TF的6本书细分特征（`div_sell`, `ma_buy`, `cs_sell` 等12个维度）也按TF权重加权汇总，输出 `book_features_weighted`，供后续结构折扣、冲突折扣使用。

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
    if abs(trend) <= 0.006:  # 高波动 + 弱趋势
        regime = 'high_vol_choppy'  # 提门槛+22%, 降杠杆*0.58
    else:
        regime = 'high_vol'         # 提门槛+12%, 降杠杆*0.75
elif vol <= 0.007 and abs(trend) >= 0.015:
    regime = 'low_vol_trend'        # 略降门槛*0.95
elif abs(trend) >= 0.015:
    regime = 'trend'                # 略降门槛*0.98
else:
    regime = 'neutral'              # 默认状态
```

Regime 还根据趋势方向做**顺逆势倾斜**: 上升趋势中做多门槛 *0.88（更容易开多），做空门槛 *1.10（更难开空）。

### Regime 对阈值和风险的影响

| Regime | 入场门槛乘数 | 杠杆乘数 | 退出门槛乘数 |
|--------|------------|---------|------------|
| neutral | 1.00 | 1.00 | 1.00 |
| trend | 0.98 | 1.00 | 1.00 |
| low_vol_trend | 0.95 | 1.00 | 1.05 |
| high_vol | 1.12 | 0.75 | 0.95 |
| high_vol_choppy | 1.22 | 0.58 | 0.90 |

---

<h2 id="section-6">六、开仓逻辑</h2>

### 6.1 信号延迟执行

**核心设计**: 消除 same-bar execution bias。

```python
# 在 bar[i] 的收盘时计算信号 → 存入 pending_ss/pending_bs
# 在 bar[i+1] 的 open 价格执行交易
exec_price = open_prices[idx]  # 当前bar的开盘价
ss, bs = pending_ss, pending_bs  # 上一根bar计算的信号
```

### 6.2 开空条件 (做空)

所有条件必须**同时满足**:

```python
_short_candidate = (
    short_cd == 0                        # 冷却期结束
    and ss >= effective_short_threshold   # SS >= 动态阈值
    and not eng.futures_short             # 无存量空仓
    and not eng.futures_long              # 无存量多仓（不同时持有）
    and sell_dom                          # SS > BS * entry_dom_ratio(1.5)
    and not in_conflict                   # 非冲突区间
    and can_open_risk                     # 风控允许
    and not micro_block_short             # 微结构不阻止
    and neutral_short_ok                  # Neutral质量门控通过
    and _bk_short_ok                      # 六书共识门控通过
    and extreme_div_short_ok             # 极端背离否决未触发
    and neutral_struct_short_ok          # 结构确认器通过
)
```

### 6.3 `effective_short_threshold` 的动态计算

```python
base = 40  # short_threshold (v4配置)
# 1. Regime动态调整 → 高波动时 *1.12~1.22
# 2. 双引擎调整 → trend引擎*0.95, reversion引擎*1.12
# 3. Regime门控 → low_vol_trend 加 +35
# 4. Regime特定阈值 → neutral 中 SS须>=45
# 5. 趋势保护 → 上升趋势中提高到 max(base, 55)
```

### 6.4 开多条件 (做多)

与开空逻辑对称，但有额外的**趋势做多增强**:
- 上升趋势中门槛降至 min(cur_long_threshold, 25)
- dominance 降低为 bs > ss 即可
- 如果 gate 后 bs=0 但趋势明确，查看 4h/24h 原始信号绕过 gate

### 6.5 仓位计算

```python
margin = available_margin * cur_margin_use  # 基础仓位

# 结构折扣 (仅neutral regime)
if struct_confirms == 0: margin *= 0.10   # 仅div驱动→极小仓
elif struct_confirms == 1: margin *= 0.20 # 微弱支撑→小仓
# 2+本确认 → 全额

# 冲突软折扣 (仅trend/high_vol regime)
if div_buy >= 50 and ma_sell >= 12:
    margin *= 0.60  # 打六折

# 杠杆动态调整
actual_lev = 5 if ss >= 50 else 3 if ss >= 35 else 2
```

---

<h2 id="section-7">七、持仓管理与退出</h2>

### 7.1 退出优先级 (空仓为例，从高到低)

| 优先级 | 退出类型 | 条件 |
|--------|---------|------|
| 1 | 强平检测 | 使用 bar HIGH 检测 intrabar 穿越 |
| 2 | 硬断路器 | PnL < -28% → 强制止损 |
| 3 | 分段止盈TP1 | PnL >= +12% → 平仓30% |
| 4 | 分段止盈TP2 | PnL >= +25% → 再平仓30% |
| 5 | 完全止盈 | PnL >= +60% → 全部平仓 |
| 6 | 追踪止盈 | 峰值 >= 19% 且回撤到峰值*50% |
| 7 | 反向信号平仓 | BS >= 40 且 SS < BS*0.7 且持仓>=8bars |
| 8 | 常规止损 | PnL < -20% → 按止损价成交 |
| 9 | 超时平仓 | 持仓 >= 48 bars |

### 7.2 分段止盈 (v3 早期锁利)

```python
# 一段止盈: +12% → 平仓30%
if pnl_r >= 0.12 and not partial_done:
    close_qty = position * 0.30
    partial_done = True

# 二段止盈: +25% → 再平仓30%  (elif 避免同bar双触发)
elif pnl_r >= 0.25 and partial_done and not partial2_done:
    close_qty = position * 0.30
    partial2_done = True

# 剩余40%仓位由追踪止盈/完全止盈管理
```

### 7.3 追踪止盈

```python
# 空仓: 当峰值利润 >= 19% (short_trail=0.19) 时启动追踪
if short_max_pnl >= 0.19:
    # 当前利润回撤到峰值的 50% 以下时平仓
    if pnl_r < short_max_pnl * 0.50:
        close_position()
```

**关键洞察**: `short_trail` 从 0.20 降到 0.19 是 WR 突破 63% 的核心手段。存在一批交易的峰值利润在 19%-20% 之间，0.20 的门槛让它们以"超时"或"止损"方式亏损退出，而 0.19 将它们转化为追踪止盈赢单。

### 7.4 冷却机制

| 事件 | 冷却时长 |
|------|---------|
| 正常开仓 | 4 bars (4h) |
| 止盈后 | 8 bars |
| 反向平仓后 | 12 bars |
| 止损后 | 16 bars |
| 连续止损 | 32 bars (16*2) |
| 硬止损后 | 20 bars |
| 强平后 | 24 bars + 跨方向12 bars |

---

<h2 id="section-8">八、结构折扣机制 (Structural Discount)</h2>

### 8.1 原理

在 neutral regime 中，divergence（占融合权重 55%）几乎没有判别力（Cohen's d = -0.04），导致 SS/BS 分数主要由背离驱动，但背离信号在震荡市中频繁假突破。

**解决方案**: 不阻止交易（避免蝴蝶效应），而是根据 **5本结构书的独立确认数量** 渐进调节仓位大小。

### 8.2 实现

```python
# 5本结构书: MA, CS, BB, VP, KDJ (排除无判别力的 DIV)
structural_sell_keys = ['ma_sell', 'cs_sell', 'bb_sell', 'vp_sell', 'kdj_sell']

# 统计 book_features_weighted 中 > 10.0 (活跃阈值) 的书的数量
confirms = sum(1 for k in keys if book_feat[k] > 10.0)

# 折扣表:
# 0本确认: margin *= 0.10  (仅divergence驱动, 33% WR)
# 1本确认: margin *= 0.20  (微弱支撑, 50% WR)
# 2本确认: margin *= 1.00  (尚可, 62.5% WR)
# 3+本确认: margin *= 1.00 (强共识)
```

**效果**: neutral 开仓空单从 v5.1 的 PnL 亏损修复为 WR=61.8%、PnL=+$21K。

---

<h2 id="section-9">九、冲突软折扣 (Conflict Soft Discount)</h2>

### 9.1 原理

在 trend/high_vol regime 中，有些空单信号"卖方极强但买方背离也很强"，这种高冲突信号的止损率极高。

### 9.2 实现

```python
# 条件: regime in ('trend', 'high_vol')
#   且 book_div_buy >= 50  (买方背离很强)
#   且 book_ma_sell >= 12  (卖方均线有活跃信号)
# 动作: margin *= 0.60  (仓位打六折)
```

---

<h2 id="section-10">十、v6.0 完整参数集</h2>

### 入场阈值

| 参数 | 值 | 说明 |
|------|------|------|
| sell_threshold | 18 | 现货卖出阈值 |
| buy_threshold | 25 | 现货买入阈值 |
| short_threshold | 40 | 空单基础阈值 |
| long_threshold | **25** | [v6: 30→25] 多单门槛降低 |
| close_short_bs | 40 | 反向平空BS阈值 |
| close_long_ss | 40 | 反向平多SS阈值 |

### 止损止盈

| 参数 | 值 | 说明 |
|------|------|------|
| short_sl | -0.20 | 空单止损 -20% |
| short_tp | 0.60 | 空单止盈 +60% |
| long_sl | -0.10 | 多单止损 -10% |
| long_tp | 0.40 | 多单止盈 +40% |
| hard_stop_loss | -0.28 | 硬断路器 -28% |
| short_trail | **0.19** | [v6: 0.20→0.19] 核心WR突破 |
| long_trail | 0.12 | 多头追踪启动 |
| trail_pullback | 0.50 | 追踪回撤容忍度 |

### 分段止盈

| 参数 | 值 | 说明 |
|------|------|------|
| partial_tp_1_early | 0.12 | TP1: +12% 平30% |
| partial_tp_2_early | 0.25 | TP2: +25% 再平30% |
| partial_tp_1_pct | 0.30 | TP1平仓比例 |
| partial_tp_2_pct | 0.30 | TP2平仓比例 |

### 结构折扣

| 参数 | 值 | 说明 |
|------|------|------|
| neutral_struct_discount_0 | **0.10** | [v6: 0.15→0.10] 0本确认 |
| neutral_struct_discount_1 | **0.20** | [v6: 0.25→0.20] 1本确认 |
| structural_discount_regimes | neutral | 仅neutral生效 |

### 冲突软折扣

| 参数 | 值 | 说明 |
|------|------|------|
| use_short_conflict_soft_discount | **True** | [v6新启用] |
| short_conflict_regimes | trend,high_vol | 生效regime |
| short_conflict_div_buy_min | 50.0 | 买方背离阈值 |
| short_conflict_ma_sell_min | 12.0 | 卖方均线阈值 |
| short_conflict_discount_mult | 0.60 | 打六折 |

### 仓位与风控

| 参数 | 值 | 说明 |
|------|------|------|
| leverage | 5 | 最大杠杆 |
| margin_use | 0.70 | 可用保证金使用比例 |
| short_max_hold | 48 | 最大持仓bars |
| cooldown | 4 | 基础冷却期 |
| reverse_min_hold_short | 8 | 反向平仓最小持仓 |

### 已验证无效的功能 (保持关闭)

| 功能 | 测试结果 |
|------|---------|
| ATR自适应止损 | 对ETH无效 |
| NoTP提前退出 | WR 62%→55%, 有害 |
| 逆势防守退出 | 效果为负 |
| 保本止损 | 压制利润 |
| 棘轮追踪 | 过度截断利润 |
| 二元门控 | 有蝴蝶效应 |
| 在线置信度学习 | 样本不足 |

---

<h2 id="section-11">十一、已知问题与优化方向</h2>

### 11.1 残存瓶颈

1. **trend/high_vol 空单胜率低**: trend 空单 52.3% WR，high_vol 空单 46.2% WR。止损在 2-11 bars 内触发，速度极快，任何退出逻辑都来不及干预。冲突软折扣仅减小仓位但无法避免亏损。

2. **Divergence 权重占比过高**: 融合权重 55% 但在 neutral 中几乎无判别力（Cohen's d ≈ -0.04）。CS/KDJ 判别力高（d ≈ 0.40）但权重不足。

3. **参数敏感度高**: `short_trail` 差 0.01（0.18 vs 0.19）= ~$15K 收益差异。意味着参数可能过拟合。

### 11.2 已尝试但失败的优化

| 优化方案 | 结果 | 失败原因 |
|---------|------|---------|
| 扩展结构折扣到 trend/high_vol | 收益 -19% | 折扣了本该盈利的 trend 交易 |
| 放宽冲突阈值 (div_buy_min 50→35) | 收益下降 | 折扣了太多盈利信号 |
| 屏蔽 high_vol 空单 | 收益下降 | 蝴蝶效应：后续交易时序变化 |
| NoTP 提前退出 (8bars) | WR 62%→55% | 过早平掉后来大赚的仓位 |
| 保本止损 (TP1后SL=break-even) | 收益下降 | 太多仓位被保本止损后反弹 |
| 棘轮追踪 (利润越高回撤越紧) | 收益下降 | 过度截断利润 |

### 11.3 建议优化方向

1. **Regime-Adaptive 融合权重**: 在 neutral 中降低 DIV 权重（55%→30%），提升 CS+KDJ 权重。在 trend 中保持当前权重。
2. **OOS/WFO 验证**: 当前参数仅在 2025 区间验证，需要 2024 OOS + Walk-Forward 确认非过拟合。
3. **Regime-Specific 止损**: neutral 空单 SL 可能需要更紧，trend 空单可能需要更宽。
4. **信号层改进**: 添加 order flow 指标、高频微结构数据等新的信号源。
5. **自适应追踪**: `short_trail` 根据 ATR 或 regime 自适应调节，降低参数敏感度。

---

### 代码组织

| 文件 | 职责 |
|------|------|
| `optimize_six_book.py` | 核心交易循环、regime检测、持仓管理 (~4200行) |
| `live_config.py` | 策略参数、版本管理、风控配置 (~740行) |
| `signal_core.py` | 六书信号计算、融合评分 |
| `multi_tf_consensus.py` | 多周期共识、TF权重、链式检测 |
| `backtest_multi_tf_daily.py` | 回测编排、数据加载、结果持久化 |
| `live_runner.py` | 实盘运行入口、阶段管理 |
