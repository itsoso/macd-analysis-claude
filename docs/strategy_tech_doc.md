# ETH/USDT 六书融合量化交易策略 v9.0 — 完整技术规格书

**交易标的**: ETH/USDT 永续合约 (Binance Futures)
**主时间框架**: 1h K线
**决策时间框架**: 15m, 1h, 4h, 24h (多周期联合决策)
**回测区间 (IS)**: 2025-01 ~ 2026-01 (12个月)
**OOS验证区间**: 2024-01 ~ 2024-12 (12个月)
**Walk-Forward验证**: 6窗口滚动 (2024Q1 ~ 2025Q4)
**初始资金**: $100,000 USDT
**策略版本**: v9.0 — 架构简化版 (B1b + P24 + Anti-Squeeze + 衍生品数据集成)
**生产配置版本**: v5 (`STRATEGY_VERSION=v5`)

> **v9.0 更新摘要** (2026-02-15):
> 1. **B1b**: 彻底禁止 neutral short（neutral:999），不再用复杂折扣/门控修补 DIV 信号失效问题
> 2. **P24**: 空头止损按 regime 差异化（trend -15%、high_vol -12%，替代全局 -20%）
> 3. **Anti-Squeeze Filter**: 显式组合条件（funding 高 + OI 上升 + taker 买强）阻止逆拥挤方向开仓
> 4. **实盘衍生品数据集成**: live_signal_generator 集成 Mark Price / Funding Rate / OI 数据获取
> 5. **架构简化**: 移除因 B1b 冗余的 neutral short 门控，决策路径从"大量门控修补"简化为"直接禁止+差异化止损"

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

> **v9.0 架构决策依据**: neutral long 是核心利润来源 (WR=85%, PF=17.67)。neutral short 贡献微弱 (PF=1.07) 且 DIV Cohen's d=-0.64（反向指标）。**四大 LLM 共识**: 与其用复杂门控修补 neutral short，不如直接禁止 (B1b)。

---

<h2 id="section-2">二、系统架构</h2>

### 2.1 五层架构 (v9.0)

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
│  → B1b neutral short 禁止 (neutral:999)                  │
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
# Veto 机制: 反向书 >=2 本超过 veto_threshold(25) → 分数 * 0.30
```

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

> **v9.0 决策**: 不再尝试修补 neutral short，而是通过 B1b 直接禁止。

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

### v9.0 Regime 对策略行为的影响

| Regime | 空单决策 (v9.0) | 止损 (P24) | 门槛乘数 | 杠杆乘数 |
|--------|----------------|-----------|---------|---------|
| **neutral** | **禁止开空 (B1b)** | N/A | 1.00 | 1.00 |
| trend | 允许, Anti-Squeeze过滤 | **-15%** | 0.98 | 1.00 |
| low_vol_trend | 允许, +35门槛 | -18% | 0.95 | 1.00 |
| high_vol | 允许, Anti-Squeeze过滤 | **-12%** | 1.12 | 0.75 |
| high_vol_choppy | 允许, Anti-Squeeze过滤 | -12% | 1.22 | 0.58 |

---

<h2 id="section-6">六、v9.0 核心变更: 开仓逻辑</h2>

### 6.1 信号延迟执行 (T+1)

```python
# 在 bar[i] 收盘时计算信号 → 存入 pending_ss/pending_bs
# 在 bar[i+1] 的 open 价格执行交易 → 消除 same-bar bias
exec_price = open_prices[idx]
ss, bs = pending_ss, pending_bs
```

### 6.2 v9.0 开空条件 (做空)

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
    # v9.0 B1b: regime_short_threshold = "neutral:999"
    # → neutral 中 SS 永远达不到 999, 等效于完全禁止
)
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

### 6.4 仓位计算

```python
margin = available_margin * cur_margin_use  # 基础仓位

# 结构折扣 (neutral long 仍有效)
if struct_confirms == 0: margin *= 0.0    # 0本→禁止
elif struct_confirms == 1: margin *= 0.05 # 1本→5%
elif struct_confirms == 2: margin *= 0.15 # 2本→15%
elif struct_confirms == 3: margin *= 0.50 # 3本→50%

# v9.0: 冲突折扣仅保留 trend/high_vol (neutral 已无空头)
if regime in ('trend', 'high_vol'):
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

### 7.3 P24 Regime-Adaptive Stop-Loss (v9.0 新增)

```python
# 按 regime 差异化止损 (替代全局 -20%):
regime_short_sl_map = {
    'neutral':        -0.12,  # 被 B1b 禁止, 理论值
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

---

<h2 id="section-8">八、衍生品数据集成 (v9.0 新增)</h2>

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

<h2 id="section-9">九、v9.0 完整参数集 (v5 配置)</h2>

### 入场参数

| 参数 | 值 | 说明 |
|------|------|------|
| short_threshold | 40 | 空单基础阈值 |
| long_threshold | 25 | 多单门槛 |
| **regime_short_threshold** | **neutral:999** | **v9.0 B1b: 禁止 neutral short** |
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

### 冲突折扣 (v9.0 简化)

| 参数 | v8.0 | v9.0 | 说明 |
|------|------|------|------|
| short_conflict_regimes | trend,high_vol,neutral | **trend,high_vol** | neutral 已被 B1b 禁止 |
| short_conflict_discount_mult | 0.60 | 0.60 | 不变 |

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
| **B1b 禁止neutral空** | **+40.78%** | **2.09** | v9.0 采用 |
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
  问题: 简单, 可解释, 可维护
```

### 11.2 被 B1b 替代（运行时不可达）的机制

| 机制 | 原状态 | v9.0状态 | 原因 |
|------|-------|---------|------|
| neutral_struct_discount (short side) | 活跃 | 不可达 | B1b 禁止 neutral short |
| short_conflict_regimes 含 'neutral' | 活跃 | 已移除 | neutral 无空头 |
| use_neutral_book_consensus | 已关 | 冗余 | |
| use_extreme_divergence_short_veto | 已关 | 冗余 | |
| use_neutral_short_structure_gate | 已关 | 冗余 | |
| P9 use_regime_adaptive_reweight | 已关 | 被 P18 取代 | |

### 11.3 四大 LLM 审计共识 (Claude/GPT Pro/Gemini/Grok)

| 建议 | v9.0 执行状态 |
|------|-------------|
| Regime-Adaptive 权重 | 代码完整实现, A/B 测试显示 E4 OOS +28.9%, 待进一步验证后启用 |
| 禁止 neutral short (B1b) | **已部署** (neutral:999) |
| Perp 专属数据 (OI/Funding) | **数据获取已完成**, Anti-Squeeze Filter 已实现 |
| 空头不对称防御 | **P24 已部署** (regime SL), P20 已部署 (追踪收紧) |
| 用 R (风险单位) 定义仓位 | P21 已实现但 A/B 显示 1.5% 太保守, 需调参 |
| 加权结构确认 (替代计数) | P23 已实现, E4 组合效果显著, 独立效果差 |
| 架构简化 ("多做减法") | **v5 配置已简化**: 移除冗余门控 |

---

<h2 id="section-12">十二、当前已知问题</h2>

### 12.1 高优先级

| # | 问题 | 影响 | 建议 |
|---|------|------|------|
| 1 | **P18 IS/OOS 分裂** | regime_adaptive 融合 OOS 优秀 (+28.9%) 但 IS 回退, 暂不敢在生产启用 | 需 Walk-Forward 验证稳定性后再启用 |
| 2 | **P21 risk_per_trade 参数过保守** | 1.5% R% 使仓位过小, 独立 A/B 收益下降 | 调参至 2.5-3.5%, 重跑 A/B |
| 3 | **P23 独立效果差** | 加权确认独立使用 OOS -8.1%, 仅与 P18 组合才有效 | 与 P18 绑定使用, 不独立启用 |
| 4 | **trend/high_vol short 胜率低** | trend WR=46.7% PF=0.84, high_vol WR=33.3% PF=0.47 | Anti-Squeeze 应可缓解, 需回测验证 |

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

| 优先级 | 编号 | 方向 | 预期收益 | 状态 |
|--------|------|------|---------|------|
| P0 | P25 | P18+P23 组合 Walk-Forward 验证 | 确认 OOS 泛化是否稳定 | 待执行 |
| P0 | P26 | P21 R% 调参 (2.5-3.5%) | 消灭"少数大亏单主导" | 待执行 |
| P1 | P27 | Funding 现金流纳入回测 | 消除收益虚高 (~2-5%) | 待执行 |
| P1 | P28 | Anti-Squeeze 回测验证 | 验证 trend/high_vol short 改善 | 待执行 |
| P2 | P29 | CS-KDJ 去相关 | 消除虚假共识, 提升胜率稳定性 | 待设计 |
| P2 | P30 | Leg 风险预算 (regime×direction) | 把资金从负alpha段迁移到正alpha段 | 待设计 |
| P3 | P31 | 动态 Regime 阈值 (滚动百分位) | 适应市场周期变化 | 待设计 |
| P3 | P32 | MAE-driven 数据驱动止损 | 替代固定百分比, 基于历史MAE分布 | 待设计 |

### 13.2 长期架构方向 (LLM 共识)

1. **策略拆腿 + 组合风险预算**: 按 (regime × direction) 拆成独立 legs, 每个 leg 有独立风控 KPI 和风险预算。当前 B1b 是这个方向的雏形（把 neutral_short 的 budget 设为 0）。

2. **从 alpha 策略到 hedge 策略**: short legs 的目标从"追求 alpha"转为"降低组合尾部风险"。仅在"崩盘风险"上升时开空（OI 快速上升 + funding 极端 + liquidation flow），仓位小、止损宽、时间短。

3. **更真实的回测口径**: fee ×2 / slippage ×2 压力测试; 成交价模型从 T+1 open 升级为包含滑点的分布模型; Mark Price 用于强平和未实现 PnL。

---

### 版本变更日志

| 版本 | 日期 | 核心变更 |
|------|------|---------|
| **v9.0** | **2026-02-15** | **B1b禁止neutral short + P24 regime SL + Anti-Squeeze Filter + 实盘衍生品数据 + 架构简化** |
| v8.1 (P20) | 2026-02-15 | P17口径统一(24h)+P20空头追踪收紧(60%→40%) |
| v8.0 (B3+P13) | 2026-02-15 | P13连续追踪止盈 + P6-P16全量诊断, OOS PF 2.33→2.51 |
| v7.0 (B3) | 2026-02 | P4→B3: neutral short 四重防护, OOS PF 1.67→2.33 |
| v6.0 | 2026-02 | P1优化: short_trail 0.20→0.19, WR 61.9%→63.1% |
| v5.1 | 2026-01 | P0前视修复 + 参数重优化 |
| v5.0 | 2026-01 | 结构折扣 + 冲突折扣初版 |
| v4.0 | 2025-12 | 六书融合 + 多周期共识 + Regime 检测 |
