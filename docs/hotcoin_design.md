# 热点币交易系统设计文档

## 1. 系统概述

基于币安全市场 Ticker 数据流，自动发现热点币种，复用六书融合策略 + 多周期共识进行信号计算，通过现货 API 执行交易的全自动系统。

**目标**: 在热点币的爆发初期介入，通过多维信号确认降低假突破风险，结合五层风控保护资金安全。

**当前状态**: Phase 1 完成 + R1-R9 (~8600 行代码, 48 个文件, 151 单测全绿)，paper 模式可运行。

## 2. 系统架构

```
                    ┌───────────────────────────────────────────────┐
                    │              HotCoinRunner (asyncio)          │
                    │                                               │
  ┌─────────┐      │  ┌─────────────┐   ┌──────────────┐          │
  │ Binance  │─WS──│→ │TickerStream │──→│AnomalyDetector│          │
  │ miniTick │      │  │ (全市场1s)   │   │  (量价异动)   │          │
  └─────────┘      │  └─────────────┘   └──────┬───────┘          │
                    │                           ↓                   │
  ┌─────────┐      │  ┌─────────────┐   ┌──────────────┐          │
  │ Binance  │─HTTP─│→│ListingMonitor│──→│              │          │
  │ 公告/API │      │  │ (新币上线)   │   │ CandidatePool│          │
  └─────────┘      │  └─────────────┘   │  (SQLite 池)  │          │
                    │  ┌─────────────┐   │              │          │
  ┌─────────┐      │  │TwitterMonitor│──→│              │          │
  │ Twitter  │─HTTP─│→│ BinanceSQ    │   └──────┬───────┘          │
  │ Binance  │      │  │ (社媒监控)   │          ↓                   │
  └─────────┘      │  └─────────────┘   ┌──────────────┐          │
                    │                    │  HotRanker    │          │
                    │                    │ (六维热度评分) │          │
                    │                    └──────┬───────┘          │
                    │                           ↓                   │
                    │  ┌─────────────┐   ┌──────────────┐          │
                    │  │  CoinFilter │←──│  Top N 候选   │          │
                    │  │ (过滤/去重)  │   └──────────────┘          │
                    │  └──────┬──────┘                              │
                    │         ↓                                     │
                    │  ┌─────────────────────────────────┐         │
                    │  │     SignalDispatcher             │         │
                    │  │  ThreadPoolExecutor (5 workers)  │         │
                    │  │  ┌───────────────────────────┐  │         │
                    │  │  │ signal_worker.py (每个币)  │  │         │
                    │  │  │  fetch_klines (多周期)     │  │         │
                    │  │  │  → _add_hot_indicators     │  │         │
                    │  │  │  → compute_signals_six     │  │         │
                    │  │  │  → calc_fusion_score_six   │  │         │
                    │  │  │  → fuse_tf_scores          │  │         │
                    │  │  └───────────────────────────┘  │         │
                    │  └──────────┬──────────────────────┘         │
                    │             ↓                                 │
                    │  ┌─────────────────────────────────┐         │
                    │  │       HotCoinSpotEngine          │         │
                    │  │  EntryExitRules (入场/出场)      │         │
                    │  │  PortfolioRisk  (五层风控)       │         │
                    │  │  CapitalAllocator (热度加权分配) │         │
                    │  │  OrderExecutor  (动态精度下单)   │         │
                    │  │  PnLTracker    (损益记录)       │         │
                    │  └─────────────────────────────────┘         │
                    └───────────────────────────────────────────────┘
```

## 3. 模块详解

### 3.1 Discovery 层 — 热点币发现

#### TickerStream (`discovery/ticker_stream.py`)
- **数据源**: 币安 `!miniTicker@arr` WebSocket (全市场 1s 推送)
- **功能**: 过滤 USDT 交易对、维护 20min 滑动窗口、计算 5min 价格变化
- **输出**: 实时 Ticker 数据 + 触发 AnomalyDetector

#### AnomalyDetector (`discovery/anomaly_detector.py`)
- **量异动**: 1min 成交量 / 20min 均量 > 5x
- **价异动**: 5min 涨幅 > 3%
- **冷却**: 同一币种 10min 内不重复告警
- **输出**: 异动币种 → CandidatePool

#### ListingMonitor (`discovery/listing_monitor.py`)
- **公告监控**: 轮询 Binance 公告 API，正则提取上线币种
- **新币检测**: 监控 exchangeInfo 发现新增交易对
- **上线前置信号**: 公告后开盘前设 listing_open_time
- **输出**: 新上线币种 → CandidatePool

#### 社媒监控 (`discovery/social_twitter.py`, `social_binance_sq.py`)
- **Twitter**: Bearer Token 搜索 crypto KOL 推文，提取币种提及频率
- **Binance Square**: 轮询社区内容，提取热门话题
- **输出**: mention_velocity + sentiment → CandidatePool

#### CandidatePool (`discovery/candidate_pool.py`)
- **存储**: 内存 dict + SQLite 持久化 (WAL 模式)
- **容量**: 最多 20 个候选币
- **生命周期**: 入池 (score ≥ 40) → 活跃 → 低分持续 10min → 出池 (score < 20)
- **冷却**: 止损后 30min 冷却期，不允许重入
- **线程安全**: threading.Lock 保护所有读写

#### HotRanker (`discovery/hot_ranker.py`)
六维热度评分 (0-100):

| 维度 | 权重 | Phase 1 | 描述 |
|------|------|---------|------|
| 公告强度 | 0.20 | 基础 | 上线信号 80-100 分 |
| 社媒扩散 | 0.15 | 基础 | mention_velocity * 10 |
| 情绪倾向 | 0.10 | 基础 | sentiment (-1~1) 映射 |
| 价格动量 | 0.25 | 完整 | 5min + 1h 涨幅 + 量比 |
| 资金流动性 | 0.20 | 完整 | 对数映射 24h 成交额 |
| 风险惩罚 | 0.10 | 完整 | 新币龄 + 涨幅过大惩罚 |

评分公式: `score = positive_normalized - risk * w_risk`

#### CoinFilter (`discovery/filters.py`)
- 黑名单/稳定币过滤
- 24h 成交额 ≥ $500K (新上线豁免)
- 24h 涨幅 ≤ 30% (FOMO 过滤)

### 3.2 Engine 层 — 信号计算

#### SignalDispatcher (`engine/signal_dispatcher.py`)
- **并发**: ThreadPoolExecutor (5 workers)
- **超时**: 60s/coin
- **异步桥接**: asyncio.wrap_executor → ThreadPoolExecutor

#### SignalWorker (`engine/signal_worker.py`)
**复用 ETH 六书策略核心**:
```
fetch_binance_klines (1m/5m/15m/1h)
  → _add_hot_indicators (MACD/KDJ/RSI/CCI/MA)
  → compute_signals_six (六书融合信号)
  → calc_fusion_score_six (卖分/买分)
  → fuse_tf_scores (多周期共识)
```

**热点币专用参数** (`engine/hot_coin_params.py`):
- MACD: 5/10/5 (快参) + 标准 12/26/9
- KDJ: 5 (比 ETH 的 9 更短)
- 时间框架权重: 1m(5) + 5m(10) + 15m(15) + 1h(20)
- 共振链要求: ≥ 2 级 (比 ETH 的 3 级更宽松)

**降级策略**: 可用周期不足一半时，信号 confidence 打 5 折。

#### EntryExitRules (`engine/entry_exit_rules.py`)
**入场条件**:
1. 信号 strength ≥ 20
2. 15m 方向确认 (bs > ss)
3. 热度 ≥ 30

**出场规则** (优先级从高到低):
1. 止损: -5% (配置可调)
2. 黑天鹅: 15min 跌幅 > 20% → 紧急全平
3. 分层止盈: T1 @ +5% 卖 30%, T2 @ +10% 卖 30% (价格跳档时取最高)
4. 追踪止损: 所有止盈层完成后，回撤 3% 全平
5. 时间止损: 持仓超 4h 全平
6. 热度衰退: 币种出池 → 全平

### 3.3 Execution 层 — 交易执行

#### OrderExecutor (`execution/order_executor.py`)
- **ExchangeInfoCache**: 1h 刷新精度信息 (线程安全双检查锁)
- **动态精度**: stepSize/tickSize → 自动格式化 qty/price
- **防重复**: 60s 去重窗口 (线程安全锁保护，check+占位原子操作)
- **预检查**: min_notional、min_qty、MARKET_LOT_SIZE
- **Paper 模式**: 记录模拟订单，不调 API
- **Live 模式**: HMAC 签名 → Spot REST API

#### PortfolioRisk (`execution/portfolio_risk.py`)
五层风控:

| 层级 | 规则 | 动作 |
|------|------|------|
| L1 | 单笔止损 (ATR 自适应 -3%~-5%) | entry_exit_rules 处理 |
| L2 | 单币亏损 > 5% 总资金 | 强制平仓 |
| L3 | 同板块敞口 > 20% | 拒绝开仓 |
| L4 | 日亏损 > 5% 总资金 | 停止新仓 |
| L5 | 总回撤 > 15% | 全清仓 + 冷却 24h |

- 峰值权益: 含未实现盈亏，避免回撤计算失真
- 日重置: UTC 0:00

#### CapitalAllocator (`execution/capital_allocator.py`)
- **公式**: `alloc = base * heat_mult * liq_mult`
- `heat_mult = 0.6 + 0.8 * (score/100)` → 0.6~1.4x
- `liq_mult = clamp(liquidity/60, 0.3, 1.0)`
- **约束**: 单币 ≤ 10% 资金, 总敞口 ≤ 40%
- **批量**: 按权重比例分配，低于 $12 不分配

#### SpotEngine (`execution/spot_engine.py`)
- 整合所有 execution 组件
- BUY 信号 → 入场检查 → 资金分配 → 下单 → 风控记录
- SELL 信号 → 现货仅支持平 BUY 仓位
- 持仓巡检: 每 10s 检查所有仓位的止盈/止损/时间/热度

#### PnLTracker (`execution/pnl_tracker.py`)
- 内存存最近 500 笔，超出自动裁剪
- 每笔 JSONL 追加写入 `data/hotcoin_trades_YYYYMMDD.jsonl`
- 统计: 总交易数、胜率、平均持仓时间、最佳/最差单笔

### 3.4 ML 层 (Phase 2 预留)

#### features_hot.py
- 复用 ETH 73 维基础特征
- 新增 ~30 维: 社交特征、横截面特征、微观结构特征
- 列名兼容: 自动适配 `MACD_BAR`/`macd_hist`、`K`/`k` 等大小写差异

#### online_learning.py / train_hotcoin.py
- Online Learning 框架 (流式更新)
- 离线训练 (Phase 2: LightGBM/LSTM 热点币专用模型)

### 3.5 Web 层

#### routes.py
- `/hotcoin/status`: 系统运行状态
- `/hotcoin/config`: 配置查看/修改
- `/hotcoin/api/status`: JSON API (前端轮询)
- 状态来源: 优先读文件快照 (3min 有效) → 回退到 runner 查询

### 3.6 Runner 主循环

```python
# 10s 周期
1. pool.remove_expired()           # 清理出池币种
2. ranker.update_scores(pool)      # 六维评分
3. pool.get_top(N, min_score=40)   # 取 Top N
4. coin_filter.apply(candidates)   # 过滤
5. dispatcher.compute_signals()    # 多周期信号 (并发)
6. spot_engine.process_signals()   # 入场/出场
7. spot_engine.check_positions()   # 持仓巡检
```

**子任务崩溃恢复**:
- 关键任务 (ticker_stream, main_loop) 崩溃 → 系统关闭
- 非关键任务 (listing_monitor, social) 崩溃 → 仅告警

## 4. 配置体系

```python
HotCoinConfig
├── DiscoveryConfig     # 发现层 (量价阈值、池容量、热度权重)
├── TradingConfig       # 交易层 (时间框架、止盈止损、黑天鹅阈值)
├── ExecutionConfig     # 执行层 (资金、风控参数、下单超时)
├── db_path             # SQLite 路径
└── log_level           # 日志级别
```

**配置优先级**: DB 保存 → 环境变量覆盖 (HOTCOIN_PAPER, HOTCOIN_CAPITAL, ...)

## 5. 数据流

```
hotcoin/data/
├── hotcoins.db                        # SQLite (coin_pool + heat_history)
├── hotcoin_runtime_status.json        # 运行状态快照 (原子写入)
└── hotcoin_trades_YYYYMMDD.jsonl      # 交易记录 (JSONL, 崩溃安全)
```

## 6. 已知问题与技术债

### 已修复 (Round 1-5，共 44 项)

<details>
<summary>展开查看全部已修复项</summary>

| 编号 | 文件 | 问题 | 修复轮次 |
|------|------|------|----------|
| P0-1 | order_executor | 防重复下单 + TOCTOU 竞态 | R1 (check+占位原子) |
| P0-2 | portfolio_risk | peak_equity 不含未实现盈亏 | R1 |
| P0-2b | portfolio_risk | deprecated utcnow() | R1 |
| P0-3 | features_hot | MACD/KDJ/RSI 列名大小写不匹配 | R1 (双向兼容) |
| P0-4 | futures_adapter | 导入路径 + 杠杆缓存 | R1 |
| P0-5 | runner | 子任务崩溃无恢复 | R2 (done callback) |
| P1-1 | hot_ranker | 风险惩罚归一化 | R1 |
| P1-2 | entry_exit_rules | 止盈跳档丢失 | R1 (取最高已达标) |
| P1-3 | signal_worker | 无降级策略 | R1 (confidence 打折) |
| R2-1 | capital_allocator | remaining_slots=0 除零 | R2 |
| R2-2 | order_executor | _recent_orders 无线程锁 | R2 (dedup_lock) |
| R2-3 | spot_engine | price=0 时静默开仓 | R2 (early return) |
| R2-4 | listing_monitor | STOP_WORDS + 正则修复 | R2 |
| R2-5 | candidate_pool | _heat_hist_ts 内存泄漏 | R2 (出池时清理) |
| R2-6 | pnl_tracker | persist 无异常处理 | R2 |
| R3-1 | candidate_pool | get_top/get_all/get 返回直接引用 | R3 (浅拷贝) |
| R3-2 | candidate_pool | update_coins_batch 覆盖实时行情 | R3 (仅更新评分字段) |
| R3-3 | candidate_pool | update_status 新方法 | R3 |
| R3-4 | ticker_stream | list.pop(0) O(N) 性能 | R3 (deque) |
| R3-5 | ticker_stream | _vol_window/_price_snapshots 无 TTL | R3 (_cleanup_stale 10min) |
| R3-6 | anomaly_detector | _alert_cooldown 无清理 | R3 (定期清理) |
| R3-7 | social_twitter | _mention_counts 无清理 | R3 (cleanup_stale + SentimentScorer) |
| R3-8 | social_binance_sq | _seen_content_ids 整体清零 | R3 (cleanup_stale) |
| R3-9 | listing_monitor | _seen_open_symbols 无清理 | R3 (定期清理) |
| R3-10 | hot_ranker | 动量评分算子优先级 | R3 |
| R3-11 | runner | _write_status_snapshot 缺 anomalies | R3 |
| R4-1 | spot_engine._partial_close | 未记录 PnL | R4 |
| R4-2 | entry_exit_rules | 黑天鹅检测用不可获取字段 | R4 (改用 price_change_5m) |
| R4-3 | spot_engine | update_coin 覆盖实时数据 | R4 (改用 update_status) |
| R4-4 | portfolio_risk.can_open | 用 entry_price 算敞口 | R4 (改用 current_prices) |
| R4-5 | order_executor | spot_limit_buy 缺去重 | R4 |
| R4-6 | signal_dispatcher | 无超时保护 | R4 (wait_for + SIGNAL_TIMEOUT_SEC) |
| R4-7 | spot_engine + order_executor | paper 模式多余 REST 调用 | R4 (hint_price) |
| R4-8 | signal_worker | 全局 config 多线程不安全 | R4 (_add_hot_indicators 参数直传) |
| R4-9 | capital_allocator | 不考虑已有敞口 | R4 (used_exposure 参数) |
| R4-10 | portfolio_risk | 日重置用 24h 窗口 | R4 (UTC 0:00 边界) |
| R4-11 | portfolio_risk | qty<=0 浮点误差 | R4 (改 < 1e-8) |
| R5-1 | candidate_pool | record_heat_history _heat_hist_ts 竞态 | R5 (移入 lock) |
| R5-2 | ticker_stream | 24h 成交额日重置导致虚假量能突增 | R5 (忽略负增量) |
| R5-3 | ticker_stream | price_change_5m 除零风险 | R5 (oldest_price>0) |
| R5-4 | online_learning | 单一标签清空 buffer | R5 (保留 + 3x 硬上限) |
| R5-5 | filters | 新币上线零流动性豁免 | R5 (10% 最低门槛) |
| R5-6 | signal_dispatcher | 异常日志缺堆栈 | R5 (exc_info) |
| R5-7 | runner | 主循环异常缺上下文 | R5 (pool_size + positions) |
| A-1 | hot_ranker + ExchangeInfoCache | coin_age_days 始终为 -1 | Phase A (_first_seen 估算) |
| A-2 | listing_monitor | 无 rate limit 退避 | Phase A (429/418 指数退避) |
| A-3 | runner + web | 缺 /hotcoin/health | Codex + Phase A (去重) |
| A-4 | runner | 缺状态机 | Codex (_compute_engine_state) |
| R7-1 | ticker_stream | tickers 属性无线程安全 | R7 (快照副本 + tickers_ref 内部引用) |
| R7-2 | anomaly_detector | _alert_cooldown 清理可能异常 | R7 (try/except + pop) |
| R7-3 | spot_engine | 缺紧急全平功能 | R7 (emergency_close_all) |
| R7-4 | order_executor | 缺资产查询 | R7 (query_account_balances) |
| R7-5 | web/routes | _runner TOCTOU 竞态 | R7 (_get_runner local binding) |
| R7-6 | web/routes | 缺紧急平仓/余额 API | R7 (/api/emergency_close + /api/balances) |
| R8-1 | spot_engine | close/partial_close PnL 用 hint_price 而非 exec_price | R8 (优先用 exec_price) |
| R8-2 | candidate_pool | remove_expired DB 写入异常可致内存/DB 不一致 | R8 (try/except + list 快照) |
| R8-3 | ticker_stream | _cleanup_stale 异常导致内存泄漏 | R8 (try/except 保护) |
| R8-4 | order_executor | 预检统计 counter 无限增长 | R8 (每小时自动重置) |
| R8-5 | signal_worker | K 线缓存超限时逐条淘汰 O(n) | R8 (批量淘汰 50 条) |
| R9-1 | dashboard.html | 缺紧急平仓 UI 和余额面板 | R9 (双确认按钮 + 余额表格) |
| R9-2 | runner | 优雅关闭无超时保护 | R9 (asyncio.wait_for 15s) |
| R9-3 | runner | _write_status_snapshot risk_summary 异常可中断 | R9 (try/except) |

</details>

### 待修复 (Backlog)

#### P1 — 可靠性

| 编号 | 文件 | 问题 |
|------|------|------|
| B-7 | signal_dispatcher | shutdown(wait=True) 可能死锁 |
| B-8 | web/routes | _runner 全局变量无线程锁 |
| B-9 | order_executor | exchangeInfo 刷新失败后 1h 用旧缓存 |
| B-11 | spot_engine | 无订单状态跟踪 (open orders 可能遗漏) |

#### P2 — 优化

| 编号 | 文件 | 问题 |
|------|------|------|
| C-2 | candidate_pool | 批量更新无事务回滚 |
| C-3 | spot_engine | 风控检查与下单间无原子保护 |
| C-5 | pnl_tracker | 内存 trades 列表非线程安全 |
| C-6 | entry_exit_rules | partial_exits 跨仓位不重置 |
| C-7 | signal_worker | K线每轮重新拉取，缺 bar-level 缓存 |
| C-8 | runner | 缺少事件日志契约 (candidate/signal/order 快照) |

## 7. 下一步规划

见 `docs/hotcoin_roadmap.md`
