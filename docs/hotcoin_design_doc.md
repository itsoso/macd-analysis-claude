# HotCoin 热点币系统设计文档（2026-02-21）

- 文档目标：统一描述 HotCoin 发现-信号-执行-监控全链路，明确当前实现状态与下一步改造计划。
- 代码范围：`hotcoin/`、`hotcoin/web/`、`templates/page_live_hotcoin_config.html`
- 代码基线：`f489b38` + R3-R5 工作区改进（未提交）

---

## 1. 目标与边界

### 1.1 系统目标

1. 在币安 USDT 市场中实时发现"热点币"。
2. 通过多周期 K 线与热度指标生成可执行交易信号。
3. 在严格风控前提下执行现货交易（默认 paper，可切 live）。
4. 提供可观测、可回放、可配置的运行与运维能力。

### 1.2 非目标（当前阶段）

1. 不做跨交易所撮合。
2. 不做链上成交级别高频策略。
3. 不做全自动资金调度到衍生品（`futures_adapter` 仍为预留能力）。

---

## 2. 当前架构

### 2.1 分层结构

1. Discovery（发现层）
- `hotcoin/discovery/ticker_stream.py`：全市场 `!miniTicker@arr` 行情流。
- `hotcoin/discovery/anomaly_detector.py`：量价异动识别。
- `hotcoin/discovery/listing_monitor.py`：上币公告监控。
- `hotcoin/discovery/social_twitter.py` / `social_binance_sq.py`：社媒信号。
- `hotcoin/discovery/candidate_pool.py`：候选池生命周期管理。
- `hotcoin/discovery/hot_ranker.py`：六维热度评分。

2. Trading（信号层）
- `hotcoin/engine/signal_dispatcher.py`：多币并发信号调度。
- `hotcoin/engine/signal_worker.py`：单币多周期计算。
- `hotcoin/engine/entry_exit_rules.py`：入场/出场规则。

3. Execution（执行与风控层）
- `hotcoin/execution/spot_engine.py`：交易编排。
- `hotcoin/execution/order_executor.py`：下单、精度、预检、去重。
- `hotcoin/execution/portfolio_risk.py`：仓位/回撤/日损风险约束。
- `hotcoin/execution/capital_allocator.py`：资金分配。
- `hotcoin/execution/pnl_tracker.py`：收益与成交统计。

4. Web/Ops（配置与观测层）
- `hotcoin/web/routes.py`：`/hotcoin/api/status`、`/hotcoin/api/precheck_stats`。
- `templates/page_live_hotcoin_config.html`：热点币独立配置页 + 预检监控卡片。
- `app.py`：`/api/live/hotcoin_config` 配置读写与校验。

### 2.2 主流程

1. `HotCoinRunner` 周期执行（`signal_loop_sec`）：
- 更新候选池 → 热度评分 → 过滤 TopN → 生成信号。
2. 执行开平仓：
- `SpotEngine.process_signals()` → 风控检查 → `OrderExecutor` 下单。
3. 状态落盘：
- 写 `hotcoin/data/hotcoin_runtime_status.json`（含候选池、信号、仓位、预检统计）。
4. Web 展示：
- 仪表盘和配置页通过 API 获取运行态。

---

## 3. 已落地改造

### 3.1 下单层稳定性与可观测性（Codex R1-R2）

1. 结构化预检失败
- 返回 `code=PRECHECK_FAILED` + `precheck_code`。
- 典型码：`BUY_MIN_NOTIONAL`、`SELL_MIN_QTY`、`SELL_MIN_NOTIONAL`。

2. 市价单精度与约束增强
- 支持 `MARKET_LOT_SIZE`（`market_qty_step` / `market_min_qty`）。
- `format_quantity(..., market=True)` 优先用市价步进。

3. 预检价格源增强
- 卖出预检优先 `/api/v3/avgPrice`，回退 `ticker/price`。

4. 去重逻辑修复（关键）
- 去重改为"原子预留 + 失败回滚"，避免并发重复下单，同时避免失败后 60 秒误拦截重试。

5. 预检统计内存保护
- 增加 `MAX_PRECHECK_SYMBOLS`（默认 500）与 LRU 裁剪，防止长期运行字典膨胀。

### 3.2 行情流健壮性（Codex R1-R2）

1. `TickerStream` 脏消息容忍
- 非 list payload 直接忽略。
- 非 dict item 忽略。
- 非法数值字段安全回退。
- 非法 `event_time` 回退到当前时间。

2. 目标效果
- 单条脏消息不触发整条 WS 链路异常重连。

### 3.3 Web 观测能力增强（Codex R1-R2）

1. 新增 `GET /hotcoin/api/precheck_stats`
- 支持优先读 runner 实时统计，回退读状态文件。

2. 配置页新增"下单预检监控"
- 总失败数、失败码 Top10、币种 Top10。
- 15s 自动刷新。

### 3.4 线程安全、内存保护、业务逻辑修正（R3-R5）

> 在 Codex commit (`f489b38`) 基础上，经过 3 轮代码审查 (R3/R4/R5) 完成 28 项改进。

**线程安全与竞态修复**

1. `candidate_pool.record_heat_history`：`_heat_hist_ts` 检查+更新整体移入 `_lock` 保护，消除竞态。
2. `candidate_pool.get_top/get_all/get`：返回浅拷贝，防止外部直接修改池内对象。
3. `candidate_pool.update_coins_batch`：仅更新评分字段，不覆盖 TickerStream 写入的实时行情。
4. `candidate_pool.update_status`：新增方法，精确修改 status 而不触碰其他字段。
5. `signal_worker._add_hot_indicators`：参数直接传入而非修改全局 `config`，确保多线程安全。

**内存保护**

6. `ticker_stream`：`_vol_window` / `_price_snapshots` 改为 `collections.deque` + 每 10min 清理不活跃币种。
7. `anomaly_detector._alert_cooldown`：定期清理过期记录。
8. `social_twitter / social_binance_sq`：`_mention_counts` / `_sentiment_cache` 定期清理 + 集成 `SentimentScorer`。
9. `listing_monitor._seen_open_symbols / _seen_article_ids`：定期清理旧记录。
10. `pnl_tracker._trades`：限制最近 500 笔内存记录。
11. `online_learning._buffer`：单一标签时保留等待更多样本 + 3x 硬上限防无限增长。

**业务逻辑修正**

12. `hot_ranker`：动量评分算子优先级修复 (`(chg / 0.10) * 50`)。
13. `entry_exit_rules`：黑天鹅检测改用可获取的 `price_change_5m` 字段 + 双向检测（做多/做空）。
14. `entry_exit_rules`：分层止盈逻辑修正 — 价格跳档时取最高已达标档。
15. `portfolio_risk._check_daily_reset`：改为 UTC 0:00 自然日边界。
16. `portfolio_risk.can_open`：传入 `current_prices` 计算真实敞口。
17. `portfolio_risk.partial_close`：余量判断改为 `pos.qty < 1e-8` 避免浮点误差。
18. `capital_allocator`：`allocate_single/batch` 传入 `used_exposure` 避免资金超配。
19. `filters`：新币上线仍需满足 10% 最低流动性门槛，防止零流动性垃圾币入池。
20. `ticker_stream`：24h 成交额回退时忽略增量，避免交易所日重置导致虚假量能突增。
21. `ticker_stream`：`price_change_5m` 显式检查 `oldest_price > 0` 防除零。

**执行层改进**

22. `order_executor.spot_limit_buy`：新增去重检查。
23. `spot_engine._partial_close`：调用 `pnl.record_trade` 记录部分平仓收益。
24. `spot_engine._try_open`：平仓/开仓传入 `hint_price`，paper 模式避免额外 REST 调用。
25. `spot_engine._try_open`：使用 `pool.update_status` 而非 `pool.update_coin` 防覆盖实时数据。

**可观测性增强**

26. `signal_dispatcher`：异常日志包含完整 `exc_info` 堆栈。
27. `runner`：主循环异常日志包含 `pool_size/positions` 上下文；无价格数据时输出 warning。
28. `runner._write_status_snapshot`：填充 `recent_anomalies` 数据。

---

## 4. 当前运行风险与不足

### P0（立即处理）

1. 缺乏"可交易状态门"
- 当前仍是"尽量执行"；缺少明确的 `tradeable / degraded / blocked` 状态机。

2. 缺乏冷却与恢复的统一语义
- 数据面异常、API 异常、风控停机之间缺统一事件总线与恢复策略。

### P1（1-2 周）

1. 回测与实盘口径一致性不足
- 候选池信号、执行滑点、成交回执在离线评估中尚未完全复现。

2. 监控告警仍偏弱
- 缺 SLO 指标（例如预检失败率、信号覆盖率、延迟分位数）。

### P2（2-6 周）

1. 多模态热点因子未形成可回测闭环
- 社媒/公告信号有接入，但权重与收益归因体系不完整。

2. 机器学习增强链路未完全产品化
- `hotcoin/ml/` 具备训练模块，但尚未形成稳定在线晋升与回滚流程。

---

## 5. 下一步规划（务实版）

> 完整路线图见 `docs/hotcoin_roadmap.md`。以下为近期 3 个阶段聚焦。

### Phase A — 剩余稳定化 + 运行状态机（本周，最高优先）

1. **清零 Phase 1.5 剩余项** (~3 项)
   - `coin_age_days` 从 exchangeInfo 提取
   - `listing_monitor` 添加 429 指数退避
   - `/hotcoin/health` 健康检查端点

2. **引入运行状态机**
   - 新增 `engine_state: tradeable | degraded | blocked`
   - 行情断流 > 60s → `degraded`（暂停新仓，继续监控持仓）
   - 关键 API 连续失败 3 次 → `blocked`
   - 风控 L5 触发 → `blocked`（24h 冷却）

3. **区分可重试/不可重试失败**
   - 网络超时 → 3 次退避重试
   - 参数/预检失败 → 不重试
   - 交易所业务失败 → 按错误码处理

验收标准：
1. 行情断流 60s 内自动降级。
2. 失败重试不会被 dedup 误拦截。
3. 配置页展示当前状态及降级原因。

### Phase B — 事件日志与回测（1-2 周）

1. **事件日志契约**
   - 统一字段：`candidate_snapshot`、`signal_snapshot`、`order_attempt`、`order_result`
   - 每轮信号计算输出完整 JSONL 事件流

2. **回放工具**
   - 从 JSONL 复现"为什么进场 / 为什么没进场"

3. **回测引擎**
   - 复用执行层预检、精度与费用模型，避免回测/实盘偏差
   - 热点币历史数据采集（30 天异动币种 K线）
   - 参数优化：止盈层级、持仓时间、热度入场阈值

4. **K线缓存优化**
   - 同一 bar 内不重复拉取，降低 API 调用量和 429 风险

验收标准：
1. 任一交易可被完整回放（输入、决策、执行、结果）。
2. 回测与 paper 统计偏差显著收敛。

### Phase C — 实盘灰度（2-4 周）

1. **实盘前准备**
   - 全面端到端测试（mock Binance API）
   - 订单状态跟踪 + 紧急平仓按钮
   - 账户余额核对（每 5min）

2. **灰度三步走**
   - Step 1: $100, max_concurrent=2, 热度≥70
   - Step 2: $300, max_concurrent=3, 热度≥50
   - Step 3: $500, max_concurrent=5, 完整策略

3. **通知系统**
   - 飞书/Telegram：开仓、平仓、风控触发、系统异常
   - 日报：每日 PnL + 持仓分布

验收标准：
1. 实盘运行 14 天，无资金安全事故。
2. 紧急平仓从触发到执行 < 5s。

---

## 6. 运维执行建议

### 6.1 启停与检查

```bash
# 启动热点币系统（默认 paper）
python -m hotcoin.runner

# 运行单测（热点币子系统）
pytest -q hotcoin/tests
```

### 6.2 关键 API

```bash
# 运行态
curl -s http://127.0.0.1:5000/hotcoin/api/status | jq .

# 下单预检统计
curl -s http://127.0.0.1:5000/hotcoin/api/precheck_stats | jq .

# 热点币配置
curl -s http://127.0.0.1:5000/api/live/hotcoin_config | jq .
```

---

## 7. 版本记录

- `2026-02-21 v3`：R3-R5 改造 — 28 项线程安全/内存保护/业务逻辑修正。83 测试全绿。
- `2026-02-21 v2`：新增预检统计监控、dedup 原子预留+回滚、TickerStream 容错、文档重构。
- `2026-02-21 v1`：初始 hotcoin 模块上线。
