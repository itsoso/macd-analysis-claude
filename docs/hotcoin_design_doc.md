# HotCoin 热点币系统设计文档（2026-02-21）

- 文档目标：统一描述 HotCoin 发现-信号-执行-监控全链路，明确当前实现状态与下一步改造计划。
- 代码范围：`hotcoin/`、`hotcoin/web/`、`templates/page_live_hotcoin_config.html`
- 代码基线：`main`（含 Phase A/B/C 最新改造）

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
- `hotcoin/web/routes.py`：`/hotcoin/api/status`、`/hotcoin/api/precheck_stats`、`/hotcoin/api/execution_metrics`。
- `templates/page_live_hotcoin_config.html`：热点币独立配置页 + 预检监控 + 运行状态/执行指标卡片。
- `hotcoin/data/hotcoin_events.jsonl`：统一事件流（可回放）。
- `app.py`：`/api/live/hotcoin_config` 配置读写与校验。

### 2.2 主流程

1. `HotCoinRunner` 周期执行（`signal_loop_sec`）：
- 更新候选池 → 热度评分 → 过滤 TopN → 生成信号。
2. 执行开平仓：
- `SpotEngine.process_signals()` → 风控检查 → `OrderExecutor` 下单。
3. 状态落盘：
- 写 `hotcoin/data/hotcoin_runtime_status.json`（含 `engine_state`、`freshness`、`execution_metrics`、候选池、信号、仓位、预检统计）。
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

### 3.5 运行状态机与执行门控（Phase A，已落地）

1. 引入 `engine_state: tradeable | degraded | blocked`
- 由 `ticker` 新鲜度、`order_errors_5m`、`risk_halted` 联合决定。
- 阈值：`ticker stale >= 90s` 进入 `degraded`，`>= 300s` 进入 `blocked`。
- 阈值：`order_errors_5m >= 3` 进入 `degraded`，`>= 10` 进入 `blocked`。

2. 开仓门控
- `spot_engine.process_signals(signals, allow_open=...)`。
- 非 `tradeable` 状态下暂停新开仓，SELL/平仓逻辑保留。

3. 运行状态可视化
- `hotcoin/api/status` 返回 `engine_state`、`engine_state_reasons`、`can_open_new_positions`、`freshness`、`execution_metrics`。
- 配置页新增运行状态面板与 5m 执行指标。

### 3.6 事件契约与回放（Phase B，已落地）

1. 统一事件流输出
- `hotcoin/runner.py` 输出 `candidate_snapshot`、`signal_snapshot`。
- `hotcoin/execution/spot_engine.py` 输出 `order_attempt`、`order_result`。
- 事件统一写入 `hotcoin/data/hotcoin_events.jsonl`。

2. 回放工具
- 新增 `scripts/replay_hotcoin_events.py`。
- 支持按 `event_type/symbol` 过滤、摘要统计、最近事件回看。

### 3.7 训练治理产物（Phase C，已落地）

1. `hotcoin/ml/train_hotcoin.py` 在 `hotness`/`trade` 训练后自动产出：
- `runtime_contract_{task}_{interval}.json`
- `promotion_decision_{task}_{interval}.json`

2. 门禁规则（当前）
- `n_samples >= 20000`
- `test_auc >= 0.55`
- 未达标标记 `research_only`，达标标记 `production`。

---

## 4. 当前运行风险与不足

### P0（立即处理）

1. 事件链路缺少 trace_id / signal_id
- 当前已有 `candidate_snapshot/signal_snapshot/order_attempt/order_result`，但跨阶段追踪仍需统一 ID。

2. 健康指标缺少告警闭环
- `/hotcoin/health` 已提供聚合状态，但尚未接入阈值告警与自动化处置策略。

3. 运行状态恢复策略仍可细化
- 已接入恢复滞后（hysteresis），但尚未按不同故障类型（行情断流/执行错误/risk halt）配置差异化恢复阈值。

### P1（1-2 周）

1. 回测与实盘口径一致性不足
- 候选池信号、执行滑点、成交回执在离线评估中尚未完全复现。

2. 监控告警仍偏弱
- 缺 SLO 告警闭环（例如预检失败率、信号覆盖率、延迟分位数阈值告警）。

### P2（2-6 周）

1. 多模态热点因子未形成可回测闭环
- 社媒/公告信号有接入，但权重与收益归因体系不完整。

2. 机器学习增强链路未完全产品化
- 训练治理产物已落地，但运行时尚未强制消费 `promotion_decision` 做线上晋升/回滚。

---

## 5. 下一步规划（务实版）

> 完整路线图见 `docs/hotcoin_roadmap.md`。以下为近期 3 个阶段聚焦。

### N1 — 运维闭环（本周，最高优先）

1. `/hotcoin/health` 聚合健康端点（已完成）
- 覆盖：WS 连接、新鲜度、risk halt、executor 错误率、状态机状态。

2. 事件日志轮转与归档（已完成）
- `hotcoin_events.jsonl` 按天或大小滚动，保留窗口 + 自动压缩。

3. 状态机恢复滞后（hysteresis）（已完成）
- 由 `degraded/blocked` 恢复到 `tradeable` 需满足连续 N 个周期正常，减少抖动。

验收标准：
1. 任意时刻可通过一个端点拿到系统健康摘要。
2. 事件日志不会无限增长。
3. 运行状态不再在阈值边缘频繁抖动。

### N2 — 回放到回测（1-2 周）

1. 建立事件到交易的可追溯链
- 事件中统一增加 `trace_id` / `signal_id`，贯通发现-信号-执行。

2. 回测引擎复用执行语义
- 复用预检、精度、费用、去重与风控，减少 paper/live 偏差。

3. K线缓存与限流策略
- 同一 bar 不重复请求，加入请求预算与 429 退避。

验收标准：
1. 任一订单可从事件流回放到策略决策路径。
2. 回测与 paper 的核心指标偏差收敛到可解释范围。

### N3 — 模型晋升产品化（2-4 周）

1. 运行时强制消费 `promotion_decision`
- 仅 `production` 模型允许在线推理，`research_only` 自动隔离。

2. 模型版本与配置指纹写入运行状态
- 状态文件和事件流包含 `model_version/model_hash/config_hash`。

3. 实盘灰度开关治理
- 资金阶梯、币种白名单、紧急平仓开关纳入统一配置页。

验收标准：
1. 模型上线/回滚有完整审计轨迹。
2. 灰度切换具备可观测、可回退、可验证流程。

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

# 执行错误率/去重拒绝率（5m）
curl -s http://127.0.0.1:5000/hotcoin/api/execution_metrics | jq .

# 健康聚合
curl -s http://127.0.0.1:5000/hotcoin/health | jq .

# 热点币配置
curl -s http://127.0.0.1:5000/api/live/hotcoin_config | jq .
```

### 6.3 事件回放

```bash
# 事件摘要
python3 scripts/replay_hotcoin_events.py --summary-only

# 按事件类型/交易对回放
python3 scripts/replay_hotcoin_events.py --event-type order_result --symbol ETHUSDT --limit 50
```

---

## 7. 版本记录

- `2026-02-21 v6`：N1 收口：状态恢复滞后（hysteresis）上线，新增状态机恢复单测与事件日志轮转单测。
- `2026-02-21 v5`：N1 持续推进：`/hotcoin/health` 聚合健康端点 + `hotcoin_events.jsonl` 自动轮转归档（含保留窗口）。
- `2026-02-21 v4`：完成 Phase A/B/C 收口：运行状态机 + 执行指标 + 事件回放 + 训练治理产物（runtime_contract/promotion_decision）。
- `2026-02-21 v4`：Phase A — coin_age_days 估算、listing_monitor 429 退避、health 端点对齐。91 测试全绿。
- `2026-02-21 v3`：R3-R5 改造 — 28 项线程安全/内存保护/业务逻辑修正。83 测试全绿。
- `2026-02-21 v2`：新增预检统计监控、dedup 原子预留+回滚、TickerStream 容错、文档重构。
- `2026-02-21 v1`：初始 hotcoin 模块上线。
