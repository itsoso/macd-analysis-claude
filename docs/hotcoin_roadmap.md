# 热点币系统开发路线图

## 当前状态

Phase 1 核心代码完成，近期完成 Phase A/B/C 收口（状态机、事件回放、训练治理门禁）+ R6-R7 优化。
热点币回归测试 102 项通过，核心 ETH 40 项通过，共 142 项全绿。Paper 模式可运行，处于实盘前治理阶段。

---

## Phase 1.5 — 稳定化 (预计 3 天)

**目标**: 修复剩余 P1 问题，确保 paper 模式稳定运行 7 天。

### 1.5.1 内存安全 ✅ 已完成 (R3-R5)
- [x] ticker_stream: 改用 `deque` + `_cleanup_stale()` 每 10min 清理不活跃币种
- [x] social_twitter: 集成 `SentimentScorer` + `cleanup_stale()` 定期清理
- [x] social_binance_sq: `cleanup_stale()` 定期清理
- [x] anomaly_detector: `_alert_cooldown` 定期清理过期记录
- [x] listing_monitor: `_seen_open_symbols / _seen_article_ids` 定期清理
- [x] pnl_tracker: `_trades` 限制最近 500 笔
- [x] online_learning: buffer 3x 硬上限防无限增长

### 1.5.2 线程安全 ✅ 已完成 (R3-R5)
- [x] candidate_pool: `get_top/get_all/get` 返回浅拷贝
- [x] candidate_pool: `record_heat_history` _heat_hist_ts 移入 lock
- [x] candidate_pool: `update_coins_batch` 仅更新评分字段
- [x] signal_worker: `_add_hot_indicators` 参数直传不修改全局 config
- [x] order_executor: dedup_lock 保护

### 1.5.3 数据完整性 ✅ 已完成
- [x] listing_monitor: STOP_WORDS + 正则修复 (R2)
- [x] ticker_stream: 24h 成交额日重置防虚假突增 (R5)
- [x] filters: 新币上线 10% 最低流动性门槛 (R5)
- [x] hot_ranker: `coin_age_days` 从 ExchangeInfoCache._first_seen 估算 (Phase A)
- [x] listing_monitor: 429/418 指数退避 60s→120s→240s→480s (Phase A)

### 1.5.4 可观测性 (部分完成)
- [x] runner: 主循环异常包含 pool_size/positions 上下文 (R5)
- [x] signal_dispatcher: 异常日志包含完整 exc_info 堆栈 (R5)
- [x] runner: `_write_status_snapshot` 填充 `recent_anomalies` (R3)
- [x] 配置页新增预检监控卡片 (Codex)
- [x] 配置页新增运行状态/执行指标面板（engine_state、freshness、error_rate）
- [x] 新增 `/hotcoin/api/execution_metrics`
- [x] `/hotcoin/health` 健康检查端点 (Codex + Phase A 去重)
- [x] `hotcoin_events.jsonl` 自动轮转归档（大小阈值 + 历史保留窗口）
- [x] trace_id 贯通 discovery → signal → order 全链路 (Phase C)
- [x] 热点币仪表盘支持单币 K 线 + 买卖点时间标注（`/hotcoin/api/chart`）
- [ ] Prometheus 指标导出 (池大小、信号数、PnL、延迟)

### 1.5.5 执行层健壮性 (部分完成)
- [x] order_executor: spot_limit_buy 去重 (R4)
- [x] spot_engine: _partial_close 记录 PnL (R4)
- [x] capital_allocator: 传入 used_exposure 避免超配 (R4)
- [x] portfolio_risk: UTC 0:00 日重置 + current_prices 真实敞口 (R4)
- [x] signal_dispatcher: asyncio.wait_for 超时保护 (R4)
- [x] exchangeInfo 刷新失败容错 (5min 重试 + get() 双保险异常兜底) (Phase B)
- [x] signal_dispatcher: shutdown(wait=True) 超时防死锁 (Phase B)
- [x] pnl_tracker: `_trades` 列表 threading.Lock 保护 (Phase B)

### 验收标准
- Paper 模式连续运行 7 天，无 OOM、无 crash、无 SQLite 锁冲突
- 日志无 ERROR 级别输出 (除网络超时类预期异常)
- 内存使用稳定 (不随时间增长)

**当前进度**: ~99%，剩余核心是 trace_id 贯通与健康告警闭环。

---

## Phase 2 — 信号增强与回测 (预计 1-2 周)

**目标**: 提升信号质量，建立可回放能力。

### 2.1 运行状态机 🔴 最高优先
- [x] 引入 `engine_state: tradeable | degraded | blocked`
  - 行情断流 `>=90s` → `degraded`，`>=300s` → `blocked`
  - `order_errors_5m >=3` → `degraded`，`>=10` → `blocked`
  - 风控 L5 触发 → `blocked` (24h 冷却)
  - 恢复滞后窗口（hysteresis）已接入：degraded=3 周期、blocked=6 周期确认恢复
- [x] 区分可重试/不可重试失败类型 (retryable/non_retryable/unknown 三级) (Phase C)
- [x] 配置页展示当前状态及降级原因

### 2.2 事件日志契约
- [x] 统一字段规范：`candidate_snapshot`、`signal_snapshot`、`order_attempt`、`order_result`
- [x] 每轮信号计算输出完整 JSONL 事件流
- [x] 回放工具：从 JSONL 复现 "为什么进场 / 为什么没进场"
- [x] 事件 `trace_id` 贯通 discovery → signal → order (Phase C)

### 2.3 回测框架
- [ ] 热点币历史数据采集 (过去 30 天异动币种 K线)
- [ ] 热点币回测引擎 (模拟 discovery → signal → execution 全流程)
- [ ] 复用执行层预检、精度与费用模型，避免回测/实盘偏差
- [ ] 参数优化: 止盈层级、持仓时间、热度入场阈值

### 2.4 K线缓存优化 ✅ 已完成 (Phase B)
- [x] 同一 bar 内不重复拉取 (bar-level 缓存, TTL=bar_sec*0.8)
- [x] 缓存 LRU 淘汰 (max 500 条), 线程安全
- [x] 减少 Binance REST API 调用量，降低 429 风险

### 2.5 ML 热点币模型
- [ ] 使用 `features_hot.py` 生成训练数据 (~30 维热点特征)
- [ ] LightGBM 热度预测模型 (目标: 预测未来 30min 涨幅 top 10%)
- [ ] Shadow 模式验证 (不修改信号，只记录对比)

### 2.6 模型治理（新增）
- [x] 训练后自动生成 `runtime_contract_{task}_{interval}.json`
- [x] 训练后自动生成 `promotion_decision_{task}_{interval}.json`
- [ ] 运行时强制读取 promotion 决策（production/research_only）
- [ ] 模型 hash/version 写入运行状态与事件流

### 验收标准
- 任一交易可被完整回放（输入、决策、执行、结果）
- 回测与 paper 统计偏差显著收敛
- 行情断流按阈值自动降级（当前 `>=90s degraded / >=300s blocked`）

---

## Phase 3 — 实盘灰度 (预计 2 周)

**目标**: 小资金 ($100-500) 实盘验证。

### 3.1 实盘前准备
- [ ] 全面端到端测试 (mock Binance API)
- [x] 订单状态跟踪: 定期查询 open orders + 过期 LIMIT 自动取消 (Phase C)
- [ ] 紧急平仓按钮 (web UI + API)
- [ ] Binance 资产查询: 每 5min 核对账户余额与系统记录

### 3.2 灰度策略
- **Step 1**: $100 资金, max_concurrent=2, 仅高热度(≥70)币种
- **Step 2**: $300 资金, max_concurrent=3, 热度≥50
- **Step 3**: $500 资金, max_concurrent=5, 完整策略

### 3.3 风控加强
- [ ] 紧急刹车: 日亏损 > 3% → 暂停 + 通知
- [ ] 网络故障处理: WebSocket 断连 > 2min → 暂停新仓
- [ ] 相关性聚类限仓 (按主题/板块)

### 3.4 通知系统
- [ ] 飞书/Telegram 通知: 开仓、平仓、风控触发、系统异常
- [ ] 日报: 每日 PnL 汇总 + 持仓分布
- [ ] 异常告警: 连续 3 笔亏损 → 通知

### 验收标准
- 实盘运行 14 天，无资金安全事故
- 紧急平仓从触发到执行 < 5s
- 每笔交易可追溯完整决策链

---

## Phase 4 — 合约扩展 (预计 2 周)

**目标**: 支持 USDT 永续合约做空。

### 4.1 合约完善
- [ ] `futures_adapter.py` 与 spot_engine 对齐 (精度格式化、预检查)
- [ ] 开仓前检查反向持仓
- [ ] 合约风控: 保证金率监控、强平预警
- [ ] 合约止盈止损: 考虑资金费率影响

### 4.2 多策略
- [ ] 做空信号: 热度下降 + 六书卖出信号
- [ ] 对冲: BTC 下跌时热点币做空对冲

---

## Phase 5 — 规模化 (长期)

### 5.1 性能
- [ ] 信号缓存: 同一 bar 内不重复计算
- [ ] 多进程架构: 信号计算从 ThreadPool 迁移到 ProcessPool (绕过 GIL)

### 5.2 GPU 训练
- [ ] 热点币专用 LSTM (短周期, 1m/5m 为主)
- [ ] TFT 热度序列预测
- [ ] 强化学习仓位优化 (PPO, 考虑多币种相互影响)
- [ ] H800 训练流程: pack_hotcoin_data.sh → train → 模型回传

### 5.3 社媒信号升级
- [ ] Twitter: KOL 权重分级 (粉丝数 > 10K → 高权重)
- [ ] 情感分析: 使用 FinBERT 替换简单规则
- [ ] 链上数据: 大额转账监控、DEX 流动性

### 5.4 扩展
- [ ] 支持 OKX/Bybit 交易所
- [ ] 币种板块自动分类 (NLP)

---

## 优先级矩阵（更新）

```
               紧急
                ↑
    Phase 1.5   │   N1
    (剩余稳定化)│   (健康端点+状态恢复)
                │
  ──────────────┼──────────────→ 重要
                │
    Phase 2.3   │   Phase 2.6
    (回测框架)  │   (模型治理产品化)
                │
                ↓
               不紧急
```

**建议执行顺序**: 1.5 剩余 → N1（健康/日志轮转/状态恢复）→ 2.3 回测 → 2.6 运行时模型治理 → 3 实盘灰度 → 4 合约

- **1.5 剩余稳定化** 工作量小 (~3 项)，尽快清零
- **运行状态机/事件日志基础能力** 已落地，下一步要补 trace_id 链路与健康告警闭环
- **2.3 回测框架** 是策略参数优化与实盘前的关键验证环节
- **2.6 模型治理** 决定后续 H800 模型能否稳定上线
- **Phase 3 实盘灰度** 需要 1.5 + 2.1 + 2.2 完成后进入
- **Phase 4 合约** 在现货跑通后再考虑

---

## 变更日志

- `2026-02-18` R7 (Opus)：TickerStream 快照副本 (Flask 线程安全)、AnomalyDetector 清理异常保护、紧急平仓 API (emergency_close_all)、资产余额查询 API、web/routes _runner TOCTOU 修复。102 hotcoin + 40 core = 142 测试全绿。
- `2026-02-21` R9：新增 `/hotcoin/api/chart` 与仪表盘 K 线买卖点可视化，补充 chart API 测试。84 测试全绿。
- `2026-02-21` R8：状态机恢复滞后（hysteresis）上线，新增 `test_runner_state_hysteresis.py`。70 测试全绿。
- `2026-02-18` R6 Review：PortfolioRisk 全方法线程锁、OrderExecutor metrics_lock、get_current_price/get_avg_price 异常日志、process_signals 复用 current_prices 减少 API 调用、_order_history 实盘修剪、主循环致命异常 (MemoryError/SystemExit) 直接关闭。87 hotcoin + 40 core = 127 测试全绿。
- `2026-02-18` Phase C：trace_id 全链路贯通 (cycle_id → TradeSignal → order events)、订单错误三级分类 (retryable/non_retryable/unknown)、订单状态对账 (query_open_orders + 过期 LIMIT 自动取消)。77 hotcoin + 40 core = 117 测试全绿。
- `2026-02-18` Phase B Fix：exchangeInfo 刷新失败 5min 重试 + get() 双保险兜底、SignalDispatcher shutdown 超时防死锁、K线 bar-level 缓存 (TTL=80% bar 周期)、PnLTracker 线程安全。62 hotcoin + 40 core = 102 测试全绿。
- `2026-02-21` R7：新增事件日志自动轮转归档（Runner），补充 health 聚合测试与归档测试。58 测试全绿。
- `2026-02-21` R6：完成状态机、事件回放、训练治理产物接入，更新路线图优先级。
- `2026-02-21` Phase A：coin_age_days 估算、listing_monitor 429 退避、health 端点去重。91 测试全绿。
- `2026-02-21` R5：五轮改进完成，44 项修复，更新路线图反映当前进度。
- `2026-02-21` R2：两轮 review 完成，16 项修复。
- `2026-02-18`：Phase 1 初始实现完成。
