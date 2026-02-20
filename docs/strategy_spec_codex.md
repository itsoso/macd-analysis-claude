# ETH/USDT 策略技术规格（Codex Runtime Review）

- 文档目标：基于**最新代码 + 实际运行日志 + 模拟交易结果 + 参数配置**，给出可执行的全局架构改进计划。
- 页面入口：`/strategy/spec-codex`（线上地址：`https://invest.executor.life/strategy/spec-codex`）
- 本地源文件：`docs/strategy_spec_codex.md`
- 评估时间：`2026-02-20 23:10 CST`
- 代码基线：`8b12f3f`（最近 5 次提交：`8b12f3f / f78a04a / 5af5516 / a97f4e7 / 0861145`）

---

## 1. 当前状态结论（先给结论）

当前系统的主问题已经从“模型效果”转为“运行架构失配”：

1. **实时运行态没有吃到最新能力**：进程长期运行，日志行为与当前代码不一致。
2. **数据面失效**：最新信号 bar 停在 `2026-02-17 17:00:00`，但引擎时间已到 `2026-02-20 22:47:59`（滞后 `77.8h`）。
3. **ML 分支在线上日志中完全不可见**：`SIGNAL` 记录中 `ml_*` 字段为 0。
4. **策略执行层实际“冻结”**：最近日志 `100` 条 `SIGNAL` 全部 `HOLD`，模拟权益横盘。

这意味着：
- 短期最优先不是继续加模型复杂度，而是先修复**数据新鲜度、运行版本一致性、上线门禁**。

---

## 2. 证据快照（2026-02-20）

### 2.1 线上/实盘日志

来源：`logs/live/trade_20260216.jsonl`、`logs/live/engine_paper.log`

- 事件计数：`FUNDING=449, BALANCE=1029, SIGNAL=100`
- `SIGNAL` 动作：`HOLD=100`
- `ml_*` 覆盖：
  - `signals_with_ml_fields=0`
  - `ml_enabled_true=0`
  - `ml_available_true=0`
- 最新信号：
  - 记录时间：`2026-02-20 22:47:59,427`
  - bar 时间：`2026-02-17 17:00:00`
  - 滞后：`77.8h`

### 2.2 运行进程状态

- 引擎进程：`PID=495`
- 启动时间：`Mon Feb 16 15:33:24 2026`
- 命令：`python live_runner.py --phase paper -y`

这说明当前日志是长期进程输出，不一定代表最新代码语义。

### 2.3 模拟交易结果（本地 performance）

来源：`data/live/performance.json`

- 已闭合交易：`5`
- 胜率：`40%`（2 胜 / 3 负）
- 平仓净收益（扣平仓手续费）：`-427.41`
- 总费用累计：`280.13`
- 最大回撤：`1.34%`
- 最近多天基本无新增成交。

### 2.4 健康检查结果

来源：`python3 check_ml_health.py --verbose`

- 模型文件完整性：通过
- 模型加载：失败（本机环境缺包）
  - `lightgbm` 缺失
  - `torch` 缺失
  - `scikit-learn` 缺失
- 端到端 `enhance_signal` 可执行但 `ml_available=False`
- 最新 live `SIGNAL` 无 `ml_*` 字段：失败

---

## 3. 代码 Review（按严重级别）

### P0（必须先修）

1. **部署脚本默认强制开启 Stacking，和当前样本门禁策略冲突**  
文件：`deploy.sh:22-32`, `deploy_local.sh:9-19`
- 现状：`ML_ENABLE_STACKING=1` 写死。
- 风险：与“小样本先禁用/门控后启用”策略冲突，导致线上行为不可预期。
- 建议：默认改为 `0`，由 `promotion_decision.json` 自动控制启停。

2. **数据获取层对主 K 线“只要有本地缓存就直接使用”，未做 live 新鲜度门禁**  
文件：`binance_fetcher.py:34-88`, `binance_fetcher.py:91-114`
- 现状：`fetch_binance_klines()` 优先本地，不检查缓存是否过旧。
- 风险：live 可能长期消费过期 bar。
- 建议：为 live 模式增加 `require_fresh=True`；超过阈值时强制 API/降级/停机。

3. **运行态版本不可追踪，难以确认“代码-模型-配置”是否一致**  
文件：`live_runner.py`, `live_trading_engine.py`, `live_signal_generator.py`
- 现状：日志无 `git_commit/config_hash/model_hash`。
- 风险：出现“代码已修复但线上无效果”时无法快速定位。
- 建议：启动时打印并落盘 runtime manifest（commit + model fingerprint + config digest）。

### P1（1-2周内）

4. **多周期共识上下文未写入 SIGNAL 结构化日志（已构造但未传递）**  
文件：`live_trading_engine.py:366-385`, `trading_logger.py:129-163`
- 现状：`log_extra` 变量构造后未传入 `log_signal(..., extra=...)`。
- 风险：无法在 JSON 日志中回放共识决策细节。
- 建议：补充 `extra=log_extra`，提高复盘可观测性。

5. **衍生品 API 失败缺少断路器，重复重试造成每轮刷新延迟和日志噪音**  
文件：`binance_fetcher.py:249-263`, `binance_fetcher.py:430-463`, `binance_fetcher.py:564-597`
- 现状：每轮刷新都会重试网络失败。
- 风险：刷新耗时抖动，日志刷屏，策略时效下降。
- 建议：增加 per-endpoint 断路器（失败 N 次后冷却 T 分钟）。

6. **运行配置落库不完整，关键策略参数常以默认值隐式生效**  
文件：`config_store.py`, `live_config.py`
- 现状：DB `strategy` 中多项关键字段为空时回退默认。
- 风险：改参数后可追踪性弱，环境漂移难排查。
- 建议：保存“完整展开配置快照”，并生成 config diff 审计。

### P2（中期）

7. **健康检查未与服务启动强绑定，仍靠人工执行**  
文件：`check_ml_health.py`, `deploy.sh`, `deploy_local.sh`
- 建议：systemd `ExecStartPre` 里执行 health check，失败则拒绝启动。

8. **训练产物契约尚未强制化（虽已改善 Multi-Horizon）**  
文件：`train_gpu.py`, `ml_live_integration.py`
- 建议：统一产物规范：`{model, meta, schema, version, hashes, promotion_decision}`。

---

## 4. 目标架构（建议）

### 4.1 四层控制闭环

1. **Data Plane（行情层）**
- 统一 DataFreshnessGate：`max_lag_hours`、`freshness_source`、`stale_policy`。
- 主 K / Mark / Funding / OI 分开 freshness 指标。
- 断路器 + 回退顺序：LocalFresh -> API -> Halt。

2. **Model Plane（模型层）**
- 训练产物必须带 schema/hash。
- promotion 门禁自动化：样本量、AUC、OOF-Test gap、特征覆盖率。
- 线上只加载 `production` 标签工件。

3. **Decision Plane（决策层）**
- 信号分三态：`tradeable` / `degraded` / `blocked`。
- 明确 fail-open/fail-closed 策略：
  - paper 可降级
  - live 执行默认 fail-closed（至少开仓 fail-closed）

4. **Ops & Observability（运维层）**
- runtime manifest 强制落地。
- 结构化日志最小集：`freshness, model_version, gating_reason, consensus_snapshot`。
- 报警：连续 stale、ML 覆盖降至 0、API 连续失败、信号全 HOLD 超阈值。

---

## 5. 分阶段落地计划（务实可执行）

### Phase A（48小时）

1. 部署脚本改默认：`ML_ENABLE_STACKING=0`。
2. 启动前健康检查（依赖、模型、别名一致性）接入 systemd `ExecStartPre`。
3. 引擎启动打印 runtime manifest（commit/config/model）。
4. `SIGNAL` 日志补齐 `consensus_*` extra 字段。

**验收标准**
- 新启动日志里有 `runtime_manifest`。
- `trade_*.jsonl` 的 `SIGNAL.data` 含 `consensus_strength`。
- `check_ml_health` 结果可被 systemd 启动前消费。

### Phase B（1-2周）

1. `binance_fetcher` 增加 live freshness 校验与断路器。
2. 数据滞后超过阈值时，进入 `degraded/blocked` 状态并报警。
3. 配置持久化改为“完整展开 + diff 审计”。

**验收标准**
- 无网络时不再每小时长时间重试刷屏。
- stale 场景下不会继续输出可交易信号。
- 配置变更具备可追踪 diff。

### Phase C（2-6周）

1. 建立模型注册与晋升（promotion）流水线。
2. 训练产物统一契约与签名校验。
3. 远程 GPU 推理服务接入生产（本机仅轻量 fallback）。

**验收标准**
- 线上模型来源可追溯到单一 promotion 记录。
- 推理服务切换后，ML 覆盖率稳定 >95%。
- 可一键回滚至前一版已签名模型。

---

## 6. H800 执行计划（训练侧）

以现有脚本为主：`scripts/run_h800_training_plan.sh`

1. `base`：`lgb + lstm(multi-horizon) + tft + cross_asset`
2. `stacking`：仅在 `n_samples >= 20000` 时执行
3. `onnx`：导出并做 smoke test
4. `report`：自动生成门禁判断（建议扩展为 `promotion_decision.json`）

建议新增产物：
- `data/ml_models/promotion_decision.json`
- `data/ml_models/runtime_contract.json`

---

## 7. 本轮建议的“先后顺序”

1. **先修运行一致性**（启动门禁 + manifest + 日志可观测）。
2. **再修数据新鲜度与断路器**（避免 stale 驱动与重试风暴）。
3. **最后再加模型复杂度**（RL/LLM/更深模型都应建立在稳定运行底座之上）。

---

## 8. 关键命令（运维/排查）

```bash
# 1) 训练后别名一致性
python3 scripts/sync_stacking_alias.py --tf 1h --check-only

# 2) 启动前健康检查
python3 check_ml_health.py --timeframe 1h --verbose

# 3) 最近 7 天 shadow 覆盖
python3 analyze_shadow_logs.py --days 7

# 4) 查看最新 SIGNAL 是否含 ml_* 字段
rg '"level": "SIGNAL"' logs/live/trade_*.jsonl | tail -n 5

# 5) 查看引擎进程启动时间（确认是否重启到新版本）
ps -p $(cat data/live/engine.pid) -o pid,lstart,command
```

---

## 9. 说明

- 你给的线上地址是页面入口；本次已更新本地源文档 `docs/strategy_spec_codex.md`。  
- 若线上站点未自动读取仓库文件，请执行你的发布流程同步该文档。
