# ETH/USDT 六书融合策略 技术规格书（Codex）

- 文档目的：给研发与其他 LLM 提供“可执行、可复核、与当前代码一致”的策略基线。
- 适用范围：`backtest_multi_tf_daily.py` / `optimize_six_book.py` / `live_config.py` / `live_signal_generator.py`。
- 更新时间：`2026-02-15 20:49:00 CST`
- 页面入口：`/strategy/spec-codex`（独立于 `/strategy/spec` 与 `/strategy/tech-doc`）

---

## 1. 执行摘要

当前代码已经具备 v9/v10 的大部分能力（Mark 口径、真实 Funding、Leg 风险预算、Risk-per-trade、Soft Veto 等），但默认运行路径仍是“保守兼容”形态。主要结论如下：

1. `STRATEGY_VERSION=v5` 已生效，且关键 v10 开关已映射到 `StrategyConfig`（本次修复）。
2. 回测主链路 `kline_store.load_klines()` 现在会保留可选 `mark_* / funding_rate / open_interest` 列（若本地文件存在这些列）。
3. 最新 v10 A/B（`data/backtests/v10/v10_backtest_20260215_192304.csv`）显示：
   - `E1_v9_v5_prod`：IS `PF=0.90`（低于 1.0），OOS `Ret=-3.83%`
   - `E4_v10_full`：IS `PF=1.31`，OOS `Ret=+7.87%`
4. 结论不是“v10 默认已经上线”，而是“v10 机制可用，但默认配置与数据链路尚未形成单轨一致”。

---

## 2. 代码事实基线

### 2.1 主流程

1. 回测入口：`backtest_multi_tf_daily.py`
2. 策略循环：`optimize_six_book.py`
3. 配置来源：`live_config.py`（`StrategyConfig` + `STRATEGY_PARAM_VERSIONS`）
4. 数据来源：`kline_store.py`（本地优先，可选 API 回退）
5. 结果落库：`multi_tf_daily_db.py`（`mtf_runs` / `mtf_daily` / `mtf_trades`）

### 2.2 实盘流程

1. 实盘入口：`live_runner.py`
2. 实盘信号：`live_signal_generator.py`
3. 实盘数据增强：`binance_fetcher.py` 的
   - `fetch_mark_price_klines`
   - `fetch_funding_rate_history`
   - `fetch_open_interest_history`
   - `merge_perp_data_into_klines`

说明：实盘链路已具备衍生品数据融合；回测链路默认未全量透传。

---

## 3. 当前默认配置（代码实测）

通过直接实例化 `StrategyConfig()`（当前环境 `STRATEGY_VERSION=v5`）得到：

| 参数 | 实际值 |
|---|---|
| `short_threshold` | `40` |
| `long_threshold` | `25` |
| `short_sl` | `-0.20` |
| `short_tp` | `0.60` |
| `cooldown` | `6` |
| `regime_short_threshold` | `neutral:60` |
| `use_regime_adaptive_fusion` | `False` |
| `use_leg_risk_budget` | `True` |
| `risk_budget_neutral_short` | `0.1` |
| `use_continuous_trail` | `True` |
| `use_regime_adaptive_sl` | `True` |
| `use_soft_veto` | `True` |
| `soft_struct_min_mult` | `0.02` |

关键点：本次已将 v5 关键开关纳入 `_resolve_param` 映射，`StrategyConfig()` 默认值与 `STRATEGY_PARAM_VERSIONS['v5']` 的关键参数对齐。

剩余注意：仍有部分字段保持 dataclass 常量默认，若后续继续扩展版本参数，建议统一走映射或统一后处理覆盖。

代码锚点：`live_config.py:323` 到 `live_config.py:340`、`live_config.py:432`。

---

## 4. 策略架构（当前实现）

### 4.1 五层架构

1. 信号层：`signal_core.py`
   - 六书打分：DIV / MA / CS / BB / VP / KDJ
   - 支持单 bar 与批量向量化计算
2. 多周期融合层：`multi_tf_consensus.py`
   - 默认决策 TF：`15m + 1h + 4h + 24h`
   - 输出 `SS/BS + meta + book_features_weighted`
3. 决策层：`optimize_six_book.py`
   - Regime 判别
   - 阈值动态化
   - 门控与仓位控制
   - 开平仓状态机（T+1 open）
4. 执行层：`strategy_futures.py` + `optimize_six_book.py`
   - 分段止盈、追踪止盈、强平、手续费、Funding
5. 持久化与展示层：`multi_tf_daily_db.py` + `app.py`
   - 数据库 + Web 页面

### 4.2 v9/v10 已实现能力

| 能力 | 实现位置 | 默认 |
|---|---|---|
| Mark 口径强平 | `optimize_six_book.py` | 关 |
| UTC 锚定 Funding 结算 | `optimize_six_book.py` | 真实 funding 默认关 |
| Leg 风险预算 | `optimize_six_book.py` | `v5` 下开 |
| Risk-per-trade | `optimize_six_book.py` | 关 |
| Weighted confirms | `optimize_six_book.py` | 关 |
| Regime-adaptive fusion | `optimize_six_book.py` | 关 |
| Soft Veto | `optimize_six_book.py` | `v5` 下开 |
| Ghost cooldown / fast-fail | `optimize_six_book.py` | 关 |

---

## 5. 数据口径与落库口径

### 5.1 回测数据链路的现实限制

`kline_store.load_klines()` 会裁剪列，但已包含衍生品可选列：

- 标准列：`open, high, low, close, volume, quote_volume`
- 扩展列：`taker_buy_base, taker_buy_quote, trades`
- 衍生品列：`mark_*`, `funding_rate`, `funding_interval_hours`, `open_interest`, `open_interest_value`

代码锚点：`kline_store.py:358` 到 `kline_store.py:361`。

影响：即使策略支持 `mark_*`/`funding_rate`/`open_interest`，如果不改加载链路，默认回测仍拿不到这些列。

### 5.2 `summary_json` 字段（当前真实口径）

`backtest_multi_tf_daily.py` 输出的是：

- `total_return_pct`
- `max_drawdown_pct`
- `win_rate_pct`
- `contract_pf`
- `portfolio_pf`
- `profit_factor`（兼容字段，等于 `contract_pf`）
- 费用与 Funding 字段

代码锚点：`backtest_multi_tf_daily.py:1042` 到 `backtest_multi_tf_daily.py:1062`。

### 5.3 run id 口径

数据库主键是 `mtf_runs.id`，不是 `run_id` 列。

代码锚点：`multi_tf_daily_db.py` 表结构定义。

---

## 6. 最新回测结果（以 CSV 为准）

## 6.1 v9 Round 3（`data/backtests/v9_ab/v9_ab_r3_20260215_180525.csv`）

| 变体 | IS Ret | IS PF | OOS Ret | OOS PF |
|---|---:|---:|---:|---:|
| `E0_baseline_fixed` | `+65.74%` | `1.14` | `-3.32%` | `1.67` |
| `E1_P23_weighted` | `+43.21%` | `0.95` | `-12.64%` | `0.72` |
| `E2_P21_risk_R` | `-9.95%` | `0.24` | `+0.06%` | `1.61` |
| `E3_P23_P21` | `+48.33%` | `1.06` | `-10.98%` | `0.87` |
| `E4_P18lite_P23` | `-6.22%` | `0.94` | `+28.91%` | `1.33` |
| `E5_full_v9` | `-4.19%` | `1.18` | `+12.74%` | `1.09` |

解读：IS/OOS 分裂明显，P18 类改造在 OOS 有改善，但 IS 不稳定。

## 6.2 v9 全量对照（`data/backtests/v9_ab/v9_full_backtest_20260215_184527.csv`）

| 变体 | IS Ret | IS PF | OOS Ret | OOS PF |
|---|---:|---:|---:|---:|
| `E0_v8_baseline` | `+67.34%` | `1.18` | `-3.83%` | `1.59` |
| `E1_v9_v5_prod` | `+46.63%` | `0.90` | `-3.83%` | `1.59` |
| `E2_v5_P18lite` | `-14.45%` | `0.44` | `+41.11%` | `2.27` |
| `E3_v5_P18lite_P23` | `+5.02%` | `0.71` | `+37.75%` | `2.13` |

结论：当前 `v5` 生产参数在 IS 的 `PF<1`，而 OOS 仍有 “仅 long 驱动” 特征。

## 6.3 v10 验证（`data/backtests/v10/v10_backtest_20260215_192304.csv`）

| 变体 | IS Ret | IS WR | IS PF | OOS Ret | OOS WR | OOS PF |
|---|---:|---:|---:|---:|---:|---:|
| `E0_v8_baseline` | `+67.34%` | `56.86%` | `1.19` | `-3.83%` | `65.12%` | `1.56` |
| `E1_v9_v5_prod` | `+46.63%` | `56.36%` | `0.90` | `-3.83%` | `65.12%` | `1.56` |
| `E2_soft_veto` | `+51.11%` | `58.87%` | `1.03` | `+7.03%` | `56.16%` | `0.92` |
| `E3_sv_softStruct` | `+51.11%` | `58.87%` | `1.03` | `+7.03%` | `56.16%` | `0.92` |
| `E4_v10_full` | `+49.14%` | `61.43%` | `1.31` | `+7.87%` | `56.16%` | `1.02` |
| `E5_v10_relaxedSL` | `+46.81%` | `58.74%` | `1.14` | `+8.36%` | `58.11%` | `1.09` |

结论：`E4/E5` 修复了 v5 的 `IS PF<1` 问题，并让 OOS 由负转正。

---

## 7. 当前关键问题（按优先级）

## P0：版本映射仍需系统化

- 现状：已修复 v5 关键开关映射（leg budget / soft veto / 连续追踪 / regime-SL）。
- 风险：版本键数量继续增加时，仍可能出现“版本表定义了但 dataclass 未映射”的漂移。
- 方向：引入统一版本覆盖层（初始化后统一 apply），减少逐字段遗漏风险。

## P0：回测数据来源仍未自动合并 perp 数据

- 本次已保留 `mark/funding/oi` 列，但前提是本地 parquet 已包含这些字段。
- 若历史本地库未预先合并 perp 数据，主回测仍会退化到纯 OHLCV 口径。

## P1：指标口径混用风险

- 新口径以 `total_return_pct/portfolio_pf` 为准。
- 旧脚本仍常用 `strategy_return/profit_factor`，且 `profit_factor` 当前等于 `contract_pf`。

## P1：实验结论分裂（IS/OOS）

- P18/P23 在不同时间段方向相反，说明参数/机制对样本体制敏感。
- 需要严格单变量和滚动验证，不宜直接全开。

---

## 8. 下一步优化路线（可执行）

### 阶段 A：先打通口径（必须先做）

1. 统一版本覆盖机制：从“逐字段 `_resolve_param`”升级为“集中覆盖”，避免新参数漏映射。
2. 打通回测 perp 数据注入：在主回测链路增加可控合并步骤，确保 `mark/funding/oi` 真实可用。
3. 固化输出口径：统一使用 `total_return_pct + contract_pf + portfolio_pf`。

### 阶段 B：策略增强（基于已验证方向）

1. 以 `E4_v10_full` 为新基线进行 A/B（不要回到 v5 直接上线）。
2. 在 `E4` 上只单独测试一个变量（例如 `regime_trend_short_sl`），避免交互干扰。
3. 强制记录 `Worst-5`、`Calmar`、`regime × side` 明细，作为上线门槛。

### 阶段 C：中期重构

1. 将 `regime × side` 拆成独立 leg（状态、预算、KPI 分离）。
2. 把 P18/P23 做成可插拔子模块，脱离主循环硬编码。
3. 增加基于 DB 的回归测试（关键 run 的结果阈值断言）。

---

## 9. 给其他 LLM 的最小上下文

可直接复制以下上下文给其他模型，避免重复误判：

```markdown
事实基线：
1) 当前策略默认版本是 STRATEGY_VERSION=v5。
2) v5 关键开关（soft veto / leg budget / continuous trail / regime-SL）已映射到 StrategyConfig 默认值。
3) 回测主数据链路已支持保留 mark/funding/oi 列，但依赖本地 parquet 里实际存在这些字段。
4) summary_json 真实字段是 total_return_pct / contract_pf / portfolio_pf。
5) 最新 v10 CSV 显示 E4_v10_full: IS PF=1.31, OOS Ret=+7.87%。
6) 最新 v5 生产对照 E1_v9_v5_prod: IS PF=0.90, OOS Ret=-3.83%。

请基于以上事实给出：
- 先做哪 3 个工程修复保证口径一致；
- 再做哪 2 个策略实验提高 OOS 稳定性；
- 每一步验收指标。
```

---

## 10. 复现命令

```bash
# v9 round3
python3 run_v9_ab_round3.py

# v9 full（v5生产对照）
python3 run_v9_full_backtest.py

# v10 验证
python3 run_v10_backtest.py

# 主回测（本地K线，不走API回退）
BACKTEST_DAILY_ALLOW_API_FALLBACK=0 python3 backtest_multi_tf_daily.py \
  --start 2025-01-01 --end 2026-01-31 --tag "codex-baseline"
```

---

## 11. 代码锚点索引

- `live_config.py`: 版本与参数默认
- `backtest_multi_tf_daily.py`: 回测编排与 `summary_json` 口径
- `optimize_six_book.py`: 策略主循环与 v9/v10 开关实现
- `kline_store.py`: 本地数据加载与列裁剪
- `multi_tf_daily_db.py`: 结果入库与快照透传
- `run_v9_ab_round3.py`: v9 A/B 实验矩阵
- `run_v10_backtest.py`: v10 改造实验矩阵
