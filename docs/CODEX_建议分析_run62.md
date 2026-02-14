# Codex 建议分析：run#55～#67 与 run#62 可上线参数

## 1. 核实结果（已接受）

- 最新优化已落库，新增 run 至 **#67**。
- **关键组合有效**：confirm35x3 + block high_vol（SS≥35 需 3 项确认 + high_vol 下禁止 SPOT_SELL）。
- K 线走本地，回测日志显示「K线数据源: 仅本地」。

---

## 2. 本轮结论与当前代码对齐

### 2.1 回测结果摘要

| Run   | 描述              | 收益      | MDD    | pPF   |
|-------|-------------------|-----------|--------|-------|
| #55   | 基线 live-v5      | +822.69%  | -5.27% | 2.16  |
| **#62** | **short_sl=-0.16** | **+881.93%** | **-4.43%** | **2.17** |
| #64   | short_sl=-0.14 保守 | +867.19%  | -3.96% | 2.16  |
| #65   | short_sl=-0.12 过紧 | +779.33%  | -5.91% | 1.94  |
| #67   | spot_sell_cap 开   | +85.07%   | 略差   | 1.54  |

- **OOS(2024)**：run#54/#63/#66 基本一致（ret +95.98%, MDD -19.24%, pPF 1.45），gate_add / hard_stop 在该段不敏感。
- **spot_sell_cap**：run#67 显示 pPF 升但收益降，**不建议启用**。

### 2.2 与仓库配置的差异

当前 **live_config.py / backtest 默认** 与 Codex 建议对比如下：

| 参数 | 当前代码 | Codex 建议（run#62 主线） | 说明 |
|------|----------|---------------------------|------|
| short_sl | v4=-0.25, v2/v3=-0.18 | **-0.16**（激进）或 **-0.14**（保守） | 需改以对齐 run#62/#64 |
| use_spot_sell_confirm | False | **True** | 启用「高分确认」 |
| spot_sell_confirm_ss | 100 | **35** | confirm35x3 = SS≥35 需确认 |
| spot_sell_confirm_min | 3 | 3 | 已一致 |
| spot_sell_regime_block | '' | **'high_vol'** | 高波动期禁止 SPOT_SELL |
| use_spot_sell_cap | False | False | 保持关闭（run#67 不建议开） |
| regime_short_gate_add | 35 | 35 | 暂停再调，保持即可 |
| hard_stop_loss | -0.28 | -0.28 | 暂停再调，保持即可 |

- **结论**：若要采纳 Codex 主线，需在 **live_config**（及回测默认 config）中：  
  - 将 **short_sl** 收敛到 **-0.16** 或 **-0.14**；  
  - 开启 **use_spot_sell_confirm=True**，**spot_sell_confirm_ss=35**，**spot_sell_regime_block='high_vol'**。  
- 代码已支持上述所有键：`optimize_six_book.py` 中 `spot_sell_regime_block` 解析为集合，`regime_label in {'high_vol', ...}` 时禁止 SPOT_SELL；confirm 逻辑已存在。

---

## 3. 策略层面下一步（Codex 指引）

- **主线参数**：先收敛到 **short_sl=-0.16**（激进）或 **-0.14**（保守），其余保持 confirm35x3 + block high_vol。
- **gate_add / hard_stop**：暂停继续调，当前数据上为低效方向。
- **下一轮重点**：改为「按月滚动 OOS」（尤其 8–9 月 SPOT_SELL 误卖段），再决定是否引入更细的卖出确认。

---

## 4. run#62 可上线参数清单（live / backtest 同步）

以下为按 **run#62** 生成的、与当前代码一致的「可上线参数」表，便于 live_config 与 `backtest_multi_tf_daily._build_default_config()` 同步。

### 4.1 需修改的项（相对当前 v4 默认）

```text
# live_config.py — StrategyConfig 或 STRATEGY_PARAM_VERSIONS
short_sl                      = -0.16        # run#62 激进；保守用 -0.14
use_spot_sell_confirm          = True
spot_sell_confirm_ss          = 35          # confirm35x3
spot_sell_confirm_min          = 3
spot_sell_regime_block         = 'high_vol'

# 保持不变
use_spot_sell_cap             = False
regime_short_gate_add          = 35
hard_stop_loss                 = -0.28
regime_short_gate_regimes      = 'low_vol_trend'
```

### 4.2 回测默认 config 同步

`backtest_multi_tf_daily.py` 中 `_build_default_config()` 已从 `_LIVE_DEFAULT`（StrategyConfig）读入上述字段，因此**只需改 live_config**，回测即与实盘一致（无额外改 backtest 的必要，除非要固定写死 run#62 而不读 StrategyConfig）。

### 4.3 建议落地方式

- **方式 A**：在 `live_config.py` 中新增 **v5**（或改 v4），把 run#62 参数写进 `STRATEGY_PARAM_VERSIONS["v5"]`，并设 `_ACTIVE_VERSION = "v5"`；同时把 `use_spot_sell_confirm` / `spot_sell_confirm_ss` / `spot_sell_regime_block` 的**默认值**改为 True / 35 / `'high_vol'`（若当前为全局默认而非版本内）。
- **方式 B**：保持 v4，仅在 StrategyConfig 的**默认值**中把上述 5 项改为 run#62 建议值，不新增版本号。

两种方式二选一即可；改完后执行一次回测与一次实盘参数打印，确认与下表一致。

### 4.4 校验用快查表（run#62）

| 配置项 | 值 |
|--------|-----|
| short_sl | -0.16 |
| use_spot_sell_confirm | True |
| spot_sell_confirm_ss | 35 |
| spot_sell_confirm_min | 3 |
| spot_sell_regime_block | high_vol |
| use_spot_sell_cap | False |
| regime_short_gate_add | 35 |
| hard_stop_loss | -0.28 |

---

**文档约定**：run#55 为「live-v5」基线（Codex 命名），与仓库内 STRATEGY_VERSION 的 v4 可能对应同一组或相邻优化；以 short_sl 与 confirm/block 组合为准对齐即可。
