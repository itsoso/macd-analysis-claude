# 建议分析：按「开仓 regime」统计开空次数 / 胜率 / 平均盈亏

## 1. 建议内容

对每个 regime 统计：

- **开空次数**：在该 regime 下开空的笔数
- **胜率**：这些空单中平仓盈利的占比
- **平均盈亏**：这些空单的总 PnL / 笔数

```python
regime_stats = {
    'trend': {'count': 0, 'wins': 0, 'total_pnl': 0},
    'low_vol_trend': {...},
    'high_vol_choppy': {...},
    'high_vol': {...},
    'neutral': {...},
}
```

---

## 2. 当前实现 vs 建议

### 2.1 已有能力

- **regime 定义**：`optimize_six_book._compute_regime_controls()` 已产出 `regime_label`，取值包括 `trend` / `low_vol_trend` / `high_vol` / `high_vol_choppy` / `neutral`。
- **每笔交易带 regime**：`strategy_futures._record_trade()` 会把当前 bar 的 `_regime_label` 写入每笔 trade 的 `regime_label`。
- **分 regime 汇总**：`backtest_multi_tf_daily.py` 已有「分市场段统计」：按 **该笔交易发生时的 regime** 汇总 `trades` / `gross_profit` / `gross_loss` / `net_pnl`（及 pPF）。

### 2.2 与建议的差异

- **现有统计是按「平仓时 regime」**：  
  CLOSE_SHORT / LIQUIDATED 等记录的是**平仓 bar** 的 `regime_label`，不是**开仓 bar** 的 regime。
- **建议需要的是「开仓时 regime」**：  
  统计「在 trend 下开了多少次空」「这些空单的胜率与平均盈亏」，必须用**开空那一刻**的 regime 作为 key。

因此需要在**开空时**记下 regime，在**平空时**把该 regime 写入平仓记录，再按「开仓 regime」做统计。

---

## 3. 实现要点

### 3.1 数据流

1. **开空**：在 `open_short()` 时把当前 `_regime_label` 存到引擎上，例如 `eng._short_open_regime`。
2. **平空**：在 CLOSE_SHORT / LIQUIDATED（以及 short 的 PARTIAL_TP）写 trade 时，把 `open_regime` 写入该笔 trade（从 `eng._short_open_regime` 取）。
3. **全平后清空**：`close_short()` / 空头 LIQUIDATED 后把 `_short_open_regime` 置为 None；PARTIAL_TP 只平一部分时保留，直到该空仓全部平完再清空。

### 3.2 需要改动的文件

| 位置 | 改动 |
|------|------|
| **strategy_futures.py** | ① `open_short()`：`self._short_open_regime = getattr(self, '_regime_label', 'unknown')`；② `close_short()` 与空头 `LIQUIDATED` 分支：在 `_record_trade(..., extra={'open_regime': self._short_open_regime})`，随后 `self._short_open_regime = None`；③ 若有其他地方直接记「平空」trade，同样传入 `open_regime` 并在全平后清空。 |
| **optimize_six_book.py** | 对 short 的 PARTIAL_TP 调用 `eng._record_trade(...)` 时，在 `extra` 中传入 `open_regime=getattr(eng, '_short_open_regime', None)`；最后一笔 PARTIAL_TP 导致空仓归零时，在调用后设 `eng._short_open_regime = None`（若引擎未在内部清空）。 |
| **backtest_multi_tf_daily.py** | ① **开空次数**：遍历 trades，`action == 'OPEN_SHORT'` 时按 `t['regime_label']` 累加 `regime_stats[r]['open_count']`；② **胜率 / 平均盈亏**：对 `action in ('CLOSE_SHORT','LIQUIDATED')` 以及 short 的 PARTIAL_TP，若有 `t.get('open_regime')`，则按 `open_regime` 累加该笔 `pnl`、以及是否盈利（wins）；③ 输出表格：每 regime 的 开空次数、平仓笔数、胜率、总 PnL、平均 PnL。 |

### 3.3 多空对称（可选）

- 若希望「按开仓 regime 统计开多」：同样在引擎中增加 `_long_open_regime`，在 OPEN_LONG 时写入、在 CLOSE_LONG / LIQUIDATED(long) / PARTIAL_TP(long) 时带入 trade，再在 backtest 里按 `open_regime` 做多头的 count/wins/total_pnl 统计。

---

## 4. 建议结论

- **建议有价值**：能直接回答「在哪个 regime 开空最多、赚/亏多少、胜率如何」，便于调 gate_add、block high_vol、hard_stop 等。
- **与现有不重复**：现有「分市场段统计」是按平仓时 regime；本建议是按开仓时 regime，两者互补。
- **实现成本**：小：引擎多 1 个字段 + 开/平仓几处传参；回测层一段按 `open_regime` 的聚合与打印。

建议按上述要点实现「按开仓 regime 的开空次数 / 胜率 / 平均盈亏」；若需要，可再补「按开仓 regime 的开多」统计。
