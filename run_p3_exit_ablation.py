"""
P3: 退出路径消融实验
逐一关闭 6 种退出机制，评估各自对收益/WR/PF 的贡献。

退出机制列表:
1. 止损 (SL) — 不可关闭（安全基线），但可测试放宽
2. 追踪止盈 (Trail TP) — short_trail/long_trail 设为极大值关闭
3. 分段止盈 (Partial TP) — use_partial_tp=False
4. 反向平仓 (Reverse Close) — close_short_bs=999 / close_long_ss=999
5. 超时退出 (Timeout) — short_max_hold / long_max_hold 设为极大值
6. 硬断路器 (Hard Stop) — hard_stop_loss 设为极小值关闭

在 IS 和 OOS 上双重验证。
"""

import json
import os
from datetime import datetime
import pandas as pd

from run_p0_oos_validation import load_base_config
from run_p1_p2_sensitivity import prepare_data, run_single_backtest


def main():
    print("=" * 100)
    print("  P3: 退出路径消融实验")
    print("=" * 100)

    # 准备数据
    print("  加载数据...")
    is_data, is_signals, needed_tfs, primary_tf, decision_tfs, is_start, is_end = \
        prepare_data('ETHUSDT', '2025-01-01', '2026-01-31')
    oos_data, oos_signals, _, _, _, oos_start, oos_end = \
        prepare_data('ETHUSDT', '2024-01-01', '2024-12-31')

    # 定义消融实验
    ablations = {
        'baseline': {},  # v6.0 基准
        'no_trail_tp': {
            'short_trail': 9.99,  # 极大值 = 永不触发
            'long_trail': 9.99,
        },
        'no_partial_tp': {
            'use_partial_tp': False,
            'use_partial_tp_2': False,
            'use_partial_tp_v3': False,
        },
        'no_reverse_close': {
            'close_short_bs': 999,  # 极大值 = 永不反向平仓
            'close_long_ss': 999,
        },
        'no_timeout': {
            'short_max_hold': 9999,  # 极大值 = 不限时
            'long_max_hold': 9999,
        },
        'no_hard_stop': {
            'hard_stop_loss': -9.99,  # 极小值 = 永不触发
        },
        'no_sl': {
            'short_sl': -9.99,  # 极宽止损 = 几乎关闭
            'long_sl': -9.99,
        },
    }

    results = []
    for name, overrides in ablations.items():
        cfg = load_base_config()
        cfg.update(overrides)

        print(f"\n  [{name}]", end=' ')
        is_r = run_single_backtest(is_data, is_signals, needed_tfs, cfg,
                                    primary_tf, decision_tfs, is_start, is_end)
        oos_r = run_single_backtest(oos_data, oos_signals, needed_tfs, cfg,
                                     primary_tf, decision_tfs, oos_start, oos_end)

        row = {
            'ablation': name,
            'is_ret': is_r['strategy_return'],
            'is_wr': is_r['win_rate'],
            'is_pf': is_r['profit_factor'],
            'is_mdd': is_r['max_drawdown'],
            'is_trades': is_r['total_trades'],
            'oos_ret': oos_r['strategy_return'],
            'oos_wr': oos_r['win_rate'],
            'oos_pf': oos_r['profit_factor'],
            'oos_mdd': oos_r['max_drawdown'],
            'oos_trades': oos_r['total_trades'],
        }
        results.append(row)
        print(f"IS: Ret={is_r['strategy_return']:+.1f}% WR={is_r['win_rate']:.1f}% PF={is_r['profit_factor']:.2f} | "
              f"OOS: Ret={oos_r['strategy_return']:+.1f}% WR={oos_r['win_rate']:.1f}% PF={oos_r['profit_factor']:.2f}")

    # 汇总表格
    rdf = pd.DataFrame(results)
    base_is_ret = rdf[rdf['ablation'] == 'baseline']['is_ret'].values[0]
    base_oos_ret = rdf[rdf['ablation'] == 'baseline']['oos_ret'].values[0]
    base_is_wr = rdf[rdf['ablation'] == 'baseline']['is_wr'].values[0]

    print(f"\n{'='*100}")
    print(f"  消融实验汇总（差值 = 关闭该机制后 - 基准）")
    print(f"{'='*100}")
    print(f"  {'实验':<20} {'IS_Ret':>8} {'ΔRet':>7} {'IS_WR':>7} {'ΔWR':>6} {'IS_PF':>7} {'OOS_Ret':>8} {'ΔRet':>7} {'OOS_WR':>7}")
    print(f"  {'-'*85}")
    for _, row in rdf.iterrows():
        d_is_ret = row['is_ret'] - base_is_ret
        d_oos_ret = row['oos_ret'] - base_oos_ret
        d_is_wr = row['is_wr'] - base_is_wr
        marker = '★' if row['ablation'] == 'baseline' else ('↓' if d_is_ret < -0.5 else ('↑' if d_is_ret > 0.5 else ''))
        print(f"  {row['ablation']:<20} {row['is_ret']:>+7.1f}% {d_is_ret:>+6.1f}% {row['is_wr']:>6.1f}% {d_is_wr:>+5.1f} {row['is_pf']:>6.2f} {row['oos_ret']:>+7.1f}% {d_oos_ret:>+6.1f}% {row['oos_wr']:>6.1f}% {marker}")

    # 保存
    export_dir = 'data/backtests/p3_exit_ablation'
    os.makedirs(export_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(export_dir, f'p3_exit_ablation_{ts}.csv')
    rdf.to_csv(csv_path, index=False)
    print(f"\n  结果已保存: {csv_path}")


if __name__ == '__main__':
    main()
