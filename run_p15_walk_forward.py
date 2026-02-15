#!/usr/bin/env python3
"""P15: Walk-Forward 验证框架
将 2024-01 ~ 2026-01 划分为滚动窗口:
  Window 1: Train 2024-01~2024-06, Test 2024-07~2024-09
  Window 2: Train 2024-04~2024-09, Test 2024-10~2024-12
  Window 3: Train 2024-07~2024-12, Test 2025-01~2025-03
  Window 4: Train 2024-10~2025-03, Test 2025-04~2025-06
  Window 5: Train 2025-01~2025-06, Test 2025-07~2025-09
  Window 6: Train 2025-04~2025-09, Test 2025-10~2025-12

在每个窗口上运行 v7.0 B3 + P13 策略，观察时间稳定性。
"""
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from run_p0_oos_validation import V6_OVERRIDES, load_base_config
from run_p1_p2_sensitivity import prepare_data
from optimize_six_book import _build_tf_score_index, run_strategy_multi_tf

# v7.0 B3 + P13 策略配置
STRATEGY_CONFIG = {
    'regime_short_threshold': 'neutral:60',
    'short_conflict_regimes': 'trend,high_vol,neutral',
    'neutral_struct_discount_0': 0.0,
    'neutral_struct_discount_1': 0.05,
    'neutral_struct_discount_2': 0.15,
    'neutral_struct_discount_3': 0.50,
    'cooldown': 6,
    'use_continuous_trail': True,
    'continuous_trail_start_pnl': 0.05,
    'continuous_trail_max_pb': 0.60,
    'continuous_trail_min_pb': 0.30,
}

# Walk-forward windows: (data_start, train_start, train_end, test_start, test_end)
# data_start: enough lookback for signals
WINDOWS = [
    ('2023-10-01', '2024-01-01', '2024-06-30', '2024-07-01', '2024-09-30'),
    ('2024-01-01', '2024-04-01', '2024-09-30', '2024-10-01', '2024-12-31'),
    ('2024-04-01', '2024-07-01', '2024-12-31', '2025-01-01', '2025-03-31'),
    ('2024-07-01', '2024-10-01', '2025-03-31', '2025-04-01', '2025-06-30'),
    ('2024-10-01', '2025-01-01', '2025-06-30', '2025-07-01', '2025-09-30'),
    ('2025-01-01', '2025-04-01', '2025-09-30', '2025-10-01', '2025-12-31'),
]


def run_window(all_data, all_signals, needed_tfs, cfg, primary_tf,
               decision_tfs, start_dt, end_dt, label):
    tf_score_map = _build_tf_score_index(all_data, all_signals, needed_tfs, cfg)
    primary_df = all_data[primary_tf]
    result = run_strategy_multi_tf(
        primary_df=primary_df,
        tf_score_map=tf_score_map,
        decision_tfs=decision_tfs,
        config=cfg,
        primary_tf=primary_tf,
        trade_days=0,
        trade_start_dt=start_dt,
        trade_end_dt=end_dt,
    )

    trades = result.get('trades', [])
    closes = [t for t in trades if t.get('action', '').startswith('CLOSE_')]
    n = len(closes)
    wins = sum(1 for t in closes if float(t.get('pnl', 0)) > 0)
    wr = wins / n * 100 if n > 0 else 0
    gp = sum(float(t.get('pnl', 0)) for t in closes if float(t.get('pnl', 0)) > 0)
    gl = abs(sum(float(t.get('pnl', 0)) for t in closes if float(t.get('pnl', 0)) <= 0))
    pf = gp / gl if gl > 0 else 0

    return {
        'label': label,
        'strategy_return': result.get('strategy_return', 0),
        'max_drawdown': result.get('max_drawdown', 0),
        'total_trades': n,
        'win_rate': wr,
        'profit_factor': pf,
    }


def main():
    print("=" * 100)
    print("  P15: Walk-Forward 验证 (v7.0 B3 + P13)")
    print("=" * 100)

    results = []
    for i, (data_start, train_start, train_end, test_start, test_end) in enumerate(WINDOWS):
        win_label = f"W{i+1}"
        print(f"\n  ── {win_label}: Train {train_start}~{train_end} | Test {test_start}~{test_end} ──")

        # 加载覆盖整个窗口的数据
        full_end = test_end
        print(f"    加载数据 {data_start}~{full_end}...")
        try:
            all_data, all_signals, needed_tfs, primary_tf, decision_tfs, _, _ = \
                prepare_data('ETHUSDT', data_start, full_end)
        except Exception as e:
            print(f"    数据加载失败: {e}")
            continue

        cfg = load_base_config()
        cfg.update(STRATEGY_CONFIG)

        # Train 阶段
        train_r = run_window(all_data, all_signals, needed_tfs, cfg, primary_tf,
                             decision_tfs, train_start, train_end, f'{win_label}_train')
        print(f"    Train: Ret={train_r['strategy_return']:+.1f}% WR={train_r['win_rate']:.1f}% "
              f"PF={train_r['profit_factor']:.2f} MDD={train_r['max_drawdown']:.1f}% T={train_r['total_trades']}")

        # Test 阶段
        test_r = run_window(all_data, all_signals, needed_tfs, cfg, primary_tf,
                            decision_tfs, test_start, test_end, f'{win_label}_test')
        print(f"    Test:  Ret={test_r['strategy_return']:+.1f}% WR={test_r['win_rate']:.1f}% "
              f"PF={test_r['profit_factor']:.2f} MDD={test_r['max_drawdown']:.1f}% T={test_r['total_trades']}")

        results.append({
            'window': win_label,
            'train_period': f'{train_start}~{train_end}',
            'test_period': f'{test_start}~{test_end}',
            'train_ret': train_r['strategy_return'],
            'train_wr': train_r['win_rate'],
            'train_pf': train_r['profit_factor'],
            'train_trades': train_r['total_trades'],
            'test_ret': test_r['strategy_return'],
            'test_wr': test_r['win_rate'],
            'test_pf': test_r['profit_factor'],
            'test_mdd': test_r['max_drawdown'],
            'test_trades': test_r['total_trades'],
        })

    # 汇总
    if results:
        rdf = pd.DataFrame(results)
        print(f"\n{'='*110}")
        print(f"  Walk-Forward 汇总")
        print(f"{'='*110}")
        print(f"  {'Window':<8} {'Train':>22} {'Test':>22} {'Train_Ret':>10} {'Test_Ret':>10} {'Test_WR':>8} {'Test_PF':>8} {'Test_MDD':>9} {'Test_N':>7}")
        print(f"  {'-'*108}")
        for _, row in rdf.iterrows():
            print(f"  {row['window']:<8} {row['train_period']:>22} {row['test_period']:>22} "
                  f"{row['train_ret']:>+9.1f}% {row['test_ret']:>+9.1f}% {row['test_wr']:>7.1f}% "
                  f"{row['test_pf']:>7.2f} {row['test_mdd']:>8.1f}% {row['test_trades']:>6}")

        # 统计
        test_rets = rdf['test_ret'].values
        test_wrs = rdf['test_wr'].values
        test_pfs = rdf['test_pf'].values
        print(f"\n  Test窗口统计:")
        print(f"    Ret: mean={np.mean(test_rets):+.1f}% std={np.std(test_rets):.1f}% min={np.min(test_rets):+.1f}% max={np.max(test_rets):+.1f}%")
        print(f"    WR:  mean={np.mean(test_wrs):.1f}% std={np.std(test_wrs):.1f}% min={np.min(test_wrs):.1f}% max={np.max(test_wrs):.1f}%")
        print(f"    PF:  mean={np.mean(test_pfs):.2f} std={np.std(test_pfs):.2f}")
        print(f"    盈利窗口数: {sum(1 for r in test_rets if r > 0)}/{len(test_rets)}")

        export_dir = 'data/backtests/walk_forward'
        os.makedirs(export_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(export_dir, f'walk_forward_{ts}.csv')
        rdf.to_csv(csv_path, index=False)
        print(f"\n  结果已保存: {csv_path}")


if __name__ == '__main__':
    main()
