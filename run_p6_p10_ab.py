#!/usr/bin/env python3
"""P6-P10+P13 综合 A/B 实验
在 v7.0 (B3) 基线上测试:
  C0: v7.0 基线 (B3)
  C1: P7 — trend/high_vol short 24h方向门控
  C2: P8 — trend/high_vol short confirms>=3 硬门槛
  C3: P7+P8 组合
  C4: P6 — ghost cooldown (正式版)
  C5: P10 — fast-fail 快速亏损退出
  C6: P13 — 追踪止盈连续化
  C7: P7+P8+P6+P10 组合
  C8: C7+P13 终极组合
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
from run_p1_p2_sensitivity import prepare_data, run_single_backtest
from optimize_six_book import _build_tf_score_index, run_strategy_multi_tf

# v7.0 B3 覆盖
B3_OVERRIDES = {
    'regime_short_threshold': 'neutral:60',
    'short_conflict_regimes': 'trend,high_vol,neutral',
    'neutral_struct_discount_0': 0.0,
    'neutral_struct_discount_1': 0.05,
    'neutral_struct_discount_2': 0.15,
    'neutral_struct_discount_3': 0.50,
    'cooldown': 6,
}

variants = {
    'C0_baseline': {},

    'C1_p7_24h_gate': {
        'use_trend_short_large_tf_gate': True,
        'trend_short_large_tf_gate_regimes': 'trend,high_vol',
        'trend_short_large_tf_bs_ratio': 1.0,
    },

    'C2_p8_confirms3': {
        'use_trend_short_min_confirms': True,
        'trend_short_min_confirms_regimes': 'trend,high_vol',
        'trend_short_min_confirms_n': 3,
    },

    'C3_p7_p8': {
        'use_trend_short_large_tf_gate': True,
        'trend_short_large_tf_gate_regimes': 'trend,high_vol',
        'use_trend_short_min_confirms': True,
        'trend_short_min_confirms_regimes': 'trend,high_vol',
        'trend_short_min_confirms_n': 3,
    },

    'C4_p6_ghost_cd': {
        'use_ghost_cooldown': True,
        'ghost_cooldown_bars': 3,
    },

    'C5_p10_fast_fail': {
        'use_fast_fail_exit': True,
        'fast_fail_max_bars': 3,
        'fast_fail_loss_threshold': -0.05,
        'fast_fail_regimes': 'trend,high_vol',
    },

    'C6_p13_cont_trail': {
        'use_continuous_trail': True,
        'continuous_trail_start_pnl': 0.05,
        'continuous_trail_max_pb': 0.60,
        'continuous_trail_min_pb': 0.30,
    },

    'C7_p7p8p6p10': {
        'use_trend_short_large_tf_gate': True,
        'trend_short_large_tf_gate_regimes': 'trend,high_vol',
        'use_trend_short_min_confirms': True,
        'trend_short_min_confirms_regimes': 'trend,high_vol',
        'trend_short_min_confirms_n': 3,
        'use_ghost_cooldown': True,
        'ghost_cooldown_bars': 3,
        'use_fast_fail_exit': True,
        'fast_fail_max_bars': 3,
        'fast_fail_loss_threshold': -0.05,
        'fast_fail_regimes': 'trend,high_vol',
    },

    'C8_all': {
        'use_trend_short_large_tf_gate': True,
        'trend_short_large_tf_gate_regimes': 'trend,high_vol',
        'use_trend_short_min_confirms': True,
        'trend_short_min_confirms_regimes': 'trend,high_vol',
        'trend_short_min_confirms_n': 3,
        'use_ghost_cooldown': True,
        'ghost_cooldown_bars': 3,
        'use_fast_fail_exit': True,
        'fast_fail_max_bars': 3,
        'fast_fail_loss_threshold': -0.05,
        'fast_fail_regimes': 'trend,high_vol',
        'use_continuous_trail': True,
        'continuous_trail_start_pnl': 0.05,
        'continuous_trail_max_pb': 0.60,
        'continuous_trail_min_pb': 0.30,
    },
}


def run_backtest_with_trades(all_data, all_signals, needed_tfs, cfg, primary_tf,
                              decision_tfs, start_dt, end_dt):
    """运行回测并返回结果和交易明细"""
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
    opens = [t for t in trades if t.get('action', '').startswith('OPEN_')]

    n_trades = len(closes)
    wins = sum(1 for t in closes if float(t.get('pnl', 0)) > 0)
    wr = wins / n_trades * 100 if n_trades > 0 else 0
    gross_profit = sum(float(t.get('pnl', 0)) for t in closes if float(t.get('pnl', 0)) > 0)
    gross_loss = abs(sum(float(t.get('pnl', 0)) for t in closes if float(t.get('pnl', 0)) <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    # 按 side+regime 统计
    regime_stats = defaultdict(lambda: {'n': 0, 'w': 0, 'pnl': 0.0})
    for i, o in enumerate(opens):
        side = o.get('direction', 'unknown')
        regime = o.get('regime_label', 'unknown')
        matching_closes = [c for c in closes if c.get('direction') == side]
        if i < len(matching_closes):
            c = matching_closes[i]
            pnl = float(c.get('pnl', 0))
            key = f"{side}|{regime}"
            regime_stats[key]['n'] += 1
            regime_stats[key]['w'] += int(pnl > 0)
            regime_stats[key]['pnl'] += pnl

    return {
        'strategy_return': result.get('strategy_return', 0),
        'max_drawdown': result.get('max_drawdown', 0),
        'total_trades': n_trades,
        'win_rate': wr,
        'profit_factor': pf,
        'regime_stats': dict(regime_stats),
    }


def main():
    print("=" * 100)
    print("  P6-P10+P13 综合 A/B 实验 (基线: v7.0 B3)")
    print("=" * 100)

    # 准备数据
    print("  加载 IS 数据 (2025-01~2026-01)...")
    is_data, is_signals, needed_tfs, primary_tf, decision_tfs, is_start, is_end = \
        prepare_data('ETHUSDT', '2025-01-01', '2026-01-31')
    print("  加载 OOS 数据 (2024-01~2024-12)...")
    oos_data, oos_signals, _, _, _, oos_start, oos_end = \
        prepare_data('ETHUSDT', '2024-01-01', '2024-12-31')

    results = []
    for name, overrides in variants.items():
        cfg = load_base_config()
        cfg.update(B3_OVERRIDES)
        cfg.update(overrides)

        print(f"\n  [{name}]", end=' ', flush=True)
        is_r = run_backtest_with_trades(is_data, is_signals, needed_tfs, cfg,
                                         primary_tf, decision_tfs, is_start, is_end)
        oos_r = run_backtest_with_trades(oos_data, oos_signals, needed_tfs, cfg,
                                          primary_tf, decision_tfs, oos_start, oos_end)

        row = {
            'variant': name,
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
            'is_regime_stats': json.dumps(is_r['regime_stats'], default=str),
            'oos_regime_stats': json.dumps(oos_r['regime_stats'], default=str),
        }
        results.append(row)
        print(f"IS: Ret={is_r['strategy_return']:+.1f}% WR={is_r['win_rate']:.1f}% PF={is_r['profit_factor']:.2f} T={is_r['total_trades']} | "
              f"OOS: Ret={oos_r['strategy_return']:+.1f}% WR={oos_r['win_rate']:.1f}% PF={oos_r['profit_factor']:.2f} T={oos_r['total_trades']}", flush=True)

        # 打印 trend/high_vol short 明细
        for period, label in [(is_r, 'IS'), (oos_r, 'OOS')]:
            for key in ['short|trend', 'short|high_vol', 'short|neutral']:
                rs = period['regime_stats'].get(key, {})
                if rs.get('n', 0) > 0:
                    rs_wr = rs['w'] / rs['n'] * 100
                    print(f"    {label} {key}: n={rs['n']} WR={rs_wr:.0f}% PnL=${rs['pnl']:+,.0f}", flush=True)

    # 汇总表格
    rdf = pd.DataFrame(results)
    base = rdf[rdf['variant'] == 'C0_baseline'].iloc[0]

    print(f"\n{'='*110}")
    print(f"  P6-P10+P13 A/B 实验汇总")
    print(f"{'='*110}")
    print(f"  {'变体':<25} {'IS_Ret':>8} {'ΔIS':>7} {'IS_WR':>7} {'IS_PF':>7} {'N':>4} | {'OOS_Ret':>8} {'ΔOOS':>7} {'OOS_WR':>7} {'OOS_PF':>7} {'N':>4}")
    print(f"  {'-'*108}")
    for _, row in rdf.iterrows():
        d_is = row['is_ret'] - base['is_ret']
        d_oos = row['oos_ret'] - base['oos_ret']
        print(f"  {row['variant']:<25} {row['is_ret']:>+7.1f}% {d_is:>+6.1f}% {row['is_wr']:>6.1f}% {row['is_pf']:>6.2f} {row['is_trades']:>4} | "
              f"{row['oos_ret']:>+7.1f}% {d_oos:>+6.1f}% {row['oos_wr']:>6.1f}% {row['oos_pf']:>6.2f} {row['oos_trades']:>4}")

    # 保存
    export_dir = 'data/backtests/p6_p10_ab'
    os.makedirs(export_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(export_dir, f'p6_p10_ab_{ts}.csv')
    rdf.to_csv(csv_path, index=False)
    print(f"\n  结果已保存: {csv_path}")
    print("  实验完成。")


if __name__ == '__main__':
    main()
