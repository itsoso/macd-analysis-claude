#!/usr/bin/env python3
"""v9.0 综合 A/B 实验 (P17 口径修正后)

基线: v8.0 (B3+P13)  — 使用 24h 决策口径 (P17 修正)
变体:
  D0: v8.0 baseline (24h 口径)
  D1: P18 — Regime-Adaptive 六维融合权重
  D2: P24 — Regime-Adaptive 止损
  D3: P20 — Short 侧 P13 追踪收紧 (max_pb 0.60→0.40)
  D4: P25/B1b — 完全禁止 neutral 空单
  D5: P18+P24 组合 (路线A核心)
  D6: P18+P24+P20 组合 (路线A完整)
  D7: B1b+P20 组合 (路线B)
  D8: P18+P24+P20+B1b 终极组合
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

# v8.0 (B3+P13) 覆盖 — 这是当前线上版本
V8_OVERRIDES = {
    # B3 参数
    'regime_short_threshold': 'neutral:60',
    'short_conflict_regimes': 'trend,high_vol,neutral',
    'neutral_struct_discount_0': 0.0,
    'neutral_struct_discount_1': 0.05,
    'neutral_struct_discount_2': 0.15,
    'neutral_struct_discount_3': 0.50,
    'cooldown': 6,
    # P13 连续追踪止盈
    'use_continuous_trail': True,
    'continuous_trail_start_pnl': 0.05,
    'continuous_trail_max_pb': 0.60,
    'continuous_trail_min_pb': 0.30,
    # v6 基础参数
    'short_threshold': 40,
    'long_threshold': 25,
    'short_sl': -0.20,
    'short_tp': 0.60,
    'long_sl': -0.10,
    'long_tp': 0.40,
    'partial_tp_1': 0.15,
    'use_partial_tp_2': True,
    'short_max_hold': 48,
    'short_trail': 0.19,
    'long_trail': 0.12,
    'trail_pullback': 0.50,
}

# 实验变体配置
variants = {
    'D0_v8_baseline': {},

    'D1_p18_regime_adaptive_fusion': {
        'use_regime_adaptive_fusion': True,
    },

    'D2_p24_regime_adaptive_sl': {
        'use_regime_adaptive_sl': True,
        'regime_neutral_short_sl': -0.12,
        'regime_trend_short_sl': -0.20,
        'regime_high_vol_short_sl': -0.15,
    },

    'D3_p20_short_trail_tight': {
        'continuous_trail_max_pb_short': 0.40,  # 空单回撤容忍从 60%→40%
    },

    'D4_b1b_block_neutral_short': {
        'regime_short_threshold': 'neutral:999',  # 完全禁止 neutral 空单
    },

    'D5_p18_p24': {
        'use_regime_adaptive_fusion': True,
        'use_regime_adaptive_sl': True,
        'regime_neutral_short_sl': -0.12,
        'regime_trend_short_sl': -0.20,
        'regime_high_vol_short_sl': -0.15,
    },

    'D6_p18_p24_p20': {
        'use_regime_adaptive_fusion': True,
        'use_regime_adaptive_sl': True,
        'regime_neutral_short_sl': -0.12,
        'regime_trend_short_sl': -0.20,
        'regime_high_vol_short_sl': -0.15,
        'continuous_trail_max_pb_short': 0.40,
    },

    'D7_b1b_p20': {
        'regime_short_threshold': 'neutral:999',
        'continuous_trail_max_pb_short': 0.40,
    },

    'D8_p18_p24_p20_b1b': {
        'use_regime_adaptive_fusion': True,
        'use_regime_adaptive_sl': True,
        'regime_neutral_short_sl': -0.12,
        'regime_trend_short_sl': -0.20,
        'regime_high_vol_short_sl': -0.15,
        'continuous_trail_max_pb_short': 0.40,
        'regime_short_threshold': 'neutral:999',
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
    print("=" * 120)
    print("  v9.0 综合 A/B 实验 (P17 口径: 24h 决策TF)")
    print("  基线: v8.0 (B3+P13)")
    print("  变体: P18(regime权重) / P24(regime止损) / P20(short追踪收紧) / B1b(禁neutral空)")
    print("=" * 120)

    # 准备数据 — P17: 使用 24h 决策口径
    print("  加载 IS 数据 (2025-01~2026-01) [24h 口径]...")
    is_data, is_signals, needed_tfs, primary_tf, decision_tfs, is_start, is_end = \
        prepare_data('ETHUSDT', '2025-01-01', '2026-01-31')
    print(f"  决策 TF: {decision_tfs}")
    print("  加载 OOS 数据 (2024-01~2024-12)...")
    oos_data, oos_signals, _, _, _, oos_start, oos_end = \
        prepare_data('ETHUSDT', '2024-01-01', '2024-12-31')

    results = []
    for name, overrides in variants.items():
        cfg = load_base_config()
        cfg.update(V8_OVERRIDES)
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
        print(f"IS: Ret={is_r['strategy_return']:+.1f}% WR={is_r['win_rate']:.1f}% PF={is_r['profit_factor']:.2f} "
              f"MDD={is_r['max_drawdown']:.1f}% T={is_r['total_trades']} | "
              f"OOS: Ret={oos_r['strategy_return']:+.1f}% WR={oos_r['win_rate']:.1f}% PF={oos_r['profit_factor']:.2f} "
              f"MDD={oos_r['max_drawdown']:.1f}% T={oos_r['total_trades']}", flush=True)

        # 打印 regime 明细
        for period, label in [(is_r, 'IS'), (oos_r, 'OOS')]:
            for key in ['short|neutral', 'short|trend', 'short|high_vol', 'long|neutral', 'long|trend']:
                rs = period['regime_stats'].get(key, {})
                if rs.get('n', 0) > 0:
                    rs_wr = rs['w'] / rs['n'] * 100
                    print(f"    {label} {key}: n={rs['n']} WR={rs_wr:.0f}% PnL=${rs['pnl']:+,.0f}", flush=True)

    # 汇总表格
    rdf = pd.DataFrame(results)
    base = rdf[rdf['variant'] == 'D0_v8_baseline'].iloc[0]

    print(f"\n{'='*130}")
    print(f"  v9.0 A/B 实验汇总 (P17 口径: 24h)")
    print(f"{'='*130}")
    print(f"  {'变体':<30} {'IS_Ret':>8} {'ΔIS':>7} {'IS_WR':>7} {'IS_PF':>7} {'MDD':>7} {'N':>4} | "
          f"{'OOS_Ret':>8} {'ΔOOS':>7} {'OOS_WR':>7} {'OOS_PF':>7} {'MDD':>7} {'N':>4}")
    print(f"  {'-'*128}")
    for _, row in rdf.iterrows():
        d_is = row['is_ret'] - base['is_ret']
        d_oos = row['oos_ret'] - base['oos_ret']
        marker = ' ★' if row['oos_pf'] > base['oos_pf'] and row['oos_ret'] > base['oos_ret'] else ''
        print(f"  {row['variant']:<30} {row['is_ret']:>+7.1f}% {d_is:>+6.1f}% {row['is_wr']:>6.1f}% {row['is_pf']:>6.2f} "
              f"{row['is_mdd']:>6.1f}% {row['is_trades']:>4} | "
              f"{row['oos_ret']:>+7.1f}% {d_oos:>+6.1f}% {row['oos_wr']:>6.1f}% {row['oos_pf']:>6.2f} "
              f"{row['oos_mdd']:>6.1f}% {row['oos_trades']:>4}{marker}")

    # 保存
    export_dir = 'data/backtests/v9_ab'
    os.makedirs(export_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(export_dir, f'v9_ab_{ts}.csv')
    rdf.to_csv(csv_path, index=False)
    print(f"\n  结果已保存: {csv_path}")

    # 最佳方案推荐
    best_oos = rdf.loc[rdf['oos_pf'].idxmax()]
    print(f"\n  ★ OOS PF 最佳方案: {best_oos['variant']}")
    print(f"    IS: Ret={best_oos['is_ret']:+.1f}% WR={best_oos['is_wr']:.1f}% PF={best_oos['is_pf']:.2f}")
    print(f"    OOS: Ret={best_oos['oos_ret']:+.1f}% WR={best_oos['oos_wr']:.1f}% PF={best_oos['oos_pf']:.2f}")

    best_oos_ret = rdf.loc[rdf['oos_ret'].idxmax()]
    if best_oos_ret['variant'] != best_oos['variant']:
        print(f"\n  ★ OOS Ret 最佳方案: {best_oos_ret['variant']}")
        print(f"    IS: Ret={best_oos_ret['is_ret']:+.1f}% WR={best_oos_ret['is_wr']:.1f}% PF={best_oos_ret['is_pf']:.2f}")
        print(f"    OOS: Ret={best_oos_ret['oos_ret']:+.1f}% WR={best_oos_ret['oos_wr']:.1f}% PF={best_oos_ret['oos_pf']:.2f}")

    print("\n  实验完成。")


if __name__ == '__main__':
    main()
