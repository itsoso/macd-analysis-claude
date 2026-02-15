#!/usr/bin/env python3
"""v9.0 A/B Round 2: P18 保守版权重调整

Round 1 发现 P18 的 IS/OOS 严重分裂:
  - neutral div_w=0.25 太激进, DIV 在 2025 震荡市中仍有价值
  - 需要找到 IS 不回退 + OOS 改善的平衡点

本轮实验:
  E0: D3 作为新基线 (v8.0 + P20)
  E1: P18-lite neutral div_w=0.45 (轻度降权, 从70%→56%)
  E2: P18-mid  neutral div_w=0.35 (中度降权, 从70%→47%)
  E3: P18-lite + P24 (regime止损)
  E4: P18-lite + P24 + B1b (禁neutral空)
  E5: P18-mid + P24 + P20
  E6: P18-lite + P24 + P20 完整
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

V8_OVERRIDES = {
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
    # P20: 空头追踪收紧
    'continuous_trail_max_pb_short': 0.40,
}


def _make_p18_overrides(neutral_div_w, neutral_ma_w=0.25):
    """生成P18配置, 只调neutral权重(trend/high_vol保持默认)"""
    return {
        'use_regime_adaptive_fusion': True,
        'regime_neutral_div_w': neutral_div_w,
        'regime_neutral_ma_w': neutral_ma_w,
    }


variants = {
    'E0_v8_p20_baseline': {},

    'E1_p18_lite': _make_p18_overrides(neutral_div_w=0.45),

    'E2_p18_mid': _make_p18_overrides(neutral_div_w=0.35),

    'E3_p18_lite_p24': {
        **_make_p18_overrides(neutral_div_w=0.45),
        'use_regime_adaptive_sl': True,
        'regime_neutral_short_sl': -0.12,
    },

    'E4_p18_lite_p24_b1b': {
        **_make_p18_overrides(neutral_div_w=0.45),
        'use_regime_adaptive_sl': True,
        'regime_neutral_short_sl': -0.12,
        'regime_short_threshold': 'neutral:999',
    },

    'E5_p18_mid_p24_p20': {
        **_make_p18_overrides(neutral_div_w=0.35),
        'use_regime_adaptive_sl': True,
        'regime_neutral_short_sl': -0.12,
    },

    'E6_p18_lite_p24_p20_full': {
        **_make_p18_overrides(neutral_div_w=0.45),
        'use_regime_adaptive_sl': True,
        'regime_neutral_short_sl': -0.12,
        'regime_high_vol_short_sl': -0.15,
    },
}


def run_backtest_with_trades(all_data, all_signals, needed_tfs, cfg, primary_tf,
                              decision_tfs, start_dt, end_dt):
    tf_score_map = _build_tf_score_index(all_data, all_signals, needed_tfs, cfg)
    primary_df = all_data[primary_tf]
    result = run_strategy_multi_tf(
        primary_df=primary_df, tf_score_map=tf_score_map,
        decision_tfs=decision_tfs, config=cfg,
        primary_tf=primary_tf, trade_days=0,
        trade_start_dt=start_dt, trade_end_dt=end_dt,
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
    print("  v9.0 A/B Round 2: P18 保守版权重实验")
    print("  基线: v8.0 + P20 (空头追踪收紧)")
    print("=" * 120)

    print("  加载 IS 数据 (2025-01~2026-01)...")
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
            'is_ret': is_r['strategy_return'], 'is_wr': is_r['win_rate'],
            'is_pf': is_r['profit_factor'], 'is_mdd': is_r['max_drawdown'],
            'is_trades': is_r['total_trades'],
            'oos_ret': oos_r['strategy_return'], 'oos_wr': oos_r['win_rate'],
            'oos_pf': oos_r['profit_factor'], 'oos_mdd': oos_r['max_drawdown'],
            'oos_trades': oos_r['total_trades'],
            'is_regime': json.dumps(is_r['regime_stats'], default=str),
            'oos_regime': json.dumps(oos_r['regime_stats'], default=str),
        }
        results.append(row)
        print(f"IS: Ret={is_r['strategy_return']:+.1f}% WR={is_r['win_rate']:.1f}% PF={is_r['profit_factor']:.2f} "
              f"MDD={is_r['max_drawdown']:.1f}% T={is_r['total_trades']} | "
              f"OOS: Ret={oos_r['strategy_return']:+.1f}% WR={oos_r['win_rate']:.1f}% PF={oos_r['profit_factor']:.2f} "
              f"MDD={oos_r['max_drawdown']:.1f}% T={oos_r['total_trades']}", flush=True)

        for period, label in [(is_r, 'IS'), (oos_r, 'OOS')]:
            for key in ['short|neutral', 'short|trend', 'short|high_vol', 'long|neutral', 'long|trend']:
                rs = period['regime_stats'].get(key, {})
                if rs.get('n', 0) > 0:
                    rs_wr = rs['w'] / rs['n'] * 100
                    print(f"    {label} {key}: n={rs['n']} WR={rs_wr:.0f}% PnL=${rs['pnl']:+,.0f}", flush=True)

    rdf = pd.DataFrame(results)
    base = rdf[rdf['variant'] == 'E0_v8_p20_baseline'].iloc[0]

    print(f"\n{'='*130}")
    print(f"  v9.0 Round 2 汇总")
    print(f"{'='*130}")
    print(f"  {'变体':<30} {'IS_Ret':>8} {'ΔIS':>7} {'IS_WR':>7} {'IS_PF':>7} {'MDD':>7} {'N':>4} | "
          f"{'OOS_Ret':>8} {'ΔOOS':>7} {'OOS_WR':>7} {'OOS_PF':>7} {'MDD':>7} {'N':>4}")
    print(f"  {'-'*128}")
    for _, row in rdf.iterrows():
        d_is = row['is_ret'] - base['is_ret']
        d_oos = row['oos_ret'] - base['oos_ret']
        marker = ' ★' if row['oos_pf'] >= base['oos_pf'] and row['is_pf'] >= base['is_pf'] * 0.95 else ''
        print(f"  {row['variant']:<30} {row['is_ret']:>+7.1f}% {d_is:>+6.1f}% {row['is_wr']:>6.1f}% {row['is_pf']:>6.2f} "
              f"{row['is_mdd']:>6.1f}% {row['is_trades']:>4} | "
              f"{row['oos_ret']:>+7.1f}% {d_oos:>+6.1f}% {row['oos_wr']:>6.1f}% {row['oos_pf']:>6.2f} "
              f"{row['oos_mdd']:>6.1f}% {row['oos_trades']:>4}{marker}")

    export_dir = 'data/backtests/v9_ab'
    os.makedirs(export_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(export_dir, f'v9_ab_r2_{ts}.csv')
    rdf.to_csv(csv_path, index=False)
    print(f"\n  结果已保存: {csv_path}")
    print("  Round 2 实验完成。")


if __name__ == '__main__':
    main()
