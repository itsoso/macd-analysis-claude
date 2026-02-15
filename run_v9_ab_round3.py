#!/usr/bin/env python3
"""v9.0 A/B Round 3: Bug Fix 验证 + P21/P23 功能实验

修复:
  - F1: P18 use_regime_adaptive_fusion 默认回退为 False
  - F2: hard_stop_loss 使用 mark price (liq_high/liq_low)
  - F3: Funding UTC 锚定 (已由 Codex 实现)

新功能:
  - P23: 加权结构确认 (基于 Cohen's d 的 alpha_weight)
  - P21: Risk-per-trade (R) 仓位模型 (ATR-based)

实验设计:
  E0: v8.0+P20 基线 (含所有 bug fix)
  E1: E0 + P23 (加权确认)
  E2: E0 + P21 (R 仓位)
  E3: E0 + P23 + P21
  E4: E0 + P18-lite (neutral div_w=0.45) + P23
  E5: E0 + P18-lite + P23 + P21 (完整 v9 candidate)
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

# v8.0 基线参数 (含 P20 空头追踪收紧)
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
    # Bug fix: 确保 P18 默认关闭
    'use_regime_adaptive_fusion': False,
}

# P23 加权确认参数
P23_OVERRIDES = {
    'use_weighted_confirms': True,
    'wc_ma_sell_w': 1.5,
    'wc_cs_sell_w': 1.4,
    'wc_kdj_sell_w': 1.0,
    'wc_vp_sell_w': 0.6,
    'wc_bb_sell_w': 0.3,
    'wc_ma_buy_w': 1.5,
    'wc_cs_buy_w': 1.4,
    'wc_kdj_buy_w': 1.0,
    'wc_vp_buy_w': 0.6,
    'wc_bb_buy_w': 0.3,
    'wc_min_score': 2.0,
    'wc_conflict_penalty_scale': 0.5,
}

# P21 Risk-per-trade 参数
P21_OVERRIDES = {
    'use_risk_per_trade': True,
    'risk_per_trade_pct': 0.015,
    'risk_stop_mode': 'atr',
    'risk_atr_mult_short': 2.5,
    'risk_atr_mult_long': 2.0,
    'risk_max_margin_pct': 0.50,
    'risk_min_margin_pct': 0.05,
}

# P18 regime-adaptive 权重 (保守版)
P18_LITE = {
    'use_regime_adaptive_fusion': True,
    'regime_neutral_div_w': 0.45,
    'regime_neutral_ma_w': 0.25,
}

variants = {
    'E0_baseline_fixed': {},

    'E1_P23_weighted': {**P23_OVERRIDES},

    'E2_P21_risk_R': {**P21_OVERRIDES},

    'E3_P23_P21': {**P23_OVERRIDES, **P21_OVERRIDES},

    'E4_P18lite_P23': {**P18_LITE, **P23_OVERRIDES},

    'E5_full_v9': {**P18_LITE, **P23_OVERRIDES, **P21_OVERRIDES},
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

    # 计算 Calmar Ratio
    calmar = 0
    if result.get('max_drawdown', 0) != 0:
        calmar = result.get('strategy_return', 0) / abs(result.get('max_drawdown', 1))

    # 最差5笔交易影响
    pnls = sorted([float(t.get('pnl', 0)) for t in closes])
    worst_5_impact = sum(pnls[:5]) if len(pnls) >= 5 else sum(pnls)

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
        'calmar': calmar,
        'worst_5_impact': worst_5_impact,
        'regime_stats': dict(regime_stats),
    }


def main():
    print("=" * 130)
    print("  v9.0 A/B Round 3: Bug Fix 验证 + P21/P23 功能实验")
    print("  基线: v8.0 + P20 + 所有 bug fix (P18默认False, hard_stop mark price)")
    print("=" * 130)

    print("\n  加载 IS 数据 (2025-01~2026-01)...")
    is_data, is_signals, needed_tfs, primary_tf, decision_tfs, is_start, is_end = \
        prepare_data('ETHUSDT', '2025-01-01', '2026-01-31')
    print(f"  决策 TF: {decision_tfs}")
    print("\n  加载 OOS 数据 (2024-01~2024-12)...")
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
            'is_trades': is_r['total_trades'], 'is_calmar': is_r['calmar'],
            'is_worst5': is_r['worst_5_impact'],
            'oos_ret': oos_r['strategy_return'], 'oos_wr': oos_r['win_rate'],
            'oos_pf': oos_r['profit_factor'], 'oos_mdd': oos_r['max_drawdown'],
            'oos_trades': oos_r['total_trades'], 'oos_calmar': oos_r['calmar'],
            'oos_worst5': oos_r['worst_5_impact'],
            'is_regime': json.dumps(is_r['regime_stats'], default=str),
            'oos_regime': json.dumps(oos_r['regime_stats'], default=str),
        }
        results.append(row)

        print(f"IS: Ret={is_r['strategy_return']:+.1f}% WR={is_r['win_rate']:.1f}% "
              f"PF={is_r['profit_factor']:.2f} MDD={is_r['max_drawdown']:.1f}% "
              f"Calmar={is_r['calmar']:.2f} W5=${is_r['worst_5_impact']:+,.0f} T={is_r['total_trades']}")
        print(f"        OOS: Ret={oos_r['strategy_return']:+.1f}% WR={oos_r['win_rate']:.1f}% "
              f"PF={oos_r['profit_factor']:.2f} MDD={oos_r['max_drawdown']:.1f}% "
              f"Calmar={oos_r['calmar']:.2f} W5=${oos_r['worst_5_impact']:+,.0f} T={oos_r['total_trades']}")

        for period, label in [(is_r, 'IS'), (oos_r, 'OOS')]:
            for key in ['short|neutral', 'short|trend', 'short|high_vol',
                         'long|neutral', 'long|trend', 'long|high_vol']:
                rs = period['regime_stats'].get(key, {})
                if rs.get('n', 0) > 0:
                    rs_wr = rs['w'] / rs['n'] * 100
                    print(f"    {label} {key}: n={rs['n']} WR={rs_wr:.0f}% PnL=${rs['pnl']:+,.0f}")

    rdf = pd.DataFrame(results)
    base = rdf[rdf['variant'] == 'E0_baseline_fixed'].iloc[0]

    print(f"\n{'='*150}")
    print(f"  v9.0 Round 3 汇总 (相对基线 E0)")
    print(f"{'='*150}")
    print(f"  {'变体':<25} {'IS_Ret':>8} {'ΔIS':>7} {'IS_WR':>7} {'IS_PF':>7} {'MDD':>7} {'Calmar':>7} {'W5':>10} {'N':>4} | "
          f"{'OOS_Ret':>8} {'ΔOOS':>7} {'OOS_WR':>7} {'OOS_PF':>7} {'MDD':>7} {'Calmar':>7} {'W5':>10} {'N':>4}")
    print(f"  {'-'*148}")
    for _, row in rdf.iterrows():
        d_is = row['is_ret'] - base['is_ret']
        d_oos = row['oos_ret'] - base['oos_ret']
        # ★ = OOS PF 不退 + IS PF 不超过 5% 回退 + Calmar 不恶化
        marker = ''
        if (row['oos_pf'] >= base['oos_pf']
                and row['is_pf'] >= base['is_pf'] * 0.95
                and row['oos_calmar'] >= base['oos_calmar'] * 0.9):
            marker = ' ★'
        print(f"  {row['variant']:<25} {row['is_ret']:>+7.1f}% {d_is:>+6.1f}% {row['is_wr']:>6.1f}% "
              f"{row['is_pf']:>6.2f} {row['is_mdd']:>6.1f}% {row['is_calmar']:>6.2f} "
              f"${row['is_worst5']:>+9,.0f} {row['is_trades']:>4} | "
              f"{row['oos_ret']:>+7.1f}% {d_oos:>+6.1f}% {row['oos_wr']:>6.1f}% "
              f"{row['oos_pf']:>6.2f} {row['oos_mdd']:>6.1f}% {row['oos_calmar']:>6.2f} "
              f"${row['oos_worst5']:>+9,.0f} {row['oos_trades']:>4}{marker}")

    export_dir = 'data/backtests/v9_ab'
    os.makedirs(export_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(export_dir, f'v9_ab_r3_{ts}.csv')
    rdf.to_csv(csv_path, index=False)
    print(f"\n  结果已保存: {csv_path}")
    print("  Round 3 实验完成。")


if __name__ == '__main__':
    main()
