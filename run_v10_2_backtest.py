#!/usr/bin/env python3
"""v10.2 Phase 2 验证回测: Leg Budget 5×2 + Regime Sigmoid + TP 禁用 + MAE

Phase 2 四项改造:
  1. MAE 追踪: trade record 加 min_pnl_r/max_pnl_r 字段
  2. Leg Budget 5×2: 差异化 regime×direction 仓位预算
  3. Regime Sigmoid: SL/杠杆/阈值连续 sigmoid 过渡 (替代硬切换)
  4. TP 禁用: trend/low_vol_trend 禁用 TP1/TP2, 仅用 P13 连续追踪

实验设计 (6 变体):
  E0: v10.1 (ATR-SL + P21)  — Phase 1 最佳变体作为基线
  E1: E0 + Leg Budget 5×2   — 差异化 regime 仓位控制
  E2: E0 + Regime Sigmoid    — SL/杠杆/阈值连续过渡
  E3: E0 + TP 禁用 (趋势)   — 趋势 regime 禁用 TP1/TP2
  E4: Phase 2 full           — Leg + Sigmoid + TP 全量
  E5: Phase 2 conservative   — E4 但 Leg Budget 更保守
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

# ── v8.0+P20 基线参数 (与 v10.1 回测一致) ──
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
    'continuous_trail_max_pb_short': 0.40,
    'use_regime_adaptive_fusion': False,
    'c6_div_weight': 0.70,
}

# ── v10.1 Phase 1 最佳 (E3: ATR-SL + P21) ──
V10_1_BEST = {
    'use_soft_veto': True,
    'soft_veto_steepness': 3.0,
    'soft_veto_midpoint': 1.0,
    'soft_struct_min_mult': 0.02,
    'use_leg_risk_budget': True,
    'risk_budget_neutral_short': 0.10,
    'c6_div_weight': 0.70,
    'use_atr_sl': True,
    'use_regime_adaptive_sl': False,
    'atr_sl_mult': 3.0,
    'atr_sl_floor': -0.25,
    'atr_sl_ceil': -0.06,
    'atr_sl_mult_neutral': 2.0,
    'atr_sl_mult_trend': 3.5,
    'atr_sl_mult_low_vol_trend': 3.0,
    'atr_sl_mult_high_vol': 2.5,
    'atr_sl_mult_high_vol_choppy': 2.0,
    'use_risk_per_trade': True,
    'risk_per_trade_pct': 0.025,
    'risk_stop_mode': 'atr',
    'risk_atr_mult_short': 3.0,
    'risk_atr_mult_long': 2.0,
    'risk_max_margin_pct': 0.40,
    'risk_min_margin_pct': 0.03,
}

# ── Phase 2 改造参数 ──
LEG_BUDGET_5x2 = {
    'risk_budget_neutral_short': 0.10,
    'risk_budget_neutral_long': 0.30,
    'risk_budget_high_vol_short': 0.50,
    'risk_budget_high_vol_long': 0.50,
    'risk_budget_high_vol_choppy_short': 0.20,
    'risk_budget_high_vol_choppy_long': 0.20,
    'risk_budget_trend_short': 0.60,
    'risk_budget_trend_long': 1.20,
    'risk_budget_low_vol_trend_short': 0.50,
    'risk_budget_low_vol_trend_long': 1.20,
}

LEG_BUDGET_CONSERVATIVE = {
    'risk_budget_neutral_short': 0.10,
    'risk_budget_neutral_long': 0.50,
    'risk_budget_high_vol_short': 0.60,
    'risk_budget_high_vol_long': 0.60,
    'risk_budget_high_vol_choppy_short': 0.30,
    'risk_budget_high_vol_choppy_long': 0.30,
    'risk_budget_trend_short': 0.70,
    'risk_budget_trend_long': 1.00,
    'risk_budget_low_vol_trend_short': 0.60,
    'risk_budget_low_vol_trend_long': 1.00,
}

REGIME_SIGMOID = {
    'use_regime_sigmoid': True,
}

TP_DISABLED_TREND = {
    'tp_disabled_regimes': ['trend', 'low_vol_trend'],
}

# 实验变体
variants = {
    'E0_v10.1_baseline':    {},
    'E1_+LegBudget5x2':    {**LEG_BUDGET_5x2},
    'E2_+RegimeSigmoid':   {**REGIME_SIGMOID},
    'E3_+TPdisabled':      {**TP_DISABLED_TREND},
    'E4_Phase2_full':      {**LEG_BUDGET_5x2, **REGIME_SIGMOID, **TP_DISABLED_TREND},
    'E5_Phase2_conserv':   {**LEG_BUDGET_CONSERVATIVE, **REGIME_SIGMOID, **TP_DISABLED_TREND},
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

    calmar = 0
    if result.get('max_drawdown', 0) != 0:
        calmar = result.get('strategy_return', 0) / abs(result.get('max_drawdown', 1))

    pnls = sorted([float(t.get('pnl', 0)) for t in closes])
    worst_5_impact = sum(pnls[:5]) if len(pnls) >= 5 else sum(pnls)

    # MAE/MFE 分析 (v10.2 新增)
    mae_list = [float(t.get('min_pnl_r', 0) or 0) for t in closes if t.get('min_pnl_r') is not None]
    mfe_list = [float(t.get('max_pnl_r', 0) or 0) for t in closes if t.get('max_pnl_r') is not None]

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

    fees_info = result.get('fees', {})
    funding_paid = fees_info.get('funding_paid', 0) if isinstance(fees_info, dict) else 0
    funding_received = fees_info.get('funding_received', 0) if isinstance(fees_info, dict) else 0

    return {
        'strategy_return': result.get('strategy_return', 0),
        'max_drawdown': result.get('max_drawdown', 0),
        'total_trades': n_trades,
        'win_rate': wr,
        'profit_factor': pf,
        'calmar': calmar,
        'worst_5_impact': worst_5_impact,
        'regime_stats': dict(regime_stats),
        'funding_paid': funding_paid,
        'funding_received': funding_received,
        'mae_stats': {
            'n': len(mae_list),
            'mean': float(np.mean(mae_list)) if mae_list else 0,
            'p10': float(np.percentile(mae_list, 10)) if mae_list else 0,
            'p25': float(np.percentile(mae_list, 25)) if mae_list else 0,
            'median': float(np.median(mae_list)) if mae_list else 0,
        },
        'mfe_stats': {
            'n': len(mfe_list),
            'mean': float(np.mean(mfe_list)) if mfe_list else 0,
            'median': float(np.median(mfe_list)) if mfe_list else 0,
        },
    }


def main():
    print("=" * 170)
    print("  v10.2 Phase 2 验证: Leg Budget 5×2 + Regime Sigmoid + TP 禁用 + MAE")
    print("  E0: v10.1基线 | E1: +Leg5×2 | E2: +Sigmoid | E3: +TP禁用")
    print("  E4: Phase2全量 | E5: Phase2保守")
    print("=" * 170)

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
        cfg.update(V10_1_BEST)
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
        }
        results.append(row)

        print(f"IS: Ret={is_r['strategy_return']:+.1f}% WR={is_r['win_rate']:.1f}% "
              f"PF={is_r['profit_factor']:.2f} MDD={is_r['max_drawdown']:.1f}% "
              f"Calmar={is_r['calmar']:.2f} W5=${is_r['worst_5_impact']:+,.0f} T={is_r['total_trades']}")
        print(f"        OOS: Ret={oos_r['strategy_return']:+.1f}% WR={oos_r['win_rate']:.1f}% "
              f"PF={oos_r['profit_factor']:.2f} MDD={oos_r['max_drawdown']:.1f}% "
              f"Calmar={oos_r['calmar']:.2f} W5=${oos_r['worst_5_impact']:+,.0f} T={oos_r['total_trades']}")

        # MAE 分析
        if is_r['mae_stats']['n'] > 0:
            mae = is_r['mae_stats']
            print(f"    IS MAE: n={mae['n']} mean={mae['mean']*100:.1f}% p10={mae['p10']*100:.1f}% "
                  f"p25={mae['p25']*100:.1f}% median={mae['median']*100:.1f}%")

        # Regime 细分
        for period, label in [(is_r, 'IS'), (oos_r, 'OOS')]:
            for key in ['short|neutral', 'short|trend', 'short|high_vol',
                         'long|neutral', 'long|trend', 'long|high_vol']:
                rs = period['regime_stats'].get(key, {})
                if rs.get('n', 0) > 0:
                    rs_wr = rs['w'] / rs['n'] * 100
                    print(f"    {label} {key}: n={rs['n']} WR={rs_wr:.0f}% PnL=${rs['pnl']:+,.0f}")

    rdf = pd.DataFrame(results)

    print(f"\n{'='*170}")
    print(f"  {'变体':<25s}  "
          f"{'IS Ret':>7s} {'IS WR':>6s} {'IS PF':>6s} {'IS MDD':>7s} {'IS W5':>10s} {'IS N':>5s}  |  "
          f"{'OOS Ret':>8s} {'OOS WR':>7s} {'OOS PF':>7s} {'OOS MDD':>8s} {'OOS Cal':>8s} {'OOS N':>6s}")
    print(f"  {'-'*162}")
    for _, r in rdf.iterrows():
        v = r['variant']
        print(f"  {v:<25s}  "
              f"{r['is_ret']:+7.1f}% {r['is_wr']:5.1f}% {r['is_pf']:6.2f} {r['is_mdd']:6.1f}% "
              f"${r['is_worst5']:>+9,.0f} {r['is_trades']:5d}  |  "
              f"{r['oos_ret']:+7.1f}% {r['oos_wr']:6.1f}% {r['oos_pf']:7.2f} {r['oos_mdd']:7.1f}% "
              f"{r['oos_calmar']:8.2f} {r['oos_trades']:5d}")

    # vs 基线比较
    base = rdf[rdf['variant'] == 'E0_v10.1_baseline'].iloc[0]
    print(f"\n  === vs E0 v10.1 基线差值 ===")
    for _, r in rdf.iterrows():
        if r['variant'] == 'E0_v10.1_baseline':
            continue
        d_ret = r['oos_ret'] - base['oos_ret']
        d_pf = r['oos_pf'] - base['oos_pf']
        d_mdd = r['oos_mdd'] - base['oos_mdd']
        d_cal = r['oos_calmar'] - base['oos_calmar']
        d_w5 = r['oos_worst5'] - base['oos_worst5']
        print(f"  {r['variant']:<25s} ΔRet={d_ret:+.1f}% ΔPF={d_pf:+.2f} "
              f"ΔMDD={d_mdd:+.1f}% ΔCalmar={d_cal:+.2f} ΔW5=${d_w5:+,.0f}")

    print(f"\n{'='*170}")
    print(f"  完成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*170}")


if __name__ == '__main__':
    main()
