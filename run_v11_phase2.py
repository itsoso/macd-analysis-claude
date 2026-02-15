#!/usr/bin/env python3
"""v11 Phase 2: Score Calibration + P18 Shrinkage + Rolling Regime

三项改造:
  2a. Score Calibration     — isotonic regression SS→p(win), E[R]
  2b. P18 Shrinkage         — w = (1-λ)*w_base + λ*w_regime
  2c. Rolling Percentile    — use_dynamic_regime_thresholds

实验设计:
  E0: v10.2 production baseline (+ Phase 1 soft antisqueeze)
  E1: E0 + Score Calibration
  E2: E0 + P18 Shrinkage (λ=0.3)
  E3: E0 + Rolling Percentile Regime
  E4: Phase 2 full (E1+E2+E3)
  E5: Phase 2 conserv (E2+E3 only, no score cal)
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
from score_calibrator import ScoreCalibrator

# ── 基线参数 (与 Phase 1 一致) ──
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

V10_2_PRODUCTION = {
    'use_soft_veto': True,
    'soft_veto_steepness': 3.0,
    'soft_veto_midpoint': 1.0,
    'soft_struct_min_mult': 0.02,
    'use_leg_risk_budget': True,
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
    'use_regime_sigmoid': True,
    'tp_disabled_regimes': ['trend', 'low_vol_trend'],
    # Phase 1b: Soft Anti-Squeeze (已验证有效)
    'use_soft_antisqueeze': True,
    'soft_antisqueeze_w_fz': 0.5,
    'soft_antisqueeze_w_oi': 0.3,
    'soft_antisqueeze_w_imb': 0.2,
    'soft_antisqueeze_midpoint': 1.5,
    'soft_antisqueeze_steepness': 2.0,
    'soft_antisqueeze_max_discount': 0.50,
}


def run_backtest(all_data, all_signals, needed_tfs, cfg,
                 primary_tf, decision_tfs, start_dt, end_dt):
    """运行回测并返回汇总指标 + trades"""
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

    n = len(closes)
    wins = sum(1 for t in closes if float(t.get('pnl', 0)) > 0)
    wr = wins / n * 100 if n > 0 else 0
    gp = sum(float(t.get('pnl', 0)) for t in closes if float(t.get('pnl', 0)) > 0)
    gl = abs(sum(float(t.get('pnl', 0)) for t in closes if float(t.get('pnl', 0)) <= 0))
    pf = gp / gl if gl > 0 else 0

    mdd = result.get('max_drawdown', 0)
    ret = result.get('strategy_return', 0)
    calmar = ret / abs(mdd) if mdd != 0 else 0

    pnls = sorted([float(t.get('pnl', 0)) for t in closes])
    w5 = sum(pnls[:5]) if len(pnls) >= 5 else sum(pnls)

    # regime 统计
    regime_stats = defaultdict(lambda: {'n': 0, 'w': 0, 'pnl': 0.0})
    for i, o in enumerate(opens):
        side = o.get('direction', 'unknown')
        regime = o.get('regime_label', 'unknown')
        matching = [c for c in closes if c.get('direction') == side]
        if i < len(matching):
            c = matching[i]
            pnl = float(c.get('pnl', 0))
            key = f"{side}|{regime}"
            regime_stats[key]['n'] += 1
            regime_stats[key]['w'] += int(pnl > 0)
            regime_stats[key]['pnl'] += pnl

    return {
        'trades': trades,
        'strategy_return': ret,
        'max_drawdown': mdd,
        'total_trades': n,
        'win_rate': wr,
        'profit_factor': pf,
        'calmar': calmar,
        'worst_5_impact': w5,
        'regime_stats': dict(regime_stats),
    }


def main():
    print("=" * 170)
    print("  v11 Phase 2: Score Calibration + P18 Shrinkage + Rolling Regime")
    print("=" * 170)

    print("\n  加载 IS 数据 (2025-01~2026-01)...")
    is_data, is_signals, needed_tfs, primary_tf, decision_tfs, is_start, is_end = \
        prepare_data('ETHUSDT', '2025-01-01', '2026-01-31')
    print(f"  决策 TF: {decision_tfs}")
    print("\n  加载 OOS 数据 (2024-01~2024-12)...")
    oos_data, oos_signals, _, _, _, oos_start, oos_end = \
        prepare_data('ETHUSDT', '2024-01-01', '2024-12-31')

    # ── Step 1: 运行基线收集交易数据用于校准 ──
    print("\n" + "=" * 100)
    print("  Step 1: 运行基线并收集校准数据")
    print("=" * 100)

    base_cfg = load_base_config()
    base_cfg.update(V8_OVERRIDES)
    base_cfg.update(V10_2_PRODUCTION)

    baseline = run_backtest(is_data, is_signals, needed_tfs, base_cfg,
                            primary_tf, decision_tfs, is_start, is_end)
    print(f"  基线 IS: Ret={baseline['strategy_return']:+.1f}% WR={baseline['win_rate']:.1f}% "
          f"PF={baseline['profit_factor']:.2f} T={baseline['total_trades']}")

    # ── Step 2: 用 IS 数据 fit Score Calibrator ──
    print("\n" + "=" * 100)
    print("  Step 2: Fit Score Calibrator (isotonic regression)")
    print("=" * 100)

    calibrator = ScoreCalibrator(min_samples=8)
    calibrator.fit(baseline['trades'], verbose=True)
    print(calibrator.summary())

    # 保存校准结果
    cal_path = os.path.join(os.path.dirname(__file__), 'score_calibration.json')
    calibrator.save(cal_path)
    print(f"\n  校准结果保存到: {cal_path}")

    # ── Step 3: 构建实验变体 ──
    # E1: Score Calibration
    SCORE_CAL = {
        'use_score_calibration': True,
        '_score_calibrator': calibrator,
        'score_cal_cost_estimate': 0.002,
        'score_cal_min_p_win': 0.40,
    }

    # E2: P18 Shrinkage
    P18_SHRINKAGE = {
        'use_regime_adaptive_fusion': True,
        'p18_shrinkage_max_lambda': 0.30,
        'p18_shrinkage_n_scale': 80.0,
    }

    # E3: Rolling Percentile Regime
    ROLLING_REGIME = {
        'use_dynamic_regime_thresholds': True,
        'dynamic_regime_lookback_bars': 2160,
        'dynamic_regime_vol_quantile': 0.80,
        'dynamic_regime_trend_quantile': 0.80,
    }

    variants = {
        'E0_baseline':       {},
        'E1_+ScoreCal':      {**SCORE_CAL},
        'E2_+P18Shrink':     {**P18_SHRINKAGE},
        'E3_+RollingRegime': {**ROLLING_REGIME},
        'E4_Phase2_full':    {**SCORE_CAL, **P18_SHRINKAGE, **ROLLING_REGIME},
        'E5_P2_conserv':     {**P18_SHRINKAGE, **ROLLING_REGIME},
    }

    # ── Step 4: 运行所有变体 ──
    print(f"\n{'='*170}")
    print(f"  Step 4: 运行 6 个变体 IS + OOS")
    print(f"{'='*170}")

    results = []
    for name, overrides in variants.items():
        cfg = load_base_config()
        cfg.update(V8_OVERRIDES)
        cfg.update(V10_2_PRODUCTION)
        cfg.update(overrides)

        print(f"\n  [{name}]", end=' ', flush=True)
        is_r = run_backtest(is_data, is_signals, needed_tfs, cfg,
                            primary_tf, decision_tfs, is_start, is_end)
        oos_r = run_backtest(oos_data, oos_signals, needed_tfs, cfg,
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

        # Regime 细分
        for period, label in [(is_r, 'IS'), (oos_r, 'OOS')]:
            for key in sorted(period['regime_stats'].keys()):
                rs = period['regime_stats'][key]
                if rs.get('n', 0) > 0:
                    rs_wr = rs['w'] / rs['n'] * 100
                    print(f"    {label} {key}: n={rs['n']} WR={rs_wr:.0f}% PnL=${rs['pnl']:+,.0f}")

    # ── Step 5: 汇总 ──
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
              f"${r['is_worst5']:>+9,.0f} {int(r['is_trades']):5d}  |  "
              f"{r['oos_ret']:+7.1f}% {r['oos_wr']:6.1f}% {r['oos_pf']:7.2f} {r['oos_mdd']:7.1f}% "
              f"{r['oos_calmar']:8.2f} {int(r['oos_trades']):5d}")

    base = rdf[rdf['variant'] == 'E0_baseline'].iloc[0]
    print(f"\n  === vs E0 基线差值 ===")
    for _, r in rdf.iterrows():
        if r['variant'] == 'E0_baseline':
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
