#!/usr/bin/env python3
"""P25: MAE-driven 自适应止损 A/B 实验

设计思路:
  1. 先跑一次基线回测, 收集含 MAE 的交易记录
  2. 用 mae_calibrator 分析 per-regime MAE 分布, 得到 P75/P90/P95 止损
  3. 用这些校准值跑 MAE-driven 止损变体
  4. 对比基线/ATR-SL/MAE 各分位数的效果

变体:
  M0: 基线 (fixed SL: short=-0.20, long=-0.10)
  M1: ATR-SL 基线 (现有 atr_sl_mult=3.0)
  M2: MAE P75 (较紧, 快速止损)
  M3: MAE P90 (推荐, 平衡保护与呼吸空间)
  M4: MAE P95 (较宽, 给更多回撤空间)
  M5: MAE P90 + P21 risk-per-trade (组合: MAE 止损 → 仓位联动)
  M6: MAE P90 + L2 leg budget (与 A/B 验证过的 leg budget 组合)
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
from mae_calibrator import analyze_mae_from_trades

# v8.0+P20 基线
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
}

# L2 leg budget (A/B 验证最优)
L2_LEG_BUDGET = {
    'use_leg_risk_budget': True,
    'risk_budget_neutral_short': 0.20,
    'risk_budget_neutral_long': 1.00,
    'risk_budget_high_vol_short': 0.60,
    'risk_budget_high_vol_long': 1.00,
    'risk_budget_high_vol_choppy_short': 0.30,
    'risk_budget_high_vol_choppy_long': 1.00,
    'risk_budget_trend_short': 0.80,
    'risk_budget_trend_long': 1.00,
    'risk_budget_low_vol_trend_short': 0.70,
    'risk_budget_low_vol_trend_long': 1.00,
}

# P21 Risk-per-trade
P21_OVERRIDES = {
    'use_risk_per_trade': True,
    'risk_per_trade_pct': 0.015,
    'risk_stop_mode': 'atr',
    'risk_atr_mult_short': 2.5,
    'risk_atr_mult_long': 2.0,
    'risk_max_margin_pct': 0.50,
    'risk_min_margin_pct': 0.05,
}


def run_backtest_with_trades(all_data, all_signals, needed_tfs, cfg, primary_tf,
                              decision_tfs, start_dt, end_dt):
    """运行回测返回结果 + 完整交易记录"""
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

    n_trades = len(closes)
    wins = sum(1 for t in closes if float(t.get('pnl', 0)) > 0)
    wr = wins / n_trades * 100 if n_trades > 0 else 0
    gross_profit = sum(float(t.get('pnl', 0)) for t in closes if float(t.get('pnl', 0)) > 0)
    gross_loss = abs(sum(float(t.get('pnl', 0)) for t in closes if float(t.get('pnl', 0)) <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0
    mdd = result.get('max_drawdown', 0)
    calmar = result.get('strategy_return', 0) / abs(mdd) if mdd != 0 else 0

    # 计算平均止损距离 (从交易记录)
    sl_distances = []
    for t in trades:
        sl_r = t.get('sig_actual_sl_r')
        if sl_r is not None:
            sl_distances.append(abs(float(sl_r)))
    avg_sl_dist = float(np.mean(sl_distances)) if sl_distances else 0

    regime_stats = defaultdict(lambda: {'n': 0, 'w': 0, 'pnl': 0.0})
    for t in closes:
        side = t.get('direction', 'unknown')
        regime = t.get('regime_label', 'unknown')
        key = f"{side}|{regime}"
        pnl_val = float(t.get('pnl', 0))
        regime_stats[key]['n'] += 1
        regime_stats[key]['pnl'] += pnl_val
        if pnl_val > 0:
            regime_stats[key]['w'] += 1

    return {
        'strategy_return': result.get('strategy_return', 0),
        'max_drawdown': mdd,
        'total_trades': n_trades,
        'win_rate': wr,
        'profit_factor': pf,
        'calmar': calmar,
        'avg_sl_dist': avg_sl_dist,
        'regime_stats': dict(regime_stats),
        'trades': trades,
    }


def build_mae_sl_overrides(mae_analysis, quantile='p90'):
    """从 MAE 分析结果构建 mae_sl_* 配置覆盖"""
    overrides = {
        'use_mae_driven_sl': True,
        'mae_sl_quantile': quantile,
    }

    q_key = f'mae_{quantile}'
    for bucket_key, bucket_data in mae_analysis.get('buckets', {}).items():
        mae_val = bucket_data.get(q_key, bucket_data.get(f'mae_{quantile}'))
        if mae_val is None:
            # 回退: 尝试从 p50/p75/p90/p95 取
            for fallback_q in ['mae_p90', 'mae_p75', 'mae_p50']:
                mae_val = bucket_data.get(fallback_q)
                if mae_val is not None:
                    break
        if mae_val is not None and float(mae_val) > 0:
            config_key = f'mae_sl_{bucket_key}'
            overrides[config_key] = round(float(mae_val), 6)

    return overrides


def main():
    print("=" * 130)
    print("  P25: MAE-driven 自适应止损 A/B 实验")
    print("  阶段1: 基线回测 → MAE 校准 → 变体回测")
    print("=" * 130)

    # 加载数据
    print("\n  加载 IS 数据 (2025-01 ~ 2026-01) ...")
    is_data, is_signals, needed_tfs, primary_tf, decision_tfs, is_start, is_end = \
        prepare_data('ETHUSDT', '2025-01-01', '2026-01-31')
    print(f"  决策 TF: {decision_tfs}")

    print("  加载 OOS 数据 (2024-01 ~ 2024-12) ...")
    oos_data, oos_signals, _, _, _, oos_start, oos_end = \
        prepare_data('ETHUSDT', '2024-01-01', '2024-12-31')

    # ================================================================
    # 阶段1: 运行基线回测, 收集 MAE 数据
    # ================================================================
    print(f"\n{'─'*80}")
    print("  阶段1: 基线回测 + MAE 校准")
    print(f"{'─'*80}")

    base_cfg = load_base_config()
    base_cfg.update(V8_OVERRIDES)

    print("  运行 IS 基线回测 (收集 MAE)...", flush=True)
    is_base = run_backtest_with_trades(is_data, is_signals, needed_tfs, base_cfg,
                                        primary_tf, decision_tfs, is_start, is_end)
    print(f"  IS 基线: Ret={is_base['strategy_return']:+.1f}% T={is_base['total_trades']}")

    # MAE 校准 (用 IS 数据, 因为要从 IS 交易中学习止损参数)
    print("\n  MAE 校准 (从 IS 交易记录):")
    mae_analysis = analyze_mae_from_trades(is_base['trades'], verbose=True)

    if not mae_analysis.get('buckets'):
        print("\n  ⚠ MAE 校准失败 (无有效交易), 无法继续实验")
        return

    # 构建各分位数的覆盖参数
    mae_p75_overrides = build_mae_sl_overrides(mae_analysis, 'p75')
    mae_p90_overrides = build_mae_sl_overrides(mae_analysis, 'p90')
    mae_p95_overrides = build_mae_sl_overrides(mae_analysis, 'p95')

    print(f"\n  MAE P75 止损参数:")
    for k, v in sorted(mae_p75_overrides.items()):
        if k.startswith('mae_sl_') and k != 'mae_sl_quantile':
            print(f"    {k}: {v}")
    print(f"  MAE P90 止损参数:")
    for k, v in sorted(mae_p90_overrides.items()):
        if k.startswith('mae_sl_') and k != 'mae_sl_quantile':
            print(f"    {k}: {v}")

    # ================================================================
    # 阶段2: A/B 变体回测
    # ================================================================
    print(f"\n{'─'*80}")
    print("  阶段2: A/B 变体回测")
    print(f"{'─'*80}")

    variants = {
        'M0_fixed_sl': {},
        'M1_atr_sl': {
            'use_atr_sl': True,
            'atr_sl_mult': 3.0,
            'atr_sl_floor': -0.25,
            'atr_sl_ceil': -0.08,
        },
        'M2_mae_p75': mae_p75_overrides,
        'M3_mae_p90': mae_p90_overrides,
        'M4_mae_p95': mae_p95_overrides,
        'M5_mae_p90_p21': {**mae_p90_overrides, **P21_OVERRIDES},
        'M6_mae_p90_L2': {**mae_p90_overrides, **L2_LEG_BUDGET},
    }

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
            'is_calmar': is_r['calmar'],
            'is_avg_sl': is_r['avg_sl_dist'],
            'oos_ret': oos_r['strategy_return'],
            'oos_wr': oos_r['win_rate'],
            'oos_pf': oos_r['profit_factor'],
            'oos_mdd': oos_r['max_drawdown'],
            'oos_trades': oos_r['total_trades'],
            'oos_calmar': oos_r['calmar'],
            'oos_avg_sl': oos_r['avg_sl_dist'],
            'is_regime_stats': json.dumps(is_r['regime_stats'], default=str),
            'oos_regime_stats': json.dumps(oos_r['regime_stats'], default=str),
        }
        results.append(row)

        print(f"\n    IS:  Ret={is_r['strategy_return']:+.1f}% WR={is_r['win_rate']:.1f}% "
              f"PF={is_r['profit_factor']:.2f} MDD={is_r['max_drawdown']:.1f}% "
              f"AvgSL={is_r['avg_sl_dist']:.3f} T={is_r['total_trades']}", flush=True)
        print(f"    OOS: Ret={oos_r['strategy_return']:+.1f}% WR={oos_r['win_rate']:.1f}% "
              f"PF={oos_r['profit_factor']:.2f} MDD={oos_r['max_drawdown']:.1f}% "
              f"AvgSL={oos_r['avg_sl_dist']:.3f} T={oos_r['total_trades']}", flush=True)

    # ================================================================
    # 汇总
    # ================================================================
    rdf = pd.DataFrame(results)
    base = rdf[rdf['variant'] == 'M0_fixed_sl'].iloc[0]

    print(f"\n{'='*140}")
    print(f"  P25 MAE-driven 止损 A/B 实验汇总")
    print(f"{'='*140}")
    print(f"  {'变体':<22} {'IS_Ret':>8} {'ΔIS':>7} {'IS_WR':>7} {'IS_PF':>7} {'MDD':>7} {'AvgSL':>7} {'N':>4} | "
          f"{'OOS_Ret':>8} {'ΔOOS':>7} {'OOS_WR':>7} {'OOS_PF':>7} {'MDD':>7} {'AvgSL':>7} {'N':>4}")
    print(f"  {'-'*138}")

    for _, row in rdf.iterrows():
        d_is = row['is_ret'] - base['is_ret']
        d_oos = row['oos_ret'] - base['oos_ret']
        marker = ''
        if row['oos_pf'] > base['oos_pf'] and row['oos_ret'] > base['oos_ret']:
            marker = ' ★'
        elif row['oos_pf'] > base['oos_pf'] or row['oos_ret'] > base['oos_ret']:
            marker = ' △'
        print(f"  {row['variant']:<22} "
              f"{row['is_ret']:>+7.1f}% {d_is:>+6.1f}% {row['is_wr']:>6.1f}% {row['is_pf']:>6.2f} "
              f"{row['is_mdd']:>6.1f}% {row['is_avg_sl']:>6.3f} {row['is_trades']:>4} | "
              f"{row['oos_ret']:>+7.1f}% {d_oos:>+6.1f}% {row['oos_wr']:>6.1f}% {row['oos_pf']:>6.2f} "
              f"{row['oos_mdd']:>6.1f}% {row['oos_avg_sl']:>6.3f} {row['oos_trades']:>4}{marker}")

    # 保存
    export_dir = 'data/backtests/mae_sl_ab'
    os.makedirs(export_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(export_dir, f'mae_sl_ab_{ts}.csv')
    rdf.to_csv(csv_path, index=False)

    # 保存 MAE 校准结果
    mae_path = os.path.join(export_dir, f'mae_calibration_{ts}.json')
    mae_analysis['generated_at'] = datetime.now().isoformat()
    with open(mae_path, 'w') as f:
        json.dump(mae_analysis, f, indent=2, ensure_ascii=False)

    print(f"\n  结果已保存: {csv_path}")
    print(f"  MAE 校准: {mae_path}")

    # 推荐
    candidates = rdf[rdf['variant'] != 'M0_fixed_sl'].copy()
    if len(candidates) > 0:
        for col in ['oos_pf', 'oos_ret', 'oos_calmar']:
            _min, _max = candidates[col].min(), candidates[col].max()
            _range = _max - _min if _max > _min else 1.0
            candidates[f'{col}_norm'] = (candidates[col] - _min) / _range
        candidates['score'] = (
            0.40 * candidates['oos_pf_norm'] +
            0.35 * candidates['oos_ret_norm'] +
            0.25 * candidates['oos_calmar_norm']
        )
        best = candidates.loc[candidates['score'].idxmax()]
        print(f"\n  ★ 综合最优: {best['variant']}")
        print(f"    IS:  Ret={best['is_ret']:+.1f}% PF={best['is_pf']:.2f} MDD={best['is_mdd']:.1f}%")
        print(f"    OOS: Ret={best['oos_ret']:+.1f}% PF={best['oos_pf']:.2f} MDD={best['oos_mdd']:.1f}%")
        print(f"    AvgSL(IS)={best['is_avg_sl']:.3f} AvgSL(OOS)={best['oos_avg_sl']:.3f}")

    print("\n  实验完成。")


if __name__ == '__main__':
    main()
