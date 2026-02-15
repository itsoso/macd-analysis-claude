#!/usr/bin/env python3
"""v10.1 Phase 1 验证回测: ATR-SL + P21绑定 + DIV权重下调 + funding_z Anti-Squeeze

Phase 1 四项改造:
  1. ATR-SL: 波动率驱动止损替代 P24 固定百分比, regime-specific ATR 乘数
  2. P21 Risk-per-trade: 仓位与止损绑定 (size = risk_budget / stop_distance)
  3. DIV 权重: 0.70→0.50 (消除最大负 alpha 锚)
  4. Anti-Squeeze: funding_z (z-score) 替代原始 funding_rate

实验设计 (7 变体):
  E0: v8 基线 (不含 v9/v10 改造)
  E1: v10.0 当前生产 (soft veto + leg budget + funding-in-pnl)
  E2: v10.0 + ATR-SL only (验证 ATR-SL 独立效果)
  E3: v10.0 + ATR-SL + P21 绑定 (验证 SL-仓位绑定)
  E4: v10.0 + DIV 0.50 only (验证 DIV 权重独立效果)
  E5: Phase 1 full (ATR-SL + P21 + DIV 0.50 + funding_z)
  E6: E5 + funding_z sensitivity (z=1.5 vs 2.0)
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

# ── v8.0+P20 基线 (与之前完全一致) ──
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
    'c6_div_weight': 0.70,  # 原始 DIV 权重 (v10.1 改为 0.50, 显式声明旧值)
    'use_atr_sl': False,    # 基线不用 ATR-SL
    'use_risk_per_trade': False,  # 基线不用 P21
}

# ── v10.0 当前生产配置 ──
V10_PRODUCTION = {
    'use_soft_veto': True,
    'soft_veto_steepness': 3.0,
    'soft_veto_midpoint': 1.0,
    'soft_struct_min_mult': 0.02,
    'use_leg_risk_budget': True,
    'risk_budget_neutral_short': 0.10,
    'c6_div_weight': 0.70,  # v10.0 仍用旧 DIV 权重
    'use_atr_sl': False,    # v10.0 不用 ATR-SL
    'use_risk_per_trade': False,  # v10.0 不用 P21
}

# ── Phase 1 改造参数 ──
ATR_SL = {
    'use_atr_sl': True,
    'use_regime_adaptive_sl': False,  # ATR-SL 取代 P24
    'atr_sl_mult': 3.0,
    'atr_sl_floor': -0.25,
    'atr_sl_ceil': -0.06,
    'atr_sl_mult_neutral': 2.0,
    'atr_sl_mult_trend': 3.5,
    'atr_sl_mult_low_vol_trend': 3.0,
    'atr_sl_mult_high_vol': 2.5,
    'atr_sl_mult_high_vol_choppy': 2.0,
}

P21_RISK_BOUND = {
    'use_risk_per_trade': True,
    'risk_per_trade_pct': 0.025,
    'risk_stop_mode': 'atr',
    'risk_atr_mult_short': 3.0,
    'risk_atr_mult_long': 2.0,
    'risk_max_margin_pct': 0.40,
    'risk_min_margin_pct': 0.03,
}

DIV_WEIGHT_50 = {
    'c6_div_weight': 0.50,
}

FUNDING_Z_ANTISQUEEZE = {
    'anti_squeeze_fz_threshold': 2.0,
}

FUNDING_Z_TIGHT = {
    'anti_squeeze_fz_threshold': 1.5,
}

# 实验变体
variants = {
    'E0_v8_baseline':     {},
    'E1_v10_prod':        {**V10_PRODUCTION},
    'E2_v10+ATR_SL':      {**V10_PRODUCTION, **ATR_SL},
    'E3_v10+ATR+P21':     {**V10_PRODUCTION, **ATR_SL, **P21_RISK_BOUND},
    'E4_v10+DIV50':       {**V10_PRODUCTION, **DIV_WEIGHT_50},
    'E5_phase1_full':     {**V10_PRODUCTION, **ATR_SL, **P21_RISK_BOUND, **DIV_WEIGHT_50, **FUNDING_Z_ANTISQUEEZE},
    'E6_phase1_fz15':     {**V10_PRODUCTION, **ATR_SL, **P21_RISK_BOUND, **DIV_WEIGHT_50, **FUNDING_Z_TIGHT},
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

    # OI 覆盖率审计 (从 primary_df.attrs 读取)
    perp_audit = primary_df.attrs.get('perp_data_audit', 'N/A')

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
        'perp_audit': perp_audit,
    }


def main():
    print("=" * 170)
    print("  v10.1 Phase 1 验证: ATR-SL + P21 Risk-Bound + DIV 50% + Funding Z-Score")
    print("  E0: v8基线 | E1: v10生产 | E2: +ATR-SL | E3: +P21绑定")
    print("  E4: +DIV50 | E5: Phase1全量 | E6: funding_z=1.5")
    print("=" * 170)

    print("\n  加载 IS 数据 (2025-01~2026-01)...")
    is_data, is_signals, needed_tfs, primary_tf, decision_tfs, is_start, is_end = \
        prepare_data('ETHUSDT', '2025-01-01', '2026-01-31')
    print(f"  决策 TF: {decision_tfs}")
    print("\n  加载 OOS 数据 (2024-01~2024-12)...")
    oos_data, oos_signals, _, _, _, oos_start, oos_end = \
        prepare_data('ETHUSDT', '2024-01-01', '2024-12-31')

    # 打印 OI 覆盖率审计 (首次运行时显示)
    _primary_df_is = is_data[primary_tf]
    _audit = _primary_df_is.attrs.get('perp_data_audit', None)
    if _audit:
        print(f"\n  === IS Period: Perp Data Coverage Audit ===")
        for line in _audit.split('\n'):
            print(f"  {line}")
    _primary_df_oos = oos_data[primary_tf]
    _audit_oos = _primary_df_oos.attrs.get('perp_data_audit', None)
    if _audit_oos:
        print(f"\n  === OOS Period: Perp Data Coverage Audit ===")
        for line in _audit_oos.split('\n'):
            print(f"  {line}")

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
            'is_funding_net': is_r['funding_received'] - is_r['funding_paid'],
            'oos_ret': oos_r['strategy_return'], 'oos_wr': oos_r['win_rate'],
            'oos_pf': oos_r['profit_factor'], 'oos_mdd': oos_r['max_drawdown'],
            'oos_trades': oos_r['total_trades'], 'oos_calmar': oos_r['calmar'],
            'oos_worst5': oos_r['worst_5_impact'],
            'oos_funding_net': oos_r['funding_received'] - oos_r['funding_paid'],
            'is_regime': json.dumps(is_r['regime_stats'], default=str),
            'oos_regime': json.dumps(oos_r['regime_stats'], default=str),
        }
        results.append(row)

        print(f"IS: Ret={is_r['strategy_return']:+.1f}% WR={is_r['win_rate']:.1f}% "
              f"PF={is_r['profit_factor']:.2f} MDD={is_r['max_drawdown']:.1f}% "
              f"Calmar={is_r['calmar']:.2f} W5=${is_r['worst_5_impact']:+,.0f} T={is_r['total_trades']}"
              f" F$={is_r['funding_received'] - is_r['funding_paid']:+,.0f}")
        print(f"        OOS: Ret={oos_r['strategy_return']:+.1f}% WR={oos_r['win_rate']:.1f}% "
              f"PF={oos_r['profit_factor']:.2f} MDD={oos_r['max_drawdown']:.1f}% "
              f"Calmar={oos_r['calmar']:.2f} W5=${oos_r['worst_5_impact']:+,.0f} T={oos_r['total_trades']}"
              f" F$={oos_r['funding_received'] - oos_r['funding_paid']:+,.0f}")

        for period, label in [(is_r, 'IS'), (oos_r, 'OOS')]:
            for key in ['short|neutral', 'short|trend', 'short|high_vol',
                         'long|neutral', 'long|trend', 'long|high_vol']:
                rs = period['regime_stats'].get(key, {})
                if rs.get('n', 0) > 0:
                    rs_wr = rs['w'] / rs['n'] * 100
                    print(f"    {label} {key}: n={rs['n']} WR={rs_wr:.0f}% PnL=${rs['pnl']:+,.0f}")

    rdf = pd.DataFrame(results)
    base = rdf[rdf['variant'] == 'E0_v8_baseline'].iloc[0]

    print(f"\n{'='*170}")
    print(f"  v10.1 Phase 1 汇总")
    print(f"{'='*170}")
    print(f"  {'变体':<24} {'IS_Ret':>8} {'ΔIS':>7} {'IS_WR':>7} {'IS_PF':>7} {'MDD':>7} {'Calmar':>7} {'W5':>10} {'N':>4} {'F$':>8} | "
          f"{'OOS_Ret':>8} {'ΔOOS':>7} {'OOS_WR':>7} {'OOS_PF':>7} {'MDD':>7} {'Calmar':>7} {'W5':>10} {'N':>4} {'F$':>8}")
    print(f"  {'-'*168}")
    for _, row in rdf.iterrows():
        d_is = row['is_ret'] - base['is_ret']
        d_oos = row['oos_ret'] - base['oos_ret']
        marker = ''
        if (row['oos_pf'] >= base['oos_pf'] * 0.95
                and row['is_pf'] >= base['is_pf'] * 0.90
                and row['oos_calmar'] >= base['oos_calmar'] * 0.9):
            marker = ' ★'
        print(f"  {row['variant']:<24} {row['is_ret']:>+7.1f}% {d_is:>+6.1f}% {row['is_wr']:>6.1f}% "
              f"{row['is_pf']:>6.2f} {row['is_mdd']:>6.1f}% {row['is_calmar']:>6.2f} "
              f"${row['is_worst5']:>+9,.0f} {row['is_trades']:>4} "
              f"${row['is_funding_net']:>+7,.0f} | "
              f"{row['oos_ret']:>+7.1f}% {d_oos:>+6.1f}% {row['oos_wr']:>6.1f}% "
              f"{row['oos_pf']:>6.2f} {row['oos_mdd']:>6.1f}% {row['oos_calmar']:>6.2f} "
              f"${row['oos_worst5']:>+9,.0f} {row['oos_trades']:>4} "
              f"${row['oos_funding_net']:>+7,.0f}{marker}")

    export_dir = 'data/backtests/v10_1'
    os.makedirs(export_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(export_dir, f'v10_1_phase1_{ts}.csv')
    rdf.to_csv(csv_path, index=False)
    print(f"\n  结果已保存: {csv_path}")
    print("  v10.1 Phase 1 验证完成。")


if __name__ == '__main__':
    main()
