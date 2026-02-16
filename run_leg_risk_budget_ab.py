#!/usr/bin/env python3
"""Leg Risk Budget A/B 实验

目标: 验证 (regime × direction) 10维仓位预算乘数的最优配置。
      Leg risk budget 在 optimize_six_book.py 中通过 _leg_budget_mult() 应用于
      开仓保证金, 乘数范围 [0.0, 2.5], 默认 1.0 (全仓)。

背景:
  - P4/P16 Cohen's d 分析表明 neutral short alpha 极弱 (d≈-0.04)
  - B1b 实验证实完全禁止 neutral short 在 IS/OOS 均有提升
  - v5 生产配置已内置一组参数 (neutral_short=0.10, trend_long=1.20 等)
  - 本实验系统性测试多组配置, 提供数据驱动的推荐参数

实验设计:
  L0: 基线 (use_leg_risk_budget=False, 所有乘数 1.0)
  L1: v5 生产配置 (来自 live_config.py v5 参数)
  L2: 保守减仓 — 仅降低弱 alpha 空头 (neutral/high_vol short)
  L3: 极端方案 — neutral_short=0.00 (等价于 B1b 硬禁止)
  L4: 仅做多优化 — 减少高风险做多 leg (neutral/high_vol long)
  L5: 温和均衡版 — v5 值的 50% 缩放 (less extreme)
  L6: 趋势聚焦 — 趋势顺向加仓, 其余全部减仓
  L7: v5 + P23 加权确认组合 (组合效应验证)

IS: 2025-01 ~ 2026-01
OOS: 2024-01 ~ 2024-12
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

# v8.0+P20 基线 (与 run_v9_ab_round3.py 一致)
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


def _build_leg_budget(**overrides):
    """构建 leg risk budget 参数字典"""
    base = {
        'use_leg_risk_budget': True,
        'risk_budget_neutral_long': 1.0,
        'risk_budget_neutral_short': 1.0,
        'risk_budget_trend_long': 1.0,
        'risk_budget_trend_short': 1.0,
        'risk_budget_low_vol_trend_long': 1.0,
        'risk_budget_low_vol_trend_short': 1.0,
        'risk_budget_high_vol_long': 1.0,
        'risk_budget_high_vol_short': 1.0,
        'risk_budget_high_vol_choppy_long': 1.0,
        'risk_budget_high_vol_choppy_short': 1.0,
    }
    base.update(overrides)
    return base


# ============================================================
#  实验变体配置
# ============================================================

# L0: 基线 — 无 leg budget
L0_NO_BUDGET = {'use_leg_risk_budget': False}

# L1: v5 生产配置 (来自 live_config.py _VERSIONS["v5"])
L1_V5_PROD = _build_leg_budget(
    risk_budget_neutral_short=0.10,
    risk_budget_neutral_long=0.30,
    risk_budget_high_vol_short=0.50,
    risk_budget_high_vol_long=0.50,
    risk_budget_high_vol_choppy_short=0.20,
    risk_budget_high_vol_choppy_long=0.20,
    risk_budget_trend_short=0.60,
    risk_budget_trend_long=1.20,
    risk_budget_low_vol_trend_short=0.50,
    risk_budget_low_vol_trend_long=1.20,
)

# L2: 保守 — 仅降低 weak-alpha 空头, 做多不动
L2_CONSERVATIVE_SHORT = _build_leg_budget(
    risk_budget_neutral_short=0.20,
    risk_budget_high_vol_short=0.60,
    risk_budget_high_vol_choppy_short=0.30,
    risk_budget_trend_short=0.80,
    risk_budget_low_vol_trend_short=0.70,
)

# L3: 极端 — neutral_short=0 (等价于 B1b 硬禁止)
L3_KILL_NEUTRAL_SHORT = _build_leg_budget(
    risk_budget_neutral_short=0.00,
    risk_budget_high_vol_choppy_short=0.10,
)

# L4: 做多侧优化 — 降低高风险做多 (neutral/high_vol long)
L4_LONG_OPTIMIZE = _build_leg_budget(
    risk_budget_neutral_long=0.50,
    risk_budget_high_vol_long=0.60,
    risk_budget_high_vol_choppy_long=0.30,
)

# L5: 温和均衡 — v5 值的 50% 缩放 (向 1.0 收缩)
L5_MODERATE = _build_leg_budget(
    risk_budget_neutral_short=0.55,   # midpoint(0.10, 1.0)
    risk_budget_neutral_long=0.65,    # midpoint(0.30, 1.0)
    risk_budget_high_vol_short=0.75,  # midpoint(0.50, 1.0)
    risk_budget_high_vol_long=0.75,
    risk_budget_high_vol_choppy_short=0.60,
    risk_budget_high_vol_choppy_long=0.60,
    risk_budget_trend_short=0.80,
    risk_budget_trend_long=1.10,      # midpoint(1.20, 1.0)
    risk_budget_low_vol_trend_short=0.75,
    risk_budget_low_vol_trend_long=1.10,
)

# L6: 趋势聚焦 — 趋势顺向加仓, 其余大幅减仓
L6_TREND_FOCUS = _build_leg_budget(
    risk_budget_neutral_short=0.10,
    risk_budget_neutral_long=0.30,
    risk_budget_high_vol_short=0.40,
    risk_budget_high_vol_long=0.40,
    risk_budget_high_vol_choppy_short=0.10,
    risk_budget_high_vol_choppy_long=0.10,
    risk_budget_trend_short=0.50,
    risk_budget_trend_long=1.50,      # 趋势顺向大幅加仓
    risk_budget_low_vol_trend_short=0.40,
    risk_budget_low_vol_trend_long=1.50,
)

# L7: v5 + P23 加权确认 (组合效应)
L7_V5_PLUS_P23 = {
    **L1_V5_PROD,
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

variants = {
    'L0_no_budget': L0_NO_BUDGET,
    'L1_v5_prod': L1_V5_PROD,
    'L2_conservative_short': L2_CONSERVATIVE_SHORT,
    'L3_kill_neutral_short': L3_KILL_NEUTRAL_SHORT,
    'L4_long_optimize': L4_LONG_OPTIMIZE,
    'L5_moderate': L5_MODERATE,
    'L6_trend_focus': L6_TREND_FOCUS,
    'L7_v5_plus_p23': L7_V5_PLUS_P23,
}


# ============================================================
#  回测执行
# ============================================================

def run_backtest_with_trades(all_data, all_signals, needed_tfs, cfg, primary_tf,
                              decision_tfs, start_dt, end_dt):
    """运行回测并返回结果和详细统计"""
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
    partials = [t for t in trades if t.get('action') == 'PARTIAL_TP']

    n_trades = len(closes)
    wins = sum(1 for t in closes if float(t.get('pnl', 0)) > 0)
    wr = wins / n_trades * 100 if n_trades > 0 else 0
    gross_profit = sum(float(t.get('pnl', 0)) for t in closes if float(t.get('pnl', 0)) > 0)
    gross_loss = abs(sum(float(t.get('pnl', 0)) for t in closes if float(t.get('pnl', 0)) <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    calmar = 0
    mdd = result.get('max_drawdown', 0)
    if mdd != 0:
        calmar = result.get('strategy_return', 0) / abs(mdd)

    # 按 side+regime 统计
    regime_stats = defaultdict(lambda: {'n': 0, 'w': 0, 'pnl': 0.0, 'margin_sum': 0.0})
    for t in opens:
        side = t.get('direction', 'unknown')
        regime = t.get('regime_label', 'unknown')
        key = f"{side}|{regime}"
        regime_stats[key]['n'] += 1
        regime_stats[key]['margin_sum'] += float(t.get('margin', 0) or 0)

    for t in closes:
        side = t.get('direction', 'unknown')
        regime = t.get('regime_label', 'unknown')
        key = f"{side}|{regime}"
        pnl_val = float(t.get('pnl', 0))
        regime_stats[key]['pnl'] += pnl_val
        if pnl_val > 0:
            regime_stats[key]['w'] += 1

    # 最差5笔影响
    pnls = sorted([float(t.get('pnl', 0)) for t in closes])
    worst_5 = sum(pnls[:5]) if len(pnls) >= 5 else sum(pnls)

    return {
        'strategy_return': result.get('strategy_return', 0),
        'max_drawdown': mdd,
        'total_trades': n_trades,
        'win_rate': wr,
        'profit_factor': pf,
        'calmar': calmar,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'worst_5_impact': worst_5,
        'n_partials': len(partials),
        'regime_stats': dict(regime_stats),
    }


def print_regime_detail(r, label):
    """打印 regime 级别细节"""
    for key in sorted(r['regime_stats'].keys()):
        rs = r['regime_stats'][key]
        if rs['n'] > 0:
            rs_wr = rs['w'] / rs['n'] * 100
            avg_margin = rs['margin_sum'] / rs['n'] if rs['n'] > 0 else 0
            print(f"    {label} {key}: n={rs['n']} WR={rs_wr:.0f}% "
                  f"PnL=${rs['pnl']:+,.0f} AvgMargin=${avg_margin:,.0f}", flush=True)


def main():
    print("=" * 130)
    print("  Leg Risk Budget A/B 实验")
    print("  目标: 验证 (regime × direction) 10维仓位预算乘数的最优配置")
    print("  基线: v8.0+P20 (use_leg_risk_budget=False)")
    print("  变体: L1(v5生产) / L2(保守) / L3(极端) / L4(做多) / L5(温和) / L6(趋势) / L7(+P23)")
    print("=" * 130)

    # 加载数据
    print("\n  加载 IS 数据 (2025-01 ~ 2026-01) ...")
    is_data, is_signals, needed_tfs, primary_tf, decision_tfs, is_start, is_end = \
        prepare_data('ETHUSDT', '2025-01-01', '2026-01-31')
    print(f"  决策 TF: {decision_tfs}")

    print("  加载 OOS 数据 (2024-01 ~ 2024-12) ...")
    oos_data, oos_signals, _, _, _, oos_start, oos_end = \
        prepare_data('ETHUSDT', '2024-01-01', '2024-12-31')

    results = []
    for name, overrides in variants.items():
        cfg = load_base_config()
        cfg.update(V8_OVERRIDES)
        cfg.update(overrides)

        print(f"\n  [{name}]", end=' ', flush=True)

        # 打印关键 budget 参数
        if cfg.get('use_leg_risk_budget'):
            ns = cfg.get('risk_budget_neutral_short', 1.0)
            nl = cfg.get('risk_budget_neutral_long', 1.0)
            ts = cfg.get('risk_budget_trend_short', 1.0)
            tl = cfg.get('risk_budget_trend_long', 1.0)
            print(f"(NS={ns:.2f} NL={nl:.2f} TS={ts:.2f} TL={tl:.2f})", end=' ', flush=True)
        else:
            print("(budget=OFF)", end=' ', flush=True)

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
            'is_gp': is_r['gross_profit'],
            'is_gl': is_r['gross_loss'],
            'is_worst5': is_r['worst_5_impact'],
            'oos_ret': oos_r['strategy_return'],
            'oos_wr': oos_r['win_rate'],
            'oos_pf': oos_r['profit_factor'],
            'oos_mdd': oos_r['max_drawdown'],
            'oos_trades': oos_r['total_trades'],
            'oos_calmar': oos_r['calmar'],
            'oos_gp': oos_r['gross_profit'],
            'oos_gl': oos_r['gross_loss'],
            'oos_worst5': oos_r['worst_5_impact'],
            'is_regime_stats': json.dumps(is_r['regime_stats'], default=str),
            'oos_regime_stats': json.dumps(oos_r['regime_stats'], default=str),
        }
        results.append(row)

        print(f"\n    IS:  Ret={is_r['strategy_return']:+.1f}% WR={is_r['win_rate']:.1f}% "
              f"PF={is_r['profit_factor']:.2f} MDD={is_r['max_drawdown']:.1f}% "
              f"Calmar={is_r['calmar']:.2f} T={is_r['total_trades']}", flush=True)
        print(f"    OOS: Ret={oos_r['strategy_return']:+.1f}% WR={oos_r['win_rate']:.1f}% "
              f"PF={oos_r['profit_factor']:.2f} MDD={oos_r['max_drawdown']:.1f}% "
              f"Calmar={oos_r['calmar']:.2f} T={oos_r['total_trades']}", flush=True)

        # 打印 regime 明细
        print_regime_detail(is_r, 'IS ')
        print_regime_detail(oos_r, 'OOS')

    # ============================================================
    #  汇总表格
    # ============================================================
    rdf = pd.DataFrame(results)
    base = rdf[rdf['variant'] == 'L0_no_budget'].iloc[0]

    print(f"\n{'='*140}")
    print(f"  Leg Risk Budget A/B 实验汇总")
    print(f"{'='*140}")
    print(f"  {'变体':<28} {'IS_Ret':>8} {'ΔIS':>7} {'IS_WR':>7} {'IS_PF':>7} {'IS_MDD':>7} {'Calmar':>7} {'N':>4} | "
          f"{'OOS_Ret':>8} {'ΔOOS':>7} {'OOS_WR':>7} {'OOS_PF':>7} {'OOS_MDD':>7} {'Calmar':>7} {'N':>4}")
    print(f"  {'-'*138}")

    for _, row in rdf.iterrows():
        d_is = row['is_ret'] - base['is_ret']
        d_oos = row['oos_ret'] - base['oos_ret']
        # 标记 OOS 优于基线
        marker = ''
        if row['oos_pf'] > base['oos_pf'] and row['oos_ret'] > base['oos_ret']:
            marker = ' ★'
        elif row['oos_pf'] > base['oos_pf'] or row['oos_ret'] > base['oos_ret']:
            marker = ' △'
        print(f"  {row['variant']:<28} "
              f"{row['is_ret']:>+7.1f}% {d_is:>+6.1f}% {row['is_wr']:>6.1f}% {row['is_pf']:>6.2f} "
              f"{row['is_mdd']:>6.1f}% {row['is_calmar']:>6.2f} {row['is_trades']:>4} | "
              f"{row['oos_ret']:>+7.1f}% {d_oos:>+6.1f}% {row['oos_wr']:>6.1f}% {row['oos_pf']:>6.2f} "
              f"{row['oos_mdd']:>6.1f}% {row['oos_calmar']:>6.2f} {row['oos_trades']:>4}{marker}")

    # ============================================================
    #  Regime-level 对比 (关键决策依据)
    # ============================================================
    print(f"\n{'='*140}")
    print(f"  Regime-level PnL 对比 (OOS)")
    print(f"{'='*140}")
    target_keys = ['short|neutral', 'short|trend', 'short|high_vol',
                   'long|neutral', 'long|trend', 'long|low_vol_trend']
    header = f"  {'变体':<28}"
    for k in target_keys:
        header += f" {k:>18}"
    print(header)
    print(f"  {'-'*138}")

    for _, row in rdf.iterrows():
        oos_rs = json.loads(row['oos_regime_stats']) if isinstance(row['oos_regime_stats'], str) else row['oos_regime_stats']
        line = f"  {row['variant']:<28}"
        for k in target_keys:
            rs = oos_rs.get(k, {})
            n = rs.get('n', 0)
            pnl = rs.get('pnl', 0.0)
            if n > 0:
                line += f" n={n:>2} ${pnl:>+8,.0f}"
            else:
                line += f" {'—':>18}"
        print(line)

    # ============================================================
    #  保存结果
    # ============================================================
    export_dir = 'data/backtests/leg_budget_ab'
    os.makedirs(export_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(export_dir, f'leg_budget_ab_{ts}.csv')
    rdf.to_csv(csv_path, index=False)
    print(f"\n  结果已保存: {csv_path}")

    # ============================================================
    #  推荐方案
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  推荐方案分析")
    print(f"{'='*100}")

    # 排除基线
    candidates = rdf[rdf['variant'] != 'L0_no_budget'].copy()

    # 多维评分: OOS PF 权重50%, OOS Ret 权重30%, IS-OOS 一致性 权重20%
    if len(candidates) > 0:
        # 归一化各指标到 [0, 1]
        for col in ['oos_pf', 'oos_ret', 'oos_calmar']:
            _min = candidates[col].min()
            _max = candidates[col].max()
            _range = _max - _min if _max > _min else 1.0
            candidates[f'{col}_norm'] = (candidates[col] - _min) / _range

        # IS-OOS 一致性: |IS_PF - OOS_PF| 越小越好
        candidates['pf_consistency'] = abs(candidates['is_pf'] - candidates['oos_pf'])
        _min_c = candidates['pf_consistency'].min()
        _max_c = candidates['pf_consistency'].max()
        _range_c = _max_c - _min_c if _max_c > _min_c else 1.0
        candidates['consistency_norm'] = 1.0 - (candidates['pf_consistency'] - _min_c) / _range_c

        # 综合得分
        candidates['score'] = (
            0.40 * candidates['oos_pf_norm'] +
            0.25 * candidates['oos_ret_norm'] +
            0.20 * candidates['oos_calmar_norm'] +
            0.15 * candidates['consistency_norm']
        )

        best = candidates.loc[candidates['score'].idxmax()]
        print(f"\n  ★ 综合最优: {best['variant']}")
        print(f"    IS:  Ret={best['is_ret']:+.1f}% PF={best['is_pf']:.2f} MDD={best['is_mdd']:.1f}%")
        print(f"    OOS: Ret={best['oos_ret']:+.1f}% PF={best['oos_pf']:.2f} MDD={best['oos_mdd']:.1f}%")
        print(f"    Calmar(OOS)={best['oos_calmar']:.2f}")
        d_oos = best['oos_ret'] - float(base['oos_ret'])
        d_is = best['is_ret'] - float(base['is_ret'])
        print(f"    vs 基线: IS {d_is:+.1f}pp, OOS {d_oos:+.1f}pp")

        # PF 最佳
        best_pf = candidates.loc[candidates['oos_pf'].idxmax()]
        if best_pf['variant'] != best['variant']:
            print(f"\n  ★ OOS PF 最佳: {best_pf['variant']}")
            print(f"    OOS: Ret={best_pf['oos_ret']:+.1f}% PF={best_pf['oos_pf']:.2f} MDD={best_pf['oos_mdd']:.1f}%")

        # Ret 最佳
        best_ret = candidates.loc[candidates['oos_ret'].idxmax()]
        if best_ret['variant'] != best['variant']:
            print(f"\n  ★ OOS Ret 最佳: {best_ret['variant']}")
            print(f"    OOS: Ret={best_ret['oos_ret']:+.1f}% PF={best_ret['oos_pf']:.2f} MDD={best_ret['oos_mdd']:.1f}%")

        # 完整排名表
        candidates_sorted = candidates.sort_values('score', ascending=False)
        print(f"\n  排名 (综合得分):")
        for rank, (_, row) in enumerate(candidates_sorted.iterrows(), 1):
            print(f"    {rank}. {row['variant']:<28} Score={row['score']:.3f} "
                  f"OOS: Ret={row['oos_ret']:+.1f}% PF={row['oos_pf']:.2f} MDD={row['oos_mdd']:.1f}%")

        # 输出推荐参数
        best_name = best['variant']
        best_cfg = variants[best_name]
        print(f"\n  推荐参数 ({best_name}):")
        for k in sorted(best_cfg.keys()):
            if k.startswith('risk_budget_') or k == 'use_leg_risk_budget':
                print(f"    {k}: {best_cfg[k]}")

    # 保存推荐配置
    recommend_path = os.path.join(export_dir, f'recommended_config_{ts}.json')
    if len(candidates) > 0:
        best_name = candidates.loc[candidates['score'].idxmax(), 'variant']
        recommend = {
            'experiment': 'leg_risk_budget_ab',
            'timestamp': ts,
            'recommended_variant': best_name,
            'config': variants[best_name],
            'metrics': {
                'oos_return': float(best['oos_ret']),
                'oos_pf': float(best['oos_pf']),
                'oos_mdd': float(best['oos_mdd']),
                'oos_calmar': float(best['oos_calmar']),
                'is_return': float(best['is_ret']),
                'is_pf': float(best['is_pf']),
            },
        }
        with open(recommend_path, 'w') as f:
            json.dump(recommend, f, indent=2, ensure_ascii=False)
        print(f"\n  推荐配置已保存: {recommend_path}")

    print("\n  实验完成。")


if __name__ == '__main__':
    main()
