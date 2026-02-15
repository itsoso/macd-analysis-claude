#!/usr/bin/env python3
"""v11 Phase 1: MAE 校准 + Anti-Squeeze Soft Penalty + 反手逻辑验证

三项改造:
  1a. MAE 分布驱动 ATR 乘数校准 — 用实际赢单 MAE P90 校准 regime-specific ATR 乘数
  1b. Anti-Squeeze soft penalty  — 硬 block → 连续 sigmoid penalty
  1c. 反手逻辑在回测中验证     — open_dominance_ratio 1.5→1.3

实验设计 (5 变体):
  E0: v10.2 production baseline       — 当前生产配置
  E1: E0 + MAE-calibrated ATR mults   — 数据驱动止损
  E2: E0 + Anti-Squeeze soft penalty  — 连续化微结构过滤
  E3: E0 + open_dominance 1.3         — 放宽开仓比率 (反手基础)
  E4: Phase 1 full (E1+E2+E3)         — 全量组合

运行方式:
  cd /path/to/project && .venv/bin/python3 run_v11_phase1.py
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

# ── v8.0+P20 基线参数 ──
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

# ── v10.2 production config (= V10_1_BEST + Phase 2) ──
V10_2_PRODUCTION = {
    # v10.1 Phase 1
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
    # v10.2 Phase 2
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
}


# ==================================================================
# 1a. MAE 校准: 收集 trade-level 数据并分析
# ==================================================================

def collect_detailed_trades(all_data, all_signals, needed_tfs, cfg,
                            primary_tf, decision_tfs, start_dt, end_dt):
    """运行回测并返回完整 trade list + 汇总指标"""
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

    calmar = 0
    mdd = result.get('max_drawdown', 0)
    ret = result.get('strategy_return', 0)
    if mdd != 0:
        calmar = ret / abs(mdd)

    pnls = sorted([float(t.get('pnl', 0)) for t in closes])
    w5 = sum(pnls[:5]) if len(pnls) >= 5 else sum(pnls)

    return {
        'trades': trades,
        'closes': closes,
        'opens': opens,
        'strategy_return': ret,
        'max_drawdown': mdd,
        'total_trades': n,
        'win_rate': wr,
        'profit_factor': pf,
        'calmar': calmar,
        'worst_5_impact': w5,
    }


def analyze_mae_by_regime(trades):
    """按 regime x direction 分析 MAE 分布，输出校准建议

    对每个 leg:
      - 赢单的 MAE P75/P90/P95
      - 赢单的 avg ATR% at entry
      - 推荐 ATR mult = abs(MAE_P90) / avg_atr_pct
    """
    closes = [t for t in trades if t.get('action', '').startswith('CLOSE_')]

    # 匹配 open-close 对
    open_map = {}  # direction -> list of opens
    for t in trades:
        if t.get('action', '').startswith('OPEN_'):
            d = t.get('direction', 'unknown')
            if d not in open_map:
                open_map[d] = []
            open_map[d].append(t)

    close_idx = defaultdict(int)
    paired = []
    for c in closes:
        d = c.get('direction', 'unknown')
        idx = close_idx[d]
        opens_for_dir = open_map.get(d, [])
        if idx < len(opens_for_dir):
            o = opens_for_dir[idx]
            paired.append({
                'direction': d,
                'regime': o.get('regime_label', 'unknown'),
                'pnl': float(c.get('pnl', 0)),
                'min_pnl_r': float(c.get('min_pnl_r', 0) or 0),
                'max_pnl_r': float(c.get('max_pnl_r', 0) or 0),
                'atr_pct': float(o.get('atr_pct', 0) or 0),
                'ss': float(o.get('ss', 0) or 0),
                'bs': float(o.get('bs', 0) or 0),
            })
            close_idx[d] += 1

    if not paired:
        print("  [MAE 分析] 无配对交易数据")
        return {}

    df = pd.DataFrame(paired)
    df['win'] = df['pnl'] > 0
    df['mae_pct'] = df['min_pnl_r'].abs()  # MAE 转为正数 (偏移幅度)

    print(f"\n{'='*100}")
    print(f"  MAE/MFE 校准分析 ({len(df)} 笔配对交易)")
    print(f"{'='*100}")

    calibration = {}

    for (regime, direction), group in df.groupby(['regime', 'direction']):
        leg = f"{direction}|{regime}"
        n_total = len(group)
        n_wins = group['win'].sum()
        n_losses = n_total - n_wins

        wins = group[group['win']]
        losses = group[~group['win']]

        print(f"\n  [{leg}] 总={n_total} 赢={n_wins} 输={n_losses} "
              f"WR={n_wins/n_total*100:.0f}%")

        if len(wins) >= 3:
            mae_vals = wins['mae_pct'].values
            print(f"    赢单 MAE:  mean={np.mean(mae_vals)*100:.1f}% "
                  f"P50={np.percentile(mae_vals, 50)*100:.1f}% "
                  f"P75={np.percentile(mae_vals, 75)*100:.1f}% "
                  f"P90={np.percentile(mae_vals, 90)*100:.1f}% "
                  f"P95={np.percentile(mae_vals, 95)*100:.1f}%")

            mfe_vals = wins['max_pnl_r'].values
            print(f"    赢单 MFE:  mean={np.mean(mfe_vals)*100:.1f}% "
                  f"P50={np.percentile(mfe_vals, 50)*100:.1f}% "
                  f"P75={np.percentile(mfe_vals, 75)*100:.1f}%")

            # ATR% at entry
            atr_vals = wins['atr_pct'].values
            avg_atr = np.mean(atr_vals) if len(atr_vals) > 0 else 0

            if avg_atr > 0:
                mae_p90 = np.percentile(mae_vals, 90)
                recommended_mult = mae_p90 / avg_atr
                print(f"    入场 ATR%: mean={avg_atr*100:.2f}%")
                print(f"    → 推荐 ATR mult = MAE_P90 / avg_ATR = "
                      f"{mae_p90*100:.1f}% / {avg_atr*100:.2f}% = {recommended_mult:.1f}")

                calibration[leg] = {
                    'n_wins': int(n_wins),
                    'n_total': int(n_total),
                    'mae_p90': float(mae_p90),
                    'avg_atr': float(avg_atr),
                    'recommended_mult': float(recommended_mult),
                    'current_mult': _get_current_mult(regime),
                }
            else:
                print(f"    入场 ATR%: 无数据")

        if len(losses) >= 3:
            loss_mae = losses['mae_pct'].values
            print(f"    输单 MAE:  mean={np.mean(loss_mae)*100:.1f}% "
                  f"P50={np.percentile(loss_mae, 50)*100:.1f}% "
                  f"P75={np.percentile(loss_mae, 75)*100:.1f}%")

    # 汇总推荐
    print(f"\n{'='*100}")
    print(f"  ATR 乘数校准推荐:")
    print(f"  {'Leg':<25s} {'当前':>6s} {'推荐':>6s} {'样本':>6s} {'方向':>8s}")
    print(f"  {'-'*60}")
    for leg, cal in sorted(calibration.items()):
        cur = cal['current_mult']
        rec = cal['recommended_mult']
        # 保守调整: 推荐值向当前值 shrink 30%
        adj = cur * 0.3 + rec * 0.7
        adj = max(1.5, min(5.0, adj))  # 限制范围
        delta = "↑" if adj > cur else "↓" if adj < cur else "="
        print(f"  {leg:<25s} {cur:6.1f} {adj:6.1f} n={cal['n_wins']:>3d} "
              f"  {delta}")
        calibration[leg]['adjusted_mult'] = float(adj)
    print(f"{'='*100}")

    return calibration


def _get_current_mult(regime):
    """返回当前 ATR 乘数"""
    mults = {
        'neutral': 2.0,
        'trend': 3.5,
        'low_vol_trend': 3.0,
        'high_vol': 2.5,
        'high_vol_choppy': 2.0,
    }
    return mults.get(regime, 3.0)


# ==================================================================
# 回测运行 + 结果汇总
# ==================================================================

def run_and_summarize(all_data, all_signals, needed_tfs, cfg,
                      primary_tf, decision_tfs, start_dt, end_dt):
    """运行回测并返回汇总指标 (不含完整 trades)"""
    r = collect_detailed_trades(all_data, all_signals, needed_tfs, cfg,
                                primary_tf, decision_tfs, start_dt, end_dt)

    # regime 统计
    regime_stats = defaultdict(lambda: {'n': 0, 'w': 0, 'pnl': 0.0})
    for i, o in enumerate(r['opens']):
        side = o.get('direction', 'unknown')
        regime = o.get('regime_label', 'unknown')
        matching = [c for c in r['closes'] if c.get('direction') == side]
        if i < len(matching):
            c = matching[i]
            pnl = float(c.get('pnl', 0))
            key = f"{side}|{regime}"
            regime_stats[key]['n'] += 1
            regime_stats[key]['w'] += int(pnl > 0)
            regime_stats[key]['pnl'] += pnl

    mae_list = [float(t.get('min_pnl_r', 0) or 0) for t in r['closes']
                if t.get('min_pnl_r') is not None]

    return {
        'strategy_return': r['strategy_return'],
        'max_drawdown': r['max_drawdown'],
        'total_trades': r['total_trades'],
        'win_rate': r['win_rate'],
        'profit_factor': r['profit_factor'],
        'calmar': r['calmar'],
        'worst_5_impact': r['worst_5_impact'],
        'regime_stats': dict(regime_stats),
        'mae_mean': float(np.mean(mae_list)) if mae_list else 0,
        'mae_p10': float(np.percentile(mae_list, 10)) if mae_list else 0,
    }


def main():
    print("=" * 170)
    print("  v11 Phase 1: MAE 校准 + Anti-Squeeze Soft + 反手验证")
    print("  E0: v10.2生产 | E1: +MAE校准ATR | E2: +SoftSqueeze | E3: +OpenDom1.3 | E4: Phase1全量")
    print("=" * 170)

    # ── 加载数据 ──
    print("\n  加载 IS 数据 (2025-01~2026-01)...")
    is_data, is_signals, needed_tfs, primary_tf, decision_tfs, is_start, is_end = \
        prepare_data('ETHUSDT', '2025-01-01', '2026-01-31')
    print(f"  决策 TF: {decision_tfs}")
    print("\n  加载 OOS 数据 (2024-01~2024-12)...")
    oos_data, oos_signals, _, _, _, oos_start, oos_end = \
        prepare_data('ETHUSDT', '2024-01-01', '2024-12-31')

    # ── Step 1: 运行基线并收集 MAE 数据 ──
    print("\n" + "=" * 100)
    print("  Step 1: 运行 v10.2 基线并收集 MAE 数据")
    print("=" * 100)

    base_cfg = load_base_config()
    base_cfg.update(V8_OVERRIDES)
    base_cfg.update(V10_2_PRODUCTION)

    baseline_result = collect_detailed_trades(
        is_data, is_signals, needed_tfs, base_cfg,
        primary_tf, decision_tfs, is_start, is_end
    )
    print(f"  基线 IS: Ret={baseline_result['strategy_return']:+.1f}% "
          f"WR={baseline_result['win_rate']:.1f}% "
          f"PF={baseline_result['profit_factor']:.2f} "
          f"MDD={baseline_result['max_drawdown']:.1f}% "
          f"T={baseline_result['total_trades']}")

    # ── Step 2: MAE 校准分析 ──
    calibration = analyze_mae_by_regime(baseline_result['trades'])

    # ── Step 3: 构建实验变体 ──
    # E1: MAE-calibrated ATR mults + regime-specific ceilings
    MAE_CALIBRATED = {}
    regime_key_map = {
        'neutral': 'atr_sl_mult_neutral',
        'trend': 'atr_sl_mult_trend',
        'low_vol_trend': 'atr_sl_mult_low_vol_trend',
        'high_vol': 'atr_sl_mult_high_vol',
        'high_vol_choppy': 'atr_sl_mult_high_vol_choppy',
    }
    # 收集每个 regime 的 MAE P90 (取 short/long 的更大值)
    regime_mae_p90 = {}
    for leg, cal in calibration.items():
        direction, regime = leg.split('|')
        if regime in regime_key_map:
            key = regime_key_map[regime]
            if key not in MAE_CALIBRATED:
                MAE_CALIBRATED[key] = cal['adjusted_mult']
            else:
                MAE_CALIBRATED[key] = (MAE_CALIBRATED[key] + cal['adjusted_mult']) / 2
            # 记录 MAE P90 用于设置 regime-specific ceiling
            old_p90 = regime_mae_p90.get(regime, 0)
            regime_mae_p90[regime] = max(old_p90, cal['mae_p90'])

    # 根据 MAE P90 设置 regime-specific ATR SL ceilings
    # ceiling = -(MAE_P90 * 1.2) — 留 20% 余量让 90% 赢单不被止损
    for regime, p90 in regime_mae_p90.items():
        ceil_val = -(p90 * 1.2)
        ceil_val = max(-0.15, min(-0.02, ceil_val))  # 限制在 -2% ~ -15%
        MAE_CALIBRATED[f'atr_sl_ceil_{regime}'] = ceil_val

    print(f"\n  MAE 校准后参数: {json.dumps({k: round(v, 4) for k, v in MAE_CALIBRATED.items()}, indent=2)}")

    # E2: Anti-Squeeze soft penalty
    SOFT_ANTISQUEEZE = {
        'use_soft_antisqueeze': True,
        'soft_antisqueeze_w_fz': 0.5,
        'soft_antisqueeze_w_oi': 0.3,
        'soft_antisqueeze_w_imb': 0.2,
        'soft_antisqueeze_midpoint': 1.5,
        'soft_antisqueeze_steepness': 2.0,
        'soft_antisqueeze_max_discount': 0.50,
    }

    # E3: 放宽开仓比率 (反手基础)
    OPEN_DOMINANCE = {
        'open_dominance_ratio': 1.3,
    }

    variants = {
        'E0_v10.2_baseline':    {},
        'E1_+MAE_ATR':         {**MAE_CALIBRATED},
        'E2_+SoftSqueeze':     {**SOFT_ANTISQUEEZE},
        'E3_+OpenDom1.3':      {**OPEN_DOMINANCE},
        'E4_Phase1_full':      {**MAE_CALIBRATED, **SOFT_ANTISQUEEZE, **OPEN_DOMINANCE},
    }

    # ── Step 4: 运行所有变体 ──
    print(f"\n{'='*170}")
    print(f"  Step 4: 运行 5 个变体 IS + OOS")
    print(f"{'='*170}")

    results = []
    for name, overrides in variants.items():
        cfg = load_base_config()
        cfg.update(V8_OVERRIDES)
        cfg.update(V10_2_PRODUCTION)
        cfg.update(overrides)

        print(f"\n  [{name}]", end=' ', flush=True)
        is_r = run_and_summarize(is_data, is_signals, needed_tfs, cfg,
                                  primary_tf, decision_tfs, is_start, is_end)
        oos_r = run_and_summarize(oos_data, oos_signals, needed_tfs, cfg,
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

    # ── Step 5: 汇总表 ──
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

    # vs 基线比较
    base = rdf[rdf['variant'] == 'E0_v10.2_baseline'].iloc[0]
    print(f"\n  === vs E0 v10.2 基线差值 ===")
    for _, r in rdf.iterrows():
        if r['variant'] == 'E0_v10.2_baseline':
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
