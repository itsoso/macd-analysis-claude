#!/usr/bin/env python3
"""CS-KDJ 去相关 A/B 实验 (GPT-4.3)

目标: 验证蜡烛图(CS)和 KDJ 去相关是否能提升信号质量。
      两书都基于短期动量, 在 neutral regime 高度相关, 导致信号冗余。

实验设计:
  D0: 基线 (CS + KDJ 独立叠加, cs_kdj_decorr="none")
  D1: Max-of-two (取 CS/KDJ 中 bonus 较大者, 消除双重计算)
  D2: 冗余折扣 60% (两者同时触发时, 总 bonus × 0.6)
  D3: 冗余折扣 40% (两者同时触发时, 总 bonus × 0.4, 更激进)
  D4: KDJ-only (去掉 CS bonus, 保留 KDJ)
  D5: CS-only (去掉 KDJ bonus, 保留 CS)
  D6: 提高 CS 权重 + Max-of-two (cs_bonus=0.10, max-of-two 让 CS 主导)

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

# v8.0+P20 基线 (与其他 A/B 脚本一致)
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

# v5 生产配置 (含 L2 leg budget + M4 MAE P95)
V5_PROD_OVERRIDES = {
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
    'use_risk_per_trade': True,
    'risk_per_trade_pct': 0.025,
    'risk_stop_mode': 'atr',
    'use_atr_sl': True,
    'atr_sl_mult': 3.0,
    'atr_sl_floor': -0.25,
    'atr_sl_ceil': -0.06,
    'use_mae_driven_sl': True,
    'mae_sl_quantile': 'p95',
    'mae_sl_floor': -0.35,
    'mae_sl_ceil': -0.04,
    'mae_sl_long_tighten': 0.85,
    'mae_sl_neutral_short': 0.0951,
    'mae_sl_neutral_long': 0.0843,
    'mae_sl_trend_short': 0.0580,
    'mae_sl_trend_long': 0.0397,
    'mae_sl_high_vol_short': 0.0756,
    'mae_sl_high_vol_long': 0.0993,
    'mae_sl_low_vol_trend_short': 0.2232,
    'mae_sl_low_vol_trend_long': 0.1071,
}


# ============================================================
#  实验变体配置
# ============================================================

variants = {
    'D0_baseline': {
        'cs_kdj_decorr': 'none',
        'label': '基线 (CS+KDJ 独立叠加)',
    },
    'D1_max_of_two': {
        'cs_kdj_decorr': 'max_of_two',
        'label': 'Max-of-two (取较大 bonus)',
    },
    'D2_redundancy_60': {
        'cs_kdj_decorr': 'redundancy_discount',
        'cs_kdj_redundancy_discount': 0.6,
        'label': '冗余折扣 60%',
    },
    'D3_redundancy_40': {
        'cs_kdj_decorr': 'redundancy_discount',
        'cs_kdj_redundancy_discount': 0.4,
        'label': '冗余折扣 40%',
    },
    'D4_kdj_only': {
        'cs_kdj_decorr': 'kdj_only',
        'label': 'KDJ-only (去掉 CS)',
    },
    'D5_cs_only': {
        'cs_kdj_decorr': 'cs_only',
        'label': 'CS-only (去掉 KDJ)',
    },
    'D6_cs_boost_max': {
        'cs_kdj_decorr': 'max_of_two',
        'cs_bonus': 0.10,
        'label': 'CS 提权 + Max-of-two',
    },
}


# ============================================================
#  回测执行
# ============================================================

def run_backtest_with_stats(all_data, all_signals, needed_tfs, cfg, primary_tf,
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

    n_trades = len(closes)
    wins = sum(1 for t in closes if float(t.get('pnl', 0)) > 0)
    wr = wins / n_trades * 100 if n_trades > 0 else 0
    gross_profit = sum(float(t.get('pnl', 0)) for t in closes if float(t.get('pnl', 0)) > 0)
    gross_loss = abs(sum(float(t.get('pnl', 0)) for t in closes if float(t.get('pnl', 0)) <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    calmar = 0
    mdd = result.get('max_drawdown', 0)
    ret = result.get('strategy_return', 0)
    if mdd != 0:
        calmar = ret / abs(mdd)

    # 按 side+regime 统计
    regime_stats = defaultdict(lambda: {'n': 0, 'w': 0, 'pnl': 0.0})
    for t in opens:
        side = t.get('direction', 'unknown')
        regime = t.get('regime_label', 'unknown')
        key = f"{side}|{regime}"
        regime_stats[key]['n'] += 1

    for t in closes:
        side = t.get('direction', 'unknown')
        regime = t.get('regime_label', 'unknown')
        key = f"{side}|{regime}"
        pnl_val = float(t.get('pnl', 0))
        regime_stats[key]['pnl'] += pnl_val
        if pnl_val > 0:
            regime_stats[key]['w'] += 1

    return {
        'strategy_return': ret,
        'max_drawdown': mdd,
        'total_trades': n_trades,
        'win_rate': wr,
        'profit_factor': pf,
        'calmar': calmar,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'regime_stats': dict(regime_stats),
    }


def main():
    print("=" * 140)
    print("  CS-KDJ 去相关 A/B 实验 (GPT-4.3)")
    print("  目标: 验证蜡烛图(CS)和 KDJ 去相关是否能提升信号质量")
    print("  策略: v8+P20 + v5生产配置 (L2 leg budget + M4 MAE P95)")
    print("=" * 140)

    # 加载数据
    print("\n  加载 IS 数据 (2025-01 ~ 2026-01) ...")
    is_data, is_signals, needed_tfs, primary_tf, decision_tfs, is_start, is_end = \
        prepare_data('ETHUSDT', '2025-01-01', '2026-01-31')
    print(f"  决策 TF: {decision_tfs}")

    print("  加载 OOS 数据 (2024-01 ~ 2024-12) ...")
    oos_data, oos_signals, _, _, _, oos_start, oos_end = \
        prepare_data('ETHUSDT', '2024-01-01', '2024-12-31')

    results = []
    for name, v in variants.items():
        cfg = load_base_config()
        cfg.update(V8_OVERRIDES)
        cfg.update(V5_PROD_OVERRIDES)
        # 应用变体配置
        for k, val in v.items():
            if k != 'label':
                cfg[k] = val

        print(f"\n  [{name}] {v['label']}", end=' ', flush=True)

        is_r = run_backtest_with_stats(is_data, is_signals, needed_tfs, cfg,
                                        primary_tf, decision_tfs, is_start, is_end)
        oos_r = run_backtest_with_stats(oos_data, oos_signals, needed_tfs, cfg,
                                         primary_tf, decision_tfs, oos_start, oos_end)

        row = {
            'variant': name,
            'label': v['label'],
            'is_ret': is_r['strategy_return'],
            'is_wr': is_r['win_rate'],
            'is_pf': is_r['profit_factor'],
            'is_mdd': is_r['max_drawdown'],
            'is_trades': is_r['total_trades'],
            'is_calmar': is_r['calmar'],
            'oos_ret': oos_r['strategy_return'],
            'oos_wr': oos_r['win_rate'],
            'oos_pf': oos_r['profit_factor'],
            'oos_mdd': oos_r['max_drawdown'],
            'oos_trades': oos_r['total_trades'],
            'oos_calmar': oos_r['calmar'],
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

    # ============================================================
    #  汇总表格
    # ============================================================
    rdf = pd.DataFrame(results)
    base = rdf[rdf['variant'] == 'D0_baseline'].iloc[0]

    print(f"\n{'='*150}")
    print(f"  CS-KDJ 去相关 A/B 实验汇总")
    print(f"{'='*150}")
    print(f"  {'变体':<25} {'模式':<22} | "
          f"{'IS_Ret':>8} {'IS_PF':>6} {'IS_MDD':>7} {'N':>4} | "
          f"{'OOS_Ret':>8} {'ΔOOS':>7} {'OOS_PF':>7} {'ΔPF':>6} {'OOS_MDD':>7} {'N':>4}")
    print(f"  {'-'*148}")

    for _, row in rdf.iterrows():
        d_oos = row['oos_ret'] - base['oos_ret']
        d_pf = row['oos_pf'] - base['oos_pf']
        marker = ''
        if row['oos_pf'] > base['oos_pf'] and row['oos_ret'] > base['oos_ret']:
            marker = ' ★'
        elif row['oos_pf'] > base['oos_pf'] or row['oos_ret'] > base['oos_ret']:
            marker = ' △'
        print(f"  {row['variant']:<25} {row['label']:<22} | "
              f"{row['is_ret']:>+7.1f}% {row['is_pf']:>5.2f} {row['is_mdd']:>6.1f}% {row['is_trades']:>4} | "
              f"{row['oos_ret']:>+7.1f}% {d_oos:>+6.1f}% {row['oos_pf']:>6.2f} {d_pf:>+5.2f} "
              f"{row['oos_mdd']:>6.1f}% {row['oos_trades']:>4}{marker}")

    # ============================================================
    #  Regime-level 对比 (neutral 是去相关重点)
    # ============================================================
    print(f"\n{'='*150}")
    print(f"  Regime-level PnL 对比 (OOS) — 重点关注 neutral")
    print(f"{'='*150}")
    target_keys = ['short|neutral', 'long|neutral', 'short|trend', 'long|trend',
                   'short|high_vol', 'long|high_vol']
    header = f"  {'变体':<25}"
    for k in target_keys:
        header += f" {k:>16}"
    print(header)
    print(f"  {'-'*125}")

    for _, row in rdf.iterrows():
        oos_rs = json.loads(row['oos_regime_stats']) if isinstance(row['oos_regime_stats'], str) else row['oos_regime_stats']
        line = f"  {row['variant']:<25}"
        for k in target_keys:
            rs = oos_rs.get(k, {})
            n = rs.get('n', 0)
            pnl = rs.get('pnl', 0.0)
            if n > 0:
                line += f" n={n:>2}${pnl:>+7,.0f}"
            else:
                line += f" {'—':>16}"
        print(line)

    # ============================================================
    #  推荐方案分析
    # ============================================================
    print(f"\n{'='*120}")
    print(f"  推荐方案分析")
    print(f"{'='*120}")

    candidates = rdf[rdf['variant'] != 'D0_baseline'].copy()

    if len(candidates) > 0:
        # 多维评分
        for col in ['oos_pf', 'oos_ret', 'oos_calmar']:
            _min = candidates[col].min()
            _max = candidates[col].max()
            _range = _max - _min if _max > _min else 1.0
            candidates[f'{col}_norm'] = (candidates[col] - _min) / _range

        # IS-OOS 一致性
        candidates['pf_consistency'] = abs(candidates['is_pf'] - candidates['oos_pf'])
        _min_c = candidates['pf_consistency'].min()
        _max_c = candidates['pf_consistency'].max()
        _range_c = _max_c - _min_c if _max_c > _min_c else 1.0
        candidates['consistency_norm'] = 1.0 - (candidates['pf_consistency'] - _min_c) / _range_c

        candidates['score'] = (
            0.40 * candidates['oos_pf_norm'] +
            0.25 * candidates['oos_ret_norm'] +
            0.20 * candidates['oos_calmar_norm'] +
            0.15 * candidates['consistency_norm']
        )

        best = candidates.loc[candidates['score'].idxmax()]
        print(f"\n  ★ 综合最优: {best['variant']} ({best['label']})")
        print(f"    IS:  Ret={best['is_ret']:+.1f}% PF={best['is_pf']:.2f} MDD={best['is_mdd']:.1f}%")
        print(f"    OOS: Ret={best['oos_ret']:+.1f}% PF={best['oos_pf']:.2f} MDD={best['oos_mdd']:.1f}%")
        d_oos = best['oos_ret'] - float(base['oos_ret'])
        d_pf = best['oos_pf'] - float(base['oos_pf'])
        print(f"    vs 基线: OOS Ret {d_oos:+.1f}pp, OOS PF {d_pf:+.3f}")

        # 完整排名
        candidates_sorted = candidates.sort_values('score', ascending=False)
        print(f"\n  排名 (综合得分):")
        for rank, (_, row) in enumerate(candidates_sorted.iterrows(), 1):
            d_r = row['oos_ret'] - float(base['oos_ret'])
            d_p = row['oos_pf'] - float(base['oos_pf'])
            print(f"    {rank}. {row['variant']:<25} Score={row['score']:.3f} "
                  f"OOS: Ret={row['oos_ret']:+.1f}%({d_r:+.1f}) PF={row['oos_pf']:.2f}({d_p:+.2f}) "
                  f"MDD={row['oos_mdd']:.1f}%")

        # 关键结论
        print(f"\n  结论:")
        if best['oos_pf'] > base['oos_pf'] and best['oos_ret'] > base['oos_ret']:
            print(f"  → CS-KDJ 去相关有效: {best['variant']} 在 OOS 上双指标超越基线")
            print(f"  → 建议更新 v5 配置: cs_kdj_decorr = '{variants[best['variant']].get('cs_kdj_decorr', 'none')}'")
        elif best['oos_pf'] > base['oos_pf']:
            print(f"  → CS-KDJ 去相关部分有效: PF 提升但收益率略降")
            print(f"  → 保守建议: 可选择性启用, 或保持基线")
        else:
            print(f"  → CS-KDJ 去相关效果不显著或负面, 建议保持当前配置")
            print(f"  → 原因可能: CS/KDJ 在 bonus 层面的冗余对最终结果影响有限")

    # ============================================================
    #  保存结果
    # ============================================================
    export_dir = 'data/backtests/cs_kdj_decorr_ab'
    os.makedirs(export_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    csv_path = os.path.join(export_dir, f'cs_kdj_decorr_{ts}.csv')
    rdf.to_csv(csv_path, index=False)
    print(f"\n  结果已保存: {csv_path}")

    if len(candidates) > 0:
        best_name = candidates.loc[candidates['score'].idxmax(), 'variant']
        recommend = {
            'experiment': 'cs_kdj_decorr_ab',
            'timestamp': ts,
            'recommended_variant': best_name,
            'config': {k: v for k, v in variants[best_name].items() if k != 'label'},
            'metrics': {
                'oos_return': float(best['oos_ret']),
                'oos_pf': float(best['oos_pf']),
                'oos_mdd': float(best['oos_mdd']),
                'is_return': float(best['is_ret']),
                'is_pf': float(best['is_pf']),
            },
        }
        json_path = os.path.join(export_dir, f'recommended_config_{ts}.json')
        with open(json_path, 'w') as f:
            json.dump(recommend, f, indent=2, ensure_ascii=False)
        print(f"  推荐配置已保存: {json_path}")

    print("\n  实验完成。")


if __name__ == '__main__':
    main()
