#!/usr/bin/env python3
"""Fee / Slippage 压力测试 (GPT-1.3)

目标: 验证策略在更高交易成本下的鲁棒性。
      回测默认值 (币安标准): taker_fee=0.05%, slippage=0.10%
      真实交易中, 高波动时段滑点可能翻倍, 非 VIP 手续费更高。

实验设计:
  F0: 基线 (默认 fee=0.05%, slippage=0.10%)
  F1: 高手续费 (fee×1.5 = 0.075%, slippage不变)
  F2: 高滑点 (fee不变, slippage×2 = 0.20%)
  F3: 双高 (fee×1.5 + slippage×2)
  F4: 极端 (fee×2 + slippage×3) — 压力极限
  F5: 低成本 (fee×0.5 = 0.025%, slippage×0.5 = 0.05%) — VIP/maker 场景
  F6: 无滑点 (fee不变, slippage=0) — 理想化参照

鲁棒性判断:
  - 如果 F3 (双高) OOS PF > 1.5 且 Ret > 0, 策略具有良好鲁棒性
  - 如果 F3 OOS PF < 1.2 或 Ret < 0, 策略对成本敏感, 需优化交易频率
  - Fee Drag 超过收益率的 50% 时为警告信号

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

from strategy_futures import FuturesEngine
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
#  Fee/Slippage 变体
# ============================================================

# 默认值 (对照)
DEFAULT_TAKER_FEE = 0.0005   # 0.05%
DEFAULT_SLIPPAGE = 0.001     # 0.10%

variants = {
    'F0_baseline': {
        'taker_fee': DEFAULT_TAKER_FEE,
        'slippage': DEFAULT_SLIPPAGE,
        'label': '基线 (0.05% + 0.10%)',
    },
    'F1_high_fee': {
        'taker_fee': DEFAULT_TAKER_FEE * 1.5,
        'slippage': DEFAULT_SLIPPAGE,
        'label': '高手续费 (0.075% + 0.10%)',
    },
    'F2_high_slippage': {
        'taker_fee': DEFAULT_TAKER_FEE,
        'slippage': DEFAULT_SLIPPAGE * 2,
        'label': '高滑点 (0.05% + 0.20%)',
    },
    'F3_double_high': {
        'taker_fee': DEFAULT_TAKER_FEE * 1.5,
        'slippage': DEFAULT_SLIPPAGE * 2,
        'label': '双高 (0.075% + 0.20%)',
    },
    'F4_extreme': {
        'taker_fee': DEFAULT_TAKER_FEE * 2,
        'slippage': DEFAULT_SLIPPAGE * 3,
        'label': '极端 (0.10% + 0.30%)',
    },
    'F5_low_cost': {
        'taker_fee': DEFAULT_TAKER_FEE * 0.5,
        'slippage': DEFAULT_SLIPPAGE * 0.5,
        'label': 'VIP/Maker (0.025% + 0.05%)',
    },
    'F6_no_slippage': {
        'taker_fee': DEFAULT_TAKER_FEE,
        'slippage': 0.0,
        'label': '零滑点 (0.05% + 0%)',
    },
}


# ============================================================
#  回测执行 (动态覆盖 FuturesEngine 类常量)
# ============================================================

def run_with_fee_override(all_data, all_signals, needed_tfs, cfg, primary_tf,
                          decision_tfs, start_dt, end_dt,
                          taker_fee, slippage):
    """运行回测, 临时覆盖 FuturesEngine 的手续费和滑点参数"""
    orig_fee = FuturesEngine.TAKER_FEE
    orig_slip = FuturesEngine.SLIPPAGE

    try:
        FuturesEngine.TAKER_FEE = taker_fee
        FuturesEngine.SLIPPAGE = slippage

        tf_score_map = _build_tf_score_index(all_data, all_signals, needed_tfs, cfg)
        primary_df = all_data[primary_tf]
        result = run_strategy_multi_tf(
            primary_df=primary_df, tf_score_map=tf_score_map,
            decision_tfs=decision_tfs, config=cfg,
            primary_tf=primary_tf, trade_days=0,
            trade_start_dt=start_dt, trade_end_dt=end_dt,
        )
    finally:
        FuturesEngine.TAKER_FEE = orig_fee
        FuturesEngine.SLIPPAGE = orig_slip

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

    fees_info = result.get('fees', {})
    total_fees = fees_info.get('total_fees', 0)
    total_slippage = fees_info.get('slippage_cost', 0)
    total_costs = fees_info.get('total_costs', 0)
    fee_drag = fees_info.get('fee_drag_pct', 0)
    final_total = result.get('final_total', 0)

    avg_pnl = np.mean([float(t.get('pnl', 0)) for t in closes]) if closes else 0
    avg_cost_per_trade = total_costs / n_trades if n_trades > 0 else 0

    return {
        'strategy_return': ret,
        'max_drawdown': mdd,
        'total_trades': n_trades,
        'win_rate': wr,
        'profit_factor': pf,
        'calmar': calmar,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'final_total': final_total,
        'total_fees': total_fees,
        'total_slippage': total_slippage,
        'total_costs': total_costs,
        'fee_drag_pct': fee_drag,
        'avg_pnl_per_trade': avg_pnl,
        'avg_cost_per_trade': avg_cost_per_trade,
        'cost_to_profit_ratio': total_costs / gross_profit * 100 if gross_profit > 0 else 0,
    }


def main():
    print("=" * 140)
    print("  Fee / Slippage 压力测试 (GPT-1.3)")
    print("  目标: 验证策略在不同交易成本下的鲁棒性")
    print("  默认值: taker_fee=0.05%, slippage=0.10%")
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

    # 构建配置
    cfg = load_base_config()
    cfg.update(V8_OVERRIDES)
    cfg.update(V5_PROD_OVERRIDES)

    results = []
    for name, v in variants.items():
        print(f"\n  [{name}] {v['label']}", end=' ', flush=True)
        print(f"(fee={v['taker_fee']*100:.3f}%, slip={v['slippage']*100:.2f}%)", end=' ', flush=True)

        is_r = run_with_fee_override(
            is_data, is_signals, needed_tfs, cfg, primary_tf, decision_tfs,
            is_start, is_end, v['taker_fee'], v['slippage'])

        oos_r = run_with_fee_override(
            oos_data, oos_signals, needed_tfs, cfg, primary_tf, decision_tfs,
            oos_start, oos_end, v['taker_fee'], v['slippage'])

        row = {
            'variant': name,
            'label': v['label'],
            'taker_fee_pct': v['taker_fee'] * 100,
            'slippage_pct': v['slippage'] * 100,
            'is_ret': is_r['strategy_return'],
            'is_wr': is_r['win_rate'],
            'is_pf': is_r['profit_factor'],
            'is_mdd': is_r['max_drawdown'],
            'is_trades': is_r['total_trades'],
            'is_calmar': is_r['calmar'],
            'is_total_costs': is_r['total_costs'],
            'is_fee_drag': is_r['fee_drag_pct'],
            'is_avg_cost_trade': is_r['avg_cost_per_trade'],
            'is_cost_profit_ratio': is_r['cost_to_profit_ratio'],
            'oos_ret': oos_r['strategy_return'],
            'oos_wr': oos_r['win_rate'],
            'oos_pf': oos_r['profit_factor'],
            'oos_mdd': oos_r['max_drawdown'],
            'oos_trades': oos_r['total_trades'],
            'oos_calmar': oos_r['calmar'],
            'oos_total_costs': oos_r['total_costs'],
            'oos_fee_drag': oos_r['fee_drag_pct'],
            'oos_avg_cost_trade': oos_r['avg_cost_per_trade'],
            'oos_cost_profit_ratio': oos_r['cost_to_profit_ratio'],
        }
        results.append(row)

        print(f"\n    IS:  Ret={is_r['strategy_return']:+.1f}% PF={is_r['profit_factor']:.2f} "
              f"MDD={is_r['max_drawdown']:.1f}% T={is_r['total_trades']} "
              f"Costs=${is_r['total_costs']:,.0f} Drag={is_r['fee_drag_pct']:.1f}%", flush=True)
        print(f"    OOS: Ret={oos_r['strategy_return']:+.1f}% PF={oos_r['profit_factor']:.2f} "
              f"MDD={oos_r['max_drawdown']:.1f}% T={oos_r['total_trades']} "
              f"Costs=${oos_r['total_costs']:,.0f} Drag={oos_r['fee_drag_pct']:.1f}%", flush=True)

    # ============================================================
    #  汇总表格 — 策略指标
    # ============================================================
    rdf = pd.DataFrame(results)
    base = rdf[rdf['variant'] == 'F0_baseline'].iloc[0]

    print(f"\n{'='*160}")
    print(f"  Fee/Slippage 压力测试汇总 — 策略指标")
    print(f"{'='*160}")
    fmt_header = (f"  {'变体':<22} {'Fee%':>6} {'Slip%':>6} | "
                  f"{'IS_Ret':>8} {'IS_PF':>6} {'IS_MDD':>7} {'IS_N':>4} | "
                  f"{'OOS_Ret':>8} {'ΔOOS':>7} {'OOS_PF':>7} {'OOS_MDD':>7} {'OOS_N':>5} | "
                  f"{'OOS_Drag':>8}")
    print(fmt_header)
    print(f"  {'-'*158}")

    for _, row in rdf.iterrows():
        d_oos = row['oos_ret'] - base['oos_ret']
        marker = ''
        if row['oos_pf'] >= 1.5 and row['oos_ret'] > 0:
            marker = ' ✓'
        elif row['oos_pf'] < 1.2 or row['oos_ret'] <= 0:
            marker = ' ✗'
        print(f"  {row['variant']:<22} {row['taker_fee_pct']:>5.3f} {row['slippage_pct']:>5.2f} | "
              f"{row['is_ret']:>+7.1f}% {row['is_pf']:>5.2f} {row['is_mdd']:>6.1f}% {row['is_trades']:>4} | "
              f"{row['oos_ret']:>+7.1f}% {d_oos:>+6.1f}% {row['oos_pf']:>6.2f} "
              f"{row['oos_mdd']:>6.1f}% {row['oos_trades']:>5} | "
              f"{row['oos_fee_drag']:>6.1f}%{marker}")

    # ============================================================
    #  汇总表格 — 成本分析
    # ============================================================
    print(f"\n{'='*160}")
    print(f"  Fee/Slippage 压力测试汇总 — 成本分析")
    print(f"{'='*160}")
    cost_header = (f"  {'变体':<22} | "
                   f"{'IS总成本':>10} {'IS费拖':>7} {'IS单笔':>8} {'IS成本/利':>9} | "
                   f"{'OOS总成本':>10} {'OOS费拖':>7} {'OOS单笔':>8} {'OOS成本/利':>9}")
    print(cost_header)
    print(f"  {'-'*120}")

    for _, row in rdf.iterrows():
        print(f"  {row['variant']:<22} | "
              f"${row['is_total_costs']:>9,.0f} {row['is_fee_drag']:>5.1f}% "
              f"${row['is_avg_cost_trade']:>7,.0f} {row['is_cost_profit_ratio']:>7.1f}% | "
              f"${row['oos_total_costs']:>9,.0f} {row['oos_fee_drag']:>5.1f}% "
              f"${row['oos_avg_cost_trade']:>7,.0f} {row['oos_cost_profit_ratio']:>7.1f}%")

    # ============================================================
    #  敏感度分析
    # ============================================================
    print(f"\n{'='*120}")
    print(f"  敏感度分析")
    print(f"{'='*120}")

    # Fee 敏感度: F0 vs F1 (fee +50%)
    f0_oos = rdf[rdf['variant'] == 'F0_baseline'].iloc[0]
    f1_oos = rdf[rdf['variant'] == 'F1_high_fee'].iloc[0]
    f2_oos = rdf[rdf['variant'] == 'F2_high_slippage'].iloc[0]

    fee_sens_ret = (f1_oos['oos_ret'] - f0_oos['oos_ret']) / 50  # per 1% fee increase
    slip_sens_ret = (f2_oos['oos_ret'] - f0_oos['oos_ret']) / 100  # per 1% slip increase

    print(f"\n  Fee 敏感度 (每 +50% fee 变化):")
    print(f"    OOS Ret: {f1_oos['oos_ret'] - f0_oos['oos_ret']:+.2f}pp")
    print(f"    OOS PF:  {f1_oos['oos_pf'] - f0_oos['oos_pf']:+.3f}")

    print(f"\n  Slippage 敏感度 (每 +100% slippage 变化):")
    print(f"    OOS Ret: {f2_oos['oos_ret'] - f0_oos['oos_ret']:+.2f}pp")
    print(f"    OOS PF:  {f2_oos['oos_pf'] - f0_oos['oos_pf']:+.3f}")

    # 盈亏平衡分析
    if f0_oos['oos_ret'] > 0:
        # 估算多少倍成本会让 OOS ret 归零 (线性外推)
        f3_oos = rdf[rdf['variant'] == 'F3_double_high'].iloc[0]
        f4_oos = rdf[rdf['variant'] == 'F4_extreme'].iloc[0]

        base_cost = f0_oos['oos_total_costs']
        f4_cost = f4_oos['oos_total_costs']
        base_ret = f0_oos['oos_ret']
        f4_ret = f4_oos['oos_ret']

        if base_cost != f4_cost and base_ret != f4_ret:
            cost_mult_at_zero = base_cost + (0 - base_ret) / (f4_ret - base_ret) * (f4_cost - base_cost)
            breakeven_mult = cost_mult_at_zero / base_cost if base_cost > 0 else float('inf')
            print(f"\n  盈亏平衡点 (线性外推):")
            print(f"    当交易成本达到基线的 {breakeven_mult:.1f}x 时, OOS 收益归零")
            if breakeven_mult > 3:
                print(f"    → 策略对成本有较强容忍度 (breakeven > 3x)")
            elif breakeven_mult > 2:
                print(f"    → 策略对成本有中等容忍度 (breakeven 2-3x)")
            else:
                print(f"    → ⚠ 策略对成本敏感 (breakeven < 2x), 需控制交易频率")

    # ============================================================
    #  鲁棒性结论
    # ============================================================
    print(f"\n{'='*120}")
    print(f"  鲁棒性结论")
    print(f"{'='*120}")

    f3 = rdf[rdf['variant'] == 'F3_double_high'].iloc[0]
    f4 = rdf[rdf['variant'] == 'F4_extreme'].iloc[0]

    robustness_level = 'UNKNOWN'
    if f3['oos_pf'] > 1.5 and f3['oos_ret'] > 0:
        if f4['oos_pf'] > 1.3 and f4['oos_ret'] > 0:
            robustness_level = 'EXCELLENT'
            print(f"\n  ★★★ 鲁棒性: 优秀")
            print(f"  即使在极端成本 (fee×2 + slip×3) 下, 策略仍然盈利")
            print(f"  F3(双高) OOS: PF={f3['oos_pf']:.2f}, Ret={f3['oos_ret']:+.1f}%")
            print(f"  F4(极端) OOS: PF={f4['oos_pf']:.2f}, Ret={f4['oos_ret']:+.1f}%")
        else:
            robustness_level = 'GOOD'
            print(f"\n  ★★ 鲁棒性: 良好")
            print(f"  在双高成本下 (fee×1.5 + slip×2) 策略仍然盈利, 但极端场景有压力")
            print(f"  F3(双高) OOS: PF={f3['oos_pf']:.2f}, Ret={f3['oos_ret']:+.1f}%")
            print(f"  F4(极端) OOS: PF={f4['oos_pf']:.2f}, Ret={f4['oos_ret']:+.1f}%")
    elif f3['oos_pf'] > 1.2 and f3['oos_ret'] > 0:
        robustness_level = 'MODERATE'
        print(f"\n  ★ 鲁棒性: 中等")
        print(f"  策略在适度成本增加下仍盈利, 但利润margin较薄")
        print(f"  F3(双高) OOS: PF={f3['oos_pf']:.2f}, Ret={f3['oos_ret']:+.1f}%")
        print(f"  建议: 优化交易频率, 减少低质量信号")
    else:
        robustness_level = 'POOR'
        print(f"\n  ⚠ 鲁棒性: 较差")
        print(f"  策略对交易成本敏感, 双高成本下利润大幅缩水或亏损")
        print(f"  F3(双高) OOS: PF={f3['oos_pf']:.2f}, Ret={f3['oos_ret']:+.1f}%")
        print(f"  建议: 1) 提高信号质量阈值 2) 减少交易频率 3) 争取 maker 费率")

    # 额外建议
    print(f"\n  交易成本优化建议:")
    f5 = rdf[rdf['variant'] == 'F5_low_cost'].iloc[0]
    cost_upside = f5['oos_ret'] - f0_oos['oos_ret']
    print(f"    1. VIP/Maker 费率提升潜力: OOS Ret +{cost_upside:.1f}pp "
          f"(从 {f0_oos['oos_ret']:+.1f}% 到 {f5['oos_ret']:+.1f}%)")
    print(f"    2. 基线 OOS fee drag = {f0_oos['oos_fee_drag']:.1f}% "
          f"(成本占初始资金的比例)")
    base_cost_ratio = f0_oos['oos_cost_profit_ratio']
    print(f"    3. 基线 OOS 成本/利润比 = {base_cost_ratio:.1f}% "
          f"({'正常' if base_cost_ratio < 30 else '偏高' if base_cost_ratio < 50 else '⚠ 过高'})")

    # ============================================================
    #  保存结果
    # ============================================================
    export_dir = 'data/backtests/fee_slippage_stress'
    os.makedirs(export_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    csv_path = os.path.join(export_dir, f'fee_slippage_stress_{ts}.csv')
    rdf.to_csv(csv_path, index=False)

    summary = {
        'experiment': 'fee_slippage_stress',
        'timestamp': ts,
        'robustness_level': robustness_level,
        'baseline': {
            'taker_fee': DEFAULT_TAKER_FEE,
            'slippage': DEFAULT_SLIPPAGE,
            'oos_return': float(f0_oos['oos_ret']),
            'oos_pf': float(f0_oos['oos_pf']),
            'oos_fee_drag': float(f0_oos['oos_fee_drag']),
        },
        'double_high': {
            'taker_fee': DEFAULT_TAKER_FEE * 1.5,
            'slippage': DEFAULT_SLIPPAGE * 2,
            'oos_return': float(f3['oos_ret']),
            'oos_pf': float(f3['oos_pf']),
            'oos_fee_drag': float(f3['oos_fee_drag']),
        },
        'extreme': {
            'taker_fee': DEFAULT_TAKER_FEE * 2,
            'slippage': DEFAULT_SLIPPAGE * 3,
            'oos_return': float(f4['oos_ret']),
            'oos_pf': float(f4['oos_pf']),
            'oos_fee_drag': float(f4['oos_fee_drag']),
        },
        'sensitivity': {
            'fee_50pct_oos_ret_delta': float(f1_oos['oos_ret'] - f0_oos['oos_ret']),
            'slip_100pct_oos_ret_delta': float(f2_oos['oos_ret'] - f0_oos['oos_ret']),
        },
    }

    json_path = os.path.join(export_dir, f'fee_slippage_summary_{ts}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  结果已保存: {csv_path}")
    print(f"  摘要已保存: {json_path}")
    print("\n  实验完成。")


if __name__ == '__main__':
    main()
