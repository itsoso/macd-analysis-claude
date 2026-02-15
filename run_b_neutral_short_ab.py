"""
B0-B3: Neutral short 入场质量 A/B 实验

B0: 基线（v6.0 不动）
B1: neutral short 仓位硬上限 0.10（所有 neutral 空单减至 10%）
B2: neutral short 强门控 — 24h方向必须空头占优 + 4h MA空头结构 + 低冲突
B3: B2 + ghost cooldown（被过滤交易仍触发冷却）

在 IS (2025-01~2026-01) 和 OOS (2024-01~2024-12) 上双重验证。
"""

import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from run_p0_oos_validation import V6_OVERRIDES, load_base_config
from run_p1_p2_sensitivity import prepare_data, run_single_backtest
from optimize_six_book import (
    _build_tf_score_index,
    run_strategy_multi_tf,
)


def run_backtest_with_trades(all_data, all_signals, needed_tfs, cfg, primary_tf,
                              decision_tfs, start_dt, end_dt):
    """运行回测并返回结果和交易列表"""
    tf_score_map = _build_tf_score_index(all_data, all_signals, needed_tfs, cfg)
    primary_df = all_data[primary_tf]
    result = run_strategy_multi_tf(
        primary_df=primary_df,
        tf_score_map=tf_score_map,
        decision_tfs=decision_tfs,
        config=cfg,
        primary_tf=primary_tf,
        trade_days=0,
        trade_start_dt=start_dt,
        trade_end_dt=end_dt,
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

    # 按 side+regime 统计
    regime_stats = defaultdict(lambda: {'n': 0, 'w': 0, 'pnl': 0.0})
    for i, o in enumerate(opens):
        side = o.get('direction', 'unknown')
        regime = o.get('regime_label', 'unknown')
        # 找对应的 CLOSE
        close_action = f'CLOSE_{side.upper()}'
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
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'regime_stats': dict(regime_stats),
    }


def main():
    print("=" * 100)
    print("  B0-B3: Neutral short 入场质量 A/B 实验")
    print("=" * 100)

    # 准备数据
    print("  加载数据...")
    is_data, is_signals, needed_tfs, primary_tf, decision_tfs, is_start, is_end = \
        prepare_data('ETHUSDT', '2025-01-01', '2026-01-31')
    oos_data, oos_signals, _, _, _, oos_start, oos_end = \
        prepare_data('ETHUSDT', '2024-01-01', '2024-12-31')

    # ============================================================
    # 定义实验变体
    # ============================================================
    variants = {
        'B0_baseline': {},

        # B1: neutral short 仓位硬上限 0.10
        # 实现: 将 ALL neutral struct discounts 设为 0.10
        'B1_neutral_short_cap010': {
            'neutral_struct_discount_0': 0.10,
            'neutral_struct_discount_1': 0.10,
            'neutral_struct_discount_2': 0.10,
            'neutral_struct_discount_3': 0.10,
            'neutral_struct_discount_4plus': 0.10,
        },

        # B1b: 更激进 — neutral short 完全禁止
        # 实现: neutral threshold 设为 999
        'B1b_neutral_short_block': {
            'regime_short_threshold': 'neutral:999',
        },

        # B2: neutral short 强门控
        # 需要代码修改 — 这里通过组合现有参数近似实现:
        # - neutral threshold 提高到 60 (只有极强信号才能做空)
        # - 冲突折扣扩展到 neutral
        # - 结构折扣更激进
        'B2_neutral_strong_gate': {
            'regime_short_threshold': 'neutral:60',
            # 在 neutral 中也启用冲突折扣
            'short_conflict_regimes': 'trend,high_vol,neutral',
            # 更激进的结构折扣
            'neutral_struct_discount_0': 0.0,    # 0确认: 直接不做
            'neutral_struct_discount_1': 0.05,   # 1确认: 5%仓位
            'neutral_struct_discount_2': 0.15,   # 2确认: 15%仓位
            'neutral_struct_discount_3': 0.50,   # 3确认: 50%仓位
            'neutral_struct_discount_4plus': 1.0, # 4+: 全额
        },

        # B2b: B2 + trend/high_vol 也加门槛
        'B2b_all_short_gate': {
            'regime_short_threshold': 'neutral:60,trend:55,high_vol:55',
            'short_conflict_regimes': 'trend,high_vol,neutral',
            'neutral_struct_discount_0': 0.0,
            'neutral_struct_discount_1': 0.05,
            'neutral_struct_discount_2': 0.15,
            'neutral_struct_discount_3': 0.50,
            'neutral_struct_discount_4plus': 1.0,
        },

        # B3: B2 + ghost cooldown
        # 当前代码不支持 ghost cooldown，先用 B2 + 更长冷却期近似
        # ghost cooldown 的效果 ≈ 不改变交易时序但减少仓位
        # 这里用 cooldown*2 近似（让被过滤的信号"占据"冷却窗口）
        'B3_b2_longer_cooldown': {
            'regime_short_threshold': 'neutral:60',
            'short_conflict_regimes': 'trend,high_vol,neutral',
            'neutral_struct_discount_0': 0.0,
            'neutral_struct_discount_1': 0.05,
            'neutral_struct_discount_2': 0.15,
            'neutral_struct_discount_3': 0.50,
            'neutral_struct_discount_4plus': 1.0,
            'cooldown': 6,  # 从4加到6
        },
    }

    results = []
    for name, overrides in variants.items():
        cfg = load_base_config()
        cfg.update(overrides)

        print(f"\n  [{name}]", end=' ')
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
            'oos_ret': oos_r['strategy_return'],
            'oos_wr': oos_r['win_rate'],
            'oos_pf': oos_r['profit_factor'],
            'oos_mdd': oos_r['max_drawdown'],
            'oos_trades': oos_r['total_trades'],
            'is_regime_stats': json.dumps(is_r['regime_stats'], default=str),
            'oos_regime_stats': json.dumps(oos_r['regime_stats'], default=str),
        }
        results.append(row)
        print(f"IS: Ret={is_r['strategy_return']:+.1f}% WR={is_r['win_rate']:.1f}% PF={is_r['profit_factor']:.2f} Trades={is_r['total_trades']} | "
              f"OOS: Ret={oos_r['strategy_return']:+.1f}% WR={oos_r['win_rate']:.1f}% PF={oos_r['profit_factor']:.2f} Trades={oos_r['total_trades']}")

        # 打印 regime 明细
        for period, label in [(is_r, 'IS'), (oos_r, 'OOS')]:
            neutral_short = period['regime_stats'].get('short|neutral', {})
            if neutral_short.get('n', 0) > 0:
                ns_wr = neutral_short['w'] / neutral_short['n'] * 100
                print(f"    {label} neutral|short: n={neutral_short['n']} W={neutral_short['w']} "
                      f"WR={ns_wr:.0f}% PnL=${neutral_short['pnl']:+,.0f}")

    # 汇总表格
    rdf = pd.DataFrame(results)
    base_is = rdf[rdf['variant'] == 'B0_baseline'].iloc[0]
    base_oos_ret = base_is['oos_ret']
    base_is_ret = base_is['is_ret']

    print(f"\n{'='*100}")
    print(f"  B0-B3 A/B 实验汇总（差值 = 变体 - B0 基线）")
    print(f"{'='*100}")
    print(f"  {'变体':<30} {'IS_Ret':>8} {'ΔRet':>7} {'IS_WR':>7} {'IS_PF':>7} {'N':>4} | {'OOS_Ret':>8} {'ΔRet':>7} {'OOS_WR':>7} {'OOS_PF':>7} {'N':>4}")
    print(f"  {'-'*105}")
    for _, row in rdf.iterrows():
        d_is = row['is_ret'] - base_is_ret
        d_oos = row['oos_ret'] - base_oos_ret
        marker = '★' if row['variant'] == 'B0_baseline' else ('✓' if d_is > 0.3 and d_oos > -0.5 else '')
        print(f"  {row['variant']:<30} {row['is_ret']:>+7.1f}% {d_is:>+6.1f}% {row['is_wr']:>6.1f}% {row['is_pf']:>6.2f} {row['is_trades']:>4} | "
              f"{row['oos_ret']:>+7.1f}% {d_oos:>+6.1f}% {row['oos_wr']:>6.1f}% {row['oos_pf']:>6.2f} {row['oos_trades']:>4} {marker}")

    # 保存
    export_dir = 'data/backtests/b_neutral_short_ab'
    os.makedirs(export_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(export_dir, f'b_neutral_short_ab_{ts}.csv')
    rdf.to_csv(csv_path, index=False)
    print(f"\n  结果已保存: {csv_path}")


if __name__ == '__main__':
    main()
