#!/usr/bin/env python3
"""
A/B 回测验证: 对比 Fix 1-5 优化前后的策略表现。

Baseline (旧参数):
  - close_long_ss=40, close_short_bs=40
  - reverse_min_hold_short=0, reverse_min_hold_long=0
  - 无 close_signal_margin

Optimized (新参数):
  - close_long_ss=60, close_short_bs=60
  - reverse_min_hold_short=8, reverse_min_hold_long=8
  - close_signal_margin=20 (净差额要求)

在 IS (60天) 和 OOS (30天) 两个窗口上分别回测。
"""

import sys
import json
import datetime

from optimize_six_book import (
    fetch_multi_tf_data,
    compute_signals_six,
    run_strategy,
)


def build_config(name, overrides=None):
    """构建回测配置 (base = v5 最优参数)"""
    base = {
        'name': name,
        'fusion_mode': 'c6_veto_4',
        'veto_threshold': 25,
        'single_pct': 0.20, 'total_pct': 0.50, 'lifetime_pct': 5.0,
        'sell_threshold': 18, 'buy_threshold': 25,
        'short_threshold': 35, 'long_threshold': 30,
        'close_short_bs': 40, 'close_long_ss': 40,
        'sell_pct': 0.55, 'margin_use': 0.70, 'lev': 5, 'max_lev': 5,
        'short_sl': -0.18, 'short_tp': 0.50, 'short_trail': 0.25,
        'short_max_hold': 72,
        'long_sl': -0.08, 'long_tp': 0.40, 'long_trail': 0.20,
        'long_max_hold': 72,
        'trail_pullback': 0.60, 'cooldown': 4, 'spot_cooldown': 12,
        'use_partial_tp': True, 'partial_tp_1': 0.20, 'partial_tp_1_pct': 0.30,
        'reverse_min_hold_short': 0, 'reverse_min_hold_long': 0,
        'initial_usdt': 100000,
    }
    if overrides:
        base.update(overrides)
    return base


def format_result(r):
    fees = r.get('fees', {})
    trades = r.get('trades', [])
    wins = sum(1 for t in trades if t.get('pnl', 0) > 0
               and t.get('action', '').startswith('CLOSE'))
    losses = sum(1 for t in trades if t.get('pnl', 0) <= 0
                 and t.get('action', '').startswith('CLOSE'))
    total_close = wins + losses
    win_rate = (wins / total_close * 100) if total_close > 0 else 0

    avg_hold = 0
    hold_counts = []
    for t in trades:
        if t.get('action', '').startswith('CLOSE') and 'bars_held' in t:
            hold_counts.append(t['bars_held'])
    if hold_counts:
        avg_hold = sum(hold_counts) / len(hold_counts)

    return {
        'alpha': r['alpha'],
        'strategy_return': r['strategy_return'],
        'buy_hold_return': r['buy_hold_return'],
        'max_drawdown': r['max_drawdown'],
        'total_trades': r['total_trades'],
        'liquidations': r['liquidations'],
        'total_costs': fees.get('total_costs', 0),
        'fee_drag_pct': fees.get('fee_drag_pct', 0),
        'win_rate': round(win_rate, 1),
        'avg_hold_bars': round(avg_hold, 1),
        'final_total': r.get('final_total', 0),
    }


def print_comparison(label, baseline, optimized):
    print(f"\n{'=' * 80}")
    print(f"  {label}")
    print(f"{'=' * 80}")
    print(f"{'指标':<20} {'Baseline':>15} {'Optimized':>15} {'差异':>15}")
    print('-' * 65)

    metrics = [
        ('Alpha %', 'alpha', '+.2f'),
        ('策略收益 %', 'strategy_return', '+.2f'),
        ('BH收益 %', 'buy_hold_return', '+.2f'),
        ('最大回撤 %', 'max_drawdown', '.2f'),
        ('交易次数', 'total_trades', 'd'),
        ('强平次数', 'liquidations', 'd'),
        ('总费用 $', 'total_costs', ',.0f'),
        ('费率拖累 %', 'fee_drag_pct', '.2f'),
        ('胜率 %', 'win_rate', '.1f'),
        ('平均持仓bars', 'avg_hold_bars', '.1f'),
        ('最终总值 $', 'final_total', ',.0f'),
    ]

    for name, key, fmt in metrics:
        bv = baseline.get(key, 0)
        ov = optimized.get(key, 0)
        diff = ov - bv
        bv_s = format(bv, fmt)
        ov_s = format(ov, fmt)
        diff_s = format(diff, fmt)
        if diff > 0:
            diff_s = '+' + diff_s
        print(f"  {name:<18} {bv_s:>15} {ov_s:>15} {diff_s:>15}")


def main():
    tf = '1h'
    is_days = 60
    oos_days = 30
    total_days = is_days + oos_days

    print("=" * 80)
    print("  A/B 回测验证: Fix 1-5 参数优化效果")
    print(f"  时间框架: {tf} | IS: {is_days}天 | OOS: {oos_days}天")
    print("=" * 80)

    # 获取数据
    print(f"\n[1/4] 获取 {total_days} 天数据...")
    all_data = fetch_multi_tf_data([tf], days=total_days)
    if tf not in all_data:
        print(f"错误: 无法获取 {tf} 数据")
        sys.exit(1)
    df = all_data[tf]
    print(f"  数据量: {len(df)} 条K线")

    # 计算信号
    print(f"\n[2/4] 计算六维信号...")
    signals = compute_signals_six(df, tf, all_data)

    # 时间窗口
    end_dt = df.index[-1]
    oos_start = end_dt - datetime.timedelta(days=oos_days)
    is_start = oos_start - datetime.timedelta(days=is_days)

    # 配置
    baseline_overrides = {
        'close_short_bs': 40,
        'close_long_ss': 40,
        'reverse_min_hold_short': 0,
        'reverse_min_hold_long': 0,
    }

    optimized_overrides = {
        'close_short_bs': 60,
        'close_long_ss': 60,
        'reverse_min_hold_short': 8,
        'reverse_min_hold_long': 8,
    }

    configs = {
        'baseline': build_config('baseline', baseline_overrides),
        'optimized': build_config('optimized', optimized_overrides),
    }

    # 运行回测
    results = {}
    for window_name, start, end in [
        ('IS', is_start, oos_start),
        ('OOS', oos_start, end_dt),
        ('FULL', is_start, end_dt),
    ]:
        results[window_name] = {}
        for cfg_name, cfg in configs.items():
            print(f"\n[3/4] 运行 {window_name} {cfg_name}...")
            r = run_strategy(
                df, signals, cfg, tf=tf, trade_days=0,
                trade_start_dt=start, trade_end_dt=end,
            )
            results[window_name][cfg_name] = format_result(r)

    # 打印对比
    print("\n" + "=" * 80)
    print("  回测结果对比")
    print("=" * 80)

    for window_name in ['IS', 'OOS', 'FULL']:
        print_comparison(
            f"{window_name} 窗口对比",
            results[window_name]['baseline'],
            results[window_name]['optimized'],
        )

    # 汇总
    print("\n" + "=" * 80)
    print("  汇总")
    print("=" * 80)
    for wn in ['IS', 'OOS', 'FULL']:
        ba = results[wn]['baseline']['alpha']
        oa = results[wn]['optimized']['alpha']
        diff = oa - ba
        marker = "✅" if diff > 0 else "⚠️"
        print(f"  {wn:<6} Alpha: baseline={ba:+.2f}% → optimized={oa:+.2f}%  "
              f"(差异={diff:+.2f}%)  {marker}")

    # 保存结果
    output = {
        'timestamp': datetime.datetime.now().isoformat(),
        'config': {
            'baseline': baseline_overrides,
            'optimized': optimized_overrides,
        },
        'results': results,
    }
    outfile = 'ab_validation_result.json'
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n详细结果已保存至 {outfile}")


if __name__ == '__main__':
    main()
