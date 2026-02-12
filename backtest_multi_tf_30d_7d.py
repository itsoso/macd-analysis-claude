"""
多周期联合决策 · 30天 vs 7天 真实回测对比

使用 optimize_six_book 优化出的最优策略配置,
在 多周期联合决策模式 下分别在最近30天和最近7天的真实币安数据上回测。
"""

import pandas as pd
import numpy as np
import json
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binance_fetcher import fetch_binance_klines
from indicators import add_all_indicators
from ma_indicators import add_moving_averages
from optimize_six_book import (
    compute_signals_six, run_strategy, run_strategy_multi_tf,
    _build_tf_score_index, calc_fusion_score_six, ALL_TIMEFRAMES
)
from strategy_futures import FuturesEngine


def fetch_data_for_tf(tf, days):
    """获取指定时间框架和天数的数据"""
    fetch_days = days + 30
    try:
        df = fetch_binance_klines("ETHUSDT", interval=tf, days=fetch_days)
        if df is not None and len(df) > 50:
            df = add_all_indicators(df)
            add_moving_averages(df, timeframe=tf)
            return df
    except Exception as e:
        print(f"  获取 {tf} 数据失败: {e}")
    return None


def main():
    print("=" * 120)
    print("  多周期联合决策 · 30天 vs 7天 真实回测对比")
    print("  数据源: 币安 ETH/USDT 真实K线 · 含手续费/滑点/资金费率")
    print("=" * 120)

    # ======================================================
    # 从优化结果中加载最优策略配置
    # ======================================================
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'optimize_six_book_result.json')
    if not os.path.exists(result_path):
        print("错误: optimize_six_book_result.json 不存在, 请先运行 optimize_six_book.py")
        return

    with open(result_path, 'r', encoding='utf-8') as f:
        opt_result = json.load(f)

    # 取全局最优配置
    global_best = opt_result.get('global_best', {})
    best_config = global_best.get('config', {})
    best_tf = global_best.get('tf', '1h')
    best_alpha = global_best.get('alpha', 0)

    print(f"\n  全局最优单TF: {best_tf} α={best_alpha:+.2f}%")
    print(f"  融合模式: {best_config.get('fusion_mode', 'c6_veto_4')}")

    # ======================================================
    # 定义多周期组合方案
    # ======================================================
    # 回测用到的所有TF
    all_tfs_needed = ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '24h']

    # 主TF候选
    primary_tf_candidates = ['1h', '2h', '4h']

    # TF组合方案
    multi_tf_combos = [
        ('核心周期', ['30m', '1h', '4h', '8h', '24h']),
        ('全周期',   ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '24h']),
        ('大周期(≥1h)', ['1h', '2h', '4h', '8h', '12h', '24h']),
        ('均衡搭配', ['15m', '1h', '4h', '12h']),
        ('中大周期', ['1h', '2h', '4h', '8h', '12h']),
        ('快慢双层', ['15m', '30m', '4h', '24h']),
    ]

    # 基础参数
    f12_base = {
        'single_pct': 0.20, 'total_pct': 0.50, 'lifetime_pct': 5.0,
        'sell_threshold': best_config.get('sell_threshold', 18),
        'buy_threshold': best_config.get('buy_threshold', 25),
        'short_threshold': best_config.get('short_threshold', 25),
        'long_threshold': best_config.get('long_threshold', 40),
        'close_short_bs': best_config.get('close_short_bs', 40),
        'close_long_ss': best_config.get('close_long_ss', 40),
        'sell_pct': best_config.get('sell_pct', 0.55),
        'margin_use': best_config.get('margin_use', 0.70),
        'lev': best_config.get('lev', 5),
        'max_lev': best_config.get('max_lev', 5),
        'short_sl': best_config.get('short_sl', -0.25),
        'short_tp': best_config.get('short_tp', 0.60),
        'short_trail': best_config.get('short_trail', 0.25),
        'short_max_hold': best_config.get('short_max_hold', 72),
        'long_sl': best_config.get('long_sl', -0.08),
        'long_tp': best_config.get('long_tp', 0.30),
        'long_trail': best_config.get('long_trail', 0.20),
        'long_max_hold': best_config.get('long_max_hold', 72),
        'trail_pullback': best_config.get('trail_pullback', 0.60),
        'cooldown': best_config.get('cooldown', 4),
        'spot_cooldown': best_config.get('spot_cooldown', 12),
        'use_partial_tp': best_config.get('use_partial_tp', False),
        'partial_tp_1': best_config.get('partial_tp_1', 0.20),
        'partial_tp_1_pct': best_config.get('partial_tp_1_pct', 0.30),
        'use_partial_tp_2': best_config.get('use_partial_tp_2', False),
        'partial_tp_2': best_config.get('partial_tp_2', 0.50),
        'partial_tp_2_pct': best_config.get('partial_tp_2_pct', 0.30),
        'use_atr_sl': best_config.get('use_atr_sl', False),
        'atr_sl_mult': best_config.get('atr_sl_mult', 3.0),
        'fusion_mode': best_config.get('fusion_mode', 'c6_veto_4'),
        'veto_threshold': best_config.get('veto_threshold', 25),
        'kdj_bonus': best_config.get('kdj_bonus', 0.09),
        'kdj_weight': best_config.get('kdj_weight', 0.15),
        'kdj_strong_mult': best_config.get('kdj_strong_mult', 1.25),
        'kdj_normal_mult': best_config.get('kdj_normal_mult', 1.12),
        'kdj_reverse_mult': best_config.get('kdj_reverse_mult', 0.70),
        'kdj_gate_threshold': best_config.get('kdj_gate_threshold', 10),
        'veto_dampen': best_config.get('veto_dampen', 0.30),
    }

    # ======================================================
    # 获取数据
    # ======================================================
    print(f"\n[1/4] 获取数据 (时间框架: {', '.join(all_tfs_needed)})...")

    all_data = {}
    for tf in all_tfs_needed:
        print(f"  获取 {tf} 数据 (90天)...")
        df = fetch_data_for_tf(tf, 90)
        if df is not None:
            all_data[tf] = df
            print(f"    {tf}: {len(df)} 条K线, {df.index[0]} ~ {df.index[-1]}")
        else:
            print(f"    {tf}: 失败!")

    available_tfs = list(all_data.keys())
    print(f"\n  可用时间框架: {', '.join(available_tfs)}")

    # 过滤组合中不可用的TF
    multi_tf_combos = [
        (name, [tf for tf in tfs if tf in available_tfs])
        for name, tfs in multi_tf_combos
    ]
    multi_tf_combos = [(n, t) for n, t in multi_tf_combos if len(t) >= 2]

    primary_tf_candidates = [tf for tf in primary_tf_candidates if tf in available_tfs]

    # ======================================================
    # 计算信号
    # ======================================================
    print(f"\n[2/4] 计算六维信号...")
    all_signals = {}
    for tf in available_tfs:
        print(f"  计算 {tf} 信号...")
        all_signals[tf] = compute_signals_six(all_data[tf], tf, all_data, max_bars=2000)
    print(f"  信号计算完成: {len(all_signals)} 个TF")

    # 构建评分索引
    print(f"\n[3/4] 构建TF评分索引...")
    tf_score_index = _build_tf_score_index(all_data, all_signals, available_tfs, f12_base)
    for tf in available_tfs:
        n_scores = len(tf_score_index.get(tf, {}))
        print(f"    {tf}: {n_scores} 个评分点")

    # ======================================================
    # 分别运行 30天 和 7天 回测
    # ======================================================
    tf_hours = {
        '10m': 1/6, '15m': 0.25, '30m': 0.5, '1h': 1, '2h': 2, '3h': 3,
        '4h': 4, '6h': 6, '8h': 8, '12h': 12, '16h': 16, '24h': 24,
    }

    periods = [
        {'days': 30, 'label': '最近30天'},
        {'days': 7,  'label': '最近7天'},
    ]

    all_period_results = {}

    # 先计算单TF基线 (30天和7天)
    single_tf_baselines = {}
    for period in periods:
        days = period['days']
        baselines = {}
        for ptf in primary_tf_candidates:
            if ptf not in all_data:
                continue
            df = all_data[ptf]
            sigs = all_signals[ptf]
            config = dict(f12_base)
            config['name'] = f'单TF_{ptf}_{days}d'
            tf_h = tf_hours.get(ptf, 1)
            config['short_max_hold'] = max(6, int(f12_base.get('short_max_hold', 72) / tf_h))
            config['long_max_hold'] = max(6, int(f12_base.get('long_max_hold', 72) / tf_h))
            config['cooldown'] = max(1, int(f12_base.get('cooldown', 4) / tf_h))
            config['spot_cooldown'] = max(2, int(f12_base.get('spot_cooldown', 12) / tf_h))
            r = run_strategy(df, sigs, config, tf=ptf, trade_days=days)
            baselines[ptf] = {
                'alpha': r['alpha'],
                'strategy_return': r['strategy_return'],
                'buy_hold_return': r['buy_hold_return'],
                'max_drawdown': r['max_drawdown'],
                'total_trades': r['total_trades'],
            }
        single_tf_baselines[days] = baselines

    for period in periods:
        days = period['days']
        label = period['label']

        print(f"\n{'=' * 120}")
        print(f"  [回测] {label} — 多周期联合决策 (trade_days={days})")
        print(f"{'=' * 120}")

        # Buy & Hold 基准
        bh_by_tf = {}
        for tf in available_tfs:
            df = all_data[tf]
            end_dt = df.index[-1]
            start_dt = end_dt - pd.Timedelta(days=days)
            start_idx = df.index.searchsorted(start_dt)
            if start_idx >= len(df):
                start_idx = 0
            sp = df['close'].iloc[start_idx]
            ep = df['close'].iloc[-1]
            bh_by_tf[tf] = round((ep / sp - 1) * 100, 2)
        bh_1h = bh_by_tf.get('1h', 0)

        # 单TF基线
        print(f"\n  单TF基线:")
        for ptf in primary_tf_candidates:
            bl = single_tf_baselines.get(days, {}).get(ptf, {})
            if bl:
                print(f"    {ptf}: α={bl['alpha']:+.2f}%  "
                      f"策略={bl['strategy_return']:+.2f}%  "
                      f"BH={bl['buy_hold_return']:+.2f}%  "
                      f"回撤={bl['max_drawdown']:.2f}%")

        # 多周期回测
        print(f"\n  {'方案':<20} {'主TF':>5} {'辅助TFs':<45} {'Alpha':>10} {'策略收益':>12} "
              f"{'BH':>8} {'回撤':>8} {'交易':>6} {'vs单TF':>10}")
        print('  ' + '-' * 130)

        period_results = []

        for combo_name, combo_tfs in multi_tf_combos:
            for ptf in primary_tf_candidates:
                if ptf not in all_data:
                    continue

                config = dict(f12_base)
                config['name'] = f'多TF_{combo_name}@{ptf}_{days}d'

                tf_h = tf_hours.get(ptf, 1)
                config['short_max_hold'] = max(6, int(f12_base.get('short_max_hold', 72) / tf_h))
                config['long_max_hold'] = max(6, int(f12_base.get('long_max_hold', 72) / tf_h))
                config['cooldown'] = max(1, int(f12_base.get('cooldown', 4) / tf_h))
                config['spot_cooldown'] = max(2, int(f12_base.get('spot_cooldown', 12) / tf_h))

                r = run_strategy_multi_tf(
                    all_data[ptf], tf_score_index, combo_tfs, config,
                    primary_tf=ptf, trade_days=days
                )

                fees = r.get('fees', {})
                baseline_alpha = single_tf_baselines.get(days, {}).get(ptf, {}).get('alpha', 0)
                vs_single = r['alpha'] - baseline_alpha

                entry = {
                    'combo_name': combo_name,
                    'primary_tf': ptf,
                    'decision_tfs': combo_tfs,
                    'alpha': r['alpha'],
                    'strategy_return': r['strategy_return'],
                    'buy_hold_return': r['buy_hold_return'],
                    'max_drawdown': r['max_drawdown'],
                    'total_trades': r['total_trades'],
                    'liquidations': r['liquidations'],
                    'total_cost': fees.get('total_costs', 0),
                    'fees': fees,
                    'vs_single_tf': round(vs_single, 2),
                }
                period_results.append(entry)

                marker = ' ★' if vs_single > 0 else ''
                print(f"  {combo_name:<20} {ptf:>5} {','.join(combo_tfs):<45} "
                      f"{r['alpha']:>+9.2f}% {r['strategy_return']:>+11.2f}% "
                      f"{r['buy_hold_return']:>+7.2f}% {r['max_drawdown']:>7.2f}% "
                      f"{r['total_trades']:>5} {vs_single:>+9.2f}%{marker}")

        period_results.sort(key=lambda x: x['alpha'], reverse=True)
        all_period_results[days] = period_results

        # 汇总
        print(f"\n  === {label} 多周期联合决策 TOP5 ===")
        for i, r in enumerate(period_results[:5]):
            print(f"    #{i+1} {r['combo_name']}@{r['primary_tf']} "
                  f"α={r['alpha']:+.2f}% (vs单TF: {r['vs_single_tf']:+.2f}%) "
                  f"回撤={r['max_drawdown']:.2f}% 交易={r['total_trades']}")

    # ======================================================
    # 30天 vs 7天 对比
    # ======================================================
    r30 = all_period_results.get(30, [])
    r7 = all_period_results.get(7, [])

    print(f"\n{'=' * 120}")
    print(f"  30天 vs 7天 多周期联合决策对比")
    print(f"{'=' * 120}")

    print(f"\n  {'方案':<30} {'30天Alpha':>12} {'7天Alpha':>12} {'差异':>10}")
    print('  ' + '-' * 70)

    for s30 in r30[:15]:
        key = f"{s30['combo_name']}@{s30['primary_tf']}"
        s7_match = next((r for r in r7
                         if r['combo_name'] == s30['combo_name']
                         and r['primary_tf'] == s30['primary_tf']), None)
        if s7_match:
            diff = s7_match['alpha'] - s30['alpha']
            print(f"  {key:<30} {s30['alpha']:>+11.2f}% {s7_match['alpha']:>+11.2f}% {diff:>+9.2f}%")

    # 总体统计
    if r30 and r7:
        print(f"\n  === 总体统计 ===")
        print(f"  {'指标':<25} {'30天':>15} {'7天':>15}")
        print('  ' + '-' * 60)
        print(f"  {'平均Alpha':<25} {np.mean([r['alpha'] for r in r30]):>+14.2f}% "
              f"{np.mean([r['alpha'] for r in r7]):>+14.2f}%")
        print(f"  {'最优Alpha':<25} {max(r['alpha'] for r in r30):>+14.2f}% "
              f"{max(r['alpha'] for r in r7):>+14.2f}%")
        print(f"  {'最差Alpha':<25} {min(r['alpha'] for r in r30):>+14.2f}% "
              f"{min(r['alpha'] for r in r7):>+14.2f}%")
        print(f"  {'盈利策略数':<25} "
              f"{sum(1 for r in r30 if r['alpha'] > 0):>15} "
              f"{sum(1 for r in r7 if r['alpha'] > 0):>15}")
        print(f"  {'平均交易数':<25} "
              f"{np.mean([r['total_trades'] for r in r30]):>15.0f} "
              f"{np.mean([r['total_trades'] for r in r7]):>15.0f}")

    # ======================================================
    # 保存结果
    # ======================================================
    def clean_json(obj):
        if isinstance(obj, dict):
            return {k: clean_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_json(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return str(obj)
        return obj

    output = {
        'description': '多周期联合决策 · 30天 vs 7天 真实回测对比',
        'run_time': datetime.now().isoformat(),
        'data_source': '币安 ETH/USDT 真实K线',
        'base_config': {
            'best_single_tf': best_tf,
            'best_single_alpha': best_alpha,
            'fusion_mode': f12_base.get('fusion_mode', 'c6_veto_4'),
        },
        'fee_model': {
            'taker_fee': '0.05%',
            'slippage': '0.1%',
            'funding_rate': '±0.01%/8h',
            'liquidation_fee': '0.5%',
        },
        'single_tf_baselines': single_tf_baselines,
        'results_30d': [{
            'rank': i + 1,
            'combo_name': r['combo_name'],
            'primary_tf': r['primary_tf'],
            'decision_tfs': r['decision_tfs'],
            'alpha': r['alpha'],
            'strategy_return': r['strategy_return'],
            'buy_hold_return': r['buy_hold_return'],
            'max_drawdown': r['max_drawdown'],
            'total_trades': r['total_trades'],
            'liquidations': r.get('liquidations', 0),
            'total_cost': r.get('total_cost', 0),
            'vs_single_tf': r.get('vs_single_tf', 0),
        } for i, r in enumerate(r30)],
        'results_7d': [{
            'rank': i + 1,
            'combo_name': r['combo_name'],
            'primary_tf': r['primary_tf'],
            'decision_tfs': r['decision_tfs'],
            'alpha': r['alpha'],
            'strategy_return': r['strategy_return'],
            'buy_hold_return': r['buy_hold_return'],
            'max_drawdown': r['max_drawdown'],
            'total_trades': r['total_trades'],
            'liquidations': r.get('liquidations', 0),
            'total_cost': r.get('total_cost', 0),
            'vs_single_tf': r.get('vs_single_tf', 0),
        } for i, r in enumerate(r7)],
    }

    if r30 and r7:
        output['summary'] = {
            '30d': {
                'avg_alpha': round(np.mean([r['alpha'] for r in r30]), 2),
                'best_alpha': round(max(r['alpha'] for r in r30), 2),
                'worst_alpha': round(min(r['alpha'] for r in r30), 2),
                'profitable_count': sum(1 for r in r30 if r['alpha'] > 0),
                'total_count': len(r30),
                'avg_trades': round(np.mean([r['total_trades'] for r in r30]), 1),
            },
            '7d': {
                'avg_alpha': round(np.mean([r['alpha'] for r in r7]), 2),
                'best_alpha': round(max(r['alpha'] for r in r7), 2),
                'worst_alpha': round(min(r['alpha'] for r in r7), 2),
                'profitable_count': sum(1 for r in r7 if r['alpha'] > 0),
                'total_count': len(r7),
                'avg_trades': round(np.mean([r['total_trades'] for r in r7]), 1),
            },
        }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'backtest_multi_tf_30d_7d_result.json')

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(clean_json(output), f, ensure_ascii=False, default=str, indent=2)

    print(f"\n结果已保存: {out_path}")
    return output


if __name__ == '__main__':
    main()
