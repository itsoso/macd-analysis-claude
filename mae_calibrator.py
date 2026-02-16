"""
Phase 1a: MAE 分布驱动的 ATR 乘数校准

从回测交易记录中提取 MAE (Maximum Adverse Excursion) 数据,
按 regime x direction 分组统计, 用 P90 MAE 校准 ATR 乘数:

    atr_mult[regime] = quantile(|MAE_pct| for wins, 0.90) / avg_atr_pct

使用方式:
    # 1. 运行校准 (自动执行回测并分析)
    python mae_calibrator.py

    # 2. 仅分析已有交易记录
    python mae_calibrator.py --analyze-only trades.json

    # 3. 指定时间框架和天数
    python mae_calibrator.py --tf 1h --days 90
"""

import json
import os
import sys
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def analyze_mae_from_trades(trades, verbose=True):
    """
    从交易记录中分析 MAE 分布。

    Parameters
    ----------
    trades : list[dict]
        完整交易记录 (来自 FuturesEngine.trades), 必须包含:
        - action: CLOSE_SHORT / CLOSE_LONG / LIQUIDATED 等平仓动作
        - direction: short / long
        - regime_label: regime 标签
        - min_pnl_r: 仓位生命周期的最大逆向偏移 (MAE)
        - max_pnl_r: 仓位生命周期的最大顺向偏移 (MFE)
        - atr_pct: 入场时的 ATR 百分比
        - pnl: 已实现盈亏

    Returns
    -------
    dict
        包含每个 (regime, direction) 桶的统计信息和推荐 ATR 乘数
    """
    close_actions = {
        'CLOSE_SHORT', 'CLOSE_LONG', 'LIQUIDATED', 'PARTIAL_TP',
    }

    # 筛选有效的平仓记录 (必须有 MAE 数据)
    valid_trades = []
    for t in trades:
        action = str(t.get('action', ''))
        # 兼容中文动作名
        if action in close_actions or '平' in action or '止损' in action or '止盈' in action:
            min_pnl_r = t.get('min_pnl_r')
            if min_pnl_r is not None:
                valid_trades.append(t)

    if verbose:
        print(f"\n{'='*70}")
        print(f"MAE 分布分析")
        print(f"{'='*70}")
        print(f"总交易数: {len(trades)}, 有效平仓 (含 MAE): {len(valid_trades)}")

    if not valid_trades:
        print("  ⚠ 没有找到含 MAE 数据的平仓记录")
        print("  提示: 需要使用最新版本的回测引擎运行, 确保 min_pnl_r/max_pnl_r 字段存在")
        return {'buckets': {}, 'recommendations': {}}

    # 按 regime x direction 分桶
    buckets = defaultdict(lambda: {
        'wins': [], 'losses': [], 'all': [],
        'mae_wins': [], 'mae_losses': [], 'mae_all': [],
        'mfe_wins': [], 'mfe_losses': [], 'mfe_all': [],
        'atr_pcts': [],
        'pnl_r_list': [],
    })

    for t in valid_trades:
        regime = str(t.get('regime_label', 'unknown'))
        direction = str(t.get('direction', 'unknown'))
        key = (regime, direction)
        pnl = float(t.get('pnl', 0))
        min_pnl_r = float(t['min_pnl_r'])
        max_pnl_r = float(t.get('max_pnl_r', 0))
        atr_pct = t.get('atr_pct')
        margin = float(t.get('margin', 0)) or 1.0

        pnl_r = pnl / margin if margin > 0 else 0
        mae_abs = abs(min_pnl_r)
        mfe_abs = abs(max_pnl_r)

        bucket = buckets[key]
        bucket['all'].append(t)
        bucket['mae_all'].append(mae_abs)
        bucket['mfe_all'].append(mfe_abs)
        bucket['pnl_r_list'].append(pnl_r)
        if atr_pct is not None and float(atr_pct) > 0:
            bucket['atr_pcts'].append(float(atr_pct))

        if pnl > 0:
            bucket['wins'].append(t)
            bucket['mae_wins'].append(mae_abs)
            bucket['mfe_wins'].append(mfe_abs)
        else:
            bucket['losses'].append(t)
            bucket['mae_losses'].append(mae_abs)
            bucket['mfe_losses'].append(mfe_abs)

    # 也按纯 direction 和纯 regime 汇总
    direction_agg = defaultdict(lambda: {'mae_wins': [], 'mae_all': [], 'atr_pcts': [], 'n': 0})
    regime_agg = defaultdict(lambda: {'mae_wins': [], 'mae_all': [], 'atr_pcts': [], 'n': 0})
    for (regime, direction), b in buckets.items():
        for field in ['mae_wins', 'mae_all', 'atr_pcts']:
            direction_agg[direction][field].extend(b[field])
            regime_agg[regime][field].extend(b[field])
        direction_agg[direction]['n'] += len(b['all'])
        regime_agg[regime]['n'] += len(b['all'])

    results = {'buckets': {}, 'recommendations': {}, 'summary': {}}

    if verbose:
        print(f"\n{'─'*70}")
        print(f"  {'Regime':<18} {'Dir':<7} {'N':>4} {'Win':>4} {'Loss':>4} "
              f"{'MAE_P50':>8} {'MAE_P75':>8} {'MAE_P90':>8} "
              f"{'MFE_P50':>8} {'AvgATR':>8} {'Rec.Mult':>9}")
        print(f"{'─'*70}")

    for (regime, direction), b in sorted(buckets.items()):
        n_all = len(b['all'])
        n_wins = len(b['wins'])
        n_losses = len(b['losses'])

        # MAE 分位数 (用赢单的 MAE, 因为我们要设置"不被震出赢单"的止损)
        mae_source = b['mae_wins'] if len(b['mae_wins']) >= 5 else b['mae_all']
        mae_arr = np.array(mae_source) if mae_source else np.array([0.0])
        mae_p50 = float(np.percentile(mae_arr, 50))
        mae_p75 = float(np.percentile(mae_arr, 75))
        mae_p90 = float(np.percentile(mae_arr, 90))
        mae_p95 = float(np.percentile(mae_arr, 95))

        # MFE 分位数
        mfe_source = b['mfe_wins'] if len(b['mfe_wins']) >= 5 else b['mfe_all']
        mfe_arr = np.array(mfe_source) if mfe_source else np.array([0.0])
        mfe_p50 = float(np.percentile(mfe_arr, 50))

        # 平均 ATR
        avg_atr = float(np.mean(b['atr_pcts'])) if b['atr_pcts'] else 0.0

        # 推荐 ATR 乘数 = P90(MAE_wins) / avg_atr_pct
        # 含义: 止损设为 atr_mult * ATR, 能覆盖 90% 赢单的最大回撤
        rec_mult = mae_p90 / avg_atr if avg_atr > 0 else 3.0
        rec_mult = max(1.5, min(5.0, rec_mult))  # 限制合理范围

        bucket_key = f"{regime}_{direction}"
        results['buckets'][bucket_key] = {
            'regime': regime,
            'direction': direction,
            'n_total': n_all,
            'n_wins': n_wins,
            'n_losses': n_losses,
            'win_rate': round(n_wins / max(1, n_all), 4),
            'mae_p50': round(mae_p50, 6),
            'mae_p75': round(mae_p75, 6),
            'mae_p90': round(mae_p90, 6),
            'mae_p95': round(mae_p95, 6),
            'mfe_p50': round(mfe_p50, 6),
            'avg_atr_pct': round(avg_atr, 6),
            'recommended_atr_mult': round(rec_mult, 3),
            'mae_source': 'wins' if len(b['mae_wins']) >= 5 else 'all',
            'mae_source_n': len(mae_source),
        }

        # 推荐的 config 参数
        results['recommendations'][f'atr_sl_mult_{regime}'] = round(rec_mult, 2)
        # 用 P95 作为 floor, P50 作为 ceil
        p95_sl = -mae_p95 if mae_p95 > 0 else -0.25
        p50_sl = -mae_p50 if mae_p50 > 0 else -0.06
        results['recommendations'][f'atr_sl_floor_{regime}'] = round(max(-0.40, p95_sl), 4)
        results['recommendations'][f'atr_sl_ceil_{regime}'] = round(min(-0.03, p50_sl), 4)

        if verbose:
            marker = '✓' if len(b['mae_wins']) >= 5 else '△'
            print(f"  {regime:<18} {direction:<7} {n_all:>4} {n_wins:>4} {n_losses:>4} "
                  f"{mae_p50:>7.4f} {mae_p75:>7.4f} {mae_p90:>7.4f} "
                  f"{mfe_p50:>7.4f} {avg_atr:>7.4f} {rec_mult:>8.3f} {marker}")

    # 汇总统计
    all_mae_wins = []
    all_atr_pcts = []
    for b in buckets.values():
        all_mae_wins.extend(b['mae_wins'])
        all_atr_pcts.extend(b['atr_pcts'])

    if all_mae_wins:
        results['summary'] = {
            'total_valid_trades': len(valid_trades),
            'total_wins': len(all_mae_wins),
            'global_mae_p50': round(float(np.percentile(all_mae_wins, 50)), 6),
            'global_mae_p75': round(float(np.percentile(all_mae_wins, 75)), 6),
            'global_mae_p90': round(float(np.percentile(all_mae_wins, 90)), 6),
            'global_avg_atr': round(float(np.mean(all_atr_pcts)), 6) if all_atr_pcts else 0,
        }

    if verbose:
        print(f"\n{'─'*70}")
        print(f"  全局汇总: {len(valid_trades)} 笔有效, {len(all_mae_wins)} 笔赢单")
        if all_mae_wins:
            print(f"  全局 MAE P50={np.percentile(all_mae_wins, 50):.4f}, "
                  f"P75={np.percentile(all_mae_wins, 75):.4f}, "
                  f"P90={np.percentile(all_mae_wins, 90):.4f}")
        print(f"\n推荐参数 (复制到 config):")
        for k, v in sorted(results['recommendations'].items()):
            print(f"  '{k}': {v},")

    return results


def run_backtest_for_mae(tf='1h', days=90, config_overrides=None):
    """
    运行一次完整回测, 收集含 MAE 数据的交易记录。

    Parameters
    ----------
    tf : str
        时间框架
    days : int
        回测天数
    config_overrides : dict
        额外的配置覆盖

    Returns
    -------
    tuple (trades, result)
        trades: 完整交易记录列表
        result: 回测结果字典
    """
    from binance_fetcher import fetch_binance_klines
    from indicators import add_all_indicators
    from ma_indicators import add_moving_averages

    print(f"\n{'='*70}")
    print(f"运行回测: tf={tf}, days={days}")
    print(f"{'='*70}")

    # 获取数据
    df = fetch_binance_klines("ETHUSDT", interval=tf, days=days)
    if df is None or len(df) < 100:
        print(f"  数据不足 ({len(df) if df is not None else 0} bars), 无法回测")
        return [], {}

    df = add_all_indicators(df)
    add_moving_averages(df, timeframe=tf)
    data_all = {tf: df}
    print(f"  数据: {len(df)} bars, {df.index[0]} → {df.index[-1]}")

    # 加载最优配置
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'optimize_six_book_result.json')
    base_config = {}
    if os.path.exists(result_path):
        with open(result_path) as f:
            result_data = json.load(f)
        gb = result_data.get('global_best', {})
        if isinstance(gb, dict) and 'config' in gb:
            base_config = dict(gb['config'])
            print(f"  加载最优配置: {gb.get('tf', '?')} α={gb.get('alpha', '?')}%")

    # 确保 MAE 追踪相关设置
    base_config.setdefault('use_atr_sl', True)

    if config_overrides:
        base_config.update(config_overrides)

    # 运行回测
    from optimize_six_book import compute_signals_six, run_strategy
    signals = compute_signals_six(df, tf, data_all, max_bars=0)
    result = run_strategy(df, signals, base_config, tf=tf, trade_days=days)

    trades = result.get('trades', [])
    print(f"\n  回测完成: α={result.get('alpha', 'N/A')}%, "
          f"trades={result.get('total_trades', 0)}, "
          f"return={result.get('strategy_return', 'N/A')}%")

    # 统计 MAE 覆盖率
    close_actions = {'CLOSE_SHORT', 'CLOSE_LONG', 'LIQUIDATED'}
    n_closes = sum(1 for t in trades
                   if str(t.get('action', '')) in close_actions or '平' in str(t.get('action', '')))
    n_with_mae = sum(1 for t in trades
                     if t.get('min_pnl_r') is not None
                     and (str(t.get('action', '')) in close_actions or '平' in str(t.get('action', ''))))
    coverage = n_with_mae / max(1, n_closes)
    print(f"  MAE 覆盖率: {n_with_mae}/{n_closes} = {coverage:.1%}")

    return trades, result


def generate_config_patch(recommendations, current_config=None):
    """
    生成配置补丁, 与当前配置对比。

    Parameters
    ----------
    recommendations : dict
        MAE 分析推荐的参数
    current_config : dict
        当前配置 (用于对比)

    Returns
    -------
    dict
        配置补丁 (仅包含需要修改的参数)
    """
    if current_config is None:
        current_config = {}

    patch = {}
    for k, v in recommendations.items():
        current = current_config.get(k)
        if current is None or abs(float(current) - float(v)) > 0.01:
            patch[k] = v

    return patch


def main():
    import argparse
    parser = argparse.ArgumentParser(description='MAE 分布驱动的 ATR 乘数校准')
    parser.add_argument('--tf', default='1h', help='时间框架 (default: 1h)')
    parser.add_argument('--days', type=int, default=90, help='回测天数 (default: 90)')
    parser.add_argument('--analyze-only', type=str, default=None,
                        help='直接分析已有的交易记录文件 (JSON)')
    parser.add_argument('--output', type=str, default='mae_calibration_result.json',
                        help='输出文件路径')
    args = parser.parse_args()

    if args.analyze_only:
        print(f"从文件加载交易记录: {args.analyze_only}")
        with open(args.analyze_only) as f:
            data = json.load(f)
        if isinstance(data, list):
            trades = data
        elif isinstance(data, dict) and 'trades' in data:
            trades = data['trades']
        else:
            print("无法识别的文件格式")
            return
        results = analyze_mae_from_trades(trades)
    else:
        trades, result = run_backtest_for_mae(tf=args.tf, days=args.days)
        if not trades:
            print("回测未产生交易, 无法校准")
            return
        results = analyze_mae_from_trades(trades)

    # 保存结果
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    results['generated_at'] = datetime.now().isoformat()
    results['params'] = {'tf': args.tf, 'days': args.days}

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n校准结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
