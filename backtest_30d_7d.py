"""
六书融合策略 · 30天 vs 7天 真实回测对比

使用optimize_six_book优化出的TOP策略配置,
分别在最近30天和最近7天的真实币安数据上回测。
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
    compute_signals_six, run_strategy, ALL_TIMEFRAMES
)
from strategy_futures import FuturesEngine


def fetch_data_for_tf(tf, days):
    """获取指定时间框架和天数的数据"""
    # 多取一些数据用于指标预热 (至少需要60根K线预热)
    fetch_days = days + 30  # 额外30天用于指标计算预热
    try:
        df = fetch_binance_klines("ETHUSDT", interval=tf, days=fetch_days)
        if df is not None and len(df) > 50:
            df = add_all_indicators(df)
            add_moving_averages(df, timeframe=tf)
            return df
    except Exception as e:
        print(f"  获取 {tf} 数据失败: {e}")
    return None


def run_single_backtest(tf, df, data_all, config, trade_days, label):
    """运行单次回测并返回详细结果"""
    signals = compute_signals_six(df, tf, data_all)
    
    # 时间周期映射
    tf_hours = {'10m': 1/6, '15m': 0.25, '30m': 0.5, '1h': 1, '2h': 2, '3h': 3,
                '4h': 4, '6h': 6, '8h': 8, '12h': 12, '16h': 16, '24h': 24}
    hours = tf_hours.get(tf, 1)
    
    # 调整持仓和冷却周期
    cfg = dict(config)
    cfg['short_max_hold'] = max(6, int(cfg.get('short_max_hold', 72) / hours))
    cfg['long_max_hold'] = max(6, int(cfg.get('long_max_hold', 72) / hours))
    cfg['cooldown'] = max(1, int(cfg.get('cooldown', 4) / hours))
    cfg['spot_cooldown'] = max(2, int(cfg.get('spot_cooldown', 12) / hours))
    cfg['name'] = label
    
    result = run_strategy(df, signals, cfg, tf=tf, trade_days=trade_days)
    return result, signals


def main():
    print("=" * 120)
    print("  六书融合策略 · 30天 vs 7天 真实回测对比")
    print("  数据源: 币安 ETH/USDT 真实K线 · 含手续费/滑点/资金费率")
    print("=" * 120)
    
    # ======================================================
    # 从优化结果中加载TOP策略配置
    # ======================================================
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'optimize_six_book_result.json')
    with open(result_path, 'r', encoding='utf-8') as f:
        opt_result = json.load(f)
    
    # 提取要测试的策略 (选取有代表性的不同策略)
    strategies = []
    seen_tags = set()
    
    for r in opt_result['global_top30']:
        tag = r['tag']
        tf = r['tf']
        key = f"{tf}_{tag}"
        if key not in seen_tags:
            seen_tags.add(key)
            strategies.append({
                'tf': tf,
                'tag': tag,
                'config': r['config'],
                'orig_alpha': r['alpha'],
            })
        if len(strategies) >= 10:
            break
    
    # 确保也包含不同融合模式的代表性策略
    for r in opt_result['global_top30']:
        mode = r['config'].get('fusion_mode', 'c6_veto')
        tf = r['tf']
        key = f"{tf}_{r['tag']}"
        if key not in seen_tags and mode not in [s['config'].get('fusion_mode', '') for s in strategies[:5]]:
            seen_tags.add(key)
            strategies.append({
                'tf': tf,
                'tag': r['tag'],
                'config': r['config'],
                'orig_alpha': r['alpha'],
            })
        if len(strategies) >= 12:
            break
    
    print(f"\n将测试 {len(strategies)} 种策略配置:")
    for i, s in enumerate(strategies):
        mode = s['config'].get('fusion_mode', '?')
        print(f"  #{i+1} [{s['tf']}] {s['tag']} (原α={s['orig_alpha']:+.2f}%, 模式={mode})")
    
    # ======================================================
    # 获取所需时间框架的数据
    # ======================================================
    needed_tfs = list(set(s['tf'] for s in strategies))
    # 也获取8h数据用于辅助信号
    if '8h' not in needed_tfs:
        needed_tfs.append('8h')
    
    print(f"\n[1/3] 获取数据 (时间框架: {', '.join(needed_tfs)})...")
    
    # 对于30天回测, 获取60天数据(多30天预热)
    # 对于7天回测, 获取37天数据(多30天预热)
    # 统一获取60天, 用trade_days参数控制实际回测区间
    data_all = {}
    for tf in sorted(needed_tfs, key=lambda x: ALL_TIMEFRAMES.index(x) if x in ALL_TIMEFRAMES else 99):
        print(f"  获取 {tf} 数据 (60天)...")
        df = fetch_data_for_tf(tf, 60)
        if df is not None:
            data_all[tf] = df
            print(f"    {tf}: {len(df)} 条K线, 时间范围: {df.index[0]} ~ {df.index[-1]}")
        else:
            print(f"    {tf}: 失败!")
    
    print(f"\n  数据获取完成, 可用时间框架: {', '.join(data_all.keys())}")
    
    # ======================================================
    # 分别在30天和7天上回测
    # ======================================================
    periods = [
        {'days': 30, 'label': '最近30天'},
        {'days': 7, 'label': '最近7天'},
    ]
    
    all_results = {}
    
    for period in periods:
        days = period['days']
        label = period['label']
        
        print(f"\n{'=' * 120}")
        print(f"  [回测] {label} (trade_days={days})")
        print(f"{'=' * 120}")
        
        period_results = []
        
        # 先计算Buy & Hold基准
        print(f"\n  各时间框架 Buy & Hold 基准:")
        bh_by_tf = {}
        for tf in data_all:
            df = data_all[tf]
            end_dt = df.index[-1]
            start_dt = end_dt - pd.Timedelta(days=days)
            start_idx = df.index.searchsorted(start_dt)
            if start_idx >= len(df):
                start_idx = 0
            start_price = df['close'].iloc[start_idx]
            end_price = df['close'].iloc[-1]
            bh_return = (end_price / start_price - 1) * 100
            bh_by_tf[tf] = {'start_price': start_price, 'end_price': end_price, 'bh_return': bh_return}
            print(f"    {tf}: ${start_price:,.2f} → ${end_price:,.2f} (BH={bh_return:+.2f}%)")
        
        print(f"\n  {'#':>3} {'时间框架':>8} {'策略标签':<40} {'Alpha':>10} {'策略收益':>12} "
              f"{'BH收益':>10} {'回撤':>8} {'交易':>6} {'强平':>4} {'费用':>10}")
        print('  ' + '-' * 130)
        
        for i, strat in enumerate(strategies):
            tf = strat['tf']
            if tf not in data_all:
                print(f"  #{i+1:>2} {tf:>8} {strat['tag']:<40} --- 数据不可用 ---")
                continue
            
            df = data_all[tf]
            config = dict(strat['config'])
            
            # 基准配置
            base_config = {
                'name': f"{strat['tag']}_{days}d",
                'fusion_mode': config.get('fusion_mode', 'c6_veto_4'),
                'veto_threshold': config.get('veto_threshold', 25),
                'single_pct': 0.20, 'total_pct': 0.50, 'lifetime_pct': 5.0,
                'sell_threshold': config.get('sell_threshold', 18),
                'buy_threshold': config.get('buy_threshold', 25),
                'short_threshold': config.get('short_threshold', 25),
                'long_threshold': config.get('long_threshold', 40),
                'close_short_bs': config.get('close_short_bs', 40),
                'close_long_ss': config.get('close_long_ss', 40),
                'sell_pct': config.get('sell_pct', 0.55),
                'margin_use': config.get('margin_use', 0.70),
                'lev': config.get('lev', 5),
                'max_lev': config.get('max_lev', 5),
                # SL/TP
                'short_sl': config.get('short_sl', -0.25),
                'short_tp': config.get('short_tp', 0.60),
                'short_trail': config.get('short_trail', 0.25),
                'short_max_hold': config.get('short_max_hold', 72),
                'long_sl': config.get('long_sl', -0.08),
                'long_tp': config.get('long_tp', 0.30),
                'long_trail': config.get('long_trail', 0.20),
                'long_max_hold': config.get('long_max_hold', 72),
                'trail_pullback': config.get('trail_pullback', 0.60),
                'cooldown': config.get('cooldown', 4),
                'spot_cooldown': config.get('spot_cooldown', 12),
                # Partial TP
                'use_partial_tp': config.get('use_partial_tp', False),
                'partial_tp_1': config.get('partial_tp_1', 0.20),
                'partial_tp_1_pct': config.get('partial_tp_1_pct', 0.30),
                # 二段止盈
                'use_partial_tp_2': config.get('use_partial_tp_2', False),
                'partial_tp_2': config.get('partial_tp_2', 0.50),
                'partial_tp_2_pct': config.get('partial_tp_2_pct', 0.30),
                # ATR
                'use_atr_sl': config.get('use_atr_sl', False),
                'atr_sl_mult': config.get('atr_sl_mult', 3.0),
                # KDJ相关
                'kdj_bonus': config.get('kdj_bonus', 0.09),
                'kdj_weight': config.get('kdj_weight', 0.15),
                'kdj_strong_mult': config.get('kdj_strong_mult', 1.25),
                'kdj_normal_mult': config.get('kdj_normal_mult', 1.12),
                'kdj_reverse_mult': config.get('kdj_reverse_mult', 0.70),
                'kdj_gate_threshold': config.get('kdj_gate_threshold', 10),
                'veto_dampen': config.get('veto_dampen', 0.30),
            }
            
            result, _ = run_single_backtest(tf, df, data_all, base_config, days,
                                            f"{strat['tag']}_{days}d")
            
            fees = result.get('fees', {})
            total_cost = fees.get('total_costs', 0)
            
            r_info = {
                'rank': i + 1,
                'tf': tf,
                'tag': strat['tag'],
                'alpha': result['alpha'],
                'strategy_return': result['strategy_return'],
                'buy_hold_return': result['buy_hold_return'],
                'max_drawdown': result['max_drawdown'],
                'total_trades': result['total_trades'],
                'liquidations': result['liquidations'],
                'total_cost': total_cost,
                'fees': fees,
                'orig_alpha': strat['orig_alpha'],
                'config': config,
                'trades': result.get('trades', []),
            }
            period_results.append(r_info)
            
            star = ' ★' if i == 0 else ''
            print(f"  #{i+1:>2} {tf:>8} {strat['tag']:<40} {result['alpha']:>+9.2f}% "
                  f"{result['strategy_return']:>+11.2f}% {result['buy_hold_return']:>+9.2f}% "
                  f"{result['max_drawdown']:>7.2f}% {result['total_trades']:>5} "
                  f"{result['liquidations']:>3} ${total_cost:>9,.0f}{star}")
        
        # 按Alpha排序
        period_results.sort(key=lambda x: x['alpha'], reverse=True)
        all_results[days] = period_results
        
        # 汇总
        print(f"\n  === {label} 汇总 ===")
        if period_results:
            best = period_results[0]
            worst = period_results[-1]
            avg_alpha = np.mean([r['alpha'] for r in period_results])
            avg_trades = np.mean([r['total_trades'] for r in period_results])
            
            print(f"  最优:     [{best['tf']}] {best['tag']} α={best['alpha']:+.2f}%")
            print(f"  最差:     [{worst['tf']}] {worst['tag']} α={worst['alpha']:+.2f}%")
            print(f"  平均Alpha: {avg_alpha:+.2f}%")
            print(f"  平均交易数: {avg_trades:.0f}")
    
    # ======================================================
    # 对比总结
    # ======================================================
    print(f"\n{'=' * 120}")
    print(f"  30天 vs 7天 对比总结")
    print(f"{'=' * 120}")
    
    r30 = all_results.get(30, [])
    r7 = all_results.get(7, [])
    
    print(f"\n  {'策略':<45} {'30天Alpha':>12} {'7天Alpha':>12} {'差异':>10}")
    print('  ' + '-' * 85)
    
    # 按策略名匹配
    for s30 in r30:
        tag = s30['tag']
        tf = s30['tf']
        s7_match = next((r for r in r7 if r['tag'] == tag and r['tf'] == tf), None)
        if s7_match:
            diff = s7_match['alpha'] - s30['alpha']
            print(f"  [{tf:>4}] {tag:<38} {s30['alpha']:>+11.2f}% {s7_match['alpha']:>+11.2f}% {diff:>+9.2f}%")
    
    # 总体统计
    if r30 and r7:
        print(f"\n  === 总体统计 ===")
        avg30 = np.mean([r['alpha'] for r in r30])
        avg7 = np.mean([r['alpha'] for r in r7])
        best30 = max(r['alpha'] for r in r30)
        best7 = max(r['alpha'] for r in r7)
        worst30 = min(r['alpha'] for r in r30)
        worst7 = min(r['alpha'] for r in r7)
        
        print(f"  {'指标':<20} {'30天':>15} {'7天':>15}")
        print('  ' + '-' * 55)
        print(f"  {'平均Alpha':<20} {avg30:>+14.2f}% {avg7:>+14.2f}%")
        print(f"  {'最优Alpha':<20} {best30:>+14.2f}% {best7:>+14.2f}%")
        print(f"  {'最差Alpha':<20} {worst30:>+14.2f}% {worst7:>+14.2f}%")
        print(f"  {'盈利策略数':<20} {sum(1 for r in r30 if r['alpha'] > 0):>15} "
              f"{sum(1 for r in r7 if r['alpha'] > 0):>15}")
        print(f"  {'平均交易数':<20} {np.mean([r['total_trades'] for r in r30]):>15.0f} "
              f"{np.mean([r['total_trades'] for r in r7]):>15.0f}")
        print(f"  {'平均费用':<20} ${np.mean([r['total_cost'] for r in r30]):>13,.0f} "
              f"${np.mean([r['total_cost'] for r in r7]):>13,.0f}")
    
    # ======================================================
    # 保存结果
    # ======================================================
    output = {
        'description': '六书融合策略 · 30天 vs 7天真实回测对比',
        'run_time': datetime.now().isoformat(),
        'data_source': '币安 ETH/USDT 真实K线',
        'fee_model': {
            'taker_fee': '0.05%',
            'slippage': '0.1%',
            'funding_rate': '±0.01%/8h',
            'liquidation_fee': '0.5%',
        },
        'results_30d': [{
            'rank': i + 1,
            'tf': r['tf'],
            'tag': r['tag'],
            'alpha': r['alpha'],
            'strategy_return': r['strategy_return'],
            'buy_hold_return': r['buy_hold_return'],
            'max_drawdown': r['max_drawdown'],
            'total_trades': r['total_trades'],
            'liquidations': r['liquidations'],
            'total_cost': r['total_cost'],
            'fees': r['fees'],
        } for i, r in enumerate(r30)],
        'results_7d': [{
            'rank': i + 1,
            'tf': r['tf'],
            'tag': r['tag'],
            'alpha': r['alpha'],
            'strategy_return': r['strategy_return'],
            'buy_hold_return': r['buy_hold_return'],
            'max_drawdown': r['max_drawdown'],
            'total_trades': r['total_trades'],
            'liquidations': r['liquidations'],
            'total_cost': r['total_cost'],
            'fees': r['fees'],
        } for i, r in enumerate(r7)],
    }
    
    # 添加汇总统计
    if r30 and r7:
        output['summary'] = {
            '30d': {
                'avg_alpha': round(np.mean([r['alpha'] for r in r30]), 2),
                'best_alpha': round(max(r['alpha'] for r in r30), 2),
                'worst_alpha': round(min(r['alpha'] for r in r30), 2),
                'profitable_count': sum(1 for r in r30 if r['alpha'] > 0),
                'total_count': len(r30),
            },
            '7d': {
                'avg_alpha': round(np.mean([r['alpha'] for r in r7]), 2),
                'best_alpha': round(max(r['alpha'] for r in r7), 2),
                'worst_alpha': round(min(r['alpha'] for r in r7), 2),
                'profitable_count': sum(1 for r in r7 if r['alpha'] > 0),
                'total_count': len(r7),
            },
        }
    
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'backtest_30d_7d_result.json')
    
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
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(clean_json(output), f, ensure_ascii=False, default=str, indent=2)
    
    print(f"\n结果已保存: {out_path}")
    
    # 最终结论
    print(f"\n{'=' * 120}")
    print(f"  最终结论")
    print(f"{'=' * 120}")
    
    if r30 and r7:
        best30_r = r30[0]
        best7_r = r7[0]
        
        print(f"\n  30天最优: [{best30_r['tf']}] {best30_r['tag']}")
        print(f"    Alpha: {best30_r['alpha']:+.2f}%  策略收益: {best30_r['strategy_return']:+.2f}%  "
              f"回撤: {best30_r['max_drawdown']:.2f}%  交易: {best30_r['total_trades']}  "
              f"费用: ${best30_r['total_cost']:,.0f}")
        
        print(f"\n  7天最优:  [{best7_r['tf']}] {best7_r['tag']}")
        print(f"    Alpha: {best7_r['alpha']:+.2f}%  策略收益: {best7_r['strategy_return']:+.2f}%  "
              f"回撤: {best7_r['max_drawdown']:.2f}%  交易: {best7_r['total_trades']}  "
              f"费用: ${best7_r['total_cost']:,.0f}")
        
        # 一致性分析
        consistent = sum(1 for s30 in r30 
                        for s7 in r7 
                        if s30['tag'] == s7['tag'] and s30['tf'] == s7['tf']
                        and (s30['alpha'] > 0) == (s7['alpha'] > 0))
        total_matched = sum(1 for s30 in r30 
                           for s7 in r7 
                           if s30['tag'] == s7['tag'] and s30['tf'] == s7['tf'])
        if total_matched > 0:
            print(f"\n  策略一致性(盈亏方向): {consistent}/{total_matched} "
                  f"({consistent/total_matched*100:.0f}%)")
    
    return output


if __name__ == '__main__':
    main()
