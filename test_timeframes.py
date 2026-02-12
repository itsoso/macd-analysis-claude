"""
测试不同时间周期组合对策略的影响

目标: 回答"10分钟/15分钟/30分钟信号是否有用?"
方法: 
  1. 获取所有可用周期的数据 (10m ~ 8h)
  2. 对每个周期独立计算信号
  3. 测试不同权重组合
  4. 使用Walk-Forward验证避免过拟合
"""

import pandas as pd
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binance_fetcher import fetch_binance_klines
from indicators import add_all_indicators
from strategy_enhanced import analyze_signals_enhanced, get_signal_at, DEFAULT_SIG
from strategy_futures import FuturesEngine
from strategy_futures_v2 import get_trend_info
from strategy_futures_v4 import _calc_top_score, _calc_bottom_score
from strategy_futures_final import run_strategy


def fetch_extended_data():
    """获取扩展周期数据 (包括短周期)"""
    print("获取扩展周期数据...")
    data = {}
    configs = [
        ('10m', 10),   # 10分钟 (Binance不支持10m, 用15m)
        ('15m', 15),   # 15分钟
        ('30m', 30),   # 30分钟
        ('1h', 30),
        ('2h', 30),
        ('4h', 30),
        ('6h', 30),
        ('8h', 60),
    ]
    for tf, days in configs:
        try:
            df = fetch_binance_klines("ETHUSDT", interval=tf, days=days)
            if df is not None and len(df) > 50:
                df = add_all_indicators(df)
                data[tf] = df
                print(f"  {tf}: {len(df)} 条 ({df.index[0]} ~ {df.index[-1]})")
        except Exception as e:
            print(f"  {tf}: 获取失败 - {e}")
    return data


def merge_signals_custom(signals_all, dt, weights):
    """自定义权重的信号合并"""
    sig = dict(DEFAULT_SIG)
    sig['top'] = 0
    sig['bottom'] = 0

    for tf, w in weights.items():
        if tf not in signals_all:
            continue
        s_tf = get_signal_at(signals_all[tf], dt)
        if not s_tf:
            continue
        sig['top'] += s_tf.get('top', 0) * w
        sig['bottom'] += s_tf.get('bottom', 0) * w
        # 合并布尔信号
        for k in DEFAULT_SIG:
            if isinstance(DEFAULT_SIG[k], bool) and s_tf.get(k):
                sig[k] = True
            elif isinstance(DEFAULT_SIG[k], (int, float)) and k not in ('top', 'bottom'):
                sig[k] = max(sig.get(k, 0), s_tf.get(k, 0))
            elif isinstance(DEFAULT_SIG[k], str) and s_tf.get(k):
                sig[k] = s_tf[k]
    return sig


def run_strategy_custom_weights(data, signals_all, config, weights):
    """使用自定义权重运行策略"""
    eng = FuturesEngine(config['name'], max_leverage=config.get('max_lev', 3))
    main_tf = config.get('main_tf', '1h')
    main_df = data[main_tf]
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[0]

    eng.max_single_margin = eng.initial_total * config.get('single_pct', 0.15)
    eng.max_margin_total = eng.initial_total * config.get('total_pct', 0.30)

    ts_sell = config.get('ts_sell', 15)
    ts_short = config.get('ts_short', 20)
    bs_close = config.get('bs_close', 35)
    bs_conflict = config.get('bs_conflict', 25)
    sell_pct = config.get('sell_pct', 0.80)
    margin_use = config.get('margin_use', 0.80)
    lev = config.get('lev', 3)
    trail_start = config.get('trail_start', 0.5)
    trail_keep = config.get('trail_keep', 0.6)
    sl = config.get('sl', -0.40)
    tp = config.get('tp', 1.5)
    max_hold = config.get('max_hold', 336)
    cd_val = config.get('cooldown', 8)

    max_pnl_r = 0; cd = 0; bars_held = 0

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        eng.check_liquidation(price, dt)
        eng.charge_funding(price, dt)
        if cd > 0: cd -= 1

        trend = get_trend_info(data, dt, price)
        sig = merge_signals_custom(signals_all, dt, weights)
        ts, ts_parts = _calc_top_score(sig, trend)
        bs = _calc_bottom_score(sig, trend)

        if ts >= ts_sell and eng.spot_eth * price > 1000:
            eng.spot_sell(price, dt, sell_pct, f"卖出 TS={ts:.0f}")

        signal_clean = bs < bs_conflict
        if cd == 0 and ts >= ts_short and signal_clean and not eng.futures_short:
            margin = eng.available_margin() * margin_use
            actual_lev = min(lev, eng.max_leverage)
            if ts >= 35: actual_lev = min(lev, eng.max_leverage)
            elif ts >= 25: actual_lev = min(2, eng.max_leverage)
            else: actual_lev = min(2, eng.max_leverage)
            eng.open_short(price, dt, margin, actual_lev, f"做空 {actual_lev}x TS={ts:.0f} BS={bs:.0f}")
            max_pnl_r = 0; bars_held = 0; cd = cd_val

        if eng.futures_short:
            bars_held += 1
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
            if pnl_r >= tp:
                eng.close_short(price, dt, f"止盈 +{pnl_r*100:.0f}%")
                max_pnl_r = 0; cd = cd_val; bars_held = 0
            elif pnl_r > max_pnl_r:
                max_pnl_r = pnl_r
            elif max_pnl_r >= trail_start:
                if pnl_r < max_pnl_r * trail_keep and eng.futures_short:
                    eng.close_short(price, dt, f"追踪止盈")
                    max_pnl_r = 0; cd = cd_val; bars_held = 0
            if eng.futures_short and bs >= bs_close:
                eng.close_short(price, dt, f"底部信号 BS={bs:.0f}")
                max_pnl_r = 0; cd = cd_val * 3; bars_held = 0
            if eng.futures_short and pnl_r < sl:
                eng.close_short(price, dt, f"止损 {pnl_r*100:.0f}%")
                max_pnl_r = 0; cd = cd_val * 2; bars_held = 0
            if eng.futures_short and bars_held >= max_hold:
                eng.close_short(price, dt, f"超时平仓")
                max_pnl_r = 0; cd = cd_val; bars_held = 0

        if idx % 4 == 0: eng.record_history(dt, price)

    if eng.futures_short: eng.close_short(main_df['close'].iloc[-1], main_df.index[-1], "期末平仓")
    return eng.get_result(main_df)


def main():
    data = fetch_extended_data()

    # 计算各周期信号
    print("\n计算各周期信号...")
    signal_configs = {
        '15m': 200,  '30m': 200,
        '1h': 168, '2h': 150, '4h': 120, '6h': 100, '8h': 90
    }
    signals_all = {}
    for tf, w in signal_configs.items():
        if tf in data and len(data[tf]) > w:
            signals_all[tf] = analyze_signals_enhanced(data[tf], w)
            print(f"  {tf}: {len(signals_all[tf])} 个信号点")
        else:
            print(f"  {tf}: 数据不足或缺失")

    # 基础配置 (书本理论, 不做优化)
    base_cfg = {
        'single_pct': 0.15, 'total_pct': 0.30,
        'ts_sell': 15, 'ts_short': 20,
        'bs_close': 35, 'bs_conflict': 25,
        'sell_pct': 0.80, 'margin_use': 0.80,
        'lev': 3, 'trail_start': 0.5, 'trail_keep': 0.6,
        'sl': -0.40, 'tp': 1.5, 'max_hold': 336, 'cooldown': 8,
    }

    # ================================================================
    # 测试1: 各单一周期的信号效果
    # ================================================================
    print(f"\n{'='*100}")
    print("  测试1: 各单一周期信号的效果 (仅使用该周期信号)")
    print(f"{'='*100}")

    for tf in ['15m', '30m', '1h', '2h', '4h', '6h', '8h']:
        if tf not in signals_all: continue
        weights = {tf: 1.0}
        cfg = {**base_cfg, 'name': f'单{tf}'}
        r = run_strategy_custom_weights(data, signals_all, cfg, weights)
        f = r.get('fees', {})
        print(f"  {tf:>4}: α={r['alpha']:+.2f}% 收益={r['strategy_return']:+.2f}% "
              f"回撤={r['max_drawdown']:.2f}% 交易={r['total_trades']}笔 费={f.get('total_costs',0):,.0f}")

    # ================================================================
    # 测试2: 不同周期组合
    # ================================================================
    print(f"\n{'='*100}")
    print("  测试2: 不同周期组合")
    print(f"{'='*100}")

    weight_combos = {
        'A: 当前(1h-8h)': {'4h': 1.0, '6h': 0.85, '8h': 0.5, '2h': 0.7, '1h': 0.3},
        'B: +15m': {'15m': 0.2, '4h': 1.0, '6h': 0.85, '8h': 0.5, '2h': 0.7, '1h': 0.3},
        'C: +30m': {'30m': 0.3, '4h': 1.0, '6h': 0.85, '8h': 0.5, '2h': 0.7, '1h': 0.3},
        'D: +15m+30m': {'15m': 0.2, '30m': 0.3, '4h': 1.0, '6h': 0.85, '8h': 0.5, '2h': 0.7, '1h': 0.3},
        'E: 短周期主导': {'15m': 0.5, '30m': 0.7, '1h': 1.0, '2h': 0.8, '4h': 0.5},
        'F: 长周期主导': {'4h': 1.0, '6h': 1.0, '8h': 0.8, '2h': 0.5, '1h': 0.2},
        'G: 全周期均衡': {'15m': 0.3, '30m': 0.5, '1h': 0.7, '2h': 0.8, '4h': 1.0, '6h': 0.8, '8h': 0.6},
        'H: 纯短周期': {'15m': 0.8, '30m': 1.0, '1h': 0.6},
        'I: 纯4h': {'4h': 1.0},
        'J: 纯8h': {'8h': 1.0},
        'K: 15m+4h共振': {'15m': 0.5, '4h': 1.0},
        'L: 30m+4h共振': {'30m': 0.5, '4h': 1.0},
    }

    combo_results = []
    for name, weights in weight_combos.items():
        cfg = {**base_cfg, 'name': name}
        r = run_strategy_custom_weights(data, signals_all, cfg, weights)
        f = r.get('fees', {})
        combo_results.append((name, r))
        print(f"  {name:<24}: α={r['alpha']:+.2f}% 收益={r['strategy_return']:+.2f}% "
              f"回撤={r['max_drawdown']:.2f}% 交易={r['total_trades']}笔 费=${f.get('total_costs',0):,.0f}")

    # ================================================================
    # 测试3: 不同执行周期 (用15m/30m作为主循环)
    # ================================================================
    print(f"\n{'='*100}")
    print("  测试3: 不同执行周期 (在更短周期上执行交易)")
    print(f"{'='*100}")

    for main_tf in ['15m', '30m', '1h']:
        if main_tf not in data: continue
        weights = {'4h': 1.0, '6h': 0.85, '8h': 0.5, '2h': 0.7, '1h': 0.3}
        if main_tf in ['15m', '30m']:
            weights[main_tf] = 0.3
        cfg = {**base_cfg, 'name': f'执行{main_tf}', 'main_tf': main_tf}
        r = run_strategy_custom_weights(data, signals_all, cfg, weights)
        f = r.get('fees', {})
        print(f"  执行周期={main_tf}: α={r['alpha']:+.2f}% 收益={r['strategy_return']:+.2f}% "
              f"回撤={r['max_drawdown']:.2f}% 交易={r['total_trades']}笔 费=${f.get('total_costs',0):,.0f}")

    # ================================================================
    # 测试4: 信号到达时间分析
    # ================================================================
    print(f"\n{'='*100}")
    print("  测试4: 各周期信号首次出现时间 (检测顶部的速度)")
    print(f"{'='*100}")

    # 找ETH价格的顶部区域 (~$3,293, 约1/17)
    main_df = data['1h']
    peak_idx = main_df['close'].idxmax()
    peak_price = main_df['close'].max()
    print(f"  ETH顶部: {peak_idx} @${peak_price:,.0f}")

    for tf in ['15m', '30m', '1h', '2h', '4h', '6h', '8h']:
        if tf not in signals_all: continue
        first_top = None
        for t, s in signals_all[tf].items():
            if s.get('top', 0) >= 3 and t >= pd.Timestamp('2026-01-10'):
                first_top = (t, s['top'])
                break
        if first_top:
            delta = peak_idx - first_top[0]
            print(f"  {tf:>4}: 首个顶部信号 {str(first_top[0])[:16]} (top={first_top[1]:.1f}, "
                  f"比顶部{'早' if delta.total_seconds()>0 else '晚'} {abs(delta.total_seconds()/3600):.0f}h)")
        else:
            print(f"  {tf:>4}: 无足够强的顶部信号")

    # 排名汇总
    print(f"\n{'='*100}")
    print("  周期组合排名")
    print(f"{'='*100}")
    ranked = sorted(combo_results, key=lambda x: x[1]['alpha'], reverse=True)
    for i, (name, r) in enumerate(ranked):
        star = ' ★' if i == 0 else ''
        print(f"  #{i+1}: {name:<24} α={r['alpha']:+.2f}%{star}")

    # 保存结果
    output = {
        'description': '多时间周期信号对比测试',
        'combo_results': [{
            'name': name,
            'alpha': r['alpha'],
            'strategy_return': r['strategy_return'],
            'max_drawdown': r['max_drawdown'],
            'total_trades': r['total_trades'],
            'fees': r.get('fees', {}),
        } for name, r in combo_results],
        'ranking': [{'rank': i+1, 'name': name, 'alpha': r['alpha']}
                    for i, (name, r) in enumerate(ranked)],
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'timeframe_test_result.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果已保存: {path}")


if __name__ == '__main__':
    main()
