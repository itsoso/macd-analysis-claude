"""
策略E(渐进减仓)的深度优化
在其基础上测试多种参数组合, 寻找最优配置
同时尝试结合策略C(趋势跟随)的思路做混合策略
"""

import pandas as pd
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binance_fetcher import fetch_binance_klines
from indicators import add_all_indicators
from divergence import ComprehensiveAnalyzer
from strategy_compare import BaseStrategy, fetch_all_data, analyze_signals, get_signal_at


def run_E_variant(data, signals_all, name, params):
    """
    渐进减仓策略变体
    params: min_ratio, sell_trigger, sell_strong, sell_weak,
            buy_trigger_exhaust, buy_trigger_score, buy_ratio,
            sell_cd, buy_cd, use_trend
    """
    s = BaseStrategy(name)
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eth_qty = s.initial_eth_value / first_price
    usdt = s.initial_usdt

    weights = params.get('weights', {'4h': 1.0, '6h': 0.85, '2h': 0.7, '1h': 0.45})
    cooldown = 0
    min_eth_ratio = params.get('min_ratio', 0.10)
    use_trend = params.get('use_trend', False)

    # 趋势判断用4h MA
    df_4h = data.get('4h', main_df)
    ma50 = df_4h['close'].rolling(50).mean()

    for idx in range(len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        if cooldown > 0: cooldown -= 1

        # 趋势
        is_downtrend = True
        if use_trend:
            for i in range(len(df_4h) - 1, -1, -1):
                if df_4h.index[i] <= dt:
                    ma_val = ma50.iloc[i] if not pd.isna(ma50.iloc[i]) else price
                    is_downtrend = price < ma_val
                    break

        w_top = w_bot = 0
        has_es = has_eb = False
        tw = 0
        for tf, w in weights.items():
            if tf in signals_all:
                sig = get_signal_at(signals_all[tf], dt)
                w_top += sig['top'] * w
                w_bot += sig['bottom'] * w
                if sig['exhaust_sell']: has_es = True
                if sig['exhaust_buy']: has_eb = True
                tw += w
        if tw > 0: w_top /= tw; w_bot /= tw

        eth_val = eth_qty * price
        total = usdt + eth_val
        eth_r = eth_val / total if total > 0 else 0

        if cooldown == 0:
            sell_trigger = params.get('sell_trigger', 20)
            sell_strong = params.get('sell_strong', 0.5)
            sell_weak = params.get('sell_weak', 0.15)
            buy_ratio = params.get('buy_ratio', 0.3)
            sell_cd = params.get('sell_cd', 6)
            buy_cd = params.get('buy_cd', 16)

            # --- 卖出逻辑 ---
            if w_top >= sell_trigger and eth_r > min_eth_ratio + 0.05:
                available_r = eth_r - min_eth_ratio

                if use_trend and is_downtrend:
                    # 下跌趋势中更积极减仓
                    if has_es or w_top >= 60:
                        sell_r = min(sell_strong * 1.2, available_r * 0.9)
                    elif w_top >= 40:
                        sell_r = min(0.35, available_r * 0.6)
                    else:
                        sell_r = min(sell_weak * 1.5, available_r * 0.4)
                else:
                    if has_es or w_top >= 60:
                        sell_r = min(sell_strong, available_r * 0.8)
                    elif w_top >= 40:
                        sell_r = min(0.3, available_r * 0.5)
                    else:
                        sell_r = min(sell_weak, available_r * 0.3)

                if sell_r > 0.05:
                    usdt, eth_qty = s.execute_sell(price, dt, usdt, eth_qty, sell_r,
                        f"减仓 T={w_top:.0f} {'↓' if use_trend and is_downtrend else ''}")
                    cooldown = sell_cd

            # --- 买入逻辑 ---
            buy_exhaust = params.get('buy_trigger_exhaust', True)
            buy_score = params.get('buy_trigger_score', 65)

            if buy_exhaust and has_eb and eth_r < 0.4:
                if not use_trend or not is_downtrend:
                    usdt, eth_qty = s.execute_buy(price, dt, usdt, eth_qty, buy_ratio,
                        "背驰加仓")
                    cooldown = buy_cd
                elif use_trend and is_downtrend:
                    # 下跌中背驰, 小仓位
                    usdt, eth_qty = s.execute_buy(price, dt, usdt, eth_qty, buy_ratio * 0.5,
                        "下跌背驰小仓位")
                    cooldown = buy_cd

            elif w_bot >= buy_score and eth_r < 0.3:
                usdt, eth_qty = s.execute_buy(price, dt, usdt, eth_qty, buy_ratio * 0.7,
                    f"强底背离={w_bot:.0f}")
                cooldown = buy_cd

        if idx % 4 == 0:
            s.history.append({'time': dt.isoformat(), 'total': round(usdt + eth_qty * price, 2)})

    return s.calc_result(main_df, first_price, usdt, eth_qty)


def run_all_optimizations():
    """运行全部优化变体"""
    data = fetch_all_data()

    print("\n计算各周期信号...")
    signal_windows = {'1h': 168, '2h': 150, '4h': 120, '6h': 100, '8h': 90}
    signals_all = {}
    for tf, df in data.items():
        w = signal_windows.get(tf, 120)
        if len(df) > w:
            signals_all[tf] = analyze_signals(df, w)
            print(f"  {tf}: {len(signals_all[tf])} 个信号点")

    variants = [
        # --- 基准 ---
        ("E-base: 原始渐进减仓", {
            'min_ratio': 0.10, 'sell_trigger': 20, 'sell_strong': 0.5, 'sell_weak': 0.15,
            'buy_trigger_exhaust': True, 'buy_trigger_score': 65, 'buy_ratio': 0.3,
            'sell_cd': 6, 'buy_cd': 16, 'use_trend': False,
        }),

        # --- 卖出触发阈值变化 ---
        ("E1: 更低卖出阈值=15", {
            'min_ratio': 0.10, 'sell_trigger': 15, 'sell_strong': 0.5, 'sell_weak': 0.15,
            'buy_trigger_exhaust': True, 'buy_trigger_score': 65, 'buy_ratio': 0.3,
            'sell_cd': 4, 'buy_cd': 16, 'use_trend': False,
        }),
        ("E2: 更高卖出阈值=30", {
            'min_ratio': 0.10, 'sell_trigger': 30, 'sell_strong': 0.5, 'sell_weak': 0.15,
            'buy_trigger_exhaust': True, 'buy_trigger_score': 65, 'buy_ratio': 0.3,
            'sell_cd': 6, 'buy_cd': 16, 'use_trend': False,
        }),

        # --- 最小仓位变化 ---
        ("E3: 最低仓位=5%", {
            'min_ratio': 0.05, 'sell_trigger': 20, 'sell_strong': 0.55, 'sell_weak': 0.18,
            'buy_trigger_exhaust': True, 'buy_trigger_score': 65, 'buy_ratio': 0.3,
            'sell_cd': 6, 'buy_cd': 16, 'use_trend': False,
        }),
        ("E4: 最低仓位=20%", {
            'min_ratio': 0.20, 'sell_trigger': 20, 'sell_strong': 0.45, 'sell_weak': 0.12,
            'buy_trigger_exhaust': True, 'buy_trigger_score': 65, 'buy_ratio': 0.3,
            'sell_cd': 6, 'buy_cd': 16, 'use_trend': False,
        }),

        # --- 冷却期调整 ---
        ("E5: 短冷却=3h卖/10h买", {
            'min_ratio': 0.10, 'sell_trigger': 20, 'sell_strong': 0.5, 'sell_weak': 0.15,
            'buy_trigger_exhaust': True, 'buy_trigger_score': 65, 'buy_ratio': 0.3,
            'sell_cd': 3, 'buy_cd': 10, 'use_trend': False,
        }),
        ("E6: 长冷却=12h卖/24h买", {
            'min_ratio': 0.10, 'sell_trigger': 20, 'sell_strong': 0.5, 'sell_weak': 0.15,
            'buy_trigger_exhaust': True, 'buy_trigger_score': 65, 'buy_ratio': 0.3,
            'sell_cd': 12, 'buy_cd': 24, 'use_trend': False,
        }),

        # --- 趋势增强 ---
        ("E7: 趋势+渐进减仓", {
            'min_ratio': 0.10, 'sell_trigger': 20, 'sell_strong': 0.5, 'sell_weak': 0.15,
            'buy_trigger_exhaust': True, 'buy_trigger_score': 65, 'buy_ratio': 0.3,
            'sell_cd': 6, 'buy_cd': 16, 'use_trend': True,
        }),

        # --- 激进减仓 ---
        ("E8: 激进减仓(强卖0.7)", {
            'min_ratio': 0.05, 'sell_trigger': 15, 'sell_strong': 0.7, 'sell_weak': 0.25,
            'buy_trigger_exhaust': True, 'buy_trigger_score': 70, 'buy_ratio': 0.25,
            'sell_cd': 4, 'buy_cd': 20, 'use_trend': False,
        }),

        # --- 不买入变体 ---
        ("E9: 只卖不买", {
            'min_ratio': 0.05, 'sell_trigger': 18, 'sell_strong': 0.55, 'sell_weak': 0.18,
            'buy_trigger_exhaust': False, 'buy_trigger_score': 999, 'buy_ratio': 0,
            'sell_cd': 5, 'buy_cd': 999, 'use_trend': False,
        }),

        # --- 权重调整: 更重4h ---
        ("E10: 重度4h权重", {
            'min_ratio': 0.10, 'sell_trigger': 20, 'sell_strong': 0.5, 'sell_weak': 0.15,
            'buy_trigger_exhaust': True, 'buy_trigger_score': 65, 'buy_ratio': 0.3,
            'sell_cd': 6, 'buy_cd': 16, 'use_trend': False,
            'weights': {'4h': 1.5, '6h': 0.6, '2h': 0.4, '1h': 0.2},
        }),

        # --- 综合最优猜测 ---
        ("E-BEST: 综合最优", {
            'min_ratio': 0.05, 'sell_trigger': 18, 'sell_strong': 0.55, 'sell_weak': 0.20,
            'buy_trigger_exhaust': True, 'buy_trigger_score': 65, 'buy_ratio': 0.25,
            'sell_cd': 5, 'buy_cd': 18, 'use_trend': True,
            'weights': {'4h': 1.2, '6h': 0.9, '8h': 0.5, '2h': 0.6, '1h': 0.3},
        }),
    ]

    print(f"\n运行 {len(variants)} 种变体...")
    print("=" * 110)

    results = []
    for name, params in variants:
        r = run_E_variant(data, signals_all, name, params)
        results.append(r)
        print(f"  {name:<30} | 收益:{r['strategy_return']:>+8.2f}% | "
              f"超额:{r['alpha']:>+7.2f}% | 回撤:{r['max_drawdown']:>7.2f}% | "
              f"交易:{r['total_trades']:>3}笔 | 资产:${r['final_total']:>10,.0f}")

    # 排名
    print("\n\n" + "=" * 110)
    print("                     最终排名 (按超额收益)")
    print("=" * 110)
    fmt = "{:>3} {:<32} {:>10} {:>10} {:>10} {:>10} {:>8} {:>12}"
    print(fmt.format("#", "策略", "策略收益", "买入持有", "超额收益", "最大回撤", "交易数", "最终资产"))
    print("-" * 110)
    for rank, r in enumerate(sorted(results, key=lambda x: x['alpha'], reverse=True), 1):
        star = " ★" if rank == 1 else ""
        print(fmt.format(
            rank,
            r['name'] + star,
            f"{r['strategy_return']:+.2f}%",
            f"{r['buy_hold_return']:+.2f}%",
            f"{r['alpha']:+.2f}%",
            f"{r['max_drawdown']:.2f}%",
            str(r['total_trades']),
            f"${r['final_total']:,.0f}",
        ))
    print("=" * 110)

    # 保存结果 (包含6个大类 + 优化变体)
    output = {
        'optimization_results': [{
            'name': r['name'],
            'summary': {k: v for k, v in r.items() if k not in ('trades', 'history')},
            'trades': r['trades'],
            'history': r['history'],
        } for r in results],
        'best_strategy': sorted(results, key=lambda x: x['alpha'], reverse=True)[0]['name'],
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'strategy_optimize_result.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果已保存: {output_path}")

    return output


if __name__ == '__main__':
    run_all_optimizations()
