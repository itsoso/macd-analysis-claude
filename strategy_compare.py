"""
策略变体对比实验
基于同一数据集, 对比6种不同策略思路

策略A: 当前多周期融合 (baseline)
策略B: 4h专注策略 — 只用4h信号, 最优周期单打
策略C: 趋势跟随防御 — MA判断趋势方向, 顺势操作
策略D: 保守高阈值 — 大幅提高信号阈值+长冷却期, 减少交易
策略E: 渐进减仓 — 下跌趋势中逐步卖出ETH, 只在强背驰买回
策略F: 仓位再平衡 — 定期将ETH:USDT维持在目标比例
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


def fetch_all_data():
    """获取所有需要的数据"""
    print("获取数据...")
    data = {}
    configs = [
        ('1h', 30), ('2h', 30), ('4h', 30), ('6h', 30), ('8h', 60),
    ]
    for tf, days in configs:
        df = fetch_binance_klines("ETHUSDT", interval=tf, days=days)
        if df is not None and len(df) > 50:
            df = add_all_indicators(df)
            data[tf] = df
            print(f"  {tf}: {len(df)} 条")
    return data


def analyze_signals(df, window):
    """对一个周期做滚动背离分析, 返回信号序列"""
    signals = {}
    step = max(1, window // 8)
    i = window
    while i < len(df):
        window_df = df.iloc[max(0, i - window):i].copy()
        try:
            analyzer = ComprehensiveAnalyzer(window_df)
            results = analyzer.analyze_all()
            sc = results.get('comprehensive_score', {})
            recs = results.get('trade_recommendations', [])
            signals[df.index[i]] = {
                'top': sc.get('top_score', 0),
                'bottom': sc.get('bottom_score', 0),
                'exhaust_sell': any(r.get('action') == 'SELL_EXHAUSTION' for r in recs),
                'exhaust_buy': any(r.get('action') == 'BUY_EXHAUSTION' for r in recs),
            }
        except Exception:
            pass
        i += step
    return signals


def get_signal_at(signals_dict, dt):
    """获取指定时间的最近信号"""
    latest = None
    for t, s in signals_dict.items():
        if t <= dt:
            latest = s
        else:
            break
    return latest or {'top': 0, 'bottom': 0, 'exhaust_sell': False, 'exhaust_buy': False}


class BaseStrategy:
    """策略基类"""

    def __init__(self, name, initial_usdt=100000, initial_eth_value=100000,
                 commission=0.001, slippage=0.0005):
        self.name = name
        self.initial_usdt = initial_usdt
        self.initial_eth_value = initial_eth_value
        self.commission = commission
        self.slippage = slippage
        self.trades = []
        self.history = []

    def execute_buy(self, price, dt, usdt, eth_qty, ratio, reason):
        """执行买入"""
        invest = usdt * ratio
        if invest < 500:
            return usdt, eth_qty
        actual_p = price * (1 + self.slippage)
        fee = invest * self.commission
        qty = (invest - fee) / actual_p
        usdt -= invest
        eth_qty += qty
        self.trades.append({
            'time': dt.isoformat(), 'action': 'BUY', 'price': round(price, 2),
            'quantity': round(qty, 4), 'value': round(invest, 2), 'fee': round(fee, 2),
            'usdt_after': round(usdt, 2), 'eth_after': round(eth_qty, 4),
            'total': round(usdt + eth_qty * price, 2), 'reason': reason,
        })
        return usdt, eth_qty

    def execute_sell(self, price, dt, usdt, eth_qty, ratio, reason):
        """执行卖出"""
        sell_qty = eth_qty * ratio
        if sell_qty * price < 500:
            return usdt, eth_qty
        actual_p = price * (1 - self.slippage)
        revenue = sell_qty * actual_p
        fee = revenue * self.commission
        usdt += revenue - fee
        eth_qty -= sell_qty
        self.trades.append({
            'time': dt.isoformat(), 'action': 'SELL', 'price': round(price, 2),
            'quantity': round(sell_qty, 4), 'value': round(revenue, 2), 'fee': round(fee, 2),
            'usdt_after': round(usdt, 2), 'eth_after': round(eth_qty, 4),
            'total': round(usdt + eth_qty * price, 2), 'reason': reason,
        })
        return usdt, eth_qty

    def calc_result(self, main_df, first_price, final_usdt, final_eth):
        last_price = main_df['close'].iloc[-1]
        final_total = final_usdt + final_eth * last_price
        initial_total = self.initial_usdt + self.initial_eth_value
        bh_eth = self.initial_eth_value / first_price
        bh_total = self.initial_usdt + bh_eth * last_price

        # Max drawdown
        if self.history:
            totals = [h['total'] for h in self.history]
            peak = totals[0]
            max_dd = 0
            for t in totals:
                peak = max(peak, t)
                dd = (t - peak) / peak
                max_dd = min(max_dd, dd)
        else:
            max_dd = 0

        return {
            'name': self.name,
            'initial_total': initial_total,
            'final_total': round(final_total, 2),
            'strategy_return': round((final_total - initial_total) / initial_total * 100, 2),
            'buy_hold_return': round((bh_total - initial_total) / initial_total * 100, 2),
            'alpha': round((final_total - initial_total) / initial_total * 100 -
                           (bh_total - initial_total) / initial_total * 100, 2),
            'max_drawdown': round(max_dd * 100, 2),
            'total_trades': len(self.trades),
            'buy_trades': sum(1 for t in self.trades if t['action'] == 'BUY'),
            'sell_trades': sum(1 for t in self.trades if t['action'] == 'SELL'),
            'final_usdt': round(final_usdt, 2),
            'final_eth': round(final_eth, 4),
            'final_eth_value': round(final_eth * last_price, 2),
            'trades': self.trades,
            'history': self.history,
        }


def run_strategy_A(data, signals_all):
    """策略A: 多周期融合 (当前baseline)"""
    s = BaseStrategy("A: 多周期融合 (baseline)")
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eth_qty = s.initial_eth_value / first_price
    usdt = s.initial_usdt

    weights = {'4h': 1.0, '6h': 0.85, '8h': 0.6, '2h': 0.7, '1h': 0.45}
    cooldown = 0

    for idx in range(len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        if cooldown > 0: cooldown -= 1

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
            if (has_es or w_top >= 35) and eth_r > 0.25:
                ratio = 0.5 if has_es or w_top >= 60 else 0.3
                usdt, eth_qty = s.execute_sell(price, dt, usdt, eth_qty, ratio,
                    f"融合顶背离={w_top:.0f}")
                cooldown = 8
            elif (has_eb or w_bot >= 45) and eth_r < 0.6:
                ratio = 0.4 if has_eb or w_bot >= 60 else 0.25
                usdt, eth_qty = s.execute_buy(price, dt, usdt, eth_qty, ratio,
                    f"融合底背离={w_bot:.0f}")
                cooldown = 8

        if idx % 4 == 0:
            s.history.append({'time': dt.isoformat(), 'total': round(usdt + eth_qty * price, 2)})

    return s.calc_result(main_df, first_price, usdt, eth_qty)


def run_strategy_B(data, signals_all):
    """策略B: 4h专注策略 — 只看4h信号, 严格阈值"""
    s = BaseStrategy("B: 4h专注策略")
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eth_qty = s.initial_eth_value / first_price
    usdt = s.initial_usdt

    if '4h' not in signals_all:
        return s.calc_result(main_df, first_price, usdt, eth_qty)

    cooldown = 0

    for idx in range(len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        if cooldown > 0: cooldown -= 1

        sig = get_signal_at(signals_all['4h'], dt)
        eth_val = eth_qty * price
        total = usdt + eth_val
        eth_r = eth_val / total if total > 0 else 0

        if cooldown == 0:
            if sig['exhaust_sell'] and eth_r > 0.2:
                usdt, eth_qty = s.execute_sell(price, dt, usdt, eth_qty, 0.6,
                    "4h背驰卖出")
                cooldown = 24
            elif sig['top'] >= 40 and eth_r > 0.3:
                usdt, eth_qty = s.execute_sell(price, dt, usdt, eth_qty, 0.35,
                    f"4h顶背离={sig['top']:.0f}")
                cooldown = 24
            elif sig['exhaust_buy'] and eth_r < 0.7:
                usdt, eth_qty = s.execute_buy(price, dt, usdt, eth_qty, 0.5,
                    "4h背驰买入")
                cooldown = 24
            elif sig['bottom'] >= 55 and eth_r < 0.5:
                usdt, eth_qty = s.execute_buy(price, dt, usdt, eth_qty, 0.3,
                    f"4h底背离={sig['bottom']:.0f}")
                cooldown = 24

        if idx % 4 == 0:
            s.history.append({'time': dt.isoformat(), 'total': round(usdt + eth_qty * price, 2)})

    return s.calc_result(main_df, first_price, usdt, eth_qty)


def run_strategy_C(data, signals_all):
    """策略C: 趋势跟随防御 — MA50判断趋势, 下跌只卖不买, 上涨才买"""
    s = BaseStrategy("C: 趋势跟随防御")
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eth_qty = s.initial_eth_value / first_price
    usdt = s.initial_usdt

    # 计算4h的MA50作为趋势判断
    df_4h = data.get('4h', main_df)
    ma50 = df_4h['close'].rolling(50).mean()

    weights = {'4h': 1.0, '6h': 0.85, '2h': 0.7}
    cooldown = 0

    for idx in range(len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        if cooldown > 0: cooldown -= 1

        # 趋势判断: 从4h MA50获取
        is_downtrend = True
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
            if is_downtrend:
                # 下跌趋势: 积极卖出, 只在强背驰买入
                if (has_es or w_top >= 25) and eth_r > 0.15:
                    ratio = 0.5 if has_es else 0.3
                    usdt, eth_qty = s.execute_sell(price, dt, usdt, eth_qty, ratio,
                        f"下跌+顶背离={w_top:.0f}")
                    cooldown = 12
                elif has_eb and eth_r < 0.4:
                    # 只有背驰才在下跌中买入
                    usdt, eth_qty = s.execute_buy(price, dt, usdt, eth_qty, 0.2,
                        "下跌中背驰抄底")
                    cooldown = 16
            else:
                # 上涨趋势: 积极买入, 背离才卖
                if (has_eb or w_bot >= 40) and eth_r < 0.7:
                    ratio = 0.4 if has_eb else 0.25
                    usdt, eth_qty = s.execute_buy(price, dt, usdt, eth_qty, ratio,
                        f"上涨+底背离={w_bot:.0f}")
                    cooldown = 12
                elif (has_es or w_top >= 50) and eth_r > 0.3:
                    usdt, eth_qty = s.execute_sell(price, dt, usdt, eth_qty, 0.4,
                        f"上涨+顶背离={w_top:.0f}")
                    cooldown = 16

        if idx % 4 == 0:
            s.history.append({'time': dt.isoformat(), 'total': round(usdt + eth_qty * price, 2)})

    return s.calc_result(main_df, first_price, usdt, eth_qty)


def run_strategy_D(data, signals_all):
    """策略D: 保守高阈值 — 只在极强信号时操作, 长冷却期"""
    s = BaseStrategy("D: 保守高阈值")
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eth_qty = s.initial_eth_value / first_price
    usdt = s.initial_usdt

    weights = {'4h': 1.0, '6h': 0.85, '8h': 0.6, '2h': 0.7}
    cooldown = 0

    for idx in range(len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        if cooldown > 0: cooldown -= 1

        w_top = w_bot = 0
        has_es = has_eb = False
        exhaust_count = 0
        tw = 0
        for tf, w in weights.items():
            if tf in signals_all:
                sig = get_signal_at(signals_all[tf], dt)
                w_top += sig['top'] * w
                w_bot += sig['bottom'] * w
                if sig['exhaust_sell']: has_es = True; exhaust_count += 1
                if sig['exhaust_buy']: has_eb = True; exhaust_count += 1
                tw += w
        if tw > 0: w_top /= tw; w_bot /= tw

        eth_val = eth_qty * price
        total = usdt + eth_val
        eth_r = eth_val / total if total > 0 else 0

        if cooldown == 0:
            # 极高阈值, 只在多周期共振时操作
            if w_top >= 55 and eth_r > 0.2:
                ratio = 0.6 if has_es else 0.4
                usdt, eth_qty = s.execute_sell(price, dt, usdt, eth_qty, ratio,
                    f"强共振卖出={w_top:.0f}")
                cooldown = 36  # 36小时冷却
            elif w_bot >= 60 and eth_r < 0.6:
                ratio = 0.5 if has_eb else 0.3
                usdt, eth_qty = s.execute_buy(price, dt, usdt, eth_qty, ratio,
                    f"强共振买入={w_bot:.0f}")
                cooldown = 36

        if idx % 4 == 0:
            s.history.append({'time': dt.isoformat(), 'total': round(usdt + eth_qty * price, 2)})

    return s.calc_result(main_df, first_price, usdt, eth_qty)


def run_strategy_E(data, signals_all):
    """策略E: 渐进减仓 — 在背离信号出现时逐步卖出, 保留10%底仓, 背驰才加仓"""
    s = BaseStrategy("E: 渐进减仓")
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eth_qty = s.initial_eth_value / first_price
    usdt = s.initial_usdt

    weights = {'4h': 1.0, '6h': 0.85, '2h': 0.7, '1h': 0.45}
    cooldown = 0
    min_eth_ratio = 0.10  # 最少保留10%仓位

    for idx in range(len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        if cooldown > 0: cooldown -= 1

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
            # 任何顶背离信号都逐步减仓
            if w_top >= 20 and eth_r > min_eth_ratio + 0.05:
                # 卖出比例随信号强度递增
                if has_es or w_top >= 60:
                    sell_r = min(0.5, (eth_r - min_eth_ratio) * 0.8)
                elif w_top >= 40:
                    sell_r = min(0.3, (eth_r - min_eth_ratio) * 0.5)
                else:
                    sell_r = min(0.15, (eth_r - min_eth_ratio) * 0.3)

                if sell_r > 0.05:
                    usdt, eth_qty = s.execute_sell(price, dt, usdt, eth_qty, sell_r,
                        f"渐进减仓 T={w_top:.0f}")
                    cooldown = 6

            # 只有强背驰才买回
            if has_eb and eth_r < 0.4:
                usdt, eth_qty = s.execute_buy(price, dt, usdt, eth_qty, 0.3,
                    "背驰加仓")
                cooldown = 16
            elif w_bot >= 65 and eth_r < 0.3:
                usdt, eth_qty = s.execute_buy(price, dt, usdt, eth_qty, 0.2,
                    f"强底背离加仓={w_bot:.0f}")
                cooldown = 16

        if idx % 4 == 0:
            s.history.append({'time': dt.isoformat(), 'total': round(usdt + eth_qty * price, 2)})

    return s.calc_result(main_df, first_price, usdt, eth_qty)


def run_strategy_F(data, signals_all):
    """策略F: 动态再平衡 — 根据背离信号动态调整ETH目标比例"""
    s = BaseStrategy("F: 动态再平衡")
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eth_qty = s.initial_eth_value / first_price
    usdt = s.initial_usdt

    weights = {'4h': 1.0, '6h': 0.85, '2h': 0.7}
    target_ratio = 0.50  # 初始目标50%
    cooldown = 0

    for idx in range(len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        if cooldown > 0: cooldown -= 1

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

        # 动态调整目标仓位
        if has_es or w_top >= 50:
            target_ratio = max(0.10, target_ratio - 0.15)
        elif has_eb or w_bot >= 55:
            target_ratio = min(0.70, target_ratio + 0.15)
        else:
            # 缓慢向中性回归
            target_ratio += (0.40 - target_ratio) * 0.005

        eth_val = eth_qty * price
        total = usdt + eth_val
        current_ratio = eth_val / total if total > 0 else 0

        # 再平衡: 偏差超过8%才操作
        if cooldown == 0 and abs(current_ratio - target_ratio) > 0.08:
            if current_ratio > target_ratio:
                # 需要卖ETH
                sell_value = (current_ratio - target_ratio) * total
                sell_qty = sell_value / price
                sell_r = sell_qty / eth_qty if eth_qty > 0 else 0
                if sell_r > 0.05:
                    usdt, eth_qty = s.execute_sell(price, dt, usdt, eth_qty, min(sell_r, 0.5),
                        f"再平衡→{target_ratio*100:.0f}% T={w_top:.0f}")
                    cooldown = 12
            else:
                # 需要买ETH
                buy_value = (target_ratio - current_ratio) * total
                buy_r = buy_value / usdt if usdt > 0 else 0
                if buy_r > 0.05:
                    usdt, eth_qty = s.execute_buy(price, dt, usdt, eth_qty, min(buy_r, 0.5),
                        f"再平衡→{target_ratio*100:.0f}% B={w_bot:.0f}")
                    cooldown = 12

        if idx % 4 == 0:
            s.history.append({'time': dt.isoformat(), 'total': round(usdt + eth_qty * price, 2)})

    return s.calc_result(main_df, first_price, usdt, eth_qty)


def run_all():
    """运行全部策略并对比"""
    # 获取数据
    data = fetch_all_data()

    # 预计算各周期信号
    print("\n计算各周期信号...")
    signal_windows = {'1h': 168, '2h': 150, '4h': 120, '6h': 100, '8h': 90}
    signals_all = {}
    for tf, df in data.items():
        w = signal_windows.get(tf, 120)
        if len(df) > w:
            signals_all[tf] = analyze_signals(df, w)
            print(f"  {tf}: {len(signals_all[tf])} 个信号点")

    # 运行各策略
    print("\n" + "=" * 80)
    print("  运行 6 种策略变体...")
    print("=" * 80)

    results = []
    for name, func in [
        ("A", run_strategy_A),
        ("B", run_strategy_B),
        ("C", run_strategy_C),
        ("D", run_strategy_D),
        ("E", run_strategy_E),
        ("F", run_strategy_F),
    ]:
        print(f"\n>>> 策略 {name}...")
        r = func(data, signals_all)
        results.append(r)
        print(f"    收益: {r['strategy_return']:+.2f}% | 超额: {r['alpha']:+.2f}% | "
              f"回撤: {r['max_drawdown']:.2f}% | 交易: {r['total_trades']}笔")

    # 汇总对比
    print("\n\n" + "=" * 100)
    print("                    策略变体对比 (初始: 10万USDT + 10万USDT的ETH)")
    print("=" * 100)
    fmt = "{:<28} {:>10} {:>10} {:>10} {:>10} {:>8} {:>12}"
    print(fmt.format("策略", "策略收益", "买入持有", "超额收益", "最大回撤", "交易数", "最终资产"))
    print("-" * 100)

    for r in sorted(results, key=lambda x: x['alpha'], reverse=True):
        print(fmt.format(
            r['name'],
            f"{r['strategy_return']:+.2f}%",
            f"{r['buy_hold_return']:+.2f}%",
            f"{r['alpha']:+.2f}%",
            f"{r['max_drawdown']:.2f}%",
            str(r['total_trades']),
            f"${r['final_total']:,.0f}",
        ))
    print("=" * 100)

    # 保存结果
    output = {
        'strategies': [{
            'name': r['name'],
            'summary': {k: v for k, v in r.items() if k not in ('trades', 'history')},
            'trades': r['trades'],
            'history': r['history'],
        } for r in results]
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'strategy_compare_result.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果已保存: {output_path}")

    return output


if __name__ == '__main__':
    run_all()
