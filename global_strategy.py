"""
全局多周期融合策略
初始配置: 100,000 USDT 现金 + 价值 100,000 USDT 的 ETH
策略: 综合多个周期背离信号, 加权投票决定买卖

核心思路:
  - 同时在多个周期上运行背离分析
  - 根据前期回测表现给各周期赋权重 (4h最优权重最高)
  - 多周期信号加权求和, 超过阈值才执行交易
  - 仓位管理: 可分批建仓/平仓, 不是全进全出
"""

import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binance_fetcher import fetch_binance_klines
from indicators import add_all_indicators
from divergence import ComprehensiveAnalyzer


# 各周期权重 (基于前期回测超额收益排名)
TF_WEIGHTS = {
    '4h':  1.0,    # 最优, 超额+37.9%
    '6h':  0.85,   # 第二, 超额+31.2%
    '8h':  0.60,   # 超额+15.9%
    '2h':  0.70,   # 超额+11.4%, 交易频率合适
    '1h':  0.45,   # 超额+7.9%
    '3h':  0.40,   # 超额+2.0%
    '30m': 0.20,   # 超额+1.3%
    '10m': 0.15,   # 超额+2.1%, 噪音大
    '15m': 0.10,   # 唯一跑输
}

# 各周期分析参数
TF_PARAMS = {
    '10m': {'window': 200, 'step': 50,  'days': 34},
    '15m': {'window': 200, 'step': 40,  'days': 30},
    '30m': {'window': 200, 'step': 30,  'days': 30},
    '1h':  {'window': 168, 'step': 24,  'days': 30},
    '2h':  {'window': 150, 'step': 18,  'days': 30},
    '3h':  {'window': 130, 'step': 15,  'days': 34},
    '4h':  {'window': 120, 'step': 12,  'days': 30},
    '6h':  {'window': 100, 'step': 8,   'days': 30},
    '8h':  {'window': 90,  'step': 6,   'days': 60},
}


class GlobalStrategy:
    """多周期融合全局策略"""

    def __init__(self,
                 initial_usdt: float = 100000.0,
                 initial_eth_value: float = 100000.0,
                 commission: float = 0.001,
                 slippage: float = 0.0005):
        self.initial_usdt = initial_usdt
        self.initial_eth_value = initial_eth_value
        self.commission = commission
        self.slippage = slippage

        # 交易记录
        self.trades = []
        self.portfolio_history = []
        self.tf_signals_log = {}  # 各周期信号记录

    def run(self) -> dict:
        """执行多周期融合回测"""
        print("\n" + "=" * 70)
        print("  全局多周期融合策略回测")
        print("  初始配置: {:,.0f} USDT + 价值 {:,.0f} USDT 的 ETH".format(
            self.initial_usdt, self.initial_eth_value))
        print("=" * 70)

        # === 第一步: 获取各周期数据并计算指标 ===
        tf_data = {}
        for tf in TF_WEIGHTS.keys():
            p = TF_PARAMS.get(tf)
            if not p:
                continue
            print(f"\n获取 {tf} 数据...")
            df = fetch_binance_klines("ETHUSDT", interval=tf, days=p['days'])
            if df is not None and len(df) >= p['window']:
                df = add_all_indicators(df)
                tf_data[tf] = df
                print(f"  {tf}: {len(df)} 条K线")
            else:
                print(f"  {tf}: 数据不足, 跳过")

        if not tf_data:
            print("无可用数据!")
            return {}

        # === 第二步: 使用1小时作为主时间轴执行交易 ===
        # 用1h K线作为主tick, 每小时检查一次信号
        main_tf = '1h'
        if main_tf not in tf_data:
            main_tf = list(tf_data.keys())[0]
        main_df = tf_data[main_tf]

        # 初始化仓位
        first_price = main_df['close'].iloc[0]
        eth_qty = self.initial_eth_value / first_price
        usdt = self.initial_usdt
        position_value = eth_qty * first_price  # ETH持仓价值

        print(f"\n初始状态: {usdt:,.0f} USDT + {eth_qty:.4f} ETH (@ ${first_price:.2f})")
        print(f"总资产: ${usdt + position_value:,.0f}")

        # === 第三步: 在各周期上做滚动分析 ===
        # 预计算各周期在各时间点的信号
        print("\n正在计算各周期背离信号...")
        tf_scores = {}  # {tf: [(datetime, top_score, bottom_score, recs), ...]}

        for tf, df in tf_data.items():
            p = TF_PARAMS[tf]
            scores = []
            i = p['window']
            while i < len(df):
                window_df = df.iloc[max(0, i - p['window']):i].copy()
                try:
                    analyzer = ComprehensiveAnalyzer(window_df)
                    results = analyzer.analyze_all()
                    sc = results.get('comprehensive_score', {})
                    recs = results.get('trade_recommendations', [])
                    scores.append({
                        'time': df.index[i],
                        'top_score': sc.get('top_score', 0),
                        'bottom_score': sc.get('bottom_score', 0),
                        'has_exhaustion_sell': any(r.get('action') == 'SELL_EXHAUSTION' for r in recs),
                        'has_exhaustion_buy': any(r.get('action') == 'BUY_EXHAUSTION' for r in recs),
                    })
                except Exception:
                    pass
                i += p['step']
            tf_scores[tf] = scores
            print(f"  {tf}: {len(scores)} 个分析点")

        # === 第四步: 逐小时执行交易决策 ===
        print("\n开始逐小时执行交易...")

        cooldown = 0  # 冷却计数器
        entry_price = first_price  # 记录买入均价

        for idx in range(len(main_df)):
            dt = main_df.index[idx]
            price = main_df['close'].iloc[idx]

            if cooldown > 0:
                cooldown -= 1

            # 获取当前时间各周期的最新信号
            weighted_top = 0
            weighted_bottom = 0
            has_sell_exhaust = False
            has_buy_exhaust = False
            signal_details = {}

            for tf, weight in TF_WEIGHTS.items():
                if tf not in tf_scores:
                    continue
                # 找到该周期在当前时间之前最近的分析点
                latest = None
                for s in tf_scores[tf]:
                    if s['time'] <= dt:
                        latest = s
                    else:
                        break

                if latest:
                    weighted_top += latest['top_score'] * weight
                    weighted_bottom += latest['bottom_score'] * weight
                    if latest['has_exhaustion_sell']:
                        has_sell_exhaust = True
                    if latest['has_exhaustion_buy']:
                        has_buy_exhaust = True
                    signal_details[tf] = {
                        'top': latest['top_score'],
                        'bottom': latest['bottom_score'],
                        'exhaust_s': latest['has_exhaustion_sell'],
                        'exhaust_b': latest['has_exhaustion_buy'],
                    }

            total_weight = sum(TF_WEIGHTS[tf] for tf in tf_scores.keys() if tf in TF_WEIGHTS)
            if total_weight > 0:
                norm_top = weighted_top / total_weight
                norm_bottom = weighted_bottom / total_weight
            else:
                norm_top = norm_bottom = 0

            # ==== 交易决策 ====
            current_eth_value = eth_qty * price
            total_assets = usdt + current_eth_value
            eth_ratio = current_eth_value / total_assets if total_assets > 0 else 0

            # 止损: ETH仓位亏损超过12%
            if eth_qty > 0 and entry_price > 0:
                pnl_pct = (price - entry_price) / entry_price
                if pnl_pct < -0.12 and cooldown == 0:
                    # 减仓50%
                    sell_qty = eth_qty * 0.5
                    revenue = sell_qty * price * (1 - self.slippage)
                    fee = revenue * self.commission
                    usdt += revenue - fee
                    eth_qty -= sell_qty
                    cooldown = 6
                    self.trades.append({
                        'time': dt.isoformat(),
                        'action': 'SELL',
                        'type': '止损减仓',
                        'price': round(price, 2),
                        'quantity': round(sell_qty, 4),
                        'value': round(revenue, 2),
                        'fee': round(fee, 2),
                        'usdt_after': round(usdt, 2),
                        'eth_after': round(eth_qty, 4),
                        'total_assets': round(usdt + eth_qty * price, 2),
                        'pnl_pct': round(pnl_pct * 100, 2),
                        'norm_top_score': round(norm_top, 1),
                        'norm_bottom_score': round(norm_bottom, 1),
                        'signal_tf': self._format_signals(signal_details),
                        'reason': f'ETH浮亏{pnl_pct*100:.1f}%, 止损减仓50%',
                    })
                    continue

            # 卖出信号 (卖点用背离, 宁早勿晚)
            sell_signal = False
            sell_reason = ''

            if has_sell_exhaust and eth_ratio > 0.2:
                sell_signal = True
                sell_reason = '多周期背驰卖出信号'
            elif norm_top >= 35 and eth_ratio > 0.3:
                sell_signal = True
                sell_reason = f'加权顶背离={norm_top:.0f}'
            elif norm_top >= 50 and eth_ratio > 0.2:
                sell_signal = True
                sell_reason = f'强顶背离={norm_top:.0f}'

            if sell_signal and cooldown == 0 and eth_qty > 0:
                # 卖出比例: 信号越强卖越多
                if has_sell_exhaust or norm_top >= 60:
                    sell_ratio = 0.6
                elif norm_top >= 45:
                    sell_ratio = 0.4
                else:
                    sell_ratio = 0.3

                sell_qty = eth_qty * sell_ratio
                actual_price = price * (1 - self.slippage)
                revenue = sell_qty * actual_price
                fee = revenue * self.commission
                usdt += revenue - fee
                eth_qty -= sell_qty
                cooldown = 8

                self.trades.append({
                    'time': dt.isoformat(),
                    'action': 'SELL',
                    'type': '背离卖出',
                    'price': round(price, 2),
                    'quantity': round(sell_qty, 4),
                    'value': round(revenue, 2),
                    'fee': round(fee, 2),
                    'usdt_after': round(usdt, 2),
                    'eth_after': round(eth_qty, 4),
                    'total_assets': round(usdt + eth_qty * price, 2),
                    'norm_top_score': round(norm_top, 1),
                    'norm_bottom_score': round(norm_bottom, 1),
                    'signal_tf': self._format_signals(signal_details),
                    'reason': sell_reason,
                })

            # 买入信号 (买点用背驰, 宁迟勿早)
            buy_signal = False
            buy_reason = ''

            if has_buy_exhaust and eth_ratio < 0.7:
                buy_signal = True
                buy_reason = '多周期背驰买入信号'
            elif norm_bottom >= 45 and eth_ratio < 0.5:
                buy_signal = True
                buy_reason = f'加权底背离={norm_bottom:.0f}'
            elif norm_bottom >= 60 and eth_ratio < 0.6:
                buy_signal = True
                buy_reason = f'强底背离={norm_bottom:.0f}'

            if buy_signal and cooldown == 0 and usdt > 1000:
                # 买入比例
                if has_buy_exhaust or norm_bottom >= 60:
                    buy_ratio = 0.5
                elif norm_bottom >= 45:
                    buy_ratio = 0.3
                else:
                    buy_ratio = 0.2

                invest = usdt * buy_ratio
                actual_price = price * (1 + self.slippage)
                fee = invest * self.commission
                buy_qty = (invest - fee) / actual_price
                eth_qty += buy_qty
                usdt -= invest
                entry_price = price  # 更新入场价
                cooldown = 8

                self.trades.append({
                    'time': dt.isoformat(),
                    'action': 'BUY',
                    'type': '背离买入',
                    'price': round(price, 2),
                    'quantity': round(buy_qty, 4),
                    'value': round(invest, 2),
                    'fee': round(fee, 2),
                    'usdt_after': round(usdt, 2),
                    'eth_after': round(eth_qty, 4),
                    'total_assets': round(usdt + eth_qty * price, 2),
                    'norm_top_score': round(norm_top, 1),
                    'norm_bottom_score': round(norm_bottom, 1),
                    'signal_tf': self._format_signals(signal_details),
                    'reason': buy_reason,
                })

            # 记录资产曲线
            if idx % 3 == 0 or idx == len(main_df) - 1:  # 每3小时记录一次
                self.portfolio_history.append({
                    'time': dt.isoformat(),
                    'usdt': round(usdt, 2),
                    'eth_qty': round(eth_qty, 4),
                    'eth_price': round(price, 2),
                    'eth_value': round(eth_qty * price, 2),
                    'total': round(usdt + eth_qty * price, 2),
                    'eth_ratio': round(eth_ratio * 100, 1),
                })

        # === 第五步: 生成报告 ===
        return self._generate_report(main_df, first_price, eth_qty, usdt)

    def _format_signals(self, details):
        """格式化各周期信号"""
        parts = []
        for tf in ['4h', '6h', '2h', '8h', '1h', '3h', '30m']:
            if tf in details:
                d = details[tf]
                flags = ''
                if d['exhaust_s']:
                    flags += '⚡卖'
                if d['exhaust_b']:
                    flags += '⚡买'
                parts.append(f"{tf}:T{d['top']:.0f}/B{d['bottom']:.0f}{flags}")
        return ' | '.join(parts[:5])  # 最多显示5个

    def _generate_report(self, main_df, first_price, final_eth, final_usdt):
        """生成报告"""
        last_price = main_df['close'].iloc[-1]
        final_total = final_usdt + final_eth * last_price
        initial_total = self.initial_usdt + self.initial_eth_value

        # 买入持有: 保持初始配置不动
        bh_eth_qty = self.initial_eth_value / first_price
        bh_total = self.initial_usdt + bh_eth_qty * last_price

        strategy_return = (final_total - initial_total) / initial_total * 100
        bh_return = (bh_total - initial_total) / initial_total * 100
        alpha = strategy_return - bh_return

        # 最大回撤
        if self.portfolio_history:
            totals = [p['total'] for p in self.portfolio_history]
            peak = totals[0]
            max_dd = 0
            for t in totals:
                if t > peak:
                    peak = t
                dd = (t - peak) / peak
                if dd < max_dd:
                    max_dd = dd
        else:
            max_dd = 0

        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']

        report = {
            'summary': {
                'initial_usdt': self.initial_usdt,
                'initial_eth_value': self.initial_eth_value,
                'initial_total': initial_total,
                'final_usdt': round(final_usdt, 2),
                'final_eth_qty': round(final_eth, 4),
                'final_eth_price': round(last_price, 2),
                'final_eth_value': round(final_eth * last_price, 2),
                'final_total': round(final_total, 2),
                'strategy_return': round(strategy_return, 2),
                'buy_hold_return': round(bh_return, 2),
                'alpha': round(alpha, 2),
                'max_drawdown': round(max_dd * 100, 2),
                'total_buy_trades': len(buy_trades),
                'total_sell_trades': len(sell_trades),
                'eth_price_change': round((last_price - first_price) / first_price * 100, 2),
            },
            'trades': self.trades,
            'portfolio_history': self.portfolio_history,
            'tf_weights': TF_WEIGHTS,
        }

        # 打印
        print("\n" + "=" * 70)
        print("              全局策略回测结果")
        print("=" * 70)
        print(f"  初始: {self.initial_usdt:>12,.0f} USDT + ETH价值 {self.initial_eth_value:>10,.0f} USDT")
        print(f"  总计: {initial_total:>12,.0f} USDT")
        print(f"  ─────────────────────────────────────")
        print(f"  终值: {final_usdt:>12,.2f} USDT + {final_eth:.4f} ETH")
        print(f"  ETH价格: ${last_price:,.2f} (变化 {(last_price-first_price)/first_price*100:+.1f}%)")
        print(f"  ETH持仓价值: ${final_eth * last_price:>10,.2f}")
        print(f"  总资产: {final_total:>12,.2f} USDT")
        print(f"  ─────────────────────────────────────")
        print(f"  策略收益:     {strategy_return:+.2f}%")
        print(f"  买入持有收益: {bh_return:+.2f}%")
        print(f"  超额收益:     {alpha:+.2f}%")
        print(f"  最大回撤:     {max_dd*100:.2f}%")
        print(f"  买入次数:     {len(buy_trades)}")
        print(f"  卖出次数:     {len(sell_trades)}")
        print("=" * 70)

        if self.trades:
            print("\n交易明细:")
            print(f"{'时间':>20} {'操作':>6} {'类型':>10} {'价格':>10} {'数量':>10} "
                  f"{'金额':>12} {'总资产':>12} {'原因'}")
            print("-" * 110)
            for t in self.trades:
                print(f"{t['time'][:16]:>20} {t['action']:>6} {t['type']:>10} "
                      f"${t['price']:>9,.2f} {t['quantity']:>9.4f} "
                      f"${t['value']:>11,.2f} ${t['total_assets']:>11,.2f} {t['reason']}")

        return report


def run_global_strategy():
    """执行全局策略"""
    strategy = GlobalStrategy(
        initial_usdt=100000.0,
        initial_eth_value=100000.0,
        commission=0.001,
        slippage=0.0005,
    )
    report = strategy.run()

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'global_strategy_result.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果已保存: {output_path}")

    return report


if __name__ == '__main__':
    run_global_strategy()
