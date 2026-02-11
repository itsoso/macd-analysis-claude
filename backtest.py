"""
回测引擎
基于《背离技术分析》背离/背驰信号的交易策略回测

策略逻辑 (遵循书中 "卖点用背离, 买点用背驰" 原则):
  买入条件 (严格): 综合底背离评分>=65 或 背驰买入信号
  卖出条件 (宽松): 综合顶背离评分>=45 或 背驰卖出信号 或 止损/止盈
  冷却期: 每次交易后至少等待 cooldown 根K线
  止损: -5%  止盈: +8%
"""

import pandas as pd
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import add_all_indicators
from divergence import ComprehensiveAnalyzer


class BacktestEngine:
    """背离策略回测引擎"""

    def __init__(self, df: pd.DataFrame,
                 initial_capital: float = 10000.0,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 position_size: float = 0.95,
                 stop_loss: float = 0.05,
                 take_profit: float = 0.08,
                 cooldown: int = 30,
                 buy_threshold: float = 65,
                 sell_threshold: float = 45):
        """
        参数:
            df: K线数据
            initial_capital: 初始资金 (USDT)
            commission: 手续费率 (0.1%)
            slippage: 滑点 (0.05%)
            position_size: 仓位比例 (95%)
            stop_loss: 止损比例 (5%)
            take_profit: 止盈比例 (8%)
            cooldown: 交易冷却K线数
            buy_threshold: 买入综合评分阈值
            sell_threshold: 卖出综合评分阈值
        """
        self.raw_df = df.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.cooldown = cooldown
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        # 状态
        self.cash = initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.trades = []
        self.equity_curve = []
        self.signals_log = []
        self.bars_since_trade = 999  # 距离上次交易的K线数

    def run(self, window_size: int = 200, step: int = 50) -> dict:
        """
        执行滚动窗口回测

        参数:
            window_size: 分析窗口大小
            step: 滚动步长 (每step根K线重新计算一次信号)
        """
        df = self.raw_df.copy()
        df = add_all_indicators(df)
        n = len(df)

        print(f"\n回测参数: 初始资金={self.initial_capital} USDT, "
              f"手续费={self.commission*100}%, 滑点={self.slippage*100}%")
        print(f"止损={self.stop_loss*100}%, 止盈={self.take_profit*100}%, "
              f"冷却期={self.cooldown}根K线")
        print(f"买入阈值={self.buy_threshold}, 卖出阈值={self.sell_threshold}")
        print(f"数据: {n} 条K线, 窗口={window_size}, 步长={step}")
        print(f"回测区间: {df.index[window_size]} ~ {df.index[-1]}")
        print("-" * 60)

        # 初始化资产曲线(窗口预热期)
        for i in range(window_size):
            price = df['close'].iloc[i]
            equity = self.cash + self.position * price
            self.equity_curve.append({
                'date': df.index[i].isoformat() if hasattr(df.index[i], 'isoformat') else str(df.index[i]),
                'equity': equity,
                'cash': self.cash,
                'position': self.position,
                'price': price
            })

        # 当前信号状态
        current_top_score = 0
        current_bottom_score = 0
        current_recs = []

        # 滚动窗口分析+逐K线执行
        i = window_size
        next_analysis = i  # 下次分析时间

        while i < n:
            price = df['close'].iloc[i]
            date = df.index[i]
            self.bars_since_trade += 1

            # 是否需要重新分析
            if i >= next_analysis:
                window_start = max(0, i - window_size)
                window_df = df.iloc[window_start:i].copy()
                try:
                    analyzer = ComprehensiveAnalyzer(window_df)
                    results = analyzer.analyze_all()
                    score = results.get('comprehensive_score', {})
                    current_top_score = score.get('top_score', 0)
                    current_bottom_score = score.get('bottom_score', 0)
                    current_recs = results.get('trade_recommendations', [])
                except Exception:
                    pass
                next_analysis = i + step

            # ---- 止损止盈检查 (优先级最高) ----
            if self.position > 0 and self.entry_price > 0:
                pnl_pct = (price - self.entry_price) / self.entry_price
                if pnl_pct <= -self.stop_loss:
                    self._execute_trade('sell', price, date, f'止损 ({pnl_pct*100:.1f}%)')
                    i += 1
                    continue
                if pnl_pct >= self.take_profit:
                    self._execute_trade('sell', price, date, f'止盈 ({pnl_pct*100:.1f}%)')
                    i += 1
                    continue

            # ---- 信号交易 (需满足冷却期) ----
            if self.bars_since_trade >= self.cooldown:
                action = self._decide(current_top_score, current_bottom_score,
                                      current_recs, price)
                if action:
                    reason = self._get_reason(action, current_top_score,
                                              current_bottom_score, current_recs)
                    self._execute_trade(action, price, date, reason)

            # 记录资产
            equity = self.cash + self.position * price
            self.equity_curve.append({
                'date': date.isoformat() if hasattr(date, 'isoformat') else str(date),
                'equity': equity,
                'cash': self.cash,
                'position': self.position,
                'price': price
            })

            i += 1

        # 收尾: 如果还有仓位以最后价格平仓
        if self.position > 0:
            last_price = df['close'].iloc[-1]
            last_date = df.index[-1]
            self._execute_trade('sell', last_price, last_date, '回测结束平仓')

        return self._generate_report(df)

    def _decide(self, top_score, bottom_score, recommendations, price):
        """交易决策"""
        # 背驰信号 (最高优先级)
        for rec in recommendations:
            if rec.get('action') == 'BUY_EXHAUSTION' and self.position == 0:
                return 'buy'
            if rec.get('action') == 'SELL_EXHAUSTION' and self.position > 0:
                return 'sell'

        # 卖点用背离 (宁早勿晚, 阈值较低)
        if top_score >= self.sell_threshold and self.position > 0:
            return 'sell'

        # 买点用背驰 (宁迟勿早, 阈值较高)
        if bottom_score >= self.buy_threshold and self.position == 0:
            return 'buy'

        return None

    def _get_reason(self, action, top_score, bottom_score, recs):
        """生成交易原因描述"""
        for rec in recs:
            if action == 'buy' and 'BUY' in rec.get('action', ''):
                return rec.get('reason', f'底背离评分{bottom_score:.0f}')
            if action == 'sell' and 'SELL' in rec.get('action', ''):
                return rec.get('reason', f'顶背离评分{top_score:.0f}')

        if action == 'buy':
            return f'综合底背离评分={bottom_score:.0f}'
        return f'综合顶背离评分={top_score:.0f}'

    def _execute_trade(self, action, price, date, reason=''):
        """执行交易"""
        date_str = date.isoformat() if hasattr(date, 'isoformat') else str(date)

        if action == 'buy' and self.position == 0:
            actual_price = price * (1 + self.slippage)
            invest = self.cash * self.position_size
            fee = invest * self.commission
            qty = (invest - fee) / actual_price
            self.position = qty
            self.entry_price = actual_price
            self.cash -= invest
            self.bars_since_trade = 0

            self.trades.append({
                'date': date_str, 'action': 'BUY',
                'price': round(actual_price, 2),
                'quantity': round(qty, 6),
                'value': round(invest, 2),
                'fee': round(fee, 2),
                'cash_after': round(self.cash, 2),
                'reason': reason
            })
            self.signals_log.append({
                'date': date_str, 'signal': 'BUY', 'price': round(price, 2)
            })

        elif action == 'sell' and self.position > 0:
            actual_price = price * (1 - self.slippage)
            revenue = self.position * actual_price
            fee = revenue * self.commission
            net_revenue = revenue - fee
            buy_cost = self.trades[-1]['value'] if self.trades else 0
            pnl = net_revenue - buy_cost
            pnl_pct = pnl / buy_cost * 100 if buy_cost > 0 else 0

            self.cash += net_revenue
            self.bars_since_trade = 0

            self.trades.append({
                'date': date_str, 'action': 'SELL',
                'price': round(actual_price, 2),
                'quantity': round(self.position, 6),
                'value': round(revenue, 2),
                'fee': round(fee, 2),
                'cash_after': round(self.cash, 2),
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
                'reason': reason
            })
            self.signals_log.append({
                'date': date_str, 'signal': 'SELL', 'price': round(price, 2)
            })
            self.position = 0
            self.entry_price = 0

    def _generate_report(self, df) -> dict:
        """生成回测报告"""
        if not self.equity_curve:
            return {}

        equity_series = pd.Series(
            [e['equity'] for e in self.equity_curve],
            index=pd.to_datetime([e['date'] for e in self.equity_curve])
        )

        final_equity = equity_series.iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        buy_hold_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]

        # 最大回撤
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()

        # 交易配对
        completed_trades = []
        buy_stack = []
        for t in self.trades:
            if t['action'] == 'BUY':
                buy_stack.append(t)
            elif t['action'] == 'SELL' and buy_stack:
                buy_t = buy_stack.pop()
                pnl = (t['value'] - t['fee']) - buy_t['value']
                completed_trades.append({
                    'buy_date': buy_t['date'],
                    'sell_date': t['date'],
                    'buy_price': buy_t['price'],
                    'sell_price': t['price'],
                    'pnl': round(pnl, 2),
                    'pnl_pct': round(pnl / buy_t['value'] * 100, 2),
                    'win': pnl > 0,
                    'buy_reason': buy_t.get('reason', ''),
                    'sell_reason': t.get('reason', ''),
                })

        win_count = sum(1 for t in completed_trades if t['win'])
        total_trades = len(completed_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0

        avg_win = np.mean([t['pnl_pct'] for t in completed_trades if t['win']]) if win_count > 0 else 0
        avg_loss = np.mean([t['pnl_pct'] for t in completed_trades if not t['win']]) if (total_trades - win_count) > 0 else 0

        # 天数
        total_days = (df.index[-1] - df.index[0]).total_seconds() / 86400 if len(df) > 1 else 0
        annual_return = ((1 + total_return) ** (365 / max(total_days, 1)) - 1) if total_days > 0 else 0

        # Sharpe
        if len(equity_series) > 1:
            returns = equity_series.pct_change().dropna()
            sharpe = (returns.mean() / returns.std() * np.sqrt(365 * 24 * 6)) if returns.std() > 0 else 0
        else:
            sharpe = 0

        # 对equity_curve采样, 避免JSON太大
        ec = self.equity_curve
        if len(ec) > 2000:
            step = len(ec) // 2000
            ec = ec[::step] + [ec[-1]]

        # kline采样
        kline_data = []
        kline_step = max(1, len(df) // 3000)
        for i in range(0, len(df), kline_step):
            kline_data.append({
                'date': df.index[i].isoformat(),
                'open': round(float(df['open'].iloc[i]), 2),
                'high': round(float(df['high'].iloc[i]), 2),
                'low': round(float(df['low'].iloc[i]), 2),
                'close': round(float(df['close'].iloc[i]), 2),
                'volume': round(float(df['volume'].iloc[i]), 4),
            })

        report = {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_equity': round(final_equity, 2),
                'total_return': round(total_return * 100, 2),
                'buy_hold_return': round(buy_hold_return * 100, 2),
                'annual_return': round(annual_return * 100, 2),
                'max_drawdown': round(max_drawdown * 100, 2),
                'sharpe_ratio': round(sharpe, 2),
                'total_trades': total_trades,
                'win_rate': round(win_rate * 100, 1),
                'win_count': win_count,
                'lose_count': total_trades - win_count,
                'avg_win_pct': round(avg_win, 2),
                'avg_loss_pct': round(avg_loss, 2),
                'total_days': round(total_days, 1),
                'commission_rate': self.commission * 100,
                'stop_loss': self.stop_loss * 100,
                'take_profit': self.take_profit * 100,
            },
            'trades': self.trades,
            'completed_trades': completed_trades,
            'equity_curve': ec,
            'signals': self.signals_log,
            'kline_data': kline_data,
        }

        # 打印
        print("\n" + "=" * 60)
        print("              回测结果摘要")
        print("=" * 60)
        print(f"  初始资金:     {self.initial_capital:,.2f} USDT")
        print(f"  最终资金:     {final_equity:,.2f} USDT")
        print(f"  策略收益:     {total_return*100:+.2f}%")
        print(f"  买入持有收益: {buy_hold_return*100:+.2f}%")
        print(f"  年化收益:     {annual_return*100:+.2f}%")
        print(f"  最大回撤:     {max_drawdown*100:.2f}%")
        print(f"  Sharpe比率:   {sharpe:.2f}")
        print(f"  总交易次数:   {total_trades}")
        print(f"  胜率:         {win_rate*100:.1f}% ({win_count}胜/{total_trades-win_count}负)")
        print(f"  平均盈利:     {avg_win:+.2f}%")
        print(f"  平均亏损:     {avg_loss:+.2f}%")
        print(f"  回测天数:     {total_days:.1f}天")
        print("=" * 60)

        return report


def run_eth_backtest(days: int = 30, interval: str = '1h') -> dict:
    """
    执行 ETH/USDT 回测

    参数:
        days: 回测天数
        interval: K线周期 (10m / 15m / 1h / 4h / 1d)
    """
    from binance_fetcher import fetch_binance_klines

    # 根据周期调整策略参数
    PARAMS = {
        '5m':  {'window': 200, 'step': 40,  'cooldown': 36, 'buy_th': 65, 'sell_th': 45,
                'sl': 0.04, 'tp': 0.06, 'label': '5分钟'},
        '10m': {'window': 200, 'step': 50,  'cooldown': 30, 'buy_th': 65, 'sell_th': 45,
                'sl': 0.05, 'tp': 0.08, 'label': '10分钟'},
        '15m': {'window': 200, 'step': 40,  'cooldown': 20, 'buy_th': 63, 'sell_th': 45,
                'sl': 0.05, 'tp': 0.08, 'label': '15分钟'},
        '30m': {'window': 200, 'step': 30,  'cooldown': 12, 'buy_th': 60, 'sell_th': 42,
                'sl': 0.06, 'tp': 0.09, 'label': '30分钟'},
        '1h':  {'window': 168, 'step': 24,  'cooldown': 18, 'buy_th': 70, 'sell_th': 40,
                'sl': 0.07, 'tp': 0.12, 'label': '1小时'},
        '2h':  {'window': 150, 'step': 18,  'cooldown': 6,  'buy_th': 55, 'sell_th': 38,
                'sl': 0.07, 'tp': 0.13, 'label': '2小时'},
        '3h':  {'window': 130, 'step': 15,  'cooldown': 4,  'buy_th': 53, 'sell_th': 37,
                'sl': 0.08, 'tp': 0.14, 'label': '3小时'},
        '4h':  {'window': 120, 'step': 12,  'cooldown': 3,  'buy_th': 50, 'sell_th': 35,
                'sl': 0.08, 'tp': 0.15, 'label': '4小时'},
        '6h':  {'window': 100, 'step': 8,   'cooldown': 2,  'buy_th': 50, 'sell_th': 35,
                'sl': 0.09, 'tp': 0.16, 'label': '6小时'},
        '8h':  {'window': 90,  'step': 6,   'cooldown': 2,  'buy_th': 50, 'sell_th': 35,
                'sl': 0.10, 'tp': 0.18, 'label': '8小时'},
        '16h': {'window': 60,  'step': 4,   'cooldown': 2,  'buy_th': 50, 'sell_th': 35,
                'sl': 0.12, 'tp': 0.22, 'label': '16小时'},
        '24h': {'window': 50,  'step': 3,   'cooldown': 1,  'buy_th': 50, 'sell_th': 35,
                'sl': 0.12, 'tp': 0.22, 'label': '24小时'},
        '32h': {'window': 45,  'step': 3,   'cooldown': 1,  'buy_th': 50, 'sell_th': 35,
                'sl': 0.15, 'tp': 0.28, 'label': '32小时'},
        '1d':  {'window': 60,  'step': 5,   'cooldown': 2,  'buy_th': 50, 'sell_th': 35,
                'sl': 0.10, 'tp': 0.20, 'label': '日线'},
    }
    p = PARAMS.get(interval, PARAMS['1h'])

    print(f"\n{'='*60}")
    print(f"  ETH/USDT {p['label']}级别背离/背驰策略回测")
    print(f"{'='*60}")

    df = fetch_binance_klines("ETHUSDT", interval=interval, days=days)
    if df is None or len(df) < p['window']:
        print(f"数据不足(需要至少{p['window']}条K线), 无法回测")
        return {}

    engine = BacktestEngine(
        df,
        initial_capital=10000.0,
        commission=0.001,
        slippage=0.0005,
        position_size=0.95,
        stop_loss=p['sl'],
        take_profit=p['tp'],
        cooldown=p['cooldown'],
        buy_threshold=p['buy_th'],
        sell_threshold=p['sell_th'],
    )

    report = engine.run(window_size=p['window'], step=p['step'])
    report['interval'] = interval
    report['interval_label'] = p['label']

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'backtest_result.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, default=str)
    print(f"\n回测结果已保存: {output_path}")

    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ETH/USDT 背离策略回测')
    parser.add_argument('--interval', '-i', default='1h',
                        choices=['5m','10m','15m','30m','1h','2h','3h','4h','6h','8h','16h','24h','32h','1d'],
                        help='K线周期 (默认: 1h)')
    parser.add_argument('--days', '-d', type=int, default=90,
                        help='回测天数 (默认: 90)')
    args = parser.parse_args()
    run_eth_backtest(days=args.days, interval=args.interval)
