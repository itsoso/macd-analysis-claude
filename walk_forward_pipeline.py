"""
Phase 3a: Walk-Forward 月度滚动验证管道

滚动窗口回测验证策略稳健性:
- IS 窗口: 6 个月 (训练/优化)
- OOS 窗口: 1 个月 (验证)
- 步长: 1 个月
- 产出 18-24 个 OOS 窗口, 要求 60%+ 窗口盈利

使用方式:
    python walk_forward_pipeline.py --tf 1h --days 720
    python walk_forward_pipeline.py --tf 4h --is-months 6 --oos-months 1
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _months_delta(dt, months):
    """简单的月份偏移 (支持正负)。"""
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, [31, 29 if year % 4 == 0 else 28,
                        31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return dt.replace(year=year, month=month, day=day)


class WalkForwardPipeline:
    """
    Walk-Forward 滚动验证管道。

    流程:
    1. 将历史数据切分为 [IS_window | OOS_window] 的滚动窗口
    2. 对每个窗口:
       a. 在 IS 窗口上运行回测 (使用当前最优参数)
       b. 在 OOS 窗口上运行回测 (使用相同参数)
       c. 记录 IS/OOS 的 alpha, return, drawdown, trades 等指标
    3. 汇总所有 OOS 窗口, 计算:
       - OOS 盈利窗口比例
       - OOS 平均 alpha
       - OOS 累计收益
       - IS vs OOS 衰减比
    """

    def __init__(self, tf='1h', is_months=6, oos_months=1, step_months=1):
        """
        Parameters
        ----------
        tf : str
            时间框架
        is_months : int
            In-Sample 窗口长度 (月)
        oos_months : int
            Out-of-Sample 窗口长度 (月)
        step_months : int
            滚动步长 (月)
        """
        self.tf = tf
        self.is_months = is_months
        self.oos_months = oos_months
        self.step_months = step_months
        self.results = []

    def run(self, df, signals_func, config, data_all=None, verbose=True):
        """
        执行 Walk-Forward 验证。

        Parameters
        ----------
        df : pd.DataFrame
            完整的 K 线数据 (含技术指标)
        signals_func : callable
            信号计算函数: (df, tf, data_all) -> signals
        config : dict
            策略配置
        data_all : dict
            多时间框架数据 (可选)
        verbose : bool
            是否打印详细信息

        Returns
        -------
        dict
            验证结果汇总
        """
        from optimize_six_book import run_strategy

        if data_all is None:
            data_all = {self.tf: df}

        start_dt = df.index[0]
        end_dt = df.index[-1]
        total_days = (end_dt - start_dt).days

        if verbose:
            print(f"\n{'='*80}")
            print(f"Walk-Forward 验证")
            print(f"{'='*80}")
            print(f"  时间框架: {self.tf}")
            print(f"  数据范围: {start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')} ({total_days} 天)")
            print(f"  IS 窗口: {self.is_months} 个月, OOS 窗口: {self.oos_months} 个月")
            print(f"  步长: {self.step_months} 个月")

        # 计算信号 (全量计算一次, 避免重复)
        if verbose:
            print(f"\n  计算信号...")
        t0 = time.time()
        signals = signals_func(df, self.tf, data_all)
        if verbose:
            print(f"  信号计算完成: {time.time()-t0:.1f}s")

        # 生成滚动窗口
        windows = []
        window_start = start_dt
        while True:
            is_start = window_start
            is_end = _months_delta(is_start, self.is_months)
            oos_start = is_end
            oos_end = _months_delta(oos_start, self.oos_months)

            if oos_end > end_dt:
                break

            windows.append({
                'is_start': is_start,
                'is_end': is_end,
                'oos_start': oos_start,
                'oos_end': oos_end,
            })
            window_start = _months_delta(window_start, self.step_months)

        if verbose:
            print(f"  窗口数: {len(windows)}")
            if windows:
                print(f"  首个窗口: IS {windows[0]['is_start'].strftime('%m/%d')}→{windows[0]['is_end'].strftime('%m/%d')}, "
                      f"OOS {windows[0]['oos_start'].strftime('%m/%d')}→{windows[0]['oos_end'].strftime('%m/%d')}")

        if not windows:
            print("  数据不足, 无法生成任何窗口")
            return {'windows': [], 'summary': {}}

        # 对每个窗口运行回测
        self.results = []
        is_days = self.is_months * 30
        oos_days = self.oos_months * 30

        if verbose:
            print(f"\n{'─'*80}")
            print(f"  {'#':>3} {'IS Period':<24} {'OOS Period':<24} "
                  f"{'IS α':>8} {'OOS α':>8} {'OOS Ret':>8} {'OOS DD':>8} {'OOS Trd':>7}")
            print(f"{'─'*80}")

        for i, w in enumerate(windows):
            # IS 回测
            is_config = dict(config)
            is_result = run_strategy(
                df, signals, is_config, tf=self.tf, trade_days=is_days,
                trade_start_dt=w['is_start'], trade_end_dt=w['is_end']
            )

            # OOS 回测 (使用相同参数)
            oos_config = dict(config)
            oos_result = run_strategy(
                df, signals, oos_config, tf=self.tf, trade_days=oos_days,
                trade_start_dt=w['oos_start'], trade_end_dt=w['oos_end']
            )

            window_result = {
                'window_idx': i,
                'is_start': w['is_start'].isoformat(),
                'is_end': w['is_end'].isoformat(),
                'oos_start': w['oos_start'].isoformat(),
                'oos_end': w['oos_end'].isoformat(),
                'is_alpha': is_result.get('alpha', 0),
                'is_return': is_result.get('strategy_return', 0),
                'is_drawdown': is_result.get('max_drawdown', 0),
                'is_trades': is_result.get('total_trades', 0),
                'oos_alpha': oos_result.get('alpha', 0),
                'oos_return': oos_result.get('strategy_return', 0),
                'oos_drawdown': oos_result.get('max_drawdown', 0),
                'oos_trades': oos_result.get('total_trades', 0),
            }
            self.results.append(window_result)

            if verbose:
                is_p = f"{w['is_start'].strftime('%Y-%m-%d')}→{w['is_end'].strftime('%m-%d')}"
                oos_p = f"{w['oos_start'].strftime('%Y-%m-%d')}→{w['oos_end'].strftime('%m-%d')}"
                oos_a = window_result['oos_alpha']
                marker = '✓' if oos_a > 0 else '✗'
                print(f"  {i+1:>3} {is_p:<24} {oos_p:<24} "
                      f"{window_result['is_alpha']:>+7.2f}% {oos_a:>+7.2f}% "
                      f"{window_result['oos_return']:>+7.2f}% {window_result['oos_drawdown']:>7.2f}% "
                      f"{window_result['oos_trades']:>5} {marker}")

        # 汇总
        summary = self._compute_summary()

        if verbose:
            self._print_summary(summary)

        return {'windows': self.results, 'summary': summary}

    def _compute_summary(self):
        """计算 Walk-Forward 汇总统计。"""
        if not self.results:
            return {}

        oos_alphas = [r['oos_alpha'] for r in self.results]
        oos_returns = [r['oos_return'] for r in self.results]
        is_alphas = [r['is_alpha'] for r in self.results]
        oos_dds = [r['oos_drawdown'] for r in self.results]
        oos_trades = [r['oos_trades'] for r in self.results]

        n_windows = len(self.results)
        n_profitable = sum(1 for a in oos_alphas if a > 0)
        pct_profitable = n_profitable / max(1, n_windows)

        # IS vs OOS 衰减比 (越接近 1.0 越好, 表示没有过拟合)
        avg_is = np.mean(is_alphas) if is_alphas else 0
        avg_oos = np.mean(oos_alphas) if oos_alphas else 0
        decay_ratio = avg_oos / max(abs(avg_is), 0.01) if avg_is != 0 else 0

        # 累计 OOS 收益 (假设连续投入)
        cum_return = 1.0
        for r in oos_returns:
            cum_return *= (1 + r / 100)
        cum_return_pct = (cum_return - 1) * 100

        # 稳定性指标
        oos_alpha_std = float(np.std(oos_alphas)) if len(oos_alphas) > 1 else 0
        oos_sharpe = avg_oos / max(oos_alpha_std, 0.01) if oos_alpha_std > 0 else 0

        return {
            'n_windows': n_windows,
            'n_profitable': n_profitable,
            'pct_profitable': round(pct_profitable, 4),
            'pass_60pct': pct_profitable >= 0.60,
            'avg_is_alpha': round(avg_is, 2),
            'avg_oos_alpha': round(avg_oos, 2),
            'median_oos_alpha': round(float(np.median(oos_alphas)), 2),
            'oos_alpha_std': round(oos_alpha_std, 2),
            'oos_alpha_sharpe': round(oos_sharpe, 3),
            'decay_ratio': round(decay_ratio, 3),
            'cum_oos_return_pct': round(cum_return_pct, 2),
            'avg_oos_drawdown': round(float(np.mean(oos_dds)), 2),
            'max_oos_drawdown': round(float(min(oos_dds)), 2) if oos_dds else 0,
            'avg_oos_trades': round(float(np.mean(oos_trades)), 1),
            'worst_oos_alpha': round(float(min(oos_alphas)), 2) if oos_alphas else 0,
            'best_oos_alpha': round(float(max(oos_alphas)), 2) if oos_alphas else 0,
        }

    def _print_summary(self, summary):
        """打印验证汇总报告。"""
        print(f"\n{'='*80}")
        print(f"Walk-Forward 汇总报告")
        print(f"{'='*80}")
        print(f"  窗口总数:        {summary['n_windows']}")
        print(f"  盈利窗口:        {summary['n_profitable']}/{summary['n_windows']} "
              f"({summary['pct_profitable']:.0%}) "
              f"{'✓ PASS' if summary['pass_60pct'] else '✗ FAIL'}")
        print(f"  IS 平均 α:       {summary['avg_is_alpha']:+.2f}%")
        print(f"  OOS 平均 α:      {summary['avg_oos_alpha']:+.2f}%")
        print(f"  OOS 中位 α:      {summary['median_oos_alpha']:+.2f}%")
        print(f"  OOS α 标准差:    {summary['oos_alpha_std']:.2f}%")
        print(f"  OOS α Sharpe:    {summary['oos_alpha_sharpe']:.3f}")
        print(f"  IS→OOS 衰减比:   {summary['decay_ratio']:.3f} "
              f"(1.0=无过拟合, <0.5=严重过拟合)")
        print(f"  累计 OOS 收益:   {summary['cum_oos_return_pct']:+.2f}%")
        print(f"  OOS 平均回撤:    {summary['avg_oos_drawdown']:.2f}%")
        print(f"  OOS 最大回撤:    {summary['max_oos_drawdown']:.2f}%")
        print(f"  OOS 最差 α:      {summary['worst_oos_alpha']:+.2f}%")
        print(f"  OOS 最佳 α:      {summary['best_oos_alpha']:+.2f}%")
        print(f"  OOS 平均交易数:  {summary['avg_oos_trades']:.1f}")

        # 诊断建议
        print(f"\n  诊断:")
        if summary['decay_ratio'] < 0.3:
            print(f"  ⚠ IS→OOS 衰减严重 ({summary['decay_ratio']:.2f}), 可能存在过拟合")
        elif summary['decay_ratio'] < 0.5:
            print(f"  △ IS→OOS 有明显衰减, 建议减少参数或增加正则化")
        else:
            print(f"  ✓ IS→OOS 衰减可接受")

        if not summary['pass_60pct']:
            print(f"  ⚠ 盈利窗口比例 < 60%, 策略稳定性不足")
        else:
            print(f"  ✓ 盈利窗口比例达标 (≥60%)")

        if summary['oos_alpha_sharpe'] < 0.5:
            print(f"  ⚠ OOS α Sharpe < 0.5, 收益波动大")
        else:
            print(f"  ✓ OOS α Sharpe 达标")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Walk-Forward 月度滚动验证')
    parser.add_argument('--tf', default='1h', help='时间框架')
    parser.add_argument('--days', type=int, default=720, help='总数据天数')
    parser.add_argument('--is-months', type=int, default=6, help='IS 窗口 (月)')
    parser.add_argument('--oos-months', type=int, default=1, help='OOS 窗口 (月)')
    parser.add_argument('--step-months', type=int, default=1, help='步长 (月)')
    parser.add_argument('--output', type=str, default='walk_forward_result.json',
                        help='输出文件')
    args = parser.parse_args()

    from binance_fetcher import fetch_binance_klines
    from indicators import add_all_indicators
    from ma_indicators import add_moving_averages
    from optimize_six_book import compute_signals_six

    # 获取数据
    print(f"获取 {args.tf} 数据 ({args.days} 天)...")
    df = fetch_binance_klines("ETHUSDT", interval=args.tf, days=args.days)
    if df is None or len(df) < 200:
        print(f"数据不足 ({len(df) if df is not None else 0} bars)")
        return

    df = add_all_indicators(df)
    add_moving_averages(df, timeframe=args.tf)
    data_all = {args.tf: df}
    print(f"数据: {len(df)} bars, {df.index[0]} → {df.index[-1]}")

    # 加载最优配置
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'optimize_six_book_result.json')
    config = {}
    if os.path.exists(result_path):
        with open(result_path) as f:
            result_data = json.load(f)
        gb = result_data.get('global_best', {})
        if isinstance(gb, dict) and 'config' in gb:
            config = dict(gb['config'])
            print(f"加载最优配置: {gb.get('tf', '?')} α={gb.get('alpha', '?')}%")

    # 运行 Walk-Forward
    pipeline = WalkForwardPipeline(
        tf=args.tf,
        is_months=args.is_months,
        oos_months=args.oos_months,
        step_months=args.step_months,
    )
    result = pipeline.run(df, compute_signals_six, config, data_all=data_all)

    # 保存
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    output_data = {
        'generated_at': datetime.now().isoformat(),
        'params': {
            'tf': args.tf,
            'days': args.days,
            'is_months': args.is_months,
            'oos_months': args.oos_months,
            'step_months': args.step_months,
        },
        **result,
    }
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
