#!/usr/bin/env python3
"""
多周期联合决策 — 逐日盈亏回测
===================================
回测 2025-01-01 ~ 2026-01-31 全区间，
将每日持仓快照与完整交易明细写入 SQLite DB，
供专属 Web 页面展示。

用法:
    python backtest_multi_tf_daily.py
"""

import json
import os
import platform
import socket
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binance_fetcher import fetch_binance_klines
from indicators import add_all_indicators
from ma_indicators import add_moving_averages
from signal_core import compute_signals_six
from optimize_six_book import (
    _build_tf_score_index,
    run_strategy_multi_tf,
)
from multi_tf_daily_db import save_run, _default_db_path

# ──────────────────────────────────────────────────────────
# 参数
# ──────────────────────────────────────────────────────────
TRADE_START = '2025-01-01'
TRADE_END = '2026-01-31'
PRIMARY_TF = '1h'

AVAILABLE_TFS = ['15m', '1h', '4h', '24h']
DECISION_TFS = ['15m', '1h', '4h', '24h']
COMBO_NAME = '四TF联合(15m+1h+4h+24h)'

# 默认策略参数（与最优配置对齐）
DEFAULT_CONFIG = {
    'name': f'多TF逐日_{COMBO_NAME}@{PRIMARY_TF}',
    'single_pct': 0.20,
    'total_pct': 0.50,
    'lifetime_pct': 5.0,
    'sell_threshold': 18,
    'buy_threshold': 25,
    'short_threshold': 25,
    'long_threshold': 40,
    'close_short_bs': 40,
    'close_long_ss': 40,
    'sell_pct': 0.55,
    'margin_use': 0.70,
    'lev': 5,
    'max_lev': 5,
    'short_sl': -0.25,
    'short_tp': 0.60,
    'short_trail': 0.25,
    'short_max_hold': 72,
    'long_sl': -0.08,
    'long_tp': 0.30,
    'long_trail': 0.20,
    'long_max_hold': 72,
    'trail_pullback': 0.60,
    'cooldown': 4,
    'spot_cooldown': 12,
    'use_partial_tp': True,
    'partial_tp_1': 0.20,
    'partial_tp_1_pct': 0.30,
    'use_partial_tp_2': False,
    'partial_tp_2': 0.50,
    'partial_tp_2_pct': 0.30,
    'use_atr_sl': False,
    'atr_sl_mult': 3.0,
    'fusion_mode': 'c6_veto_4',
    'veto_threshold': 25,
    'kdj_bonus': 0.09,
    'kdj_weight': 0.15,
    'kdj_strong_mult': 1.25,
    'kdj_normal_mult': 1.12,
    'kdj_reverse_mult': 0.70,
    'kdj_gate_threshold': 10,
    'veto_dampen': 0.30,
    'use_live_gate': False,
    'consensus_min_strength': 40,
    'coverage_min': 0.0,
    'use_regime_aware': False,
    'use_protections': False,
}

TF_HOURS = {
    '10m': 1/6, '15m': 0.25, '30m': 0.5, '1h': 1, '2h': 2, '3h': 3,
    '4h': 4, '6h': 6, '8h': 8, '12h': 12, '16h': 16, '24h': 24,
}


def _scale_runtime_config(base_config, primary_tf):
    """按主周期缩放 hold/cooldown."""
    config = dict(base_config)
    tf_h = TF_HOURS.get(primary_tf, 1)
    config['short_max_hold'] = max(6, int(config.get('short_max_hold', 72) / tf_h))
    config['long_max_hold'] = max(6, int(config.get('long_max_hold', 72) / tf_h))
    config['cooldown'] = max(1, int(config.get('cooldown', 4) / tf_h))
    config['spot_cooldown'] = max(2, int(config.get('spot_cooldown', 12) / tf_h))
    return config


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


def _history_to_daily(history, trades, initial_capital, trade_start, trade_end):
    """
    将 FuturesEngine 的 history 快照列表 + trades 列表
    转换为逐日记录，覆盖 trade_start ~ trade_end 每一天。
    """
    if not history:
        return []

    # 按天分组 history
    hist_by_day = defaultdict(list)
    for h in history:
        day_str = h['time'][:10]
        hist_by_day[day_str].append(h)

    # 按天分组 trades
    trades_by_day = defaultdict(list)
    for t in trades:
        day_str = (t.get('time') or '')[:10]
        trades_by_day[day_str].append(t)

    # 生成日期范围
    start = pd.Timestamp(trade_start)
    end = pd.Timestamp(trade_end)
    date_range = pd.date_range(start, end, freq='D')

    daily_records = []
    peak = initial_capital
    last_snapshot = None

    for dt in date_range:
        day_str = dt.strftime('%Y-%m-%d')
        day_hists = hist_by_day.get(day_str, [])

        # 取当天最后一个快照，若无则沿用前一天
        snap = day_hists[-1] if day_hists else last_snapshot
        if snap is None:
            continue
        last_snapshot = snap

        total = snap.get('total', initial_capital)
        peak = max(peak, total)
        drawdown = round((total - peak) / peak * 100, 2) if peak > 0 else 0
        return_pct = round((total / initial_capital - 1) * 100, 2)

        day_trades = trades_by_day.get(day_str, [])
        day_pnl = sum(t.get('pnl', 0) for t in day_trades if t.get('pnl'))

        rec = {
            'date': day_str,
            'eth_price': snap.get('eth_price'),
            'total_value': round(total, 2),
            'usdt': snap.get('usdt'),
            'frozen_margin': snap.get('frozen_margin', 0),
            'long_pnl': snap.get('long_pnl', 0),
            'short_pnl': snap.get('short_pnl', 0),
            'spot_eth_value': snap.get('spot_eth_value', 0),
            'return_pct': return_pct,
            'drawdown_pct': drawdown,
            'has_long': snap.get('long_pnl', 0) != 0,
            'has_short': snap.get('short_pnl', 0) != 0,
            'long_entry': None,
            'long_qty': None,
            'short_entry': None,
            'short_qty': None,
            'day_trades': len(day_trades),
            'day_pnl': round(day_pnl, 2),
        }

        # 从 trade 记录中提取最新持仓信息
        for t in reversed(day_trades):
            if t.get('has_long') and t.get('long_entry'):
                rec['has_long'] = True
                rec['long_entry'] = t['long_entry']
                rec['long_qty'] = t.get('long_qty')
                break
        for t in reversed(day_trades):
            if t.get('has_short') and t.get('short_entry'):
                rec['has_short'] = True
                rec['short_entry'] = t['short_entry']
                rec['short_qty'] = t.get('short_qty')
                break

        # 也从 history 快照推断 (如果当天没交易但有持仓)
        if snap.get('long_pnl', 0) != 0:
            rec['has_long'] = True
        if snap.get('short_pnl', 0) != 0:
            rec['has_short'] = True

        daily_records.append(rec)

    return daily_records


def _normalize_trade(t):
    """统一清洗交易记录"""
    def _num(v):
        if v is None:
            return None
        try:
            return round(float(v), 4)
        except (TypeError, ValueError):
            return None

    return {
        'time': str(t.get('time', '')),
        'action': t.get('action', ''),
        'direction': t.get('direction', ''),
        'market_price': _num(t.get('market_price')),
        'exec_price': _num(t.get('exec_price')),
        'quantity': _num(t.get('quantity')),
        'notional_value': _num(t.get('notional_value')),
        'margin': _num(t.get('margin')),
        'margin_released': _num(t.get('margin_released')),
        'leverage': t.get('leverage'),
        'fee': _num(t.get('fee')),
        'slippage_cost': _num(t.get('slippage_cost')),
        'total_cost': _num(t.get('total_cost')),
        'pnl': _num(t.get('pnl')),
        'entry_price': _num(t.get('entry_price')),
        'after_usdt': _num(t.get('after_usdt')),
        'after_total': _num(t.get('after_total')),
        'after_frozen_margin': _num(t.get('after_frozen_margin')),
        'after_available': _num(t.get('after_available')),
        'has_long': bool(t.get('has_long')),
        'has_short': bool(t.get('has_short')),
        'long_entry': _num(t.get('long_entry')),
        'long_qty': _num(t.get('long_qty')),
        'short_entry': _num(t.get('short_entry')),
        'short_qty': _num(t.get('short_qty')),
        'cum_spot_fees': _num(t.get('cum_spot_fees')),
        'cum_futures_fees': _num(t.get('cum_futures_fees')),
        'cum_funding_paid': _num(t.get('cum_funding_paid')),
        'cum_slippage': _num(t.get('cum_slippage')),
        'reason': str(t.get('reason', '')),
    }


def main():
    t0 = time.time()
    print("=" * 80)
    print("  多周期联合决策 — 逐日盈亏回测")
    print(f"  区间: {TRADE_START} ~ {TRADE_END}")
    print(f"  主TF: {PRIMARY_TF}  |  决策TFs: {', '.join(DECISION_TFS)}")
    print("=" * 80)

    # ── 1. 获取数据 ──
    # 需要足够长的历史来覆盖 trade_start 之前的预热期
    # 从现在回溯到 2024-09-01 约 530 天，加缓冲取 560 天
    history_days = 560
    print(f"\n[1/4] 获取数据 ({history_days}天)...")

    all_data = {}
    for tf in AVAILABLE_TFS:
        print(f"  获取 {tf} 数据...")
        df = fetch_data_for_tf(tf, history_days)
        if df is not None:
            all_data[tf] = df
            print(f"    {tf}: {len(df)} 条K线, {df.index[0]} ~ {df.index[-1]}")
        else:
            print(f"    {tf}: 失败!")

    available_tfs = [tf for tf in AVAILABLE_TFS if tf in all_data]
    decision_tfs = [tf for tf in DECISION_TFS if tf in available_tfs]
    if len(decision_tfs) < 2:
        print("❌ 可用TF不足2个, 无法执行多周期决策")
        sys.exit(1)

    if PRIMARY_TF not in all_data:
        print(f"❌ 主TF {PRIMARY_TF} 数据获取失败")
        sys.exit(1)

    print(f"\n  可用TFs: {', '.join(available_tfs)}")
    print(f"  决策TFs: {', '.join(decision_tfs)}")

    # ── 2. 计算信号 ──
    print(f"\n[2/4] 计算六维信号...")
    all_signals = {}
    for tf in available_tfs:
        print(f"  计算 {tf} 信号...")
        all_signals[tf] = compute_signals_six(all_data[tf], tf, all_data, max_bars=2000)
    print(f"  信号计算完成: {len(all_signals)} 个TF")

    # ── 3. 构建评分索引 ──
    print(f"\n[3/4] 构建TF评分索引...")
    config = _scale_runtime_config(DEFAULT_CONFIG, PRIMARY_TF)
    tf_score_index = _build_tf_score_index(all_data, all_signals, available_tfs, config)
    for tf in available_tfs:
        n_scores = len(tf_score_index.get(tf, {}))
        print(f"    {tf}: {n_scores} 个评分点")

    # ── 4. 运行多周期回测 ──
    print(f"\n[4/4] 运行多周期联合决策回测...")
    trade_start_dt = pd.Timestamp(TRADE_START)
    trade_end_dt = pd.Timestamp(TRADE_END) + pd.Timedelta(hours=23, minutes=59)

    # trade_days 设为 0 或 None，因为我们显式指定了 start/end
    result = run_strategy_multi_tf(
        primary_df=all_data[PRIMARY_TF],
        tf_score_map=tf_score_index,
        decision_tfs=decision_tfs,
        config=config,
        primary_tf=PRIMARY_TF,
        trade_days=0,
        trade_start_dt=trade_start_dt,
        trade_end_dt=trade_end_dt,
    )

    # ── 结果提取 ──
    history = result.get('history', [])
    raw_trades = result.get('trades', [])
    fees = result.get('fees', {})

    initial_capital = result.get('initial_total', 200000)
    final_total = result.get('final_total', 0)
    strategy_return = result.get('strategy_return', 0)
    buy_hold_return = result.get('buy_hold_return', 0)
    alpha = result.get('alpha', 0)
    max_drawdown = result.get('max_drawdown', 0)

    # 清洗 trades
    trades = [_normalize_trade(t) for t in raw_trades]

    # 转成逐日记录
    daily_records = _history_to_daily(
        history, raw_trades, initial_capital,
        TRADE_START, TRADE_END,
    )

    # 计算胜率等交易统计
    close_actions = ['CLOSE_LONG', 'CLOSE_SHORT', 'LIQUIDATED']
    close_trades = [t for t in trades if t['action'] in close_actions]
    wins = [t for t in close_trades if (t.get('pnl') or 0) > 0]
    losses = [t for t in close_trades if (t.get('pnl') or 0) <= 0]
    win_rate = round(len(wins) / len(close_trades) * 100, 2) if close_trades else 0
    avg_win = round(sum(t['pnl'] for t in wins) / len(wins), 2) if wins else 0
    avg_loss = round(sum(t['pnl'] for t in losses) / len(losses), 2) if losses else 0
    profit_factor = round(
        abs(sum(t['pnl'] for t in wins)) / abs(sum(t['pnl'] for t in losses)), 2
    ) if losses and sum(t['pnl'] for t in losses) != 0 else 999

    elapsed = time.time() - t0

    summary = {
        'initial_capital': initial_capital,
        'final_capital': final_total,
        'total_return_pct': strategy_return,
        'buy_hold_return_pct': buy_hold_return,
        'alpha_pct': alpha,
        'max_drawdown_pct': max_drawdown,
        'total_trades': len(raw_trades),
        'close_trades': len(close_trades),
        'win_count': len(wins),
        'loss_count': len(losses),
        'win_rate_pct': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'total_fees': fees.get('total_fees', 0),
        'total_slippage': fees.get('slippage_cost', 0),
        'total_costs': fees.get('total_costs', 0),
        'funding_paid': fees.get('funding_paid', 0),
        'funding_received': fees.get('funding_received', 0),
        'net_funding': fees.get('net_funding', 0),
        'fee_drag_pct': fees.get('fee_drag_pct', 0),
        'liquidations': result.get('liquidations', 0),
    }

    run_meta = {
        'start_date': TRADE_START,
        'end_date': TRADE_END,
        'primary_tf': PRIMARY_TF,
        'decision_tfs': decision_tfs,
        'combo_name': COMBO_NAME,
        'leverage': config.get('lev', 5),
        'initial_capital': initial_capital,
        'runner': 'backtest_multi_tf_daily.py',
        'host': socket.gethostname(),
        'run_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_sec': round(elapsed, 1),
        'config': {k: v for k, v in config.items() if not k.startswith('_')},
    }

    # ── 打印汇总 ──
    print(f"\n{'=' * 80}")
    print(f"  回测完成！ ({elapsed:.1f}秒)")
    print(f"{'=' * 80}")
    print(f"  区间:        {TRADE_START} ~ {TRADE_END}")
    print(f"  初始资金:    ${initial_capital:,.0f}")
    print(f"  期末资金:    ${final_total:,.0f}")
    print(f"  策略收益:    {strategy_return:+.2f}%")
    print(f"  买入持有:    {buy_hold_return:+.2f}%")
    print(f"  Alpha:       {alpha:+.2f}%")
    print(f"  最大回撤:    {max_drawdown:.2f}%")
    print(f"  交易次数:    {len(raw_trades)} (平仓 {len(close_trades)})")
    print(f"  胜率:        {win_rate:.1f}%")
    print(f"  盈亏比:      {profit_factor:.2f}")
    print(f"  总费用:      ${fees.get('total_costs', 0):,.2f}")
    print(f"  逐日记录:    {len(daily_records)} 天")

    # ── 保存到 DB ──
    db_path = _default_db_path()
    run_id = save_run(
        db_path=db_path,
        run_meta=run_meta,
        summary=summary,
        daily_records=daily_records,
        trades=trades,
    )
    print(f"\n💾 结果已写入 DB: {db_path} (run_id={run_id})")

    return run_id


if __name__ == '__main__':
    main()
