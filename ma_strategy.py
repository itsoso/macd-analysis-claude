"""
均线技术分析 — 策略回测模块

基于《均线技术分析》(邱立波著) 全书策略:
  S1: 葛南维法则 — 严格按照八大买卖法则
  S2: 金叉死叉 — 双均线交叉策略
  S3: 多头排列 — 均线排列趋势跟踪
  S4: 特殊形态 — 银山谷/金山谷/死亡谷等
  S5: 均线粘合突破 — 粘合后发散方向交易
  S6: 综合策略 — 融合全部信号的最优策略
  S7: 保守组合 — 高阈值+低杠杆+严格风控
  S8: 激进做空 — 空头信号加权, 利用下跌获利

初始资金: 10万USDT + 价值10万USDT的ETH
数据: 币安 ETH/USDT, 1h K线
支持: 现货买卖 + 合约做多/做空, 完整手续费模型
"""

import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binance_fetcher import fetch_binance_klines
from indicators import add_all_indicators
from strategy_futures import FuturesEngine
from ma_indicators import (
    add_moving_averages, compute_ma_signals, extract_signal_timeline,
    granville_rules, detect_ma_arrangement, detect_golden_cross,
    detect_death_cross, detect_ma_convergence, detect_ma_bond,
    detect_silver_valley, detect_gold_valley, detect_death_valley,
    detect_dragon_emerge, detect_cloud_support, detect_head_up,
    detect_head_down, price_ma_distance, ma_slope,
)


# ======================================================
#   数据获取
# ======================================================
def fetch_data():
    """获取1h主数据 + 辅助周期数据"""
    print("获取数据...")
    data = {}
    configs = [
        ('1h', 60),    # 主周期: 1小时, 60天(确保均线有足够计算窗口)
        ('4h', 60),    # 辅助: 大趋势
        ('15m', 30),   # 辅助: 短线信号
    ]
    for tf, days in configs:
        df = fetch_binance_klines("ETHUSDT", interval=tf, days=days)
        if df is not None and len(df) > 50:
            df = add_all_indicators(df)
            add_moving_averages(df, timeframe=tf)
            data[tf] = df
            print(f"  {tf}: {len(df)} 条 ({df.index[0]} ~ {df.index[-1]})")
    return data


# ======================================================
#   策略执行器
# ======================================================
def run_ma_strategy(data, config, trade_days=None):
    """
    均线策略回测执行器

    Parameters:
        data: dict of DataFrames
        config: 策略配置
        trade_days: 实际交易天数, None=全部

    Returns:
        dict: 回测结果
    """
    eng = FuturesEngine(config['name'], max_leverage=config.get('max_lev', 5))
    main_df = data['1h']

    # 确定交易起始时间
    start_dt = None
    if trade_days and trade_days < 60:
        start_dt = main_df.index[-1] - pd.Timedelta(days=trade_days)

    # 初始ETH仓位
    init_idx = 0
    if start_dt:
        init_idx = main_df.index.searchsorted(start_dt)
        if init_idx >= len(main_df):
            init_idx = 0
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[init_idx]

    # 风控参数
    eng.max_single_margin = eng.initial_total * config.get('single_pct', 0.15)
    eng.max_margin_total = eng.initial_total * config.get('total_pct', 0.40)
    eng.max_lifetime_margin = eng.initial_total * config.get('lifetime_pct', 5.0)

    # 计算均线信号
    signals = compute_ma_signals(main_df, timeframe='1h')
    buy_scores = signals['buy_score']
    sell_scores = signals['sell_score']
    arrangement = signals['arrangement']
    granville = signals['granville']

    # 策略参数
    buy_threshold = config.get('buy_threshold', 20)
    sell_threshold = config.get('sell_threshold', 20)
    short_threshold = config.get('short_threshold', 30)
    long_threshold = config.get('long_threshold', 30)
    sell_pct = config.get('sell_pct', 0.40)
    buy_pct = config.get('buy_pct', 0.25)
    margin_use = config.get('margin_use', 0.50)
    lev = config.get('lev', 3)

    # 风控
    short_sl = config.get('short_sl', -0.20)
    short_tp = config.get('short_tp', 0.60)
    short_trail = config.get('short_trail', 0.25)
    short_max_hold = config.get('short_max_hold', 72)  # 72h = 3天
    long_sl = config.get('long_sl', -0.15)
    long_tp = config.get('long_tp', 0.50)
    long_trail = config.get('long_trail', 0.20)
    long_max_hold = config.get('long_max_hold', 72)
    cooldown = config.get('cooldown', 4)  # 4h冷却
    spot_cooldown = config.get('spot_cooldown', 12)  # 12h冷却

    # 特殊信号权重
    use_granville = config.get('use_granville', True)
    use_patterns = config.get('use_patterns', True)
    use_crosses = config.get('use_crosses', True)
    use_arrangement = config.get('use_arrangement', True)

    # 状态变量
    short_cd = 0
    long_cd = 0
    spot_cd = 0
    short_bars = 0
    long_bars = 0
    short_max_pnl = 0
    long_max_pnl = 0
    short_just_opened = False
    long_just_opened = False

    for idx in range(max(60, init_idx if not start_dt else 0), len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]

        # start_dt之前只记录不交易
        if start_dt and dt < start_dt:
            if idx % 4 == 0:
                eng.record_history(dt, price)
            continue

        eng.check_liquidation(price, dt)
        short_just_opened = False
        long_just_opened = False

        # 资金费率(每8h)
        eng.funding_counter += 1
        if eng.funding_counter % 8 == 0:
            is_neg = (eng.funding_counter * 7 + 3) % 10 < 3
            rate = FuturesEngine.FUNDING_RATE if not is_neg else -FuturesEngine.FUNDING_RATE * 0.5
            if eng.futures_long:
                cost = eng.futures_long.quantity * price * rate
                eng.usdt -= cost
                if cost > 0:
                    eng.total_funding_paid += cost
                else:
                    eng.total_funding_received += abs(cost)
            if eng.futures_short:
                income = eng.futures_short.quantity * price * rate
                eng.usdt += income
                if income > 0:
                    eng.total_funding_received += income
                else:
                    eng.total_funding_paid += abs(income)

        if short_cd > 0: short_cd -= 1
        if long_cd > 0: long_cd -= 1
        if spot_cd > 0: spot_cd -= 1

        # 获取当前信号
        bs = float(buy_scores.iloc[idx]) if idx < len(buy_scores) else 0
        ss = float(sell_scores.iloc[idx]) if idx < len(sell_scores) else 0
        arr = str(arrangement.iloc[idx]) if idx < len(arrangement) else 'mixed'

        # 额外信号加成(根据策略配置)
        extra_buy = 0
        extra_sell = 0
        reasons_buy = []
        reasons_sell = []

        if use_granville and idx < len(granville):
            g = granville.iloc[idx]
            if g.get('rule_1', False): reasons_buy.append('葛1')
            if g.get('rule_2', False): reasons_buy.append('葛2')
            if g.get('rule_3', False): reasons_buy.append('葛3')
            if g.get('rule_4', False): reasons_buy.append('葛4')
            if g.get('rule_5', False): reasons_sell.append('葛5')
            if g.get('rule_6', False): reasons_sell.append('葛6')
            if g.get('rule_7', False): reasons_sell.append('葛7')
            if g.get('rule_8', False): reasons_sell.append('葛8')

        if use_arrangement:
            if arr == 'bullish':
                extra_buy += 10
                reasons_buy.append('多头排列')
            elif arr == 'bearish':
                extra_sell += 10
                reasons_sell.append('空头排列')

        if use_patterns and idx < len(main_df):
            for pname, pseries in signals['patterns'].items():
                val = pseries.iloc[idx]
                if isinstance(val, bool) and val:
                    if pname in ('silver_valley', 'gold_valley', 'dragon_emerge',
                                 'cloud_support', 'head_up'):
                        extra_buy += 15
                        reasons_buy.append(pname)
                    elif pname in ('death_valley', 'head_down'):
                        extra_sell += 15
                        reasons_sell.append(pname)
                elif isinstance(val, str):
                    if val == 'bullish_break':
                        extra_buy += 15
                        reasons_buy.append('粘合上突')
                    elif val == 'bearish_break':
                        extra_sell += 15
                        reasons_sell.append('粘合下突')

        if use_crosses:
            for cname, cseries in signals['crosses'].items():
                if cseries.iloc[idx]:
                    if 'gc' in cname:
                        extra_buy += 10
                        reasons_buy.append(cname)
                    else:
                        extra_sell += 10
                        reasons_sell.append(cname)

        total_buy = bs + extra_buy
        total_sell = ss + extra_sell

        # 信号冲突检测(改进版: 比例判断)
        if total_buy > 0 and total_sell > 0:
            ratio = min(total_buy, total_sell) / max(total_buy, total_sell)
            in_conflict = ratio >= 0.6 and min(total_buy, total_sell) >= 15
        else:
            in_conflict = False

        # ========================================
        # 决策1: 卖出现货
        # ========================================
        can_sell = (total_sell >= sell_threshold and spot_cd == 0
                    and not in_conflict and eng.spot_eth * price > 500)
        if can_sell:
            reason = f"卖出 SS={total_sell:.0f} {' '.join(reasons_sell[:3])}"
            eng.spot_sell(price, dt, sell_pct, reason)
            spot_cd = spot_cooldown

        # ========================================
        # 决策2: 开空仓
        # ========================================
        sell_dominates = (total_sell > total_buy * 1.5) if total_buy > 0 else True
        short_ok = (short_cd == 0 and total_sell >= short_threshold
                    and not eng.futures_short and sell_dominates)
        if short_ok:
            margin = eng.available_margin() * margin_use
            actual_lev = min(lev, eng.max_leverage)
            if total_sell >= 50:
                actual_lev = min(lev, eng.max_leverage)
            elif total_sell >= 35:
                actual_lev = min(min(lev, 3), eng.max_leverage)
            else:
                actual_lev = min(2, eng.max_leverage)

            reason = f"做空 {actual_lev}x SS={total_sell:.0f} {' '.join(reasons_sell[:3])}"
            eng.open_short(price, dt, margin, actual_lev, reason)
            short_max_pnl = 0
            short_bars = 0
            short_cd = cooldown
            short_just_opened = True

        # ========================================
        # 决策3: 管理空仓
        # ========================================
        if eng.futures_short and not short_just_opened:
            short_bars += 1
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin

            if pnl_r >= short_tp:
                eng.close_short(price, dt, f"止盈 +{pnl_r * 100:.0f}%")
                short_max_pnl = 0; short_cd = cooldown * 2; short_bars = 0
            else:
                if pnl_r > short_max_pnl:
                    short_max_pnl = pnl_r
                if short_max_pnl >= short_trail and eng.futures_short:
                    if pnl_r < short_max_pnl * 0.60:
                        eng.close_short(price, dt,
                                        f"追踪止盈 max={short_max_pnl*100:.0f}% now={pnl_r*100:.0f}%")
                        short_max_pnl = 0; short_cd = cooldown; short_bars = 0

                # 买入信号平空(BS明显主导)
                if eng.futures_short and total_buy >= config.get('close_short_bs', 35):
                    buy_dominant = (total_sell < total_buy * 0.7) if total_buy > 0 else True
                    if buy_dominant:
                        eng.close_short(price, dt, f"买信号平空 BS={total_buy:.0f}")
                        short_max_pnl = 0; short_cd = cooldown * 3; short_bars = 0

                if eng.futures_short and pnl_r < short_sl:
                    eng.close_short(price, dt, f"止损 {pnl_r*100:.0f}%")
                    short_max_pnl = 0; short_cd = cooldown * 4; short_bars = 0

                if eng.futures_short and short_bars >= short_max_hold:
                    eng.close_short(price, dt, f"超时平仓 {short_bars}h")
                    short_max_pnl = 0; short_cd = cooldown; short_bars = 0
        elif eng.futures_short and short_just_opened:
            short_bars = 1

        # ========================================
        # 决策4: 买入现货
        # ========================================
        can_buy = (total_buy >= buy_threshold and spot_cd == 0
                   and not in_conflict and eng.available_usdt() > 500)
        if can_buy:
            buy_amount = eng.available_usdt() * buy_pct
            reason = f"买入 BS={total_buy:.0f} {' '.join(reasons_buy[:3])}"
            eng.spot_buy(price, dt, buy_amount, reason)
            spot_cd = spot_cooldown

        # ========================================
        # 决策5: 开多仓
        # ========================================
        buy_dominates = (total_buy > total_sell * 1.5) if total_sell > 0 else True
        long_ok = (long_cd == 0 and total_buy >= long_threshold
                   and not eng.futures_long and buy_dominates)
        if long_ok:
            margin = eng.available_margin() * margin_use
            actual_lev = min(lev, eng.max_leverage)
            if total_buy >= 50:
                actual_lev = min(lev, eng.max_leverage)
            elif total_buy >= 35:
                actual_lev = min(min(lev, 3), eng.max_leverage)
            else:
                actual_lev = min(2, eng.max_leverage)

            reason = f"做多 {actual_lev}x BS={total_buy:.0f} {' '.join(reasons_buy[:3])}"
            eng.open_long(price, dt, margin, actual_lev, reason)
            long_max_pnl = 0
            long_bars = 0
            long_cd = cooldown
            long_just_opened = True

        # ========================================
        # 决策6: 管理多仓
        # ========================================
        if eng.futures_long and not long_just_opened:
            long_bars += 1
            pnl_r = eng.futures_long.calc_pnl(price) / eng.futures_long.margin

            if pnl_r >= long_tp:
                eng.close_long(price, dt, f"止盈 +{pnl_r*100:.0f}%")
                long_max_pnl = 0; long_cd = cooldown * 2; long_bars = 0
            else:
                if pnl_r > long_max_pnl:
                    long_max_pnl = pnl_r
                if long_max_pnl >= long_trail and eng.futures_long:
                    if pnl_r < long_max_pnl * 0.60:
                        eng.close_long(price, dt,
                                       f"追踪止盈 max={long_max_pnl*100:.0f}% now={pnl_r*100:.0f}%")
                        long_max_pnl = 0; long_cd = cooldown; long_bars = 0

                # 卖出信号平多(SS明显主导)
                if eng.futures_long and total_sell >= config.get('close_long_ss', 35):
                    sell_dominant = (total_buy < total_sell * 0.7) if total_sell > 0 else True
                    if sell_dominant:
                        eng.close_long(price, dt, f"卖信号平多 SS={total_sell:.0f}")
                        long_max_pnl = 0; long_cd = cooldown * 3; long_bars = 0

                if eng.futures_long and pnl_r < long_sl:
                    eng.close_long(price, dt, f"止损 {pnl_r*100:.0f}%")
                    long_max_pnl = 0; long_cd = cooldown * 4; long_bars = 0

                if eng.futures_long and long_bars >= long_max_hold:
                    eng.close_long(price, dt, f"超时平仓 {long_bars}h")
                    long_max_pnl = 0; long_cd = cooldown; long_bars = 0
        elif eng.futures_long and long_just_opened:
            long_bars = 1

        # 记录历史
        if idx % 4 == 0:
            eng.record_history(dt, price)

    # 期末平仓
    last_price = main_df['close'].iloc[-1]
    last_dt = main_df.index[-1]
    if eng.futures_short:
        eng.close_short(last_price, last_dt, "期末平仓")
    if eng.futures_long:
        eng.close_long(last_price, last_dt, "期末平仓")

    # 计算结果
    if start_dt:
        trade_df = main_df[main_df.index >= start_dt]
        if len(trade_df) > 1:
            return eng.get_result(trade_df)
    return eng.get_result(main_df)


# ======================================================
#   策略变体定义
# ======================================================
def get_strategies():
    """定义8种均线策略变体"""
    base = {
        'single_pct': 0.15, 'total_pct': 0.40, 'lifetime_pct': 5.0,
        'buy_threshold': 20, 'sell_threshold': 20,
        'short_threshold': 30, 'long_threshold': 30,
        'close_short_bs': 35, 'close_long_ss': 35,
        'sell_pct': 0.40, 'buy_pct': 0.25, 'margin_use': 0.50, 'lev': 3,
        'short_sl': -0.20, 'short_tp': 0.60, 'short_trail': 0.25,
        'short_max_hold': 72, 'long_sl': -0.15, 'long_tp': 0.50,
        'long_trail': 0.20, 'long_max_hold': 72,
        'cooldown': 4, 'spot_cooldown': 12, 'max_lev': 5,
        'use_granville': True, 'use_patterns': True,
        'use_crosses': True, 'use_arrangement': True,
    }

    strategies = [
        # S1: 葛南维法则 — 纯用八大法则, 不用特殊形态
        {**base, 'name': 'S1: 葛南维法则',
         'use_patterns': False, 'use_crosses': False, 'use_arrangement': False,
         'buy_threshold': 15, 'sell_threshold': 15,
         'short_threshold': 25, 'long_threshold': 25},

        # S2: 金叉死叉 — 纯用均线交叉
        {**base, 'name': 'S2: 金叉死叉',
         'use_granville': False, 'use_patterns': False, 'use_arrangement': False,
         'buy_threshold': 15, 'sell_threshold': 15,
         'short_threshold': 25, 'long_threshold': 25},

        # S3: 多头/空头排列 — 趋势跟踪
        {**base, 'name': 'S3: 排列趋势',
         'use_granville': False, 'use_crosses': False, 'use_patterns': False,
         'buy_threshold': 10, 'sell_threshold': 10,
         'short_threshold': 20, 'long_threshold': 20,
         'short_max_hold': 120, 'long_max_hold': 120},

        # S4: 特殊形态 — 银山谷/金山谷/死亡谷等
        {**base, 'name': 'S4: 特殊形态',
         'use_granville': False, 'use_crosses': False, 'use_arrangement': False,
         'buy_threshold': 15, 'sell_threshold': 15,
         'short_threshold': 25, 'long_threshold': 25},

        # S5: 粘合突破 — 均线粘合后的方向突破
        {**base, 'name': 'S5: 粘合突破',
         'use_granville': True, 'use_patterns': True,
         'buy_threshold': 25, 'sell_threshold': 25,
         'short_threshold': 35, 'long_threshold': 35,
         'lev': 4, 'margin_use': 0.60},

        # S6: 综合策略 — 全部信号融合
        {**base, 'name': 'S6: 综合策略',
         'buy_threshold': 25, 'sell_threshold': 25,
         'short_threshold': 35, 'long_threshold': 35,
         'sell_pct': 0.50, 'buy_pct': 0.30},

        # S7: 保守组合 — 高阈值+低杠杆
        {**base, 'name': 'S7: 保守组合',
         'buy_threshold': 30, 'sell_threshold': 30,
         'short_threshold': 45, 'long_threshold': 45,
         'lev': 2, 'max_lev': 2, 'margin_use': 0.30,
         'short_sl': -0.12, 'long_sl': -0.10,
         'sell_pct': 0.30, 'buy_pct': 0.20,
         'spot_cooldown': 24},

        # S8: 激进做空 — 低阈值开空, 高杠杆
        {**base, 'name': 'S8: 激进做空',
         'sell_threshold': 15, 'short_threshold': 20,
         'buy_threshold': 25, 'long_threshold': 40,
         'lev': 5, 'max_lev': 5, 'margin_use': 0.70,
         'single_pct': 0.20, 'total_pct': 0.50,
         'sell_pct': 0.60, 'buy_pct': 0.20,
         'short_sl': -0.30, 'short_tp': 0.80},
    ]
    return strategies


# ======================================================
#   信号统计
# ======================================================
def analyze_signal_distribution(df, signals):
    """分析信号分布统计"""
    buy_scores = signals['buy_score']
    sell_scores = signals['sell_score']

    stats = {
        'total_bars': len(df),
        'buy_signals': {
            'count_gt_15': int((buy_scores > 15).sum()),
            'count_gt_30': int((buy_scores > 30).sum()),
            'count_gt_50': int((buy_scores > 50).sum()),
            'max_score': float(buy_scores.max()),
            'avg_when_active': float(buy_scores[buy_scores > 15].mean()) if (buy_scores > 15).any() else 0,
        },
        'sell_signals': {
            'count_gt_15': int((sell_scores > 15).sum()),
            'count_gt_30': int((sell_scores > 30).sum()),
            'count_gt_50': int((sell_scores > 50).sum()),
            'max_score': float(sell_scores.max()),
            'avg_when_active': float(sell_scores[sell_scores > 15].mean()) if (sell_scores > 15).any() else 0,
        },
        'arrangement': {
            'bullish_pct': float((signals['arrangement'] == 'bullish').sum() / len(df) * 100),
            'bearish_pct': float((signals['arrangement'] == 'bearish').sum() / len(df) * 100),
            'mixed_pct': float((signals['arrangement'] == 'mixed').sum() / len(df) * 100),
        },
        'patterns': {},
    }

    # 形态统计
    for pname, pseries in signals['patterns'].items():
        if pseries.dtype == bool:
            stats['patterns'][pname] = int(pseries.sum())
        else:
            stats['patterns'][pname] = {
                v: int((pseries == v).sum()) for v in pseries.unique() if v != 'none'
            }

    # 葛南维法则统计
    stats['granville'] = {}
    for r in ['rule_1', 'rule_2', 'rule_3', 'rule_4',
              'rule_5', 'rule_6', 'rule_7', 'rule_8']:
        stats['granville'][r] = int(signals['granville'][r].sum())

    return stats


# ======================================================
#   主函数
# ======================================================
def main(trade_days=None):
    """
    trade_days: 实际交易天数(默认7天)
    """
    if trade_days is None:
        trade_days = 7

    data = fetch_data()

    if '1h' not in data:
        print("错误: 无法获取1h数据")
        return

    main_df = data['1h']

    # 计算信号
    print("\n计算均线信号...")
    signals = compute_ma_signals(main_df, timeframe='1h')

    # 信号统计
    stats = analyze_signal_distribution(main_df, signals)
    print(f"  买入信号: {stats['buy_signals']['count_gt_15']}次(>15分) "
          f"{stats['buy_signals']['count_gt_30']}次(>30分) "
          f"最高={stats['buy_signals']['max_score']:.0f}")
    print(f"  卖出信号: {stats['sell_signals']['count_gt_15']}次(>15分) "
          f"{stats['sell_signals']['count_gt_30']}次(>30分) "
          f"最高={stats['sell_signals']['max_score']:.0f}")
    print(f"  多头排列: {stats['arrangement']['bullish_pct']:.1f}% "
          f"空头排列: {stats['arrangement']['bearish_pct']:.1f}%")
    print(f"  葛南维法则: 买入{sum(stats['granville'][f'rule_{i}'] for i in range(1,5))}次 "
          f"卖出{sum(stats['granville'][f'rule_{i}'] for i in range(5,9))}次")

    # 运行策略
    strategies = get_strategies()
    all_results = []

    trade_label = f"最近{trade_days}天"
    start_dt = main_df.index[-1] - pd.Timedelta(days=trade_days)
    trade_start_str = str(start_dt)[:16]
    trade_end_str = str(main_df.index[-1])[:16]

    print(f"\n{'=' * 110}")
    print(f"  均线技术分析回测 · {len(strategies)}种策略 · {trade_label}")
    print(f"  信号数据: {main_df.index[0]} ~ {main_df.index[-1]} ({len(main_df)}根1h K线)")
    print(f"  交易区间: {trade_start_str} ~ {trade_end_str}")
    print(f"  初始: 10万USDT + 价值10万USDT的ETH")
    print(f"{'=' * 110}")

    print(f"\n{'策略':<24} {'α':>8} {'策略收益':>10} {'BH收益':>10} {'回撤':>8} "
          f"{'交易':>6} {'强平':>4} {'费用':>10}")
    print('-' * 110)

    for cfg in strategies:
        r = run_ma_strategy(data, cfg, trade_days=trade_days)
        all_results.append(r)
        fees = r.get('fees', {})
        print(f"  {cfg['name']:<22} {r['alpha']:>+7.2f}% {r['strategy_return']:>+9.2f}% "
              f"{r['buy_hold_return']:>+9.2f}% {r['max_drawdown']:>7.2f}% "
              f"{r['total_trades']:>5} {r['liquidations']:>3} "
              f"${fees.get('total_costs', 0):>9,.0f}")

    # 排名
    ranked = sorted(all_results, key=lambda x: x['alpha'], reverse=True)
    print(f"\n{'=' * 110}")
    print(f"  策略排名")
    print(f"{'=' * 110}")
    for i, r in enumerate(ranked):
        star = ' ★' if i == 0 else ''
        print(f"  #{i + 1}: {r['name']:<24} α={r['alpha']:+.2f}%{star} "
              f"| 收益={r['strategy_return']:+.2f}% 回撤={r['max_drawdown']:.2f}%")

    # 最佳策略交易明细
    best = ranked[0]
    print(f"\n{'=' * 110}")
    print(f"  最佳策略: {best['name']} · 交易明细")
    print(f"{'=' * 110}")
    for t in best.get('trades', []):
        action = t.get('action', '')
        direction = t.get('direction', '')
        reason = t.get('reason', '')
        prc = t.get('price', 0)
        lev_t = t.get('leverage', 1)
        total = t.get('total', 0)
        print(f"  {str(t['time'])[:16]}  {action:<14} {direction:<6} @${prc:>8,.2f} "
              f"{lev_t}x  总${total:>10,.0f}  {reason[:60]}")

    # 费用明细
    bf = best.get('fees', {})
    print(f"\n  费用明细:")
    print(f"    现货手续费: ${bf.get('spot_fees', 0):,.2f}")
    print(f"    合约手续费: ${bf.get('futures_fees', 0):,.2f}")
    print(f"    资金费率(净): ${bf.get('net_funding', 0):,.2f}")
    print(f"    滑点成本: ${bf.get('slippage_cost', 0):,.2f}")
    print(f"    总费用: ${bf.get('total_costs', 0):,.2f}")

    # 保存结果
    output = {
        'description': f'均线技术分析回测 · {trade_label}',
        'book': '《均线技术分析》邱立波著',
        'run_time': datetime.now().isoformat(),
        'data_range': f"{main_df.index[0]} ~ {main_df.index[-1]}",
        'trade_range': f"{trade_start_str} ~ {trade_end_str}",
        'trade_days': trade_days,
        'total_bars': len(main_df),
        'initial_capital': '10万USDT + 价值10万USDT的ETH',
        'timeframe': '1h',
        'signal_stats': stats,
        'results': [{
            'name': r['name'],
            'alpha': r['alpha'],
            'strategy_return': r['strategy_return'],
            'buy_hold_return': r['buy_hold_return'],
            'max_drawdown': r['max_drawdown'],
            'total_trades': r['total_trades'],
            'liquidations': r['liquidations'],
            'final_total': r['final_total'],
            'final_usdt': r['final_usdt'],
            'final_spot_eth': r['final_spot_eth'],
            'fees': r.get('fees', {}),
            'trades': r.get('trades', []),
            'history': r.get('history', []),
        } for r in all_results],
        'ranking': [{'rank': i + 1, 'name': r['name'], 'alpha': r['alpha']}
                    for i, r in enumerate(ranked)],
        'best_strategy': {
            'name': best['name'],
            'alpha': best['alpha'],
            'strategy_return': best['strategy_return'],
            'trades_count': best['total_trades'],
        },
        'book_outline': {
            'title': '均线技术分析',
            'author': '邱立波',
            'chapters': [
                {
                    'chapter': '第一章 均线概述',
                    'sections': [
                        '均线定义及原理(SMA/EMA/WMA)',
                        '均线参数设置(5/10/20/30/60/120/240日)',
                        '均线特点: 平均成本、支撑压力、助涨助跌、趋势确认',
                        '均线分类: 短期(5/10日)、中期(20/30/60日)、长期(120/240日)',
                    ],
                },
                {
                    'chapter': '第二章 葛南维买卖法则',
                    'sections': [
                        '法则1: 均线走平上穿 → 买入',
                        '法则2: 回调不破均线 → 买入',
                        '法则3: 短暂跌破后回升 → 买入',
                        '法则4: 暴跌远离均线 → 超卖反弹买入',
                        '法则5: 暴涨远离均线 → 超买回落卖出',
                        '法则6: 均线走平下穿 → 卖出',
                        '法则7: 反弹不破均线 → 卖出',
                        '法则8: 均线上方徘徊转跌 → 卖出',
                    ],
                },
                {
                    'chapter': '第三章 均线实战应用',
                    'sections': [
                        '单条均线: 5日/10日/20日/60日/120日/240日',
                        '双线组合: 黄金交叉(金叉)、死亡交叉(死叉)',
                        '均线排列: 多头排列(看涨)、空头排列(看跌)',
                        '均线粘合: 多条均线汇聚→大行情信号',
                        '收敛发散: 粘合后向上/向下发散',
                    ],
                },
                {
                    'chapter': '第四章 均线特殊形态',
                    'sections': [
                        '银山谷: 三线交叉向上三角(初步买入)',
                        '金山谷: 银山谷后回调再次形成(可靠买入)',
                        '死亡谷: 三线交叉向下三角(强烈卖出)',
                        '蛟龙出海: 大阳线突破多条均线(强烈买入)',
                        '烘云托月: 均线托住价格形成支撑',
                        '逐浪上升/下降: 波浪式运行',
                        '首次上穿/下穿: 长期趋势转变信号',
                        '均线粘合后突破: 变盘方向确认',
                    ],
                },
            ],
        },
    }

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'ma_strategy_result.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果已保存: {path}")
    return output


if __name__ == '__main__':
    trade_days = 7  # 默认7天
    if len(sys.argv) > 1:
        try:
            trade_days = int(sys.argv[1])
            if trade_days <= 0 or trade_days > 60:
                print(f"天数范围: 1-60, 输入: {trade_days}")
                trade_days = 7
        except ValueError:
            print(f"无效参数: {sys.argv[1]}, 使用默认7天")
    main(trade_days=trade_days)
