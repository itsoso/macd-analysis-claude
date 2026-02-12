"""
双书融合策略 — 背离技术分析 × 均线技术分析

核心思路:
  背离分析 → 捕捉拐点(何时反转)
  均线分析 → 确认趋势(方向是否成立)
  两者结合 → 减少假信号, 提升胜率

融合方式:
  1. 共振确认: 背离+均线同时给出信号 → 高确信度交易
  2. 均线过滤: 用均线排列/趋势过滤背离信号(只顺势交易)
  3. 背离触发+均线确认: 背离产生信号, 等均线确认后入场
  4. 分层加仓: 第一信号开仓, 第二信号加仓

初始: 10万USDT + 价值10万USDT的ETH
数据: 币安 ETH/USDT, 1h K线
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
from strategy_enhanced import analyze_signals_enhanced, get_signal_at, DEFAULT_SIG
from strategy_futures import FuturesEngine
from strategy_futures_v4 import _calc_top_score, _calc_bottom_score
from ma_indicators import (
    add_moving_averages, compute_ma_signals,
    detect_ma_arrangement, detect_golden_cross, detect_death_cross,
    detect_ma_convergence, price_ma_distance, ma_slope,
    granville_rules,
)


# ======================================================
#   数据获取
# ======================================================
def fetch_data():
    """获取多周期数据"""
    print("获取数据...")
    data = {}
    configs = [
        ('1h', 60),
        ('4h', 60),
        ('8h', 60),
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
#   趋势判断(融合MA+价格)
# ======================================================
def get_combined_trend(data, dt, price):
    """融合均线和价格的趋势判断"""
    trend = {
        'is_downtrend': False, 'is_uptrend': False,
        'ma_bearish': False, 'ma_bullish': False,
        'ma_slope_down': False, 'ma_slope_up': False,
    }
    df1h = data.get('1h')
    if df1h is None:
        return trend

    idx = df1h.index.searchsorted(dt)
    if idx < 30 or idx >= len(df1h):
        return trend

    # MA排列
    if all(c in df1h.columns for c in ['MA5', 'MA10', 'MA20']):
        ma5 = df1h['MA5'].iloc[idx]
        ma10 = df1h['MA10'].iloc[idx]
        ma20 = df1h['MA20'].iloc[idx]

        if ma5 > ma10 > ma20:
            trend['ma_bullish'] = True
        elif ma5 < ma10 < ma20:
            trend['ma_bearish'] = True

    # MA20斜率
    if 'MA20' in df1h.columns:
        ma20_now = df1h['MA20'].iloc[idx]
        ma20_prev = df1h['MA20'].iloc[max(0, idx - 5)]
        if ma20_prev > 0:
            slope_pct = (ma20_now - ma20_prev) / ma20_prev * 100
            if slope_pct < -0.3:
                trend['ma_slope_down'] = True
            elif slope_pct > 0.3:
                trend['ma_slope_up'] = True

    # 短期趋势(用于背离的趋势加成)
    close_5 = df1h['close'].iloc[max(0, idx - 5):idx].mean()
    close_20 = df1h['close'].iloc[max(0, idx - 20):idx].mean()
    if close_5 < close_20 * 0.99:
        trend['is_downtrend'] = True
    elif close_5 > close_20 * 1.01:
        trend['is_uptrend'] = True

    return trend


# ======================================================
#   融合信号计算
# ======================================================
def calc_combined_signals(data, div_signals_1h, div_signals_8h, ma_signals, dt, idx, config):
    """
    计算融合信号

    Returns:
        sell_score, buy_score, reasons_sell, reasons_buy
    """
    # ---- 背离信号 ----
    sig_1h = get_signal_at(div_signals_1h, dt) or dict(DEFAULT_SIG)
    sig_8h = get_signal_at(div_signals_8h, dt) or dict(DEFAULT_SIG)

    # 融合1h(主) + 8h(辅)
    merged_div = dict(DEFAULT_SIG)
    merged_div['top'] = 0
    merged_div['bottom'] = 0
    for sig_src, w in [(sig_1h, 1.0), (sig_8h, 0.5)]:
        merged_div['top'] += sig_src.get('top', 0) * w
        merged_div['bottom'] += sig_src.get('bottom', 0) * w
        for k in DEFAULT_SIG:
            if isinstance(DEFAULT_SIG[k], bool) and sig_src.get(k):
                merged_div[k] = True
            elif isinstance(DEFAULT_SIG[k], (int, float)) and k not in ('top', 'bottom'):
                merged_div[k] = max(merged_div.get(k, 0), sig_src.get(k, 0))

    trend = get_combined_trend(data, dt, idx)
    div_ts, div_parts = _calc_top_score(merged_div, trend)
    div_bs = _calc_bottom_score(merged_div, trend)

    # ---- 均线信号 ----
    ma_buy = float(ma_signals['buy_score'].iloc[idx]) if idx < len(ma_signals['buy_score']) else 0
    ma_sell = float(ma_signals['sell_score'].iloc[idx]) if idx < len(ma_signals['sell_score']) else 0
    ma_arr = str(ma_signals['arrangement'].iloc[idx]) if idx < len(ma_signals['arrangement']) else 'mixed'

    # 均线交叉
    ma_gc = any(ma_signals['crosses'][k].iloc[idx] for k in ma_signals['crosses']
                if 'gc' in k and idx < len(ma_signals['crosses'][k]))
    ma_dc = any(ma_signals['crosses'][k].iloc[idx] for k in ma_signals['crosses']
                if 'dc' in k and idx < len(ma_signals['crosses'][k]))

    # ---- 融合策略 ----
    mode = config.get('fusion_mode', 'weighted')
    div_w = config.get('div_weight', 0.6)
    ma_w = config.get('ma_weight', 0.4)

    sell_score = 0
    buy_score = 0
    reasons_sell = []
    reasons_buy = []

    if mode == 'weighted':
        # 加权融合: 两个系统的分数加权相加
        sell_score = div_ts * div_w + ma_sell * ma_w
        buy_score = div_bs * div_w + ma_buy * ma_w
        if div_ts > 0: reasons_sell.append(f"背离TS={div_ts:.0f}")
        if ma_sell > 0: reasons_sell.append(f"均线SS={ma_sell:.0f}")
        if div_bs > 0: reasons_buy.append(f"背离BS={div_bs:.0f}")
        if ma_buy > 0: reasons_buy.append(f"均线BS={ma_buy:.0f}")

    elif mode == 'resonance':
        # 共振确认: 两个系统都必须给出信号, 分数相乘(高确信度)
        if div_ts >= 15 and ma_sell >= 10:
            sell_score = (div_ts + ma_sell) * 1.2  # 共振加成20%
            reasons_sell.append(f"共振! 背离TS={div_ts:.0f}+均线SS={ma_sell:.0f}")
        if div_bs >= 15 and ma_buy >= 10:
            buy_score = (div_bs + ma_buy) * 1.2
            reasons_buy.append(f"共振! 背离BS={div_bs:.0f}+均线BS={ma_buy:.0f}")

    elif mode == 'ma_filter':
        # 均线过滤: 背离产生信号, 但只有均线趋势一致才执行
        # 卖出: 背离顶部 + 均线空头排列或死叉
        if div_ts >= 15:
            ma_confirm = (ma_arr == 'bearish') or ma_dc or (ma_sell >= 15)
            if ma_confirm:
                sell_score = div_ts * 1.3  # 趋势确认加成
                reasons_sell.append(f"背离TS={div_ts:.0f}")
                if ma_arr == 'bearish': reasons_sell.append("空头排列确认")
                if ma_dc: reasons_sell.append("死叉确认")
            else:
                sell_score = div_ts * 0.5  # 无确认则大幅削弱
                reasons_sell.append(f"背离TS={div_ts:.0f}(无均线确认)")

        # 买入: 背离底部 + 均线多头排列或金叉
        if div_bs >= 15:
            ma_confirm = (ma_arr == 'bullish') or ma_gc or (ma_buy >= 15)
            if ma_confirm:
                buy_score = div_bs * 1.3
                reasons_buy.append(f"背离BS={div_bs:.0f}")
                if ma_arr == 'bullish': reasons_buy.append("多头排列确认")
                if ma_gc: reasons_buy.append("金叉确认")
            else:
                buy_score = div_bs * 0.5
                reasons_buy.append(f"背离BS={div_bs:.0f}(无均线确认)")

    elif mode == 'div_filter':
        # 反过来: 均线产生信号, 背离确认拐点
        if ma_sell >= 15:
            div_confirm = div_ts >= 15
            if div_confirm:
                sell_score = ma_sell * 1.3
                reasons_sell.append(f"均线SS={ma_sell:.0f}+背离TS={div_ts:.0f}确认")
            else:
                sell_score = ma_sell * 0.6
                reasons_sell.append(f"均线SS={ma_sell:.0f}(无背离确认)")

        if ma_buy >= 15:
            div_confirm = div_bs >= 15
            if div_confirm:
                buy_score = ma_buy * 1.3
                reasons_buy.append(f"均线BS={ma_buy:.0f}+背离BS={div_bs:.0f}确认")
            else:
                buy_score = ma_buy * 0.6
                reasons_buy.append(f"均线BS={ma_buy:.0f}(无背离确认)")

    # 通用加成: 排列方向
    if ma_arr == 'bearish':
        sell_score *= 1.1
        if 'bearish' not in str(reasons_sell): reasons_sell.append("空头排列")
    elif ma_arr == 'bullish':
        buy_score *= 1.1
        if 'bullish' not in str(reasons_buy): reasons_buy.append("多头排列")

    # 背离特殊形态加成
    if div_parts:
        reasons_sell.extend(div_parts[:2])

    return sell_score, buy_score, reasons_sell, reasons_buy


# ======================================================
#   策略执行器
# ======================================================
def run_combined_strategy(data, div_signals_1h, div_signals_8h, ma_signals, config,
                          trade_days=None):
    """融合策略回测执行器"""
    eng = FuturesEngine(config['name'], max_leverage=config.get('max_lev', 5))
    main_df = data['1h']

    start_dt = None
    if trade_days and trade_days < 60:
        start_dt = main_df.index[-1] - pd.Timedelta(days=trade_days)

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

    # 策略参数
    sell_threshold = config.get('sell_threshold', 25)
    buy_threshold = config.get('buy_threshold', 25)
    short_threshold = config.get('short_threshold', 35)
    long_threshold = config.get('long_threshold', 35)
    sell_pct = config.get('sell_pct', 0.45)
    buy_pct = config.get('buy_pct', 0.25)
    margin_use = config.get('margin_use', 0.50)
    lev = config.get('lev', 3)
    cooldown = config.get('cooldown', 4)
    spot_cooldown = config.get('spot_cooldown', 12)

    # 风控
    short_sl = config.get('short_sl', -0.20)
    short_tp = config.get('short_tp', 0.60)
    short_trail = config.get('short_trail', 0.25)
    short_max_hold = config.get('short_max_hold', 72)
    long_sl = config.get('long_sl', -0.15)
    long_tp = config.get('long_tp', 0.50)
    long_trail = config.get('long_trail', 0.20)
    long_max_hold = config.get('long_max_hold', 72)

    # 状态
    short_cd = 0; long_cd = 0; spot_cd = 0
    short_bars = 0; long_bars = 0
    short_max_pnl = 0; long_max_pnl = 0

    for idx in range(60, len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]

        if start_dt and dt < start_dt:
            if idx % 4 == 0:
                eng.record_history(dt, price)
            continue

        eng.check_liquidation(price, dt)
        short_just_opened = False
        long_just_opened = False

        # 资金费率
        eng.funding_counter += 1
        if eng.funding_counter % 8 == 0:
            is_neg = (eng.funding_counter * 7 + 3) % 10 < 3
            rate = FuturesEngine.FUNDING_RATE if not is_neg else -FuturesEngine.FUNDING_RATE * 0.5
            if eng.futures_long:
                cost = eng.futures_long.quantity * price * rate
                eng.usdt -= cost
                if cost > 0: eng.total_funding_paid += cost
                else: eng.total_funding_received += abs(cost)
            if eng.futures_short:
                income = eng.futures_short.quantity * price * rate
                eng.usdt += income
                if income > 0: eng.total_funding_received += income
                else: eng.total_funding_paid += abs(income)

        if short_cd > 0: short_cd -= 1
        if long_cd > 0: long_cd -= 1
        if spot_cd > 0: spot_cd -= 1

        # 计算融合信号
        ss, bs, r_sell, r_buy = calc_combined_signals(
            data, div_signals_1h, div_signals_8h, ma_signals, dt, idx, config)

        # 冲突检测
        if ss > 0 and bs > 0:
            ratio = min(ss, bs) / max(ss, bs)
            in_conflict = ratio >= 0.6 and min(ss, bs) >= 15
        else:
            in_conflict = False

        # ---- 卖出现货 ----
        if ss >= sell_threshold and spot_cd == 0 and not in_conflict and eng.spot_eth * price > 500:
            eng.spot_sell(price, dt, sell_pct, f"卖出 {' '.join(r_sell[:3])}")
            spot_cd = spot_cooldown

        # ---- 开空仓 ----
        sell_dom = (ss > bs * 1.5) if bs > 0 else True
        if short_cd == 0 and ss >= short_threshold and not eng.futures_short and sell_dom:
            margin = eng.available_margin() * margin_use
            actual_lev = min(lev if ss >= 50 else min(lev, 3) if ss >= 35 else 2, eng.max_leverage)
            eng.open_short(price, dt, margin, actual_lev,
                           f"做空 {actual_lev}x {' '.join(r_sell[:3])}")
            short_max_pnl = 0; short_bars = 0; short_cd = cooldown
            short_just_opened = True

        # ---- 管理空仓 ----
        if eng.futures_short and not short_just_opened:
            short_bars += 1
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
            if pnl_r >= short_tp:
                eng.close_short(price, dt, f"止盈 +{pnl_r*100:.0f}%")
                short_max_pnl = 0; short_cd = cooldown * 2; short_bars = 0
            else:
                if pnl_r > short_max_pnl: short_max_pnl = pnl_r
                if short_max_pnl >= short_trail and eng.futures_short:
                    if pnl_r < short_max_pnl * 0.60:
                        eng.close_short(price, dt, f"追踪止盈 max={short_max_pnl*100:.0f}%")
                        short_max_pnl = 0; short_cd = cooldown; short_bars = 0
                if eng.futures_short and bs >= config.get('close_short_bs', 40):
                    bs_dom = (ss < bs * 0.7) if bs > 0 else True
                    if bs_dom:
                        eng.close_short(price, dt, f"买信号平空 BS={bs:.0f}")
                        short_max_pnl = 0; short_cd = cooldown * 3; short_bars = 0
                if eng.futures_short and pnl_r < short_sl:
                    eng.close_short(price, dt, f"止损 {pnl_r*100:.0f}%")
                    short_max_pnl = 0; short_cd = cooldown * 4; short_bars = 0
                if eng.futures_short and short_bars >= short_max_hold:
                    eng.close_short(price, dt, f"超时 {short_bars}h")
                    short_max_pnl = 0; short_cd = cooldown; short_bars = 0
        elif eng.futures_short and short_just_opened:
            short_bars = 1

        # ---- 买入现货 ----
        if bs >= buy_threshold and spot_cd == 0 and not in_conflict and eng.available_usdt() > 500:
            eng.spot_buy(price, dt, eng.available_usdt() * buy_pct,
                         f"买入 {' '.join(r_buy[:3])}")
            spot_cd = spot_cooldown

        # ---- 开多仓 ----
        buy_dom = (bs > ss * 1.5) if ss > 0 else True
        if long_cd == 0 and bs >= long_threshold and not eng.futures_long and buy_dom:
            margin = eng.available_margin() * margin_use
            actual_lev = min(lev if bs >= 50 else min(lev, 3) if bs >= 35 else 2, eng.max_leverage)
            eng.open_long(price, dt, margin, actual_lev,
                          f"做多 {actual_lev}x {' '.join(r_buy[:3])}")
            long_max_pnl = 0; long_bars = 0; long_cd = cooldown
            long_just_opened = True

        # ---- 管理多仓 ----
        if eng.futures_long and not long_just_opened:
            long_bars += 1
            pnl_r = eng.futures_long.calc_pnl(price) / eng.futures_long.margin
            if pnl_r >= long_tp:
                eng.close_long(price, dt, f"止盈 +{pnl_r*100:.0f}%")
                long_max_pnl = 0; long_cd = cooldown * 2; long_bars = 0
            else:
                if pnl_r > long_max_pnl: long_max_pnl = pnl_r
                if long_max_pnl >= long_trail and eng.futures_long:
                    if pnl_r < long_max_pnl * 0.60:
                        eng.close_long(price, dt, f"追踪止盈 max={long_max_pnl*100:.0f}%")
                        long_max_pnl = 0; long_cd = cooldown; long_bars = 0
                if eng.futures_long and ss >= config.get('close_long_ss', 40):
                    ss_dom = (bs < ss * 0.7) if ss > 0 else True
                    if ss_dom:
                        eng.close_long(price, dt, f"卖信号平多 SS={ss:.0f}")
                        long_max_pnl = 0; long_cd = cooldown * 3; long_bars = 0
                if eng.futures_long and pnl_r < long_sl:
                    eng.close_long(price, dt, f"止损 {pnl_r*100:.0f}%")
                    long_max_pnl = 0; long_cd = cooldown * 4; long_bars = 0
                if eng.futures_long and long_bars >= long_max_hold:
                    eng.close_long(price, dt, f"超时 {long_bars}h")
                    long_max_pnl = 0; long_cd = cooldown; long_bars = 0
        elif eng.futures_long and long_just_opened:
            long_bars = 1

        if idx % 4 == 0:
            eng.record_history(dt, price)

    # 期末平仓
    last_price = main_df['close'].iloc[-1]
    last_dt = main_df.index[-1]
    if eng.futures_short: eng.close_short(last_price, last_dt, "期末平仓")
    if eng.futures_long: eng.close_long(last_price, last_dt, "期末平仓")

    if start_dt:
        trade_df = main_df[main_df.index >= start_dt]
        if len(trade_df) > 1:
            return eng.get_result(trade_df)
    return eng.get_result(main_df)


# ======================================================
#   策略变体
# ======================================================
def get_strategies():
    """8种融合策略"""
    base = {
        'single_pct': 0.15, 'total_pct': 0.40, 'lifetime_pct': 5.0,
        'sell_threshold': 25, 'buy_threshold': 25,
        'short_threshold': 35, 'long_threshold': 35,
        'close_short_bs': 40, 'close_long_ss': 40,
        'sell_pct': 0.45, 'buy_pct': 0.25, 'margin_use': 0.50, 'lev': 3,
        'short_sl': -0.20, 'short_tp': 0.60, 'short_trail': 0.25,
        'short_max_hold': 72, 'long_sl': -0.15, 'long_tp': 0.50,
        'long_trail': 0.20, 'long_max_hold': 72,
        'cooldown': 4, 'spot_cooldown': 12, 'max_lev': 5,
        'fusion_mode': 'weighted', 'div_weight': 0.6, 'ma_weight': 0.4,
    }

    return [
        # C1: 加权融合(标准) — 背离60%+均线40%
        {**base, 'name': 'C1: 加权融合(6:4)',
         'div_weight': 0.6, 'ma_weight': 0.4},

        # C2: 加权融合(均线主导) — 均线60%+背离40%
        {**base, 'name': 'C2: 加权融合(4:6)',
         'div_weight': 0.4, 'ma_weight': 0.6},

        # C3: 共振确认 — 两个系统都给出信号才交易
        {**base, 'name': 'C3: 共振确认',
         'fusion_mode': 'resonance',
         'sell_threshold': 30, 'buy_threshold': 30,
         'short_threshold': 40, 'long_threshold': 40},

        # C4: 均线过滤背离 — 背离产生信号, 均线确认方向
        {**base, 'name': 'C4: 均线过滤背离',
         'fusion_mode': 'ma_filter',
         'sell_threshold': 20, 'buy_threshold': 20,
         'short_threshold': 30, 'long_threshold': 30},

        # C5: 背离过滤均线 — 均线产生信号, 背离确认拐点
        {**base, 'name': 'C5: 背离过滤均线',
         'fusion_mode': 'div_filter',
         'sell_threshold': 20, 'buy_threshold': 20,
         'short_threshold': 30, 'long_threshold': 30},

        # C6: 激进融合做空 — 加权融合 + 高杠杆 + 低阈值
        {**base, 'name': 'C6: 激进融合做空',
         'div_weight': 0.7, 'ma_weight': 0.3,
         'sell_threshold': 18, 'short_threshold': 25,
         'buy_threshold': 25, 'long_threshold': 40,
         'lev': 5, 'max_lev': 5, 'margin_use': 0.70,
         'single_pct': 0.20, 'total_pct': 0.50,
         'sell_pct': 0.55, 'short_sl': -0.30, 'short_tp': 0.80},

        # C7: 保守融合 — 共振+高阈值+低杠杆
        {**base, 'name': 'C7: 保守融合',
         'fusion_mode': 'resonance',
         'sell_threshold': 40, 'buy_threshold': 40,
         'short_threshold': 50, 'long_threshold': 50,
         'lev': 2, 'max_lev': 2, 'margin_use': 0.30,
         'sell_pct': 0.30, 'buy_pct': 0.20,
         'short_sl': -0.12, 'long_sl': -0.10,
         'spot_cooldown': 24},

        # C8: 趋势加速 — 均线确认趋势+背离触发入场+大仓位
        {**base, 'name': 'C8: 趋势加速',
         'fusion_mode': 'ma_filter',
         'div_weight': 0.7, 'ma_weight': 0.3,
         'sell_threshold': 20, 'short_threshold': 28,
         'buy_threshold': 20, 'long_threshold': 28,
         'lev': 4, 'margin_use': 0.65,
         'single_pct': 0.20, 'total_pct': 0.50,
         'short_max_hold': 96, 'long_max_hold': 96},
    ]


# ======================================================
#   主函数
# ======================================================
def main(trade_days=None):
    if trade_days is None:
        trade_days = 7

    data = fetch_data()
    if '1h' not in data:
        print("错误: 无法获取1h数据")
        return

    main_df = data['1h']

    # 计算背离信号
    print("\n计算背离信号...")
    div_signals_1h = analyze_signals_enhanced(main_df, 168)
    print(f"  1h背离: {len(div_signals_1h)} 个信号点")

    div_signals_8h = {}
    if '8h' in data:
        div_signals_8h = analyze_signals_enhanced(data['8h'], 90)
        print(f"  8h背离: {len(div_signals_8h)} 个信号点")

    # 计算均线信号
    print("计算均线信号...")
    ma_signals = compute_ma_signals(main_df, timeframe='1h')
    ma_buy_count = int((ma_signals['buy_score'] > 15).sum())
    ma_sell_count = int((ma_signals['sell_score'] > 15).sum())
    print(f"  均线买入: {ma_buy_count}次 | 均线卖出: {ma_sell_count}次")

    # 运行策略
    strategies = get_strategies()
    all_results = []

    start_dt = main_df.index[-1] - pd.Timedelta(days=trade_days)
    trade_start = str(start_dt)[:16]
    trade_end = str(main_df.index[-1])[:16]

    print(f"\n{'=' * 110}")
    print(f"  双书融合策略 · {len(strategies)}种 · 最近{trade_days}天")
    print(f"  信号: {len(main_df)}根1h K线 | 交易: {trade_start} ~ {trade_end}")
    print(f"  融合: 背离技术(MACD/KDJ/CCI/RSI/量价) × 均线技术(葛南维/排列/交叉/形态)")
    print(f"{'=' * 110}")

    print(f"\n{'策略':<24} {'α':>8} {'收益':>10} {'BH':>10} {'回撤':>8} "
          f"{'交易':>6} {'强平':>4} {'费用':>10}")
    print('-' * 110)

    for cfg in strategies:
        r = run_combined_strategy(data, div_signals_1h, div_signals_8h, ma_signals,
                                  cfg, trade_days=trade_days)
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
    print(f"\n  最佳策略: {best['name']} · 前20笔交易")
    for t in best.get('trades', [])[:20]:
        print(f"  {str(t['time'])[:16]}  {t.get('action',''):<14} @${t.get('price',0):>8,.2f} "
              f"{t.get('leverage',1)}x  ${t.get('total',0):>10,.0f}  {t.get('reason','')[:55]}")

    bf = best.get('fees', {})
    print(f"\n  费用: 现货${bf.get('spot_fees',0):,.0f} 合约${bf.get('futures_fees',0):,.0f} "
          f"资金费${bf.get('net_funding',0):,.0f} 滑点${bf.get('slippage_cost',0):,.0f} "
          f"总${bf.get('total_costs',0):,.0f}")

    # 保存
    output = {
        'description': f'双书融合策略 · 最近{trade_days}天',
        'books': ['《背离技术分析》江南小隐', '《均线技术分析》邱立波'],
        'run_time': datetime.now().isoformat(),
        'data_range': f"{main_df.index[0]} ~ {main_df.index[-1]}",
        'trade_range': f"{trade_start} ~ {trade_end}",
        'trade_days': trade_days,
        'total_bars': len(main_df),
        'initial_capital': '10万USDT + 价值10万USDT的ETH',
        'timeframe': '1h',
        'signal_summary': {
            'div_1h_count': len(div_signals_1h),
            'div_8h_count': len(div_signals_8h),
            'ma_buy_count': ma_buy_count,
            'ma_sell_count': ma_sell_count,
        },
        'results': [{
            'name': r['name'], 'alpha': r['alpha'],
            'strategy_return': r['strategy_return'],
            'buy_hold_return': r['buy_hold_return'],
            'max_drawdown': r['max_drawdown'],
            'total_trades': r['total_trades'],
            'liquidations': r['liquidations'],
            'final_total': r['final_total'],
            'fees': r.get('fees', {}),
            'trades': r.get('trades', []),
            'history': r.get('history', []),
        } for r in all_results],
        'ranking': [{'rank': i + 1, 'name': r['name'], 'alpha': r['alpha']}
                    for i, r in enumerate(ranked)],
        'best_strategy': {
            'name': best['name'], 'alpha': best['alpha'],
            'strategy_return': best['strategy_return'],
            'trades_count': best['total_trades'],
        },
    }

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'combined_strategy_result.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果已保存: {path}")
    return output


if __name__ == '__main__':
    td = 7
    if len(sys.argv) > 1:
        try:
            td = max(1, min(60, int(sys.argv[1])))
        except ValueError:
            pass
    main(trade_days=td)
