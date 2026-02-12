"""
布林带策略 — 基于《布林线》(John Bollinger) 珍藏版

核心指标:
  布林带: MA(20) ± 2σ
  %B: (价格-下轨) / (上轨-下轨)  → 0~1, 超买>1, 超卖<0
  BandWidth: (上轨-下轨) / 中轨  → 衡量波动率
  Squeeze: 带宽收窄(低波动) → 即将爆发

核心策略:
  1. 均值回归: 价格触及上轨做空, 触及下轨做多
  2. 趋势跟随: "走在带上"(Walking the Bands) — 强势持续触及上/下轨
  3. W底 / M顶: 经典双底/双顶形态+%B确认
  4. Squeeze突破: 带宽缩至极低后的方向突破
  5. %B + 指标: %B与MFI/RSI等结合判断超买超卖

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
from strategy_futures import FuturesEngine


# ======================================================
#   布林带指标计算
# ======================================================
def compute_bollinger(df, period=20, std_mult=2.0):
    """计算布林带及衍生指标"""
    close = df['close']

    # 中轨 = SMA(period)
    df['bb_mid'] = close.rolling(period, min_periods=period).mean()

    # 标准差
    rolling_std = close.rolling(period, min_periods=period).std()

    # 上轨 / 下轨
    df['bb_upper'] = df['bb_mid'] + std_mult * rolling_std
    df['bb_lower'] = df['bb_mid'] - std_mult * rolling_std

    # %B = (价格 - 下轨) / (上轨 - 下轨)
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_pct_b'] = (close - df['bb_lower']) / bb_range.replace(0, np.nan)

    # BandWidth = (上轨 - 下轨) / 中轨 * 100
    df['bb_bandwidth'] = bb_range / df['bb_mid'].replace(0, np.nan) * 100

    # BandWidth的移动平均(用于检测Squeeze)
    df['bb_bw_sma'] = df['bb_bandwidth'].rolling(120, min_periods=20).mean()

    # %B的均线(平滑)
    df['bb_pct_b_sma'] = df['bb_pct_b'].rolling(5, min_periods=3).mean()

    # 布林带斜率(中轨方向)
    df['bb_mid_slope'] = (df['bb_mid'] - df['bb_mid'].shift(5)) / df['bb_mid'].shift(5) * 100

    return df


def detect_squeeze(df, i, lookback=120):
    """检测Squeeze(带宽缩窄): 当前带宽 < 过去N期最低的前20%"""
    if i < lookback:
        return False, 0

    bw = df['bb_bandwidth'].iloc[max(0, i - lookback):i + 1]
    bw_valid = bw.dropna()
    if len(bw_valid) < 20:
        return False, 0

    current_bw = df['bb_bandwidth'].iloc[i]
    if pd.isna(current_bw):
        return False, 0

    percentile = (bw_valid < current_bw).sum() / len(bw_valid) * 100
    is_squeeze = percentile < 20  # 低于20百分位 = Squeeze
    return is_squeeze, percentile


def detect_w_bottom(df, i, lookback=30):
    """W底形态: 两个低点(第二个%B更高), 中间有反弹"""
    if i < lookback + 5:
        return False, 0

    pct_b = df['bb_pct_b'].iloc[max(0, i - lookback):i + 1]
    close = df['close'].iloc[max(0, i - lookback):i + 1]

    # 找到最近两个%B低于0.2的点
    low_points = []
    in_low = False
    for j in range(len(pct_b)):
        val = pct_b.iloc[j]
        if pd.isna(val):
            continue
        if val < 0.2 and not in_low:
            in_low = True
            low_points.append((j, val, close.iloc[j]))
        elif val > 0.5:
            in_low = False

    if len(low_points) < 2:
        return False, 0

    # 最近两个低点
    p1, p2 = low_points[-2], low_points[-1]

    # W底: 第二个低点的价格更低或差不多, 但%B更高
    if p2[1] > p1[1] and p2[2] <= p1[2] * 1.02:
        # 中间有反弹(%B > 0.5)
        mid_vals = pct_b.iloc[p1[0]:p2[0]]
        if any(v > 0.5 for v in mid_vals if not pd.isna(v)):
            score = min(int((p2[1] - p1[1]) * 100) + 30, 80)
            return True, score

    return False, 0


def detect_m_top(df, i, lookback=30):
    """M顶形态: 两个高点(第二个%B更低), 中间有回落"""
    if i < lookback + 5:
        return False, 0

    pct_b = df['bb_pct_b'].iloc[max(0, i - lookback):i + 1]
    close = df['close'].iloc[max(0, i - lookback):i + 1]

    high_points = []
    in_high = False
    for j in range(len(pct_b)):
        val = pct_b.iloc[j]
        if pd.isna(val):
            continue
        if val > 0.8 and not in_high:
            in_high = True
            high_points.append((j, val, close.iloc[j]))
        elif val < 0.5:
            in_high = False

    if len(high_points) < 2:
        return False, 0

    p1, p2 = high_points[-2], high_points[-1]

    if p2[1] < p1[1] and p2[2] >= p1[2] * 0.98:
        mid_vals = pct_b.iloc[p1[0]:p2[0]]
        if any(v < 0.5 for v in mid_vals if not pd.isna(v)):
            score = min(int((p1[1] - p2[1]) * 100) + 30, 80)
            return True, score

    return False, 0


# ======================================================
#   布林带信号综合评分
# ======================================================
def compute_bollinger_scores(df):
    """计算每根K线的布林带综合得分"""
    df = compute_bollinger(df)

    sell_scores = pd.Series(0.0, index=df.index)
    buy_scores = pd.Series(0.0, index=df.index)
    signal_names = pd.Series('', index=df.index, dtype=str)

    for i in range(25, len(df)):
        ss = 0  # 卖出分
        bs = 0  # 买入分
        reasons = []

        pct_b = df['bb_pct_b'].iloc[i]
        bw = df['bb_bandwidth'].iloc[i]
        slope = df['bb_mid_slope'].iloc[i]
        close = df['close'].iloc[i]
        upper = df['bb_upper'].iloc[i]
        lower = df['bb_lower'].iloc[i]
        mid = df['bb_mid'].iloc[i]

        if pd.isna(pct_b) or pd.isna(bw):
            continue

        # === 1. %B 超买/超卖 ===
        if pct_b > 1.0:
            # 价格突破上轨 — 可能超买或强势
            if slope < 0:
                ss += 25
                reasons.append(f"%B={pct_b:.2f}超买+下行")
            elif slope > 0.5:
                # 强势上行,触及上轨 = "走在上轨"(趋势延续)
                bs += 10  # 轻微看涨
                reasons.append(f"走在上轨%B={pct_b:.2f}")
            else:
                ss += 15
                reasons.append(f"%B={pct_b:.2f}超买")

        elif pct_b < 0:
            if slope > 0:
                bs += 25
                reasons.append(f"%B={pct_b:.2f}超卖+上行")
            elif slope < -0.5:
                ss += 10
                reasons.append(f"走在下轨%B={pct_b:.2f}")
            else:
                bs += 15
                reasons.append(f"%B={pct_b:.2f}超卖")

        elif pct_b > 0.85:
            ss += int((pct_b - 0.85) * 100)
            reasons.append(f"%B={pct_b:.2f}偏高")
        elif pct_b < 0.15:
            bs += int((0.15 - pct_b) * 100)
            reasons.append(f"%B={pct_b:.2f}偏低")

        # === 2. Squeeze 检测 ===
        is_squeeze, squeeze_pct = detect_squeeze(df, i)
        if is_squeeze:
            # Squeeze中 — 准备爆发,看最新方向
            if close > mid and slope > 0:
                bs += 20
                reasons.append(f"Squeeze向上突破(P={squeeze_pct:.0f})")
            elif close < mid and slope < 0:
                ss += 20
                reasons.append(f"Squeeze向下突破(P={squeeze_pct:.0f})")
            else:
                reasons.append(f"Squeeze({squeeze_pct:.0f}%)")

        # === 3. 带宽扩张(波动率增大) ===
        bw_sma = df['bb_bw_sma'].iloc[i]
        if not pd.isna(bw_sma) and bw > bw_sma * 1.5:
            # 带宽扩张 = 趋势行情
            if close > upper:
                bs += 15
                reasons.append("带宽扩张+突破上轨")
            elif close < lower:
                ss += 15
                reasons.append("带宽扩张+跌破下轨")

        # === 4. W底 / M顶 ===
        is_w, w_score = detect_w_bottom(df, i)
        if is_w:
            bs += w_score
            reasons.append(f"W底({w_score})")

        is_m, m_score = detect_m_top(df, i)
        if is_m:
            ss += m_score
            reasons.append(f"M顶({m_score})")

        # === 5. 中轨支撑/阻力 ===
        prev_close = df['close'].iloc[i - 1]
        if prev_close < mid and close > mid and slope > 0:
            bs += 10
            reasons.append("上穿中轨")
        elif prev_close > mid and close < mid and slope < 0:
            ss += 10
            reasons.append("下穿中轨")

        # === 6. 收缩后方向确认 ===
        if i >= 5:
            recent_bw = df['bb_bandwidth'].iloc[i - 5:i + 1]
            bw_expanding = all(
                recent_bw.iloc[j] > recent_bw.iloc[j - 1]
                for j in range(1, len(recent_bw))
                if not pd.isna(recent_bw.iloc[j]) and not pd.isna(recent_bw.iloc[j - 1])
            )
            if bw_expanding and len(recent_bw.dropna()) >= 4:
                if close > upper:
                    bs += 15
                    reasons.append("带宽连续扩张+上破")
                elif close < lower:
                    ss += 15
                    reasons.append("带宽连续扩张+下破")

        sell_scores.iloc[i] = min(ss, 100)
        buy_scores.iloc[i] = min(bs, 100)
        signal_names.iloc[i] = ','.join(reasons[:3])

    return sell_scores, buy_scores, signal_names


# ======================================================
#   布林带策略回测
# ======================================================
def fetch_data():
    print("获取数据...")
    data = {}
    for tf, days in [('1h', 60), ('4h', 60)]:
        df = fetch_binance_klines("ETHUSDT", interval=tf, days=days)
        if df is not None and len(df) > 50:
            df = add_all_indicators(df)
            data[tf] = df
            print(f"  {tf}: {len(df)} 条 ({df.index[0]} ~ {df.index[-1]})")
    return data


def run_bollinger_strategy(data, config, trade_days=None):
    """布林带策略回测"""
    eng = FuturesEngine(config['name'], max_leverage=config.get('max_lev', 5))
    main_df = data['1h']

    sell_sc, buy_sc, sig_names = compute_bollinger_scores(main_df)

    # 4h辅助
    sell_sc_4h = pd.Series(0.0, index=main_df.index)
    buy_sc_4h = pd.Series(0.0, index=main_df.index)
    if '4h' in data:
        s4, b4, _ = compute_bollinger_scores(data['4h'])
        for ii in range(len(data['4h'])):
            dt4 = data['4h'].index[ii]
            idx1h = main_df.index.searchsorted(dt4)
            if idx1h < len(main_df):
                for k in range(idx1h, min(idx1h + 4, len(main_df))):
                    sell_sc_4h.iloc[k] = max(sell_sc_4h.iloc[k], s4.iloc[ii] * 0.4)
                    buy_sc_4h.iloc[k] = max(buy_sc_4h.iloc[k], b4.iloc[ii] * 0.4)

    start_dt = None
    if trade_days and trade_days < 60:
        start_dt = main_df.index[-1] - pd.Timedelta(days=trade_days)

    init_idx = 0
    if start_dt:
        init_idx = main_df.index.searchsorted(start_dt)
        if init_idx >= len(main_df): init_idx = 0
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[init_idx]

    eng.max_single_margin = eng.initial_total * config.get('single_pct', 0.15)
    eng.max_margin_total = eng.initial_total * config.get('total_pct', 0.40)
    eng.max_lifetime_margin = eng.initial_total * 5.0

    sell_threshold = config.get('sell_threshold', 30)
    buy_threshold = config.get('buy_threshold', 30)
    short_threshold = config.get('short_threshold', 40)
    long_threshold = config.get('long_threshold', 40)
    sell_pct = config.get('sell_pct', 0.40)
    margin_use = config.get('margin_use', 0.50)
    lev = config.get('lev', 3)
    cooldown = config.get('cooldown', 4)
    spot_cooldown = config.get('spot_cooldown', 12)

    short_sl = config.get('short_sl', -0.20)
    short_tp = config.get('short_tp', 0.60)
    short_trail = config.get('short_trail', 0.25)
    short_max_hold = config.get('short_max_hold', 72)
    long_sl = config.get('long_sl', -0.15)
    long_tp = config.get('long_tp', 0.50)
    long_trail = config.get('long_trail', 0.20)
    long_max_hold = config.get('long_max_hold', 72)

    short_cd = 0; long_cd = 0; spot_cd = 0
    short_bars = 0; long_bars = 0
    short_max_pnl = 0; long_max_pnl = 0

    for idx in range(30, len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]

        if start_dt and dt < start_dt:
            if idx % 4 == 0: eng.record_history(dt, price)
            continue

        eng.check_liquidation(price, dt)
        short_just_opened = False; long_just_opened = False

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

        ss = sell_sc.iloc[idx] + sell_sc_4h.iloc[idx]
        bs = buy_sc.iloc[idx] + buy_sc_4h.iloc[idx]
        reason = sig_names.iloc[idx][:60] if sig_names.iloc[idx] else ''

        in_conflict = False
        if ss > 0 and bs > 0:
            ratio = min(ss, bs) / max(ss, bs)
            in_conflict = ratio >= 0.6 and min(ss, bs) >= 15

        # 卖出现货
        if ss >= sell_threshold and spot_cd == 0 and not in_conflict and eng.spot_eth * price > 500:
            eng.spot_sell(price, dt, sell_pct, f"BB卖 {reason}")
            spot_cd = spot_cooldown

        # 开空
        sell_dom = (ss > bs * 1.5) if bs > 0 else True
        if short_cd == 0 and ss >= short_threshold and not eng.futures_short and sell_dom:
            margin = eng.available_margin() * margin_use
            actual_lev = min(lev if ss >= 60 else 2, eng.max_leverage)
            eng.open_short(price, dt, margin, actual_lev, f"BB空 {actual_lev}x {reason}")
            short_max_pnl = 0; short_bars = 0; short_cd = cooldown; short_just_opened = True

        # 管理空仓
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
                        eng.close_short(price, dt, "追踪止盈")
                        short_max_pnl = 0; short_cd = cooldown; short_bars = 0
                if eng.futures_short and bs >= 50:
                    eng.close_short(price, dt, "反向信号平空")
                    short_max_pnl = 0; short_cd = cooldown * 3; short_bars = 0
                if eng.futures_short and pnl_r < short_sl:
                    eng.close_short(price, dt, f"止损 {pnl_r*100:.0f}%")
                    short_max_pnl = 0; short_cd = cooldown * 4; short_bars = 0
                if eng.futures_short and short_bars >= short_max_hold:
                    eng.close_short(price, dt, "超时")
                    short_max_pnl = 0; short_cd = cooldown; short_bars = 0
        elif eng.futures_short and short_just_opened:
            short_bars = 1

        # 买入现货
        if bs >= buy_threshold and spot_cd == 0 and not in_conflict and eng.available_usdt() > 500:
            eng.spot_buy(price, dt, eng.available_usdt() * 0.25, f"BB买 {reason}")
            spot_cd = spot_cooldown

        # 开多
        buy_dom = (bs > ss * 1.5) if ss > 0 else True
        if long_cd == 0 and bs >= long_threshold and not eng.futures_long and buy_dom:
            margin = eng.available_margin() * margin_use
            actual_lev = min(lev if bs >= 60 else 2, eng.max_leverage)
            eng.open_long(price, dt, margin, actual_lev, f"BB多 {actual_lev}x {reason}")
            long_max_pnl = 0; long_bars = 0; long_cd = cooldown; long_just_opened = True

        # 管理多仓
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
                        eng.close_long(price, dt, "追踪止盈")
                        long_max_pnl = 0; long_cd = cooldown; long_bars = 0
                if eng.futures_long and ss >= 50:
                    eng.close_long(price, dt, "反向信号平多")
                    long_max_pnl = 0; long_cd = cooldown * 3; long_bars = 0
                if eng.futures_long and pnl_r < long_sl:
                    eng.close_long(price, dt, f"止损 {pnl_r*100:.0f}%")
                    long_max_pnl = 0; long_cd = cooldown * 4; long_bars = 0
                if eng.futures_long and long_bars >= long_max_hold:
                    eng.close_long(price, dt, "超时")
                    long_max_pnl = 0; long_cd = cooldown; long_bars = 0
        elif eng.futures_long and long_just_opened:
            long_bars = 1

        if idx % 4 == 0: eng.record_history(dt, price)

    last_price = main_df['close'].iloc[-1]
    last_dt = main_df.index[-1]
    if eng.futures_short: eng.close_short(last_price, last_dt, "期末平仓")
    if eng.futures_long: eng.close_long(last_price, last_dt, "期末平仓")

    if start_dt:
        trade_df = main_df[main_df.index >= start_dt]
        if len(trade_df) > 1: return eng.get_result(trade_df)
    return eng.get_result(main_df)


def get_strategies():
    base = {
        'single_pct': 0.15, 'total_pct': 0.40,
        'sell_threshold': 30, 'buy_threshold': 30,
        'short_threshold': 45, 'long_threshold': 45,
        'sell_pct': 0.40, 'margin_use': 0.50, 'lev': 3,
        'short_sl': -0.20, 'short_tp': 0.60, 'short_trail': 0.25,
        'short_max_hold': 72, 'long_sl': -0.15, 'long_tp': 0.50,
        'long_trail': 0.20, 'long_max_hold': 72,
        'cooldown': 4, 'spot_cooldown': 12, 'max_lev': 5,
    }

    return [
        {**base, 'name': 'B1: 标准布林带'},

        {**base, 'name': 'B2: 激进做空',
         'sell_threshold': 20, 'short_threshold': 30,
         'lev': 5, 'margin_use': 0.70, 'sell_pct': 0.55,
         'short_sl': -0.30, 'short_tp': 0.80},

        {**base, 'name': 'B3: Squeeze专注',
         'sell_threshold': 35, 'buy_threshold': 35,
         'short_threshold': 50, 'long_threshold': 50,
         'lev': 4, 'margin_use': 0.60,
         'short_max_hold': 96, 'long_max_hold': 96},

        {**base, 'name': 'B4: 保守均值回归',
         'sell_threshold': 40, 'buy_threshold': 40,
         'short_threshold': 55, 'long_threshold': 55,
         'lev': 2, 'margin_use': 0.30, 'sell_pct': 0.30,
         'short_sl': -0.12, 'long_sl': -0.10,
         'spot_cooldown': 24},
    ]


def main(trade_days=None):
    if trade_days is None:
        trade_days = 30

    data = fetch_data()
    if '1h' not in data:
        print("错误: 无法获取1h数据")
        return

    main_df = data['1h']

    print("\n计算布林带指标...")
    main_df = compute_bollinger(main_df)

    # 统计
    squeeze_count = 0
    for i in range(120, len(main_df)):
        sq, _ = detect_squeeze(main_df, i)
        if sq: squeeze_count += 1
    print(f"  Squeeze事件: {squeeze_count}次")

    w_count = 0; m_count = 0
    for i in range(50, len(main_df)):
        w, _ = detect_w_bottom(main_df, i)
        if w: w_count += 1
        m, _ = detect_m_top(main_df, i)
        if m: m_count += 1
    print(f"  W底: {w_count}次 | M顶: {m_count}次")

    strategies = get_strategies()
    all_results = []

    start_dt = main_df.index[-1] - pd.Timedelta(days=trade_days)
    trade_start = str(start_dt)[:16]
    trade_end = str(main_df.index[-1])[:16]

    print(f"\n{'=' * 100}")
    print(f"  布林带策略 · 《布林线》 · {len(strategies)}种 · 最近{trade_days}天")
    print(f"  数据: {len(main_df)}根1h K线 | 交易: {trade_start} ~ {trade_end}")
    print(f"{'=' * 100}")

    print(f"\n{'策略':<20} {'α':>8} {'收益':>10} {'BH':>10} {'回撤':>8} {'交易':>6} {'费用':>10}")
    print('-' * 100)

    for cfg in strategies:
        r = run_bollinger_strategy(data, cfg, trade_days=trade_days)
        all_results.append(r)
        fees = r.get('fees', {})
        print(f"  {cfg['name']:<18} {r['alpha']:>+7.2f}% {r['strategy_return']:>+9.2f}% "
              f"{r['buy_hold_return']:>+9.2f}% {r['max_drawdown']:>7.2f}% "
              f"{r['total_trades']:>5} ${fees.get('total_costs', 0):>9,.0f}")

    ranked = sorted(all_results, key=lambda x: x['alpha'], reverse=True)
    print(f"\n排名:")
    for i, r in enumerate(ranked):
        star = ' ★' if i == 0 else ''
        print(f"  #{i + 1}: {r['name']:<20} α={r['alpha']:+.2f}%{star}")

    output = {
        'description': f'布林带策略 · 最近{trade_days}天',
        'book': '《布林线》John Bollinger',
        'run_time': datetime.now().isoformat(),
        'trade_days': trade_days,
        'squeeze_count': squeeze_count,
        'w_bottom_count': w_count,
        'm_top_count': m_count,
        'initial_capital': '10万USDT + 价值10万USDT的ETH',
        'timeframe': '1h',
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
    }

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'bollinger_result.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果已保存: {path}")
    return output


if __name__ == '__main__':
    td = 30
    if len(sys.argv) > 1:
        try: td = max(1, min(60, int(sys.argv[1])))
        except ValueError: pass
    main(trade_days=td)
