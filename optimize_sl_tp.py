"""
多时间框架止盈止损优化器

基于F12(C6+三书否决)最佳策略,系统性优化:
1. 止损(SL)参数
2. 止盈(TP)参数
3. 追踪止盈(Trailing Stop)参数
4. 最大持仓时间(Max Hold)
5. 冷却期(Cooldown)
6. 信号阈值

在12个时间周期上全面测试:
10min, 15min, 30min, 1h, 2h, 3h, 4h, 6h, 8h, 12h, 16h, 24h
"""

import pandas as pd
import numpy as np
import json
import sys
import os
import time
from datetime import datetime
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binance_fetcher import fetch_binance_klines
from indicators import add_all_indicators
from strategy_enhanced import analyze_signals_enhanced, get_signal_at, DEFAULT_SIG
from strategy_futures import FuturesEngine
from strategy_futures_v4 import _calc_top_score, _calc_bottom_score
from ma_indicators import add_moving_averages, compute_ma_signals
from candlestick_patterns import compute_candlestick_scores
from bollinger_strategy import compute_bollinger_scores
from volume_price_strategy import compute_volume_price_scores


# ======================================================
#   多周期数据获取
# ======================================================
ALL_TIMEFRAMES = ['10m', '15m', '30m', '1h', '2h', '3h', '4h', '6h', '8h', '12h', '16h', '24h']

def fetch_multi_tf_data(timeframes=None, days=60):
    """获取多时间周期数据"""
    if timeframes is None:
        timeframes = ALL_TIMEFRAMES
    data = {}
    for tf in timeframes:
        print(f"\n--- 获取 {tf} 数据 ---")
        try:
            df = fetch_binance_klines("ETHUSDT", interval=tf, days=days)
            if df is not None and len(df) > 50:
                df = add_all_indicators(df)
                add_moving_averages(df, timeframe=tf)
                data[tf] = df
                print(f"  {tf}: {len(df)} 条K线")
            else:
                print(f"  {tf}: 数据不足, 跳过")
        except Exception as e:
            print(f"  {tf}: 获取失败 - {e}")
    return data


def compute_signals_for_tf(df, tf, data_all):
    """为指定时间框架计算五维信号"""
    signals = {}

    # 1. 背离信号(主周期)
    lookback = max(60, min(200, len(df) // 3))
    div_signals = analyze_signals_enhanced(df, lookback)
    signals['div'] = div_signals

    # 8h辅助背离(如果可用)
    signals['div_8h'] = {}
    if '8h' in data_all and tf not in ('8h', '12h', '16h', '24h'):
        signals['div_8h'] = analyze_signals_enhanced(data_all['8h'], 90)

    # 2. 均线信号
    ma_signals = compute_ma_signals(df, timeframe=tf)
    signals['ma'] = ma_signals

    # 3. 蜡烛图
    cs_sell, cs_buy, cs_names = compute_candlestick_scores(df)
    signals['cs_sell'] = cs_sell
    signals['cs_buy'] = cs_buy

    # 4. 布林带
    bb_sell, bb_buy, bb_names = compute_bollinger_scores(df)
    signals['bb_sell'] = bb_sell
    signals['bb_buy'] = bb_buy

    # 5. 量价
    vp_sell, vp_buy, vp_names = compute_volume_price_scores(df)
    signals['vp_sell'] = vp_sell
    signals['vp_buy'] = vp_buy

    return signals


def calc_fusion_score_generic(signals, df, idx, dt, config):
    """通用的五维融合评分(c6_veto模式)"""
    price = df['close'].iloc[idx]

    # 1. 背离
    sig_main = get_signal_at(signals['div'], dt) or dict(DEFAULT_SIG)
    sig_8h = get_signal_at(signals.get('div_8h', {}), dt) or dict(DEFAULT_SIG)

    merged = dict(DEFAULT_SIG)
    merged['top'] = 0; merged['bottom'] = 0
    for sig_src, w in [(sig_main, 1.0), (sig_8h, 0.5)]:
        merged['top'] += sig_src.get('top', 0) * w
        merged['bottom'] += sig_src.get('bottom', 0) * w
        for k in DEFAULT_SIG:
            if isinstance(DEFAULT_SIG[k], bool) and sig_src.get(k):
                merged[k] = True
            elif isinstance(DEFAULT_SIG[k], (int, float)) and k not in ('top', 'bottom'):
                merged[k] = max(merged.get(k, 0), sig_src.get(k, 0))

    trend = {'is_downtrend': False, 'is_uptrend': False,
             'ma_bearish': False, 'ma_bullish': False,
             'ma_slope_down': False, 'ma_slope_up': False}
    if idx >= 30:
        c5 = df['close'].iloc[max(0, idx-5):idx].mean()
        c20 = df['close'].iloc[max(0, idx-20):idx].mean()
        if c5 < c20 * 0.99: trend['is_downtrend'] = True
        elif c5 > c20 * 1.01: trend['is_uptrend'] = True

    div_sell, _ = _calc_top_score(merged, trend)
    div_buy = _calc_bottom_score(merged, trend)

    # 2. 均线
    ma_sell = float(signals['ma']['sell_score'].iloc[idx]) if idx < len(signals['ma']['sell_score']) else 0
    ma_buy = float(signals['ma']['buy_score'].iloc[idx]) if idx < len(signals['ma']['buy_score']) else 0

    # 3-5. K线、布林、量价
    cs_sell = float(signals['cs_sell'].iloc[idx]) if idx < len(signals['cs_sell']) else 0
    cs_buy = float(signals['cs_buy'].iloc[idx]) if idx < len(signals['cs_buy']) else 0
    bb_sell = float(signals['bb_sell'].iloc[idx]) if idx < len(signals['bb_sell']) else 0
    bb_buy = float(signals['bb_buy'].iloc[idx]) if idx < len(signals['bb_buy']) else 0
    vp_sell = float(signals['vp_sell'].iloc[idx]) if idx < len(signals['vp_sell']) else 0
    vp_buy = float(signals['vp_buy'].iloc[idx]) if idx < len(signals['vp_buy']) else 0

    # C6 veto逻辑
    base_sell = div_sell * 0.7 + ma_sell * 0.3
    base_buy = div_buy * 0.7 + ma_buy * 0.3

    # MA排列加成
    if idx < len(signals['ma']['sell_score']):
        ma_data = signals['ma']
        if 'arrangement' in ma_data:
            arr_series = ma_data.get('arrangement', None)
            if arr_series is not None and hasattr(arr_series, 'iloc') and idx < len(arr_series):
                try:
                    arr_val = float(arr_series.iloc[idx])
                    if arr_val < 0: base_sell *= 1.10
                    elif arr_val > 0: base_buy *= 1.10
                except (ValueError, TypeError):
                    pass

    veto_threshold = config.get('veto_threshold', 25)

    # 否决逻辑
    sell_vetoes = sum(1 for s in [bb_buy, vp_buy, cs_buy] if s >= veto_threshold)
    buy_vetoes = sum(1 for s in [bb_sell, vp_sell, cs_sell] if s >= veto_threshold)

    if sell_vetoes >= 2:
        sell_score = base_sell * 0.3
    else:
        sell_bonus = 0
        if bb_sell >= 15: sell_bonus += 0.10
        if vp_sell >= 15: sell_bonus += 0.08
        if cs_sell >= 25: sell_bonus += 0.06
        sell_score = base_sell * (1 + sell_bonus)

    if buy_vetoes >= 2:
        buy_score = base_buy * 0.3
    else:
        buy_bonus = 0
        if bb_buy >= 15: buy_bonus += 0.10
        if vp_buy >= 15: buy_bonus += 0.08
        if cs_buy >= 25: buy_bonus += 0.06
        buy_score = base_buy * (1 + buy_bonus)

    return sell_score, buy_score


# ======================================================
#   通用回测引擎(适配多时间框架)
# ======================================================
def run_strategy(df, signals, config, tf='1h', trade_days=30):
    """在指定时间框架上运行策略回测"""
    eng = FuturesEngine(config.get('name', 'opt'), max_leverage=config.get('max_lev', 5))

    start_dt = None
    if trade_days and trade_days < 60:
        start_dt = df.index[-1] - pd.Timedelta(days=trade_days)

    init_idx = 0
    if start_dt:
        init_idx = df.index.searchsorted(start_dt)
        if init_idx >= len(df): init_idx = 0
    eng.spot_eth = eng.initial_eth_value / df['close'].iloc[init_idx]

    eng.max_single_margin = eng.initial_total * config.get('single_pct', 0.20)
    eng.max_margin_total = eng.initial_total * config.get('total_pct', 0.50)
    eng.max_lifetime_margin = eng.initial_total * config.get('lifetime_pct', 5.0)

    sell_threshold = config.get('sell_threshold', 18)
    buy_threshold = config.get('buy_threshold', 25)
    short_threshold = config.get('short_threshold', 25)
    long_threshold = config.get('long_threshold', 40)
    sell_pct = config.get('sell_pct', 0.55)
    margin_use = config.get('margin_use', 0.70)
    lev = config.get('lev', 5)
    cooldown = config.get('cooldown', 4)
    spot_cooldown = config.get('spot_cooldown', 12)

    short_sl = config.get('short_sl', -0.30)
    short_tp = config.get('short_tp', 0.80)
    short_trail = config.get('short_trail', 0.25)
    short_max_hold = config.get('short_max_hold', 72)
    long_sl = config.get('long_sl', -0.15)
    long_tp = config.get('long_tp', 0.50)
    long_trail = config.get('long_trail', 0.20)
    long_max_hold = config.get('long_max_hold', 72)

    # 追踪止盈回撤比例
    trail_pullback = config.get('trail_pullback', 0.60)

    # 动态止盈(根据波动率调整)
    use_dynamic_tp = config.get('use_dynamic_tp', False)
    # 分段止盈
    use_partial_tp = config.get('use_partial_tp', False)
    partial_tp_1 = config.get('partial_tp_1', 0.30)  # 第一段止盈点
    partial_tp_1_pct = config.get('partial_tp_1_pct', 0.40)  # 止盈比例

    # 波动率自适应止损
    use_atr_sl = config.get('use_atr_sl', False)
    atr_sl_mult = config.get('atr_sl_mult', 3.0)

    short_cd = 0; long_cd = 0; spot_cd = 0
    short_bars = 0; long_bars = 0
    short_max_pnl = 0; long_max_pnl = 0
    short_partial_done = False; long_partial_done = False

    warmup = max(60, int(len(df) * 0.05))

    # 根据时间框架计算funding间隔(8小时/K线周期)
    tf_hours = {'10m': 1/6, '15m': 0.25, '30m': 0.5, '1h': 1, '2h': 2, '3h': 3,
                '4h': 4, '6h': 6, '8h': 8, '12h': 12, '16h': 16, '24h': 24}
    bars_per_8h = max(1, int(8 / tf_hours.get(tf, 1)))

    record_interval = max(1, len(df) // 500)

    for idx in range(warmup, len(df)):
        dt = df.index[idx]
        price = df['close'].iloc[idx]

        if start_dt and dt < start_dt:
            if idx % record_interval == 0: eng.record_history(dt, price)
            continue

        eng.check_liquidation(price, dt)
        short_just_opened = False; long_just_opened = False

        # 资金费率
        eng.funding_counter += 1
        if eng.funding_counter % bars_per_8h == 0:
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

        ss, bs = calc_fusion_score_generic(signals, df, idx, dt, config)

        in_conflict = False
        if ss > 0 and bs > 0:
            ratio = min(ss, bs) / max(ss, bs)
            in_conflict = ratio >= 0.6 and min(ss, bs) >= 15

        # ATR自适应止损
        actual_short_sl = short_sl
        actual_long_sl = long_sl
        if use_atr_sl and idx >= 20:
            high = df['high'].iloc[max(0,idx-14):idx]
            low = df['low'].iloc[max(0,idx-14):idx]
            close_prev = df['close'].iloc[max(0,idx-15):idx-1]
            if len(high) > 0 and len(close_prev) > 0:
                min_len = min(len(high), len(low), len(close_prev))
                tr = pd.Series([max(h-l, abs(h-c), abs(l-c))
                                for h, l, c in zip(high[-min_len:], low[-min_len:], close_prev[-min_len:])])
                atr = tr.mean()
                atr_pct = atr / price
                actual_short_sl = max(short_sl, -(atr_pct * atr_sl_mult))
                actual_long_sl = max(long_sl, -(atr_pct * atr_sl_mult))

        # 动态止盈(根据波动率)
        actual_short_tp = short_tp
        actual_long_tp = long_tp
        if use_dynamic_tp and idx >= 20:
            recent_returns = df['close'].iloc[max(0,idx-20):idx].pct_change().dropna()
            if len(recent_returns) > 5:
                vol = recent_returns.std()
                if vol > 0.03:  # 高波动
                    actual_short_tp = short_tp * 1.3
                    actual_long_tp = long_tp * 1.3
                elif vol < 0.01:  # 低波动
                    actual_short_tp = short_tp * 0.7
                    actual_long_tp = long_tp * 0.7

        # 卖出现货
        if ss >= sell_threshold and spot_cd == 0 and not in_conflict and eng.spot_eth * price > 500:
            eng.spot_sell(price, dt, sell_pct, f"卖出 SS={ss:.0f}")
            spot_cd = spot_cooldown

        # 开空
        sell_dom = (ss > bs * 1.5) if bs > 0 else True
        if short_cd == 0 and ss >= short_threshold and not eng.futures_short and sell_dom:
            margin = eng.available_margin() * margin_use
            actual_lev = min(lev if ss >= 50 else min(lev, 3) if ss >= 35 else 2, eng.max_leverage)
            eng.open_short(price, dt, margin, actual_lev, f"开空 {actual_lev}x")
            short_max_pnl = 0; short_bars = 0; short_cd = cooldown
            short_just_opened = True; short_partial_done = False

        # 管理空仓
        if eng.futures_short and not short_just_opened:
            short_bars += 1
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin

            # 分段止盈
            if use_partial_tp and not short_partial_done and pnl_r >= partial_tp_1:
                # 平掉一部分
                old_qty = eng.futures_short.quantity
                partial_qty = old_qty * partial_tp_1_pct
                partial_pnl = (eng.futures_short.entry_price - price) * partial_qty
                eng.usdt += eng.futures_short.margin * partial_tp_1_pct + partial_pnl
                fee = partial_qty * price * FuturesEngine.TAKER_FEE
                eng.usdt -= fee; eng.total_futures_fees += fee
                eng.futures_short.quantity = old_qty - partial_qty
                eng.futures_short.margin *= (1 - partial_tp_1_pct)
                short_partial_done = True
                eng.trades.append({'time': str(dt), 'action': '分段止盈空',
                    'price': price, 'pnl': partial_pnl, 'reason': f'分段TP1 +{pnl_r*100:.0f}%'})

            if pnl_r >= actual_short_tp:
                eng.close_short(price, dt, f"止盈 +{pnl_r*100:.0f}%")
                short_max_pnl = 0; short_cd = cooldown * 2; short_bars = 0
            else:
                if pnl_r > short_max_pnl: short_max_pnl = pnl_r
                if short_max_pnl >= short_trail and eng.futures_short:
                    if pnl_r < short_max_pnl * trail_pullback:
                        eng.close_short(price, dt, "追踪止盈")
                        short_max_pnl = 0; short_cd = cooldown; short_bars = 0
                if eng.futures_short and bs >= config.get('close_short_bs', 40):
                    bs_dom = (ss < bs * 0.7) if bs > 0 else True
                    if bs_dom:
                        eng.close_short(price, dt, f"反向平空 BS={bs:.0f}")
                        short_max_pnl = 0; short_cd = cooldown * 3; short_bars = 0
                if eng.futures_short and pnl_r < actual_short_sl:
                    eng.close_short(price, dt, f"止损 {pnl_r*100:.0f}%")
                    short_max_pnl = 0; short_cd = cooldown * 4; short_bars = 0
                if eng.futures_short and short_bars >= short_max_hold:
                    eng.close_short(price, dt, "超时")
                    short_max_pnl = 0; short_cd = cooldown; short_bars = 0
        elif eng.futures_short and short_just_opened:
            short_bars = 1

        # 买入现货
        if bs >= buy_threshold and spot_cd == 0 and not in_conflict and eng.available_usdt() > 500:
            eng.spot_buy(price, dt, eng.available_usdt() * 0.25, f"买入 BS={bs:.0f}")
            spot_cd = spot_cooldown

        # 开多
        buy_dom = (bs > ss * 1.5) if ss > 0 else True
        if long_cd == 0 and bs >= long_threshold and not eng.futures_long and buy_dom:
            margin = eng.available_margin() * margin_use
            actual_lev = min(lev if bs >= 50 else min(lev, 3) if bs >= 35 else 2, eng.max_leverage)
            eng.open_long(price, dt, margin, actual_lev, f"开多 {actual_lev}x")
            long_max_pnl = 0; long_bars = 0; long_cd = cooldown
            long_just_opened = True; long_partial_done = False

        # 管理多仓
        if eng.futures_long and not long_just_opened:
            long_bars += 1
            pnl_r = eng.futures_long.calc_pnl(price) / eng.futures_long.margin

            # 分段止盈
            if use_partial_tp and not long_partial_done and pnl_r >= partial_tp_1:
                old_qty = eng.futures_long.quantity
                partial_qty = old_qty * partial_tp_1_pct
                partial_pnl = (price - eng.futures_long.entry_price) * partial_qty
                eng.usdt += eng.futures_long.margin * partial_tp_1_pct + partial_pnl
                fee = partial_qty * price * FuturesEngine.TAKER_FEE
                eng.usdt -= fee; eng.total_futures_fees += fee
                eng.futures_long.quantity = old_qty - partial_qty
                eng.futures_long.margin *= (1 - partial_tp_1_pct)
                long_partial_done = True
                eng.trades.append({'time': str(dt), 'action': '分段止盈多',
                    'price': price, 'pnl': partial_pnl, 'reason': f'分段TP1 +{pnl_r*100:.0f}%'})

            if pnl_r >= actual_long_tp:
                eng.close_long(price, dt, f"止盈 +{pnl_r*100:.0f}%")
                long_max_pnl = 0; long_cd = cooldown * 2; long_bars = 0
            else:
                if pnl_r > long_max_pnl: long_max_pnl = pnl_r
                if long_max_pnl >= long_trail and eng.futures_long:
                    if pnl_r < long_max_pnl * trail_pullback:
                        eng.close_long(price, dt, "追踪止盈")
                        long_max_pnl = 0; long_cd = cooldown; long_bars = 0
                if eng.futures_long and ss >= config.get('close_long_ss', 40):
                    ss_dom = (bs < ss * 0.7) if ss > 0 else True
                    if ss_dom:
                        eng.close_long(price, dt, f"反向平多 SS={ss:.0f}")
                        long_max_pnl = 0; long_cd = cooldown * 3; long_bars = 0
                if eng.futures_long and pnl_r < actual_long_sl:
                    eng.close_long(price, dt, f"止损 {pnl_r*100:.0f}%")
                    long_max_pnl = 0; long_cd = cooldown * 4; long_bars = 0
                if eng.futures_long and long_bars >= long_max_hold:
                    eng.close_long(price, dt, "超时")
                    long_max_pnl = 0; long_cd = cooldown; long_bars = 0
        elif eng.futures_long and long_just_opened:
            long_bars = 1

        if idx % record_interval == 0:
            eng.record_history(dt, price)

    # 期末平仓
    last_price = df['close'].iloc[-1]
    last_dt = df.index[-1]
    if eng.futures_short: eng.close_short(last_price, last_dt, "期末平仓")
    if eng.futures_long: eng.close_long(last_price, last_dt, "期末平仓")

    if start_dt:
        trade_df = df[df.index >= start_dt]
        if len(trade_df) > 1: return eng.get_result(trade_df)
    return eng.get_result(df)


# ======================================================
#   止盈止损参数空间
# ======================================================
def get_sl_tp_variants():
    """获取止盈止损参数变体"""
    variants = []

    # ---- 基准(当前F12) ----
    variants.append({
        'tag': 'F12基准',
        'short_sl': -0.30, 'short_tp': 0.80, 'short_trail': 0.25, 'short_max_hold': 72,
        'long_sl': -0.15, 'long_tp': 0.50, 'long_trail': 0.20, 'long_max_hold': 72,
        'trail_pullback': 0.60, 'cooldown': 4, 'spot_cooldown': 12,
    })

    # ---- Phase 1: 止损优化 ----
    for ssl in [-0.10, -0.15, -0.20, -0.25, -0.35]:
        for lsl in [-0.08, -0.12, -0.18, -0.20]:
            if ssl == -0.30 and lsl == -0.15: continue  # skip baseline
            variants.append({
                'tag': f'SL空{ssl*100:.0f}%多{lsl*100:.0f}%',
                'short_sl': ssl, 'short_tp': 0.80, 'short_trail': 0.25, 'short_max_hold': 72,
                'long_sl': lsl, 'long_tp': 0.50, 'long_trail': 0.20, 'long_max_hold': 72,
                'trail_pullback': 0.60, 'cooldown': 4, 'spot_cooldown': 12,
            })

    # ---- Phase 2: 止盈优化 ----
    for stp in [0.40, 0.60, 1.00, 1.20, 1.50]:
        for ltp in [0.30, 0.40, 0.60, 0.80]:
            if stp == 0.80 and ltp == 0.50: continue
            variants.append({
                'tag': f'TP空{stp*100:.0f}%多{ltp*100:.0f}%',
                'short_sl': -0.30, 'short_tp': stp, 'short_trail': 0.25, 'short_max_hold': 72,
                'long_sl': -0.15, 'long_tp': ltp, 'long_trail': 0.20, 'long_max_hold': 72,
                'trail_pullback': 0.60, 'cooldown': 4, 'spot_cooldown': 12,
            })

    # ---- Phase 3: 追踪止盈优化 ----
    for trail in [0.15, 0.20, 0.30, 0.40]:
        for pullback in [0.45, 0.50, 0.55, 0.65, 0.70]:
            if trail == 0.25 and pullback == 0.60: continue
            variants.append({
                'tag': f'Trail{trail*100:.0f}%回撤{pullback*100:.0f}%',
                'short_sl': -0.30, 'short_tp': 0.80, 'short_trail': trail, 'short_max_hold': 72,
                'long_sl': -0.15, 'long_tp': 0.50, 'long_trail': trail, 'long_max_hold': 72,
                'trail_pullback': pullback, 'cooldown': 4, 'spot_cooldown': 12,
            })

    # ---- Phase 4: 最大持仓时间 ----
    for hold in [24, 36, 48, 96, 120, 168]:
        variants.append({
            'tag': f'Hold{hold}bars',
            'short_sl': -0.30, 'short_tp': 0.80, 'short_trail': 0.25, 'short_max_hold': hold,
            'long_sl': -0.15, 'long_tp': 0.50, 'long_trail': 0.20, 'long_max_hold': hold,
            'trail_pullback': 0.60, 'cooldown': 4, 'spot_cooldown': 12,
        })

    # ---- Phase 5: 冷却期 ----
    for cd in [2, 3, 6, 8]:
        for scd in [6, 8, 16, 24]:
            if cd == 4 and scd == 12: continue
            variants.append({
                'tag': f'CD{cd}期货_CD{scd}现货',
                'short_sl': -0.30, 'short_tp': 0.80, 'short_trail': 0.25, 'short_max_hold': 72,
                'long_sl': -0.15, 'long_tp': 0.50, 'long_trail': 0.20, 'long_max_hold': 72,
                'trail_pullback': 0.60, 'cooldown': cd, 'spot_cooldown': scd,
            })

    # ---- Phase 6: 增强策略 ----
    # ATR自适应止损
    for atr_mult in [2.0, 2.5, 3.0, 4.0]:
        variants.append({
            'tag': f'ATR自适应SL({atr_mult}x)',
            'short_sl': -0.30, 'short_tp': 0.80, 'short_trail': 0.25, 'short_max_hold': 72,
            'long_sl': -0.15, 'long_tp': 0.50, 'long_trail': 0.20, 'long_max_hold': 72,
            'trail_pullback': 0.60, 'cooldown': 4, 'spot_cooldown': 12,
            'use_atr_sl': True, 'atr_sl_mult': atr_mult,
        })

    # 动态止盈(波动率调整)
    variants.append({
        'tag': '动态TP(波动率)',
        'short_sl': -0.30, 'short_tp': 0.80, 'short_trail': 0.25, 'short_max_hold': 72,
        'long_sl': -0.15, 'long_tp': 0.50, 'long_trail': 0.20, 'long_max_hold': 72,
        'trail_pullback': 0.60, 'cooldown': 4, 'spot_cooldown': 12,
        'use_dynamic_tp': True,
    })

    # 分段止盈
    for pt1 in [0.20, 0.30, 0.40]:
        for pt1_pct in [0.30, 0.50]:
            variants.append({
                'tag': f'分段TP@{pt1*100:.0f}%平{pt1_pct*100:.0f}%',
                'short_sl': -0.30, 'short_tp': 0.80, 'short_trail': 0.25, 'short_max_hold': 72,
                'long_sl': -0.15, 'long_tp': 0.50, 'long_trail': 0.20, 'long_max_hold': 72,
                'trail_pullback': 0.60, 'cooldown': 4, 'spot_cooldown': 12,
                'use_partial_tp': True, 'partial_tp_1': pt1, 'partial_tp_1_pct': pt1_pct,
            })

    return variants


# ======================================================
#   主函数
# ======================================================
def main():
    trade_days = 30
    print("=" * 120)
    print("  多时间框架止盈止损优化器")
    print("  基于F12(C6+三书否决)策略 · 12个时间周期 · 系统性参数搜索")
    print("=" * 120)

    # 获取所有时间框架数据
    print("\n[1/4] 获取多时间框架数据...")
    all_data = fetch_multi_tf_data(ALL_TIMEFRAMES, days=60)
    available_tfs = sorted(all_data.keys(), key=lambda x: ALL_TIMEFRAMES.index(x) if x in ALL_TIMEFRAMES else 99)
    print(f"\n可用时间框架: {', '.join(available_tfs)}")

    # 预计算各时间框架信号
    print("\n[2/4] 预计算各时间框架的五维信号...")
    all_signals = {}
    for tf in available_tfs:
        print(f"\n  计算 {tf} 信号:")
        all_signals[tf] = compute_signals_for_tf(all_data[tf], tf, all_data)
        print(f"    {tf} 信号计算完成")

    # F12基准配置
    f12_base = {
        'name': 'F12基准',
        'fusion_mode': 'c6_veto',
        'veto_threshold': 25,
        'single_pct': 0.20, 'total_pct': 0.50, 'lifetime_pct': 5.0,
        'sell_threshold': 18, 'buy_threshold': 25,
        'short_threshold': 25, 'long_threshold': 40,
        'close_short_bs': 40, 'close_long_ss': 40,
        'sell_pct': 0.55, 'margin_use': 0.70, 'lev': 5, 'max_lev': 5,
    }

    # 获取参数变体
    sl_tp_variants = get_sl_tp_variants()
    print(f"\n[3/4] 参数变体: {len(sl_tp_variants)}种")

    # ============ Phase A: 各时间框架上测试基准策略 ============
    print(f"\n{'=' * 120}")
    print(f"  Phase A: 各时间框架基准性能(F12)")
    print(f"{'=' * 120}")

    tf_baseline_results = {}
    print(f"\n{'时间框架':<10} {'K线数':>8} {'Alpha':>10} {'策略收益':>12} {'BH收益':>12} "
          f"{'回撤':>8} {'交易':>6} {'强平':>4} {'费用':>10}")
    print('-' * 100)

    for tf in available_tfs:
        config = {**f12_base, **sl_tp_variants[0]}  # F12基准
        config['name'] = f'F12_{tf}'

        # 根据时间框架调整max_hold (归一化到约3天)
        tf_hours = {'10m': 1/6, '15m': 0.25, '30m': 0.5, '1h': 1, '2h': 2, '3h': 3,
                    '4h': 4, '6h': 6, '8h': 8, '12h': 12, '16h': 16, '24h': 24}
        hours = tf_hours.get(tf, 1)
        config['short_max_hold'] = max(6, int(72 / hours))
        config['long_max_hold'] = max(6, int(72 / hours))
        # 冷却期同理调整
        config['cooldown'] = max(1, int(4 / hours))
        config['spot_cooldown'] = max(2, int(12 / hours))

        r = run_strategy(all_data[tf], all_signals[tf], config, tf=tf, trade_days=trade_days)
        fees = r.get('fees', {})
        tf_baseline_results[tf] = r
        r['config_max_hold'] = config['short_max_hold']
        r['config_cooldown'] = config['cooldown']

        print(f"  {tf:<8} {len(all_data[tf]):>8} {r['alpha']:>+9.2f}% {r['strategy_return']:>+11.2f}% "
              f"{r['buy_hold_return']:>+11.2f}% {r['max_drawdown']:>7.2f}% "
              f"{r['total_trades']:>5} {r['liquidations']:>3} "
              f"${fees.get('total_costs', 0):>9,.0f}")

    # 找到表现最好的3个时间框架
    tf_ranked = sorted(tf_baseline_results.items(), key=lambda x: x[1]['alpha'], reverse=True)
    top_tfs = [t[0] for t in tf_ranked[:3]]
    print(f"\n  TOP3时间框架: {', '.join(top_tfs)}")
    print(f"    #{1}: {tf_ranked[0][0]} α={tf_ranked[0][1]['alpha']:+.2f}%")
    print(f"    #{2}: {tf_ranked[1][0]} α={tf_ranked[1][1]['alpha']:+.2f}%")
    print(f"    #{3}: {tf_ranked[2][0]} α={tf_ranked[2][1]['alpha']:+.2f}%")

    # ============ Phase B: 在TOP3时间框架上优化止盈止损 ============
    print(f"\n{'=' * 120}")
    print(f"  Phase B: 在TOP时间框架上系统优化止盈止损")
    print(f"{'=' * 120}")

    all_opt_results = []

    for tf in top_tfs:
        print(f"\n  === {tf} 优化开始 ({len(sl_tp_variants)}种参数变体) ===")
        tf_hours_val = tf_hours.get(tf, 1)

        results_for_tf = []
        for i, var in enumerate(sl_tp_variants):
            config = {**f12_base, **var}
            config['name'] = f'{var["tag"]}_{tf}'

            # 根据时间框架归一化持仓时间
            raw_hold = var.get('short_max_hold', 72)
            config['short_max_hold'] = max(6, int(raw_hold / tf_hours_val))
            config['long_max_hold'] = max(6, int(var.get('long_max_hold', 72) / tf_hours_val))
            config['cooldown'] = max(1, int(var.get('cooldown', 4) / tf_hours_val))
            config['spot_cooldown'] = max(2, int(var.get('spot_cooldown', 12) / tf_hours_val))

            r = run_strategy(all_data[tf], all_signals[tf], config, tf=tf, trade_days=trade_days)
            results_for_tf.append({
                'tf': tf,
                'tag': var['tag'],
                'alpha': r['alpha'],
                'strategy_return': r['strategy_return'],
                'buy_hold_return': r['buy_hold_return'],
                'max_drawdown': r['max_drawdown'],
                'total_trades': r['total_trades'],
                'liquidations': r['liquidations'],
                'fees': r.get('fees', {}).get('total_costs', 0),
                'final_total': r.get('final_total', 0),
                'config': var,
            })

            if (i + 1) % 20 == 0:
                print(f"    进度: {i+1}/{len(sl_tp_variants)}")

        # 排序
        results_for_tf.sort(key=lambda x: x['alpha'], reverse=True)
        all_opt_results.extend(results_for_tf)

        print(f"\n  {tf} TOP10参数:")
        print(f"  {'排名':>4} {'参数标签':<30} {'Alpha':>10} {'收益':>10} {'回撤':>8} {'交易':>6}")
        print('  ' + '-' * 80)
        for i, r in enumerate(results_for_tf[:10]):
            star = ' ★' if i == 0 else ''
            print(f"  #{i+1:>3} {r['tag']:<30} {r['alpha']:>+9.2f}% "
                  f"{r['strategy_return']:>+9.2f}% {r['max_drawdown']:>7.2f}% {r['total_trades']:>5}{star}")

    # ============ Phase C: 综合最优组合 ============
    print(f"\n{'=' * 120}")
    print(f"  Phase C: 全局最优参数组合")
    print(f"{'=' * 120}")

    # 按alpha排序所有结果
    all_opt_results.sort(key=lambda x: x['alpha'], reverse=True)

    print(f"\n  全局TOP20:")
    print(f"  {'排名':>4} {'时间框架':>8} {'参数标签':<30} {'Alpha':>10} {'收益':>12} {'回撤':>8} {'交易':>6} {'费用':>10}")
    print('  ' + '-' * 110)
    for i, r in enumerate(all_opt_results[:20]):
        star = ' ★★★' if i == 0 else ' ★★' if i <= 2 else ' ★' if i <= 4 else ''
        print(f"  #{i+1:>3} {r['tf']:>8} {r['tag']:<30} {r['alpha']:>+9.2f}% "
              f"{r['strategy_return']:>+11.2f}% {r['max_drawdown']:>7.2f}% "
              f"{r['total_trades']:>5} ${r['fees']:>9,.0f}{star}")

    # ============ Phase D: 最优参数组合精细搜索 ============
    # 取全局TOP5的参数进行交叉组合
    if len(all_opt_results) >= 3:
        print(f"\n{'=' * 120}")
        print(f"  Phase D: 精细组合搜索(基于TOP参数)")
        print(f"{'=' * 120}")

        top1 = all_opt_results[0]
        top1_tf = top1['tf']
        top1_cfg = top1['config']

        # 获取各维度最佳参数
        best_sl = sorted([r for r in all_opt_results if r['tag'].startswith('SL')],
                         key=lambda x: x['alpha'], reverse=True)
        best_tp = sorted([r for r in all_opt_results if r['tag'].startswith('TP')],
                         key=lambda x: x['alpha'], reverse=True)
        best_trail = sorted([r for r in all_opt_results if r['tag'].startswith('Trail')],
                           key=lambda x: x['alpha'], reverse=True)
        best_hold = sorted([r for r in all_opt_results if r['tag'].startswith('Hold')],
                          key=lambda x: x['alpha'], reverse=True)

        # 构建精细组合
        fine_variants = []
        sl_opts = [best_sl[0]['config']] if best_sl else [sl_tp_variants[0]]
        tp_opts = [best_tp[0]['config']] if best_tp else [sl_tp_variants[0]]
        trail_opts = [best_trail[0]['config']] if best_trail else [sl_tp_variants[0]]
        hold_opts = [best_hold[0]['config']] if best_hold else [sl_tp_variants[0]]

        # 组合最佳SL + 最佳TP
        for sl_cfg in sl_opts[:2]:
            for tp_cfg in tp_opts[:2]:
                for trail_cfg in trail_opts[:2]:
                    combined = {**sl_tp_variants[0]}  # 基准
                    combined['short_sl'] = sl_cfg.get('short_sl', -0.30)
                    combined['long_sl'] = sl_cfg.get('long_sl', -0.15)
                    combined['short_tp'] = tp_cfg.get('short_tp', 0.80)
                    combined['long_tp'] = tp_cfg.get('long_tp', 0.50)
                    combined['short_trail'] = trail_cfg.get('short_trail', 0.25)
                    combined['long_trail'] = trail_cfg.get('long_trail', 0.20)
                    combined['trail_pullback'] = trail_cfg.get('trail_pullback', 0.60)
                    combined['tag'] = (f"组合SL{combined['short_sl']*100:.0f}_"
                                      f"TP{combined['short_tp']*100:.0f}_"
                                      f"TR{combined['short_trail']*100:.0f}")
                    fine_variants.append(combined)

        # 加入增强策略的最优组合
        best_enhanced = sorted([r for r in all_opt_results
                               if 'ATR' in r['tag'] or '动态' in r['tag'] or '分段' in r['tag']],
                              key=lambda x: x['alpha'], reverse=True)

        if best_enhanced:
            for enh in best_enhanced[:3]:
                combined = {**enh['config']}
                # 覆盖最佳SL/TP
                if best_sl:
                    combined['short_sl'] = best_sl[0]['config'].get('short_sl', combined.get('short_sl', -0.30))
                    combined['long_sl'] = best_sl[0]['config'].get('long_sl', combined.get('long_sl', -0.15))
                if best_tp:
                    combined['short_tp'] = best_tp[0]['config'].get('short_tp', combined.get('short_tp', 0.80))
                    combined['long_tp'] = best_tp[0]['config'].get('long_tp', combined.get('long_tp', 0.50))
                combined['tag'] = f"精选{enh['tag'][:15]}+最佳SL/TP"
                fine_variants.append(combined)

        # 在最优时间框架上测试精细组合
        fine_results = []
        tf_for_fine = top1_tf
        print(f"\n  在 {tf_for_fine} 上测试 {len(fine_variants)} 种精细组合...")

        for var in fine_variants:
            config = {**f12_base, **var}
            config['name'] = var['tag']
            tf_h = tf_hours.get(tf_for_fine, 1)
            config['short_max_hold'] = max(6, int(var.get('short_max_hold', 72) / tf_h))
            config['long_max_hold'] = max(6, int(var.get('long_max_hold', 72) / tf_h))
            config['cooldown'] = max(1, int(var.get('cooldown', 4) / tf_h))
            config['spot_cooldown'] = max(2, int(var.get('spot_cooldown', 12) / tf_h))

            r = run_strategy(all_data[tf_for_fine], all_signals[tf_for_fine],
                           config, tf=tf_for_fine, trade_days=trade_days)
            fine_results.append({
                'tf': tf_for_fine,
                'tag': var['tag'],
                'alpha': r['alpha'],
                'strategy_return': r['strategy_return'],
                'buy_hold_return': r['buy_hold_return'],
                'max_drawdown': r['max_drawdown'],
                'total_trades': r['total_trades'],
                'liquidations': r['liquidations'],
                'fees': r.get('fees', {}).get('total_costs', 0),
                'final_total': r.get('final_total', 0),
                'config': var,
                'full_result': r,
            })

        fine_results.sort(key=lambda x: x['alpha'], reverse=True)

        print(f"\n  精细组合TOP10:")
        print(f"  {'排名':>4} {'参数标签':<35} {'Alpha':>10} {'收益':>12} {'回撤':>8} {'交易':>6}")
        print('  ' + '-' * 90)
        for i, r in enumerate(fine_results[:10]):
            star = ' ★★★' if i == 0 else ''
            print(f"  #{i+1:>3} {r['tag']:<35} {r['alpha']:>+9.2f}% "
                  f"{r['strategy_return']:>+11.2f}% {r['max_drawdown']:>7.2f}% "
                  f"{r['total_trades']:>5}{star}")

        all_opt_results.extend(fine_results)
        all_opt_results.sort(key=lambda x: x['alpha'], reverse=True)

    # ============ 保存结果 ============
    print(f"\n[4/4] 保存结果...")

    # 全局最优
    global_best = all_opt_results[0] if all_opt_results else None

    output = {
        'description': f'多时间框架止盈止损优化 · 最近{trade_days}天',
        'run_time': datetime.now().isoformat(),
        'available_timeframes': available_tfs,
        'total_variants_tested': len(all_opt_results),
        'trade_days': trade_days,

        'baseline_by_tf': [{
            'tf': tf,
            'alpha': tf_baseline_results[tf]['alpha'],
            'strategy_return': tf_baseline_results[tf]['strategy_return'],
            'buy_hold_return': tf_baseline_results[tf]['buy_hold_return'],
            'max_drawdown': tf_baseline_results[tf]['max_drawdown'],
            'total_trades': tf_baseline_results[tf]['total_trades'],
        } for tf in available_tfs if tf in tf_baseline_results],

        'top_timeframes': top_tfs,

        'global_top20': [{
            'rank': i + 1,
            'tf': r['tf'],
            'tag': r['tag'],
            'alpha': r['alpha'],
            'strategy_return': r['strategy_return'],
            'max_drawdown': r['max_drawdown'],
            'total_trades': r['total_trades'],
            'fees': r['fees'],
            'config': r['config'],
        } for i, r in enumerate(all_opt_results[:20])],

        'global_best': {
            'tf': global_best['tf'],
            'tag': global_best['tag'],
            'alpha': global_best['alpha'],
            'strategy_return': global_best['strategy_return'],
            'max_drawdown': global_best['max_drawdown'],
            'total_trades': global_best['total_trades'],
            'config': global_best['config'],
            'full_result': global_best.get('full_result', {}),
        } if global_best else None,

        'optimization_summary': {
            'best_sl': best_sl[0] if best_sl else None,
            'best_tp': best_tp[0] if best_tp else None,
            'best_trail': best_trail[0] if best_trail else None,
            'best_hold': best_hold[0] if best_hold else None,
        },
    }

    # 清理不可序列化的对象
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()
                    if k != 'full_result'}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return str(obj)
        return obj

    output_clean = clean_for_json(output)

    # 保存全局最优的完整结果
    if global_best and 'full_result' in global_best:
        output_clean['global_best_trades'] = global_best['full_result'].get('trades', [])
        output_clean['global_best_history'] = global_best['full_result'].get('history', [])
        output_clean['global_best_fees'] = global_best['full_result'].get('fees', {})

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'optimize_sl_tp_result.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output_clean, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果已保存: {path}")

    # 最终总结
    print(f"\n{'=' * 120}")
    print(f"  优化完成总结")
    print(f"{'=' * 120}")
    print(f"\n  测试时间框架: {len(available_tfs)}个 ({', '.join(available_tfs)})")
    print(f"  参数变体总数: {len(all_opt_results)}")
    if global_best:
        print(f"\n  🏆 全局最优策略:")
        print(f"     时间框架: {global_best['tf']}")
        print(f"     参数标签: {global_best['tag']}")
        print(f"     Alpha:    {global_best['alpha']:+.2f}%")
        print(f"     策略收益: {global_best['strategy_return']:+.2f}%")
        print(f"     最大回撤: {global_best['max_drawdown']:.2f}%")
        print(f"     交易次数: {global_best['total_trades']}")
        cfg = global_best['config']
        print(f"\n  最优参数:")
        print(f"     空头止损: {cfg.get('short_sl', -0.30)*100:.0f}%")
        print(f"     空头止盈: {cfg.get('short_tp', 0.80)*100:.0f}%")
        print(f"     多头止损: {cfg.get('long_sl', -0.15)*100:.0f}%")
        print(f"     多头止盈: {cfg.get('long_tp', 0.50)*100:.0f}%")
        print(f"     追踪止盈: {cfg.get('short_trail', 0.25)*100:.0f}%")
        print(f"     回撤比例: {cfg.get('trail_pullback', 0.60)*100:.0f}%")
        print(f"     最大持仓: {cfg.get('short_max_hold', 72)} bars")
        if cfg.get('use_atr_sl'): print(f"     ATR止损:  {cfg.get('atr_sl_mult', 3.0)}x")
        if cfg.get('use_dynamic_tp'): print(f"     动态止盈: 开启")
        if cfg.get('use_partial_tp'): print(f"     分段止盈: @{cfg.get('partial_tp_1',0.3)*100:.0f}% 平{cfg.get('partial_tp_1_pct',0.5)*100:.0f}%")

    print(f"\n  F12基准对比(1h):")
    if '1h' in tf_baseline_results:
        bl = tf_baseline_results['1h']
        print(f"     Alpha: {bl['alpha']:+.2f}%")
        diff = global_best['alpha'] - bl['alpha'] if global_best else 0
        print(f"     优化提升: {diff:+.2f}%")

    return output


if __name__ == '__main__':
    main()
