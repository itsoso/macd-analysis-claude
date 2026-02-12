"""
六书融合多时间框架止盈止损优化器

在五书优化器(optimize_sl_tp.py)基础上升级:
1. 新增KDJ作为第6维信号(基于《随机指标KDJ：波段操作精解》)
2. 支持多种融合模式: c6_veto_4 / kdj_weighted / kdj_timing
3. KDJ特有参数优化: KDJ权重、否决阈值、确认强度
4. 目标: 超越五书最优 α=+86.69% (精选分段TP@20%平30%+最佳SL/TP)

在12个时间周期上全面测试:
10min, 15min, 30min, 1h, 2h, 3h, 4h, 6h, 8h, 12h, 16h, 24h

真实回测条件:
- 币安 ETH/USDT 真实K线数据
- Taker手续费 0.05%, Maker 0.02%
- 资金费率每8小时 ±0.01%
- 逐仓模式, 5%维持保证金率强平
- 滑点模拟
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
from kdj_strategy import compute_kdj_scores


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


def compute_signals_six(df, tf, data_all, max_bars=0):
    """为指定时间框架计算六维信号(含KDJ)

    参数:
        df: 主周期 DataFrame (已含指标)
        tf: 时间框架字符串
        data_all: 多周期数据 dict
        max_bars: >0 时只保留尾部 max_bars 根K线用于计算，大幅降低耗时。
                  0 = 不限制 (全量计算，用于回测)。
    """
    # ---- 尾部截断优化 ----
    if max_bars > 0 and len(df) > max_bars:
        df = df.iloc[-max_bars:].copy()

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

    # 6. KDJ ★新增★
    kdj_sell, kdj_buy, kdj_names = compute_kdj_scores(df)
    signals['kdj_sell'] = kdj_sell
    signals['kdj_buy'] = kdj_buy

    return signals


# ======================================================
#   六维融合评分(支持多模式)
# ======================================================
def calc_fusion_score_six(signals, df, idx, dt, config):
    """六维融合评分 — 支持c6_veto_4/kdj_weighted/kdj_timing三种模式"""
    price = df['close'].iloc[idx]
    mode = config.get('fusion_mode', 'c6_veto_4')

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
        if c5 < c20 * 0.99: trend['is_downtrend'] = True; trend['ma_bearish'] = True
        elif c5 > c20 * 1.01: trend['is_uptrend'] = True; trend['ma_bullish'] = True

    div_sell, _ = _calc_top_score(merged, trend)
    div_buy = _calc_bottom_score(merged, trend)

    # 2. 均线
    ma_sell = float(signals['ma']['sell_score'].iloc[idx]) if idx < len(signals['ma']['sell_score']) else 0
    ma_buy = float(signals['ma']['buy_score'].iloc[idx]) if idx < len(signals['ma']['buy_score']) else 0

    # 3-6. K线、布林、量价、KDJ
    cs_sell = float(signals['cs_sell'].iloc[idx]) if idx < len(signals['cs_sell']) else 0
    cs_buy = float(signals['cs_buy'].iloc[idx]) if idx < len(signals['cs_buy']) else 0
    bb_sell = float(signals['bb_sell'].iloc[idx]) if idx < len(signals['bb_sell']) else 0
    bb_buy = float(signals['bb_buy'].iloc[idx]) if idx < len(signals['bb_buy']) else 0
    vp_sell = float(signals['vp_sell'].iloc[idx]) if idx < len(signals['vp_sell']) else 0
    vp_buy = float(signals['vp_buy'].iloc[idx]) if idx < len(signals['vp_buy']) else 0
    kdj_sell = float(signals['kdj_sell'].iloc[idx]) if idx < len(signals['kdj_sell']) else 0
    kdj_buy = float(signals['kdj_buy'].iloc[idx]) if idx < len(signals['kdj_buy']) else 0

    # MA排列加成
    ma_arr_bonus_sell = 1.0
    ma_arr_bonus_buy = 1.0
    if idx < len(signals['ma']['sell_score']):
        ma_data = signals['ma']
        if 'arrangement' in ma_data:
            arr_series = ma_data.get('arrangement', None)
            if arr_series is not None and hasattr(arr_series, 'iloc') and idx < len(arr_series):
                try:
                    arr_val = float(arr_series.iloc[idx])
                    if arr_val < 0: ma_arr_bonus_sell = 1.10
                    elif arr_val > 0: ma_arr_bonus_buy = 1.10
                except (ValueError, TypeError):
                    pass

    if mode == 'c6_veto_4':
        # === C6底座 + 四书否决权 ===
        base_sell = (div_sell * 0.7 + ma_sell * 0.3) * ma_arr_bonus_sell
        base_buy = (div_buy * 0.7 + ma_buy * 0.3) * ma_arr_bonus_buy

        veto_threshold = config.get('veto_threshold', 25)
        kdj_bonus = config.get('kdj_bonus', 0.09)
        bb_bonus = config.get('bb_bonus', 0.10)
        vp_bonus = config.get('vp_bonus', 0.08)
        cs_bonus = config.get('cs_bonus', 0.06)

        # 否决逻辑: 4个辅助系统, 至少2个强烈反对 → 削弱
        sell_vetoes = sum(1 for s in [bb_buy, vp_buy, cs_buy, kdj_buy] if s >= veto_threshold)
        buy_vetoes = sum(1 for s in [bb_sell, vp_sell, cs_sell, kdj_sell] if s >= veto_threshold)

        veto_dampen = config.get('veto_dampen', 0.30)

        if sell_vetoes >= 2:
            sell_score = base_sell * veto_dampen
        else:
            sb = 0
            if bb_sell >= 15: sb += bb_bonus
            if vp_sell >= 15: sb += vp_bonus
            if cs_sell >= 25: sb += cs_bonus
            if kdj_sell >= 15: sb += kdj_bonus
            sell_score = base_sell * (1 + sb)

        if buy_vetoes >= 2:
            buy_score = base_buy * veto_dampen
        else:
            bb_ = 0
            if bb_buy >= 15: bb_ += bb_bonus
            if vp_buy >= 15: bb_ += vp_bonus
            if cs_buy >= 25: bb_ += cs_bonus
            if kdj_buy >= 15: bb_ += kdj_bonus
            buy_score = base_buy * (1 + bb_)

    elif mode == 'c6_veto':
        # 五书兼容模式(无KDJ) — 用作对照
        base_sell = (div_sell * 0.7 + ma_sell * 0.3) * ma_arr_bonus_sell
        base_buy = (div_buy * 0.7 + ma_buy * 0.3) * ma_arr_bonus_buy
        veto_threshold = config.get('veto_threshold', 25)

        sell_vetoes = sum(1 for s in [bb_buy, vp_buy, cs_buy] if s >= veto_threshold)
        buy_vetoes = sum(1 for s in [bb_sell, vp_sell, cs_sell] if s >= veto_threshold)

        if sell_vetoes >= 2:
            sell_score = base_sell * 0.3
        else:
            sb = 0
            if bb_sell >= 15: sb += 0.10
            if vp_sell >= 15: sb += 0.08
            if cs_sell >= 25: sb += 0.06
            sell_score = base_sell * (1 + sb)

        if buy_vetoes >= 2:
            buy_score = base_buy * 0.3
        else:
            bb_ = 0
            if bb_buy >= 15: bb_ += 0.10
            if vp_buy >= 15: bb_ += 0.08
            if cs_buy >= 25: bb_ += 0.06
            buy_score = base_buy * (1 + bb_)

    elif mode == 'kdj_weighted':
        # KDJ加权模式: KDJ参与基础分计算
        kdj_w = config.get('kdj_weight', 0.20)
        div_w = config.get('div_weight', 0.55)
        ma_w = 1.0 - div_w - kdj_w  # 剩余给均线
        base_sell = (div_sell * div_w + ma_sell * ma_w + kdj_sell * kdj_w) * ma_arr_bonus_sell
        base_buy = (div_buy * div_w + ma_buy * ma_w + kdj_buy * kdj_w) * ma_arr_bonus_buy

        sb = 0
        if bb_sell >= 15: sb += 0.08
        if vp_sell >= 15: sb += 0.06
        if cs_sell >= 25: sb += 0.05
        sell_score = base_sell * (1 + sb)

        bb_ = 0
        if bb_buy >= 15: bb_ += 0.08
        if vp_buy >= 15: bb_ += 0.06
        if cs_buy >= 25: bb_ += 0.05
        buy_score = base_buy * (1 + bb_)

    elif mode == 'kdj_timing':
        # KDJ择时模式: C6基础上KDJ做乘法确认
        base_sell = (div_sell * 0.7 + ma_sell * 0.3) * ma_arr_bonus_sell
        base_buy = (div_buy * 0.7 + ma_buy * 0.3) * ma_arr_bonus_buy

        kdj_strong = config.get('kdj_strong_mult', 1.25)
        kdj_normal = config.get('kdj_normal_mult', 1.12)
        kdj_reverse = config.get('kdj_reverse_mult', 0.70)

        if kdj_sell >= 30:
            sell_score = base_sell * kdj_strong
        elif kdj_sell >= 15:
            sell_score = base_sell * kdj_normal
        elif kdj_buy >= 25:
            sell_score = base_sell * kdj_reverse
        else:
            sell_score = base_sell

        if kdj_buy >= 30:
            buy_score = base_buy * kdj_strong
        elif kdj_buy >= 15:
            buy_score = base_buy * kdj_normal
        elif kdj_sell >= 25:
            buy_score = base_buy * kdj_reverse
        else:
            buy_score = base_buy

        if bb_sell >= 15: sell_score *= 1.08
        if vp_sell >= 15: sell_score *= 1.06
        if bb_buy >= 15: buy_score *= 1.08
        if vp_buy >= 15: buy_score *= 1.06

    elif mode == 'kdj_gate':
        # KDJ门控模式: KDJ不满足条件直接不开仓
        base_sell = (div_sell * 0.7 + ma_sell * 0.3) * ma_arr_bonus_sell
        base_buy = (div_buy * 0.7 + ma_buy * 0.3) * ma_arr_bonus_buy

        kdj_gate_threshold = config.get('kdj_gate_threshold', 10)

        # KDJ必须同向才能开仓
        if kdj_sell < kdj_gate_threshold and base_sell > 0:
            sell_score = base_sell * 0.4  # KDJ不配合, 大幅削弱
        else:
            sb = 0
            if bb_sell >= 15: sb += 0.10
            if vp_sell >= 15: sb += 0.08
            if kdj_sell >= 20: sb += 0.12
            sell_score = base_sell * (1 + sb)

        if kdj_buy < kdj_gate_threshold and base_buy > 0:
            buy_score = base_buy * 0.4
        else:
            bb_ = 0
            if bb_buy >= 15: bb_ += 0.10
            if vp_buy >= 15: bb_ += 0.08
            if kdj_buy >= 20: bb_ += 0.12
            buy_score = base_buy * (1 + bb_)

    else:
        # fallback
        sell_score = div_sell * 0.5 + ma_sell * 0.2 + kdj_sell * 0.15 + bb_sell * 0.08 + vp_sell * 0.05 + cs_sell * 0.02
        buy_score = div_buy * 0.5 + ma_buy * 0.2 + kdj_buy * 0.15 + bb_buy * 0.08 + vp_buy * 0.05 + cs_buy * 0.02

    return sell_score, buy_score


# ======================================================
#   通用回测引擎(六书版)
# ======================================================
def run_strategy(df, signals, config, tf='1h', trade_days=30):
    """在指定时间框架上运行六书策略回测"""
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
    trail_pullback = config.get('trail_pullback', 0.60)

    use_dynamic_tp = config.get('use_dynamic_tp', False)
    use_partial_tp = config.get('use_partial_tp', False)
    partial_tp_1 = config.get('partial_tp_1', 0.30)
    partial_tp_1_pct = config.get('partial_tp_1_pct', 0.40)

    use_atr_sl = config.get('use_atr_sl', False)
    atr_sl_mult = config.get('atr_sl_mult', 3.0)

    # 二段止盈
    use_partial_tp_2 = config.get('use_partial_tp_2', False)
    partial_tp_2 = config.get('partial_tp_2', 0.50)
    partial_tp_2_pct = config.get('partial_tp_2_pct', 0.30)

    short_cd = 0; long_cd = 0; spot_cd = 0
    short_bars = 0; long_bars = 0
    short_max_pnl = 0; long_max_pnl = 0
    short_partial_done = False; long_partial_done = False
    short_partial2_done = False; long_partial2_done = False

    warmup = max(60, int(len(df) * 0.05))

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

        # 记录强平前的仓位状态
        had_short_before_liq = eng.futures_short is not None
        had_long_before_liq = eng.futures_long is not None
        eng.check_liquidation(price, dt)
        # 强平后设置冷却期, 防止同bar立即重新开仓
        if had_short_before_liq and eng.futures_short is None:
            short_cd = max(short_cd, cooldown * 6)
            short_bars = 0; short_max_pnl = 0
        if had_long_before_liq and eng.futures_long is None:
            long_cd = max(long_cd, cooldown * 6)
            long_bars = 0; long_max_pnl = 0
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

        ss, bs = calc_fusion_score_six(signals, df, idx, dt, config)

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

        # 动态止盈
        actual_short_tp = short_tp
        actual_long_tp = long_tp
        if use_dynamic_tp and idx >= 20:
            recent_returns = df['close'].iloc[max(0,idx-20):idx].pct_change().dropna()
            if len(recent_returns) > 5:
                vol = recent_returns.std()
                if vol > 0.03:
                    actual_short_tp = short_tp * 1.3
                    actual_long_tp = long_tp * 1.3
                elif vol < 0.01:
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
            short_just_opened = True; short_partial_done = False; short_partial2_done = False

        # 管理空仓
        if eng.futures_short and not short_just_opened:
            short_bars += 1
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin

            # 一段止盈 (含滑点 + 修复frozen_margin泄漏)
            if use_partial_tp and not short_partial_done and pnl_r >= partial_tp_1:
                old_qty = eng.futures_short.quantity
                partial_qty = old_qty * partial_tp_1_pct
                actual_close_p = price * (1 + FuturesEngine.SLIPPAGE)  # 空头平仓买入, 价格偏高
                partial_pnl = (eng.futures_short.entry_price - actual_close_p) * partial_qty
                margin_released = eng.futures_short.margin * partial_tp_1_pct
                eng.usdt += margin_released + partial_pnl
                eng.frozen_margin -= margin_released  # 修复: 释放冻结保证金
                fee = partial_qty * actual_close_p * FuturesEngine.TAKER_FEE
                slippage_cost = partial_qty * price * FuturesEngine.SLIPPAGE
                eng.usdt -= fee; eng.total_futures_fees += fee
                eng.total_slippage_cost += slippage_cost
                eng.futures_short.quantity = old_qty - partial_qty
                eng.futures_short.margin *= (1 - partial_tp_1_pct)
                short_partial_done = True
                eng.trades.append({'time': str(dt), 'action': '分段止盈空',
                    'price': price, 'pnl': partial_pnl, 'reason': f'分段TP1 +{pnl_r*100:.0f}%'})

            # 二段止盈 (含滑点 + 修复frozen_margin泄漏, 使用elif避免同bar双触发)
            elif use_partial_tp_2 and short_partial_done and not short_partial2_done and pnl_r >= partial_tp_2:
                old_qty = eng.futures_short.quantity
                partial_qty = old_qty * partial_tp_2_pct
                actual_close_p = price * (1 + FuturesEngine.SLIPPAGE)
                partial_pnl = (eng.futures_short.entry_price - actual_close_p) * partial_qty
                margin_released = eng.futures_short.margin * partial_tp_2_pct
                eng.usdt += margin_released + partial_pnl
                eng.frozen_margin -= margin_released
                fee = partial_qty * actual_close_p * FuturesEngine.TAKER_FEE
                slippage_cost = partial_qty * price * FuturesEngine.SLIPPAGE
                eng.usdt -= fee; eng.total_futures_fees += fee
                eng.total_slippage_cost += slippage_cost
                eng.futures_short.quantity = old_qty - partial_qty
                eng.futures_short.margin *= (1 - partial_tp_2_pct)
                short_partial2_done = True
                eng.trades.append({'time': str(dt), 'action': '分段止盈空2',
                    'price': price, 'pnl': partial_pnl, 'reason': f'分段TP2 +{pnl_r*100:.0f}%'})

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
            long_just_opened = True; long_partial_done = False; long_partial2_done = False

        # 管理多仓
        if eng.futures_long and not long_just_opened:
            long_bars += 1
            pnl_r = eng.futures_long.calc_pnl(price) / eng.futures_long.margin

            # 一段止盈 (含滑点 + 修复frozen_margin泄漏)
            if use_partial_tp and not long_partial_done and pnl_r >= partial_tp_1:
                old_qty = eng.futures_long.quantity
                partial_qty = old_qty * partial_tp_1_pct
                actual_close_p = price * (1 - FuturesEngine.SLIPPAGE)  # 多头平仓卖出, 价格偏低
                partial_pnl = (actual_close_p - eng.futures_long.entry_price) * partial_qty
                margin_released = eng.futures_long.margin * partial_tp_1_pct
                eng.usdt += margin_released + partial_pnl
                eng.frozen_margin -= margin_released  # 修复: 释放冻结保证金
                fee = partial_qty * actual_close_p * FuturesEngine.TAKER_FEE
                slippage_cost = partial_qty * price * FuturesEngine.SLIPPAGE
                eng.usdt -= fee; eng.total_futures_fees += fee
                eng.total_slippage_cost += slippage_cost
                eng.futures_long.quantity = old_qty - partial_qty
                eng.futures_long.margin *= (1 - partial_tp_1_pct)
                long_partial_done = True
                eng.trades.append({'time': str(dt), 'action': '分段止盈多',
                    'price': price, 'pnl': partial_pnl, 'reason': f'分段TP1 +{pnl_r*100:.0f}%'})

            # 二段止盈 (含滑点 + 修复frozen_margin泄漏, 使用elif避免同bar双触发)
            elif use_partial_tp_2 and long_partial_done and not long_partial2_done and pnl_r >= partial_tp_2:
                old_qty = eng.futures_long.quantity
                partial_qty = old_qty * partial_tp_2_pct
                actual_close_p = price * (1 - FuturesEngine.SLIPPAGE)
                partial_pnl = (actual_close_p - eng.futures_long.entry_price) * partial_qty
                margin_released = eng.futures_long.margin * partial_tp_2_pct
                eng.usdt += margin_released + partial_pnl
                eng.frozen_margin -= margin_released
                fee = partial_qty * actual_close_p * FuturesEngine.TAKER_FEE
                slippage_cost = partial_qty * price * FuturesEngine.SLIPPAGE
                eng.usdt -= fee; eng.total_futures_fees += fee
                eng.total_slippage_cost += slippage_cost
                eng.futures_long.quantity = old_qty - partial_qty
                eng.futures_long.margin *= (1 - partial_tp_2_pct)
                long_partial2_done = True
                eng.trades.append({'time': str(dt), 'action': '分段止盈多2',
                    'price': price, 'pnl': partial_pnl, 'reason': f'分段TP2 +{pnl_r*100:.0f}%'})

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
                    ss_dom = (bs < ss * 0.7) if bs > 0 else True
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
#   多周期联合决策回测引擎
# ======================================================

# 各时间框架权重(与 live_runner 保持一致)
_MTF_WEIGHT = {
    '10m': 2, '15m': 3, '30m': 5,
    '1h': 8, '2h': 10, '3h': 12,
    '4h': 15, '6h': 18, '8h': 20,
    '12h': 22, '16h': 25, '24h': 28,
}
_MTF_MINUTES = {
    '10m':10, '15m':15, '30m':30,
    '1h':60, '2h':120, '3h':180, '4h':240, '6h':360,
    '8h':480, '12h':720, '16h':960, '24h':1440,
}


def _build_tf_score_index(all_data, all_signals, tfs, config):
    """
    预计算每个 TF 在每个时间戳的 (sell_score, buy_score)。
    返回 {tf: {timestamp: (ss, bs)}} 的字典, 供回测快速查询。
    """
    tf_score_map = {}
    for tf in tfs:
        df = all_data[tf]
        sigs = all_signals[tf]
        score_dict = {}
        warmup = max(60, int(len(df) * 0.05))
        for idx in range(warmup, len(df)):
            dt = df.index[idx]
            ss, bs = calc_fusion_score_six(sigs, df, idx, dt, config)
            score_dict[dt] = (float(ss), float(bs))
        tf_score_map[tf] = score_dict
    return tf_score_map


def _get_tf_score_at(tf_score_map, tf, dt):
    """在 tf 的评分索引中查找离 dt 最近且 <= dt 的评分"""
    scores = tf_score_map.get(tf)
    if not scores:
        return 0.0, 0.0

    # 精确匹配
    if dt in scores:
        return scores[dt]

    # 查找最近的 <= dt 的时间戳
    candidates = [t for t in scores if t <= dt]
    if not candidates:
        return 0.0, 0.0
    nearest = max(candidates)

    # 如果最近评分太老 (超过该TF的2个周期), 视为无效
    tf_mins = _MTF_MINUTES.get(tf, 60)
    if (dt - nearest).total_seconds() > tf_mins * 60 * 2:
        return 0.0, 0.0

    return scores[nearest]


def calc_multi_tf_consensus(tf_score_map, decision_tfs, dt, config):
    """
    在指定时刻 dt 计算多周期加权共识评分

    返回: (consensus_ss, consensus_bs, meta_dict)
      - consensus_ss: 加权卖出分
      - consensus_bs: 加权买入分
      - meta_dict: 包含共振链、大周期信号等调节信息

    调节规则:
      1. 加权平均各TF的 ss/bs (大周期权重高)
      2. 大小周期反向时: 信号衰减
      3. 共振链(≥3级连续同向): 信号增强
    """
    tf_scores = {}
    for tf in decision_tfs:
        ss, bs = _get_tf_score_at(tf_score_map, tf, dt)
        tf_scores[tf] = (ss, bs)

    total_weight = sum(_MTF_WEIGHT.get(tf, 5) for tf in decision_tfs)
    if total_weight == 0:
        return 0.0, 0.0, {}

    # --- 1. 加权平均 ---
    weighted_ss = sum(tf_scores[tf][0] * _MTF_WEIGHT.get(tf, 5) for tf in decision_tfs) / total_weight
    weighted_bs = sum(tf_scores[tf][1] * _MTF_WEIGHT.get(tf, 5) for tf in decision_tfs) / total_weight

    # --- 2. 大周期定调 (≥4h) ---
    large_tfs = [tf for tf in decision_tfs if _MTF_MINUTES.get(tf, 0) >= 240]
    small_tfs = [tf for tf in decision_tfs if _MTF_MINUTES.get(tf, 0) < 240]

    large_ss_avg = 0; large_bs_avg = 0
    if large_tfs:
        large_w = sum(_MTF_WEIGHT.get(tf, 5) for tf in large_tfs)
        large_ss_avg = sum(tf_scores[tf][0] * _MTF_WEIGHT.get(tf, 5) for tf in large_tfs) / large_w
        large_bs_avg = sum(tf_scores[tf][1] * _MTF_WEIGHT.get(tf, 5) for tf in large_tfs) / large_w

    small_ss_avg = 0; small_bs_avg = 0
    if small_tfs:
        small_w = sum(_MTF_WEIGHT.get(tf, 5) for tf in small_tfs)
        small_ss_avg = sum(tf_scores[tf][0] * _MTF_WEIGHT.get(tf, 5) for tf in small_tfs) / small_w
        small_bs_avg = sum(tf_scores[tf][1] * _MTF_WEIGHT.get(tf, 5) for tf in small_tfs) / small_w

    # --- 3. 各TF方向判定 (用于共振链检测) ---
    short_threshold = config.get('short_threshold', 25)
    long_threshold = config.get('long_threshold', 40)
    directions = []
    for tf in sorted(decision_tfs, key=lambda t: _MTF_MINUTES.get(t, 0)):
        ss, bs = tf_scores[tf]
        if ss >= short_threshold and ss > bs * 1.3:
            directions.append((tf, 'short'))
        elif bs >= long_threshold and bs > ss * 1.3:
            directions.append((tf, 'long'))
        else:
            directions.append((tf, 'hold'))

    # --- 4. 共振链检测 ---
    best_chain_len = 0
    best_chain_dir = 'hold'
    best_chain_has_4h = False

    for target in ['long', 'short']:
        chain = []
        gap = 0
        for tf, d in directions:
            if d == target:
                chain.append(tf)
                gap = 0
            elif d == 'hold' and chain and gap == 0:
                gap += 1
                continue
            else:
                if len(chain) > best_chain_len:
                    best_chain_len = len(chain)
                    best_chain_dir = target
                    best_chain_has_4h = any(_MTF_MINUTES.get(t, 0) >= 240 for t in chain)
                chain = []
                gap = 0
                if d == target:
                    chain = [tf]
        if len(chain) > best_chain_len:
            best_chain_len = len(chain)
            best_chain_dir = target
            best_chain_has_4h = any(_MTF_MINUTES.get(t, 0) >= 240 for t in chain)

    # --- 5. 应用调节因子 ---
    boost = 1.0

    # 5a. 大小周期反向 → 衰减
    if large_tfs and small_tfs:
        # 大周期看空 + 小周期看多 → 衰减买入
        if large_ss_avg > large_bs_avg * 1.3 and small_bs_avg > small_ss_avg * 1.3:
            weighted_bs *= 0.5  # 小周期多在大周期空的环境下减弱
            weighted_ss *= 1.15  # 空方信号轻微增强
        # 大周期看多 + 小周期看空 → 衰减卖出
        elif large_bs_avg > large_ss_avg * 1.3 and small_ss_avg > small_bs_avg * 1.3:
            weighted_ss *= 0.5
            weighted_bs *= 1.15

    # 5b. 共振链加成
    if best_chain_len >= 3 and best_chain_has_4h:
        chain_boost = 1.0 + best_chain_len * 0.08  # 3级+8%, 4级+16%, ...
        if best_chain_dir == 'short':
            weighted_ss *= chain_boost
        else:
            weighted_bs *= chain_boost
    elif best_chain_len >= 2:
        chain_boost = 1.0 + best_chain_len * 0.04  # 2级+8%
        if best_chain_dir == 'short':
            weighted_ss *= chain_boost
        else:
            weighted_bs *= chain_boost

    meta = {
        'chain_len': best_chain_len,
        'chain_dir': best_chain_dir,
        'chain_has_4h': best_chain_has_4h,
        'large_ss': round(large_ss_avg, 1),
        'large_bs': round(large_bs_avg, 1),
        'small_ss': round(small_ss_avg, 1),
        'small_bs': round(small_bs_avg, 1),
    }

    return float(weighted_ss), float(weighted_bs), meta


def run_strategy_multi_tf(primary_df, tf_score_map, decision_tfs, config,
                          primary_tf='1h', trade_days=30):
    """
    多周期联合决策回测引擎

    与 run_strategy 的区别:
      - 在每根K线(主TF)决策时, 查询所有辅助TF的最新信号
      - 通过 calc_multi_tf_consensus 生成加权共识 ss/bs
      - 交易引擎与仓位管理逻辑复用 run_strategy

    参数:
      primary_df: 主时间框架 DataFrame (决定决策频率)
      tf_score_map: _build_tf_score_index 的输出
      decision_tfs: 参与决策的时间框架列表
      config: 策略参数
      primary_tf: 主TF名 (用于计算持仓周期等)
      trade_days: 回测天数
    """
    eng = FuturesEngine(config.get('name', 'multi_tf'), max_leverage=config.get('max_lev', 5))

    start_dt = None
    if trade_days and trade_days < 60:
        start_dt = primary_df.index[-1] - pd.Timedelta(days=trade_days)

    init_idx = 0
    if start_dt:
        init_idx = primary_df.index.searchsorted(start_dt)
        if init_idx >= len(primary_df): init_idx = 0
    eng.spot_eth = eng.initial_eth_value / primary_df['close'].iloc[init_idx]

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
    trail_pullback = config.get('trail_pullback', 0.60)

    use_dynamic_tp = config.get('use_dynamic_tp', False)
    use_partial_tp = config.get('use_partial_tp', False)
    partial_tp_1 = config.get('partial_tp_1', 0.30)
    partial_tp_1_pct = config.get('partial_tp_1_pct', 0.40)

    use_atr_sl = config.get('use_atr_sl', False)
    atr_sl_mult = config.get('atr_sl_mult', 3.0)

    use_partial_tp_2 = config.get('use_partial_tp_2', False)
    partial_tp_2 = config.get('partial_tp_2', 0.50)
    partial_tp_2_pct = config.get('partial_tp_2_pct', 0.30)

    short_cd = 0; long_cd = 0; spot_cd = 0
    short_bars = 0; long_bars = 0
    short_max_pnl = 0; long_max_pnl = 0
    short_partial_done = False; long_partial_done = False
    short_partial2_done = False; long_partial2_done = False

    warmup = max(60, int(len(primary_df) * 0.05))

    tf_hours = {'10m': 1/6, '15m': 0.25, '30m': 0.5, '1h': 1, '2h': 2, '3h': 3,
                '4h': 4, '6h': 6, '8h': 8, '12h': 12, '16h': 16, '24h': 24}
    bars_per_8h = max(1, int(8 / tf_hours.get(primary_tf, 1)))

    record_interval = max(1, len(primary_df) // 500)

    for idx in range(warmup, len(primary_df)):
        dt = primary_df.index[idx]
        price = primary_df['close'].iloc[idx]

        if start_dt and dt < start_dt:
            if idx % record_interval == 0: eng.record_history(dt, price)
            continue

        had_short_before_liq = eng.futures_short is not None
        had_long_before_liq = eng.futures_long is not None
        eng.check_liquidation(price, dt)
        if had_short_before_liq and eng.futures_short is None:
            short_cd = max(short_cd, cooldown * 6)
            short_bars = 0; short_max_pnl = 0
        if had_long_before_liq and eng.futures_long is None:
            long_cd = max(long_cd, cooldown * 6)
            long_bars = 0; long_max_pnl = 0
        short_just_opened = False; long_just_opened = False

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

        # ★★★ 核心区别: 用多周期共识评分替代单TF评分 ★★★
        ss, bs, _meta = calc_multi_tf_consensus(tf_score_map, decision_tfs, dt, config)

        in_conflict = False
        if ss > 0 and bs > 0:
            ratio = min(ss, bs) / max(ss, bs)
            in_conflict = ratio >= 0.6 and min(ss, bs) >= 15

        # ATR自适应止损
        actual_short_sl = short_sl
        actual_long_sl = long_sl
        if use_atr_sl and idx >= 20:
            high = primary_df['high'].iloc[max(0,idx-14):idx]
            low = primary_df['low'].iloc[max(0,idx-14):idx]
            close_prev = primary_df['close'].iloc[max(0,idx-15):idx-1]
            if len(high) > 0 and len(close_prev) > 0:
                min_len = min(len(high), len(low), len(close_prev))
                tr = pd.Series([max(h-l, abs(h-c), abs(l-c))
                                for h, l, c in zip(high[-min_len:], low[-min_len:], close_prev[-min_len:])])
                atr = tr.mean()
                atr_pct = atr / price
                actual_short_sl = max(short_sl, -(atr_pct * atr_sl_mult))
                actual_long_sl = max(long_sl, -(atr_pct * atr_sl_mult))

        # 动态止盈
        actual_short_tp = short_tp
        actual_long_tp = long_tp
        if use_dynamic_tp and idx >= 20:
            recent_returns = primary_df['close'].iloc[max(0,idx-20):idx].pct_change().dropna()
            if len(recent_returns) > 5:
                vol = recent_returns.std()
                if vol > 0.03:
                    actual_short_tp = short_tp * 1.3
                    actual_long_tp = long_tp * 1.3
                elif vol < 0.01:
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
            short_just_opened = True; short_partial_done = False; short_partial2_done = False

        # 管理空仓
        if eng.futures_short and not short_just_opened:
            short_bars += 1
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin

            if use_partial_tp and not short_partial_done and pnl_r >= partial_tp_1:
                old_qty = eng.futures_short.quantity
                partial_qty = old_qty * partial_tp_1_pct
                actual_close_p = price * (1 + FuturesEngine.SLIPPAGE)
                partial_pnl = (eng.futures_short.entry_price - actual_close_p) * partial_qty
                margin_released = eng.futures_short.margin * partial_tp_1_pct
                eng.usdt += margin_released + partial_pnl
                eng.frozen_margin -= margin_released
                fee = partial_qty * actual_close_p * FuturesEngine.TAKER_FEE
                slippage_cost = partial_qty * price * FuturesEngine.SLIPPAGE
                eng.usdt -= fee; eng.total_futures_fees += fee
                eng.total_slippage_cost += slippage_cost
                eng.futures_short.quantity = old_qty - partial_qty
                eng.futures_short.margin *= (1 - partial_tp_1_pct)
                short_partial_done = True
                eng.trades.append({'time': str(dt), 'action': '分段止盈空',
                    'price': price, 'pnl': partial_pnl, 'reason': f'分段TP1 +{pnl_r*100:.0f}%'})

            elif use_partial_tp_2 and short_partial_done and not short_partial2_done and pnl_r >= partial_tp_2:
                old_qty = eng.futures_short.quantity
                partial_qty = old_qty * partial_tp_2_pct
                actual_close_p = price * (1 + FuturesEngine.SLIPPAGE)
                partial_pnl = (eng.futures_short.entry_price - actual_close_p) * partial_qty
                margin_released = eng.futures_short.margin * partial_tp_2_pct
                eng.usdt += margin_released + partial_pnl
                eng.frozen_margin -= margin_released
                fee = partial_qty * actual_close_p * FuturesEngine.TAKER_FEE
                slippage_cost = partial_qty * price * FuturesEngine.SLIPPAGE
                eng.usdt -= fee; eng.total_futures_fees += fee
                eng.total_slippage_cost += slippage_cost
                eng.futures_short.quantity = old_qty - partial_qty
                eng.futures_short.margin *= (1 - partial_tp_2_pct)
                short_partial2_done = True
                eng.trades.append({'time': str(dt), 'action': '分段止盈空2',
                    'price': price, 'pnl': partial_pnl, 'reason': f'分段TP2 +{pnl_r*100:.0f}%'})

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
            long_just_opened = True; long_partial_done = False; long_partial2_done = False

        # 管理多仓
        if eng.futures_long and not long_just_opened:
            long_bars += 1
            pnl_r = eng.futures_long.calc_pnl(price) / eng.futures_long.margin

            if use_partial_tp and not long_partial_done and pnl_r >= partial_tp_1:
                old_qty = eng.futures_long.quantity
                partial_qty = old_qty * partial_tp_1_pct
                actual_close_p = price * (1 - FuturesEngine.SLIPPAGE)
                partial_pnl = (actual_close_p - eng.futures_long.entry_price) * partial_qty
                margin_released = eng.futures_long.margin * partial_tp_1_pct
                eng.usdt += margin_released + partial_pnl
                eng.frozen_margin -= margin_released
                fee = partial_qty * actual_close_p * FuturesEngine.TAKER_FEE
                slippage_cost = partial_qty * price * FuturesEngine.SLIPPAGE
                eng.usdt -= fee; eng.total_futures_fees += fee
                eng.total_slippage_cost += slippage_cost
                eng.futures_long.quantity = old_qty - partial_qty
                eng.futures_long.margin *= (1 - partial_tp_1_pct)
                long_partial_done = True
                eng.trades.append({'time': str(dt), 'action': '分段止盈多',
                    'price': price, 'pnl': partial_pnl, 'reason': f'分段TP1 +{pnl_r*100:.0f}%'})

            elif use_partial_tp_2 and long_partial_done and not long_partial2_done and pnl_r >= partial_tp_2:
                old_qty = eng.futures_long.quantity
                partial_qty = old_qty * partial_tp_2_pct
                actual_close_p = price * (1 - FuturesEngine.SLIPPAGE)
                partial_pnl = (actual_close_p - eng.futures_long.entry_price) * partial_qty
                margin_released = eng.futures_long.margin * partial_tp_2_pct
                eng.usdt += margin_released + partial_pnl
                eng.frozen_margin -= margin_released
                fee = partial_qty * actual_close_p * FuturesEngine.TAKER_FEE
                slippage_cost = partial_qty * price * FuturesEngine.SLIPPAGE
                eng.usdt -= fee; eng.total_futures_fees += fee
                eng.total_slippage_cost += slippage_cost
                eng.futures_long.quantity = old_qty - partial_qty
                eng.futures_long.margin *= (1 - partial_tp_2_pct)
                long_partial2_done = True
                eng.trades.append({'time': str(dt), 'action': '分段止盈多2',
                    'price': price, 'pnl': partial_pnl, 'reason': f'分段TP2 +{pnl_r*100:.0f}%'})

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
                    ss_dom = (bs < ss * 0.7) if bs > 0 else True
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
    last_price = primary_df['close'].iloc[-1]
    last_dt = primary_df.index[-1]
    if eng.futures_short: eng.close_short(last_price, last_dt, "期末平仓")
    if eng.futures_long: eng.close_long(last_price, last_dt, "期末平仓")

    if start_dt:
        trade_df = primary_df[primary_df.index >= start_dt]
        if len(trade_df) > 1: return eng.get_result(trade_df)
    return eng.get_result(primary_df)


# ======================================================
#   参数空间
# ======================================================
def get_all_variants():
    """获取全部参数变体(六书增强版)"""
    variants = []

    # ---------- 基准: 五书最优配置(用六书评分函数跑) ----------
    prev_best = {
        'tag': '五书最优(六书评分)',
        'fusion_mode': 'c6_veto_4',
        'short_sl': -0.25, 'short_tp': 0.60, 'short_trail': 0.25, 'short_max_hold': 72,
        'long_sl': -0.08, 'long_tp': 0.30, 'long_trail': 0.20, 'long_max_hold': 72,
        'trail_pullback': 0.60, 'cooldown': 4, 'spot_cooldown': 12,
        'use_partial_tp': True, 'partial_tp_1': 0.20, 'partial_tp_1_pct': 0.30,
    }
    variants.append(prev_best)

    # ---------- Phase 1: 融合模式对比 ----------
    for mode in ['c6_veto', 'c6_veto_4', 'kdj_weighted', 'kdj_timing', 'kdj_gate']:
        v = {**prev_best}
        v['tag'] = f'模式_{mode}'
        v['fusion_mode'] = mode
        variants.append(v)

    # ---------- Phase 2: KDJ否决阈值 ----------
    for vt in [15, 20, 25, 30, 35]:
        v = {**prev_best}
        v['tag'] = f'否决阈值_{vt}'
        v['fusion_mode'] = 'c6_veto_4'
        v['veto_threshold'] = vt
        variants.append(v)

    # ---------- Phase 3: KDJ确认奖励权重 ----------
    for kb in [0.05, 0.08, 0.12, 0.15, 0.20]:
        v = {**prev_best}
        v['tag'] = f'KDJ奖励_{kb*100:.0f}%'
        v['fusion_mode'] = 'c6_veto_4'
        v['kdj_bonus'] = kb
        variants.append(v)

    # ---------- Phase 4: KDJ加权模式权重搜索 ----------
    for kw in [0.10, 0.15, 0.20, 0.25, 0.30]:
        for dw in [0.45, 0.50, 0.55, 0.60]:
            if kw + dw > 0.85: continue  # 确保均线权重>0
            v = {**prev_best}
            v['tag'] = f'加权_KDJ{kw*100:.0f}_背离{dw*100:.0f}'
            v['fusion_mode'] = 'kdj_weighted'
            v['kdj_weight'] = kw
            v['div_weight'] = dw
            variants.append(v)

    # ---------- Phase 5: KDJ择时乘数 ----------
    for strong in [1.15, 1.20, 1.25, 1.30, 1.40]:
        for normal in [1.05, 1.10, 1.15, 1.20]:
            for reverse in [0.50, 0.60, 0.70, 0.80]:
                if strong <= normal: continue
                v = {**prev_best}
                v['tag'] = f'择时_强{strong}_弱{normal}_反{reverse}'
                v['fusion_mode'] = 'kdj_timing'
                v['kdj_strong_mult'] = strong
                v['kdj_normal_mult'] = normal
                v['kdj_reverse_mult'] = reverse
                variants.append(v)

    # ---------- Phase 6: KDJ门控阈值 ----------
    for gt in [5, 8, 10, 12, 15, 20]:
        v = {**prev_best}
        v['tag'] = f'门控_KDJ>{gt}'
        v['fusion_mode'] = 'kdj_gate'
        v['kdj_gate_threshold'] = gt
        variants.append(v)

    # ---------- Phase 7: 止损微调(围绕最优) ----------
    for ssl in [-0.15, -0.20, -0.25, -0.30, -0.35]:
        for lsl in [-0.06, -0.08, -0.10, -0.12, -0.15]:
            if ssl == -0.25 and lsl == -0.08: continue
            v = {**prev_best}
            v['tag'] = f'SL空{ssl*100:.0f}%多{lsl*100:.0f}%'
            v['short_sl'] = ssl; v['long_sl'] = lsl
            variants.append(v)

    # ---------- Phase 8: 止盈微调 ----------
    for stp in [0.40, 0.50, 0.60, 0.70, 0.80, 1.00]:
        for ltp in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
            if stp == 0.60 and ltp == 0.30: continue
            v = {**prev_best}
            v['tag'] = f'TP空{stp*100:.0f}%多{ltp*100:.0f}%'
            v['short_tp'] = stp; v['long_tp'] = ltp
            variants.append(v)

    # ---------- Phase 9: 追踪止盈 ----------
    for trail in [0.15, 0.20, 0.25, 0.30, 0.35]:
        for pullback in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            if trail == 0.25 and pullback == 0.60: continue
            v = {**prev_best}
            v['tag'] = f'Trail{trail*100:.0f}%回撤{pullback*100:.0f}%'
            v['short_trail'] = trail; v['long_trail'] = trail
            v['trail_pullback'] = pullback
            variants.append(v)

    # ---------- Phase 10: 分段止盈参数 ----------
    for pt1 in [0.15, 0.20, 0.25, 0.30, 0.40]:
        for pt1_pct in [0.20, 0.25, 0.30, 0.40, 0.50]:
            if pt1 == 0.20 and pt1_pct == 0.30: continue
            v = {**prev_best}
            v['tag'] = f'分段TP@{pt1*100:.0f}%平{pt1_pct*100:.0f}%'
            v['partial_tp_1'] = pt1; v['partial_tp_1_pct'] = pt1_pct
            variants.append(v)

    # ---------- Phase 11: 二段止盈 ----------
    for pt1 in [0.15, 0.20]:
        for pt1_pct in [0.25, 0.30]:
            for pt2 in [0.40, 0.50, 0.60]:
                for pt2_pct in [0.25, 0.30, 0.40]:
                    v = {**prev_best}
                    v['tag'] = f'双段TP@{pt1*100:.0f}/{pt2*100:.0f}%平{pt1_pct*100:.0f}/{pt2_pct*100:.0f}%'
                    v['partial_tp_1'] = pt1; v['partial_tp_1_pct'] = pt1_pct
                    v['use_partial_tp_2'] = True
                    v['partial_tp_2'] = pt2; v['partial_tp_2_pct'] = pt2_pct
                    variants.append(v)

    # ---------- Phase 12: 信号阈值 ----------
    for sell_t in [15, 18, 22]:
        for short_t in [20, 25, 30]:
            for buy_t in [20, 25, 30]:
                for long_t in [35, 40, 45]:
                    if sell_t == 18 and short_t == 25 and buy_t == 25 and long_t == 40: continue
                    v = {**prev_best}
                    v['tag'] = f'阈值S{sell_t}_空{short_t}_B{buy_t}_多{long_t}'
                    v['sell_threshold'] = sell_t; v['short_threshold'] = short_t
                    v['buy_threshold'] = buy_t; v['long_threshold'] = long_t
                    variants.append(v)

    # ---------- Phase 13: 持仓时间 ----------
    for hold in [36, 48, 72, 96, 120, 168]:
        v = {**prev_best}
        v['tag'] = f'Hold{hold}bars'
        v['short_max_hold'] = hold; v['long_max_hold'] = hold
        variants.append(v)

    # ---------- Phase 14: ATR止损 + 分段止盈组合 ----------
    for atr_m in [2.0, 2.5, 3.0, 3.5]:
        v = {**prev_best}
        v['tag'] = f'ATR{atr_m}x+分段TP'
        v['use_atr_sl'] = True; v['atr_sl_mult'] = atr_m
        variants.append(v)

    # ---------- Phase 15: 杠杆和仓位 ----------
    for l in [3, 4, 5, 7]:
        for mu in [0.50, 0.60, 0.70, 0.80]:
            if l == 5 and mu == 0.70: continue
            v = {**prev_best}
            v['tag'] = f'杠杆{l}x仓位{mu*100:.0f}%'
            v['lev'] = l; v['margin_use'] = mu
            variants.append(v)

    # ---------- Phase 16: 否决削弱比例 ----------
    for vd in [0.15, 0.20, 0.30, 0.40, 0.50]:
        v = {**prev_best}
        v['tag'] = f'否决削弱{vd*100:.0f}%'
        v['fusion_mode'] = 'c6_veto_4'
        v['veto_dampen'] = vd
        variants.append(v)

    return variants


# ======================================================
#   主函数
# ======================================================
def main():
    trade_days = 30
    print("=" * 120)
    print("  六书融合多时间框架止盈止损优化器")
    print("  基于六书(含KDJ)信号 · 12个时间周期 · 系统性参数搜索")
    print(f"  目标: 超越五书最优 α=+86.69%")
    print("=" * 120)

    # 获取所有时间框架数据
    print("\n[1/5] 获取多时间框架数据...")
    all_data = fetch_multi_tf_data(ALL_TIMEFRAMES, days=60)
    available_tfs = sorted(all_data.keys(), key=lambda x: ALL_TIMEFRAMES.index(x) if x in ALL_TIMEFRAMES else 99)
    print(f"\n可用时间框架: {', '.join(available_tfs)}")

    # 预计算各时间框架六维信号
    print("\n[2/5] 预计算各时间框架的六维信号(含KDJ)...")
    all_signals = {}
    for tf in available_tfs:
        print(f"\n  计算 {tf} 六维信号:")
        all_signals[tf] = compute_signals_six(all_data[tf], tf, all_data)
        print(f"    {tf} 六维信号计算完成")

    # 基准配置(五书最优参数 + 六书评分)
    f12_base = {
        'name': '六书基准',
        'fusion_mode': 'c6_veto_4',
        'veto_threshold': 25,
        'single_pct': 0.20, 'total_pct': 0.50, 'lifetime_pct': 5.0,
        'sell_threshold': 18, 'buy_threshold': 25,
        'short_threshold': 25, 'long_threshold': 40,
        'close_short_bs': 40, 'close_long_ss': 40,
        'sell_pct': 0.55, 'margin_use': 0.70, 'lev': 5, 'max_lev': 5,
    }

    tf_hours = {'10m': 1/6, '15m': 0.25, '30m': 0.5, '1h': 1, '2h': 2, '3h': 3,
                '4h': 4, '6h': 6, '8h': 8, '12h': 12, '16h': 16, '24h': 24}

    # 获取参数变体
    all_variants = get_all_variants()
    print(f"\n[3/5] 参数变体: {len(all_variants)}种")

    # ============ Phase A: 各时间框架基准测试 ============
    print(f"\n{'=' * 120}")
    print(f"  Phase A: 各时间框架基准性能(六书C6+四书否决)")
    print(f"{'=' * 120}")

    # 五书最优参数
    prev_best_sl_tp = {
        'short_sl': -0.25, 'short_tp': 0.60, 'short_trail': 0.25, 'short_max_hold': 72,
        'long_sl': -0.08, 'long_tp': 0.30, 'long_trail': 0.20, 'long_max_hold': 72,
        'trail_pullback': 0.60, 'cooldown': 4, 'spot_cooldown': 12,
        'use_partial_tp': True, 'partial_tp_1': 0.20, 'partial_tp_1_pct': 0.30,
    }

    tf_baseline_results = {}
    print(f"\n{'时间框架':<10} {'K线数':>8} {'Alpha':>10} {'策略收益':>12} {'BH收益':>12} "
          f"{'回撤':>8} {'交易':>6} {'强平':>4} {'费用':>10}")
    print('-' * 100)

    for tf in available_tfs:
        config = {**f12_base, **prev_best_sl_tp}
        config['name'] = f'六书_{tf}'
        hours = tf_hours.get(tf, 1)
        config['short_max_hold'] = max(6, int(72 / hours))
        config['long_max_hold'] = max(6, int(72 / hours))
        config['cooldown'] = max(1, int(4 / hours))
        config['spot_cooldown'] = max(2, int(12 / hours))

        r = run_strategy(all_data[tf], all_signals[tf], config, tf=tf, trade_days=trade_days)
        fees = r.get('fees', {})
        tf_baseline_results[tf] = r

        print(f"  {tf:<8} {len(all_data[tf]):>8} {r['alpha']:>+9.2f}% {r['strategy_return']:>+11.2f}% "
              f"{r['buy_hold_return']:>+11.2f}% {r['max_drawdown']:>7.2f}% "
              f"{r['total_trades']:>5} {r['liquidations']:>3} "
              f"${fees.get('total_costs', 0):>9,.0f}")

    tf_ranked = sorted(tf_baseline_results.items(), key=lambda x: x[1]['alpha'], reverse=True)
    top_tfs = [t[0] for t in tf_ranked[:4]]  # 选TOP4
    print(f"\n  TOP4时间框架: {', '.join(top_tfs)}")
    for i, (tf, r) in enumerate(tf_ranked[:4]):
        print(f"    #{i+1}: {tf} α={r['alpha']:+.2f}%")

    # ============ Phase B: 大规模参数搜索 ============
    print(f"\n{'=' * 120}")
    print(f"  Phase B: 在TOP时间框架上系统优化({len(all_variants)}种变体)")
    print(f"{'=' * 120}")

    all_opt_results = []

    for tf in top_tfs:
        print(f"\n  === {tf} 优化开始 ({len(all_variants)}种参数变体) ===")
        tf_h = tf_hours.get(tf, 1)
        results_for_tf = []

        for i, var in enumerate(all_variants):
            config = {**f12_base, **var}
            config['name'] = f'{var["tag"]}_{tf}'

            raw_hold = var.get('short_max_hold', 72)
            config['short_max_hold'] = max(6, int(raw_hold / tf_h))
            config['long_max_hold'] = max(6, int(var.get('long_max_hold', 72) / tf_h))
            config['cooldown'] = max(1, int(var.get('cooldown', 4) / tf_h))
            config['spot_cooldown'] = max(2, int(var.get('spot_cooldown', 12) / tf_h))

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

            if (i + 1) % 50 == 0:
                print(f"    进度: {i+1}/{len(all_variants)}")

        results_for_tf.sort(key=lambda x: x['alpha'], reverse=True)
        all_opt_results.extend(results_for_tf)

        print(f"\n  {tf} TOP15参数:")
        print(f"  {'排名':>4} {'参数标签':<40} {'Alpha':>10} {'收益':>10} {'回撤':>8} {'交易':>6}")
        print('  ' + '-' * 90)
        for i, r in enumerate(results_for_tf[:15]):
            star = ' ★' if i == 0 else ''
            print(f"  #{i+1:>3} {r['tag']:<40} {r['alpha']:>+9.2f}% "
                  f"{r['strategy_return']:>+9.2f}% {r['max_drawdown']:>7.2f}% {r['total_trades']:>5}{star}")

    # ============ Phase C: 全局TOP排序 ============
    print(f"\n{'=' * 120}")
    print(f"  Phase C: 全局最优参数组合")
    print(f"{'=' * 120}")

    all_opt_results.sort(key=lambda x: x['alpha'], reverse=True)

    print(f"\n  全局TOP30:")
    print(f"  {'排名':>4} {'时间框架':>8} {'参数标签':<40} {'Alpha':>10} {'收益':>12} {'回撤':>8} {'交易':>6} {'费用':>10}")
    print('  ' + '-' * 120)
    for i, r in enumerate(all_opt_results[:30]):
        star = ' ★★★' if i == 0 else ' ★★' if i <= 2 else ' ★' if i <= 4 else ''
        print(f"  #{i+1:>3} {r['tf']:>8} {r['tag']:<40} {r['alpha']:>+9.2f}% "
              f"{r['strategy_return']:>+11.2f}% {r['max_drawdown']:>7.2f}% "
              f"{r['total_trades']:>5} ${r['fees']:>9,.0f}{star}")

    # ============ Phase D: 精细组合搜索 ============
    if len(all_opt_results) >= 5:
        print(f"\n{'=' * 120}")
        print(f"  Phase D: 精细组合搜索(交叉最优参数)")
        print(f"{'=' * 120}")

        top1 = all_opt_results[0]
        top1_tf = top1['tf']
        top1_cfg = top1['config']

        # 从各维度提取最佳
        def best_of(prefix, results):
            return sorted([r for r in results if r['tag'].startswith(prefix)],
                         key=lambda x: x['alpha'], reverse=True)

        best_mode = best_of('模式_', all_opt_results)
        best_veto = best_of('否决阈值_', all_opt_results)
        best_kdj_bonus = best_of('KDJ奖励_', all_opt_results)
        best_sl = best_of('SL', all_opt_results)
        best_tp = best_of('TP', all_opt_results)
        best_trail = best_of('Trail', all_opt_results)
        best_partial = best_of('分段TP', all_opt_results)
        best_dual = best_of('双段TP', all_opt_results)
        best_threshold = best_of('阈值', all_opt_results)
        best_lev = best_of('杠杆', all_opt_results)
        best_gate = best_of('门控', all_opt_results)
        best_timing = best_of('择时', all_opt_results)
        best_weighted = best_of('加权', all_opt_results)
        best_atr = best_of('ATR', all_opt_results)
        best_dampen = best_of('否决削弱', all_opt_results)

        fine_variants = []

        # 组合1: 最佳模式 + 最佳SL + 最佳TP + 最佳分段
        if best_mode and best_sl and best_tp:
            base_mode_cfg = best_mode[0]['config'] if best_mode else top1_cfg
            base_sl_cfg = best_sl[0]['config'] if best_sl else {}
            base_tp_cfg = best_tp[0]['config'] if best_tp else {}
            base_partial_cfg = best_partial[0]['config'] if best_partial else {}
            base_trail_cfg = best_trail[0]['config'] if best_trail else {}

            combined = {**prev_best_sl_tp}
            combined['fusion_mode'] = base_mode_cfg.get('fusion_mode', 'c6_veto_4')
            if best_veto: combined['veto_threshold'] = best_veto[0]['config'].get('veto_threshold', 25)
            if best_kdj_bonus: combined['kdj_bonus'] = best_kdj_bonus[0]['config'].get('kdj_bonus', 0.09)
            combined['short_sl'] = base_sl_cfg.get('short_sl', -0.25)
            combined['long_sl'] = base_sl_cfg.get('long_sl', -0.08)
            combined['short_tp'] = base_tp_cfg.get('short_tp', 0.60)
            combined['long_tp'] = base_tp_cfg.get('long_tp', 0.30)
            if base_trail_cfg:
                combined['short_trail'] = base_trail_cfg.get('short_trail', 0.25)
                combined['long_trail'] = base_trail_cfg.get('long_trail', 0.20)
                combined['trail_pullback'] = base_trail_cfg.get('trail_pullback', 0.60)
            if base_partial_cfg:
                combined['partial_tp_1'] = base_partial_cfg.get('partial_tp_1', 0.20)
                combined['partial_tp_1_pct'] = base_partial_cfg.get('partial_tp_1_pct', 0.30)
            combined['tag'] = '组合A_最佳模式+SL+TP+分段'
            fine_variants.append(combined)

        # 组合2: 全局TOP1参数 + 最佳分段
        if best_partial:
            combined2 = {**top1_cfg}
            combined2['partial_tp_1'] = best_partial[0]['config'].get('partial_tp_1', 0.20)
            combined2['partial_tp_1_pct'] = best_partial[0]['config'].get('partial_tp_1_pct', 0.30)
            combined2['tag'] = '组合B_TOP1+最佳分段'
            fine_variants.append(combined2)

        # 组合3: 最佳模式 + ATR止损 + 最佳TP
        if best_atr and best_tp:
            combined3 = {**prev_best_sl_tp}
            combined3['use_atr_sl'] = True
            combined3['atr_sl_mult'] = best_atr[0]['config'].get('atr_sl_mult', 3.0)
            combined3['short_tp'] = best_tp[0]['config'].get('short_tp', 0.60)
            combined3['long_tp'] = best_tp[0]['config'].get('long_tp', 0.30)
            if best_mode: combined3['fusion_mode'] = best_mode[0]['config'].get('fusion_mode', 'c6_veto_4')
            combined3['tag'] = '组合C_ATR止损+最佳TP'
            fine_variants.append(combined3)

        # 组合4: 最佳阈值 + 最佳否决 + 最佳SL/TP
        if best_threshold and best_sl and best_tp:
            combined4 = {**prev_best_sl_tp}
            combined4['sell_threshold'] = best_threshold[0]['config'].get('sell_threshold', 18)
            combined4['short_threshold'] = best_threshold[0]['config'].get('short_threshold', 25)
            combined4['buy_threshold'] = best_threshold[0]['config'].get('buy_threshold', 25)
            combined4['long_threshold'] = best_threshold[0]['config'].get('long_threshold', 40)
            combined4['short_sl'] = best_sl[0]['config'].get('short_sl', -0.25)
            combined4['long_sl'] = best_sl[0]['config'].get('long_sl', -0.08)
            combined4['short_tp'] = best_tp[0]['config'].get('short_tp', 0.60)
            combined4['long_tp'] = best_tp[0]['config'].get('long_tp', 0.30)
            if best_veto: combined4['veto_threshold'] = best_veto[0]['config'].get('veto_threshold', 25)
            combined4['tag'] = '组合D_阈值+否决+SL/TP'
            fine_variants.append(combined4)

        # 组合5: 双段止盈 + 最佳SL/TP
        if best_dual and best_sl and best_tp:
            combined5 = {**best_dual[0]['config']}
            combined5['short_sl'] = best_sl[0]['config'].get('short_sl', -0.25)
            combined5['long_sl'] = best_sl[0]['config'].get('long_sl', -0.08)
            combined5['short_tp'] = best_tp[0]['config'].get('short_tp', 0.60)
            combined5['long_tp'] = best_tp[0]['config'].get('long_tp', 0.30)
            combined5['tag'] = '组合E_双段TP+最佳SL/TP'
            fine_variants.append(combined5)

        # 组合6: 最佳杠杆 + 最佳SL/TP + 最佳分段
        if best_lev and best_sl and best_tp:
            combined6 = {**prev_best_sl_tp}
            combined6['lev'] = best_lev[0]['config'].get('lev', 5)
            combined6['margin_use'] = best_lev[0]['config'].get('margin_use', 0.70)
            combined6['short_sl'] = best_sl[0]['config'].get('short_sl', -0.25)
            combined6['long_sl'] = best_sl[0]['config'].get('long_sl', -0.08)
            combined6['short_tp'] = best_tp[0]['config'].get('short_tp', 0.60)
            combined6['long_tp'] = best_tp[0]['config'].get('long_tp', 0.30)
            combined6['tag'] = '组合F_杠杆+SL/TP+分段'
            fine_variants.append(combined6)

        # 组合7: TOP1完整配置 + 否决削弱优化
        if best_dampen:
            combined7 = {**top1_cfg}
            combined7['veto_dampen'] = best_dampen[0]['config'].get('veto_dampen', 0.30)
            combined7['tag'] = '组合G_TOP1+否决削弱'
            fine_variants.append(combined7)

        # 组合8: KDJ门控 + 最佳SL/TP + 分段
        if best_gate and best_sl and best_tp:
            combined8 = {**prev_best_sl_tp}
            combined8['fusion_mode'] = 'kdj_gate'
            combined8['kdj_gate_threshold'] = best_gate[0]['config'].get('kdj_gate_threshold', 10)
            combined8['short_sl'] = best_sl[0]['config'].get('short_sl', -0.25)
            combined8['long_sl'] = best_sl[0]['config'].get('long_sl', -0.08)
            combined8['short_tp'] = best_tp[0]['config'].get('short_tp', 0.60)
            combined8['long_tp'] = best_tp[0]['config'].get('long_tp', 0.30)
            combined8['tag'] = '组合H_KDJ门控+SL/TP'
            fine_variants.append(combined8)

        # 组合9: KDJ择时 + 最佳SL/TP + 分段
        if best_timing and best_sl and best_tp:
            combined9 = {**prev_best_sl_tp}
            combined9['fusion_mode'] = 'kdj_timing'
            combined9['kdj_strong_mult'] = best_timing[0]['config'].get('kdj_strong_mult', 1.25)
            combined9['kdj_normal_mult'] = best_timing[0]['config'].get('kdj_normal_mult', 1.12)
            combined9['kdj_reverse_mult'] = best_timing[0]['config'].get('kdj_reverse_mult', 0.70)
            combined9['short_sl'] = best_sl[0]['config'].get('short_sl', -0.25)
            combined9['long_sl'] = best_sl[0]['config'].get('long_sl', -0.08)
            combined9['short_tp'] = best_tp[0]['config'].get('short_tp', 0.60)
            combined9['long_tp'] = best_tp[0]['config'].get('long_tp', 0.30)
            combined9['tag'] = '组合I_KDJ择时+SL/TP'
            fine_variants.append(combined9)

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
        print(f"  {'排名':>4} {'参数标签':<45} {'Alpha':>10} {'收益':>12} {'回撤':>8} {'交易':>6}")
        print('  ' + '-' * 100)
        for i, r in enumerate(fine_results[:10]):
            star = ' ★★★' if i == 0 else ''
            print(f"  #{i+1:>3} {r['tag']:<45} {r['alpha']:>+9.2f}% "
                  f"{r['strategy_return']:>+11.2f}% {r['max_drawdown']:>7.2f}% "
                  f"{r['total_trades']:>5}{star}")

        all_opt_results.extend(fine_results)
        all_opt_results.sort(key=lambda x: x['alpha'], reverse=True)

    # ============ Phase E: 多周期联合决策回测 ============
    print(f"\n{'=' * 120}")
    print(f"  Phase E: 多周期联合决策回测 (加权共识 + 共振链检测)")
    print(f"{'=' * 120}")

    # 取全局最优参数作为基础配置
    best_cfg_for_multi = all_opt_results[0]['config'] if all_opt_results else prev_best_sl_tp
    best_single_tf = all_opt_results[0]['tf'] if all_opt_results else '4h'
    best_single_alpha = all_opt_results[0]['alpha'] if all_opt_results else 0

    print(f"\n  单TF最优基线: {best_single_tf} α={best_single_alpha:+.2f}%")
    print(f"  基础参数: {best_cfg_for_multi.get('tag', '默认')}")

    # 预计算所有TF的评分索引
    print(f"\n  [E1] 预计算各TF评分索引...")
    tf_score_index = _build_tf_score_index(all_data, all_signals, available_tfs, {**f12_base, **best_cfg_for_multi})
    for tf in available_tfs:
        n_scores = len(tf_score_index.get(tf, {}))
        print(f"    {tf}: {n_scores} 个评分点")

    # 定义多种TF组合方案
    multi_tf_combos = []

    # 方案1: 全TF (所有可用周期)
    multi_tf_combos.append(('全周期', available_tfs))

    # 方案2: 大周期为主 (≥1h)
    large_only = [tf for tf in available_tfs if _MTF_MINUTES.get(tf, 0) >= 60]
    if len(large_only) >= 3:
        multi_tf_combos.append(('大周期(≥1h)', large_only))

    # 方案3: 核心周期 (30m, 1h, 4h, 8h, 24h)
    core_tfs = [tf for tf in ['30m', '1h', '4h', '8h', '24h'] if tf in available_tfs]
    if len(core_tfs) >= 3:
        multi_tf_combos.append(('核心周期', core_tfs))

    # 方案4: 小+大搭配 (15m, 1h, 4h, 12h)
    balanced_tfs = [tf for tf in ['15m', '1h', '4h', '12h'] if tf in available_tfs]
    if len(balanced_tfs) >= 3:
        multi_tf_combos.append(('均衡搭配', balanced_tfs))

    # 方案5: TOP3单TF周期
    top3_single = [t[0] for t in tf_ranked[:3]]
    if len(top3_single) >= 2:
        multi_tf_combos.append(('TOP3单TF', top3_single))

    # 方案6: 中大周期 (1h, 2h, 4h, 8h, 12h)
    mid_large = [tf for tf in ['1h', '2h', '4h', '8h', '12h'] if tf in available_tfs]
    if len(mid_large) >= 3:
        multi_tf_combos.append(('中大周期', mid_large))

    # 各主TF x 各组合方案
    primary_tf_candidates = ['1h', '2h', '4h']
    primary_tf_candidates = [tf for tf in primary_tf_candidates if tf in available_tfs]

    multi_tf_results = []

    print(f"\n  [E2] 运行多周期联合决策回测...")
    print(f"  主TF: {primary_tf_candidates}")
    print(f"  组合方案: {len(multi_tf_combos)}种")
    print(f"\n  {'方案':<25} {'主TF':>5} {'辅助TFs':<45} {'Alpha':>10} {'收益':>12} {'回撤':>8} {'交易':>6}")
    print('  ' + '-' * 120)

    for combo_name, combo_tfs in multi_tf_combos:
        for ptf in primary_tf_candidates:
            if ptf not in all_data:
                continue

            config = {**f12_base, **best_cfg_for_multi}
            config['name'] = f'多TF_{combo_name}@{ptf}'

            tf_h = tf_hours.get(ptf, 1)
            config['short_max_hold'] = max(6, int(best_cfg_for_multi.get('short_max_hold', 72) / tf_h))
            config['long_max_hold'] = max(6, int(best_cfg_for_multi.get('long_max_hold', 72) / tf_h))
            config['cooldown'] = max(1, int(best_cfg_for_multi.get('cooldown', 4) / tf_h))
            config['spot_cooldown'] = max(2, int(best_cfg_for_multi.get('spot_cooldown', 12) / tf_h))

            r = run_strategy_multi_tf(
                all_data[ptf], tf_score_index, combo_tfs, config,
                primary_tf=ptf, trade_days=trade_days
            )
            fees = r.get('fees', {})
            entry = {
                'combo_name': combo_name,
                'primary_tf': ptf,
                'decision_tfs': combo_tfs,
                'alpha': r['alpha'],
                'strategy_return': r['strategy_return'],
                'buy_hold_return': r['buy_hold_return'],
                'max_drawdown': r['max_drawdown'],
                'total_trades': r['total_trades'],
                'liquidations': r['liquidations'],
                'fees': fees.get('total_costs', 0),
                'final_total': r.get('final_total', 0),
                'full_result': r,
            }
            multi_tf_results.append(entry)

            vs_single = r['alpha'] - best_single_alpha
            marker = ' ★' if vs_single > 0 else ''
            print(f"  {combo_name:<25} {ptf:>5} {','.join(combo_tfs):<45} "
                  f"{r['alpha']:>+9.2f}% {r['strategy_return']:>+11.2f}% "
                  f"{r['max_drawdown']:>7.2f}% {r['total_trades']:>5}{marker}")

    multi_tf_results.sort(key=lambda x: x['alpha'], reverse=True)

    print(f"\n  {'─' * 120}")
    print(f"  多周期联合决策 TOP5:")
    for i, r in enumerate(multi_tf_results[:5]):
        vs = r['alpha'] - best_single_alpha
        print(f"    #{i+1} {r['combo_name']}@{r['primary_tf']} "
              f"α={r['alpha']:+.2f}% (vs单TF最优: {vs:+.2f}%) "
              f"回撤={r['max_drawdown']:.2f}% 交易={r['total_trades']}")

    if multi_tf_results:
        best_multi = multi_tf_results[0]
        vs = best_multi['alpha'] - best_single_alpha
        print(f"\n  ★ 多周期最优: {best_multi['combo_name']}@{best_multi['primary_tf']}")
        print(f"    Alpha: {best_multi['alpha']:+.2f}% (vs单TF: {vs:+.2f}%)")
        print(f"    收益: {best_multi['strategy_return']:+.2f}% | 回撤: {best_multi['max_drawdown']:.2f}%")
        print(f"    参与TFs: {','.join(best_multi['decision_tfs'])}")

    # ============ 保存结果 ============
    print(f"\n[5/5] 保存结果...")

    global_best = all_opt_results[0] if all_opt_results else None

    # 清理不可序列化
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

    output = {
        'description': f'六书融合多时间框架优化 · 最近{trade_days}天 · 含KDJ',
        'run_time': datetime.now().isoformat(),
        'available_timeframes': available_tfs,
        'total_variants_tested': len(all_opt_results),
        'trade_days': trade_days,
        'previous_best_alpha': 86.69,

        'baseline_by_tf': [{
            'tf': tf,
            'alpha': tf_baseline_results[tf]['alpha'],
            'strategy_return': tf_baseline_results[tf]['strategy_return'],
            'buy_hold_return': tf_baseline_results[tf]['buy_hold_return'],
            'max_drawdown': tf_baseline_results[tf]['max_drawdown'],
            'total_trades': tf_baseline_results[tf]['total_trades'],
        } for tf in available_tfs if tf in tf_baseline_results],

        'top_timeframes': top_tfs,

        'global_top30': [{
            'rank': i + 1,
            'tf': r['tf'],
            'tag': r['tag'],
            'alpha': r['alpha'],
            'strategy_return': r['strategy_return'],
            'max_drawdown': r['max_drawdown'],
            'total_trades': r['total_trades'],
            'fees': r['fees'],
            'config': r['config'],
        } for i, r in enumerate(all_opt_results[:30])],

        'global_best': {
            'tf': global_best['tf'],
            'tag': global_best['tag'],
            'alpha': global_best['alpha'],
            'strategy_return': global_best['strategy_return'],
            'max_drawdown': global_best['max_drawdown'],
            'total_trades': global_best['total_trades'],
            'config': global_best['config'],
        } if global_best else None,

        # Phase E: 多周期联合决策结果
        'multi_tf_results': [{
            'rank': i + 1,
            'combo_name': r['combo_name'],
            'primary_tf': r['primary_tf'],
            'decision_tfs': r['decision_tfs'],
            'alpha': r['alpha'],
            'strategy_return': r['strategy_return'],
            'buy_hold_return': r.get('buy_hold_return', 0),
            'max_drawdown': r['max_drawdown'],
            'total_trades': r['total_trades'],
            'fees': r['fees'],
        } for i, r in enumerate(multi_tf_results[:20])],
        'multi_tf_best': {
            'combo_name': multi_tf_results[0]['combo_name'],
            'primary_tf': multi_tf_results[0]['primary_tf'],
            'decision_tfs': multi_tf_results[0]['decision_tfs'],
            'alpha': multi_tf_results[0]['alpha'],
            'strategy_return': multi_tf_results[0]['strategy_return'],
            'max_drawdown': multi_tf_results[0]['max_drawdown'],
            'total_trades': multi_tf_results[0]['total_trades'],
            'vs_single_tf': multi_tf_results[0]['alpha'] - best_single_alpha,
        } if multi_tf_results else None,
    }

    # 添加完整best trades/history
    if global_best and 'full_result' in global_best:
        output['global_best_trades'] = global_best['full_result'].get('trades', [])
        output['global_best_history'] = global_best['full_result'].get('history', [])
        output['global_best_fees'] = global_best['full_result'].get('fees', {})

    output_clean = clean_for_json(output)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'optimize_six_book_result.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output_clean, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果已保存: {path}")

    # 最终总结
    print(f"\n{'=' * 120}")
    print(f"  六书融合优化完成总结")
    print(f"{'=' * 120}")
    print(f"\n  测试时间框架: {len(available_tfs)}个 ({', '.join(available_tfs)})")
    print(f"  参数变体总数: {len(all_opt_results)}")
    print(f"  五书前最优:   α=+86.69%")
    if global_best:
        improvement = global_best['alpha'] - 86.69
        print(f"\n  ★ 六书全局最优策略:")
        print(f"     时间框架: {global_best['tf']}")
        print(f"     参数标签: {global_best['tag']}")
        print(f"     Alpha:    {global_best['alpha']:+.2f}%")
        print(f"     策略收益: {global_best['strategy_return']:+.2f}%")
        print(f"     最大回撤: {global_best['max_drawdown']:.2f}%")
        print(f"     交易次数: {global_best['total_trades']}")
        print(f"\n     vs 五书最优: {improvement:+.2f}% {'★ 超越!' if improvement > 0 else '未超越'}")

        cfg = global_best['config']
        print(f"\n  最优参数:")
        print(f"     融合模式:   {cfg.get('fusion_mode', 'c6_veto_4')}")
        print(f"     空头止损:   {cfg.get('short_sl', -0.25)*100:.0f}%")
        print(f"     空头止盈:   {cfg.get('short_tp', 0.60)*100:.0f}%")
        print(f"     多头止损:   {cfg.get('long_sl', -0.08)*100:.0f}%")
        print(f"     多头止盈:   {cfg.get('long_tp', 0.30)*100:.0f}%")
        print(f"     追踪止盈:   {cfg.get('short_trail', 0.25)*100:.0f}%")
        print(f"     回撤比例:   {cfg.get('trail_pullback', 0.60)*100:.0f}%")
        print(f"     最大持仓:   {cfg.get('short_max_hold', 72)} bars")
        if cfg.get('use_partial_tp'): print(f"     分段止盈:   @{cfg.get('partial_tp_1',0.2)*100:.0f}% 平{cfg.get('partial_tp_1_pct',0.3)*100:.0f}%")
        if cfg.get('use_partial_tp_2'): print(f"     二段止盈:   @{cfg.get('partial_tp_2',0.5)*100:.0f}% 平{cfg.get('partial_tp_2_pct',0.3)*100:.0f}%")
        if cfg.get('use_atr_sl'): print(f"     ATR止损:    {cfg.get('atr_sl_mult', 3.0)}x")
        if cfg.get('kdj_bonus'): print(f"     KDJ奖励:    {cfg.get('kdj_bonus', 0.09)*100:.0f}%")
        if cfg.get('veto_threshold'): print(f"     否决阈值:   {cfg.get('veto_threshold', 25)}")

    # 多周期联合决策对比
    if multi_tf_results:
        best_m = multi_tf_results[0]
        vs_single = best_m['alpha'] - best_single_alpha
        print(f"\n  ★ 多周期联合决策最优:")
        print(f"     方案:      {best_m['combo_name']}@{best_m['primary_tf']}")
        print(f"     参与TFs:   {','.join(best_m['decision_tfs'])}")
        print(f"     Alpha:     {best_m['alpha']:+.2f}%")
        print(f"     策略收益:  {best_m['strategy_return']:+.2f}%")
        print(f"     最大回撤:  {best_m['max_drawdown']:.2f}%")
        print(f"     交易次数:  {best_m['total_trades']}")
        print(f"\n     vs 单TF最优({best_single_tf}): {vs_single:+.2f}% "
              f"{'★ 多周期更优!' if vs_single > 0 else '单TF更优'}")

    return output


if __name__ == '__main__':
    main()
