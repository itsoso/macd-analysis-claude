"""
六书融合策略 — 背离 × 均线 × 蜡烛图 × 布林带 × 量价分析 × KDJ波段

新增第6本书:
  6. 《随机指标KDJ：波段操作精解》(凌波) → KDJ超买超卖/金叉死叉/背离/
     四撞顶底/KD-MACD柱线/二次交叉/回测不破 → 短线波段拐点

融合升级:
  - 从"C6+三书否决"升级为"C6+四书否决"
  - KDJ作为第4个否决/确认维度
  - 否决逻辑: 至少2/4个辅助系统强烈反对 → 削弱信号
  - KDJ擅长短线拐点, 为波段操作提供更精准的入场/出场时机

初始: 10万USDT + 价值10万USDT的ETH
数据: 币安 ETH/USDT, 1h K线
"""

import pandas as pd
import numpy as np
import json
import sys
import os
import time
from datetime import datetime

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
#   数据获取
# ======================================================
def fetch_data():
    """获取多周期数据"""
    print("获取数据...")
    data = {}
    for tf, days in [('1h', 60), ('4h', 60), ('8h', 60)]:
        df = fetch_binance_klines("ETHUSDT", interval=tf, days=days)
        if df is not None and len(df) > 50:
            df = add_all_indicators(df)
            add_moving_averages(df, timeframe=tf)
            data[tf] = df
            print(f"  {tf}: {len(df)} 条 ({df.index[0]} ~ {df.index[-1]})")
    return data


# ======================================================
#   六维信号计算
# ======================================================
def compute_all_signals(data):
    """预计算所有六个系统的信号"""
    main_df = data['1h']
    signals = {}

    # 1. 背离信号
    print("  [1/6] 计算背离信号...")
    div_signals_1h = analyze_signals_enhanced(main_df, 168)
    div_signals_8h = {}
    if '8h' in data:
        div_signals_8h = analyze_signals_enhanced(data['8h'], 90)
    signals['div_1h'] = div_signals_1h
    signals['div_8h'] = div_signals_8h
    print(f"    1h背离: {len(div_signals_1h)} | 8h背离: {len(div_signals_8h)}")

    # 2. 均线信号
    print("  [2/6] 计算均线信号...")
    ma_signals = compute_ma_signals(main_df, timeframe='1h')
    signals['ma'] = ma_signals
    ma_buy = int((ma_signals['buy_score'] > 15).sum())
    ma_sell = int((ma_signals['sell_score'] > 15).sum())
    print(f"    均线买: {ma_buy} | 均线卖: {ma_sell}")

    # 3. 蜡烛图形态
    print("  [3/6] 扫描蜡烛图形态...")
    cs_sell, cs_buy, cs_names = compute_candlestick_scores(main_df)
    signals['cs_sell'] = cs_sell
    signals['cs_buy'] = cs_buy
    signals['cs_names'] = cs_names
    cs_active = int((cs_sell > 0).sum() + (cs_buy > 0).sum())
    print(f"    K线形态信号: {cs_active}个")

    # 4. 布林带
    print("  [4/6] 计算布林带指标...")
    bb_sell, bb_buy, bb_names = compute_bollinger_scores(main_df)
    signals['bb_sell'] = bb_sell
    signals['bb_buy'] = bb_buy
    signals['bb_names'] = bb_names
    print(f"    布林带信号: {int((bb_sell > 15).sum() + (bb_buy > 15).sum())}个")

    # 5. 量价分析
    print("  [5/6] 计算量价指标...")
    vp_sell, vp_buy, vp_names = compute_volume_price_scores(main_df)
    signals['vp_sell'] = vp_sell
    signals['vp_buy'] = vp_buy
    signals['vp_names'] = vp_names
    print(f"    量价信号: {int((vp_sell > 15).sum() + (vp_buy > 15).sum())}个")

    # 6. KDJ波段 ★新增★
    print("  [6/6] 计算KDJ波段信号...")
    kdj_sell, kdj_buy, kdj_names = compute_kdj_scores(main_df)
    signals['kdj_sell'] = kdj_sell
    signals['kdj_buy'] = kdj_buy
    signals['kdj_names'] = kdj_names
    kdj_active = int((kdj_sell > 15).sum() + (kdj_buy > 15).sum())
    print(f"    KDJ信号: {kdj_active}个")

    return signals


# ======================================================
#   六维融合评分
# ======================================================
def calc_fusion_score(signals, data, idx, dt, config):
    """
    六维融合信号计算(升级版)
    返回: sell_score, buy_score, reasons_sell, reasons_buy
    """
    mode = config.get('fusion_mode', 'c6_veto_4')
    main_df = data['1h']
    price = main_df['close'].iloc[idx]

    # --- 获取六个系统的分数 ---
    # 1. 背离
    sig_1h = get_signal_at(signals['div_1h'], dt) or dict(DEFAULT_SIG)
    sig_8h = get_signal_at(signals['div_8h'], dt) or dict(DEFAULT_SIG)
    merged_div = dict(DEFAULT_SIG)
    merged_div['top'] = 0; merged_div['bottom'] = 0
    for sig_src, w in [(sig_1h, 1.0), (sig_8h, 0.5)]:
        merged_div['top'] += sig_src.get('top', 0) * w
        merged_div['bottom'] += sig_src.get('bottom', 0) * w
        for k in DEFAULT_SIG:
            if isinstance(DEFAULT_SIG[k], bool) and sig_src.get(k):
                merged_div[k] = True
            elif isinstance(DEFAULT_SIG[k], (int, float)) and k not in ('top', 'bottom'):
                merged_div[k] = max(merged_div.get(k, 0), sig_src.get(k, 0))

    # 简化趋势判断
    trend = {'is_downtrend': False, 'is_uptrend': False,
             'ma_bearish': False, 'ma_bullish': False,
             'ma_slope_down': False, 'ma_slope_up': False}
    if idx >= 30:
        c5 = main_df['close'].iloc[max(0, idx - 5):idx].mean()
        c20 = main_df['close'].iloc[max(0, idx - 20):idx].mean()
        if c5 < c20 * 0.99: trend['is_downtrend'] = True; trend['ma_bearish'] = True
        elif c5 > c20 * 1.01: trend['is_uptrend'] = True; trend['ma_bullish'] = True

    div_sell, _ = _calc_top_score(merged_div, trend)
    div_buy = _calc_bottom_score(merged_div, trend)

    # 2. 均线
    ma_data = signals['ma']
    ma_sell = float(ma_data['sell_score'].iloc[idx]) if idx < len(ma_data['sell_score']) else 0
    ma_buy = float(ma_data['buy_score'].iloc[idx]) if idx < len(ma_data['buy_score']) else 0

    # 3-6. 其余四个系统
    cs_sell = float(signals['cs_sell'].iloc[idx]) if idx < len(signals['cs_sell']) else 0
    cs_buy = float(signals['cs_buy'].iloc[idx]) if idx < len(signals['cs_buy']) else 0
    bb_sell = float(signals['bb_sell'].iloc[idx]) if idx < len(signals['bb_sell']) else 0
    bb_buy = float(signals['bb_buy'].iloc[idx]) if idx < len(signals['bb_buy']) else 0
    vp_sell = float(signals['vp_sell'].iloc[idx]) if idx < len(signals['vp_sell']) else 0
    vp_buy = float(signals['vp_buy'].iloc[idx]) if idx < len(signals['vp_buy']) else 0
    kdj_sell = float(signals['kdj_sell'].iloc[idx]) if idx < len(signals['kdj_sell']) else 0
    kdj_buy = float(signals['kdj_buy'].iloc[idx]) if idx < len(signals['kdj_buy']) else 0

    r_sell = []
    r_buy = []

    if mode == 'c6_veto_4':
        # === C6底座 + 四书否决权(升级版) ===
        base_sell = div_sell * 0.7 + ma_sell * 0.3
        base_buy = div_buy * 0.7 + ma_buy * 0.3

        if div_sell > 0: r_sell.append(f"背离{div_sell:.0f}")
        if ma_sell > 0: r_sell.append(f"均线{ma_sell:.0f}")
        if div_buy > 0: r_buy.append(f"背离{div_buy:.0f}")
        if ma_buy > 0: r_buy.append(f"均线{ma_buy:.0f}")

        # MA排列加成
        if 'arrangement' in ma_data:
            arr_series = ma_data.get('arrangement', None)
            if arr_series is not None and hasattr(arr_series, 'iloc') and idx < len(arr_series):
                try:
                    arr_val = float(arr_series.iloc[idx])
                    if arr_val < 0: base_sell *= 1.10
                    elif arr_val > 0: base_buy *= 1.10
                except (ValueError, TypeError):
                    pass

        # 否决逻辑: 4个辅助系统, 至少2个强烈反对 → 削弱
        veto_threshold = config.get('veto_threshold', 25)
        sell_vetoes = sum(1 for s in [bb_buy, vp_buy, cs_buy, kdj_buy] if s >= veto_threshold)
        buy_vetoes = sum(1 for s in [bb_sell, vp_sell, cs_sell, kdj_sell] if s >= veto_threshold)

        if sell_vetoes >= 2:
            sell_score = base_sell * 0.3
            r_sell.append(f"被否决({sell_vetoes}/4)")
        else:
            sell_bonus = 0
            if bb_sell >= 15: sell_bonus += 0.10; r_sell.append(f"BB{bb_sell:.0f}")
            if vp_sell >= 15: sell_bonus += 0.08; r_sell.append(f"量价{vp_sell:.0f}")
            if cs_sell >= 25: sell_bonus += 0.06; r_sell.append(f"K线{cs_sell:.0f}")
            if kdj_sell >= 15: sell_bonus += 0.09; r_sell.append(f"KDJ{kdj_sell:.0f}")
            sell_score = base_sell * (1 + sell_bonus)

        if buy_vetoes >= 2:
            buy_score = base_buy * 0.3
            r_buy.append(f"被否决({buy_vetoes}/4)")
        else:
            buy_bonus = 0
            if bb_buy >= 15: buy_bonus += 0.10; r_buy.append(f"BB{bb_buy:.0f}")
            if vp_buy >= 15: buy_bonus += 0.08; r_buy.append(f"量价{vp_buy:.0f}")
            if cs_buy >= 25: buy_bonus += 0.06; r_buy.append(f"K线{cs_buy:.0f}")
            if kdj_buy >= 15: buy_bonus += 0.09; r_buy.append(f"KDJ{kdj_buy:.0f}")
            buy_score = base_buy * (1 + buy_bonus)

    elif mode == 'c6_veto':
        # 兼容: 五书模式(不含KDJ), 用于对比
        base_sell = div_sell * 0.7 + ma_sell * 0.3
        base_buy = div_buy * 0.7 + ma_buy * 0.3

        if div_sell > 0: r_sell.append(f"背离{div_sell:.0f}")
        if ma_sell > 0: r_sell.append(f"均线{ma_sell:.0f}")
        if div_buy > 0: r_buy.append(f"背离{div_buy:.0f}")
        if ma_buy > 0: r_buy.append(f"均线{ma_buy:.0f}")

        veto_threshold = config.get('veto_threshold', 25)
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

    elif mode == 'kdj_weighted':
        # KDJ加权模式: 在基础分上给KDJ更高权重
        base_sell = div_sell * 0.55 + ma_sell * 0.25 + kdj_sell * 0.20
        base_buy = div_buy * 0.55 + ma_buy * 0.25 + kdj_buy * 0.20

        if div_sell > 0: r_sell.append(f"背离{div_sell:.0f}")
        if ma_sell > 0: r_sell.append(f"均线{ma_sell:.0f}")
        if kdj_sell > 0: r_sell.append(f"KDJ{kdj_sell:.0f}")
        if div_buy > 0: r_buy.append(f"背离{div_buy:.0f}")
        if ma_buy > 0: r_buy.append(f"均线{ma_buy:.0f}")
        if kdj_buy > 0: r_buy.append(f"KDJ{kdj_buy:.0f}")

        # 确认加成
        sell_bonus = 0
        if bb_sell >= 15: sell_bonus += 0.08; r_sell.append(f"BB{bb_sell:.0f}")
        if vp_sell >= 15: sell_bonus += 0.06; r_sell.append(f"量价{vp_sell:.0f}")
        if cs_sell >= 25: sell_bonus += 0.05; r_sell.append(f"K线{cs_sell:.0f}")
        sell_score = base_sell * (1 + sell_bonus)

        buy_bonus = 0
        if bb_buy >= 15: buy_bonus += 0.08; r_buy.append(f"BB{bb_buy:.0f}")
        if vp_buy >= 15: buy_bonus += 0.06; r_buy.append(f"量价{vp_buy:.0f}")
        if cs_buy >= 25: buy_bonus += 0.05; r_buy.append(f"K线{cs_buy:.0f}")
        buy_score = base_buy * (1 + buy_bonus)

    elif mode == 'kdj_timing':
        # KDJ择时模式: C6信号基础上, KDJ提供精确入场时机
        base_sell = div_sell * 0.7 + ma_sell * 0.3
        base_buy = div_buy * 0.7 + ma_buy * 0.3

        # KDJ作为择时确认
        kdj_sell_confirm = 1.0
        kdj_buy_confirm = 1.0

        if kdj_sell >= 30:
            kdj_sell_confirm = 1.25  # KDJ强卖 → 大幅增强
            r_sell.append(f"KDJ强卖{kdj_sell:.0f}")
        elif kdj_sell >= 15:
            kdj_sell_confirm = 1.12
            r_sell.append(f"KDJ卖{kdj_sell:.0f}")
        elif kdj_buy >= 25:
            kdj_sell_confirm = 0.7  # KDJ看多 → 削弱卖出
            r_sell.append(f"KDJ反向{kdj_buy:.0f}")

        if kdj_buy >= 30:
            kdj_buy_confirm = 1.25
            r_buy.append(f"KDJ强买{kdj_buy:.0f}")
        elif kdj_buy >= 15:
            kdj_buy_confirm = 1.12
            r_buy.append(f"KDJ买{kdj_buy:.0f}")
        elif kdj_sell >= 25:
            kdj_buy_confirm = 0.7
            r_buy.append(f"KDJ反向{kdj_sell:.0f}")

        sell_score = base_sell * kdj_sell_confirm
        buy_score = base_buy * kdj_buy_confirm

        # 其他系统确认
        if bb_sell >= 15: sell_score *= 1.08; r_sell.append(f"BB{bb_sell:.0f}")
        if vp_sell >= 15: sell_score *= 1.06; r_sell.append(f"量价{vp_sell:.0f}")
        if bb_buy >= 15: buy_score *= 1.08; r_buy.append(f"BB{bb_buy:.0f}")
        if vp_buy >= 15: buy_score *= 1.06; r_buy.append(f"量价{vp_buy:.0f}")

    else:
        # fallback: 简单加权
        sell_score = div_sell * 0.5 + ma_sell * 0.2 + kdj_sell * 0.15 + bb_sell * 0.08 + vp_sell * 0.05 + cs_sell * 0.02
        buy_score = div_buy * 0.5 + ma_buy * 0.2 + kdj_buy * 0.15 + bb_buy * 0.08 + vp_buy * 0.05 + cs_buy * 0.02

    return sell_score, buy_score, r_sell, r_buy


# ======================================================
#   策略执行器
# ======================================================
def run_fusion_strategy(data, signals, config, trade_days=None):
    """六书融合策略回测"""
    eng = FuturesEngine(config['name'], max_leverage=config.get('max_lev', 5))
    main_df = data['1h']

    start_dt = None
    if trade_days and trade_days < 60:
        start_dt = main_df.index[-1] - pd.Timedelta(days=trade_days)

    init_idx = 0
    if start_dt:
        init_idx = main_df.index.searchsorted(start_dt)
        if init_idx >= len(main_df): init_idx = 0
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[init_idx]

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
    long_max_hold = config.get('long_max_hold', 120)
    trail_pullback = config.get('trail_pullback', 0.60)

    # 分段止盈参数
    use_partial_tp = config.get('use_partial_tp', False)
    partial_tp_1 = config.get('partial_tp_1', 0.20)
    partial_tp_1_pct = config.get('partial_tp_1_pct', 0.30)

    short_bars = 0; long_bars = 0
    short_max_pnl = 0; long_max_pnl = 0
    last_short_close = 0; last_long_close = 0
    last_spot_sell = 0; last_spot_buy = 0
    short_partial_done = False; long_partial_done = False

    for i in range(max(30, init_idx), len(main_df)):
        dt = main_df.index[i]
        price = main_df['close'].iloc[i]

        ss, bs, r_s, r_b = calc_fusion_score(signals, data, i, dt, config)

        short_just_opened = False
        long_just_opened = False

        # ====== 管理空仓 ======
        if eng.futures_short and not short_just_opened:
            short_bars += 1
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin

            # 分段止盈
            if use_partial_tp and not short_partial_done and pnl_r >= partial_tp_1:
                old_qty = eng.futures_short.quantity
                partial_qty = old_qty * partial_tp_1_pct
                partial_pnl = (eng.futures_short.entry_price - price) * partial_qty
                eng.usdt += eng.futures_short.margin * partial_tp_1_pct + partial_pnl
                fee = partial_qty * price * FuturesEngine.TAKER_FEE
                eng.usdt -= fee
                eng.total_futures_fees += fee
                eng.futures_short.quantity = old_qty - partial_qty
                eng.futures_short.margin *= (1 - partial_tp_1_pct)
                short_partial_done = True

            short_max_pnl = max(short_max_pnl, pnl_r)

            if pnl_r >= short_tp:
                eng.close_short(price, dt, f"止盈 +{pnl_r*100:.0f}%")
            elif short_max_pnl >= short_trail:
                if pnl_r < short_max_pnl * trail_pullback:
                    eng.close_short(price, dt, "追踪止盈")
            elif eng.futures_short and bs >= config.get('close_short_bs', 40):
                eng.close_short(price, dt, "反向信号平仓")
            elif pnl_r < short_sl:
                eng.close_short(price, dt, f"止损 {pnl_r*100:.0f}%")
            elif short_bars >= short_max_hold:
                eng.close_short(price, dt, "超时")

            if not eng.futures_short:
                short_bars = 0; short_max_pnl = 0
                last_short_close = i; short_partial_done = False

        # ====== 管理多仓 ======
        if eng.futures_long and not long_just_opened:
            long_bars += 1
            pnl_r = eng.futures_long.calc_pnl(price) / eng.futures_long.margin

            if use_partial_tp and not long_partial_done and pnl_r >= partial_tp_1:
                old_qty = eng.futures_long.quantity
                partial_qty = old_qty * partial_tp_1_pct
                partial_pnl = (price - eng.futures_long.entry_price) * partial_qty
                eng.usdt += eng.futures_long.margin * partial_tp_1_pct + partial_pnl
                fee = partial_qty * price * FuturesEngine.TAKER_FEE
                eng.usdt -= fee
                eng.total_futures_fees += fee
                eng.futures_long.quantity = old_qty - partial_qty
                eng.futures_long.margin *= (1 - partial_tp_1_pct)
                long_partial_done = True

            long_max_pnl = max(long_max_pnl, pnl_r)

            if pnl_r >= long_tp:
                eng.close_long(price, dt, f"止盈 +{pnl_r*100:.0f}%")
            elif long_max_pnl >= long_trail:
                if pnl_r < long_max_pnl * trail_pullback:
                    eng.close_long(price, dt, "追踪止盈")
            elif eng.futures_long and ss >= config.get('close_long_ss', 40):
                eng.close_long(price, dt, "反向信号平仓")
            elif pnl_r < long_sl:
                eng.close_long(price, dt, f"止损 {pnl_r*100:.0f}%")
            elif long_bars >= long_max_hold:
                eng.close_long(price, dt, "超时")

            if not eng.futures_long:
                long_bars = 0; long_max_pnl = 0
                last_long_close = i; long_partial_done = False

        # ====== 现货交易 ======
        if ss >= sell_threshold and not eng.futures_short and (i - last_spot_sell >= spot_cooldown):
            if eng.spot_eth * price > 200:
                eng.spot_sell(price, dt, sell_pct, f"六书卖 {' '.join(r_s[:3])}")
                last_spot_sell = i

        if bs >= buy_threshold and (i - last_spot_buy >= spot_cooldown):
            buy_budget = eng.available_usdt() * 0.25
            if buy_budget > 200:
                eng.spot_buy(price, dt, buy_budget, f"六书买 {' '.join(r_b[:3])}")
                last_spot_buy = i

        # ====== 开空仓 ======
        if ss >= short_threshold and not eng.futures_short and (i - last_short_close >= cooldown):
            avail = eng.usdt * margin_use
            margin = min(avail, eng.max_single_margin)
            if margin > 50:
                eng.open_short(price, dt, margin, lev, ' '.join(r_s[:3]))
                short_just_opened = True
                short_bars = 0; short_max_pnl = 0; short_partial_done = False

        # ====== 开多仓 ======
        if bs >= long_threshold and not eng.futures_long and (i - last_long_close >= cooldown):
            avail = eng.usdt * margin_use
            margin = min(avail, eng.max_single_margin)
            if margin > 50:
                eng.open_long(price, dt, margin, lev, ' '.join(r_b[:3]))
                long_just_opened = True
                long_bars = 0; long_max_pnl = 0; long_partial_done = False

        # 资金费率 (每8小时)
        if i % 8 == 0:
            eng.charge_funding(price, dt)

    # 结算
    final_price = main_df['close'].iloc[-1]
    final_dt = main_df.index[-1]
    if eng.futures_short: eng.close_short(final_price, final_dt, "结算")
    if eng.futures_long: eng.close_long(final_price, final_dt, "结算")

    # 构建交易数据子集
    if start_dt:
        trade_df = main_df.loc[start_dt:]
        if len(trade_df) > 1:
            return eng.get_result(trade_df)
    return eng.get_result(main_df)


# ======================================================
#   策略配置(对比: 五书 vs 六书)
# ======================================================
def get_strategies():
    base = {
        'sell_threshold': 18, 'short_threshold': 25,
        'buy_threshold': 25, 'long_threshold': 40,
        'lev': 5, 'max_lev': 5, 'margin_use': 0.70,
        'single_pct': 0.20, 'total_pct': 0.50,
        'sell_pct': 0.55, 'short_sl': -0.25, 'short_tp': 0.60,
        'long_sl': -0.08, 'long_tp': 0.30,
        'short_trail': 0.25, 'short_max_hold': 72,
        'long_trail': 0.20, 'long_max_hold': 120,
        'trail_pullback': 0.60,
        'cooldown': 4, 'spot_cooldown': 12,
        'close_short_bs': 40, 'close_long_ss': 40,
        'use_partial_tp': True,
        'partial_tp_1': 0.20, 'partial_tp_1_pct': 0.30,
    }

    return [
        # S1: 五书基准(不含KDJ) — 用于对比
        {**base, 'name': 'S1: 五书基准(C6+三书否决)',
         'fusion_mode': 'c6_veto',
         'veto_threshold': 25},

        # S2: 六书C6+四书否决 — 核心升级
        {**base, 'name': 'S2: 六书(C6+四书否决)',
         'fusion_mode': 'c6_veto_4',
         'veto_threshold': 25},

        # S3: 六书C6+四书否决(低阈值)
        {**base, 'name': 'S3: 六书(低否决阈值)',
         'fusion_mode': 'c6_veto_4',
         'veto_threshold': 20},

        # S4: KDJ加权模式
        {**base, 'name': 'S4: KDJ加权',
         'fusion_mode': 'kdj_weighted'},

        # S5: KDJ择时模式
        {**base, 'name': 'S5: KDJ择时',
         'fusion_mode': 'kdj_timing'},

        # S6: 六书否决+无分段止盈(消融测试)
        {**base, 'name': 'S6: 六书否决(无分段TP)',
         'fusion_mode': 'c6_veto_4',
         'veto_threshold': 25,
         'use_partial_tp': False,
         'short_sl': -0.30, 'short_tp': 0.80,
         'long_sl': -0.15, 'long_tp': 0.50},

        # S7: 六书否决+高阈值(保守)
        {**base, 'name': 'S7: 六书否决(高阈值)',
         'fusion_mode': 'c6_veto_4',
         'veto_threshold': 30,
         'sell_threshold': 22, 'short_threshold': 30,
         'buy_threshold': 30, 'long_threshold': 45},

        # S8: 六书否决+激进做空
        {**base, 'name': 'S8: 六书否决(激进做空)',
         'fusion_mode': 'c6_veto_4',
         'veto_threshold': 25,
         'sell_threshold': 15, 'short_threshold': 20,
         'lev': 5, 'margin_use': 0.80,
         'sell_pct': 0.60},
    ]


# ======================================================
#   主函数
# ======================================================
def main(trade_days=None):
    if trade_days is None:
        trade_days = 30

    print("=" * 80)
    print("  六书融合策略回测")
    print("  背离×均线×蜡烛图×布林带×量价×KDJ — 六维信号融合")
    print("=" * 80)

    data = fetch_data()
    if '1h' not in data:
        print("无法获取1h数据"); return

    print(f"\n{'='*60}")
    print(f"  计算六维信号...")
    print(f"{'='*60}")
    signals = compute_all_signals(data)

    strategies = get_strategies()
    results = []

    for cfg in strategies:
        t0 = time.time()
        result = run_fusion_strategy(data, signals, cfg, trade_days=trade_days)
        elapsed = time.time() - t0
        results.append(result)
        alpha = result.get('alpha', 0)
        ret = result.get('strategy_return', 0)
        trades = result.get('total_trades', 0)
        alpha_str = f"+{alpha:.2f}%" if alpha >= 0 else f"{alpha:.2f}%"
        print(f"  {cfg['name'][:35]:35s} | α={alpha_str:>10s} | 收益={ret:>8.2f}% | 交易={trades:>3d} | {elapsed:.1f}s")

    # 排序
    results.sort(key=lambda x: x.get('alpha', -999), reverse=True)

    print(f"\n{'='*80}")
    print(f"  排名结果")
    print(f"{'='*80}")
    for rank, r in enumerate(results, 1):
        alpha = r.get('alpha', 0)
        name = r.get('name', '')
        print(f"  #{rank} {name[:40]:40s} α = {alpha:+.2f}%")

    # 对比五书 vs 六书
    five_book_alpha = None
    six_book_alpha = None
    best_six = None
    for r in results:
        name = r.get('name', '')
        alpha = r.get('alpha', 0)
        if '五书基准' in name:
            five_book_alpha = alpha
        if '六书' in name and (best_six is None or alpha > best_six.get('alpha', -999)):
            best_six = r
    
    if best_six:
        six_book_alpha = best_six.get('alpha', 0)

    print(f"\n{'='*80}")
    if five_book_alpha is not None and six_book_alpha is not None:
        improvement = six_book_alpha - five_book_alpha
        print(f"  五书基准 α = {five_book_alpha:+.2f}%")
        print(f"  六书最佳 α = {six_book_alpha:+.2f}% ({best_six.get('name', '')})")
        print(f"  提升: {improvement:+.2f}%")
    print(f"{'='*80}")

    # 保存
    output = {
        'description': '六书融合策略 — 背离×均线×蜡烛图×布林带×量价×KDJ',
        'run_time': datetime.now().isoformat(),
        'trade_days': trade_days,
        'strategies': results,
        'best_strategy': results[0] if results else None,
        'five_book_alpha': five_book_alpha,
        'six_book_alpha': six_book_alpha,
        'improvement': (six_book_alpha - five_book_alpha) if five_book_alpha is not None and six_book_alpha is not None else None,
    }

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'six_book_fusion_result.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果已保存: {path}")


if __name__ == '__main__':
    trade_days = 30
    if len(sys.argv) > 1:
        trade_days = int(sys.argv[1])
    main(trade_days)
