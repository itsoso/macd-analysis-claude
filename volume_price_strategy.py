"""
量价分析策略 — 基于利弗莫尔/威科夫(Wyckoff)量价分析理论

核心理论(利弗莫尔/威科夫):
  1. 量价关系: 成交量确认价格趋势
     - 价涨量增 = 健康上涨    价跌量增 = 健康下跌
     - 价涨量缩 = 上涨乏力    价跌量缩 = 下跌乏力
  2. 量价背离: 价格新高但量能下降 = 见顶
  3. 放量反转: 在趋势末端出现巨量+反转 = 高潮(Climax)
  4. 缩量盘整 → 放量突破: 蓄势后爆发
  5. OBV(能量潮): 累积成交量方向判断资金流
  6. 利弗莫尔关键点: 等待确认,在关键位置加仓

核心指标:
  OBV: On-Balance Volume (累积成交量)
  VWAP: 成交量加权平均价
  Volume Ratio: 成交量相对倍数
  A/D Line: 累积/派发线
  MFI: 资金流量指数(Money Flow Index)

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
#   量价指标计算
# ======================================================
def compute_volume_indicators(df):
    """计算全套量价指标"""
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # === 1. OBV (能量潮) ===
    obv = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]
    df['obv'] = obv
    df['obv_sma20'] = obv.rolling(20, min_periods=5).mean()
    df['obv_sma5'] = obv.rolling(5, min_periods=3).mean()

    # === 2. Volume SMA ===
    df['vol_sma20'] = volume.rolling(20, min_periods=5).mean()
    df['vol_sma5'] = volume.rolling(5, min_periods=3).mean()
    df['vol_ratio'] = volume / df['vol_sma20'].replace(0, np.nan)  # 量比

    # === 3. VWAP (成交量加权平均价) - 滚动20周期 ===
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).rolling(20, min_periods=5).sum()
    cum_vol = volume.rolling(20, min_periods=5).sum()
    df['vwap'] = cum_tp_vol / cum_vol.replace(0, np.nan)

    # === 4. A/D Line (累积/派发线) ===
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    clv = clv.fillna(0)
    ad = (clv * volume).cumsum()
    df['ad_line'] = ad
    df['ad_sma20'] = ad.rolling(20, min_periods=5).mean()

    # === 5. MFI (资金流量指数, 14周期) ===
    mfi_period = 14
    tp = typical_price
    raw_mf = tp * volume
    pos_mf = pd.Series(0.0, index=df.index)
    neg_mf = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        if tp.iloc[i] > tp.iloc[i - 1]:
            pos_mf.iloc[i] = raw_mf.iloc[i]
        elif tp.iloc[i] < tp.iloc[i - 1]:
            neg_mf.iloc[i] = raw_mf.iloc[i]

    pos_sum = pos_mf.rolling(mfi_period, min_periods=mfi_period).sum()
    neg_sum = neg_mf.rolling(mfi_period, min_periods=mfi_period).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    df['mfi'] = 100 - (100 / (1 + mfr))

    # === 6. 价格变化率 ===
    df['price_change'] = close.pct_change()
    df['price_change_5'] = close.pct_change(5)

    # === 7. 量价相关性(滚动20期) ===
    df['vol_price_corr'] = close.rolling(20, min_periods=10).corr(volume)

    return df


# ======================================================
#   量价形态检测
# ======================================================
def detect_volume_climax(df, i, lookback=50):
    """放量高潮: 成交量突然放大到平均的3倍+"""
    if i < lookback:
        return None
    vol_ratio = df['vol_ratio'].iloc[i]
    if pd.isna(vol_ratio):
        return None

    if vol_ratio >= 3.0:
        price_change = df['price_change'].iloc[i]
        if pd.isna(price_change):
            return None

        if price_change > 0.02:
            # 放量上涨 — 可能是买入高潮(见顶)
            return {'type': 'buying_climax', 'score': -35,
                    'reason': f'买入高潮(量比{vol_ratio:.1f}x)'}
        elif price_change < -0.02:
            # 放量下跌 — 可能是卖出高潮(见底)
            return {'type': 'selling_climax', 'score': 35,
                    'reason': f'卖出高潮(量比{vol_ratio:.1f}x)'}
    return None


def detect_volume_divergence(df, i, lookback=20):
    """量价背离: 价格创新高/低但成交量下降"""
    if i < lookback + 5:
        return None

    price_now = df['close'].iloc[i]
    vol_now = df['vol_sma5'].iloc[i]
    if pd.isna(vol_now):
        return None

    # 回看区间
    prices = df['close'].iloc[max(0, i - lookback):i]
    vols = df['vol_sma5'].iloc[max(0, i - lookback):i]

    if len(prices) < 10 or len(vols.dropna()) < 10:
        return None

    price_max = prices.max()
    price_min = prices.min()
    vol_at_max = vols.iloc[prices.values.argmax()] if not pd.isna(vols.iloc[prices.values.argmax()]) else None
    vol_at_min = vols.iloc[prices.values.argmin()] if not pd.isna(vols.iloc[prices.values.argmin()]) else None

    # 顶背离: 价格接近或超过前高, 但量能下降
    if price_now >= price_max * 0.98 and vol_at_max is not None:
        if vol_now < vol_at_max * 0.7:
            return {'type': 'bearish_divergence', 'score': -30,
                    'reason': f'量价顶背离(量降{(1-vol_now/vol_at_max)*100:.0f}%)'}

    # 底背离: 价格接近或低于前低, 但量能下降(卖压减弱)
    if price_now <= price_min * 1.02 and vol_at_min is not None:
        if vol_now < vol_at_min * 0.7:
            return {'type': 'bullish_divergence', 'score': 30,
                    'reason': f'量价底背离(量降{(1-vol_now/vol_at_min)*100:.0f}%)'}

    return None


def detect_volume_breakout(df, i, lookback=20):
    """缩量蓄势 → 放量突破"""
    if i < lookback + 5:
        return None

    # 最近5期平均量比
    recent_ratio = df['vol_ratio'].iloc[max(0, i - 2):i + 1].mean()
    # 前面的量比(蓄势期)
    prior_ratio = df['vol_ratio'].iloc[max(0, i - lookback):max(0, i - 3)].mean()

    if pd.isna(recent_ratio) or pd.isna(prior_ratio):
        return None

    # 蓄势: 前期缩量(量比<0.7), 当前放量(量比>1.5)
    if prior_ratio < 0.7 and recent_ratio > 1.5:
        price_change_5 = df['price_change_5'].iloc[i]
        if pd.isna(price_change_5):
            return None

        if price_change_5 > 0.02:
            return {'type': 'volume_breakout_up', 'score': 40,
                    'reason': f'缩量蓄势→放量突破上行'}
        elif price_change_5 < -0.02:
            return {'type': 'volume_breakout_down', 'score': -40,
                    'reason': f'缩量蓄势→放量突破下行'}
    return None


def detect_obv_divergence(df, i, lookback=20):
    """OBV背离"""
    if i < lookback + 5:
        return None

    obv_now = df['obv'].iloc[i]
    price_now = df['close'].iloc[i]
    obv_sma = df['obv_sma20'].iloc[i]

    if pd.isna(obv_sma):
        return None

    # OBV斜率 vs 价格斜率
    obv_5_ago = df['obv'].iloc[max(0, i - 5)]
    price_5_ago = df['close'].iloc[max(0, i - 5)]

    if price_5_ago == 0:
        return None

    price_slope = (price_now - price_5_ago) / price_5_ago
    obv_range = df['obv'].iloc[max(0, i - 20):i + 1]
    obv_std = obv_range.std()
    if obv_std == 0:
        return None
    obv_slope = (obv_now - obv_5_ago) / obv_std

    # 价格上涨但OBV下降 = 看跌
    if price_slope > 0.01 and obv_slope < -0.3:
        return {'type': 'obv_bearish_div', 'score': -25,
                'reason': 'OBV背离(价涨量退)'}

    # 价格下跌但OBV上升 = 看涨
    if price_slope < -0.01 and obv_slope > 0.3:
        return {'type': 'obv_bullish_div', 'score': 25,
                'reason': 'OBV背离(价跌量增)'}

    return None


# ======================================================
#   量价信号综合评分
# ======================================================
def compute_volume_price_scores(df):
    """计算每根K线的量价综合得分"""
    df = compute_volume_indicators(df)

    sell_scores = pd.Series(0.0, index=df.index)
    buy_scores = pd.Series(0.0, index=df.index)
    signal_names = pd.Series('', index=df.index, dtype=str)

    for i in range(25, len(df)):
        ss = 0; bs = 0
        reasons = []

        vol_ratio = df['vol_ratio'].iloc[i]
        mfi = df['mfi'].iloc[i]
        vwap = df['vwap'].iloc[i]
        close = df['close'].iloc[i]
        obv = df['obv'].iloc[i]
        obv_sma = df['obv_sma20'].iloc[i]
        ad = df['ad_line'].iloc[i]
        ad_sma = df['ad_sma20'].iloc[i]
        corr = df['vol_price_corr'].iloc[i]
        price_chg = df['price_change'].iloc[i]

        # === 1. MFI 超买/超卖 ===
        if not pd.isna(mfi):
            if mfi > 80:
                ss += 20
                reasons.append(f'MFI超买({mfi:.0f})')
            elif mfi < 20:
                bs += 20
                reasons.append(f'MFI超卖({mfi:.0f})')
            elif mfi > 70:
                ss += 10
                reasons.append(f'MFI偏高({mfi:.0f})')
            elif mfi < 30:
                bs += 10
                reasons.append(f'MFI偏低({mfi:.0f})')

        # === 2. VWAP 关系 ===
        if not pd.isna(vwap):
            vwap_dist = (close - vwap) / vwap * 100
            if vwap_dist > 2:
                ss += 10
                reasons.append(f'高于VWAP({vwap_dist:.1f}%)')
            elif vwap_dist < -2:
                bs += 10
                reasons.append(f'低于VWAP({vwap_dist:.1f}%)')

        # === 3. 量价关系(基本) ===
        if not pd.isna(vol_ratio) and not pd.isna(price_chg):
            if vol_ratio > 1.5:
                if price_chg > 0.01:
                    # 放量上涨 - 趋势确认(但若MFI已超买则警惕)
                    if not pd.isna(mfi) and mfi > 70:
                        ss += 10
                        reasons.append('放量上涨(高位警惕)')
                    else:
                        bs += 15
                        reasons.append(f'放量上涨({vol_ratio:.1f}x)')
                elif price_chg < -0.01:
                    if not pd.isna(mfi) and mfi < 30:
                        bs += 10
                        reasons.append('放量下跌(低位企稳)')
                    else:
                        ss += 15
                        reasons.append(f'放量下跌({vol_ratio:.1f}x)')
            elif vol_ratio < 0.5:
                if price_chg > 0.01:
                    ss += 10
                    reasons.append('缩量上涨(量能不足)')
                elif price_chg < -0.01:
                    bs += 10
                    reasons.append('缩量下跌(卖压减弱)')

        # === 4. OBV 趋势 ===
        if not pd.isna(obv_sma):
            if obv > obv_sma * 1.05:
                bs += 10
                reasons.append('OBV在均线上方')
            elif obv < obv_sma * 0.95:
                ss += 10
                reasons.append('OBV在均线下方')

        # === 5. A/D线 ===
        if not pd.isna(ad_sma):
            if ad > ad_sma:
                bs += 8
            elif ad < ad_sma:
                ss += 8

        # === 6. 量价高潮 ===
        climax = detect_volume_climax(df, i)
        if climax:
            if climax['score'] > 0:
                bs += abs(climax['score'])
            else:
                ss += abs(climax['score'])
            reasons.append(climax['reason'])

        # === 7. 量价背离 ===
        div = detect_volume_divergence(df, i)
        if div:
            if div['score'] > 0:
                bs += abs(div['score'])
            else:
                ss += abs(div['score'])
            reasons.append(div['reason'])

        # === 8. 缩量→放量突破 ===
        brk = detect_volume_breakout(df, i)
        if brk:
            if brk['score'] > 0:
                bs += abs(brk['score'])
            else:
                ss += abs(brk['score'])
            reasons.append(brk['reason'])

        # === 9. OBV背离 ===
        obv_div = detect_obv_divergence(df, i)
        if obv_div:
            if obv_div['score'] > 0:
                bs += abs(obv_div['score'])
            else:
                ss += abs(obv_div['score'])
            reasons.append(obv_div['reason'])

        # === 10. 量价相关性 ===
        if not pd.isna(corr):
            if corr < -0.5:
                # 负相关: 价格和成交量反向 = 警示
                ss += 10
                reasons.append(f'量价负相关({corr:.2f})')

        sell_scores.iloc[i] = min(ss, 100)
        buy_scores.iloc[i] = min(bs, 100)
        signal_names.iloc[i] = ','.join(reasons[:3])

    return sell_scores, buy_scores, signal_names


# ======================================================
#   量价策略回测
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


def run_volume_price_strategy(data, config, trade_days=None):
    """量价策略回测"""
    eng = FuturesEngine(config['name'], max_leverage=config.get('max_lev', 5))
    main_df = data['1h']

    sell_sc, buy_sc, sig_names = compute_volume_price_scores(main_df)

    # 4h辅助
    sell_sc_4h = pd.Series(0.0, index=main_df.index)
    buy_sc_4h = pd.Series(0.0, index=main_df.index)
    if '4h' in data:
        s4, b4, _ = compute_volume_price_scores(data['4h'])
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
            eng.spot_sell(price, dt, sell_pct, f"量价卖 {reason}")
            spot_cd = spot_cooldown

        # 开空
        sell_dom = (ss > bs * 1.5) if bs > 0 else True
        if short_cd == 0 and ss >= short_threshold and not eng.futures_short and sell_dom:
            margin = eng.available_margin() * margin_use
            actual_lev = min(lev if ss >= 60 else 2, eng.max_leverage)
            eng.open_short(price, dt, margin, actual_lev, f"量价空 {actual_lev}x {reason}")
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
            eng.spot_buy(price, dt, eng.available_usdt() * 0.25, f"量价买 {reason}")
            spot_cd = spot_cooldown

        # 开多
        buy_dom = (bs > ss * 1.5) if ss > 0 else True
        if long_cd == 0 and bs >= long_threshold and not eng.futures_long and buy_dom:
            margin = eng.available_margin() * margin_use
            actual_lev = min(lev if bs >= 60 else 2, eng.max_leverage)
            eng.open_long(price, dt, margin, actual_lev, f"量价多 {actual_lev}x {reason}")
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
        {**base, 'name': 'V1: 标准量价'},

        {**base, 'name': 'V2: 激进做空',
         'sell_threshold': 20, 'short_threshold': 30,
         'lev': 5, 'margin_use': 0.70, 'sell_pct': 0.55,
         'short_sl': -0.30, 'short_tp': 0.80},

        {**base, 'name': 'V3: 高潮捕捉',
         'sell_threshold': 25, 'buy_threshold': 25,
         'short_threshold': 35, 'long_threshold': 35,
         'lev': 4, 'margin_use': 0.60,
         'short_max_hold': 48, 'long_max_hold': 48,
         'cooldown': 2, 'spot_cooldown': 8},

        {**base, 'name': 'V4: 保守量价',
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

    print("\n计算量价指标...")
    main_df = compute_volume_indicators(main_df)

    # 统计
    climax_count = 0
    div_count = 0
    brk_count = 0
    for i in range(50, len(main_df)):
        if detect_volume_climax(main_df, i): climax_count += 1
        if detect_volume_divergence(main_df, i): div_count += 1
        if detect_volume_breakout(main_df, i): brk_count += 1
    print(f"  量价高潮: {climax_count}次 | 量价背离: {div_count}次 | 放量突破: {brk_count}次")

    strategies = get_strategies()
    all_results = []

    start_dt = main_df.index[-1] - pd.Timedelta(days=trade_days)
    trade_start = str(start_dt)[:16]
    trade_end = str(main_df.index[-1])[:16]

    print(f"\n{'=' * 100}")
    print(f"  量价分析策略 · 利弗莫尔/威科夫 · {len(strategies)}种 · 最近{trade_days}天")
    print(f"  数据: {len(main_df)}根1h K线 | 交易: {trade_start} ~ {trade_end}")
    print(f"{'=' * 100}")

    print(f"\n{'策略':<20} {'α':>8} {'收益':>10} {'BH':>10} {'回撤':>8} {'交易':>6} {'费用':>10}")
    print('-' * 100)

    for cfg in strategies:
        r = run_volume_price_strategy(data, cfg, trade_days=trade_days)
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
        'description': f'量价分析策略 · 最近{trade_days}天',
        'book': '利弗莫尔/威科夫量价分析',
        'run_time': datetime.now().isoformat(),
        'trade_days': trade_days,
        'climax_count': climax_count,
        'divergence_count': div_count,
        'breakout_count': brk_count,
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
                        'volume_price_result.json')
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
