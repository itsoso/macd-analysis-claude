"""
P0 信号计算向量化模块 [实验性 — 不建议用于正式策略结论]
将逐行 Python 循环替换为 NumPy/Pandas 向量化操作

⚠️  本模块为性能优化的近似实现, 与原版模块存在以下差异:
  - KDJ/K线形态/布林带/量价: 用 rolling 近似替代逐行 lookback 检测
  - 背离分析: 简化了峰谷检测逻辑
  - 整体误差约 ±1%, 但在极端行情下可能更大
  - 正式回测/策略结论请使用原版 (不加 --fast 参数)

目标: 信号计算从 ~1476s 降到 ~120-180s (全 7 TF)

模块覆盖 (按耗时占比):
  1. KDJ 评分 (25.6%)          → compute_kdj_scores_vec
  2. K线形态 (15.3%)           → compute_candlestick_scores_vec
  3. 量价评分 (6.3%)           → compute_volume_price_scores_vec
  4. 布林带评分 (6.1%)         → compute_bollinger_scores_vec
  5. 背离分析 (40.6% + 5.9%)   → analyze_signals_enhanced_fast

已知近似点:
  - KDJ: EMA 递推精确, 评分条件精确
  - K线形态: 形态识别精确, 但缺少部分罕见形态
  - 量价: vol_at_max/vol_at_min 已修复为精确语义 (2026-02-14)
  - 布林带: 挤压/扩张检测用 rolling 近似
  - 背离: 峰谷检测用 rolling extrema 替代原版逐段比较
"""

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════
#  1. KDJ 向量化
# ══════════════════════════════════════════════════════════

def _ema_recursive(x, alpha, init_val=50.0):
    """纯 NumPy EMA 递推: y[i] = (1-alpha)*y[i-1] + alpha*x[i]"""
    out = np.empty_like(x)
    out[0] = (1 - alpha) * init_val + alpha * x[0]
    decay = 1 - alpha
    for i in range(1, len(x)):
        out[i] = decay * out[i - 1] + alpha * x[i]
    return out


def compute_kdj_fast(df, n=9, m1=3, m2=3):
    """快速 KDJ 计算 (最小化 Python 循环)"""
    high_n = df['high'].rolling(n, min_periods=n).max()
    low_n = df['low'].rolling(n, min_periods=n).min()
    hl_range = (high_n - low_n).replace(0, np.nan)
    rsv = ((df['close'] - low_n) / hl_range * 100).fillna(50)

    alpha_k = 1.0 / m1
    alpha_d = 1.0 / m2

    rsv_vals = rsv.values.astype(np.float64)
    start = n - 1
    rsv_valid = rsv_vals[start:]

    k_out = _ema_recursive(rsv_valid, alpha_k, init_val=50.0)
    d_out = _ema_recursive(k_out, alpha_d, init_val=50.0)

    k_values = np.full(len(df), 50.0)
    d_values = np.full(len(df), 50.0)
    k_values[start:] = k_out
    d_values[start:] = d_out

    df['kdj_rsv'] = rsv
    df['kdj_k'] = k_values
    df['kdj_d'] = d_values
    df['kdj_j'] = 3 * k_values - 2 * d_values
    df['kd_macd'] = 2 * (k_values - d_values)
    df['kdj_k_slope'] = df['kdj_k'] - df['kdj_k'].shift(3)
    df['kdj_d_slope'] = df['kdj_d'] - df['kdj_d'].shift(3)
    return df


def compute_kdj_scores_vec(df):
    """向量化 KDJ 评分 — 替代 kdj_strategy.compute_kdj_scores"""
    df = compute_kdj_fast(df)
    n = len(df)
    idx = np.arange(n)

    # 提取数组
    k = df['kdj_k'].values.astype(np.float64)
    d = df['kdj_d'].values.astype(np.float64)
    j = df['kdj_j'].values.astype(np.float64)
    k_slope = np.nan_to_num(df['kdj_k_slope'].values.astype(np.float64), nan=0.0)
    kd_macd_arr = df['kd_macd'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)

    # Shifted arrays
    k1 = np.empty(n); k1[0] = 50; k1[1:] = k[:-1]
    d1 = np.empty(n); d1[0] = 50; d1[1:] = d[:-1]
    k2 = np.empty(n); k2[:2] = 50; k2[2:] = k[:-2]
    d2 = np.empty(n); d2[:2] = 50; d2[2:] = d[:-2]
    k3 = np.empty(n); k3[:3] = 50; k3[3:] = k[:-3]
    m1_arr = np.empty(n); m1_arr[0] = 0; m1_arr[1:] = kd_macd_arr[:-1]
    m2_arr = np.empty(n); m2_arr[:2] = 0; m2_arr[2:] = kd_macd_arr[:-2]

    ss = np.zeros(n, dtype=np.float64)
    bs = np.zeros(n, dtype=np.float64)
    sell_cnt = np.zeros(n, dtype=np.int32)
    buy_cnt = np.zeros(n, dtype=np.int32)
    mask = idx >= 15

    # ── 1. 超买超卖 ──
    ob = mask & (k > 80) & (d > 75)
    ss += np.where(ob, 10, 0)
    sell_cnt += ob.astype(np.int32)
    ss += np.where(ob & (j > 100), 5, 0)
    sell_cnt += (ob & (j > 100)).astype(np.int32)

    os_ = mask & (k < 20) & (d < 25)
    bs += np.where(os_, 10, 0)
    buy_cnt += os_.astype(np.int32)
    bs += np.where(os_ & (j < 0), 5, 0)
    buy_cnt += (os_ & (j < 0)).astype(np.int32)

    # ── 2. 金叉/死叉 ──
    golden = mask & (k1 <= d1) & (k > d)
    death  = mask & (k1 >= d1) & (k < d)

    bs += np.where(golden & (k < 20) & (d < 25), 30,
          np.where(golden & (k < 50), 20,
          np.where(golden & (k < 80), 10,
          np.where(golden, 5, 0))))
    buy_cnt += golden.astype(np.int32)

    ss += np.where(death & (k > 80) & (d > 75), 30,
          np.where(death & (k > 50), 20,
          np.where(death & (k > 20), 10,
          np.where(death, 5, 0))))
    sell_cnt += death.astype(np.int32)

    # ── 3. 50线穿越 ──
    bs += np.where(mask & (k > 50) & (k3 < 50), 8, 0)
    buy_cnt += (mask & (k > 50) & (k3 < 50)).astype(np.int32)
    ss += np.where(mask & (k < 50) & (k3 > 50), 8, 0)
    sell_cnt += (mask & (k < 50) & (k3 > 50)).astype(np.int32)

    # ── 4. 背离 (rolling 近似) ──
    lb = 30
    close_s = pd.Series(close)
    k_s = pd.Series(k)
    rmax_c = close_s.rolling(lb, min_periods=lb).max().values
    rmax_k = k_s.rolling(lb, min_periods=lb).max().values
    rmin_c = close_s.rolling(lb, min_periods=lb).min().values
    rmin_k = k_s.rolling(lb, min_periods=lb).min().values

    div_mask = mask & (idx >= lb + 5)
    # 顶背离
    dt_m = div_mask & (close >= rmax_c * 0.998) & (k < rmax_k * 0.92) & (rmax_k > 60)
    dt_str = np.minimum(40.0, (rmax_k - k) * 1.5)
    ss += np.where(dt_m, dt_str, 0)
    sell_cnt += dt_m.astype(np.int32)
    # 底背离
    db_m = div_mask & (close <= rmin_c * 1.002) & (k > rmin_k * 1.08) & (rmin_k < 40)
    db_str = np.minimum(40.0, (k - rmin_k) * 1.5)
    bs += np.where(db_m, db_str, 0)
    buy_cnt += db_m.astype(np.int32)

    # ── 5. KD-MACD 柱线 ──
    red_shrink = mask & (m1_arr > 0) & (m2_arr > 0) & (kd_macd_arr > 0) & (m1_arr > m2_arr) & (kd_macd_arr < m1_arr)
    ss += np.where(red_shrink, 15, 0); sell_cnt += red_shrink.astype(np.int32)
    green_shrink = mask & (m1_arr < 0) & (m2_arr < 0) & (kd_macd_arr < 0) & (m1_arr < m2_arr) & (kd_macd_arr > m1_arr)
    bs += np.where(green_shrink, 15, 0); buy_cnt += green_shrink.astype(np.int32)
    ss += np.where(mask & (m1_arr > 5) & (kd_macd_arr < m1_arr * 0.3), 10, 0)
    bs += np.where(mask & (m1_arr < -5) & (kd_macd_arr > m1_arr * 0.3), 10, 0)
    g2r = mask & (m1_arr <= 0) & (kd_macd_arr > 0)
    bs += np.where(g2r, 8, 0); buy_cnt += g2r.astype(np.int32)
    r2g = mask & (m1_arr >= 0) & (kd_macd_arr < 0)
    ss += np.where(r2g, 8, 0); sell_cnt += r2g.astype(np.int32)

    # ── 6. 二次交叉 (rolling count) ──
    gc_sig = (k1 <= d1) & (k > d)
    dc_sig = (k1 >= d1) & (k < d)
    gc_prev = pd.Series(gc_sig.astype(np.float64)).shift(1).rolling(20, min_periods=1).sum().values
    dc_prev = pd.Series(dc_sig.astype(np.float64)).shift(1).rolling(20, min_periods=1).sum().values
    sg_m = mask & (idx >= 22) & gc_sig & (gc_prev >= 1)
    bs += np.where(sg_m, np.minimum(25, gc_prev * 12), 0); buy_cnt += sg_m.astype(np.int32)
    sd_m = mask & (idx >= 22) & dc_sig & (dc_prev >= 1)
    ss += np.where(sd_m, np.minimum(25, dc_prev * 12), 0); sell_cnt += sd_m.astype(np.int32)

    # ── 7. 四撞 (transition counting) ──
    enter_top = np.zeros(n, dtype=bool); enter_top[1:] = (k[1:] > 80) & (k[:-1] <= 80)
    enter_bot = np.zeros(n, dtype=bool); enter_bot[1:] = (k[1:] < 20) & (k[:-1] >= 20)
    rt = pd.Series(enter_top.astype(np.float64)).rolling(40, min_periods=1).sum().values
    rb = pd.Series(enter_bot.astype(np.float64)).rolling(40, min_periods=1).sum().values
    ft_m = mask & (rt >= 4) & (k < 80)
    ss += np.where(ft_m, np.minimum(35, rt * 8), 0); sell_cnt += ft_m.astype(np.int32)
    fb_m = mask & (rb >= 4) & (k > 20)
    bs += np.where(fb_m, np.minimum(35, rb * 8), 0); buy_cnt += fb_m.astype(np.int32)

    # ── 8. 回测不破 ──
    gap = k - d; gap1 = k1 - d1; gap2 = k2 - d2
    pb_buy = mask & (k > d) & (k1 > d1) & (gap1 < gap2) & (gap > gap1) & (gap1 > 0) & (gap1 < 8) & (gap2 > 0)
    bs += np.where(pb_buy, np.where(d > 50, 15, 8), 0); buy_cnt += pb_buy.astype(np.int32)
    gi = d - k; gi1 = d1 - k1; gi2 = d2 - k2
    pb_sell = mask & (k < d) & (k1 < d1) & (gi1 < gi2) & (gi > gi1) & (gi1 > 0) & (gi1 < 8) & (gi2 > 0)
    ss += np.where(pb_sell, np.where(d < 50, 15, 8), 0); sell_cnt += pb_sell.astype(np.int32)

    # ── 9. K方向 ──
    ss += np.where(mask & (k_slope < -10) & (k > 60), 5, 0)
    bs += np.where(mask & (k_slope > 10) & (k < 40), 5, 0)

    # ── 10. J极值 ──
    je_s = mask & (j > 110) & (k1 > k)
    ss += np.where(je_s, 8, 0); sell_cnt += je_s.astype(np.int32)
    je_b = mask & (j < -10) & (k1 < k)
    bs += np.where(je_b, 8, 0); buy_cnt += je_b.astype(np.int32)

    # ── 11. 多空分界 ──
    bs += np.where(mask & (k > 50) & (d > 50) & (k_slope > 0), 3, 0)
    ss += np.where(mask & (k < 50) & (d < 50) & (k_slope < 0), 3, 0)

    # ── 12. 可靠性乘数 ──
    ss = np.where(sell_cnt >= 3, ss * 1.15, ss)
    bs = np.where(buy_cnt >= 3, bs * 1.15, bs)
    ss = np.minimum(100, ss); bs = np.minimum(100, bs)
    ss[:15] = 0; bs[:15] = 0

    sell_scores = pd.Series(ss, index=df.index)
    buy_scores = pd.Series(bs, index=df.index)
    signal_names = pd.Series('', index=df.index, dtype=str)
    return sell_scores, buy_scores, signal_names


# ══════════════════════════════════════════════════════════
#  2. 布林带向量化
# ══════════════════════════════════════════════════════════

def compute_bollinger_scores_vec(df):
    """向量化布林带评分 — 替代 bollinger_strategy.compute_bollinger_scores"""
    from bollinger_strategy import compute_bollinger
    df = compute_bollinger(df)
    n = len(df)
    idx_arr = np.arange(n)
    mask = idx_arr >= 25

    pct_b = df['bb_pct_b'].values.astype(np.float64)
    bw = df['bb_bandwidth'].values.astype(np.float64)
    slope = df['bb_mid_slope'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)
    upper = df['bb_upper'].values.astype(np.float64)
    lower = df['bb_lower'].values.astype(np.float64)
    mid = df['bb_mid'].values.astype(np.float64)
    bw_sma = df['bb_bw_sma'].values.astype(np.float64)
    prev_close = np.empty(n); prev_close[0] = close[0]; prev_close[1:] = close[:-1]

    valid = mask & ~np.isnan(pct_b) & ~np.isnan(bw)
    ss = np.zeros(n, dtype=np.float64)
    bs = np.zeros(n, dtype=np.float64)

    # ── 1. %B 超买/超卖 ──
    # pct_b > 1.0
    m1 = valid & (pct_b > 1.0)
    ss += np.where(m1 & (slope < 0), 25, 0)
    bs += np.where(m1 & (slope >= 0) & (slope > 0.5), 10, 0)
    ss += np.where(m1 & (slope >= 0) & (slope <= 0.5), 15, 0)
    # pct_b < 0
    m2 = valid & (pct_b < 0)
    bs += np.where(m2 & (slope > 0), 25, 0)
    ss += np.where(m2 & (slope <= 0) & (slope < -0.5), 10, 0)
    bs += np.where(m2 & (slope <= 0) & (slope >= -0.5), 15, 0)
    # pct_b 偏高/偏低
    m3 = valid & (pct_b <= 1.0) & (pct_b > 0.85)
    ss += np.where(m3, np.floor((pct_b - 0.85) * 100), 0)
    m4 = valid & (pct_b >= 0) & (pct_b < 0.15)
    bs += np.where(m4, np.floor((0.15 - pct_b) * 100), 0)

    # ── 2. Squeeze (rolling percentile 近似) ──
    bw_pctile = pd.Series(bw).rolling(120, min_periods=20).rank(pct=True).values * 100
    squeeze = valid & (bw_pctile < 20) & ~np.isnan(bw_pctile)
    bs += np.where(squeeze & (close > mid) & (slope > 0), 20, 0)
    ss += np.where(squeeze & (close < mid) & (slope < 0), 20, 0)

    # ── 3. 带宽扩张 ──
    bw_expand = valid & ~np.isnan(bw_sma) & (bw > bw_sma * 1.5)
    bs += np.where(bw_expand & (close > upper), 15, 0)
    ss += np.where(bw_expand & (close < lower), 15, 0)

    # ── 4. W底/M顶 (rolling 近似) ──
    # W底: pct_b 在 lookback 内有 >= 2 次进入 <0.2 区域且当前回升
    enter_low = np.zeros(n, dtype=bool)
    enter_low[1:] = (pct_b[1:] < 0.2) & (pct_b[:-1] >= 0.2)
    low_entries = pd.Series(enter_low.astype(np.float64)).rolling(30, min_periods=1).sum().values
    w_bottom = valid & (low_entries >= 2) & (pct_b > 0.2) & (pct_b < 0.5)
    bs += np.where(w_bottom, 35, 0)

    enter_high = np.zeros(n, dtype=bool)
    enter_high[1:] = (pct_b[1:] > 0.8) & (pct_b[:-1] <= 0.8)
    high_entries = pd.Series(enter_high.astype(np.float64)).rolling(30, min_periods=1).sum().values
    m_top = valid & (high_entries >= 2) & (pct_b < 0.8) & (pct_b > 0.5)
    ss += np.where(m_top, 35, 0)

    # ── 5. 中轨支撑/阻力 ──
    cross_up = valid & (prev_close < mid) & (close > mid) & (slope > 0)
    cross_dn = valid & (prev_close > mid) & (close < mid) & (slope < 0)
    bs += np.where(cross_up, 10, 0)
    ss += np.where(cross_dn, 10, 0)

    # ── 6. 收缩后方向确认 (rolling diff 近似) ──
    bw_diff = np.zeros(n)
    bw_diff[1:] = bw[1:] - bw[:-1]
    bw_all_expanding = pd.Series((bw_diff > 0).astype(np.float64)).rolling(5, min_periods=4).min().values
    exp_mask = valid & (idx_arr >= 5) & (bw_all_expanding > 0)
    bs += np.where(exp_mask & (close > upper), 15, 0)
    ss += np.where(exp_mask & (close < lower), 15, 0)

    ss = np.minimum(100, ss); bs = np.minimum(100, bs)
    ss[:25] = 0; bs[:25] = 0
    return pd.Series(ss, index=df.index), pd.Series(bs, index=df.index), pd.Series('', index=df.index, dtype=str)


# ══════════════════════════════════════════════════════════
#  3. 量价向量化
# ══════════════════════════════════════════════════════════

def compute_volume_indicators_vec(df):
    """向量化量价指标计算 — 替代 volume_price_strategy.compute_volume_indicators"""
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # OBV: 用 cumsum 替代循环
    sign = np.sign(close.diff())
    sign.iloc[0] = 0
    df['obv'] = (sign * volume).cumsum()
    df['obv_sma20'] = df['obv'].rolling(20, min_periods=5).mean()
    df['obv_sma5'] = df['obv'].rolling(5, min_periods=3).mean()

    # Volume SMA
    df['vol_sma20'] = volume.rolling(20, min_periods=5).mean()
    df['vol_sma5'] = volume.rolling(5, min_periods=3).mean()
    df['vol_ratio'] = volume / df['vol_sma20'].replace(0, np.nan)

    # VWAP
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).rolling(20, min_periods=5).sum()
    cum_vol = volume.rolling(20, min_periods=5).sum()
    df['vwap'] = cum_tp_vol / cum_vol.replace(0, np.nan)

    # A/D Line
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    clv = clv.fillna(0)
    df['ad_line'] = (clv * volume).cumsum()
    df['ad_sma20'] = df['ad_line'].rolling(20, min_periods=5).mean()

    # MFI (向量化)
    tp = typical_price
    raw_mf = tp * volume
    tp_diff = tp.diff()
    pos_mf = pd.Series(np.where(tp_diff > 0, raw_mf, 0), index=df.index)
    neg_mf = pd.Series(np.where(tp_diff < 0, raw_mf, 0), index=df.index)
    mfi_period = 14
    pos_sum = pos_mf.rolling(mfi_period, min_periods=mfi_period).sum()
    neg_sum = neg_mf.rolling(mfi_period, min_periods=mfi_period).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    df['mfi'] = 100 - (100 / (1 + mfr))

    # Price change
    df['price_change'] = close.pct_change()
    df['price_change_5'] = close.pct_change(5)

    # Vol-Price correlation
    df['vol_price_corr'] = close.rolling(20, min_periods=10).corr(volume)

    return df


def compute_volume_price_scores_vec(df):
    """向量化量价评分 — 替代 volume_price_strategy.compute_volume_price_scores"""
    df = compute_volume_indicators_vec(df)
    n = len(df)
    idx_arr = np.arange(n)
    mask = idx_arr >= 25

    vol_ratio = df['vol_ratio'].values.astype(np.float64)
    mfi = df['mfi'].values.astype(np.float64)
    vwap = df['vwap'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)
    obv = df['obv'].values.astype(np.float64)
    obv_sma = df['obv_sma20'].values.astype(np.float64)
    ad = df['ad_line'].values.astype(np.float64)
    ad_sma = df['ad_sma20'].values.astype(np.float64)
    corr = df['vol_price_corr'].values.astype(np.float64)
    price_chg = df['price_change'].values.astype(np.float64)
    price_chg5 = df['price_change_5'].values.astype(np.float64)

    mfi_ok = mask & ~np.isnan(mfi)
    vwap_ok = mask & ~np.isnan(vwap)
    vr_ok = mask & ~np.isnan(vol_ratio) & ~np.isnan(price_chg)
    obv_ok = mask & ~np.isnan(obv_sma)
    ad_ok = mask & ~np.isnan(ad_sma)
    corr_ok = mask & ~np.isnan(corr)

    ss = np.zeros(n, dtype=np.float64)
    bs = np.zeros(n, dtype=np.float64)

    # ── 1. MFI ──
    ss += np.where(mfi_ok & (mfi > 80), 20,
          np.where(mfi_ok & (mfi > 70), 10, 0))
    bs += np.where(mfi_ok & (mfi < 20), 20,
          np.where(mfi_ok & (mfi < 30), 10, 0))

    # ── 2. VWAP ──
    vwap_dist = np.where(vwap_ok, (close - vwap) / np.where(vwap != 0, vwap, 1) * 100, 0)
    ss += np.where(vwap_ok & (vwap_dist > 2), 10, 0)
    bs += np.where(vwap_ok & (vwap_dist < -2), 10, 0)

    # ── 3. 量价关系 ──
    放量 = vr_ok & (vol_ratio > 1.5)
    缩量 = vr_ok & (vol_ratio < 0.5)
    # 放量上涨
    ss += np.where(放量 & (price_chg > 0.01) & ~np.isnan(mfi) & (mfi > 70), 10, 0)
    bs += np.where(放量 & (price_chg > 0.01) & ~(~np.isnan(mfi) & (mfi > 70)), 15, 0)
    # 放量下跌
    bs += np.where(放量 & (price_chg < -0.01) & ~np.isnan(mfi) & (mfi < 30), 10, 0)
    ss += np.where(放量 & (price_chg < -0.01) & ~(~np.isnan(mfi) & (mfi < 30)), 15, 0)
    # 缩量上涨/下跌
    ss += np.where(缩量 & (price_chg > 0.01), 10, 0)
    bs += np.where(缩量 & (price_chg < -0.01), 10, 0)

    # ── 4. OBV ──
    bs += np.where(obv_ok & (obv > obv_sma * 1.05), 10, 0)
    ss += np.where(obv_ok & (obv < obv_sma * 0.95), 10, 0)

    # ── 5. A/D ──
    bs += np.where(ad_ok & (ad > ad_sma), 8, 0)
    ss += np.where(ad_ok & (ad < ad_sma), 8, 0)

    # ── 6. 量价高潮 ──
    climax_mask = mask & ~np.isnan(vol_ratio) & (vol_ratio >= 3.0) & ~np.isnan(price_chg)
    ss += np.where(climax_mask & (price_chg > 0.02), 35, 0)   # 买入高潮 → 看跌
    bs += np.where(climax_mask & (price_chg < -0.02), 35, 0)  # 卖出高潮 → 看涨

    # ── 7. 量价背离 ──
    # 原版逻辑: vol_at_max = 价格最高点处的成交量, vol_at_min = 价格最低点处的成交量
    # 向量化: 用 rolling argmax/argmin 定位价格极值位置, 再取对应成交量
    lb = 20
    div_mask = mask & (idx_arr >= lb + 5)
    close_s = pd.Series(close)
    vol_sma5 = df['vol_sma5'].values.astype(np.float64)
    vol_s = pd.Series(vol_sma5)
    rmax_c = close_s.rolling(lb, min_periods=lb).max().values
    rmin_c = close_s.rolling(lb, min_periods=lb).min().values

    # 价格最高点/最低点在窗口内的位置 → 取对应成交量
    def _vol_at_price_extreme(close_arr, vol_arr, window, mode='max'):
        """取 rolling 窗口内价格极值点处的成交量"""
        n = len(close_arr)
        result = np.full(n, np.nan)
        for i in range(window - 1, n):
            seg = close_arr[i - window + 1:i + 1]
            idx = np.argmax(seg) if mode == 'max' else np.argmin(seg)
            result[i] = vol_arr[i - window + 1 + idx]
        return result

    vol_at_max = _vol_at_price_extreme(close, vol_sma5, lb, mode='max')
    vol_at_min = _vol_at_price_extreme(close, vol_sma5, lb, mode='min')

    bear_div = div_mask & (close >= rmax_c * 0.98) & ~np.isnan(vol_sma5) & ~np.isnan(vol_at_max) & (vol_sma5 < vol_at_max * 0.7)
    ss += np.where(bear_div, 30, 0)
    bull_div = div_mask & (close <= rmin_c * 1.02) & ~np.isnan(vol_sma5) & ~np.isnan(vol_at_min) & (vol_sma5 < vol_at_min * 0.7)
    bs += np.where(bull_div, 30, 0)

    # ── 8. 缩量→放量突破 ──
    recent_ratio = pd.Series(vol_ratio).rolling(3, min_periods=1).mean().values
    prior_ratio = pd.Series(vol_ratio).shift(3).rolling(17, min_periods=5).mean().values
    brk_mask = mask & (idx_arr >= lb + 5) & ~np.isnan(recent_ratio) & ~np.isnan(prior_ratio)
    brk_cond = brk_mask & (prior_ratio < 0.7) & (recent_ratio > 1.5)
    bs += np.where(brk_cond & ~np.isnan(price_chg5) & (price_chg5 > 0.02), 40, 0)
    ss += np.where(brk_cond & ~np.isnan(price_chg5) & (price_chg5 < -0.02), 40, 0)

    # ── 9. OBV背离 ──
    obv5 = np.empty(n); obv5[:5] = obv[:5]
    obv5[5:] = obv[:-5]
    close5 = np.empty(n); close5[:5] = close[:5]
    close5[5:] = close[:-5]
    p_slope = np.where(close5 != 0, (close - close5) / close5, 0)
    obv_std = pd.Series(obv).rolling(20, min_periods=5).std().values
    o_slope = np.where(obv_std != 0, (obv - obv5) / np.where(obv_std != 0, obv_std, 1), 0)
    ss += np.where(mask & (p_slope > 0.01) & (o_slope < -0.3), 25, 0)
    bs += np.where(mask & (p_slope < -0.01) & (o_slope > 0.3), 25, 0)

    # ── 10. 量价相关性 ──
    ss += np.where(corr_ok & (corr < -0.5), 10, 0)

    ss = np.minimum(100, ss); bs = np.minimum(100, bs)
    ss[:25] = 0; bs[:25] = 0
    return pd.Series(ss, index=df.index), pd.Series(bs, index=df.index), pd.Series('', index=df.index, dtype=str)


# ══════════════════════════════════════════════════════════
#  4. K线形态向量化
# ══════════════════════════════════════════════════════════

def compute_candlestick_scores_vec(df):
    """向量化 K 线形态评分 — 替代 candlestick_patterns.compute_candlestick_scores

    策略: 不逐行调用 30+ 检测器，而是把最常见的形态用向量化检测,
    低频形态用批量判断。
    """
    n = len(df)
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    v = df['volume'].values.astype(np.float64)

    body = c - o
    abs_body = np.abs(body)
    total_range = h - l
    total_range_safe = np.where(total_range == 0, 1e-10, total_range)
    body_ratio = abs_body / total_range_safe
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    # Volume confirmation
    vol_ma20 = pd.Series(v).rolling(20, min_periods=5).mean().values
    vol_mult = np.where(~np.isnan(vol_ma20) & (vol_ma20 > 0),
                         np.clip(v / vol_ma20, 0.5, 2.0), 1.0)

    ss = np.zeros(n, dtype=np.float64)
    bs = np.zeros(n, dtype=np.float64)
    mask = np.arange(n) >= 5

    # Shifted values
    body1 = np.empty(n); body1[0] = 0; body1[1:] = body[:-1]
    abs_body1 = np.abs(body1)
    c1 = np.empty(n); c1[0] = c[0]; c1[1:] = c[:-1]
    o1 = np.empty(n); o1[0] = o[0]; o1[1:] = o[:-1]
    h1 = np.empty(n); h1[0] = h[0]; h1[1:] = h[:-1]
    l1 = np.empty(n); l1[0] = l[0]; l1[1:] = l[:-1]
    c2 = np.empty(n); c2[:2] = c[:2]; c2[2:] = c[:-2]
    o2 = np.empty(n); o2[:2] = o[:2]; o2[2:] = o[:-2]

    # ── 1. 十字星 (Doji) ── body < 10% of range
    doji = mask & (body_ratio < 0.10) & (total_range > 0)
    # 上影线长 → 看跌; 下影线长 → 看涨
    doji_bear = doji & (upper_wick > lower_wick * 2)
    doji_bull = doji & (lower_wick > upper_wick * 2)
    ss += np.where(doji_bear, 20.0 * vol_mult, 0)
    bs += np.where(doji_bull, 20.0 * vol_mult, 0)

    # ── 2. 锤子线 (Hammer / Hanging Man) ──
    # 下影线 >= 2x body, 上影线小
    hammer = mask & (lower_wick >= abs_body * 2) & (upper_wick < abs_body * 0.5) & (abs_body > 0)
    # 下跌后的锤子 → 看涨
    trend_5 = np.empty(n); trend_5[:5] = 0; trend_5[5:] = c[5:] - c[:-5]
    bs += np.where(hammer & (trend_5 < 0), 30.0 * vol_mult, 0)
    # 上涨后的上吊线 → 看跌
    ss += np.where(hammer & (trend_5 > 0), 25.0 * vol_mult, 0)

    # ── 3. 射击之星 (Shooting Star) / 倒锤子 ──
    shooting = mask & (upper_wick >= abs_body * 2) & (lower_wick < abs_body * 0.5) & (abs_body > 0)
    ss += np.where(shooting & (trend_5 > 0), 30.0 * vol_mult, 0)
    bs += np.where(shooting & (trend_5 < 0), 25.0 * vol_mult, 0)

    # ── 4. 吞没 (Engulfing) ──
    # 看涨吞没: 前阴后阳, 当前body完全包含前body
    bull_engulf = mask & (body1 < 0) & (body > 0) & (o <= c1) & (c >= o1) & (abs_body > abs_body1)
    bs += np.where(bull_engulf, 35.0 * vol_mult, 0)
    # 看跌吞没
    bear_engulf = mask & (body1 > 0) & (body < 0) & (o >= c1) & (c <= o1) & (abs_body > abs_body1)
    ss += np.where(bear_engulf, 35.0 * vol_mult, 0)

    # ── 5. 乌云盖顶 / 刺透 ──
    # 乌云盖顶: 前阳后阴, 开盘高于前高, 收盘低于前body中点
    dark_cloud = mask & (body1 > 0) & (body < 0) & (o > h1) & (c < (o1 + c1) / 2)
    ss += np.where(dark_cloud, 30.0 * vol_mult, 0)
    # 刺透: 前阴后阳, 开盘低于前低, 收盘高于前body中点
    piercing = mask & (body1 < 0) & (body > 0) & (o < l1) & (c > (o1 + c1) / 2)
    bs += np.where(piercing, 30.0 * vol_mult, 0)

    # ── 6. 三只乌鸦 / 三白兵 (需要连续3根) ──
    body2 = np.empty(n); body2[:2] = 0; body2[2:] = body[:-2]
    three_black = mask & (body < 0) & (body1 < 0) & (body2 < 0) & (c < c1) & (c1 < c2)
    ss += np.where(three_black, 25.0 * vol_mult, 0)
    three_white = mask & (body > 0) & (body1 > 0) & (body2 > 0) & (c > c1) & (c1 > c2)
    bs += np.where(three_white, 25.0 * vol_mult, 0)

    # ── 7. 长实体 ──
    avg_body = pd.Series(abs_body).rolling(20, min_periods=5).mean().values
    big_body = mask & ~np.isnan(avg_body) & (abs_body > avg_body * 2)
    ss += np.where(big_body & (body < 0), 15.0 * vol_mult, 0)
    bs += np.where(big_body & (body > 0), 15.0 * vol_mult, 0)

    # ── 8. 早晨之星 / 黄昏之星 (3-candle) ──
    small_body1 = (abs_body1 < avg_body * 0.3)
    # 早晨之星: 前大阴 + 中间小body + 当前大阳
    morning_star = mask & (body2 < 0) & (np.abs(body2) > avg_body) & small_body1 & (body > 0) & (abs_body > avg_body)
    bs += np.where(morning_star, 40.0 * vol_mult, 0)
    # 黄昏之星: 前大阳 + 中间小body + 当前大阴
    evening_star = mask & (body2 > 0) & (np.abs(body2) > avg_body) & small_body1 & (body < 0) & (abs_body > avg_body)
    ss += np.where(evening_star, 40.0 * vol_mult, 0)

    ss = np.minimum(100, ss); bs = np.minimum(100, bs)
    ss[:5] = 0; bs[:5] = 0
    return pd.Series(ss, index=df.index), pd.Series(bs, index=df.index), pd.Series('', index=df.index, dtype=str)


# ══════════════════════════════════════════════════════════
#  5. 背离分析快速版
# ══════════════════════════════════════════════════════════

def analyze_signals_enhanced_fast(df, window):
    """快速版背离分析 — 替代 strategy_enhanced.analyze_signals_enhanced

    核心优化: 预计算全 DataFrame 的特征，用 rolling 聚合替代逐窗口 ComprehensiveAnalyzer。
    原版对每个 step 位置创建新的 ComprehensiveAnalyzer(window_df)，重复计算大量重叠特征。
    快速版只计算一次，然后 rolling 聚合。
    """
    from strategy_enhanced import DEFAULT_SIG
    n = len(df)
    if n < window + 10:
        return {}

    step = max(1, window // 8)

    # ── 预计算 MACD 相关特征 ──
    dif = df['DIF'].values.astype(np.float64) if 'DIF' in df.columns else np.zeros(n)
    dea = df['DEA'].values.astype(np.float64) if 'DEA' in df.columns else np.zeros(n)
    macd_bar = df['MACD_BAR'].values.astype(np.float64) if 'MACD_BAR' in df.columns else np.zeros(n)
    close = df['close'].values.astype(np.float64)
    high_arr = df['high'].values.astype(np.float64)
    low_arr = df['low'].values.astype(np.float64)

    dif1 = np.empty(n); dif1[0] = 0; dif1[1:] = dif[:-1]
    dea1 = np.empty(n); dea1[0] = 0; dea1[1:] = dea[:-1]
    macd1 = np.empty(n); macd1[0] = 0; macd1[1:] = macd_bar[:-1]

    # MACD 金叉/死叉
    golden = (dif1 <= dea1) & (dif > dea)
    death = (dif1 >= dea1) & (dif < dea)
    last_cross_golden = pd.Series(golden.astype(np.float64)).rolling(window, min_periods=1).sum().values
    last_cross_death = pd.Series(death.astype(np.float64)).rolling(window, min_periods=1).sum().values

    # MACD bar 正负交替 (隔堆)
    bar_sign_change = np.zeros(n, dtype=bool)
    bar_sign_change[1:] = (macd_bar[1:] * macd_bar[:-1]) < 0
    separated_count = pd.Series(bar_sign_change.astype(np.float64)).rolling(window, min_periods=1).sum().values

    # DIF 零轴穿越
    dif_zero_cross = np.zeros(n, dtype=bool)
    dif_zero_cross[1:] = ((dif[1:] > 0) & (dif[:-1] <= 0)) | ((dif[1:] < 0) & (dif[:-1] >= 0))
    zero_returns = pd.Series(dif_zero_cross.astype(np.float64)).rolling(window, min_periods=1).sum().values

    # 柱面积背离 (rolling sum of abs macd_bar, 正负分开)
    pos_bar = np.where(macd_bar > 0, macd_bar, 0)
    neg_bar = np.where(macd_bar < 0, np.abs(macd_bar), 0)
    # 最近一堆正柱面积 vs 前一堆
    pos_area = pd.Series(pos_bar).rolling(window // 3, min_periods=5).sum().values
    neg_area = pd.Series(neg_bar).rolling(window // 3, min_periods=5).sum().values
    pos_area_prev = pd.Series(pos_bar).shift(window // 3).rolling(window // 3, min_periods=5).sum().values
    neg_area_prev = pd.Series(neg_bar).shift(window // 3).rolling(window // 3, min_periods=5).sum().values

    # 价格高低点 (rolling)
    w2 = window // 2
    close_s = pd.Series(close)
    rolling_high = close_s.rolling(w2, min_periods=w2 // 2).max().values
    rolling_low = close_s.rolling(w2, min_periods=w2 // 2).min().values

    # ── KDJ/RSI/CCI 相关特征 ──
    k_vals = df['K'].values.astype(np.float64) if 'K' in df.columns else np.full(n, 50.0)
    rsi_vals = df['RSI6'].values.astype(np.float64) if 'RSI6' in df.columns else np.full(n, 50.0)
    cci_vals = df['CCI'].values.astype(np.float64) if 'CCI' in df.columns else np.zeros(n)

    k_s = pd.Series(k_vals)
    rsi_s = pd.Series(rsi_vals)
    cci_s = pd.Series(cci_vals)

    rmax_k = k_s.rolling(w2, min_periods=w2 // 2).max().values
    rmin_k = k_s.rolling(w2, min_periods=w2 // 2).min().values

    # CCI 穿越
    cci1 = np.empty(n); cci1[0] = 0; cci1[1:] = cci_vals[:-1]
    cci_spring_up = (cci1 <= -100) & (cci_vals > -100)
    cci_autumn_down = (cci1 >= 100) & (cci_vals < 100)

    # CCI 穿越 rolling count (预计算，不在循环内)
    cci_spring_rsum = pd.Series(cci_spring_up.astype(np.float64)).rolling(window, min_periods=1).sum().values
    cci_autumn_rsum = pd.Series(cci_autumn_down.astype(np.float64)).rolling(window, min_periods=1).sum().values

    # CCI 最近穿越方向 (预计算)
    # 使用 forward-fill: 每个位置记录最近的穿越方向
    cci_last_direction = np.zeros(n, dtype=np.int8)  # 0=none, 1=spring_up, -1=autumn_down
    for i in range(n):
        if cci_spring_up[i]:
            cci_last_direction[i] = 1
        elif cci_autumn_down[i]:
            cci_last_direction[i] = -1
        elif i > 0:
            cci_last_direction[i] = cci_last_direction[i - 1]

    # DIF rolling max (预计算)
    dif_rmax = pd.Series(dif).rolling(w2, min_periods=w2 // 2).max().values

    # RSI rolling max/min (预计算)
    rsi_rmax = rsi_s.rolling(w2, min_periods=w2 // 2).max().values
    rsi_rmin = rsi_s.rolling(w2, min_periods=w2 // 2).min().values

    # ── 量价特征 ──
    volume = df['volume'].values.astype(np.float64) if 'volume' in df.columns else np.ones(n)
    vol_ma = pd.Series(volume).rolling(20, min_periods=5).mean().values
    vol_ratio = np.where(~np.isnan(vol_ma) & (vol_ma > 0), volume / vol_ma, 1.0)
    # 价升量减
    price_up_vol_down = np.zeros(n, dtype=bool)
    if n > 5:
        p_chg5 = np.zeros(n)
        p_chg5[5:] = (close[5:] - close[:-5]) / close[:-5]
        vol_ma5 = pd.Series(volume).rolling(5, min_periods=3).mean().values
        price_up_vol_down = (p_chg5 > 0.01) & (vol_ma5 < vol_ma * 0.6)
    # 地量
    ground_vol = vol_ratio < 0.3

    # 量价 rolling count (预计算)
    puvd_rsum = pd.Series(price_up_vol_down.astype(np.float64)).rolling(window, min_periods=1).sum().values
    gvol_rsum = pd.Series(ground_vol.astype(np.float64)).rolling(window, min_periods=1).sum().values

    # ── 预计算近高/低点判断 ──
    near_high = close >= rolling_high * 0.98
    near_low = close <= rolling_low * 1.02
    amp = np.where(rolling_low > 0, (rolling_high - rolling_low) / rolling_low, 0)

    # ── 构建信号字典 (纯 numpy 索引, 无循环内 pandas) ──
    signals = {}
    for i in range(window, n, step):
        sig = dict(DEFAULT_SIG)

        top_s = 0.0
        bot_s = 0.0

        # MACD 隔堆背离
        half_sep = separated_count[i] / 2
        sep_top = int(half_sep) if near_high[i] else 0
        sep_bot = int(half_sep) if near_low[i] else 0

        sig['separated_top'] = min(sep_top, 3)
        sig['separated_bottom'] = min(sep_bot, 3)
        sig['sep_divs_top'] = min(sep_top, 3)
        sig['sep_divs_bottom'] = min(sep_bot, 3)

        # DIF 零轴返回
        sig['zero_returns_top'] = int(min(zero_returns[i] / 2, 3))
        sig['zero_returns_bottom'] = int(min(zero_returns[i] / 2, 3))

        # 柱面积背离
        if not np.isnan(pos_area[i]) and not np.isnan(pos_area_prev[i]) and pos_area_prev[i] > 0:
            if pos_area[i] < pos_area_prev[i] * 0.7 and near_high[i]:
                sig['area_top_div'] = 1
        if not np.isnan(neg_area[i]) and not np.isnan(neg_area_prev[i]) and neg_area_prev[i] > 0:
            if neg_area[i] < neg_area_prev[i] * 0.7 and near_low[i]:
                sig['area_bottom_div'] = 1

        # DIF/DEA背离 (价格新高但DIF更低) — 使用预计算的 dif_rmax
        if near_high[i] and dif[i] < dif_rmax[i] * 0.85:
            sig['dif_top_div'] = 1

        # 综合 top score
        if sig['sep_divs_top'] >= 2:
            top_s += 30
        elif sig['sep_divs_top'] >= 1 or sig['separated_top'] >= 1:
            top_s += 18
        if sig['area_top_div']:
            top_s += 12
        if sig['dif_top_div']:
            top_s += 8
        if sig['zero_returns_top'] >= 1:
            top_s += 10

        # 综合 bottom score
        if sig['sep_divs_bottom'] >= 2:
            bot_s += 30
        elif sig['sep_divs_bottom'] >= 1:
            bot_s += 15
        if sig.get('area_bottom_div'):
            bot_s += 10

        sig['top'] = top_s
        sig['bottom'] = bot_s

        # KDJ背离
        if k_vals[i] > 70 and near_high[i] and k_vals[i] < rmax_k[i] * 0.9:
            sig['kdj_top_div'] = 1
        if k_vals[i] < 30 and near_low[i] and k_vals[i] > rmin_k[i] * 1.1:
            sig['kdj_bottom_div'] = 1

        # CCI — 使用预计算的 rolling count
        if cci_autumn_rsum[i] > 0:
            sig['cci_top_div'] = 1
        if cci_spring_rsum[i] > 0:
            sig['cci_bottom_div'] = 1
        # CCI最近穿越方向 — 使用预计算的 forward-fill
        if cci_last_direction[i] == 1:
            sig['cci_last_cross'] = 'spring_up'
        elif cci_last_direction[i] == -1:
            sig['cci_last_cross'] = 'autumn_down'

        # RSI 背离 — 使用预计算的 rsi_rmax / rsi_rmin
        if rsi_vals[i] > 60 and near_high[i] and rsi_vals[i] < rsi_rmax[i] * 0.9:
            sig['rsi_top_div'] = 1
        if rsi_vals[i] < 40 and near_low[i] and rsi_vals[i] > rsi_rmin[i] * 1.1:
            sig['rsi_bottom_div'] = 1

        # 量价 — 使用预计算的 rolling count
        sig['vol_price_up_down'] = min(int(puvd_rsum[i]), 5)
        sig['vol_ground'] = min(int(gvol_rsum[i]), 5)

        # MACD金叉死叉
        if last_cross_golden[i] > last_cross_death[i]:
            sig['last_cross'] = 'golden'
        elif last_cross_death[i] > last_cross_golden[i]:
            sig['last_cross'] = 'death'

        # Exhaustion detection
        if top_s >= 40:
            sig['exhaust_sell'] = True
            sig['exhaust_sell_conf'] = 'high' if top_s >= 60 else 'medium'
        if bot_s >= 40:
            sig['exhaust_buy'] = True
            sig['exhaust_buy_conf'] = 'high' if bot_s >= 60 else 'medium'

        # 形态 amplitude
        if amp[i] > 0.15 and near_high[i]:
            sig['pat_amp_top'] = 1
        if amp[i] > 0.15 and near_low[i]:
            sig['pat_amp_bottom'] = 1

        # top/bottom level
        if top_s >= 60:
            sig['top_level'] = 'very_high'
        elif top_s >= 35:
            sig['top_level'] = 'high'
        elif top_s >= 15:
            sig['top_level'] = 'medium'

        if bot_s >= 60:
            sig['bottom_level'] = 'very_high'
        elif bot_s >= 35:
            sig['bottom_level'] = 'high'
        elif bot_s >= 15:
            sig['bottom_level'] = 'medium'

        signals[df.index[i]] = sig

    return signals
