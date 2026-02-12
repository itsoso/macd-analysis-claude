"""
均线技术分析指标模块

基于《均线技术分析》(邱立波著) 全书四章内容:
  第一章: 均线概述 — MA定义/参数/分类/组合
  第二章: 葛南维八大买卖法则
  第三章: 均线实战 — 单线/双线/排列
  第四章: 特殊形态 — 银山谷/金山谷/死亡谷等17种形态

适用于加密货币(ETH/USDT)的15分钟~日线级别分析
"""

import pandas as pd
import numpy as np


# ======================================================
#  第一章: 均线计算与基础
# ======================================================

# 标准均线周期(书中推荐)
MA_PERIODS = [5, 10, 20, 30, 60, 120, 240]

# 加密货币适配周期(24/7交易, 无休市)
CRYPTO_MA_PERIODS = {
    '15m': [5, 10, 20, 60, 120, 240, 480],   # 15m: 1.25h~5天
    '1h':  [5, 10, 20, 30, 60, 120, 240],     # 1h: 5h~10天
    '4h':  [5, 10, 20, 30, 60, 120],          # 4h: 20h~20天
    '8h':  [5, 10, 20, 30, 60],               # 8h: 40h~20天
    '1d':  [5, 10, 20, 30, 60, 120, 240],     # 日线: 标准
}


def calc_sma(series, period):
    """简单移动平均线 (SMA)"""
    return series.rolling(window=period, min_periods=period).mean()


def calc_ema(series, period):
    """指数移动平均线 (EMA) — 对近期价格赋予更高权重"""
    return series.ewm(span=period, adjust=False).mean()


def calc_wma(series, period):
    """加权移动平均线 (WMA) — 线性递增权重"""
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def add_moving_averages(df, periods=None, timeframe='1h', ma_type='sma'):
    """
    为DataFrame添加多条均线

    Parameters:
        df: 含'close'列的DataFrame
        periods: 均线周期列表, None则使用timeframe对应的默认周期
        timeframe: 时间周期 ('15m', '1h', '4h', '8h', '1d')
        ma_type: 均线类型 ('sma', 'ema', 'wma')
    """
    if periods is None:
        periods = CRYPTO_MA_PERIODS.get(timeframe, MA_PERIODS)

    calc_fn = {'sma': calc_sma, 'ema': calc_ema, 'wma': calc_wma}.get(ma_type, calc_sma)

    for p in periods:
        col = f'MA{p}'
        if col not in df.columns:
            df[col] = calc_fn(df['close'], p)

    return df


# ======================================================
#  第一章: 均线特性分析
# ======================================================

def ma_slope(df, ma_col, lookback=3):
    """均线斜率(趋势方向): >0上升, <0下降, ≈0走平"""
    ma = df[ma_col]
    slope = (ma - ma.shift(lookback)) / ma.shift(lookback) * 100  # 百分比
    return slope


def price_ma_distance(df, ma_col):
    """价格与均线的乖离率(%)"""
    return (df['close'] - df[ma_col]) / df[ma_col] * 100


def ma_support_pressure(df, ma_col, threshold_pct=0.5):
    """
    均线支撑/压力判定

    Returns:
        Series: 'support' / 'pressure' / 'none'
    """
    dist = price_ma_distance(df, ma_col)
    result = pd.Series('none', index=df.index)
    # 价格在均线上方且接近 → 支撑
    result[(dist > 0) & (dist < threshold_pct)] = 'support'
    # 价格在均线下方且接近 → 压力
    result[(dist < 0) & (dist > -threshold_pct)] = 'pressure'
    return result


# ======================================================
#  第二章: 葛南维八大买卖法则
# ======================================================

def granville_rules(df, ma_col='MA20', distance_threshold=3.0, slope_lookback=5):
    """
    葛南维(Granville)均线八大买卖法则

    Parameters:
        df: 含'close'和均线列的DataFrame
        ma_col: 参考均线列名
        distance_threshold: 乖离率阈值(%)
        slope_lookback: 斜率计算回看期

    Returns:
        DataFrame with columns: rule_1..rule_8, buy_signal, sell_signal, signal_strength
    """
    close = df['close']
    ma = df[ma_col]
    slope = ma_slope(df, ma_col, slope_lookback)
    dist = price_ma_distance(df, ma_col)

    # 价格穿越均线
    cross_up = (close > ma) & (close.shift(1) <= ma.shift(1))    # 上穿
    cross_down = (close < ma) & (close.shift(1) >= ma.shift(1))  # 下穿

    # 均线方向
    ma_rising = slope > 0.1
    ma_falling = slope < -0.1
    ma_flat = slope.abs() < 0.1

    # === 买入法则 ===

    # 法则1: 均线走平或上升, 价格从下方上穿均线 → 买入
    rule_1 = cross_up & (ma_rising | ma_flat)

    # 法则2: 价格在均线上方运行, 回调不破均线, 再度上升 → 买入
    above_ma = close > ma
    was_above = close.shift(1) > ma.shift(1)
    near_ma = dist.abs() < 1.5  # 接近均线
    price_rising = close > close.shift(1)
    rule_2 = above_ma & was_above & near_ma & price_rising & ma_rising

    # 法则3: 价格短暂跌破均线, 但均线仍上升, 价格快速回到均线上方 → 买入
    just_crossed_down = cross_down.shift(1) | cross_down.shift(2) | cross_down.shift(3)
    rule_3 = above_ma & just_crossed_down & ma_rising

    # 法则4: 价格暴跌远离均线, 乖离率过大 → 超卖反弹买入
    rule_4 = (dist < -distance_threshold) & ma_falling.shift(1).fillna(False)

    # === 卖出法则 ===

    # 法则5: 价格暴涨远离均线, 乖离率过大 → 超买回落卖出
    rule_5 = (dist > distance_threshold) & ma_rising.shift(1).fillna(False)

    # 法则6: 均线走平或下降, 价格从上方下穿均线 → 卖出
    rule_6 = cross_down & (ma_falling | ma_flat)

    # 法则7: 价格反弹未能突破均线, 均线转跌 → 卖出
    below_ma = close < ma
    was_below = close.shift(1) < ma.shift(1)
    near_ma_below = (dist > -1.5) & (dist < 0)
    price_falling = close < close.shift(1)
    rule_7 = below_ma & was_below & near_ma_below & price_falling & ma_falling

    # 法则8: 价格在均线上方徘徊, 均线持续下跌 → 卖出
    rule_8 = above_ma & (dist < 1.0) & ma_falling

    # 综合信号
    buy_signal = rule_1 | rule_2 | rule_3 | rule_4
    sell_signal = rule_5 | rule_6 | rule_7 | rule_8

    # 信号强度 (-100 ~ +100)
    strength = pd.Series(0.0, index=df.index)
    strength += rule_1.astype(float) * 25   # 趋势确认买入
    strength += rule_2.astype(float) * 20   # 回调支撑买入
    strength += rule_3.astype(float) * 15   # 假破位买入
    strength += rule_4.astype(float) * 30   # 超卖反弹(强)
    strength -= rule_5.astype(float) * 30   # 超买回落(强)
    strength -= rule_6.astype(float) * 25   # 趋势确认卖出
    strength -= rule_7.astype(float) * 20   # 反弹失败卖出
    strength -= rule_8.astype(float) * 15   # 均线压制卖出

    result = pd.DataFrame({
        'rule_1': rule_1, 'rule_2': rule_2, 'rule_3': rule_3, 'rule_4': rule_4,
        'rule_5': rule_5, 'rule_6': rule_6, 'rule_7': rule_7, 'rule_8': rule_8,
        'buy_signal': buy_signal, 'sell_signal': sell_signal,
        'signal_strength': strength.clip(-100, 100),
    }, index=df.index)
    return result


# ======================================================
#  第三章: 均线交叉与排列
# ======================================================

def detect_golden_cross(df, fast_col='MA5', slow_col='MA20'):
    """
    黄金交叉: 短期均线从下方上穿长期均线

    Returns: bool Series
    """
    fast = df[fast_col]
    slow = df[slow_col]
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def detect_death_cross(df, fast_col='MA5', slow_col='MA20'):
    """
    死亡交叉: 短期均线从上方下穿长期均线

    Returns: bool Series
    """
    fast = df[fast_col]
    slow = df[slow_col]
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))


def detect_ma_arrangement(df, ma_cols=None):
    """
    均线排列检测

    Returns:
        Series: 'bullish'(多头排列) / 'bearish'(空头排列) / 'mixed'(混合)

    多头排列: MA5 > MA10 > MA20 > MA60 (短>中>长)
    空头排列: MA5 < MA10 < MA20 < MA60 (短<中<长)
    """
    if ma_cols is None:
        available = [c for c in ['MA5', 'MA10', 'MA20', 'MA60'] if c in df.columns]
        if len(available) < 3:
            return pd.Series('mixed', index=df.index)
        ma_cols = available

    result = pd.Series('mixed', index=df.index)

    # 检查多头排列: 每条短期均线 > 长期均线
    bullish = pd.Series(True, index=df.index)
    for i in range(len(ma_cols) - 1):
        bullish &= df[ma_cols[i]] > df[ma_cols[i + 1]]
    result[bullish] = 'bullish'

    # 检查空头排列: 每条短期均线 < 长期均线
    bearish = pd.Series(True, index=df.index)
    for i in range(len(ma_cols) - 1):
        bearish &= df[ma_cols[i]] < df[ma_cols[i + 1]]
    result[bearish] = 'bearish'

    return result


def detect_ma_convergence(df, ma_cols=None, threshold_pct=0.5):
    """
    均线粘合检测: 多条均线汇聚到很小的范围

    Parameters:
        threshold_pct: 均线最大价差占价格的百分比

    Returns:
        bool Series (True=粘合状态)
    """
    if ma_cols is None:
        ma_cols = [c for c in ['MA5', 'MA10', 'MA20', 'MA30', 'MA60']
                   if c in df.columns]
    if len(ma_cols) < 3:
        return pd.Series(False, index=df.index)

    ma_values = df[ma_cols]
    ma_range = ma_values.max(axis=1) - ma_values.min(axis=1)
    ma_avg = ma_values.mean(axis=1)
    convergence_pct = ma_range / ma_avg * 100

    return convergence_pct < threshold_pct


def detect_ma_divergence(df, ma_cols=None, lookback=10, threshold=0.3):
    """
    均线发散检测: 粘合后均线开始发散

    Returns:
        Series: 'bullish_diverge'(向上发散) / 'bearish_diverge'(向下发散) / 'none'
    """
    if ma_cols is None:
        ma_cols = [c for c in ['MA5', 'MA10', 'MA20', 'MA60'] if c in df.columns]
    if len(ma_cols) < 3:
        return pd.Series('none', index=df.index)

    # 检查前期是否粘合
    was_converged = detect_ma_convergence(df, ma_cols, threshold_pct=0.8)
    recent_converged = pd.Series(False, index=df.index)
    for shift in range(1, lookback + 1):
        recent_converged |= was_converged.shift(shift).fillna(False)

    # 当前是否发散
    ma_range = df[ma_cols].max(axis=1) - df[ma_cols].min(axis=1)
    ma_avg = df[ma_cols].mean(axis=1)
    current_spread = ma_range / ma_avg * 100
    is_diverging = current_spread > threshold

    # 方向判断
    fastest = df[ma_cols[0]]
    slowest = df[ma_cols[-1]]

    result = pd.Series('none', index=df.index)
    result[recent_converged & is_diverging & (fastest > slowest)] = 'bullish_diverge'
    result[recent_converged & is_diverging & (fastest < slowest)] = 'bearish_diverge'

    return result


# ======================================================
#  第四章: 17种均线特殊形态
# ======================================================

def detect_silver_valley(df, fast='MA5', mid='MA10', slow='MA20', lookback=30):
    """
    银山谷: 短中长三条均线交叉形成尖头向上的三角形
    条件: 1) 此前空头排列或下跌
          2) 快线上穿中线和慢线, 中线上穿慢线
          3) 形成向上的三角形

    Returns: bool Series
    """
    if not all(c in df.columns for c in [fast, mid, slow]):
        return pd.Series(False, index=df.index)

    # 快线上穿慢线
    fast_cross_slow = (df[fast] > df[slow]) & (df[fast].shift(1) <= df[slow].shift(1))
    # 快线上穿中线
    fast_cross_mid = (df[fast] > df[mid]) & (df[fast].shift(1) <= df[mid].shift(1))
    # 中线上穿慢线
    mid_cross_slow = (df[mid] > df[slow]) & (df[mid].shift(1) <= df[slow].shift(1))

    # 近期有交叉(lookback范围内)
    recent_cross = pd.Series(False, index=df.index)
    for shift in range(0, lookback):
        recent_cross |= (fast_cross_slow.shift(shift).fillna(False) |
                         fast_cross_mid.shift(shift).fillna(False) |
                         mid_cross_slow.shift(shift).fillna(False))

    # 当前多头排列
    bullish = (df[fast] > df[mid]) & (df[mid] > df[slow])

    # 此前下跌(慢线下降)
    was_falling = df[slow].shift(lookback) > df[slow].shift(1)

    return bullish & recent_cross & was_falling


def detect_gold_valley(df, fast='MA5', mid='MA10', slow='MA20', lookback=60):
    """
    金山谷: 银山谷之后回调再次形成向上三角, 且位置高于银山谷
    更可靠的买入信号

    Returns: bool Series
    """
    silver = detect_silver_valley(df, fast, mid, slow, lookback=20)

    # 检查前期是否有银山谷
    had_silver = pd.Series(False, index=df.index)
    for shift in range(20, lookback):
        had_silver |= silver.shift(shift).fillna(False)

    # 当前再次形成多头排列
    bullish = (df[fast] > df[mid]) & (df[mid] > df[slow])

    # 中间有回调(快线曾低于中线)
    had_pullback = pd.Series(False, index=df.index)
    for shift in range(5, 30):
        had_pullback |= (df[fast].shift(shift) < df[mid].shift(shift))

    # 当前价格高于银山谷时
    return bullish & had_silver & had_pullback


def detect_death_valley(df, fast='MA5', mid='MA10', slow='MA20', lookback=30):
    """
    死亡谷: 短中长三条均线交叉形成尖头向下的三角形
    强烈卖出信号

    Returns: bool Series
    """
    if not all(c in df.columns for c in [fast, mid, slow]):
        return pd.Series(False, index=df.index)

    # 快线下穿慢线
    fast_cross_slow = (df[fast] < df[slow]) & (df[fast].shift(1) >= df[slow].shift(1))
    # 中线下穿慢线
    mid_cross_slow = (df[mid] < df[slow]) & (df[mid].shift(1) >= df[slow].shift(1))

    recent_cross = pd.Series(False, index=df.index)
    for shift in range(0, lookback):
        recent_cross |= (fast_cross_slow.shift(shift).fillna(False) |
                         mid_cross_slow.shift(shift).fillna(False))

    # 当前空头排列
    bearish = (df[fast] < df[mid]) & (df[mid] < df[slow])

    # 此前上涨(慢线上升)
    was_rising = df[slow].shift(lookback) < df[slow].shift(1)

    return bearish & recent_cross & was_rising


def detect_dragon_emerge(df, ma_cols=None, lookback=5):
    """
    蛟龙出海: 一根大阳线突破多条均线(突破3条以上均线)
    强烈买入信号

    Returns: bool Series
    """
    if ma_cols is None:
        ma_cols = [c for c in ['MA5', 'MA10', 'MA20', 'MA30', 'MA60']
                   if c in df.columns]

    close = df['close']
    open_price = df['open']

    # 大阳线: 涨幅 > 2%
    is_big_bullish = (close - open_price) / open_price > 0.02

    # 突破多条均线: 开盘低于均线, 收盘高于均线
    crosses_count = pd.Series(0, index=df.index)
    for col in ma_cols:
        crossed = (open_price < df[col]) & (close > df[col])
        crosses_count += crossed.astype(int)

    return is_big_bullish & (crosses_count >= 3)


def detect_cloud_support(df, fast='MA5', mid='MA10', slow='MA20', lookback=10):
    """
    烘云托月: 短期均线在上方, 中长期均线在下方形成"云层"托住价格
    均线多头排列且价格回调到中期均线附近获得支撑

    Returns: bool Series
    """
    if not all(c in df.columns for c in [fast, mid, slow]):
        return pd.Series(False, index=df.index)

    close = df['close']

    # 多头排列
    bullish = (df[fast] > df[mid]) & (df[mid] > df[slow])

    # 价格回调到中期均线附近(在中期和短期均线之间)
    near_mid = (close > df[mid]) & (close < df[fast])

    # 中长期均线形成支撑带(差距小)
    cloud_tight = (df[mid] - df[slow]).abs() / df[slow] * 100 < 1.5

    # 前期有上涨
    was_rising = close.shift(lookback) < close

    return bullish & near_mid & cloud_tight & was_rising


def detect_wave_up(df, fast='MA5', slow='MA20', lookback=30, min_waves=2):
    """
    逐浪上升: 价格在均线上方波浪式上涨, 每次回调到均线获支撑后再创新高

    Returns: bool Series
    """
    if not all(c in df.columns for c in [fast, slow]):
        return pd.Series(False, index=df.index)

    close = df['close']
    result = pd.Series(False, index=df.index)

    for i in range(lookback, len(df)):
        window = df.iloc[i - lookback:i + 1]
        wclose = window['close']
        wma = window[slow]

        if len(wclose) < lookback:
            continue

        # 整体上升趋势
        if wclose.iloc[-1] <= wclose.iloc[0]:
            continue

        # 大部分时间在均线上方
        above_ratio = (wclose > wma).sum() / len(wclose)
        if above_ratio < 0.6:
            continue

        # 检测波浪: 寻找局部低点(碰到均线)和高点
        touches = 0
        for j in range(5, len(wclose) - 1):
            dist_pct = (wclose.iloc[j] - wma.iloc[j]) / wma.iloc[j] * 100
            if 0 <= dist_pct < 1.0:  # 接近均线
                touches += 1

        if touches >= min_waves:
            result.iloc[i] = True

    return result


def detect_wave_down(df, fast='MA5', slow='MA20', lookback=30, min_waves=2):
    """
    逐浪下降: 价格在均线下方波浪式下跌, 每次反弹到均线遇阻后再创新低

    Returns: bool Series
    """
    if not all(c in df.columns for c in [fast, slow]):
        return pd.Series(False, index=df.index)

    close = df['close']
    result = pd.Series(False, index=df.index)

    for i in range(lookback, len(df)):
        window = df.iloc[i - lookback:i + 1]
        wclose = window['close']
        wma = window[slow]

        if len(wclose) < lookback:
            continue

        # 整体下降趋势
        if wclose.iloc[-1] >= wclose.iloc[0]:
            continue

        # 大部分时间在均线下方
        below_ratio = (wclose < wma).sum() / len(wclose)
        if below_ratio < 0.6:
            continue

        # 检测波浪: 寻找局部高点(接触均线)
        touches = 0
        for j in range(5, len(wclose) - 1):
            dist_pct = (wclose.iloc[j] - wma.iloc[j]) / wma.iloc[j] * 100
            if -1.0 < dist_pct <= 0:
                touches += 1

        if touches >= min_waves:
            result.iloc[i] = True

    return result


def detect_head_up(df, fast='MA5', mid='MA10', slow='MA20'):
    """
    首次上穿(多头首次金叉): 快线首次上穿中线和慢线
    经过长期下跌后的首次金叉, 信号可靠度高

    Returns: bool Series
    """
    if not all(c in df.columns for c in [fast, mid, slow]):
        return pd.Series(False, index=df.index)

    # 快线上穿慢线
    cross = (df[fast] > df[slow]) & (df[fast].shift(1) <= df[slow].shift(1))

    # 前期为空头排列(至少20个bar)
    was_bearish = pd.Series(True, index=df.index)
    for shift in range(5, 25):
        was_bearish &= (df[fast].shift(shift) < df[slow].shift(shift))

    return cross & was_bearish


def detect_head_down(df, fast='MA5', mid='MA10', slow='MA20'):
    """
    首次下穿(空头首次死叉): 快线首次下穿中线和慢线
    经过长期上涨后的首次死叉, 信号可靠度高

    Returns: bool Series
    """
    if not all(c in df.columns for c in [fast, mid, slow]):
        return pd.Series(False, index=df.index)

    cross = (df[fast] < df[slow]) & (df[fast].shift(1) >= df[slow].shift(1))

    # 前期为多头排列
    was_bullish = pd.Series(True, index=df.index)
    for shift in range(5, 25):
        was_bullish &= (df[fast].shift(shift) > df[slow].shift(shift))

    return cross & was_bullish


def detect_ma_bond(df, ma_cols=None, bond_bars=10, threshold_pct=0.3):
    """
    均线粘合后发散(书中第三章重点):
    多条均线汇聚一处(粘合), 然后突然发散 → 大行情启动信号

    Returns:
        Series: 'bullish_break'(向上突破) / 'bearish_break'(向下突破) / 'bonding'(粘合中) / 'none'
    """
    if ma_cols is None:
        ma_cols = [c for c in ['MA5', 'MA10', 'MA20', 'MA30'] if c in df.columns]
    if len(ma_cols) < 3:
        return pd.Series('none', index=df.index)

    converged = detect_ma_convergence(df, ma_cols, threshold_pct=threshold_pct)

    # 检查前期是否有粘合
    was_bonding = pd.Series(False, index=df.index)
    for shift in range(1, bond_bars + 1):
        was_bonding |= converged.shift(shift).fillna(False)

    # 当前发散方向
    fastest = df[ma_cols[0]]
    slowest = df[ma_cols[-1]]
    spread = (fastest - slowest) / slowest * 100

    result = pd.Series('none', index=df.index)
    result[converged] = 'bonding'
    result[was_bonding & (spread > threshold_pct * 1.5)] = 'bullish_break'
    result[was_bonding & (spread < -threshold_pct * 1.5)] = 'bearish_break'

    return result


# ======================================================
#  综合信号评分系统
# ======================================================

def compute_ma_signals(df, timeframe='1h'):
    """
    综合均线信号分析

    Returns:
        dict: {
            'buy_score': float (-100~100),
            'sell_score': float (-100~100),
            'arrangement': str,
            'granville_rules': dict,
            'patterns': dict,
            'crosses': dict,
        }
    """
    periods = CRYPTO_MA_PERIODS.get(timeframe, MA_PERIODS)
    add_moving_averages(df, periods, timeframe)

    # 确定可用的均线列
    fast = 'MA5' if 'MA5' in df.columns else None
    mid = 'MA10' if 'MA10' in df.columns else None
    slow = 'MA20' if 'MA20' in df.columns else None
    long_ma = 'MA60' if 'MA60' in df.columns else None

    if not all([fast, mid, slow]):
        return {'buy_score': 0, 'sell_score': 0, 'arrangement': 'unknown',
                'granville_rules': {}, 'patterns': {}, 'crosses': {}}

    # 葛南维法则
    granville = granville_rules(df, ma_col=slow)

    # 均线排列
    arr_cols = [c for c in [fast, mid, slow, long_ma] if c is not None]
    arrangement = detect_ma_arrangement(df, arr_cols)

    # 交叉信号
    gc_5_20 = detect_golden_cross(df, fast, slow)
    dc_5_20 = detect_death_cross(df, fast, slow)
    gc_10_20 = detect_golden_cross(df, mid, slow) if mid else pd.Series(False, index=df.index)
    dc_10_20 = detect_death_cross(df, mid, slow) if mid else pd.Series(False, index=df.index)

    # 特殊形态
    silver = detect_silver_valley(df, fast, mid, slow)
    gold = detect_gold_valley(df, fast, mid, slow)
    death = detect_death_valley(df, fast, mid, slow)
    dragon = detect_dragon_emerge(df)
    cloud = detect_cloud_support(df, fast, mid, slow)
    bond = detect_ma_bond(df)
    convergence = detect_ma_convergence(df)
    head_up = detect_head_up(df, fast, mid, slow)
    head_down = detect_head_down(df, fast, mid, slow)

    # 综合评分
    buy_score = pd.Series(0.0, index=df.index)
    sell_score = pd.Series(0.0, index=df.index)

    # 葛南维法则
    buy_score += granville['signal_strength'].clip(lower=0)
    sell_score += (-granville['signal_strength']).clip(lower=0)

    # 排列
    buy_score[arrangement == 'bullish'] += 15
    sell_score[arrangement == 'bearish'] += 15

    # 交叉
    buy_score[gc_5_20] += 20
    sell_score[dc_5_20] += 20
    buy_score[gc_10_20] += 15
    sell_score[dc_10_20] += 15

    # 特殊形态
    buy_score[silver] += 25
    buy_score[gold] += 35
    sell_score[death] += 35
    buy_score[dragon] += 30
    buy_score[cloud] += 20
    buy_score[head_up] += 30
    sell_score[head_down] += 30
    buy_score[bond == 'bullish_break'] += 25
    sell_score[bond == 'bearish_break'] += 25

    # 粘合状态: 不给方向分, 但标记为"即将变盘"
    buy_score[convergence] += 5
    sell_score[convergence] += 5

    return {
        'buy_score': buy_score.clip(0, 100),
        'sell_score': sell_score.clip(0, 100),
        'arrangement': arrangement,
        'granville': granville,
        'crosses': {
            'gc_5_20': gc_5_20, 'dc_5_20': dc_5_20,
            'gc_10_20': gc_10_20, 'dc_10_20': dc_10_20,
        },
        'patterns': {
            'silver_valley': silver, 'gold_valley': gold,
            'death_valley': death, 'dragon_emerge': dragon,
            'cloud_support': cloud, 'head_up': head_up,
            'head_down': head_down, 'bond': bond,
            'convergence': convergence,
        },
    }


def get_latest_signals(df, signals, n=1):
    """获取最新N个bar的信号状态"""
    results = []
    for i in range(-n, 0):
        idx = df.index[i]
        buy = float(signals['buy_score'].iloc[i])
        sell = float(signals['sell_score'].iloc[i])
        arr = str(signals['arrangement'].iloc[i])

        active_patterns = []
        for name, series in signals['patterns'].items():
            val = series.iloc[i]
            if isinstance(val, bool) and val:
                active_patterns.append(name)
            elif isinstance(val, str) and val not in ('none', 'mixed'):
                active_patterns.append(f"{name}:{val}")

        active_rules = []
        gran = signals['granville']
        for r in ['rule_1', 'rule_2', 'rule_3', 'rule_4',
                   'rule_5', 'rule_6', 'rule_7', 'rule_8']:
            if gran[r].iloc[i]:
                active_rules.append(r)

        active_crosses = []
        for name, series in signals['crosses'].items():
            if series.iloc[i]:
                active_crosses.append(name)

        results.append({
            'time': str(idx),
            'price': float(df['close'].iloc[i]),
            'buy_score': buy,
            'sell_score': sell,
            'arrangement': arr,
            'patterns': active_patterns,
            'rules': active_rules,
            'crosses': active_crosses,
        })
    return results


# ======================================================
#  辅助: 信号时间序列提取(用于回测)
# ======================================================

def extract_signal_timeline(df, signals, min_score=15):
    """
    提取所有有效信号点, 用于回测引擎消费

    Returns:
        list of dict: [{'time': ..., 'type': 'buy'/'sell', 'score': ...,
                        'reasons': [...], 'price': ...}, ...]
    """
    timeline = []
    buy_scores = signals['buy_score']
    sell_scores = signals['sell_score']

    for i in range(len(df)):
        bs = float(buy_scores.iloc[i])
        ss = float(sell_scores.iloc[i])

        if bs >= min_score or ss >= min_score:
            reasons = []
            # 葛南维法则
            gran = signals['granville']
            rule_names = {
                'rule_1': '葛1:均线上穿', 'rule_2': '葛2:回调支撑',
                'rule_3': '葛3:假破位回升', 'rule_4': '葛4:超卖反弹',
                'rule_5': '葛5:超买回落', 'rule_6': '葛6:均线下穿',
                'rule_7': '葛7:反弹失败', 'rule_8': '葛8:均线压制',
            }
            for r, name in rule_names.items():
                if gran[r].iloc[i]:
                    reasons.append(name)

            # 特殊形态
            pattern_names = {
                'silver_valley': '银山谷', 'gold_valley': '金山谷',
                'death_valley': '死亡谷', 'dragon_emerge': '蛟龙出海',
                'cloud_support': '烘云托月', 'head_up': '首次上穿',
                'head_down': '首次下穿',
            }
            for p, name in pattern_names.items():
                val = signals['patterns'][p].iloc[i]
                if isinstance(val, bool) and val:
                    reasons.append(name)

            bond_val = signals['patterns']['bond'].iloc[i]
            if bond_val == 'bullish_break':
                reasons.append('粘合向上突破')
            elif bond_val == 'bearish_break':
                reasons.append('粘合向下突破')

            # 交叉
            cross_names = {
                'gc_5_20': '金叉5/20', 'dc_5_20': '死叉5/20',
                'gc_10_20': '金叉10/20', 'dc_10_20': '死叉10/20',
            }
            for c, name in cross_names.items():
                if signals['crosses'][c].iloc[i]:
                    reasons.append(name)

            # 排列
            arr = str(signals['arrangement'].iloc[i])
            if arr == 'bullish':
                reasons.append('多头排列')
            elif arr == 'bearish':
                reasons.append('空头排列')

            sig_type = 'buy' if bs > ss else 'sell'
            timeline.append({
                'time': df.index[i],
                'type': sig_type,
                'buy_score': bs,
                'sell_score': ss,
                'score': max(bs, ss),
                'price': float(df['close'].iloc[i]),
                'reasons': reasons,
            })

    return timeline
