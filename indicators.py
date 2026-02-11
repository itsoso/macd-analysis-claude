"""
技术指标计算模块
实现书中涉及的所有技术指标: MACD, KDJ, CCI, RSI, 均线等
"""

import numpy as np
import pandas as pd
import config as cfg


# ============================================================
# 移动平均线 (第二章第四/五节)
# ============================================================
def calc_ma(close: pd.Series, period: int) -> pd.Series:
    """计算简单移动平均线 (SMA)"""
    return close.rolling(window=period, min_periods=1).mean()


def calc_ema(close: pd.Series, period: int) -> pd.Series:
    """计算指数移动平均线 (EMA)"""
    return close.ewm(span=period, adjust=False).mean()


def add_ma_columns(df: pd.DataFrame) -> pd.DataFrame:
    """为DataFrame添加所有常用均线列"""
    periods = [cfg.MA_SHORT, cfg.MA_MID, cfg.MA_LONG, cfg.MA_LONG2,
               cfg.MA_LONG3, cfg.MA_LONG4]
    for p in periods:
        df[f'MA{p}'] = calc_ma(df['close'], p)
    return df


# ============================================================
# MACD 指标 (第三章)
# DIF(白线) = EMA(fast) - EMA(slow)
# DEA(黄线) = EMA(DIF, signal)
# BAR(彩柱线) = (DIF - DEA) * 2   (部分软件乘2)
# ============================================================
def calc_macd(close: pd.Series,
              fast: int = cfg.MACD_FAST,
              slow: int = cfg.MACD_SLOW,
              signal: int = cfg.MACD_SIGNAL) -> pd.DataFrame:
    """
    计算MACD指标
    返回包含 DIF, DEA, MACD_BAR 的DataFrame
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow                    # DIF线(白线)
    dea = dif.ewm(span=signal, adjust=False).mean()  # DEA线(黄线)
    bar = (dif - dea) * 2                        # 彩柱线(红绿柱)

    result = pd.DataFrame({
        'DIF': dif,
        'DEA': dea,
        'MACD_BAR': bar
    })
    return result


# ============================================================
# KDJ 指标 (第五章第一节)
# RSV = (close - low_n) / (high_n - low_n) * 100
# K = SMA(RSV, M1)
# D = SMA(K, M2)
# J = 3*K - 2*D
# ============================================================
def calc_kdj(high: pd.Series, low: pd.Series, close: pd.Series,
             n: int = cfg.KDJ_N,
             m1: int = cfg.KDJ_M1,
             m2: int = cfg.KDJ_M2) -> pd.DataFrame:
    """
    计算KDJ指标
    书中说明:
    - J线: 方向敏感线, >100超买, <0超卖
    - K线: 快速确认线, >90超买, <10超卖
    - D线: 慢速主干线, >80超买, <20超卖
    - 金叉: KD值在20以下, K上穿D
    - 死叉: KD值在80以上, K下穿D
    """
    low_n = low.rolling(window=n, min_periods=1).min()
    high_n = high.rolling(window=n, min_periods=1).max()

    rsv = (close - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)  # 处理除零

    # 使用递推方式计算K和D (模拟SMA)
    k = pd.Series(np.zeros(len(close)), index=close.index)
    d = pd.Series(np.zeros(len(close)), index=close.index)

    k.iloc[0] = 50
    d.iloc[0] = 50

    for i in range(1, len(close)):
        k.iloc[i] = (m1 - 1) / m1 * k.iloc[i - 1] + 1 / m1 * rsv.iloc[i]
        d.iloc[i] = (m2 - 1) / m2 * d.iloc[i - 1] + 1 / m2 * k.iloc[i]

    j = 3 * k - 2 * d

    return pd.DataFrame({'K': k, 'D': d, 'J': j})


# ============================================================
# CCI 指标 (第五章第二节)
# TP = (High + Low + Close) / 3
# CCI = (TP - SMA(TP, N)) / (0.015 * MeanDeviation)
# 春天线 +100, 秋天线 -100
# ============================================================
def calc_cci(high: pd.Series, low: pd.Series, close: pd.Series,
             n: int = cfg.CCI_N) -> pd.Series:
    """
    计算CCI指标
    书中说明:
    - +100以上为超买区(买方力量加强, 考虑买入 - 顺势)
    - -100以下为超卖区(卖方力量加强, 考虑卖出 - 顺势)
    - -100到+100之间为震荡区
    """
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=n, min_periods=1).mean()
    # 平均绝对偏差
    mad = tp.rolling(window=n, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci = (tp - sma_tp) / (0.015 * mad)
    cci = cci.fillna(0)
    return cci


# ============================================================
# RSI 指标 (第五章第三节)
# RSI = 上升平均数 / (上升平均数 + 下跌平均数) * 100
# ============================================================
def calc_rsi(close: pd.Series, period: int = cfg.RSI_SHORT) -> pd.Series:
    """
    计算RSI指标
    书中说明:
    - 80-100: 超强(超买), 考虑卖出
    - 50-80:  强势区域, 持有
    - 20-50:  弱势区域, 观望
    - 0-20:   超弱(超卖), 考虑买入
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    return rsi


# ============================================================
# 结构性顶底识别 (第二章: 峰线/谷线)
# 三根相邻K线, 中间K线高低点均为最高/最低 -> 峰线/谷线
# ============================================================
def find_peaks(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    识别结构性顶部(峰线)
    三根相邻K线, 中间K线的高点和低点都是三根中最高的
    返回布尔Series, True表示该位置是峰线
    """
    n = len(high)
    peaks = pd.Series(False, index=high.index)
    for i in range(1, n - 1):
        if (high.iloc[i] > high.iloc[i - 1] and
            high.iloc[i] > high.iloc[i + 1] and
            low.iloc[i] > low.iloc[i - 1] and
            low.iloc[i] > low.iloc[i + 1]):
            peaks.iloc[i] = True
    return peaks


def find_valleys(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    识别结构性底部(谷线)
    三根相邻K线, 中间K线的高点和低点都是三根中最低的
    返回布尔Series, True表示该位置是谷线
    """
    n = len(high)
    valleys = pd.Series(False, index=high.index)
    for i in range(1, n - 1):
        if (high.iloc[i] < high.iloc[i - 1] and
            high.iloc[i] < high.iloc[i + 1] and
            low.iloc[i] < low.iloc[i - 1] and
            low.iloc[i] < low.iloc[i + 1]):
            valleys.iloc[i] = True
    return valleys


def find_swing_highs(high: pd.Series, order: int = 5) -> pd.Series:
    """
    识别摆动高点 (更宽松的峰值识别, 用于趋势段划分)
    order: 高点两侧需要有多少根较低的K线
    """
    n = len(high)
    peaks = pd.Series(False, index=high.index)
    for i in range(order, n - order):
        is_peak = True
        for j in range(1, order + 1):
            if high.iloc[i] <= high.iloc[i - j] or high.iloc[i] <= high.iloc[i + j]:
                is_peak = False
                break
        if is_peak:
            peaks.iloc[i] = True
    return peaks


def find_swing_lows(low: pd.Series, order: int = 5) -> pd.Series:
    """
    识别摆动低点 (更宽松的谷值识别, 用于趋势段划分)
    """
    n = len(low)
    valleys = pd.Series(False, index=low.index)
    for i in range(order, n - order):
        is_valley = True
        for j in range(1, order + 1):
            if low.iloc[i] >= low.iloc[i - j] or low.iloc[i] >= low.iloc[i + j]:
                is_valley = False
                break
        if is_valley:
            valleys.iloc[i] = True
    return valleys


# ============================================================
# 趋势段划分工具
# ============================================================
def identify_trend_segments(df: pd.DataFrame, order: int = 5):
    """
    识别上涨段和下跌段
    返回: (up_segments, down_segments)
    每个segment: {'start_idx': int, 'end_idx': int, 'start_price': float, 'end_price': float,
                   'amplitude': float, 'duration': int}
    """
    swing_highs = find_swing_highs(df['high'], order)
    swing_lows = find_swing_lows(df['low'], order)

    # 收集所有拐点并排序
    points = []
    for i in swing_highs[swing_highs].index:
        idx = df.index.get_loc(i)
        points.append(('high', idx, df['high'].iloc[idx]))
    for i in swing_lows[swing_lows].index:
        idx = df.index.get_loc(i)
        points.append(('low', idx, df['low'].iloc[idx]))

    points.sort(key=lambda x: x[1])

    # 去除连续同向拐点, 只保留极值
    filtered = []
    for p in points:
        if not filtered:
            filtered.append(p)
        elif p[0] != filtered[-1][0]:
            filtered.append(p)
        else:
            # 同向: 高点取更高, 低点取更低
            if p[0] == 'high' and p[2] > filtered[-1][2]:
                filtered[-1] = p
            elif p[0] == 'low' and p[2] < filtered[-1][2]:
                filtered[-1] = p

    up_segments = []
    down_segments = []

    for i in range(1, len(filtered)):
        prev = filtered[i - 1]
        curr = filtered[i]
        segment = {
            'start_idx': prev[1],
            'end_idx': curr[1],
            'start_price': prev[2],
            'end_price': curr[2],
            'amplitude': abs(curr[2] - prev[2]),
            'duration': curr[1] - prev[1]
        }
        if prev[0] == 'low' and curr[0] == 'high':
            up_segments.append(segment)
        elif prev[0] == 'high' and curr[0] == 'low':
            down_segments.append(segment)

    return up_segments, down_segments


# ============================================================
# MACD 彩柱堆识别 (第三章第三节)
# ============================================================
def identify_bar_groups(bar: pd.Series):
    """
    识别MACD彩柱堆(连续同色的柱状线组)
    返回: list of dict, 每个dict包含:
        - 'type': 'red' 或 'green'
        - 'start_idx': 开始位置
        - 'end_idx': 结束位置
        - 'max_length': 最大柱线长度(绝对值)
        - 'area': 面积(绝对值之和)
        - 'max_bar_idx': 最长柱线位置
    """
    groups = []
    if len(bar) == 0:
        return groups

    current_type = 'red' if bar.iloc[0] >= 0 else 'green'
    start = 0

    for i in range(1, len(bar) + 1):
        if i < len(bar):
            new_type = 'red' if bar.iloc[i] >= 0 else 'green'
        else:
            new_type = None

        if new_type != current_type or i == len(bar):
            end = i - 1
            segment = bar.iloc[start:i]
            abs_segment = segment.abs()

            group = {
                'type': current_type,
                'start_idx': start,
                'end_idx': end,
                'max_length': abs_segment.max(),
                'area': abs_segment.sum(),
                'max_bar_idx': start + abs_segment.argmax()
            }
            groups.append(group)

            if i < len(bar):
                start = i
                current_type = new_type

    return groups


# ============================================================
# MACD 金叉/死叉识别 (第三章第一节)
# ============================================================
def find_macd_crosses(dif: pd.Series, dea: pd.Series):
    """
    识别MACD金叉和死叉
    金叉: DIF从下向上穿过DEA, 彩柱由绿转红
    死叉: DIF从上向下穿过DEA, 彩柱由红转绿

    返回: list of dict
        {'type': 'golden'/'death', 'idx': int, 'position': 'above_zero'/'below_zero',
         'dif': float, 'dea': float}
    """
    crosses = []
    for i in range(1, len(dif)):
        prev_diff = dif.iloc[i - 1] - dea.iloc[i - 1]
        curr_diff = dif.iloc[i] - dea.iloc[i]

        if prev_diff <= 0 and curr_diff > 0:
            cross_type = 'golden'
        elif prev_diff >= 0 and curr_diff < 0:
            cross_type = 'death'
        else:
            continue

        position = 'above_zero' if dif.iloc[i] > 0 else 'below_zero'

        crosses.append({
            'type': cross_type,
            'idx': i,
            'position': position,
            'dif': dif.iloc[i],
            'dea': dea.iloc[i]
        })

    return crosses


# ============================================================
# KDJ 金叉/死叉识别 (第五章第一节)
# ============================================================
def find_kdj_crosses(k: pd.Series, d: pd.Series):
    """
    识别KDJ金叉和死叉
    金叉: KD值在20以下, K上穿D
    死叉: KD值在80以上, K下穿D
    """
    crosses = []
    for i in range(1, len(k)):
        prev_diff = k.iloc[i - 1] - d.iloc[i - 1]
        curr_diff = k.iloc[i] - d.iloc[i]

        if prev_diff <= 0 and curr_diff > 0:
            # K上穿D
            if k.iloc[i] <= cfg.KDJ_OVERSOLD or d.iloc[i] <= cfg.KDJ_OVERSOLD:
                crosses.append({
                    'type': 'golden',
                    'idx': i,
                    'k': k.iloc[i],
                    'd': d.iloc[i],
                    'valid': True  # 在20以下的金叉才有效
                })
            else:
                crosses.append({
                    'type': 'golden',
                    'idx': i,
                    'k': k.iloc[i],
                    'd': d.iloc[i],
                    'valid': False
                })
        elif prev_diff >= 0 and curr_diff < 0:
            # K下穿D
            if k.iloc[i] >= cfg.KDJ_OVERBOUGHT or d.iloc[i] >= cfg.KDJ_OVERBOUGHT:
                crosses.append({
                    'type': 'death',
                    'idx': i,
                    'k': k.iloc[i],
                    'd': d.iloc[i],
                    'valid': True  # 在80以上的死叉才有效
                })
            else:
                crosses.append({
                    'type': 'death',
                    'idx': i,
                    'k': k.iloc[i],
                    'd': d.iloc[i],
                    'valid': False
                })

    return crosses


# ============================================================
# 辅助函数
# ============================================================
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """为DataFrame添加所有技术指标"""
    df = df.copy()

    # 均线
    df = add_ma_columns(df)

    # MACD
    macd_df = calc_macd(df['close'])
    df['DIF'] = macd_df['DIF']
    df['DEA'] = macd_df['DEA']
    df['MACD_BAR'] = macd_df['MACD_BAR']

    # KDJ
    kdj_df = calc_kdj(df['high'], df['low'], df['close'])
    df['K'] = kdj_df['K']
    df['D'] = kdj_df['D']
    df['J'] = kdj_df['J']

    # CCI
    df['CCI'] = calc_cci(df['high'], df['low'], df['close'])

    # RSI
    df['RSI6'] = calc_rsi(df['close'], cfg.RSI_SHORT)
    df['RSI12'] = calc_rsi(df['close'], cfg.RSI_LONG)

    return df
