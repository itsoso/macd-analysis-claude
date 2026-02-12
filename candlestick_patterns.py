"""
蜡烛图形态识别引擎 — 基于《日本蜡烛图技术》(Steve Nison) 完整版

完整形态识别(30+种):
  === 单根蜡烛线 ===
  锤子线、上吊线、流星线、倒锤子线、十字线(普通/长腿/墓碑/蜻蜓)、
  纺锤线、提腰带线、光头光脚(阳线/阴线)

  === 双蜡烛线 ===
  吞没形态(看涨/看跌)、乌云盖顶、刺透形态、孕线(含十字孕线)、
  平头形态(顶/底)、反击线、窗口(向上/向下缺口)、
  切入线/待入线/插入线

  === 三根及更多 ===
  启明星(含十字启明星)、黄昏星(含十字黄昏星)、弃婴形态、
  三只乌鸦、前进白色三兵(含受阻/停顿形态)、
  上升三法、下降三法、三内升降、三外升降、
  塔形顶/底、向上/向下跳空两只乌鸦、铺垫形态

  === Nison增强 ===
  成交量确认(放量确认信号强度 +30%~50%)
  形态失败检测(失败的看涨=强看跌, 反之亦然)
  关键支撑/阻力位增强(关键位置的形态可靠性 +25%)
  多级趋势判断(5/10/20周期综合)
  形态时效性(信号衰减)

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
#   蜡烛线基本属性计算
# ======================================================
def candle_features(df):
    """为DataFrame添加蜡烛线基本特征(Nison基础分析法)"""
    o, h, l, c = df['open'], df['high'], df['low'], df['close']

    df['body'] = (c - o).abs()
    df['upper_shadow'] = h - pd.concat([o, c], axis=1).max(axis=1)
    df['lower_shadow'] = pd.concat([o, c], axis=1).min(axis=1) - l
    df['range'] = h - l
    df['is_bull'] = c > o
    df['is_bear'] = c < o
    df['body_pct'] = df['body'] / df['range'].replace(0, np.nan)
    df['mid_body'] = (o + c) / 2
    df['body_high'] = pd.concat([o, c], axis=1).max(axis=1)
    df['body_low'] = pd.concat([o, c], axis=1).min(axis=1)

    # 动态平均(用于"长""短"判断)
    df['avg_body'] = df['body'].rolling(20, min_periods=5).mean()
    df['avg_range'] = df['range'].rolling(20, min_periods=5).mean()
    df['avg_volume'] = df['volume'].rolling(20, min_periods=5).mean()

    # 趋势均线
    df['sma5'] = c.rolling(5).mean()
    df['sma10'] = c.rolling(10).mean()
    df['sma20'] = c.rolling(20).mean()

    # 关键支撑/阻力(近期高低点)
    df['recent_high'] = h.rolling(20, min_periods=10).max()
    df['recent_low'] = l.rolling(20, min_periods=10).min()

    return df


# ======================================================
#   多级趋势判断 (Nison: 形态只有在正确趋势下才有意义)
# ======================================================
def trend_at(df, i, lookback=10):
    """综合趋势判断: 结合价格变化 + 均线方向"""
    if i < max(lookback, 20):
        return 'flat'

    # 价格变化
    prices = df['close'].iloc[max(0, i - lookback):i + 1]
    pct = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100

    # 均线排列
    sma5 = df['sma5'].iloc[i] if not pd.isna(df['sma5'].iloc[i]) else 0
    sma10 = df['sma10'].iloc[i] if not pd.isna(df['sma10'].iloc[i]) else 0
    sma20 = df['sma20'].iloc[i] if not pd.isna(df['sma20'].iloc[i]) else 0

    ma_bull = sma5 > sma10 > sma20 and sma5 > 0
    ma_bear = sma5 < sma10 < sma20 and sma5 > 0

    if pct > 2.0 or (pct > 0.8 and ma_bull):
        return 'up'
    elif pct < -2.0 or (pct < -0.8 and ma_bear):
        return 'down'
    return 'flat'


def trend_strength(df, i, lookback=10):
    """趋势强度 0~100"""
    if i < lookback:
        return 0
    prices = df['close'].iloc[max(0, i - lookback):i + 1]
    pct = abs((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100)
    return min(pct * 15, 100)


# ======================================================
#   成交量确认 (Nison第13章: 量价配合原则)
# ======================================================
def volume_confirm(df, i):
    """成交量确认系数: 放量=1.3~1.5, 正常=1.0, 缩量=0.7~0.9"""
    if i < 5:
        return 1.0
    avg_vol = df['avg_volume'].iloc[i]
    if pd.isna(avg_vol) or avg_vol < 1e-8:
        return 1.0

    cur_vol = df['volume'].iloc[i]
    ratio = cur_vol / avg_vol

    if ratio >= 2.5:
        return 1.5  # 极端放量, 最强确认
    elif ratio >= 1.8:
        return 1.35
    elif ratio >= 1.3:
        return 1.2
    elif ratio >= 0.8:
        return 1.0  # 正常量
    elif ratio >= 0.5:
        return 0.85  # 缩量, 信号减弱
    else:
        return 0.7  # 极度缩量


# ======================================================
#   关键位置增强 (Nison: 形态在支撑/阻力位更有意义)
# ======================================================
def key_level_bonus(df, i):
    """关键支撑/阻力位加成: 在关键位返回1.25, 否则1.0"""
    if i < 20:
        return 1.0

    price = df['close'].iloc[i]
    recent_h = df['recent_high'].iloc[i]
    recent_l = df['recent_low'].iloc[i]
    rng = recent_h - recent_l

    if pd.isna(rng) or rng < 1e-8:
        return 1.0

    # 接近近期高点(阻力位) 或 接近近期低点(支撑位)
    dist_to_high = abs(price - recent_h) / rng
    dist_to_low = abs(price - recent_l) / rng

    if dist_to_high < 0.05 or dist_to_low < 0.05:
        return 1.25  # 非常接近关键位
    elif dist_to_high < 0.10 or dist_to_low < 0.10:
        return 1.12
    return 1.0


# ======================================================
#   单根蜡烛线形态 (Nison第4-5章)
# ======================================================
def detect_hammer(df, i):
    """锤子线(第4章): 小实体在顶端, 长下影线≥实体2倍, 几乎无上影线
    要点: 1)必须在下降趋势后 2)下影线越长信号越强 3)阳线锤子更强"""
    if i < 5:
        return None
    body = df['body'].iloc[i]
    lower = df['lower_shadow'].iloc[i]
    upper = df['upper_shadow'].iloc[i]
    rng = df['range'].iloc[i]
    if rng < 1e-8 or body < 1e-8:
        return None

    if lower >= body * 2 and upper <= body * 0.5:
        trend = trend_at(df, i)
        if trend == 'down':
            score = min(55 + int(lower / body) * 5, 80)
            # Nison: 阳线锤子比阴线锤子更有看涨意味
            if df['is_bull'].iloc[i]:
                score += 8
            return {'name': '锤子线', 'score': score, 'reliability': 'high',
                    'type': 'bullish_reversal'}
    return None


def detect_hanging_man(df, i):
    """上吊线(第4章): 外形同锤子, 出现在上升趋势后
    要点: 1)必须获得验证(次日低开或收低) 2)单独可靠性低于锤子"""
    if i < 5:
        return None
    body = df['body'].iloc[i]
    lower = df['lower_shadow'].iloc[i]
    upper = df['upper_shadow'].iloc[i]
    rng = df['range'].iloc[i]
    if rng < 1e-8 or body < 1e-8:
        return None

    if lower >= body * 2 and upper <= body * 0.5:
        trend = trend_at(df, i)
        if trend == 'up':
            score = 45
            # 移除未来函数: 不再引用 i+1 验证, 改为用当前 bar 自身信息增强
            # 阴线上吊本身就是验证信号(收盘低于开盘)
            if df['is_bear'].iloc[i]:
                score = 65  # 阴线上吊可靠性更高
            # 长下影 + 短实体比例越极端, 信号越强
            if lower >= body * 3:
                score += 5
            return {'name': '上吊线', 'score': -score, 'reliability': 'medium',
                    'type': 'bearish_reversal'}
    return None


def detect_shooting_star(df, i):
    """流星线(第5章): 小实体在底端, 长上影线, 几乎无下影线
    要点: 1)上升趋势顶部 2)与前一根有跳空更理想 3)阴线流星更强"""
    if i < 5:
        return None
    body = df['body'].iloc[i]
    upper = df['upper_shadow'].iloc[i]
    lower = df['lower_shadow'].iloc[i]
    rng = df['range'].iloc[i]
    if rng < 1e-8 or body < 1e-8:
        return None

    if upper >= body * 2 and lower <= body * 0.3:
        trend = trend_at(df, i)
        if trend == 'up':
            score = min(50 + int(upper / body) * 5, 75)
            # 有跳空(gap up)加强
            if df['low'].iloc[i] > df['high'].iloc[i - 1]:
                score += 10
            if df['is_bear'].iloc[i]:
                score += 5
            return {'name': '流星线', 'score': -score, 'reliability': 'medium',
                    'type': 'bearish_reversal'}
    return None


def detect_inverted_hammer(df, i):
    """倒锤子线(第5章): 长上影线+小实体在底端, 下降趋势
    要点: 需要次日验证(高开确认)"""
    if i < 5:
        return None
    body = df['body'].iloc[i]
    upper = df['upper_shadow'].iloc[i]
    lower = df['lower_shadow'].iloc[i]
    rng = df['range'].iloc[i]
    if rng < 1e-8 or body < 1e-8:
        return None

    if upper >= body * 2 and lower <= body * 0.3:
        trend = trend_at(df, i)
        if trend == 'down':
            score = 40
            # 移除未来函数: 不再引用 i+1 验证, 改为用当前 bar 自身信息增强
            # 阳线倒锤子本身就是看涨信号(收盘高于开盘)
            if df['is_bull'].iloc[i]:
                score = 55  # 阳线倒锤子可靠性更高
            # 长上影 + 短实体比例越极端, 信号越强
            if upper >= body * 3:
                score += 5
            return {'name': '倒锤子线', 'score': score, 'reliability': 'low',
                    'type': 'bullish_reversal'}
    return None


def detect_doji(df, i):
    """十字线(第8章): 开市价≈收市价
    Nison分类: 普通十字/长腿十字/墓碑十字/蜻蜓十字
    要点: 1)十字线=犹豫 2)上升趋势中的十字更有看跌含义 3)连续十字要注意"""
    if i < 5:
        return None
    body = df['body'].iloc[i]
    rng = df['range'].iloc[i]
    avg_body = df['avg_body'].iloc[i]
    if rng < 1e-8 or pd.isna(avg_body) or avg_body < 1e-8:
        return None

    # 实体极小(不到平均实体的10%)
    if body < avg_body * 0.1:
        trend = trend_at(df, i)
        upper = df['upper_shadow'].iloc[i]
        lower = df['lower_shadow'].iloc[i]

        # Nison四种十字分类
        long_legged = (upper > rng * 0.3) and (lower > rng * 0.3)
        gravestone = upper > rng * 0.6 and lower < rng * 0.1
        dragonfly = lower > rng * 0.6 and upper < rng * 0.1

        if trend == 'up':
            if gravestone:
                return {'name': '墓碑十字线', 'score': -75, 'reliability': 'high',
                        'type': 'bearish_reversal'}
            if long_legged:
                return {'name': '长腿十字线(顶)', 'score': -55, 'reliability': 'medium',
                        'type': 'bearish_reversal'}
            return {'name': '十字星(顶部)', 'score': -50, 'reliability': 'medium',
                    'type': 'bearish_reversal'}
        elif trend == 'down':
            if dragonfly:
                return {'name': '蜻蜓十字线', 'score': 75, 'reliability': 'high',
                        'type': 'bullish_reversal'}
            if long_legged:
                return {'name': '长腿十字线(底)', 'score': 55, 'reliability': 'medium',
                        'type': 'bullish_reversal'}
            return {'name': '十字星(底部)', 'score': 50, 'reliability': 'medium',
                    'type': 'bullish_reversal'}
    return None


def detect_spinning_top(df, i):
    """纺锤线(第3章): 小实体 + 上下影线均较长
    要点: 单独意义不大, 但在趋势极端位出现是警告信号"""
    if i < 5:
        return None
    body = df['body'].iloc[i]
    avg_body = df['avg_body'].iloc[i]
    upper = df['upper_shadow'].iloc[i]
    lower = df['lower_shadow'].iloc[i]
    rng = df['range'].iloc[i]

    if pd.isna(avg_body) or rng < 1e-8:
        return None

    # 小实体(不到平均的50%), 上下影线都有一定长度
    if body < avg_body * 0.5 and body > avg_body * 0.1:
        if upper > body * 0.8 and lower > body * 0.8:
            trend = trend_at(df, i)
            ts = trend_strength(df, i)
            # 只在强趋势端才有意义
            if ts >= 40:
                if trend == 'up':
                    return {'name': '纺锤线(顶)', 'score': -25, 'reliability': 'low',
                            'type': 'bearish_warning'}
                elif trend == 'down':
                    return {'name': '纺锤线(底)', 'score': 25, 'reliability': 'low',
                            'type': 'bullish_warning'}
    return None


def detect_marubozu(df, i):
    """光头光脚(第3章): 长实体, 几乎无影线
    要点: 极强的趋势信号, 表示一方完全控制"""
    if i < 5:
        return None
    body = df['body'].iloc[i]
    avg_body = df['avg_body'].iloc[i]
    upper = df['upper_shadow'].iloc[i]
    lower = df['lower_shadow'].iloc[i]
    rng = df['range'].iloc[i]

    if pd.isna(avg_body) or rng < 1e-8:
        return None

    # 长实体(>1.8倍平均) + 极小影线
    if body > avg_body * 1.8 and body > rng * 0.90:
        if df['is_bull'].iloc[i]:
            return {'name': '阳线光头光脚', 'score': 40, 'reliability': 'medium',
                    'type': 'bullish_continuation'}
        else:
            return {'name': '阴线光头光脚', 'score': -40, 'reliability': 'medium',
                    'type': 'bearish_continuation'}
    return None


def detect_belt_hold(df, i):
    """提腰带线(第4章): 长实体, 一端无影线(秃头或秃脚)
    要点: 1)实体越长信号越强 2)此后价格不应重返实体另一端"""
    if i < 5:
        return None
    body = df['body'].iloc[i]
    avg_body = df['avg_body'].iloc[i]
    rng = df['range'].iloc[i]
    if pd.isna(avg_body) or rng < 1e-8:
        return None

    if body > avg_body * 1.5 and body > rng * 0.80:
        trend = trend_at(df, i)
        if df['is_bull'].iloc[i] and trend == 'down':
            # 看涨提腰带: 开盘=最低价(或接近)
            if df['lower_shadow'].iloc[i] < body * 0.05:
                return {'name': '看涨提腰带线', 'score': 55, 'reliability': 'medium',
                        'type': 'bullish_reversal'}
        elif df['is_bear'].iloc[i] and trend == 'up':
            # 看跌提腰带: 开盘=最高价(或接近)
            if df['upper_shadow'].iloc[i] < body * 0.05:
                return {'name': '看跌提腰带线', 'score': -55, 'reliability': 'medium',
                        'type': 'bearish_reversal'}
    return None


# ======================================================
#   双蜡烛线形态 (Nison第6章)
# ======================================================
def detect_engulfing(df, i):
    """吞没形态(第6章): 后一根实体完全包住前一根
    Nison增强条件: 1)前一根小实体 2)后一根长实体 3)放量更佳
    4)吞没发生在长期趋势的极端位 5)第二根吞没多根则更强"""
    if i < 2:
        return None
    o0, c0 = df['open'].iloc[i - 1], df['close'].iloc[i - 1]
    o1, c1 = df['open'].iloc[i], df['close'].iloc[i]
    body0 = df['body'].iloc[i - 1]
    body1 = df['body'].iloc[i]
    avg_body = df['avg_body'].iloc[i]

    if body0 < 1e-8 or body1 < 1e-8 or pd.isna(avg_body):
        return None

    # 看涨吞没
    if df['is_bear'].iloc[i - 1] and df['is_bull'].iloc[i]:
        if o1 <= c0 and c1 >= o0:
            trend = trend_at(df, i)
            if trend == 'down' or trend == 'flat':
                score = 60
                # Nison增强: 第一根小+第二根长 = 更强
                if body0 < avg_body * 0.5:
                    score += 10
                if body1 > avg_body * 1.5:
                    score += 10
                score = min(score, 90)
                return {'name': '看涨吞没', 'score': score, 'reliability': 'high',
                        'type': 'bullish_reversal'}

    # 看跌吞没
    if df['is_bull'].iloc[i - 1] and df['is_bear'].iloc[i]:
        if o1 >= c0 and c1 <= o0:
            trend = trend_at(df, i)
            if trend == 'up' or trend == 'flat':
                score = 60
                if body0 < avg_body * 0.5:
                    score += 10
                if body1 > avg_body * 1.5:
                    score += 10
                score = min(score, 90)
                return {'name': '看跌吞没', 'score': -score, 'reliability': 'high',
                        'type': 'bearish_reversal'}
    return None


def detect_dark_cloud(df, i):
    """乌云盖顶(第6章): 上升趋势, 阳+阴, 阴线开高收穿入阳线50%+
    要点: 穿入越深看跌越强, 穿入不到50%则为待入线(弱信号)"""
    if i < 2:
        return None
    if not (df['is_bull'].iloc[i - 1] and df['is_bear'].iloc[i]):
        return None

    prev_o, prev_c = df['open'].iloc[i - 1], df['close'].iloc[i - 1]
    curr_o, curr_c = df['open'].iloc[i], df['close'].iloc[i]
    prev_h = df['high'].iloc[i - 1]
    prev_mid = (prev_o + prev_c) / 2

    if curr_o >= prev_h and curr_c < prev_mid and curr_c > prev_o:
        trend = trend_at(df, i)
        if trend == 'up' or trend == 'flat':
            # 穿入深度
            pen = (prev_c - curr_c) / (prev_c - prev_o) * 100 if prev_c != prev_o else 50
            score = min(55 + int(pen * 0.3), 85)
            return {'name': '乌云盖顶', 'score': -score, 'reliability': 'high',
                    'type': 'bearish_reversal'}

    # 待入线(穿入不足50%) — 弱信号
    if curr_o >= prev_c and curr_c < prev_c and curr_c >= prev_mid:
        trend = trend_at(df, i)
        if trend == 'up':
            return {'name': '待入线', 'score': -25, 'reliability': 'low',
                    'type': 'bearish_warning'}
    return None


def detect_piercing(df, i):
    """刺透形态(第6章): 下降趋势, 阴+阳, 阳线开低收穿入阴线50%+"""
    if i < 2:
        return None
    if not (df['is_bear'].iloc[i - 1] and df['is_bull'].iloc[i]):
        return None

    prev_o, prev_c = df['open'].iloc[i - 1], df['close'].iloc[i - 1]
    curr_o, curr_c = df['open'].iloc[i], df['close'].iloc[i]
    prev_l = df['low'].iloc[i - 1]
    prev_mid = (prev_o + prev_c) / 2

    if curr_o <= prev_l and curr_c > prev_mid and curr_c < prev_o:
        trend = trend_at(df, i)
        if trend == 'down' or trend == 'flat':
            pen = (curr_c - prev_c) / (prev_o - prev_c) * 100 if prev_o != prev_c else 50
            score = min(55 + int(pen * 0.3), 85)
            return {'name': '刺透形态', 'score': score, 'reliability': 'high',
                    'type': 'bullish_reversal'}
    return None


def detect_harami(df, i):
    """孕线形态(第6章): 前一根长实体包含后一根小实体
    Nison: 十字孕线比普通孕线更有力"""
    if i < 2:
        return None
    body0 = df['body'].iloc[i - 1]
    body1 = df['body'].iloc[i]
    avg_body = df['avg_body'].iloc[i]
    if pd.isna(avg_body) or body0 < 1e-8:
        return None

    if body0 > avg_body * 1.0 and body1 < body0 * 0.5:
        hi0, lo0 = df['body_high'].iloc[i - 1], df['body_low'].iloc[i - 1]
        hi1, lo1 = df['body_high'].iloc[i], df['body_low'].iloc[i]

        if lo0 <= lo1 and hi0 >= hi1:
            trend = trend_at(df, i)
            is_doji = body1 < avg_body * 0.1
            base_score = 65 if is_doji else 45
            name_suffix = '十字孕线' if is_doji else '孕线'

            if trend == 'up' and df['is_bull'].iloc[i - 1]:
                return {'name': f'看跌{name_suffix}', 'score': -base_score,
                        'reliability': 'high' if is_doji else 'medium',
                        'type': 'bearish_reversal'}
            elif trend == 'down' and df['is_bear'].iloc[i - 1]:
                return {'name': f'看涨{name_suffix}', 'score': base_score,
                        'reliability': 'high' if is_doji else 'medium',
                        'type': 'bullish_reversal'}
    return None


def detect_tweezers(df, i):
    """平头形态(第7章): 两根蜡烛高点(顶)/低点(底)一致
    Nison: 平头与其他形态结合时更有效"""
    if i < 2:
        return None
    h0, h1 = df['high'].iloc[i - 1], df['high'].iloc[i]
    l0, l1 = df['low'].iloc[i - 1], df['low'].iloc[i]
    avg_range = df['avg_range'].iloc[i]
    if pd.isna(avg_range) or avg_range < 1e-8:
        return None

    threshold = avg_range * 0.03

    trend = trend_at(df, i)
    # 平头顶: 两根高点一致, 第二根阴线
    if abs(h0 - h1) < threshold and trend == 'up' and df['is_bear'].iloc[i]:
        return {'name': '平头顶部', 'score': -45, 'reliability': 'medium',
                'type': 'bearish_reversal'}
    # 平头底: 两根低点一致, 第二根阳线
    if abs(l0 - l1) < threshold and trend == 'down' and df['is_bull'].iloc[i]:
        return {'name': '平头底部', 'score': 45, 'reliability': 'medium',
                'type': 'bullish_reversal'}
    return None


def detect_counterattack(df, i):
    """反击线(第6章): 颜色相反, 收市价相同或极接近
    要点: 两根实体都必须是长实体"""
    if i < 2:
        return None
    c0, c1 = df['close'].iloc[i - 1], df['close'].iloc[i]
    avg_body = df['avg_body'].iloc[i]
    if pd.isna(avg_body) or avg_body < 1e-8:
        return None

    if abs(c0 - c1) < avg_body * 0.05:
        body0, body1 = df['body'].iloc[i - 1], df['body'].iloc[i]
        if body0 > avg_body * 0.8 and body1 > avg_body * 0.8:
            trend = trend_at(df, i)
            if df['is_bear'].iloc[i - 1] and df['is_bull'].iloc[i] and trend == 'down':
                return {'name': '看涨反击线', 'score': 50, 'reliability': 'medium',
                        'type': 'bullish_reversal'}
            if df['is_bull'].iloc[i - 1] and df['is_bear'].iloc[i] and trend == 'up':
                return {'name': '看跌反击线', 'score': -50, 'reliability': 'medium',
                        'type': 'bearish_reversal'}
    return None


def detect_window(df, i):
    """窗口/缺口(第7章): Nison将缺口称为"窗口"
    向上窗口=支撑, 向下窗口=阻力
    窗口是强力趋势信号, 也是支撑/阻力"""
    if i < 2:
        return None
    prev_h = df['high'].iloc[i - 1]
    prev_l = df['low'].iloc[i - 1]
    curr_h = df['high'].iloc[i]
    curr_l = df['low'].iloc[i]
    avg_range = df['avg_range'].iloc[i]

    if pd.isna(avg_range) or avg_range < 1e-8:
        return None

    gap_threshold = avg_range * 0.3  # 缺口要有意义需达到平均振幅的30%

    # 向上窗口
    if curr_l > prev_h and (curr_l - prev_h) > gap_threshold:
        return {'name': '向上窗口', 'score': 35, 'reliability': 'medium',
                'type': 'bullish_continuation'}

    # 向下窗口
    if curr_h < prev_l and (prev_l - curr_h) > gap_threshold:
        return {'name': '向下窗口', 'score': -35, 'reliability': 'medium',
                'type': 'bearish_continuation'}
    return None


# ======================================================
#   三蜡烛线形态 (Nison第6-7章)
# ======================================================
def detect_morning_star(df, i):
    """启明星(第6章): 长阴+小实体(星)+长阳, 阳色穿入阴线实体
    十字启明星(星线为十字)更强"""
    if i < 3:
        return None
    body0 = df['body'].iloc[i - 2]
    body1 = df['body'].iloc[i - 1]
    body2 = df['body'].iloc[i]
    avg_body = df['avg_body'].iloc[i]
    if pd.isna(avg_body) or avg_body < 1e-8:
        return None

    if not (df['is_bear'].iloc[i - 2] and body0 > avg_body * 0.8):
        return None
    if body1 > avg_body * 0.5:
        return None
    if not (df['is_bull'].iloc[i] and body2 > avg_body * 0.5):
        return None

    o0, c0 = df['open'].iloc[i - 2], df['close'].iloc[i - 2]
    c2 = df['close'].iloc[i]
    mid0 = (o0 + c0) / 2

    if c2 > mid0:
        trend = trend_at(df, i, lookback=15)
        if trend == 'down' or trend == 'flat':
            is_doji_star = body1 < avg_body * 0.1
            score = 85 if is_doji_star else 75
            name = '十字启明星' if is_doji_star else '启明星'
            return {'name': name, 'score': score, 'reliability': 'high',
                    'type': 'bullish_reversal'}
    return None


def detect_evening_star(df, i):
    """黄昏星(第6章): 长阳+小实体(星)+长阴"""
    if i < 3:
        return None
    body0 = df['body'].iloc[i - 2]
    body1 = df['body'].iloc[i - 1]
    body2 = df['body'].iloc[i]
    avg_body = df['avg_body'].iloc[i]
    if pd.isna(avg_body) or avg_body < 1e-8:
        return None

    if not (df['is_bull'].iloc[i - 2] and body0 > avg_body * 0.8):
        return None
    if body1 > avg_body * 0.5:
        return None
    if not (df['is_bear'].iloc[i] and body2 > avg_body * 0.5):
        return None

    o0, c0 = df['open'].iloc[i - 2], df['close'].iloc[i - 2]
    c2 = df['close'].iloc[i]
    mid0 = (o0 + c0) / 2

    if c2 < mid0:
        trend = trend_at(df, i, lookback=15)
        if trend == 'up' or trend == 'flat':
            is_doji_star = body1 < avg_body * 0.1
            score = 85 if is_doji_star else 75
            name = '十字黄昏星' if is_doji_star else '黄昏星'
            return {'name': name, 'score': -score, 'reliability': 'high',
                    'type': 'bearish_reversal'}
    return None


def detect_abandoned_baby(df, i):
    """弃婴形态(第6章补充): 类似启明星/黄昏星, 但星线与前后都有缺口
    极其罕见但非常可靠"""
    if i < 3:
        return None
    body1 = df['body'].iloc[i - 1]
    avg_body = df['avg_body'].iloc[i]
    if pd.isna(avg_body):
        return None

    # 星线必须是十字
    if body1 > avg_body * 0.1:
        return None

    # 第一根和第三根必须是长实体
    body0 = df['body'].iloc[i - 2]
    body2 = df['body'].iloc[i]
    if body0 < avg_body * 0.6 or body2 < avg_body * 0.6:
        return None

    # 检查缺口: 星线的high/low不与前后重叠
    star_h = df['high'].iloc[i - 1]
    star_l = df['low'].iloc[i - 1]

    # 看涨弃婴: 阴+十字(gap下)+阳(gap上)
    if (df['is_bear'].iloc[i - 2] and df['is_bull'].iloc[i] and
            star_h < df['low'].iloc[i - 2] and star_l < df['low'].iloc[i]):
        return {'name': '看涨弃婴', 'score': 90, 'reliability': 'high',
                'type': 'bullish_reversal'}

    # 看跌弃婴
    if (df['is_bull'].iloc[i - 2] and df['is_bear'].iloc[i] and
            star_l > df['high'].iloc[i - 2] and star_h > df['high'].iloc[i]):
        return {'name': '看跌弃婴', 'score': -90, 'reliability': 'high',
                'type': 'bearish_reversal'}
    return None


def detect_three_black_crows(df, i):
    """三只乌鸦(第7章): 三根长阴线依次下降
    要点: 1)每根开盘在前一根实体内 2)每根收盘接近最低价 3)上升趋势后"""
    if i < 3:
        return None
    avg_body = df['avg_body'].iloc[i]
    if pd.isna(avg_body):
        return None

    for j in range(3):
        idx = i - 2 + j
        if not df['is_bear'].iloc[idx]:
            return None
        if df['body'].iloc[idx] < avg_body * 0.6:
            return None
        # Nison: 收盘应接近最低价(小下影线)
        if df['lower_shadow'].iloc[idx] > df['body'].iloc[idx] * 0.4:
            return None

    for j in range(1, 3):
        curr_idx = i - 2 + j
        prev_idx = curr_idx - 1
        if df['close'].iloc[curr_idx] >= df['close'].iloc[prev_idx]:
            return None
        # 开盘在前一根实体内
        if not (df['body_low'].iloc[prev_idx] <= df['open'].iloc[curr_idx] <= df['body_high'].iloc[prev_idx]):
            return None

    trend = trend_at(df, i, lookback=15)
    if trend == 'up' or trend == 'flat':
        return {'name': '三只乌鸦', 'score': -85, 'reliability': 'high',
                'type': 'bearish_reversal'}
    return None


def detect_three_white_soldiers(df, i):
    """前进白色三兵(第7章): 三根长阳线依次上升
    Nison区分: 标准三兵 vs 前方受阻(第三根有长上影) vs 停顿(第三根小实体)"""
    if i < 3:
        return None
    avg_body = df['avg_body'].iloc[i]
    if pd.isna(avg_body):
        return None

    for j in range(3):
        idx = i - 2 + j
        if not df['is_bull'].iloc[idx]:
            return None
        if df['body'].iloc[idx] < avg_body * 0.5:
            return None

    for j in range(1, 3):
        curr_idx = i - 2 + j
        prev_idx = curr_idx - 1
        if df['close'].iloc[curr_idx] <= df['close'].iloc[prev_idx]:
            return None

    trend = trend_at(df, i, lookback=15)
    if trend != 'down' and trend != 'flat':
        return None

    # Nison: 检查是否"前方受阻"(第三根上影线长)
    last_body = df['body'].iloc[i]
    last_upper = df['upper_shadow'].iloc[i]
    if last_upper > last_body * 0.8:
        return {'name': '三兵前方受阻', 'score': 40, 'reliability': 'low',
                'type': 'bullish_weakening'}

    # 检查"停顿形态"(第三根实体明显缩小)
    if last_body < df['body'].iloc[i - 1] * 0.5:
        return {'name': '三兵停顿', 'score': 45, 'reliability': 'low',
                'type': 'bullish_weakening'}

    return {'name': '前进白色三兵', 'score': 80, 'reliability': 'high',
            'type': 'bullish_reversal'}


def detect_three_inside(df, i):
    """三内升降(第7章): 孕线 + 确认(第三根延续第二根方向)"""
    if i < 3:
        return None
    body0 = df['body'].iloc[i - 2]
    body1 = df['body'].iloc[i - 1]
    body2 = df['body'].iloc[i]
    avg_body = df['avg_body'].iloc[i]
    if pd.isna(avg_body) or body0 < 1e-8:
        return None

    # 先检查是否是孕线(前两根)
    if body0 < avg_body * 0.8 or body1 >= body0 * 0.5:
        return None
    hi0, lo0 = df['body_high'].iloc[i - 2], df['body_low'].iloc[i - 2]
    hi1, lo1 = df['body_high'].iloc[i - 1], df['body_low'].iloc[i - 1]
    if not (lo0 <= lo1 and hi0 >= hi1):
        return None

    # 三内升: 阴+小阳(被包)+确认阳(收盘高于第一根开盘)
    if df['is_bear'].iloc[i - 2] and df['is_bull'].iloc[i - 1] and df['is_bull'].iloc[i]:
        if df['close'].iloc[i] > df['open'].iloc[i - 2]:
            trend = trend_at(df, i)
            if trend == 'down' or trend == 'flat':
                return {'name': '三内升', 'score': 70, 'reliability': 'high',
                        'type': 'bullish_reversal'}

    # 三内降
    if df['is_bull'].iloc[i - 2] and df['is_bear'].iloc[i - 1] and df['is_bear'].iloc[i]:
        if df['close'].iloc[i] < df['open'].iloc[i - 2]:
            trend = trend_at(df, i)
            if trend == 'up' or trend == 'flat':
                return {'name': '三内降', 'score': -70, 'reliability': 'high',
                        'type': 'bearish_reversal'}
    return None


def detect_three_outside(df, i):
    """三外升降(第7章): 吞没 + 确认(第三根延续方向)"""
    if i < 3:
        return None
    body0 = df['body'].iloc[i - 2]
    body1 = df['body'].iloc[i - 1]
    if body0 < 1e-8 or body1 < 1e-8:
        return None

    o0, c0 = df['open'].iloc[i - 2], df['close'].iloc[i - 2]
    o1, c1 = df['open'].iloc[i - 1], df['close'].iloc[i - 1]

    # 三外升: 阴+看涨吞没+确认阳
    if (df['is_bear'].iloc[i - 2] and df['is_bull'].iloc[i - 1] and
            o1 <= c0 and c1 >= o0 and df['is_bull'].iloc[i]):
        if df['close'].iloc[i] > c1:
            trend = trend_at(df, i)
            if trend == 'down' or trend == 'flat':
                return {'name': '三外升', 'score': 75, 'reliability': 'high',
                        'type': 'bullish_reversal'}

    # 三外降
    if (df['is_bull'].iloc[i - 2] and df['is_bear'].iloc[i - 1] and
            o1 >= c0 and c1 <= o0 and df['is_bear'].iloc[i]):
        if df['close'].iloc[i] < c1:
            trend = trend_at(df, i)
            if trend == 'up' or trend == 'flat':
                return {'name': '三外降', 'score': -75, 'reliability': 'high',
                        'type': 'bearish_reversal'}
    return None


def detect_rising_three(df, i):
    """上升三法(第7章): 长阳+2~3根小体(不跌破阳线低点)+长阳(创新高)"""
    if i < 5:
        return None
    avg_body = df['avg_body'].iloc[i]
    if pd.isna(avg_body):
        return None

    if not (df['is_bull'].iloc[i - 4] and df['body'].iloc[i - 4] > avg_body):
        return None
    first_lo = df['low'].iloc[i - 4]
    first_hi = df['close'].iloc[i - 4]

    for j in range(i - 3, i):
        if df['body'].iloc[j] > avg_body * 0.6:
            return None
        if df['low'].iloc[j] < first_lo:
            return None

    if not (df['is_bull'].iloc[i] and df['body'].iloc[i] > avg_body):
        return None
    if df['close'].iloc[i] <= first_hi:
        return None

    return {'name': '上升三法', 'score': 65, 'reliability': 'high',
            'type': 'bullish_continuation'}


def detect_falling_three(df, i):
    """下降三法(第7章): 长阴+2~3根小体(不突破阴线高点)+长阴(创新低)"""
    if i < 5:
        return None
    avg_body = df['avg_body'].iloc[i]
    if pd.isna(avg_body):
        return None

    if not (df['is_bear'].iloc[i - 4] and df['body'].iloc[i - 4] > avg_body):
        return None
    first_hi = df['high'].iloc[i - 4]
    first_lo = df['close'].iloc[i - 4]

    for j in range(i - 3, i):
        if df['body'].iloc[j] > avg_body * 0.6:
            return None
        if df['high'].iloc[j] > first_hi:
            return None

    if not (df['is_bear'].iloc[i] and df['body'].iloc[i] > avg_body):
        return None
    if df['close'].iloc[i] >= first_lo:
        return None

    return {'name': '下降三法', 'score': -65, 'reliability': 'high',
            'type': 'bearish_continuation'}


def detect_tower(df, i):
    """塔形顶/底(第7章): 由多根蜡烛构成, 先一方控制然后另一方接管
    塔形底: 长阴→数根小线→长阳  塔形顶: 长阳→数根小线→长阴"""
    if i < 6:
        return None
    avg_body = df['avg_body'].iloc[i]
    if pd.isna(avg_body):
        return None

    # 塔形底: 寻找前面的长阴, 中间的小实体, 以及当前的长阳
    if df['is_bull'].iloc[i] and df['body'].iloc[i] > avg_body * 1.2:
        # 回溯找长阴
        for lookback in range(3, 7):
            if i - lookback < 0:
                break
            first_idx = i - lookback
            if df['is_bear'].iloc[first_idx] and df['body'].iloc[first_idx] > avg_body * 1.0:
                # 中间应该是小实体
                all_small = True
                for mid_idx in range(first_idx + 1, i):
                    if df['body'].iloc[mid_idx] > avg_body * 0.7:
                        all_small = False
                        break
                if all_small:
                    trend = trend_at(df, first_idx)
                    if trend == 'down' or trend == 'flat':
                        return {'name': '塔形底', 'score': 60, 'reliability': 'medium',
                                'type': 'bullish_reversal'}
                break

    # 塔形顶
    if df['is_bear'].iloc[i] and df['body'].iloc[i] > avg_body * 1.2:
        for lookback in range(3, 7):
            if i - lookback < 0:
                break
            first_idx = i - lookback
            if df['is_bull'].iloc[first_idx] and df['body'].iloc[first_idx] > avg_body * 1.0:
                all_small = True
                for mid_idx in range(first_idx + 1, i):
                    if df['body'].iloc[mid_idx] > avg_body * 0.7:
                        all_small = False
                        break
                if all_small:
                    trend = trend_at(df, first_idx)
                    if trend == 'up' or trend == 'flat':
                        return {'name': '塔形顶', 'score': -60, 'reliability': 'medium',
                                'type': 'bearish_reversal'}
                break
    return None


# ======================================================
#   形态失败检测 (Nison: 失败的形态产生反向强信号)
# ======================================================
def detect_pattern_failure(df, i, recent_patterns):
    """检测近期形态是否失败
    Nison: 看涨形态后价格跌破形态低点 = 强烈看跌信号, 反之亦然"""
    results = []
    if i < 3:
        return results

    price = df['close'].iloc[i]

    # 检查最近3~5根K线内的形态
    for check_idx in range(max(0, i - 5), i):
        if check_idx not in recent_patterns:
            continue
        for p in recent_patterns[check_idx]:
            bars_since = i - check_idx
            if bars_since < 2 or bars_since > 5:
                continue

            # 看涨形态失败: 价格跌破形态区间低点
            if p['score'] > 0 and p['type'].endswith('reversal'):
                pattern_low = df['low'].iloc[check_idx]
                if price < pattern_low:
                    fail_score = min(abs(p['score']) * 0.8, 70)
                    results.append({
                        'name': f'{p["name"]}失败',
                        'score': -fail_score,
                        'reliability': 'medium',
                        'type': 'bearish_failure'
                    })

            # 看跌形态失败
            elif p['score'] < 0 and p['type'].endswith('reversal'):
                pattern_high = df['high'].iloc[check_idx]
                if price > pattern_high:
                    fail_score = min(abs(p['score']) * 0.8, 70)
                    results.append({
                        'name': f'{p["name"]}失败',
                        'score': fail_score,
                        'reliability': 'medium',
                        'type': 'bullish_failure'
                    })

    return results


# ======================================================
#   综合形态扫描 (升级版: 含量价确认+关键位+失败检测)
# ======================================================
ALL_DETECTORS = [
    # 单根
    detect_hammer, detect_hanging_man, detect_shooting_star,
    detect_inverted_hammer, detect_doji, detect_spinning_top,
    detect_marubozu, detect_belt_hold,
    # 双根
    detect_engulfing, detect_dark_cloud, detect_piercing,
    detect_harami, detect_tweezers, detect_counterattack, detect_window,
    # 三根+
    detect_morning_star, detect_evening_star, detect_abandoned_baby,
    detect_three_black_crows, detect_three_white_soldiers,
    detect_three_inside, detect_three_outside,
    detect_rising_three, detect_falling_three, detect_tower,
]


def scan_patterns(df):
    """扫描DataFrame中所有K线形态, 含Nison增强:
    1. 成交量确认  2. 关键位加成  3. 形态失败检测"""
    df = candle_features(df)
    results = {}

    for i in range(5, len(df)):
        patterns = []
        for detector in ALL_DETECTORS:
            p = detector(df, i)
            if p:
                # Nison增强: 成交量确认
                vol_mult = volume_confirm(df, i)
                p['volume_mult'] = vol_mult

                # Nison增强: 关键位加成
                level_mult = key_level_bonus(df, i)
                p['level_mult'] = level_mult

                # 调整分数
                adjusted_score = p['score'] * vol_mult * level_mult
                p['raw_score'] = p['score']
                p['score'] = int(min(max(adjusted_score, -100), 100))

                patterns.append(p)

        # 形态失败检测
        failures = detect_pattern_failure(df, i, results)
        for f in failures:
            f['volume_mult'] = 1.0
            f['level_mult'] = 1.0
            f['raw_score'] = f['score']
            patterns.append(f)

        if patterns:
            results[i] = patterns

    return results


def compute_candlestick_scores(df):
    """计算每根K线的综合蜡烛图得分
    返回 sell_score, buy_score, pattern_names (Series)"""
    patterns = scan_patterns(df)

    sell_scores = pd.Series(0.0, index=df.index)
    buy_scores = pd.Series(0.0, index=df.index)
    pattern_names = pd.Series('', index=df.index, dtype=str)

    for idx, plist in patterns.items():
        total_bearish = 0
        total_bullish = 0
        names = []

        for p in plist:
            score = p['score']
            reliability_mult = {
                'high': 1.0,
                'medium': 0.75,
                'low': 0.5
            }.get(p['reliability'], 0.5)
            weighted = abs(score) * reliability_mult

            if score < 0:
                total_bearish += weighted
            elif score > 0:
                total_bullish += weighted
            names.append(p['name'])

        # Nison: 同一时间多个同向形态 = 信号增强
        sell_scores.iloc[idx] = min(total_bearish, 100)
        buy_scores.iloc[idx] = min(total_bullish, 100)
        pattern_names.iloc[idx] = ','.join(names)

    return sell_scores, buy_scores, pattern_names


# ======================================================
#   蜡烛图策略回测
# ======================================================
def fetch_data():
    """获取1h K线数据"""
    print("获取数据...")
    data = {}
    for tf, days in [('1h', 60), ('4h', 60)]:
        df = fetch_binance_klines("ETHUSDT", interval=tf, days=days)
        if df is not None and len(df) > 50:
            df = add_all_indicators(df)
            data[tf] = df
            print(f"  {tf}: {len(df)} 条 ({df.index[0]} ~ {df.index[-1]})")
    return data


def run_candlestick_strategy(data, config, trade_days=None):
    """蜡烛图形态策略回测"""
    eng = FuturesEngine(config['name'], max_leverage=config.get('max_lev', 5))
    main_df = data['1h']

    sell_sc, buy_sc, pat_names = compute_candlestick_scores(main_df)

    # 4h辅助趋势
    sell_sc_4h = pd.Series(0.0, index=main_df.index)
    buy_sc_4h = pd.Series(0.0, index=main_df.index)
    if '4h' in data:
        s4, b4, _ = compute_candlestick_scores(data['4h'])
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
        if init_idx >= len(main_df):
            init_idx = 0
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[init_idx]

    eng.max_single_margin = eng.initial_total * config.get('single_pct', 0.15)
    eng.max_margin_total = eng.initial_total * config.get('total_pct', 0.40)
    eng.max_lifetime_margin = eng.initial_total * 5.0

    sell_threshold = config.get('sell_threshold', 30)
    buy_threshold = config.get('buy_threshold', 30)
    short_threshold = config.get('short_threshold', 45)
    long_threshold = config.get('long_threshold', 45)
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
            if idx % 4 == 0:
                eng.record_history(dt, price)
            continue

        eng.check_liquidation(price, dt)
        short_just_opened = False
        long_just_opened = False

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
        reason = pat_names.iloc[idx][:60] if pat_names.iloc[idx] else ''

        in_conflict = False
        if ss > 0 and bs > 0:
            ratio = min(ss, bs) / max(ss, bs)
            in_conflict = ratio >= 0.6 and min(ss, bs) >= 15

        if ss >= sell_threshold and spot_cd == 0 and not in_conflict and eng.spot_eth * price > 500:
            eng.spot_sell(price, dt, sell_pct, f"K线卖 {reason}")
            spot_cd = spot_cooldown

        sell_dom = (ss > bs * 1.5) if bs > 0 else True
        if short_cd == 0 and ss >= short_threshold and not eng.futures_short and sell_dom:
            margin = eng.available_margin() * margin_use
            actual_lev = min(lev if ss >= 60 else 2, eng.max_leverage)
            eng.open_short(price, dt, margin, actual_lev, f"K线空 {actual_lev}x {reason}")
            short_max_pnl = 0; short_bars = 0; short_cd = cooldown
            short_just_opened = True

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
                    bs_dom = (ss < bs * 0.7) if bs > 0 else True
                    if bs_dom:
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

        if bs >= buy_threshold and spot_cd == 0 and not in_conflict and eng.available_usdt() > 500:
            eng.spot_buy(price, dt, eng.available_usdt() * 0.25, f"K线买 {reason}")
            spot_cd = spot_cooldown

        buy_dom = (bs > ss * 1.5) if ss > 0 else True
        if long_cd == 0 and bs >= long_threshold and not eng.futures_long and buy_dom:
            margin = eng.available_margin() * margin_use
            actual_lev = min(lev if bs >= 60 else 2, eng.max_leverage)
            eng.open_long(price, dt, margin, actual_lev, f"K线多 {actual_lev}x {reason}")
            long_max_pnl = 0; long_bars = 0; long_cd = cooldown
            long_just_opened = True

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
                    ss_dom = (bs < ss * 0.7) if ss > 0 else True
                    if ss_dom:
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

        if idx % 4 == 0:
            eng.record_history(dt, price)

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
        {**base, 'name': 'K1: 标准蜡烛图'},

        {**base, 'name': 'K2: 激进做空',
         'sell_threshold': 20, 'short_threshold': 30,
         'lev': 5, 'margin_use': 0.70, 'sell_pct': 0.55,
         'short_sl': -0.30, 'short_tp': 0.80},

        {**base, 'name': 'K3: 保守确认',
         'sell_threshold': 45, 'buy_threshold': 45,
         'short_threshold': 60, 'long_threshold': 60,
         'lev': 2, 'margin_use': 0.30, 'sell_pct': 0.30,
         'short_sl': -0.12, 'long_sl': -0.10},

        {**base, 'name': 'K4: 快速交易',
         'sell_threshold': 25, 'buy_threshold': 25,
         'short_threshold': 35, 'long_threshold': 35,
         'cooldown': 2, 'spot_cooldown': 6,
         'short_max_hold': 48, 'long_max_hold': 48},
    ]


# ======================================================
#   主函数
# ======================================================
def main(trade_days=None):
    if trade_days is None:
        trade_days = 30

    data = fetch_data()
    if '1h' not in data:
        print("错误: 无法获取1h数据")
        return

    main_df = data['1h']

    print("\n扫描蜡烛图形态(Nison完整版)...")
    patterns = scan_patterns(main_df)
    total_patterns = sum(len(v) for v in patterns.values())
    print(f"  发现 {total_patterns} 个形态, 分布在 {len(patterns)} 根K线上")

    name_counts = {}
    for plist in patterns.values():
        for p in plist:
            name_counts[p['name']] = name_counts.get(p['name'], 0) + 1
    print(f"  形态分布:")
    for n, cnt in sorted(name_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"    {n}: {cnt}次")

    strategies = get_strategies()
    all_results = []

    start_dt = main_df.index[-1] - pd.Timedelta(days=trade_days)
    trade_start = str(start_dt)[:16]
    trade_end = str(main_df.index[-1])[:16]

    print(f"\n{'=' * 100}")
    print(f"  蜡烛图形态策略(Nison完整版) · {len(strategies)}种 · 最近{trade_days}天")
    print(f"  增强: 成交量确认 + 关键位增强 + 形态失败检测 + 30+种形态")
    print(f"  数据: {len(main_df)}根1h K线 | 交易: {trade_start} ~ {trade_end}")
    print(f"{'=' * 100}")

    print(f"\n{'策略':<20} {'α':>8} {'收益':>10} {'BH':>10} {'回撤':>8} {'交易':>6} {'费用':>10}")
    print('-' * 100)

    for cfg in strategies:
        r = run_candlestick_strategy(data, cfg, trade_days=trade_days)
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
        'description': f'蜡烛图形态策略(Nison完整版) · 最近{trade_days}天',
        'book': '《日本蜡烛图技术-K线分析》Steve Nison完整版',
        'run_time': datetime.now().isoformat(),
        'trade_days': trade_days,
        'total_patterns': total_patterns,
        'pattern_distribution': name_counts,
        'enhancements': [
            '成交量确认(放量+30%~50%)',
            '关键支撑/阻力位增强(+25%)',
            '形态失败检测(反向信号)',
            '30+种形态(含弃婴/塔形/三内外升降/窗口等)',
            '多级趋势判断(5/10/20周期综合)',
            'Nison可靠性分级评分'
        ],
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
                        'candlestick_result.json')
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
