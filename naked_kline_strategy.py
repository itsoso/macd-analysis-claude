"""
裸K线交易法 — 许佳聪《裸K线交易法》核心策略代码实现
=======================================================

本模块完全基于价格行为(Price Action)，不使用任何技术指标(MACD/RSI/KDJ等)。
仅依赖K线形态 + 关键位(支撑/阻力) + 趋势结构做出交易决策。

核心章节对应:
  第一部分: 市场结构与趋势 → identify_trend(), find_swing_points()
  第二部分: 关键位(支撑/阻力) → find_key_levels()
  第三部分: Pin Bar 信号 → detect_pin_bar()
  第四部分: Inside Bar 信号 → detect_inside_bar()
  第五部分: Outside Bar / 吞没 → detect_engulfing()
  第六部分: Fakey(假突破)信号 → detect_fakey()
  第七部分: 双K反转 → detect_two_bar_reversal()
  第八部分: 组合信号评分 → score_signal()
  第九部分: 资金管理 → calc_position_size()

交易原则:
  1. 顺势交易为主: 在趋势方向上寻找入场信号
  2. 关键位过滤: 信号必须出现在关键支撑/阻力位附近
  3. 逆势只在极强反转信号+关键位共振时交易
  4. 每笔交易风险不超过账户的2%
  5. 至少1:1.5的盈亏比
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict


# ======================================================
#   数据结构
# ======================================================

@dataclass
class Signal:
    """一个交易信号"""
    time: str                # K线时间
    pattern: str             # 形态名称
    direction: str           # 'long' / 'short'
    strength: int            # 信号强度 0-100
    entry_price: float       # 建议入场价
    stop_loss: float         # 止损价
    take_profit_1: float     # 止盈1 (1:1.5)
    take_profit_2: float     # 止盈2 (1:2.5)
    at_key_level: bool       # 是否在关键位
    with_trend: bool         # 是否顺势
    trend: str               # 'up' / 'down' / 'range'
    risk_reward: float       # 盈亏比
    notes: str = ''          # 备注

    def to_dict(self):
        return asdict(self)


@dataclass
class KeyLevel:
    """关键支撑/阻力位"""
    price: float
    level_type: str          # 'support' / 'resistance' / 'both'
    touches: int             # 被触及次数
    strength: int            # 强度 1-5


# ======================================================
#   第一部分: 市场结构与趋势识别 (纯价格行为)
# ======================================================

def find_swing_points(df: pd.DataFrame, left: int = 5, right: int = 5) -> Tuple[List, List]:
    """
    识别摆动高点(Swing High)和摆动低点(Swing Low)。

    书中核心概念: 市场结构由一系列高点和低点构成。
    - 上升趋势: 更高的高点(HH) + 更高的低点(HL)
    - 下降趋势: 更低的低点(LL) + 更低的高点(LH)

    Parameters
    ----------
    df : DataFrame with 'high', 'low' columns
    left : int  左侧需要多少根K线确认
    right : int 右侧需要多少根K线确认
    """
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)

    swing_highs = []  # [(index, price), ...]
    swing_lows = []

    for i in range(left, n - right):
        # Swing High: 比左右各 left/right 根K线的高点都高
        is_sh = True
        for j in range(1, left + 1):
            if highs[i] <= highs[i - j]:
                is_sh = False
                break
        if is_sh:
            for j in range(1, right + 1):
                if highs[i] <= highs[i + j]:
                    is_sh = False
                    break
        if is_sh:
            swing_highs.append((i, float(highs[i])))

        # Swing Low: 比左右各 left/right 根K线的低点都低
        is_sl = True
        for j in range(1, left + 1):
            if lows[i] >= lows[i - j]:
                is_sl = False
                break
        if is_sl:
            for j in range(1, right + 1):
                if lows[i] >= lows[i + j]:
                    is_sl = False
                    break
        if is_sl:
            swing_lows.append((i, float(lows[i])))

    return swing_highs, swing_lows


def identify_trend(df: pd.DataFrame, idx: int, lookback: int = 60) -> str:
    """
    用纯价格行为判断趋势方向。

    书中方法: 观察最近的摆动高点和低点序列。
    - HH + HL = 上升趋势
    - LL + LH = 下降趋势
    - 其他 = 震荡/盘整

    Parameters
    ----------
    idx : 当前K线索引 (只使用 idx 之前的数据)
    lookback : 回看周期
    """
    start = max(0, idx - lookback)
    window = df.iloc[start:idx + 1]

    if len(window) < 20:
        return 'range'

    # 在窗口内寻找摆动点 (使用较小的left/right以适应窗口)
    swing_highs, swing_lows = find_swing_points(window, left=3, right=3)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return 'range'

    # 取最近的2-3个摆动点做判断
    recent_highs = [p for _, p in swing_highs[-3:]]
    recent_lows = [p for _, p in swing_lows[-3:]]

    # 高点递增 + 低点递增 → 上升趋势
    highs_rising = all(recent_highs[i] < recent_highs[i + 1]
                       for i in range(len(recent_highs) - 1))
    lows_rising = all(recent_lows[i] < recent_lows[i + 1]
                      for i in range(len(recent_lows) - 1))

    # 高点递减 + 低点递减 → 下降趋势
    highs_falling = all(recent_highs[i] > recent_highs[i + 1]
                        for i in range(len(recent_highs) - 1))
    lows_falling = all(recent_lows[i] > recent_lows[i + 1]
                       for i in range(len(recent_lows) - 1))

    if highs_rising and lows_rising:
        return 'up'
    elif highs_falling and lows_falling:
        return 'down'

    # 辅助判断: 价格相对于20日高低的位置
    h20 = window['high'].max()
    l20 = window['low'].min()
    price = float(df['close'].iloc[idx])
    pos = (price - l20) / (h20 - l20) if h20 > l20 else 0.5

    if pos > 0.65 and (highs_rising or lows_rising):
        return 'up'
    elif pos < 0.35 and (highs_falling or lows_falling):
        return 'down'

    return 'range'


# ======================================================
#   第二部分: 关键位(支撑/阻力)识别
# ======================================================

def find_key_levels(df: pd.DataFrame, idx: int, lookback: int = 120,
                    tolerance_pct: float = 0.005) -> List[KeyLevel]:
    """
    识别关键支撑/阻力位。

    书中方法:
    1. 找出历史上价格多次触及但未能突破的价位
    2. 用摆动高点的聚集区作为阻力, 摆动低点聚集区作为支撑
    3. 合并接近的价位(容差范围内)

    Parameters
    ----------
    tolerance_pct : 价格容差(占当前价格的百分比)
    """
    start = max(0, idx - lookback)
    window = df.iloc[start:idx + 1]

    if len(window) < 20:
        return []

    current_price = float(df['close'].iloc[idx])
    tol = current_price * tolerance_pct

    # 收集所有摆动高点和低点
    swing_highs, swing_lows = find_swing_points(window, left=3, right=3)

    # 合并所有关键价位
    raw_levels = []
    for _, price in swing_highs:
        raw_levels.append(('resistance', price))
    for _, price in swing_lows:
        raw_levels.append(('support', price))

    if not raw_levels:
        return []

    # 聚类: 将接近的价位合并
    raw_levels.sort(key=lambda x: x[1])
    clusters = []
    current_cluster = [raw_levels[0]]

    for i in range(1, len(raw_levels)):
        if raw_levels[i][1] - current_cluster[-1][1] <= tol:
            current_cluster.append(raw_levels[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [raw_levels[i]]
    clusters.append(current_cluster)

    # 构建关键位
    key_levels = []
    for cluster in clusters:
        prices = [p for _, p in cluster]
        types = [t for t, _ in cluster]
        avg_price = np.mean(prices)
        touches = len(cluster)

        has_support = 'support' in types
        has_resistance = 'resistance' in types

        if has_support and has_resistance:
            level_type = 'both'
        elif has_support:
            level_type = 'support'
        else:
            level_type = 'resistance'

        # 强度: 触及次数越多越强
        strength = min(5, touches)

        key_levels.append(KeyLevel(
            price=round(float(avg_price), 2),
            level_type=level_type,
            touches=touches,
            strength=strength,
        ))

    return key_levels


def is_at_key_level(price: float, key_levels: List[KeyLevel],
                    tolerance_pct: float = 0.01) -> Tuple[bool, Optional[KeyLevel]]:
    """检查价格是否在关键位附近"""
    for kl in key_levels:
        if abs(price - kl.price) / price <= tolerance_pct:
            return True, kl
    return False, None


# ======================================================
#   第三部分: Pin Bar (影线反转信号)
# ======================================================

def detect_pin_bar(df: pd.DataFrame, idx: int) -> Optional[Dict]:
    """
    检测 Pin Bar (长影线反转信号)。

    书中定义:
    1. 一端有明显的长影线(>=实体的2倍)
    2. 实体较小, 位于K线的一端
    3. 另一端影线很短或没有

    看涨Pin Bar(锤子线): 长下影线, 实体在上方
    看跌Pin Bar(射击之星): 长上影线, 实体在下方
    """
    if idx < 1:
        return None

    o = float(df['open'].iloc[idx])
    h = float(df['high'].iloc[idx])
    l = float(df['low'].iloc[idx])
    c = float(df['close'].iloc[idx])

    body = abs(c - o)
    total_range = h - l

    if total_range < 1e-8 or body < 1e-8:
        return None

    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l

    body_ratio = body / total_range  # 实体占K线总长比例

    # Pin Bar 条件: 实体小(< 35%总长), 一端影线 >= 2倍实体
    if body_ratio > 0.35:
        return None

    # 看涨 Pin Bar (锤子线): 长下影线
    if lower_shadow >= body * 2.0 and upper_shadow <= body * 0.5:
        # 下影线越长, 信号越强
        strength = min(80, int(40 + (lower_shadow / body) * 8))
        if lower_shadow >= body * 3:
            strength = min(90, strength + 10)

        return {
            'pattern': 'pin_bar_bullish',
            'name': '看涨Pin Bar(锤子线)',
            'direction': 'long',
            'strength': strength,
            'stop_loss': l,  # 止损在Pin Bar最低点下方
            'entry_trigger': h,  # 突破Pin Bar高点入场
        }

    # 看跌 Pin Bar (射击之星): 长上影线
    if upper_shadow >= body * 2.0 and lower_shadow <= body * 0.5:
        strength = min(80, int(40 + (upper_shadow / body) * 8))
        if upper_shadow >= body * 3:
            strength = min(90, strength + 10)

        return {
            'pattern': 'pin_bar_bearish',
            'name': '看跌Pin Bar(射击之星)',
            'direction': 'short',
            'strength': strength,
            'stop_loss': h,
            'entry_trigger': l,
        }

    return None


# ======================================================
#   第四部分: Inside Bar (孕线/内包线)
# ======================================================

def detect_inside_bar(df: pd.DataFrame, idx: int) -> Optional[Dict]:
    """
    检测 Inside Bar (内包线)。

    书中定义:
    1. 当前K线的高点低于前一根K线的高点
    2. 当前K线的低点高于前一根K线的低点
    3. 即当前K线完全被前一根K线"包含"

    交易方向:
    - 上升趋势中: 突破母线高点做多
    - 下降趋势中: 突破母线低点做空
    - 震荡中: 等待突破方向
    """
    if idx < 1:
        return None

    # 母线 (前一根)
    mother_h = float(df['high'].iloc[idx - 1])
    mother_l = float(df['low'].iloc[idx - 1])
    mother_body = abs(float(df['close'].iloc[idx - 1]) - float(df['open'].iloc[idx - 1]))

    # 子线 (当前)
    h = float(df['high'].iloc[idx])
    l = float(df['low'].iloc[idx])

    # Inside Bar 条件
    if h < mother_h and l > mother_l:
        # 子线越小 (相对于母线), 蓄能越强
        child_range = h - l
        mother_range = mother_h - mother_l
        if mother_range < 1e-8:
            return None

        compression = 1.0 - (child_range / mother_range)
        strength = min(75, int(35 + compression * 50))

        # 母线如果是大实体, 信号更强
        if mother_body > mother_range * 0.6:
            strength = min(85, strength + 10)

        return {
            'pattern': 'inside_bar',
            'name': '内包线(Inside Bar)',
            'direction': 'pending',  # 方向由趋势/突破确定
            'strength': strength,
            'mother_high': mother_h,
            'mother_low': mother_l,
            'long_entry': mother_h,    # 突破母线高点做多
            'short_entry': mother_l,   # 突破母线低点做空
            'long_stop': mother_l,     # 做多止损在母线低点
            'short_stop': mother_h,    # 做空止损在母线高点
        }

    return None


# ======================================================
#   第五部分: Outside Bar / 吞没线 (Engulfing)
# ======================================================

def detect_engulfing(df: pd.DataFrame, idx: int) -> Optional[Dict]:
    """
    检测吞没线(Engulfing / Outside Bar)。

    书中定义:
    1. 当前K线的高点高于前一根K线的高点
    2. 当前K线的低点低于前一根K线的低点
    3. 当前K线完全"吞没"了前一根K线

    看涨吞没: 当前收阳(close > open), 吞没前一根阴线
    看跌吞没: 当前收阴(close < open), 吞没前一根阳线
    """
    if idx < 1:
        return None

    prev_o = float(df['open'].iloc[idx - 1])
    prev_c = float(df['close'].iloc[idx - 1])
    prev_h = float(df['high'].iloc[idx - 1])
    prev_l = float(df['low'].iloc[idx - 1])

    curr_o = float(df['open'].iloc[idx])
    curr_c = float(df['close'].iloc[idx])
    curr_h = float(df['high'].iloc[idx])
    curr_l = float(df['low'].iloc[idx])

    # 当前K线范围必须包含前一根K线
    if not (curr_h > prev_h and curr_l < prev_l):
        return None

    curr_body = abs(curr_c - curr_o)
    prev_body = abs(prev_c - prev_o)
    curr_range = curr_h - curr_l

    if curr_range < 1e-8 or prev_body < 1e-8:
        return None

    # 看涨吞没: 当前收阳, 前一根收阴
    if curr_c > curr_o and prev_c < prev_o:
        strength = min(80, int(40 + (curr_body / prev_body) * 10))
        # 实体吞没实体更强
        if curr_c > prev_o and curr_o < prev_c:
            strength = min(90, strength + 10)

        return {
            'pattern': 'engulfing_bullish',
            'name': '看涨吞没线',
            'direction': 'long',
            'strength': strength,
            'stop_loss': curr_l,
            'entry_trigger': curr_h,
        }

    # 看跌吞没: 当前收阴, 前一根收阳
    if curr_c < curr_o and prev_c > prev_o:
        strength = min(80, int(40 + (curr_body / prev_body) * 10))
        if curr_c < prev_o and curr_o > prev_c:
            strength = min(90, strength + 10)

        return {
            'pattern': 'engulfing_bearish',
            'name': '看跌吞没线',
            'direction': 'short',
            'strength': strength,
            'stop_loss': curr_h,
            'entry_trigger': curr_l,
        }

    return None


# ======================================================
#   第六部分: Fakey (假突破)
# ======================================================

def detect_fakey(df: pd.DataFrame, idx: int) -> Optional[Dict]:
    """
    检测 Fakey 信号 (假突破)。

    书中定义:
    1. 先出现一个 Inside Bar
    2. 下一根K线突破了母线的一端
    3. 但收盘价回到母线范围内(假突破)
    4. 交易方向: 与假突破方向相反

    这是书中最高级的信号之一, 胜率最高。
    """
    if idx < 2:
        return None

    # 第一根: 母线 (idx-2)
    mother_h = float(df['high'].iloc[idx - 2])
    mother_l = float(df['low'].iloc[idx - 2])

    # 第二根: Inside Bar (idx-1)
    ib_h = float(df['high'].iloc[idx - 1])
    ib_l = float(df['low'].iloc[idx - 1])

    # 确认 idx-1 是 Inside Bar
    if not (ib_h < mother_h and ib_l > mother_l):
        return None

    # 第三根: 假突破K线 (idx)
    curr_o = float(df['open'].iloc[idx])
    curr_h = float(df['high'].iloc[idx])
    curr_l = float(df['low'].iloc[idx])
    curr_c = float(df['close'].iloc[idx])

    # 向上假突破: 突破母线高点但收盘回落
    if curr_h > mother_h and curr_c < mother_h and curr_c < curr_o:
        strength = 70
        # 突破幅度越大但收回越多, 信号越强
        breakout = curr_h - mother_h
        pullback = curr_h - curr_c
        if pullback > breakout * 2:
            strength = min(90, strength + 15)

        return {
            'pattern': 'fakey_bearish',
            'name': '看跌假突破(Fakey)',
            'direction': 'short',
            'strength': strength,
            'stop_loss': curr_h,
            'entry_trigger': mother_l,
        }

    # 向下假突破: 突破母线低点但收盘回升
    if curr_l < mother_l and curr_c > mother_l and curr_c > curr_o:
        strength = 70
        breakout = mother_l - curr_l
        pullback = curr_c - curr_l
        if pullback > breakout * 2:
            strength = min(90, strength + 15)

        return {
            'pattern': 'fakey_bullish',
            'name': '看涨假突破(Fakey)',
            'direction': 'long',
            'strength': strength,
            'stop_loss': curr_l,
            'entry_trigger': mother_h,
        }

    return None


# ======================================================
#   第七部分: 双K反转
# ======================================================

def detect_two_bar_reversal(df: pd.DataFrame, idx: int) -> Optional[Dict]:
    """
    检测双K反转形态。

    书中定义:
    1. 两根连续K线, 方向相反
    2. 第二根K线的收盘接近第一根K线的开盘
    3. 两根K线的总范围类似一个大Pin Bar

    看涨双K反转: 第一根大阴线 + 第二根大阳线 (收复失地)
    看跌双K反转: 第一根大阳线 + 第二根大阴线
    """
    if idx < 1:
        return None

    o1 = float(df['open'].iloc[idx - 1])
    c1 = float(df['close'].iloc[idx - 1])
    h1 = float(df['high'].iloc[idx - 1])
    l1 = float(df['low'].iloc[idx - 1])

    o2 = float(df['open'].iloc[idx])
    c2 = float(df['close'].iloc[idx])
    h2 = float(df['high'].iloc[idx])
    l2 = float(df['low'].iloc[idx])

    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    total_range = max(h1, h2) - min(l1, l2)

    if total_range < 1e-8 or body1 < 1e-8 or body2 < 1e-8:
        return None

    # 两根都需要是有意义的实体
    range1 = h1 - l1
    range2 = h2 - l2
    if body1 < range1 * 0.4 or body2 < range2 * 0.4:
        return None

    # 看涨双K反转: 阴 + 阳
    if c1 < o1 and c2 > o2:
        # 第二根收盘接近或超过第一根开盘
        recovery = (c2 - l1) / (h1 - l1) if h1 > l1 else 0
        if recovery >= 0.6 and body2 >= body1 * 0.7:
            strength = min(75, int(40 + recovery * 30))
            return {
                'pattern': 'two_bar_reversal_bullish',
                'name': '看涨双K反转',
                'direction': 'long',
                'strength': strength,
                'stop_loss': min(l1, l2),
                'entry_trigger': max(h1, h2),
            }

    # 看跌双K反转: 阳 + 阴
    if c1 > o1 and c2 < o2:
        recovery = (h1 - c2) / (h1 - l1) if h1 > l1 else 0
        if recovery >= 0.6 and body2 >= body1 * 0.7:
            strength = min(75, int(40 + recovery * 30))
            return {
                'pattern': 'two_bar_reversal_bearish',
                'name': '看跌双K反转',
                'direction': 'short',
                'strength': strength,
                'stop_loss': max(h1, h2),
                'entry_trigger': min(l1, l2),
            }

    return None


# ======================================================
#   第八部分: 综合信号评分与过滤
# ======================================================

def score_signal(raw_signal: Dict, trend: str, at_key_level: bool,
                 key_level: Optional[KeyLevel] = None) -> Optional[Signal]:
    """
    对原始形态信号做综合评分与过滤。

    书中核心原则:
    1. 顺势 + 关键位 = 最强信号 (A级)
    2. 顺势 + 无关键位 = 可交易信号 (B级)
    3. 逆势 + 关键位 + 强形态 = 谨慎交易 (C级)
    4. 逆势 + 无关键位 = 不交易

    盈亏比要求: 至少 1:1.5
    """
    direction = raw_signal['direction']
    base_strength = raw_signal['strength']

    # ── 趋势加/减分 ──
    with_trend = False
    if direction == 'long' and trend == 'up':
        base_strength += 15
        with_trend = True
    elif direction == 'short' and trend == 'down':
        base_strength += 15
        with_trend = True
    elif direction == 'long' and trend == 'down':
        base_strength -= 20  # 逆势做多, 大幅减分
    elif direction == 'short' and trend == 'up':
        base_strength -= 20  # 逆势做空, 大幅减分
    # 震荡市不加不减

    # Inside Bar 方向由趋势决定
    if direction == 'pending':
        if trend == 'up':
            direction = 'long'
            with_trend = True
            raw_signal['stop_loss'] = raw_signal['long_stop']
            raw_signal['entry_trigger'] = raw_signal['long_entry']
        elif trend == 'down':
            direction = 'short'
            with_trend = True
            raw_signal['stop_loss'] = raw_signal['short_stop']
            raw_signal['entry_trigger'] = raw_signal['short_entry']
        else:
            return None  # 震荡市不交易Inside Bar

    # ── 关键位加/减分 ──
    if at_key_level:
        base_strength += 15
        if key_level and key_level.strength >= 3:
            base_strength += 5
    else:
        base_strength -= 10

    # ── 过滤: 逆势 + 无关键位 = 不交易 ──
    if not with_trend and not at_key_level:
        return None

    # ── 过滤: 总分过低 ──
    if base_strength < 40:
        return None

    base_strength = min(100, max(0, base_strength))

    # ── 计算止盈价和盈亏比 ──
    stop_loss = raw_signal['stop_loss']
    entry = raw_signal.get('entry_trigger', stop_loss)

    if direction == 'long':
        risk = entry - stop_loss
        if risk <= 0:
            return None
        tp1 = entry + risk * 1.5  # 1:1.5
        tp2 = entry + risk * 2.5  # 1:2.5
        rr = 1.5
    else:
        risk = stop_loss - entry
        if risk <= 0:
            return None
        tp1 = entry - risk * 1.5
        tp2 = entry - risk * 2.5
        rr = 1.5

    return Signal(
        time='',  # 由 scan_bar() 在外层设置
        pattern=raw_signal['pattern'],
        direction=direction,
        strength=base_strength,
        entry_price=round(entry, 2),
        stop_loss=round(stop_loss, 2),
        take_profit_1=round(tp1, 2),
        take_profit_2=round(tp2, 2),
        at_key_level=at_key_level,
        with_trend=with_trend,
        trend=trend,
        risk_reward=round(rr, 2),
        notes=raw_signal.get('name', ''),
    )


# ======================================================
#   第九部分: 资金管理
# ======================================================

def calc_position_size(account_balance: float, entry: float,
                       stop_loss: float, risk_pct: float = 0.02,
                       leverage: int = 1) -> float:
    """
    计算仓位大小。

    书中原则: 每笔交易风险 = 账户的 1-2%
    仓位大小 = 风险金额 / (入场价 - 止损价)

    Parameters
    ----------
    account_balance : 账户余额
    entry : 入场价
    stop_loss : 止损价
    risk_pct : 风险百分比 (默认2%)
    leverage : 杠杆倍数
    """
    risk_amount = account_balance * risk_pct
    price_risk = abs(entry - stop_loss)

    if price_risk <= 0:
        return 0.0

    # 基础仓位 (无杠杆)
    position_size = risk_amount / price_risk
    # 实际需要的保证金
    notional = position_size * entry
    margin_needed = notional / leverage

    # 不超过账户的30% (保守约束)
    max_margin = account_balance * 0.30
    if margin_needed > max_margin:
        position_size = max_margin * leverage / entry

    return round(position_size, 6)


# ======================================================
#   综合扫描: 对单根K线检测所有形态
# ======================================================

def scan_bar(df: pd.DataFrame, idx: int, key_levels: List[KeyLevel],
             trend: str) -> List[Signal]:
    """
    扫描单根K线的所有裸K线形态, 返回过滤后的信号列表。

    检测顺序(优先级): Fakey > Pin Bar > Engulfing > Two Bar Reversal > Inside Bar
    """
    signals = []
    price = float(df['close'].iloc[idx])

    at_kl, kl = is_at_key_level(price, key_levels, tolerance_pct=0.015)

    detectors = [
        detect_fakey,
        detect_pin_bar,
        detect_engulfing,
        detect_two_bar_reversal,
        detect_inside_bar,
    ]

    for detector in detectors:
        raw = detector(df, idx)
        if raw is None:
            continue

        sig = score_signal(raw, trend, at_kl, kl)
        if sig is not None:
            sig.time = str(df.index[idx])
            signals.append(sig)

    # 按强度排序, 只保留最强的信号 (同方向去重)
    signals.sort(key=lambda s: s.strength, reverse=True)
    seen_dirs = set()
    unique = []
    for s in signals:
        if s.direction not in seen_dirs:
            unique.append(s)
            seen_dirs.add(s.direction)
    return unique


# ======================================================
#   全量扫描: 生成逐日信号
# ======================================================

def generate_daily_signals(df: pd.DataFrame, warmup: int = 60) -> List[Signal]:
    """
    对整个 DataFrame 逐日扫描, 生成所有裸K线信号。

    Parameters
    ----------
    df : 日线 DataFrame (含 open/high/low/close)
    warmup : 预热期(前N根不交易, 用于建立关键位和趋势)
    """
    all_signals = []

    for idx in range(warmup, len(df)):
        trend = identify_trend(df, idx)
        key_levels = find_key_levels(df, idx)
        day_signals = scan_bar(df, idx, key_levels, trend)
        all_signals.extend(day_signals)

    return all_signals


if __name__ == '__main__':
    print("裸K线交易法 — 许佳聪")
    print("核心形态: Pin Bar / Inside Bar / Engulfing / Fakey / 双K反转")
    print("请运行 backtest_naked_kline.py 进行回测")
