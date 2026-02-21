"""
Pump 阶段检测器

基于 K线 OHLCV + taker_buy_base 数据，检测 5 个 Pump 阶段:
  NORMAL       — 正常波动
  ACCUMULATION — 放量不涨 (吸筹)
  EARLY_PUMP   — 动量 3-10%, 量放大
  MAIN_PUMP    — 动量 >10%, 加速上涨
  DISTRIBUTION — 量大但涨不动, 长上影线 (派发)

评分 4 因子加权:
  成交量 30% | 价格 30% | 订单流 20% | 模式 20%
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class PumpPhase(Enum):
    NORMAL = "normal"
    ACCUMULATION = "accumulation"
    EARLY_PUMP = "early_pump"
    MAIN_PUMP = "main_pump"
    DISTRIBUTION = "distribution"


@dataclass
class PumpResult:
    phase: PumpPhase
    score: float            # 0-100
    volume_z: float         # 成交量 Z-Score
    momentum: float         # 5-bar 涨幅 %
    acceleration: float     # 动量变化率
    buy_ratio: float        # taker_buy / total_volume
    suggested_action: str   # "WATCH" | "CONSIDER_BUY" | "HOLD" | "PREPARE_SELL" | "SELL"


_MIN_BARS = 30


def detect_pump_phase(df: pd.DataFrame, lookback: int = 100) -> PumpResult:
    """从 K线 DataFrame 检测 Pump 阶段。

    Parameters
    ----------
    df : DataFrame with columns: close, high, low, volume, (optional) taker_buy_base
    lookback : 回溯计算窗口

    Returns
    -------
    PumpResult
    """
    if df is None or len(df) < _MIN_BARS:
        return PumpResult(
            phase=PumpPhase.NORMAL, score=0, volume_z=0,
            momentum=0, acceleration=0, buy_ratio=0.5,
            suggested_action="WATCH",
        )

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    # taker_buy_base 可能缺失
    has_taker = "taker_buy_base" in df.columns
    taker_buy = df["taker_buy_base"].values if has_taker else None

    n = len(close)
    lb = min(lookback, n)

    # ---- 1. 成交量 Z-Score ----
    vol_window = volume[-lb:]
    vol_mean = np.mean(vol_window)
    vol_std = np.std(vol_window)
    current_vol = volume[-1]
    volume_z = (current_vol - vol_mean) / vol_std if vol_std > 0 else 0.0

    # ---- 2. 价格动量 (5-bar 涨幅%) ----
    bars_back = min(5, n - 1)
    ref_price = close[-(bars_back + 1)]
    momentum = ((close[-1] / ref_price) - 1) * 100 if ref_price > 0 else 0.0

    # 前一周期动量 (用于加速度)
    if n > bars_back + 2:
        prev_ref = close[-(bars_back + 2)]
        prev_close = close[-2]
        prev_momentum = ((prev_close / prev_ref) - 1) * 100 if prev_ref > 0 else 0.0
    else:
        prev_momentum = 0.0
    acceleration = momentum - prev_momentum

    # ---- 3. 买卖比 ----
    if taker_buy is not None and current_vol > 0:
        buy_ratio = float(taker_buy[-1] / current_vol)
    else:
        buy_ratio = 0.5

    # ---- 4. 突破检测 ----
    is_breakout = False
    if n >= 21:
        high_20 = np.max(high[-21:-1])
        is_breakout = close[-1] > high_20

    # ---- 评分 ----
    vol_score = min(100, max(0, volume_z * 20))
    if volume_z > 5:
        vol_score = min(100, vol_score + 20)

    price_score = min(50, abs(momentum) * 5)
    if acceleration > 0:
        price_score += min(30, acceleration * 10)
    if is_breakout:
        price_score += 20
    price_score = min(100, max(0, price_score))

    flow_score = min(100, max(0, abs(buy_ratio - 0.5) * 200))
    if buy_ratio < 0.5:
        flow_score = -flow_score  # 卖压为负分

    pattern_score = 0.0
    if vol_score > 50 and price_score > 30:
        pattern_score += 30
    if momentum > 0 and acceleration > 0:
        pattern_score += 25
    if buy_ratio > 0.55:
        pattern_score += 25
    if is_breakout:
        pattern_score += 20
    pattern_score = min(100, pattern_score)

    # 加权综合 (flow_score 可能为负, 用 max(0, ...) 截断)
    total_score = (
        vol_score * 0.30
        + price_score * 0.30
        + max(0, flow_score) * 0.20
        + pattern_score * 0.20
    )
    total_score = max(0, min(100, total_score))

    # ---- 阶段判定 ----
    phase = _determine_phase(total_score, vol_score, price_score, flow_score, momentum, acceleration)
    suggested = _suggest_action(phase, total_score)

    return PumpResult(
        phase=phase,
        score=round(total_score, 1),
        volume_z=round(volume_z, 2),
        momentum=round(momentum, 2),
        acceleration=round(acceleration, 2),
        buy_ratio=round(buy_ratio, 3),
        suggested_action=suggested,
    )


def _determine_phase(
    total: float, vol: float, price: float, flow: float,
    momentum: float, accel: float,
) -> PumpPhase:
    # 派发优先: 强卖压信号不受总分门槛限制
    if momentum < -5:
        return PumpPhase.DISTRIBUTION
    if flow < 0 and price > 30:
        return PumpPhase.DISTRIBUTION
    if total < 30:
        return PumpPhase.NORMAL
    # 主升: 动量 >10% 且加速
    if momentum > 10 and accel > 0:
        return PumpPhase.MAIN_PUMP
    # 早期拉升: 动量 3-10%
    if 3 <= momentum <= 10 and total > 40:
        return PumpPhase.EARLY_PUMP
    # 吸筹: 放量不涨
    if vol > 60 and price < 30:
        return PumpPhase.ACCUMULATION
    # fallback
    if total > 40:
        return PumpPhase.EARLY_PUMP
    return PumpPhase.NORMAL


def _suggest_action(phase: PumpPhase, score: float) -> str:
    if phase == PumpPhase.EARLY_PUMP and score > 50:
        return "CONSIDER_BUY"
    if phase == PumpPhase.MAIN_PUMP:
        return "HOLD"
    if phase == PumpPhase.DISTRIBUTION:
        return "PREPARE_SELL"
    if phase == PumpPhase.ACCUMULATION:
        return "WATCH"
    return "WATCH"
