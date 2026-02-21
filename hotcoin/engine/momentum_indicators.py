"""
短周期专用指标 — 连续K线 / 放量比 / 快速动量

为热点币快速决策提供辅助指标, 补充六书策略不覆盖的超短周期特征。
"""

import numpy as np
import pandas as pd
from typing import Dict


def consecutive_green_bars(df: pd.DataFrame, lookback: int = 10) -> int:
    """最近连续阳线数量。"""
    closes = df["close"].values[-lookback:]
    opens = df["open"].values[-lookback:]
    count = 0
    for i in range(len(closes) - 1, -1, -1):
        if closes[i] > opens[i]:
            count += 1
        else:
            break
    return count


def consecutive_red_bars(df: pd.DataFrame, lookback: int = 10) -> int:
    """最近连续阴线数量。"""
    closes = df["close"].values[-lookback:]
    opens = df["open"].values[-lookback:]
    count = 0
    for i in range(len(closes) - 1, -1, -1):
        if closes[i] < opens[i]:
            count += 1
        else:
            break
    return count


def volume_surge_ratio(df: pd.DataFrame, short: int = 3, long: int = 20) -> float:
    """短期成交量 / 长期均值。"""
    vol = df["volume"].values
    if len(vol) < long:
        return 1.0
    short_avg = np.mean(vol[-short:])
    long_avg = np.mean(vol[-long:])
    return short_avg / long_avg if long_avg > 0 else 1.0


def price_momentum(df: pd.DataFrame, periods: list = None) -> Dict[str, float]:
    """多周期涨幅。"""
    if periods is None:
        periods = [1, 3, 5, 10]
    close = df["close"].values
    result = {}
    for p in periods:
        if len(close) > p and close[-p - 1] > 0:
            result[f"mom_{p}"] = (close[-1] - close[-p - 1]) / close[-p - 1]
        else:
            result[f"mom_{p}"] = 0.0
    return result


def body_to_range_ratio(df: pd.DataFrame) -> float:
    """最新K线实体/振幅比 (大阳/大阴线检测)。"""
    if len(df) < 1:
        return 0.0
    row = df.iloc[-1]
    rng = row["high"] - row["low"]
    if rng <= 0:
        return 0.0
    body = abs(row["close"] - row["open"])
    return body / rng


def compute_hot_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """综合计算热点币辅助指标。"""
    result = {
        "consecutive_green": consecutive_green_bars(df),
        "consecutive_red": consecutive_red_bars(df),
        "vol_surge_3_20": volume_surge_ratio(df, 3, 20),
        "body_range_ratio": body_to_range_ratio(df),
    }
    result.update(price_momentum(df))
    return result
