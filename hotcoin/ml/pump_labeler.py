"""
Pump 事件自动标注管道

从历史 K 线数据中自动识别 pump 事件并标注 5 个阶段:
  0 - normal:       正常波动
  1 - accumulation: 放量不涨 (吸筹)
  2 - early_pump:   初期拉升 (动量 3-8%)
  3 - main_pump:    主升浪 (动量 >8%, 加速)
  4 - distribution:  派发 (量大不涨, 长上影线)

用法:
    from hotcoin.ml.pump_labeler import label_pump_phases
    labels = label_pump_phases(df, pump_threshold=0.08, vol_surge=3.0)
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger("hotcoin.pump_labeler")


def find_pump_events(
    df: pd.DataFrame,
    window: int = 5,
    pump_threshold: float = 0.08,
    vol_surge: float = 3.0,
) -> list:
    """
    扫描 K 线找到所有 pump 事件。

    pump 定义: window bar 内涨幅 > pump_threshold 且成交量 > vol_surge × 均量。

    Returns
    -------
    list of dict: [{peak_idx, start_idx, return_pct, vol_ratio}, ...]
    """
    close = df["close"].values
    volume = df["volume"].values
    n = len(close)

    if n < 30:
        return []

    # 滚动均量 (20 bar)
    vol_ma20 = pd.Series(volume).rolling(20, min_periods=10).mean().values

    events = []
    i = window
    while i < n:
        # window bar 涨幅
        ret = (close[i] / close[i - window] - 1) if close[i - window] > 0 else 0
        # 窗口内平均成交量 vs pump 前的长期均量
        vol_avg_window = np.mean(volume[i - window : i + 1])
        pre_pump_idx = max(0, i - window)
        vol_baseline = vol_ma20[pre_pump_idx] if vol_ma20[pre_pump_idx] > 0 else vol_ma20[i]
        vol_ratio = vol_avg_window / vol_baseline if vol_baseline > 0 else 0

        if ret >= pump_threshold and vol_ratio >= vol_surge:
            events.append({
                "peak_idx": i,
                "start_idx": i - window,
                "return_pct": ret,
                "vol_ratio": vol_ratio,
            })
            # 跳过 pump 区间避免重复检测
            i += window
        else:
            i += 1

    log.info("发现 %d 个 pump 事件 (threshold=%.1f%%, vol_surge=%.1fx)",
             len(events), pump_threshold * 100, vol_surge)
    return events


def _has_accumulation_pattern(df: pd.DataFrame, idx: int) -> bool:
    """检查 idx 位置是否有吸筹特征: 放量但价格不涨。"""
    if idx < 5:
        return False
    close = df["close"].values
    volume = df["volume"].values
    vol_ma = pd.Series(volume).rolling(20, min_periods=10).mean().values

    # 近 3 bar 成交量 > 1.5x 均量, 但涨幅 < 2%
    vol_ratio = np.mean(volume[idx - 2 : idx + 1]) / vol_ma[idx] if vol_ma[idx] > 0 else 0
    price_change = abs(close[idx] / close[idx - 3] - 1) if close[idx - 3] > 0 else 0
    return vol_ratio > 1.5 and price_change < 0.02


def _has_distribution_pattern(df: pd.DataFrame, idx: int) -> bool:
    """检查 idx 位置是否有派发特征: 量大不涨 + 长上影线。"""
    if idx >= len(df):
        return False
    high = df["high"].values[idx]
    low = df["low"].values[idx]
    close = df["close"].values[idx]
    open_ = df["open"].values[idx]

    rng = high - low
    if rng <= 0:
        return False

    # 上影线占比 > 40%
    upper_wick = high - max(open_, close)
    upper_ratio = upper_wick / rng

    # 成交量放大
    volume = df["volume"].values
    vol_ma = pd.Series(volume).rolling(20, min_periods=10).mean().values
    vol_ratio = volume[idx] / vol_ma[idx] if vol_ma[idx] > 0 else 0

    return upper_ratio > 0.4 or (vol_ratio > 2.0 and close < open_)


def label_pump_phases(
    df: pd.DataFrame,
    pump_threshold: float = 0.08,
    vol_surge: float = 3.0,
    window: int = 5,
    accum_lookback: int = 10,
    early_lookback: int = 5,
    dist_lookahead: int = 5,
) -> pd.Series:
    """
    为整个 DataFrame 标注 pump 阶段。

    Parameters
    ----------
    df : K 线 DataFrame (需含 open/high/low/close/volume)
    pump_threshold : pump 涨幅阈值
    vol_surge : 成交量放大倍数阈值
    window : pump 检测窗口
    accum_lookback : accumulation 回溯 bar 数
    early_lookback : early_pump 回溯 bar 数
    dist_lookahead : distribution 前瞻 bar 数

    Returns
    -------
    pd.Series: 与 df 同 index, 值为 0-4 的整数标签
    """
    events = find_pump_events(df, window, pump_threshold, vol_surge)
    labels = np.zeros(len(df), dtype=int)

    for ev in events:
        peak = ev["peak_idx"]
        start = ev["start_idx"]

        # main_pump: pump 窗口内
        for j in range(start, min(peak + 1, len(df))):
            labels[j] = 3  # main_pump

        # early_pump: pump 前 early_lookback bar
        ep_start = max(0, start - early_lookback)
        for j in range(ep_start, start):
            if labels[j] == 0:
                labels[j] = 2  # early_pump

        # accumulation: pump 前 accum_lookback bar (如果有吸筹特征)
        acc_start = max(0, start - accum_lookback)
        for j in range(acc_start, ep_start):
            if labels[j] == 0 and _has_accumulation_pattern(df, j):
                labels[j] = 1  # accumulation

        # distribution: pump 后 dist_lookahead bar (如果有派发特征)
        dist_end = min(len(df), peak + 1 + dist_lookahead)
        for j in range(peak + 1, dist_end):
            if labels[j] == 0 and _has_distribution_pattern(df, j):
                labels[j] = 4  # distribution

    # 统计
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique, counts))
    log.info("标签分布: %s (total=%d)", dist, len(labels))

    return pd.Series(labels, index=df.index, name="pump_label")
