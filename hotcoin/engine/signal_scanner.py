"""
信号扫描器 — S1-S5 信号检测 + F1-F3 反欺诈过滤

基于 K线 OHLCV + taker_buy_base 数据:

信号:
  S1 放量收高   — turnover_z > 2 且 close_pos > 0.7 且 涨幅 > 0      (+25)
  S2 突破       — close > 20 周期高点, 50 周期额外 +10                 (+20~30)
  S3 挤压突破   — BB 窄 + S2 + 放量                                    (+30)
  S4 买压优势   — taker_buy_ratio > 0.65 且 (涨幅 > 0 或 S2)          (+15)
  S5 持续买入   — 连续 3 bar taker_buy_ratio > 0.55                   (+10)

过滤:
  F1 疑似刷量   — 放量但收低位且跌
  F2 拉高出货   — 长上影线 + 放量
  F3 流动性不足 — 24h 成交额 < $200K
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class ScanResult:
    """单周期扫描结果。"""
    symbol: str
    timeframe: str
    signals: List[str] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)
    signal_reasons: Dict[str, str] = field(default_factory=dict)
    filter_reasons: Dict[str, str] = field(default_factory=dict)
    raw_score: float = 0.0


@dataclass
class AggregatedScan:
    """跨周期聚合结果。"""
    symbol: str
    s1_tfs: List[str] = field(default_factory=list)
    s2_tfs: List[str] = field(default_factory=list)
    s3_tfs: List[str] = field(default_factory=list)
    s4_tfs: List[str] = field(default_factory=list)
    s5_tfs: List[str] = field(default_factory=list)
    f1_hit: bool = False
    f2_hit: bool = False
    f3_hit: bool = False
    active_signals: List[str] = field(default_factory=list)
    active_filters: List[str] = field(default_factory=list)


_MIN_BARS = 30


def scan_timeframe(
    df: pd.DataFrame,
    tf: str,
    symbol: str = "",
    quote_vol_24h: float = 0,
) -> ScanResult:
    """对单个周期 K线检测 S1-S5 + F1-F3。"""
    result = ScanResult(symbol=symbol, timeframe=tf)

    if df is None or len(df) < _MIN_BARS:
        return result

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open_price = df["open"].values
    volume = df["volume"].values

    has_quote_vol = "quote_volume" in df.columns
    quote_vol = df["quote_volume"].values if has_quote_vol else volume * close

    has_taker = "taker_buy_base" in df.columns
    taker_buy = df["taker_buy_base"].values if has_taker else None

    n = len(close)
    cur_close = close[-1]
    cur_high = high[-1]
    cur_low = low[-1]
    cur_open = open_price[-1]
    cur_vol = volume[-1]

    # ---- 特征计算 ----

    # turnover Z-Score (成交额)
    lb = min(200, n)
    qv_window = quote_vol[-lb:]
    qv_mean = np.mean(qv_window)
    qv_std = np.std(qv_window)
    turnover_z = (quote_vol[-1] - qv_mean) / qv_std if qv_std > 0 else 0.0

    # close position
    bar_range = cur_high - cur_low
    close_pos = (cur_close - cur_low) / bar_range if bar_range > 0 else 0.5

    # 涨幅 (当 bar)
    ret_tf = ((cur_close / close[-2]) - 1) * 100 if n >= 2 and close[-2] > 0 else 0.0

    # taker buy ratio
    buy_ratio = float(taker_buy[-1] / cur_vol) if (taker_buy is not None and cur_vol > 0) else 0.5

    # 上影线占比
    upper_wick = cur_high - max(cur_close, cur_open)
    wick_ratio = upper_wick / bar_range if bar_range > 0 else 0.0

    # 突破检测
    breakout_20 = False
    breakout_50 = False
    if n >= 21:
        high_20 = np.max(high[-21:-1])
        breakout_20 = cur_close > high_20
    if n >= 51:
        high_50 = np.max(high[-51:-1])
        breakout_50 = cur_close > high_50

    # BB 挤压检测
    squeeze_flag = False
    if n >= 200:
        sma20 = pd.Series(close).rolling(20).mean().values
        std20 = pd.Series(close).rolling(20).std().values
        if sma20[-1] and sma20[-1] > 0:
            bbw = (std20 * 2) / sma20
            current_bbw = bbw[-1]
            if not (np.isnan(current_bbw) or current_bbw <= 0):
                bbw_valid = bbw[~np.isnan(bbw)]
                if len(bbw_valid) > 0:
                    percentile = (bbw_valid < current_bbw).sum() / len(bbw_valid) * 100
                    squeeze_flag = percentile < 20

    # 连续买入检测
    consecutive_buy = 0
    if taker_buy is not None:
        for i in range(min(5, n)):
            idx = -(i + 1)
            v = volume[idx]
            if v > 0 and taker_buy[idx] / v > 0.55:
                consecutive_buy += 1
            else:
                break

    # ---- S1: 放量收高 ----
    if turnover_z > 2.0 and close_pos > 0.7 and ret_tf > 0:
        result.signals.append("S1")
        result.signal_reasons["S1"] = (
            f"放量(z={turnover_z:.1f}) + 收高位({close_pos:.0%}) + 涨({ret_tf:+.2f}%)"
        )
        result.raw_score += 25

    # ---- S2: 突破 ----
    if breakout_20 and ret_tf > 0:
        result.signals.append("S2")
        score_s2 = 20
        reason = f"突破20周期高点"
        if breakout_50:
            score_s2 += 10
            reason += " (含50周期)"
        result.signal_reasons["S2"] = reason
        result.raw_score += score_s2

    # ---- S3: 挤压突破 ----
    if squeeze_flag and breakout_20 and turnover_z > 2.0:
        result.signals.append("S3")
        result.signal_reasons["S3"] = f"布林压缩后放量突破 (z={turnover_z:.1f})"
        result.raw_score += 30

    # ---- S4: 买压优势 ----
    if buy_ratio > 0.65 and (ret_tf > 0 or breakout_20):
        result.signals.append("S4")
        result.signal_reasons["S4"] = (
            f"买压优势 (ratio={buy_ratio:.2f}) + {'突破' if breakout_20 else '上涨'}"
        )
        result.raw_score += 15

    # ---- S5: 持续买入 ----
    if consecutive_buy >= 3:
        result.signals.append("S5")
        result.signal_reasons["S5"] = f"连续{consecutive_buy}根K线买入主导"
        result.raw_score += 10

    # ---- F1: 疑似刷量 ----
    if turnover_z > 2.0 and close_pos < 0.55 and ret_tf <= 0:
        result.filters.append("F1")
        result.filter_reasons["F1"] = (
            f"疑似刷量: 放量(z={turnover_z:.1f}) 但收低位({close_pos:.0%}) 且跌({ret_tf:+.2f}%)"
        )

    # ---- F2: 拉高出货 ----
    if wick_ratio > 0.6 and turnover_z > 2.0:
        result.filters.append("F2")
        result.filter_reasons["F2"] = f"疑似拉高出货: 上影线占{wick_ratio:.0%}"

    # ---- F3: 流动性不足 ----
    if quote_vol_24h > 0 and quote_vol_24h < 200_000:
        result.filters.append("F3")
        result.filter_reasons["F3"] = f"流动性不足: 24h成交额=${quote_vol_24h:,.0f}"

    return result


def aggregate_scans(results: List[ScanResult]) -> AggregatedScan:
    """跨周期聚合信号和过滤结果。"""
    if not results:
        return AggregatedScan(symbol="")

    symbol = results[0].symbol
    agg = AggregatedScan(symbol=symbol)

    for r in results:
        for sig in r.signals:
            tag = f"{sig}@{r.timeframe}"
            agg.active_signals.append(tag)
            if sig == "S1":
                agg.s1_tfs.append(r.timeframe)
            elif sig == "S2":
                agg.s2_tfs.append(r.timeframe)
            elif sig == "S3":
                agg.s3_tfs.append(r.timeframe)
            elif sig == "S4":
                agg.s4_tfs.append(r.timeframe)
            elif sig == "S5":
                agg.s5_tfs.append(r.timeframe)

        for flt in r.filters:
            if flt == "F1":
                agg.f1_hit = True
            elif flt == "F2":
                agg.f2_hit = True
            elif flt == "F3":
                agg.f3_hit = True
            if flt not in agg.active_filters:
                agg.active_filters.append(flt)

    return agg
