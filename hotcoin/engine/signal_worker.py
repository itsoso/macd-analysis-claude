"""
单币 Signal Worker — 复用 signal_core + multi_tf_consensus

为单个热点币执行:
  1. 获取多周期 K 线
  2. 添加指标 (使用 HOT_COIN_INDICATOR_PARAMS 快参数)
  3. compute_signals_six → calc_fusion_score_six
  4. fuse_tf_scores 多周期共识
  5. 输出交易信号
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from hotcoin.engine.hot_coin_params import (
    HOT_COIN_FUSION_CONFIG,
    HOT_COIN_INDICATOR_PARAMS,
    HOT_COIN_TIMEFRAMES,
    HOT_COIN_CONSENSUS_CONFIG,
    HOT_COIN_KLINE_PARAMS,
)

log = logging.getLogger("hotcoin.worker")

_P = HOT_COIN_INDICATOR_PARAMS


def _add_hot_indicators(df):
    """线程安全的指标计算 — 直接传参, 不依赖全局 config。"""
    from indicators import calc_macd, calc_kdj, calc_cci, calc_rsi, add_ma_columns

    df = df.copy()
    df = add_ma_columns(df)

    macd_df = calc_macd(df["close"], fast=_P["macd_fast"], slow=_P["macd_slow"], signal=_P["macd_signal"])
    df["DIF"] = macd_df["DIF"]
    df["DEA"] = macd_df["DEA"]
    df["MACD_BAR"] = macd_df["MACD_BAR"]

    macd_alt = calc_macd(df["close"], fast=_P.get("macd_fast_alt", 5), slow=_P.get("macd_slow_alt", 10), signal=_P.get("macd_signal_alt", 5))
    df["DIF_FAST"] = macd_alt["DIF"]
    df["DEA_FAST"] = macd_alt["DEA"]
    df["MACD_BAR_FAST"] = macd_alt["MACD_BAR"]

    kdj_df = calc_kdj(df["high"], df["low"], df["close"], n=_P["kdj_period"])
    df["K"] = kdj_df["K"]
    df["D"] = kdj_df["D"]
    df["J"] = kdj_df["J"]

    df["CCI"] = calc_cci(df["high"], df["low"], df["close"])
    df["RSI6"] = calc_rsi(df["close"], _P.get("rsi_short", 6))
    df["RSI12"] = calc_rsi(df["close"], _P.get("rsi_long", 12))
    return df


@dataclass
class TradeSignal:
    """交易信号。"""
    symbol: str
    action: str            # "BUY" | "SELL" | "HOLD"
    strength: int = 0      # 绝对值, 越大越强
    confidence: float = 0.0
    reason: str = ""
    sell_score: float = 0.0
    buy_score: float = 0.0
    consensus: dict = field(default_factory=dict)
    tf_details: dict = field(default_factory=dict)
    computed_at: float = 0.0


def compute_signal_for_symbol(symbol: str, timeframes: Optional[List[str]] = None) -> TradeSignal:
    """
    为单个 symbol 计算多周期六书融合信号。

    这是同步阻塞函数, 由 SignalDispatcher 通过 ThreadPoolExecutor 调用。
    使用 _add_hot_indicators 确保线程安全 (不修改全局 config)。
    """
    tfs = timeframes or HOT_COIN_TIMEFRAMES
    t0 = time.time()

    try:
        from binance_fetcher import fetch_binance_klines
        from ma_indicators import add_moving_averages
        from signal_core import compute_signals_six, calc_fusion_score_six
        from multi_tf_consensus import fuse_tf_scores
    except ImportError as e:
        log.error("导入失败: %s", e)
        return TradeSignal(symbol=symbol, action="HOLD", reason=f"import error: {e}")

    tf_scores = {}
    tf_details = {}

    for tf in tfs:
        try:
            kline_cfg = HOT_COIN_KLINE_PARAMS.get(tf, {"days": 7, "min_bars": 50})

            df = fetch_binance_klines(symbol, interval=tf, days=kline_cfg["days"])
            if df is None or len(df) < kline_cfg["min_bars"]:
                bars = len(df) if df is not None else 0
                log.debug("%s %s 数据不足 (%d bars)", symbol, tf, bars)
                continue

            df = _add_hot_indicators(df)
            add_moving_averages(df, timeframe=tf)

            data_all = {tf: df}
            signals = compute_signals_six(df, tf, data_all, max_bars=500)

            idx = len(df) - 1
            dt = df.index[idx]
            ss, bs = calc_fusion_score_six(signals, df, idx, dt, HOT_COIN_FUSION_CONFIG)

            tf_scores[tf] = (ss, bs)
            tf_details[tf] = {
                "ss": round(ss, 1),
                "bs": round(bs, 1),
                "bars": len(df),
                "last_close": float(df["close"].iloc[-1]),
            }

        except Exception as e:
            log.warning("%s %s 信号计算失败: %s", symbol, tf, e)
            continue

    if not tf_scores:
        return TradeSignal(
            symbol=symbol, action="HOLD",
            reason="所有周期信号计算失败",
            computed_at=time.time(),
        )

    # 降级警告: 可用周期不足一半时降低信号可信度
    degraded = len(tf_scores) < len(tfs) / 2

    if degraded:
        log.warning("%s 信号降级: 仅 %d/%d 周期可用 (%s)",
                    symbol, len(tf_scores), len(tfs), list(tf_scores.keys()))

    # 多周期共识
    try:
        computed_tfs = list(tf_scores.keys())
        consensus = fuse_tf_scores(tf_scores, computed_tfs, config=HOT_COIN_CONSENSUS_CONFIG)
    except Exception as e:
        log.warning("%s 多周期共识失败: %s", symbol, e)
        return TradeSignal(
            symbol=symbol, action="HOLD",
            tf_details=tf_details,
            reason=f"consensus error: {e}",
            computed_at=time.time(),
        )

    decision = consensus.get("decision", {})
    direction = decision.get("direction", "neutral")
    strength = decision.get("strength", 0)
    actionable = decision.get("actionable", False)

    if actionable and direction == "long":
        action = "BUY"
    elif actionable and direction == "short":
        action = "SELL"
    else:
        action = "HOLD"

    elapsed = time.time() - t0
    log.debug("%s 信号完成 %.1fs: %s strength=%d (%s)",
              symbol, elapsed, action, strength, decision.get("label", ""))

    confidence = min(1.0, abs(strength) / 100)
    if degraded:
        confidence *= 0.5  # 降级信号降低可信度

    return TradeSignal(
        symbol=symbol,
        action=action,
        strength=strength,
        confidence=confidence,
        reason=decision.get("reason", decision.get("label", "")),
        sell_score=consensus.get("weighted_ss", 0),
        buy_score=consensus.get("weighted_bs", 0),
        consensus=consensus,
        tf_details=tf_details,
        computed_at=time.time(),
    )
