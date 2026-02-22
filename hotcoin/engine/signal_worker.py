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
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from hotcoin.engine.hot_coin_params import (
    HOT_COIN_FUSION_CONFIG,
    HOT_COIN_INDICATOR_PARAMS,
    HOT_COIN_TIMEFRAMES,
    HOT_COIN_CONSENSUS_CONFIG,
    HOT_COIN_KLINE_PARAMS,
)

log = logging.getLogger("hotcoin.worker")

_P = HOT_COIN_INDICATOR_PARAMS

# ---------------------------------------------------------------------------
# K线 bar-level 缓存: 同一个 bar 周期内复用上次拉取的 DataFrame
# ---------------------------------------------------------------------------
_TF_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
    "6h": 21600, "8h": 28800, "12h": 43200, "1d": 86400, "24h": 86400,
}

_kline_cache: Dict[Tuple[str, str], Tuple[pd.DataFrame, float]] = {}
_kline_cache_lock = threading.Lock()
_KLINE_CACHE_MAX = 500
_KLINE_CACHE_PRUNE_BATCH = 50  # 淘汰时一次性删除最老的 N 条, 避免频繁淘汰

# 全局 AlertScorer 实例 (保持 score_history 用于 L3 连续高分检测)
from hotcoin.engine.alert_scorer import AlertScorer as _AlertScorerCls
_alert_scorer = _AlertScorerCls()


def _get_cached_klines(symbol: str, tf: str, days: int, min_bars: int):
    """从缓存或 API 获取 K线。缓存 key=(symbol, tf), 有效期 = bar 周期的 80%。"""
    from binance_fetcher import fetch_binance_klines

    key = (symbol, tf)
    bar_sec = _TF_SECONDS.get(tf, 300)
    ttl = max(bar_sec * 0.8, 30)
    now = time.time()

    with _kline_cache_lock:
        cached = _kline_cache.get(key)
        if cached is not None:
            df_cached, ts = cached
            if now - ts < ttl and len(df_cached) >= min_bars:
                return df_cached

    df = fetch_binance_klines(symbol, interval=tf, days=days)
    if df is not None and len(df) >= min_bars:
        with _kline_cache_lock:
            _kline_cache[key] = (df, now)
            if len(_kline_cache) > _KLINE_CACHE_MAX:
                by_age = sorted(_kline_cache, key=lambda k: _kline_cache[k][1])
                for old_key in by_age[:_KLINE_CACHE_PRUNE_BATCH]:
                    _kline_cache.pop(old_key, None)
    return df


def clear_kline_cache():
    """清空 K线缓存 (测试/重置用)。"""
    with _kline_cache_lock:
        _kline_cache.clear()


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
    trace_id: str = ""
    # Pump / Signal / Alert 增强
    pump_phase: str = ""
    pump_score: float = 0.0
    alert_level: str = ""
    alert_score: float = 0.0
    active_signals: list = field(default_factory=list)
    active_filters: list = field(default_factory=list)


def compute_signal_for_symbol(symbol: str, timeframes: Optional[List[str]] = None) -> TradeSignal:
    """
    为单个 symbol 计算多周期六书融合信号。

    这是同步阻塞函数, 由 SignalDispatcher 通过 ThreadPoolExecutor 调用。
    使用 _add_hot_indicators 确保线程安全 (不修改全局 config)。
    """
    tfs = timeframes or HOT_COIN_TIMEFRAMES
    t0 = time.time()

    try:
        from ma_indicators import add_moving_averages
        from signal_core import compute_signals_six, calc_fusion_score_six
        from multi_tf_consensus import fuse_tf_scores
    except ImportError as e:
        log.error("导入失败: %s", e)
        return TradeSignal(symbol=symbol, action="HOLD", reason=f"import error: {e}")

    tf_scores = {}
    tf_details = {}
    tf_raw_dfs: Dict[str, pd.DataFrame] = {}  # 收集原始 df 供 Pump/Scan 复用

    for tf in tfs:
        try:
            kline_cfg = HOT_COIN_KLINE_PARAMS.get(tf, {"days": 7, "min_bars": 50})

            df = _get_cached_klines(symbol, tf, kline_cfg["days"], kline_cfg["min_bars"])
            if df is None or len(df) < kline_cfg["min_bars"]:
                bars = len(df) if df is not None else 0
                log.debug("%s %s 数据不足 (%d bars)", symbol, tf, bars)
                continue

            tf_raw_dfs[tf] = df  # 缓存原始 df (信号扫描需要 OHLCV 原始列)

            df = _add_hot_indicators(df)
            add_moving_averages(df, timeframe=tf)

            max_bars = kline_cfg.get("max_bars", 500)
            if len(df) > max_bars:
                df = df.iloc[-max_bars:].copy()
            data_all = {tf: df}
            signals = compute_signals_six(df, tf, data_all, max_bars=max_bars)

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

    # --- Pump + Signal Scan + Alert (增强层, 不影响原有逻辑) ---
    pump_phase = ""
    pump_score = 0.0
    alert_level = ""
    alert_score = 0.0
    active_signals_list = []
    active_filters_list = []
    try:
        from hotcoin.engine.pump_detector import detect_pump_phase
        from hotcoin.engine.signal_scanner import scan_timeframe, aggregate_scans

        # 从已有 df 获取 quote_vol_24h (用于 F3 流动性过滤)
        quote_vol_24h = 0.0
        _tf_minutes = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
                       "1h": 60, "2h": 120, "4h": 240}
        for _tf, _df in tf_raw_dfs.items():
            if "quote_volume" in _df.columns and len(_df) > 0:
                bars_per_day = 1440 / _tf_minutes.get(_tf, 5)
                sample_n = min(len(_df), max(1, int(bars_per_day)))
                avg_vol = float(_df["quote_volume"].iloc[-sample_n:].mean())
                if not (avg_vol != avg_vol):  # NaN check
                    quote_vol_24h = avg_vol * bars_per_day
                break

        # Pump 检测: 按优先级 5m > 15m > 1m > 其他
        pump_result = None
        _pump_tf_pref = ["5m", "15m", "1m", "3m", "30m", "1h"]
        for ptf in _pump_tf_pref:
            if ptf in tf_raw_dfs:
                pump_result = detect_pump_phase(tf_raw_dfs[ptf])
                break
        # fallback: 用第一个可用 tf
        if pump_result is None and tf_raw_dfs:
            pump_result = detect_pump_phase(next(iter(tf_raw_dfs.values())))

        # Signal Scan: 直接复用已缓存的 df (无需重新拉取)
        scan_results = []
        for tf, df_scan in tf_raw_dfs.items():
            scan_results.append(
                scan_timeframe(df_scan, tf, symbol=symbol, quote_vol_24h=quote_vol_24h))

        if scan_results:
            agg_scan = aggregate_scans(scan_results)
            alert = _alert_scorer.score(pump_result, agg_scan, tf_details)

            pump_phase = alert.pump_phase
            pump_score = pump_result.score if pump_result else 0.0
            alert_level = alert.level
            alert_score = alert.total_score
            active_signals_list = alert.active_signals
            active_filters_list = alert.active_filters

            if alert_level != "NONE":
                log.info("%s 预警 %s(%s) score=%.0f pump=%s signals=%s filters=%s",
                         symbol, alert_level, alert.level_name, alert_score,
                         pump_phase, active_signals_list, active_filters_list)
    except Exception:
        log.warning("%s Pump/Alert 增强层异常", symbol, exc_info=True)

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
        pump_phase=pump_phase,
        pump_score=pump_score,
        alert_level=alert_level,
        alert_score=alert_score,
        active_signals=active_signals_list,
        active_filters=active_filters_list,
    )
