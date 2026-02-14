from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(slots=True)
class TripleSwordConfig:
    kdj_n: int = 9
    kdj_m1: int = 3
    kdj_m2: int = 3
    rsi_period: int = 14
    wr_period: int = 14
    rsi_overbought: float = 80.0
    rsi_oversold: float = 20.0
    wr_overbought: float = -20.0
    wr_oversold: float = -80.0
    min_confluence: int = 2


def _require_ohlc(df: pd.DataFrame) -> None:
    required = {"high", "low", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")


def calc_kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
) -> pd.DataFrame:
    low_n = low.rolling(window=n, min_periods=1).min()
    high_n = high.rolling(window=n, min_periods=1).max()

    denom = (high_n - low_n).replace(0, np.nan)
    rsv = ((close - low_n) / denom * 100).fillna(50.0)

    k = np.full(len(close), 50.0, dtype=np.float64)
    d = np.full(len(close), 50.0, dtype=np.float64)
    alpha_k = 1.0 / m1
    alpha_d = 1.0 / m2

    for i in range(1, len(close)):
        k[i] = (1.0 - alpha_k) * k[i - 1] + alpha_k * float(rsv.iloc[i])
        d[i] = (1.0 - alpha_d) * d[i - 1] + alpha_d * k[i]

    j = 3.0 * k - 2.0 * d
    return pd.DataFrame({"K": k, "D": d, "J": j}, index=close.index)


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder smoothing (alpha=1/period)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)
    return rsi


def calc_wr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest_n = high.rolling(window=period, min_periods=1).max()
    lowest_n = low.rolling(window=period, min_periods=1).min()
    denom = (highest_n - lowest_n).replace(0, np.nan)
    wr = (-100.0 * (highest_n - close) / denom).fillna(-50.0)
    return wr


def _cross_up(curr: pd.Series, prev: pd.Series, level: float) -> pd.Series:
    return (curr > level) & (prev <= level)


def _cross_down(curr: pd.Series, prev: pd.Series, level: float) -> pd.Series:
    return (curr < level) & (prev >= level)


def add_triple_sword_features(
    df: pd.DataFrame,
    config: TripleSwordConfig | None = None,
) -> pd.DataFrame:
    _require_ohlc(df)
    cfg = config or TripleSwordConfig()

    out = df.copy()
    kdj = calc_kdj(out["high"], out["low"], out["close"], cfg.kdj_n, cfg.kdj_m1, cfg.kdj_m2)
    out["kdj_k"] = kdj["K"]
    out["kdj_d"] = kdj["D"]
    out["kdj_j"] = kdj["J"]
    out["rsi"] = calc_rsi(out["close"], cfg.rsi_period)
    out["wr"] = calc_wr(out["high"], out["low"], out["close"], cfg.wr_period)

    out["kdj_golden_cross"] = (out["kdj_k"] > out["kdj_d"]) & (
        out["kdj_k"].shift(1) <= out["kdj_d"].shift(1)
    )
    out["kdj_death_cross"] = (out["kdj_k"] < out["kdj_d"]) & (
        out["kdj_k"].shift(1) >= out["kdj_d"].shift(1)
    )

    out["rsi_cross_up_oversold"] = _cross_up(out["rsi"], out["rsi"].shift(1), cfg.rsi_oversold)
    out["rsi_cross_down_overbought"] = _cross_down(
        out["rsi"], out["rsi"].shift(1), cfg.rsi_overbought
    )

    out["wr_cross_up_oversold"] = _cross_up(out["wr"], out["wr"].shift(1), cfg.wr_oversold)
    out["wr_cross_down_overbought"] = _cross_down(
        out["wr"], out["wr"].shift(1), cfg.wr_overbought
    )

    buy_score = (
        out["kdj_golden_cross"].astype(int)
        + out["rsi_cross_up_oversold"].astype(int)
        + out["wr_cross_up_oversold"].astype(int)
    )
    sell_score = (
        out["kdj_death_cross"].astype(int)
        + out["rsi_cross_down_overbought"].astype(int)
        + out["wr_cross_down_overbought"].astype(int)
    )
    out["buy_score"] = buy_score
    out["sell_score"] = sell_score

    decisions: list[Literal["BUY", "SELL", "HOLD"]] = []
    for b, s in zip(buy_score, sell_score):
        if b >= cfg.min_confluence and s == 0:
            decisions.append("BUY")
        elif s >= cfg.min_confluence and b == 0:
            decisions.append("SELL")
        else:
            decisions.append("HOLD")
    out["triple_sword_decision"] = decisions
    return out
