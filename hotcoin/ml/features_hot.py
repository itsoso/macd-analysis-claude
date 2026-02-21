"""
热点币特征工程

在现有 ETH 73 维基础特征之上, 新增 ~30 维热点币专用特征:
  - 社交特征 (Phase 2 填充)
  - 横截面特征 (相对全市场排名)
  - 微观结构特征

标签定义:
  - hotness: 未来 N 分钟涨幅在同时段全市场排名的分位数 (top 10% = 1)
  - trade: 进入后 30min 内能否获得 >3% 收益 (含流动性约束)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

log = logging.getLogger("hotcoin.features")


# ---------------------------------------------------------------------------
# 社交特征 (Phase 2 — 由外部填充)
# ---------------------------------------------------------------------------

SOCIAL_FEATURES = [
    "mention_velocity",         # 提及频率变化率
    "sentiment_score",          # NLP 情绪分数 (-1~1)
    "kol_count",                # KOL 提及数量
    "announcement_recency_h",   # 距最近公告小时数
]

# ---------------------------------------------------------------------------
# 横截面特征 (相对全市场)
# ---------------------------------------------------------------------------

CROSS_SECTIONAL_FEATURES = [
    "volume_rank_pct",          # 24h 成交量排名百分位
    "return_rank_pct",          # 涨幅排名百分位
    "corr_btc_20",              # 与 BTC 的 20bar 相关性
    "sector_momentum",          # 板块动量 (同板块均涨幅)
]

# ---------------------------------------------------------------------------
# 微观结构特征
# ---------------------------------------------------------------------------

MICRO_FEATURES = [
    "bid_ask_spread",           # 买卖价差 (Phase 2)
    "orderbook_imbalance",      # 盘口不平衡度 (Phase 2)
    "consecutive_green",        # 连续阳线数
    "consecutive_red",          # 连续阴线数
    "vol_surge_3_20",           # 短期/长期成交量比
    "body_range_ratio",         # K线实体/振幅比
    "mom_1", "mom_3", "mom_5", "mom_10",  # 多周期动量
]


def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算基础技术面特征 (复用现有 ETH 特征工程)。

    假设 df 已经通过 add_all_indicators() + add_moving_averages() 处理过。
    """
    features = pd.DataFrame(index=df.index)

    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # 价格动量
    for p in [1, 3, 5, 10, 20]:
        features[f"ret_{p}"] = c.pct_change(p)

    # 波动率
    features["hvol_5"] = c.pct_change().rolling(5).std()
    features["hvol_20"] = c.pct_change().rolling(20).std()

    # ATR
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    features["atr_14"] = tr.rolling(14).mean()
    features["atr_pct"] = features["atr_14"] / c

    # 成交量特征
    features["vol_ratio_5_20"] = v.rolling(5).mean() / v.rolling(20).mean().clip(lower=1)
    features["vol_change"] = v.pct_change()

    # MACD (indicators.py outputs "MACD_BAR")
    for macd_col in ["MACD_BAR", "macd_hist"]:
        if macd_col in df.columns:
            features["macd_hist"] = df[macd_col]
            features["macd_hist_diff"] = df[macd_col].diff()
            break

    # RSI (indicators.py outputs "RSI6" / "RSI12")
    for rsi_col in ["RSI6", "RSI12", "rsi"]:
        if rsi_col in df.columns:
            features["rsi"] = df[rsi_col]
            break

    # KDJ (indicators.py outputs uppercase "K", "D", "J")
    for col_upper, col_lower in [("K", "k"), ("D", "d"), ("J", "j")]:
        for col in [col_upper, col_lower]:
            if col in df.columns:
                features[col_lower] = df[col]
                break

    # 布林带
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        bb_width = df["bb_upper"] - df["bb_lower"]
        features["bb_pct"] = (c - df["bb_lower"]) / bb_width.clip(lower=1e-8)
        features["bb_width_pct"] = bb_width / c

    return features


def compute_hot_features(df: pd.DataFrame, cross_data: Optional[Dict] = None) -> pd.DataFrame:
    """
    计算热点币专用特征 (在 base_features 之上)。

    cross_data: 横截面数据 dict, 可选:
      {
        "volume_rank_pct": float,
        "return_rank_pct": float,
        "corr_btc_20": float,
        "sector_momentum": float,
      }
    """
    features = compute_base_features(df)

    c = df["close"].values
    o = df["open"].values
    v = df["volume"].values

    # 连续阳线/阴线
    green = (c > o).astype(int)
    red = (c < o).astype(int)
    n = len(c)
    consec_green = np.zeros(n)
    consec_red = np.zeros(n)
    for i in range(1, n):
        if green[i]:
            consec_green[i] = consec_green[i - 1] + 1
        if red[i]:
            consec_red[i] = consec_red[i - 1] + 1
    features["consecutive_green"] = consec_green
    features["consecutive_red"] = consec_red

    # 成交量突增比
    v_short = pd.Series(v).rolling(3).mean()
    v_long = pd.Series(v).rolling(20).mean().clip(lower=1)
    features["vol_surge_3_20"] = (v_short / v_long).values

    # 实体/振幅比
    rng = df["high"].values - df["low"].values
    body = np.abs(c - o)
    features["body_range_ratio"] = np.where(rng > 0, body / rng, 0)

    # 多周期动量
    for p in [1, 3, 5, 10]:
        if len(c) > p:
            features[f"mom_{p}"] = pd.Series(c).pct_change(p).values

    # 横截面特征 (外部传入)
    if cross_data:
        for feat_name, feat_val in cross_data.items():
            features[feat_name] = feat_val

    # 社交特征占位 (Phase 2 由外部填充)
    for col in SOCIAL_FEATURES:
        if col not in features.columns:
            features[col] = 0.0

    return features


def make_hotness_labels(returns_df: pd.DataFrame, window: int = 15,
                        percentile: float = 0.90) -> pd.Series:
    """
    热度标签: 未来 window 分钟涨幅在全市场排名 >= percentile 为 1。

    returns_df: DataFrame, index=timestamp, columns=symbols, values=return_N_min
    """
    ranks = returns_df.rank(axis=1, pct=True)
    labels = (ranks >= percentile).astype(int)
    return labels


def make_trade_labels(df: pd.DataFrame, forward_window: int = 30,
                      min_return: float = 0.03,
                      min_volume_24h: float = 1_000_000) -> pd.Series:
    """
    交易标签: 未来 forward_window 根K线内最大收益 >= min_return。

    使用向量化 rolling max, 避免 O(N * forward_window) 循环。
    """
    close = df["close"]
    future_max = close[::-1].rolling(forward_window, min_periods=1).max()[::-1].shift(-1)
    max_ret = (future_max - close) / close.clip(lower=1e-10)

    labels = (max_ret >= min_return).astype(int)

    if "quote_volume_24h" in df.columns:
        labels = labels & (df["quote_volume_24h"] >= min_volume_24h)

    labels.iloc[-forward_window:] = 0
    return labels.astype(int)
