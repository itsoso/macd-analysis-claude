"""
热点币快参数集

对比 ETH 慢参数, 热点币使用更短周期、更灵敏的参数。
这些参数通过 config_dict 传递给 signal_core.calc_fusion_score_six()，
而非修改 signal_core.py 源码。
"""

# 六书融合评分的配置覆盖 (传入 calc_fusion_score_six 的 config dict)
HOT_COIN_FUSION_CONFIG = {
    "fusion_mode": "c6_veto_4",
    # 阈值更灵敏 (ETH: sell=18/buy=25/short=35/long=30)
    "sell_threshold": 15,
    "buy_threshold": 20,
    "short_threshold": 30,
    "long_threshold": 25,
    # 权重微调: 增加动量权重, 降低背离权重 (小币走势更粗暴)
    "w_div": 0.10,
    "w_ma": 0.15,
    "w_candle": 0.10,
    "w_boll": 0.20,
    "w_vol": 0.25,
    "w_kdj": 0.20,
}

# 指标计算参数 (传入各子模块)
HOT_COIN_INDICATOR_PARAMS = {
    "macd_fast": 6,
    "macd_slow": 13,
    "macd_signal": 5,
    "macd_fast_alt": 3,
    "macd_slow_alt": 8,
    "macd_signal_alt": 3,
    "kdj_period": 5,
    "rsi_short": 6,
    "rsi_long": 12,
    "bollinger_period": 10,
    "bollinger_std": 2.5,
    "volume_ma_period": 10,
    "ma_periods": [3, 5, 10, 20],
}

# 多周期决策的周期列表和权重
# multi_tf_consensus.py 已内置所有 TF 权重:
#   {1m:1, 5m:1, 15m:3, 1h:8, ...}
# 热点币只使用 1m/5m/15m/1h
HOT_COIN_TIMEFRAMES = ["1m", "5m", "15m", "1h"]

# 多周期共识阈值 (传入 fuse_tf_scores 的 config)
HOT_COIN_CONSENSUS_CONFIG = {
    "short_threshold": 20,     # 比 ETH 25 更灵敏
    "long_threshold": 35,      # 比 ETH 40 更灵敏
    "coverage_min": 0.5,
}

# K 线获取参数
HOT_COIN_KLINE_PARAMS = {
    "1m": {"days": 1, "min_bars": 30},
    "5m": {"days": 3, "min_bars": 30},
    "15m": {"days": 7, "min_bars": 50},
    "1h": {"days": 30, "min_bars": 50},
}
