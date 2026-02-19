"""
生成模拟 K线数据用于测试训练管线 (无需外网)

用法:
    python ml/generate_mock_data.py
"""

import os
import sys
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ml.config import SYMBOL, TIMEFRAMES

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")


def generate_mock_klines(tf: str, n_bars: int, start_price: float = 2000.0):
    """生成带有真实统计特性的模拟 K线数据"""
    np.random.seed(42)

    # 时间索引
    freq_map = {
        "1h": "1h", "4h": "4h", "8h": "8h", "24h": "24h",
        "15m": "15min", "1d": "1D",
    }
    freq = freq_map.get(tf, "1h")
    dates = pd.date_range("2022-01-01", periods=n_bars, freq=freq)

    # 生成价格序列 (几何布朗运动 + 趋势切换)
    mu = 0.0001  # 微小正漂移
    sigma = 0.02  # 波动率
    returns = np.random.normal(mu, sigma, n_bars)

    # 加入趋势切换
    regime_len = n_bars // 5
    for i in range(0, n_bars, regime_len):
        end = min(i + regime_len, n_bars)
        trend = np.random.choice([-0.001, 0.0, 0.001])
        returns[i:end] += trend

    close = start_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n_bars)))
    open_ = close * (1 + np.random.normal(0, 0.003, n_bars))

    # 确保 high >= max(open, close), low <= min(open, close)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = np.random.lognormal(mean=10, sigma=1, size=n_bars)
    quote_volume = volume * close

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "quote_volume": quote_volume,
        "taker_buy_base": volume * 0.5,
        "taker_buy_quote": quote_volume * 0.5,
    }, index=dates)

    return df


def generate_all():
    os.makedirs(RAW_DIR, exist_ok=True)

    bars_map = {"1h": 17520, "4h": 4380, "8h": 2190, "24h": 730}

    for tf in TIMEFRAMES:
        n_bars = bars_map.get(tf, 5000)
        print(f"[mock] 生成 {SYMBOL} {tf} 模拟数据 ({n_bars} bars)...")

        df = generate_mock_klines(tf, n_bars)

        # 添加指标
        from indicators import add_all_indicators
        df_ind = add_all_indicators(df)

        # 保存
        raw_path = os.path.join(RAW_DIR, f"{SYMBOL}_{tf}_raw.parquet")
        ind_path = os.path.join(RAW_DIR, f"{SYMBOL}_{tf}_indicators.parquet")
        df.to_parquet(raw_path)
        df_ind.to_parquet(ind_path)
        print(f"  -> 保存到 {ind_path} ({df_ind.shape[1]} 列)")

    print(f"\n[mock] 模拟数据已生成在 {RAW_DIR}/")


if __name__ == "__main__":
    generate_all()
