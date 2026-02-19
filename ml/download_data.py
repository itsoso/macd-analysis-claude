"""
数据下载脚本 — 在可访问外网的机器上运行

从币安 API 下载 K线数据并保存为 Parquet 文件。
然后 SCP 到 H800 训练机。

用法 (在外网机器上):
    cd /workspace/macd-analysis-claude
    python ml/download_data.py
    # 数据保存在 ml/data/raw/

然后 SCP:
    scp -r ml/data/raw/ panbaokun@H800:~/macd-analysis/ml/data/raw/
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from binance_fetcher import fetch_binance_klines
from indicators import add_all_indicators
from ml.config import SYMBOL, TIMEFRAMES, FETCH_DAYS

RAW_DIR = os.path.join("ml", "data", "raw")


def download_all():
    os.makedirs(RAW_DIR, exist_ok=True)

    for tf in TIMEFRAMES:
        print(f"[download] 获取 {SYMBOL} {tf} (days={FETCH_DAYS})...")
        df = fetch_binance_klines(symbol=SYMBOL, interval=tf, days=FETCH_DAYS)
        if df is None or len(df) == 0:
            print(f"  !! 获取失败, 跳过 {tf}")
            continue

        # 保存原始 OHLCV
        raw_path = os.path.join(RAW_DIR, f"{SYMBOL}_{tf}_raw.parquet")
        df.to_parquet(raw_path)
        print(f"  -> {len(df)} bars, 保存到 {raw_path}")

        # 添加指标后保存
        df_ind = add_all_indicators(df)
        ind_path = os.path.join(RAW_DIR, f"{SYMBOL}_{tf}_indicators.parquet")
        df_ind.to_parquet(ind_path)
        print(f"  -> 含指标版本: {df_ind.shape[1]} 列, 保存到 {ind_path}")

    print(f"\n[download] 完成! 数据在 {RAW_DIR}/")
    print(f"可以 SCP 到 H800:")
    print(f"  scp -r {RAW_DIR}/ panbaokun@H800:~/macd-analysis/{RAW_DIR}/")


if __name__ == "__main__":
    download_all()
