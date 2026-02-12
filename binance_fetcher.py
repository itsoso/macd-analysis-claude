"""
币安 ETH/USDT 数据获取模块
使用币安公开API获取K线数据, 无需API Key
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError


BINANCE_KLINE_URL = "https://api.binance.com/api/v3/klines"

INTERVAL_MAP = {
    '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
    '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
    '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M',
    '10m': '15m',  # 币安没有10m, 用15m代替后重采样
}


def fetch_binance_klines(symbol: str = "ETHUSDT",
                         interval: str = "15m",
                         days: int = 30,
                         limit_per_request: int = 1000) -> pd.DataFrame:
    """
    从币安获取K线数据

    参数:
        symbol: 交易对 (如 ETHUSDT)
        interval: K线周期 (1m/5m/15m/30m/1h/4h/1d)
        days: 获取最近多少天的数据
        limit_per_request: 每次请求最多返回条数

    返回: DataFrame with open, high, low, close, volume
    """
    all_klines = []
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    # 非标准周期: 用更小周期获取后重采样
    actual_interval = interval
    need_resample = False
    resample_rule = None

    RESAMPLE_MAP = {
        '10m': ('5m',  '10min'),
        '3h':  ('1h',  '3h'),
        '16h': ('4h',  '16h'),
        '24h': ('4h',  '24h'),
        '32h': ('4h',  '32h'),
    }

    if interval in RESAMPLE_MAP:
        actual_interval, resample_rule = RESAMPLE_MAP[interval]
        need_resample = True
        days = int(days * 1.15)  # 多取一些数据用于重采样对齐
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    current_start = start_time

    print(f"正在从币安获取 {symbol} {interval} K线数据 (最近{days}天)...")

    while current_start < end_time:
        url = (f"{BINANCE_KLINE_URL}?symbol={symbol}"
               f"&interval={actual_interval}"
               f"&startTime={current_start}"
               f"&limit={limit_per_request}")

        try:
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode())
        except Exception as e:
            print(f"  请求失败: {e}, 重试中...")
            time.sleep(2)
            try:
                req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urlopen(req, timeout=15) as response:
                    data = json.loads(response.read().decode())
            except Exception as e2:
                print(f"  重试仍失败: {e2}")
                break

        if not data:
            break

        all_klines.extend(data)
        current_start = data[-1][0] + 1  # 下一批从最后一条之后开始

        if len(data) < limit_per_request:
            break

        time.sleep(0.2)  # 避免限速

    if not all_klines:
        print("未获取到数据!")
        return pd.DataFrame()

    # 解析数据
    df = pd.DataFrame(all_klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('date')

    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ── 剔除未闭合的最后一根K线 ──
    # close_time 是该K线的结束时间戳(ms), 如果 > 当前时间说明K线尚未收盘
    now_ms = int(time.time() * 1000)
    df['_close_time_ms'] = pd.to_numeric(df['close_time'], errors='coerce')
    n_before = len(df)
    df = df[df['_close_time_ms'] <= now_ms]
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  剔除 {n_dropped} 根未闭合K线")
    df = df.drop(columns=['_close_time_ms'])

    df = df[['open', 'high', 'low', 'close', 'volume', 'quote_volume']].copy()
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    # 如果需要重采样到10分钟
    if need_resample and resample_rule:
        print(f"  将 {actual_interval} 数据重采样为 {interval}...")
        df = df.resample(resample_rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'quote_volume': 'sum'
        }).dropna()

    # 去掉时区信息使后续处理一致
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    print(f"  获取完成: {len(df)} 条 {interval} K线 "
          f"({df.index[0].strftime('%Y-%m-%d %H:%M')} ~ "
          f"{df.index[-1].strftime('%Y-%m-%d %H:%M')})")

    return df


if __name__ == '__main__':
    df = fetch_binance_klines("ETHUSDT", interval="10m", days=30)
    print(f"\n数据预览:\n{df.tail(10)}")
    print(f"\n数据统计:\n{df.describe()}")
