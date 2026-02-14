"""
币安 ETH/USDT 数据获取模块
优先使用本地 Parquet 缓存, 无本地数据时回退到币安API
"""

import os
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

# ── 本地缓存路径 ──
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_KLINE_DIR = os.path.join(_BASE_DIR, 'data', 'klines')


def _try_load_local(symbol: str, interval: str, days: int) -> pd.DataFrame | None:
    """
    尝试从本地 Parquet 加载K线数据。
    策略: 本地有数据就用本地, 最大化利用缓存 (回测数据不需要实时)。
    只有在本地数据太少 (<100条) 或完全没有时才返回 None。
    """
    path = os.path.join(_KLINE_DIR, symbol, f'{interval}.parquet')
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        # 区分文件不存在/损坏/权限等，避免静默传播到回测
        import sys
        print(f"[binance_fetcher] 读取 Parquet 失败 {path!r}: {type(e).__name__}: {e}", file=sys.stderr)
        return None

    if df is None or len(df) < 100:
        return None

    # 计算请求的起止时间
    end_dt = pd.Timestamp.now()
    start_dt = end_dt - pd.Timedelta(days=days)

    local_start = df.index[0]
    local_end = df.index[-1]

    # 如果请求的起始时间早于本地数据, 仍然使用本地数据
    # (回测时 days 往往设很大做buffer, 实际只需要本地覆盖范围)
    # 只有当本地数据完全不覆盖请求范围时才放弃
    if local_end < start_dt:
        return None

    # 如果请求起始早于本地数据起始, 从本地起始开始
    actual_start = max(start_dt - pd.Timedelta(days=5), local_start)
    df = df[df.index >= actual_start]

    # 确保去时区
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # 只保留标准列 (与API返回格式一致)
    standard_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
    keep = [c for c in standard_cols if c in df.columns]
    df = df[keep].copy()
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    print(f"  [本地] {symbol} {interval} 加载 {len(df)} 条K线 "
          f"({df.index[0].strftime('%Y-%m-%d %H:%M')} ~ "
          f"{df.index[-1].strftime('%Y-%m-%d %H:%M')})")

    return df


def fetch_binance_klines(symbol: str = "ETHUSDT",
                         interval: str = "15m",
                         days: int = 30,
                         limit_per_request: int = 1000,
                         force_api: bool = False) -> pd.DataFrame:
    """
    获取K线数据: 优先本地 Parquet, 无则走API

    参数:
        symbol: 交易对 (如 ETHUSDT)
        interval: K线周期 (1m/5m/15m/30m/1h/4h/1d/24h)
        days: 获取最近多少天的数据
        limit_per_request: 每次API请求最多返回条数
        force_api: 强制走API (跳过本地缓存)

    返回: DataFrame with open, high, low, close, volume, quote_volume
    """
    # ── 先尝试本地缓存 ──
    if not force_api:
        local_df = _try_load_local(symbol, interval, days)
        if local_df is not None:
            return local_df

    # ── 本地无数据, 走API ──
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

    print(f"正在从币安API获取 {symbol} {interval} K线数据 (最近{days}天)...")

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
