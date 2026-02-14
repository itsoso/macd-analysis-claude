#!/usr/bin/env python3
"""
本地K线数据存储层
==================
从币安下载 ETH/USDT 多时间维度K线数据，保存为 Parquet 文件。
回测时直接读取本地数据，不再走API。

存储路径: data/klines/{symbol}/{interval}.parquet
支持周期: 15m, 1h, 4h, 24h (24h 从 4h 重采样)

用法:
    # 下载/更新全部数据
    python kline_store.py download

    # 查看本地数据状态
    python kline_store.py status

    # 校验数据完整性
    python kline_store.py verify

    # 在回测代码中加载
    from kline_store import load_klines
    df = load_klines('ETHUSDT', '1h', '2025-01-01', '2026-01-31')
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from urllib.request import urlopen, Request

# ──────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KLINE_DIR = os.path.join(BASE_DIR, 'data', 'klines')

BINANCE_KLINE_URL = "https://api.binance.com/api/v3/klines"

# 需要下载的原生周期 (币安原生支持的)
NATIVE_INTERVALS = ['15m', '30m', '1h', '4h', '8h', '12h']
# 重采样周期: 目标 -> (源周期, pandas resample rule)
RESAMPLE_INTERVALS = {
    '24h': ('4h', '24h'),
}
# 全部可用周期
ALL_INTERVALS = NATIVE_INTERVALS + list(RESAMPLE_INTERVALS.keys())

# 默认下载时间范围
DEFAULT_START = '2024-01-01'
DEFAULT_END = '2026-02-28'

# K线列定义 (与币安API返回对齐)
KLINE_COLUMNS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
    'taker_buy_quote', 'ignore'
]

# 每个周期的分钟数 (用于完整性校验)
INTERVAL_MINUTES = {
    '15m': 15, '1h': 60, '4h': 240, '24h': 1440,
    '10m': 10, '30m': 30, '2h': 120, '6h': 360,
    '8h': 480, '12h': 720,
}


# ──────────────────────────────────────────────────────────
# 下载
# ──────────────────────────────────────────────────────────
def _fetch_raw_klines(symbol, interval, start_ms, end_ms, limit=1000):
    """从币安API拉取原始K线数据, 返回 list of lists"""
    all_klines = []
    current_start = start_ms

    while current_start < end_ms:
        url = (f"{BINANCE_KLINE_URL}?symbol={symbol}"
               f"&interval={interval}"
               f"&startTime={current_start}"
               f"&endTime={end_ms}"
               f"&limit={limit}")

        for attempt in range(3):
            try:
                req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urlopen(req, timeout=30) as response:
                    data = json.loads(response.read().decode())
                break
            except Exception as e:
                if attempt == 2:
                    print(f"    ✗ 请求失败 ({e}), 跳过")
                    return all_klines
                time.sleep(2 ** attempt)

        if not data:
            break

        all_klines.extend(data)
        current_start = data[-1][0] + 1

        if len(data) < limit:
            break

        time.sleep(0.15)  # 避免限速

    return all_klines


def _raw_to_dataframe(raw_klines, interval):
    """将原始K线数据转为 DataFrame"""
    if not raw_klines:
        return pd.DataFrame()

    df = pd.DataFrame(raw_klines, columns=KLINE_COLUMNS)
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('date')

    numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                    'quote_volume', 'taker_buy_base', 'taker_buy_quote', 'trades']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 保留完整列 (包括 taker 数据, 供微结构分析)
    keep_cols = ['open', 'high', 'low', 'close', 'volume',
                 'quote_volume', 'taker_buy_base', 'taker_buy_quote', 'trades',
                 'open_time', 'close_time']
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # 剔除未闭合K线
    now_ms = int(time.time() * 1000)
    if 'close_time' in df.columns:
        ct = pd.to_numeric(df['close_time'], errors='coerce')
        df = df[ct <= now_ms]

    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    # 去掉时区
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df


def _resample_df(df, target_interval, rule):
    """将小周期K线重采样为大周期"""
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum',
    }
    if 'taker_buy_base' in df.columns:
        agg['taker_buy_base'] = 'sum'
    if 'taker_buy_quote' in df.columns:
        agg['taker_buy_quote'] = 'sum'
    if 'trades' in df.columns:
        agg['trades'] = 'sum'

    result = df.resample(rule).agg(agg).dropna(subset=['close'])
    return result


def _save_path(symbol, interval):
    """获取本地存储路径"""
    d = os.path.join(KLINE_DIR, symbol)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f'{interval}.parquet')


def download_klines(symbol='ETHUSDT', intervals=None,
                    start_date=DEFAULT_START, end_date=DEFAULT_END):
    """
    下载K线数据并保存到本地 Parquet 文件。
    支持增量更新: 如果本地已有数据, 只下载缺失部分。
    """
    if intervals is None:
        intervals = ALL_INTERVALS

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)  # 包含end_date当天
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(min(end_dt.timestamp(), time.time()) * 1000)

    print(f"{'=' * 70}")
    print(f"  K线数据下载: {symbol}")
    print(f"  时间范围: {start_date} ~ {end_date}")
    print(f"  目标周期: {', '.join(intervals)}")
    print(f"{'=' * 70}")

    # 第一步: 下载原生周期
    native_data = {}
    for interval in NATIVE_INTERVALS:
        if interval not in intervals and not any(
            src == interval for src, _ in RESAMPLE_INTERVALS.values()
            if RESAMPLE_INTERVALS.get(interval) is not None or interval in [v[0] for v in RESAMPLE_INTERVALS.values()]
        ):
            # 检查是否作为重采样源被需要
            needed_as_source = any(
                src == interval for target, (src, _) in RESAMPLE_INTERVALS.items()
                if target in intervals
            )
            if interval not in intervals and not needed_as_source:
                continue

        path = _save_path(symbol, interval)
        existing_df = None
        actual_start_ms = start_ms

        # 增量更新: 加载已有数据
        if os.path.exists(path):
            try:
                existing_df = pd.read_parquet(path)
                if len(existing_df) > 0:
                    last_ts = existing_df.index[-1]
                    last_ms = int(last_ts.timestamp() * 1000)
                    # 从已有数据的最后一根之后开始下载
                    if last_ms > start_ms:
                        actual_start_ms = last_ms + 1
                        print(f"\n  [{interval}] 已有 {len(existing_df)} 条, "
                              f"最后: {last_ts}, 增量下载...")
            except Exception:
                existing_df = None

        if actual_start_ms >= end_ms:
            print(f"\n  [{interval}] 数据已是最新, 跳过")
            if existing_df is not None:
                native_data[interval] = existing_df
            continue

        print(f"\n  [{interval}] 下载中 "
              f"({pd.Timestamp(actual_start_ms, unit='ms')} ~ "
              f"{pd.Timestamp(end_ms, unit='ms')})...")

        raw = _fetch_raw_klines(symbol, interval, actual_start_ms, end_ms)
        new_df = _raw_to_dataframe(raw, interval)

        if len(new_df) > 0:
            print(f"    获取 {len(new_df)} 条新数据")

        # 合并已有数据
        if existing_df is not None and len(existing_df) > 0 and len(new_df) > 0:
            combined = pd.concat([existing_df, new_df])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
        elif existing_df is not None and len(existing_df) > 0:
            combined = existing_df
        else:
            combined = new_df

        # 裁剪到目标范围
        combined = combined[combined.index >= start_dt]
        if combined.index.max() > end_dt:
            combined = combined[combined.index <= end_dt]

        # 保存
        if len(combined) > 0:
            combined.to_parquet(path, engine='pyarrow')
            print(f"    ✓ 保存 {len(combined)} 条 → {path}")
            print(f"      范围: {combined.index[0]} ~ {combined.index[-1]}")
            native_data[interval] = combined
        else:
            print(f"    ✗ 无数据")

    # 第二步: 重采样生成衍生周期
    for target, (source, rule) in RESAMPLE_INTERVALS.items():
        if target not in intervals:
            continue

        if source not in native_data:
            print(f"\n  [{target}] 缺少源数据 {source}, 跳过")
            continue

        print(f"\n  [{target}] 从 {source} 重采样...")
        resampled = _resample_df(native_data[source], target, rule)

        if len(resampled) > 0:
            path = _save_path(symbol, target)
            resampled.to_parquet(path, engine='pyarrow')
            print(f"    ✓ 保存 {len(resampled)} 条 → {path}")
            print(f"      范围: {resampled.index[0]} ~ {resampled.index[-1]}")

    print(f"\n{'=' * 70}")
    print(f"  下载完成!")
    print(f"{'=' * 70}")


# ──────────────────────────────────────────────────────────
# 加载 (供回测使用)
# ──────────────────────────────────────────────────────────
def load_klines(symbol='ETHUSDT', interval='1h',
                start=None, end=None, with_indicators=True,
                allow_api_fallback=True):
    """
    从本地加载K线数据。可配置是否在本地缺失时回退到API。

    参数:
        symbol: 交易对
        interval: 周期 (15m/1h/4h/24h)
        start: 起始日期 (str 或 Timestamp, 可选)
        end: 结束日期 (str 或 Timestamp, 可选)
        with_indicators: 是否自动添加技术指标
        allow_api_fallback: 本地缺失时是否允许回退API

    返回: DataFrame (与 fetch_binance_klines + add_all_indicators 输出格式一致)
    """
    path = _save_path(symbol, interval)

    if not os.path.exists(path):
        if not allow_api_fallback:
            print(f"  [{interval}] 本地无数据, 且已禁用API回退")
            return None
        print(f"  [{interval}] 本地无数据, 回退到API...")
        from binance_fetcher import fetch_binance_klines
        days = 800  # 足够覆盖回测范围
        if start:
            start_dt = pd.Timestamp(start)
            days = max(30, (pd.Timestamp.now() - start_dt).days + 60)
        df = fetch_binance_klines(symbol, interval=interval, days=days)
        if df is not None and with_indicators:
            from indicators import add_all_indicators
            from ma_indicators import add_moving_averages
            df = add_all_indicators(df)
            add_moving_averages(df, timeframe=interval)
        return df

    df = pd.read_parquet(path)
    print(f"  [{interval}] 本地加载 {len(df)} 条K线 "
          f"({df.index[0]} ~ {df.index[-1]})")

    # 时间范围裁剪
    if start:
        start_dt = pd.Timestamp(start)
        # 保留预热数据: 向前多留200根K线
        warmup_bars = 200
        minutes = INTERVAL_MINUTES.get(interval, 60)
        warmup_delta = pd.Timedelta(minutes=minutes * warmup_bars)
        df = df[df.index >= (start_dt - warmup_delta)]

    if end:
        end_dt = pd.Timestamp(end) + pd.Timedelta(days=1)
        df = df[df.index <= end_dt]

    # 确保去时区
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # 只保留回测需要的标准列 (与 fetch_binance_klines 输出对齐)
    standard_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
    extra_cols = ['taker_buy_base', 'taker_buy_quote', 'trades']
    keep = [c for c in standard_cols + extra_cols if c in df.columns]
    df = df[keep].copy()

    if with_indicators:
        from indicators import add_all_indicators
        from ma_indicators import add_moving_averages
        df = add_all_indicators(df)
        add_moving_averages(df, timeframe=interval)

    return df


# ──────────────────────────────────────────────────────────
# 状态查看 & 校验
# ──────────────────────────────────────────────────────────
def show_status(symbol='ETHUSDT'):
    """显示本地K线数据状态"""
    d = os.path.join(KLINE_DIR, symbol)
    if not os.path.exists(d):
        print(f"  无本地数据: {d}")
        return

    print(f"\n{'=' * 70}")
    print(f"  本地K线数据状态: {symbol}")
    print(f"{'=' * 70}")
    print(f"  {'周期':>6} | {'条数':>8} | {'起始':>20} | {'结束':>20} | {'文件大小':>10}")
    print(f"  {'-' * 68}")

    for f in sorted(os.listdir(d)):
        if not f.endswith('.parquet'):
            continue
        interval = f.replace('.parquet', '')
        path = os.path.join(d, f)
        try:
            df = pd.read_parquet(path)
            size = os.path.getsize(path)
            size_str = f"{size / 1024 / 1024:.1f}MB" if size > 1024 * 1024 else f"{size / 1024:.0f}KB"
            print(f"  {interval:>6} | {len(df):>8,} | {str(df.index[0])[:19]:>20} | "
                  f"{str(df.index[-1])[:19]:>20} | {size_str:>10}")
        except Exception as e:
            print(f"  {interval:>6} | 读取失败: {e}")

    print()


def verify_completeness(symbol='ETHUSDT', start_date=DEFAULT_START, end_date=DEFAULT_END):
    """校验数据完整性: 检查缺失的K线"""
    d = os.path.join(KLINE_DIR, symbol)
    if not os.path.exists(d):
        print("  无本地数据")
        return False

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    all_ok = True

    print(f"\n{'=' * 70}")
    print(f"  数据完整性校验: {symbol} ({start_date} ~ {end_date})")
    print(f"{'=' * 70}")

    for interval in ALL_INTERVALS:
        path = _save_path(symbol, interval)
        if not os.path.exists(path):
            print(f"  [{interval}] ✗ 文件不存在")
            all_ok = False
            continue

        df = pd.read_parquet(path)
        # 裁剪到检查范围
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]

        if len(df) == 0:
            print(f"  [{interval}] ✗ 范围内无数据")
            all_ok = False
            continue

        minutes = INTERVAL_MINUTES.get(interval, 60)
        freq = pd.Timedelta(minutes=minutes)

        # 构建期望的时间索引
        expected = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq)
        actual = set(df.index)

        missing = [t for t in expected if t not in actual]
        gap_pct = len(missing) / len(expected) * 100 if len(expected) > 0 else 0

        # 检查数值异常
        null_rows = df[['open', 'high', 'low', 'close']].isnull().any(axis=1).sum()
        zero_close = (df['close'] == 0).sum()

        status = "✓" if gap_pct < 1.0 and null_rows == 0 and zero_close == 0 else "✗"
        if status == "✗":
            all_ok = False

        print(f"  [{interval}] {status} 实际:{len(df):,}条 | "
              f"期望:{len(expected):,}条 | 缺失:{len(missing)}({gap_pct:.2f}%) | "
              f"空值:{null_rows} | 零值:{zero_close}")

        if missing and len(missing) <= 10:
            for t in missing[:5]:
                print(f"         缺失: {t}")

    print(f"\n  {'全部通过 ✓' if all_ok else '存在问题 ✗'}")
    return all_ok


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='K线数据本地存储管理')
    parser.add_argument('action', choices=['download', 'status', 'verify'],
                        help='download=下载/更新, status=查看状态, verify=校验完整性')
    parser.add_argument('--symbol', default='ETHUSDT', help='交易对')
    parser.add_argument('--start', default=DEFAULT_START, help='起始日期')
    parser.add_argument('--end', default=DEFAULT_END, help='结束日期')
    parser.add_argument('--intervals', nargs='+', default=None,
                        help='指定周期 (默认全部)')
    args = parser.parse_args()

    if args.action == 'download':
        download_klines(args.symbol, args.intervals, args.start, args.end)
    elif args.action == 'status':
        show_status(args.symbol)
    elif args.action == 'verify':
        verify_completeness(args.symbol, args.start, args.end)


if __name__ == '__main__':
    main()
