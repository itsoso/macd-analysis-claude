"""
Phase 3d: OI 自采集落库

定时任务每小时从 Binance API 拉取最新 Open Interest snapshot,
解决回测中 OI 长期历史缺失的根因。

存储: data/oi/ 目录, Parquet 格式 (按月分文件)

使用方式:
    # 一次性采集
    python oi_collector.py --symbol ETHUSDT

    # 以守护进程模式运行 (每小时采集)
    python oi_collector.py --symbol ETHUSDT --daemon

    # 查看已采集数据
    python oi_collector.py --symbol ETHUSDT --show

    # 回填历史 (尝试从 API 获取尽可能多的历史数据)
    python oi_collector.py --symbol ETHUSDT --backfill --days 30

    # 配合 cron 使用 (推荐):
    # 0 * * * * cd /opt/macd-analysis && /opt/macd-analysis/venv/bin/python oi_collector.py --symbol ETHUSDT >> /var/log/oi_collector.log 2>&1
"""

import os
import sys
import json
import time
import logging
import requests
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

_OI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'oi')
_BASE_URL = 'https://fapi.binance.com'


def fetch_current_oi(symbol='ETHUSDT'):
    """
    获取当前 Open Interest。

    Returns
    -------
    dict or None
        {'openInterest': float, 'symbol': str, 'time': int}
    """
    try:
        url = f'{_BASE_URL}/fapi/v1/openInterest'
        resp = requests.get(url, params={'symbol': symbol}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'symbol': symbol,
                'openInterest': float(data.get('openInterest', 0)),
                'time': int(data.get('time', 0)),
            }
        else:
            logger.warning(f"OI fetch failed: {resp.status_code} {resp.text[:200]}")
            return None
    except Exception as e:
        logger.error(f"OI fetch error: {e}")
        return None


def fetch_oi_history(symbol='ETHUSDT', period='1h', limit=500):
    """
    获取 OI 历史数据 (Binance openInterestHist, 最多 30 天)。

    Parameters
    ----------
    symbol : str
        交易对
    period : str
        时间间隔: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
    limit : int
        最大返回条数 (最大 500)

    Returns
    -------
    list[dict]
        OI 历史记录
    """
    try:
        url = f'{_BASE_URL}/futures/data/openInterestHist'
        resp = requests.get(url, params={
            'symbol': symbol,
            'period': period,
            'limit': min(limit, 500),
        }, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            records = []
            for item in data:
                records.append({
                    'timestamp': datetime.utcfromtimestamp(
                        int(item['timestamp']) / 1000
                    ).isoformat(),
                    'symbol': symbol,
                    'openInterest': float(item.get('sumOpenInterest', 0)),
                    'openInterestValue': float(item.get('sumOpenInterestValue', 0)),
                })
            return records
        else:
            logger.warning(f"OI hist fetch failed: {resp.status_code}")
            return []
    except Exception as e:
        logger.error(f"OI hist fetch error: {e}")
        return []


def fetch_funding_rate(symbol='ETHUSDT', limit=100):
    """
    获取近期资金费率。

    Returns
    -------
    list[dict]
    """
    try:
        url = f'{_BASE_URL}/fapi/v1/fundingRate'
        resp = requests.get(url, params={
            'symbol': symbol,
            'limit': min(limit, 1000),
        }, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            records = []
            for item in data:
                records.append({
                    'timestamp': datetime.utcfromtimestamp(
                        int(item['fundingTime']) / 1000
                    ).isoformat(),
                    'symbol': symbol,
                    'fundingRate': float(item.get('fundingRate', 0)),
                    'markPrice': float(item.get('markPrice', 0)),
                })
            return records
        else:
            logger.warning(f"Funding rate fetch failed: {resp.status_code}")
            return []
    except Exception as e:
        logger.error(f"Funding rate fetch error: {e}")
        return []


def save_oi_snapshot(records, symbol='ETHUSDT'):
    """
    保存 OI 数据到 Parquet 文件 (按月分文件)。

    Parameters
    ----------
    records : list[dict]
        OI 数据记录
    symbol : str
        交易对
    """
    import pandas as pd

    os.makedirs(_OI_DIR, exist_ok=True)

    if not records:
        return

    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # 按月分文件
    months = df.index.to_period('M').unique()
    for month in months:
        month_str = str(month)
        filepath = os.path.join(_OI_DIR, f'{symbol}_oi_{month_str}.parquet')

        month_data = df[df.index.to_period('M') == month]

        if os.path.exists(filepath):
            existing = pd.read_parquet(filepath)
            combined = pd.concat([existing, month_data])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined.sort_index(inplace=True)
            combined.to_parquet(filepath)
            logger.info(f"OI updated: {filepath} ({len(combined)} records)")
        else:
            month_data.to_parquet(filepath)
            logger.info(f"OI created: {filepath} ({len(month_data)} records)")


def load_oi_data(symbol='ETHUSDT', start=None, end=None):
    """
    加载已采集的 OI 数据。

    Parameters
    ----------
    symbol : str
        交易对
    start : str or datetime
        起始时间
    end : str or datetime
        结束时间

    Returns
    -------
    pd.DataFrame or None
    """
    import pandas as pd
    import glob

    pattern = os.path.join(_OI_DIR, f'{symbol}_oi_*.parquet')
    files = sorted(glob.glob(pattern))

    if not files:
        return None

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs)
    df = df[~df.index.duplicated(keep='last')]
    df.sort_index(inplace=True)

    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]

    return df


def collect_once(symbol='ETHUSDT'):
    """
    执行一次采集: 当前 OI + 历史 OI (最近 30 天) + 资金费率。
    """
    logger.info(f"开始采集 {symbol} OI 数据...")

    # 1. 当前 OI snapshot
    current = fetch_current_oi(symbol)
    if current:
        logger.info(f"当前 OI: {current['openInterest']:.4f} {symbol}")
        save_oi_snapshot([current], symbol)

    # 2. 历史 OI (尽量多拉)
    history = fetch_oi_history(symbol, period='1h', limit=500)
    if history:
        logger.info(f"历史 OI: {len(history)} 条记录")
        save_oi_snapshot(history, symbol)

    # 3. 资金费率
    funding = fetch_funding_rate(symbol, limit=100)
    if funding:
        import pandas as pd
        os.makedirs(_OI_DIR, exist_ok=True)
        df = pd.DataFrame(funding)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        filepath = os.path.join(_OI_DIR, f'{symbol}_funding.parquet')
        if os.path.exists(filepath):
            existing = pd.read_parquet(filepath)
            combined = pd.concat([existing, df])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined.sort_index(inplace=True)
            combined.to_parquet(filepath)
        else:
            df.to_parquet(filepath)
        logger.info(f"资金费率: {len(funding)} 条记录")

    logger.info("采集完成")


def run_daemon(symbol='ETHUSDT', interval_sec=3600):
    """
    守护进程模式: 每小时采集一次。

    Parameters
    ----------
    symbol : str
        交易对
    interval_sec : int
        采集间隔 (秒, 默认 3600)
    """
    import signal as sig

    _stop = [False]

    def _handler(signum, frame):
        logger.info("OI collector daemon stopping...")
        _stop[0] = True

    sig.signal(sig.SIGINT, _handler)
    sig.signal(sig.SIGTERM, _handler)

    logger.info(f"OI collector daemon started: {symbol}, interval={interval_sec}s")

    while not _stop[0]:
        try:
            collect_once(symbol)
        except Exception as e:
            logger.error(f"采集异常: {e}")

        # 等待下一次采集
        next_run = time.time() + interval_sec
        while time.time() < next_run and not _stop[0]:
            time.sleep(10)

    logger.info("OI collector daemon stopped")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='OI 自采集落库')
    parser.add_argument('--symbol', default='ETHUSDT', help='交易对')
    parser.add_argument('--daemon', action='store_true', help='守护进程模式')
    parser.add_argument('--show', action='store_true', help='查看已采集数据')
    parser.add_argument('--backfill', action='store_true', help='回填历史数据')
    parser.add_argument('--days', type=int, default=30, help='回填天数')
    parser.add_argument('--interval', type=int, default=3600,
                        help='采集间隔 (秒, daemon 模式)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    if args.show:
        df = load_oi_data(args.symbol)
        if df is None or len(df) == 0:
            print(f"无数据: {args.symbol}")
            return
        print(f"\n{args.symbol} OI 数据:")
        print(f"  总记录数: {len(df)}")
        print(f"  时间范围: {df.index[0]} → {df.index[-1]}")
        print(f"  覆盖天数: {(df.index[-1] - df.index[0]).days}")
        print(f"\n最近数据:")
        print(df.tail(10).to_string())
    elif args.backfill:
        print(f"回填 {args.symbol} 历史 OI ({args.days} 天)...")
        history = fetch_oi_history(args.symbol, period='1h', limit=500)
        if history:
            save_oi_snapshot(history, args.symbol)
            print(f"  保存 {len(history)} 条记录")
        else:
            print("  无法获取历史数据 (API 限制约 30 天)")
        # 也获取资金费率
        collect_once(args.symbol)
    elif args.daemon:
        run_daemon(args.symbol, interval_sec=args.interval)
    else:
        collect_once(args.symbol)


if __name__ == '__main__':
    main()
