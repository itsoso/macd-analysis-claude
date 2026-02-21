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
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from urllib.error import HTTPError


BINANCE_KLINE_URL = "https://api.binance.com/api/v3/klines"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
BINANCE_FUTURES_MARK_KLINE_URL = f"{BINANCE_FUTURES_BASE}/fapi/v1/markPriceKlines"
BINANCE_FUTURES_FUNDING_RATE_URL = f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate"
BINANCE_FUTURES_OI_HIST_URL = f"{BINANCE_FUTURES_BASE}/futures/data/openInterestHist"

INTERVAL_MAP = {
    '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
    '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
    '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M',
    '10m': '15m',  # 币安没有10m, 用15m代替后重采样
}

# 各周期对应的小时数 (用于缓存新鲜度检查)
INTERVAL_MAP_HOURS = {
    '1m': 1/60, '3m': 3/60, '5m': 5/60, '10m': 10/60, '15m': 0.25,
    '30m': 0.5, '1h': 1.0, '2h': 2.0, '3h': 3.0, '4h': 4.0,
    '6h': 6.0, '8h': 8.0, '12h': 12.0, '16h': 16.0, '24h': 24.0, '1d': 24.0,
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
    standard_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                     'taker_buy_base', 'taker_buy_quote']
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
                         force_api: bool = False,
                         require_fresh: bool = False,
                         max_lag_hours: float | None = None,
                         allow_api_fallback: bool = True) -> pd.DataFrame:
    """
    获取K线数据: 优先本地 Parquet, 无则走API

    参数:
        symbol: 交易对 (如 ETHUSDT)
        interval: K线周期 (1m/5m/15m/30m/1h/4h/1d/24h)
        days: 获取最近多少天的数据
        limit_per_request: 每次API请求最多返回条数
        force_api: 强制走API (跳过本地缓存)
        require_fresh: True 时要求本地缓存满足新鲜度，否则尝试 API 回退
        max_lag_hours: 本地缓存允许的最大滞后小时数 (None=自动 max(2*周期, 6h))
        allow_api_fallback: 本地缓存不可用/过期时是否允许回退 API

    返回: DataFrame with open, high, low, close, volume, quote_volume
    """
    # ── 先尝试本地缓存 ──
    if not force_api:
        local_df = _try_load_local(symbol, interval, days)
        if local_df is not None:
            if not require_fresh:
                return local_df

            interval_hours = INTERVAL_MAP_HOURS.get(interval, 1.0)
            cfg_lag = 0.0
            if max_lag_hours is not None:
                try:
                    cfg_lag = max(0.0, float(max_lag_hours))
                except Exception:
                    cfg_lag = 0.0
            allowed_lag = max(cfg_lag, interval_hours * 2.0, 6.0)
            cache_lag = (datetime.now() - pd.Timestamp(local_df.index[-1])).total_seconds() / 3600.0

            if cache_lag <= allowed_lag:
                return local_df

            print(
                f"  本地 K线缓存过期: {symbol} {interval} lag={cache_lag:.1f}h "
                f"(max={allowed_lag:.1f}h)"
            )
            if not allow_api_fallback:
                print("  已禁用 API 回退，继续使用本地过期缓存")
                return local_df

    if not allow_api_fallback and not force_api:
        print(f"  [本地] {symbol} {interval} 缓存缺失/无效, 已禁用 API 回退")
        return pd.DataFrame()

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

        data = _api_get_json(url, max_retries=2)

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

    # 保留 taker_buy 列用于微结构分析 (taker imbalance)
    for col in ['taker_buy_base', 'taker_buy_quote']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    keep_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                 'taker_buy_base', 'taker_buy_quote']
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    # 如果需要重采样到10分钟
    if need_resample and resample_rule:
        print(f"  将 {actual_interval} 数据重采样为 {interval}...")
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'quote_volume': 'sum',
        }
        for _tc in ['taker_buy_base', 'taker_buy_quote']:
            if _tc in df.columns:
                agg_dict[_tc] = 'sum'
        df = df.resample(resample_rule).agg(agg_dict).dropna()

    # 去掉时区信息使后续处理一致
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    print(f"  获取完成: {len(df)} 条 {interval} K线 "
          f"({df.index[0].strftime('%Y-%m-%d %H:%M')} ~ "
          f"{df.index[-1].strftime('%Y-%m-%d %H:%M')})")

    return df


# ═══════════════════════════════════════════════════════════════════════
# P22: Binance Futures 衍生品数据获取 (Mark Price / Funding Rate)
# ═══════════════════════════════════════════════════════════════════════

_MARK_DIR = os.path.join(_BASE_DIR, 'data', 'mark_klines')
_FUNDING_DIR = os.path.join(_BASE_DIR, 'data', 'funding_rates')
_OI_DIR = os.path.join(_BASE_DIR, 'data', 'open_interest')

# API 断路器: endpoint 连续失败后进入冷却，避免每个 tick 都卡在重试
_API_CIRCUIT_COOLDOWN_SEC = int(os.environ.get("BINANCE_API_CIRCUIT_COOLDOWN_SEC", "300"))
_API_CIRCUIT_OPEN_UNTIL: dict[str, float] = {}
_API_CIRCUIT_LAST_WARN_TS: dict[str, float] = {}


def _api_endpoint_key(url: str) -> str:
    """将 URL 归一化为 endpoint key (host + path)。"""
    try:
        parsed = urlparse(url)
        return f"{parsed.netloc}{parsed.path}"
    except Exception:
        return url


def _is_api_circuit_open(endpoint_key: str) -> bool:
    """检查 endpoint 是否处于断路器冷却期。"""
    until_ts = _API_CIRCUIT_OPEN_UNTIL.get(endpoint_key, 0.0)
    now_ts = time.time()
    if until_ts <= now_ts:
        if endpoint_key in _API_CIRCUIT_OPEN_UNTIL:
            _API_CIRCUIT_OPEN_UNTIL.pop(endpoint_key, None)
            _API_CIRCUIT_LAST_WARN_TS.pop(endpoint_key, None)
        return False

    # 最多每 60s 提示一次，避免刷屏
    last_warn = _API_CIRCUIT_LAST_WARN_TS.get(endpoint_key, 0.0)
    if now_ts - last_warn >= 60:
        remain = max(0.0, until_ts - now_ts)
        print(f"  API 断路器生效: {endpoint_key} 暂停请求 {remain:.0f}s")
        _API_CIRCUIT_LAST_WARN_TS[endpoint_key] = now_ts
    return True


def _open_api_circuit(endpoint_key: str, reason: str = "") -> None:
    """打开 endpoint 断路器。"""
    until_ts = time.time() + max(1, _API_CIRCUIT_COOLDOWN_SEC)
    _API_CIRCUIT_OPEN_UNTIL[endpoint_key] = until_ts
    _API_CIRCUIT_LAST_WARN_TS[endpoint_key] = time.time()
    extra = f" ({reason})" if reason else ""
    print(
        f"  API 断路器开启: {endpoint_key}{extra}, "
        f"cooldown={_API_CIRCUIT_COOLDOWN_SEC}s"
    )


def _api_get_json(url: str, max_retries: int = 3) -> list:
    """通用 API GET 请求, 带重试。HTTP 4xx 客户端错误不重试(永久失败)。"""
    endpoint_key = _api_endpoint_key(url)
    if _is_api_circuit_open(endpoint_key):
        return []

    for attempt in range(max_retries):
        try:
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req, timeout=20) as response:
                _API_CIRCUIT_OPEN_UNTIL.pop(endpoint_key, None)
                _API_CIRCUIT_LAST_WARN_TS.pop(endpoint_key, None)
                return json.loads(response.read().decode())
        except HTTPError as e:
            # 4xx 客户端错误 (如 400 startTime 超出限制) 无需重试
            if 400 <= e.code < 500:
                print(f"  请求最终失败: {e}")
                return []
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  请求失败 (尝试 {attempt+1}/{max_retries}): {e}, {wait}s 后重试...")
                time.sleep(wait)
            else:
                print(f"  请求最终失败: {e}")
                _open_api_circuit(endpoint_key, reason=f"http_{e.code}")
                return []
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  请求失败 (尝试 {attempt+1}/{max_retries}): {e}, {wait}s 后重试...")
                time.sleep(wait)
            else:
                print(f"  请求最终失败: {e}")
                _open_api_circuit(endpoint_key, reason=type(e).__name__)
                return []


def fetch_mark_price_klines(symbol: str = "ETHUSDT",
                            interval: str = "15m",
                            days: int = 365,
                            force_api: bool = False,
                            allow_api_fallback: bool = True) -> pd.DataFrame:
    """
    获取 Binance Futures Mark Price K线数据

    Mark Price = 现货指数 + 资金费率衰减移动平均, 用于计算未实现 PnL 和强平
    与标准 K 线格式相同, 但基于 Mark Price 而非 Last Trade Price

    返回: DataFrame with mark_open, mark_high, mark_low, mark_close columns
    """
    # ── 先尝试本地缓存 (含新鲜度检查) ──
    if not force_api:
        cache_path = os.path.join(_MARK_DIR, symbol, f'{interval}.parquet')
        if os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                if df is not None and len(df) >= 100:
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    df = df[~df.index.duplicated(keep='first')].sort_index()
                    # 新鲜度检查: 缓存过期 (>6h) 则通过 API 重新拉取
                    interval_hours = INTERVAL_MAP_HOURS.get(interval, 1.0)
                    max_lag_hours = max(interval_hours * 2, 6.0)
                    cache_lag_hours = (datetime.now() - df.index[-1]).total_seconds() / 3600
                    if cache_lag_hours > max_lag_hours:
                        print(f"  Mark kline 缓存已过期 ({cache_lag_hours:.1f}h > {max_lag_hours:.1f}h), 重新从API获取...")
                    else:
                        print(f"  [本地] {symbol} mark {interval} 加载 {len(df)} 条 "
                              f"({df.index[0].strftime('%Y-%m-%d %H:%M')} ~ "
                              f"{df.index[-1].strftime('%Y-%m-%d %H:%M')})")
                        return df
            except Exception as e:
                print(f"  Mark kline 缓存读取失败: {e}")

    if not allow_api_fallback and not force_api:
        print(f"  [本地] {symbol} mark {interval} 缓存缺失/无效, 已禁用 API 回退")
        return pd.DataFrame()

    # ── API 获取 ──
    # Mark klines 支持的 interval 与标准 klines 相同
    actual_interval = INTERVAL_MAP.get(interval, interval)
    # mark price klines limit 最大 1500
    limit = 1500

    all_klines = []
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    current_start = start_time

    print(f"正在从 Binance Futures API 获取 {symbol} Mark Price {interval} K线 (最近{days}天)...")

    while current_start < end_time:
        url = (f"{BINANCE_FUTURES_MARK_KLINE_URL}?symbol={symbol}"
               f"&interval={actual_interval}"
               f"&startTime={current_start}"
               f"&limit={limit}")

        data = _api_get_json(url)
        if not data:
            break

        all_klines.extend(data)
        current_start = data[-1][0] + 1

        if len(data) < limit:
            break
        time.sleep(0.15)

    if not all_klines:
        print("  未获取到 Mark Price 数据!")
        return pd.DataFrame()

    # Mark Price Klines 格式同标准 Klines
    df = pd.DataFrame(all_klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('date')

    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 重命名为 mark_ 前缀
    df = df[['open', 'high', 'low', 'close']].copy()
    df.columns = ['mark_open', 'mark_high', 'mark_low', 'mark_close']
    df = df[~df.index.duplicated(keep='first')].sort_index()

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # 剔除未闭合 K 线 (最后一条如果 close_time > now)
    # (Mark klines 的 volume 字段通常为 0, 无法用 close_time 判断, 简单剔除最后一条)
    if len(df) > 1:
        last_expected_close = df.index[-1] + pd.Timedelta(actual_interval)
        if last_expected_close > pd.Timestamp.now():
            df = df.iloc[:-1]

    # 如果原始 interval 需要重采样 (如 24h)
    RESAMPLE_MAP = {
        '10m': '10min', '3h': '3h', '16h': '16h',
        '24h': '24h', '32h': '32h',
    }
    if interval in RESAMPLE_MAP:
        rule = RESAMPLE_MAP[interval]
        df = df.resample(rule).agg({
            'mark_open': 'first',
            'mark_high': 'max',
            'mark_low': 'min',
            'mark_close': 'last',
        }).dropna()

    # ── 保存缓存 ──
    cache_dir = os.path.join(_MARK_DIR, symbol)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'{interval}.parquet')
    df.to_parquet(cache_path)

    print(f"  Mark Price K线获取完成: {len(df)} 条 "
          f"({df.index[0].strftime('%Y-%m-%d %H:%M')} ~ "
          f"{df.index[-1].strftime('%Y-%m-%d %H:%M')})")
    print(f"  已缓存至: {cache_path}")

    return df


def fetch_funding_rate_history(symbol: str = "ETHUSDT",
                               days: int = 365,
                               force_api: bool = False,
                               allow_api_fallback: bool = True) -> pd.DataFrame:
    """
    获取 Binance Futures 历史资金费率

    Binance funding 结算周期:
    - 标准: 每 8h (00:00/08:00/16:00 UTC)
    - 动态: 当费率触及上下限时可缩至 4h 或 1h (2025-05-02 后)

    返回: DataFrame with columns:
        - funding_rate: 资金费率 (如 0.0001 表示 0.01%)
        - mark_price: 结算时的 mark price
        - funding_interval_hours: 当前结算间隔 (小时)
    """
    def _normalize_funding_df(_df: pd.DataFrame) -> pd.DataFrame:
        """统一 funding 索引为 UTC整点(无时区), 去重排序.

        Binance fundingTime 理论上是整点结算；缓存中的浮点时间漂移会导致
        merge 审计 `orig_count` 被低估，因此这里做一次强制标准化。
        """
        if _df is None or len(_df) == 0:
            return pd.DataFrame()
        _df = _df.copy()
        # 仅保留策略需要的列，兼容历史缓存字段差异
        keep_cols = [c for c in ['funding_rate', 'mark_price_at_funding', 'funding_interval_hours'] if c in _df.columns]
        if keep_cols:
            _df = _df[keep_cols]
        # 索引标准化
        _idx = pd.to_datetime(_df.index, errors='coerce')
        _df = _df.loc[~_idx.isna()].copy()
        _idx = pd.to_datetime(_df.index)
        if getattr(_idx, 'tz', None) is not None:
            _idx = _idx.tz_localize(None)
        # 回归整点，消除秒/毫秒漂移
        _df.index = _idx.round('1h')
        _df = _df[~_df.index.duplicated(keep='last')].sort_index()
        return _df

    # ── 先尝试本地缓存 ──
    if not force_api:
        cache_path = os.path.join(_FUNDING_DIR, f'{symbol}_funding.parquet')
        if os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                if df is not None and len(df) >= 10:
                    df = _normalize_funding_df(df)
                    end_dt = pd.Timestamp.now()
                    start_dt = end_dt - pd.Timedelta(days=days)
                    if allow_api_fallback:
                        # 实盘/在线模式下，要求缓存不超过24小时，过期则尝试API刷新
                        if df.index[-1] >= (end_dt - pd.Timedelta(hours=24)):
                            df = df[df.index >= start_dt]
                            df = df[~df.index.duplicated(keep='first')].sort_index()
                            print(f"  [本地] {symbol} funding rate 加载 {len(df)} 条 "
                                  f"({df.index[0].strftime('%Y-%m-%d %H:%M')} ~ "
                                  f"{df.index[-1].strftime('%Y-%m-%d %H:%M')})")
                            return df
                    else:
                        # 回测离线模式：不要求“新鲜度”，只要本地有历史即可
                        df = df[df.index >= start_dt]
                        df = df[~df.index.duplicated(keep='first')].sort_index()
                        print(f"  [本地] {symbol} funding rate(离线) 加载 {len(df)} 条 "
                              f"({df.index[0].strftime('%Y-%m-%d %H:%M')} ~ "
                              f"{df.index[-1].strftime('%Y-%m-%d %H:%M')})")
                        return df
            except Exception as e:
                print(f"  Funding rate 缓存读取失败: {e}")

    if not allow_api_fallback and not force_api:
        print(f"  [本地] {symbol} funding 缓存缺失/无效, 已禁用 API 回退")
        return pd.DataFrame()

    # ── API 获取 ──
    # fundingRate API limit=1000, 8h 间隔约 3 条/天, 1000 条 ≈ 333 天
    all_records = []
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    current_start = start_time

    print(f"正在从 Binance Futures API 获取 {symbol} 资金费率 (最近{days}天)...")

    while current_start < end_time:
        url = (f"{BINANCE_FUTURES_FUNDING_RATE_URL}?symbol={symbol}"
               f"&startTime={current_start}"
               f"&endTime={end_time}"
               f"&limit=1000")

        data = _api_get_json(url)
        if not data:
            break

        all_records.extend(data)
        # 下一批从最后一条之后
        current_start = data[-1]['fundingTime'] + 1

        if len(data) < 1000:
            break
        time.sleep(0.15)

    if not all_records:
        print("  未获取到 Funding Rate 数据!")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    # fundingTime 强制转整数毫秒，避免浮点精度导致的秒/毫秒漂移
    _ft_ms = pd.to_numeric(df['fundingTime'], errors='coerce').dropna()
    df = df.loc[_ft_ms.index].copy()
    df['fundingTime'] = _ft_ms.astype('int64')
    df['date'] = pd.to_datetime(df['fundingTime'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.set_index('date')

    df['funding_rate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
    if 'markPrice' in df.columns:
        df['mark_price_at_funding'] = pd.to_numeric(df['markPrice'], errors='coerce')
    else:
        df['mark_price_at_funding'] = np.nan

    # 计算 funding 间隔 (小时)
    # 标准 8h, 但 2025-05-02 后可能动态变化
    time_diffs = df.index.to_series().diff()
    df['funding_interval_hours'] = time_diffs.dt.total_seconds() / 3600
    # 第一条无法计算差分, 设为 8h
    df.loc[df.index[0], 'funding_interval_hours'] = 8.0
    # 清理异常值 (间隔 < 0.5h 或 > 24h)
    df['funding_interval_hours'] = df['funding_interval_hours'].clip(0.5, 24.0)

    df = df[['funding_rate', 'mark_price_at_funding', 'funding_interval_hours']].copy()
    df = _normalize_funding_df(df)

    # ── 保存缓存 ──
    os.makedirs(_FUNDING_DIR, exist_ok=True)
    cache_path = os.path.join(_FUNDING_DIR, f'{symbol}_funding.parquet')
    df.to_parquet(cache_path)

    print(f"  Funding Rate 获取完成: {len(df)} 条 "
          f"({df.index[0].strftime('%Y-%m-%d %H:%M')} ~ "
          f"{df.index[-1].strftime('%Y-%m-%d %H:%M')})")
    print(f"  平均间隔: {df['funding_interval_hours'].mean():.1f}h, "
          f"最短: {df['funding_interval_hours'].min():.1f}h, "
          f"最长: {df['funding_interval_hours'].max():.1f}h")
    print(f"  费率: 均值={df['funding_rate'].mean()*100:.4f}%, "
          f"中位={df['funding_rate'].median()*100:.4f}%, "
          f"标准差={df['funding_rate'].std()*100:.4f}%")
    print(f"  已缓存至: {cache_path}")

    return df


def fetch_open_interest_history(symbol: str = "ETHUSDT",
                                interval: str = "15m",
                                days: int = 365,
                                force_api: bool = False,
                                allow_api_fallback: bool = True) -> pd.DataFrame:
    """
    获取 Binance Futures 历史 Open Interest 数据

    Binance /futures/data/openInterestHist 支持的 period:
    5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d

    返回: DataFrame with columns:
        - open_interest: 持仓量 (合约张数, sumOpenInterest)
        - open_interest_value: 持仓价值 (USDT, sumOpenInterestValue)
    """
    # 映射 interval -> OI API 支持的 period
    OI_PERIOD_MAP = {
        '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '2h': '2h', '4h': '4h',
        '6h': '6h', '12h': '12h', '1d': '1d',
        '10m': '15m',  # 回退到 15m
    }
    period = OI_PERIOD_MAP.get(interval, '15m')

    # ── 先尝试本地缓存 ──
    if not force_api:
        cache_path = os.path.join(_OI_DIR, symbol, f'{period}.parquet')
        if os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                if df is not None and len(df) >= 50:
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    end_dt = pd.Timestamp.now()
                    start_dt = end_dt - pd.Timedelta(days=days)
                    if allow_api_fallback:
                        # 实盘/在线模式下，要求缓存不超过24小时，过期则尝试API刷新
                        if df.index[-1] >= (end_dt - pd.Timedelta(hours=24)):
                            df = df[df.index >= start_dt]
                            df = df[~df.index.duplicated(keep='first')].sort_index()
                            print(f"  [本地] {symbol} OI {period} 加载 {len(df)} 条 "
                                  f"({df.index[0].strftime('%Y-%m-%d %H:%M')} ~ "
                                  f"{df.index[-1].strftime('%Y-%m-%d %H:%M')})")
                            return df
                    else:
                        # 回测离线模式：不要求“新鲜度”，只要本地有历史即可
                        df = df[df.index >= start_dt]
                        df = df[~df.index.duplicated(keep='first')].sort_index()
                        print(f"  [本地] {symbol} OI {period}(离线) 加载 {len(df)} 条 "
                              f"({df.index[0].strftime('%Y-%m-%d %H:%M')} ~ "
                              f"{df.index[-1].strftime('%Y-%m-%d %H:%M')})")
                        return df
            except Exception as e:
                print(f"  OI 缓存读取失败: {e}")

    if not allow_api_fallback and not force_api:
        print(f"  [本地] {symbol} OI {period} 缓存缺失/无效, 已禁用 API 回退")
        return pd.DataFrame()

    # ── API 获取 ──
    # openInterestHist: limit=500, 按 period 粒度返回
    # Binance OI API startTime 上限约 30 天，需分段拉取
    CHUNK_DAYS = 29
    all_records = []
    final_end = int(datetime.now().timestamp() * 1000)
    request_start = datetime.now() - timedelta(days=days)

    print(f"正在从 Binance Futures API 获取 {symbol} Open Interest {period} "
          f"(最近{days}天, 分{(days // CHUNK_DAYS) + 1}段)...")

    chunk_idx = 0
    while request_start < datetime.now():
        chunk_end_dt = min(request_start + timedelta(days=CHUNK_DAYS), datetime.now())
        chunk_start_ms = int(request_start.timestamp() * 1000)
        chunk_end_ms = int(chunk_end_dt.timestamp() * 1000)
        current_start = chunk_start_ms

        while current_start < chunk_end_ms:
            url = (f"{BINANCE_FUTURES_OI_HIST_URL}?symbol={symbol}"
                   f"&period={period}"
                   f"&startTime={current_start}"
                   f"&endTime={chunk_end_ms}"
                   f"&limit=500")

            data = _api_get_json(url)
            if not data:
                break

            all_records.extend(data)
            current_start = data[-1]['timestamp'] + 1

            if len(data) < 500:
                break
            time.sleep(0.15)

        request_start = chunk_end_dt + timedelta(seconds=1)
        chunk_idx += 1
        if chunk_idx % 4 == 0:
            print(f"  ... 已拉取 {len(all_records)} 条 (第{chunk_idx}段)")
        time.sleep(0.2)

    if not all_records:
        print("  未获取到 Open Interest 数据!")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('date')

    df['open_interest'] = pd.to_numeric(df['sumOpenInterest'], errors='coerce')
    df['open_interest_value'] = pd.to_numeric(df['sumOpenInterestValue'], errors='coerce')

    df = df[['open_interest', 'open_interest_value']].copy()
    df = df[~df.index.duplicated(keep='first')].sort_index()

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # ── 保存缓存 ──
    cache_dir = os.path.join(_OI_DIR, symbol)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'{period}.parquet')
    df.to_parquet(cache_path)

    print(f"  OI 获取完成: {len(df)} 条 "
          f"({df.index[0].strftime('%Y-%m-%d %H:%M')} ~ "
          f"{df.index[-1].strftime('%Y-%m-%d %H:%M')})")
    oi_mean = df['open_interest_value'].mean()
    oi_max = df['open_interest_value'].max()
    print(f"  OI 价值: 均值=${oi_mean/1e6:.1f}M, 峰值=${oi_max/1e6:.1f}M")
    print(f"  已缓存至: {cache_path}")

    return df


def merge_perp_data_into_klines(kline_df: pd.DataFrame,
                                 mark_df: pd.DataFrame | None = None,
                                 funding_df: pd.DataFrame | None = None,
                                 oi_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    将 Mark Price K线、Funding Rate 和 Open Interest 数据合并到标准 K 线 DataFrame

    合并策略:
    - mark_high/mark_low/mark_close: 按时间戳对齐 (reindex + ffill)
    - funding_rate: 前向填充到每根 K 线 bar (结算前使用上一期费率, 防前视)
    - funding_interval_hours: 同上
    - open_interest/open_interest_value: 前向填充 (每根 bar 看到的是最近一次快照)

    返回: 合并后的 DataFrame (原始列 + mark_* + funding_* + open_interest*)
    """
    result = kline_df.copy()

    if mark_df is not None and len(mark_df) > 0:
        # 按时间对齐, 用 reindex + ffill (mark klines 与 klines 应该时间对齐)
        for col in ['mark_open', 'mark_high', 'mark_low', 'mark_close']:
            if col in mark_df.columns:
                aligned = mark_df[col].reindex(result.index, method='ffill')
                result[col] = aligned

    if funding_df is not None and len(funding_df) > 0:
        # Funding rate: 前向填充 (每根 bar 看到的是最近一次结算的费率)
        for col in ['funding_rate', 'funding_interval_hours']:
            if col in funding_df.columns:
                aligned = funding_df[col].reindex(result.index, method='ffill')
                result[col] = aligned

    if oi_df is not None and len(oi_df) > 0:
        # Open Interest: 前向填充到每根 bar
        for col in ['open_interest', 'open_interest_value']:
            if col in oi_df.columns:
                aligned = oi_df[col].reindex(result.index, method='ffill')
                result[col] = aligned

    # ── v10.1: 衍生品数据覆盖率审计 ──
    _audit_lines = []
    _audit_dict = {}
    _total_bars = len(result)
    for col_name, src_label in [
        ('funding_rate', 'Funding Rate'),
        ('open_interest', 'Open Interest'),
        ('open_interest_value', 'OI Value'),
        ('mark_close', 'Mark Price'),
    ]:
        if col_name not in result.columns:
            _audit_lines.append(f"  {src_label}: NOT PRESENT")
            continue
        _col = result[col_name]
        _non_null = int(_col.notna().sum())
        _coverage = _non_null / _total_bars if _total_bars > 0 else 0.0
        # 计算最长连续 forward-fill (staleness): 原始非NaN的位置之间的最大间隔
        _orig_mask = pd.Series(False, index=result.index)
        if col_name in ['open_interest', 'open_interest_value'] and oi_df is not None and len(oi_df) > 0 and col_name in oi_df.columns:
            _orig_mask.loc[_orig_mask.index.isin(oi_df.index)] = True
        elif col_name == 'funding_rate' and funding_df is not None and len(funding_df) > 0 and col_name in funding_df.columns:
            _orig_mask.loc[_orig_mask.index.isin(funding_df.index)] = True
        elif col_name == 'mark_close' and mark_df is not None and len(mark_df) > 0 and col_name in mark_df.columns:
            _orig_mask.loc[_orig_mask.index.isin(mark_df.index)] = True
        _orig_count = int(_orig_mask.sum())
        _orig_coverage = _orig_count / _total_bars if _total_bars > 0 else 0.0
        # 最长连续 stale 段 (原始数据点之间的最大间隔)
        _max_stale = 0
        _max_internal_stale = 0
        if _orig_count > 0:
            _orig_positions = _orig_mask[_orig_mask].index
            if len(_orig_positions) > 1:
                _orig_iloc = [result.index.get_loc(p) for p in _orig_positions]
                _gaps = [_orig_iloc[i+1] - _orig_iloc[i] for i in range(len(_orig_iloc)-1)]
                _max_internal_stale = max(_gaps) if _gaps else 0
                _max_stale = _max_internal_stale
            # 也检查第一个原始点之前和最后一个之后的间隔
            _first_iloc = result.index.get_loc(_orig_positions[0])
            _last_iloc = result.index.get_loc(_orig_positions[-1])
            _max_stale = max(_max_stale, _first_iloc, _total_bars - 1 - _last_iloc)
        else:
            _max_stale = _total_bars
            _max_internal_stale = _total_bars
        _audit_lines.append(
            f"  {src_label}: coverage={_coverage:.1%} "
            f"(orig_points={_orig_count}/{_total_bars}, orig_coverage={_orig_coverage:.1%}, "
            f"max_stale_bars={_max_stale}, max_internal_stale_bars={_max_internal_stale})"
        )
        _audit_dict[col_name] = {
            'coverage': _coverage,
            'orig_count': _orig_count,
            'total_bars': _total_bars,
            'orig_coverage': _orig_coverage,
            'max_stale_bars': int(_max_stale),
            'max_internal_stale_bars': int(_max_internal_stale),
        }
    if _audit_lines:
        import logging
        _log = logging.getLogger('merge_perp_audit')
        _log.info("=== Perp Data Coverage Audit (%d bars) ===\n%s",
                  _total_bars, '\n'.join(_audit_lines))
        # 存储审计结果到 DataFrame attrs (供回测脚本读取)
        result.attrs['perp_data_audit'] = '\n'.join(_audit_lines)
        result.attrs['perp_data_audit_dict'] = _audit_dict

    return result


if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'kline'

    if mode == 'kline':
        df = fetch_binance_klines("ETHUSDT", interval="10m", days=30)
        print(f"\n数据预览:\n{df.tail(10)}")
        print(f"\n数据统计:\n{df.describe()}")

    elif mode == 'mark':
        df = fetch_mark_price_klines("ETHUSDT", interval="15m", days=90)
        print(f"\n数据预览:\n{df.tail(10)}")

    elif mode == 'funding':
        df = fetch_funding_rate_history("ETHUSDT", days=365)
        print(f"\n数据预览:\n{df.tail(20)}")

    elif mode == 'oi':
        df = fetch_open_interest_history("ETHUSDT", interval="15m", days=90)
        print(f"\n数据预览:\n{df.tail(20)}")

    elif mode == 'all':
        # 获取所有数据并合并
        symbol = "ETHUSDT"
        days = 365
        print(f"\n{'='*80}")
        print(f"  获取 {symbol} 完整衍生品数据 (最近 {days} 天)")
        print(f"{'='*80}")

        kline_df = fetch_binance_klines(symbol, interval="15m", days=days)
        mark_df = fetch_mark_price_klines(symbol, interval="15m", days=days)
        funding_df = fetch_funding_rate_history(symbol, days=days)
        oi_df = fetch_open_interest_history(symbol, interval="15m", days=days)

        if len(kline_df) > 0:
            merged = merge_perp_data_into_klines(kline_df, mark_df, funding_df, oi_df)
            print(f"\n合并后 DataFrame: {len(merged)} 行, 列: {list(merged.columns)}")
            print(f"\n数据预览:\n{merged.tail(10)}")
            # 检查覆盖率
            for col in ['mark_high', 'mark_low', 'funding_rate', 'funding_interval_hours',
                         'open_interest', 'open_interest_value', 'taker_buy_quote']:
                if col in merged.columns:
                    coverage = merged[col].notna().mean() * 100
                    print(f"  {col} 覆盖率: {coverage:.1f}%")
    else:
        print(f"用法: python3 binance_fetcher.py [kline|mark|funding|oi|all]")


# ── 热点币系统用 API (hotcoin/) ──────────────────────────────────

def fetch_all_tickers_24h() -> list:
    """
    获取全市场 24h 行情 (REST 方式, 用于初始化/补充 WebSocket 数据)。
    GET /api/v3/ticker/24hr — 无需认证, weight=40。
    返回: list of dict, 每个含 symbol, priceChangePercent, quoteVolume 等。
    """
    url = "https://api.binance.com/api/v3/ticker/24hr"
    try:
        resp = _api_get_json(url, timeout=15)
        if isinstance(resp, list):
            return [t for t in resp if t.get("symbol", "").endswith("USDT")]
        return []
    except Exception as e:
        print(f"[binance_fetcher] fetch_all_tickers_24h 失败: {e}")
        return []


def fetch_exchange_info(quote_asset: str = "USDT") -> dict:
    """
    获取交易所信息 (交易对精度/过滤器)。
    GET /api/v3/exchangeInfo — 无需认证。
    返回: dict, key=symbol, value=symbol_info (含 filters)。
    """
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        resp = _api_get_json(url, timeout=15)
        result = {}
        for sym_info in resp.get("symbols", []):
            if sym_info.get("quoteAsset") == quote_asset and sym_info.get("status") == "TRADING":
                result[sym_info["symbol"]] = sym_info
        return result
    except Exception as e:
        print(f"[binance_fetcher] fetch_exchange_info 失败: {e}")
        return {}
