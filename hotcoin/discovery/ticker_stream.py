"""
全市场行情流 — WebSocket !miniTicker@arr

单连接订阅全部 USDT 交易对 miniTicker，每秒推送一次。
收到数据后交给 AnomalyDetector 做异动检测，异动币种送入候选池。
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

try:
    import websockets
    import websockets.client
except ImportError:
    websockets = None  # type: ignore

import requests

from hotcoin.config import DiscoveryConfig, BASE_WS_URL, BASE_REST_URL

log = logging.getLogger("hotcoin.ticker")


@dataclass
class MiniTicker:
    """!miniTicker@arr 推送的单条数据。"""
    symbol: str
    close: float
    open_price: float
    high: float
    low: float
    base_volume: float
    quote_volume: float            # 24h USDT 成交额
    event_time: int                # ms timestamp

    # 运行时由 TickerStream 维护
    price_change_24h: float = 0.0  # (close - open) / open
    volume_1m: float = 0.0         # 最近 1min 累计成交额 (滑动窗口)
    avg_volume_20m: float = 0.0    # 20min 滑动均值
    price_5m_ago: float = 0.0
    price_change_5m: float = 0.0


class TickerStream:
    """WebSocket 全市场行情流 + 异动检测。"""

    RECONNECT_DELAY = 5
    MAX_RECONNECT_DELAY = 60

    def __init__(self, config: DiscoveryConfig, anomaly_detector, pool):
        self.config = config
        self.detector = anomaly_detector
        self.pool = pool
        self._tickers: Dict[str, MiniTicker] = {}

        # 1min 滑动窗口: symbol → deque of (ts_sec, quote_volume_delta)
        self._vol_window: Dict[str, deque] = {}
        # 5min 价格快照: symbol → deque of (ts_sec, price)
        self._price_snapshots: Dict[str, deque] = {}
        self._last_cleanup = 0.0

    @property
    def tickers(self) -> Dict[str, MiniTicker]:
        return self._tickers

    async def run(self, shutdown: asyncio.Event):
        if websockets is None:
            log.error("websockets 库未安装, pip install websockets")
            return

        url = f"{BASE_WS_URL}/ws/!miniTicker@arr"
        delay = self.RECONNECT_DELAY

        while not shutdown.is_set():
            try:
                log.info("连接全市场 miniTicker: %s", url)
                async with websockets.client.connect(url, ping_interval=20, ping_timeout=60) as ws:
                    delay = self.RECONNECT_DELAY
                    log.info("miniTicker 连接成功")
                    async for raw in ws:
                        if shutdown.is_set():
                            break
                        self._handle_message(raw)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("miniTicker 连接异常, %ds 后重连", delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.MAX_RECONNECT_DELAY)

    def _handle_message(self, raw: str):
        try:
            arr = json.loads(raw)
        except json.JSONDecodeError:
            return

        now = time.time()
        for item in arr:
            sym = item.get("s", "")
            if not sym.endswith("USDT"):
                continue

            close = float(item.get("c", 0))
            open_price = float(item.get("o", 0))
            quote_vol = float(item.get("q", 0))
            if close <= 0 or open_price <= 0:
                continue

            ticker = MiniTicker(
                symbol=sym,
                close=close,
                open_price=open_price,
                high=float(item.get("h", 0)),
                low=float(item.get("l", 0)),
                base_volume=float(item.get("v", 0)),
                quote_volume=quote_vol,
                event_time=int(item.get("E", 0)),
                price_change_24h=(close - open_price) / open_price if open_price else 0,
            )

            prev = self._tickers.get(sym)
            self._update_volume_window(sym, now, quote_vol, prev)
            self._update_price_snapshots(sym, now, close)

            ticker.volume_1m = self._get_volume_1m(sym)
            ticker.avg_volume_20m = self._get_avg_volume_20m(sym)
            snap = self._price_snapshots.get(sym, [])
            if snap:
                ticker.price_5m_ago = snap[0][1]
                ticker.price_change_5m = (close - snap[0][1]) / snap[0][1] if snap[0][1] else 0

            self._tickers[sym] = ticker

            anomaly = self.detector.detect(sym, ticker)
            if anomaly:
                self.pool.on_anomaly(anomaly)

        # 每 10 分钟清理不活跃币种的数据
        if now - self._last_cleanup > 600:
            self._cleanup_stale(now)
            self._last_cleanup = now

    def _update_volume_window(self, sym: str, now: float, quote_vol: float,
                              prev: Optional[MiniTicker]):
        window = self._vol_window.get(sym)
        if window is None:
            window = deque()
            self._vol_window[sym] = window
        delta = 0.0
        if prev and quote_vol >= prev.quote_volume:
            delta = quote_vol - prev.quote_volume
        elif prev:
            delta = quote_vol
        window.append((now, delta))
        cutoff = now - 1200
        while window and window[0][0] < cutoff:
            window.popleft()

    def _update_price_snapshots(self, sym: str, now: float, price: float):
        snaps = self._price_snapshots.get(sym)
        if snaps is None:
            snaps = deque()
            self._price_snapshots[sym] = snaps
        snaps.append((now, price))
        cutoff = now - 300
        while snaps and snaps[0][0] < cutoff:
            snaps.popleft()

    def _cleanup_stale(self, now: float):
        """清理 30 分钟无更新的币种数据, 防止内存泄漏。"""
        stale_cutoff = now - 1800
        stale = [s for s, t in self._tickers.items() if t.event_time / 1000 < stale_cutoff]
        for sym in stale:
            self._tickers.pop(sym, None)
            self._vol_window.pop(sym, None)
            self._price_snapshots.pop(sym, None)
        if stale:
            log.debug("清理 %d 个不活跃币种数据", len(stale))

    def _get_volume_1m(self, sym: str) -> float:
        window = self._vol_window.get(sym, [])
        if not window:
            return 0.0
        cutoff = window[-1][0] - 60
        return sum(v for t, v in window if t >= cutoff)

    def _get_avg_volume_20m(self, sym: str) -> float:
        window = self._vol_window.get(sym, [])
        if not window:
            return 0.0
        total = sum(v for _, v in window)
        span = max(window[-1][0] - window[0][0], 60)
        return total / (span / 60) if span > 0 else 0.0
