"""
新币上线监控

两个数据源:
  1. GET /sapi/v1/spot/open-symbol-list  — 即将上线的交易对及时间
  2. 币安公告 CMS API 轮询 — 检测 "Will List" 类公告, 正则提取币种
"""

import asyncio
import logging
import re
import time
from typing import List, Optional, Set

import os

import requests

from hotcoin.config import DiscoveryConfig, BASE_REST_URL
from hotcoin.discovery.anomaly_detector import AnomalySignal

_API_KEY = os.environ.get("BINANCE_API_KEY", "")

log = logging.getLogger("hotcoin.listing")

_ANNOUNCEMENT_URL = (
    "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
)
_WILL_LIST_RE = re.compile(
    r"(?:will list|will add|新增|上线)\s+([A-Z][A-Z0-9]{1,9})",
    re.IGNORECASE,
)
_TICKER_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9})(?:USDT)?\b")
_STOP_WORDS = frozenset({
    "THE", "FOR", "AND", "NEW", "WILL", "LIST", "ADD", "ALL", "HAS",
    "NOT", "ARE", "ITS", "WITH", "SPOT", "TRADE", "BINANCE", "USDT",
    "MARGIN", "FUTURES", "TOKEN", "COIN", "PAIR", "MARKET", "OPEN",
    "UPDATE", "SEED", "ZONE", "ALPHA", "TRADING", "PRICE",
})


class ListingMonitor:
    """新币上线监控 — 异步后台任务。"""

    _BACKOFF_BASE = 60
    _BACKOFF_MAX = 480

    def __init__(self, config: DiscoveryConfig, pool):
        self.config = config
        self.pool = pool
        self._seen_open_symbols: Set[str] = set()
        self._seen_article_ids: Set[int] = set()
        self._last_seen_cleanup = 0.0
        self._listing_backoff = 0
        self._announce_backoff = 0

    async def run(self, shutdown: asyncio.Event):
        log.info("新币上线监控启动 (listing=%ds, announcement=%ds)",
                 self.config.listing_poll_sec, self.config.announcement_poll_sec)

        last_listing = 0.0
        last_announce = 0.0

        while not shutdown.is_set():
            now = time.time()
            listing_interval = self.config.listing_poll_sec + self._listing_backoff
            announce_interval = self.config.announcement_poll_sec + self._announce_backoff
            try:
                if now - last_listing >= listing_interval:
                    ok = await asyncio.to_thread(self._check_open_symbol_list)
                    last_listing = now
                    self._listing_backoff = 0 if ok else min(
                        self._listing_backoff * 2 or self._BACKOFF_BASE,
                        self._BACKOFF_MAX,
                    )

                if now - last_announce >= announce_interval:
                    ok = await asyncio.to_thread(self._check_announcements)
                    last_announce = now
                    self._announce_backoff = 0 if ok else min(
                        self._announce_backoff * 2 or self._BACKOFF_BASE,
                        self._BACKOFF_MAX,
                    )
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("上线监控异常")

            # 每 24h 清理 _seen 集合, 防止无限增长
            if now - self._last_seen_cleanup > 86400:
                if len(self._seen_article_ids) > 500:
                    self._seen_article_ids.clear()
                if len(self._seen_open_symbols) > 200:
                    self._seen_open_symbols.clear()
                self._last_seen_cleanup = now

            await asyncio.sleep(10)

    def _check_open_symbol_list(self) -> bool:
        """调用 /sapi/v1/spot/open-symbol-list 发现即将上线的交易对 (需 API Key)。
        返回 True 表示成功, False 表示需退避。"""
        if not _API_KEY:
            return True
        try:
            headers = {"X-MBX-APIKEY": _API_KEY}
            resp = requests.get(
                f"{BASE_REST_URL}/sapi/v1/spot/open-symbol-list",
                headers=headers, timeout=10,
            )
            if resp.status_code == 429 or resp.status_code == 418:
                log.warning("open-symbol-list 触发限频 (%d), 启动退避", resp.status_code)
                return False
            if resp.status_code != 200:
                log.debug("open-symbol-list 返回 %d", resp.status_code)
                return True

            data = resp.json()
            if not isinstance(data, list):
                return True

            for entry in data:
                symbols = entry.get("symbols", [])
                open_time = entry.get("openTime", 0)
                for sym in symbols:
                    if sym in self._seen_open_symbols:
                        continue
                    if not sym.endswith("USDT"):
                        continue
                    self._seen_open_symbols.add(sym)
                    log.info("发现即将上线: %s (openTime=%d)", sym, open_time)
                    self.pool.on_listing(sym, open_time)
            return True
        except requests.RequestException as e:
            log.warning("open-symbol-list 请求失败: %s", e)
            return False

    def _check_announcements(self) -> bool:
        """轮询币安公告, 检测 'Will List' 类新币上线公告。
        返回 True 表示成功, False 表示需退避。"""
        try:
            resp = requests.get(
                _ANNOUNCEMENT_URL,
                params={
                    "type": 1,
                    "catalogId": 48,  # New Cryptocurrency Listing
                    "pageNo": 1,
                    "pageSize": 10,
                },
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if resp.status_code == 429 or resp.status_code == 418:
                log.warning("公告 API 触发限频 (%d), 启动退避", resp.status_code)
                return False
            if resp.status_code != 200:
                return True

            data = resp.json()
            articles = data.get("data", {}).get("catalogs", [{}])
            if isinstance(articles, list) and articles:
                articles = articles[0].get("articles", [])

            for art in articles:
                art_id = art.get("id", 0)
                if art_id in self._seen_article_ids:
                    continue
                self._seen_article_ids.add(art_id)

                title = art.get("title", "")
                code = art.get("code", "")
                self._parse_listing_announcement(title, code)
            return True

        except requests.RequestException as e:
            log.debug("公告查询失败: %s", e)
            return False
        except Exception:
            log.exception("公告解析异常")
            return True

    def _parse_listing_announcement(self, title: str, code: str):
        """从公告标题中提取新币符号。"""
        m = _WILL_LIST_RE.search(title)
        if m:
            token = m.group(1).upper()
            sym = f"{token}USDT"
            log.info("公告发现新币: %s (title=%s)", sym, title[:80])
            self.pool.on_listing(sym, open_time=0)
            return

        if any(kw in title.lower() for kw in ("will list", "new listing", "上线")):
            tickers = _TICKER_RE.findall(title)
            for t in tickers:
                if len(t) >= 2 and t not in _STOP_WORDS:
                    sym = f"{t}USDT"
                    log.info("公告提取币种: %s (title=%s)", sym, title[:80])
                    self.pool.on_listing(sym, open_time=0)
