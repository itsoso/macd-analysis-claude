"""
币安广场 (Binance Square) 帖子监控

无官方 API, 通过逆向 Web 接口轮询热帖。
频率: 5min / 次 (保守, 避免封禁)。

提取:
  - 热门帖子中的币种提及
  - 官方账号动态
  - 带 $ 标签的币种
"""

import asyncio
import json
import logging
import re
import time
from typing import Dict, List, Set

import requests

log = logging.getLogger("hotcoin.square")

_SQUARE_FEED_URL = "https://www.binance.com/bapi/composite/v1/public/content/getContentListByTab"
_SQUARE_TRENDING_URL = "https://www.binance.com/bapi/composite/v1/public/content/getTrendingContent"

_TICKER_RE = re.compile(r"\$([A-Z]{2,10})\b")
_COMMON_WORDS = {"THE", "FOR", "AND", "NOT", "BUT", "ALL", "NEW", "TOP", "USD"}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


class BinanceSquareMonitor:
    """币安广场帖子监控器。"""

    def __init__(self, pool):
        self.pool = pool
        self._seen_content_ids: Set[str] = set()
        self._mention_counts: Dict[str, List[float]] = {}

    async def run(self, shutdown: asyncio.Event, interval_sec: int = 300):
        log.info("币安广场监控启动 (间隔 %ds)", interval_sec)
        while not shutdown.is_set():
            try:
                await asyncio.to_thread(self._poll_feed)
                await asyncio.to_thread(self._poll_trending)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("广场监控异常")
            await asyncio.sleep(interval_sec)

    def _poll_feed(self):
        """轮询广场信息流。"""
        try:
            resp = requests.post(
                _SQUARE_FEED_URL,
                headers=_HEADERS,
                json={"pageNo": 1, "pageSize": 20, "tab": "hot"},
                timeout=15,
            )
            if resp.status_code != 200:
                log.debug("广场 feed 返回 %d", resp.status_code)
                return

            data = resp.json()
            contents = data.get("data", {}).get("contents", [])
            self._process_contents(contents)
        except requests.RequestException as e:
            log.debug("广场 feed 请求失败: %s", e)

    def _poll_trending(self):
        """轮询广场趋势内容。"""
        try:
            resp = requests.post(
                _SQUARE_TRENDING_URL,
                headers=_HEADERS,
                json={"pageNo": 1, "pageSize": 10},
                timeout=15,
            )
            if resp.status_code != 200:
                return

            data = resp.json()
            contents = data.get("data", {}).get("contents", [])
            self._process_contents(contents)
        except requests.RequestException:
            pass

    def _process_contents(self, contents: list):
        """解析帖子, 提取币种提及。"""
        now = time.time()
        for item in contents:
            content_id = str(item.get("id", item.get("contentId", "")))
            if content_id in self._seen_content_ids:
                continue
            self._seen_content_ids.add(content_id)

            title = item.get("title", "")
            body = item.get("body", item.get("content", ""))
            text = f"{title} {body}"

            tickers = set()
            for m in _TICKER_RE.findall(text):
                if m not in _COMMON_WORDS and len(m) >= 2:
                    tickers.add(m)

            # 也检查标签
            tags = item.get("tags", [])
            if isinstance(tags, list):
                for tag in tags:
                    t = str(tag).upper()
                    if 2 <= len(t) <= 10 and t not in _COMMON_WORDS:
                        tickers.add(t)

            for ticker in tickers:
                symbol = f"{ticker}USDT"
                mentions = self._mention_counts.setdefault(symbol, [])
                mentions.append(now)
                cutoff = now - 3600
                self._mention_counts[symbol] = [t for t in mentions if t >= cutoff]

                log.info("广场提及: %s (content=%s)", symbol, content_id)
                self.pool.on_social_mention(
                    symbol=symbol,
                    source="binance_square",
                    kol_id="",
                    mention_count_1h=len(self._mention_counts[symbol]),
                )

        # 限制 seen 集合大小
        if len(self._seen_content_ids) > 5000:
            self._seen_content_ids = set(list(self._seen_content_ids)[-2000:])

    def get_mention_velocity(self, symbol: str) -> float:
        mentions = self._mention_counts.get(symbol, [])
        now = time.time()
        return len([t for t in mentions if now - t < 3600])
