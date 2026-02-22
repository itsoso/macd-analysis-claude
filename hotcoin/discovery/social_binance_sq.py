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
from collections import deque
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
        self._sentiment_cache: Dict[str, float] = {}
        self._recent_posts: deque = deque(maxlen=100)
        self._started_at = 0.0
        self._running = False
        self._last_poll_ts = 0.0
        self._last_success_ts = 0.0
        self._last_error = ""
        self._error_count = 0

        try:
            from hotcoin.discovery.sentiment_scorer import SentimentScorer
            self._scorer = SentimentScorer()
        except Exception:
            self._scorer = None

    async def run(self, shutdown: asyncio.Event, interval_sec: int = 300):
        log.info("币安广场监控启动 (间隔 %ds)", interval_sec)
        self._started_at = self._started_at or time.time()
        self._running = True
        while not shutdown.is_set():
            try:
                self._last_poll_ts = time.time()
                feed_ok, _ = await asyncio.to_thread(self._poll_feed)
                trend_ok, _ = await asyncio.to_thread(self._poll_trending)
                if feed_ok or trend_ok:
                    self._last_success_ts = time.time()
                    self._last_error = ""
            except asyncio.CancelledError:
                break
            except Exception:
                self._error_count += 1
                self._last_error = "poll_exception"
                log.exception("广场监控异常")
            await asyncio.sleep(interval_sec)
        self._running = False

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
                self._error_count += 1
                self._last_error = f"feed_http_{resp.status_code}"
                return False, 0

            data = resp.json()
            contents = data.get("data", {}).get("contents", [])
            new_posts = self._process_contents(contents)
            return True, new_posts
        except requests.RequestException as e:
            log.debug("广场 feed 请求失败: %s", e)
            self._error_count += 1
            self._last_error = f"feed_req_error:{type(e).__name__}"
            return False, 0

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
                self._error_count += 1
                self._last_error = f"trend_http_{resp.status_code}"
                return False, 0

            data = resp.json()
            contents = data.get("data", {}).get("contents", [])
            new_posts = self._process_contents(contents)
            return True, new_posts
        except requests.RequestException:
            self._error_count += 1
            self._last_error = "trend_req_error"
            return False, 0

    def _process_contents(self, contents: list):
        """解析帖子, 提取币种提及。"""
        now = time.time()
        new_posts = 0
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

            sentiment = self._scorer.score(text) if self._scorer else 0.0

            if tickers:
                new_posts += 1
                self._recent_posts.append({
                    "id": content_id,
                    "title": title[:120],
                    "body": (body or "")[:200],
                    "tickers": sorted(tickers),
                    "sentiment": round(sentiment, 2),
                    "source": "binance_square",
                    "ts": now,
                })

            for ticker in tickers:
                symbol = f"{ticker}USDT"
                mentions = self._mention_counts.setdefault(symbol, [])
                mentions.append(now)
                cutoff = now - 3600
                self._mention_counts[symbol] = [t for t in mentions if t >= cutoff]

                if sentiment != 0.0:
                    self._sentiment_cache[symbol] = sentiment

                log.info("广场提及: %s (content=%s, sentiment=%.2f)", symbol, content_id, sentiment)
                self.pool.on_social_mention(
                    symbol=symbol,
                    source="binance_square",
                    kol_id="",
                    mention_count_1h=len(self._mention_counts[symbol]),
                    sentiment=self._sentiment_cache.get(symbol, 0.0),
                )

        if len(self._seen_content_ids) > 5000:
            self._seen_content_ids.clear()
        return new_posts

    def cleanup_stale(self):
        """清理不活跃 symbol 的过期提及记录。"""
        now = time.time()
        cutoff = now - 3600
        stale = [s for s, ts in self._mention_counts.items() if not ts or ts[-1] < cutoff]
        for s in stale:
            del self._mention_counts[s]
        stale_s = [s for s in self._sentiment_cache if s not in self._mention_counts]
        for s in stale_s:
            del self._sentiment_cache[s]

    def get_recent_posts(self, limit: int = 50) -> list:
        """返回最近的帖子列表 (新→旧)。"""
        posts = list(self._recent_posts)
        posts.reverse()
        return posts[:limit]

    def get_mention_velocity(self, symbol: str) -> float:
        mentions = self._mention_counts.get(symbol, [])
        now = time.time()
        return len([t for t in mentions if now - t < 3600])

    def status(self) -> dict:
        return {
            "enabled": True,
            "running": bool(self._running),
            "started_at": self._started_at,
            "last_poll_ts": self._last_poll_ts,
            "last_success_ts": self._last_success_ts,
            "last_error": self._last_error,
            "error_count": int(self._error_count),
            "recent_posts": len(self._recent_posts),
        }
