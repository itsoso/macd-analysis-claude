"""
X/Twitter KOL 监控 — API v2 Filtered Stream

通过 Twitter API v2 的 Filtered Stream 订阅 50-100 个加密 KOL 的推文,
实时提取币种提及和情绪信号。

需要 $200/月 Basic 套餐 (Phase 2)。
环境变量: TWITTER_BEARER_TOKEN
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Set

import requests

log = logging.getLogger("hotcoin.twitter")

BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN", "")
API_BASE = "https://api.twitter.com/2"

# 加密货币 KOL 账号 (可配置)
DEFAULT_KOLS = [
    "caborundum",      # 典型示例, 实际使用时替换
    "CryptoCapo_",
    "crypto_birb",
    "AltcoinGordon",
    "CryptoBusy",
]

_TICKER_RE = re.compile(r"\$([A-Z]{2,10})\b")
_CASHTAG_RE = re.compile(r"#([A-Z]{2,10})(?:USDT)?\b")
_COMMON_WORDS = {"THE", "FOR", "AND", "NOT", "BUT", "ALL", "NEW", "TOP", "USD", "ETH", "BTC"}


class TwitterMonitor:
    """X/Twitter KOL 推文监控器。"""

    def __init__(self, pool, bearer_token: str = ""):
        self.pool = pool
        self.token = bearer_token or BEARER_TOKEN
        self._headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self._mention_counts: Dict[str, List[float]] = {}  # symbol → [timestamps]
        self._kol_set: Set[str] = set(DEFAULT_KOLS)
        self._sentiment_cache: Dict[str, float] = {}

        try:
            from hotcoin.discovery.sentiment_scorer import SentimentScorer
            self._scorer = SentimentScorer()
        except Exception:
            self._scorer = None

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    async def run(self, shutdown: asyncio.Event):
        if not self.enabled:
            log.warning("Twitter 监控未启用 (TWITTER_BEARER_TOKEN 未设置)")
            return

        log.info("Twitter KOL 监控启动 (%d 个 KOL)", len(self._kol_set))
        await self._setup_rules()

        while not shutdown.is_set():
            try:
                await asyncio.to_thread(self._stream_tweets)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Twitter stream 异常, 30s 后重连")
                await asyncio.sleep(30)

    def _stream_tweets(self):
        """连接 Filtered Stream, 解析推文。"""
        url = f"{API_BASE}/tweets/search/stream"
        params = {"tweet.fields": "author_id,created_at,entities", "expansions": "author_id"}
        try:
            resp = requests.get(url, headers=self._headers, params=params, stream=True, timeout=90)
            if resp.status_code == 429:
                log.warning("Twitter rate limited, 等待 60s")
                time.sleep(60)
                return
            if resp.status_code != 200:
                log.warning("Twitter stream 返回 %d: %s", resp.status_code, resp.text[:200])
                return

            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    self._process_tweet(data)
                except json.JSONDecodeError:
                    continue
        except requests.RequestException as e:
            log.debug("Twitter stream 断开: %s", e)

    def _process_tweet(self, data: dict):
        """从推文中提取币种提及。"""
        tweet = data.get("data", {})
        text = tweet.get("text", "")
        author_id = tweet.get("author_id", "")
        created_at = tweet.get("created_at", "")

        # 提取 $TICKER 和 #TICKER
        tickers = set()
        for m in _TICKER_RE.findall(text):
            if m not in _COMMON_WORDS and len(m) >= 2:
                tickers.add(m)
        for m in _CASHTAG_RE.findall(text):
            if m not in _COMMON_WORDS and len(m) >= 2:
                tickers.add(m)

        now = time.time()
        sentiment = self._scorer.score(text) if self._scorer else 0.0

        for ticker in tickers:
            symbol = f"{ticker}USDT"
            log.info("Twitter 提及: %s (author=%s, sentiment=%.2f)", symbol, author_id, sentiment)

            mentions = self._mention_counts.setdefault(symbol, [])
            mentions.append(now)
            cutoff = now - 3600
            self._mention_counts[symbol] = [t for t in mentions if t >= cutoff]

            if sentiment != 0.0:
                self._sentiment_cache[symbol] = sentiment

            self.pool.on_social_mention(
                symbol=symbol,
                source="twitter",
                kol_id=author_id,
                mention_count_1h=len(self._mention_counts[symbol]),
                sentiment=self._sentiment_cache.get(symbol, 0.0),
            )

    async def _setup_rules(self):
        """设置 Filtered Stream 规则 (关键词过滤)。"""
        url = f"{API_BASE}/tweets/search/stream/rules"
        try:
            existing = requests.get(url, headers=self._headers, timeout=10).json()
            ids = [r["id"] for r in existing.get("data", [])]
            if ids:
                requests.post(url, headers=self._headers, json={"delete": {"ids": ids}}, timeout=10)

            rules = [
                {"value": "crypto OR #altcoin OR $BTC OR $ETH -is:retweet lang:en", "tag": "crypto_general"},
                {"value": "binance listing OR launchpool OR new coin -is:retweet", "tag": "binance_listing"},
            ]
            requests.post(url, headers=self._headers, json={"add": rules}, timeout=10)
            log.info("Twitter stream 规则已设置 (%d 条)", len(rules))
        except Exception as e:
            log.warning("设置 Twitter 规则失败: %s", e)

    def cleanup_stale(self):
        """清理不活跃 symbol 的过期提及记录。"""
        now = time.time()
        cutoff = now - 3600
        stale = [s for s, ts in self._mention_counts.items() if not ts or ts[-1] < cutoff]
        for s in stale:
            del self._mention_counts[s]
        stale_s = [s for s, _ in self._sentiment_cache.items() if s not in self._mention_counts]
        for s in stale_s:
            del self._sentiment_cache[s]

    def get_mention_velocity(self, symbol: str) -> float:
        """返回最近 1h 提及速率 (次/小时)。"""
        mentions = self._mention_counts.get(symbol, [])
        now = time.time()
        recent = [t for t in mentions if now - t < 3600]
        return len(recent)
