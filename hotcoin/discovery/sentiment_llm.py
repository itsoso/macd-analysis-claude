"""
LLM 情感分析器

用 Claude API 分析币种相关文本的情感倾向, 填充 hot_ranker Phase 2 维度。

输出:
    {
        "sentiment": float,       # -1.0 (极度看空) ~ +1.0 (极度看多)
        "confidence": float,      # 0-1
        "keywords": ["pump", "moon"],
        "summary": "..."
    }

用法:
    from hotcoin.discovery.sentiment_llm import analyze_sentiment
    result = analyze_sentiment("BTC breaking ATH, massive volume incoming!")
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional

log = logging.getLogger("hotcoin.sentiment")

_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
_CACHE: Dict[str, dict] = {}
_CACHE_TTL = 300  # 5 分钟缓存
_CACHE_TS: Dict[str, float] = {}
_TIMEOUT = 3.0

_SYSTEM_PROMPT = """You are a cryptocurrency market sentiment analyzer.
Analyze the given text and return ONLY valid JSON:
- sentiment: float from -1.0 (extremely bearish) to +1.0 (extremely bullish)
- confidence: float 0-1 (how confident you are in the sentiment assessment)
- keywords: array of up to 5 key sentiment-driving words/phrases
- summary: one-sentence summary of the sentiment"""

# 关键词规则降级
_BULLISH_WORDS = frozenset({
    "moon", "pump", "bullish", "breakout", "ath", "surge", "rally",
    "buy", "long", "rocket", "gem", "100x", "listing", "launch",
    "看多", "拉盘", "暴涨", "突破", "新高", "上线",
})
_BEARISH_WORDS = frozenset({
    "dump", "crash", "bearish", "scam", "rug", "sell", "short",
    "dead", "delist", "hack", "exploit", "warning",
    "看空", "砸盘", "暴跌", "下线", "骗局", "跑路",
})


def analyze_sentiment(
    text: str,
    symbol: str = "",
    cache_key: Optional[str] = None,
) -> dict:
    """
    分析文本情感。LLM 优先, 关键词降级。

    Returns
    -------
    dict: {sentiment, confidence, keywords, summary}
    """
    key = cache_key or text[:100]
    now = time.time()

    # 缓存命中 (TTL 内)
    if key in _CACHE and now - _CACHE_TS.get(key, 0) < _CACHE_TTL:
        return _CACHE[key]

    result = None

    # 尝试 LLM
    if _ANTHROPIC_KEY:
        try:
            result = _analyze_with_llm(text, symbol)
        except Exception as e:
            log.debug("LLM 情感分析降级: %s", e)

    # 降级关键词
    if result is None:
        result = _analyze_with_keywords(text)

    _CACHE[key] = result
    _CACHE_TS[key] = now

    # 缓存清理 (超过 200 条时清理最旧的)
    if len(_CACHE) > 200:
        oldest = sorted(_CACHE_TS, key=_CACHE_TS.get)[:50]
        for k in oldest:
            _CACHE.pop(k, None)
            _CACHE_TS.pop(k, None)

    return result


def analyze_batch(texts: List[str], symbol: str = "") -> dict:
    """
    批量分析多条文本, 返回聚合情感。

    适用于: 多条社交媒体帖子 → 综合情感评分。
    """
    if not texts:
        return {"sentiment": 0.0, "confidence": 0.0, "keywords": [], "summary": "no data"}

    sentiments = []
    all_keywords = []
    for text in texts[:10]:  # 最多分析 10 条
        r = analyze_sentiment(text, symbol)
        sentiments.append(r["sentiment"] * r["confidence"])
        all_keywords.extend(r.get("keywords", []))

    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
    # 聚合置信度: 样本越多越高, 上限 0.9
    confidence = min(0.9, 0.3 + len(sentiments) * 0.06)

    # 去重关键词, 保留频率最高的 5 个
    from collections import Counter
    top_kw = [w for w, _ in Counter(all_keywords).most_common(5)]

    return {
        "sentiment": round(max(-1, min(1, avg_sentiment)), 3),
        "confidence": round(confidence, 2),
        "keywords": top_kw,
        "summary": f"Aggregated from {len(sentiments)} texts",
        "n_texts": len(sentiments),
    }


def _analyze_with_llm(text: str, symbol: str = "") -> Optional[dict]:
    """用 Claude API 分析情感。"""
    try:
        import anthropic
    except ImportError:
        return None

    client = anthropic.Anthropic(api_key=_ANTHROPIC_KEY)
    user_text = text[:500]
    if symbol:
        user_text = f"[{symbol}] {user_text}"

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_text}],
            timeout=_TIMEOUT,
        )
        raw = response.content[0].text.strip()

        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        if "sentiment" not in parsed:
            return None

        return {
            "sentiment": float(parsed["sentiment"]),
            "confidence": float(parsed.get("confidence", 0.7)),
            "keywords": parsed.get("keywords", [])[:5],
            "summary": parsed.get("summary", "")[:200],
        }
    except Exception as e:
        log.debug("Claude sentiment API 失败: %s", e)
        return None


def _analyze_with_keywords(text: str) -> dict:
    """关键词规则降级。"""
    text_lower = text.lower()
    words = set(text_lower.split())

    bull_count = len(words & _BULLISH_WORDS)
    bear_count = len(words & _BEARISH_WORDS)
    total = bull_count + bear_count

    if total == 0:
        return {
            "sentiment": 0.0,
            "confidence": 0.2,
            "keywords": [],
            "summary": "No sentiment keywords detected",
        }

    sentiment = (bull_count - bear_count) / total
    matched = list((words & _BULLISH_WORDS) | (words & _BEARISH_WORDS))

    return {
        "sentiment": round(sentiment, 3),
        "confidence": round(min(0.6, 0.2 + total * 0.1), 2),
        "keywords": matched[:5],
        "summary": f"Keyword-based: {bull_count} bullish, {bear_count} bearish",
    }
