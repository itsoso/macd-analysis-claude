"""
LLM 公告解析器

用 Claude API 结构化提取币安公告信息, 降级到正则。

用法:
    from hotcoin.discovery.announcement_llm import parse_announcement
    result = parse_announcement("Binance Will List XYZ (XYZ)")

环境变量:
    ANTHROPIC_API_KEY: Claude API Key
"""

import json
import logging
import os
import re
import time
from typing import Dict, Optional

log = logging.getLogger("hotcoin.announcement_llm")

_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
_CACHE: Dict[str, dict] = {}  # announcement_id -> parsed result
_TIMEOUT = 3.0

# 正则降级用
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

_SYSTEM_PROMPT = """You are a cryptocurrency exchange announcement parser.
Extract structured information from the announcement title and body.
Return ONLY valid JSON with these fields:
- event_type: one of "listing", "delisting", "airdrop", "upgrade", "partnership", "other"
- symbols: array of trading pair symbols (e.g. ["XYZUSDT"])
- listing_time: ISO 8601 timestamp if mentioned, null otherwise
- confidence: float 0-1
- summary: one-sentence English summary"""


def parse_announcement(
    title: str,
    body: str = "",
    announcement_id: Optional[str] = None,
) -> dict:
    """
    解析公告, 返回结构化 JSON。

    优先用 Claude API, 失败降级到正则。
    """
    # 缓存命中
    cache_key = announcement_id or title
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    result = None

    # 尝试 LLM
    if _ANTHROPIC_KEY:
        try:
            result = _parse_with_llm(title, body)
        except Exception as e:
            log.warning("LLM 解析失败, 降级正则: %s", e)

    # 降级正则
    if result is None:
        result = _parse_with_regex(title)

    _CACHE[cache_key] = result
    return result


def _parse_with_llm(title: str, body: str = "") -> Optional[dict]:
    """用 Claude API 解析公告。"""
    try:
        import anthropic
    except ImportError:
        log.debug("anthropic SDK 未安装")
        return None

    client = anthropic.Anthropic(api_key=_ANTHROPIC_KEY)
    user_text = f"Title: {title}"
    if body:
        user_text += f"\nBody: {body[:500]}"

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_text}],
            timeout=_TIMEOUT,
        )
        text = response.content[0].text.strip()

        # 提取 JSON (可能被 markdown 包裹)
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        parsed = json.loads(text)

        # 验证必要字段
        if "event_type" not in parsed or "symbols" not in parsed:
            log.warning("LLM 返回缺少必要字段: %s", text[:100])
            return None

        log.info("LLM 解析成功: %s → %s", title[:60], parsed.get("event_type"))
        return parsed

    except json.JSONDecodeError as e:
        log.warning("LLM 返回非 JSON: %s", e)
        return None
    except Exception as e:
        log.warning("Claude API 调用失败: %s", e)
        return None


def _parse_with_regex(title: str) -> dict:
    """正则降级解析。"""
    result = {
        "event_type": "other",
        "symbols": [],
        "listing_time": None,
        "confidence": 0.5,
        "summary": title[:100],
    }

    title_lower = title.lower()

    # 事件类型检测
    if any(kw in title_lower for kw in ("will list", "new listing", "上线", "will add")):
        result["event_type"] = "listing"
        result["confidence"] = 0.7
    elif any(kw in title_lower for kw in ("delist", "remove", "下线")):
        result["event_type"] = "delisting"
        result["confidence"] = 0.7
    elif any(kw in title_lower for kw in ("airdrop", "空投")):
        result["event_type"] = "airdrop"
        result["confidence"] = 0.6
    elif any(kw in title_lower for kw in ("upgrade", "maintenance", "升级", "维护")):
        result["event_type"] = "upgrade"
        result["confidence"] = 0.6

    # 提取币种
    m = _WILL_LIST_RE.search(title)
    if m:
        token = m.group(1).upper()
        result["symbols"] = [f"{token}USDT"]
        result["confidence"] = min(result["confidence"] + 0.1, 0.9)
    else:
        tickers = _TICKER_RE.findall(title)
        symbols = [f"{t}USDT" for t in tickers if len(t) >= 2 and t not in _STOP_WORDS]
        result["symbols"] = symbols[:3]  # 最多 3 个

    return result
