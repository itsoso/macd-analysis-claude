"""
NLP 情绪评分 — 轻量级实现

Phase 2 策略:
  1. 先用关键词 + 规则 (零依赖, 即时可用)
  2. 可选 fasttext 模型 (需安装 fasttext 包)
  3. 未来可替换为 LLM API 调用

评分范围: -1.0 (极度负面) ~ +1.0 (极度正面)
"""

import logging
import re
from typing import Optional

log = logging.getLogger("hotcoin.sentiment")

# 加密货币领域正/负面关键词
_POSITIVE_WORDS = {
    "moon", "pump", "rally", "bullish", "breakout", "ath", "surge", "rocket",
    "launch", "listing", "buy", "long", "gem", "undervalued", "accumulate",
    "partnership", "adoption", "upgrade", "mainnet", "airdrop", "staking",
    "burn", "deflationary", "bullrun", "100x", "10x", "launch",
    "上线", "利好", "暴涨", "突破", "新高", "看涨", "牛市",
}

_NEGATIVE_WORDS = {
    "dump", "crash", "scam", "rug", "bearish", "sell", "short", "warning",
    "hack", "exploit", "delist", "delisted", "ponzi", "fraud", "dead",
    "collapse", "liquidation", "rekt", "fud", "overvalued", "bubble",
    "下架", "暴跌", "骗局", "崩盘", "割韭菜", "跑路", "利空", "看跌", "熊市",
}

_UNCERTAIN_WORDS = {
    "rumor", "rumour", "maybe", "might", "unconfirmed", "speculate",
    "possibly", "alleged", "uncertain",
    "传言", "可能", "未确认",
}


class SentimentScorer:
    """
    轻量情绪评分器。

    Phase 2: 关键词 + 规则引擎
    Phase 3: 可替换为 fasttext / 小模型
    """

    def __init__(self):
        self._fasttext_model = None

    def score(self, text: str) -> float:
        """
        返回情绪分数 -1.0 ~ +1.0。

        正面词 → 正分, 负面词 → 负分, 不确定词 → 降低置信度。
        """
        if not text:
            return 0.0

        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))

        pos_count = sum(1 for w in words if w in _POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in _NEGATIVE_WORDS)
        unc_count = sum(1 for w in words if w in _UNCERTAIN_WORDS)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        raw = (pos_count - neg_count) / total  # -1 ~ +1

        # 不确定词降低置信度
        if unc_count > 0:
            raw *= 0.7

        return max(-1.0, min(1.0, raw))

    def score_batch(self, texts: list) -> list:
        return [self.score(t) for t in texts]

    def try_load_fasttext(self, model_path: str) -> bool:
        """尝试加载 fasttext 模型 (可选)。"""
        try:
            import fasttext
            self._fasttext_model = fasttext.load_model(model_path)
            log.info("fasttext 模型已加载: %s", model_path)
            return True
        except ImportError:
            log.warning("fasttext 未安装, 使用关键词引擎")
        except Exception as e:
            log.warning("fasttext 模型加载失败: %s", e)
        return False
