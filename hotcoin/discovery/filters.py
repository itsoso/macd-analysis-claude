"""
候选币过滤器

过滤规则:
  - 排除稳定币
  - 排除黑名单
  - 流动性 > $500K 24h 成交额
  - FOMO 过滤: 24h 涨幅 > 30% 不追高
  - 冷却中的币种不参与
"""

import logging
from typing import List

from hotcoin.config import DiscoveryConfig, STABLECOINS, BLACKLIST_SYMBOLS

log = logging.getLogger("hotcoin.filter")


class CoinFilter:
    """候选币过滤链。"""

    def __init__(self, config: DiscoveryConfig):
        self.config = config

    def apply(self, coins: list) -> list:
        """过滤候选币列表, 返回通过所有条件的币种。"""
        result = []
        for coin in coins:
            reason = self._check(coin)
            if reason:
                log.debug("过滤 %s: %s", coin.symbol, reason)
                continue
            result.append(coin)
        return result

    def _check(self, coin) -> str:
        """返回过滤原因, 空字符串表示通过。"""
        sym = coin.symbol

        if sym in STABLECOINS:
            return "稳定币"
        if sym in BLACKLIST_SYMBOLS:
            return "黑名单"

        is_listing = coin.has_listing_signal
        if is_listing:
            listing_min = self.config.min_quote_volume_24h * 0.1
            if coin.quote_volume_24h < listing_min:
                return f"新币流动性过低 (${coin.quote_volume_24h:,.0f} < ${listing_min:,.0f})"
        elif coin.quote_volume_24h < self.config.min_quote_volume_24h:
            return f"流动性不足 (${coin.quote_volume_24h:,.0f} < ${self.config.min_quote_volume_24h:,.0f})"

        if coin.price_change_24h > self.config.max_price_change_24h:
            return f"FOMO过滤 (24h={coin.price_change_24h:+.1%} > {self.config.max_price_change_24h:+.1%})"

        if coin.status == "cooling":
            return "冷却中"

        return ""
