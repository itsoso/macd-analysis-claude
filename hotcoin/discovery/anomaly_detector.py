"""
量价异动检测 — 基于 miniTicker 每秒判定

检测条件 (全部满足):
  1. 成交量突增: 1min 量 > 20min 均值 * N 倍
  2. 价格急涨: 5min 涨幅 > M%
  3. 流动性门槛: 24h 成交额 > $500K USDT
  4. 排除追高: 24h 涨幅 < 30%
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Set

from hotcoin.config import DiscoveryConfig, STABLECOINS, BLACKLIST_SYMBOLS

log = logging.getLogger("hotcoin.anomaly")


@dataclass
class AnomalySignal:
    """异动信号。"""
    symbol: str
    volume_surge_ratio: float       # 成交量放大倍数
    price_change_5m: float          # 5min 涨幅
    price_change_24h: float         # 24h 涨幅
    quote_volume_24h: float         # 24h USDT 成交额
    detected_at: float              # unix timestamp

    @property
    def summary(self) -> str:
        return (f"{self.symbol} vol_surge={self.volume_surge_ratio:.1f}x "
                f"chg5m={self.price_change_5m:+.2%} "
                f"chg24h={self.price_change_24h:+.2%} "
                f"vol24h=${self.quote_volume_24h:,.0f}")


class AnomalyDetector:
    """全市场成交量异动扫描器。"""

    def __init__(self, config: DiscoveryConfig):
        self.config = config
        # 防重复告警: symbol → last_alert_ts
        self._alert_cooldown: Dict[str, float] = {}
        self._cooldown_sec = 120  # 同一币种 2 分钟内不重复告警
        self._excluded: Set[str] = STABLECOINS | BLACKLIST_SYMBOLS

    def detect(self, symbol: str, ticker) -> Optional[AnomalySignal]:
        if symbol in self._excluded:
            return None

        # 条件 1: 成交量突增
        if ticker.avg_volume_20m <= 0:
            return None
        vol_ratio = ticker.volume_1m / ticker.avg_volume_20m
        if vol_ratio < self.config.volume_surge_ratio:
            return None

        # 条件 2: 5min 涨幅
        if ticker.price_change_5m < self.config.price_surge_5m_pct:
            return None

        # 条件 3: 流动性门槛
        if ticker.quote_volume < self.config.min_quote_volume_24h:
            return None

        # 条件 4: FOMO 过滤
        if ticker.price_change_24h > self.config.max_price_change_24h:
            return None

        # 冷却检查
        now = time.time()
        last = self._alert_cooldown.get(symbol, 0)
        if now - last < self._cooldown_sec:
            return None
        self._alert_cooldown[symbol] = now

        if len(self._alert_cooldown) > 500:
            try:
                expired = [s for s, t in self._alert_cooldown.items() if now - t > self._cooldown_sec * 5]
                for s in expired:
                    self._alert_cooldown.pop(s, None)
            except Exception:
                log.debug("清理告警冷却记录异常", exc_info=True)

        signal = AnomalySignal(
            symbol=symbol,
            volume_surge_ratio=vol_ratio,
            price_change_5m=ticker.price_change_5m,
            price_change_24h=ticker.price_change_24h,
            quote_volume_24h=ticker.quote_volume,
            detected_at=now,
        )
        log.info("异动检测: %s", signal.summary)
        return signal
