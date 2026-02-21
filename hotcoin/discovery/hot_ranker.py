"""
六维热度评分

Phase 1 仅实现:
  w4 价格动量 (0.25)
  w5 资金流动性 (0.20)
  w6 风险惩罚 (0.10)
合计权重 0.55, 归一化到 0-100 分。

Phase 2 补充:
  w1 公告强度 (0.20)
  w2 社媒扩散 (0.15)
  w3 情绪倾向 (0.10)
"""

import logging
import math
import time

from hotcoin.config import DiscoveryConfig

log = logging.getLogger("hotcoin.ranker")


class HotRanker:
    """六维热度评分引擎。"""

    def __init__(self, config: DiscoveryConfig, coin_age_fn=None):
        self.cfg = config
        self._coin_age_fn = coin_age_fn

    def update_scores(self, pool):
        """对候选池中所有币种重新评分 (单次批量 commit)。"""
        coins = pool.get_all()
        for coin in coins:
            self._compute_score(coin)
        pool.update_coins_batch(coins)
        for coin in coins:
            pool.record_heat_history(coin)

    def _compute_score(self, coin):
        """计算六维热度评分, 0-100 范围。"""
        now = time.time()

        if callable(self._coin_age_fn):
            try:
                coin.coin_age_days = self._coin_age_fn(coin.symbol)
            except Exception:
                pass

        # --- 维度 1: 公告强度 (Phase 2 完善) ---
        s_announce = 0.0
        if coin.has_listing_signal:
            s_announce = 80.0
            if coin.listing_open_time > 0:
                hours_until = max(0, (coin.listing_open_time / 1000 - now) / 3600)
                if hours_until < 1:
                    s_announce = 100.0
                elif hours_until < 24:
                    s_announce = 90.0

        # --- 维度 2: 社媒扩散 (Phase 2) ---
        s_social = min(100, coin.mention_velocity * 10)

        # --- 维度 3: 情绪倾向 (Phase 2) ---
        s_sentiment = max(0, min(100, (coin.sentiment + 1) * 50))

        # --- 维度 4: 价格动量 ---
        s_momentum = self._score_momentum(coin)

        # --- 维度 5: 资金流动性 ---
        s_liquidity = self._score_liquidity(coin)

        # --- 维度 6: 风险惩罚 ---
        s_risk = self._score_risk_penalty(coin)

        # 加权合成: 仅对有数据的维度加权, 避免零维度稀释评分
        dims = [
            (self.cfg.w_announcement, s_announce),
            (self.cfg.w_social, s_social),
            (self.cfg.w_sentiment, s_sentiment),
            (self.cfg.w_momentum, s_momentum),
            (self.cfg.w_liquidity, s_liquidity),
        ]
        # 有效维度: 分数 > 0 或情绪维度(默认50)始终参与
        active_weight = 0.0
        positive_raw = 0.0
        for w, s in dims:
            positive_raw += w * s
            if s > 0:
                active_weight += w

        # 归一化: 按有效维度权重归一化, 保证动量币不被空维度稀释
        # 最小权重 = momentum + liquidity = 0.45, 避免极端膨胀
        effective_weight = max(active_weight, self.cfg.w_momentum + self.cfg.w_liquidity)
        positive_score = (positive_raw / effective_weight) if effective_weight > 0 else 0

        # 风险惩罚: w_risk_penalty 直接控制最大扣减幅度
        # e.g. w_risk_penalty=0.10, s_risk=100 → 扣减 10 分
        score = max(0, min(100, positive_score - s_risk * self.cfg.w_risk_penalty))

        coin.score_announcement = s_announce
        coin.score_social = s_social
        coin.score_sentiment = s_sentiment
        coin.score_momentum = s_momentum
        coin.score_liquidity = s_liquidity
        coin.score_risk_penalty = s_risk
        coin.heat_score = round(score, 1)
        coin.last_score_update = now

        # 低分跟踪
        if score < self.cfg.pool_exit_score:
            if coin.low_score_since == 0:
                coin.low_score_since = now
        else:
            coin.low_score_since = 0

    def _score_momentum(self, coin) -> float:
        """价格动量评分: 0-100。
        5min 涨幅 10% → 满分 50; 1h 涨幅 15% → 满分 30; 量比加成 20。
        """
        score = 0.0
        if coin.price_change_5m > 0:
            score += min(50, (coin.price_change_5m / 0.10) * 50)
        if coin.price_change_1h > 0:
            score += min(30, (coin.price_change_1h / 0.15) * 30)
        if coin.volume_surge_ratio > 1:
            vol_bonus = min(20, math.log2(max(1, coin.volume_surge_ratio)) * 5)
            score += vol_bonus
        return min(100, max(0, score))

    def _score_liquidity(self, coin) -> float:
        """资金流动性评分: 0-100。"""
        vol_24h = coin.quote_volume_24h
        if vol_24h <= 0:
            return 0.0
        # 对数映射: $500K → 30, $5M → 60, $50M → 90
        score = max(0, math.log10(max(1, vol_24h / 1000)) * 20)
        # 成交量突增加成
        if coin.volume_surge_ratio > 3:
            score += min(20, coin.volume_surge_ratio * 2)
        return min(100, score)

    def _score_risk_penalty(self, coin) -> float:
        """风险惩罚: 0-100, 越高越危险。"""
        penalty = 0.0
        # 新币龄惩罚 (< 7 天高风险)
        if 0 <= coin.coin_age_days < 7:
            penalty += 30
        elif 0 <= coin.coin_age_days < 30:
            penalty += 15
        # 24h 涨幅过大 (可能是拉盘)
        if coin.price_change_24h > 0.50:
            penalty += 40
        elif coin.price_change_24h > 0.30:
            penalty += 20
        # 5min 涨幅异常 (> 15% — 可能闪崩前兆)
        if coin.price_change_5m > 0.15:
            penalty += 20
        return min(100, penalty)
