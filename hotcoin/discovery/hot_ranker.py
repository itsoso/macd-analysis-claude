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

    _ML_SHADOW = True  # True = ML 只记录不影响评分

    def __init__(self, config: DiscoveryConfig, coin_age_fn=None):
        self.cfg = config
        self._coin_age_fn = coin_age_fn
        self._ml_predictor = None

    def update_scores(self, pool):
        """对候选池中所有币种重新评分 (单次批量 commit)。"""
        coins = pool.get_all()
        scored = []
        for coin in coins:
            try:
                self._compute_score(coin)
                scored.append(coin)
            except Exception:
                log.warning("评分失败 %s, 跳过", coin.symbol, exc_info=True)
        if scored:
            pool.update_coins_batch(scored)
            for coin in scored:
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

        # --- 维度 2: 社媒扩散 (Phase 2 — LLM 情感增强) ---
        s_social = min(100, coin.mention_velocity * 10)

        # --- 维度 3: 情绪倾向 (Phase 2 — LLM 情感增强) ---
        s_sentiment = max(0, min(100, (coin.sentiment + 1) * 50))

        # LLM 情感覆盖 (如果有社交文本数据)
        llm_sentiment = self._get_llm_sentiment(coin)
        if llm_sentiment is not None:
            s_sentiment = max(0, min(100, (llm_sentiment + 1) * 50))

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

        # 归一化: 按有效维度权重归一化, 保证动量币/公告币不被空维度稀释
        # 最小权重 0.35: 避免单维度极端膨胀, 同时不过度抑制公告币
        effective_weight = max(active_weight, 0.35)
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

        # --- ML 热度增强 (shadow 模式) ---
        ml_prob = self._predict_hotness_ml(coin)
        if ml_prob is not None:
            ml_score = ml_prob * 100  # 0-1 → 0-100
            if not self._ML_SHADOW:
                # 非 shadow: ML 分数占 20% 权重融合
                score = score * 0.8 + ml_score * 0.2
                coin.heat_score = round(max(0, min(100, score)), 1)
            else:
                log.debug("%s ML hotness=%.1f (shadow, rule=%.1f)",
                          coin.symbol, ml_score, coin.heat_score)

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

    def _predict_hotness_ml(self, coin) -> float:
        """用 ML 模型预测热度概率, 返回 0-1 或 None。"""
        try:
            if self._ml_predictor is None:
                from hotcoin.ml.predictor_hot import get_predictor
                self._ml_predictor = get_predictor()

            # 构造简化特征 (从 coin 属性)
            import pandas as pd
            features = pd.DataFrame([{
                "ret_1": coin.price_change_5m,
                "ret_5": coin.price_change_1h / 12 if coin.price_change_1h else 0,
                "vol_ratio_5_20": coin.volume_surge_ratio,
                "vol_change": coin.volume_surge_ratio - 1,
                "consecutive_green": 0,
                "consecutive_red": 0,
                "vol_surge_3_20": coin.volume_surge_ratio,
                "body_range_ratio": 0.5,
                "mom_1": coin.price_change_5m,
                "mom_3": coin.price_change_5m * 3,
                "mom_5": coin.price_change_1h / 12 * 5 if coin.price_change_1h else 0,
                "mom_10": coin.price_change_1h / 6 if coin.price_change_1h else 0,
            }])
            return self._ml_predictor.predict_hotness(features)
        except Exception as e:
            log.debug("ML hotness 预测失败: %s", e)
            return None

    def _get_llm_sentiment(self, coin) -> float:
        """获取 LLM 情感分析结果, 返回 -1~1 或 None。"""
        try:
            # 只在有社交文本时调用
            texts = getattr(coin, "social_texts", None)
            if not texts:
                return None

            from hotcoin.discovery.sentiment_llm import analyze_batch
            result = analyze_batch(texts, symbol=coin.symbol)
            if result["confidence"] > 0.3:
                return result["sentiment"]
            return None
        except Exception:
            return None
