"""
热度加权资金分配

规则:
  - 高热度 + 高流动性 → 更多资金
  - 新币/低流动性 → 少量试探性仓位
  - 单币上限: 总资金 * max_single_position_pct
  - 随持仓数增加递减 (避免过度分散)
"""

import logging
from typing import List, Dict

from hotcoin.config import ExecutionConfig

log = logging.getLogger("hotcoin.alloc")


class CapitalAllocator:
    """热度加权资金分配器。"""

    def __init__(self, config: ExecutionConfig):
        self.config = config

    def allocate_single(self, heat_score: float, liquidity_score: float,
                        current_positions: int) -> float:
        """
        为单个币种分配资金金额 (USDT)。

        heat_score: 0-100
        liquidity_score: 0-100
        current_positions: 当前持仓数
        """
        capital = self.config.initial_capital
        max_total = capital * self.config.max_total_exposure_pct
        max_single = capital * self.config.max_single_position_pct

        if current_positions >= self.config.max_concurrent_positions:
            return 0.0

        # 基础分配: 总敞口均分
        remaining_slots = self.config.max_concurrent_positions - current_positions
        base_alloc = max_total / self.config.max_concurrent_positions

        # 热度加权: heat 高 → 乘数增大 (0.6 ~ 1.4)
        heat_mult = 0.6 + 0.8 * (heat_score / 100)

        # 流动性折扣: 低流动性 → 降低仓位
        liq_mult = min(1.0, max(0.3, liquidity_score / 60))

        alloc = base_alloc * heat_mult * liq_mult
        alloc = min(alloc, max_single)

        # 最低下单金额检查
        if alloc < 12:  # Binance 最低 ~$10
            return 0.0

        return round(alloc, 2)

    def allocate_batch(self, candidates: List[Dict]) -> Dict[str, float]:
        """
        批量分配: 对一组候选币同时分配资金。

        candidates: list of {"symbol": str, "heat_score": float, "liquidity_score": float}
        返回: {symbol: alloc_usdt}
        """
        if not candidates:
            return {}

        capital = self.config.initial_capital
        max_total = capital * self.config.max_total_exposure_pct
        max_single = capital * self.config.max_single_position_pct

        # 计算权重
        weights = {}
        for c in candidates:
            w = c["heat_score"] * max(1, c.get("liquidity_score", 50))
            weights[c["symbol"]] = max(0.1, w)

        total_w = sum(weights.values())
        allocations = {}
        for sym, w in weights.items():
            alloc = max_total * (w / total_w) if total_w > 0 else 0
            alloc = min(alloc, max_single)
            if alloc >= 12:
                allocations[sym] = round(alloc, 2)

        return allocations
