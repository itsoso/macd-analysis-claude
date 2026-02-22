"""
热度加权资金分配

规则:
  - 高热度 + 高流动性 → 更多资金
  - 新币/低流动性 → 少量试探性仓位
  - 单币上限: 总资金 * max_single_position_pct
  - 考虑已占资金, 按剩余可用额度分配
"""

import logging
from typing import List, Dict

from hotcoin.config import ExecutionConfig

log = logging.getLogger("hotcoin.alloc")


class CapitalAllocator:
    """热度加权资金分配器。"""

    def __init__(self, config: ExecutionConfig):
        self.config = config

    # 最小分配金额 (Binance 下限约 $10, 留余量)
    MIN_ALLOC_USDT = 12.0

    def allocate_single(self, heat_score: float, liquidity_score: float,
                        current_positions: int,
                        used_exposure: float = 0.0,
                        alert_level: str = "",
                        pump_phase: str = "") -> float:
        """
        为单个币种分配资金金额 (USDT)。

        heat_score: 0-100
        liquidity_score: 0-100
        current_positions: 当前持仓数
        used_exposure: 已占用的总敞口金额 (USDT)
        alert_level: 预警等级 (NONE/L1/L2/L3)
        pump_phase: 动量阶段 (normal/accumulation/early_pump/main_pump/distribution)
        """
        # 派发阶段拒绝开仓
        if pump_phase == "distribution":
            return 0.0

        capital = self.config.initial_capital
        max_total = capital * self.config.max_total_exposure_pct
        max_single = capital * self.config.max_single_position_pct

        if current_positions >= self.config.max_concurrent_positions:
            return 0.0

        remaining_slots = self.config.max_concurrent_positions - current_positions
        remaining_capital = max(0, max_total - used_exposure)

        if remaining_capital < self.MIN_ALLOC_USDT or remaining_slots <= 0:
            return 0.0

        base_alloc = remaining_capital / remaining_slots

        clamped_heat = max(0.0, min(100.0, heat_score))
        heat_mult = 0.6 + 0.8 * (clamped_heat / 100)
        liq_mult = min(1.0, max(0.3, liquidity_score / 60))

        # 预警等级加成: L2 +20%, L3 +40%
        alert_mult = 1.0
        if alert_level == "L3":
            alert_mult = 1.4
        elif alert_level == "L2":
            alert_mult = 1.2

        alloc = base_alloc * heat_mult * liq_mult * alert_mult
        alloc = min(alloc, max_single, remaining_capital)

        if alloc < self.MIN_ALLOC_USDT:
            return 0.0

        return round(alloc, 2)

    def allocate_batch(self, candidates: List[Dict],
                       current_positions: int = 0,
                       used_exposure: float = 0.0) -> Dict[str, float]:
        """
        批量分配: 对一组候选币同时分配资金。

        candidates: list of {"symbol": str, "heat_score": float, "liquidity_score": float}
        current_positions: 当前已有持仓数
        used_exposure: 当前已占用的总敞口
        返回: {symbol: alloc_usdt}
        """
        if not candidates:
            return {}

        capital = self.config.initial_capital
        max_total = capital * self.config.max_total_exposure_pct
        max_single = capital * self.config.max_single_position_pct

        remaining_capital = max(0, max_total - used_exposure)
        available_slots = max(0, self.config.max_concurrent_positions - current_positions)

        # 限制候选数量不超过可用槽位
        candidates = candidates[:available_slots]
        if not candidates or remaining_capital < self.MIN_ALLOC_USDT:
            return {}

        # 过滤掉派发阶段的币
        candidates = [c for c in candidates if c.get("pump_phase") != "distribution"]
        if not candidates:
            return {}

        weights = {}
        for c in candidates:
            w = c["heat_score"] * max(1, c.get("liquidity_score", 50))
            # 预警加成
            al = c.get("alert_level", "")
            if al == "L3":
                w *= 1.4
            elif al == "L2":
                w *= 1.2
            weights[c["symbol"]] = max(0.1, w)

        total_w = sum(weights.values())
        allocations = {}
        total_alloc = 0.0
        for sym, w in weights.items():
            budget_left = remaining_capital - total_alloc
            if budget_left < self.MIN_ALLOC_USDT:
                break
            alloc = remaining_capital * (w / total_w) if total_w > 0 else 0
            alloc = min(alloc, max_single, budget_left)
            if alloc >= self.MIN_ALLOC_USDT:
                allocations[sym] = round(alloc, 2)
                total_alloc += alloc

        return allocations
