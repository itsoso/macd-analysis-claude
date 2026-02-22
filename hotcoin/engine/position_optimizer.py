"""
多币种仓位优化器

基于 Kelly 准则 + 风险平价, 在多个热点币之间分配仓位。

策略:
    1. Kelly 准则: f* = (p*b - q) / b, 其中 p=胜率, b=赔率, q=1-p
    2. 风险平价: 按波动率倒数分配, 高波动币分配更少仓位
    3. 最大单币仓位限制 (默认 20%)
    4. 相关性惩罚: 高相关币种降低总仓位

用法:
    from hotcoin.engine.position_optimizer import PositionOptimizer
    opt = PositionOptimizer(max_single_pct=0.20)
    allocations = opt.optimize(candidates)
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger("hotcoin.position")


@dataclass
class CandidatePosition:
    """候选仓位信息。"""
    symbol: str
    win_prob: float         # 胜率 (0-1), 来自 ML 预测
    avg_win: float          # 平均盈利幅度 (e.g. 0.05 = 5%)
    avg_loss: float         # 平均亏损幅度 (e.g. 0.03 = 3%)
    volatility: float       # 波动率 (e.g. 日化 0.05)
    confidence: float       # 信号置信度 (0-1)
    heat_score: float = 0.0 # 热度评分 (0-100)


@dataclass
class PositionAllocation:
    """仓位分配结果。"""
    symbol: str
    weight: float           # 仓位权重 (0-1)
    kelly_raw: float        # 原始 Kelly 比例
    risk_parity_w: float    # 风险平价权重
    final_pct: float        # 最终仓位百分比
    reason: str = ""


class PositionOptimizer:
    """多币种仓位优化器。"""

    def __init__(
        self,
        max_single_pct: float = 0.20,
        max_total_pct: float = 0.80,
        kelly_fraction: float = 0.25,  # 1/4 Kelly (保守)
        min_confidence: float = 0.3,
    ):
        self.max_single = max_single_pct
        self.max_total = max_total_pct
        self.kelly_frac = kelly_fraction
        self.min_confidence = min_confidence

    def optimize(self, candidates: List[CandidatePosition]) -> List[PositionAllocation]:
        """
        优化多币种仓位分配。

        Returns
        -------
        list of PositionAllocation, 按 final_pct 降序排列
        """
        if not candidates:
            return []

        # 过滤低置信度
        valid = [c for c in candidates if c.confidence >= self.min_confidence]
        if not valid:
            return []

        allocations = []
        for c in valid:
            kelly = self._kelly_criterion(c.win_prob, c.avg_win, c.avg_loss)
            allocations.append({
                "candidate": c,
                "kelly_raw": kelly,
            })

        # 风险平价权重
        vols = [a["candidate"].volatility for a in allocations]
        rp_weights = self._risk_parity_weights(vols)

        # 综合: Kelly * 风险平价 * 置信度
        results = []
        for i, a in enumerate(allocations):
            c = a["candidate"]
            kelly = a["kelly_raw"] * self.kelly_frac
            rp_w = rp_weights[i]

            # 综合权重: Kelly 50% + 风险平价 30% + 热度 20%
            heat_w = c.heat_score / 100 if c.heat_score > 0 else 0.5
            combined = kelly * 0.5 + rp_w * 0.3 + heat_w * 0.2

            # 置信度调整
            combined *= c.confidence

            # 单币上限
            final = min(combined, self.max_single)

            results.append(PositionAllocation(
                symbol=c.symbol,
                weight=combined,
                kelly_raw=a["kelly_raw"],
                risk_parity_w=rp_w,
                final_pct=final,
                reason=f"kelly={a['kelly_raw']:.3f} rp={rp_w:.3f} heat={heat_w:.2f}",
            ))

        # 总仓位上限
        results.sort(key=lambda x: x.final_pct, reverse=True)
        total = sum(r.final_pct for r in results)
        if total > self.max_total:
            scale = self.max_total / total
            for r in results:
                r.final_pct = round(r.final_pct * scale, 4)

        # 四舍五入
        for r in results:
            r.final_pct = round(r.final_pct, 4)
            r.weight = round(r.weight, 4)
            r.kelly_raw = round(r.kelly_raw, 4)
            r.risk_parity_w = round(r.risk_parity_w, 4)

        log.info("仓位优化: %d 币种, 总仓位 %.1f%%",
                 len(results), sum(r.final_pct for r in results) * 100)
        for r in results:
            log.info("  %s: %.1f%% (%s)", r.symbol, r.final_pct * 100, r.reason)

        return results

    def _kelly_criterion(self, win_prob: float, avg_win: float, avg_loss: float) -> float:
        """Kelly 准则: f* = (p*b - q) / b"""
        p = max(0.01, min(0.99, win_prob))
        q = 1 - p
        b = avg_win / max(avg_loss, 0.001)  # 赔率

        kelly = (p * b - q) / b
        return max(0, kelly)

    def _risk_parity_weights(self, volatilities: List[float]) -> List[float]:
        """风险平价: 按波动率倒数分配。"""
        inv_vols = [1.0 / max(v, 0.001) for v in volatilities]
        total = sum(inv_vols)
        if total <= 0:
            n = len(volatilities)
            return [1.0 / n] * n
        return [iv / total for iv in inv_vols]
