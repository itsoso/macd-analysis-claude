"""
三级预警评分器

评分维度 (满分 100):
  动量/结构  35%  — 多周期动量、突破计数、挤压信号
  成交量/模式 35% — S1 信号、收盘位置
  资金流向   20%  — 买压比例、S4/S5
  模式综合   10%  — Pump 阶段加成

预警级别:
  L1 预警     (≥60): 有 S1 或 S4 触发
  L2 起飞     (≥75): 中周期(15m+) 有 S2/S3 + 短周期(1m/5m) 有 S1
  L3 确认加速 (≥85): L2 条件 + 连续 2 轮高分

过滤惩罚: F1 → score * 0.5, F2 → score * 0.2, F3 → score * 0.7
"""

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from hotcoin.engine.pump_detector import PumpPhase, PumpResult
from hotcoin.engine.signal_scanner import AggregatedScan


@dataclass
class AlertResult:
    symbol: str
    total_score: float = 0.0
    level: str = "NONE"          # "NONE" | "L1" | "L2" | "L3"
    level_name: str = ""         # "" | "预警" | "起飞" | "确认加速"
    action_hint: str = ""
    # 4 维评分
    momentum_score: float = 0.0  # 0-35
    volume_score: float = 0.0    # 0-35
    flow_score: float = 0.0      # 0-20
    pattern_score: float = 0.0   # 0-10
    filter_penalty: float = 0.0  # 乘法惩罚后扣减的分
    # 关键信息
    pump_phase: str = "normal"
    active_signals: List[str] = field(default_factory=list)
    active_filters: List[str] = field(default_factory=list)


# 周期权重 (短→长, sum 不需要 = 100)
_TF_WEIGHTS: Dict[str, float] = {
    "1m": 5, "3m": 8, "5m": 10, "15m": 15,
    "30m": 15, "1h": 20, "2h": 20, "4h": 20,
}

_SHORT_TFS = {"1m", "3m", "5m"}
_MEDIUM_TFS = {"15m", "30m", "1h", "2h", "4h"}


class AlertScorer:
    """三级预警评分器。"""

    def __init__(self):
        self._score_history: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def score(
        self,
        pump: Optional[PumpResult],
        scan: Optional[AggregatedScan],
        tf_details: Optional[dict] = None,
    ) -> AlertResult:
        symbol = scan.symbol if scan else ""
        result = AlertResult(symbol=symbol)

        if scan is None:
            return result

        # ---- 动量/结构 (0-35) ----
        momentum = self._calc_momentum(scan, tf_details or {})

        # ---- 成交量/模式 (0-35) ----
        volume = self._calc_volume(scan)

        # ---- 资金流向 (0-20) ----
        flow = self._calc_flow(scan)

        # ---- 模式综合 (0-10) ----
        pattern = self._calc_pattern(pump)

        raw_total = momentum + volume + flow + pattern

        # ---- 过滤惩罚 (乘法) ----
        penalty_mult = 1.0
        if scan.f2_hit:
            penalty_mult *= 0.2
        if scan.f1_hit:
            penalty_mult *= 0.5
        if scan.f3_hit:
            penalty_mult *= 0.7

        final_score = max(0, min(100, raw_total * penalty_mult))
        filter_penalty = raw_total - final_score

        result.momentum_score = round(momentum, 1)
        result.volume_score = round(volume, 1)
        result.flow_score = round(flow, 1)
        result.pattern_score = round(pattern, 1)
        result.filter_penalty = round(filter_penalty, 1)
        result.total_score = round(final_score, 1)
        result.active_signals = list(scan.active_signals)
        result.active_filters = list(scan.active_filters)
        result.pump_phase = pump.phase.value if pump else "normal"

        # ---- 预警级别 (线程安全读写 history) ----
        with self._lock:
            level, name, hint = self._determine_level(final_score, scan, symbol)
            result.level = level
            result.level_name = name
            result.action_hint = hint

            # 更新历史 (用于 L3 检测)
            hist = self._score_history.setdefault(symbol, [])
            hist.append(final_score)
            if len(hist) > 10:
                self._score_history[symbol] = hist[-10:]

        return result

    def _calc_momentum(self, scan: AggregatedScan, tf_details: dict) -> float:
        """动量/结构评分, 最高 35 分。"""
        score = 0.0
        # 突破信号 (S2)
        for tf in scan.s2_tfs:
            w = _TF_WEIGHTS.get(tf, 10) / 100
            score += 5 * w
        score = min(15, score)

        # 挤压突破 (S3)
        score += min(10, len(scan.s3_tfs) * 5)

        # 多周期一致性 (从 tf_details 的 buy/sell 方向判断)
        if tf_details:
            long_count = sum(1 for d in tf_details.values()
                             if isinstance(d, dict) and d.get("bs", 0) > d.get("ss", 0))
            if long_count >= 3:
                score += 5
            elif long_count >= 2:
                score += 3

        return min(35, score)

    def _calc_volume(self, scan: AggregatedScan) -> float:
        """成交量/模式评分, 最高 35 分。"""
        score = 0.0
        # S1 信号
        for tf in scan.s1_tfs:
            w = _TF_WEIGHTS.get(tf, 10) / 100
            score += 8 * w
        score = min(20, score)

        # 多周期 S1 加成
        if len(scan.s1_tfs) >= 2:
            score += 5
        if len(scan.s1_tfs) >= 3:
            score += 5

        return min(35, score)

    def _calc_flow(self, scan: AggregatedScan) -> float:
        """资金流向评分, 最高 20 分。"""
        score = 0.0
        # S4 买压优势
        if scan.s4_tfs:
            score += min(10, len(scan.s4_tfs) * 5)
        # S5 持续买入
        if scan.s5_tfs:
            score += min(10, len(scan.s5_tfs) * 5)
        return min(20, score)

    def _calc_pattern(self, pump: Optional[PumpResult]) -> float:
        """模式综合评分, 最高 10 分。"""
        if pump is None:
            return 0.0
        if pump.phase == PumpPhase.MAIN_PUMP:
            return 10
        if pump.phase == PumpPhase.EARLY_PUMP:
            return 7
        if pump.phase == PumpPhase.ACCUMULATION:
            return 3
        return 0

    def _determine_level(
        self, score: float, scan: AggregatedScan, symbol: str,
    ) -> tuple:
        """判定预警级别。返回 (level, name, hint)。"""
        # L1 条件
        l1 = score >= 60 and (scan.s1_tfs or scan.s4_tfs)
        # L2 条件
        has_medium = any(tf in _MEDIUM_TFS for tf in (scan.s2_tfs + scan.s3_tfs))
        has_short_s1 = any(tf in _SHORT_TFS for tf in scan.s1_tfs)
        l2 = score >= 75 and has_medium and has_short_s1

        # L3 条件: L2 + 连续 2 轮高分
        sustained = False
        hist = self._score_history.get(symbol, [])
        if len(hist) >= 2:
            sustained = all(s >= 75 for s in hist[-2:])
        l3 = l2 and sustained

        if l3:
            return "L3", "确认加速", "趋势确认, 可加仓追踪"
        if l2:
            return "L2", "起飞", "重点盯盘, 寻找入场"
        if l1:
            return "L1", "预警", "加入观察列表"
        return "NONE", "", ""
