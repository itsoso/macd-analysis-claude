"""
入场/出场规则 — 热点币专用

入场: 15m 定方向 → 5m 触发 → 1m 执行
出场: ATR 止损 + 分层止盈 + 时间止损 + 热度衰退 + 黑天鹅
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from hotcoin.config import TradingConfig

log = logging.getLogger("hotcoin.rules")


@dataclass
class EntryDecision:
    allow: bool
    reason: str
    suggested_sl_pct: float = -0.05
    suggested_size_pct: float = 0.10


@dataclass
class ExitDecision:
    should_exit: bool
    reason: str
    exit_pct: float = 1.0      # 全部平仓的比例
    is_emergency: bool = False


class HotCoinEntryExit:
    """入场/出场规则引擎。"""

    def __init__(self, config: TradingConfig):
        self.config = config

    def should_enter(self, signal, coin=None) -> EntryDecision:
        """
        判断是否入场。

        signal: TradeSignal
        coin: HotCoin (可选, 提供额外上下文)
        """
        if signal.action not in ("BUY", "SELL"):
            return EntryDecision(allow=False, reason="信号为 HOLD")

        if signal.strength < self.config.min_consensus_strength:
            return EntryDecision(
                allow=False,
                reason=f"strength={signal.strength} < {self.config.min_consensus_strength}",
            )

        # 检查 15m 方向确认
        tf_details = signal.tf_details or {}
        confirm_tf = self.config.entry_confirm_tf
        if confirm_tf in tf_details:
            td = tf_details[confirm_tf]
            ss, bs = td.get("ss", 0), td.get("bs", 0)
            if signal.action == "BUY" and bs <= ss:
                return EntryDecision(allow=False, reason=f"{confirm_tf} 方向未确认 (bs={bs:.0f} <= ss={ss:.0f})")
            if signal.action == "SELL" and ss <= bs:
                return EntryDecision(allow=False, reason=f"{confirm_tf} 方向未确认 (ss={ss:.0f} <= bs={bs:.0f})")

        # 如果有候选币信息, 检查热度
        if coin and coin.heat_score < 30:
            return EntryDecision(allow=False, reason=f"热度过低 ({coin.heat_score:.0f})")

        # F1-F3 反欺诈过滤
        filters = getattr(signal, "active_filters", None) or []
        if "F2" in filters:
            return EntryDecision(allow=False, reason="F2过滤: 疑似拉高出货")
        if "F1" in filters and "F3" in filters:
            return EntryDecision(allow=False, reason="F1+F3过滤: 刷量且低流动性")

        # Pump 阶段过滤: 派发阶段不入场
        pump_phase = getattr(signal, "pump_phase", "")
        if pump_phase == "distribution":
            return EntryDecision(allow=False, reason="Pump阶段=派发, 不入场")

        sl = self.config.default_sl_pct
        return EntryDecision(allow=True, reason="OK", suggested_sl_pct=sl)

    def check_exit(self, position: dict, current_price: float,
                   coin=None) -> Optional[ExitDecision]:
        """
        检查是否应该出场。

        position: dict with keys: entry_price, side, qty, entry_time, partial_exits
        """
        entry_price = position.get("entry_price", 0)
        if entry_price <= 0:
            return None

        side = position.get("side", "BUY")
        entry_time = position.get("entry_time", time.time())
        partial_exits = position.get("partial_exits", 0)  # 已执行的分层止盈次数
        holding_sec = time.time() - entry_time

        if side == "BUY":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        # 1. 止损
        if pnl_pct <= self.config.default_sl_pct:
            return ExitDecision(
                should_exit=True,
                reason=f"止损触发 ({pnl_pct:+.2%} <= {self.config.default_sl_pct:+.2%})",
                exit_pct=1.0,
            )

        # 1.5 Pump 派发阶段退出 (盈利时部分平仓, 仅触发一次)
        if coin and partial_exits == 0:
            cp = getattr(coin, "pump_phase", "")
            if cp == "distribution" and pnl_pct > 0:
                return ExitDecision(
                    should_exit=True,
                    reason=f"Pump进入派发阶段, 获利退出 ({pnl_pct:+.2%})",
                    exit_pct=0.5,
                )

        # 2. 黑天鹅 (5min 剧烈反向波动)
        if coin:
            chg = getattr(coin, "price_change_5m", 0)
            if side == "BUY" and chg < self.config.black_swan_pct:
                return ExitDecision(
                    should_exit=True,
                    reason=f"黑天鹅 (5m={chg:+.2%})",
                    exit_pct=1.0,
                    is_emergency=True,
                )
            if side == "SELL" and chg > abs(self.config.black_swan_pct):
                return ExitDecision(
                    should_exit=True,
                    reason=f"黑天鹅做空 (5m={chg:+.2%})",
                    exit_pct=1.0,
                    is_emergency=True,
                )

        # 3. 分层止盈 (跳过已执行的档位, 触发最高已达标的未执行档位)
        best_tier = None
        for tier_idx, (tp_pct, exit_ratio) in enumerate(self.config.take_profit_tiers):
            if tier_idx < partial_exits:
                continue
            if pnl_pct >= tp_pct:
                best_tier = (tier_idx, tp_pct, exit_ratio)
            else:
                break
        if best_tier is not None:
            tier_idx, tp_pct, exit_ratio = best_tier
            return ExitDecision(
                should_exit=True,
                reason=f"止盈T{tier_idx+1} ({pnl_pct:+.2%} >= {tp_pct:+.2%})",
                exit_pct=exit_ratio,
            )

        # 4. 追踪止损 (已过所有止盈层后)
        if partial_exits >= len(self.config.take_profit_tiers):
            max_pnl = position.get("max_pnl_pct", pnl_pct)
            trailing = max_pnl - self.config.trailing_stop_pct
            if pnl_pct < trailing:
                return ExitDecision(
                    should_exit=True,
                    reason=f"追踪止损 (回撤 {max_pnl:+.2%} → {pnl_pct:+.2%})",
                    exit_pct=1.0,
                )

        # 5. 时间止损
        max_hold = self.config.max_hold_minutes * 60
        if holding_sec > max_hold:
            return ExitDecision(
                should_exit=True,
                reason=f"时间止损 ({holding_sec/60:.0f}min > {self.config.max_hold_minutes}min)",
                exit_pct=1.0,
            )

        # 6. 热度衰退
        if coin and coin.status == "cooling":
            return ExitDecision(
                should_exit=True,
                reason="热度衰退, 币种已出池",
                exit_pct=1.0,
            )

        return None
