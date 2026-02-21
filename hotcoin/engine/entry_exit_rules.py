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

        # 2. 黑天鹅 (15min 跌超阈值)
        if coin and hasattr(coin, "price_change_15m"):
            chg = getattr(coin, "price_change_15m", 0)
            if side == "BUY" and chg < self.config.black_swan_pct:
                return ExitDecision(
                    should_exit=True,
                    reason=f"黑天鹅 (15m={chg:+.2%})",
                    exit_pct=1.0,
                    is_emergency=True,
                )

        # 3. 分层止盈
        for tier_idx, (tp_pct, exit_ratio) in enumerate(self.config.take_profit_tiers):
            if pnl_pct >= tp_pct and partial_exits <= tier_idx:
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
