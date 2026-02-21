"""
五层组合风控

L1 单笔: 止损 -3%~-5% (ATR 自适应)
L2 单币: 单币最大亏损 -5% 总资金, 冷却 30min
L3 行业聚集: 同板块 ≤ 20%
L4 日度: 日亏损 -5% 总资金 → 停新仓
L5 组合: 总回撤 -15% → 全清仓 + 冷却 24h
附加: 流动性门槛 / FOMO 过滤
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from hotcoin.config import ExecutionConfig

log = logging.getLogger("hotcoin.risk")


@dataclass
class Position:
    """活跃持仓。"""
    symbol: str
    side: str                       # "BUY" / "SELL"
    entry_price: float
    qty: float
    entry_time: float               # unix ts
    sector: str = "unknown"         # "meme" / "defi" / "ai" / "l2" / ...
    partial_exits: int = 0
    max_pnl_pct: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class RiskState:
    """风控状态。"""
    daily_pnl: float = 0.0
    daily_start_time: float = 0.0
    peak_equity: float = 0.0
    total_equity: float = 0.0
    halted: bool = False
    halt_until: float = 0.0
    halt_reason: str = ""


class PortfolioRisk:
    """五层组合风控引擎。"""

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.state = RiskState(
            peak_equity=config.initial_capital,
            total_equity=config.initial_capital,
            daily_start_time=time.time(),
        )
        self._positions: Dict[str, Position] = {}

    @property
    def positions(self) -> Dict[str, Position]:
        return self._positions

    @property
    def num_positions(self) -> int:
        return len(self._positions)

    # ------------------------------------------------------------------
    # 开仓前检查
    # ------------------------------------------------------------------

    def can_open(self, symbol: str, quote_amount: float, sector: str = "unknown",
                 current_prices: Optional[Dict[str, float]] = None) -> tuple:
        """
        检查是否允许开新仓。
        返回 (allowed: bool, reason: str)。
        """
        self._check_daily_reset()

        # L5: 全局暂停
        if self.state.halted:
            if time.time() < self.state.halt_until:
                return False, f"L5 组合暂停 ({self.state.halt_reason})"
            else:
                self.state.halted = False
                log.info("L5 冷却结束, 恢复交易")

        # L4: 日度亏损限额
        daily_loss_limit = self.config.initial_capital * self.config.daily_max_loss_pct
        if self.state.daily_pnl <= -daily_loss_limit:
            return False, f"L4 日亏损达限 ({self.state.daily_pnl:+.2f} <= -{daily_loss_limit:.2f})"

        # 最大并发仓位
        if len(self._positions) >= self.config.max_concurrent_positions:
            return False, f"并发仓位已满 ({len(self._positions)}/{self.config.max_concurrent_positions})"

        # 单币重复
        if symbol in self._positions:
            return False, f"{symbol} 已有持仓"

        # 单币资金上限
        max_single = self.config.initial_capital * self.config.max_single_position_pct
        if quote_amount > max_single:
            return False, f"单币金额超限 (${quote_amount:.0f} > ${max_single:.0f})"

        # 总敞口 (使用当前市价而非入场价)
        prices = current_prices or {}
        total_exposure = sum(
            p.qty * prices.get(p.symbol, p.entry_price)
            for p in self._positions.values()
        ) + quote_amount
        max_exposure = self.config.initial_capital * self.config.max_total_exposure_pct
        if total_exposure > max_exposure:
            return False, f"总敞口超限 (${total_exposure:.0f} > ${max_exposure:.0f})"

        # L3: 行业聚集 (同样使用当前市价)
        sector_exposure = sum(
            p.qty * prices.get(p.symbol, p.entry_price)
            for p in self._positions.values()
            if p.sector == sector
        ) + quote_amount
        max_sector = self.config.initial_capital * self.config.max_sector_exposure_pct
        if sector_exposure > max_sector:
            return False, f"L3 板块聚集超限 ({sector}: ${sector_exposure:.0f} > ${max_sector:.0f})"

        return True, "OK"

    # ------------------------------------------------------------------
    # 持仓管理
    # ------------------------------------------------------------------

    def open_position(self, symbol: str, side: str, price: float,
                      qty: float, sector: str = "unknown"):
        pos = Position(
            symbol=symbol, side=side, entry_price=price,
            qty=qty, entry_time=time.time(), sector=sector,
        )
        self._positions[symbol] = pos
        log.info("开仓 %s %s qty=%.6f @ $%.4f", side, symbol, qty, price)

    def close_position(self, symbol: str, price: float, reason: str = "") -> float:
        """平仓, 返回已实现 PnL。"""
        pos = self._positions.pop(symbol, None)
        if not pos:
            return 0.0

        if pos.side == "BUY":
            pnl = (price - pos.entry_price) * pos.qty
        else:
            pnl = (pos.entry_price - price) * pos.qty

        self.state.daily_pnl += pnl
        self.state.total_equity += pnl
        if self.state.total_equity > self.state.peak_equity:
            self.state.peak_equity = self.state.total_equity

        log.info("平仓 %s %s PnL=$%.2f (%s)", pos.side, symbol, pnl, reason)
        return pnl

    def partial_close(self, symbol: str, price: float, pct: float, reason: str = "") -> float:
        """部分平仓。"""
        pos = self._positions.get(symbol)
        if not pos:
            return 0.0

        close_qty = pos.qty * pct
        if pos.side == "BUY":
            pnl = (price - pos.entry_price) * close_qty
        else:
            pnl = (pos.entry_price - price) * close_qty

        pos.qty -= close_qty
        pos.partial_exits += 1
        pos.realized_pnl += pnl
        self.state.daily_pnl += pnl
        self.state.total_equity += pnl

        log.info("部分平仓 %s %.0f%% qty=%.6f PnL=$%.2f (%s)",
                 symbol, pct * 100, close_qty, pnl, reason)

        if pos.qty < 1e-8:
            self._positions.pop(symbol, None)

        return pnl

    # ------------------------------------------------------------------
    # 实时风控检查
    # ------------------------------------------------------------------

    def check_all(self, current_prices: Dict[str, float]):
        """对所有持仓执行风控检查, 返回需要处理的列表。"""
        actions = []
        self._check_daily_reset()

        # L5: 总回撤检查
        unrealized = sum(self._calc_unrealized_pnl(p, current_prices.get(p.symbol, p.entry_price))
                         for p in self._positions.values())
        current_equity = self.state.total_equity + unrealized
        drawdown = (self.state.peak_equity - current_equity) / self.state.peak_equity if self.state.peak_equity > 0 else 0

        if drawdown >= self.config.total_drawdown_halt_pct:
            log.warning("L5 总回撤 %.1f%% >= %.1f%%, 全清仓!", drawdown * 100,
                        self.config.total_drawdown_halt_pct * 100)
            for sym in list(self._positions.keys()):
                actions.append(("CLOSE_ALL", sym, "L5 总回撤暂停"))
            self.state.halted = True
            self.state.halt_until = time.time() + self.config.cooling_after_halt_sec
            self.state.halt_reason = f"总回撤 {drawdown:.1%}"
            return actions

        # L2: 单币最大亏损
        for sym, pos in list(self._positions.items()):
            price = current_prices.get(sym, pos.entry_price)
            pnl = self._calc_unrealized_pnl(pos, price)
            max_loss = self.config.initial_capital * self.config.single_coin_max_loss_pct
            if pnl <= -max_loss:
                actions.append(("CLOSE", sym, f"L2 单币亏损 ${pnl:.2f}"))

            # 更新最大盈利 (用于追踪止损)
            pnl_pct = pnl / (pos.entry_price * pos.qty) if pos.entry_price * pos.qty > 0 else 0
            if pnl_pct > pos.max_pnl_pct:
                pos.max_pnl_pct = pnl_pct

        return actions

    def _calc_unrealized_pnl(self, pos: Position, current_price: float) -> float:
        if pos.side == "BUY":
            return (current_price - pos.entry_price) * pos.qty
        else:
            return (pos.entry_price - current_price) * pos.qty

    def _check_daily_reset(self):
        """每日 UTC 0:00 重置日度统计 (按自然日边界)。"""
        import datetime
        now = time.time()
        utc_today_start = datetime.datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()
        if self.state.daily_start_time < utc_today_start:
            log.info("日度风控重置 (daily_pnl=$%.2f)", self.state.daily_pnl)
            self.state.daily_pnl = 0.0
            self.state.daily_start_time = utc_today_start

    def get_summary(self) -> dict:
        return {
            "positions": len(self._positions),
            "total_equity": round(self.state.total_equity, 2),
            "daily_pnl": round(self.state.daily_pnl, 2),
            "peak_equity": round(self.state.peak_equity, 2),
            "halted": self.state.halted,
        }
