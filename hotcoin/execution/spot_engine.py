"""
现货交易引擎

整合:
  - SignalDispatcher (信号)
  - EntryExitRules (入场/出场判断)
  - PortfolioRisk (五层风控)
  - OrderExecutor (下单)
  - CapitalAllocator (资金分配)
  - PnLTracker (损益记录)
"""

import logging
import time
from typing import Dict, List, Optional

from hotcoin.config import HotCoinConfig
from hotcoin.discovery.candidate_pool import HotCoin, CandidatePool
from hotcoin.engine.signal_worker import TradeSignal
from hotcoin.engine.entry_exit_rules import HotCoinEntryExit
from hotcoin.execution.order_executor import OrderExecutor
from hotcoin.execution.portfolio_risk import PortfolioRisk
from hotcoin.execution.capital_allocator import CapitalAllocator
from hotcoin.execution.pnl_tracker import PnLTracker

log = logging.getLogger("hotcoin.spot")


class HotCoinSpotEngine:
    """
    现货交易引擎 — 管理多币种并发持仓。

    核心流程:
      1. 收到信号 → 入场检查 (EntryExitRules + PortfolioRisk)
      2. 分配资金 (CapitalAllocator)
      3. 下单 (OrderExecutor)
      4. 持仓管理: 止盈/止损/时间止损
      5. PnL 记录
    """

    def __init__(self, config: HotCoinConfig, pool: CandidatePool):
        self.config = config
        self.pool = pool

        self.rules = HotCoinEntryExit(config.trading)
        self.risk = PortfolioRisk(config.execution)
        self.allocator = CapitalAllocator(config.execution)
        self.executor = OrderExecutor(config.execution)
        self.pnl = PnLTracker()

    @property
    def num_positions(self) -> int:
        return self.risk.num_positions

    @property
    def positions(self) -> dict:
        return self.risk.positions

    def process_signals(self, signals: List[TradeSignal]):
        """处理一批交易信号。"""
        for sig in signals:
            if sig.action == "HOLD":
                continue

            if sig.action == "SELL":
                self._handle_sell_signal(sig)
                continue

            coin = self.pool.get(sig.symbol)
            entry_decision = self.rules.should_enter(sig, coin)

            if not entry_decision.allow:
                log.debug("拒绝 %s %s: %s", sig.action, sig.symbol, entry_decision.reason)
                continue

            self._try_open(sig, coin, entry_decision)

    def _handle_sell_signal(self, sig: TradeSignal):
        """
        现货模式不支持开空:
        - 若已有 BUY 持仓: 视为平仓信号
        - 若无仓位: 仅记录并忽略
        """
        pos = self.risk.positions.get(sig.symbol)
        if not pos:
            log.debug("忽略 SELL %s: 现货无仓位可平", sig.symbol)
            return

        if pos.side != "BUY":
            log.warning("忽略 SELL %s: 现货仅支持平 BUY 持仓 (current=%s)", sig.symbol, pos.side)
            return

        price = self.executor.get_current_price(sig.symbol)
        if price <= 0:
            log.warning("忽略 SELL %s: 无法获取价格", sig.symbol)
            return
        self._close_position(sig.symbol, price, f"SELL signal: {sig.reason or 'signal exit'}")

    def _try_open(self, sig: TradeSignal, coin: Optional[HotCoin], entry_decision):
        """尝试开仓。"""
        # 分配资金
        heat_score = coin.heat_score if coin else 50
        liquidity_score = coin.score_liquidity if coin else 50
        alloc = self.allocator.allocate_single(
            heat_score=heat_score,
            liquidity_score=liquidity_score,
            current_positions=self.risk.num_positions,
        )

        if alloc <= 0:
            log.debug("%s 分配资金为 0", sig.symbol)
            return

        # 风控检查
        ok, reason = self.risk.can_open(sig.symbol, alloc)
        if not ok:
            log.info("风控拒绝 %s: %s", sig.symbol, reason)
            return

        # 现货开仓只支持 BUY
        result = self.executor.spot_market_buy(sig.symbol, alloc)

        price = result.get("price", 0) or self.executor.get_current_price(sig.symbol)
        qty = result.get("qty", 0) or (alloc / price if price else 0)

        if price > 0 and qty > 0:
            self.risk.open_position(sig.symbol, "BUY", price, qty)
            if coin:
                coin.status = "trading"
                self.pool.update_coin(coin)
            log.info("开仓 %s %s $%.2f @ $%.6f (heat=%.0f)",
                     sig.action, sig.symbol, alloc, price,
                     heat_score)

    def check_positions(self, current_prices: Dict[str, float]):
        """
        持仓巡检: 止盈/止损/时间止损/热度衰退/黑天鹅。
        """
        # 五层风控检查
        risk_actions = self.risk.check_all(current_prices)
        for action_type, symbol, reason in risk_actions:
            if action_type in ("CLOSE", "CLOSE_ALL"):
                price = current_prices.get(symbol, 0)
                if price > 0:
                    self._close_position(symbol, price, reason)

        # 逐仓检查 EntryExitRules
        for sym, pos in list(self.risk.positions.items()):
            price = current_prices.get(sym, 0)
            if price <= 0:
                continue

            coin = self.pool.get(sym)
            position_dict = {
                "entry_price": pos.entry_price,
                "side": pos.side,
                "qty": pos.qty,
                "entry_time": pos.entry_time,
                "partial_exits": pos.partial_exits,
                "max_pnl_pct": pos.max_pnl_pct,
            }

            exit_decision = self.rules.check_exit(position_dict, price, coin)
            if exit_decision and exit_decision.should_exit:
                if exit_decision.exit_pct >= 1.0:
                    self._close_position(sym, price, exit_decision.reason)
                else:
                    self._partial_close(sym, price, exit_decision.exit_pct, exit_decision.reason)

    def _close_position(self, symbol: str, price: float, reason: str):
        """全部平仓。"""
        pos = self.risk.positions.get(symbol)
        if not pos:
            return

        result = self.executor.spot_market_sell(symbol, pos.qty)
        if result.get("error") and not self.config.execution.use_paper_trading:
            log.error("平仓下单失败 %s: %s — 保留仓位", symbol, result.get("error"))
            return

        pnl = self.risk.close_position(symbol, price, reason)
        self.pnl.record_trade(
            symbol=symbol, side=pos.side,
            entry_price=pos.entry_price, exit_price=price,
            qty=pos.qty, entry_time=pos.entry_time,
            reason=reason,
        )

        if pnl < 0:
            self.pool.set_cooling(symbol)

    def _partial_close(self, symbol: str, price: float, pct: float, reason: str):
        """部分平仓。"""
        pos = self.risk.positions.get(symbol)
        if not pos:
            return

        close_qty = pos.qty * pct
        result = self.executor.spot_market_sell(symbol, close_qty)
        if result.get("error") and not self.config.execution.use_paper_trading:
            log.error("部分平仓下单失败 %s: %s — 保留仓位", symbol, result.get("error"))
            return

        self.risk.partial_close(symbol, price, pct, reason)

    def get_status(self) -> dict:
        """返回引擎状态摘要。"""
        return {
            "positions": self.risk.num_positions,
            "risk_summary": self.risk.get_summary(),
            "pnl_summary": self.pnl.get_summary(),
            "paper": self.config.execution.use_paper_trading,
        }
