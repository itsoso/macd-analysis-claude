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

    def __init__(self, config: HotCoinConfig, pool: CandidatePool, event_sink=None):
        self.config = config
        self.pool = pool
        self._event_sink = event_sink

        self.rules = HotCoinEntryExit(config.trading)
        self.risk = PortfolioRisk(config.execution)
        self.allocator = CapitalAllocator(config.execution)
        self.executor = OrderExecutor(config.execution)
        self.pnl = PnLTracker()

    def _emit_event(self, event_type: str, payload: dict, trace_id: str = ""):
        if not callable(self._event_sink):
            return
        try:
            self._event_sink(event_type, payload, trace_id=trace_id)
        except Exception:
            log.debug("event_sink 写入失败: %s", event_type, exc_info=True)

    @property
    def num_positions(self) -> int:
        return self.risk.num_positions

    @property
    def positions(self) -> dict:
        return self.risk.positions

    def process_signals(self, signals: List[TradeSignal], allow_open: bool = True):
        """处理一批交易信号。"""
        for sig in signals:
            if sig.action == "HOLD":
                continue

            if sig.action == "SELL":
                self._handle_sell_signal(sig)
                continue
            if not allow_open:
                log.debug("忽略开仓信号 %s %s: allow_open=False", sig.action, sig.symbol)
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
        heat_score = coin.heat_score if coin else 50
        liquidity_score = coin.score_liquidity if coin else 50

        current_price = self.executor.get_current_price(sig.symbol) or 0
        if current_price <= 0:
            log.warning("放弃开仓 %s: 无法获取当前价格", sig.symbol)
            return

        prices = {sig.symbol: current_price}

        used = sum(
            p.qty * prices.get(p.symbol, p.entry_price)
            for p in self.risk.positions.values()
        )
        alloc = self.allocator.allocate_single(
            heat_score=heat_score,
            liquidity_score=liquidity_score,
            current_positions=self.risk.num_positions,
            used_exposure=used,
        )

        if alloc <= 0:
            log.debug("%s 分配资金为 0", sig.symbol)
            return

        # 风控检查 (传入实时价格)
        ok, reason = self.risk.can_open(sig.symbol, alloc, current_prices=prices)
        if not ok:
            log.info("风控拒绝 %s: %s", sig.symbol, reason)
            return

        tid = sig.trace_id
        self._emit_event("order_attempt", {
            "symbol": sig.symbol,
            "side": "BUY",
            "intent": "open",
            "quote_amount": round(float(alloc), 8),
            "hint_price": round(float(current_price), 8),
            "reason": sig.reason,
            "strength": int(sig.strength),
            "confidence": float(sig.confidence),
            "ts": time.time(),
        }, trace_id=tid)
        result = self.executor.spot_market_buy(sig.symbol, alloc, hint_price=current_price)
        self._emit_event("order_result", {
            "symbol": sig.symbol,
            "side": "BUY",
            "intent": "open",
            "ok": not bool(result.get("error")),
            "error": result.get("error"),
            "error_class": result.get("error_class", ""),
            "code": result.get("code"),
            "precheck_code": result.get("precheck_code"),
            "status": result.get("status"),
            "ts": time.time(),
        }, trace_id=tid)
        if result.get("error"):
            log.error("开仓下单失败 %s: %s", sig.symbol, result.get("error"))
            return

        price, qty = self._extract_fill_price_qty(sig.symbol, alloc, result)

        if (not self.config.execution.use_paper_trading
                and str(result.get("status", "")).upper() not in ("FILLED", "PARTIALLY_FILLED")
                and qty <= 0):
            log.error("开仓未成交 %s: status=%s result=%s",
                      sig.symbol, result.get("status"), result)
            return

        if price > 0 and qty > 0:
            self.risk.open_position(sig.symbol, "BUY", price, qty)
            self.pool.update_status(sig.symbol, "trading")
            log.info("开仓 %s %s $%.2f @ $%.6f (heat=%.0f)",
                     sig.action, sig.symbol, alloc, price,
                     heat_score)
        else:
            log.error("开仓结果异常 %s: price=%.6f qty=%.8f result=%s",
                      sig.symbol, price, qty, result)

    @staticmethod
    def _to_float(v, default=0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default

    def _extract_fill_price_qty(self, symbol: str, alloc: float, result: dict) -> tuple:
        """从订单回执中尽量还原真实成交价和成交量。"""
        qty = self._to_float(result.get("qty"), 0.0)
        if qty <= 0:
            qty = self._to_float(result.get("executedQty"), 0.0)

        price = self._to_float(result.get("price"), 0.0)

        fills = result.get("fills")
        if price <= 0 and isinstance(fills, list) and fills:
            total_qty = 0.0
            total_quote = 0.0
            for f in fills:
                p = self._to_float(f.get("price"), 0.0)
                q = self._to_float(f.get("qty"), 0.0)
                if p > 0 and q > 0:
                    total_qty += q
                    total_quote += p * q
            if total_qty > 0:
                if qty <= 0:
                    qty = total_qty
                price = total_quote / total_qty

        if price <= 0 and qty > 0:
            cum_quote = self._to_float(result.get("cummulativeQuoteQty"), 0.0)
            if cum_quote > 0:
                price = cum_quote / qty

        if price <= 0:
            price = self.executor.get_current_price(symbol)
        if qty <= 0 and price > 0:
            qty = alloc / price if alloc > 0 else 0.0

        return price, qty

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

        self._emit_event("order_attempt", {
            "symbol": symbol,
            "side": "SELL",
            "intent": "close",
            "qty": round(float(pos.qty), 8),
            "hint_price": round(float(price), 8),
            "reason": reason,
            "ts": time.time(),
        })
        result = self.executor.spot_market_sell(symbol, pos.qty, hint_price=price)
        self._emit_event("order_result", {
            "symbol": symbol,
            "side": "SELL",
            "intent": "close",
            "ok": not bool(result.get("error")),
            "error": result.get("error"),
            "error_class": result.get("error_class", ""),
            "code": result.get("code"),
            "precheck_code": result.get("precheck_code"),
            "status": result.get("status"),
            "reason": reason,
            "ts": time.time(),
        })
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
        self._emit_event("order_attempt", {
            "symbol": symbol,
            "side": "SELL",
            "intent": "partial_close",
            "qty": round(float(close_qty), 8),
            "hint_price": round(float(price), 8),
            "reason": reason,
            "ts": time.time(),
        })
        result = self.executor.spot_market_sell(symbol, close_qty, hint_price=price)
        self._emit_event("order_result", {
            "symbol": symbol,
            "side": "SELL",
            "intent": "partial_close",
            "ok": not bool(result.get("error")),
            "error": result.get("error"),
            "error_class": result.get("error_class", ""),
            "code": result.get("code"),
            "precheck_code": result.get("precheck_code"),
            "status": result.get("status"),
            "reason": reason,
            "ts": time.time(),
        })
        if result.get("error") and not self.config.execution.use_paper_trading:
            log.error("部分平仓下单失败 %s: %s — 保留仓位", symbol, result.get("error"))
            return

        pnl = self.risk.partial_close(symbol, price, pct, reason)
        self.pnl.record_trade(
            symbol=symbol, side=pos.side,
            entry_price=pos.entry_price, exit_price=price,
            qty=close_qty, entry_time=pos.entry_time,
            reason=reason,
        )

    def reconcile_open_orders(self) -> dict:
        """
        查询 Binance 未成交订单, 与内部持仓对账。
        返回 {stale_orders: int, canceled: int} 摘要。
        """
        if self.config.execution.use_paper_trading:
            return {"stale_orders": 0, "canceled": 0}
        try:
            open_orders = self.executor.query_open_orders()
        except Exception:
            log.exception("对账查询失败")
            return {"stale_orders": 0, "canceled": 0, "error": "query_failed"}

        stale = 0
        canceled = 0
        now = time.time()
        for order in open_orders:
            sym = order.get("symbol", "")
            order_time_ms = int(order.get("time", 0))
            age_sec = (now * 1000 - order_time_ms) / 1000 if order_time_ms else 0
            # 超过 5 分钟未成交的 LIMIT 订单视为过期
            if age_sec > 300 and order.get("type") == "LIMIT":
                oid = int(order.get("orderId", 0))
                log.warning("过期 LIMIT 订单 %s #%d (%.0fs), 尝试取消", sym, oid, age_sec)
                result = self.executor.cancel_order(sym, oid)
                if not result.get("error"):
                    canceled += 1
                stale += 1

        if stale:
            log.info("对账: %d 过期订单, %d 已取消", stale, canceled)
        return {"stale_orders": stale, "canceled": canceled}

    def get_status(self) -> dict:
        """返回引擎状态摘要。"""
        return {
            "positions": self.risk.num_positions,
            "risk_summary": self.risk.get_summary(),
            "pnl_summary": self.pnl.get_summary(),
            "paper": self.config.execution.use_paper_trading,
        }
