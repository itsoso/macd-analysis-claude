"""
统一下单模块 — 动态精度 + 限价优先

通过 exchangeInfo 缓存每个交易对的 stepSize / tickSize / minNotional,
下单时自动格式化数量和价格精度。

支持模式:
  - Paper Trading (默认): 仅记录, 不调用 API
  - Live Trading: 调用币安 Spot REST API
"""

import hashlib
import hmac
import logging
import math
import os
import threading
import time
from collections import deque
from decimal import Decimal, ROUND_DOWN
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlencode

import requests

from hotcoin.config import ExecutionConfig, BASE_REST_URL

log = logging.getLogger("hotcoin.order")


@dataclass
class SymbolInfo:
    """交易对精度信息, 来自 exchangeInfo。"""
    symbol: str
    base_asset: str
    quote_asset: str
    qty_precision: int          # stepSize 推导的小数位数
    price_precision: int        # tickSize 推导的小数位数
    quote_precision: int        # quoteAssetPrecision
    qty_step: float             # LOT_SIZE.stepSize
    market_qty_step: float      # MARKET_LOT_SIZE.stepSize (市价单)
    price_tick: float           # PRICE_FILTER.tickSize
    min_notional: float         # 最小下单金额
    min_qty: float              # 最小下单数量
    market_min_qty: float       # MARKET_LOT_SIZE.minQty
    status: str = "TRADING"


class ExchangeInfoCache:
    """exchangeInfo 缓存, 每小时刷新。"""

    REFRESH_INTERVAL = 3600  # 1h

    def __init__(self):
        self._cache: Dict[str, SymbolInfo] = {}
        self._last_refresh = 0.0
        self._lock = threading.Lock()

    def get(self, symbol: str) -> Optional[SymbolInfo]:
        if time.time() - self._last_refresh > self.REFRESH_INTERVAL:
            with self._lock:
                if time.time() - self._last_refresh > self.REFRESH_INTERVAL:
                    self._refresh()
        return self._cache.get(symbol)

    def _refresh(self):
        try:
            resp = requests.get(
                f"{BASE_REST_URL}/api/v3/exchangeInfo",
                timeout=15,
            )
            if resp.status_code != 200:
                log.warning("exchangeInfo 返回 %d", resp.status_code)
                return

            data = resp.json()
            for sym_info in data.get("symbols", []):
                sym = sym_info.get("symbol", "")
                if not sym.endswith("USDT"):
                    continue

                qty_prec = 8
                price_prec = 8
                qty_step = 0.0
                market_qty_step = 0.0
                price_tick = 0.0
                min_qty = 0.0
                market_min_qty = 0.0
                min_notional = 10.0

                for f in sym_info.get("filters", []):
                    ft = f.get("filterType", "")
                    if ft == "LOT_SIZE":
                        step = float(f.get("stepSize", "0.00000001"))
                        qty_prec = self._precision_from_step(step)
                        qty_step = step
                        min_qty = float(f.get("minQty", "0"))
                    elif ft == "MARKET_LOT_SIZE":
                        step = float(f.get("stepSize", "0.00000001"))
                        market_qty_step = step if step > 0 else 0.0
                        market_min_qty = float(f.get("minQty", "0"))
                    elif ft == "PRICE_FILTER":
                        tick = float(f.get("tickSize", "0.00000001"))
                        price_prec = self._precision_from_step(tick)
                        price_tick = tick
                    elif ft == "NOTIONAL":
                        min_notional = float(f.get("minNotional", "10"))
                    elif ft == "MIN_NOTIONAL":
                        min_notional = float(f.get("minNotional", "10"))

                self._cache[sym] = SymbolInfo(
                    symbol=sym,
                    base_asset=sym_info.get("baseAsset", ""),
                    quote_asset=sym_info.get("quoteAsset", "USDT"),
                    qty_precision=qty_prec,
                    price_precision=price_prec,
                    quote_precision=int(sym_info.get("quoteAssetPrecision", 8)),
                    qty_step=qty_step,
                    market_qty_step=market_qty_step,
                    price_tick=price_tick,
                    min_notional=min_notional,
                    min_qty=min_qty,
                    market_min_qty=market_min_qty,
                    status=sym_info.get("status", "TRADING"),
                )

            self._last_refresh = time.time()
            log.info("exchangeInfo 刷新: %d 个 USDT 交易对", len(self._cache))
        except Exception:
            log.exception("exchangeInfo 刷新失败")

    @staticmethod
    def _precision_from_step(step: float) -> int:
        if step <= 0:
            return 8
        s = f"{step:.10f}".rstrip("0")
        if "." not in s:
            return 0
        return len(s.split(".")[1])


class OrderExecutor:
    """
    统一下单执行器。

    paper=True 时仅记录, 不调用 API。
    """

    DEDUP_WINDOW_SEC = 60  # 防重复下单窗口
    MAX_PRECHECK_SYMBOLS = 500
    METRICS_WINDOW_SEC = 300
    METRICS_EVENT_LIMIT = 5000

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.paper = config.use_paper_trading
        self._exchange_info = ExchangeInfoCache()
        self._api_key = os.environ.get("BINANCE_API_KEY", "")
        self._api_secret = os.environ.get("BINANCE_SECRET_KEY", "")
        self._order_history: list = []
        self._max_history = 1000
        self._recent_orders: Dict[str, float] = {}  # symbol+side → timestamp
        self._dedup_lock = threading.Lock()
        self._precheck_failures: Dict[str, int] = {}
        self._precheck_failures_by_symbol: Dict[str, Dict[str, int]] = {}
        self._precheck_symbol_last_seen: Dict[str, float] = {}
        self._runtime_events: Dict[str, deque] = {
            "order_attempt": deque(),
            "order_success": deque(),
            "order_error": deque(),
            "precheck_failed": deque(),
            "dedup_rejected": deque(),
        }

    @staticmethod
    def _quantize_down(value: float, step: float = 0.0, precision: int = 8) -> float:
        """按 step/tick 向下截断；若无 step 则按 precision 截断。"""
        if value <= 0:
            return 0.0
        if step and step > 0:
            d_val = Decimal(str(value))
            d_step = Decimal(str(step))
            units = (d_val / d_step).to_integral_value(rounding=ROUND_DOWN)
            return float(units * d_step)
        precision = max(0, min(int(precision), 12))
        factor = 10 ** precision
        return math.floor(value * factor) / factor

    def _record_precheck_failure(self, symbol: str, precheck_code: str):
        self._precheck_failures[precheck_code] = self._precheck_failures.get(precheck_code, 0) + 1
        sym_stat = self._precheck_failures_by_symbol.setdefault(symbol, {})
        sym_stat[precheck_code] = sym_stat.get(precheck_code, 0) + 1
        self._precheck_symbol_last_seen[symbol] = time.time()
        self._record_runtime_event("precheck_failed")
        self._prune_precheck_symbol_stats()

    def _prune_precheck_symbol_stats(self):
        if len(self._precheck_failures_by_symbol) <= self.MAX_PRECHECK_SYMBOLS:
            return
        overflow = len(self._precheck_failures_by_symbol) - self.MAX_PRECHECK_SYMBOLS
        for sym, _ in sorted(self._precheck_symbol_last_seen.items(), key=lambda kv: kv[1])[:overflow]:
            self._precheck_failures_by_symbol.pop(sym, None)
            self._precheck_symbol_last_seen.pop(sym, None)

    def _record_runtime_event(self, event_type: str, ts: Optional[float] = None):
        q = self._runtime_events.get(event_type)
        if q is None:
            return
        now = ts if ts is not None else time.time()
        q.append(now)
        cutoff = now - self.METRICS_WINDOW_SEC
        while q and q[0] < cutoff:
            q.popleft()
        while len(q) > self.METRICS_EVENT_LIMIT:
            q.popleft()

    def _count_runtime_events(self, event_type: str, window_sec: int, now: Optional[float] = None) -> int:
        q = self._runtime_events.get(event_type)
        if not q:
            return 0
        current = now if now is not None else time.time()
        cutoff = current - max(1, int(window_sec))
        while q and q[0] < cutoff:
            q.popleft()
        return len(q)

    def get_runtime_metrics(self, window_sec: int = 300) -> dict:
        now = time.time()
        attempts = self._count_runtime_events("order_attempt", window_sec, now)
        precheck_failed = self._count_runtime_events("precheck_failed", window_sec, now)
        dedup_rejected = self._count_runtime_events("dedup_rejected", window_sec, now)
        order_error = self._count_runtime_events("order_error", window_sec, now)
        order_success = self._count_runtime_events("order_success", window_sec, now)
        precheck_fail_rate = (precheck_failed / attempts) if attempts > 0 else 0.0
        order_error_rate = (order_error / attempts) if attempts > 0 else 0.0
        return {
            "window_sec": int(window_sec),
            "order_attempts_5m": attempts,
            "precheck_failures_5m": precheck_failed,
            "dedup_rejects_5m": dedup_rejected,
            "order_errors_5m": order_error,
            "order_success_5m": order_success,
            "precheck_fail_rate_5m": round(precheck_fail_rate, 4),
            "order_error_rate_5m": round(order_error_rate, 4),
        }

    def get_precheck_stats(self) -> dict:
        total = sum(self._precheck_failures.values())
        return {
            "total": total,
            "by_code": dict(self._precheck_failures),
            "by_symbol": {k: dict(v) for k, v in self._precheck_failures_by_symbol.items()},
        }

    def _precheck_failed(self, symbol: str, precheck_code: str, message: str) -> dict:
        self._record_precheck_failure(symbol, precheck_code)
        return {
            "code": "PRECHECK_FAILED",
            "precheck_code": precheck_code,
            "symbol": symbol,
            "error": message,
        }

    def _precheck_buy_quote(self, symbol: str, quote_qty: float) -> Optional[tuple[str, str]]:
        if quote_qty <= 0:
            return "BUY_INVALID_QUOTE_QTY", "quoteOrderQty 必须 > 0"
        info = self._exchange_info.get(symbol)
        min_notional = float(getattr(info, "min_notional", 10.0)) if info else 10.0
        if quote_qty + 1e-12 < min_notional:
            return "BUY_MIN_NOTIONAL", (
                f"quoteOrderQty 低于最小名义金额 ({quote_qty:.8f} < {min_notional:.8f})"
            )
        return None

    def _precheck_sell_qty(
        self, symbol: str, qty: float, est_price: float = 0.0, market: bool = True
    ) -> Optional[tuple[str, str]]:
        if qty <= 0:
            return "SELL_INVALID_QTY", "quantity 必须 > 0"
        info = self._exchange_info.get(symbol)
        if info:
            min_qty = float(getattr(info, "min_qty", 0.0))
            if market:
                market_min_qty = float(getattr(info, "market_min_qty", 0.0))
                if market_min_qty > 0:
                    min_qty = market_min_qty
            if qty + 1e-12 < min_qty:
                return "SELL_MIN_QTY", (
                    f"quantity 低于最小下单数量 ({qty:.12f} < {min_qty:.12f})"
                )
            min_notional = float(getattr(info, "min_notional", 10.0))
            if est_price > 0 and (qty * est_price) + 1e-12 < min_notional:
                return "SELL_MIN_NOTIONAL", (
                    "quantity*price 低于最小名义金额 "
                    f"({qty * est_price:.8f} < {min_notional:.8f})"
                )
        return None

    def _precheck_limit_buy(self, symbol: str, qty: float, price: float) -> Optional[tuple[str, str]]:
        if qty <= 0:
            return "LIMIT_BUY_INVALID_QTY", "quantity 必须 > 0"
        if price <= 0:
            return "LIMIT_BUY_INVALID_PRICE", "price 必须 > 0"
        info = self._exchange_info.get(symbol)
        if info:
            min_qty = float(getattr(info, "min_qty", 0.0))
            if qty + 1e-12 < min_qty:
                return "LIMIT_BUY_MIN_QTY", (
                    f"quantity 低于最小下单数量 ({qty:.12f} < {min_qty:.12f})"
                )
            min_notional = float(getattr(info, "min_notional", 10.0))
            notional = qty * price
            if notional + 1e-12 < min_notional:
                return "LIMIT_BUY_MIN_NOTIONAL", (
                    f"limit notional 低于最小名义金额 ({notional:.8f} < {min_notional:.8f})"
                )
        return None

    def format_quantity(self, symbol: str, qty: float, market: bool = False) -> str:
        """根据 exchangeInfo 格式化数量。market=True 时优先 MARKET_LOT_SIZE。"""
        info = self._exchange_info.get(symbol)
        prec = int(getattr(info, "qty_precision", 6)) if info else 6
        step = float(getattr(info, "qty_step", 0.0)) if info else 0.0
        if info and market:
            market_step = float(getattr(info, "market_qty_step", 0.0))
            if market_step > 0:
                step = market_step
                prec = ExchangeInfoCache._precision_from_step(market_step)
        truncated = self._quantize_down(qty, step=step, precision=prec)
        formatted = f"{truncated:.{max(0, prec)}f}"
        return formatted

    def format_price(self, symbol: str, price: float) -> str:
        """根据 exchangeInfo 格式化价格。"""
        info = self._exchange_info.get(symbol)
        prec = info.price_precision if info else 2
        tick = info.price_tick if info else 0.0
        truncated = self._quantize_down(price, step=tick, precision=prec)
        return f"{truncated:.{prec}f}"

    def format_quote_quantity(self, symbol: str, quote_qty: float) -> str:
        """格式化 quoteOrderQty (USDT 金额), 与价格精度解耦。"""
        info = self._exchange_info.get(symbol)
        prec = info.quote_precision if info else 2
        prec = max(0, min(int(prec), 8))
        truncated = self._quantize_down(quote_qty, precision=prec)
        return f"{truncated:.{prec}f}"

    def get_min_notional(self, symbol: str) -> float:
        info = self._exchange_info.get(symbol)
        return info.min_notional if info else 10.0

    def _check_dedup(self, symbol: str, side: str) -> Optional[str]:
        """防重复下单(原子预留): 同一 symbol+side 在窗口内不允许重复。"""
        key = f"{symbol}:{side}"
        now = time.time()
        with self._dedup_lock:
            expired = [k for k, t in self._recent_orders.items() if now - t > self.DEDUP_WINDOW_SEC]
            for k in expired:
                del self._recent_orders[k]
            last = self._recent_orders.get(key, 0)
            if now - last < self.DEDUP_WINDOW_SEC:
                return f"防重复: {symbol} {side} 距上次下单仅 {now - last:.0f}s (窗口 {self.DEDUP_WINDOW_SEC}s)"
            self._recent_orders[key] = now
        return None

    def _rollback_dedup(self, symbol: str, side: str):
        """下单失败/预检失败时释放预留，允许立即重试。"""
        with self._dedup_lock:
            self._recent_orders.pop(f"{symbol}:{side}", None)

    def spot_market_buy(self, symbol: str, quote_qty: float, hint_price: float = 0) -> dict:
        """现货市价买入 (按 USDT 金额)。hint_price: 可选, 避免 paper 模式额外请求。"""
        dedup_err = self._check_dedup(symbol, "BUY")
        if dedup_err:
            self._record_runtime_event("dedup_rejected")
            return {"error": dedup_err, "code": "DEDUP_REJECTED", "symbol": symbol}
        self._record_runtime_event("order_attempt")

        quote_qty_s = self.format_quote_quantity(symbol, quote_qty)
        quote_qty_v = float(quote_qty_s)
        precheck_err = self._precheck_buy_quote(symbol, quote_qty_v)
        if precheck_err:
            self._rollback_dedup(symbol, "BUY")
            return self._precheck_failed(symbol, precheck_err[0], precheck_err[1])

        if self.paper:
            result = self._paper_order(symbol, "BUY", quote_qty=quote_qty_v, price=hint_price)
            if result.get("error"):
                self._rollback_dedup(symbol, "BUY")
                self._record_runtime_event("order_error")
            else:
                self._record_runtime_event("order_success")
            return result

        params = {
            "symbol": symbol,
            "side": "BUY",
            "type": "MARKET",
            "quoteOrderQty": quote_qty_s,
        }
        result = self._send_spot_order(params)
        if result.get("error"):
            self._rollback_dedup(symbol, "BUY")
        return result

    def spot_market_sell(self, symbol: str, qty: float, hint_price: float = 0) -> dict:
        """现货市价卖出 (按数量)。hint_price: 可选, 避免 paper 模式额外请求。"""
        dedup_err = self._check_dedup(symbol, "SELL")
        if dedup_err:
            self._record_runtime_event("dedup_rejected")
            return {"error": dedup_err, "code": "DEDUP_REJECTED", "symbol": symbol}
        self._record_runtime_event("order_attempt")

        qty_s = self.format_quantity(symbol, qty, market=True)
        qty_v = float(qty_s)
        if hint_price > 0:
            est_price = hint_price
        else:
            est_price = self.get_avg_price(symbol) or self.get_current_price(symbol)
        precheck_err = self._precheck_sell_qty(symbol, qty_v, est_price=est_price, market=True)
        if precheck_err:
            self._rollback_dedup(symbol, "SELL")
            return self._precheck_failed(symbol, precheck_err[0], precheck_err[1])

        if self.paper:
            result = self._paper_order(symbol, "SELL", qty=qty_v, price=hint_price)
            if result.get("error"):
                self._rollback_dedup(symbol, "SELL")
                self._record_runtime_event("order_error")
            else:
                self._record_runtime_event("order_success")
            return result

        params = {
            "symbol": symbol,
            "side": "SELL",
            "type": "MARKET",
            "quantity": qty_s,
        }
        result = self._send_spot_order(params)
        if result.get("error"):
            self._rollback_dedup(symbol, "SELL")
        return result

    def spot_limit_buy(self, symbol: str, qty: float, price: float) -> dict:
        """现货限价买入。"""
        dedup_err = self._check_dedup(symbol, "BUY")
        if dedup_err:
            self._record_runtime_event("dedup_rejected")
            return {"error": dedup_err, "code": "DEDUP_REJECTED", "symbol": symbol}
        self._record_runtime_event("order_attempt")

        qty_s = self.format_quantity(symbol, qty)
        qty_v = float(qty_s)
        price_s = self.format_price(symbol, price)
        price_v = float(price_s)

        precheck_err = self._precheck_limit_buy(symbol, qty_v, price_v)
        if precheck_err:
            self._rollback_dedup(symbol, "BUY")
            return self._precheck_failed(symbol, precheck_err[0], precheck_err[1])

        if self.paper:
            result = self._paper_order(symbol, "BUY", qty=qty_v, price=price_v)
            if result.get("error"):
                self._rollback_dedup(symbol, "BUY")
                self._record_runtime_event("order_error")
            else:
                self._record_runtime_event("order_success")
            return result

        params = {
            "symbol": symbol,
            "side": "BUY",
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": qty_s,
            "price": price_s,
        }
        result = self._send_spot_order(params)
        if result.get("error"):
            self._rollback_dedup(symbol, "BUY")
        return result

    def get_current_price(self, symbol: str) -> float:
        """获取最新价格。"""
        try:
            resp = requests.get(
                f"{BASE_REST_URL}/api/v3/ticker/price",
                params={"symbol": symbol},
                timeout=5,
            )
            if resp.status_code == 200:
                return float(resp.json().get("price", 0))
        except Exception:
            pass
        return 0.0

    def get_avg_price(self, symbol: str) -> float:
        """获取 Binance avgPrice（近 5 分钟成交均价）用于名义金额预检。"""
        try:
            resp = requests.get(
                f"{BASE_REST_URL}/api/v3/avgPrice",
                params={"symbol": symbol},
                timeout=5,
            )
            if resp.status_code == 200:
                return float(resp.json().get("price", 0))
        except Exception:
            pass
        return 0.0

    def _send_spot_order(self, params: dict) -> dict:
        """发送现货订单到币安 API。"""
        url = f"{BASE_REST_URL}/api/v3/order"
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000

        query = urlencode(sorted(params.items()))
        signature = hmac.new(
            self._api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()

        headers = {"X-MBX-APIKEY": self._api_key}
        try:
            resp = requests.post(
                url, params=query + "&signature=" + signature,
                headers=headers, timeout=10,
            )
            result = resp.json()
            if resp.status_code == 200:
                log.info("下单成功: %s %s %s", params.get("symbol"),
                         params.get("side"), params.get("type"))
                self._order_history.append(result)
                self._record_runtime_event("order_success")
            else:
                log.error("下单失败: %s", result)
                result["error"] = result.get("msg", f"HTTP {resp.status_code}")
                self._record_runtime_event("order_error")
            return result
        except Exception as e:
            log.exception("下单异常")
            self._record_runtime_event("order_error")
            return {"error": str(e)}

    def _paper_order(self, symbol: str, side: str,
                     qty: float = 0, quote_qty: float = 0,
                     price: float = 0) -> dict:
        """纸面交易记录。"""
        current_price = price or self.get_current_price(symbol)
        if not qty and quote_qty and current_price:
            qty = quote_qty / current_price

        order = {
            "symbol": symbol,
            "side": side,
            "type": "PAPER",
            "price": current_price,
            "qty": qty,
            "quote_qty": quote_qty,
            "time": time.time(),
        }
        self._order_history.append(order)
        if len(self._order_history) > self._max_history:
            self._order_history = self._order_history[-self._max_history:]
        log.info("[PAPER] %s %s qty=%.6f @ $%.4f", side, symbol, qty, current_price)
        return order
