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
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
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
    min_notional: float         # 最小下单金额
    min_qty: float              # 最小下单数量
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
                min_qty = 0.0
                min_notional = 10.0

                for f in sym_info.get("filters", []):
                    ft = f.get("filterType", "")
                    if ft == "LOT_SIZE":
                        step = float(f.get("stepSize", "0.00000001"))
                        qty_prec = self._precision_from_step(step)
                        min_qty = float(f.get("minQty", "0"))
                    elif ft == "PRICE_FILTER":
                        tick = float(f.get("tickSize", "0.00000001"))
                        price_prec = self._precision_from_step(tick)
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
                    min_notional=min_notional,
                    min_qty=min_qty,
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

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.paper = config.use_paper_trading
        self._exchange_info = ExchangeInfoCache()
        self._api_key = os.environ.get("BINANCE_API_KEY", "")
        self._api_secret = os.environ.get("BINANCE_SECRET_KEY", "")
        self._order_history: list = []
        self._max_history = 1000

    def format_quantity(self, symbol: str, qty: float) -> str:
        """根据 exchangeInfo 格式化数量。"""
        info = self._exchange_info.get(symbol)
        prec = info.qty_precision if info else 6
        formatted = f"{qty:.{prec}f}"
        return formatted

    def format_price(self, symbol: str, price: float) -> str:
        """根据 exchangeInfo 格式化价格。"""
        info = self._exchange_info.get(symbol)
        prec = info.price_precision if info else 2
        return f"{price:.{prec}f}"

    def format_quote_quantity(self, symbol: str, quote_qty: float) -> str:
        """格式化 quoteOrderQty (USDT 金额), 与价格精度解耦。"""
        info = self._exchange_info.get(symbol)
        prec = info.quote_precision if info else 2
        prec = max(0, min(int(prec), 8))
        return f"{quote_qty:.{prec}f}"

    def get_min_notional(self, symbol: str) -> float:
        info = self._exchange_info.get(symbol)
        return info.min_notional if info else 10.0

    def spot_market_buy(self, symbol: str, quote_qty: float) -> dict:
        """现货市价买入 (按 USDT 金额)。"""
        if self.paper:
            return self._paper_order(symbol, "BUY", quote_qty=quote_qty)

        params = {
            "symbol": symbol,
            "side": "BUY",
            "type": "MARKET",
            "quoteOrderQty": self.format_quote_quantity(symbol, quote_qty),
        }
        return self._send_spot_order(params)

    def spot_market_sell(self, symbol: str, qty: float) -> dict:
        """现货市价卖出 (按数量)。"""
        if self.paper:
            return self._paper_order(symbol, "SELL", qty=qty)

        params = {
            "symbol": symbol,
            "side": "SELL",
            "type": "MARKET",
            "quantity": self.format_quantity(symbol, qty),
        }
        return self._send_spot_order(params)

    def spot_limit_buy(self, symbol: str, qty: float, price: float) -> dict:
        """现货限价买入。"""
        if self.paper:
            return self._paper_order(symbol, "BUY", qty=qty, price=price)

        params = {
            "symbol": symbol,
            "side": "BUY",
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": self.format_quantity(symbol, qty),
            "price": self.format_price(symbol, price),
        }
        return self._send_spot_order(params)

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
            else:
                log.error("下单失败: %s", result)
                result["error"] = result.get("msg", f"HTTP {resp.status_code}")
            return result
        except Exception as e:
            log.exception("下单异常")
            return {"error": str(e)}

    def _paper_order(self, symbol: str, side: str,
                     qty: float = 0, quote_qty: float = 0,
                     price: float = 0) -> dict:
        """纸面交易记录。"""
        current_price = self.get_current_price(symbol) or price
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
