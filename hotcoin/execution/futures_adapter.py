"""
合约交易适配器

对有永续合约的币种提供合约交易能力。
复用现有 order_manager.py 的 BinanceOrderManager (合约端点)。
默认不启用, 需显式配置。
"""

import logging
import os
from typing import Optional

log = logging.getLogger("hotcoin.futures")


class FuturesAdapter:
    """
    合约适配层 — 仅对支持合约的币种可用。

    桥接 hotcoin 系统和现有 BinanceOrderManager (合约)。
    """

    def __init__(self, leverage: int = 2, use_paper: bool = True):
        self.leverage = leverage
        self.use_paper = use_paper
        self._order_manager = None
        self._futures_symbols: set = set()
        self._leverage_set: set = set()  # 已设置杠杆的交易对

    def init(self):
        """延迟初始化, 仅在需要时加载。"""
        if self._order_manager is not None:
            return

        try:
            import sys, os
            # 确保项目根目录在 sys.path 中
            root = os.path.join(os.path.dirname(__file__), "..", "..")
            if root not in sys.path:
                sys.path.insert(0, root)
            from live_config import TradingPhase, APIConfig
            from order_manager import BinanceOrderManager

            api_key = os.environ.get("BINANCE_API_KEY", "")
            api_secret = os.environ.get("BINANCE_SECRET_KEY", "")

            if not api_key or not api_secret:
                log.warning("合约 API 密钥未配置")
                return

            self._order_manager = BinanceOrderManager(
                api_key=api_key,
                api_secret=api_secret,
                phase=TradingPhase.TESTNET if self.use_paper else TradingPhase.MAINNET,
            )
            log.info("合约适配器初始化完成 (leverage=%d, paper=%s)",
                     self.leverage, self.use_paper)
        except ImportError as e:
            log.warning("合约依赖导入失败: %s", e)

    def has_futures(self, symbol: str) -> bool:
        """检查是否支持永续合约。"""
        if not self._futures_symbols:
            self._load_futures_symbols()
        return symbol in self._futures_symbols

    def _load_futures_symbols(self):
        """加载支持合约的交易对。"""
        try:
            import requests
            resp = requests.get(
                "https://fapi.binance.com/fapi/v1/exchangeInfo",
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                for sym_info in data.get("symbols", []):
                    if sym_info.get("status") == "TRADING":
                        self._futures_symbols.add(sym_info["symbol"])
                log.info("合约交易对: %d 个", len(self._futures_symbols))
        except Exception as e:
            log.warning("获取合约交易对失败: %s", e)

    def _usdt_to_quantity(self, symbol: str, usdt_amount: float) -> float:
        """将 USDT 金额转为合约数量 (含杠杆)。"""
        try:
            import requests
            resp = requests.get(
                "https://fapi.binance.com/fapi/v1/ticker/price",
                params={"symbol": symbol}, timeout=5,
            )
            if resp.status_code == 200:
                price = float(resp.json().get("price", 0))
                if price > 0:
                    return (usdt_amount * self.leverage) / price
        except Exception as e:
            log.warning("获取 %s 合约价格失败: %s", symbol, e)
        return 0.0

    def open_long(self, symbol: str, usdt_amount: float) -> Optional[dict]:
        """开多 (传入 USDT 金额, 自动换算为合约数量)。"""
        self.init()
        if not self._order_manager:
            return None

        if self.use_paper:
            log.info("[PAPER-FUTURES] LONG %s $%.2f lev=%d", symbol, usdt_amount, self.leverage)
            return {"paper": True, "side": "BUY", "symbol": symbol, "amount": usdt_amount}

        try:
            qty = self._usdt_to_quantity(symbol, usdt_amount)
            if qty <= 0:
                log.error("合约开多 %s: 数量计算失败", symbol)
                return None
            self._ensure_leverage(symbol)
            result = self._order_manager._place_order(symbol, "BUY", qty, reduce_only=False)
            return result
        except Exception as e:
            log.error("合约开多失败 %s: %s", symbol, e)
            return None

    def open_short(self, symbol: str, usdt_amount: float) -> Optional[dict]:
        """开空 (传入 USDT 金额, 自动换算为合约数量)。"""
        self.init()
        if not self._order_manager:
            return None

        if self.use_paper:
            log.info("[PAPER-FUTURES] SHORT %s $%.2f lev=%d", symbol, usdt_amount, self.leverage)
            return {"paper": True, "side": "SELL", "symbol": symbol, "amount": usdt_amount}

        try:
            qty = self._usdt_to_quantity(symbol, usdt_amount)
            if qty <= 0:
                log.error("合约开空 %s: 数量计算失败", symbol)
                return None
            self._ensure_leverage(symbol)
            result = self._order_manager._place_order(symbol, "SELL", qty, reduce_only=False)
            return result
        except Exception as e:
            log.error("合约开空失败 %s: %s", symbol, e)
            return None

    def _ensure_leverage(self, symbol: str):
        """设置杠杆 (每个交易对仅设置一次)。"""
        if symbol not in self._leverage_set:
            self._order_manager.set_leverage(symbol, self.leverage)
            self._leverage_set.add(symbol)

    def close_position(self, symbol: str, side: str, qty: float) -> Optional[dict]:
        """平仓。"""
        self.init()
        if not self._order_manager:
            return None

        close_side = "SELL" if side == "BUY" else "BUY"
        if self.use_paper:
            log.info("[PAPER-FUTURES] CLOSE %s %s qty=%.6f", close_side, symbol, qty)
            return {"paper": True, "side": close_side, "symbol": symbol, "qty": qty}

        try:
            result = self._order_manager.market_order(symbol, close_side, qty, reduce_only=True)
            return result
        except Exception as e:
            log.error("合约平仓失败 %s: %s", symbol, e)
            return None
