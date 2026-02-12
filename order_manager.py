"""
Binance 订单管理器
支持: 主网/测试网 / 合约交易 / 订单生命周期管理
"""

import hashlib
import hmac
import json
import os
import time
from datetime import datetime
from typing import Optional, Dict, List
from urllib.parse import urlencode

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from live_config import TradingPhase, APIConfig


class BinanceOrderManager:
    """Binance 合约订单管理器"""

    # API 端点
    ENDPOINTS = {
        "account": "/fapi/v2/account",
        "balance": "/fapi/v2/balance",
        "position": "/fapi/v2/positionRisk",
        "order": "/fapi/v1/order",
        "open_orders": "/fapi/v1/openOrders",
        "all_orders": "/fapi/v1/allOrders",
        "leverage": "/fapi/v1/leverage",
        "margin_type": "/fapi/v1/marginType",
        "klines": "/fapi/v1/klines",
        "ticker_price": "/fapi/v1/ticker/price",
        "exchange_info": "/fapi/v1/exchangeInfo",
        "funding_rate": "/fapi/v1/fundingRate",
        "income": "/fapi/v1/income",
    }

    def __init__(self, api_config: APIConfig, phase: TradingPhase,
                 logger=None, notifier=None):
        self.phase = phase
        self.logger = logger
        self.notifier = notifier

        # 获取对应阶段的凭据和端点
        self.api_key, self.api_secret = api_config.get_credentials(phase)
        self.base_url = api_config.get_base_url(phase)

        self.session = requests.Session() if HAS_REQUESTS else None
        if self.session:
            self.session.headers.update({
                "X-MBX-APIKEY": self.api_key,
                "Content-Type": "application/x-www-form-urlencoded",
            })

        # 订单追踪
        self._order_history: List[dict] = []
        self._pending_orders: Dict[str, dict] = {}

    # ============================================================
    # 签名 & 请求
    # ============================================================
    def _sign(self, params: dict) -> dict:
        """HMAC SHA256 签名"""
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        query = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params

    def _request(self, method: str, endpoint: str,
                 params: dict = None, signed: bool = True) -> dict:
        """发送 API 请求"""
        if not self.session:
            raise RuntimeError("requests 库未安装")

        url = self.base_url + endpoint
        params = params or {}

        if signed:
            params = self._sign(params)

        try:
            if method == "GET":
                resp = self.session.get(url, params=params, timeout=10)
            elif method == "POST":
                resp = self.session.post(url, data=params, timeout=10)
            elif method == "DELETE":
                resp = self.session.delete(url, params=params, timeout=10)
            else:
                raise ValueError(f"不支持的 HTTP 方法: {method}")

            result = resp.json()

            if resp.status_code != 200:
                error_msg = result.get("msg", str(result))
                error_code = result.get("code", resp.status_code)
                if self.logger:
                    self.logger.error(
                        f"API 错误 [{error_code}] {endpoint}: {error_msg}"
                    )
                raise BinanceAPIError(error_code, error_msg)

            return result

        except requests.exceptions.Timeout:
            if self.logger:
                self.logger.error(f"API 超时: {endpoint}")
            raise
        except requests.exceptions.ConnectionError:
            if self.logger:
                self.logger.error(f"API 连接失败: {endpoint}")
            raise

    # ============================================================
    # 账户信息
    # ============================================================
    def get_account_info(self) -> dict:
        """获取账户信息"""
        return self._request("GET", self.ENDPOINTS["account"])

    def get_balance(self, asset: str = "USDT") -> dict:
        """获取指定资产余额"""
        balances = self._request("GET", self.ENDPOINTS["balance"])
        for b in balances:
            if b["asset"] == asset:
                return {
                    "balance": float(b["balance"]),
                    "available": float(b["availableBalance"]),
                    "unrealized_pnl": float(b.get("crossUnPnl", 0)),
                }
        return {"balance": 0, "available": 0, "unrealized_pnl": 0}

    def get_positions(self, symbol: str = None) -> list:
        """获取持仓信息"""
        params = {}
        if symbol:
            params["symbol"] = symbol
        positions = self._request("GET", self.ENDPOINTS["position"], params)
        # 过滤有仓位的
        active = [p for p in positions if float(p.get("positionAmt", 0)) != 0]
        return active

    # ============================================================
    # 交易设置
    # ============================================================
    def set_leverage(self, symbol: str, leverage: int) -> dict:
        """设置杠杆"""
        return self._request("POST", self.ENDPOINTS["leverage"], {
            "symbol": symbol,
            "leverage": leverage,
        })

    def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> dict:
        """设置保证金模式 (ISOLATED / CROSSED)"""
        try:
            return self._request("POST", self.ENDPOINTS["margin_type"], {
                "symbol": symbol,
                "marginType": margin_type,
            })
        except BinanceAPIError as e:
            # -4046 表示已经是该模式
            if e.code == -4046:
                return {"msg": "already set"}
            raise

    # ============================================================
    # 下单操作
    # ============================================================
    def market_open_long(self, symbol: str, quantity: float,
                         reason: str = "") -> dict:
        """市价开多"""
        return self._place_order(
            symbol=symbol, side="BUY", quantity=quantity,
            order_type="MARKET", reason=reason
        )

    def market_open_short(self, symbol: str, quantity: float,
                          reason: str = "") -> dict:
        """市价开空"""
        return self._place_order(
            symbol=symbol, side="SELL", quantity=quantity,
            order_type="MARKET", reason=reason
        )

    def market_close_long(self, symbol: str, quantity: float,
                          reason: str = "") -> dict:
        """市价平多"""
        return self._place_order(
            symbol=symbol, side="SELL", quantity=quantity,
            order_type="MARKET", reduce_only=True, reason=reason
        )

    def market_close_short(self, symbol: str, quantity: float,
                           reason: str = "") -> dict:
        """市价平空"""
        return self._place_order(
            symbol=symbol, side="BUY", quantity=quantity,
            order_type="MARKET", reduce_only=True, reason=reason
        )

    def place_stop_loss(self, symbol: str, side: str,
                        stop_price: float, quantity: float,
                        reason: str = "") -> dict:
        """设置止损单"""
        order_side = "SELL" if side == "LONG" else "BUY"
        return self._place_order(
            symbol=symbol, side=order_side, quantity=quantity,
            order_type="STOP_MARKET", stop_price=stop_price,
            reduce_only=True, reason=reason
        )

    def place_take_profit(self, symbol: str, side: str,
                          stop_price: float, quantity: float,
                          reason: str = "") -> dict:
        """设置止盈单"""
        order_side = "SELL" if side == "LONG" else "BUY"
        return self._place_order(
            symbol=symbol, side=order_side, quantity=quantity,
            order_type="TAKE_PROFIT_MARKET", stop_price=stop_price,
            reduce_only=True, reason=reason
        )

    def _place_order(self, symbol: str, side: str, quantity: float,
                     order_type: str = "MARKET", price: float = None,
                     stop_price: float = None, reduce_only: bool = False,
                     reason: str = "") -> dict:
        """通用下单"""
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": self._format_quantity(symbol, quantity),
        }

        if price and order_type == "LIMIT":
            params["price"] = f"{price:.2f}"
            params["timeInForce"] = "GTC"

        if stop_price:
            params["stopPrice"] = f"{stop_price:.2f}"

        if reduce_only:
            params["reduceOnly"] = "true"

        if self.logger:
            self.logger.info(
                f"下单: {side} {symbol} qty={quantity:.4f} "
                f"type={order_type} reason={reason}"
            )

        result = self._request("POST", self.ENDPOINTS["order"], params)

        # 记录订单
        order_record = {
            "order_id": result.get("orderId"),
            "client_order_id": result.get("clientOrderId"),
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "price": float(result.get("avgPrice", 0) or result.get("price", 0)),
            "status": result.get("status"),
            "time": datetime.now().isoformat(),
            "reason": reason,
        }
        self._order_history.append(order_record)

        if self.logger:
            self.logger.info(
                f"订单成交: order_id={result.get('orderId')} "
                f"status={result.get('status')} "
                f"avg_price={result.get('avgPrice', 'N/A')}"
            )

        return result

    # ============================================================
    # 订单管理
    # ============================================================
    def cancel_order(self, symbol: str, order_id: int) -> dict:
        """撤销订单"""
        return self._request("DELETE", self.ENDPOINTS["order"], {
            "symbol": symbol,
            "orderId": order_id,
        })

    def cancel_all_orders(self, symbol: str) -> dict:
        """撤销所有挂单"""
        return self._request("DELETE", self.ENDPOINTS["open_orders"], {
            "symbol": symbol,
        })

    def get_order_status(self, symbol: str, order_id: int) -> dict:
        """查询订单状态"""
        return self._request("GET", self.ENDPOINTS["order"], {
            "symbol": symbol,
            "orderId": order_id,
        })

    def get_open_orders(self, symbol: str = None) -> list:
        """获取未成交订单"""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", self.ENDPOINTS["open_orders"], params)

    # ============================================================
    # 一键平仓 (Kill Switch)
    # ============================================================
    def close_all_positions(self, symbol: str = None) -> list:
        """
        紧急平仓所有持仓
        返回: 平仓结果列表
        """
        results = []

        # 1. 撤销所有挂单
        if symbol:
            try:
                self.cancel_all_orders(symbol)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"撤单失败: {e}")

        # 2. 获取所有持仓
        positions = self.get_positions(symbol)

        # 3. 逐个平仓
        for pos in positions:
            pos_symbol = pos["symbol"]
            pos_amt = float(pos["positionAmt"])
            if pos_amt == 0:
                continue

            try:
                if pos_amt > 0:  # 多仓
                    result = self.market_close_long(
                        pos_symbol, abs(pos_amt), "KILL_SWITCH"
                    )
                else:  # 空仓
                    result = self.market_close_short(
                        pos_symbol, abs(pos_amt), "KILL_SWITCH"
                    )
                results.append({"symbol": pos_symbol, "result": result, "error": None})
            except Exception as e:
                results.append({"symbol": pos_symbol, "result": None, "error": str(e)})
                if self.logger:
                    self.logger.error(f"平仓失败 {pos_symbol}: {e}")

        return results

    # ============================================================
    # 市场数据
    # ============================================================
    def get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        result = self._request("GET", self.ENDPOINTS["ticker_price"],
                               {"symbol": symbol}, signed=False)
        return float(result["price"])

    def get_klines(self, symbol: str, interval: str,
                   limit: int = 100) -> list:
        """获取K线数据"""
        result = self._request("GET", self.ENDPOINTS["klines"], {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }, signed=False)
        return result

    def get_funding_rate(self, symbol: str) -> float:
        """获取当前资金费率"""
        result = self._request("GET", self.ENDPOINTS["funding_rate"], {
            "symbol": symbol,
            "limit": 1,
        }, signed=False)
        if result:
            return float(result[0]["fundingRate"])
        return 0

    def get_exchange_info(self, symbol: str = None) -> dict:
        """获取交易规则"""
        result = self._request("GET", self.ENDPOINTS["exchange_info"],
                               signed=False)
        if symbol:
            for s in result.get("symbols", []):
                if s["symbol"] == symbol:
                    return s
        return result

    def get_income_history(self, symbol: str = None,
                           income_type: str = None,
                           limit: int = 100) -> list:
        """获取收入历史 (手续费、资金费率等)"""
        params = {"limit": limit}
        if symbol:
            params["symbol"] = symbol
        if income_type:
            params["incomeType"] = income_type
        return self._request("GET", self.ENDPOINTS["income"], params)

    # ============================================================
    # 辅助方法
    # ============================================================
    def _format_quantity(self, symbol: str, quantity: float) -> str:
        """格式化数量精度 (ETHUSDT 精度为 3)"""
        # ETH 默认精度
        precision = 3
        return f"{quantity:.{precision}f}"

    def get_order_history(self) -> list:
        """获取本地订单历史"""
        return self._order_history.copy()

    def test_connection(self) -> bool:
        """测试 API 连接"""
        try:
            result = self._request("GET", self.ENDPOINTS["ticker_price"],
                                   {"symbol": "ETHUSDT"}, signed=False)
            price = float(result["price"])
            if self.logger:
                self.logger.info(
                    f"API 连接测试成功 [{self.phase.value}] "
                    f"ETH/USDT: ${price:.2f}"
                )
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"API 连接测试失败: {e}")
            return False


class PaperOrderManager:
    """
    纸上交易订单管理器 - 模拟下单，不调用真实 API
    仅获取实时价格用于信号计算
    """

    def __init__(self, api_config: APIConfig = None, phase: TradingPhase = None,
                 logger=None, initial_capital: float = 100000):
        self.logger = logger
        self.initial_capital = initial_capital
        self._usdt = initial_capital
        self._positions = {}  # symbol -> position info
        self._order_counter = 0
        self._order_history = []

        # 如果提供了 API 配置，用于获取实时价格
        self._api_config = api_config
        self._phase = phase
        self._real_api = None
        if api_config:
            try:
                self._real_api = BinanceOrderManager(
                    api_config, phase or TradingPhase.PAPER, logger
                )
            except Exception:
                pass

    def get_current_price(self, symbol: str) -> float:
        """获取当前市场价格"""
        if self._real_api:
            try:
                return self._real_api.get_current_price(symbol)
            except Exception:
                pass
        return 0

    def get_klines(self, symbol: str, interval: str,
                   limit: int = 100) -> list:
        """获取K线数据"""
        if self._real_api:
            return self._real_api.get_klines(symbol, interval, limit)
        return []

    def get_funding_rate(self, symbol: str) -> float:
        if self._real_api:
            try:
                return self._real_api.get_funding_rate(symbol)
            except Exception:
                pass
        return 0

    def market_open_long(self, symbol: str, quantity: float,
                         reason: str = "") -> dict:
        return self._simulate_order(symbol, "BUY", quantity, reason)

    def market_open_short(self, symbol: str, quantity: float,
                          reason: str = "") -> dict:
        return self._simulate_order(symbol, "SELL", quantity, reason)

    def market_close_long(self, symbol: str, quantity: float,
                          reason: str = "") -> dict:
        return self._simulate_order(symbol, "SELL", quantity, reason,
                                    reduce_only=True)

    def market_close_short(self, symbol: str, quantity: float,
                           reason: str = "") -> dict:
        return self._simulate_order(symbol, "BUY", quantity, reason,
                                    reduce_only=True)

    def _simulate_order(self, symbol: str, side: str, quantity: float,
                        reason: str, reduce_only: bool = False) -> dict:
        """模拟下单"""
        self._order_counter += 1
        price = self.get_current_price(symbol)

        result = {
            "orderId": f"PAPER_{self._order_counter}",
            "clientOrderId": f"paper_{self._order_counter}",
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity,
            "avgPrice": str(price),
            "status": "FILLED",
            "paper": True,
        }

        self._order_history.append({
            **result,
            "time": datetime.now().isoformat(),
            "reason": reason,
        })

        if self.logger:
            action = "PAPER_"
            action += ("CLOSE_" if reduce_only else "OPEN_")
            action += ("LONG" if side == "BUY" else "SHORT")
            self.logger.log_trade(
                action=action, symbol=symbol,
                side=side, price=price,
                qty=quantity, reason=reason
            )

        return result

    def place_stop_loss(self, *args, **kwargs):
        return {"orderId": "PAPER_SL", "status": "NEW"}

    def place_take_profit(self, *args, **kwargs):
        return {"orderId": "PAPER_TP", "status": "NEW"}

    def cancel_all_orders(self, *args, **kwargs):
        return {}

    def close_all_positions(self, *args, **kwargs):
        return []

    def get_balance(self, asset="USDT"):
        return {"balance": self._usdt, "available": self._usdt,
                "unrealized_pnl": 0}

    def get_positions(self, symbol=None):
        return []

    def set_leverage(self, symbol, leverage):
        return {"leverage": leverage}

    def set_margin_type(self, symbol, margin_type="ISOLATED"):
        return {"msg": "paper mode"}

    def get_order_history(self):
        return self._order_history.copy()

    def get_exchange_info(self, symbol=None):
        return {}

    def get_income_history(self, **kwargs):
        return []

    def test_connection(self) -> bool:
        price = self.get_current_price("ETHUSDT")
        if price > 0:
            if self.logger:
                self.logger.info(f"Paper 模式 API 测试成功 ETH/USDT: ${price:.2f}")
            return True
        if self.logger:
            self.logger.info("Paper 模式 - 无实时价格源，使用历史数据")
        return True


class BinanceAPIError(Exception):
    """Binance API 错误"""
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


def create_order_manager(config, logger=None, notifier=None):
    """工厂函数 - 根据配置创建对应的订单管理器"""
    if config.phase == TradingPhase.PAPER:
        return PaperOrderManager(
            api_config=config.api,
            phase=config.phase,
            logger=logger,
            initial_capital=config.initial_capital,
        )
    else:
        return BinanceOrderManager(
            api_config=config.api,
            phase=config.phase,
            logger=logger,
            notifier=notifier,
        )
