"""
交易日志系统
5级结构化日志: TRADE / SIGNAL / RISK / BALANCE / FUNDING
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

# ============================================================
# 自定义日志级别
# ============================================================
TRADE = 25
SIGNAL = 24
RISK = 23
BALANCE = 22
FUNDING = 21

logging.addLevelName(TRADE, "TRADE")
logging.addLevelName(SIGNAL, "SIGNAL")
logging.addLevelName(RISK, "RISK")
logging.addLevelName(BALANCE, "BALANCE")
logging.addLevelName(FUNDING, "FUNDING")


class JSONFormatter(logging.Formatter):
    """JSON 格式日志 - 便于后续分析"""

    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, 'trade_data'):
            log_entry["data"] = record.trade_data
        return json.dumps(log_entry, ensure_ascii=False, default=str)


class TradingLogger:
    """交易专用日志器"""

    def __init__(self, log_dir: str = "logs/live", name: str = "trading"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(FUNDING)  # 捕获所有自定义级别
        self.logger.handlers = []  # 清除旧 handler

        # --- 控制台输出 ---
        console = logging.StreamHandler()
        console.setLevel(FUNDING)
        console_fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)-8s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console.setFormatter(console_fmt)
        self.logger.addHandler(console)

        # --- 文件输出 (每日轮转) ---
        today = datetime.now().strftime("%Y%m%d")

        # 可读格式文件
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"trade_{today}.log"),
            encoding='utf-8'
        )
        file_handler.setLevel(FUNDING)
        file_handler.setFormatter(console_fmt)
        self.logger.addHandler(file_handler)

        # JSON 格式文件 (便于分析)
        json_handler = logging.FileHandler(
            os.path.join(log_dir, f"trade_{today}.jsonl"),
            encoding='utf-8'
        )
        json_handler.setLevel(FUNDING)
        json_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(json_handler)

        # --- 统计计数器 ---
        self._trade_count = 0
        self._signal_count = 0
        self._risk_event_count = 0

    # ============================================================
    # TRADE 级别 - 每笔开仓/平仓/部分止盈
    # ============================================================
    def log_trade(self, action: str, symbol: str, side: str,
                  price: float, qty: float, margin: float = 0,
                  leverage: int = 0, fee: float = 0, slippage: float = 0,
                  pnl: float = 0, reason: str = "",
                  order_id: str = "", extra: dict = None):
        """
        记录交易事件
        action: OPEN_LONG / OPEN_SHORT / CLOSE_LONG / CLOSE_SHORT /
                PARTIAL_TP / STOP_LOSS / LIQUIDATION / PAPER_TRADE
        """
        self._trade_count += 1
        msg = (
            f"{action} {symbol} {side} "
            f"price={price:.2f} qty={qty:.4f} "
            f"margin={margin:.2f} lev={leverage}x "
            f"fee=${fee:.2f} slip=${slippage:.2f} "
            f"pnl=${pnl:.2f} reason=\"{reason}\""
        )
        if order_id:
            msg += f" order_id={order_id}"

        record = self.logger.makeRecord(
            self.logger.name, TRADE, "", 0, msg, (), None
        )
        record.trade_data = {
            "action": action, "symbol": symbol, "side": side,
            "price": price, "qty": qty, "margin": margin,
            "leverage": leverage, "fee": fee, "slippage": slippage,
            "pnl": pnl, "reason": reason, "order_id": order_id,
            "trade_number": self._trade_count,
            **(extra or {})
        }
        self.logger.handle(record)

    # ============================================================
    # SIGNAL 级别 - 每次信号计算
    # ============================================================
    def log_signal(self, sell_score: float, buy_score: float,
                   components: dict, conflict: bool = False,
                   action_taken: str = "HOLD",
                   timestamp: str = "", extra: dict = None):
        """
        记录信号计算结果
        components: {'div': x, 'ma': x, 'kdj': x, 'bb': x, 'vp': x, 'cs': x}
        """
        self._signal_count += 1
        c = components
        msg = (
            f"SS={sell_score:.1f} BS={buy_score:.1f} "
            f"div_s={c.get('div_s', 0):.0f} div_b={c.get('div_b', 0):.0f} "
            f"ma_s={c.get('ma_s', 0):.0f} ma_b={c.get('ma_b', 0):.0f} "
            f"kdj_s={c.get('kdj_s', 0):.0f} kdj_b={c.get('kdj_b', 0):.0f} "
            f"bb_s={c.get('bb_s', 0):.0f} bb_b={c.get('bb_b', 0):.0f} "
            f"vp_s={c.get('vp_s', 0):.0f} vp_b={c.get('vp_b', 0):.0f} "
            f"cs_s={c.get('cs_s', 0):.0f} cs_b={c.get('cs_b', 0):.0f} "
            f"conflict={'YES' if conflict else 'NO'} "
            f"action={action_taken}"
        )
        if timestamp:
            msg = f"[{timestamp}] " + msg

        record = self.logger.makeRecord(
            self.logger.name, SIGNAL, "", 0, msg, (), None
        )
        record.trade_data = {
            "sell_score": sell_score, "buy_score": buy_score,
            "components": components, "conflict": conflict,
            "action_taken": action_taken, "timestamp": timestamp,
            "signal_number": self._signal_count,
            **(extra or {})
        }
        self.logger.handle(record)

    # ============================================================
    # RISK 级别 - 风险事件
    # ============================================================
    def log_risk(self, event_type: str, message: str,
                 current_value: float = 0, threshold: float = 0,
                 action: str = "", extra: dict = None):
        """
        记录风险事件
        event_type: STOP_LOSS / LIQUIDATION / CIRCUIT_BREAKER /
                    MARGIN_WARNING / DRAWDOWN_ALERT / KILL_SWITCH /
                    MAX_LOSS_DAILY / MAX_LOSS_WEEKLY / CONSECUTIVE_LOSS
        """
        self._risk_event_count += 1
        msg = (
            f"{event_type}: {message} "
            f"value={current_value:.4f} threshold={threshold:.4f} "
            f"action={action}"
        )

        record = self.logger.makeRecord(
            self.logger.name, RISK, "", 0, msg, (), None
        )
        record.trade_data = {
            "event_type": event_type, "message": message,
            "current_value": current_value, "threshold": threshold,
            "action": action, "risk_event_number": self._risk_event_count,
            **(extra or {})
        }
        self.logger.handle(record)

    # ============================================================
    # BALANCE 级别 - 定时余额快照
    # ============================================================
    def log_balance(self, total_equity: float, usdt: float,
                    unrealized_pnl: float = 0, frozen_margin: float = 0,
                    available_margin: float = 0,
                    positions: list = None, extra: dict = None):
        """记录余额快照"""
        msg = (
            f"equity=${total_equity:.2f} usdt=${usdt:.2f} "
            f"unrealized=${unrealized_pnl:.2f} "
            f"frozen=${frozen_margin:.2f} avail=${available_margin:.2f}"
        )
        if positions:
            for p in positions:
                msg += (
                    f" | {p['side']} entry={p['entry_price']:.2f} "
                    f"qty={p['qty']:.4f} pnl={p['pnl']:.2f}"
                )

        record = self.logger.makeRecord(
            self.logger.name, BALANCE, "", 0, msg, (), None
        )
        record.trade_data = {
            "total_equity": total_equity, "usdt": usdt,
            "unrealized_pnl": unrealized_pnl,
            "frozen_margin": frozen_margin,
            "available_margin": available_margin,
            "positions": positions or [],
            **(extra or {})
        }
        self.logger.handle(record)

    # ============================================================
    # FUNDING 级别 - 资金费率
    # ============================================================
    def log_funding(self, rate: float, direction: str,
                    amount: float, cumulative: float,
                    symbol: str = "ETHUSDT", extra: dict = None):
        """记录资金费率"""
        msg = (
            f"{symbol} rate={rate:+.4%} direction={direction} "
            f"amount=${amount:+.2f} cumulative=${cumulative:.2f}"
        )

        record = self.logger.makeRecord(
            self.logger.name, FUNDING, "", 0, msg, (), None
        )
        record.trade_data = {
            "rate": rate, "direction": direction,
            "amount": amount, "cumulative": cumulative,
            "symbol": symbol,
            **(extra or {})
        }
        self.logger.handle(record)

    # ============================================================
    # 通用方法
    # ============================================================
    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def get_stats(self) -> dict:
        return {
            "total_trades": self._trade_count,
            "total_signals": self._signal_count,
            "total_risk_events": self._risk_event_count,
        }
