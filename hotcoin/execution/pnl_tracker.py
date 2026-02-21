"""
实时 PnL 跟踪器

记录所有交易损益、持仓浮动盈亏, 提供统计摘要。
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List

log = logging.getLogger("hotcoin.pnl")


@dataclass
class TradeRecord:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float
    holding_sec: float
    reason: str
    entry_time: float
    exit_time: float


class PnLTracker:
    """交易损益追踪。"""

    def __init__(self, data_dir: str = ""):
        self._trades: List[TradeRecord] = []
        self._data_dir = data_dir or os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(self._data_dir, exist_ok=True)

    def record_trade(self, symbol: str, side: str, entry_price: float,
                     exit_price: float, qty: float, entry_time: float,
                     reason: str = "") -> TradeRecord:
        pnl = (exit_price - entry_price) * qty if side == "BUY" else (entry_price - exit_price) * qty
        pnl_pct = (exit_price - entry_price) / entry_price if side == "BUY" else (entry_price - exit_price) / entry_price

        record = TradeRecord(
            symbol=symbol, side=side,
            entry_price=entry_price, exit_price=exit_price,
            qty=qty, pnl=pnl, pnl_pct=pnl_pct,
            holding_sec=time.time() - entry_time,
            reason=reason,
            entry_time=entry_time, exit_time=time.time(),
        )
        self._trades.append(record)
        self._persist(record)
        return record

    def _persist(self, record: TradeRecord):
        """追加写入 JSONL 文件。"""
        ts = time.strftime("%Y%m%d", time.gmtime(record.exit_time))
        path = os.path.join(self._data_dir, f"hotcoin_trades_{ts}.jsonl")
        entry = {
            "symbol": record.symbol,
            "side": record.side,
            "entry_price": record.entry_price,
            "exit_price": record.exit_price,
            "qty": record.qty,
            "pnl": round(record.pnl, 4),
            "pnl_pct": round(record.pnl_pct, 4),
            "holding_sec": round(record.holding_sec),
            "reason": record.reason,
            "entry_time": record.entry_time,
            "exit_time": record.exit_time,
        }
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_summary(self) -> dict:
        if not self._trades:
            return {"total_trades": 0}

        wins = [t for t in self._trades if t.pnl > 0]
        losses = [t for t in self._trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in self._trades)
        avg_hold = sum(t.holding_sec for t in self._trades) / len(self._trades) if self._trades else 0

        return {
            "total_trades": len(self._trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self._trades) if self._trades else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / len(self._trades), 2),
            "avg_holding_min": round(avg_hold / 60, 1),
            "best_trade": round(max(t.pnl for t in self._trades), 2) if self._trades else 0,
            "worst_trade": round(min(t.pnl for t in self._trades), 2) if self._trades else 0,
        }
