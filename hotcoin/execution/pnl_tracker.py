"""
实时 PnL 跟踪器

记录所有交易损益、持仓浮动盈亏, 提供统计摘要。
"""

import json
import logging
import os
import threading
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

    MAX_IN_MEMORY = 500

    def __init__(self, data_dir: str = ""):
        self._trades: List[TradeRecord] = []
        self._lock = threading.Lock()
        self._data_dir = data_dir or os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(self._data_dir, exist_ok=True)

    def record_trade(self, symbol: str, side: str, entry_price: float,
                     exit_price: float, qty: float, entry_time: float,
                     reason: str = "") -> TradeRecord:
        pnl = (exit_price - entry_price) * qty if side == "BUY" else (entry_price - exit_price) * qty
        safe_entry = entry_price if entry_price > 0 else 1e-10
        pnl_pct = (exit_price - entry_price) / safe_entry if side == "BUY" else (entry_price - exit_price) / safe_entry

        record = TradeRecord(
            symbol=symbol, side=side,
            entry_price=entry_price, exit_price=exit_price,
            qty=qty, pnl=pnl, pnl_pct=pnl_pct,
            holding_sec=time.time() - entry_time,
            reason=reason,
            entry_time=entry_time, exit_time=time.time(),
        )
        with self._lock:
            self._trades.append(record)
            if len(self._trades) > self.MAX_IN_MEMORY:
                self._trades = self._trades[-self.MAX_IN_MEMORY:]
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
        try:
            with open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            log.exception("PnL 记录写入失败: %s", path)

    def get_summary(self) -> dict:
        with self._lock:
            snapshot = list(self._trades)
        if not snapshot:
            return {"total_trades": 0}

        wins = [t for t in snapshot if t.pnl > 0]
        losses = [t for t in snapshot if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in snapshot)
        avg_hold = sum(t.holding_sec for t in snapshot) / len(snapshot)

        gross_profit = sum(t.pnl for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in snapshot:
            cum_pnl += t.pnl
            peak = max(peak, cum_pnl)
            dd = peak - cum_pnl
            max_dd = max(max_dd, dd)

        return {
            "total_trades": len(snapshot),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(snapshot), 4),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / len(snapshot), 2),
            "avg_holding_min": round(avg_hold / 60, 1),
            "best_trade": round(max(t.pnl for t in snapshot), 2),
            "worst_trade": round(min(t.pnl for t in snapshot), 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
            "max_drawdown": round(max_dd, 2),
        }
