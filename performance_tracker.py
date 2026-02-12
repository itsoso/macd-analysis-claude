"""
绩效追踪器
实盘 vs 回测对比 / 胜率统计 / 滑点分析 / 费用分析
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional, List, Dict


class PerformanceTracker:
    """交易绩效追踪和分析"""

    def __init__(self, initial_capital: float, data_dir: str = "data/live"):
        self.initial_capital = initial_capital
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # 交易记录
        self.trades: List[dict] = []
        # 权益曲线
        self.equity_curve: List[dict] = []
        # 每日统计
        self.daily_stats: Dict[str, dict] = {}
        # 滑点记录
        self.slippage_records: List[dict] = []
        # 费用累计
        self.total_fees: float = 0
        self.total_funding: float = 0
        # 峰值权益
        self.peak_equity: float = initial_capital
        self.max_drawdown: float = 0
        self.max_drawdown_duration: int = 0  # 天数
        self._drawdown_start: Optional[str] = None

        # 加载历史数据
        self._load()

    # ============================================================
    # 记录方法
    # ============================================================
    def record_trade(self, trade: dict):
        """
        记录一笔交易
        trade: {
            "time", "action", "side", "price", "qty",
            "margin", "leverage", "fee", "pnl", "reason",
            "expected_price"(可选), "actual_price"(可选)
        }
        """
        trade["trade_id"] = len(self.trades) + 1
        if "time" not in trade:
            trade["time"] = datetime.now().isoformat()
        self.trades.append(trade)

        # 更新费用
        self.total_fees += trade.get("fee", 0)

        # 滑点记录
        if "expected_price" in trade and "actual_price" in trade:
            expected = trade["expected_price"]
            actual = trade["actual_price"]
            if expected > 0:
                slip = abs(actual - expected) / expected
                self.slippage_records.append({
                    "time": trade["time"],
                    "expected": expected,
                    "actual": actual,
                    "slippage_pct": slip,
                    "side": trade.get("side", ""),
                })

        self._save()

    def record_equity(self, equity: float, timestamp: str = None):
        """记录权益快照"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        self.equity_curve.append({
            "time": timestamp,
            "equity": equity,
        })

        # 更新峰值和回撤
        if equity > self.peak_equity:
            self.peak_equity = equity
            self._drawdown_start = None

        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

    def record_funding(self, amount: float):
        """记录资金费率"""
        self.total_funding += amount

    def record_daily(self, date: str, stats: dict):
        """记录每日统计"""
        self.daily_stats[date] = stats
        self._save()

    # ============================================================
    # 统计分析
    # ============================================================
    def get_summary(self) -> dict:
        """获取完整绩效汇总"""
        closed_trades = [t for t in self.trades
                        if t.get("action", "").startswith("CLOSE")]

        total_trades = len(closed_trades)
        wins = [t for t in closed_trades if t.get("pnl", 0) > 0]
        losses = [t for t in closed_trades if t.get("pnl", 0) < 0]

        total_pnl = sum(t.get("pnl", 0) for t in closed_trades)
        total_win_pnl = sum(t.get("pnl", 0) for t in wins)
        total_loss_pnl = sum(t.get("pnl", 0) for t in losses)

        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        avg_win = total_win_pnl / len(wins) if wins else 0
        avg_loss = total_loss_pnl / len(losses) if losses else 0
        profit_factor = (total_win_pnl / abs(total_loss_pnl)
                        if total_loss_pnl != 0 else float('inf'))

        # 最大连续亏损
        max_consec_loss = 0
        current_consec = 0
        for t in closed_trades:
            if t.get("pnl", 0) < 0:
                current_consec += 1
                max_consec_loss = max(max_consec_loss, current_consec)
            else:
                current_consec = 0

        # 当前权益
        current_equity = (self.equity_curve[-1]["equity"]
                         if self.equity_curve else self.initial_capital)
        # 口径统一: 初始资金缺失时避免除零，退化为0收益
        capital_base = self.initial_capital if self.initial_capital > 0 else 0
        total_return = ((current_equity - capital_base) / capital_base
                        if capital_base > 0 else 0)

        # 滑点统计
        avg_slippage = 0
        if self.slippage_records:
            avg_slippage = sum(s["slippage_pct"]
                              for s in self.slippage_records) / len(self.slippage_records)

        return {
            "initial_capital": capital_base,
            "current_equity": current_equity,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "win_rate_pct": win_rate * 100,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_consecutive_losses": max_consec_loss,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown * 100,
            "total_fees": self.total_fees,
            "total_funding": self.total_funding,
            "fee_ratio": self.total_fees / total_pnl if total_pnl > 0 else 0,
            "avg_slippage": avg_slippage,
            "slippage_samples": len(self.slippage_records),
            "equity_data_points": len(self.equity_curve),
        }

    def get_daily_summary(self, date: str = None) -> dict:
        """获取某日统计"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        day_trades = [t for t in self.trades
                     if t.get("time", "").startswith(date)
                     and t.get("action", "").startswith("CLOSE")]

        pnl = sum(t.get("pnl", 0) for t in day_trades)
        fees = sum(t.get("fee", 0) for t in day_trades)
        wins = sum(1 for t in day_trades if t.get("pnl", 0) > 0)

        return {
            "date": date,
            "trades": len(day_trades),
            "pnl": pnl,
            "fees": fees,
            "wins": wins,
            "losses": len(day_trades) - wins,
            "net_pnl": pnl - fees,
        }

    def compare_with_backtest(self, backtest_result: dict) -> dict:
        """与回测结果对比"""
        live = self.get_summary()

        # 回测结果兼容两种口径:
        # 1) ratio: 0.12  2) percent: 12.0
        def _to_ratio(v):
            try:
                x = float(v)
            except (TypeError, ValueError):
                return 0.0
            return x / 100.0 if abs(x) > 1.0 else x

        bt_return = _to_ratio(
            backtest_result.get("total_return",
                                backtest_result.get("strategy_return", 0))
        )
        bt_win_rate = _to_ratio(backtest_result.get("win_rate", 0))
        bt_max_dd = _to_ratio(backtest_result.get("max_drawdown", 0))
        bt_trades = backtest_result.get("total_trades", 0)

        return {
            "return": {
                "live": live["total_return"],
                "backtest": bt_return,
                "deviation": live["total_return"] - bt_return,
            },
            "win_rate": {
                "live": live["win_rate"],
                "backtest": bt_win_rate,
                "deviation": live["win_rate"] - bt_win_rate,
            },
            "max_drawdown": {
                "live": live["max_drawdown"],
                "backtest": bt_max_dd,
                "deviation": live["max_drawdown"] - bt_max_dd,
            },
            "trades_per_day": {
                "live": live["total_trades"],
                "backtest": bt_trades,
            },
            "fee_impact": {
                "total_fees": live["total_fees"],
                "total_funding": live["total_funding"],
                "fee_ratio": live["fee_ratio"],
            },
            "slippage": {
                "avg_actual": live["avg_slippage"],
                "model_assumed": 0.001,
                "deviation": live["avg_slippage"] - 0.001,
            },
        }

    # ============================================================
    # 持久化
    # ============================================================
    def _save(self):
        """保存绩效数据"""
        filepath = os.path.join(self.data_dir, "performance.json")
        data = {
            "initial_capital": self.initial_capital,
            "trades": self.trades[-200:],  # 最近200笔
            "equity_curve": self.equity_curve[-1000:],
            "daily_stats": dict(list(self.daily_stats.items())[-30:]),
            "slippage_records": self.slippage_records[-100:],
            "total_fees": self.total_fees,
            "total_funding": self.total_funding,
            "peak_equity": self.peak_equity,
            "max_drawdown": self.max_drawdown,
            "saved_at": datetime.now().isoformat(),
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _load(self):
        """加载绩效数据"""
        filepath = os.path.join(self.data_dir, "performance.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                self.initial_capital = data.get("initial_capital", self.initial_capital)
                self.trades = data.get("trades", [])
                self.equity_curve = data.get("equity_curve", [])
                self.daily_stats = data.get("daily_stats", {})
                self.slippage_records = data.get("slippage_records", [])
                self.total_fees = data.get("total_fees", 0)
                self.total_funding = data.get("total_funding", 0)
                self.peak_equity = data.get("peak_equity", self.initial_capital)
                self.max_drawdown = data.get("max_drawdown", 0)
            except (json.JSONDecodeError, KeyError):
                pass

    def export_report(self, filepath: str = None) -> str:
        """导出绩效报告"""
        if filepath is None:
            filepath = os.path.join(
                self.data_dir,
                f"report_{datetime.now():%Y%m%d_%H%M}.json"
            )

        report = {
            "summary": self.get_summary(),
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "daily_stats": self.daily_stats,
            "generated_at": datetime.now().isoformat(),
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return filepath
