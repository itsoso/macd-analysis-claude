"""
风险管理器
熔断机制 / 回撤监控 / 连续亏损检测 / Kill Switch / 仓位限制
"""

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List

from live_config import RiskConfig, TradingPhase


@dataclass
class TradeRecord:
    """单笔交易记录"""
    timestamp: str
    action: str     # OPEN / CLOSE
    side: str       # LONG / SHORT
    price: float
    qty: float
    pnl: float      # 已实现盈亏
    fee: float
    reason: str


@dataclass
class RiskState:
    """风控状态 - 可持久化"""
    # 资金峰值 (用于计算回撤)
    peak_equity: float = 0
    # 当日已实现盈亏
    daily_pnl: float = 0
    daily_date: str = ""
    # 本周已实现盈亏
    weekly_pnl: float = 0
    weekly_start: str = ""
    # 连续亏损计数
    consecutive_losses: int = 0
    # 策略暂停状态
    is_paused: bool = False
    pause_reason: str = ""
    pause_time: str = ""
    # Kill Switch
    kill_switch_active: bool = False
    # 交易历史 (最近50笔)
    recent_trades: List[dict] = field(default_factory=list)
    # 累计统计
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_pnl: float = 0
    total_fees: float = 0
    max_drawdown: float = 0

    def to_dict(self) -> dict:
        return {
            "peak_equity": self.peak_equity,
            "daily_pnl": self.daily_pnl,
            "daily_date": self.daily_date,
            "weekly_pnl": self.weekly_pnl,
            "weekly_start": self.weekly_start,
            "consecutive_losses": self.consecutive_losses,
            "is_paused": self.is_paused,
            "pause_reason": self.pause_reason,
            "pause_time": self.pause_time,
            "kill_switch_active": self.kill_switch_active,
            "recent_trades": self.recent_trades[-50:],
            "total_trades": self.total_trades,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "total_pnl": self.total_pnl,
            "total_fees": self.total_fees,
            "max_drawdown": self.max_drawdown,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'RiskState':
        state = cls()
        for k, v in d.items():
            if hasattr(state, k):
                setattr(state, k, v)
        return state


class RiskManager:
    """风险管理器"""

    def __init__(self, config: RiskConfig, initial_capital: float,
                 logger=None, notifier=None, state_file: str = None):
        self.config = config
        self.initial_capital = initial_capital
        self.logger = logger
        self.notifier = notifier
        self.state_file = state_file

        # 加载或初始化状态
        self.state = self._load_state()
        if self.state.peak_equity == 0:
            self.state.peak_equity = initial_capital

    # ============================================================
    # 核心检查: 是否允许开仓
    # ============================================================
    def can_open_position(self, side: str, margin: float,
                          current_equity: float,
                          current_frozen_margin: float) -> tuple:
        """
        检查是否允许开仓
        返回: (allowed: bool, reason: str)
        """
        # Kill Switch
        if self.state.kill_switch_active:
            return False, "Kill Switch 已激活，所有交易已暂停"

        # 策略暂停
        if self.state.is_paused:
            return False, f"策略已暂停: {self.state.pause_reason}"

        # 紧急平仓标志
        if self.config.emergency_close_all:
            return False, "紧急平仓模式，禁止新开仓"

        # --- 仓位限制 ---
        # 单仓限制
        max_single = self.initial_capital * self.config.max_single_position_pct
        if margin > max_single:
            return False, (f"单仓保证金 ${margin:.0f} 超过限制 "
                          f"${max_single:.0f} ({self.config.max_single_position_pct:.0%})")

        # 总仓位限制
        max_total = self.initial_capital * self.config.max_total_margin_pct
        if current_frozen_margin + margin > max_total:
            return False, (f"总保证金 ${current_frozen_margin + margin:.0f} "
                          f"超过限制 ${max_total:.0f}")

        # 资金储备
        remaining = current_equity - current_frozen_margin - margin
        min_reserve = self.initial_capital * self.config.reserve_pct
        if remaining < min_reserve:
            return False, (f"剩余资金 ${remaining:.0f} 低于储备要求 "
                          f"${min_reserve:.0f} ({self.config.reserve_pct:.0%})")

        return True, "OK"

    # ============================================================
    # 交易后更新
    # ============================================================
    def on_trade_closed(self, pnl: float, fee: float,
                        current_equity: float):
        """交易平仓后更新风控状态"""
        today = datetime.now().strftime("%Y-%m-%d")
        week_start = (datetime.now() - timedelta(days=datetime.now().weekday())
                      ).strftime("%Y-%m-%d")

        # 更新日/周盈亏
        if self.state.daily_date != today:
            self.state.daily_pnl = 0
            self.state.daily_date = today
        self.state.daily_pnl += pnl

        if self.state.weekly_start != week_start:
            self.state.weekly_pnl = 0
            self.state.weekly_start = week_start
        self.state.weekly_pnl += pnl

        # 更新统计
        self.state.total_trades += 1
        self.state.total_pnl += pnl
        self.state.total_fees += fee
        if pnl > 0:
            self.state.total_wins += 1
            self.state.consecutive_losses = 0
        elif pnl < 0:
            self.state.total_losses += 1
            self.state.consecutive_losses += 1

        # 更新峰值和回撤
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity
        drawdown = (self.state.peak_equity - current_equity) / self.state.peak_equity
        if drawdown > self.state.max_drawdown:
            self.state.max_drawdown = drawdown

        # 记录交易
        self.state.recent_trades.append({
            "time": datetime.now().isoformat(),
            "pnl": pnl,
            "fee": fee,
            "equity": current_equity,
        })
        if len(self.state.recent_trades) > 50:
            self.state.recent_trades = self.state.recent_trades[-50:]

        # --- 触发熔断检查 ---
        self._check_circuit_breakers(current_equity)

        # 持久化
        self._save_state()

    # ============================================================
    # 熔断检查
    # ============================================================
    def _check_circuit_breakers(self, current_equity: float):
        """检查所有熔断条件"""

        # 1. 日最大亏损
        daily_loss_ratio = abs(self.state.daily_pnl) / self.initial_capital
        if self.state.daily_pnl < 0 and daily_loss_ratio >= self.config.max_daily_loss_pct:
            self._trigger_pause(
                "MAX_LOSS_DAILY",
                f"日亏损 ${self.state.daily_pnl:.2f} "
                f"({daily_loss_ratio:.1%}) 触发日限",
                daily_loss_ratio,
                self.config.max_daily_loss_pct
            )

        # 2. 周最大亏损
        weekly_loss_ratio = abs(self.state.weekly_pnl) / self.initial_capital
        if (self.state.weekly_pnl < 0 and
                weekly_loss_ratio >= self.config.max_weekly_loss_pct):
            self._trigger_pause(
                "MAX_LOSS_WEEKLY",
                f"周亏损 ${self.state.weekly_pnl:.2f} "
                f"({weekly_loss_ratio:.1%}) 触发周限",
                weekly_loss_ratio,
                self.config.max_weekly_loss_pct
            )

        # 3. 连续亏损
        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            self._trigger_pause(
                "CONSECUTIVE_LOSS",
                f"连续 {self.state.consecutive_losses} 笔亏损",
                self.state.consecutive_losses,
                self.config.max_consecutive_losses
            )

        # 4. 最大回撤
        drawdown = ((self.state.peak_equity - current_equity) /
                    self.state.peak_equity) if self.state.peak_equity > 0 else 0
        if drawdown >= self.config.max_drawdown_pct:
            self._trigger_pause(
                "DRAWDOWN_ALERT",
                f"回撤 {drawdown:.1%} 超过阈值",
                drawdown,
                self.config.max_drawdown_pct
            )

    def _trigger_pause(self, reason_type: str, message: str,
                       current_value: float, threshold: float):
        """触发策略暂停"""
        self.state.is_paused = True
        self.state.pause_reason = f"{reason_type}: {message}"
        self.state.pause_time = datetime.now().isoformat()

        if self.logger:
            self.logger.log_risk(
                reason_type, message,
                current_value=current_value,
                threshold=threshold,
                action="STRATEGY_PAUSED"
            )

        if self.notifier:
            self.notifier.notify_risk(
                reason_type, message,
                current_value=current_value,
                threshold=threshold,
                action="策略已自动暂停，需要人工审查后恢复"
            )

    # ============================================================
    # 定时检查 (每根K线调用)
    # ============================================================
    def check_positions(self, current_price: float,
                        current_equity: float,
                        positions: list) -> list:
        """
        检查持仓风险，返回需要执行的动作列表
        返回: [{"action": "CLOSE", "side": "LONG", "reason": "..."}]
        """
        actions = []

        # 更新峰值
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity

        # 回撤预警 (达到阈值的 80% 时预警)
        if self.state.peak_equity > 0:
            drawdown = (self.state.peak_equity - current_equity) / self.state.peak_equity
            warn_threshold = self.config.max_drawdown_pct * 0.8
            if drawdown >= warn_threshold and not self.state.is_paused:
                if self.logger:
                    self.logger.log_risk(
                        "DRAWDOWN_WARNING",
                        f"回撤 {drawdown:.1%} 接近阈值 {self.config.max_drawdown_pct:.1%}",
                        current_value=drawdown,
                        threshold=self.config.max_drawdown_pct,
                        action="WARNING"
                    )

        # Kill Switch 检查
        if self.state.kill_switch_active and positions:
            for p in positions:
                actions.append({
                    "action": "CLOSE",
                    "side": p.get("side", ""),
                    "reason": "Kill Switch 紧急平仓"
                })

        return actions

    # ============================================================
    # 手动控制
    # ============================================================
    def activate_kill_switch(self, reason: str = "手动触发"):
        """激活一键平仓"""
        self.state.kill_switch_active = True
        if self.logger:
            self.logger.log_risk("KILL_SWITCH", reason, action="ALL_POSITIONS_CLOSE")
        if self.notifier:
            self.notifier.notify_risk("KILL_SWITCH", reason, action="紧急平仓所有仓位")
        self._save_state()

    def deactivate_kill_switch(self):
        """解除一键平仓"""
        self.state.kill_switch_active = False
        if self.logger:
            self.logger.info("Kill Switch 已解除")
        self._save_state()

    def resume_trading(self, reason: str = "人工审查后恢复"):
        """恢复交易 (从暂停状态)"""
        self.state.is_paused = False
        self.state.pause_reason = ""
        self.state.pause_time = ""
        self.state.consecutive_losses = 0  # 重置
        if self.logger:
            self.logger.info(f"交易已恢复: {reason}")
        if self.notifier:
            self.notifier.notify_system("RESUME", f"交易已恢复: {reason}")
        self._save_state()

    # ============================================================
    # 滑点检查
    # ============================================================
    def check_slippage(self, expected_price: float,
                       actual_price: float, side: str) -> tuple:
        """
        检查滑点是否在可接受范围
        返回: (acceptable: bool, slippage_pct: float)
        """
        if expected_price == 0:
            return True, 0

        slippage = abs(actual_price - expected_price) / expected_price

        if slippage > self.config.max_slippage_pct:
            if self.logger:
                self.logger.log_risk(
                    "SLIPPAGE_HIGH",
                    f"滑点 {slippage:.3%} 超过阈值 {self.config.max_slippage_pct:.3%}",
                    current_value=slippage,
                    threshold=self.config.max_slippage_pct,
                    action="TRADE_REJECTED"
                )
            return False, slippage

        return True, slippage

    # ============================================================
    # 杠杆约束
    # ============================================================
    def constrain_leverage(self, requested_leverage: int) -> int:
        """约束杠杆不超过配置上限"""
        return min(requested_leverage, self.config.max_leverage)

    def constrain_margin(self, requested_margin: float,
                         current_equity: float,
                         current_frozen: float) -> float:
        """约束保证金不超过限制"""
        # 单仓限制
        max_single = self.initial_capital * self.config.max_single_position_pct
        margin = min(requested_margin, max_single)

        # 总仓限制
        max_total = self.initial_capital * self.config.max_total_margin_pct
        max_additional = max(0, max_total - current_frozen)
        margin = min(margin, max_additional)

        # 保留储备
        available = current_equity - current_frozen
        min_reserve = self.initial_capital * self.config.reserve_pct
        max_from_reserve = max(0, available - min_reserve)
        margin = min(margin, max_from_reserve)

        return margin

    # ============================================================
    # 状态持久化
    # ============================================================
    def _save_state(self):
        """保存风控状态"""
        if not self.state_file:
            return
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2, ensure_ascii=False)

    def _load_state(self) -> RiskState:
        """加载风控状态"""
        if self.state_file and os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                return RiskState.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                pass
        return RiskState()

    def reset_stats(self, new_capital: float = 0):
        """重置累计统计 (资金变更或重新开始时调用)

        Parameters
        ----------
        new_capital : float
            新的初始资金。如果为 0, 使用 self.initial_capital。
        """
        cap = new_capital if new_capital > 0 else self.initial_capital
        self.state.peak_equity = cap
        self.state.daily_pnl = 0
        self.state.daily_date = ""
        self.state.weekly_pnl = 0
        self.state.weekly_start = ""
        self.state.consecutive_losses = 0
        self.state.is_paused = False
        self.state.pause_reason = ""
        self.state.pause_time = ""
        self.state.kill_switch_active = False
        self.state.recent_trades = []
        self.state.total_trades = 0
        self.state.total_wins = 0
        self.state.total_losses = 0
        self.state.total_pnl = 0
        self.state.total_fees = 0
        self.state.max_drawdown = 0
        self.initial_capital = cap
        self._save_state()
        if self.logger:
            self.logger.info(f"风控统计已重置, 新初始资金: ${cap:,.2f}")

    def get_status_report(self) -> dict:
        """获取风控状态报告"""
        return {
            "is_paused": self.state.is_paused,
            "pause_reason": self.state.pause_reason,
            "kill_switch": self.state.kill_switch_active,
            "peak_equity": self.state.peak_equity,
            "daily_pnl": self.state.daily_pnl,
            "weekly_pnl": self.state.weekly_pnl,
            "consecutive_losses": self.state.consecutive_losses,
            "max_drawdown": self.state.max_drawdown,
            "total_trades": self.state.total_trades,
            "win_rate": (self.state.total_wins / self.state.total_trades
                        if self.state.total_trades > 0 else 0),
            "total_pnl": self.state.total_pnl,
            "total_fees": self.state.total_fees,
        }
