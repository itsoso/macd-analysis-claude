"""
PortfolioRisk 五层风控单元测试
"""

import time
import pytest

from hotcoin.config import ExecutionConfig
from hotcoin.execution.portfolio_risk import PortfolioRisk


@pytest.fixture
def risk():
    config = ExecutionConfig(initial_capital=10000, max_concurrent_positions=3)
    return PortfolioRisk(config)


def test_can_open_basic(risk):
    ok, reason = risk.can_open("TESTUSDT", 500)
    assert ok is True


def test_max_concurrent(risk):
    for i in range(3):
        risk.open_position(f"T{i}USDT", "BUY", 1.0, 100)
    ok, reason = risk.can_open("T3USDT", 100)
    assert ok is False
    assert "仓位已满" in reason


def test_duplicate_symbol(risk):
    risk.open_position("ETHUSDT", "BUY", 2000, 0.1)
    ok, reason = risk.can_open("ETHUSDT", 200)
    assert ok is False
    assert "已有持仓" in reason


def test_single_position_limit(risk):
    # max_single = 10000 * 0.10 = 1000
    ok, reason = risk.can_open("BIGUSDT", 2000)
    assert ok is False
    assert "单币金额超限" in reason


def test_close_position_pnl(risk):
    risk.open_position("WINUSDT", "BUY", 100, 1.0)
    pnl = risk.close_position("WINUSDT", 110, reason="止盈")
    assert pnl == 10.0
    assert risk.state.daily_pnl == 10.0


def test_l5_drawdown_halt(risk):
    risk.open_position("LOSEUSDT", "BUY", 100, 10)
    # current_price drops from 100 to 80 → unrealized = -200
    # drawdown = (10000 - 9800) / 10000 = 2% < 15%, no halt

    # Force large loss
    risk.close_position("LOSEUSDT", 80, "test")
    # PnL = -200, equity = 9800

    risk.open_position("LOSE2", "BUY", 100, 100)
    actions = risk.check_all({"LOSE2": 50})  # unrealized = -5000
    # equity = 9800 - 5000 = 4800, drawdown = (10000-4800)/10000 = 52% > 15%
    assert any(a[0] == "CLOSE_ALL" for a in actions)
    assert risk.state.halted is True


def test_daily_loss_halt(risk):
    # daily max loss = 10000 * 0.05 = 500
    risk.open_position("DAY1", "BUY", 100, 6)
    risk.close_position("DAY1", 10, "大亏")
    # PnL = -540

    ok, reason = risk.can_open("DAY2", 100)
    assert ok is False
    assert "L4" in reason
