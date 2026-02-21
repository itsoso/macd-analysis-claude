"""
SpotEngine + CapitalAllocator 测试
"""

import os
import tempfile
import pytest

from hotcoin.config import HotCoinConfig, DiscoveryConfig, TradingConfig, ExecutionConfig
from hotcoin.discovery.candidate_pool import CandidatePool, HotCoin
from hotcoin.execution.spot_engine import HotCoinSpotEngine
from hotcoin.execution.capital_allocator import CapitalAllocator
from hotcoin.engine.signal_worker import TradeSignal


@pytest.fixture
def engine():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        config = HotCoinConfig(
            db_path=db_path,
            execution=ExecutionConfig(
                initial_capital=10000,
                max_concurrent_positions=3,
                use_paper_trading=True,
            ),
        )
        pool = CandidatePool(db_path, config.discovery)
        eng = HotCoinSpotEngine(config, pool)
        yield eng
        pool.close()


def test_allocator_single():
    config = ExecutionConfig(initial_capital=10000, max_concurrent_positions=5)
    alloc = CapitalAllocator(config)

    # High heat + high liquidity → more allocation
    a1 = alloc.allocate_single(heat_score=80, liquidity_score=70, current_positions=0)
    # Low heat → less
    a2 = alloc.allocate_single(heat_score=30, liquidity_score=70, current_positions=0)
    assert a1 > a2 > 0


def test_allocator_max_positions():
    config = ExecutionConfig(initial_capital=10000, max_concurrent_positions=3)
    alloc = CapitalAllocator(config)

    a = alloc.allocate_single(heat_score=80, liquidity_score=70, current_positions=3)
    assert a == 0.0


def test_allocator_batch():
    config = ExecutionConfig(initial_capital=50000, max_single_position_pct=0.20)
    alloc = CapitalAllocator(config)

    candidates = [
        {"symbol": "PEPEUSDT", "heat_score": 80, "liquidity_score": 70},
        {"symbol": "DOGEUSDT", "heat_score": 30, "liquidity_score": 30},
    ]
    result = alloc.allocate_batch(candidates)
    assert "PEPEUSDT" in result
    assert "DOGEUSDT" in result
    assert result["PEPEUSDT"] > result["DOGEUSDT"]


def test_sell_signal_without_position_is_ignored(engine, monkeypatch):
    buy_called = {"n": 0}

    def _fake_buy(_symbol, _quote):
        buy_called["n"] += 1
        return {"price": 1.0, "qty": 1.0}

    monkeypatch.setattr(engine.executor, "spot_market_buy", _fake_buy)

    engine.process_signals([
        TradeSignal(symbol="ETHUSDT", action="SELL", strength=30, confidence=0.8, reason="bearish"),
    ])

    assert buy_called["n"] == 0
    assert engine.num_positions == 0


def test_sell_signal_closes_existing_buy_position(engine, monkeypatch):
    engine.risk.open_position("ETHUSDT", "BUY", 2000.0, 0.1)
    sold = {"n": 0}

    def _fake_get_price(_symbol):
        return 1990.0

    def _fake_sell(_symbol, _qty):
        sold["n"] += 1
        return {"price": 1990.0, "qty": _qty}

    monkeypatch.setattr(engine.executor, "get_current_price", _fake_get_price)
    monkeypatch.setattr(engine.executor, "spot_market_sell", _fake_sell)

    engine.process_signals([
        TradeSignal(symbol="ETHUSDT", action="SELL", strength=40, confidence=0.9, reason="exit"),
    ])

    assert sold["n"] == 1
    assert engine.num_positions == 0
