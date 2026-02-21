"""
R6 线程安全与优化测试:
  P0  PortfolioRisk 锁保护
  P1  OrderExecutor metrics_lock
  P1  消除冗余价格调用
  P2  _order_history 修剪
"""

import os
import tempfile
import threading
import time

import pytest


# ---- PortfolioRisk 并发读写安全 ----

def test_portfolio_risk_concurrent_open_close():
    from hotcoin.config import ExecutionConfig
    from hotcoin.execution.portfolio_risk import PortfolioRisk

    risk = PortfolioRisk(ExecutionConfig(
        initial_capital=100000,
        max_concurrent_positions=50,
    ))
    errors = []

    def _writer(tid):
        try:
            for i in range(20):
                sym = f"T{tid}_{i}USDT"
                ok, _ = risk.can_open(sym, 100)
                if ok:
                    risk.open_position(sym, "BUY", 100.0, 1.0)
                    time.sleep(0.001)
                    risk.close_position(sym, 101.0, "test")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=_writer, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert risk.num_positions == 0


def test_portfolio_risk_summary_under_concurrent_writes():
    from hotcoin.config import ExecutionConfig
    from hotcoin.execution.portfolio_risk import PortfolioRisk

    risk = PortfolioRisk(ExecutionConfig(initial_capital=10000))
    errors = []

    def _reader():
        try:
            for _ in range(50):
                s = risk.get_summary()
                assert "positions" in s
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    def _writer():
        try:
            for i in range(20):
                sym = f"RW{i}USDT"
                risk.open_position(sym, "BUY", 50.0, 1.0)
                time.sleep(0.001)
                risk.close_position(sym, 51.0, "test")
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=_reader)
    t2 = threading.Thread(target=_writer)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert not errors


# ---- OrderExecutor metrics lock ----

def test_runtime_metrics_concurrent():
    from hotcoin.config import ExecutionConfig
    from hotcoin.execution.order_executor import OrderExecutor

    executor = OrderExecutor(ExecutionConfig(use_paper_trading=True))
    errors = []

    def _recorder():
        try:
            for _ in range(50):
                executor._record_runtime_event("order_attempt")
                executor._record_runtime_event("order_success")
        except Exception as e:
            errors.append(e)

    def _reader():
        try:
            for _ in range(50):
                m = executor.get_runtime_metrics()
                assert "order_attempts_5m" in m
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=_recorder),
        threading.Thread(target=_recorder),
        threading.Thread(target=_reader),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


# ---- process_signals uses current_prices ----

def test_process_signals_uses_passed_prices(monkeypatch):
    from hotcoin.config import HotCoinConfig, ExecutionConfig
    from hotcoin.discovery.candidate_pool import CandidatePool, HotCoin
    from hotcoin.execution.spot_engine import HotCoinSpotEngine
    from hotcoin.engine.signal_worker import TradeSignal

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        config = HotCoinConfig(
            db_path=db_path,
            execution=ExecutionConfig(
                initial_capital=10000,
                use_paper_trading=True,
                enable_order_execution=True,
            ),
        )
        pool = CandidatePool(db_path, config.discovery)
        engine = HotCoinSpotEngine(config, pool)

        api_calls = {"n": 0}
        original_get_price = engine.executor.get_current_price

        def _tracking_price(symbol):
            api_calls["n"] += 1
            return original_get_price(symbol)

        monkeypatch.setattr(engine.executor, "get_current_price", _tracking_price)

        sig = TradeSignal(symbol="ETHUSDT", action="BUY", strength=80, confidence=0.8)
        engine.process_signals(
            [sig], allow_open=True,
            current_prices={"ETHUSDT": 2000.0},
        )

        assert api_calls["n"] == 0, "Should not call API when current_prices provided"


# ---- _order_history trimmed in live mode ----

def test_order_history_trimmed_in_send(monkeypatch):
    from hotcoin.config import ExecutionConfig
    from hotcoin.execution.order_executor import OrderExecutor

    executor = OrderExecutor(ExecutionConfig(use_paper_trading=False))
    executor._api_key = "test"
    executor._api_secret = "test"
    executor._max_history = 5
    executor._order_history = [{"id": i} for i in range(5)]

    class FakeResp:
        status_code = 200
        def json(self):
            return {"status": "FILLED", "orderId": 999}

    monkeypatch.setattr("hotcoin.execution.order_executor.requests.post", lambda *a, **kw: FakeResp())

    executor._send_spot_order({"symbol": "ETHUSDT", "side": "BUY", "type": "MARKET"})

    assert len(executor._order_history) == 5
    assert executor._order_history[-1]["orderId"] == 999
