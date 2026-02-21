"""
trace_id 贯通、错误分类、订单对账 测试
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---- trace_id flows through TradeSignal ----

def test_trade_signal_trace_id():
    from hotcoin.engine.signal_worker import TradeSignal

    sig = TradeSignal(symbol="ETHUSDT", action="BUY", trace_id="abc123")
    assert sig.trace_id == "abc123"


def test_trade_signal_trace_id_default():
    from hotcoin.engine.signal_worker import TradeSignal

    sig = TradeSignal(symbol="ETHUSDT", action="HOLD")
    assert sig.trace_id == ""


# ---- error classification ----

def test_classify_retryable_http():
    from hotcoin.execution.order_executor import OrderExecutor

    for code in (429, 418, 500, 502, 503, 504):
        assert OrderExecutor.classify_order_error(code, 0) == "retryable"


def test_classify_retryable_api_codes():
    from hotcoin.execution.order_executor import OrderExecutor

    for api_code in (-1000, -1001, -1003, -1006, -1007, -1015):
        assert OrderExecutor.classify_order_error(400, api_code) == "retryable"


def test_classify_non_retryable_client_error():
    from hotcoin.execution.order_executor import OrderExecutor

    assert OrderExecutor.classify_order_error(400, -1100) == "non_retryable"
    assert OrderExecutor.classify_order_error(401, 0) == "non_retryable"
    assert OrderExecutor.classify_order_error(403, 0) == "non_retryable"


def test_classify_success():
    from hotcoin.execution.order_executor import OrderExecutor

    assert OrderExecutor.classify_order_error(200, 0) == "non_retryable"


# ---- order_executor query/cancel (paper mode) ----

def test_query_open_orders_paper():
    from hotcoin.config import ExecutionConfig
    from hotcoin.execution.order_executor import OrderExecutor

    executor = OrderExecutor(ExecutionConfig(use_paper_trading=True))
    assert executor.query_open_orders() == []


def test_cancel_order_paper():
    from hotcoin.config import ExecutionConfig
    from hotcoin.execution.order_executor import OrderExecutor

    executor = OrderExecutor(ExecutionConfig(use_paper_trading=True))
    result = executor.cancel_order("ETHUSDT", 12345)
    assert result["status"] == "CANCELED"


# ---- spot_engine reconcile (paper mode) ----

def test_reconcile_paper_noop():
    import os, tempfile
    from hotcoin.config import HotCoinConfig, ExecutionConfig
    from hotcoin.discovery.candidate_pool import CandidatePool
    from hotcoin.execution.spot_engine import HotCoinSpotEngine

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        config = HotCoinConfig(
            db_path=db_path,
            execution=ExecutionConfig(use_paper_trading=True),
        )
        pool = CandidatePool(db_path, config.discovery)
        engine = HotCoinSpotEngine(config, pool)
        result = engine.reconcile_open_orders()
        assert result["stale_orders"] == 0
        assert result["canceled"] == 0


# ---- emit_event includes trace_id ----

def test_emit_event_trace_id():
    import os, tempfile
    from hotcoin.config import HotCoinConfig, ExecutionConfig
    from hotcoin.discovery.candidate_pool import CandidatePool
    from hotcoin.execution.spot_engine import HotCoinSpotEngine

    captured = []

    def _sink(event_type, payload, trace_id=""):
        captured.append({"event_type": event_type, "trace_id": trace_id})

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        config = HotCoinConfig(
            db_path=db_path,
            execution=ExecutionConfig(use_paper_trading=True),
        )
        pool = CandidatePool(db_path, config.discovery)
        engine = HotCoinSpotEngine(config, pool, event_sink=_sink)
        engine._emit_event("test_event", {"foo": 1}, trace_id="trace_abc")

        assert len(captured) == 1
        assert captured[0]["trace_id"] == "trace_abc"


# ---- send_spot_order error_class in result ----

def test_send_order_timeout_retryable(monkeypatch):
    import requests as req_lib
    from hotcoin.config import ExecutionConfig
    from hotcoin.execution.order_executor import OrderExecutor

    executor = OrderExecutor(ExecutionConfig(use_paper_trading=False))
    executor._api_key = "test"
    executor._api_secret = "test"

    def _timeout_post(*args, **kwargs):
        raise req_lib.exceptions.Timeout("read timed out")

    monkeypatch.setattr("hotcoin.execution.order_executor.requests.post", _timeout_post)
    result = executor._send_spot_order({"symbol": "ETHUSDT", "side": "BUY", "type": "MARKET"})

    assert result.get("error_class") == "retryable"
    assert "timeout" in result.get("error", "").lower()
