"""
OrderExecutor 精度相关测试
"""

from types import SimpleNamespace

from hotcoin.config import ExecutionConfig
from hotcoin.execution.order_executor import OrderExecutor


def test_spot_market_buy_uses_quote_precision_floor(monkeypatch):
    cfg = ExecutionConfig(use_paper_trading=False)
    executor = OrderExecutor(cfg)

    monkeypatch.setattr(
        executor._exchange_info,
        "get",
        lambda _symbol: SimpleNamespace(
            qty_precision=6,
            price_precision=4,
            quote_precision=3,
            qty_step=0.000001,
            market_qty_step=0.00001,
            price_tick=0.0001,
            min_notional=10.0,
            min_qty=0.0001,
            market_min_qty=0.0002,
        ),
    )

    captured = {}

    def _fake_send(params):
        captured.update(params)
        return {"ok": True}

    monkeypatch.setattr(executor, "_send_spot_order", _fake_send)
    executor.spot_market_buy("ETHUSDT", 12.34567)

    assert captured["quoteOrderQty"] == "12.345"
    assert captured["side"] == "BUY"
    assert captured["type"] == "MARKET"


def test_format_uses_floor_step_and_tick(monkeypatch):
    cfg = ExecutionConfig(use_paper_trading=False)
    executor = OrderExecutor(cfg)

    monkeypatch.setattr(
        executor._exchange_info,
        "get",
        lambda _symbol: SimpleNamespace(
            qty_precision=2,
            price_precision=4,
            quote_precision=2,
            qty_step=0.01,
            market_qty_step=0.1,
            price_tick=0.0005,
            min_notional=5.0,
            min_qty=0.01,
            market_min_qty=0.2,
        ),
    )

    assert executor.format_quantity("ETHUSDT", 1.239) == "1.23"
    assert executor.format_price("ETHUSDT", 123.45678) == "123.4565"


def test_market_buy_precheck_min_notional(monkeypatch):
    cfg = ExecutionConfig(use_paper_trading=False)
    executor = OrderExecutor(cfg)

    monkeypatch.setattr(
        executor._exchange_info,
        "get",
        lambda _symbol: SimpleNamespace(
            qty_precision=6,
            price_precision=2,
            quote_precision=2,
            qty_step=0.000001,
            market_qty_step=0.00001,
            price_tick=0.01,
            min_notional=10.0,
            min_qty=0.0001,
            market_min_qty=0.0002,
        ),
    )

    sent = {"n": 0}

    def _fake_send(_params):
        sent["n"] += 1
        return {"ok": True}

    monkeypatch.setattr(executor, "_send_spot_order", _fake_send)
    out = executor.spot_market_buy("ETHUSDT", 9.99)

    assert out.get("code") == "PRECHECK_FAILED"
    assert out.get("precheck_code") == "BUY_MIN_NOTIONAL"
    assert sent["n"] == 0


def test_market_sell_precheck_min_qty(monkeypatch):
    cfg = ExecutionConfig(use_paper_trading=False)
    executor = OrderExecutor(cfg)

    monkeypatch.setattr(
        executor._exchange_info,
        "get",
        lambda _symbol: SimpleNamespace(
            qty_precision=4,
            price_precision=2,
            quote_precision=2,
            qty_step=0.001,
            market_qty_step=0.001,
            price_tick=0.01,
            min_notional=5.0,
            min_qty=0.01,
            market_min_qty=0.01,
        ),
    )
    monkeypatch.setattr(executor, "get_avg_price", lambda _symbol: 100.0)
    monkeypatch.setattr(executor, "get_current_price", lambda _symbol: 90.0)

    sent = {"n": 0}

    def _fake_send(_params):
        sent["n"] += 1
        return {"ok": True}

    monkeypatch.setattr(executor, "_send_spot_order", _fake_send)
    out = executor.spot_market_sell("ETHUSDT", 0.0099)

    assert out.get("code") == "PRECHECK_FAILED"
    assert out.get("precheck_code") == "SELL_MIN_QTY"
    assert sent["n"] == 0


def test_market_quantity_prefers_market_lot_size(monkeypatch):
    cfg = ExecutionConfig(use_paper_trading=False)
    executor = OrderExecutor(cfg)

    monkeypatch.setattr(
        executor._exchange_info,
        "get",
        lambda _symbol: SimpleNamespace(
            qty_precision=6,
            price_precision=2,
            quote_precision=2,
            qty_step=0.001,
            market_qty_step=0.01,
            price_tick=0.01,
            min_notional=5.0,
            min_qty=0.001,
            market_min_qty=0.01,
        ),
    )

    assert executor.format_quantity("ETHUSDT", 1.239, market=False) == "1.239000"
    assert executor.format_quantity("ETHUSDT", 1.239, market=True) == "1.23"


def test_precheck_stats_increment(monkeypatch):
    cfg = ExecutionConfig(use_paper_trading=False)
    executor = OrderExecutor(cfg)

    monkeypatch.setattr(
        executor._exchange_info,
        "get",
        lambda _symbol: SimpleNamespace(
            qty_precision=6,
            price_precision=2,
            quote_precision=2,
            qty_step=0.000001,
            market_qty_step=0.00001,
            price_tick=0.01,
            min_notional=10.0,
            min_qty=0.0001,
            market_min_qty=0.0002,
        ),
    )
    monkeypatch.setattr(executor, "_send_spot_order", lambda _params: {"ok": True})

    _ = executor.spot_market_buy("ETHUSDT", 8.0)
    _ = executor.spot_market_buy("ETHUSDT", 9.0)

    stats = executor.get_precheck_stats()
    assert stats["total"] == 2
    assert stats["by_code"]["BUY_MIN_NOTIONAL"] == 2
    assert stats["by_symbol"]["ETHUSDT"]["BUY_MIN_NOTIONAL"] == 2


def test_paper_buy_prefers_hint_price_without_price_fetch(monkeypatch):
    cfg = ExecutionConfig(use_paper_trading=True)
    executor = OrderExecutor(cfg)

    monkeypatch.setattr(
        executor._exchange_info,
        "get",
        lambda _symbol: SimpleNamespace(
            qty_precision=6,
            price_precision=2,
            quote_precision=2,
            qty_step=0.000001,
            market_qty_step=0.00001,
            price_tick=0.01,
            min_notional=10.0,
            min_qty=0.0001,
            market_min_qty=0.0002,
        ),
    )

    def _should_not_fetch(_symbol):
        raise AssertionError("paper 模式提供 hint_price 时不应再请求价格")

    monkeypatch.setattr(executor, "get_current_price", _should_not_fetch)

    out = executor.spot_market_buy("ETHUSDT", 20.0, hint_price=2000.0)
    assert out["price"] == 2000.0
    assert out["qty"] > 0


def test_market_buy_dedup_not_set_when_send_failed(monkeypatch):
    cfg = ExecutionConfig(use_paper_trading=False)
    executor = OrderExecutor(cfg)

    monkeypatch.setattr(
        executor._exchange_info,
        "get",
        lambda _symbol: SimpleNamespace(
            qty_precision=6,
            price_precision=2,
            quote_precision=2,
            qty_step=0.000001,
            market_qty_step=0.00001,
            price_tick=0.01,
            min_notional=10.0,
            min_qty=0.0001,
            market_min_qty=0.0002,
        ),
    )
    monkeypatch.setattr(executor, "_send_spot_order", lambda _params: {"error": "network"})

    out1 = executor.spot_market_buy("ETHUSDT", 20.0)
    out2 = executor.spot_market_buy("ETHUSDT", 20.0)

    assert out1.get("error") == "network"
    assert out2.get("code") != "DEDUP_REJECTED"


def test_market_buy_dedup_set_after_success(monkeypatch):
    cfg = ExecutionConfig(use_paper_trading=False)
    executor = OrderExecutor(cfg)

    monkeypatch.setattr(
        executor._exchange_info,
        "get",
        lambda _symbol: SimpleNamespace(
            qty_precision=6,
            price_precision=2,
            quote_precision=2,
            qty_step=0.000001,
            market_qty_step=0.00001,
            price_tick=0.01,
            min_notional=10.0,
            min_qty=0.0001,
            market_min_qty=0.0002,
        ),
    )
    monkeypatch.setattr(executor, "_send_spot_order", lambda _params: {"status": "FILLED"})

    out1 = executor.spot_market_buy("ETHUSDT", 20.0)
    out2 = executor.spot_market_buy("ETHUSDT", 20.0)

    assert out1.get("status") == "FILLED"
    assert out2.get("code") == "DEDUP_REJECTED"


def test_precheck_symbol_stats_pruned_by_lru(monkeypatch):
    cfg = ExecutionConfig(use_paper_trading=False)
    executor = OrderExecutor(cfg)
    monkeypatch.setattr(executor, "MAX_PRECHECK_SYMBOLS", 2, raising=False)

    ts = {"v": 0}

    def _fake_time():
        ts["v"] += 1
        return float(ts["v"])

    monkeypatch.setattr("hotcoin.execution.order_executor.time.time", _fake_time)

    executor._record_precheck_failure("AAAUSDT", "BUY_MIN_NOTIONAL")
    executor._record_precheck_failure("BBBUSDT", "BUY_MIN_NOTIONAL")
    executor._record_precheck_failure("CCCUSDT", "BUY_MIN_NOTIONAL")

    stats = executor.get_precheck_stats()
    assert len(stats["by_symbol"]) == 2
    assert "AAAUSDT" not in stats["by_symbol"]
