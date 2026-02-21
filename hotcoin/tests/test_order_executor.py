"""
OrderExecutor 精度相关测试
"""

from types import SimpleNamespace

from hotcoin.config import ExecutionConfig
from hotcoin.execution.order_executor import OrderExecutor


def test_spot_market_buy_uses_quote_precision(monkeypatch):
    cfg = ExecutionConfig(use_paper_trading=False)
    executor = OrderExecutor(cfg)

    monkeypatch.setattr(
        executor._exchange_info,
        "get",
        lambda _symbol: SimpleNamespace(
            qty_precision=6,
            price_precision=4,
            quote_precision=3,
            min_notional=10.0,
        ),
    )

    captured = {}

    def _fake_send(params):
        captured.update(params)
        return {"ok": True}

    monkeypatch.setattr(executor, "_send_spot_order", _fake_send)
    executor.spot_market_buy("ETHUSDT", 12.34567)

    assert captured["quoteOrderQty"] == "12.346"
    assert captured["side"] == "BUY"
    assert captured["type"] == "MARKET"
