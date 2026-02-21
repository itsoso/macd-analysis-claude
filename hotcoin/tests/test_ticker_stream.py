import json

from hotcoin.config import DiscoveryConfig
from hotcoin.discovery.ticker_stream import TickerStream


class _FakeDetector:
    def detect(self, sym, ticker):
        if sym == "ETHUSDT":
            return {"symbol": sym, "close": ticker.close}
        return None


class _FakePool:
    def __init__(self):
        self.anomalies = []

    def on_anomaly(self, anomaly):
        self.anomalies.append(anomaly)


def test_handle_message_ignores_non_list_payload():
    stream = TickerStream(DiscoveryConfig(), _FakeDetector(), _FakePool())

    stream._handle_message('{"result": null}')

    assert stream.tickers == {}


def test_handle_message_tolerates_malformed_items_and_keeps_valid_one():
    pool = _FakePool()
    stream = TickerStream(DiscoveryConfig(), _FakeDetector(), pool)

    payload = [
        {"s": "ETHUSDT", "c": "2000", "o": "1900", "q": "12345", "h": "2100", "l": "1800", "v": "100", "E": "bad-ts"},
        "not-a-dict",
        {"s": "BTCUSDT", "c": "bad-close", "o": "1", "q": "2"},
        {"s": "FOO", "c": "1", "o": "1", "q": "1"},
    ]

    stream._handle_message(json.dumps(payload))

    assert "ETHUSDT" in stream.tickers
    t = stream.tickers["ETHUSDT"]
    assert t.close == 2000.0
    assert t.event_time > 0
    assert len(pool.anomalies) == 1
    assert pool.anomalies[0]["symbol"] == "ETHUSDT"
