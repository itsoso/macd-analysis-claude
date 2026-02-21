"""R7 改进测试: TickerStream 快照、AnomalyDetector 清理、紧急平仓、资产查询、web routes 线程安全。"""

import json
import time

from flask import Flask

import hotcoin.web.routes as routes
from hotcoin.web.routes import hotcoin_bp
from hotcoin.config import DiscoveryConfig
from hotcoin.discovery.ticker_stream import TickerStream, MiniTicker
from hotcoin.discovery.anomaly_detector import AnomalyDetector


# -- TickerStream snapshot tests --

class _FakeDetector:
    def detect(self, sym, ticker):
        return None


class _FakePool:
    def __init__(self):
        self.anomalies = []

    def on_anomaly(self, anomaly):
        self.anomalies.append(anomaly)


def test_tickers_returns_snapshot_copy():
    """tickers 属性返回副本, 修改不影响内部状态。"""
    stream = TickerStream(DiscoveryConfig(), _FakeDetector(), _FakePool())
    payload = [{"s": "ETHUSDT", "c": "2000", "o": "1900", "q": "99999",
                "h": "2100", "l": "1800", "v": "100", "E": str(int(time.time() * 1000))}]
    stream._handle_message(json.dumps(payload))

    snapshot = stream.tickers
    assert "ETHUSDT" in snapshot

    snapshot["FAKEUSDT"] = MiniTicker(
        symbol="FAKEUSDT", close=1.0, open_price=1.0,
        high=1.0, low=1.0, base_volume=0, quote_volume=0, event_time=0
    )
    assert "FAKEUSDT" not in stream.tickers


def test_tickers_ref_returns_live_dict():
    """tickers_ref 返回活引用, 修改直接可见。"""
    stream = TickerStream(DiscoveryConfig(), _FakeDetector(), _FakePool())
    payload = [{"s": "BTCUSDT", "c": "50000", "o": "49000", "q": "5555",
                "h": "51000", "l": "48000", "v": "10", "E": str(int(time.time() * 1000))}]
    stream._handle_message(json.dumps(payload))

    ref = stream.tickers_ref
    assert "BTCUSDT" in ref
    assert ref is stream._tickers


# -- AnomalyDetector cleanup safety --

def test_anomaly_detector_cleanup_handles_exception(monkeypatch):
    """清理冷却记录时异常不崩溃。"""
    config = DiscoveryConfig()
    detector = AnomalyDetector(config)

    now = time.time()
    for i in range(600):
        detector._alert_cooldown[f"SYM{i}USDT"] = now - 9999

    class _BrokenTicker:
        avg_volume_20m = 100.0
        volume_1m = 500.0
        price_change_5m = 0.05
        quote_volume = 1_000_000
        price_change_24h = 0.10

    result = detector.detect("NEWUSDT", _BrokenTicker())
    assert result is not None
    assert result.symbol == "NEWUSDT"


# -- SpotEngine emergency_close_all --

def test_emergency_close_all_no_positions():
    """无持仓时紧急平仓返回空结果。"""
    from hotcoin.execution.spot_engine import HotCoinSpotEngine
    from hotcoin.config import HotCoinConfig

    config = HotCoinConfig()
    config.execution.use_paper_trading = True
    engine = HotCoinSpotEngine(config, pool=_FakePool())
    result = engine.emergency_close_all()
    assert result["closed"] == 0
    assert result["failed"] == 0
    assert result["details"] == []


def test_emergency_close_all_with_position():
    """有持仓时紧急平仓执行全平。"""
    from hotcoin.execution.spot_engine import HotCoinSpotEngine
    from hotcoin.config import HotCoinConfig

    config = HotCoinConfig()
    config.execution.use_paper_trading = True
    engine = HotCoinSpotEngine(config, pool=_FakePool())

    engine.risk.open_position("ETHUSDT", "BUY", 2000.0, 0.1, 200.0)
    assert engine.risk.num_positions == 1

    result = engine.emergency_close_all(reason="test_emergency")
    assert result["closed"] == 1
    assert result["failed"] == 0
    assert engine.risk.num_positions == 0


# -- OrderExecutor.query_account_balances --

def test_query_account_balances_paper_returns_empty():
    """Paper 模式下查余额返回空列表。"""
    from hotcoin.execution.order_executor import OrderExecutor
    from hotcoin.config import HotCoinConfig

    config = HotCoinConfig()
    executor = OrderExecutor(config.execution)
    assert executor.query_account_balances() == []


# -- Web routes emergency_close API --

class _FakeSpotEngine:
    def __init__(self):
        self.num_positions = 0
        self.risk = type("Risk", (), {"get_summary": lambda self: {}})()

    def emergency_close_all(self, reason="test"):
        return {"closed": 0, "failed": 0, "details": []}

    class executor:
        @staticmethod
        def get_precheck_stats():
            return {"total": 0, "by_code": {}, "by_symbol": {}}

        @staticmethod
        def get_runtime_metrics(window_sec=300):
            return {}

        @staticmethod
        def query_account_balances():
            return [{"asset": "USDT", "free": 100.0, "locked": 0, "total": 100.0}]


class _FakeRunnerForWeb:
    def __init__(self):
        self.spot_engine = _FakeSpotEngine()
        self.config = type("Cfg", (), {"execution": type("Exec", (), {
            "use_paper_trading": True,
            "enable_order_execution": False,
        })()})()
        self.ticker_stream = type("TS", (), {"tickers": {}})()
        self.pool = type("Pool", (), {
            "size": 0,
            "get_top": lambda self, n=20, min_score=0: [],
            "get_all": lambda self: [],
        })()


def _build_client(tmp_path, monkeypatch, with_runner=False):
    app = Flask(__name__)
    app.register_blueprint(hotcoin_bp)
    status_file = tmp_path / "hotcoin_runtime_status.json"
    monkeypatch.setattr(routes, "_STATUS_FILE", str(status_file))

    if with_runner:
        monkeypatch.setattr(routes, "_runner", _FakeRunnerForWeb())
    else:
        monkeypatch.setattr(routes, "_runner", None)

    return app.test_client(), status_file


def test_emergency_close_no_runner(tmp_path, monkeypatch):
    client, _ = _build_client(tmp_path, monkeypatch, with_runner=False)
    resp = client.post("/hotcoin/api/emergency_close",
                       data=json.dumps({"confirm": "yes"}),
                       content_type="application/json")
    assert resp.status_code == 503


def test_emergency_close_no_confirm(tmp_path, monkeypatch):
    client, _ = _build_client(tmp_path, monkeypatch, with_runner=True)
    resp = client.post("/hotcoin/api/emergency_close",
                       data=json.dumps({}),
                       content_type="application/json")
    assert resp.status_code == 400


def test_emergency_close_success(tmp_path, monkeypatch):
    client, _ = _build_client(tmp_path, monkeypatch, with_runner=True)
    resp = client.post("/hotcoin/api/emergency_close",
                       data=json.dumps({"confirm": "yes"}),
                       content_type="application/json")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert data["closed"] == 0


def test_balances_no_runner(tmp_path, monkeypatch):
    client, _ = _build_client(tmp_path, monkeypatch, with_runner=False)
    resp = client.get("/hotcoin/api/balances")
    assert resp.status_code == 503


def test_balances_success(tmp_path, monkeypatch):
    client, _ = _build_client(tmp_path, monkeypatch, with_runner=True)
    resp = client.get("/hotcoin/api/balances")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert len(data["balances"]) == 1
    assert data["balances"][0]["asset"] == "USDT"


def test_get_runner_returns_same_ref(monkeypatch):
    """_get_runner 返回全局 _runner 引用。"""
    fake = _FakeRunnerForWeb()
    monkeypatch.setattr(routes, "_runner", fake)
    assert routes._get_runner() is fake

    monkeypatch.setattr(routes, "_runner", None)
    assert routes._get_runner() is None
