import json
import time

from flask import Flask

import hotcoin.web.routes as routes
from hotcoin.web.routes import hotcoin_bp


class _FakeExecutor:
    def __init__(self, stats):
        self._stats = stats

    def get_precheck_stats(self):
        return self._stats


class _FakeSpotEngine:
    def __init__(self, stats):
        self.executor = _FakeExecutor(stats)
        self.num_positions = 0


class _FakeRunner:
    def __init__(self, stats):
        self.spot_engine = _FakeSpotEngine(stats)
        self.config = type("Cfg", (), {"execution": type("Exec", (), {
            "use_paper_trading": True,
            "enable_order_execution": False,
        })()})()
        self.ticker_stream = type("TS", (), {"tickers": {}})()
        self.pool = type("Pool", (), {"size": 0, "get_top": lambda self, n=20, min_score=0: []})()


def _build_client(tmp_path, monkeypatch):
    app = Flask(__name__)
    app.register_blueprint(hotcoin_bp)

    status_file = tmp_path / "hotcoin_runtime_status.json"
    monkeypatch.setattr(routes, "_runner", None)
    monkeypatch.setattr(routes, "_STATUS_FILE", str(status_file))

    return app.test_client(), status_file


def test_precheck_stats_default_when_no_runner_and_no_file(tmp_path, monkeypatch):
    client, _ = _build_client(tmp_path, monkeypatch)

    resp = client.get("/hotcoin/api/precheck_stats")
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["source"] == "none"
    assert data["total"] == 0
    assert data["by_code"] == {}
    assert data["by_symbol"] == {}


def test_precheck_stats_from_status_file(tmp_path, monkeypatch):
    client, status_file = _build_client(tmp_path, monkeypatch)
    status_file.write_text(
        json.dumps(
            {
                "ts": 1700000000,
                "precheck_stats": {
                    "total": 3,
                    "by_code": {"BUY_MIN_NOTIONAL": 2, "SELL_MIN_QTY": 1},
                    "by_symbol": {"ETHUSDT": {"BUY_MIN_NOTIONAL": 2}, "SOLUSDT": {"SELL_MIN_QTY": 1}},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    resp = client.get("/hotcoin/api/precheck_stats")
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["source"] == "status_file"
    assert data["total"] == 3
    assert data["by_code"]["BUY_MIN_NOTIONAL"] == 2
    assert data["ts"] == 1700000000


def test_precheck_stats_runner_has_priority_over_file(tmp_path, monkeypatch):
    client, status_file = _build_client(tmp_path, monkeypatch)
    status_file.write_text(
        json.dumps({"ts": 1700000000, "precheck_stats": {"total": 99}}, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        routes,
        "_runner",
        _FakeRunner({"total": 5, "by_code": {"BUY_MIN_NOTIONAL": 5}, "by_symbol": {"ETHUSDT": {"BUY_MIN_NOTIONAL": 5}}}),
    )

    resp = client.get("/hotcoin/api/precheck_stats")
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["source"] == "runner"
    assert data["total"] == 5
    assert data["by_code"]["BUY_MIN_NOTIONAL"] == 5
    assert data["ts"] >= time.time() - 10


def test_status_returns_default_precheck_stats_for_legacy_status_file(tmp_path, monkeypatch):
    client, status_file = _build_client(tmp_path, monkeypatch)
    status_file.write_text(json.dumps({"pool_size": 1}, ensure_ascii=False), encoding="utf-8")

    resp = client.get("/hotcoin/api/status")
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["pool_size"] == 1
    assert "precheck_stats" in data
    assert data["precheck_stats"]["total"] == 0
