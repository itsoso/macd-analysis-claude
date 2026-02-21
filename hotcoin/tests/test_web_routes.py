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

    def get_runtime_metrics(self, window_sec=300):
        return {
            "window_sec": int(window_sec),
            "order_attempts_5m": 7,
            "precheck_failures_5m": 2,
            "dedup_rejects_5m": 1,
            "order_errors_5m": 1,
            "order_success_5m": 4,
            "precheck_fail_rate_5m": 0.2857,
            "order_error_rate_5m": 0.1429,
        }


class _FakeRisk:
    def __init__(self, halted=False):
        self._halted = halted

    def get_summary(self):
        return {"halted": bool(self._halted)}


class _FakeSpotEngine:
    def __init__(self, stats, halted=False):
        self.executor = _FakeExecutor(stats)
        self.risk = _FakeRisk(halted=halted)
        self.num_positions = 0


class _FakeRunner:
    def __init__(self, stats, halted=False, tickers=None):
        self.spot_engine = _FakeSpotEngine(stats, halted=halted)
        self.config = type("Cfg", (), {"execution": type("Exec", (), {
            "use_paper_trading": True,
            "enable_order_execution": False,
        })()})()
        self.ticker_stream = type("TS", (), {"tickers": tickers if tickers is not None else {}})()
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
    assert data["execution_metrics"]["order_attempts_5m"] == 0
    assert data["engine_state"] == "stopped"
    assert data["state_recovery_pending"] is None
    assert data["can_open_new_positions"] is False


def test_execution_metrics_default_when_no_runner_and_no_file(tmp_path, monkeypatch):
    client, _ = _build_client(tmp_path, monkeypatch)

    resp = client.get("/hotcoin/api/execution_metrics")
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["source"] == "none"
    assert data["order_attempts_5m"] == 0
    assert data["order_errors_5m"] == 0


def test_execution_metrics_from_status_file(tmp_path, monkeypatch):
    client, status_file = _build_client(tmp_path, monkeypatch)
    status_file.write_text(
        json.dumps(
            {
                "ts": 1700000000,
                "execution_metrics": {
                    "order_attempts_5m": 10,
                    "order_errors_5m": 2,
                    "order_error_rate_5m": 0.2,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    resp = client.get("/hotcoin/api/execution_metrics")
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["source"] == "status_file"
    assert data["order_attempts_5m"] == 10
    assert data["order_errors_5m"] == 2
    assert data["ts"] == 1700000000


def test_execution_metrics_runner_has_priority_over_file(tmp_path, monkeypatch):
    client, status_file = _build_client(tmp_path, monkeypatch)
    status_file.write_text(
        json.dumps(
            {"ts": 1700000000, "execution_metrics": {"order_attempts_5m": 99}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        routes,
        "_runner",
        _FakeRunner({"total": 0, "by_code": {}, "by_symbol": {}}),
    )

    resp = client.get("/hotcoin/api/execution_metrics")
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["source"] == "runner"
    assert data["order_attempts_5m"] == 7
    assert data["order_errors_5m"] == 1
    assert data["ts"] >= time.time() - 10


def test_health_stopped_when_runner_not_attached(tmp_path, monkeypatch):
    client, _ = _build_client(tmp_path, monkeypatch)

    resp = client.get("/hotcoin/health")
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["status"] == "stopped"
    assert data["can_trade"] is False
    assert data["runner_attached"] is False
    assert "runner_not_attached" in data["reasons"]


def test_health_ok_when_tradeable_and_fresh(tmp_path, monkeypatch):
    client, status_file = _build_client(tmp_path, monkeypatch)
    monkeypatch.setattr(
        routes,
        "_runner",
        _FakeRunner(
            {"total": 0, "by_code": {}, "by_symbol": {}},
            halted=False,
            tickers={"ETHUSDT": object()},
        ),
    )
    status_file.write_text(
        json.dumps(
            {
                "ts": time.time(),
                "engine_state": "tradeable",
                "engine_state_reasons": [],
                "ws_connected": True,
                "freshness": {"latest_ticker_age_sec": 12.3, "priced_symbols": 8},
                "execution_metrics": {"order_errors_5m": 0, "order_attempts_5m": 9},
                "risk_halted": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    resp = client.get("/hotcoin/health")
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["status"] == "ok"
    assert data["can_trade"] is True
    assert data["engine_state"] == "tradeable"
    assert data["runner_attached"] is True
    assert data["status_snapshot_fresh"] is True


def test_health_blocked_when_risk_halted(tmp_path, monkeypatch):
    client, _ = _build_client(tmp_path, monkeypatch)
    monkeypatch.setattr(
        routes,
        "_runner",
        _FakeRunner(
            {"total": 0, "by_code": {}, "by_symbol": {}},
            halted=True,
            tickers={"ETHUSDT": object()},
        ),
    )

    resp = client.get("/hotcoin/health")
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["status"] == "blocked"
    assert data["can_trade"] is False
    assert data["risk_halted"] is True
    assert "risk_halted" in data["reasons"]


def test_chart_requires_symbol(tmp_path, monkeypatch):
    client, _ = _build_client(tmp_path, monkeypatch)
    resp = client.get("/hotcoin/api/chart")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"] == "invalid_symbol"


def test_chart_rejects_invalid_interval(tmp_path, monkeypatch):
    client, _ = _build_client(tmp_path, monkeypatch)
    resp = client.get("/hotcoin/api/chart?symbol=ETHUSDT&interval=10m")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"] == "invalid_interval"


def test_chart_returns_klines_and_markers(tmp_path, monkeypatch):
    client, _ = _build_client(tmp_path, monkeypatch)

    monkeypatch.setattr(
        routes,
        "_load_chart_klines",
        lambda symbol, interval, days, bars_limit=800: [
            {"time": 1700000000, "open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1, "volume": 100.0},
            {"time": 1700000060, "open": 1.1, "high": 1.3, "low": 1.0, "close": 1.25, "volume": 120.0},
        ],
    )
    monkeypatch.setattr(
        routes,
        "_load_order_markers",
        lambda symbol, since_ts, until_ts, limit=200, trace_id_filter="": (
            [
                {
                    "time": 1700000000,
                    "position": "belowBar",
                    "color": "#00d68f",
                    "shape": "arrowUp",
                    "text": "BUY open 10:00 @1.1",
                }
            ],
            [
                {
                    "time": 1700000000,
                    "time_text": "2023-11-14 10:00:00",
                    "side": "BUY",
                    "intent": "open",
                    "price": 1.1,
                    "qty": 10.0,
                    "source": "events",
                }
            ],
        ),
    )

    resp = client.get("/hotcoin/api/chart?symbol=ETHUSDT&interval=5m&days=3")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["symbol"] == "ETHUSDT"
    assert data["interval"] == "5m"
    assert len(data["klines"]) == 2
    assert len(data["markers"]) == 1
    assert len(data["trade_points"]) == 1


def test_chart_invalid_trace_id(tmp_path, monkeypatch):
    client, _ = _build_client(tmp_path, monkeypatch)
    resp = client.get("/hotcoin/api/chart?symbol=ETHUSDT&trace_id=bad*trace")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"] == "invalid_trace_id"


def test_chart_passes_trace_id_filter_to_loader(tmp_path, monkeypatch):
    client, _ = _build_client(tmp_path, monkeypatch)
    captured = {"trace_id_filter": None}

    monkeypatch.setattr(
        routes,
        "_load_chart_klines",
        lambda symbol, interval, days, bars_limit=800: [
            {"time": 1700000000, "open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1, "volume": 100.0},
            {"time": 1700000060, "open": 1.1, "high": 1.3, "low": 1.0, "close": 1.25, "volume": 120.0},
        ],
    )

    def _fake_load_order_markers(symbol, since_ts, until_ts, limit=200, trace_id_filter=""):
        captured["trace_id_filter"] = trace_id_filter
        return [], []

    monkeypatch.setattr(routes, "_load_order_markers", _fake_load_order_markers)

    resp = client.get("/hotcoin/api/chart?symbol=ETHUSDT&interval=5m&days=3&trace_id=cycle_001")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["trace_id_filter"] == "cycle_001"
    assert captured["trace_id_filter"] == "cycle_001"


def test_load_order_markers_links_related_events_by_trace_id(tmp_path, monkeypatch):
    events_file = tmp_path / "hotcoin_events.jsonl"
    status_file = tmp_path / "hotcoin_runtime_status.json"
    t0 = 1700000000.0
    trace_id = "cycle123"
    events = [
        {
            "ts": t0 - 30,
            "trace_id": trace_id,
            "event_type": "candidate_snapshot",
            "payload": {"top_symbols": ["ETHUSDT", "SOLUSDT"], "pool_size": 20, "candidate_count": 5},
        },
        {
            "ts": t0 - 20,
            "trace_id": trace_id,
            "event_type": "signal_snapshot",
            "payload": {
                "signals": [{"symbol": "ETHUSDT", "action": "BUY", "strength": 4}],
                "actionable_count": 1,
            },
        },
        {
            "ts": t0 - 10,
            "trace_id": trace_id,
            "event_type": "order_attempt",
            "payload": {"symbol": "ETHUSDT", "side": "BUY", "intent": "open", "qty": 1.2},
        },
        {
            "ts": t0,
            "trace_id": trace_id,
            "event_type": "order_result",
            "payload": {"symbol": "ETHUSDT", "side": "BUY", "intent": "open", "ok": True, "executed_price": 2010.5},
        },
        {
            "ts": t0 + 5,
            "trace_id": trace_id,
            "event_type": "order_result",
            "payload": {"symbol": "BTCUSDT", "side": "BUY", "intent": "open", "ok": True, "executed_price": 50000.0},
        },
    ]
    events_file.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n", encoding="utf-8")

    monkeypatch.setattr(routes, "_EVENTS_FILE", str(events_file))
    monkeypatch.setattr(routes, "_STATUS_FILE", str(status_file))
    monkeypatch.setattr(routes, "_runner", None)

    markers, points = routes._load_order_markers("ETHUSDT", t0 - 120, t0 + 120, limit=20)
    assert len(markers) == 1
    assert len(points) == 1
    p = points[0]
    assert p["trace_id"] == trace_id
    rel_types = [e["event_type"] for e in p["related_events"]]
    assert rel_types == ["candidate_snapshot", "signal_snapshot", "order_attempt", "order_result"]
    for e in p["related_events"]:
        assert isinstance(e.get("raw"), dict)
        assert e["raw"]["event_type"] == e["event_type"]


def test_load_order_markers_fallback_to_time_window_when_trace_missing(tmp_path, monkeypatch):
    events_file = tmp_path / "hotcoin_events.jsonl"
    status_file = tmp_path / "hotcoin_runtime_status.json"
    t0 = 1700001000.0
    events = [
        {
            "ts": t0 - 20,
            "event_type": "candidate_snapshot",
            "payload": {"top_symbols": ["ETHUSDT"], "pool_size": 12, "candidate_count": 3},
        },
        {
            "ts": t0 - 10,
            "event_type": "signal_snapshot",
            "payload": {
                "signals": [{"symbol": "ETHUSDT", "action": "BUY", "strength": 3}],
                "actionable_count": 1,
            },
        },
        {
            "ts": t0 - 3,
            "event_type": "order_attempt",
            "payload": {"symbol": "ETHUSDT", "side": "BUY", "intent": "open", "qty": 0.8},
        },
        {
            "ts": t0,
            "event_type": "order_result",
            "payload": {"symbol": "ETHUSDT", "side": "BUY", "intent": "open", "ok": True, "executed_price": 2021.2},
        },
    ]
    events_file.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n", encoding="utf-8")

    monkeypatch.setattr(routes, "_EVENTS_FILE", str(events_file))
    monkeypatch.setattr(routes, "_STATUS_FILE", str(status_file))
    monkeypatch.setattr(routes, "_runner", None)

    markers, points = routes._load_order_markers("ETHUSDT", t0 - 120, t0 + 120, limit=20)
    assert len(markers) == 1
    assert len(points) == 1
    p = points[0]
    assert p["trace_id"] == ""
    rel_types = [e["event_type"] for e in p["related_events"]]
    assert "order_result" in rel_types
    assert "signal_snapshot" in rel_types
    assert any(isinstance(e.get("raw"), dict) and "payload" in e["raw"] for e in p["related_events"])


def test_load_order_markers_trace_id_filter_only_keeps_target_trace(tmp_path, monkeypatch):
    events_file = tmp_path / "hotcoin_events.jsonl"
    status_file = tmp_path / "hotcoin_runtime_status.json"
    t0 = 1700002000.0
    events = [
        {
            "ts": t0 - 10,
            "trace_id": "target_trace",
            "event_type": "order_result",
            "payload": {"symbol": "ETHUSDT", "side": "BUY", "intent": "open", "ok": True, "executed_price": 2100.0},
        },
        {
            "ts": t0,
            "trace_id": "other_trace",
            "event_type": "order_result",
            "payload": {"symbol": "ETHUSDT", "side": "BUY", "intent": "open", "ok": True, "executed_price": 2200.0},
        },
    ]
    events_file.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n", encoding="utf-8")
    monkeypatch.setattr(routes, "_EVENTS_FILE", str(events_file))
    monkeypatch.setattr(routes, "_STATUS_FILE", str(status_file))
    monkeypatch.setattr(routes, "_runner", None)

    markers, points = routes._load_order_markers(
        "ETHUSDT",
        t0 - 120,
        t0 + 120,
        limit=20,
        trace_id_filter="target_trace",
    )
    assert len(markers) == 1
    assert len(points) == 1
    assert points[0]["trace_id"] == "target_trace"
