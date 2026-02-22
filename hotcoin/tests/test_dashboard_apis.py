"""Dashboard 增强 API 测试: /api/positions, /api/trades, /api/chart?scores=1"""

import json
import os
import tempfile
import time

import pytest
from flask import Flask

from hotcoin.web.routes import hotcoin_bp


def _make_app():
    app = Flask(__name__)
    app.register_blueprint(hotcoin_bp, url_prefix="/hotcoin")
    return app


class TestPositionsAPI:
    def test_positions_without_runner(self):
        app = _make_app()
        with app.test_client() as c:
            resp = c.get("/hotcoin/api/positions")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["ok"] is True
            assert data["positions"] == []

    def test_positions_response_structure(self):
        app = _make_app()
        with app.test_client() as c:
            resp = c.get("/hotcoin/api/positions")
            data = resp.get_json()
            assert "pnl_summary" in data
            assert "ts" in data


class TestTradesAPI:
    def test_trades_without_runner_and_no_files(self):
        app = _make_app()
        with app.test_client() as c:
            resp = c.get("/hotcoin/api/trades")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["ok"] is True
            assert isinstance(data["trades"], list)

    def test_trades_reads_from_jsonl(self):
        tmpdir = tempfile.mkdtemp()
        trades_file = os.path.join(tmpdir, "hotcoin_trades_20260218.jsonl")
        record = {
            "symbol": "TESTUSDT", "side": "BUY",
            "entry_price": 1.0, "exit_price": 1.1,
            "qty": 10, "pnl": 1.0, "pnl_pct": 0.1,
            "holding_sec": 60, "reason": "tp",
            "entry_time": time.time() - 60, "exit_time": time.time(),
        }
        with open(trades_file, "w") as f:
            f.write(json.dumps(record) + "\n")

        import hotcoin.web.routes as routes
        orig = routes._TRADES_GLOB_DIR
        routes._TRADES_GLOB_DIR = tmpdir
        try:
            app = _make_app()
            with app.test_client() as c:
                resp = c.get("/hotcoin/api/trades?limit=10")
                data = resp.get_json()
                assert data["ok"] is True
                assert len(data["trades"]) == 1
                assert data["trades"][0]["symbol"] == "TESTUSDT"
        finally:
            routes._TRADES_GLOB_DIR = orig


class TestChartScores:
    def test_chart_without_scores_param(self):
        """不传 scores=1 时不应返回 scores 字段（或为空）。"""
        from unittest.mock import patch

        app = _make_app()
        fake_klines = [
            {"time": int(time.time()) - 300 * i,
             "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000}
            for i in range(5)
        ]
        fake_klines.sort(key=lambda k: k["time"])

        with patch("hotcoin.web.routes._load_chart_klines", return_value=fake_klines), \
             patch("hotcoin.web.routes._load_order_markers", return_value=([], [])):
            with app.test_client() as c:
                resp = c.get("/hotcoin/api/chart?symbol=TESTUSDT&interval=5m&days=1")
                data = resp.get_json()
                assert data.get("scores") == {}

    def test_chart_with_scores_param(self):
        """传 scores=1 时应调用 _compute_bar_scores。"""
        from unittest.mock import patch

        app = _make_app()
        fake_klines = [
            {"time": int(time.time()) - 300 * i,
             "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000}
            for i in range(5)
        ]
        fake_klines.sort(key=lambda k: k["time"])
        fake_scores = {str(fake_klines[0]["time"]): {"ss": 45.0, "bs": 20.0}}

        with patch("hotcoin.web.routes._load_chart_klines", return_value=fake_klines), \
             patch("hotcoin.web.routes._load_order_markers", return_value=([], [])), \
             patch("hotcoin.web.routes._compute_bar_scores", return_value=fake_scores):
            with app.test_client() as c:
                resp = c.get("/hotcoin/api/chart?symbol=TESTUSDT&interval=5m&days=1&scores=1")
                data = resp.get_json()
                assert len(data["scores"]) == 1
                ts_key = list(data["scores"].keys())[0]
                assert data["scores"][ts_key]["ss"] == 45.0
