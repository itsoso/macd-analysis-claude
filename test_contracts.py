import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import app as app_module
from indicators import add_all_indicators
from live_config import TradingPhase
from live_trading_engine import LiveTradingEngine
from ma_indicators import add_moving_averages
from optimize_six_book import (
    compute_signals_six,
    calc_fusion_score_six,
    run_strategy,
    run_strategy_multi_tf,
)
from multi_tf_consensus import fuse_tf_scores


@pytest.fixture
def client():
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as c:
        with c.session_transaction() as sess:
            sess["logged_in"] = True
            sess["username"] = "admin"
        yield c


def _make_signal_fixture_data():
    """Deterministic OHLCV fixture used to freeze signal outputs."""
    np.random.seed(7)
    n = 240
    idx = pd.date_range("2025-01-01", periods=n, freq="h")

    base = 2600
    noise = np.random.normal(0, 12, n).cumsum()
    trend = np.linspace(-80, 120, n)

    close = np.maximum(base + noise + trend, 100)
    open_ = close + np.random.normal(0, 4, n)
    high = np.maximum(open_, close) + np.abs(np.random.normal(0, 8, n))
    low = np.minimum(open_, close) - np.abs(np.random.normal(0, 8, n))
    volume = np.abs(np.random.normal(900, 180, n)) + 100
    quote = volume * close

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": quote,
        },
        index=idx,
    )

    df = add_all_indicators(df)
    add_moving_averages(df, timeframe="1h")

    aux = (
        df.resample("8h")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "quote_volume": "sum",
            }
        )
        .dropna()
    )
    aux = add_all_indicators(aux)

    signals = compute_signals_six(df, "1h", {"1h": df, "8h": aux})
    return df, signals


def _make_backtest_cfg(mode="c6_veto_4"):
    return {
        "name": "contract-smoke",
        "fusion_mode": mode,
        "single_pct": 0.2,
        "total_pct": 0.5,
        "lifetime_pct": 5.0,
        "sell_threshold": 18,
        "buy_threshold": 25,
        "short_threshold": 25,
        "long_threshold": 40,
        "close_short_bs": 40,
        "close_long_ss": 40,
        "sell_pct": 0.55,
        "margin_use": 0.70,
        "lev": 3,
        "max_lev": 5,
        "cooldown": 4,
        "spot_cooldown": 12,
    }


@pytest.fixture(scope="module")
def signal_fixture_data():
    return _make_signal_fixture_data()


@pytest.mark.parametrize(
    "idx, mode, expected_ss, expected_bs",
    [
        (120, "c6_veto_4", 20.58, 17.25),
        (120, "kdj_weighted", 53.9, 21.0945),
        (120, "kdj_timing", 48.02, 18.75),
        (120, "kdj_gate", 27.44, 16.8),
        (160, "c6_veto_4", 76.659906, 0.0),
        (160, "kdj_weighted", 59.344504, 0.6),
        (160, "kdj_timing", 75.240278, 0.0),
        (160, "kdj_gate", 28.392558, 0.0),
        (200, "c6_veto_4", 44.145009, 0.7838),
        (200, "kdj_weighted", 46.081083, 2.340204),
        (200, "kdj_timing", 47.67661, 2.093687),
        (200, "kdj_gate", 43.085529, 1.045067),
        (239, "c6_veto_4", 12.816566, 18.2),
        (239, "kdj_weighted", 10.070159, 15.4),
        (239, "kdj_timing", 12.816566, 18.2),
        (239, "kdj_gate", 5.126626, 7.28),
    ],
)
def test_fusion_score_signal_contract(signal_fixture_data, idx, mode, expected_ss, expected_bs):
    """Signal contract: refactors must not silently change fused scores."""
    df, signals = signal_fixture_data

    cfg = {
        "fusion_mode": mode,
        "veto_threshold": 25,
        "kdj_bonus": 0.09,
        "veto_dampen": 0.3,
        "kdj_weight": 0.2,
        "div_weight": 0.55,
        "kdj_strong_mult": 1.25,
        "kdj_normal_mult": 1.12,
        "kdj_reverse_mult": 0.7,
        "kdj_gate_threshold": 10,
    }

    dt = df.index[idx]
    ss, bs = calc_fusion_score_six(signals, df, idx, dt, cfg)

    assert ss == pytest.approx(expected_ss, abs=1e-6)
    assert bs == pytest.approx(expected_bs, abs=1e-6)


def test_api_requires_login_schema():
    """API auth contract: unauthenticated API must return JSON schema, not HTML."""
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as c:
        resp = c.get("/api/live/status")
    assert resp.status_code == 401
    data = resp.get_json()
    assert data["success"] is False
    assert data["login_required"] is True
    assert "error" in data


def test_api_live_status_schema(client, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(app_module, "_detect_engine_process", lambda: {"running": False})

    resp = client.get("/api/live/status")
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["success"] is True
    assert set(["engine", "risk", "performance", "process"]).issubset(data.keys())
    assert isinstance(data["process"], dict)
    assert "running" in data["process"]


def test_api_live_test_signal_multi_schema(client, monkeypatch, tmp_path):
    """Response schema expected by page_live_control.js renderMultiSignalResult."""
    monkeypatch.setattr(app_module, "BASE_DIR", str(tmp_path))

    output_payload = {
        "timeframes": ["15m", "1h", "4h"],
        "results": [
            {
                "tf": "15m",
                "ok": True,
                "action": "OPEN_LONG",
                "sell_score": 10.0,
                "buy_score": 35.0,
                "price": 2500.0,
                "elapsed": 0.4,
            },
            {
                "tf": "1h",
                "ok": True,
                "action": "HOLD",
                "sell_score": 12.0,
                "buy_score": 14.0,
                "price": 2501.0,
                "elapsed": 0.6,
            },
        ],
        "consensus": {
            "decision": {
                "direction": "long",
                "label": "📈 大周期看多 + 小周期确认",
                "strength": 62,
                "actionable": True,
                "reason": "contract-test",
            }
        },
        "total_elapsed": 1.0,
    }

    def fake_run(cmd, capture_output, text, timeout, cwd):
        assert "--test-signal-multi" in cmd
        output_idx = cmd.index("-o") + 1
        out_path = cmd[output_idx]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(app_module.subprocess, "run", fake_run)

    resp = client.post(
        "/api/live/test_signal_multi",
        json={"timeframes": ["15m", "1h", "4h"]},
    )
    assert resp.status_code == 200
    data = resp.get_json()

    assert set(["success", "output", "error", "data"]).issubset(data.keys())
    assert data["success"] is True
    assert isinstance(data["data"], dict)
    assert set(["timeframes", "results", "consensus", "total_elapsed"]).issubset(data["data"].keys())


def test_api_live_logs_schema_when_empty(client, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "BASE_DIR", str(tmp_path))

    resp = client.get("/api/live/logs?lines=50")
    assert resp.status_code == 200
    data = resp.get_json()

    assert set(["success", "logs", "file"]).issubset(data.keys())
    assert data["success"] is True
    assert isinstance(data["logs"], str)
    assert isinstance(data["file"], str)


def test_run_strategy_smoke_contract(signal_fixture_data):
    """Backtest executor contract: unified core should remain callable."""
    df, signals = signal_fixture_data
    result = run_strategy(df, signals, _make_backtest_cfg(), tf="1h", trade_days=7)

    assert isinstance(result, dict)
    assert set(["strategy_return", "buy_hold_return", "max_drawdown", "total_trades"]).issubset(
        result.keys()
    )


def test_run_strategy_multi_tf_smoke_contract(signal_fixture_data):
    """Multi-TF executor contract: wrapper should work after executor merge."""
    df, signals = signal_fixture_data
    cfg = _make_backtest_cfg()

    warmup = max(60, int(len(df) * 0.05))
    score_map = {"1h": {}}
    for idx in range(warmup, len(df)):
        dt = df.index[idx]
        ss, bs = calc_fusion_score_six(signals, df, idx, dt, cfg)
        score_map["1h"][dt] = (float(ss), float(bs))

    result = run_strategy_multi_tf(
        primary_df=df,
        tf_score_map=score_map,
        decision_tfs=["1h"],
        config=cfg,
        primary_tf="1h",
        trade_days=7,
    )

    assert isinstance(result, dict)
    assert set(["strategy_return", "buy_hold_return", "max_drawdown", "total_trades"]).issubset(
        result.keys()
    )


# ================================================================
# fuse_tf_scores 统一共识算法测试
# ================================================================

class TestFuseTfScores:
    """Contract tests for the unified fuse_tf_scores function."""

    def test_output_schema(self):
        """fuse_tf_scores must return all expected keys."""
        tf_scores = {"1h": (20, 55), "4h": (15, 60)}
        result = fuse_tf_scores(tf_scores, ["1h", "4h"])

        required_keys = {
            "weighted_ss", "weighted_bs", "tf_scores", "coverage",
            "decision", "meta",
            "long_tfs", "short_tfs", "hold_tfs",
            "weighted_scores", "resonance_chains", "large_tf_signal",
            "direction",
        }
        assert required_keys.issubset(result.keys())

        # decision sub-schema
        dec = result["decision"]
        assert set(["direction", "strength", "actionable", "label", "reason"]).issubset(dec.keys())
        assert isinstance(dec["strength"], (int, float))
        assert isinstance(dec["actionable"], bool)

    def test_coverage_full(self):
        """All TFs available → coverage = 1.0."""
        tf_scores = {"15m": (10, 50), "1h": (15, 55), "4h": (12, 60)}
        result = fuse_tf_scores(tf_scores, ["15m", "1h", "4h"])
        assert result["coverage"] == 1.0

    def test_coverage_partial(self):
        """Some TFs missing → coverage < 1.0."""
        tf_scores = {"1h": (15, 55)}  # 15m and 4h missing
        result = fuse_tf_scores(tf_scores, ["15m", "1h", "4h"])
        assert 0 < result["coverage"] < 1.0

    def test_coverage_fail_closed(self):
        """Coverage below threshold → not actionable."""
        tf_scores = {"15m": (5, 70)}  # only lightest TF, heavy TFs missing
        result = fuse_tf_scores(
            tf_scores, ["15m", "1h", "4h", "8h", "24h"],
            config={"coverage_min": 0.5}
        )
        assert result["decision"]["actionable"] is False
        assert "覆盖不足" in result["decision"]["label"]

    def test_empty_tf_scores(self):
        """No TFs available → hold, not actionable."""
        result = fuse_tf_scores({}, ["1h", "4h"])
        assert result["decision"]["direction"] == "hold"
        assert result["decision"]["actionable"] is False
        assert result["coverage"] == 0.0

    def test_strong_long_consensus(self):
        """All TFs strongly long → direction should be long."""
        tf_scores = {
            "15m": (5, 65),
            "30m": (3, 70),
            "1h": (8, 60),
            "4h": (10, 55),
            "8h": (5, 62),
            "24h": (2, 68),
        }
        result = fuse_tf_scores(
            tf_scores,
            ["15m", "30m", "1h", "4h", "8h", "24h"],
        )
        assert result["decision"]["direction"] == "long"
        assert result["decision"]["actionable"] is True
        assert result["weighted_bs"] > result["weighted_ss"]

    def test_strong_short_consensus(self):
        """All TFs strongly short → direction should be short."""
        tf_scores = {
            "15m": (60, 5),
            "30m": (65, 3),
            "1h": (58, 8),
            "4h": (55, 10),
            "8h": (62, 5),
            "24h": (68, 2),
        }
        result = fuse_tf_scores(
            tf_scores,
            ["15m", "30m", "1h", "4h", "8h", "24h"],
        )
        assert result["decision"]["direction"] == "short"
        assert result["weighted_ss"] > result["weighted_bs"]

    def test_cross_direction_dampening(self):
        """Large TF long + small TF short → small TF's short dampened."""
        tf_scores_aligned = {
            "15m": (10, 55),
            "1h": (10, 55),
            "4h": (10, 55),
        }
        result_aligned = fuse_tf_scores(tf_scores_aligned, ["15m", "1h", "4h"])

        tf_scores_cross = {
            "15m": (55, 10),   # small TF bearish
            "1h": (55, 10),    # small TF bearish
            "4h": (10, 55),    # large TF bullish
        }
        result_cross = fuse_tf_scores(tf_scores_cross, ["15m", "1h", "4h"])
        # Cross-direction should dampen the weighted_ss
        assert result_cross["weighted_ss"] < result_aligned["weighted_bs"]

    def test_resonance_chain_boost(self):
        """Long chain with 4h+ → weighted_bs boosted."""
        tf_base = {
            "15m": (5, 60),
            "30m": (5, 60),
            "1h": (5, 60),
            "4h": (5, 60),
        }
        result = fuse_tf_scores(tf_base, ["15m", "30m", "1h", "4h"])
        assert result["meta"]["chain_len"] >= 3
        # The chain boost should push weighted_bs above the raw average
        raw_avg = sum(60 * w for w in [3, 5, 8, 15]) / sum([3, 5, 8, 15])
        assert result["weighted_bs"] > raw_avg

    def test_backtest_and_live_consistency(self):
        """
        Core invariant: fuse_tf_scores should produce identical scores
        whether called from backtest path or live path.
        """
        from optimize_six_book import calc_multi_tf_consensus, _get_tf_score_at

        tf_scores = {
            "15m": (30, 45),
            "1h": (20, 50),
            "4h": (25, 55),
        }
        decision_tfs = ["15m", "1h", "4h"]
        config = {"short_threshold": 25, "long_threshold": 40}

        # "Live" path: direct fuse_tf_scores call
        live_result = fuse_tf_scores(tf_scores, decision_tfs, config)

        # "Backtest" path: via calc_multi_tf_consensus
        # Build a score_map with a single timestamp for deterministic lookup
        from datetime import datetime
        dt = datetime(2025, 6, 1, 12, 0, 0)
        score_map = {tf: {dt: scores} for tf, scores in tf_scores.items()}
        bt_ss, bt_bs, bt_meta = calc_multi_tf_consensus(
            score_map, decision_tfs, dt,
            {**config, "coverage_min": 0.0},
        )

        assert live_result["weighted_ss"] == pytest.approx(bt_ss, abs=1e-6)
        assert live_result["weighted_bs"] == pytest.approx(bt_bs, abs=1e-6)

    def test_tf_detail_included(self):
        """tf_scores output should contain per-TF detail."""
        tf_scores = {"1h": (20, 55), "4h": (15, 60)}
        result = fuse_tf_scores(tf_scores, ["1h", "4h"])

        assert "1h" in result["tf_scores"]
        assert "4h" in result["tf_scores"]
        detail_1h = result["tf_scores"]["1h"]
        assert "ss" in detail_1h and "bs" in detail_1h and "dir" in detail_1h


class _DummyLogger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


class _RaisingSignalGenerator:
    def compute_multi_tf_consensus(self, _decision_tfs):
        raise RuntimeError("boom")


def test_multi_tf_gate_exception_fail_closed_when_execute_trades():
    """Live mode must fail-closed on multi-TF consensus exceptions."""
    engine = LiveTradingEngine.__new__(LiveTradingEngine)
    engine.signal_generator = _RaisingSignalGenerator()
    engine._decision_tfs = ["15m", "1h", "4h"]
    engine._consensus_min_strength = 50
    engine.logger = _DummyLogger()
    engine.phase = TradingPhase.SMALL_LIVE
    engine.config = SimpleNamespace(execute_trades=True)

    sig = SimpleNamespace(action="OPEN_LONG", reason="base")
    out = LiveTradingEngine._apply_multi_tf_gate(engine, sig)

    assert out.action == "HOLD"
    assert "fail-closed" in out.reason


def test_multi_tf_gate_exception_fail_open_when_paper():
    """Paper mode should degrade to single-TF on multi-TF exceptions."""
    engine = LiveTradingEngine.__new__(LiveTradingEngine)
    engine.signal_generator = _RaisingSignalGenerator()
    engine._decision_tfs = ["15m", "1h", "4h"]
    engine._consensus_min_strength = 50
    engine.logger = _DummyLogger()
    engine.phase = TradingPhase.PAPER
    engine.config = SimpleNamespace(execute_trades=False)

    sig = SimpleNamespace(action="OPEN_LONG", reason="base")
    out = LiveTradingEngine._apply_multi_tf_gate(engine, sig)

    assert out.action == "OPEN_LONG"
    assert out.reason == "base"


def test_cmd_test_signal_multi_uses_fuse_tf_scores(monkeypatch):
    """CLI multi-TF test path must use the unified fusion function."""
    import live_runner

    called = {}

    def fake_compute_single_tf(tf, _base_config):
        if tf == "15m":
            return {
                "tf": tf, "ok": True, "price": 2500.0,
                "sell_score": 12.0, "buy_score": 36.0,
                "action": "OPEN_LONG", "reason": "", "elapsed": 0.1,
            }
        return {
            "tf": tf, "ok": True, "price": 2501.0,
            "sell_score": 20.0, "buy_score": 42.0,
            "action": "OPEN_LONG", "reason": "", "elapsed": 0.1,
        }

    def fake_fuse_tf_scores(tf_scores, decision_tfs, config):
        called["tf_scores"] = tf_scores
        called["decision_tfs"] = decision_tfs
        called["config"] = config
        return {
            "long_tfs": ["15m", "1h"],
            "short_tfs": [],
            "hold_tfs": [],
            "weighted_scores": {
                "long": 100.0, "short": 0.0, "net": 100.0,
                "long_raw": 1, "short_raw": 0, "total_weight": 1,
            },
            "resonance_chains": [],
            "large_tf_signal": {"direction": "neutral", "tfs": []},
            "decision": {
                "direction": "long",
                "label": "test",
                "strength": 80,
                "reason": "test",
                "actionable": True,
            },
        }

    monkeypatch.setattr(live_runner, "_compute_single_tf", fake_compute_single_tf)
    monkeypatch.setattr(live_runner, "fuse_tf_scores", fake_fuse_tf_scores)
    monkeypatch.setattr(
        live_runner,
        "create_default_config",
        lambda _phase: SimpleNamespace(
            strategy=SimpleNamespace(short_threshold=25, long_threshold=40)
        ),
    )
    monkeypatch.setattr(
        live_runner.os.path,
        "exists",
        lambda p: False,
    )

    args = SimpleNamespace(timeframe="15m,1h", output=None)
    results = live_runner.cmd_test_signal_multi(args)

    assert len(results) == 2
    assert called["decision_tfs"] == ["15m", "1h"]
    assert called["config"]["short_threshold"] == 25
    assert called["config"]["long_threshold"] == 40
    assert called["tf_scores"]["15m"] == (12.0, 36.0)
    assert called["tf_scores"]["1h"] == (20.0, 42.0)
