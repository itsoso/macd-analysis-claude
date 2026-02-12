import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import app as app_module
from indicators import add_all_indicators
from ma_indicators import add_moving_averages
from optimize_six_book import (
    compute_signals_six,
    calc_fusion_score_six,
    run_strategy,
    run_strategy_multi_tf,
)


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
