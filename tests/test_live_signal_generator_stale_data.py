from types import SimpleNamespace

import pandas as pd

import live_signal_generator as lsg


class _NoopLogger:
    def info(self, _msg):
        return None

    def warning(self, _msg):
        return None

    def error(self, _msg):
        return None


def _make_kline_df(hours_ago_end: int = 10, rows: int = 120):
    end = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(hours=hours_ago_end)
    idx = pd.date_range(end=end, periods=rows, freq="1h")
    return pd.DataFrame(
        {
            "open": 2000.0,
            "high": 2010.0,
            "low": 1990.0,
            "close": 2005.0,
            "volume": 1000.0,
        },
        index=idx,
    )


def _make_cfg(allow_stale=False, max_lag_hours=2.0):
    return SimpleNamespace(
        symbol="ETHUSDT",
        timeframe="1h",
        lookback_days=60,
        allow_stale_klines=allow_stale,
        max_kline_lag_hours=max_lag_hours,
    )


def test_refresh_data_blocks_stale_klines(monkeypatch):
    monkeypatch.setattr(lsg, "fetch_binance_klines", lambda *args, **kwargs: _make_kline_df(hours_ago_end=12))
    monkeypatch.setattr(lsg, "fetch_mark_price_klines", lambda *args, **kwargs: None)
    monkeypatch.setattr(lsg, "fetch_funding_rate_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(lsg, "fetch_open_interest_history", lambda *args, **kwargs: None)

    gen = lsg.LiveSignalGenerator(_make_cfg(allow_stale=False, max_lag_hours=2.0), logger=_NoopLogger())
    ok = gen.refresh_data(force=True)

    assert ok is False
    assert gen._data_stale is True
    assert gen._signals is None


def test_refresh_data_allows_stale_when_config_enabled(monkeypatch):
    monkeypatch.setattr(lsg, "fetch_binance_klines", lambda *args, **kwargs: _make_kline_df(hours_ago_end=12))
    monkeypatch.setattr(lsg, "fetch_mark_price_klines", lambda *args, **kwargs: None)
    monkeypatch.setattr(lsg, "fetch_funding_rate_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(lsg, "fetch_open_interest_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(lsg, "add_all_indicators", lambda df: df)
    monkeypatch.setattr(lsg, "add_moving_averages", lambda df, timeframe: None)
    monkeypatch.setattr(lsg, "compute_signals_six", lambda *args, **kwargs: {"ok": True})

    gen = lsg.LiveSignalGenerator(_make_cfg(allow_stale=True, max_lag_hours=2.0), logger=_NoopLogger())
    ok = gen.refresh_data(force=True)

    assert ok is True
    assert gen._data_stale is True
    assert gen._signals == {"ok": True}


def test_get_current_data_info_reports_stale_status(monkeypatch):
    monkeypatch.setattr(lsg, "fetch_binance_klines", lambda *args, **kwargs: _make_kline_df(hours_ago_end=12))
    monkeypatch.setattr(lsg, "fetch_mark_price_klines", lambda *args, **kwargs: None)
    monkeypatch.setattr(lsg, "fetch_funding_rate_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(lsg, "fetch_open_interest_history", lambda *args, **kwargs: None)

    gen = lsg.LiveSignalGenerator(_make_cfg(allow_stale=False, max_lag_hours=2.0), logger=_NoopLogger())
    ok = gen.refresh_data(force=True)

    assert ok is False
    info = gen.get_current_data_info()
    assert info["status"] == "stale"
    assert info["data_stale"] is True
    assert info["data_lag_hours"] > info["max_allowed_lag_hours"]


def test_compute_latest_signal_emits_ml_status_when_disabled(monkeypatch):
    monkeypatch.setattr(lsg, "fetch_binance_klines", lambda *args, **kwargs: _make_kline_df(hours_ago_end=0))
    monkeypatch.setattr(lsg, "fetch_mark_price_klines", lambda *args, **kwargs: None)
    monkeypatch.setattr(lsg, "fetch_funding_rate_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(lsg, "fetch_open_interest_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(lsg, "add_all_indicators", lambda df: df)
    monkeypatch.setattr(lsg, "add_moving_averages", lambda df, timeframe: None)
    monkeypatch.setattr(lsg, "compute_signals_six", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(lsg, "calc_fusion_score_six", lambda *args, **kwargs: (30.0, 25.0))

    cfg = _make_cfg(allow_stale=True, max_lag_hours=2.0)
    cfg.use_ml_enhancement = False
    cfg.ml_enhancement_shadow_mode = True

    gen = lsg.LiveSignalGenerator(cfg, logger=_NoopLogger())
    gen._infer_regime_label = lambda *args, **kwargs: "neutral"
    gen._build_fusion_config = lambda *args, **kwargs: {}
    gen._extract_components = lambda *args, **kwargs: {}

    sig = gen.compute_latest_signal()
    assert sig is not None
    comps = sig.components
    assert comps["ml_enabled"] is False
    assert comps["ml_available"] is False
    assert comps["ml_reason"] == "disabled_by_config"
