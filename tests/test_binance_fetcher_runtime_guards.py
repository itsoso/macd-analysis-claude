import time
from urllib.error import URLError

import numpy as np
import pandas as pd

import binance_fetcher as bf


def _make_local_df(hours_ago_end: int = 72, rows: int = 120) -> pd.DataFrame:
    end = pd.Timestamp.now().tz_localize(None) - pd.Timedelta(hours=hours_ago_end)
    idx = pd.date_range(end=end, periods=rows, freq="1h")
    return pd.DataFrame(
        {
            "open": np.full(rows, 100.0),
            "high": np.full(rows, 110.0),
            "low": np.full(rows, 90.0),
            "close": np.full(rows, 105.0),
            "volume": np.full(rows, 1000.0),
            "quote_volume": np.full(rows, 100000.0),
            "taker_buy_base": np.full(rows, 500.0),
            "taker_buy_quote": np.full(rows, 50000.0),
        },
        index=idx,
    )


def test_api_circuit_breaker_short_circuits_repeated_failures(monkeypatch):
    calls = {"n": 0}

    def _boom(*_args, **_kwargs):
        calls["n"] += 1
        raise URLError("network down")

    monkeypatch.setattr(bf, "urlopen", _boom)
    monkeypatch.setattr(bf.time, "sleep", lambda *_args, **_kwargs: None)

    bf._API_CIRCUIT_OPEN_UNTIL.clear()
    bf._API_CIRCUIT_LAST_WARN_TS.clear()

    url = "https://fapi.binance.com/fapi/v1/fundingRate?symbol=ETHUSDT&limit=10"
    out1 = bf._api_get_json(url, max_retries=1)
    assert out1 == []
    assert calls["n"] == 1

    endpoint_key = bf._api_endpoint_key(url)
    assert bf._API_CIRCUIT_OPEN_UNTIL.get(endpoint_key, 0) > time.time()

    out2 = bf._api_get_json(url, max_retries=3)
    assert out2 == []
    # 第二次命中断路器，不应再次触发网络请求
    assert calls["n"] == 1


def test_fetch_binance_klines_uses_api_when_local_cache_is_stale(monkeypatch):
    local_df = _make_local_df(hours_ago_end=72)
    api_calls = {"n": 0}

    def _fake_api(*_args, **_kwargs):
        api_calls["n"] += 1
        now_ms = int(time.time() * 1000)
        open_time = now_ms - 3600_000
        close_time = now_ms - 1
        return [[
            open_time, "100", "110", "90", "106", "1000",
            close_time, "100000", "100", "500", "50000", "0",
        ]]

    monkeypatch.setattr(bf, "_try_load_local", lambda *_args, **_kwargs: local_df)
    monkeypatch.setattr(bf, "_api_get_json", _fake_api)

    out = bf.fetch_binance_klines(
        "ETHUSDT",
        interval="1h",
        days=30,
        require_fresh=True,
        max_lag_hours=2.0,
        allow_api_fallback=True,
    )

    assert len(out) == 1
    assert api_calls["n"] >= 1
    assert out.index[-1] > local_df.index[-1]


def test_fetch_binance_klines_respects_no_api_fallback_when_cache_stale(monkeypatch):
    local_df = _make_local_df(hours_ago_end=72)
    api_calls = {"n": 0}

    def _fake_api(*_args, **_kwargs):
        api_calls["n"] += 1
        return []

    monkeypatch.setattr(bf, "_try_load_local", lambda *_args, **_kwargs: local_df)
    monkeypatch.setattr(bf, "_api_get_json", _fake_api)

    out = bf.fetch_binance_klines(
        "ETHUSDT",
        interval="1h",
        days=30,
        require_fresh=True,
        max_lag_hours=2.0,
        allow_api_fallback=False,
    )

    assert api_calls["n"] == 0
    assert len(out) == len(local_df)
    assert out.index[-1] == local_df.index[-1]
