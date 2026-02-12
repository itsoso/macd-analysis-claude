import numpy as np
import pandas as pd
import pytest

import backtest_multi_tf_30d_7d as bt_60307
import optimize_six_book as opt_six
from candlestick_patterns import (
    candle_features,
    detect_hanging_man,
    detect_inverted_hammer,
)


def _make_ohlcv(n=240, start=100.0, step=1.0):
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    close = np.array([start + step * i for i in range(n)], dtype=float)
    open_ = close - (0.2 if step >= 0 else -0.2)
    high = np.maximum(open_, close) + 0.3
    low = np.minimum(open_, close) - 0.3
    volume = np.full(n, 1000.0)
    quote_volume = volume * close
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": quote_volume,
        },
        index=idx,
    )


def _base_exec_config():
    return {
        "name": "realism-test",
        "single_pct": 0.2,
        "total_pct": 0.5,
        "lifetime_pct": 5.0,
        "sell_threshold": 18,
        "buy_threshold": 25,
        "short_threshold": 25,
        "long_threshold": 40,
        "close_short_bs": 999,
        "close_long_ss": 999,
        "sell_pct": 0.0,
        "margin_use": 0.7,
        "lev": 3,
        "max_lev": 5,
        "cooldown": 1,
        "spot_cooldown": 1,
        "short_sl": -0.99,
        "long_sl": -0.99,
        "short_tp": 9.0,
        "long_tp": 9.0,
        "short_trail": 9.0,
        "long_trail": 9.0,
        "short_max_hold": 9999,
        "long_max_hold": 9999,
    }


def test_hanging_man_does_not_depend_on_next_bar():
    df = _make_ohlcv(n=40, start=100, step=1.0)
    i = 25
    df.iloc[i, df.columns.get_loc("open")] = 130.0
    df.iloc[i, df.columns.get_loc("close")] = 129.8
    df.iloc[i, df.columns.get_loc("high")] = 130.05
    df.iloc[i, df.columns.get_loc("low")] = 128.9

    f1 = candle_features(df.copy())
    p1 = detect_hanging_man(f1, i)
    assert p1 is not None

    df2 = df.copy()
    df2.iloc[i + 1, df2.columns.get_loc("close")] = 10.0
    f2 = candle_features(df2)
    p2 = detect_hanging_man(f2, i)

    assert p2 == p1


def test_inverted_hammer_does_not_depend_on_next_bar():
    df = _make_ohlcv(n=40, start=200, step=-1.0)
    i = 25
    df.iloc[i, df.columns.get_loc("open")] = 170.0
    df.iloc[i, df.columns.get_loc("close")] = 170.2
    df.iloc[i, df.columns.get_loc("high")] = 171.3
    df.iloc[i, df.columns.get_loc("low")] = 170.15

    f1 = candle_features(df.copy())
    p1 = detect_inverted_hammer(f1, i)
    assert p1 is not None

    df2 = df.copy()
    df2.iloc[i + 1, df2.columns.get_loc("close")] = 1000.0
    f2 = candle_features(df2)
    p2 = detect_inverted_hammer(f2, i)

    assert p2 == p1


def test_execution_uses_next_bar_open_after_signal_close():
    df = _make_ohlcv(n=100, start=100, step=1.0)
    warmup = max(60, int(len(df) * 0.05))

    def score_provider(idx, _dt, _price):
        if idx == warmup:
            return 80.0, 0.0
        return 0.0, 0.0

    result = opt_six._run_strategy_core(
        primary_df=df,
        config=_base_exec_config(),
        primary_tf="1h",
        trade_days=30,
        score_provider=score_provider,
    )

    open_short = next(t for t in result["trades"] if t["action"] == "OPEN_SHORT")
    expected_idx = warmup + 1
    assert open_short["time"] == df.index[expected_idx].isoformat()
    assert open_short["price"] == pytest.approx(float(df["open"].iloc[expected_idx]), abs=1e-9)


def test_trade_days_60_is_windowed_not_full_history():
    df = _make_ohlcv(n=2200, start=100, step=0.05)  # > 90 days hourly
    start_dt = df.index[-1] - pd.Timedelta(days=60)
    early_cutoff = start_dt - pd.Timedelta(hours=2)

    def score_provider(_idx, dt, _price):
        if dt <= early_cutoff:
            return 80.0, 0.0
        return 0.0, 0.0

    result = opt_six._run_strategy_core(
        primary_df=df,
        config=_base_exec_config(),
        primary_tf="1h",
        trade_days=60,
        score_provider=score_provider,
    )

    assert result["total_trades"] == 0


def test_slice_train_window_non_overlapping():
    df = _make_ohlcv(n=24 * 200, start=100, step=0.01)  # 200 days hourly
    train = bt_60307._slice_train_window(df, test_days=60, train_days=60)
    test_start = df.index[-1] - pd.Timedelta(days=60)

    assert len(train) > 0
    assert train.index.max() < test_start
    assert train.index.min() >= (test_start - pd.Timedelta(days=60))


def test_apply_conservative_risk_clamps_aggressive_config():
    cfg = {"single_pct": 0.2, "lev": 5, "margin_use": 0.7, "max_lev": 5}
    safe = bt_60307._apply_conservative_risk(cfg)

    assert safe["single_pct"] <= 0.10
    assert safe["lev"] <= 3
    assert safe["margin_use"] <= 0.50
    assert safe["max_lev"] <= 3
