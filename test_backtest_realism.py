import numpy as np
import pandas as pd
import pytest

import backtest_multi_tf_30d_7d as bt_60307
import optimize_six_book as opt_six
import multi_tf_consensus
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
    df = _make_ohlcv(n=260, start=100, step=1.0)
    warmup = 200

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
    market_price = open_short.get("price", open_short.get("market_price"))
    assert market_price == pytest.approx(float(df["open"].iloc[expected_idx]), abs=1e-9)


def test_run_strategy_core_blocks_dual_futures_open(monkeypatch):
    """策略层不应在已有空单时再开多（反向同理）。"""
    df = _make_ohlcv(n=400, start=100, step=0.1)
    warmup = 200

    def score_provider(idx, _dt, _price):
        if idx < warmup:
            return 0.0, 0.0
        # 先触发开空，随后持续给做多信号；若缺少持仓互斥保护，会在空单未平时开多
        if idx == warmup:
            return 85.0, 0.0
        return 0.0, 85.0

    class _GuardEngine(opt_six.FuturesEngine):
        def open_long(self, price, dt, margin, leverage, reason, **kwargs):
            assert self.futures_short is None, "OPEN_LONG called while short position is active"
            return super().open_long(price, dt, margin, leverage, reason, **kwargs)

        def open_short(self, price, dt, margin, leverage, reason, **kwargs):
            assert self.futures_long is None, "OPEN_SHORT called while long position is active"
            return super().open_short(price, dt, margin, leverage, reason, **kwargs)

    monkeypatch.setattr(opt_six, "FuturesEngine", _GuardEngine)

    cfg = _base_exec_config()
    cfg.update(
        {
            "short_threshold": 20,
            "long_threshold": 20,
            "close_short_bs": 999,
            "close_long_ss": 999,
            "short_sl": -0.99,
            "long_sl": -0.99,
            "hard_stop_loss": -0.99,
            "short_max_hold": 9999,
            "long_max_hold": 9999,
            "use_partial_tp": False,
            "use_partial_tp_2": False,
            "use_trend_enhance": False,
            "cooldown": 1,
            "spot_cooldown": 1,
        }
    )

    result = opt_six._run_strategy_core(
        primary_df=df,
        config=cfg,
        primary_tf="1h",
        trade_days=30,
        score_provider=score_provider,
    )

    # 测试有效性：确实开过空；且不存在在空单存续期间开多
    assert any(t["action"] == "OPEN_SHORT" for t in result["trades"])
    assert not any(t["action"] == "OPEN_LONG" for t in result["trades"])


def test_reverse_close_then_open_same_bar():
    """反向信号应先平旧仓，再允许同bar开新方向仓位。"""
    df = _make_ohlcv(n=420, start=100, step=0.1)
    warmup = 200

    def score_provider(idx, _dt, _price):
        if idx < warmup:
            return 0.0, 0.0
        if idx == warmup:
            return 85.0, 0.0   # 先开空
        if idx == warmup + 1:
            return 0.0, 85.0   # 下一根触发反向平空 + 开多
        return 0.0, 0.0

    cfg = _base_exec_config()
    cfg.update(
        {
            "short_threshold": 20,
            "long_threshold": 20,
            "close_short_bs": 20,
            "close_long_ss": 20,
            "cooldown": 1,
            "spot_cooldown": 1,
            "use_trend_enhance": False,
            "reverse_min_hold_short": 0,
            "reverse_min_hold_long": 0,
        }
    )

    result = opt_six._run_strategy_core(
        primary_df=df,
        config=cfg,
        primary_tf="1h",
        trade_days=30,
        score_provider=score_provider,
    )

    trades = result["trades"]
    close_short_idx = next(i for i, t in enumerate(trades) if t["action"] == "CLOSE_SHORT")
    open_long_idx = next(i for i, t in enumerate(trades) if t["action"] == "OPEN_LONG")

    # 关键断言: 同一bar先平空再开多，避免“错过反手”
    assert close_short_idx < open_long_idx
    assert trades[close_short_idx]["time"] == trades[open_long_idx]["time"]


def test_neutral_quality_gate_blocks_low_confidence_entry(monkeypatch):
    df = _make_ohlcv(n=420, start=100, step=0.0)
    warmup = 200

    original_regime = opt_six._compute_regime_controls

    def _forced_neutral(primary_df, idx, config, precomputed=None):
        ctl = original_regime(primary_df, idx, config, precomputed=precomputed)
        ctl["regime_label"] = "neutral"
        return ctl

    monkeypatch.setattr(opt_six, "_compute_regime_controls", _forced_neutral)

    low_conf_meta = {
        "decision": {"direction": "long", "actionable": True, "strength": 70},
        "chain_len": 3,
        "chain_dir": "short",
        "chain_has_4h": True,
        "large_ss": 60,
        "large_bs": 10,
    }

    def score_provider(idx, _dt, _price):
        if idx < warmup:
            return 0.0, 0.0, {}
        return 82.0, 18.0, low_conf_meta

    cfg = _base_exec_config()
    cfg.update(
        {
            "short_threshold": 20,
            "close_short_bs": 999,
            "use_neutral_quality_gate": True,
            "neutral_min_strength": 45,
            "neutral_min_streak": 1,
        }
    )

    result = opt_six._run_strategy_core(
        primary_df=df,
        config=cfg,
        primary_tf="1h",
        trade_days=30,
        score_provider=score_provider,
    )

    assert not any(t["action"] == "OPEN_SHORT" for t in result["trades"])
    gate_stats = result.get("neutral_quality_gate", {})
    assert gate_stats.get("short_blocked", 0) > 0
    assert gate_stats.get("blocked_reason_counts", {}).get("decision_conflict", 0) > 0


def test_neutral_quality_gate_allows_consistent_entry(monkeypatch):
    df = _make_ohlcv(n=420, start=100, step=0.0)
    warmup = 200

    original_regime = opt_six._compute_regime_controls

    def _forced_neutral(primary_df, idx, config, precomputed=None):
        ctl = original_regime(primary_df, idx, config, precomputed=precomputed)
        ctl["regime_label"] = "neutral"
        return ctl

    monkeypatch.setattr(opt_six, "_compute_regime_controls", _forced_neutral)

    high_conf_meta = {
        "decision": {"direction": "short", "actionable": True, "strength": 75},
        "chain_len": 3,
        "chain_dir": "short",
        "chain_has_4h": True,
        "large_ss": 62,
        "large_bs": 8,
    }

    def score_provider(idx, _dt, _price):
        if idx < warmup:
            return 0.0, 0.0, {}
        return 84.0, 16.0, high_conf_meta

    cfg = _base_exec_config()
    cfg.update(
        {
            "short_threshold": 20,
            "close_short_bs": 999,
            "use_neutral_quality_gate": True,
            "neutral_min_score_gap": 12,
        }
    )

    result = opt_six._run_strategy_core(
        primary_df=df,
        config=cfg,
        primary_tf="1h",
        trade_days=30,
        score_provider=score_provider,
    )

    short_opens = [t for t in result["trades"] if t["action"] == "OPEN_SHORT"]
    assert len(short_opens) >= 1
    assert pd.Timestamp(short_opens[0]["time"]) >= df.index[warmup + 1]
    gate_stats = result.get("neutral_quality_gate", {})
    assert gate_stats.get("short_blocked", 0) == 0


def test_neutral_short_structure_gate_blocks_bullish_large_tf(monkeypatch):
    df = _make_ohlcv(n=420, start=100, step=0.0)
    warmup = 200

    original_regime = opt_six._compute_regime_controls

    def _forced_neutral(primary_df, idx, config, precomputed=None):
        ctl = original_regime(primary_df, idx, config, precomputed=precomputed)
        ctl["regime_label"] = "neutral"
        return ctl

    monkeypatch.setattr(opt_six, "_compute_regime_controls", _forced_neutral)

    conflict_meta = {
        "decision": {"direction": "short", "actionable": True, "strength": 80},
        "book_features_by_tf": {
            "4h": {"div_sell": 12, "div_buy": 36, "ma_sell": 9, "ma_buy": 24, "vp_sell": 6, "vp_buy": 16},
            "24h": {"div_sell": 10, "div_buy": 34, "ma_sell": 8, "ma_buy": 21, "vp_sell": 4, "vp_buy": 14},
        },
    }

    def score_provider(idx, _dt, _price):
        if idx < warmup:
            return 0.0, 0.0, {}
        return 86.0, 14.0, conflict_meta

    cfg = _base_exec_config()
    cfg.update(
        {
            "short_threshold": 20,
            "close_short_bs": 999,
            "use_neutral_short_structure_gate": True,
            "neutral_short_structure_large_tfs": "4h,24h",
            "neutral_short_structure_need_min_tfs": 1,
            "neutral_short_structure_min_agree": 1,
            "neutral_short_structure_fail_open": True,
        }
    )

    result = opt_six._run_strategy_core(
        primary_df=df,
        config=cfg,
        primary_tf="1h",
        trade_days=30,
        score_provider=score_provider,
    )

    assert not any(t["action"] == "OPEN_SHORT" for t in result["trades"])
    nss = result.get("neutral_short_structure_gate", {})
    assert nss.get("blocked", 0) > 0
    assert nss.get("reason_counts", {}).get("large_tf_bull_conflict", 0) > 0


def test_neutral_short_structure_gate_allows_bearish_large_tf(monkeypatch):
    df = _make_ohlcv(n=420, start=100, step=0.0)
    warmup = 200

    original_regime = opt_six._compute_regime_controls

    def _forced_neutral(primary_df, idx, config, precomputed=None):
        ctl = original_regime(primary_df, idx, config, precomputed=precomputed)
        ctl["regime_label"] = "neutral"
        return ctl

    monkeypatch.setattr(opt_six, "_compute_regime_controls", _forced_neutral)

    support_meta = {
        "decision": {"direction": "short", "actionable": True, "strength": 82},
        "book_features_by_tf": {
            "4h": {"div_sell": 34, "div_buy": 12, "ma_sell": 25, "ma_buy": 10, "vp_sell": 18, "vp_buy": 7},
            "24h": {"div_sell": 29, "div_buy": 11, "ma_sell": 21, "ma_buy": 9, "vp_sell": 14, "vp_buy": 5},
        },
    }

    def score_provider(idx, _dt, _price):
        if idx < warmup:
            return 0.0, 0.0, {}
        return 86.0, 14.0, support_meta

    cfg = _base_exec_config()
    cfg.update(
        {
            "short_threshold": 20,
            "close_short_bs": 999,
            "use_neutral_short_structure_gate": True,
            "neutral_short_structure_large_tfs": "4h,24h",
            "neutral_short_structure_need_min_tfs": 1,
            "neutral_short_structure_min_agree": 1,
            "neutral_short_structure_fail_open": True,
        }
    )

    result = opt_six._run_strategy_core(
        primary_df=df,
        config=cfg,
        primary_tf="1h",
        trade_days=30,
        score_provider=score_provider,
    )

    assert any(t["action"] == "OPEN_SHORT" for t in result["trades"])
    nss = result.get("neutral_short_structure_gate", {})
    assert nss.get("blocked", 0) == 0
    assert nss.get("support_avg", 0.0) > 0


def test_short_adverse_exit_closes_losing_short_early():
    df = _make_ohlcv(n=440, start=100, step=1.0)
    warmup = 200

    open_meta = {
        "decision": {"direction": "short", "actionable": True, "strength": 80},
        "chain_dir": "short",
        "book_features_weighted": {"ma_sell": 22, "ma_buy": 10},
    }
    reverse_meta = {
        "decision": {"direction": "long", "actionable": True, "strength": 80},
        "chain_dir": "long",
        "book_features_weighted": {
            "div_sell": 8, "div_buy": 22,
            "ma_sell": 8, "ma_buy": 24,
            "cs_sell": 6, "cs_buy": 18,
            "bb_sell": 7, "bb_buy": 16,
            "vp_sell": 8, "vp_buy": 17,
            "kdj_sell": 6, "kdj_buy": 19,
        },
    }

    def score_provider(idx, _dt, _price):
        if idx < warmup:
            return 0.0, 0.0, {}
        if idx == warmup:
            return 88.0, 0.0, open_meta
        return 20.0, 82.0, reverse_meta

    cfg = _base_exec_config()
    cfg.update(
        {
            "short_threshold": 20,
            "close_short_bs": 999,   # 禁用常规反向平空，隔离测试目标
            "hard_stop_loss": -0.99,  # 禁用硬止损
            "short_sl": -0.99,        # 禁用常规止损
            "use_short_adverse_exit": True,
            "short_adverse_min_bars": 2,
            "short_adverse_loss_r": -0.02,
            "short_adverse_bs": 60,
            "short_adverse_bs_dom_ratio": 0.9,
            "short_adverse_ss_cap": 40,
            "short_adverse_ma_conflict_gap": 8,
            "short_adverse_need_chain_long": True,
            "short_adverse_regimes": "",  # 不限制 regime，避免依赖 regime 分类细节
        }
    )

    result = opt_six._run_strategy_core(
        primary_df=df,
        config=cfg,
        primary_tf="1h",
        trade_days=30,
        score_provider=score_provider,
    )

    close_shorts = [t for t in result["trades"] if t["action"] == "CLOSE_SHORT"]
    assert any("防守平空" in (t.get("reason") or "") for t in close_shorts)
    sa = result.get("short_adverse_exit", {})
    assert sa.get("triggered", 0) >= 1


def test_extreme_divergence_short_veto_blocks_trend_short(monkeypatch):
    df = _make_ohlcv(n=420, start=100, step=0.2)
    warmup = 200

    original_regime = opt_six._compute_regime_controls

    def _forced_trend(primary_df, idx, config, precomputed=None):
        ctl = original_regime(primary_df, idx, config, precomputed=precomputed)
        ctl["regime_label"] = "trend"
        return ctl

    monkeypatch.setattr(opt_six, "_compute_regime_controls", _forced_trend)

    meta = {
        "decision": {"direction": "short", "actionable": True, "strength": 90},
        "book_features_weighted": {
            "div_sell": 95,
            "div_buy": 20,
            "ma_sell": 12,
            "ma_buy": 8,
            "cs_sell": 6,
            "cs_buy": 4,
            "bb_sell": 7,
            "bb_buy": 3,
            "vp_sell": 9,
            "vp_buy": 5,
            "kdj_sell": 8,
            "kdj_buy": 3,
        },
    }

    def score_provider(idx, _dt, _price):
        if idx < warmup:
            return 0.0, 0.0, {}
        return 90.0, 10.0, meta

    cfg = _base_exec_config()
    cfg.update(
        {
            "short_threshold": 20,
            "use_extreme_divergence_short_veto": True,
            "extreme_div_short_threshold": 85.0,
            "extreme_div_short_regimes": "trend,high_vol",
        }
    )

    result = opt_six._run_strategy_core(
        primary_df=df,
        config=cfg,
        primary_tf="1h",
        trade_days=30,
        score_provider=score_provider,
    )

    assert not any(t["action"] == "OPEN_SHORT" for t in result["trades"])
    dv = result.get("extreme_div_short_veto", {})
    assert dv.get("blocked", 0) > 0
    assert dv.get("reason_counts", {}).get("extreme_divergence", 0) > 0


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


def test_explicit_trade_start_end_dt_limits_execution_window():
    df = _make_ohlcv(n=24 * 60, start=100, step=0.2)  # 60天小时线
    start_dt = df.index[24 * 10]   # 第10天
    end_dt = df.index[24 * 20]     # 第20天

    def score_provider(_idx, _dt, _price):
        return 80.0, 0.0  # 持续开空信号

    result = opt_six._run_strategy_core(
        primary_df=df,
        config=_base_exec_config(),
        primary_tf="1h",
        trade_days=0,
        score_provider=score_provider,
        trade_start_dt=start_dt,
        trade_end_dt=end_dt,
    )

    assert result["total_trades"] > 0
    trade_times = [pd.Timestamp(t["time"]) for t in result["trades"]]
    assert min(trade_times) >= start_dt
    assert max(trade_times) <= end_dt


def test_slice_train_window_non_overlapping():
    df = _make_ohlcv(n=24 * 200, start=100, step=0.01)  # 200 days hourly
    train = bt_60307._slice_train_window(df, test_days=60, train_days=60, purge_days=7)
    test_start = df.index[-1] - pd.Timedelta(days=60)
    train_end = test_start - pd.Timedelta(days=7)  # purge gap

    assert len(train) > 0
    # 训练集应在 purge 间隔之前结束
    assert train.index.max() < train_end
    # 训练集不与测试集重叠
    assert train.index.max() < test_start


def test_apply_conservative_risk_clamps_aggressive_config():
    cfg = {"single_pct": 0.2, "lev": 5, "margin_use": 0.7, "max_lev": 5}
    safe = bt_60307._apply_conservative_risk(cfg)

    assert safe["single_pct"] <= 0.10
    assert safe["lev"] <= 3
    assert safe["margin_use"] <= 0.50
    assert safe["max_lev"] <= 3


def test_format_period_results_export_includes_full_trade_details_by_default():
    raw_trade = {
        "time": "2025-01-01T01:00:00",
        "action": "OPEN_SHORT",
        "direction": "short",
        "market_price": 2000.0,
        "exec_price": 1998.0,
        "quantity": 1.2,
        "notional_value": 2397.6,
        "fee": 1.1,
        "slippage_cost": 0.8,
        "total_cost": 1.9,
        "leverage": 3,
        "margin": 799.2,
        "pnl": 0.0,
        "reason": "signal",
        "extra_note": "keep-me",
    }
    results = [
        {
            "combo_name": "组合A",
            "primary_tf": "1h",
            "decision_tfs": ["30m", "1h", "4h"],
            "alpha": 12.3,
            "strategy_return": 8.8,
            "buy_hold_return": -3.5,
            "max_drawdown": -9.2,
            "total_trades": 1,
            "trade_details": [raw_trade],
        }
    ]

    formatted = bt_60307._format_period_results_export(results, include_full_trades=True)

    assert len(formatted) == 1
    row = formatted[0]
    assert row["trade_count"] == 1
    assert "trade_details" in row
    assert len(row["trade_details"]) == 1
    trade = row["trade_details"][0]
    for field in bt_60307.TRADE_RECORD_FIELDS:
        assert field in trade
    assert trade["extra_note"] == "keep-me"


def test_format_period_results_export_can_disable_full_trade_details():
    results = [
        {
            "combo_name": "组合B",
            "primary_tf": "2h",
            "decision_tfs": ["1h", "2h", "8h"],
            "alpha": 5.0,
            "strategy_return": 2.0,
            "buy_hold_return": 1.0,
            "max_drawdown": -4.0,
            "total_trades": 1,
            "trade_details": [{"time": "2025-01-01T01:00:00", "action": "OPEN_LONG"}],
        }
    ]

    formatted = bt_60307._format_period_results_export(results, include_full_trades=False)

    assert len(formatted) == 1
    row = formatted[0]
    assert row["trade_count"] == 1
    assert "trade_details" not in row


def test_calc_multi_tf_consensus_passes_configured_coverage_min(monkeypatch):
    called = {}

    def fake_fuse(tf_scores, decision_tfs, config):
        called["config"] = dict(config)
        return {
            "weighted_ss": 10.0,
            "weighted_bs": 20.0,
            "meta": {},
            "decision": {"direction": "long", "actionable": True, "strength": 80},
            "coverage": 1.0,
            "weighted_scores": {},
        }

    # 必须 patch optimize_six_book 模块中已绑定的名称,
    # 因为 optimize_six_book.py 用 'from multi_tf_consensus import fuse_tf_scores'
    monkeypatch.setattr(opt_six, "fuse_tf_scores", fake_fuse)
    dt = pd.Timestamp("2025-01-01 00:00:00")
    score_map = {"1h": {dt: (10.0, 20.0)}}
    ss, bs, _meta = opt_six.calc_multi_tf_consensus(
        score_map,
        ["1h"],
        dt,
        {"short_threshold": 25, "long_threshold": 40, "coverage_min": 0.75},
    )

    assert ss == 10.0
    assert bs == 20.0
    assert called["config"]["coverage_min"] == 0.75


def test_calc_multi_tf_consensus_coverage_counts_only_valid_tf_scores():
    dt = pd.Timestamp("2025-01-01 00:00:00")
    # 1h 在 dt 有效；4h 只有未来评分点，应视为无效，不应计入 coverage
    score_map = {
        "1h": {dt: (10.0, 50.0)},
        "4h": {dt + pd.Timedelta(hours=6): (20.0, 40.0)},
    }

    ss, bs, meta = opt_six.calc_multi_tf_consensus(
        score_map,
        ["1h", "4h"],
        dt,
        {"short_threshold": 25, "long_threshold": 40, "coverage_min": 0.5},
    )

    # 只应按 1h 权重计 coverage: 8 / (8+15)
    cov = 8 / 23
    assert meta["coverage"] == pytest.approx(cov, abs=1e-3)
    # coverage<1 时融合分数会按 coverage 衰减
    assert ss == pytest.approx(10.0 * cov, abs=1e-6)
    assert bs == pytest.approx(50.0 * cov, abs=1e-6)
    assert meta["decision"]["actionable"] is False
    assert "覆盖不足" in meta["decision"]["label"]


def test_calc_multi_tf_consensus_includes_book_features_meta():
    dt = pd.Timestamp("2025-01-01 00:00:00")
    score_map = {
        "1h": {dt: (30.0, 10.0)},
        "4h": {dt: (50.0, 5.0)},
        "__feature_map__": {
            "1h": {dt: {"div_sell": 20.0, "ma_sell": 15.0, "vp_sell": 10.0}},
            "4h": {dt: {"div_sell": 40.0, "ma_sell": 25.0, "vp_sell": 12.0}},
        },
    }

    ss, bs, meta = opt_six.calc_multi_tf_consensus(
        score_map,
        ["1h", "4h"],
        dt,
        {"short_threshold": 25, "long_threshold": 40, "coverage_min": 0.0},
    )

    assert ss > bs
    assert "book_features_by_tf" in meta
    assert "book_features_weighted" in meta
    assert "1h" in meta["book_features_by_tf"]
    assert "4h" in meta["book_features_by_tf"]
    # 4h权重(15) > 1h权重(8)，加权后的div_sell应靠近4h值
    assert meta["book_features_weighted"]["div_sell"] > 30.0
    assert meta["book_features_weighted"]["ma_sell"] > 20.0


def test_oos_selection_uses_multi_tf_objective(monkeypatch):
    df_ref = _make_ohlcv(n=24 * 220, start=100, step=0.02)
    df_fast = _make_ohlcv(n=24 * 220, start=100, step=0.02)
    df_mid = _make_ohlcv(n=24 * 220, start=100, step=0.02)
    df_slow = _make_ohlcv(n=24 * 220, start=100, step=0.02)
    all_data = {
        "1h": df_ref,
        "15m": df_fast,
        "4h": df_mid,
        "12h": df_slow,
    }

    calls = {"multi": 0, "single": 0, "decision_tfs": []}

    def fake_compute_signals_six(_df, _tf, _all, max_bars=0):
        return {"ok": True, "max_bars": max_bars}

    def fake_build_tf_score_index(_all_data, _all_signals, _tfs, _cfg):
        return {"dummy": {}}

    def fake_run_strategy(*_args, **_kwargs):
        calls["single"] += 1
        raise AssertionError("OOS should use multi-TF objective, not single-TF run_strategy")

    def fake_run_strategy_multi_tf(primary_df, tf_score_map, decision_tfs, config, primary_tf="1h", trade_days=30, **_kwargs):
        calls["multi"] += 1
        calls["decision_tfs"].append(tuple(decision_tfs))
        alpha_map = {
            "candidate_top": 20.0,
            "candidate_best": 10.0,
            "fallback_cfg": 12.0,
            "generic_conservative": 2.0,
            "generic_moderate": 3.0,
            "generic_high_bar": 1.0,
        }
        alpha = alpha_map.get(config.get("tag", ""), 0.0)
        return {
            "alpha": alpha,
            "max_drawdown": -5.0,
            "strategy_return": alpha,
            "buy_hold_return": 0.0,
            "total_trades": 5,
            "liquidations": 0,
            "fees": {"total_costs": 0.0},
        }

    monkeypatch.setattr(bt_60307, "compute_signals_six", fake_compute_signals_six)
    monkeypatch.setattr(bt_60307, "_build_tf_score_index", fake_build_tf_score_index)
    monkeypatch.setattr(bt_60307, "run_strategy", fake_run_strategy)
    monkeypatch.setattr(bt_60307, "run_strategy_multi_tf", fake_run_strategy_multi_tf)

    opt_result = {
        "global_best": {"config": {"tag": "candidate_best"}},
        "global_top30": [{"config": {"tag": "candidate_top"}}],
    }
    fallback_cfg = {"tag": "fallback_cfg", "coverage_min": 0.5}

    selected_cfg, meta = bt_60307._select_oos_base_config(
        opt_result=opt_result,
        all_data=all_data,
        test_days=60,
        train_days=60,
        tf_hours={"1h": 1, "15m": 0.25, "4h": 4, "12h": 12},
        fallback_config=fallback_cfg,
        ref_tf="1h",
        top_n=2,
        conservative=False,
        decision_tfs=["15m", "1h", "4h", "12h"],
    )

    assert calls["single"] == 0
    assert calls["multi"] > 0
    assert all(tfs == ("15m", "1h", "4h", "12h") for tfs in calls["decision_tfs"])
    assert selected_cfg["tag"] == "candidate_top"
    assert meta["objective_mode"] == "multi_tf"
    assert meta["decision_tfs"] == ["15m", "1h", "4h", "12h"]


def test_oos_combo_selection_selects_per_combo_primary(monkeypatch):
    df_ref = _make_ohlcv(n=24 * 220, start=100, step=0.02)
    all_data = {
        "15m": df_ref.copy(),
        "1h": df_ref.copy(),
        "2h": df_ref.copy(),
        "4h": df_ref.copy(),
        "12h": df_ref.copy(),
    }

    def fake_compute_signals_six(_df, _tf, _all, max_bars=0):
        return {"ok": True, "max_bars": max_bars}

    def fake_build_tf_score_index(_all_data, _all_signals, _tfs, _cfg):
        return {"dummy": {}}

    def fake_run_strategy_multi_tf(primary_df=None, tf_score_map=None, decision_tfs=None, config=None, primary_tf="1h", **_kwargs):
        key = f"{'+'.join(decision_tfs)}@{primary_tf}"
        tag = config.get("tag", "")
        alpha_map = {
            "15m+1h+4h+12h@1h": {"candidate_top": 16.0, "candidate_best": 7.0, "fallback_cfg": 10.0},
            "1h+2h+4h@2h": {"candidate_top": 4.0, "candidate_best": 12.0, "fallback_cfg": 8.0},
        }
        alpha = alpha_map.get(key, {}).get(tag, 0.0)
        return {
            "alpha": alpha,
            "max_drawdown": -6.0,
            "strategy_return": alpha,
            "buy_hold_return": 0.0,
            "total_trades": 4,
            "liquidations": 0,
            "fees": {"total_costs": 0.0},
        }

    monkeypatch.setattr(bt_60307, "compute_signals_six", fake_compute_signals_six)
    monkeypatch.setattr(bt_60307, "_build_tf_score_index", fake_build_tf_score_index)
    monkeypatch.setattr(bt_60307, "run_strategy_multi_tf", fake_run_strategy_multi_tf)

    opt_result = {
        "global_best": {"config": {"tag": "candidate_best"}},
        "global_top30": [{"config": {"tag": "candidate_top"}}],
    }
    fallback_cfg = {"tag": "fallback_cfg", "coverage_min": 0.5}
    combos = [
        ("均衡搭配", ["15m", "1h", "4h", "12h"]),
        ("中大周期", ["1h", "2h", "4h"]),
    ]

    combo_map, meta = bt_60307._select_oos_combo_configs(
        opt_result=opt_result,
        all_data=all_data,
        test_days=60,
        train_days=60,
        tf_hours={"15m": 0.25, "1h": 1, "2h": 2, "4h": 4, "12h": 12},
        fallback_config=fallback_cfg,
        ref_tf="1h",
        combos=combos,
        primary_tfs=["1h", "2h"],
        top_n=2,
        conservative=False,
    )

    key_a = bt_60307._combo_primary_key("均衡搭配", "1h", ["15m", "1h", "4h", "12h"])
    key_b = bt_60307._combo_primary_key("中大周期", "2h", ["1h", "2h", "4h"])
    assert combo_map[key_a]["tag"] == "candidate_top"
    assert combo_map[key_b]["tag"] == "candidate_best"
    assert meta["mode"] == "strict_oos_combo"
    assert meta["objective_mode"] == "multi_tf_combo"
    assert meta["selected_targets"] >= 2


def test_microstructure_overlay_can_block_crowded_direction():
    ss, bs, overlay = opt_six._apply_microstructure_overlay(
        ss=20.0,
        bs=20.0,
        micro_state={"valid": True, "long_score": 0.0, "short_score": 3.0, "basis_z": 0.5},
        config={"use_microstructure": True},
    )

    assert overlay["block_long"] is True
    assert overlay["block_short"] is False
    assert ss > 20.0
    assert bs < 20.0


def test_vol_target_scale_uses_previous_completed_bar():
    vol_ann = pd.Series([np.nan, 1.70, 0.30], index=pd.date_range("2025-01-01", periods=3, freq="h"))
    scale, rv, active = opt_six._compute_vol_target_scale(
        vol_ann,
        idx=2,
        config={
            "vol_target_annual": 0.85,
            "vol_target_min_scale": 0.45,
            "vol_target_max_scale": 1.35,
        },
    )

    assert active is True
    assert rv == pytest.approx(1.70)
    assert scale == pytest.approx(0.5, abs=1e-6)


def test_run_strategy_multi_tf_live_gate_blocks_non_actionable(monkeypatch):
    df = _make_ohlcv(n=180, start=100, step=0.2)

    def fake_consensus(_map, _tfs, _dt, _cfg):
        return 80.0, 0.0, {
            "decision": {"direction": "short", "actionable": False, "strength": 100},
            "coverage": 1.0,
        }

    monkeypatch.setattr(opt_six, "calc_multi_tf_consensus", fake_consensus)

    cfg = _base_exec_config()
    cfg.update({"use_live_gate": True, "consensus_min_strength": 40, "coverage_min": 0.5})
    result = opt_six.run_strategy_multi_tf(
        primary_df=df,
        tf_score_map={"1h": {}},
        decision_tfs=["1h"],
        config=cfg,
        primary_tf="1h",
        trade_days=30,
    )

    assert result["total_trades"] == 0


def test_run_strategy_multi_tf_live_gate_allows_actionable_direction(monkeypatch):
    df = _make_ohlcv(n=180, start=120, step=-0.2)

    def fake_consensus(_map, _tfs, _dt, _cfg):
        return 80.0, 0.0, {
            "decision": {"direction": "short", "actionable": True, "strength": 90},
            "coverage": 1.0,
        }

    monkeypatch.setattr(opt_six, "calc_multi_tf_consensus", fake_consensus)

    cfg = _base_exec_config()
    cfg.update({"use_live_gate": True, "consensus_min_strength": 40, "coverage_min": 0.5})
    result = opt_six.run_strategy_multi_tf(
        primary_df=df,
        tf_score_map={"1h": {}},
        decision_tfs=["1h"],
        config=cfg,
        primary_tf="1h",
        trade_days=30,
    )

    assert any(t["action"] == "OPEN_SHORT" for t in result["trades"])


def test_build_walk_forward_windows_respects_train_test_split():
    df = _make_ohlcv(n=24 * 260, start=100, step=0.01)  # ~260 days
    windows = bt_60307._build_walk_forward_windows(
        df,
        train_days=90,
        test_days=7,
        step_days=7,
        windows=4,
    )

    assert len(windows) == 4
    assert windows[0]["test_end"] < windows[-1]["test_end"]
    for w in windows:
        assert w["train_end"] == w["test_start"]
        assert w["train_start"] < w["train_end"]
        assert w["train_start"] >= df.index[0]


def test_summarize_walk_forward_rankings_penalizes_unstable_combo():
    rows = []
    for i, a in enumerate([10.0, 8.5, 9.0], start=1):
        rows.append(
            {
                "window_id": f"wf_{i}",
                "combo_name": "稳健组",
                "primary_tf": "1h",
                "decision_tfs": ["30m", "1h", "4h"],
                "alpha": a,
                "max_drawdown": -10.0,
                "total_trades": 10,
            }
        )
    for i, a in enumerate([20.0, -6.0, 22.0], start=1):
        rows.append(
            {
                "window_id": f"wf_{i}",
                "combo_name": "波动组",
                "primary_tf": "1h",
                "decision_tfs": ["15m", "1h", "8h"],
                "alpha": a,
                "max_drawdown": -30.0,
                "total_trades": 12,
            }
        )

    rankings = bt_60307._summarize_walk_forward_rankings(rows, min_windows=3)

    assert len(rankings) == 2
    assert rankings[0]["combo"] == "稳健组@1h"
    assert rankings[0]["min_alpha"] > rankings[1]["min_alpha"]


def _make_choppy_ohlcv(n=240, start=100.0):
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    x = np.arange(n, dtype=float)
    trend = start + 0.01 * x
    noise = 2.2 * np.sin(x * 0.9) + np.where((x % 2) == 0, 1.6, -1.6)
    close = trend + noise
    open_ = close + np.where((x % 3) == 0, 0.8, -0.8)
    high = np.maximum(open_, close) + 1.2
    low = np.minimum(open_, close) - 1.2
    volume = np.full(n, 1200.0)
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


def _make_downtrend_ohlcv(n=240, start=220.0):
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    x = np.arange(n, dtype=float)
    close = start - 0.45 * x + 0.25 * np.sin(x / 4.0)
    open_ = close + 0.15
    high = np.maximum(open_, close) + 0.4
    low = np.minimum(open_, close) - 0.4
    volume = np.full(n, 900.0)
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


def test_regime_controls_high_vol_choppy_become_more_conservative():
    df = _make_choppy_ohlcv()
    cfg = {
        "use_regime_aware": True,
        "short_threshold": 25,
        "long_threshold": 40,
        "close_short_bs": 40,
        "close_long_ss": 40,
        "lev": 5,
        "margin_use": 0.50,
    }

    controls = opt_six._compute_regime_controls(df, len(df) - 1, cfg)

    assert controls["short_threshold"] > 25
    assert controls["long_threshold"] > 40
    assert controls["close_short_bs"] <= 40
    assert controls["close_long_ss"] <= 40
    assert controls["lev"] <= 3
    assert controls["margin_use"] < 0.50


def test_regime_controls_downtrend_biases_toward_short_side():
    df = _make_downtrend_ohlcv()
    cfg = {
        "use_regime_aware": True,
        "short_threshold": 25,
        "long_threshold": 40,
        "close_short_bs": 40,
        "close_long_ss": 40,
        "lev": 5,
        "margin_use": 0.50,
    }

    controls = opt_six._compute_regime_controls(df, len(df) - 1, cfg)

    assert controls["short_threshold"] < 25
    assert controls["long_threshold"] > 40


def test_regime_controls_disabled_keeps_original_parameters():
    df = _make_choppy_ohlcv()
    cfg = {
        "use_regime_aware": False,
        "short_threshold": 25,
        "long_threshold": 40,
        "close_short_bs": 41,
        "close_long_ss": 42,
        "lev": 5,
        "margin_use": 0.50,
    }
    controls = opt_six._compute_regime_controls(df, len(df) - 1, cfg)

    assert controls["short_threshold"] == 25
    assert controls["long_threshold"] == 40
    assert controls["close_short_bs"] == 41
    assert controls["close_long_ss"] == 42
    assert controls["lev"] == 5
    assert controls["margin_use"] == pytest.approx(0.50)


def test_extract_realized_pnl_from_trade_parses_close_and_liquidation():
    close_trade = {"action": "CLOSE_SHORT", "reason": "反向平空 BS=52 PnL=-123.45"}
    liq_trade = {"action": "LIQUIDATED", "reason": "空仓强平,损失678.9(含清算费12.3)"}
    open_trade = {"action": "OPEN_SHORT", "reason": "开空 3x"}

    assert opt_six._extract_realized_pnl_from_trade(close_trade) == pytest.approx(-123.45)
    assert opt_six._extract_realized_pnl_from_trade(liq_trade) == pytest.approx(-678.9)
    assert opt_six._extract_realized_pnl_from_trade(open_trade) is None


def test_protection_daily_lock_resets_next_day():
    cfg = {
        "use_protections": True,
        "prot_daily_loss_limit_pct": 0.03,
        "prot_global_dd_limit_pct": 0.5,
    }
    state = opt_six._init_protection_state(cfg, initial_equity=100000.0)
    t1 = pd.Timestamp("2025-01-01 10:00:00")
    t2 = pd.Timestamp("2025-01-01 12:00:00")
    t3 = pd.Timestamp("2025-01-02 00:00:00")

    opt_six._update_protection_risk_state(state, t1, equity=100000.0, idx=1, config=cfg)
    allowed, reason = opt_six._protection_entry_allowed(state, 1)
    assert allowed is True
    assert reason is None

    opt_six._update_protection_risk_state(state, t2, equity=96000.0, idx=2, config=cfg)
    allowed, reason = opt_six._protection_entry_allowed(state, 2)
    assert allowed is False
    assert reason == "daily_loss_limit"

    opt_six._update_protection_risk_state(state, t3, equity=96000.0, idx=3, config=cfg)
    allowed, reason = opt_six._protection_entry_allowed(state, 3)
    assert allowed is True
    assert reason is None


def test_protection_loss_streak_blocks_entries_for_cooldown():
    cfg = {
        "use_protections": True,
        "prot_loss_streak_limit": 2,
        "prot_loss_streak_cooldown_bars": 5,
    }
    state = opt_six._init_protection_state(cfg, initial_equity=100000.0)
    opt_six._apply_loss_streak_protection(state, pnl=-10.0, idx=10, config=cfg)
    allowed, _ = opt_six._protection_entry_allowed(state, 11)
    assert allowed is True

    opt_six._apply_loss_streak_protection(state, pnl=-8.0, idx=12, config=cfg)
    allowed, reason = opt_six._protection_entry_allowed(state, 13)
    assert allowed is False
    assert reason == "loss_streak_cooldown"

    allowed, reason = opt_six._protection_entry_allowed(state, 18)
    assert allowed is True
    assert reason is None


def test_protection_global_halt_blocks_entries():
    cfg = {
        "use_protections": True,
        "prot_daily_loss_limit_pct": 0.5,
        "prot_global_dd_limit_pct": 0.10,
    }
    state = opt_six._init_protection_state(cfg, initial_equity=100000.0)
    t = pd.Timestamp("2025-01-01 10:00:00")
    opt_six._update_protection_risk_state(state, t, equity=89000.0, idx=1, config=cfg)

    allowed, reason = opt_six._protection_entry_allowed(state, 1)
    assert allowed is False
    assert reason == "global_halt"


def test_run_strategy_core_with_protections_reduces_reentry_after_losses():
    df = _make_ohlcv(n=180, start=100, step=0.5)  # 持续上涨, 空头容易连续止损

    def score_provider(_idx, _dt, _price):
        return 80.0, 0.0  # 持续给出开空信号

    cfg_plain = _base_exec_config()
    cfg_plain.update({"short_sl": -0.01, "cooldown": 1})
    r_plain = opt_six._run_strategy_core(
        primary_df=df,
        config=cfg_plain,
        primary_tf="1h",
        trade_days=30,
        score_provider=score_provider,
    )

    cfg_prot = dict(cfg_plain)
    cfg_prot.update(
        {
            "use_protections": True,
            "prot_loss_streak_limit": 1,
            "prot_loss_streak_cooldown_bars": 40,
            "prot_daily_loss_limit_pct": 0.50,
            "prot_global_dd_limit_pct": 0.95,
        }
    )
    r_prot = opt_six._run_strategy_core(
        primary_df=df,
        config=cfg_prot,
        primary_tf="1h",
        trade_days=30,
        score_provider=score_provider,
    )

    assert r_prot["total_trades"] < r_plain["total_trades"]
    assert "protections" in r_prot
    assert r_prot["protections"]["streak_lock_count"] >= 1
