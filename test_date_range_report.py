import pandas as pd
import pytest

import date_range_report as dr


def test_normalize_trade_records_keeps_schema_and_extra_fields():
    raw = [
        {
            "time": "2025-01-01T00:00:00",
            "action": "OPEN_SHORT",
            "direction": "short",
            "market_price": 2000.0,
            "notional_value": 1000.0,
            "fee": 1.0,
            "custom_note": "x",
        }
    ]

    rows = dr.normalize_trade_records(raw)

    assert len(rows) == 1
    row = rows[0]
    for field in dr.TRADE_RECORD_FIELDS:
        assert field in row
    assert row["custom_note"] == "x"


def test_build_daily_trade_summary_covers_full_date_range():
    trades = [
        {
            "time": "2025-01-01T10:00:00",
            "action": "OPEN_SHORT",
            "notional_value": 1000.0,
            "fee": 1.0,
            "slippage_cost": 0.5,
            "total_cost": 1.5,
            "pnl": 0.0,
        },
        {
            "time": "2025-01-01T20:00:00",
            "action": "CLOSE_SHORT",
            "notional_value": 1100.0,
            "fee": 1.1,
            "slippage_cost": 0.6,
            "total_cost": 1.7,
            "pnl": 50.0,
        },
        {
            "time": "2025-01-03T09:00:00",
            "action": "LIQUIDATED",
            "notional_value": 900.0,
            "fee": 4.0,
            "slippage_cost": 0.8,
            "total_cost": 4.8,
            "pnl": -120.0,
        },
    ]

    daily = dr.build_daily_trade_summary(trades, "2025-01-01", "2025-01-03")

    assert list(daily.index.strftime("%Y-%m-%d")) == ["2025-01-01", "2025-01-02", "2025-01-03"]
    assert int(daily.loc[pd.Timestamp("2025-01-01"), "trade_count"]) == 2
    assert int(daily.loc[pd.Timestamp("2025-01-01"), "open_short_count"]) == 1
    assert int(daily.loc[pd.Timestamp("2025-01-01"), "close_short_count"]) == 1
    assert daily.loc[pd.Timestamp("2025-01-01"), "realized_pnl"] == pytest.approx(50.0)
    assert int(daily.loc[pd.Timestamp("2025-01-03"), "liquidations"]) == 1
    assert daily.loc[pd.Timestamp("2025-01-03"), "realized_pnl"] == pytest.approx(-120.0)
    assert int(daily.loc[pd.Timestamp("2025-01-02"), "trade_count"]) == 0


def test_build_daily_equity_positions_uses_daily_last_and_trade_state():
    history = [
        {
            "time": "2025-01-01T01:00:00",
            "total": 200000.0,
            "usdt": 100000.0,
            "spot_eth_value": 100000.0,
            "long_pnl": 0.0,
            "short_pnl": 0.0,
            "frozen_margin": 0.0,
            "eth_price": 2000.0,
        },
        {
            "time": "2025-01-01T23:00:00",
            "total": 201000.0,
            "usdt": 101000.0,
            "spot_eth_value": 100000.0,
            "long_pnl": 0.0,
            "short_pnl": 0.0,
            "frozen_margin": 300.0,
            "eth_price": 2020.0,
        },
        {
            "time": "2025-01-02T22:00:00",
            "total": 198000.0,
            "usdt": 98000.0,
            "spot_eth_value": 100000.0,
            "long_pnl": 0.0,
            "short_pnl": -500.0,
            "frozen_margin": 280.0,
            "eth_price": 1980.0,
        },
        {
            "time": "2025-01-03T23:00:00",
            "total": 205000.0,
            "usdt": 102000.0,
            "spot_eth_value": 103000.0,
            "long_pnl": 0.0,
            "short_pnl": 0.0,
            "frozen_margin": 0.0,
            "eth_price": 2050.0,
        },
    ]
    trades = [
        {
            "time": "2025-01-01T10:00:00",
            "action": "OPEN_SHORT",
            "has_long": False,
            "has_short": True,
            "long_qty": None,
            "short_qty": 2.0,
        },
        {
            "time": "2025-01-02T11:00:00",
            "action": "CLOSE_SHORT",
            "has_long": False,
            "has_short": False,
            "long_qty": None,
            "short_qty": None,
        },
    ]

    daily = dr.build_daily_equity_positions(
        history=history,
        trades=trades,
        start_date="2025-01-01",
        end_date="2025-01-03",
    )

    assert list(daily.index.strftime("%Y-%m-%d")) == ["2025-01-01", "2025-01-02", "2025-01-03"]
    assert daily.loc[pd.Timestamp("2025-01-01"), "total"] == pytest.approx(201000.0)
    assert daily.loc[pd.Timestamp("2025-01-02"), "total"] == pytest.approx(198000.0)
    assert daily.loc[pd.Timestamp("2025-01-03"), "total"] == pytest.approx(205000.0)
    assert daily.loc[pd.Timestamp("2025-01-02"), "daily_pnl"] == pytest.approx(-3000.0)
    assert daily.loc[pd.Timestamp("2025-01-03"), "daily_pnl"] == pytest.approx(7000.0)
    assert bool(daily.loc[pd.Timestamp("2025-01-01"), "has_short"]) is True
    assert bool(daily.loc[pd.Timestamp("2025-01-02"), "has_short"]) is False
    assert bool(daily.loc[pd.Timestamp("2025-01-03"), "has_short"]) is False


def test_save_and_load_latest_report_from_db(tmp_path):
    db_path = tmp_path / "report.db"
    report = {
        "run_meta": {
            "runner": "pytest",
            "symbol": "ETHUSDT",
            "primary_tf": "1h",
            "decision_tfs": ["30m", "1h", "4h"],
            "start_date": "2025-01-01",
            "end_date": "2025-01-03",
        },
        "summary": {
            "strategy_return": 12.3,
            "alpha": 5.6,
            "max_drawdown": -8.8,
            "total_trades": 9,
        },
        "daily_records": [
            {"date": "2025-01-01", "total": 200000.0, "daily_pnl": 0.0},
            {"date": "2025-01-02", "total": 199000.0, "daily_pnl": -1000.0},
        ],
        "trades": [
            {"time": "2025-01-01T10:00:00", "action": "OPEN_SHORT", "fee": 1.0},
            {"time": "2025-01-02T10:00:00", "action": "CLOSE_SHORT", "fee": 1.2},
        ],
    }

    run_id = dr.save_report_to_db(str(db_path), report)
    assert isinstance(run_id, int)
    assert run_id > 0

    loaded = dr.load_latest_report_from_db(str(db_path))
    assert loaded is not None
    assert loaded["run_meta"]["runner"] == "pytest"
    assert loaded["summary"]["strategy_return"] == pytest.approx(12.3)
    assert len(loaded["daily_records"]) == 2
    assert len(loaded["trades"]) == 2
