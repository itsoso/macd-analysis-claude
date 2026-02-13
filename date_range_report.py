import json
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd


TRADE_RECORD_FIELDS = [
    "time",
    "action",
    "direction",
    "market_price",
    "exec_price",
    "quantity",
    "notional_value",
    "fee",
    "slippage_cost",
    "total_cost",
    "leverage",
    "margin",
    "margin_released",
    "pnl",
    "entry_price",
    "partial_ratio",
    "after_usdt",
    "after_spot_eth",
    "after_frozen_margin",
    "after_total",
    "after_available",
    "has_long",
    "has_short",
    "long_entry",
    "long_qty",
    "short_entry",
    "short_qty",
    "cum_spot_fees",
    "cum_futures_fees",
    "cum_funding_paid",
    "cum_slippage",
    "reason",
]


_ZERO_TRADE_COLS = [
    "trade_count",
    "open_long_count",
    "close_long_count",
    "open_short_count",
    "close_short_count",
    "spot_buy_count",
    "spot_sell_count",
    "partial_tp_count",
    "liquidations",
    "notional_turnover",
    "total_fee",
    "total_slippage",
    "total_cost",
    "realized_pnl",
]


def _to_date_index(start_date, end_date):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError(f"end_date({end_date}) must be >= start_date({start_date})")
    return pd.date_range(start, end, freq="D"), start, end


def _safe_num(v, default=0.0):
    if v is None:
        return default
    try:
        if isinstance(v, str) and not v.strip():
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def normalize_trade_records(trades):
    rows = []
    for trade in trades or []:
        row = {k: trade.get(k) for k in TRADE_RECORD_FIELDS}
        for k, v in trade.items():
            if k not in row:
                row[k] = v
        rows.append(row)
    return rows


def build_daily_trade_summary(trades, start_date, end_date):
    dates, start, end = _to_date_index(start_date, end_date)
    if not trades:
        return pd.DataFrame(0.0, index=dates, columns=_ZERO_TRADE_COLS)

    rows = normalize_trade_records(trades)
    tdf = pd.DataFrame(rows)
    if tdf.empty:
        return pd.DataFrame(0.0, index=dates, columns=_ZERO_TRADE_COLS)

    tdf["time"] = pd.to_datetime(tdf["time"], errors="coerce")
    tdf = tdf.dropna(subset=["time"])
    tdf = tdf[(tdf["time"] >= start) & (tdf["time"] <= (end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)))]
    if tdf.empty:
        return pd.DataFrame(0.0, index=dates, columns=_ZERO_TRADE_COLS)

    tdf["date"] = tdf["time"].dt.normalize()
    tdf["fee"] = tdf["fee"].map(_safe_num)
    tdf["slippage_cost"] = tdf["slippage_cost"].map(_safe_num)
    tdf["total_cost"] = tdf["total_cost"].map(_safe_num)
    tdf["notional_value"] = tdf["notional_value"].map(_safe_num)
    tdf["pnl"] = tdf["pnl"].map(_safe_num)
    tdf["trade_count"] = 1
    tdf["open_long_count"] = (tdf["action"] == "OPEN_LONG").astype(int)
    tdf["close_long_count"] = (tdf["action"] == "CLOSE_LONG").astype(int)
    tdf["open_short_count"] = (tdf["action"] == "OPEN_SHORT").astype(int)
    tdf["close_short_count"] = (tdf["action"] == "CLOSE_SHORT").astype(int)
    tdf["spot_buy_count"] = (tdf["action"] == "SPOT_BUY").astype(int)
    tdf["spot_sell_count"] = (tdf["action"] == "SPOT_SELL").astype(int)
    tdf["partial_tp_count"] = (tdf["action"] == "PARTIAL_TP").astype(int)
    tdf["liquidations"] = (tdf["action"] == "LIQUIDATED").astype(int)

    # 仅平仓相关动作计入已实现盈亏
    pnl_actions = {"CLOSE_LONG", "CLOSE_SHORT", "LIQUIDATED", "PARTIAL_TP"}
    tdf["realized_pnl"] = np.where(tdf["action"].isin(pnl_actions), tdf["pnl"], 0.0)
    tdf["notional_turnover"] = tdf["notional_value"]
    tdf["total_fee"] = tdf["fee"]
    tdf["total_slippage"] = tdf["slippage_cost"]

    daily = (
        tdf.groupby("date")[_ZERO_TRADE_COLS]
        .sum()
        .reindex(dates)
        .fillna(0.0)
    )
    return daily


def build_daily_equity_positions(history, trades, start_date, end_date):
    dates, _start, end = _to_date_index(start_date, end_date)

    hdf = pd.DataFrame(history or [])
    base_cols = [
        "total",
        "usdt",
        "spot_eth_value",
        "long_pnl",
        "short_pnl",
        "frozen_margin",
        "eth_price",
    ]
    if hdf.empty:
        out = pd.DataFrame(0.0, index=dates, columns=base_cols)
    else:
        hdf["time"] = pd.to_datetime(hdf["time"], errors="coerce")
        hdf = hdf.dropna(subset=["time"]).sort_values("time")
        hdf = hdf[hdf["time"] <= (end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]
        if hdf.empty:
            out = pd.DataFrame(0.0, index=dates, columns=base_cols)
        else:
            for col in base_cols:
                hdf[col] = hdf[col].map(_safe_num)
            hdf["date"] = hdf["time"].dt.normalize()
            out = hdf.groupby("date")[base_cols].last().reindex(dates)
            out = out.ffill().fillna(0.0)

    # 从交易快照中提取每日仓位状态
    pos_cols = ["has_long", "has_short", "long_qty", "short_qty", "long_entry", "short_entry"]
    tdf = pd.DataFrame(normalize_trade_records(trades))
    if tdf.empty:
        pos = pd.DataFrame(index=dates, columns=pos_cols)
        pos["has_long"] = False
        pos["has_short"] = False
        pos["long_qty"] = 0.0
        pos["short_qty"] = 0.0
        pos["long_entry"] = np.nan
        pos["short_entry"] = np.nan
    else:
        tdf["time"] = pd.to_datetime(tdf["time"], errors="coerce")
        tdf = tdf.dropna(subset=["time"]).sort_values("time")
        tdf = tdf[tdf["time"] <= (end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]
        if tdf.empty:
            pos = pd.DataFrame(index=dates, columns=pos_cols)
            pos["has_long"] = False
            pos["has_short"] = False
            pos["long_qty"] = 0.0
            pos["short_qty"] = 0.0
            pos["long_entry"] = np.nan
            pos["short_entry"] = np.nan
        else:
            tdf["date"] = tdf["time"].dt.normalize()
            for c in ["long_qty", "short_qty", "long_entry", "short_entry"]:
                tdf[c] = tdf[c].map(_safe_num)
            tdf["has_long"] = tdf["has_long"].astype(bool)
            tdf["has_short"] = tdf["has_short"].astype(bool)
            pos = tdf.groupby("date")[pos_cols].last().reindex(dates).ffill()
            pos["has_long"] = pos["has_long"].fillna(False).astype(bool)
            pos["has_short"] = pos["has_short"].fillna(False).astype(bool)
            pos["long_qty"] = pos["long_qty"].fillna(0.0)
            pos["short_qty"] = pos["short_qty"].fillna(0.0)

    daily = out.join(pos, how="left")
    daily["daily_pnl"] = daily["total"].diff().fillna(0.0)
    prev_total = daily["total"].shift(1)
    daily["daily_return_pct"] = np.where(prev_total > 0, daily["daily_pnl"] / prev_total * 100.0, 0.0)
    daily["equity_peak"] = daily["total"].cummax()
    daily["drawdown_pct"] = np.where(
        daily["equity_peak"] > 0,
        (daily["total"] - daily["equity_peak"]) / daily["equity_peak"] * 100.0,
        0.0,
    )
    return daily


def init_report_db(db_path):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS report_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                runner TEXT,
                symbol TEXT,
                primary_tf TEXT,
                decision_tfs_json TEXT,
                start_date TEXT,
                end_date TEXT,
                summary_json TEXT,
                meta_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS report_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                data_json TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES report_runs(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS report_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                seq INTEGER NOT NULL,
                time TEXT,
                action TEXT,
                direction TEXT,
                market_price REAL,
                exec_price REAL,
                quantity REAL,
                notional_value REAL,
                fee REAL,
                total_cost REAL,
                data_json TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES report_runs(id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_report_daily_run_date ON report_daily(run_id, date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_report_trades_run_seq ON report_trades(run_id, seq)")
        conn.commit()
    finally:
        conn.close()


def save_report_to_db(db_path, report):
    init_report_db(db_path)
    run_meta = dict(report.get("run_meta", {}))
    summary = dict(report.get("summary", {}))
    daily_records = list(report.get("daily_records", []))
    trades = normalize_trade_records(report.get("trades", []))

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO report_runs(
                created_at, runner, symbol, primary_tf, decision_tfs_json,
                start_date, end_date, summary_json, meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(),
                run_meta.get("runner"),
                run_meta.get("symbol"),
                run_meta.get("primary_tf"),
                json.dumps(run_meta.get("decision_tfs", []), ensure_ascii=False),
                run_meta.get("start_date"),
                run_meta.get("end_date"),
                json.dumps(summary, ensure_ascii=False),
                json.dumps(run_meta, ensure_ascii=False),
            ),
        )
        run_id = int(cur.lastrowid)

        for row in daily_records:
            cur.execute(
                "INSERT INTO report_daily(run_id, date, data_json) VALUES (?, ?, ?)",
                (
                    run_id,
                    str(row.get("date")),
                    json.dumps(row, ensure_ascii=False),
                ),
            )

        for i, tr in enumerate(trades, start=1):
            cur.execute(
                """
                INSERT INTO report_trades(
                    run_id, seq, time, action, direction, market_price, exec_price,
                    quantity, notional_value, fee, total_cost, data_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    i,
                    tr.get("time"),
                    tr.get("action"),
                    tr.get("direction"),
                    _safe_num(tr.get("market_price"), None),
                    _safe_num(tr.get("exec_price"), None),
                    _safe_num(tr.get("quantity"), None),
                    _safe_num(tr.get("notional_value"), None),
                    _safe_num(tr.get("fee"), None),
                    _safe_num(tr.get("total_cost"), None),
                    json.dumps(tr, ensure_ascii=False),
                ),
            )
        conn.commit()
        return run_id
    finally:
        conn.close()


def load_latest_report_from_db(db_path):
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        run = cur.execute(
            "SELECT * FROM report_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not run:
            return None
        run_id = int(run["id"])
        daily_rows = cur.execute(
            "SELECT data_json FROM report_daily WHERE run_id = ? ORDER BY date ASC, id ASC",
            (run_id,),
        ).fetchall()
        trade_rows = cur.execute(
            "SELECT data_json FROM report_trades WHERE run_id = ? ORDER BY seq ASC",
            (run_id,),
        ).fetchall()

        run_meta = json.loads(run["meta_json"]) if run["meta_json"] else {}
        summary = json.loads(run["summary_json"]) if run["summary_json"] else {}
        daily_records = [json.loads(r["data_json"]) for r in daily_rows]
        trades = [json.loads(r["data_json"]) for r in trade_rows]

        return {
            "run_id": run_id,
            "created_at": run["created_at"],
            "run_meta": run_meta,
            "summary": summary,
            "daily_records": daily_records,
            "trades": trades,
            "trade_record_fields": TRADE_RECORD_FIELDS,
        }
    finally:
        conn.close()
