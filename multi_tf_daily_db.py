"""
多周期联合决策 — SQLite 逐日盈亏数据库存储层
===============================================
表结构:
  mtf_runs      — 每次回测运行的元数据和汇总
  mtf_daily     — 每日持仓 + 盈亏快照
  mtf_trades    — 每笔交易完整明细
"""

import json
import os
import sqlite3
from datetime import datetime

DB_FILENAME = 'multi_tf_daily_backtest.db'


def _default_db_path():
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, 'data', 'backtests', DB_FILENAME)


def init_db(db_path=None):
    db_path = db_path or _default_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS mtf_runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      TEXT NOT NULL,
                runner          TEXT,
                host            TEXT,
                start_date      TEXT,
                end_date        TEXT,
                primary_tf      TEXT,
                decision_tfs    TEXT,
                combo_name      TEXT,
                leverage        INTEGER,
                initial_capital REAL,
                summary_json    TEXT,
                meta_json       TEXT
            );

            CREATE TABLE IF NOT EXISTS mtf_daily (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          INTEGER NOT NULL,
                date            TEXT NOT NULL,
                eth_price       REAL,
                total_value     REAL,
                usdt            REAL,
                frozen_margin   REAL,
                long_pnl        REAL,
                short_pnl       REAL,
                spot_eth_value  REAL,
                return_pct      REAL,
                drawdown_pct    REAL,
                has_long        INTEGER,
                has_short       INTEGER,
                long_entry      REAL,
                long_qty        REAL,
                short_entry     REAL,
                short_qty       REAL,
                day_trades      INTEGER,
                day_pnl         REAL,
                data_json       TEXT,
                FOREIGN KEY(run_id) REFERENCES mtf_runs(id)
            );

            CREATE TABLE IF NOT EXISTS mtf_trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          INTEGER NOT NULL,
                seq             INTEGER NOT NULL,
                time            TEXT,
                action          TEXT,
                direction       TEXT,
                market_price    REAL,
                exec_price      REAL,
                quantity        REAL,
                notional_value  REAL,
                margin          REAL,
                leverage        INTEGER,
                fee             REAL,
                slippage_cost   REAL,
                pnl             REAL,
                after_total     REAL,
                reason          TEXT,
                data_json       TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES mtf_runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_mtf_daily_run_date
                ON mtf_daily(run_id, date);
            CREATE INDEX IF NOT EXISTS idx_mtf_trades_run_seq
                ON mtf_trades(run_id, seq);
        """)
        conn.commit()
    finally:
        conn.close()


def save_run(db_path, run_meta, summary, daily_records, trades):
    """保存一次完整回测到 DB。Returns run_id."""
    db_path = db_path or _default_db_path()
    init_db(db_path)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO mtf_runs(
                created_at, runner, host, start_date, end_date,
                primary_tf, decision_tfs, combo_name,
                leverage, initial_capital, summary_json, meta_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.now().isoformat(),
            run_meta.get('runner', 'local'),
            run_meta.get('host', ''),
            run_meta.get('start_date', ''),
            run_meta.get('end_date', ''),
            run_meta.get('primary_tf', '1h'),
            json.dumps(run_meta.get('decision_tfs', []), ensure_ascii=False),
            run_meta.get('combo_name', ''),
            run_meta.get('leverage', 5),
            run_meta.get('initial_capital', 200000),
            json.dumps(summary, ensure_ascii=False),
            json.dumps(run_meta, ensure_ascii=False),
        ))
        run_id = cur.lastrowid

        for rec in daily_records:
            cur.execute("""
                INSERT INTO mtf_daily(
                    run_id, date, eth_price, total_value, usdt,
                    frozen_margin, long_pnl, short_pnl, spot_eth_value,
                    return_pct, drawdown_pct,
                    has_long, has_short, long_entry, long_qty,
                    short_entry, short_qty,
                    day_trades, day_pnl, data_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                run_id,
                rec.get('date', ''),
                rec.get('eth_price'),
                rec.get('total_value'),
                rec.get('usdt'),
                rec.get('frozen_margin'),
                rec.get('long_pnl'),
                rec.get('short_pnl'),
                rec.get('spot_eth_value'),
                rec.get('return_pct'),
                rec.get('drawdown_pct'),
                1 if rec.get('has_long') else 0,
                1 if rec.get('has_short') else 0,
                rec.get('long_entry'),
                rec.get('long_qty'),
                rec.get('short_entry'),
                rec.get('short_qty'),
                rec.get('day_trades', 0),
                rec.get('day_pnl', 0),
                json.dumps(rec, ensure_ascii=False),
            ))

        for i, tr in enumerate(trades, 1):
            cur.execute("""
                INSERT INTO mtf_trades(
                    run_id, seq, time, action, direction,
                    market_price, exec_price, quantity, notional_value,
                    margin, leverage, fee, slippage_cost, pnl,
                    after_total, reason, data_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                run_id, i,
                tr.get('time'),
                tr.get('action'),
                tr.get('direction'),
                tr.get('market_price'),
                tr.get('exec_price'),
                tr.get('quantity'),
                tr.get('notional_value'),
                tr.get('margin'),
                tr.get('leverage'),
                tr.get('fee'),
                tr.get('slippage_cost'),
                tr.get('pnl'),
                tr.get('after_total'),
                tr.get('reason'),
                json.dumps(tr, ensure_ascii=False),
            ))

        conn.commit()
        return run_id
    finally:
        conn.close()


def load_latest_run(db_path=None):
    """加载最新一次回测的完整数据"""
    db_path = db_path or _default_db_path()
    if not os.path.exists(db_path):
        return None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        run = cur.execute(
            "SELECT * FROM mtf_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not run:
            return None
        run_id = run['id']

        daily_rows = cur.execute(
            "SELECT * FROM mtf_daily WHERE run_id = ? ORDER BY date ASC",
            (run_id,)
        ).fetchall()

        trade_rows = cur.execute(
            "SELECT data_json FROM mtf_trades WHERE run_id = ? ORDER BY seq ASC",
            (run_id,)
        ).fetchall()

        run_meta = json.loads(run['meta_json']) if run['meta_json'] else {}
        summary = json.loads(run['summary_json']) if run['summary_json'] else {}

        daily_records = []
        for r in daily_rows:
            rec = json.loads(r['data_json']) if r['data_json'] else {}
            rec['drawdown_pct'] = r['drawdown_pct']
            daily_records.append(rec)

        trades = [json.loads(r['data_json']) for r in trade_rows]

        equity_curve = [
            {
                'date': r['date'],
                'total': r['total_value'],
                'price': r['eth_price'],
                'drawdown': r['drawdown_pct'],
            }
            for r in daily_rows
        ]

        return {
            'run_id': run_id,
            'run_meta': run_meta,
            'summary': summary,
            'daily_records': daily_records,
            'trade_details': trades,
            'equity_curve': equity_curve,
        }
    finally:
        conn.close()
