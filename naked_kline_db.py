"""
裸K线交易法 — SQLite 数据库存储层
==================================
将回测结果（逐日持仓盈亏 + 交易明细 + 汇总）写入 SQLite，
而非 JSON 文件。支持多次运行历史查询。

表结构:
  nk_runs       — 每次回测运行的元数据和汇总
  nk_daily      — 每日K线 + 持仓 + 盈亏快照
  nk_trades     — 每笔交易完整明细
  nk_signals    — 每日检测到的信号
  nk_book       — 章节解读（静态数据）
"""

import json
import os
import sqlite3
from datetime import datetime


DB_FILENAME = 'naked_kline_backtest.db'


def _default_db_path():
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, 'data', 'backtests', DB_FILENAME)


def init_db(db_path=None):
    """创建表结构"""
    db_path = db_path or _default_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS nk_runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT NOT NULL,
                runner      TEXT,
                host        TEXT,
                start_date  TEXT,
                end_date    TEXT,
                leverage    INTEGER,
                risk_pct    REAL,
                initial_capital REAL,
                summary_json TEXT,
                meta_json   TEXT,
                book_json   TEXT
            );

            CREATE TABLE IF NOT EXISTS nk_daily (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      INTEGER NOT NULL,
                date        TEXT NOT NULL,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                trend       TEXT,
                signal_desc TEXT,
                pos_direction   TEXT,
                pos_entry_price REAL,
                pos_quantity    REAL,
                pos_unrealized_pnl REAL,
                pos_pnl_ratio   REAL,
                pos_stop_loss   REAL,
                pos_tp1         REAL,
                pos_bars_held   INTEGER,
                total_value REAL,
                usdt        REAL,
                return_pct  REAL,
                drawdown_pct REAL,
                data_json   TEXT,
                FOREIGN KEY(run_id) REFERENCES nk_runs(id)
            );

            CREATE TABLE IF NOT EXISTS nk_trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      INTEGER NOT NULL,
                seq         INTEGER NOT NULL,
                time        TEXT,
                action      TEXT,
                direction   TEXT,
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
                close_reason    TEXT,
                pattern         TEXT,
                signal_strength INTEGER,
                at_key_level    INTEGER,
                with_trend      INTEGER,
                bars_held       INTEGER,
                data_json       TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES nk_runs(id)
            );

            CREATE TABLE IF NOT EXISTS nk_signals (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      INTEGER NOT NULL,
                date        TEXT NOT NULL,
                pattern     TEXT,
                direction   TEXT,
                strength    INTEGER,
                entry_price REAL,
                stop_loss   REAL,
                tp1         REAL,
                at_key_level INTEGER,
                with_trend  INTEGER,
                trend       TEXT,
                notes       TEXT,
                FOREIGN KEY(run_id) REFERENCES nk_runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_nk_daily_run_date
                ON nk_daily(run_id, date);
            CREATE INDEX IF NOT EXISTS idx_nk_trades_run_seq
                ON nk_trades(run_id, seq);
            CREATE INDEX IF NOT EXISTS idx_nk_signals_run_date
                ON nk_signals(run_id, date);
        """)
        conn.commit()
    finally:
        conn.close()


def save_run(db_path, run_meta, summary, book_analysis,
             daily_records, trades, signals, equity_curve):
    """
    保存一次完整的回测运行到 DB。

    Returns: run_id
    """
    db_path = db_path or _default_db_path()
    init_db(db_path)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()

        # ── 写入 run ──
        cur.execute("""
            INSERT INTO nk_runs(
                created_at, runner, host, start_date, end_date,
                leverage, risk_pct, initial_capital,
                summary_json, meta_json, book_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            run_meta.get('runner', 'local'),
            run_meta.get('host', ''),
            run_meta.get('start_date', ''),
            run_meta.get('end_date', ''),
            run_meta.get('leverage', 3),
            run_meta.get('risk_per_trade', 0.02),
            run_meta.get('initial_capital', 100000),
            json.dumps(summary, ensure_ascii=False),
            json.dumps(run_meta, ensure_ascii=False),
            json.dumps(book_analysis, ensure_ascii=False),
        ))
        run_id = cur.lastrowid

        # ── 计算每日回撤并写入 daily ──
        peak = run_meta.get('initial_capital', 100000)
        for rec in daily_records:
            tv = rec.get('total_value', peak)
            if tv > peak:
                peak = tv
            dd_pct = round((peak - tv) / peak * 100, 2) if peak > 0 else 0

            pos = rec.get('position') or {}
            cur.execute("""
                INSERT INTO nk_daily(
                    run_id, date, open, high, low, close,
                    trend, signal_desc,
                    pos_direction, pos_entry_price, pos_quantity,
                    pos_unrealized_pnl, pos_pnl_ratio,
                    pos_stop_loss, pos_tp1, pos_bars_held,
                    total_value, usdt, return_pct, drawdown_pct,
                    data_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                run_id,
                rec.get('date', ''),
                rec.get('open'),
                rec.get('high'),
                rec.get('low'),
                rec.get('close'),
                rec.get('trend', ''),
                rec.get('signal', ''),
                pos.get('direction'),
                pos.get('entry_price'),
                pos.get('quantity'),
                pos.get('unrealized_pnl'),
                pos.get('pnl_ratio'),
                pos.get('stop_loss'),
                pos.get('tp1'),
                pos.get('bars_held'),
                rec.get('total_value'),
                rec.get('usdt'),
                rec.get('return_pct'),
                dd_pct,
                json.dumps(rec, ensure_ascii=False),
            ))

        # ── 写入 trades ──
        for i, tr in enumerate(trades, 1):
            cur.execute("""
                INSERT INTO nk_trades(
                    run_id, seq, time, action, direction,
                    market_price, exec_price, quantity, notional_value,
                    margin, leverage, fee, slippage_cost, pnl,
                    after_total, close_reason, pattern, signal_strength,
                    at_key_level, with_trend, bars_held, data_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                tr.get('close_reason'),
                tr.get('pattern'),
                tr.get('signal_strength'),
                1 if tr.get('at_key_level') else 0,
                1 if tr.get('with_trend') else 0,
                tr.get('bars_held'),
                json.dumps(tr, ensure_ascii=False),
            ))

        # ── 写入 signals ──
        for sig in signals:
            sig_dict = sig if isinstance(sig, dict) else sig.to_dict()
            cur.execute("""
                INSERT INTO nk_signals(
                    run_id, date, pattern, direction, strength,
                    entry_price, stop_loss, tp1,
                    at_key_level, with_trend, trend, notes
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                run_id,
                sig_dict.get('time', ''),
                sig_dict.get('pattern', ''),
                sig_dict.get('direction', ''),
                sig_dict.get('strength'),
                sig_dict.get('entry_price'),
                sig_dict.get('stop_loss'),
                sig_dict.get('take_profit_1'),
                1 if sig_dict.get('at_key_level') else 0,
                1 if sig_dict.get('with_trend') else 0,
                sig_dict.get('trend', ''),
                sig_dict.get('notes', ''),
            ))

        conn.commit()
        return run_id
    finally:
        conn.close()


def load_latest_run(db_path=None):
    """加载最新一次回测运行的完整数据"""
    db_path = db_path or _default_db_path()
    if not os.path.exists(db_path):
        return None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()

        # 最新 run
        run = cur.execute(
            "SELECT * FROM nk_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not run:
            return None
        run_id = run['id']

        # daily records
        daily_rows = cur.execute(
            "SELECT * FROM nk_daily WHERE run_id = ? ORDER BY date ASC",
            (run_id,)
        ).fetchall()

        # trades
        trade_rows = cur.execute(
            "SELECT * FROM nk_trades WHERE run_id = ? ORDER BY seq ASC",
            (run_id,)
        ).fetchall()

        # signals
        signal_rows = cur.execute(
            "SELECT * FROM nk_signals WHERE run_id = ? ORDER BY date ASC",
            (run_id,)
        ).fetchall()

        run_meta = json.loads(run['meta_json']) if run['meta_json'] else {}
        summary = json.loads(run['summary_json']) if run['summary_json'] else {}
        book = json.loads(run['book_json']) if run['book_json'] else {}

        daily_records = []
        for r in daily_rows:
            rec = json.loads(r['data_json']) if r['data_json'] else {}
            rec['drawdown_pct'] = r['drawdown_pct']
            # 补充 OHLC 数据
            if 'high' not in rec:
                rec['high'] = r['high']
                rec['low'] = r['low']
                rec['close'] = r['close']
            daily_records.append(rec)

        trades = [json.loads(r['data_json']) for r in trade_rows]

        signals = []
        for r in signal_rows:
            signals.append({
                'date': r['date'],
                'pattern': r['pattern'],
                'direction': r['direction'],
                'strength': r['strength'],
                'entry_price': r['entry_price'],
                'stop_loss': r['stop_loss'],
                'tp1': r['tp1'],
                'at_key_level': bool(r['at_key_level']),
                'with_trend': bool(r['with_trend']),
                'trend': r['trend'],
                'notes': r['notes'],
            })

        # 构建 equity_curve
        equity_curve = []
        for r in daily_rows:
            equity_curve.append({
                'date': r['date'],
                'total': r['total_value'],
                'price': r['close'],
                'drawdown': r['drawdown_pct'],
            })

        # 信号统计
        sig_by_pattern = {}
        sig_by_dir = {}
        for s in signals:
            sig_by_pattern[s['pattern']] = sig_by_pattern.get(s['pattern'], 0) + 1
            sig_by_dir[s['direction']] = sig_by_dir.get(s['direction'], 0) + 1

        return {
            'run_id': run_id,
            'run_meta': run_meta,
            'summary': summary,
            'book_analysis': book,
            'daily_records': daily_records,
            'trade_details': trades,
            'signals': signals,
            'equity_curve': equity_curve,
            'signal_summary': {
                'total_signals': len(signals),
                'by_pattern': sig_by_pattern,
                'by_direction': sig_by_dir,
            },
        }
    finally:
        conn.close()


def list_runs(db_path=None):
    """列出所有历史回测运行"""
    db_path = db_path or _default_db_path()
    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, created_at, runner, host, start_date, end_date, summary_json FROM nk_runs ORDER BY id DESC"
        ).fetchall()
        result = []
        for r in rows:
            summary = json.loads(r['summary_json']) if r['summary_json'] else {}
            result.append({
                'run_id': r['id'],
                'created_at': r['created_at'],
                'runner': r['runner'],
                'host': r['host'],
                'start_date': r['start_date'],
                'end_date': r['end_date'],
                'total_return_pct': summary.get('total_return_pct'),
                'max_drawdown_pct': summary.get('max_drawdown_pct'),
                'total_trades': summary.get('total_trades'),
                'win_rate_pct': summary.get('win_rate_pct'),
            })
        return result
    finally:
        conn.close()
