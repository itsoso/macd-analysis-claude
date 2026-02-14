"""
多周期联合决策 — SQLite 逐日盈亏数据库存储层
===============================================
表结构:
  mtf_runs      — 每次回测运行的元数据、策略快照和汇总
  mtf_daily     — 每日持仓 + 盈亏快照
  mtf_trades    — 每笔交易完整明细

版本化设计:
  每次回测自动分配 run_id，并记录 version_tag（策略版本标签）
  和 strategy_snapshot（关键开关/参数快照），供前端对比分析。
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
                meta_json       TEXT,
                version_tag     TEXT,
                strategy_snapshot TEXT,
                experiment_notes TEXT
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

        # 为旧DB添加新列 (如果不存在)
        try:
            conn.execute("ALTER TABLE mtf_runs ADD COLUMN version_tag TEXT")
        except sqlite3.OperationalError:
            pass  # 列已存在
        try:
            conn.execute("ALTER TABLE mtf_runs ADD COLUMN strategy_snapshot TEXT")
        except sqlite3.OperationalError:
            pass  # 列已存在
        try:
            conn.execute("ALTER TABLE mtf_runs ADD COLUMN experiment_notes TEXT")
        except sqlite3.OperationalError:
            pass  # 列已存在

        conn.commit()
    finally:
        conn.close()


def _build_strategy_snapshot(config):
    """从策略config中提取关键开关和参数，生成策略快照用于对比"""
    if not config:
        return {}
    # 关键开关
    snapshot = {
        'use_trend_enhance': config.get('use_trend_enhance', False),
        'trend_floor_ratio': config.get('trend_floor_ratio'),
        'min_base_eth_ratio': config.get('min_base_eth_ratio'),
        'use_microstructure': config.get('use_microstructure', False),
        'use_dual_engine': config.get('use_dual_engine', False),
        'use_vol_target': config.get('use_vol_target', False),
        'use_live_gate': config.get('use_live_gate', False),
        'use_regime_aware': config.get('use_regime_aware', False),
        'use_protections': config.get('use_protections', False),
        # 核心参数
        'sell_threshold': config.get('sell_threshold'),
        'buy_threshold': config.get('buy_threshold'),
        'short_threshold': config.get('short_threshold'),
        'long_threshold': config.get('long_threshold'),
        'sell_pct': config.get('sell_pct'),
        'margin_use': config.get('margin_use'),
        'lev': config.get('lev'),
        'short_sl': config.get('short_sl'),
        'short_tp': config.get('short_tp'),
        'long_sl': config.get('long_sl'),
        'long_tp': config.get('long_tp'),
        'short_max_hold': config.get('short_max_hold'),
        'long_max_hold': config.get('long_max_hold'),
        'consensus_min_strength': config.get('consensus_min_strength'),
        'coverage_min': config.get('coverage_min'),
        # 保护参数
        'prot_global_dd_limit_pct': config.get('prot_global_dd_limit_pct'),
        'prot_global_halt_recovery_pct': config.get('prot_global_halt_recovery_pct'),
        # ── Codex run#62 新增：confirm / block / gate / 退出规则 ──
        'use_spot_sell_confirm': config.get('use_spot_sell_confirm', False),
        'spot_sell_confirm_ss': config.get('spot_sell_confirm_ss'),
        'spot_sell_confirm_min': config.get('spot_sell_confirm_min'),
        'spot_sell_regime_block': config.get('spot_sell_regime_block', ''),
        'use_regime_short_gate': config.get('use_regime_short_gate', False),
        'regime_short_gate_add': config.get('regime_short_gate_add'),
        'regime_short_gate_regimes': config.get('regime_short_gate_regimes', ''),
        'hard_stop_loss': config.get('hard_stop_loss'),
        'use_dual_macd': config.get('use_dual_macd', False),
        'dual_macd_bonus': config.get('dual_macd_bonus'),
        # P1a / P1b
        'no_tp_exit_bars': config.get('no_tp_exit_bars', 0),
        'no_tp_exit_min_pnl': config.get('no_tp_exit_min_pnl'),
        'neutral_mid_ss_sell_ratio': config.get('neutral_mid_ss_sell_ratio'),
        # 实验3: regime-specific short_threshold
        'regime_short_threshold': config.get('regime_short_threshold'),
    }
    return snapshot


def _auto_version_tag(config):
    """根据策略config自动生成版本标签"""
    parts = []
    if config.get('use_trend_enhance'):
        parts.append('趋势v3')
    if config.get('use_microstructure'):
        parts.append('微结构')
    if config.get('use_dual_engine'):
        parts.append('双引擎')
    if config.get('use_vol_target'):
        parts.append('波动目标')
    if config.get('use_live_gate'):
        parts.append('LiveGate')
    if config.get('use_regime_aware'):
        parts.append('Regime')
    if not parts:
        parts.append('基线')
    return ' + '.join(parts)


def save_run(db_path, run_meta, summary, daily_records, trades,
             version_tag=None, experiment_notes=None):
    """保存一次完整回测到 DB。Returns run_id.

    Args:
        experiment_notes: 实验说明文本 (Markdown)，描述本轮回测目的、变体、结论等。
    """
    db_path = db_path or _default_db_path()
    init_db(db_path)

    config = run_meta.get('config', {})
    strategy_snapshot = _build_strategy_snapshot(config)
    if not version_tag:
        version_tag = _auto_version_tag(config)

    # experiment_notes 也可从 run_meta 中取
    if not experiment_notes:
        experiment_notes = run_meta.get('experiment_notes', '')

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO mtf_runs(
                created_at, runner, host, start_date, end_date,
                primary_tf, decision_tfs, combo_name,
                leverage, initial_capital, summary_json, meta_json,
                version_tag, strategy_snapshot, experiment_notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            version_tag,
            json.dumps(strategy_snapshot, ensure_ascii=False),
            experiment_notes or '',
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


def list_runs(db_path=None):
    """列出所有回测运行，用于版本选择器"""
    db_path = db_path or _default_db_path()
    if not os.path.exists(db_path):
        return []

    init_db(db_path)  # 确保新列存在

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        rows = cur.execute("""
            SELECT id, created_at, runner, host, start_date, end_date,
                   primary_tf, decision_tfs, combo_name, leverage,
                   initial_capital, summary_json, version_tag, strategy_snapshot,
                   experiment_notes
            FROM mtf_runs ORDER BY id DESC
        """).fetchall()

        result = []
        for r in rows:
            summary = json.loads(r['summary_json']) if r['summary_json'] else {}
            snapshot = json.loads(r['strategy_snapshot']) if r['strategy_snapshot'] else {}
            notes = ''
            try:
                notes = r['experiment_notes'] or ''
            except (KeyError, IndexError):
                pass
            result.append({
                'run_id': r['id'],
                'created_at': r['created_at'],
                'host': r['host'],
                'start_date': r['start_date'],
                'end_date': r['end_date'],
                'combo_name': r['combo_name'],
                'version_tag': r['version_tag'] or f"Run #{r['id']}",
                'strategy_snapshot': snapshot,
                'experiment_notes': notes,
                'total_return_pct': summary.get('total_return_pct'),
                'alpha_pct': summary.get('alpha_pct'),
                'max_drawdown_pct': summary.get('max_drawdown_pct'),
                'win_rate_pct': summary.get('win_rate_pct'),
                'profit_factor': summary.get('profit_factor'),
                'total_trades': summary.get('total_trades'),
                'close_trades': summary.get('close_trades'),
            })
        return result
    finally:
        conn.close()


def load_run_by_id(run_id, db_path=None):
    """按 run_id 加载回测的完整数据"""
    db_path = db_path or _default_db_path()
    if not os.path.exists(db_path):
        return None

    init_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        run = cur.execute(
            "SELECT * FROM mtf_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if not run:
            return None

        return _load_run_data(cur, run)
    finally:
        conn.close()


def load_latest_run(db_path=None):
    """加载最新一次回测的完整数据"""
    db_path = db_path or _default_db_path()
    if not os.path.exists(db_path):
        return None

    init_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        run = cur.execute(
            "SELECT * FROM mtf_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not run:
            return None

        return _load_run_data(cur, run)
    finally:
        conn.close()


def _load_run_data(cur, run):
    """内部：从 run 行加载完整数据"""
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

    # 读取版本信息
    version_tag = None
    strategy_snapshot = {}
    experiment_notes = ''
    try:
        version_tag = run['version_tag']
        strategy_snapshot = json.loads(run['strategy_snapshot']) if run['strategy_snapshot'] else {}
        experiment_notes = run['experiment_notes'] or ''
    except (KeyError, IndexError):
        pass

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
        'version_tag': version_tag or f"Run #{run_id}",
        'strategy_snapshot': strategy_snapshot,
        'experiment_notes': experiment_notes,
        'daily_records': daily_records,
        'trade_details': trades,
        'equity_curve': equity_curve,
    }


def update_experiment_notes(run_id, notes, db_path=None):
    """更新指定 run 的实验说明 (用于回填历史记录)。"""
    db_path = db_path or _default_db_path()
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE mtf_runs SET experiment_notes = ? WHERE id = ?",
            (notes, run_id),
        )
        conn.commit()
    finally:
        conn.close()
