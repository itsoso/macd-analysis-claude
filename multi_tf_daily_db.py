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


def _open_conn(db_path):
    """打开 SQLite 连接，启用 WAL 模式 + 性能 pragma。"""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-8000")      # 8 MB page cache
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def init_db(db_path=None):
    db_path = db_path or _default_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = _open_conn(db_path)
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
        'trend_enhance_engine_gate': config.get('trend_enhance_engine_gate', False),
        'trend_floor_ratio': config.get('trend_floor_ratio'),
        'min_base_eth_ratio': config.get('min_base_eth_ratio'),
        'use_microstructure': config.get('use_microstructure', False),
        'use_dual_engine': config.get('use_dual_engine', False),
        'use_vol_target': config.get('use_vol_target', False),
        'use_live_gate': config.get('use_live_gate', False),
        'use_regime_aware': config.get('use_regime_aware', False),
        'use_protections': config.get('use_protections', False),
        # 核心参数
        'fusion_mode': config.get('fusion_mode'),
        'veto_threshold': config.get('veto_threshold'),
        'sell_threshold': config.get('sell_threshold'),
        'buy_threshold': config.get('buy_threshold'),
        'short_threshold': config.get('short_threshold'),
        'long_threshold': config.get('long_threshold'),
        'close_short_bs': config.get('close_short_bs'),
        'close_long_ss': config.get('close_long_ss'),
        'sell_pct': config.get('sell_pct'),
        'margin_use': config.get('margin_use'),
        'lev': config.get('lev'),
        'short_sl': config.get('short_sl'),
        'short_tp': config.get('short_tp'),
        'long_sl': config.get('long_sl'),
        'long_tp': config.get('long_tp'),
        'short_trail': config.get('short_trail'),
        'long_trail': config.get('long_trail'),
        'trail_pullback': config.get('trail_pullback'),
        'short_max_hold': config.get('short_max_hold'),
        'long_max_hold': config.get('long_max_hold'),
        'cooldown': config.get('cooldown'),
        'spot_cooldown': config.get('spot_cooldown'),
        'consensus_min_strength': config.get('consensus_min_strength'),
        'coverage_min': config.get('coverage_min'),
        # 分段止盈
        'use_partial_tp': config.get('use_partial_tp', False),
        'partial_tp_1': config.get('partial_tp_1'),
        'partial_tp_1_pct': config.get('partial_tp_1_pct'),
        'use_partial_tp_2': config.get('use_partial_tp_2', False),
        'partial_tp_2': config.get('partial_tp_2'),
        'partial_tp_2_pct': config.get('partial_tp_2_pct'),
        'use_partial_tp_v3': config.get('use_partial_tp_v3', False),
        'partial_tp_1_early': config.get('partial_tp_1_early'),
        'partial_tp_2_early': config.get('partial_tp_2_early'),
        # ATR 止损
        'use_atr_sl': config.get('use_atr_sl', False),
        'atr_sl_mult': config.get('atr_sl_mult'),
        'atr_sl_floor': config.get('atr_sl_floor'),
        'atr_sl_ceil': config.get('atr_sl_ceil'),
        # 保护参数
        'prot_global_dd_limit_pct': config.get('prot_global_dd_limit_pct'),
        'prot_global_halt_recovery_pct': config.get('prot_global_halt_recovery_pct'),
        # ── Codex run#62 新增：confirm / block / gate / 退出规则 ──
        'use_spot_sell_confirm': config.get('use_spot_sell_confirm', False),
        'spot_sell_confirm_ss': config.get('spot_sell_confirm_ss'),
        'spot_sell_confirm_min': config.get('spot_sell_confirm_min'),
        'use_neutral_spot_sell_layer': config.get('use_neutral_spot_sell_layer', False),
        'neutral_spot_sell_confirm_thr': config.get('neutral_spot_sell_confirm_thr'),
        'neutral_spot_sell_min_confirms_any': config.get('neutral_spot_sell_min_confirms_any'),
        'neutral_spot_sell_strong_confirms': config.get('neutral_spot_sell_strong_confirms'),
        'neutral_spot_sell_full_ss_min': config.get('neutral_spot_sell_full_ss_min'),
        'neutral_spot_sell_weak_ss_min': config.get('neutral_spot_sell_weak_ss_min'),
        'neutral_spot_sell_weak_pct_cap': config.get('neutral_spot_sell_weak_pct_cap'),
        'neutral_spot_sell_block_ss_min': config.get('neutral_spot_sell_block_ss_min'),
        'spot_sell_regime_block': config.get('spot_sell_regime_block', ''),
        'use_stagnation_reentry': config.get('use_stagnation_reentry', False),
        'stagnation_reentry_days': config.get('stagnation_reentry_days'),
        'stagnation_reentry_regimes': config.get('stagnation_reentry_regimes'),
        'stagnation_reentry_min_spot_ratio': config.get('stagnation_reentry_min_spot_ratio'),
        'stagnation_reentry_buy_pct': config.get('stagnation_reentry_buy_pct'),
        'stagnation_reentry_min_usdt': config.get('stagnation_reentry_min_usdt'),
        'stagnation_reentry_cooldown_days': config.get('stagnation_reentry_cooldown_days'),
        'use_regime_short_gate': config.get('use_regime_short_gate', False),
        'regime_short_gate_add': config.get('regime_short_gate_add'),
        'regime_short_gate_regimes': config.get('regime_short_gate_regimes', ''),
        'hard_stop_loss': config.get('hard_stop_loss'),
        'use_dual_macd': config.get('use_dual_macd', False),
        'dual_macd_bonus': config.get('dual_macd_bonus'),
        # P1a / P1b
        'no_tp_exit_bars': config.get('no_tp_exit_bars', 0),
        'no_tp_exit_min_pnl': config.get('no_tp_exit_min_pnl'),
        'no_tp_exit_regimes': config.get('no_tp_exit_regimes'),
        'no_tp_exit_short_bars': config.get('no_tp_exit_short_bars'),
        'no_tp_exit_short_min_pnl': config.get('no_tp_exit_short_min_pnl'),
        'no_tp_exit_short_loss_floor': config.get('no_tp_exit_short_loss_floor'),
        'no_tp_exit_short_regimes': config.get('no_tp_exit_short_regimes'),
        'no_tp_exit_long_bars': config.get('no_tp_exit_long_bars'),
        'no_tp_exit_long_min_pnl': config.get('no_tp_exit_long_min_pnl'),
        'no_tp_exit_long_loss_floor': config.get('no_tp_exit_long_loss_floor'),
        'no_tp_exit_long_regimes': config.get('no_tp_exit_long_regimes'),
        'reverse_min_hold_short': config.get('reverse_min_hold_short', 0),
        'reverse_min_hold_long': config.get('reverse_min_hold_long', 0),
        'neutral_mid_ss_sell_ratio': config.get('neutral_mid_ss_sell_ratio'),
        # 实验3: regime-specific short_threshold
        'regime_short_threshold': config.get('regime_short_threshold'),
        # neutral 结构化信号质量门控
        'use_neutral_quality_gate': config.get('use_neutral_quality_gate', False),
        'neutral_min_score_gap': config.get('neutral_min_score_gap'),
        'neutral_min_strength': config.get('neutral_min_strength'),
        'neutral_min_streak': config.get('neutral_min_streak'),
        'neutral_nochain_extra_gap': config.get('neutral_nochain_extra_gap'),
        'neutral_large_conflict_ratio': config.get('neutral_large_conflict_ratio'),
        # neutral 六书共识门控
        'use_neutral_book_consensus': config.get('use_neutral_book_consensus', False),
        'neutral_book_sell_threshold': config.get('neutral_book_sell_threshold'),
        'neutral_book_buy_threshold': config.get('neutral_book_buy_threshold'),
        'neutral_book_min_confirms': config.get('neutral_book_min_confirms'),
        'neutral_book_max_conflicts': config.get('neutral_book_max_conflicts'),
        'neutral_book_cs_kdj_threshold_adj': config.get('neutral_book_cs_kdj_threshold_adj'),
        # neutral 结构质量渐进折扣
        'use_neutral_structural_discount': config.get('use_neutral_structural_discount', False),
        'neutral_struct_activity_thr': config.get('neutral_struct_activity_thr'),
        'neutral_struct_discount_0': config.get('neutral_struct_discount_0'),
        'neutral_struct_discount_1': config.get('neutral_struct_discount_1'),
        'neutral_struct_discount_2': config.get('neutral_struct_discount_2'),
        'neutral_struct_discount_3': config.get('neutral_struct_discount_3'),
        'neutral_struct_discount_4plus': config.get('neutral_struct_discount_4plus'),
        'structural_discount_short_regimes': config.get('structural_discount_short_regimes'),
        'structural_discount_long_regimes': config.get('structural_discount_long_regimes'),
        # 信号置信度学习层
        'use_confidence_learning': config.get('use_confidence_learning', False),
        'confidence_min_raw': config.get('confidence_min_raw'),
        'confidence_min_posterior': config.get('confidence_min_posterior'),
        'confidence_min_samples': config.get('confidence_min_samples'),
        'confidence_block_after_samples': config.get('confidence_block_after_samples'),
        'confidence_threshold_gain': config.get('confidence_threshold_gain'),
        'confidence_threshold_min_mult': config.get('confidence_threshold_min_mult'),
        'confidence_threshold_max_mult': config.get('confidence_threshold_max_mult'),
        'confidence_prior_alpha': config.get('confidence_prior_alpha'),
        'confidence_prior_beta': config.get('confidence_prior_beta'),
        'confidence_win_pnl_r': config.get('confidence_win_pnl_r'),
        'confidence_loss_pnl_r': config.get('confidence_loss_pnl_r'),
        'print_signal_features': config.get('print_signal_features', True),
        'signal_replay_top_n': config.get('signal_replay_top_n', 10),
        # neutral short 结构确认器
        'use_neutral_short_structure_gate': config.get('use_neutral_short_structure_gate', False),
        'neutral_short_structure_large_tfs': config.get('neutral_short_structure_large_tfs'),
        'neutral_short_structure_need_min_tfs': config.get('neutral_short_structure_need_min_tfs'),
        'neutral_short_structure_min_agree': config.get('neutral_short_structure_min_agree'),
        'neutral_short_structure_div_gap': config.get('neutral_short_structure_div_gap'),
        'neutral_short_structure_ma_gap': config.get('neutral_short_structure_ma_gap'),
        'neutral_short_structure_vp_gap': config.get('neutral_short_structure_vp_gap'),
        'neutral_short_structure_fail_open': config.get('neutral_short_structure_fail_open'),
        'neutral_short_structure_soften_weak': config.get('neutral_short_structure_soften_weak'),
        'neutral_short_structure_soften_mult': config.get('neutral_short_structure_soften_mult'),
        # 空单冲突软折扣
        'use_short_conflict_soft_discount': config.get('use_short_conflict_soft_discount', False),
        'short_conflict_regimes': config.get('short_conflict_regimes'),
        'short_conflict_div_buy_min': config.get('short_conflict_div_buy_min'),
        'short_conflict_ma_sell_min': config.get('short_conflict_ma_sell_min'),
        'short_conflict_discount_mult': config.get('short_conflict_discount_mult'),
        'use_long_conflict_soft_discount': config.get('use_long_conflict_soft_discount', False),
        'long_conflict_regimes': config.get('long_conflict_regimes'),
        'long_conflict_div_sell_min': config.get('long_conflict_div_sell_min'),
        'long_conflict_ma_buy_min': config.get('long_conflict_ma_buy_min'),
        'long_conflict_discount_mult': config.get('long_conflict_discount_mult'),
        # long 高置信错单候选门控（A/B）
        'use_long_high_conf_gate_a': config.get('use_long_high_conf_gate_a', False),
        'long_high_conf_gate_a_conf_min': config.get('long_high_conf_gate_a_conf_min'),
        'long_high_conf_gate_a_regime': config.get('long_high_conf_gate_a_regime'),
        'use_long_high_conf_gate_b': config.get('use_long_high_conf_gate_b', False),
        'long_high_conf_gate_b_conf_min': config.get('long_high_conf_gate_b_conf_min'),
        'long_high_conf_gate_b_regime': config.get('long_high_conf_gate_b_regime'),
        'long_high_conf_gate_b_vp_buy_min': config.get('long_high_conf_gate_b_vp_buy_min'),
        # 空单逆势防守退出（结构化风控）
        'use_short_adverse_exit': config.get('use_short_adverse_exit', False),
        'short_adverse_min_bars': config.get('short_adverse_min_bars'),
        'short_adverse_loss_r': config.get('short_adverse_loss_r'),
        'short_adverse_bs': config.get('short_adverse_bs'),
        'short_adverse_bs_dom_ratio': config.get('short_adverse_bs_dom_ratio'),
        'short_adverse_ss_cap': config.get('short_adverse_ss_cap'),
        'short_adverse_require_bs_dom': config.get('short_adverse_require_bs_dom'),
        'short_adverse_ma_conflict_gap': config.get('short_adverse_ma_conflict_gap'),
        'short_adverse_conflict_thr': config.get('short_adverse_conflict_thr'),
        'short_adverse_min_conflicts': config.get('short_adverse_min_conflicts'),
        'short_adverse_need_cs_kdj': config.get('short_adverse_need_cs_kdj'),
        'short_adverse_large_bs_min': config.get('short_adverse_large_bs_min'),
        'short_adverse_large_ratio': config.get('short_adverse_large_ratio'),
        'short_adverse_need_chain_long': config.get('short_adverse_need_chain_long'),
        'short_adverse_regimes': config.get('short_adverse_regimes'),
        # 极端 divergence 做空否决
        'use_extreme_divergence_short_veto': config.get('use_extreme_divergence_short_veto', False),
        'extreme_div_short_threshold': config.get('extreme_div_short_threshold'),
        'extreme_div_short_confirm_thr': config.get('extreme_div_short_confirm_thr'),
        'extreme_div_short_min_confirms': config.get('extreme_div_short_min_confirms'),
        'extreme_div_short_regimes': config.get('extreme_div_short_regimes'),
        # 引擎模式倍率
        'trend_engine_entry_mult': config.get('trend_engine_entry_mult'),
        'trend_engine_hold_mult': config.get('trend_engine_hold_mult'),
        'reversion_engine_entry_mult': config.get('reversion_engine_entry_mult'),
        'reversion_engine_hold_mult': config.get('reversion_engine_hold_mult'),
        # 波动目标
        'vol_target_annual': config.get('vol_target_annual'),
        'vol_target_lookback_bars': config.get('vol_target_lookback_bars'),
        # S2: 保本止损
        'use_breakeven_after_tp1': config.get('use_breakeven_after_tp1', False),
        'breakeven_buffer': config.get('breakeven_buffer', 0.01),
        # S3: 棘轮追踪止损
        'use_ratchet_trail': config.get('use_ratchet_trail', False),
        'ratchet_trail_tiers': config.get('ratchet_trail_tiers', ''),
        # S5: 信号质量止损
        'use_ss_quality_sl': config.get('use_ss_quality_sl', False),
        'ss_quality_sl_threshold': config.get('ss_quality_sl_threshold', 50),
        'ss_quality_sl_mult': config.get('ss_quality_sl_mult', 0.70),
        # P9/P18/P24: Regime 自适应
        'use_regime_adaptive_reweight': config.get('use_regime_adaptive_reweight', False),
        'regime_neutral_ss_dampen': config.get('regime_neutral_ss_dampen'),
        'regime_neutral_bs_boost': config.get('regime_neutral_bs_boost'),
        'use_regime_adaptive_fusion': config.get('use_regime_adaptive_fusion', False),
        'regime_trend_div_w': config.get('regime_trend_div_w'),
        'regime_trend_ma_w': config.get('regime_trend_ma_w'),
        'regime_low_vol_trend_div_w': config.get('regime_low_vol_trend_div_w'),
        'regime_low_vol_trend_ma_w': config.get('regime_low_vol_trend_ma_w'),
        'regime_neutral_div_w': config.get('regime_neutral_div_w'),
        'regime_neutral_ma_w': config.get('regime_neutral_ma_w'),
        'regime_high_vol_div_w': config.get('regime_high_vol_div_w'),
        'regime_high_vol_ma_w': config.get('regime_high_vol_ma_w'),
        'regime_high_vol_choppy_div_w': config.get('regime_high_vol_choppy_div_w'),
        'regime_high_vol_choppy_ma_w': config.get('regime_high_vol_choppy_ma_w'),
        'use_regime_adaptive_sl': config.get('use_regime_adaptive_sl', False),
        'regime_neutral_short_sl': config.get('regime_neutral_short_sl'),
        'regime_trend_short_sl': config.get('regime_trend_short_sl'),
        'regime_low_vol_trend_short_sl': config.get('regime_low_vol_trend_short_sl'),
        'regime_high_vol_short_sl': config.get('regime_high_vol_short_sl'),
        'regime_high_vol_choppy_short_sl': config.get('regime_high_vol_choppy_short_sl'),
        # V9: Mark/Funding 真实化口径
        'use_mark_price_for_liquidation': config.get('use_mark_price_for_liquidation', False),
        'mark_price_col': config.get('mark_price_col'),
        'mark_high_col': config.get('mark_high_col'),
        'mark_low_col': config.get('mark_low_col'),
        'use_real_funding_rate': config.get('use_real_funding_rate', False),
        'funding_rate_col': config.get('funding_rate_col'),
        'funding_interval_hours': config.get('funding_interval_hours'),
        'funding_interval_hours_col': config.get('funding_interval_hours_col'),
        # V9: Leg 风险预算
        'use_leg_risk_budget': config.get('use_leg_risk_budget', False),
        'risk_budget_neutral_long': config.get('risk_budget_neutral_long'),
        'risk_budget_neutral_short': config.get('risk_budget_neutral_short'),
        'risk_budget_trend_long': config.get('risk_budget_trend_long'),
        'risk_budget_trend_short': config.get('risk_budget_trend_short'),
        'risk_budget_low_vol_trend_long': config.get('risk_budget_low_vol_trend_long'),
        'risk_budget_low_vol_trend_short': config.get('risk_budget_low_vol_trend_short'),
        'risk_budget_high_vol_long': config.get('risk_budget_high_vol_long'),
        'risk_budget_high_vol_short': config.get('risk_budget_high_vol_short'),
        'risk_budget_high_vol_choppy_long': config.get('risk_budget_high_vol_choppy_long'),
        'risk_budget_high_vol_choppy_short': config.get('risk_budget_high_vol_choppy_short'),
        # 止损后冷却倍数
        'short_sl_cd_mult': config.get('short_sl_cd_mult'),
        'long_sl_cd_mult': config.get('long_sl_cd_mult'),
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

    conn = _open_conn(db_path)
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
            run_meta.get('initial_capital', 100000),
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


def list_runs(db_path=None, limit=0, offset=0):
    """列出回测运行，用于版本选择器。

    Args:
        limit: 返回条数上限，0 表示全部。
        offset: 跳过前 N 条。
    """
    db_path = db_path or _default_db_path()
    if not os.path.exists(db_path):
        return []

    init_db(db_path)  # 确保新列存在

    conn = _open_conn(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()

        # 列表只取轻量列，不取 strategy_snapshot / experiment_notes / meta_json
        sql = """
            SELECT id, created_at, host, start_date, end_date,
                   combo_name, summary_json, version_tag
            FROM mtf_runs ORDER BY id DESC
        """
        params = []
        if limit > 0:
            sql += " LIMIT ? OFFSET ?"
            params = [limit, offset]
        rows = cur.execute(sql, params).fetchall()

        result = []
        for r in rows:
            summary = json.loads(r['summary_json']) if r['summary_json'] else {}
            result.append({
                'run_id': r['id'],
                'created_at': r['created_at'],
                'host': r['host'],
                'start_date': r['start_date'],
                'end_date': r['end_date'],
                'combo_name': r['combo_name'],
                'version_tag': r['version_tag'] or f"Run #{r['id']}",
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


def load_run_by_id(run_id, db_path=None, include_trades=True):
    """按 run_id 加载回测数据。include_trades=False 返回轻量首屏数据。"""
    db_path = db_path or _default_db_path()
    if not os.path.exists(db_path):
        return None

    init_db(db_path)

    conn = _open_conn(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        run = cur.execute(
            "SELECT * FROM mtf_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if not run:
            return None

        return _load_run_data(cur, run, include_trades=include_trades)
    finally:
        conn.close()


def load_latest_run(db_path=None, include_trades=True):
    """加载最新一次回测数据。include_trades=False 返回轻量首屏数据。"""
    db_path = db_path or _default_db_path()
    if not os.path.exists(db_path):
        return None

    init_db(db_path)

    conn = _open_conn(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        run = cur.execute(
            "SELECT * FROM mtf_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not run:
            return None

        return _load_run_data(cur, run, include_trades=include_trades)
    finally:
        conn.close()


def _load_run_data(cur, run, include_trades=True):
    """内部：从 run 行加载完整数据（性能优化版）。

    Args:
        include_trades: False 时不返回完整 trade_details，
            改为轻量的 trade_summary + trade_marks，首屏体积减少 ~50%。
    """
    run_id = run['id']

    # ── daily: 只取必要列 ──
    daily_rows = cur.execute("""
        SELECT date, eth_price, total_value, usdt, frozen_margin,
               long_pnl, short_pnl, spot_eth_value,
               return_pct, drawdown_pct,
               has_long, has_short, long_entry, long_qty,
               short_entry, short_qty, day_trades, day_pnl
        FROM mtf_daily WHERE run_id = ? ORDER BY date ASC
    """, (run_id,)).fetchall()

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

    # 构建 daily_records（从结构化列，不 parse JSON）
    daily_records = []
    for r in daily_rows:
        daily_records.append({
            'date': r['date'],
            'eth_price': r['eth_price'],
            'total_value': r['total_value'],
            'usdt': r['usdt'],
            'frozen_margin': r['frozen_margin'],
            'long_pnl': r['long_pnl'],
            'short_pnl': r['short_pnl'],
            'spot_eth_value': r['spot_eth_value'],
            'return_pct': r['return_pct'],
            'drawdown_pct': r['drawdown_pct'],
            'has_long': bool(r['has_long']),
            'has_short': bool(r['has_short']),
            'long_entry': r['long_entry'],
            'long_qty': r['long_qty'],
            'short_entry': r['short_entry'],
            'short_qty': r['short_qty'],
            'day_trades': r['day_trades'],
            'day_pnl': r['day_pnl'],
        })

    result = {
        'run_id': run_id,
        'run_meta': run_meta,
        'summary': summary,
        'version_tag': version_tag or f"Run #{run_id}",
        'strategy_snapshot': strategy_snapshot,
        'experiment_notes': experiment_notes,
        'daily_records': daily_records,
    }

    # ── trades 处理 ──
    if include_trades:
        # 完整模式：返回 trade_details + equity_curve（兼容旧调用，如导出 Excel）
        trade_rows = cur.execute("""
            SELECT time, action, direction, market_price, exec_price,
                   quantity, notional_value, margin, leverage, fee,
                   slippage_cost, pnl, after_total, reason
            FROM mtf_trades WHERE run_id = ? ORDER BY seq ASC
        """, (run_id,)).fetchall()
        trades = []
        for r in trade_rows:
            trades.append({
                'time': r['time'], 'action': r['action'],
                'direction': r['direction'],
                'market_price': r['market_price'],
                'exec_price': r['exec_price'],
                'quantity': r['quantity'],
                'notional_value': r['notional_value'],
                'margin': r['margin'], 'leverage': r['leverage'],
                'fee': r['fee'], 'slippage_cost': r['slippage_cost'],
                'pnl': r['pnl'], 'after_total': r['after_total'],
                'reason': r['reason'],
            })
        result['trade_details'] = trades
        result['equity_curve'] = [
            {'date': r['date'], 'total': r['total_value'],
             'price': r['eth_price'], 'drawdown': r['drawdown_pct']}
            for r in daily_rows
        ]
    else:
        # 轻量模式：只返回 trade_summary + trade_marks（首屏用）
        trade_rows = cur.execute("""
            SELECT time, action, pnl
            FROM mtf_trades WHERE run_id = ? ORDER BY seq ASC
        """, (run_id,)).fetchall()

        # trade_summary: 按 action 汇总数量和 PnL
        from collections import defaultdict
        counts = defaultdict(int)
        pnls = defaultdict(float)
        for r in trade_rows:
            a = r['action']
            counts[a] += 1
            if r['pnl']:
                pnls[a] += r['pnl']
        result['trade_summary'] = {
            'counts': dict(counts),
            'pnls': dict(pnls),
            'total': len(trade_rows),
        }

        # trade_marks: 轻量数组，仅供逐日时间线显示
        # 格式: [date_str, action, pnl] 极致精简
        result['trade_marks'] = [
            [(r['time'] or '')[:10], r['action'], r['pnl']]
            for r in trade_rows
        ]

    return result


def load_trades_by_run(run_id, db_path=None):
    """单独加载某 run 的完整交易记录（延迟加载用）。"""
    db_path = db_path or _default_db_path()
    if not os.path.exists(db_path):
        return []
    conn = _open_conn(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT time, action, direction, market_price, exec_price,
                   quantity, notional_value, margin, leverage, fee,
                   slippage_cost, pnl, after_total, reason
            FROM mtf_trades WHERE run_id = ? ORDER BY seq ASC
        """, (run_id,)).fetchall()
        return [{
            'time': r['time'], 'action': r['action'],
            'direction': r['direction'],
            'market_price': r['market_price'],
            'exec_price': r['exec_price'],
            'quantity': r['quantity'],
            'notional_value': r['notional_value'],
            'margin': r['margin'], 'leverage': r['leverage'],
            'fee': r['fee'], 'slippage_cost': r['slippage_cost'],
            'pnl': r['pnl'], 'after_total': r['after_total'],
            'reason': r['reason'],
        } for r in rows]
    finally:
        conn.close()


def update_experiment_notes(run_id, notes, db_path=None):
    """更新指定 run 的实验说明 (用于回填历史记录)。"""
    db_path = db_path or _default_db_path()
    init_db(db_path)
    conn = _open_conn(db_path)
    try:
        conn.execute(
            "UPDATE mtf_runs SET experiment_notes = ? WHERE id = ?",
            (notes, run_id),
        )
        conn.commit()
    finally:
        conn.close()
