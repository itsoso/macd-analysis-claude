#!/usr/bin/env python3
"""
多周期联合决策 — 逐日盈亏回测
===================================
将每日持仓快照与完整交易明细写入 SQLite DB，
供专属 Web 页面展示。

用法:
    python backtest_multi_tf_daily.py                           # 默认区间
    python backtest_multi_tf_daily.py --start 2025-06-01 --end 2025-12-31
    python backtest_multi_tf_daily.py --start 2025-01-01 --end 2026-01-31 --tag "趋势v3基线"
"""

import argparse
import json
import os
import platform
import socket
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import add_all_indicators
from ma_indicators import add_moving_averages
from signal_core import compute_signals_six, compute_signals_six_multiprocess
from live_config import StrategyConfig, get_strategy_version
from kline_store import load_klines
from optimize_six_book import (
    _build_tf_score_index,
    run_strategy_multi_tf,
)
from multi_tf_daily_db import save_run, _default_db_path

# ──────────────────────────────────────────────────────────
# 参数 (可通过 CLI 覆盖)
# ──────────────────────────────────────────────────────────
DEFAULT_TRADE_START = '2025-01-01'
DEFAULT_TRADE_END = '2026-01-31'
_LIVE_DEFAULT = StrategyConfig()
PRIMARY_TF = _LIVE_DEFAULT.timeframe

DECISION_TFS = list(_LIVE_DEFAULT.decision_timeframes)
FALLBACK_DECISION_TFS = list(_LIVE_DEFAULT.decision_timeframes_fallback)
AVAILABLE_TFS = list(dict.fromkeys([PRIMARY_TF, *DECISION_TFS, *FALLBACK_DECISION_TFS]))
COMBO_NAME = f"四TF联合({'+'.join(DECISION_TFS)})"

# 默认策略参数（与最优配置对齐）
def _build_default_config():
    cfg = {
        'name': f'多TF逐日_{COMBO_NAME}@{PRIMARY_TF}',
        # ── 初始资金 ──
        'initial_usdt': 100000,       # 10万USDT起步
        'initial_eth_value': 0,       # 0 ETH (纯U起步, 方便核对)
        'single_pct': _LIVE_DEFAULT.single_pct,
        'total_pct': _LIVE_DEFAULT.total_pct,
        'lifetime_pct': 5.0,
        'sell_threshold': _LIVE_DEFAULT.sell_threshold,
        'buy_threshold': _LIVE_DEFAULT.buy_threshold,
        'short_threshold': _LIVE_DEFAULT.short_threshold,
        'long_threshold': _LIVE_DEFAULT.long_threshold,
        'close_short_bs': _LIVE_DEFAULT.close_short_bs,
        'close_long_ss': _LIVE_DEFAULT.close_long_ss,
        'sell_pct': _LIVE_DEFAULT.sell_pct,
        'margin_use': _LIVE_DEFAULT.margin_use,
        'lev': _LIVE_DEFAULT.leverage,
        'max_lev': _LIVE_DEFAULT.max_lev,
        'short_sl': _LIVE_DEFAULT.short_sl,
        'short_tp': _LIVE_DEFAULT.short_tp,
        'short_trail': _LIVE_DEFAULT.short_trail,
        'short_max_hold': _LIVE_DEFAULT.short_max_hold,
        'long_sl': _LIVE_DEFAULT.long_sl,
        'long_tp': _LIVE_DEFAULT.long_tp,
        'long_trail': _LIVE_DEFAULT.long_trail,
        'long_max_hold': _LIVE_DEFAULT.long_max_hold,
        'trail_pullback': _LIVE_DEFAULT.trail_pullback,
        'cooldown': _LIVE_DEFAULT.cooldown,
        'spot_cooldown': _LIVE_DEFAULT.spot_cooldown,
        'use_partial_tp': _LIVE_DEFAULT.use_partial_tp,
        'partial_tp_1': _LIVE_DEFAULT.partial_tp_1,
        'partial_tp_1_pct': _LIVE_DEFAULT.partial_tp_1_pct,
        'use_partial_tp_2': _LIVE_DEFAULT.use_partial_tp_2,
        'partial_tp_2': _LIVE_DEFAULT.partial_tp_2,
        'partial_tp_2_pct': _LIVE_DEFAULT.partial_tp_2_pct,
        'use_atr_sl': _LIVE_DEFAULT.use_atr_sl,
        'atr_sl_mult': _LIVE_DEFAULT.atr_sl_mult,
        'atr_sl_floor': _LIVE_DEFAULT.atr_sl_floor,
        'atr_sl_ceil': _LIVE_DEFAULT.atr_sl_ceil,
        # [已移除] use_short_suppress
        # 硬断路器
        'hard_stop_loss': _LIVE_DEFAULT.hard_stop_loss,
        # Regime-aware 做空门控
        'use_regime_short_gate': _LIVE_DEFAULT.use_regime_short_gate,
        'regime_short_gate_add': _LIVE_DEFAULT.regime_short_gate_add,
        'regime_short_gate_regimes': _LIVE_DEFAULT.regime_short_gate_regimes,
        # SPOT_SELL 高分确认过滤
        'use_spot_sell_confirm': _LIVE_DEFAULT.use_spot_sell_confirm,
        'spot_sell_confirm_ss': _LIVE_DEFAULT.spot_sell_confirm_ss,
        'spot_sell_confirm_min': _LIVE_DEFAULT.spot_sell_confirm_min,
        # SPOT_SELL 尾部风控
        'use_spot_sell_cap': _LIVE_DEFAULT.use_spot_sell_cap,
        'spot_sell_max_pct': _LIVE_DEFAULT.spot_sell_max_pct,
        'spot_sell_regime_block': _LIVE_DEFAULT.spot_sell_regime_block,
        # ── 实验参数（默认关闭/中性） ──
        # NoTP 提前退出（长短独立 + regime 白名单）
        'no_tp_exit_bars': _LIVE_DEFAULT.no_tp_exit_bars,  # 旧参数, 兼容
        'no_tp_exit_min_pnl': _LIVE_DEFAULT.no_tp_exit_min_pnl,  # 旧参数, 兼容
        'no_tp_exit_regimes': _LIVE_DEFAULT.no_tp_exit_regimes,  # 旧参数, 兼容
        'no_tp_exit_short_bars': _LIVE_DEFAULT.no_tp_exit_short_bars,
        'no_tp_exit_short_min_pnl': _LIVE_DEFAULT.no_tp_exit_short_min_pnl,
        'no_tp_exit_short_loss_floor': _LIVE_DEFAULT.no_tp_exit_short_loss_floor,
        'no_tp_exit_short_regimes': _LIVE_DEFAULT.no_tp_exit_short_regimes,
        'no_tp_exit_long_bars': _LIVE_DEFAULT.no_tp_exit_long_bars,
        'no_tp_exit_long_min_pnl': _LIVE_DEFAULT.no_tp_exit_long_min_pnl,
        'no_tp_exit_long_loss_floor': _LIVE_DEFAULT.no_tp_exit_long_loss_floor,
        'no_tp_exit_long_regimes': _LIVE_DEFAULT.no_tp_exit_long_regimes,
        # 反向平仓最小持仓 bars（防抖）
        'reverse_min_hold_short': _LIVE_DEFAULT.reverse_min_hold_short,
        'reverse_min_hold_long': _LIVE_DEFAULT.reverse_min_hold_long,
        # neutral 中分段 SS 卖出降仓
        'neutral_mid_ss_sell_ratio': 1.0,  # 1.0=不调整
        'neutral_mid_ss_lo': 50.0,
        'neutral_mid_ss_hi': 70.0,
        # regime-specific short_threshold → 移至下方 S1 区段, 使用 _LIVE_DEFAULT
        # v3 分段止盈
        'use_partial_tp_v3': _LIVE_DEFAULT.use_partial_tp_v3,
        'partial_tp_1_early': _LIVE_DEFAULT.partial_tp_1_early,
        'partial_tp_2_early': _LIVE_DEFAULT.partial_tp_2_early,
        'fusion_mode': _LIVE_DEFAULT.fusion_mode,
        'veto_threshold': _LIVE_DEFAULT.veto_threshold,
        'kdj_bonus': _LIVE_DEFAULT.kdj_bonus,
        'kdj_weight': _LIVE_DEFAULT.kdj_weight,
        'div_weight': _LIVE_DEFAULT.div_weight,
        'kdj_strong_mult': _LIVE_DEFAULT.kdj_strong_mult,
        'kdj_normal_mult': _LIVE_DEFAULT.kdj_normal_mult,
        'kdj_reverse_mult': _LIVE_DEFAULT.kdj_reverse_mult,
        'kdj_gate_threshold': _LIVE_DEFAULT.kdj_gate_threshold,
        'veto_dampen': _LIVE_DEFAULT.veto_dampen,
        'bb_bonus': _LIVE_DEFAULT.bb_bonus,
        'vp_bonus': _LIVE_DEFAULT.vp_bonus,
        'cs_bonus': _LIVE_DEFAULT.cs_bonus,
        # ── 实盘口径对齐（与 live 引擎一致） ──
        'use_live_gate': True,
        'consensus_min_strength': _LIVE_DEFAULT.consensus_min_strength,
        'coverage_min': _LIVE_DEFAULT.coverage_min,
        'use_regime_aware': True,
        'regime_vol_high': _LIVE_DEFAULT.regime_vol_high,
        'regime_vol_low': _LIVE_DEFAULT.regime_vol_low,
        'regime_trend_strong': _LIVE_DEFAULT.regime_trend_strong,
        'regime_trend_weak': _LIVE_DEFAULT.regime_trend_weak,
        'regime_atr_high': _LIVE_DEFAULT.regime_atr_high,
        'regime_lookback_bars': _LIVE_DEFAULT.regime_lookback_bars,
        'regime_atr_bars': _LIVE_DEFAULT.regime_atr_bars,
        'use_protections': True,
        'prot_loss_streak_limit': 3,
        'prot_loss_streak_cooldown_bars': 24,
        'prot_daily_loss_limit_pct': 0.03,
        'prot_global_dd_limit_pct': 0.15,  # 15%回撤触发停机(放宽, 原0.12)
        'prot_close_on_global_halt': True,
        # ── 趋势持仓保护 ──
        'use_trend_enhance': _LIVE_DEFAULT.use_trend_enhance,
        'trend_floor_ratio': _LIVE_DEFAULT.trend_floor_ratio,
        'min_base_eth_ratio': _LIVE_DEFAULT.min_base_eth_ratio,
        'trend_enhance_engine_gate': _LIVE_DEFAULT.trend_enhance_engine_gate,
        # ── global_halt 恢复机制 ──
        'prot_global_halt_recovery_pct': 0.06,  # 回撤收窄到6%时恢复交易
        # ── 微结构增强 ── (回测中关闭, 隔离趋势保护v3效果)
        'use_microstructure': False,  # _LIVE_DEFAULT.use_microstructure,
        'micro_lookback_bars': _LIVE_DEFAULT.micro_lookback_bars,
        'micro_imbalance_threshold': _LIVE_DEFAULT.micro_imbalance_threshold,
        'micro_oi_trend_z': _LIVE_DEFAULT.micro_oi_trend_z,
        'micro_basis_extreme_z': _LIVE_DEFAULT.micro_basis_extreme_z,
        'micro_basis_crowded_z': _LIVE_DEFAULT.micro_basis_crowded_z,
        'micro_funding_extreme': _LIVE_DEFAULT.micro_funding_extreme,
        'micro_participation_trend': _LIVE_DEFAULT.micro_participation_trend,
        'micro_funding_proxy_mult': _LIVE_DEFAULT.micro_funding_proxy_mult,
        'micro_score_boost': _LIVE_DEFAULT.micro_score_boost,
        'micro_score_dampen': _LIVE_DEFAULT.micro_score_dampen,
        'micro_margin_mult_step': _LIVE_DEFAULT.micro_margin_mult_step,
        'micro_mode_override': _LIVE_DEFAULT.micro_mode_override,
        # ── 双引擎 ── (回测中关闭, 隔离趋势保护v3效果)
        'use_dual_engine': False,  # _LIVE_DEFAULT.use_dual_engine,
        'entry_dominance_ratio': _LIVE_DEFAULT.entry_dominance_ratio,
        'trend_engine_entry_mult': _LIVE_DEFAULT.trend_engine_entry_mult,
        'trend_engine_exit_mult': _LIVE_DEFAULT.trend_engine_exit_mult,
        'trend_engine_hold_mult': _LIVE_DEFAULT.trend_engine_hold_mult,
        'trend_engine_risk_mult': _LIVE_DEFAULT.trend_engine_risk_mult,
        'trend_engine_dominance_ratio': _LIVE_DEFAULT.trend_engine_dominance_ratio,
        'reversion_engine_entry_mult': _LIVE_DEFAULT.reversion_engine_entry_mult,
        'reversion_engine_exit_mult': _LIVE_DEFAULT.reversion_engine_exit_mult,
        'reversion_engine_hold_mult': _LIVE_DEFAULT.reversion_engine_hold_mult,
        'reversion_engine_risk_mult': _LIVE_DEFAULT.reversion_engine_risk_mult,
        'reversion_engine_dominance_ratio': _LIVE_DEFAULT.reversion_engine_dominance_ratio,
        # ── 波动目标仓位 ── (回测中关闭, 隔离趋势保护v3效果)
        'use_vol_target': False,  # _LIVE_DEFAULT.use_vol_target,
        'vol_target_annual': _LIVE_DEFAULT.vol_target_annual,
        'vol_target_lookback_bars': _LIVE_DEFAULT.vol_target_lookback_bars,
        'vol_target_min_scale': _LIVE_DEFAULT.vol_target_min_scale,
        'vol_target_max_scale': _LIVE_DEFAULT.vol_target_max_scale,
        # ── S1: Neutral regime做空门槛覆盖 ──
        'regime_short_threshold': _LIVE_DEFAULT.regime_short_threshold,
        # ── S1.5: neutral 信号质量门控 ──
        'use_neutral_quality_gate': _LIVE_DEFAULT.use_neutral_quality_gate,
        'neutral_min_score_gap': _LIVE_DEFAULT.neutral_min_score_gap,
        'neutral_min_strength': _LIVE_DEFAULT.neutral_min_strength,
        'neutral_min_streak': _LIVE_DEFAULT.neutral_min_streak,
        'neutral_nochain_extra_gap': _LIVE_DEFAULT.neutral_nochain_extra_gap,
        'neutral_large_conflict_ratio': _LIVE_DEFAULT.neutral_large_conflict_ratio,
        # ── 信号置信度学习层 ──
        'use_confidence_learning': _LIVE_DEFAULT.use_confidence_learning,
        'confidence_min_raw': _LIVE_DEFAULT.confidence_min_raw,
        'confidence_min_posterior': _LIVE_DEFAULT.confidence_min_posterior,
        'confidence_min_samples': _LIVE_DEFAULT.confidence_min_samples,
        'confidence_block_after_samples': _LIVE_DEFAULT.confidence_block_after_samples,
        'confidence_threshold_gain': _LIVE_DEFAULT.confidence_threshold_gain,
        'confidence_threshold_min_mult': _LIVE_DEFAULT.confidence_threshold_min_mult,
        'confidence_threshold_max_mult': _LIVE_DEFAULT.confidence_threshold_max_mult,
        'confidence_prior_alpha': _LIVE_DEFAULT.confidence_prior_alpha,
        'confidence_prior_beta': _LIVE_DEFAULT.confidence_prior_beta,
        'confidence_win_pnl_r': _LIVE_DEFAULT.confidence_win_pnl_r,
        'confidence_loss_pnl_r': _LIVE_DEFAULT.confidence_loss_pnl_r,
        # ── Neutral 六书共识门控 ──
        'use_neutral_book_consensus': _LIVE_DEFAULT.use_neutral_book_consensus,
        'neutral_book_sell_threshold': _LIVE_DEFAULT.neutral_book_sell_threshold,
        'neutral_book_buy_threshold': _LIVE_DEFAULT.neutral_book_buy_threshold,
        'neutral_book_min_confirms': _LIVE_DEFAULT.neutral_book_min_confirms,
        'neutral_book_max_conflicts': _LIVE_DEFAULT.neutral_book_max_conflicts,
        'neutral_book_cs_kdj_threshold_adj': _LIVE_DEFAULT.neutral_book_cs_kdj_threshold_adj,
        # Neutral 结构质量渐进折扣
        'use_neutral_structural_discount': _LIVE_DEFAULT.use_neutral_structural_discount,
        'neutral_struct_activity_thr': _LIVE_DEFAULT.neutral_struct_activity_thr,
        'neutral_struct_discount_0': _LIVE_DEFAULT.neutral_struct_discount_0,
        'neutral_struct_discount_1': _LIVE_DEFAULT.neutral_struct_discount_1,
        'neutral_struct_discount_2': _LIVE_DEFAULT.neutral_struct_discount_2,
        'neutral_struct_discount_3': _LIVE_DEFAULT.neutral_struct_discount_3,
        'neutral_struct_discount_4plus': _LIVE_DEFAULT.neutral_struct_discount_4plus,
        'structural_discount_short_regimes': _LIVE_DEFAULT.structural_discount_short_regimes,
        'structural_discount_long_regimes': _LIVE_DEFAULT.structural_discount_long_regimes,
        # neutral short 结构确认器
        'use_neutral_short_structure_gate': _LIVE_DEFAULT.use_neutral_short_structure_gate,
        'neutral_short_structure_large_tfs': _LIVE_DEFAULT.neutral_short_structure_large_tfs,
        'neutral_short_structure_need_min_tfs': _LIVE_DEFAULT.neutral_short_structure_need_min_tfs,
        'neutral_short_structure_min_agree': _LIVE_DEFAULT.neutral_short_structure_min_agree,
        'neutral_short_structure_div_gap': _LIVE_DEFAULT.neutral_short_structure_div_gap,
        'neutral_short_structure_ma_gap': _LIVE_DEFAULT.neutral_short_structure_ma_gap,
        'neutral_short_structure_vp_gap': _LIVE_DEFAULT.neutral_short_structure_vp_gap,
        'neutral_short_structure_fail_open': _LIVE_DEFAULT.neutral_short_structure_fail_open,
        'neutral_short_structure_soften_weak': _LIVE_DEFAULT.neutral_short_structure_soften_weak,
        'neutral_short_structure_soften_mult': _LIVE_DEFAULT.neutral_short_structure_soften_mult,
        # 空单冲突软折扣（趋势/高波动）
        'use_short_conflict_soft_discount': _LIVE_DEFAULT.use_short_conflict_soft_discount,
        'short_conflict_regimes': _LIVE_DEFAULT.short_conflict_regimes,
        'short_conflict_div_buy_min': _LIVE_DEFAULT.short_conflict_div_buy_min,
        'short_conflict_ma_sell_min': _LIVE_DEFAULT.short_conflict_ma_sell_min,
        'short_conflict_discount_mult': _LIVE_DEFAULT.short_conflict_discount_mult,
        'use_long_conflict_soft_discount': _LIVE_DEFAULT.use_long_conflict_soft_discount,
        'long_conflict_regimes': _LIVE_DEFAULT.long_conflict_regimes,
        'long_conflict_div_sell_min': _LIVE_DEFAULT.long_conflict_div_sell_min,
        'long_conflict_ma_buy_min': _LIVE_DEFAULT.long_conflict_ma_buy_min,
        'long_conflict_discount_mult': _LIVE_DEFAULT.long_conflict_discount_mult,
        # 空单逆势防守退出（结构化风险控制）
        'use_short_adverse_exit': _LIVE_DEFAULT.use_short_adverse_exit,
        'short_adverse_min_bars': _LIVE_DEFAULT.short_adverse_min_bars,
        'short_adverse_loss_r': _LIVE_DEFAULT.short_adverse_loss_r,
        'short_adverse_bs': _LIVE_DEFAULT.short_adverse_bs,
        'short_adverse_bs_dom_ratio': _LIVE_DEFAULT.short_adverse_bs_dom_ratio,
        'short_adverse_ss_cap': _LIVE_DEFAULT.short_adverse_ss_cap,
        'short_adverse_require_bs_dom': _LIVE_DEFAULT.short_adverse_require_bs_dom,
        'short_adverse_ma_conflict_gap': _LIVE_DEFAULT.short_adverse_ma_conflict_gap,
        'short_adverse_conflict_thr': _LIVE_DEFAULT.short_adverse_conflict_thr,
        'short_adverse_min_conflicts': _LIVE_DEFAULT.short_adverse_min_conflicts,
        'short_adverse_need_cs_kdj': _LIVE_DEFAULT.short_adverse_need_cs_kdj,
        'short_adverse_large_bs_min': _LIVE_DEFAULT.short_adverse_large_bs_min,
        'short_adverse_large_ratio': _LIVE_DEFAULT.short_adverse_large_ratio,
        'short_adverse_need_chain_long': _LIVE_DEFAULT.short_adverse_need_chain_long,
        'short_adverse_regimes': _LIVE_DEFAULT.short_adverse_regimes,
        # 极端 divergence 做空否决
        'use_extreme_divergence_short_veto': _LIVE_DEFAULT.use_extreme_divergence_short_veto,
        'extreme_div_short_threshold': _LIVE_DEFAULT.extreme_div_short_threshold,
        'extreme_div_short_confirm_thr': _LIVE_DEFAULT.extreme_div_short_confirm_thr,
        'extreme_div_short_min_confirms': _LIVE_DEFAULT.extreme_div_short_min_confirms,
        'extreme_div_short_regimes': _LIVE_DEFAULT.extreme_div_short_regimes,
        # 回测日志复盘
        'print_signal_features': _LIVE_DEFAULT.print_signal_features,
        'signal_replay_top_n': _LIVE_DEFAULT.signal_replay_top_n,
        # ── S2: 保本止损 ──
        'use_breakeven_after_tp1': _LIVE_DEFAULT.use_breakeven_after_tp1,
        'breakeven_buffer': _LIVE_DEFAULT.breakeven_buffer,
        # ── S3: 棘轮追踪止损 ──
        'use_ratchet_trail': _LIVE_DEFAULT.use_ratchet_trail,
        'ratchet_trail_tiers': _LIVE_DEFAULT.ratchet_trail_tiers,
        # ── S5: 信号质量止损 ──
        'use_ss_quality_sl': _LIVE_DEFAULT.use_ss_quality_sl,
        'ss_quality_sl_threshold': _LIVE_DEFAULT.ss_quality_sl_threshold,
        'ss_quality_sl_mult': _LIVE_DEFAULT.ss_quality_sl_mult,
        # ── 止损后冷却倍数(v5.2) ──
        'short_sl_cd_mult': _LIVE_DEFAULT.short_sl_cd_mult,
        'long_sl_cd_mult': _LIVE_DEFAULT.long_sl_cd_mult,
    }
    return cfg


DEFAULT_CONFIG = _build_default_config()

TF_HOURS = {
    '10m': 1/6, '15m': 0.25, '30m': 0.5, '1h': 1, '2h': 2, '3h': 3,
    '4h': 4, '6h': 6, '8h': 8, '12h': 12, '16h': 16, '24h': 24,
}


def _scale_runtime_config(base_config, primary_tf):
    """按主周期缩放 hold/cooldown."""
    config = dict(base_config)
    tf_h = TF_HOURS.get(primary_tf, 1)
    config['short_max_hold'] = max(6, int(config.get('short_max_hold', 72) / tf_h))
    config['long_max_hold'] = max(6, int(config.get('long_max_hold', 72) / tf_h))
    config['cooldown'] = max(1, int(config.get('cooldown', 4) / tf_h))
    config['spot_cooldown'] = max(2, int(config.get('spot_cooldown', 12) / tf_h))
    return config


def fetch_data_for_tf(tf, days, allow_api_fallback=False):
    """优先从本地K线库读取；可配置是否允许API回退。"""
    fetch_days = days + 30
    start_dt = (pd.Timestamp.now().tz_localize(None) - pd.Timedelta(days=fetch_days)).strftime('%Y-%m-%d')
    try:
        df = load_klines(
            symbol="ETHUSDT",
            interval=tf,
            start=start_dt,
            end=None,
            with_indicators=False,
            allow_api_fallback=allow_api_fallback,
        )
        if df is not None and len(df) > 50:
            df = add_all_indicators(df)
            add_moving_averages(df, timeframe=tf)
            return df
    except Exception as e:
        print(f"  获取 {tf} 数据失败: {e}")
    return None


def _history_to_daily(history, trades, initial_capital, trade_start, trade_end):
    """
    将 FuturesEngine 的 history 快照列表 + trades 列表
    转换为逐日记录，覆盖 trade_start ~ trade_end 每一天。
    """
    if not history:
        return []

    # 按天分组 history
    hist_by_day = defaultdict(list)
    for h in history:
        day_str = h['time'][:10]
        hist_by_day[day_str].append(h)

    # 按天分组 trades
    trades_by_day = defaultdict(list)
    for t in trades:
        day_str = (t.get('time') or '')[:10]
        trades_by_day[day_str].append(t)

    day_trade_stats = {}
    for day_str, day_trades in trades_by_day.items():
        day_pnl = sum(t.get('pnl', 0) for t in day_trades if t.get('pnl'))
        has_long = False
        has_short = False
        long_entry = None
        long_qty = None
        short_entry = None
        short_qty = None
        for t in reversed(day_trades):
            if not has_long and t.get('has_long') and t.get('long_entry'):
                has_long = True
                long_entry = t.get('long_entry')
                long_qty = t.get('long_qty')
            if not has_short and t.get('has_short') and t.get('short_entry'):
                has_short = True
                short_entry = t.get('short_entry')
                short_qty = t.get('short_qty')
            if has_long and has_short:
                break
        day_trade_stats[day_str] = {
            'day_trades': len(day_trades),
            'day_pnl': round(day_pnl, 2),
            'has_long': has_long,
            'has_short': has_short,
            'long_entry': long_entry,
            'long_qty': long_qty,
            'short_entry': short_entry,
            'short_qty': short_qty,
        }

    # 生成日期范围
    start = pd.Timestamp(trade_start)
    end = pd.Timestamp(trade_end)
    date_range = pd.date_range(start, end, freq='D')

    daily_records = []
    peak = initial_capital
    last_snapshot = None

    for dt in date_range:
        day_str = dt.strftime('%Y-%m-%d')
        day_hists = hist_by_day.get(day_str, [])

        # 取当天最后一个快照，若无则沿用前一天
        snap = day_hists[-1] if day_hists else last_snapshot
        if snap is None:
            continue
        last_snapshot = snap

        total = snap.get('total', initial_capital)
        peak = max(peak, total)
        drawdown = round((total - peak) / peak * 100, 2) if peak > 0 else 0
        return_pct = round((total / initial_capital - 1) * 100, 2)

        day_stat = day_trade_stats.get(day_str)
        if day_stat is None:
            day_stat = {
                'day_trades': 0,
                'day_pnl': 0.0,
                'has_long': False,
                'has_short': False,
                'long_entry': None,
                'long_qty': None,
                'short_entry': None,
                'short_qty': None,
            }

        rec = {
            'date': day_str,
            'eth_price': snap.get('eth_price'),
            'total_value': round(total, 2),
            'usdt': snap.get('usdt'),
            'frozen_margin': snap.get('frozen_margin', 0),
            'long_pnl': snap.get('long_pnl', 0),
            'short_pnl': snap.get('short_pnl', 0),
            'spot_eth_value': snap.get('spot_eth_value', 0),
            'return_pct': return_pct,
            'drawdown_pct': drawdown,
            'has_long': snap.get('long_pnl', 0) != 0,
            'has_short': snap.get('short_pnl', 0) != 0,
            'long_entry': day_stat['long_entry'],
            'long_qty': day_stat['long_qty'],
            'short_entry': day_stat['short_entry'],
            'short_qty': day_stat['short_qty'],
            'day_trades': day_stat['day_trades'],
            'day_pnl': day_stat['day_pnl'],
        }
        if day_stat['has_long']:
            rec['has_long'] = True
        if day_stat['has_short']:
            rec['has_short'] = True

        # 也从 history 快照推断 (如果当天没交易但有持仓)
        if snap.get('long_pnl', 0) != 0:
            rec['has_long'] = True
        if snap.get('short_pnl', 0) != 0:
            rec['has_short'] = True

        daily_records.append(rec)

    return daily_records


def _normalize_trade(t):
    """统一清洗交易记录"""
    def _num(v):
        if v is None:
            return None
        try:
            return round(float(v), 4)
        except (TypeError, ValueError):
            return None

    out = {
        'time': str(t.get('time', '')),
        'action': t.get('action', ''),
        'direction': t.get('direction', ''),
        'market_price': _num(t.get('market_price')),
        'exec_price': _num(t.get('exec_price')),
        'quantity': _num(t.get('quantity')),
        'notional_value': _num(t.get('notional_value')),
        'margin': _num(t.get('margin')),
        'margin_released': _num(t.get('margin_released')),
        'leverage': t.get('leverage'),
        'fee': _num(t.get('fee')),
        'slippage_cost': _num(t.get('slippage_cost')),
        'total_cost': _num(t.get('total_cost')),
        'pnl': _num(t.get('pnl')),
        'entry_price': _num(t.get('entry_price')),
        'after_usdt': _num(t.get('after_usdt')),
        'after_spot_eth': _num(t.get('after_spot_eth')),  # 现货ETH持仓(修复: 之前遗漏)
        'after_spot_value': _num(t.get('after_spot_value')),  # 现货ETH价值
        'after_total': _num(t.get('after_total')),
        'after_frozen_margin': _num(t.get('after_frozen_margin')),
        'after_available': _num(t.get('after_available')),
        'has_long': bool(t.get('has_long')),
        'has_short': bool(t.get('has_short')),
        'long_entry': _num(t.get('long_entry')),
        'long_qty': _num(t.get('long_qty')),
        'short_entry': _num(t.get('short_entry')),
        'short_qty': _num(t.get('short_qty')),
        'cum_spot_fees': _num(t.get('cum_spot_fees')),
        'cum_futures_fees': _num(t.get('cum_futures_fees')),
        'cum_funding_paid': _num(t.get('cum_funding_paid')),
        'cum_slippage': _num(t.get('cum_slippage')),
        'reason': str(t.get('reason', '')),
        # 结构化观测字段 (分市场段统计 + 后续归因分析)
        'regime_label': t.get('regime_label', 'unknown'),
        'ss': _num(t.get('ss')),
        'bs': _num(t.get('bs')),
        'atr_pct': _num(t.get('atr_pct')),
    }
    for k, v in t.items():
        if k in out:
            continue
        if k.startswith('sig_') or k.startswith('book_'):
            if isinstance(v, (int, float, np.number)):
                out[k] = _num(v)
            else:
                out[k] = v
    return out


def _build_signal_replay_report(trades, top_n=10):
    """基于交易日志复盘信号质量：开仓→分段TP→平仓配对。"""
    active = {'short': None, 'long': None}
    samples = []

    def _finalize(direction, close_trade):
        ctx = active.get(direction)
        if not ctx:
            return
        entry = ctx['entry']
        total_pnl = float(ctx.get('partial_pnl', 0.0)) + float(close_trade.get('pnl') or 0.0)
        entry_margin = float(entry.get('margin') or 0.0)
        pnl_r = total_pnl / entry_margin if entry_margin > 0 else 0.0
        samples.append({
            'direction': direction,
            'entry_time': entry.get('time'),
            'exit_time': close_trade.get('time'),
            'regime': entry.get('regime_label', 'unknown'),
            'conf_raw': float(entry.get('sig_conf_raw') or 0.0),
            'conf_eff': float(entry.get('sig_conf_effective') or 0.0),
            'conf_bucket': str(entry.get('sig_conf_bucket', 'na')),
            'ss': float(entry.get('ss') or 0.0),
            'bs': float(entry.get('bs') or 0.0),
            'entry_margin': entry_margin,
            'total_pnl': round(total_pnl, 4),
            'pnl_r': round(pnl_r, 6),
            'exit_action': close_trade.get('action'),
            'exit_reason': close_trade.get('reason'),
            'sig_chain_len': int(entry.get('sig_chain_len') or 0),
            'sig_coverage': float(entry.get('sig_coverage') or 0.0),
            'book_div_sell': float(entry.get('book_div_sell') or 0.0),
            'book_div_buy': float(entry.get('book_div_buy') or 0.0),
            'book_ma_sell': float(entry.get('book_ma_sell') or 0.0),
            'book_ma_buy': float(entry.get('book_ma_buy') or 0.0),
            'book_vp_sell': float(entry.get('book_vp_sell') or 0.0),
            'book_vp_buy': float(entry.get('book_vp_buy') or 0.0),
        })
        active[direction] = None

    for t in trades:
        action = str(t.get('action', ''))
        direction = str(t.get('direction', ''))
        if action == 'OPEN_SHORT':
            active['short'] = {'entry': t, 'partial_pnl': 0.0}
        elif action == 'OPEN_LONG':
            active['long'] = {'entry': t, 'partial_pnl': 0.0}
        elif action == 'PARTIAL_TP' and direction in ('short', 'long'):
            if active.get(direction):
                active[direction]['partial_pnl'] += float(t.get('pnl') or 0.0)
        elif action in ('CLOSE_SHORT', 'LIQUIDATED') and direction == 'short':
            _finalize('short', t)
        elif action in ('CLOSE_LONG', 'LIQUIDATED') and direction == 'long':
            _finalize('long', t)

    if not samples:
        return {}

    bucket_stats = {}
    bucket_map = defaultdict(list)
    for s in samples:
        bucket_map[s['conf_bucket']].append(s)
    for bk, arr in bucket_map.items():
        n = len(arr)
        wins = sum(1 for x in arr if x['pnl_r'] > 0)
        losses = sum(1 for x in arr if x['pnl_r'] < 0)
        avg_r = float(np.mean([x['pnl_r'] for x in arr])) if arr else 0.0
        bucket_stats[bk] = {
            'n': n,
            'wins': wins,
            'losses': losses,
            'win_rate': round(wins / n, 4) if n else 0.0,
            'avg_pnl_r': round(avg_r, 6),
        }

    high_conf_misses = sorted(
        [x for x in samples if x['conf_raw'] >= 0.70 and x['pnl_r'] < 0],
        key=lambda x: (x['conf_raw'], -x['pnl_r']),
        reverse=True,
    )[:max(1, int(top_n))]

    return {
        'samples': len(samples),
        'bucket_stats': bucket_stats,
        'high_conf_misses': high_conf_misses,
    }


def _print_signal_feature_samples(trades, limit=12):
    """打印开仓信号特征样本，便于人工复盘。"""
    rows = [t for t in trades if t.get('action') in ('OPEN_LONG', 'OPEN_SHORT')]
    if not rows:
        return
    print(f"\n  {'─' * 60}")
    print("  信号特征样本 (开仓)")
    print(f"  {'─' * 60}")
    print("  time                act   R          conf   cov  chain   ss    bs  div_s  div_b  ma_s  ma_b")
    for t in rows[:max(1, int(limit))]:
        print(
            f"  {str(t.get('time', ''))[:19]:<19} "
            f"{str(t.get('action', '')):<9} "
            f"{str(t.get('regime_label', 'unknown')):<10} "
            f"{float(t.get('sig_conf_raw') or 0.0):>5.2f} "
            f"{float(t.get('sig_coverage') or 0.0):>5.2f} "
            f"{int(t.get('sig_chain_len') or 0):>5d} "
            f"{float(t.get('ss') or 0.0):>5.0f} "
            f"{float(t.get('bs') or 0.0):>5.0f} "
            f"{float(t.get('book_div_sell') or 0.0):>6.1f} "
            f"{float(t.get('book_div_buy') or 0.0):>6.1f} "
            f"{float(t.get('book_ma_sell') or 0.0):>5.1f} "
            f"{float(t.get('book_ma_buy') or 0.0):>5.1f}"
        )


def _save_signal_feature_log_csv(trades, run_id, trade_start, trade_end):
    rows = [t for t in trades if any(k.startswith('sig_') or k.startswith('book_') for k in t.keys())]
    if not rows:
        return None
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, 'logs', 'signal_audit')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'run_{run_id}_{trade_start}_{trade_end}_signal_features.csv')
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8')
    return path


def main(trade_start=None, trade_end=None, version_tag=None, experiment_notes=None):
    t0 = time.time()
    perf_log = {}  # 性能日志: 阶段 -> 耗时(秒)

    # CLI 参数解析
    parser = argparse.ArgumentParser(description='多周期联合决策回测')
    parser.add_argument('--start', type=str, default=None, help='回测起始日 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='回测结束日 (YYYY-MM-DD)')
    parser.add_argument('--tag', type=str, default=None, help='策略版本标签')
    parser.add_argument('--notes', type=str, default=None,
                        help='实验说明文本，会存入DB方便查阅')
    parser.add_argument('--fast', action='store_true', default=False,
                        help='[实验性] P0向量化信号计算, 结果与原版存在近似偏差(±1%%), 不建议作为正式策略结论依据')
    parser.add_argument('--override', action='append', default=[],
                        help='覆盖配置参数, 格式: key=value (可多次使用, bool用true/false)')
    args, _ = parser.parse_known_args()

    TRADE_START = args.start or trade_start or DEFAULT_TRADE_START
    TRADE_END = args.end or trade_end or DEFAULT_TRADE_END
    if args.tag:
        version_tag = args.tag
    if args.notes:
        experiment_notes = args.notes

    # ── 应用 --override 参数 ──
    if args.override:
        for ov in args.override:
            if '=' not in ov:
                print(f"  ⚠️  忽略无效 override: {ov} (格式: key=value)")
                continue
            k, v = ov.split('=', 1)
            k = k.strip()
            if k not in DEFAULT_CONFIG:
                print(f"  ⚠️  忽略未知 override key: {k}")
                continue
            orig = DEFAULT_CONFIG[k]
            # 类型推断: 先尝试 JSON (支持 dict/list)，再按原值类型
            if v.strip().startswith('{') or v.strip().startswith('['):
                import json as _json
                try:
                    DEFAULT_CONFIG[k] = _json.loads(v)
                except Exception:
                    DEFAULT_CONFIG[k] = v
            elif isinstance(orig, bool):
                DEFAULT_CONFIG[k] = v.lower() in ('true', '1', 'yes')
            elif orig is None:
                # None 默认值: 尝试数字, 否则字符串
                try:
                    DEFAULT_CONFIG[k] = int(v)
                except ValueError:
                    try:
                        DEFAULT_CONFIG[k] = float(v)
                    except ValueError:
                        DEFAULT_CONFIG[k] = v if v.lower() != 'none' else None
            elif isinstance(orig, int):
                DEFAULT_CONFIG[k] = int(v)
            elif isinstance(orig, float):
                DEFAULT_CONFIG[k] = float(v)
            else:
                DEFAULT_CONFIG[k] = v
            print(f"  ✏️  override: {k} = {DEFAULT_CONFIG[k]} (原值: {orig})")

    preferred_combo_name = f"四TF联合({'+'.join(DECISION_TFS)})"
    print("=" * 80)
    print("  多周期联合决策 — 逐日盈亏回测")
    print(f"  区间: {TRADE_START} ~ {TRADE_END}")
    if version_tag:
        print(f"  版本标签: {version_tag}")
    print(f"  主TF: {PRIMARY_TF}  |  决策TFs: {', '.join(DECISION_TFS)}")
    use_fast_signals = args.fast
    if use_fast_signals:
        print("  " + "!" * 60)
        print("  ⚠️  --fast 模式已启用 (实验性近似算法)")
        print("  ⚠️  信号计算与原版存在偏差, 不建议作为正式策略结论依据")
        print("  ⚠️  正式回测请去掉 --fast 参数")
        print("  " + "!" * 60)
    allow_api_fallback = os.getenv('BACKTEST_DAILY_ALLOW_API_FALLBACK', '0') == '1'
    print(f"  K线数据源: {'本地优先+API回退' if allow_api_fallback else '仅本地'}")
    print(f"  信号加速: {'⚠️ ON (P0向量化/实验性)' if use_fast_signals else 'OFF (原版精确)'}")
    print(f"  策略参数版本: {get_strategy_version()} (STRATEGY_VERSION 环境变量可切换)")
    # 显示关键开关状态
    trend_gate_mode = 'gated' if DEFAULT_CONFIG.get('trend_enhance_engine_gate') else 'decoupled'
    print(f"  趋势保护v3: {'ON' if DEFAULT_CONFIG.get('use_trend_enhance') else 'OFF'}({trend_gate_mode})"
          f"  |  微结构: {'ON' if DEFAULT_CONFIG.get('use_microstructure') else 'OFF'}"
          f"  |  双引擎: {'ON' if DEFAULT_CONFIG.get('use_dual_engine') else 'OFF'}"
          f"  |  波动目标: {'ON' if DEFAULT_CONFIG.get('use_vol_target') else 'OFF'}")
    print("=" * 80)

    # ── 1. 获取数据 ──
    # 需要足够长的历史来覆盖 trade_start 之前的预热期
    # 动态计算: 从现在到 TRADE_START 的天数 + 缓冲
    _days_to_start = (pd.Timestamp.now() - pd.Timestamp(TRADE_START)).days
    history_days = max(560, _days_to_start + 90)
    print(f"\n[1/4] 获取数据 ({history_days}天)...")

    t_phase1 = time.time()
    fetch_workers = max(1, min(len(AVAILABLE_TFS), int(os.getenv('BACKTEST_DAILY_FETCH_WORKERS', '3'))))
    print(f"  抓取并发: {fetch_workers}")
    all_data = {}

    def _fetch_tf_batch(tf_list):
        if not tf_list:
            return
        if fetch_workers == 1:
            for tf in tf_list:
                t_tf = time.time()
                print(f"  获取 {tf} 数据...")
                df = fetch_data_for_tf(tf, history_days, allow_api_fallback=allow_api_fallback)
                if df is not None:
                    all_data[tf] = df
                    elapsed_tf = time.time() - t_tf
                    print(f"    {tf}: {len(df)} 条K线, {df.index[0]} ~ {df.index[-1]}  [{elapsed_tf:.2f}s]")
                else:
                    print(f"    {tf}: 失败!")
            return

        start_map = {}
        with ThreadPoolExecutor(max_workers=fetch_workers) as executor:
            futures = {}
            for tf in tf_list:
                print(f"  获取 {tf} 数据...")
                start_map[tf] = time.time()
                futures[executor.submit(fetch_data_for_tf, tf, history_days, allow_api_fallback)] = tf
            for future in as_completed(futures):
                tf = futures[future]
                elapsed_tf = time.time() - start_map.get(tf, time.time())
                try:
                    df = future.result()
                except Exception as e:
                    df = None
                    print(f"    {tf}: 失败! {e}")
                if df is not None:
                    all_data[tf] = df
                    print(f"    {tf}: {len(df)} 条K线, {df.index[0]} ~ {df.index[-1]}  [{elapsed_tf:.2f}s]")
                else:
                    print(f"    {tf}: 失败!")

    # 阶段1: 先抓主TF + 优先决策TF
    phase1_tfs = list(dict.fromkeys([PRIMARY_TF, *DECISION_TFS]))
    _fetch_tf_batch(phase1_tfs)

    # 阶段2: 仅当优先决策TF不足时，再补抓 fallback TF
    pref_decision_available = [tf for tf in DECISION_TFS if tf in all_data]
    if len(pref_decision_available) < 2:
        phase2_tfs = [tf for tf in FALLBACK_DECISION_TFS if tf not in all_data]
        if phase2_tfs:
            print("  优先决策TF不足，补抓 fallback TF...")
            _fetch_tf_batch(phase2_tfs)

    perf_log['1_data_load'] = time.time() - t_phase1

    available_tfs = [tf for tf in AVAILABLE_TFS if tf in all_data]
    decision_tfs = [tf for tf in DECISION_TFS if tf in available_tfs]
    tf_source = "preferred"
    if len(decision_tfs) < 2:
        decision_tfs = [tf for tf in FALLBACK_DECISION_TFS if tf in available_tfs]
        tf_source = "fallback"
    if len(decision_tfs) < 2:
        print("❌ 可用TF不足2个, 无法执行多周期决策")
        sys.exit(1)

    if PRIMARY_TF not in all_data:
        print(f"❌ 主TF {PRIMARY_TF} 数据获取失败")
        sys.exit(1)

    combo_name = f"多TF联合({'+'.join(decision_tfs)})"
    print(f"\n  可用TFs: {', '.join(available_tfs)}")
    print(f"  决策TFs({tf_source}): {', '.join(decision_tfs)}")
    if tf_source == "fallback":
        print(f"  说明: 优先组合 {preferred_combo_name} 不完整，已自动回退")

    score_tfs = list(dict.fromkeys([PRIMARY_TF, *decision_tfs]))

    # ── 2. 计算信号 ──
    # ⚠️ 必须使用 max_bars=0 全量计算！
    # max_bars>0 会截断df到尾部N根，但 _build_tf_score_index 用全量df的idx
    # 去 .iloc 索引截断后的信号Series，导致严重错位（100%不一致）。
    print(f"\n[2/4] 计算六维信号 (全量, max_bars=0)...")
    t_phase2 = time.time()
    signal_workers = max(1, min(len(score_tfs), int(os.getenv('BACKTEST_DAILY_SIGNAL_WORKERS', '2'))))
    use_multiprocess = os.getenv('BACKTEST_MULTIPROCESS', '1') == '1'  # 默认启用模块级多进程
    print(f"  信号并发: {signal_workers}  |  目标TF: {', '.join(score_tfs)}  |  多进程: {'ON' if use_multiprocess else 'OFF'}")
    all_signals = {}
    if use_multiprocess and not use_fast_signals:
        # 模块级多进程: 24个任务分发到多核, 瓶颈=最慢单模块(~40s)
        mp_workers = int(os.getenv('BACKTEST_MP_WORKERS', '0')) or None  # 0=auto
        all_signals = compute_signals_six_multiprocess(all_data, score_tfs, max_workers=mp_workers)
    elif signal_workers == 1:
        for tf in score_tfs:
            t_tf = time.time()
            print(f"  计算 {tf} 信号 ({len(all_data[tf])} bars)...")
            all_signals[tf] = compute_signals_six(all_data[tf], tf, all_data, max_bars=0, fast=use_fast_signals)
            elapsed_tf = time.time() - t_tf
            print(f"    {tf} 信号完成  [{elapsed_tf:.2f}s]")
    else:
        start_map = {}
        with ThreadPoolExecutor(max_workers=signal_workers) as executor:
            futures = {}
            for tf in score_tfs:
                print(f"  计算 {tf} 信号 ({len(all_data[tf])} bars)...")
                start_map[tf] = time.time()
                futures[executor.submit(compute_signals_six, all_data[tf], tf, all_data, 0, use_fast_signals)] = tf
            for future in as_completed(futures):
                tf = futures[future]
                elapsed_tf = time.time() - start_map.get(tf, time.time())
                try:
                    all_signals[tf] = future.result()
                    print(f"    {tf} 信号完成  [{elapsed_tf:.2f}s]")
                except Exception as e:
                    print(f"    {tf} 信号失败: {e}")
                    raise
    perf_log['2_signal_calc'] = time.time() - t_phase2
    print(f"  信号计算完成: {len(all_signals)} 个TF  [总计 {perf_log['2_signal_calc']:.2f}s]")

    # 打印子模块 profiling
    sub_perf_total = {}
    for tf in score_tfs:
        sub_perf = all_signals[tf].get('_perf', {})
        if sub_perf:
            parts = '  '.join(f"{k}={v:.1f}s" for k, v in sorted(sub_perf.items()))
            print(f"    {tf:>4s} 细分: {parts}")
            for k, v in sub_perf.items():
                sub_perf_total[k] = sub_perf_total.get(k, 0) + v
    if sub_perf_total:
        print(f"    {'合计':>4s} 细分: {'  '.join(f'{k}={v:.1f}s' for k, v in sorted(sub_perf_total.items()))}")

    # ── 3. 构建评分索引 ──
    print(f"\n[3/4] 构建TF评分索引...")
    t_phase3 = time.time()
    config = _scale_runtime_config(DEFAULT_CONFIG, PRIMARY_TF)
    config['name'] = f"多TF逐日_{combo_name}@{PRIMARY_TF}"
    tf_score_index = _build_tf_score_index(all_data, all_signals, score_tfs, config)
    perf_log['3_score_index'] = time.time() - t_phase3
    for tf in score_tfs:
        n_scores = len(tf_score_index.get(tf, {}))
        print(f"    {tf}: {n_scores} 个评分点")
    print(f"  评分索引构建完成  [总计 {perf_log['3_score_index']:.2f}s]")

    # ── 4. 运行多周期回测 ──
    print(f"\n[4/4] 运行多周期联合决策回测...")
    t_phase4 = time.time()
    trade_start_dt = pd.Timestamp(TRADE_START)
    trade_end_dt = pd.Timestamp(TRADE_END) + pd.Timedelta(hours=23, minutes=59)

    # trade_days 设为 0 或 None，因为我们显式指定了 start/end
    result = run_strategy_multi_tf(
        primary_df=all_data[PRIMARY_TF],
        tf_score_map=tf_score_index,
        decision_tfs=decision_tfs,
        config=config,
        primary_tf=PRIMARY_TF,
        trade_days=0,
        trade_start_dt=trade_start_dt,
        trade_end_dt=trade_end_dt,
    )
    perf_log['4_strategy_run'] = time.time() - t_phase4

    # ── 结果提取 ──
    history = result.get('history', [])
    raw_trades = result.get('trades', [])
    fees = result.get('fees', {})

    initial_capital = result.get('initial_total', 100000)
    final_total = result.get('final_total', 0)
    strategy_return = result.get('strategy_return', 0)
    buy_hold_return = result.get('buy_hold_return', 0)
    alpha = result.get('alpha', 0)
    max_drawdown = result.get('max_drawdown', 0)

    # 清洗 trades
    trades = [_normalize_trade(t) for t in raw_trades]

    # 转成逐日记录
    daily_records = _history_to_daily(
        history, raw_trades, initial_capital,
        TRADE_START, TRADE_END,
    )

    # 计算胜率等交易统计
    # ── contract_pf: 仅合约平仓 (CLOSE_LONG/CLOSE_SHORT/LIQUIDATED) ──
    close_actions = {'CLOSE_LONG', 'CLOSE_SHORT', 'LIQUIDATED'}
    close_trades = []
    wins = []
    losses = []
    for t in trades:
        if t['action'] not in close_actions:
            continue
        close_trades.append(t)
        if (t.get('pnl') or 0) > 0:
            wins.append(t)
        else:
            losses.append(t)
    win_rate = round(len(wins) / len(close_trades) * 100, 2) if close_trades else 0
    avg_win = round(sum(t['pnl'] for t in wins) / len(wins), 2) if wins else 0
    avg_loss = round(sum(t['pnl'] for t in losses) / len(losses), 2) if losses else 0
    contract_pf = round(
        abs(sum(t['pnl'] for t in wins)) / abs(sum(t['pnl'] for t in losses)), 2
    ) if losses and sum(t['pnl'] for t in losses) != 0 else 999

    # ── portfolio_pf: 全量 PnL (含 PARTIAL_TP / SPOT_SELL) ──
    pnl_actions = {'CLOSE_LONG', 'CLOSE_SHORT', 'LIQUIDATED', 'PARTIAL_TP', 'SPOT_SELL'}
    all_pnl_trades = [t for t in trades if t['action'] in pnl_actions and t.get('pnl') is not None]
    all_wins = [t for t in all_pnl_trades if (t.get('pnl') or 0) > 0]
    all_losses = [t for t in all_pnl_trades if (t.get('pnl') or 0) < 0]
    gross_profit = sum(t['pnl'] for t in all_wins)
    gross_loss = abs(sum(t['pnl'] for t in all_losses))
    portfolio_pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999

    # 兼容性: profit_factor 保持为 contract_pf
    profit_factor = contract_pf

    elapsed = time.time() - t0

    summary = {
        'initial_capital': initial_capital,
        'final_capital': final_total,
        'total_return_pct': strategy_return,
        'buy_hold_return_pct': buy_hold_return,
        'alpha_pct': alpha,
        'max_drawdown_pct': max_drawdown,
        'total_trades': len(raw_trades),
        'close_trades': len(close_trades),
        'win_count': len(wins),
        'loss_count': len(losses),
        'win_rate_pct': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'contract_pf': contract_pf,
        'portfolio_pf': portfolio_pf,
        'total_fees': fees.get('total_fees', 0),
        'total_slippage': fees.get('slippage_cost', 0),
        'total_costs': fees.get('total_costs', 0),
        'funding_paid': fees.get('funding_paid', 0),
        'funding_received': fees.get('funding_received', 0),
        'net_funding': fees.get('net_funding', 0),
        'fee_drag_pct': fees.get('fee_drag_pct', 0),
        'liquidations': result.get('liquidations', 0),
    }
    if result.get('neutral_quality_gate'):
        summary['neutral_quality_gate'] = result.get('neutral_quality_gate')
    if result.get('book_consensus_gate'):
        summary['book_consensus_gate'] = result.get('book_consensus_gate')
    if result.get('neutral_short_structure_gate'):
        summary['neutral_short_structure_gate'] = result.get('neutral_short_structure_gate')
    if result.get('structural_discount'):
        summary['structural_discount'] = result.get('structural_discount')
    if result.get('confidence_learning'):
        summary['confidence_learning'] = result.get('confidence_learning')
    if result.get('short_adverse_exit'):
        summary['short_adverse_exit'] = result.get('short_adverse_exit')
    if result.get('short_conflict_soft_discount'):
        summary['short_conflict_soft_discount'] = result.get('short_conflict_soft_discount')
    if result.get('long_conflict_soft_discount'):
        summary['long_conflict_soft_discount'] = result.get('long_conflict_soft_discount')
    if result.get('extreme_div_short_veto'):
        summary['extreme_div_short_veto'] = result.get('extreme_div_short_veto')
    signal_replay = _build_signal_replay_report(
        trades, top_n=int(config.get('signal_replay_top_n', 10))
    )
    if signal_replay:
        summary['signal_replay'] = signal_replay

    run_meta = {
        'start_date': TRADE_START,
        'end_date': TRADE_END,
        'primary_tf': PRIMARY_TF,
        'decision_tfs': decision_tfs,
        'combo_name': combo_name,
        'leverage': config.get('lev', 5),
        'initial_capital': initial_capital,
        'signal_mode': 'fast' if use_fast_signals else 'original',
        'strategy_version': get_strategy_version(),
        'multiprocess': use_multiprocess,
        'runner': 'backtest_multi_tf_daily.py',
        'host': socket.gethostname(),
        'run_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_sec': round(elapsed, 1),
        'config': {k: v for k, v in config.items() if not k.startswith('_')},
    }

    # ── 打印汇总 ──
    print(f"\n{'=' * 80}")
    print(f"  回测完成！ ({elapsed:.1f}秒)")
    print(f"{'=' * 80}")
    print(f"  区间:        {TRADE_START} ~ {TRADE_END}")
    print(f"  初始资金:    ${initial_capital:,.0f}")
    print(f"  期末资金:    ${final_total:,.0f}")
    print(f"  策略收益:    {strategy_return:+.2f}%")
    print(f"  买入持有:    {buy_hold_return:+.2f}%")
    print(f"  Alpha:       {alpha:+.2f}%")
    print(f"  最大回撤:    {max_drawdown:.2f}%")
    print(f"  交易次数:    {len(raw_trades)} (平仓 {len(close_trades)})")
    print(f"  胜率:        {win_rate:.1f}%")
    print(f"  合约PF:      {contract_pf:.2f} (仅 CLOSE_LONG/SHORT/LIQUIDATED)")
    print(f"  组合PF:      {portfolio_pf:.2f} (含 PARTIAL_TP / SPOT_SELL)")
    print(f"  总费用:      ${fees.get('total_costs', 0):,.2f}")
    print(f"  逐日记录:    {len(daily_records)} 天")
    if summary.get('neutral_quality_gate'):
        ng = summary['neutral_quality_gate']
        br = ng.get('blocked_reason_counts', {}) or {}
        br_text = ', '.join(f"{k}:{v}" for k, v in sorted(br.items())) if br else '-'
        print(f"  Neutral门控: short_blocked={ng.get('short_blocked', 0)} "
              f"long_blocked={ng.get('long_blocked', 0)} reasons[{br_text}]")
    if summary.get('book_consensus_gate'):
        bcg = summary['book_consensus_gate']
        bcr = bcg.get('reason_counts', {}) or {}
        bcr_text = ', '.join(f"{k}:{v}" for k, v in sorted(bcr.items())) if bcr else '-'
        print(f"  六书共识:    eval={bcg.get('evaluated', 0)} "
              f"short_blk={bcg.get('short_blocked', 0)} "
              f"long_blk={bcg.get('long_blocked', 0)} "
              f"cs_kdj_adj={bcg.get('cs_kdj_threshold_adj_count', 0)} "
              f"reasons[{bcr_text}]")
    if summary.get('neutral_short_structure_gate'):
        nss = summary['neutral_short_structure_gate']
        rr = nss.get('reason_counts', {}) or {}
        rr_text = ', '.join(f"{k}:{v}" for k, v in sorted(rr.items())) if rr else '-'
        print(f"  Neutral结构: evaluated={nss.get('evaluated', 0)} "
              f"blocked={nss.get('blocked', 0)} support_avg={nss.get('support_avg', 0.0):.2f} "
              f"reasons[{rr_text}]")
    if summary.get('structural_discount'):
        sd = summary['structural_discount']
        cd = sd.get('confirm_distribution', {})
        cd_text = ' '.join(f"{k}c:{v}" for k, v in sorted(cd.items(), key=lambda x: int(x[0])))
        print(f"  结构折扣:    eval={sd.get('evaluated', 0)} "
              f"discounted={sd.get('discount_applied', 0)} avg_mult={sd.get('avg_mult', 0.0):.3f} "
              f"dist[{cd_text}]")
    if summary.get('confidence_learning'):
        cl = summary['confidence_learning']
        cbr = cl.get('blocked_reason_counts', {}) or {}
        cbr_text = ', '.join(f"{k}:{v}" for k, v in sorted(cbr.items())) if cbr else '-'
        print(f"  置信学习:    updates={cl.get('updates', 0)} "
              f"wins={cl.get('wins', 0)} losses={cl.get('losses', 0)} "
              f"short_blocked={cl.get('short_blocked', 0)} long_blocked={cl.get('long_blocked', 0)} "
              f"reasons[{cbr_text}]")
    if summary.get('short_adverse_exit'):
        sa = summary['short_adverse_exit']
        rr = sa.get('reason_counts', {}) or {}
        rr_text = ', '.join(f"{k}:{v}" for k, v in sorted(rr.items())) if rr else '-'
        print(f"  逆势防守:    eval={sa.get('evaluated', 0)} "
              f"triggered={sa.get('triggered', 0)} avg_pnl_r={sa.get('avg_trigger_pnl_r', 0.0):+.4f} "
              f"reasons[{rr_text}]")
    if summary.get('short_conflict_soft_discount'):
        sc = summary['short_conflict_soft_discount']
        rr = sc.get('reason_counts', {}) or {}
        rr_text = ', '.join(f"{k}:{v}" for k, v in sorted(rr.items())) if rr else '-'
        print(f"  空单冲突折扣: eval={sc.get('evaluated', 0)} "
              f"triggered={sc.get('triggered', 0)} avg_mult={sc.get('avg_mult_on_triggered', 1.0):.3f} "
              f"reasons[{rr_text}]")
    if summary.get('long_conflict_soft_discount'):
        lc = summary['long_conflict_soft_discount']
        rr = lc.get('reason_counts', {}) or {}
        rr_text = ', '.join(f"{k}:{v}" for k, v in sorted(rr.items())) if rr else '-'
        print(f"  多单冲突折扣: eval={lc.get('evaluated', 0)} "
              f"triggered={lc.get('triggered', 0)} avg_mult={lc.get('avg_mult_on_triggered', 1.0):.3f} "
              f"reasons[{rr_text}]")
    if summary.get('extreme_div_short_veto'):
        dv = summary['extreme_div_short_veto']
        dr = dv.get('reason_counts', {}) or {}
        dr_text = ', '.join(f"{k}:{v}" for k, v in sorted(dr.items())) if dr else '-'
        print(f"  极端Divergence: eval={dv.get('evaluated', 0)} "
              f"blocked={dv.get('blocked', 0)} avg_div={dv.get('avg_div_sell', 0.0):.2f} "
              f"reasons[{dr_text}]")
    if signal_replay:
        print(f"\n  {'─' * 60}")
        print("  信号复盘 (开仓→平仓)")
        print(f"  {'─' * 60}")
        print(f"  样本数: {signal_replay.get('samples', 0)}")
        bstats = signal_replay.get('bucket_stats', {})
        if bstats:
            print("  置信度分桶:")
            for bk in ('high', 'mid', 'low', 'off', 'na'):
                if bk not in bstats:
                    continue
                bs = bstats[bk]
                print(f"    {bk:<4s} n={bs.get('n', 0):>4d} "
                      f"wr={bs.get('win_rate', 0.0):>6.2%} "
                      f"avg_r={bs.get('avg_pnl_r', 0.0):>+8.4f}")
        misses = signal_replay.get('high_conf_misses', [])
        if misses:
            print("  高置信错单(top):")
            for m in misses[:5]:
                print(f"    {str(m.get('entry_time', ''))[:19]} {m.get('direction')} "
                      f"conf={m.get('conf_raw', 0):.2f} pnl_r={m.get('pnl_r', 0):+.4f} "
                      f"R={m.get('regime', 'unknown')} ss={m.get('ss', 0):.0f} bs={m.get('bs', 0):.0f}")
    if bool(config.get('print_signal_features', True)):
        _print_signal_feature_samples(trades, limit=12)

    # ── 分市场段统计 (按 regime 拆分) ──
    pnl_actions_set = {'CLOSE_LONG', 'CLOSE_SHORT', 'LIQUIDATED', 'PARTIAL_TP', 'SPOT_SELL'}
    regime_stats = defaultdict(lambda: {'trades': 0, 'gross_profit': 0, 'gross_loss': 0, 'net_pnl': 0})
    for t in trades:
        if t['action'] not in pnl_actions_set or t.get('pnl') is None:
            continue
        regime = t.get('regime_label', 'unknown')
        pnl_val = t['pnl']
        rs = regime_stats[regime]
        rs['trades'] += 1
        rs['net_pnl'] += pnl_val
        if pnl_val > 0:
            rs['gross_profit'] += pnl_val
        else:
            rs['gross_loss'] += pnl_val

    if regime_stats:
        print(f"\n  {'─' * 60}")
        print(f"  分市场段统计 (regime)")
        print(f"  {'─' * 60}")
        print(f"  {'Regime':<20s} {'笔数':>5s} {'净PnL':>12s} {'毛利':>12s} {'毛损':>12s} {'pPF':>6s}")
        for regime in sorted(regime_stats.keys()):
            rs = regime_stats[regime]
            gl = abs(rs['gross_loss'])
            ppf = round(rs['gross_profit'] / gl, 2) if gl > 0 else 999
            print(f"  {regime:<20s} {rs['trades']:>5d} {rs['net_pnl']:>+12,.0f} "
                  f"{rs['gross_profit']:>12,.0f} {rs['gross_loss']:>12,.0f} {ppf:>6.2f}")
        summary['regime_breakdown'] = {
            r: {
                'trades': s['trades'],
                'net_pnl': round(s['net_pnl'], 2),
                'gross_profit': round(s['gross_profit'], 2),
                'gross_loss': round(s['gross_loss'], 2),
                'portfolio_pf': round(s['gross_profit'] / abs(s['gross_loss']), 2) if s['gross_loss'] != 0 else 999,
            }
            for r, s in regime_stats.items()
        }

    # ── 保存到 DB ──
    t_db = time.time()
    db_path = _default_db_path()
    run_id = save_run(
        db_path=db_path,
        run_meta=run_meta,
        summary=summary,
        daily_records=daily_records,
        trades=trades,
        version_tag=version_tag,
        experiment_notes=experiment_notes,
    )
    perf_log['5_db_save'] = time.time() - t_db
    print(f"\n💾 结果已写入 DB: {db_path} (run_id={run_id})")
    signal_log_path = _save_signal_feature_log_csv(trades, run_id, TRADE_START, TRADE_END)
    if signal_log_path:
        print(f"🧾 信号特征日志: {signal_log_path}")

    # ── 性能瓶颈日志 ──
    total_elapsed = time.time() - t0
    perf_log['total'] = total_elapsed
    print(f"\n{'─' * 60}")
    print(f"  性能分析 (瓶颈诊断)")
    print(f"{'─' * 60}")
    for phase, sec in sorted(perf_log.items()):
        pct = sec / total_elapsed * 100 if total_elapsed > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  {phase:<20} {sec:>7.2f}s  ({pct:>5.1f}%)  {bar}")
    print(f"{'─' * 60}")

    return run_id


if __name__ == '__main__':
    main()
