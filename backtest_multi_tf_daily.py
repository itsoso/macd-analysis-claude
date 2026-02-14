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
        # <12h 空单抑制
        'use_short_suppress': _LIVE_DEFAULT.use_short_suppress,
        'short_suppress_ss_min': _LIVE_DEFAULT.short_suppress_ss_min,
        # SPOT_SELL 高分确认过滤
        'use_spot_sell_confirm': _LIVE_DEFAULT.use_spot_sell_confirm,
        'spot_sell_confirm_ss': _LIVE_DEFAULT.spot_sell_confirm_ss,
        'spot_sell_confirm_min': _LIVE_DEFAULT.spot_sell_confirm_min,
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
        'use_trend_enhance': True,
        'trend_floor_ratio': 0.50,  # 上调至0.50: 趋势中持有更多ETH，减少过度卖出
        'min_base_eth_ratio': 0.0,  # 禁用最低持仓限制, 允许灵活调仓
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

    return {
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
    }


def main(trade_start=None, trade_end=None, version_tag=None):
    t0 = time.time()
    perf_log = {}  # 性能日志: 阶段 -> 耗时(秒)

    # CLI 参数解析
    parser = argparse.ArgumentParser(description='多周期联合决策回测')
    parser.add_argument('--start', type=str, default=None, help='回测起始日 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='回测结束日 (YYYY-MM-DD)')
    parser.add_argument('--tag', type=str, default=None, help='策略版本标签')
    parser.add_argument('--fast', action='store_true', default=False,
                        help='[实验性] P0向量化信号计算, 结果与原版存在近似偏差(±1%%), 不建议作为正式策略结论依据')
    args, _ = parser.parse_known_args()

    TRADE_START = args.start or trade_start or DEFAULT_TRADE_START
    TRADE_END = args.end or trade_end or DEFAULT_TRADE_END
    if args.tag:
        version_tag = args.tag

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
    print(f"  趋势保护v3: {'ON' if DEFAULT_CONFIG.get('use_trend_enhance') else 'OFF'}"
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

    initial_capital = result.get('initial_total', 200000)
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
    profit_factor = round(
        abs(sum(t['pnl'] for t in wins)) / abs(sum(t['pnl'] for t in losses)), 2
    ) if losses and sum(t['pnl'] for t in losses) != 0 else 999

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
        'total_fees': fees.get('total_fees', 0),
        'total_slippage': fees.get('slippage_cost', 0),
        'total_costs': fees.get('total_costs', 0),
        'funding_paid': fees.get('funding_paid', 0),
        'funding_received': fees.get('funding_received', 0),
        'net_funding': fees.get('net_funding', 0),
        'fee_drag_pct': fees.get('fee_drag_pct', 0),
        'liquidations': result.get('liquidations', 0),
    }

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
    print(f"  盈亏比:      {profit_factor:.2f}")
    print(f"  总费用:      ${fees.get('total_costs', 0):,.2f}")
    print(f"  逐日记录:    {len(daily_records)} 天")

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
    )
    perf_log['5_db_save'] = time.time() - t_db
    print(f"\n💾 结果已写入 DB: {db_path} (run_id={run_id})")

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
