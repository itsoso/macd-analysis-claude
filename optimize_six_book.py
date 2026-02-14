"""
六书融合多时间框架止盈止损优化器

在五书优化器(optimize_sl_tp.py)基础上升级:
1. 新增KDJ作为第6维信号(基于《随机指标KDJ：波段操作精解》)
2. 支持多种融合模式: c6_veto_4 / kdj_weighted / kdj_timing
3. KDJ特有参数优化: KDJ权重、否决阈值、确认强度
4. 目标: 超越五书最优 α=+86.69% (精选分段TP@20%平30%+最佳SL/TP)

在12个时间周期上全面测试:
10min, 15min, 30min, 1h, 2h, 3h, 4h, 6h, 8h, 12h, 16h, 24h

真实回测条件:
- 币安 ETH/USDT 真实K线数据
- Taker手续费 0.05%, Maker 0.02%
- 资金费率每8小时 ±0.01%
- 逐仓模式, 5%维持保证金率强平
- 滑点模拟
"""

import pandas as pd
import numpy as np
import json
import sys
import os
import time
import re
from bisect import bisect_right
from datetime import datetime
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binance_fetcher import fetch_binance_klines
from indicators import add_all_indicators
from strategy_enhanced import analyze_signals_enhanced, get_signal_at, DEFAULT_SIG
from strategy_futures import FuturesEngine
from strategy_futures_v4 import _calc_top_score, _calc_bottom_score
from ma_indicators import add_moving_averages, compute_ma_signals
from candlestick_patterns import compute_candlestick_scores
from bollinger_strategy import compute_bollinger_scores
from volume_price_strategy import compute_volume_price_scores
from kdj_strategy import compute_kdj_scores
from signal_core import (
    compute_signals_six as _compute_signals_six_core,
    calc_fusion_score_six as _calc_fusion_score_six_core,
)
from multi_tf_consensus import fuse_tf_scores


# ======================================================
#   多周期数据获取
# ======================================================
ALL_TIMEFRAMES = ['10m', '15m', '30m', '1h', '2h', '3h', '4h', '6h', '8h', '12h', '16h', '24h']

def fetch_multi_tf_data(timeframes=None, days=60):
    """获取多时间周期数据"""
    if timeframes is None:
        timeframes = ALL_TIMEFRAMES
    data = {}
    for tf in timeframes:
        print(f"\n--- 获取 {tf} 数据 ---")
        try:
            df = fetch_binance_klines("ETHUSDT", interval=tf, days=days)
            if df is not None and len(df) > 50:
                df = add_all_indicators(df)
                add_moving_averages(df, timeframe=tf)
                data[tf] = df
                print(f"  {tf}: {len(df)} 条K线")
            else:
                print(f"  {tf}: 数据不足, 跳过")
        except Exception as e:
            print(f"  {tf}: 获取失败 - {e}")
    return data


def compute_signals_six(df, tf, data_all, max_bars=0):
    """兼容包装: 委托给 signal_core.compute_signals_six。"""
    return _compute_signals_six_core(df, tf, data_all, max_bars=max_bars)


# ======================================================
#   六维融合评分(支持多模式)
# ======================================================
def calc_fusion_score_six(signals, df, idx, dt, config):
    """兼容包装: 委托给 signal_core.calc_fusion_score_six。"""
    return _calc_fusion_score_six_core(signals, df, idx, dt, config)


_PNL_RE = re.compile(r"PnL=([+-]?\d+(?:\.\d+)?)")
_LIQ_LOSS_RE = re.compile(r"损失([+-]?\d+(?:\.\d+)?)")


def _extract_realized_pnl_from_trade(trade):
    """
    从成交记录中提取平仓已实现盈亏(USDT)。
    返回 None 表示该成交不计入连亏统计。
    """
    action = str(trade.get('action', ''))
    reason = str(trade.get('reason', ''))
    pnl_field = trade.get('pnl', None)
    if action in ('CLOSE_LONG', 'CLOSE_SHORT', 'LIQUIDATED') and pnl_field is not None:
        try:
            return float(pnl_field)
        except (TypeError, ValueError):
            pass
    if action in ('CLOSE_LONG', 'CLOSE_SHORT'):
        m = _PNL_RE.search(reason)
        if m:
            return float(m.group(1))
        return None
    if action == 'LIQUIDATED':
        m = _LIQ_LOSS_RE.search(reason)
        if m:
            return -abs(float(m.group(1)))
        return -abs(float(trade.get('fee', 0.0) or 0.0))
    return None


def _init_protection_state(config, initial_equity):
    enabled = bool(config.get('use_protections', False))
    return {
        'enabled': enabled,
        'daily_date': None,
        'daily_start_equity': float(initial_equity),
        'daily_locked': False,
        'daily_pnl_pct': 0.0,
        'equity_peak': float(initial_equity),
        'drawdown_from_peak_pct': 0.0,
        'global_halt': False,
        'global_halt_closed': False,  # 停机后是否已执行平仓
        'entry_block_until_idx': -1,
        'loss_streak': 0,
        'stats': {
            'daily_lock_count': 0,
            'streak_lock_count': 0,
            'global_halt_triggered': 0,
            'global_halt_recovered': 0,
            'blocked_bars': 0,
            'max_loss_streak': 0,
            'blocked_reason_counts': {},
        },
    }


def _update_protection_risk_state(state, dt, equity, idx, config):
    if not state.get('enabled', False):
        return state

    day = pd.Timestamp(dt).date().isoformat()
    if state.get('daily_date') != day:
        state['daily_date'] = day
        state['daily_start_equity'] = float(equity)
        state['daily_locked'] = False

    daily_start = float(state.get('daily_start_equity', equity) or equity)
    if daily_start > 0:
        state['daily_pnl_pct'] = (float(equity) - daily_start) / daily_start
    else:
        state['daily_pnl_pct'] = 0.0

    daily_limit = float(config.get('prot_daily_loss_limit_pct', 0.03))
    if not state.get('daily_locked', False) and state['daily_pnl_pct'] <= -daily_limit:
        state['daily_locked'] = True
        state['stats']['daily_lock_count'] += 1

    state['equity_peak'] = max(float(state.get('equity_peak', equity) or equity), float(equity))
    peak = float(state['equity_peak']) if float(state['equity_peak']) > 0 else float(equity)
    if peak > 0:
        state['drawdown_from_peak_pct'] = (float(equity) - peak) / peak
    else:
        state['drawdown_from_peak_pct'] = 0.0

    global_dd_limit = float(config.get('prot_global_dd_limit_pct', 0.15))
    if (not state.get('global_halt', False)) and state['drawdown_from_peak_pct'] <= -global_dd_limit:
        state['global_halt'] = True
        state['stats']['global_halt_triggered'] += 1

    # ── global_halt 恢复机制: 回撤收窄到恢复阈值时解除停机 ──
    # 避免一次大回撤永久杀死策略
    if state.get('global_halt', False):
        recovery_threshold = float(config.get('prot_global_halt_recovery_pct', 0.06))
        if state['drawdown_from_peak_pct'] > -recovery_threshold:
            state['global_halt'] = False
            state['global_halt_closed'] = False  # 重置平仓标记, 下次停机可再执行平仓
            state['stats']['global_halt_recovered'] = int(state['stats'].get('global_halt_recovered', 0)) + 1

    return state


def _apply_loss_streak_protection(state, pnl, idx, config):
    if not state.get('enabled', False) or pnl is None:
        return state

    if float(pnl) < 0:
        state['loss_streak'] = int(state.get('loss_streak', 0)) + 1
    else:
        state['loss_streak'] = 0

    state['stats']['max_loss_streak'] = max(
        int(state['stats'].get('max_loss_streak', 0)),
        int(state['loss_streak']),
    )

    streak_limit = int(config.get('prot_loss_streak_limit', 3))
    cooldown_bars = int(config.get('prot_loss_streak_cooldown_bars', 24))
    if state['loss_streak'] >= streak_limit:
        state['entry_block_until_idx'] = max(
            int(state.get('entry_block_until_idx', -1)),
            int(idx) + cooldown_bars,
        )
        state['stats']['streak_lock_count'] += 1
        state['loss_streak'] = 0

    return state


def _protection_entry_allowed(state, idx):
    if not state.get('enabled', False):
        return True, None
    if state.get('global_halt', False):
        return False, 'global_halt'
    if state.get('daily_locked', False):
        return False, 'daily_loss_limit'
    if int(idx) <= int(state.get('entry_block_until_idx', -1)):
        return False, 'loss_streak_cooldown'
    return True, None


def _build_regime_precomputed(primary_df, config):
    """预计算 regime 所需滚动统计，减少逐 bar 切片开销。"""
    if not config.get('use_regime_aware', False):
        return None

    lookback = max(2, int(config.get('regime_lookback_bars', 48)))
    atr_bars = max(2, int(config.get('regime_atr_bars', 14)))
    close = primary_df['close']
    high = primary_df['high']
    low = primary_df['low']

    ret = close.pct_change()
    vol_window = max(2, lookback - 1)
    regime_vol = ret.rolling(vol_window, min_periods=8).std()

    ma_fast = close.rolling(12, min_periods=1).mean()
    ma_slow = close.rolling(48, min_periods=1).mean()

    close_prev = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close_prev).abs(),
            (low - close_prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_pct = (tr.rolling(atr_bars + 1, min_periods=3).mean() / close).replace([np.inf, -np.inf], np.nan)

    return {
        'regime_vol': regime_vol,
        'ma_fast': ma_fast,
        'ma_slow': ma_slow,
        'atr_pct': atr_pct,
    }


def _compute_regime_controls(primary_df, idx, config, precomputed=None):
    """
    根据当前市场状态动态调整阈值与风险参数。
    仅使用 <= idx 的历史数据, 防止未来函数。
    """
    base = {
        'sell_threshold': float(config.get('sell_threshold', 18)),
        'buy_threshold': float(config.get('buy_threshold', 25)),
        'short_threshold': float(config.get('short_threshold', 25)),
        'long_threshold': float(config.get('long_threshold', 40)),
        'close_short_bs': float(config.get('close_short_bs', 40)),
        'close_long_ss': float(config.get('close_long_ss', 40)),
        'lev': int(config.get('lev', 5)),
        'margin_use': float(config.get('margin_use', 0.70)),
        'regime_label': 'static',
    }
    if not config.get('use_regime_aware', False):
        return base

    lookback = int(config.get('regime_lookback_bars', 48))
    atr_bars = int(config.get('regime_atr_bars', 14))
    if idx < max(20, lookback):
        return base

    price = float(primary_df['close'].iloc[idx]) if idx < len(primary_df) else float(primary_df['close'].iloc[-1])
    if price <= 0:
        return base

    if precomputed is not None:
        vol_s = precomputed.get('regime_vol')
        ma_fast_s = precomputed.get('ma_fast')
        ma_slow_s = precomputed.get('ma_slow')
        atr_pct_s = precomputed.get('atr_pct')
        vol = float(vol_s.iloc[idx]) if vol_s is not None and idx < len(vol_s) else float('nan')
        ma_fast = float(ma_fast_s.iloc[idx]) if ma_fast_s is not None and idx < len(ma_fast_s) else float('nan')
        ma_slow = float(ma_slow_s.iloc[idx]) if ma_slow_s is not None and idx < len(ma_slow_s) else float('nan')
        atr_pct = float(atr_pct_s.iloc[idx]) if atr_pct_s is not None and idx < len(atr_pct_s) else 0.0
        if not np.isfinite(vol):
            return base
        if not np.isfinite(ma_fast) or not np.isfinite(ma_slow):
            return base
        if not np.isfinite(atr_pct):
            atr_pct = 0.0
    else:
        start = max(0, idx - lookback + 1)
        win = primary_df.iloc[start:idx + 1]
        if len(win) < 20:
            return base
        close = win['close']
        returns = close.pct_change().dropna()
        if len(returns) < 8:
            return base
        vol = float(returns.std())
        ma_fast = float(close.tail(min(12, len(close))).mean())
        ma_slow = float(close.tail(min(48, len(close))).mean())

        atr_start = max(0, idx - atr_bars)
        h = primary_df['high'].iloc[atr_start:idx + 1]
        l = primary_df['low'].iloc[atr_start:idx + 1]
        c_prev = primary_df['close'].shift(1).iloc[atr_start:idx + 1]
        min_len = min(len(h), len(l), len(c_prev))
        atr_pct = 0.0
        if min_len > 2:
            tr = pd.Series(
                [
                    max(hi - lo, abs(hi - cp), abs(lo - cp))
                    for hi, lo, cp in zip(h.tail(min_len), l.tail(min_len), c_prev.tail(min_len))
                ]
            )
            atr_pct = float(tr.mean() / price) if price > 0 else 0.0

    trend = (ma_fast - ma_slow) / price
    abs_trend = abs(trend)

    vol_high = float(config.get('regime_vol_high', 0.020))
    vol_low = float(config.get('regime_vol_low', 0.007))
    trend_strong = float(config.get('regime_trend_strong', 0.015))
    trend_weak = float(config.get('regime_trend_weak', 0.006))
    atr_high = float(config.get('regime_atr_high', 0.018))

    high_vol = vol >= vol_high or atr_pct >= atr_high
    low_vol = vol <= vol_low and atr_pct <= atr_high * 0.7
    strong_trend = abs_trend >= trend_strong
    weak_trend = abs_trend <= trend_weak

    short_mult = 1.0
    long_mult = 1.0
    close_mult = 1.0
    risk_mult = 1.0
    regime_label = 'neutral'

    # 波动大且趋势弱: 提高入场门槛 + 降杠杆 + 提前退出
    if high_vol and weak_trend:
        short_mult *= 1.22
        long_mult *= 1.22
        close_mult *= 0.90
        risk_mult *= 0.58
        regime_label = 'high_vol_choppy'
    elif high_vol:
        short_mult *= 1.12
        long_mult *= 1.12
        close_mult *= 0.95
        risk_mult *= 0.75
        regime_label = 'high_vol'
    elif low_vol and strong_trend:
        short_mult *= 0.95
        long_mult *= 0.95
        close_mult *= 1.05
        regime_label = 'low_vol_trend'
    elif strong_trend:
        short_mult *= 0.98
        long_mult *= 0.98
        regime_label = 'trend'

    # 趋势方向倾斜: 顺势更易进场, 逆势更难
    if trend >= trend_weak:
        long_mult *= 0.88
        short_mult *= 1.10
    elif trend <= -trend_weak:
        short_mult *= 0.88
        long_mult *= 1.10

    def _clip(v, base_v, lo=0.75, hi=1.45):
        return float(np.clip(v, base_v * lo, base_v * hi))

    short_threshold = _clip(base['short_threshold'] * short_mult, base['short_threshold'])
    long_threshold = _clip(base['long_threshold'] * long_mult, base['long_threshold'])
    sell_threshold = _clip(base['sell_threshold'] * short_mult, base['sell_threshold'], lo=0.80, hi=1.40)
    buy_threshold = _clip(base['buy_threshold'] * long_mult, base['buy_threshold'], lo=0.80, hi=1.40)
    close_short_bs = _clip(base['close_short_bs'] * close_mult, base['close_short_bs'], lo=0.75, hi=1.25)
    close_long_ss = _clip(base['close_long_ss'] * close_mult, base['close_long_ss'], lo=0.75, hi=1.25)

    lev = int(max(1, min(base['lev'], round(base['lev'] * risk_mult))))
    margin_use = float(np.clip(base['margin_use'] * risk_mult, 0.10, base['margin_use']))

    return {
        'sell_threshold': sell_threshold,
        'buy_threshold': buy_threshold,
        'short_threshold': short_threshold,
        'long_threshold': long_threshold,
        'close_short_bs': close_short_bs,
        'close_long_ss': close_long_ss,
        'lev': lev,
        'margin_use': margin_use,
        'regime_label': regime_label,
        'regime_metrics': {
            'vol': vol,
            'atr_pct': atr_pct,
            'trend': trend,
            'high_vol': high_vol,
            'strong_trend': strong_trend,
        },
    }


def _build_microstructure_features(primary_df, config):
    """
    预计算微结构特征（资金费率/基差/持仓代理/主动买卖失衡）。
    仅使用当前K线中可得字段；缺失时自动回退到中性代理，确保兼容旧数据。
    """
    lookback = max(8, int(config.get('micro_lookback_bars', 48)))
    minp = max(5, lookback // 3)

    close = pd.to_numeric(primary_df.get('close', pd.Series(0.0, index=primary_df.index)), errors='coerce')
    volume = pd.to_numeric(primary_df.get('volume', pd.Series(0.0, index=primary_df.index)), errors='coerce')
    quote_volume = primary_df.get('quote_volume')
    if quote_volume is None:
        quote_volume = close * volume
    quote_volume = pd.to_numeric(quote_volume, errors='coerce').replace(0, np.nan)

    taker_buy_quote = primary_df.get('taker_buy_quote')
    if taker_buy_quote is None:
        taker_buy_quote = quote_volume * 0.5
    taker_buy_quote = pd.to_numeric(taker_buy_quote, errors='coerce')

    # 主动买卖失衡: [-1, 1]
    taker_buy_ratio = (taker_buy_quote / quote_volume).clip(0.0, 1.0).fillna(0.5)
    taker_imbalance = ((taker_buy_ratio - 0.5) * 2.0).clip(-1.0, 1.0)

    # 成交参与度（作为 OI 缺失时的代理）
    qv_median = quote_volume.rolling(lookback, min_periods=minp).median()
    participation = (quote_volume / qv_median.replace(0, np.nan)).clip(0.0, 6.0).fillna(1.0)

    # 基差: 优先使用外部提供字段，否则用价格相对滚动VWAP偏离作为代理
    if 'basis' in primary_df.columns:
        basis_raw = pd.to_numeric(primary_df['basis'], errors='coerce')
    else:
        pv = (close * quote_volume).rolling(lookback, min_periods=minp).sum()
        qv = quote_volume.rolling(lookback, min_periods=minp).sum().replace(0, np.nan)
        vwap = pv / qv
        basis_raw = (close - vwap) / vwap.replace(0, np.nan)
    basis_mean = basis_raw.rolling(lookback, min_periods=minp).mean()
    basis_std = basis_raw.rolling(lookback, min_periods=minp).std(ddof=0).replace(0, np.nan)
    basis_z = ((basis_raw - basis_mean) / basis_std).clip(-5.0, 5.0).fillna(0.0)

    # 持仓量: 优先真实 OI；缺失时用 quote_volume 作为资金参与代理
    oi_series = None
    for col in ('open_interest', 'oi'):
        if col in primary_df.columns:
            oi_series = pd.to_numeric(primary_df[col], errors='coerce')
            break
    if oi_series is None:
        oi_series = quote_volume
    oi_mean = oi_series.rolling(lookback, min_periods=minp).mean()
    oi_std = oi_series.rolling(lookback, min_periods=minp).std(ddof=0).replace(0, np.nan)
    oi_z = ((oi_series - oi_mean) / oi_std).clip(-5.0, 5.0).fillna(0.0)

    # 资金费率: 优先真实 funding_rate；缺失时用基差映射为资金费代理
    if 'funding_rate' in primary_df.columns:
        funding_rate = pd.to_numeric(primary_df['funding_rate'], errors='coerce').fillna(0.0)
    else:
        funding_mult = float(config.get('micro_funding_proxy_mult', 0.35))
        funding_rate = (basis_raw * funding_mult).clip(-0.0020, 0.0020).fillna(0.0)
    fr_mean = funding_rate.rolling(lookback, min_periods=minp).mean()
    fr_std = funding_rate.rolling(lookback, min_periods=minp).std(ddof=0).replace(0, np.nan)
    funding_z = ((funding_rate - fr_mean) / fr_std).clip(-5.0, 5.0).fillna(0.0)

    return pd.DataFrame(
        {
            'taker_imbalance': taker_imbalance,
            'participation': participation,
            'basis_z': basis_z,
            'oi_z': oi_z,
            'funding_rate': funding_rate,
            'funding_z': funding_z,
        },
        index=primary_df.index,
    )


def _compute_microstructure_state(micro_df, idx, config):
    """
    计算当前bar的微结构方向偏置:
      - long_score / short_score: 趋势延续或拥挤反转共识得分
      - mode_hint: trend / reversion / neutral
    """
    neutral = {
        'valid': False,
        'long_score': 0.0,
        'short_score': 0.0,
        'mode_hint': 'neutral',
        'basis_z': 0.0,
        'oi_z': 0.0,
        'funding_rate': 0.0,
        'taker_imbalance': 0.0,
        'participation': 1.0,
    }
    if micro_df is None or idx <= 0 or idx >= len(micro_df):
        return neutral

    row = micro_df.iloc[idx - 1]  # 决策在当前bar open执行，只能用上一bar完成值
    vals = {
        'basis_z': float(row.get('basis_z', 0.0) or 0.0),
        'oi_z': float(row.get('oi_z', 0.0) or 0.0),
        'funding_rate': float(row.get('funding_rate', 0.0) or 0.0),
        'taker_imbalance': float(row.get('taker_imbalance', 0.0) or 0.0),
        'participation': float(row.get('participation', 1.0) or 1.0),
    }

    imb_thr = float(config.get('micro_imbalance_threshold', 0.08))
    oi_thr = float(config.get('micro_oi_trend_z', 0.8))
    basis_thr = float(config.get('micro_basis_extreme_z', 1.2))
    funding_thr = float(config.get('micro_funding_extreme', 0.0006))
    part_thr = float(config.get('micro_participation_trend', 1.15))

    long_score = 0.0
    short_score = 0.0

    # 主动成交失衡（延续）
    if vals['taker_imbalance'] >= imb_thr:
        long_score += 1.0
    elif vals['taker_imbalance'] <= -imb_thr:
        short_score += 1.0

    # OI/资金参与放大（延续）
    if vals['oi_z'] >= oi_thr and vals['participation'] >= part_thr:
        if vals['taker_imbalance'] >= 0:
            long_score += 1.0
        if vals['taker_imbalance'] <= 0:
            short_score += 1.0

    # 基差 + 资金费拥挤（反转）
    if vals['basis_z'] >= basis_thr and vals['funding_rate'] >= funding_thr:
        short_score += 1.5
    if vals['basis_z'] <= -basis_thr and vals['funding_rate'] <= -funding_thr:
        long_score += 1.5

    mode_hint = 'neutral'
    if abs(vals['basis_z']) >= basis_thr * 1.3 and vals['participation'] < 1.0:
        mode_hint = 'reversion'
    elif vals['oi_z'] >= oi_thr and abs(vals['taker_imbalance']) >= imb_thr:
        mode_hint = 'trend'

    return {
        'valid': True,
        'long_score': float(long_score),
        'short_score': float(short_score),
        'mode_hint': mode_hint,
        **vals,
    }


def _apply_microstructure_overlay(ss, bs, micro_state, config):
    """将微结构偏置叠加到融合分数与开仓限制。"""
    overlay = {
        'block_long': False,
        'block_short': False,
        'margin_mult': 1.0,
    }
    if not config.get('use_microstructure', False):
        return ss, bs, overlay
    if not isinstance(micro_state, dict) or not micro_state.get('valid', False):
        return ss, bs, overlay

    diff = float(micro_state.get('long_score', 0.0)) - float(micro_state.get('short_score', 0.0))
    diff = float(np.clip(diff, -4.0, 4.0))
    boost = float(config.get('micro_score_boost', 0.08))
    damp = float(config.get('micro_score_dampen', 0.10))
    margin_step = float(config.get('micro_margin_mult_step', 0.06))

    if diff > 0:
        bs *= 1.0 + boost * diff
        ss *= max(0.20, 1.0 - damp * diff)
        overlay['margin_mult'] *= 1.0 + margin_step * diff
        if diff >= 2.5:
            overlay['block_short'] = True
    elif diff < 0:
        d = abs(diff)
        ss *= 1.0 + boost * d
        bs *= max(0.20, 1.0 - damp * d)
        overlay['margin_mult'] *= 1.0 + margin_step * d
        if d >= 2.5:
            overlay['block_long'] = True

    # 拥挤状态降风险（即使方向一致也避免过热追单）
    if abs(float(micro_state.get('basis_z', 0.0))) >= float(config.get('micro_basis_crowded_z', 2.2)):
        overlay['margin_mult'] *= 0.80

    overlay['margin_mult'] = float(np.clip(overlay['margin_mult'], 0.50, 1.35))
    return float(ss), float(bs), overlay


def _resolve_engine_mode(regime_label, micro_state, config):
    """
    双引擎切换:
      - trend: 趋势跟随
      - reversion: 均值回归/反转
    """
    if not config.get('use_dual_engine', False):
        return 'single'

    mode = 'trend' if regime_label in ('trend', 'low_vol_trend') else 'reversion'
    if config.get('micro_mode_override', True) and isinstance(micro_state, dict):
        hint = micro_state.get('mode_hint', 'neutral')
        if hint in ('trend', 'reversion'):
            mode = hint
    return mode


def _build_realized_vol_series(primary_df, primary_tf, config):
    """构建年化波动率序列（用于波动目标仓位）。"""
    if not config.get('use_vol_target', False):
        return None

    tf_hours = {
        '10m': 1 / 6, '15m': 0.25, '30m': 0.5, '1h': 1, '2h': 2, '3h': 3,
        '4h': 4, '6h': 6, '8h': 8, '12h': 12, '16h': 16, '24h': 24,
    }
    tf_h = float(tf_hours.get(primary_tf, 1.0))
    bars_per_year = max(1, int(round(24.0 * 365.0 / tf_h)))
    lookback = max(10, int(config.get('vol_target_lookback_bars', 48)))
    minp = max(6, lookback // 3)
    ret = primary_df['close'].pct_change()
    vol_ann = ret.rolling(lookback, min_periods=minp).std(ddof=0) * np.sqrt(bars_per_year)
    return vol_ann.replace([np.inf, -np.inf], np.nan)


def _compute_vol_target_scale(vol_ann, idx, config):
    """根据实时年化波动率计算仓位缩放。"""
    if vol_ann is None or idx <= 0 or idx > len(vol_ann):
        return 1.0, None, False

    rv = float(vol_ann.iloc[idx - 1])  # 仅使用上一bar完成数据
    if not np.isfinite(rv) or rv <= 0:
        return 1.0, rv, False

    target = float(config.get('vol_target_annual', 0.85))
    min_s = float(config.get('vol_target_min_scale', 0.45))
    max_s = float(config.get('vol_target_max_scale', 1.35))
    scale = float(np.clip(target / rv, min_s, max_s))
    return scale, rv, True


# ======================================================
#   通用回测引擎(六书版)
# ======================================================
def _run_strategy_core(
    primary_df,
    config,
    primary_tf,
    trade_days,
    score_provider,
    trade_start_dt=None,
    trade_end_dt=None,
    tf_score_map=None,
    decision_tfs=None,
):
    """统一交易执行循环: 单TF/多TF共用。
    tf_score_map, decision_tfs: 可选, 供趋势做多直接读取大周期原始信号。
    """
    eng = FuturesEngine(config.get('name', 'opt'), max_leverage=config.get('max_lev', 5))

    def _norm_ts(ts):
        if ts is None:
            return None
        v = pd.Timestamp(ts)
        if v.tz is not None:
            v = v.tz_localize(None)
        return v

    start_dt = _norm_ts(trade_start_dt)
    end_dt = _norm_ts(trade_end_dt)
    # 未显式指定起始时间时，按 trade_days 截窗。
    if start_dt is None and trade_days and trade_days > 0:
        start_dt = primary_df.index[-1] - pd.Timedelta(days=trade_days)

    index_values = primary_df.index
    close_series = primary_df['close']
    open_series = primary_df['open']
    high_series = primary_df['high']
    low_series = primary_df['low']
    close_prices = close_series.to_numpy(dtype=float, copy=False)
    open_prices = open_series.to_numpy(dtype=float, copy=False)

    init_idx = 0
    if start_dt:
        init_idx = index_values.searchsorted(start_dt)
        if init_idx >= len(primary_df):
            init_idx = 0
    eng.spot_eth = eng.initial_eth_value / float(close_prices[init_idx])
    eng.spot_cost_basis = eng.initial_eth_value  # 初始ETH成本基准 = 初始分配金额

    eng.max_single_margin = eng.initial_total * config.get('single_pct', 0.20)
    eng.max_margin_total = eng.initial_total * config.get('total_pct', 0.50)
    eng.max_lifetime_margin = eng.initial_total * config.get('lifetime_pct', 5.0)

    sell_threshold = config.get('sell_threshold', 18)
    buy_threshold = config.get('buy_threshold', 25)
    short_threshold = config.get('short_threshold', 25)
    long_threshold = config.get('long_threshold', 40)
    sell_pct = config.get('sell_pct', 0.55)
    margin_use = config.get('margin_use', 0.70)
    lev = config.get('lev', 5)
    cooldown = config.get('cooldown', 4)
    spot_cooldown = config.get('spot_cooldown', 12)

    short_sl = config.get('short_sl', -0.30)
    short_tp = config.get('short_tp', 0.80)
    short_trail = config.get('short_trail', 0.25)
    short_max_hold = config.get('short_max_hold', 72)
    long_sl = config.get('long_sl', -0.15)
    long_tp = config.get('long_tp', 0.50)
    long_trail = config.get('long_trail', 0.20)
    long_max_hold = config.get('long_max_hold', 72)
    trail_pullback = config.get('trail_pullback', 0.60)

    use_dynamic_tp = config.get('use_dynamic_tp', False)
    use_partial_tp = config.get('use_partial_tp', False)
    partial_tp_1 = config.get('partial_tp_1', 0.30)
    partial_tp_1_pct = config.get('partial_tp_1_pct', 0.40)

    use_atr_sl = config.get('use_atr_sl', False)
    atr_sl_mult = config.get('atr_sl_mult', 3.0)

    # 二段止盈
    use_partial_tp_2 = config.get('use_partial_tp_2', False)
    partial_tp_2 = config.get('partial_tp_2', 0.50)
    partial_tp_2_pct = config.get('partial_tp_2_pct', 0.30)

    short_cd = 0; long_cd = 0; spot_cd = 0
    short_bars = 0; long_bars = 0
    short_max_pnl = 0; long_max_pnl = 0
    short_partial_done = False; long_partial_done = False
    short_partial2_done = False; long_partial2_done = False

    # 趋势持续状态 (跨bar持久, 带滞后)
    _trend_up_active = False  # 上升趋势一旦激活, 直到明确反转才关闭

    warmup = max(60, int(len(primary_df) * 0.05))

    tf_hours = {
        '10m': 1/6, '15m': 0.25, '30m': 0.5, '1h': 1, '2h': 2, '3h': 3,
        '4h': 4, '6h': 6, '8h': 8, '12h': 12, '16h': 16, '24h': 24,
    }
    bars_per_8h = max(1, int(8 / tf_hours.get(primary_tf, 1)))

    record_interval = max(1, len(primary_df) // 500)

    # ── 延迟信号: 上一根K线产生的信号, 记录在 pending_* 中,
    #    在下一根K线的 open 价执行。消除 same-bar execution bias。
    pending_ss = 0.0
    pending_bs = 0.0
    has_pending_signal = False
    prot_state = _init_protection_state(config, eng.total_value(primary_df['close'].iloc[init_idx]))
    trade_cursor = 0
    micro_df = _build_microstructure_features(primary_df, config) if config.get('use_microstructure', False) else None
    vol_ann = _build_realized_vol_series(primary_df, primary_tf, config)
    regime_precomputed = _build_regime_precomputed(primary_df, config)

    atr_pct_series = None
    if use_atr_sl:
        close_prev = close_series.shift(1)
        tr = pd.concat(
            [
                (high_series - low_series).abs(),
                (high_series - close_prev).abs(),
                (low_series - close_prev).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_pct_series = (tr.rolling(14, min_periods=3).mean() / close_series).replace([np.inf, -np.inf], np.nan)

    dyn_vol_series = None
    if use_dynamic_tp:
        dyn_vol_series = close_series.pct_change().rolling(19, min_periods=6).std().replace([np.inf, -np.inf], np.nan)

    trend_ema10 = None
    trend_ema30 = None
    if bool(config.get('use_trend_enhance', False)):
        trend_ema10 = close_series.ewm(span=10, adjust=False).mean()
        trend_ema30 = close_series.ewm(span=30, adjust=False).mean()

    for idx in range(warmup, len(primary_df)):
        dt = index_values[idx]
        if end_dt is not None and dt > end_dt:
            break
        price = float(close_prices[idx])
        # 交易执行价 = 当前 bar 的 open (模拟"收到上根信号后在下根开盘执行")
        exec_price = float(open_prices[idx])

        # 月度重置累计保证金额度, 避免长周期回测耗尽
        eng.maybe_reset_lifetime_monthly(dt)

        if start_dt and dt < start_dt:
            if idx % record_interval == 0:
                eng.record_history(dt, price)
            # 预热阶段也要生成信号供下一根使用(但不交易)
            pending_ss, pending_bs = score_provider(idx, dt, price)
            has_pending_signal = True
            continue

        # 记录强平前的仓位状态 (强平用实时价格, 不受信号延迟影响)
        had_short_before_liq = eng.futures_short is not None
        had_long_before_liq = eng.futures_long is not None
        eng.check_liquidation(price, dt)
        # 强平后设置冷却期, 防止同bar立即重新开仓
        if had_short_before_liq and eng.futures_short is None:
            short_cd = max(short_cd, cooldown * 6)
            short_bars = 0; short_max_pnl = 0
        if had_long_before_liq and eng.futures_long is None:
            long_cd = max(long_cd, cooldown * 6)
            long_bars = 0; long_max_pnl = 0
        short_just_opened = False; long_just_opened = False

        # 资金费率
        eng.funding_counter += 1
        if eng.funding_counter % bars_per_8h == 0:
            is_neg = (eng.funding_counter * 7 + 3) % 10 < 3
            rate = FuturesEngine.FUNDING_RATE if not is_neg else -FuturesEngine.FUNDING_RATE * 0.5
            if eng.futures_long:
                cost = eng.futures_long.quantity * price * rate
                eng.usdt -= cost
                if cost > 0: eng.total_funding_paid += cost
                else: eng.total_funding_received += abs(cost)
            if eng.futures_short:
                income = eng.futures_short.quantity * price * rate
                eng.usdt += income
                if income > 0: eng.total_funding_received += income
                else: eng.total_funding_paid += abs(income)

        if short_cd > 0: short_cd -= 1
        if long_cd > 0: long_cd -= 1
        if spot_cd > 0: spot_cd -= 1

        # 组合层风控: 日内亏损预算 + 全局回撤停机 + 连亏冷却
        _update_protection_risk_state(prot_state, dt, eng.total_value(price), idx, config)
        if (
            prot_state.get('enabled', False)
            and prot_state.get('global_halt', False)
            and not prot_state.get('global_halt_closed', False)
            and config.get('prot_close_on_global_halt', True)
        ):
            if eng.futures_short:
                eng.close_short(exec_price, dt, "全局停机平空")
                short_bars = 0
            if eng.futures_long:
                eng.close_long(exec_price, dt, "全局停机平多")
                long_bars = 0
            prot_state['global_halt_closed'] = True

        can_open_risk, blocked_reason = _protection_entry_allowed(prot_state, idx)
        if prot_state.get('enabled', False) and not can_open_risk:
            prot_state['stats']['blocked_bars'] += 1
            brc = prot_state['stats'].setdefault('blocked_reason_counts', {})
            brc[blocked_reason] = int(brc.get(blocked_reason, 0)) + 1

        # ── 使用上一根 bar 的信号做决策 (消除 same-bar bias) ──
        # 如果还没有 pending 信号(第一根交易bar), 使用前一根 bar 的信号
        if not has_pending_signal and idx > warmup:
            pending_ss, pending_bs = score_provider(idx - 1,
                                                     index_values[idx - 1],
                                                     float(close_prices[idx - 1]))
            has_pending_signal = True

        ss, bs = (pending_ss, pending_bs) if has_pending_signal else (0.0, 0.0)

        # ATR自适应止损
        actual_short_sl = short_sl
        actual_long_sl = long_sl
        if atr_pct_series is not None and idx >= 20:
            atr_pct = float(atr_pct_series.iloc[idx - 1]) if idx - 1 < len(atr_pct_series) else float('nan')
            if np.isfinite(atr_pct) and atr_pct > 0:
                actual_short_sl = max(short_sl, -(atr_pct * atr_sl_mult))
                actual_long_sl = max(long_sl, -(atr_pct * atr_sl_mult))

        # 动态止盈
        actual_short_tp = short_tp
        actual_long_tp = long_tp
        if dyn_vol_series is not None and idx >= 20:
            vol = float(dyn_vol_series.iloc[idx - 1]) if idx - 1 < len(dyn_vol_series) else float('nan')
            if np.isfinite(vol):
                if vol > 0.03:
                    actual_short_tp = short_tp * 1.3
                    actual_long_tp = long_tp * 1.3
                elif vol < 0.01:
                    actual_short_tp = short_tp * 0.7
                    actual_long_tp = long_tp * 0.7

        # Regime-aware 动态阈值与风险控制 (仅使用历史数据)
        regime_ctl = _compute_regime_controls(primary_df, idx, config, precomputed=regime_precomputed)
        # 注入 regime_label 供多周期融合动态调权 (P1-2)
        config['_regime_label'] = regime_ctl.get('regime_label', 'neutral')
        cur_sell_threshold = float(regime_ctl['sell_threshold'])
        cur_buy_threshold = float(regime_ctl['buy_threshold'])
        cur_short_threshold = float(regime_ctl['short_threshold'])
        cur_long_threshold = float(regime_ctl['long_threshold'])
        cur_close_short_bs = float(regime_ctl['close_short_bs'])
        cur_close_long_ss = float(regime_ctl['close_long_ss'])
        cur_lev = int(regime_ctl['lev'])
        cur_margin_use = float(regime_ctl['margin_use'])

        # 微结构状态（资金费率/基差/OI/主动买卖）
        micro_state = _compute_microstructure_state(micro_df, idx, config)

        # 双引擎切换: 趋势跟随 vs 反转回归
        engine_mode = _resolve_engine_mode(regime_ctl.get('regime_label', 'neutral'), micro_state, config)
        entry_mult = 1.0
        exit_mult = 1.0
        hold_mult = 1.0
        risk_mult = 1.0
        entry_dom_ratio = float(config.get('entry_dominance_ratio', 1.5))
        if engine_mode == 'trend':
            entry_mult = float(config.get('trend_engine_entry_mult', 0.95))
            exit_mult = float(config.get('trend_engine_exit_mult', 1.05))
            hold_mult = float(config.get('trend_engine_hold_mult', 1.35))
            risk_mult = float(config.get('trend_engine_risk_mult', 1.10))
            entry_dom_ratio = float(config.get('trend_engine_dominance_ratio', 1.35))
        elif engine_mode == 'reversion':
            entry_mult = float(config.get('reversion_engine_entry_mult', 1.12))
            exit_mult = float(config.get('reversion_engine_exit_mult', 0.90))
            hold_mult = float(config.get('reversion_engine_hold_mult', 0.70))
            risk_mult = float(config.get('reversion_engine_risk_mult', 0.75))
            entry_dom_ratio = float(config.get('reversion_engine_dominance_ratio', 1.75))

        cur_sell_threshold = float(np.clip(cur_sell_threshold * entry_mult, 5.0, 95.0))
        cur_buy_threshold = float(np.clip(cur_buy_threshold * entry_mult, 5.0, 95.0))
        cur_short_threshold = float(np.clip(cur_short_threshold * entry_mult, 5.0, 95.0))
        cur_long_threshold = float(np.clip(cur_long_threshold * entry_mult, 5.0, 95.0))
        cur_close_short_bs = float(np.clip(cur_close_short_bs * exit_mult, 5.0, 95.0))
        cur_close_long_ss = float(np.clip(cur_close_long_ss * exit_mult, 5.0, 95.0))
        cur_lev = int(max(1, round(cur_lev * risk_mult)))
        cur_margin_use = float(np.clip(cur_margin_use * risk_mult, 0.05, 0.95))

        # 微结构叠加到融合分数 + 开仓限制
        ss, bs, micro_overlay = _apply_microstructure_overlay(ss, bs, micro_state, config)
        micro_block_long = bool(micro_overlay.get('block_long', False))
        micro_block_short = bool(micro_overlay.get('block_short', False))
        cur_margin_use = float(np.clip(cur_margin_use * float(micro_overlay.get('margin_mult', 1.0)), 0.05, 0.95))

        # 波动目标仓位: 高波动自动降风险，低波动允许恢复风险预算
        vol_scale, _rv_ann, vol_active = _compute_vol_target_scale(vol_ann, idx, config)
        if vol_active:
            cur_margin_use = float(np.clip(cur_margin_use * vol_scale, 0.05, 0.95))
            cur_lev = int(max(1, round(cur_lev * np.sqrt(vol_scale))))

        # 冲突区间过滤
        in_conflict = False
        if ss > 0 and bs > 0:
            ratio = min(ss, bs) / max(ss, bs)
            in_conflict = ratio >= 0.6 and min(ss, bs) >= 15

        # 是否启用趋势增强（反转引擎中禁用趋势偏置）
        trend_enhance_active = bool(config.get('use_trend_enhance', False)) and engine_mode != 'reversion'

        # ── 趋势持仓保护 (Trend Floor v3 — 带滞后 + 事后检查) ──
        trend_floor_active = False
        effective_sell_pct = sell_pct
        effective_sell_threshold = cur_sell_threshold
        effective_spot_cooldown = spot_cooldown
        trend_is_up = False  # 供后续做多/抑空逻辑使用
        if trend_enhance_active and idx >= 30 and trend_ema10 is not None and trend_ema30 is not None:
            ema10 = float(trend_ema10.iloc[idx])
            ema30 = float(trend_ema30.iloc[idx])

            # 趋势激活条件: EMA10 > EMA30 (不要求price > ema10, 避免敏感波动)
            trend_enter = ema10 > ema30 * 1.005  # 略高于1:1才激活
            # 趋势退出条件: EMA10 < EMA30 * 0.98 (明确反转才退出, 滞后保护)
            trend_exit = ema10 < ema30 * 0.98

            if trend_enter:
                _trend_up_active = True
            elif trend_exit:
                _trend_up_active = False
            # else: 保持上一个bar的趋势状态 (滞后)

            if _trend_up_active:
                trend_is_up = True
                trend_floor_ratio = float(config.get('trend_floor_ratio', 0.50))
                total_val = eng.total_value(price)
                eth_val = eng.spot_eth * price
                eth_ratio = eth_val / total_val if total_val > 0 else 0

                if eth_ratio <= trend_floor_ratio:
                    trend_floor_active = True  # 低于 floor, 完全禁止卖出
                else:
                    # 超出 floor 的部分可以少量卖出 (最多10%)
                    effective_sell_pct = min(sell_pct, 0.10)
                # 提高卖出阈值: 只有很强的卖出信号才触发
                effective_sell_threshold = max(cur_sell_threshold, 55)
                # 拉长现货冷却期到48bar (防快速连续卖出)
                effective_spot_cooldown = max(spot_cooldown, 48)

        # 永久最低持仓保护: 任何情况下不卖穿 min_base_eth_ratio
        # 使用"事后检查": 模拟卖出后的ETH比例, 若低于底线则阻止
        if not trend_floor_active and trend_enhance_active:
            total_val = eng.total_value(price)
            eth_val = eng.spot_eth * price
            eth_ratio = eth_val / total_val if total_val > 0 else 0
            min_base_ratio = float(config.get('min_base_eth_ratio', 0.15))
            if eth_ratio <= min_base_ratio:
                trend_floor_active = True
            else:
                # 事后检查: 如果卖出后会低于底线, 也阻止
                post_sell_eth_val = eth_val * (1 - effective_sell_pct)
                # 卖出后 USDT 增加, 总值近似不变 (短期)
                post_sell_ratio = post_sell_eth_val / total_val if total_val > 0 else 0
                floor_check = trend_floor_ratio if trend_is_up else min_base_ratio
                if post_sell_ratio < floor_check:
                    # 缩小卖出比例, 恰好卖到底线
                    max_sell = 1.0 - (floor_check * total_val / eth_val) if eth_val > 0 else 0
                    if max_sell <= 0.02:
                        trend_floor_active = True  # 几乎不能卖
                    else:
                        effective_sell_pct = max(0.02, max_sell)

        # 卖出现货 (用 exec_price 执行, 即本bar open)
        if not trend_floor_active and ss >= effective_sell_threshold and spot_cd == 0 and not in_conflict and eng.spot_eth * exec_price > 500:
            eng.spot_sell(exec_price, dt, effective_sell_pct, f"卖出 SS={ss:.0f}")
            spot_cd = effective_spot_cooldown

        # 开空 (用 exec_price 执行, 即本bar open)
        sell_dom = (ss > bs * entry_dom_ratio) if bs > 0 else True
        # 趋势中抑制做空: 提高阈值 + 要求更强的卖方优势
        effective_short_threshold = cur_short_threshold
        if trend_is_up and trend_enhance_active:
            effective_short_threshold = max(cur_short_threshold, 55)
            sell_dom = (ss > bs * max(2.5, entry_dom_ratio)) if bs > 0 else True
        if short_cd == 0 and ss >= effective_short_threshold and not eng.futures_short and sell_dom and not in_conflict and can_open_risk and not micro_block_short:
            margin = eng.available_margin() * cur_margin_use
            actual_lev = min(cur_lev if ss >= 50 else min(cur_lev, 3) if ss >= 35 else 2, eng.max_leverage)
            eng.open_short(exec_price, dt, margin, actual_lev, f"开空 {actual_lev}x")
            short_max_pnl = 0; short_bars = 0; short_cd = cooldown
            short_just_opened = True; short_partial_done = False; short_partial2_done = False

        # 管理空仓
        if eng.futures_short and not short_just_opened:
            short_bars += 1
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin

            # 一段止盈 (含滑点 + 修复frozen_margin泄漏)
            if use_partial_tp and not short_partial_done and pnl_r >= partial_tp_1:
                old_qty = eng.futures_short.quantity
                partial_qty = old_qty * partial_tp_1_pct
                actual_close_p = price * (1 + FuturesEngine.SLIPPAGE)  # 空头平仓买入, 价格偏高
                partial_pnl = (eng.futures_short.entry_price - actual_close_p) * partial_qty
                _entry_p = eng.futures_short.entry_price
                margin_released = eng.futures_short.margin * partial_tp_1_pct
                eng.usdt += margin_released + partial_pnl
                eng.frozen_margin -= margin_released  # 修复: 释放冻结保证金
                fee = partial_qty * actual_close_p * FuturesEngine.TAKER_FEE
                slippage_cost = partial_qty * price * FuturesEngine.SLIPPAGE
                eng.usdt -= fee; eng.total_futures_fees += fee
                eng.total_slippage_cost += slippage_cost
                eng.futures_short.quantity = old_qty - partial_qty
                eng.futures_short.margin *= (1 - partial_tp_1_pct)
                short_partial_done = True
                eng._record_trade(dt, price, 'PARTIAL_TP', 'short', partial_qty,
                    partial_qty * actual_close_p, fee,
                    eng.futures_short.leverage if eng.futures_short else 0,
                    f'分段TP1空 +{pnl_r*100:.0f}%',
                    exec_price=actual_close_p, slippage_cost=slippage_cost,
                    pnl=partial_pnl, entry_price=_entry_p,
                    margin_released=margin_released, partial_ratio=partial_tp_1_pct)

            # 二段止盈 (含滑点 + 修复frozen_margin泄漏, 使用elif避免同bar双触发)
            elif use_partial_tp_2 and short_partial_done and not short_partial2_done and pnl_r >= partial_tp_2:
                old_qty = eng.futures_short.quantity
                partial_qty = old_qty * partial_tp_2_pct
                actual_close_p = price * (1 + FuturesEngine.SLIPPAGE)
                partial_pnl = (eng.futures_short.entry_price - actual_close_p) * partial_qty
                _entry_p = eng.futures_short.entry_price
                margin_released = eng.futures_short.margin * partial_tp_2_pct
                eng.usdt += margin_released + partial_pnl
                eng.frozen_margin -= margin_released
                fee = partial_qty * actual_close_p * FuturesEngine.TAKER_FEE
                slippage_cost = partial_qty * price * FuturesEngine.SLIPPAGE
                eng.usdt -= fee; eng.total_futures_fees += fee
                eng.total_slippage_cost += slippage_cost
                eng.futures_short.quantity = old_qty - partial_qty
                eng.futures_short.margin *= (1 - partial_tp_2_pct)
                short_partial2_done = True
                eng._record_trade(dt, price, 'PARTIAL_TP', 'short', partial_qty,
                    partial_qty * actual_close_p, fee,
                    eng.futures_short.leverage if eng.futures_short else 0,
                    f'分段TP2空 +{pnl_r*100:.0f}%',
                    exec_price=actual_close_p, slippage_cost=slippage_cost,
                    pnl=partial_pnl, entry_price=_entry_p,
                    margin_released=margin_released, partial_ratio=partial_tp_2_pct)

            if pnl_r >= actual_short_tp:
                eng.close_short(price, dt, f"止盈 +{pnl_r*100:.0f}%")
                short_max_pnl = 0; short_cd = cooldown * 2; short_bars = 0
            else:
                if pnl_r > short_max_pnl: short_max_pnl = pnl_r
                if short_max_pnl >= short_trail and eng.futures_short:
                    if pnl_r < short_max_pnl * trail_pullback:
                        eng.close_short(price, dt, "追踪止盈")
                        short_max_pnl = 0; short_cd = cooldown; short_bars = 0
                if eng.futures_short and bs >= cur_close_short_bs:
                    bs_dom = (ss < bs * 0.7) if bs > 0 else True
                    if bs_dom:
                        # 信号驱动平仓用 exec_price (当前bar open), 因为信号来自上一根bar
                        eng.close_short(exec_price, dt, f"反向平空 BS={bs:.0f}")
                        short_max_pnl = 0; short_cd = cooldown * 3; short_bars = 0
                if eng.futures_short and pnl_r < actual_short_sl:
                    eng.close_short(price, dt, f"止损 {pnl_r*100:.0f}%")
                    short_max_pnl = 0; short_cd = cooldown * 4; short_bars = 0
                if eng.futures_short and short_bars >= int(max(3, short_max_hold * hold_mult)):
                    # 超时平仓为主观决策, 用 exec_price
                    eng.close_short(exec_price, dt, "超时")
                    short_max_pnl = 0; short_cd = cooldown; short_bars = 0
        elif eng.futures_short and short_just_opened:
            short_bars = 1

        # 买入现货 (用 exec_price 执行, 即本bar open)
        # 趋势中主动补仓: 降低买入门槛, 加大买入比例
        effective_buy_threshold = cur_buy_threshold
        effective_buy_pct = 0.25
        if trend_is_up and trend_enhance_active:
            total_val = eng.total_value(exec_price)
            eth_val = eng.spot_eth * exec_price
            eth_ratio = eth_val / total_val if total_val > 0 else 0
            target_ratio = float(config.get('trend_floor_ratio', 0.50))
            if eth_ratio < target_ratio:
                # ETH不足目标, 积极补仓
                effective_buy_threshold = min(cur_buy_threshold, 18)
                effective_buy_pct = min(0.50, (target_ratio - eth_ratio) + 0.15)
        if bs >= effective_buy_threshold and spot_cd == 0 and not in_conflict and eng.available_usdt() > 500 and can_open_risk:
            eng.spot_buy(exec_price, dt, eng.available_usdt() * effective_buy_pct, f"买入 BS={bs:.0f}")
            spot_cd = spot_cooldown

        # 开多 (用 exec_price 执行, 即本bar open)
        buy_dom = (bs > ss * entry_dom_ratio) if ss > 0 else True
        # 趋势跟踪做多: 当 live gate 过滤后 bs=0 时, 直接用大周期原始信号判断
        effective_long_threshold = cur_long_threshold
        trend_long_bs = bs  # 默认用 gate 后的分数
        if trend_is_up and trend_enhance_active:
            effective_long_threshold = min(cur_long_threshold, 25)
            buy_dom = (bs > ss) if ss > 0 else True
            # 如果 gate 后 bs 为 0 但趋势明确, 查看大周期原始信号
            if bs < 10 and tf_score_map and decision_tfs:
                # 从 4h 和 24h 直接获取原始评分
                raw_bs_list = []
                for tf in decision_tfs:
                    tf_mins = _MTF_MINUTES.get(tf, 60)
                    if tf_mins >= 240:  # 只看 4h 及以上
                        raw_ss, raw_bs = _get_tf_score_at(tf_score_map, tf, dt)
                        if raw_bs > 0:
                            raw_bs_list.append(raw_bs)
                # 至少1个大周期看多, 且无大周期看空
                if raw_bs_list:
                    raw_big_bs = sum(raw_bs_list) / len(raw_bs_list)
                    raw_big_ss_list = []
                    for tf in decision_tfs:
                        tf_mins = _MTF_MINUTES.get(tf, 60)
                        if tf_mins >= 240:
                            raw_ss, _ = _get_tf_score_at(tf_score_map, tf, dt)
                            if raw_ss > 0:
                                raw_big_ss_list.append(raw_ss)
                    raw_big_ss = sum(raw_big_ss_list) / len(raw_big_ss_list) if raw_big_ss_list else 0
                    if raw_big_bs > raw_big_ss * 1.2:  # 大周期买方占优
                        trend_long_bs = raw_big_bs
                        buy_dom = True
        if long_cd == 0 and trend_long_bs >= effective_long_threshold and not eng.futures_long and buy_dom and not in_conflict and can_open_risk and not micro_block_long:
            margin = eng.available_margin() * cur_margin_use
            actual_lev = min(cur_lev if bs >= 50 else min(cur_lev, 3) if bs >= 35 else 2, eng.max_leverage)
            eng.open_long(exec_price, dt, margin, actual_lev, f"开多 {actual_lev}x")
            long_max_pnl = 0; long_bars = 0; long_cd = cooldown
            long_just_opened = True; long_partial_done = False; long_partial2_done = False

        # 管理多仓
        if eng.futures_long and not long_just_opened:
            long_bars += 1
            pnl_r = eng.futures_long.calc_pnl(price) / eng.futures_long.margin

            # 一段止盈 (含滑点 + 修复frozen_margin泄漏)
            if use_partial_tp and not long_partial_done and pnl_r >= partial_tp_1:
                old_qty = eng.futures_long.quantity
                partial_qty = old_qty * partial_tp_1_pct
                actual_close_p = price * (1 - FuturesEngine.SLIPPAGE)  # 多头平仓卖出, 价格偏低
                partial_pnl = (actual_close_p - eng.futures_long.entry_price) * partial_qty
                _entry_p = eng.futures_long.entry_price
                margin_released = eng.futures_long.margin * partial_tp_1_pct
                eng.usdt += margin_released + partial_pnl
                eng.frozen_margin -= margin_released  # 修复: 释放冻结保证金
                fee = partial_qty * actual_close_p * FuturesEngine.TAKER_FEE
                slippage_cost = partial_qty * price * FuturesEngine.SLIPPAGE
                eng.usdt -= fee; eng.total_futures_fees += fee
                eng.total_slippage_cost += slippage_cost
                eng.futures_long.quantity = old_qty - partial_qty
                eng.futures_long.margin *= (1 - partial_tp_1_pct)
                long_partial_done = True
                eng._record_trade(dt, price, 'PARTIAL_TP', 'long', partial_qty,
                    partial_qty * actual_close_p, fee,
                    eng.futures_long.leverage if eng.futures_long else 0,
                    f'分段TP1多 +{pnl_r*100:.0f}%',
                    exec_price=actual_close_p, slippage_cost=slippage_cost,
                    pnl=partial_pnl, entry_price=_entry_p,
                    margin_released=margin_released, partial_ratio=partial_tp_1_pct)

            # 二段止盈 (含滑点 + 修复frozen_margin泄漏, 使用elif避免同bar双触发)
            elif use_partial_tp_2 and long_partial_done and not long_partial2_done and pnl_r >= partial_tp_2:
                old_qty = eng.futures_long.quantity
                partial_qty = old_qty * partial_tp_2_pct
                actual_close_p = price * (1 - FuturesEngine.SLIPPAGE)
                partial_pnl = (actual_close_p - eng.futures_long.entry_price) * partial_qty
                _entry_p = eng.futures_long.entry_price
                margin_released = eng.futures_long.margin * partial_tp_2_pct
                eng.usdt += margin_released + partial_pnl
                eng.frozen_margin -= margin_released
                fee = partial_qty * actual_close_p * FuturesEngine.TAKER_FEE
                slippage_cost = partial_qty * price * FuturesEngine.SLIPPAGE
                eng.usdt -= fee; eng.total_futures_fees += fee
                eng.total_slippage_cost += slippage_cost
                eng.futures_long.quantity = old_qty - partial_qty
                eng.futures_long.margin *= (1 - partial_tp_2_pct)
                long_partial2_done = True
                eng._record_trade(dt, price, 'PARTIAL_TP', 'long', partial_qty,
                    partial_qty * actual_close_p, fee,
                    eng.futures_long.leverage if eng.futures_long else 0,
                    f'分段TP2多 +{pnl_r*100:.0f}%',
                    exec_price=actual_close_p, slippage_cost=slippage_cost,
                    pnl=partial_pnl, entry_price=_entry_p,
                    margin_released=margin_released, partial_ratio=partial_tp_2_pct)

            if pnl_r >= actual_long_tp:
                eng.close_long(price, dt, f"止盈 +{pnl_r*100:.0f}%")
                long_max_pnl = 0; long_cd = cooldown * 2; long_bars = 0
            else:
                if pnl_r > long_max_pnl: long_max_pnl = pnl_r
                if long_max_pnl >= long_trail and eng.futures_long:
                    if pnl_r < long_max_pnl * trail_pullback:
                        eng.close_long(price, dt, "追踪止盈")
                        long_max_pnl = 0; long_cd = cooldown; long_bars = 0
                if eng.futures_long and ss >= cur_close_long_ss:
                    ss_dom = (bs < ss * 0.7) if bs > 0 else True
                    if ss_dom:
                        # 信号驱动平仓用 exec_price (当前bar open), 因为信号来自上一根bar
                        eng.close_long(exec_price, dt, f"反向平多 SS={ss:.0f}")
                        long_max_pnl = 0; long_cd = cooldown * 3; long_bars = 0
                if eng.futures_long and pnl_r < actual_long_sl:
                    eng.close_long(price, dt, f"止损 {pnl_r*100:.0f}%")
                    long_max_pnl = 0; long_cd = cooldown * 4; long_bars = 0
                if eng.futures_long and long_bars >= int(max(3, long_max_hold * hold_mult)):
                    # 超时平仓为主观决策, 用 exec_price
                    eng.close_long(exec_price, dt, "超时")
                    long_max_pnl = 0; long_cd = cooldown; long_bars = 0
        elif eng.futures_long and long_just_opened:
            long_bars = 1

        if len(eng.trades) > trade_cursor:
            if prot_state.get('enabled', False):
                for tr in eng.trades[trade_cursor:]:
                    pnl = _extract_realized_pnl_from_trade(tr)
                    _apply_loss_streak_protection(prot_state, pnl, idx, config)
            trade_cursor = len(eng.trades)

        if idx % record_interval == 0:
            eng.record_history(dt, price)

        # ── 在当前 bar 收盘后计算信号, 供下一根 bar 执行 ──
        pending_ss, pending_bs = score_provider(idx, dt, price)
        has_pending_signal = True

    # 期末平仓: 若指定 end_dt，则在 end_dt 所在窗口结束价结算。
    settle_df = primary_df
    if end_dt is not None:
        settle_df = primary_df[primary_df.index <= end_dt]
        if len(settle_df) == 0:
            settle_df = primary_df.iloc[[0]]
    last_price = settle_df['close'].iloc[-1]
    last_dt = settle_df.index[-1]
    if eng.futures_short:
        eng.close_short(last_price, last_dt, "期末平仓")
    if eng.futures_long:
        eng.close_long(last_price, last_dt, "期末平仓")

    trade_df = primary_df
    if start_dt is not None:
        trade_df = trade_df[trade_df.index >= start_dt]
    if end_dt is not None:
        trade_df = trade_df[trade_df.index <= end_dt]

    if len(trade_df) > 1:
        result = eng.get_result(trade_df)
        if prot_state.get('enabled', False):
            result['protections'] = {
                **prot_state.get('stats', {}),
                'daily_locked': bool(prot_state.get('daily_locked', False)),
                'global_halt': bool(prot_state.get('global_halt', False)),
                'loss_streak': int(prot_state.get('loss_streak', 0)),
                'entry_block_until_idx': int(prot_state.get('entry_block_until_idx', -1)),
                'daily_pnl_pct': round(float(prot_state.get('daily_pnl_pct', 0.0)) * 100, 2),
                'drawdown_from_peak_pct': round(float(prot_state.get('drawdown_from_peak_pct', 0.0)) * 100, 2),
            }
        return result

    result = eng.get_result(primary_df)
    if prot_state.get('enabled', False):
        result['protections'] = {
            **prot_state.get('stats', {}),
            'daily_locked': bool(prot_state.get('daily_locked', False)),
            'global_halt': bool(prot_state.get('global_halt', False)),
            'loss_streak': int(prot_state.get('loss_streak', 0)),
            'entry_block_until_idx': int(prot_state.get('entry_block_until_idx', -1)),
            'daily_pnl_pct': round(float(prot_state.get('daily_pnl_pct', 0.0)) * 100, 2),
            'drawdown_from_peak_pct': round(float(prot_state.get('drawdown_from_peak_pct', 0.0)) * 100, 2),
        }
    return result


def run_strategy(
    df,
    signals,
    config,
    tf='1h',
    trade_days=30,
    trade_start_dt=None,
    trade_end_dt=None,
):
    """在指定时间框架上运行六书策略回测。"""

    def _single_tf_score(idx, dt, _price):
        return calc_fusion_score_six(signals, df, idx, dt, config)

    return _run_strategy_core(
        df,
        config,
        tf,
        trade_days,
        _single_tf_score,
        trade_start_dt=trade_start_dt,
        trade_end_dt=trade_end_dt,
    )


# ======================================================
#   多周期联合决策回测引擎
# ======================================================

# 各时间框架权重(与 live_runner 保持一致)
_MTF_WEIGHT = {
    '10m': 2, '15m': 3, '30m': 5,
    '1h': 8, '2h': 10, '3h': 12,
    '4h': 15, '6h': 18, '8h': 20,
    '12h': 22, '16h': 25, '24h': 28,
}
_MTF_MINUTES = {
    '10m':10, '15m':15, '30m':30,
    '1h':60, '2h':120, '3h':180, '4h':240, '6h':360,
    '8h':480, '12h':720, '16h':960, '24h':1440,
}


def _build_tf_score_index(all_data, all_signals, tfs, config):
    """
    预计算每个 TF 在每个时间戳的 (sell_score, buy_score)。
    返回 {tf: {timestamp: (ss, bs)}} 的字典, 供回测快速查询。

    P1 优化: 使用 calc_fusion_score_six_batch 向量化批量计算,
    从逐 bar 循环 (41s) 降至 <3s。
    """
    from signal_core import calc_fusion_score_six_batch

    tf_score_map = {'__lookup_cache__': {}}
    for tf in tfs:
        df = all_data[tf]
        sigs = all_signals[tf]
        warmup = max(60, int(len(df) * 0.05))

        score_dict, ordered_ts = calc_fusion_score_six_batch(
            sigs, df, config, warmup=warmup,
        )

        tf_score_map[tf] = score_dict
        tf_score_map['__lookup_cache__'][tf] = {
            'times': ordered_ts,
            'cursor': 0,
            'n': len(ordered_ts),
        }
    return tf_score_map


def _ensure_tf_lookup_cache(tf_score_map, tf, scores):
    """确保 tf 的有序时间缓存可用。"""
    cache_root = tf_score_map.setdefault('__lookup_cache__', {})
    cache = cache_root.get(tf)
    n_scores = len(scores)
    if cache is None or int(cache.get('n', -1)) != n_scores:
        ordered_ts = list(scores.keys())
        if ordered_ts and any(ordered_ts[i] > ordered_ts[i + 1] for i in range(len(ordered_ts) - 1)):
            ordered_ts = sorted(ordered_ts)
        cache = {'times': ordered_ts, 'cursor': 0, 'n': n_scores}
        cache_root[tf] = cache
    return cache


def _get_tf_score_at(tf_score_map, tf, dt, with_valid=False):
    """
    在 tf 的评分索引中查找离 dt 最近且 <= dt 的评分。
    使用时效衰减: 评分随年龄(age)平滑衰减，而非二值截断，
    减少临界点抖动和信号跳变。
    """
    scores = tf_score_map.get(tf)
    if not scores:
        return (0.0, 0.0, False) if with_valid else (0.0, 0.0)

    # 精确匹配
    if dt in scores:
        ss, bs = scores[dt]
        ss = float(ss)
        bs = float(bs)
        return (ss, bs, True) if with_valid else (ss, bs)

    cache = _ensure_tf_lookup_cache(tf_score_map, tf, scores)
    times = cache.get('times', [])
    if not times:
        return (0.0, 0.0, False) if with_valid else (0.0, 0.0)

    if dt < times[0]:
        return (0.0, 0.0, False) if with_valid else (0.0, 0.0)

    if dt >= times[-1]:
        pos = len(times) - 1
    else:
        cursor = int(cache.get('cursor', 0))
        if cursor < 0 or cursor >= len(times):
            cursor = 0
        if times[cursor] <= dt:
            while cursor + 1 < len(times) and times[cursor + 1] <= dt:
                cursor += 1
            pos = cursor
        else:
            pos = bisect_right(times, dt) - 1
            if pos < 0:
                return (0.0, 0.0, False) if with_valid else (0.0, 0.0)
    cache['cursor'] = pos
    nearest = times[pos]

    # 时效衰减: 1个周期内=100%, 2个周期=50%, 3个周期=25%, >4个周期≈0
    tf_mins = _MTF_MINUTES.get(tf, 60)
    age_secs = (dt - nearest).total_seconds()
    period_secs = tf_mins * 60
    age_periods = age_secs / period_secs if period_secs > 0 else 999

    if age_periods <= 1.0:
        decay = 1.0  # 1个周期内, 完全有效
    elif age_periods >= 4.0:
        decay = 0.0  # 4个周期以上, 视为失效
    else:
        # 1~4个周期之间, 线性衰减: 1期=100%, 4期=0%
        decay = max(0.0, 1.0 - (age_periods - 1.0) / 3.0)

    if decay <= 0.01:
        return (0.0, 0.0, False) if with_valid else (0.0, 0.0)

    ss, bs = scores[nearest]
    ss = float(ss) * decay
    bs = float(bs) * decay
    return (ss, bs, True) if with_valid else (ss, bs)


def calc_multi_tf_consensus(tf_score_map, decision_tfs, dt, config):
    """
    在指定时刻 dt 计算多周期加权共识评分。

    委托给 multi_tf_consensus.fuse_tf_scores 统一实现,
    保持与实盘引擎完全相同的融合逻辑。

    返回: (consensus_ss, consensus_bs, meta_dict)
    """
    # 从预计算索引中查找当前时刻各TF的分数
    tf_scores = {}
    for tf in decision_tfs:
        ss, bs, valid = _get_tf_score_at(tf_score_map, tf, dt, with_valid=True)
        if valid:
            tf_scores[tf] = (ss, bs)

    # 覆盖率门控可配置:
    # - 传统回测: coverage_min=0.0 (不门控)
    # - 实盘口径回放: coverage_min=0.5 (与 live 一致)
    fuse_config = {
        'short_threshold': config.get('short_threshold', 25),
        'long_threshold': config.get('long_threshold', 40),
        'coverage_min': config.get('coverage_min', 0.0),
        # 参数化融合阈值
        'dominance_ratio': config.get('dominance_ratio', 1.3),
        'chain_boost_per_tf': config.get('chain_boost_per_tf', 0.08),
        'chain_boost_weak_per_tf': config.get('chain_boost_weak_per_tf', 0.04),
        # Regime驱动动态TF权重 (由 _run_strategy_core 通过 config 注入)
        '_regime_label': config.get('_regime_label', 'neutral'),
    }
    result = fuse_tf_scores(tf_scores, decision_tfs, fuse_config)
    meta = dict(result.get("meta", {}))
    meta["decision"] = result.get("decision", {})
    meta["coverage"] = result.get("coverage", 0.0)
    meta["weighted_scores"] = result.get("weighted_scores", {})
    return result["weighted_ss"], result["weighted_bs"], meta


def _apply_live_parity_gate(ss, bs, meta, config):
    """
    可选: 回测中复用实盘共识门控口径 (actionable/direction/strength)。
    """
    if not config.get('use_live_gate', False):
        return ss, bs

    decision = meta.get("decision", {}) if isinstance(meta, dict) else {}
    direction = decision.get("direction", "hold")
    strength = float(decision.get("strength", 0))
    actionable = bool(decision.get("actionable", False))
    min_strength = float(config.get("consensus_min_strength", 40))
    short_threshold = float(config.get("short_threshold", 25))
    long_threshold = float(config.get("long_threshold", 40))

    if not actionable:
        return 0.0, 0.0
    if direction not in ("long", "short"):
        return 0.0, 0.0
    if strength < min_strength:
        return 0.0, 0.0

    if direction == "long":
        return 0.0, max(float(bs), long_threshold)
    return max(float(ss), short_threshold), 0.0


def run_strategy_multi_tf(
    primary_df,
    tf_score_map,
    decision_tfs,
    config,
    primary_tf='1h',
    trade_days=30,
    trade_start_dt=None,
    trade_end_dt=None,
):
    """多周期联合决策回测引擎。"""

    def _multi_tf_score(_idx, dt, _price):
        ss, bs, meta = calc_multi_tf_consensus(tf_score_map, decision_tfs, dt, config)
        ss, bs = _apply_live_parity_gate(ss, bs, meta, config)
        return ss, bs

    return _run_strategy_core(
        primary_df,
        config=config,
        primary_tf=primary_tf,
        trade_days=trade_days,
        score_provider=_multi_tf_score,
        trade_start_dt=trade_start_dt,
        trade_end_dt=trade_end_dt,
        tf_score_map=tf_score_map,
        decision_tfs=decision_tfs,
    )


# ======================================================
#   参数空间
# ======================================================
def get_all_variants():
    """获取全部参数变体(六书增强版)"""
    variants = []

    # ---------- 基准: 五书最优配置(用六书评分函数跑) ----------
    prev_best = {
        'tag': '五书最优(六书评分)',
        'fusion_mode': 'c6_veto_4',
        'short_sl': -0.25, 'short_tp': 0.60, 'short_trail': 0.25, 'short_max_hold': 72,
        'long_sl': -0.08, 'long_tp': 0.30, 'long_trail': 0.20, 'long_max_hold': 72,
        'trail_pullback': 0.60, 'cooldown': 4, 'spot_cooldown': 12,
        'use_partial_tp': True, 'partial_tp_1': 0.20, 'partial_tp_1_pct': 0.30,
    }
    variants.append(prev_best)

    # ---------- Phase 1: 融合模式对比 ----------
    for mode in ['c6_veto', 'c6_veto_4', 'kdj_weighted', 'kdj_timing', 'kdj_gate']:
        v = {**prev_best}
        v['tag'] = f'模式_{mode}'
        v['fusion_mode'] = mode
        variants.append(v)

    # ---------- Phase 2: KDJ否决阈值 ----------
    for vt in [15, 20, 25, 30, 35]:
        v = {**prev_best}
        v['tag'] = f'否决阈值_{vt}'
        v['fusion_mode'] = 'c6_veto_4'
        v['veto_threshold'] = vt
        variants.append(v)

    # ---------- Phase 3: KDJ确认奖励权重 ----------
    for kb in [0.05, 0.08, 0.12, 0.15, 0.20]:
        v = {**prev_best}
        v['tag'] = f'KDJ奖励_{kb*100:.0f}%'
        v['fusion_mode'] = 'c6_veto_4'
        v['kdj_bonus'] = kb
        variants.append(v)

    # ---------- Phase 4: KDJ加权模式权重搜索 ----------
    for kw in [0.10, 0.15, 0.20, 0.25, 0.30]:
        for dw in [0.45, 0.50, 0.55, 0.60]:
            if kw + dw > 0.85: continue  # 确保均线权重>0
            v = {**prev_best}
            v['tag'] = f'加权_KDJ{kw*100:.0f}_背离{dw*100:.0f}'
            v['fusion_mode'] = 'kdj_weighted'
            v['kdj_weight'] = kw
            v['div_weight'] = dw
            variants.append(v)

    # ---------- Phase 5: KDJ择时乘数 ----------
    for strong in [1.15, 1.20, 1.25, 1.30, 1.40]:
        for normal in [1.05, 1.10, 1.15, 1.20]:
            for reverse in [0.50, 0.60, 0.70, 0.80]:
                if strong <= normal: continue
                v = {**prev_best}
                v['tag'] = f'择时_强{strong}_弱{normal}_反{reverse}'
                v['fusion_mode'] = 'kdj_timing'
                v['kdj_strong_mult'] = strong
                v['kdj_normal_mult'] = normal
                v['kdj_reverse_mult'] = reverse
                variants.append(v)

    # ---------- Phase 6: KDJ门控阈值 ----------
    for gt in [5, 8, 10, 12, 15, 20]:
        v = {**prev_best}
        v['tag'] = f'门控_KDJ>{gt}'
        v['fusion_mode'] = 'kdj_gate'
        v['kdj_gate_threshold'] = gt
        variants.append(v)

    # ---------- Phase 7: 止损微调(围绕最优) ----------
    for ssl in [-0.15, -0.20, -0.25, -0.30, -0.35]:
        for lsl in [-0.06, -0.08, -0.10, -0.12, -0.15]:
            if ssl == -0.25 and lsl == -0.08: continue
            v = {**prev_best}
            v['tag'] = f'SL空{ssl*100:.0f}%多{lsl*100:.0f}%'
            v['short_sl'] = ssl; v['long_sl'] = lsl
            variants.append(v)

    # ---------- Phase 8: 止盈微调 ----------
    for stp in [0.40, 0.50, 0.60, 0.70, 0.80, 1.00]:
        for ltp in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
            if stp == 0.60 and ltp == 0.30: continue
            v = {**prev_best}
            v['tag'] = f'TP空{stp*100:.0f}%多{ltp*100:.0f}%'
            v['short_tp'] = stp; v['long_tp'] = ltp
            variants.append(v)

    # ---------- Phase 9: 追踪止盈 ----------
    for trail in [0.15, 0.20, 0.25, 0.30, 0.35]:
        for pullback in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            if trail == 0.25 and pullback == 0.60: continue
            v = {**prev_best}
            v['tag'] = f'Trail{trail*100:.0f}%回撤{pullback*100:.0f}%'
            v['short_trail'] = trail; v['long_trail'] = trail
            v['trail_pullback'] = pullback
            variants.append(v)

    # ---------- Phase 10: 分段止盈参数 ----------
    for pt1 in [0.15, 0.20, 0.25, 0.30, 0.40]:
        for pt1_pct in [0.20, 0.25, 0.30, 0.40, 0.50]:
            if pt1 == 0.20 and pt1_pct == 0.30: continue
            v = {**prev_best}
            v['tag'] = f'分段TP@{pt1*100:.0f}%平{pt1_pct*100:.0f}%'
            v['partial_tp_1'] = pt1; v['partial_tp_1_pct'] = pt1_pct
            variants.append(v)

    # ---------- Phase 11: 二段止盈 ----------
    for pt1 in [0.15, 0.20]:
        for pt1_pct in [0.25, 0.30]:
            for pt2 in [0.40, 0.50, 0.60]:
                for pt2_pct in [0.25, 0.30, 0.40]:
                    v = {**prev_best}
                    v['tag'] = f'双段TP@{pt1*100:.0f}/{pt2*100:.0f}%平{pt1_pct*100:.0f}/{pt2_pct*100:.0f}%'
                    v['partial_tp_1'] = pt1; v['partial_tp_1_pct'] = pt1_pct
                    v['use_partial_tp_2'] = True
                    v['partial_tp_2'] = pt2; v['partial_tp_2_pct'] = pt2_pct
                    variants.append(v)

    # ---------- Phase 12: 信号阈值 ----------
    for sell_t in [15, 18, 22]:
        for short_t in [20, 25, 30]:
            for buy_t in [20, 25, 30]:
                for long_t in [35, 40, 45]:
                    if sell_t == 18 and short_t == 25 and buy_t == 25 and long_t == 40: continue
                    v = {**prev_best}
                    v['tag'] = f'阈值S{sell_t}_空{short_t}_B{buy_t}_多{long_t}'
                    v['sell_threshold'] = sell_t; v['short_threshold'] = short_t
                    v['buy_threshold'] = buy_t; v['long_threshold'] = long_t
                    variants.append(v)

    # ---------- Phase 13: 持仓时间 ----------
    for hold in [36, 48, 72, 96, 120, 168]:
        v = {**prev_best}
        v['tag'] = f'Hold{hold}bars'
        v['short_max_hold'] = hold; v['long_max_hold'] = hold
        variants.append(v)

    # ---------- Phase 14: ATR止损 + 分段止盈组合 ----------
    for atr_m in [2.0, 2.5, 3.0, 3.5]:
        v = {**prev_best}
        v['tag'] = f'ATR{atr_m}x+分段TP'
        v['use_atr_sl'] = True; v['atr_sl_mult'] = atr_m
        variants.append(v)

    # ---------- Phase 15: 杠杆和仓位 ----------
    for l in [3, 4, 5, 7]:
        for mu in [0.50, 0.60, 0.70, 0.80]:
            if l == 5 and mu == 0.70: continue
            v = {**prev_best}
            v['tag'] = f'杠杆{l}x仓位{mu*100:.0f}%'
            v['lev'] = l; v['margin_use'] = mu
            variants.append(v)

    # ---------- Phase 16: 否决削弱比例 ----------
    for vd in [0.15, 0.20, 0.30, 0.40, 0.50]:
        v = {**prev_best}
        v['tag'] = f'否决削弱{vd*100:.0f}%'
        v['fusion_mode'] = 'c6_veto_4'
        v['veto_dampen'] = vd
        variants.append(v)

    return variants


# ======================================================
#   主函数
# ======================================================
def main():
    trade_days = 30
    print("=" * 120)
    print("  六书融合多时间框架止盈止损优化器")
    print("  基于六书(含KDJ)信号 · 12个时间周期 · 系统性参数搜索")
    print(f"  目标: 超越五书最优 α=+86.69%")
    print("=" * 120)

    # 获取所有时间框架数据
    print("\n[1/5] 获取多时间框架数据...")
    all_data = fetch_multi_tf_data(ALL_TIMEFRAMES, days=60)
    available_tfs = sorted(all_data.keys(), key=lambda x: ALL_TIMEFRAMES.index(x) if x in ALL_TIMEFRAMES else 99)
    print(f"\n可用时间框架: {', '.join(available_tfs)}")

    # 预计算各时间框架六维信号
    print("\n[2/5] 预计算各时间框架的六维信号(含KDJ)...")
    all_signals = {}
    for tf in available_tfs:
        print(f"\n  计算 {tf} 六维信号:")
        all_signals[tf] = compute_signals_six(all_data[tf], tf, all_data)
        print(f"    {tf} 六维信号计算完成")

    # 基准配置(五书最优参数 + 六书评分)
    f12_base = {
        'name': '六书基准',
        'fusion_mode': 'c6_veto_4',
        'veto_threshold': 25,
        'single_pct': 0.20, 'total_pct': 0.50, 'lifetime_pct': 5.0,
        'sell_threshold': 18, 'buy_threshold': 25,
        'short_threshold': 25, 'long_threshold': 40,
        'close_short_bs': 40, 'close_long_ss': 40,
        'sell_pct': 0.55, 'margin_use': 0.70, 'lev': 5, 'max_lev': 5,
    }

    tf_hours = {'10m': 1/6, '15m': 0.25, '30m': 0.5, '1h': 1, '2h': 2, '3h': 3,
                '4h': 4, '6h': 6, '8h': 8, '12h': 12, '16h': 16, '24h': 24}

    # 获取参数变体
    all_variants = get_all_variants()
    print(f"\n[3/5] 参数变体: {len(all_variants)}种")

    # ============ Phase A: 各时间框架基准测试 ============
    print(f"\n{'=' * 120}")
    print(f"  Phase A: 各时间框架基准性能(六书C6+四书否决)")
    print(f"{'=' * 120}")

    # 五书最优参数
    prev_best_sl_tp = {
        'short_sl': -0.25, 'short_tp': 0.60, 'short_trail': 0.25, 'short_max_hold': 72,
        'long_sl': -0.08, 'long_tp': 0.30, 'long_trail': 0.20, 'long_max_hold': 72,
        'trail_pullback': 0.60, 'cooldown': 4, 'spot_cooldown': 12,
        'use_partial_tp': True, 'partial_tp_1': 0.20, 'partial_tp_1_pct': 0.30,
    }

    tf_baseline_results = {}
    print(f"\n{'时间框架':<10} {'K线数':>8} {'Alpha':>10} {'策略收益':>12} {'BH收益':>12} "
          f"{'回撤':>8} {'交易':>6} {'强平':>4} {'费用':>10}")
    print('-' * 100)

    for tf in available_tfs:
        config = {**f12_base, **prev_best_sl_tp}
        config['name'] = f'六书_{tf}'
        hours = tf_hours.get(tf, 1)
        config['short_max_hold'] = max(6, int(72 / hours))
        config['long_max_hold'] = max(6, int(72 / hours))
        config['cooldown'] = max(1, int(4 / hours))
        config['spot_cooldown'] = max(2, int(12 / hours))

        r = run_strategy(all_data[tf], all_signals[tf], config, tf=tf, trade_days=trade_days)
        fees = r.get('fees', {})
        tf_baseline_results[tf] = r

        print(f"  {tf:<8} {len(all_data[tf]):>8} {r['alpha']:>+9.2f}% {r['strategy_return']:>+11.2f}% "
              f"{r['buy_hold_return']:>+11.2f}% {r['max_drawdown']:>7.2f}% "
              f"{r['total_trades']:>5} {r['liquidations']:>3} "
              f"${fees.get('total_costs', 0):>9,.0f}")

    tf_ranked = sorted(tf_baseline_results.items(), key=lambda x: x[1]['alpha'], reverse=True)
    top_tfs = [t[0] for t in tf_ranked[:4]]  # 选TOP4
    print(f"\n  TOP4时间框架: {', '.join(top_tfs)}")
    for i, (tf, r) in enumerate(tf_ranked[:4]):
        print(f"    #{i+1}: {tf} α={r['alpha']:+.2f}%")

    # ============ Phase B: 大规模参数搜索 ============
    print(f"\n{'=' * 120}")
    print(f"  Phase B: 在TOP时间框架上系统优化({len(all_variants)}种变体)")
    print(f"{'=' * 120}")

    all_opt_results = []

    for tf in top_tfs:
        print(f"\n  === {tf} 优化开始 ({len(all_variants)}种参数变体) ===")
        tf_h = tf_hours.get(tf, 1)
        results_for_tf = []

        for i, var in enumerate(all_variants):
            config = {**f12_base, **var}
            config['name'] = f'{var["tag"]}_{tf}'

            raw_hold = var.get('short_max_hold', 72)
            config['short_max_hold'] = max(6, int(raw_hold / tf_h))
            config['long_max_hold'] = max(6, int(var.get('long_max_hold', 72) / tf_h))
            config['cooldown'] = max(1, int(var.get('cooldown', 4) / tf_h))
            config['spot_cooldown'] = max(2, int(var.get('spot_cooldown', 12) / tf_h))

            r = run_strategy(all_data[tf], all_signals[tf], config, tf=tf, trade_days=trade_days)
            results_for_tf.append({
                'tf': tf,
                'tag': var['tag'],
                'alpha': r['alpha'],
                'strategy_return': r['strategy_return'],
                'buy_hold_return': r['buy_hold_return'],
                'max_drawdown': r['max_drawdown'],
                'total_trades': r['total_trades'],
                'liquidations': r['liquidations'],
                'fees': r.get('fees', {}).get('total_costs', 0),
                'final_total': r.get('final_total', 0),
                'config': var,
            })

            if (i + 1) % 50 == 0:
                print(f"    进度: {i+1}/{len(all_variants)}")

        results_for_tf.sort(key=lambda x: x['alpha'], reverse=True)
        all_opt_results.extend(results_for_tf)

        print(f"\n  {tf} TOP15参数:")
        print(f"  {'排名':>4} {'参数标签':<40} {'Alpha':>10} {'收益':>10} {'回撤':>8} {'交易':>6}")
        print('  ' + '-' * 90)
        for i, r in enumerate(results_for_tf[:15]):
            star = ' ★' if i == 0 else ''
            print(f"  #{i+1:>3} {r['tag']:<40} {r['alpha']:>+9.2f}% "
                  f"{r['strategy_return']:>+9.2f}% {r['max_drawdown']:>7.2f}% {r['total_trades']:>5}{star}")

    # ============ Phase C: 全局TOP排序 ============
    print(f"\n{'=' * 120}")
    print(f"  Phase C: 全局最优参数组合")
    print(f"{'=' * 120}")

    all_opt_results.sort(key=lambda x: x['alpha'], reverse=True)

    print(f"\n  全局TOP30:")
    print(f"  {'排名':>4} {'时间框架':>8} {'参数标签':<40} {'Alpha':>10} {'收益':>12} {'回撤':>8} {'交易':>6} {'费用':>10}")
    print('  ' + '-' * 120)
    for i, r in enumerate(all_opt_results[:30]):
        star = ' ★★★' if i == 0 else ' ★★' if i <= 2 else ' ★' if i <= 4 else ''
        print(f"  #{i+1:>3} {r['tf']:>8} {r['tag']:<40} {r['alpha']:>+9.2f}% "
              f"{r['strategy_return']:>+11.2f}% {r['max_drawdown']:>7.2f}% "
              f"{r['total_trades']:>5} ${r['fees']:>9,.0f}{star}")

    # ============ Phase D: 精细组合搜索 ============
    if len(all_opt_results) >= 5:
        print(f"\n{'=' * 120}")
        print(f"  Phase D: 精细组合搜索(交叉最优参数)")
        print(f"{'=' * 120}")

        top1 = all_opt_results[0]
        top1_tf = top1['tf']
        top1_cfg = top1['config']

        # 从各维度提取最佳
        def best_of(prefix, results):
            return sorted([r for r in results if r['tag'].startswith(prefix)],
                         key=lambda x: x['alpha'], reverse=True)

        best_mode = best_of('模式_', all_opt_results)
        best_veto = best_of('否决阈值_', all_opt_results)
        best_kdj_bonus = best_of('KDJ奖励_', all_opt_results)
        best_sl = best_of('SL', all_opt_results)
        best_tp = best_of('TP', all_opt_results)
        best_trail = best_of('Trail', all_opt_results)
        best_partial = best_of('分段TP', all_opt_results)
        best_dual = best_of('双段TP', all_opt_results)
        best_threshold = best_of('阈值', all_opt_results)
        best_lev = best_of('杠杆', all_opt_results)
        best_gate = best_of('门控', all_opt_results)
        best_timing = best_of('择时', all_opt_results)
        best_weighted = best_of('加权', all_opt_results)
        best_atr = best_of('ATR', all_opt_results)
        best_dampen = best_of('否决削弱', all_opt_results)

        fine_variants = []

        # 组合1: 最佳模式 + 最佳SL + 最佳TP + 最佳分段
        if best_mode and best_sl and best_tp:
            base_mode_cfg = best_mode[0]['config'] if best_mode else top1_cfg
            base_sl_cfg = best_sl[0]['config'] if best_sl else {}
            base_tp_cfg = best_tp[0]['config'] if best_tp else {}
            base_partial_cfg = best_partial[0]['config'] if best_partial else {}
            base_trail_cfg = best_trail[0]['config'] if best_trail else {}

            combined = {**prev_best_sl_tp}
            combined['fusion_mode'] = base_mode_cfg.get('fusion_mode', 'c6_veto_4')
            if best_veto: combined['veto_threshold'] = best_veto[0]['config'].get('veto_threshold', 25)
            if best_kdj_bonus: combined['kdj_bonus'] = best_kdj_bonus[0]['config'].get('kdj_bonus', 0.09)
            combined['short_sl'] = base_sl_cfg.get('short_sl', -0.25)
            combined['long_sl'] = base_sl_cfg.get('long_sl', -0.08)
            combined['short_tp'] = base_tp_cfg.get('short_tp', 0.60)
            combined['long_tp'] = base_tp_cfg.get('long_tp', 0.30)
            if base_trail_cfg:
                combined['short_trail'] = base_trail_cfg.get('short_trail', 0.25)
                combined['long_trail'] = base_trail_cfg.get('long_trail', 0.20)
                combined['trail_pullback'] = base_trail_cfg.get('trail_pullback', 0.60)
            if base_partial_cfg:
                combined['partial_tp_1'] = base_partial_cfg.get('partial_tp_1', 0.20)
                combined['partial_tp_1_pct'] = base_partial_cfg.get('partial_tp_1_pct', 0.30)
            combined['tag'] = '组合A_最佳模式+SL+TP+分段'
            fine_variants.append(combined)

        # 组合2: 全局TOP1参数 + 最佳分段
        if best_partial:
            combined2 = {**top1_cfg}
            combined2['partial_tp_1'] = best_partial[0]['config'].get('partial_tp_1', 0.20)
            combined2['partial_tp_1_pct'] = best_partial[0]['config'].get('partial_tp_1_pct', 0.30)
            combined2['tag'] = '组合B_TOP1+最佳分段'
            fine_variants.append(combined2)

        # 组合3: 最佳模式 + ATR止损 + 最佳TP
        if best_atr and best_tp:
            combined3 = {**prev_best_sl_tp}
            combined3['use_atr_sl'] = True
            combined3['atr_sl_mult'] = best_atr[0]['config'].get('atr_sl_mult', 3.0)
            combined3['short_tp'] = best_tp[0]['config'].get('short_tp', 0.60)
            combined3['long_tp'] = best_tp[0]['config'].get('long_tp', 0.30)
            if best_mode: combined3['fusion_mode'] = best_mode[0]['config'].get('fusion_mode', 'c6_veto_4')
            combined3['tag'] = '组合C_ATR止损+最佳TP'
            fine_variants.append(combined3)

        # 组合4: 最佳阈值 + 最佳否决 + 最佳SL/TP
        if best_threshold and best_sl and best_tp:
            combined4 = {**prev_best_sl_tp}
            combined4['sell_threshold'] = best_threshold[0]['config'].get('sell_threshold', 18)
            combined4['short_threshold'] = best_threshold[0]['config'].get('short_threshold', 25)
            combined4['buy_threshold'] = best_threshold[0]['config'].get('buy_threshold', 25)
            combined4['long_threshold'] = best_threshold[0]['config'].get('long_threshold', 40)
            combined4['short_sl'] = best_sl[0]['config'].get('short_sl', -0.25)
            combined4['long_sl'] = best_sl[0]['config'].get('long_sl', -0.08)
            combined4['short_tp'] = best_tp[0]['config'].get('short_tp', 0.60)
            combined4['long_tp'] = best_tp[0]['config'].get('long_tp', 0.30)
            if best_veto: combined4['veto_threshold'] = best_veto[0]['config'].get('veto_threshold', 25)
            combined4['tag'] = '组合D_阈值+否决+SL/TP'
            fine_variants.append(combined4)

        # 组合5: 双段止盈 + 最佳SL/TP
        if best_dual and best_sl and best_tp:
            combined5 = {**best_dual[0]['config']}
            combined5['short_sl'] = best_sl[0]['config'].get('short_sl', -0.25)
            combined5['long_sl'] = best_sl[0]['config'].get('long_sl', -0.08)
            combined5['short_tp'] = best_tp[0]['config'].get('short_tp', 0.60)
            combined5['long_tp'] = best_tp[0]['config'].get('long_tp', 0.30)
            combined5['tag'] = '组合E_双段TP+最佳SL/TP'
            fine_variants.append(combined5)

        # 组合6: 最佳杠杆 + 最佳SL/TP + 最佳分段
        if best_lev and best_sl and best_tp:
            combined6 = {**prev_best_sl_tp}
            combined6['lev'] = best_lev[0]['config'].get('lev', 5)
            combined6['margin_use'] = best_lev[0]['config'].get('margin_use', 0.70)
            combined6['short_sl'] = best_sl[0]['config'].get('short_sl', -0.25)
            combined6['long_sl'] = best_sl[0]['config'].get('long_sl', -0.08)
            combined6['short_tp'] = best_tp[0]['config'].get('short_tp', 0.60)
            combined6['long_tp'] = best_tp[0]['config'].get('long_tp', 0.30)
            combined6['tag'] = '组合F_杠杆+SL/TP+分段'
            fine_variants.append(combined6)

        # 组合7: TOP1完整配置 + 否决削弱优化
        if best_dampen:
            combined7 = {**top1_cfg}
            combined7['veto_dampen'] = best_dampen[0]['config'].get('veto_dampen', 0.30)
            combined7['tag'] = '组合G_TOP1+否决削弱'
            fine_variants.append(combined7)

        # 组合8: KDJ门控 + 最佳SL/TP + 分段
        if best_gate and best_sl and best_tp:
            combined8 = {**prev_best_sl_tp}
            combined8['fusion_mode'] = 'kdj_gate'
            combined8['kdj_gate_threshold'] = best_gate[0]['config'].get('kdj_gate_threshold', 10)
            combined8['short_sl'] = best_sl[0]['config'].get('short_sl', -0.25)
            combined8['long_sl'] = best_sl[0]['config'].get('long_sl', -0.08)
            combined8['short_tp'] = best_tp[0]['config'].get('short_tp', 0.60)
            combined8['long_tp'] = best_tp[0]['config'].get('long_tp', 0.30)
            combined8['tag'] = '组合H_KDJ门控+SL/TP'
            fine_variants.append(combined8)

        # 组合9: KDJ择时 + 最佳SL/TP + 分段
        if best_timing and best_sl and best_tp:
            combined9 = {**prev_best_sl_tp}
            combined9['fusion_mode'] = 'kdj_timing'
            combined9['kdj_strong_mult'] = best_timing[0]['config'].get('kdj_strong_mult', 1.25)
            combined9['kdj_normal_mult'] = best_timing[0]['config'].get('kdj_normal_mult', 1.12)
            combined9['kdj_reverse_mult'] = best_timing[0]['config'].get('kdj_reverse_mult', 0.70)
            combined9['short_sl'] = best_sl[0]['config'].get('short_sl', -0.25)
            combined9['long_sl'] = best_sl[0]['config'].get('long_sl', -0.08)
            combined9['short_tp'] = best_tp[0]['config'].get('short_tp', 0.60)
            combined9['long_tp'] = best_tp[0]['config'].get('long_tp', 0.30)
            combined9['tag'] = '组合I_KDJ择时+SL/TP'
            fine_variants.append(combined9)

        # 在最优时间框架上测试精细组合
        fine_results = []
        tf_for_fine = top1_tf
        print(f"\n  在 {tf_for_fine} 上测试 {len(fine_variants)} 种精细组合...")

        for var in fine_variants:
            config = {**f12_base, **var}
            config['name'] = var['tag']
            tf_h = tf_hours.get(tf_for_fine, 1)
            config['short_max_hold'] = max(6, int(var.get('short_max_hold', 72) / tf_h))
            config['long_max_hold'] = max(6, int(var.get('long_max_hold', 72) / tf_h))
            config['cooldown'] = max(1, int(var.get('cooldown', 4) / tf_h))
            config['spot_cooldown'] = max(2, int(var.get('spot_cooldown', 12) / tf_h))

            r = run_strategy(all_data[tf_for_fine], all_signals[tf_for_fine],
                           config, tf=tf_for_fine, trade_days=trade_days)
            fine_results.append({
                'tf': tf_for_fine,
                'tag': var['tag'],
                'alpha': r['alpha'],
                'strategy_return': r['strategy_return'],
                'buy_hold_return': r['buy_hold_return'],
                'max_drawdown': r['max_drawdown'],
                'total_trades': r['total_trades'],
                'liquidations': r['liquidations'],
                'fees': r.get('fees', {}).get('total_costs', 0),
                'final_total': r.get('final_total', 0),
                'config': var,
                'full_result': r,
            })

        fine_results.sort(key=lambda x: x['alpha'], reverse=True)

        print(f"\n  精细组合TOP10:")
        print(f"  {'排名':>4} {'参数标签':<45} {'Alpha':>10} {'收益':>12} {'回撤':>8} {'交易':>6}")
        print('  ' + '-' * 100)
        for i, r in enumerate(fine_results[:10]):
            star = ' ★★★' if i == 0 else ''
            print(f"  #{i+1:>3} {r['tag']:<45} {r['alpha']:>+9.2f}% "
                  f"{r['strategy_return']:>+11.2f}% {r['max_drawdown']:>7.2f}% "
                  f"{r['total_trades']:>5}{star}")

        all_opt_results.extend(fine_results)
        all_opt_results.sort(key=lambda x: x['alpha'], reverse=True)

    # ============ Phase E: 多周期联合决策回测 ============
    print(f"\n{'=' * 120}")
    print(f"  Phase E: 多周期联合决策回测 (加权共识 + 共振链检测)")
    print(f"{'=' * 120}")

    # 取全局最优参数作为基础配置
    best_cfg_for_multi = all_opt_results[0]['config'] if all_opt_results else prev_best_sl_tp
    best_single_tf = all_opt_results[0]['tf'] if all_opt_results else '4h'
    best_single_alpha = all_opt_results[0]['alpha'] if all_opt_results else 0

    print(f"\n  单TF最优基线: {best_single_tf} α={best_single_alpha:+.2f}%")
    print(f"  基础参数: {best_cfg_for_multi.get('tag', '默认')}")

    # 预计算所有TF的评分索引
    print(f"\n  [E1] 预计算各TF评分索引...")
    tf_score_index = _build_tf_score_index(all_data, all_signals, available_tfs, {**f12_base, **best_cfg_for_multi})
    for tf in available_tfs:
        n_scores = len(tf_score_index.get(tf, {}))
        print(f"    {tf}: {n_scores} 个评分点")

    # 定义多种TF组合方案
    multi_tf_combos = []

    # 方案1: 全TF (所有可用周期)
    multi_tf_combos.append(('全周期', available_tfs))

    # 方案2: 大周期为主 (≥1h)
    large_only = [tf for tf in available_tfs if _MTF_MINUTES.get(tf, 0) >= 60]
    if len(large_only) >= 3:
        multi_tf_combos.append(('大周期(≥1h)', large_only))

    # 方案3: 核心周期 (30m, 1h, 4h, 8h, 24h)
    core_tfs = [tf for tf in ['30m', '1h', '4h', '8h', '24h'] if tf in available_tfs]
    if len(core_tfs) >= 3:
        multi_tf_combos.append(('核心周期', core_tfs))

    # 方案4: 小+大搭配 (15m, 1h, 4h, 12h)
    balanced_tfs = [tf for tf in ['15m', '1h', '4h', '12h'] if tf in available_tfs]
    if len(balanced_tfs) >= 3:
        multi_tf_combos.append(('均衡搭配', balanced_tfs))

    # 方案5: TOP3单TF周期
    top3_single = [t[0] for t in tf_ranked[:3]]
    if len(top3_single) >= 2:
        multi_tf_combos.append(('TOP3单TF', top3_single))

    # 方案6: 中大周期 (1h, 2h, 4h, 8h, 12h)
    mid_large = [tf for tf in ['1h', '2h', '4h', '8h', '12h'] if tf in available_tfs]
    if len(mid_large) >= 3:
        multi_tf_combos.append(('中大周期', mid_large))

    # 各主TF x 各组合方案
    primary_tf_candidates = ['1h', '2h', '4h']
    primary_tf_candidates = [tf for tf in primary_tf_candidates if tf in available_tfs]

    multi_tf_results = []

    print(f"\n  [E2] 运行多周期联合决策回测...")
    print(f"  主TF: {primary_tf_candidates}")
    print(f"  组合方案: {len(multi_tf_combos)}种")
    print(f"\n  {'方案':<25} {'主TF':>5} {'辅助TFs':<45} {'Alpha':>10} {'收益':>12} {'回撤':>8} {'交易':>6}")
    print('  ' + '-' * 120)

    for combo_name, combo_tfs in multi_tf_combos:
        for ptf in primary_tf_candidates:
            if ptf not in all_data:
                continue

            config = {**f12_base, **best_cfg_for_multi}
            config['name'] = f'多TF_{combo_name}@{ptf}'

            tf_h = tf_hours.get(ptf, 1)
            config['short_max_hold'] = max(6, int(best_cfg_for_multi.get('short_max_hold', 72) / tf_h))
            config['long_max_hold'] = max(6, int(best_cfg_for_multi.get('long_max_hold', 72) / tf_h))
            config['cooldown'] = max(1, int(best_cfg_for_multi.get('cooldown', 4) / tf_h))
            config['spot_cooldown'] = max(2, int(best_cfg_for_multi.get('spot_cooldown', 12) / tf_h))

            r = run_strategy_multi_tf(
                all_data[ptf], tf_score_index, combo_tfs, config,
                primary_tf=ptf, trade_days=trade_days
            )
            fees = r.get('fees', {})
            entry = {
                'combo_name': combo_name,
                'primary_tf': ptf,
                'decision_tfs': combo_tfs,
                'alpha': r['alpha'],
                'strategy_return': r['strategy_return'],
                'buy_hold_return': r['buy_hold_return'],
                'max_drawdown': r['max_drawdown'],
                'total_trades': r['total_trades'],
                'liquidations': r['liquidations'],
                'fees': fees.get('total_costs', 0),
                'final_total': r.get('final_total', 0),
                'full_result': r,
            }
            multi_tf_results.append(entry)

            vs_single = r['alpha'] - best_single_alpha
            marker = ' ★' if vs_single > 0 else ''
            print(f"  {combo_name:<25} {ptf:>5} {','.join(combo_tfs):<45} "
                  f"{r['alpha']:>+9.2f}% {r['strategy_return']:>+11.2f}% "
                  f"{r['max_drawdown']:>7.2f}% {r['total_trades']:>5}{marker}")

    multi_tf_results.sort(key=lambda x: x['alpha'], reverse=True)

    print(f"\n  {'─' * 120}")
    print(f"  多周期联合决策 TOP5:")
    for i, r in enumerate(multi_tf_results[:5]):
        vs = r['alpha'] - best_single_alpha
        print(f"    #{i+1} {r['combo_name']}@{r['primary_tf']} "
              f"α={r['alpha']:+.2f}% (vs单TF最优: {vs:+.2f}%) "
              f"回撤={r['max_drawdown']:.2f}% 交易={r['total_trades']}")

    if multi_tf_results:
        best_multi = multi_tf_results[0]
        vs = best_multi['alpha'] - best_single_alpha
        print(f"\n  ★ 多周期最优: {best_multi['combo_name']}@{best_multi['primary_tf']}")
        print(f"    Alpha: {best_multi['alpha']:+.2f}% (vs单TF: {vs:+.2f}%)")
        print(f"    收益: {best_multi['strategy_return']:+.2f}% | 回撤: {best_multi['max_drawdown']:.2f}%")
        print(f"    参与TFs: {','.join(best_multi['decision_tfs'])}")

    # ============ 保存结果 ============
    print(f"\n[5/5] 保存结果...")

    global_best = all_opt_results[0] if all_opt_results else None

    # 清理不可序列化
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()
                    if k != 'full_result'}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return str(obj)
        return obj

    output = {
        'description': f'六书融合多时间框架优化 · 最近{trade_days}天 · 含KDJ',
        'run_time': datetime.now().isoformat(),
        'available_timeframes': available_tfs,
        'total_variants_tested': len(all_opt_results),
        'trade_days': trade_days,
        'previous_best_alpha': 86.69,

        'baseline_by_tf': [{
            'tf': tf,
            'alpha': tf_baseline_results[tf]['alpha'],
            'strategy_return': tf_baseline_results[tf]['strategy_return'],
            'buy_hold_return': tf_baseline_results[tf]['buy_hold_return'],
            'max_drawdown': tf_baseline_results[tf]['max_drawdown'],
            'total_trades': tf_baseline_results[tf]['total_trades'],
        } for tf in available_tfs if tf in tf_baseline_results],

        'top_timeframes': top_tfs,

        'global_top30': [{
            'rank': i + 1,
            'tf': r['tf'],
            'tag': r['tag'],
            'alpha': r['alpha'],
            'strategy_return': r['strategy_return'],
            'max_drawdown': r['max_drawdown'],
            'total_trades': r['total_trades'],
            'fees': r['fees'],
            'config': r['config'],
        } for i, r in enumerate(all_opt_results[:30])],

        'global_best': {
            'tf': global_best['tf'],
            'tag': global_best['tag'],
            'alpha': global_best['alpha'],
            'strategy_return': global_best['strategy_return'],
            'max_drawdown': global_best['max_drawdown'],
            'total_trades': global_best['total_trades'],
            'config': global_best['config'],
        } if global_best else None,

        # Phase E: 多周期联合决策结果
        'multi_tf_results': [{
            'rank': i + 1,
            'combo_name': r['combo_name'],
            'primary_tf': r['primary_tf'],
            'decision_tfs': r['decision_tfs'],
            'alpha': r['alpha'],
            'strategy_return': r['strategy_return'],
            'buy_hold_return': r.get('buy_hold_return', 0),
            'max_drawdown': r['max_drawdown'],
            'total_trades': r['total_trades'],
            'fees': r['fees'],
        } for i, r in enumerate(multi_tf_results[:20])],
        'multi_tf_best': {
            'combo_name': multi_tf_results[0]['combo_name'],
            'primary_tf': multi_tf_results[0]['primary_tf'],
            'decision_tfs': multi_tf_results[0]['decision_tfs'],
            'alpha': multi_tf_results[0]['alpha'],
            'strategy_return': multi_tf_results[0]['strategy_return'],
            'max_drawdown': multi_tf_results[0]['max_drawdown'],
            'total_trades': multi_tf_results[0]['total_trades'],
            'vs_single_tf': multi_tf_results[0]['alpha'] - best_single_alpha,
        } if multi_tf_results else None,
    }

    # 添加完整best trades/history
    if global_best and 'full_result' in global_best:
        output['global_best_trades'] = global_best['full_result'].get('trades', [])
        output['global_best_history'] = global_best['full_result'].get('history', [])
        output['global_best_fees'] = global_best['full_result'].get('fees', {})

    output_clean = clean_for_json(output)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'optimize_six_book_result.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output_clean, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果已保存: {path}")

    # 最终总结
    print(f"\n{'=' * 120}")
    print(f"  六书融合优化完成总结")
    print(f"{'=' * 120}")
    print(f"\n  测试时间框架: {len(available_tfs)}个 ({', '.join(available_tfs)})")
    print(f"  参数变体总数: {len(all_opt_results)}")
    print(f"  五书前最优:   α=+86.69%")
    if global_best:
        improvement = global_best['alpha'] - 86.69
        print(f"\n  ★ 六书全局最优策略:")
        print(f"     时间框架: {global_best['tf']}")
        print(f"     参数标签: {global_best['tag']}")
        print(f"     Alpha:    {global_best['alpha']:+.2f}%")
        print(f"     策略收益: {global_best['strategy_return']:+.2f}%")
        print(f"     最大回撤: {global_best['max_drawdown']:.2f}%")
        print(f"     交易次数: {global_best['total_trades']}")
        print(f"\n     vs 五书最优: {improvement:+.2f}% {'★ 超越!' if improvement > 0 else '未超越'}")

        cfg = global_best['config']
        print(f"\n  最优参数:")
        print(f"     融合模式:   {cfg.get('fusion_mode', 'c6_veto_4')}")
        print(f"     空头止损:   {cfg.get('short_sl', -0.25)*100:.0f}%")
        print(f"     空头止盈:   {cfg.get('short_tp', 0.60)*100:.0f}%")
        print(f"     多头止损:   {cfg.get('long_sl', -0.08)*100:.0f}%")
        print(f"     多头止盈:   {cfg.get('long_tp', 0.30)*100:.0f}%")
        print(f"     追踪止盈:   {cfg.get('short_trail', 0.25)*100:.0f}%")
        print(f"     回撤比例:   {cfg.get('trail_pullback', 0.60)*100:.0f}%")
        print(f"     最大持仓:   {cfg.get('short_max_hold', 72)} bars")
        if cfg.get('use_partial_tp'): print(f"     分段止盈:   @{cfg.get('partial_tp_1',0.2)*100:.0f}% 平{cfg.get('partial_tp_1_pct',0.3)*100:.0f}%")
        if cfg.get('use_partial_tp_2'): print(f"     二段止盈:   @{cfg.get('partial_tp_2',0.5)*100:.0f}% 平{cfg.get('partial_tp_2_pct',0.3)*100:.0f}%")
        if cfg.get('use_atr_sl'): print(f"     ATR止损:    {cfg.get('atr_sl_mult', 3.0)}x")
        if cfg.get('kdj_bonus'): print(f"     KDJ奖励:    {cfg.get('kdj_bonus', 0.09)*100:.0f}%")
        if cfg.get('veto_threshold'): print(f"     否决阈值:   {cfg.get('veto_threshold', 25)}")

    # 多周期联合决策对比
    if multi_tf_results:
        best_m = multi_tf_results[0]
        vs_single = best_m['alpha'] - best_single_alpha
        print(f"\n  ★ 多周期联合决策最优:")
        print(f"     方案:      {best_m['combo_name']}@{best_m['primary_tf']}")
        print(f"     参与TFs:   {','.join(best_m['decision_tfs'])}")
        print(f"     Alpha:     {best_m['alpha']:+.2f}%")
        print(f"     策略收益:  {best_m['strategy_return']:+.2f}%")
        print(f"     最大回撤:  {best_m['max_drawdown']:.2f}%")
        print(f"     交易次数:  {best_m['total_trades']}")
        print(f"\n     vs 单TF最优({best_single_tf}): {vs_single:+.2f}% "
              f"{'★ 多周期更优!' if vs_single > 0 else '单TF更优'}")

    return output


if __name__ == '__main__':
    main()
