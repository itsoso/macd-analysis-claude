"""
合约做空策略 Phase 3 — 基于S策略深度分析的精细优化

Phase 2关键发现:
  S策略收益+144%, 但63笔空仓净亏-$738(胜率2%)
  所有利润来源=现货减仓择时(隔堆信号触发ETH减仓, 然后ETH下跌获利)
  空仓问题: 3x杠杆+短持仓+高频开平→滑点和手续费吃掉利润

优化方向:
  1. 保留并强化现货减仓择时(核心利润来源)
  2. 大幅优化空仓: 更严格过滤/降低杠杆/追踪止盈/更长持仓
  3. 参数网格搜索找最优配置

策略变体:
  S0: S原版(对照基线)
  S1: 纯现货减仓(不开空仓) — 验证空仓是否有正贡献
  S2: 选择性做空 — 只在极强信号时做空, 2x杠杆, 追踪止盈
  S3: 4h级联做空 — 隔堆信号+4h趋势确认才开空
  S4: 极值区做空 — 隔堆信号+KDJ/CCI极值才开空
  S5: 长持空仓 — 更宽止损+追踪止盈, 让利润奔跑
  S6: 最优融合 — 综合S1-S5最佳元素
"""

import pandas as pd
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binance_fetcher import fetch_binance_klines
from indicators import add_all_indicators
from divergence import ComprehensiveAnalyzer
from strategy_enhanced import (
    analyze_signals_enhanced, get_realtime_indicators,
    get_signal_at, DEFAULT_SIG, fetch_all_data
)
from strategy_futures import (
    FuturesEngine, FuturesPosition,
    _merge_signals
)
from strategy_futures_v2 import get_tf_signal, get_trend_info


def _calc_precision_score(sig, trend):
    """计算隔堆精确信号分数(复用S策略核心逻辑)"""
    has_separated = (sig.get('separated_top', 0) >= 1 or
                     sig.get('sep_divs_top', 0) >= 1)
    has_area_div = sig.get('area_top_div', 0) >= 1
    has_zero_return_sep = (sig.get('sep_divs_top', 0) >= 1 and
                           sig.get('zero_returns_top', 0) >= 1)
    has_dif_div_trend = (sig.get('dif_top_div', 0) >= 1 and
                         trend['is_downtrend'])
    has_double_separated = sig.get('sep_divs_top', 0) >= 2

    score = 0
    parts = []
    if has_double_separated:
        score += 25; parts.append("双隔堆")
    elif has_separated:
        score += 15; parts.append("隔堆")
    if has_zero_return_sep:
        score += 12; parts.append("零轴+隔堆")
    if has_area_div:
        score += 10; parts.append("面积背离")
    if has_dif_div_trend:
        score += 8; parts.append("DIF+↓趋势")
    if sig.get('exhaust_sell'):
        score += 15; parts.append("背驰")
    if trend['is_downtrend']:
        score = int(score * 1.2)
    return score, parts


def _calc_bottom_score(sig):
    """计算底部信号分数"""
    score = 0
    if sig.get('separated_bottom', 0) >= 1 or sig.get('sep_divs_bottom', 0) >= 1:
        score += 15
    if sig.get('exhaust_buy'): score += 15
    if sig.get('sep_divs_bottom', 0) >= 2: score += 10
    if sig['bottom'] >= 30: score += sig['bottom'] * 0.15
    return score


# ======================================================
#   S0: 原版S策略(基线)
# ======================================================
def run_S0(data, signals_all):
    eng = FuturesEngine("S0: 原版基线", max_leverage=3)
    main_df = data['1h']
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[0]
    cooldown = 0; min_r = 0.05

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
        eng.check_liquidation(price, dt); eng.charge_funding(price, dt)
        if cooldown > 0: cooldown -= 1

        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)
        ps, parts = _calc_precision_score(sig, trend)
        total = eng.total_value(price)
        eth_r = eng.spot_eth * price / total if total > 0 else 0

        if cooldown == 0:
            if ps >= 15 and not eng.futures_short:
                if eth_r > min_r + 0.05:
                    sr = min(0.15 + ps * 0.005, 0.5)
                    sr = min(sr, (eth_r - min_r) * 0.9)
                    if sr > 0.05:
                        eng.spot_sell(price, dt, sr, f"减仓 {','.join(parts[:2])}")
                lev = 3 if ps >= 25 else 2
                margin = eng.available_margin() * (0.5 if ps >= 25 else 0.3)
                eng.open_short(price, dt, margin, lev,
                    f"做空{lev}x PS={ps} {','.join(parts[:3])}")
                cooldown = 8

            bs = _calc_bottom_score(sig)
            if eng.futures_short and bs >= 12:
                eng.close_short(price, dt, f"底信号 BS={bs:.0f}")
                cooldown = 6

            if eng.futures_short:
                pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
                if pnl_r > 0.7:
                    eng.close_short(price, dt, f"止盈{pnl_r*100:.0f}%")
                    cooldown = 4
                elif pnl_r < -0.35:
                    eng.close_short(price, dt, f"止损{pnl_r*100:.0f}%")
                    cooldown = 10

        if idx % 4 == 0: eng.record_history(dt, price)

    if eng.futures_short: eng.close_short(main_df['close'].iloc[-1], main_df.index[-1], "结束")
    return eng.get_result(main_df)


# ======================================================
#   S1: 纯现货减仓(不开空仓)
# ======================================================
def run_S1(data, signals_all):
    """验证: 不开空仓, 只用隔堆信号做现货减仓择时"""
    eng = FuturesEngine("S1: 纯现货减仓(无空)", max_leverage=1)
    main_df = data['1h']
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[0]
    cooldown = 0; min_r = 0.05

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
        if cooldown > 0: cooldown -= 1

        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)
        ps, parts = _calc_precision_score(sig, trend)
        total = eng.total_value(price)
        eth_r = eng.spot_eth * price / total if total > 0 else 0

        if cooldown == 0 and ps >= 15:
            if eth_r > min_r + 0.05:
                sr = min(0.15 + ps * 0.005, 0.5)
                sr = min(sr, (eth_r - min_r) * 0.9)
                if sr > 0.05:
                    eng.spot_sell(price, dt, sr, f"减仓 PS={ps} {','.join(parts[:2])}")
                    cooldown = 8

            # 底部加仓
            bs = _calc_bottom_score(sig)
            if bs >= 20 and eth_r < 0.35:
                invest = eng.available_usdt() * 0.1
                eng.spot_buy(price, dt, invest, f"加仓 BS={bs:.0f}")
                cooldown = 10

        if idx % 4 == 0: eng.record_history(dt, price)
    return eng.get_result(main_df)


# ======================================================
#   S2: 选择性做空(极强信号+2x+追踪)
# ======================================================
def run_S2(data, signals_all):
    """只在极强隔堆信号做空, 2x杠杆, 追踪止盈"""
    eng = FuturesEngine("S2: 选择性做空+追踪", max_leverage=2)
    main_df = data['1h']
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[0]
    cooldown = 0; min_r = 0.05; max_pnl_r = 0

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
        eng.check_liquidation(price, dt); eng.charge_funding(price, dt)
        if cooldown > 0: cooldown -= 1

        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)
        ps, parts = _calc_precision_score(sig, trend)
        total = eng.total_value(price)
        eth_r = eng.spot_eth * price / total if total > 0 else 0

        if cooldown == 0:
            # 现货减仓(所有信号)
            if ps >= 15 and eth_r > min_r + 0.05:
                sr = min(0.15 + ps * 0.005, 0.5)
                sr = min(sr, (eth_r - min_r) * 0.9)
                if sr > 0.05:
                    eng.spot_sell(price, dt, sr, f"减仓 PS={ps}")
                    cooldown = 6

            # 做空(只在极强信号: PS >= 25 = 双隔堆/零轴+隔堆/背驰级别)
            if ps >= 25 and not eng.futures_short and trend['is_downtrend']:
                margin = eng.available_margin() * 0.4
                eng.open_short(price, dt, margin, 2,
                    f"精选做空 PS={ps} {','.join(parts[:2])}")
                max_pnl_r = 0
                cooldown = 10

            # 底信号平空
            bs = _calc_bottom_score(sig)
            if eng.futures_short and bs >= 15:
                eng.close_short(price, dt, f"底信号 BS={bs:.0f}")
                max_pnl_r = 0; cooldown = 8

        # 追踪止盈(每K线检查)
        if eng.futures_short:
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
            max_pnl_r = max(max_pnl_r, pnl_r)
            if max_pnl_r >= 0.4:
                trail = max_pnl_r * 0.7
                if pnl_r < trail:
                    eng.close_short(price, dt,
                        f"追踪止盈 max={max_pnl_r*100:.0f}% now={pnl_r*100:.0f}%")
                    max_pnl_r = 0; cooldown = 5
            if pnl_r < -0.4:
                eng.close_short(price, dt, f"止损{pnl_r*100:.0f}%")
                max_pnl_r = 0; cooldown = 10

        if idx % 4 == 0: eng.record_history(dt, price)

    if eng.futures_short: eng.close_short(main_df['close'].iloc[-1], main_df.index[-1], "结束")
    return eng.get_result(main_df)


# ======================================================
#   S3: 4h级联做空
# ======================================================
def run_S3(data, signals_all):
    """隔堆信号+4h趋势确认才开空"""
    eng = FuturesEngine("S3: 4h级联做空", max_leverage=3)
    main_df = data['1h']
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[0]
    cooldown = 0; min_r = 0.05; max_pnl_r = 0

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
        eng.check_liquidation(price, dt); eng.charge_funding(price, dt)
        if cooldown > 0: cooldown -= 1

        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)
        sig_4h = get_tf_signal(signals_all, '4h', dt)
        ps, parts = _calc_precision_score(sig, trend)
        total = eng.total_value(price)
        eth_r = eng.spot_eth * price / total if total > 0 else 0

        # 4h级别确认
        big_tf_sell = (sig_4h['top'] >= 15 or sig_4h.get('exhaust_sell') or
                       sig_4h.get('sep_divs_top', 0) >= 1)

        if cooldown == 0:
            # 现货减仓(隔堆信号即可)
            if ps >= 15 and eth_r > min_r + 0.05:
                sr = min(0.15 + ps * 0.005, 0.5)
                sr = min(sr, (eth_r - min_r) * 0.9)
                if sr > 0.05:
                    eng.spot_sell(price, dt, sr, f"减仓 PS={ps}")
                    cooldown = 6

            # 做空: 必须有4h确认
            if ps >= 18 and big_tf_sell and not eng.futures_short:
                lev = 3 if ps >= 30 else 2
                margin = eng.available_margin() * (0.45 if ps >= 30 else 0.3)
                eng.open_short(price, dt, margin, lev,
                    f"4h级联做空{lev}x PS={ps} 4h={sig_4h['top']:.0f}")
                max_pnl_r = 0; cooldown = 8

            bs = _calc_bottom_score(sig)
            if eng.futures_short and bs >= 12:
                eng.close_short(price, dt, f"底信号 BS={bs:.0f}")
                max_pnl_r = 0; cooldown = 6

        # 追踪止盈
        if eng.futures_short:
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
            max_pnl_r = max(max_pnl_r, pnl_r)
            if max_pnl_r >= 0.35:
                trail = max_pnl_r * 0.65
                if pnl_r < trail:
                    eng.close_short(price, dt, f"追踪 max={max_pnl_r*100:.0f}%")
                    max_pnl_r = 0; cooldown = 5
            if pnl_r < -0.35:
                eng.close_short(price, dt, f"止损{pnl_r*100:.0f}%")
                max_pnl_r = 0; cooldown = 10

        if idx % 4 == 0: eng.record_history(dt, price)

    if eng.futures_short: eng.close_short(main_df['close'].iloc[-1], main_df.index[-1], "结束")
    return eng.get_result(main_df)


# ======================================================
#   S4: 极值区做空
# ======================================================
def run_S4(data, signals_all):
    """隔堆信号+KDJ/CCI极值才开空"""
    eng = FuturesEngine("S4: 极值区做空", max_leverage=3)
    main_df = data['1h']
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[0]
    cooldown = 0; min_r = 0.05; max_pnl_r = 0

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
        ind = get_realtime_indicators(main_df, idx)
        eng.check_liquidation(price, dt); eng.charge_funding(price, dt)
        if cooldown > 0: cooldown -= 1

        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)
        ps, parts = _calc_precision_score(sig, trend)
        total = eng.total_value(price)
        eth_r = eng.spot_eth * price / total if total > 0 else 0

        k = ind.get('K'); cci = ind.get('CCI'); rsi = ind.get('RSI6')
        extreme = 0
        if k and k > 70: extreme += 1
        if cci and cci > 80: extreme += 1
        if rsi and rsi > 65: extreme += 1

        if cooldown == 0:
            # 现货减仓(隔堆信号即可)
            if ps >= 15 and eth_r > min_r + 0.05:
                sr = min(0.15 + ps * 0.005, 0.5)
                sr = min(sr, (eth_r - min_r) * 0.9)
                if sr > 0.05:
                    eng.spot_sell(price, dt, sr, f"减仓 PS={ps}")
                    cooldown = 6

            # 做空: 隔堆信号 + 至少1个极值指标
            if ps >= 15 and extreme >= 1 and not eng.futures_short:
                lev = 3 if (ps >= 25 and extreme >= 2) else 2
                margin = eng.available_margin() * (0.45 if extreme >= 2 else 0.3)
                eng.open_short(price, dt, margin, lev,
                    f"极值做空{lev}x PS={ps} x{extreme} K={k or 0:.0f}")
                max_pnl_r = 0; cooldown = 8

            bs = _calc_bottom_score(sig)
            k_low = k and k < 30
            if eng.futures_short and (bs >= 12 or k_low):
                eng.close_short(price, dt, f"底信号 BS={bs:.0f}")
                max_pnl_r = 0; cooldown = 6

        if eng.futures_short:
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
            max_pnl_r = max(max_pnl_r, pnl_r)
            if max_pnl_r >= 0.35:
                trail = max_pnl_r * 0.65
                if pnl_r < trail:
                    eng.close_short(price, dt, f"追踪 max={max_pnl_r*100:.0f}%")
                    max_pnl_r = 0; cooldown = 5
            if pnl_r < -0.4:
                eng.close_short(price, dt, f"止损{pnl_r*100:.0f}%")
                max_pnl_r = 0; cooldown = 10

        if idx % 4 == 0: eng.record_history(dt, price)

    if eng.futures_short: eng.close_short(main_df['close'].iloc[-1], main_df.index[-1], "结束")
    return eng.get_result(main_df)


# ======================================================
#   S5: 长持空仓
# ======================================================
def run_S5(data, signals_all):
    """更宽止损+追踪止盈, 减少交易频率, 让利润奔跑"""
    eng = FuturesEngine("S5: 长持空仓", max_leverage=2)
    main_df = data['1h']
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[0]
    cooldown = 0; min_r = 0.05; max_pnl_r = 0

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
        eng.check_liquidation(price, dt); eng.charge_funding(price, dt)
        if cooldown > 0: cooldown -= 1

        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)
        ps, parts = _calc_precision_score(sig, trend)
        total = eng.total_value(price)
        eth_r = eng.spot_eth * price / total if total > 0 else 0

        if cooldown == 0:
            if ps >= 15 and eth_r > min_r + 0.05:
                sr = min(0.15 + ps * 0.005, 0.5)
                sr = min(sr, (eth_r - min_r) * 0.9)
                if sr > 0.05:
                    eng.spot_sell(price, dt, sr, f"减仓 PS={ps}")

            # 做空: 强信号+下跌趋势, 2x杠杆
            if ps >= 20 and trend['is_downtrend'] and not eng.futures_short:
                margin = eng.available_margin() * 0.35
                eng.open_short(price, dt, margin, 2,
                    f"长持做空 PS={ps} {','.join(parts[:2])}")
                max_pnl_r = 0
                cooldown = 12  # 更长冷却

            # 只在非常强的底信号时平空
            bs = _calc_bottom_score(sig)
            if eng.futures_short and bs >= 20:
                eng.close_short(price, dt, f"强底信号 BS={bs:.0f}")
                max_pnl_r = 0; cooldown = 12

        # 宽松追踪止盈
        if eng.futures_short:
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
            max_pnl_r = max(max_pnl_r, pnl_r)
            # 盈利超50%才开始追踪, 保留60%
            if max_pnl_r >= 0.5:
                trail = max_pnl_r * 0.6
                if pnl_r < trail:
                    eng.close_short(price, dt, f"追踪 max={max_pnl_r*100:.0f}%")
                    max_pnl_r = 0; cooldown = 8
            # 宽止损-50%
            if pnl_r < -0.5:
                eng.close_short(price, dt, f"止损{pnl_r*100:.0f}%")
                max_pnl_r = 0; cooldown = 12

        if idx % 4 == 0: eng.record_history(dt, price)

    if eng.futures_short: eng.close_short(main_df['close'].iloc[-1], main_df.index[-1], "结束")
    return eng.get_result(main_df)


# ======================================================
#   S6: 最优融合
# ======================================================
def run_S6(data, signals_all):
    """综合最佳元素:
    - 现货: S1的纯减仓逻辑(核心利润)
    - 做空: S2的选择性(PS>=25) + S3的4h确认 + S5的追踪止盈
    - 加仓: 底背驰+地量时少量加仓"""
    eng = FuturesEngine("S6: 最优融合", max_leverage=3)
    main_df = data['1h']
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[0]
    vol_ma20 = main_df['volume'].rolling(20).mean()
    cooldown = 0; min_r = 0.05; max_pnl_r = 0

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
        ind = get_realtime_indicators(main_df, idx)
        eng.check_liquidation(price, dt); eng.charge_funding(price, dt)
        if cooldown > 0: cooldown -= 1

        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)
        sig_4h = get_tf_signal(signals_all, '4h', dt)
        ps, parts = _calc_precision_score(sig, trend)
        total = eng.total_value(price)
        eth_r = eng.spot_eth * price / total if total > 0 else 0

        # 4h确认
        big_tf_sell = (sig_4h['top'] >= 12 or sig_4h.get('exhaust_sell') or
                       sig_4h.get('sep_divs_top', 0) >= 1)

        # 极值指标
        k = ind.get('K'); cci = ind.get('CCI')
        extreme = (k and k > 70) or (cci and cci > 80)

        # 量能
        avg_vol = vol_ma20.iloc[idx] if idx >= 20 else 1
        vol_ratio = main_df['volume'].iloc[idx] / avg_vol if avg_vol > 0 else 1
        is_ground_vol = vol_ratio < 0.3

        if cooldown == 0:
            # === 核心: 现货减仓 (宽松触发, PS>=12即减仓) ===
            if ps >= 12 and eth_r > min_r + 0.03:
                sr = min(0.12 + ps * 0.006, 0.55)
                sr = min(sr, (eth_r - min_r) * 0.9)
                if sr > 0.05:
                    eng.spot_sell(price, dt, sr, f"减仓 PS={ps} {','.join(parts[:2])}")
                    cooldown = 5

            # === 精选做空: 极强信号 + (4h确认 或 极值区) ===
            short_ok = (ps >= 25 and (big_tf_sell or extreme) and
                        trend['is_downtrend'] and not eng.futures_short)
            if short_ok:
                lev = 3 if ps >= 35 else 2
                margin = eng.available_margin() * (0.45 if ps >= 35 else 0.3)
                eng.open_short(price, dt, margin, lev,
                    f"精选做空{lev}x PS={ps} {'4h✓' if big_tf_sell else ''} "
                    f"{'极值' if extreme else ''}")
                max_pnl_r = 0; cooldown = 10

            # === 平空: 强底信号 ===
            bs = _calc_bottom_score(sig)
            if eng.futures_short and bs >= 15:
                eng.close_short(price, dt, f"底信号 BS={bs:.0f}")
                max_pnl_r = 0; cooldown = 8

            # === 底部加仓: 地量+底背驰 ===
            if is_ground_vol and sig.get('exhaust_buy') and eth_r < 0.3:
                invest = eng.available_usdt() * 0.08
                eng.spot_buy(price, dt, invest, f"地量底背驰加仓")
                cooldown = 12

        # 追踪止盈(宽松版)
        if eng.futures_short:
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
            max_pnl_r = max(max_pnl_r, pnl_r)
            if max_pnl_r >= 0.4:
                trail = max_pnl_r * 0.65
                if pnl_r < trail:
                    eng.close_short(price, dt, f"追踪 max={max_pnl_r*100:.0f}%")
                    max_pnl_r = 0; cooldown = 6
            if pnl_r < -0.4:
                eng.close_short(price, dt, f"止损{pnl_r*100:.0f}%")
                max_pnl_r = 0; cooldown = 10

        if idx % 4 == 0: eng.record_history(dt, price)

    if eng.futures_short: eng.close_short(main_df['close'].iloc[-1], main_df.index[-1], "结束")
    return eng.get_result(main_df)


# ======================================================
#   参数网格优化(基于S6框架)
# ======================================================
def run_param_grid(data, signals_all):
    """在S6框架上做参数网格搜索"""
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    vol_ma20 = main_df['volume'].rolling(20).mean()

    params_grid = [
        {'name': 'G1: 低门槛+2x',    'sell_th': 10, 'short_th': 22, 'lev': 2, 'cd': 6,  'sl': 0.35, 'trail_start': 0.3},
        {'name': 'G2: 高门槛+3x',    'sell_th': 18, 'short_th': 30, 'lev': 3, 'cd': 10, 'sl': 0.4,  'trail_start': 0.5},
        {'name': 'G3: 超低门槛+频繁', 'sell_th': 8,  'short_th': 20, 'lev': 2, 'cd': 4,  'sl': 0.3,  'trail_start': 0.25},
        {'name': 'G4: 只减仓+偶尔空', 'sell_th': 10, 'short_th': 35, 'lev': 2, 'cd': 12, 'sl': 0.5,  'trail_start': 0.4},
        {'name': 'G5: 激进3x低门槛',  'sell_th': 12, 'short_th': 20, 'lev': 3, 'cd': 8,  'sl': 0.35, 'trail_start': 0.35},
        {'name': 'G6: 保守长持',      'sell_th': 15, 'short_th': 28, 'lev': 2, 'cd': 15, 'sl': 0.5,  'trail_start': 0.5},
    ]

    grid_results = []
    for p in params_grid:
        eng = FuturesEngine(p['name'], max_leverage=p['lev'])
        eng.spot_eth = eng.initial_eth_value / first_price
        cooldown = 0; min_r = 0.05; max_pnl_r = 0

        for idx in range(20, len(main_df)):
            dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
            eng.check_liquidation(price, dt); eng.charge_funding(price, dt)
            if cooldown > 0: cooldown -= 1

            trend = get_trend_info(data, dt, price)
            sig = _merge_signals(signals_all, dt)
            sig_4h = get_tf_signal(signals_all, '4h', dt)
            ps, parts = _calc_precision_score(sig, trend)
            total = eng.total_value(price)
            eth_r = eng.spot_eth * price / total if total > 0 else 0

            big_tf = (sig_4h['top'] >= 12 or sig_4h.get('exhaust_sell'))
            avg_vol = vol_ma20.iloc[idx] if idx >= 20 else 1
            is_ground = (main_df['volume'].iloc[idx] / avg_vol < 0.3) if avg_vol > 0 else False

            if cooldown == 0:
                if ps >= p['sell_th'] and eth_r > min_r + 0.03:
                    sr = min(0.12 + ps * 0.006, 0.55)
                    sr = min(sr, (eth_r - min_r) * 0.9)
                    if sr > 0.05:
                        eng.spot_sell(price, dt, sr, f"减仓 PS={ps}")
                        cooldown = max(3, p['cd'] // 2)

                if (ps >= p['short_th'] and (big_tf or trend['is_downtrend']) and
                        not eng.futures_short):
                    margin = eng.available_margin() * 0.35
                    eng.open_short(price, dt, margin, p['lev'],
                        f"做空{p['lev']}x PS={ps}")
                    max_pnl_r = 0; cooldown = p['cd']

                bs = _calc_bottom_score(sig)
                if eng.futures_short and bs >= 15:
                    eng.close_short(price, dt, f"底信号")
                    max_pnl_r = 0; cooldown = p['cd']

                if is_ground and sig.get('exhaust_buy') and eth_r < 0.3:
                    eng.spot_buy(price, dt, eng.available_usdt() * 0.08, "地量加仓")
                    cooldown = 12

            if eng.futures_short:
                pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
                max_pnl_r = max(max_pnl_r, pnl_r)
                if max_pnl_r >= p['trail_start']:
                    trail = max_pnl_r * 0.65
                    if pnl_r < trail:
                        eng.close_short(price, dt, f"追踪")
                        max_pnl_r = 0; cooldown = p['cd'] // 2
                if pnl_r < -p['sl']:
                    eng.close_short(price, dt, f"止损")
                    max_pnl_r = 0; cooldown = p['cd']

            if idx % 4 == 0: eng.record_history(dt, price)

        if eng.futures_short:
            eng.close_short(main_df['close'].iloc[-1], main_df.index[-1], "结束")
        grid_results.append(eng.get_result(main_df))

    return grid_results


# ======================================================
def run_all():
    data = fetch_all_data()

    print("\n计算各周期增强信号...")
    signal_windows = {'1h': 168, '2h': 150, '4h': 120, '6h': 100, '8h': 90}
    signals_all = {}
    for tf, df in data.items():
        w = signal_windows.get(tf, 120)
        if len(df) > w:
            signals_all[tf] = analyze_signals_enhanced(df, w)
            print(f"  {tf}: {len(signals_all[tf])} 个信号点")

    strategies = [
        ("S0", run_S0),
        ("S1", run_S1),
        ("S2", run_S2),
        ("S3", run_S3),
        ("S4", run_S4),
        ("S5", run_S5),
        ("S6", run_S6),
    ]

    print(f"\n{'='*150}")
    print(f"  Phase 3: S策略深度优化 · 隔堆做空精细调优")
    print(f"  初始: 100,000 USDT + 价值100,000 USDT的ETH")
    print(f"  关键发现: S原版空仓净亏-$738, 利润全部来自现货减仓择时")
    print(f"{'='*150}")

    results = []
    for name, func in strategies:
        print(f"\n>>> 策略 {name}...")
        r = func(data, signals_all)
        results.append(r)
        f = r.get('fees', {})
        liq = f" 强平:{r['liquidations']}" if r['liquidations'] > 0 else ""
        print(f"    α: {r['alpha']:+.2f}% | 收益: {r['strategy_return']:+.2f}% | "
              f"回撤: {r['max_drawdown']:.2f}% | 交易: {r['total_trades']}笔{liq} | "
              f"费用: ${f.get('total_costs', 0):,.0f} ({f.get('fee_drag_pct', 0):.2f}%)")

    # 参数网格
    print(f"\n>>> 参数网格搜索 (6种配置)...")
    grid = run_param_grid(data, signals_all)
    for r in grid:
        f = r.get('fees', {})
        print(f"    {r['name']:<24} α: {r['alpha']:+.2f}% | 交易: {r['total_trades']}笔 | "
              f"费用: ${f.get('total_costs', 0):,.0f} ({f.get('fee_drag_pct', 0):.2f}%)")
    results.extend(grid)

    # 排名
    ranked = sorted(results, key=lambda x: x['alpha'], reverse=True)
    print(f"\n\n{'='*150}")
    print("                    Phase 3 排名 (按超额收益) · 含费用影响")
    print(f"{'='*150}")
    fmt = "{:>3} {:<34} {:>9} {:>9} {:>8} {:>6} {:>10} {:>10} {:>9} {:>12}"
    print(fmt.format("#", "策略", "收益", "超额α", "回撤", "笔数",
                      "总费用", "滑点", "费率%", "最终资产"))
    print("-" * 150)
    for rank, r in enumerate(ranked, 1):
        star = " ★" if rank == 1 else ""
        f = r.get('fees', {})
        print(fmt.format(
            rank, r['name'] + star,
            f"{r['strategy_return']:+.1f}%",
            f"{r['alpha']:+.1f}%",
            f"{r['max_drawdown']:.1f}%",
            str(r['total_trades']),
            f"${f.get('total_fees', 0):,.0f}",
            f"${f.get('slippage_cost', 0):,.0f}",
            f"{f.get('fee_drag_pct', 0):.2f}%",
            f"${r['final_total']:,.0f}",
        ))
    print("=" * 150)

    # S0 vs S1 对比
    s0 = next((r for r in results if r['name'].startswith('S0')), None)
    s1 = next((r for r in results if r['name'].startswith('S1')), None)
    if s0 and s1:
        print(f"\n  空仓价值分析:")
        print(f"    S0(有空仓): α={s0['alpha']:+.2f}%, 费用={s0.get('fees',{}).get('total_costs',0):,.0f}")
        print(f"    S1(无空仓): α={s1['alpha']:+.2f}%, 费用={s1.get('fees',{}).get('total_costs',0):,.0f}")
        diff = s0['alpha'] - s1['alpha']
        print(f"    空仓贡献: {diff:+.2f}% {'(空仓有正贡献)' if diff > 0 else '(空仓拖累收益!)'}")

    # 保存
    output = {
        'phase': 3,
        'futures_results': [{
            'name': r['name'],
            'summary': {k: v for k, v in r.items() if k not in ('trades', 'history')},
            'trades': r['trades'],
            'history': r['history'],
        } for r in results],
        'best_strategy': ranked[0]['name'],
        'ranking': [{'rank': i + 1, 'name': r['name'], 'alpha': r['alpha'],
                      'return': r['strategy_return'], 'max_dd': r['max_drawdown'],
                      'trades': r['total_trades'], 'fees': r.get('fees', {})}
                     for i, r in enumerate(ranked)],
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'strategy_futures_v3_result.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果已保存: {path}")
    return output


if __name__ == '__main__':
    run_all()
