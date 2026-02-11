"""
合约做空策略 Phase 5 — 终极优化

Phase 4冠军: A3(巨额空仓) α=+26.0%, 仅5笔交易
分析: 在$3,279全清ETH + $7K保证金3x做空, 持仓26天, 空仓赚$8,695
优化空间:
  1. A3仅用$7K保证金(可用上限$10K), 浪费30%容量
  2. 未利用2月初的第二轮强信号(TOP=100)
  3. 未在底部开多捕捉反弹
  4. 可以更早减仓(1/13有8h弱信号)

策略变体:
  B1: A3满仓版(用满$10K保证金)
  B2: 双空仓(1/16首空 + 后续强信号追加空)
  B3: 多空双向(空仓+底部做多)
  B4: 分批入场(现货渐进减仓+空仓分批开)
  B5: 早鸟版(1/13弱信号先减仓一部分)
  B6: 终极A3++(融合所有最佳元素)
"""

import pandas as pd
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_enhanced import (
    analyze_signals_enhanced, get_realtime_indicators,
    DEFAULT_SIG, fetch_all_data
)
from strategy_futures import FuturesEngine, _merge_signals
from strategy_futures_v2 import get_tf_signal, get_trend_info
from strategy_futures_v4 import _calc_top_score, _calc_bottom_score


# ======================================================
#   B1: A3满仓版(保证金用到极限)
# ======================================================
def run_B1(data, signals_all):
    """A3逻辑但保证金用满100%"""
    eng = FuturesEngine("B1: A3满仓版", max_leverage=3)
    main_df = data['1h']
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[0]
    short_opened = False; max_pnl_r = 0

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
        eng.check_liquidation(price, dt); eng.charge_funding(price, dt)
        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)
        ts, parts = _calc_top_score(sig, trend)

        if ts >= 15 and not short_opened:
            if eng.spot_eth * price > 1000:
                eng.spot_sell(price, dt, 0.92, f"全清 TS={ts}")
            if not eng.futures_short and ts >= 20:
                # 用满可用保证金
                margin = eng.available_margin() * 1.0
                lev = 3 if ts >= 35 else 2
                eng.open_short(price, dt, margin, lev,
                    f"满仓做空{lev}x TS={ts} {','.join(parts[:3])}")
                max_pnl_r = 0; short_opened = True
        elif short_opened and ts >= 12 and eng.spot_eth * price > 500:
            eng.spot_sell(price, dt, 0.8, "追卖")

        if eng.futures_short:
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
            max_pnl_r = max(max_pnl_r, pnl_r)
            if max_pnl_r >= 1.2:
                trail = max_pnl_r * 0.55
                if pnl_r < trail:
                    eng.close_short(price, dt, f"追踪 max={max_pnl_r*100:.0f}%")
                    max_pnl_r = 0; short_opened = False
            if pnl_r < -0.6:
                eng.close_short(price, dt, f"止损{pnl_r*100:.0f}%")
                max_pnl_r = 0; short_opened = False

        if idx % 4 == 0: eng.record_history(dt, price)
    if eng.futures_short: eng.close_short(main_df['close'].iloc[-1], main_df.index[-1], "结束")
    return eng.get_result(main_df)


# ======================================================
#   B2: 双空仓(两波做空)
# ======================================================
def run_B2(data, signals_all):
    """第一波信号做空 + 第二波强信号追加做空"""
    eng = FuturesEngine("B2: 双空仓", max_leverage=3)
    main_df = data['1h']
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[0]
    max_pnl_r = 0; short_count = 0; cd = 0

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
        eng.check_liquidation(price, dt); eng.charge_funding(price, dt)
        if cd > 0: cd -= 1
        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)
        ts, parts = _calc_top_score(sig, trend)
        bs = _calc_bottom_score(sig, trend)

        if ts >= 15 and eng.spot_eth * price > 500:
            eng.spot_sell(price, dt, 0.92, f"全清 TS={ts}")

        # 开空: 最多2笔
        if (cd == 0 and ts >= 20 and not eng.futures_short and
                short_count < 2 and trend['is_downtrend']):
            margin = eng.available_margin() * 0.95
            lev = 3 if ts >= 35 else 2
            eng.open_short(price, dt, margin, lev,
                f"做空#{short_count+1} {lev}x TS={ts} {','.join(parts[:3])}")
            max_pnl_r = 0; short_count += 1
            cd = 48  # 2天冷却

        # 平空
        if eng.futures_short:
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
            max_pnl_r = max(max_pnl_r, pnl_r)
            # 第一笔空: 宽松追踪(让利润跑)
            if max_pnl_r >= 1.0:
                trail = max_pnl_r * 0.55
                if pnl_r < trail:
                    eng.close_short(price, dt, f"追踪 max={max_pnl_r*100:.0f}%")
                    max_pnl_r = 0; cd = 24
            # 强底信号
            if bs >= 40:
                eng.close_short(price, dt, f"强底 BS={bs:.0f}")
                max_pnl_r = 0; cd = 24
            if pnl_r < -0.6:
                eng.close_short(price, dt, f"止损{pnl_r*100:.0f}%")
                max_pnl_r = 0; cd = 36

        if idx % 4 == 0: eng.record_history(dt, price)
    if eng.futures_short: eng.close_short(main_df['close'].iloc[-1], main_df.index[-1], "结束")
    return eng.get_result(main_df)


# ======================================================
#   B3: 多空双向(空仓+底部做多)
# ======================================================
def run_B3(data, signals_all):
    """做空为主, 底部做多捕捉反弹"""
    eng = FuturesEngine("B3: 多空双向", max_leverage=3)
    main_df = data['1h']
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[0]
    max_pnl_r = 0; trade_count = 0; cd = 0

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
        eng.check_liquidation(price, dt); eng.charge_funding(price, dt)
        if cd > 0: cd -= 1
        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)
        ts, parts = _calc_top_score(sig, trend)
        bs = _calc_bottom_score(sig, trend)

        # 清仓现货
        if ts >= 15 and eng.spot_eth * price > 500:
            eng.spot_sell(price, dt, 0.92, f"全清 TS={ts}")

        if cd == 0 and not eng.futures_short and not eng.futures_long and trade_count < 5:
            # 做空
            if ts >= 20 and trend['is_downtrend']:
                margin = eng.available_margin() * 0.9
                lev = 3 if ts >= 35 else 2
                eng.open_short(price, dt, margin, lev,
                    f"做空{lev}x TS={ts}")
                max_pnl_r = 0; trade_count += 1; cd = 24
            # 做多(底部强信号)
            elif bs >= 35:
                margin = eng.available_margin() * 0.5
                lev = 2
                eng.open_long(price, dt, margin, lev,
                    f"底部做多 BS={bs:.0f}")
                max_pnl_r = 0; trade_count += 1; cd = 12

        # 平空
        if eng.futures_short:
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
            max_pnl_r = max(max_pnl_r, pnl_r)
            if max_pnl_r >= 0.8:
                trail = max_pnl_r * 0.55
                if pnl_r < trail:
                    eng.close_short(price, dt, f"追踪 max={max_pnl_r*100:.0f}%")
                    max_pnl_r = 0; cd = 12
            if bs >= 35:
                eng.close_short(price, dt, f"强底 BS={bs:.0f}")
                max_pnl_r = 0; cd = 8
            if pnl_r < -0.6:
                eng.close_short(price, dt, f"止损")
                max_pnl_r = 0; cd = 24

        # 平多
        if eng.futures_long:
            pnl_r = eng.futures_long.calc_pnl(price) / eng.futures_long.margin
            max_pnl_r = max(max_pnl_r, pnl_r)
            if max_pnl_r >= 0.3:
                trail = max_pnl_r * 0.6
                if pnl_r < trail:
                    eng.close_long(price, dt, f"追踪多 max={max_pnl_r*100:.0f}%")
                    max_pnl_r = 0; cd = 8
            if ts >= 20:
                eng.close_long(price, dt, f"转空 TS={ts}")
                max_pnl_r = 0; cd = 4
            if pnl_r < -0.5:
                eng.close_long(price, dt, f"止损多")
                max_pnl_r = 0; cd = 12

        if idx % 4 == 0: eng.record_history(dt, price)
    if eng.futures_short: eng.close_short(main_df['close'].iloc[-1], main_df.index[-1], "结束")
    if eng.futures_long: eng.close_long(main_df['close'].iloc[-1], main_df.index[-1], "结束")
    return eng.get_result(main_df)


# ======================================================
#   B4: 分批入场
# ======================================================
def run_B4(data, signals_all):
    """现货分3批卖出, 空仓分2批开"""
    eng = FuturesEngine("B4: 分批入场", max_leverage=3)
    main_df = data['1h']
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[0]
    sell_batch = 0; short_batch = 0; max_pnl_r = 0; cd = 0

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
        eng.check_liquidation(price, dt); eng.charge_funding(price, dt)
        if cd > 0: cd -= 1
        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)
        ts, parts = _calc_top_score(sig, trend)
        bs = _calc_bottom_score(sig, trend)

        # 分批现货卖出
        if ts >= 12 and eng.spot_eth * price > 500:
            if sell_batch == 0:
                eng.spot_sell(price, dt, 0.60, f"首批卖出 TS={ts}")
                sell_batch = 1; cd = 4
            elif sell_batch == 1 and cd == 0:
                eng.spot_sell(price, dt, 0.70, f"二批卖出 TS={ts}")
                sell_batch = 2; cd = 4
            elif sell_batch >= 2 and cd == 0:
                eng.spot_sell(price, dt, 0.85, f"清仓 TS={ts}")
                sell_batch = 3

        # 分批做空
        if (cd == 0 and ts >= 20 and trend['is_downtrend'] and
                not eng.futures_short and short_batch < 2):
            margin = eng.available_margin() * 0.9
            lev = 3 if ts >= 35 else 2
            eng.open_short(price, dt, margin, lev,
                f"空仓#{short_batch+1} {lev}x TS={ts}")
            max_pnl_r = 0; short_batch += 1
            cd = 72  # 3天冷却

        if eng.futures_short:
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
            max_pnl_r = max(max_pnl_r, pnl_r)
            if max_pnl_r >= 1.0:
                trail = max_pnl_r * 0.55
                if pnl_r < trail:
                    eng.close_short(price, dt, f"追踪 max={max_pnl_r*100:.0f}%")
                    max_pnl_r = 0; cd = 24
            if bs >= 40:
                eng.close_short(price, dt, f"强底 BS={bs:.0f}")
                max_pnl_r = 0; cd = 24
            if pnl_r < -0.55:
                eng.close_short(price, dt, f"止损")
                max_pnl_r = 0; cd = 36

        if idx % 4 == 0: eng.record_history(dt, price)
    if eng.futures_short: eng.close_short(main_df['close'].iloc[-1], main_df.index[-1], "结束")
    return eng.get_result(main_df)


# ======================================================
#   B5: 早鸟版(更早减仓)
# ======================================================
def run_B5(data, signals_all):
    """1/13弱信号先减仓30%, 1/16强信号再全清+做空"""
    eng = FuturesEngine("B5: 早鸟版", max_leverage=3)
    main_df = data['1h']
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[0]
    max_pnl_r = 0; short_opened = False; early_sold = False

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
        eng.check_liquidation(price, dt); eng.charge_funding(price, dt)
        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)
        ts, parts = _calc_top_score(sig, trend)
        ind = get_realtime_indicators(main_df, idx)

        # 早期弱信号: 降低门槛, 先卖30%
        k = ind.get('K', 50); rsi = ind.get('RSI6', 50)
        early_signal = (ts >= 5 or (k > 75 and rsi > 70))
        if not early_sold and early_signal and eng.spot_eth * price > 20000:
            eng.spot_sell(price, dt, 0.35, f"早鸟减仓 TS={ts} K={k:.0f}")
            early_sold = True

        # 强信号: 全清 + 做空
        if ts >= 15 and not short_opened:
            if eng.spot_eth * price > 1000:
                eng.spot_sell(price, dt, 0.92, f"全清 TS={ts}")
            if not eng.futures_short and ts >= 20:
                margin = eng.available_margin() * 0.95
                lev = 3 if ts >= 35 else 2
                eng.open_short(price, dt, margin, lev,
                    f"满仓做空{lev}x TS={ts} {','.join(parts[:3])}")
                max_pnl_r = 0; short_opened = True
        elif short_opened and eng.spot_eth * price > 500:
            eng.spot_sell(price, dt, 0.8, "追卖")

        if eng.futures_short:
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
            max_pnl_r = max(max_pnl_r, pnl_r)
            if max_pnl_r >= 1.2:
                trail = max_pnl_r * 0.55
                if pnl_r < trail:
                    eng.close_short(price, dt, f"追踪 max={max_pnl_r*100:.0f}%")
                    max_pnl_r = 0; short_opened = False
            if pnl_r < -0.6:
                eng.close_short(price, dt, f"止损")
                max_pnl_r = 0; short_opened = False

        if idx % 4 == 0: eng.record_history(dt, price)
    if eng.futures_short: eng.close_short(main_df['close'].iloc[-1], main_df.index[-1], "结束")
    return eng.get_result(main_df)


# ======================================================
#   B6: 终极A3++(融合所有最佳元素)
# ======================================================
def run_B6(data, signals_all):
    """早鸟减仓 + 满仓做空 + 双空仓 + 底部做多 = 理论最优"""
    eng = FuturesEngine("B6: 终极A3++", max_leverage=3)
    main_df = data['1h']
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[0]
    max_pnl_r = 0; short_count = 0; long_count = 0
    cd_short = 0; cd_long = 0; early_sold = False

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
        eng.check_liquidation(price, dt); eng.charge_funding(price, dt)
        if cd_short > 0: cd_short -= 1
        if cd_long > 0: cd_long -= 1

        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)
        ts, parts = _calc_top_score(sig, trend)
        bs = _calc_bottom_score(sig, trend)
        ind = get_realtime_indicators(main_df, idx)

        # === 1. 早鸟减仓(弱信号就开始卖) ===
        k = ind.get('K', 50); rsi = ind.get('RSI6', 50)
        if not early_sold and (ts >= 5 or (k > 75 and rsi > 70)):
            if eng.spot_eth * price > 20000:
                eng.spot_sell(price, dt, 0.35, f"早鸟 TS={ts} K={k:.0f}")
                early_sold = True

        # === 2. 强信号全清 + 做空 ===
        if ts >= 15 and eng.spot_eth * price > 500:
            eng.spot_sell(price, dt, 0.92, f"全清 TS={ts}")

        # 做空(最多2笔, 间隔48h)
        if (cd_short == 0 and ts >= 20 and trend['is_downtrend'] and
                not eng.futures_short and short_count < 2):
            margin = eng.available_margin() * 0.95
            lev = 3 if ts >= 35 else 2
            eng.open_short(price, dt, margin, lev,
                f"做空#{short_count+1} {lev}x TS={ts} {','.join(parts[:3])}")
            max_pnl_r = 0; short_count += 1; cd_short = 48

        # === 3. 底部做多(捕捉反弹) ===
        if (cd_long == 0 and bs >= 35 and not eng.futures_long and
                not eng.futures_short and long_count < 2):
            margin = eng.available_margin() * 0.4
            eng.open_long(price, dt, margin, 2, f"底部做多 BS={bs:.0f}")
            max_pnl_r = 0; long_count += 1; cd_long = 24

        # === 平空: 宽松追踪 ===
        if eng.futures_short:
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
            max_pnl_r = max(max_pnl_r, pnl_r)
            if max_pnl_r >= 1.0:
                trail = max_pnl_r * 0.55
                if pnl_r < trail:
                    eng.close_short(price, dt, f"追踪 max={max_pnl_r*100:.0f}%")
                    max_pnl_r = 0; cd_short = 24
            if bs >= 40:
                eng.close_short(price, dt, f"强底 BS={bs:.0f}")
                max_pnl_r = 0; cd_short = 12
            if pnl_r < -0.6:
                eng.close_short(price, dt, f"止损")
                max_pnl_r = 0; cd_short = 36

        # === 平多: 快速止盈 ===
        if eng.futures_long:
            pnl_r = eng.futures_long.calc_pnl(price) / eng.futures_long.margin
            max_pnl_r = max(max_pnl_r, pnl_r)
            if max_pnl_r >= 0.25:
                trail = max_pnl_r * 0.6
                if pnl_r < trail:
                    eng.close_long(price, dt, f"止盈多")
                    max_pnl_r = 0; cd_long = 8
            if ts >= 20:
                eng.close_long(price, dt, f"转空信号")
                max_pnl_r = 0; cd_long = 4
            if pnl_r < -0.4:
                eng.close_long(price, dt, f"止损多")
                max_pnl_r = 0; cd_long = 12

        if idx % 4 == 0: eng.record_history(dt, price)
    if eng.futures_short: eng.close_short(main_df['close'].iloc[-1], main_df.index[-1], "结束")
    if eng.futures_long: eng.close_long(main_df['close'].iloc[-1], main_df.index[-1], "结束")
    return eng.get_result(main_df)


# ======================================================
#   参数精调网格(B6框架)
# ======================================================
def run_param_grid(data, signals_all):
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]

    configs = [
        {'name': 'Q1: 满仓单空+无追卖',  'sell_pct': 0.95, 'short_pct': 1.0, 'max_shorts': 1, 'cd': 72, 'early': False, 'do_long': False},
        {'name': 'Q2: 满仓双空+早鸟',    'sell_pct': 0.92, 'short_pct': 0.95, 'max_shorts': 2, 'cd': 48, 'early': True, 'do_long': False},
        {'name': 'Q3: 满仓双空+做多',    'sell_pct': 0.92, 'short_pct': 0.9, 'max_shorts': 2, 'cd': 48, 'early': True, 'do_long': True},
        {'name': 'Q4: 超满仓单空',      'sell_pct': 0.98, 'short_pct': 1.0, 'max_shorts': 1, 'cd': 72, 'early': True, 'do_long': False},
        {'name': 'Q5: 三重空仓',       'sell_pct': 0.92, 'short_pct': 0.9, 'max_shorts': 3, 'cd': 36, 'early': True, 'do_long': False},
        {'name': 'Q6: 保守双空+做多',    'sell_pct': 0.85, 'short_pct': 0.7, 'max_shorts': 2, 'cd': 72, 'early': True, 'do_long': True},
    ]

    results = []
    for p in configs:
        eng = FuturesEngine(p['name'], max_leverage=3)
        eng.spot_eth = eng.initial_eth_value / first_price
        max_pnl_r = 0; short_count = 0; long_count = 0
        cd_s = 0; cd_l = 0; early_sold = False

        for idx in range(20, len(main_df)):
            dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
            eng.check_liquidation(price, dt); eng.charge_funding(price, dt)
            if cd_s > 0: cd_s -= 1
            if cd_l > 0: cd_l -= 1

            trend = get_trend_info(data, dt, price)
            sig = _merge_signals(signals_all, dt)
            ts, parts = _calc_top_score(sig, trend)
            bs = _calc_bottom_score(sig, trend)
            ind = get_realtime_indicators(main_df, idx)

            k = ind.get('K', 50); rsi = ind.get('RSI6', 50)
            if p['early'] and not early_sold and (ts >= 5 or (k > 75 and rsi > 70)):
                if eng.spot_eth * price > 20000:
                    eng.spot_sell(price, dt, 0.35, "早鸟")
                    early_sold = True

            if ts >= 15 and eng.spot_eth * price > 500:
                eng.spot_sell(price, dt, p['sell_pct'], f"清仓")

            if (cd_s == 0 and ts >= 20 and trend['is_downtrend'] and
                    not eng.futures_short and short_count < p['max_shorts']):
                margin = eng.available_margin() * p['short_pct']
                lev = 3 if ts >= 35 else 2
                eng.open_short(price, dt, margin, lev, f"做空{lev}x")
                max_pnl_r = 0; short_count += 1; cd_s = p['cd']

            if p['do_long'] and cd_l == 0 and bs >= 35 and not eng.futures_long and not eng.futures_short and long_count < 2:
                margin = eng.available_margin() * 0.4
                eng.open_long(price, dt, margin, 2, "做多")
                max_pnl_r = 0; long_count += 1; cd_l = 24

            if eng.futures_short:
                pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
                max_pnl_r = max(max_pnl_r, pnl_r)
                if max_pnl_r >= 1.0:
                    if pnl_r < max_pnl_r * 0.55:
                        eng.close_short(price, dt, "追踪"); max_pnl_r = 0; cd_s = 24
                if bs >= 40:
                    eng.close_short(price, dt, "强底"); max_pnl_r = 0; cd_s = 12
                if pnl_r < -0.6:
                    eng.close_short(price, dt, "止损"); max_pnl_r = 0; cd_s = 36

            if eng.futures_long:
                pnl_r = eng.futures_long.calc_pnl(price) / eng.futures_long.margin
                max_pnl_r = max(max_pnl_r, pnl_r)
                if max_pnl_r >= 0.25:
                    if pnl_r < max_pnl_r * 0.6:
                        eng.close_long(price, dt, "止盈多"); max_pnl_r = 0; cd_l = 8
                if ts >= 20:
                    eng.close_long(price, dt, "转空"); max_pnl_r = 0; cd_l = 4
                if pnl_r < -0.4:
                    eng.close_long(price, dt, "止损多"); max_pnl_r = 0; cd_l = 12

            if idx % 4 == 0: eng.record_history(dt, price)
        if eng.futures_short: eng.close_short(main_df['close'].iloc[-1], main_df.index[-1], "结束")
        if eng.futures_long: eng.close_long(main_df['close'].iloc[-1], main_df.index[-1], "结束")
        results.append(eng.get_result(main_df))
    return results


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
        ("B1", run_B1),
        ("B2", run_B2),
        ("B3", run_B3),
        ("B4", run_B4),
        ("B5", run_B5),
        ("B6", run_B6),
    ]

    print(f"\n{'='*140}")
    print(f"  Phase 5: 终极优化 · A3巨额空仓的极致强化")
    print(f"  A3冠军: α=+26.0%, 仅5笔交易, $7K保证金3x做空持26天赚$8,695")
    print(f"{'='*140}")

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

    print(f"\n>>> 参数精调网格 (6种配置)...")
    grid = run_param_grid(data, signals_all)
    for r in grid:
        f = r.get('fees', {})
        print(f"    {r['name']:<22} α: {r['alpha']:+.2f}% | 交易: {r['total_trades']}笔 | "
              f"费用: ${f.get('total_costs', 0):,.0f} ({f.get('fee_drag_pct', 0):.2f}%)")
    results.extend(grid)

    ranked = sorted(results, key=lambda x: x['alpha'], reverse=True)
    print(f"\n\n{'='*140}")
    print("             Phase 5 终极排名 (最大盈利策略)")
    print(f"{'='*140}")
    fmt = "{:>3} {:<28} {:>9} {:>9} {:>8} {:>6} {:>10} {:>9} {:>12}"
    print(fmt.format("#", "策略", "收益", "超额α", "回撤", "笔数", "总费用", "费率%", "最终资产"))
    print("-" * 140)
    for rank, r in enumerate(ranked, 1):
        star = " ★" if rank == 1 else ""
        f = r.get('fees', {})
        print(fmt.format(
            rank, r['name'] + star,
            f"{r['strategy_return']:+.1f}%",
            f"{r['alpha']:+.1f}%",
            f"{r['max_drawdown']:.1f}%",
            str(r['total_trades']),
            f"${f.get('total_costs', 0):,.0f}",
            f"{f.get('fee_drag_pct', 0):.2f}%",
            f"${r['final_total']:,.0f}",
        ))
    print("=" * 140)

    # vs Phase 4 A3
    print(f"\n  vs Phase 4冠军(A3): α=+26.04%")
    if ranked[0]['alpha'] > 26.04:
        print(f"  Phase 5冠军: {ranked[0]['name']} α=+{ranked[0]['alpha']:.2f}% → 提升 +{ranked[0]['alpha']-26.04:.2f}%!")
    else:
        print(f"  Phase 5冠军: {ranked[0]['name']} α=+{ranked[0]['alpha']:.2f}%")

    output = {
        'phase': 5,
        'futures_results': [{
            'name': r['name'],
            'summary': {k: v for k, v in r.items() if k not in ('trades', 'history')},
            'trades': r['trades'],
            'history': r['history'],
        } for r in results],
        'best_strategy': ranked[0]['name'],
        'ranking': [{'rank': i+1, 'name': r['name'], 'alpha': r['alpha'],
                      'return': r['strategy_return'], 'max_dd': r['max_drawdown'],
                      'trades': r['total_trades'], 'fees': r.get('fees', {})}
                     for i, r in enumerate(ranked)],
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'strategy_futures_v5_result.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果已保存: {path}")
    return output


if __name__ == '__main__':
    run_all()
