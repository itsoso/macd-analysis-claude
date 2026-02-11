"""
合约做空策略 Phase 4 — 引擎修正后的真实优化

Phase 3修正后的真实排名:
  P1最佳: N(纯合约双向) α=+21.3%, 仅16笔交易, 首笔空仓持467小时
  P2最佳: U(追踪止盈) α=+19.6%, 175笔但太多噪音交易
  P3最佳: S3(4h级联) α=+17.7%, 仅11笔现货卖出, 0笔空仓

核心发现:
  1. 交易越少alpha越高(费用是核心拖累)
  2. N的第一笔空仓(467小时)贡献了大部分利润
  3. S3实际上没开过空仓, 纯靠现货择时卖出
  4. 同K线开平空仓 = 每次亏$24(滑点+手续费)

优化方向: 更早更大量卖出ETH + 极少量高确信度长持做空

策略变体:
  A1: 极简择时(纯现货减仓, 第一信号即大量卖出)
  A2: N增强版(更宽止损+追踪止盈, 只捕大波段)
  A3: 巨额空仓(单笔大空仓, 持数周, 搭配现货全清)
  A4: 选择性做空(现货快速减仓 + 仅最强信号开空)
  A5: 完美时机(渐进减仓 + 底部精确加仓)
  A6: 终极融合(A1-A5最佳元素)
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


def _calc_top_score(sig, trend):
    """计算顶部综合评分(融合所有指标)"""
    score = 0
    parts = []

    # MACD背离(隔堆/面积/DIF)
    if sig.get('sep_divs_top', 0) >= 2:
        score += 30; parts.append("双隔堆")
    elif sig.get('sep_divs_top', 0) >= 1 or sig.get('separated_top', 0) >= 1:
        score += 18; parts.append("隔堆")
    if sig.get('area_top_div', 0) >= 1:
        score += 12; parts.append("面积背离")
    if sig.get('dif_top_div', 0) >= 1:
        score += 8; parts.append("DIF背离")

    # 背驰
    if sig.get('exhaust_sell'):
        score += 18; parts.append("背驰")

    # 零轴
    if sig.get('zero_returns_top', 0) >= 1:
        score += 10; parts.append("零轴回")

    # 通用顶部评分
    if sig.get('top', 0) >= 20:
        score += sig['top'] * 0.2; parts.append(f"TOP{sig['top']:.0f}")

    # 趋势加成
    if trend['is_downtrend']:
        score = int(score * 1.3)
        parts.append("↓趋势")

    return score, parts


def _calc_bottom_score(sig, trend):
    """计算底部综合评分"""
    score = 0
    if sig.get('sep_divs_bottom', 0) >= 2: score += 30
    elif sig.get('sep_divs_bottom', 0) >= 1 or sig.get('separated_bottom', 0) >= 1:
        score += 15
    if sig.get('exhaust_buy'): score += 18
    if sig.get('area_bottom_div', 0) >= 1: score += 10
    if sig.get('bottom', 0) >= 20: score += sig['bottom'] * 0.15
    if trend.get('is_uptrend'): score = int(score * 1.2)
    return score


# ======================================================
#   A1: 极简择时(纯现货减仓)
# ======================================================
def run_A1(data, signals_all):
    """第一个强信号就大量卖出, 不做空, 不回购"""
    eng = FuturesEngine("A1: 极简择时", max_leverage=1)
    main_df = data['1h']
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[0]
    sold = False

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)
        ts, parts = _calc_top_score(sig, trend)

        if not sold and ts >= 15:
            # 第一信号: 卖出80%
            eng.spot_sell(price, dt, 0.80, f"首信号大卖 TS={ts} {','.join(parts[:2])}")
            sold = True
        elif sold and ts >= 12 and eng.spot_eth * price > 1000:
            # 后续信号: 继续卖
            eng.spot_sell(price, dt, 0.5, f"追卖 TS={ts}")

        if idx % 4 == 0: eng.record_history(dt, price)
    return eng.get_result(main_df)


# ======================================================
#   A2: N增强版(超长持空仓)
# ======================================================
def run_A2(data, signals_all):
    """基于N的思路: 极少交易, 超长持仓, 宽止损"""
    eng = FuturesEngine("A2: N增强版", max_leverage=3)
    main_df = data['1h']
    # N策略不持现货, 纯合约
    eng.spot_eth = 0
    max_pnl_r = 0; trade_count = 0

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
        eng.check_liquidation(price, dt); eng.charge_funding(price, dt)

        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)
        sig_4h = get_tf_signal(signals_all, '4h', dt)
        ts, parts = _calc_top_score(sig, trend)
        bs = _calc_bottom_score(sig, trend)
        ind = get_realtime_indicators(main_df, idx)

        # 极度选择性: 只有最强信号才交易
        if not eng.futures_short and not eng.futures_long:
            # 做空条件: TS >= 25 + 4h确认
            big_top = sig_4h['top'] >= 10 or sig_4h.get('exhaust_sell')
            if ts >= 25 and (big_top or trend['is_downtrend']) and trade_count < 6:
                lev = 3 if ts >= 40 else 2
                margin = eng.available_margin() * (0.6 if ts >= 40 else 0.4)
                eng.open_short(price, dt, margin, lev,
                    f"做空{lev}x TS={ts} {','.join(parts[:3])}")
                max_pnl_r = 0; trade_count += 1

            # 做多条件: BS >= 30
            elif bs >= 30 and trade_count < 6:
                lev = 3 if bs >= 45 else 2
                margin = eng.available_margin() * 0.4
                eng.open_long(price, dt, margin, lev,
                    f"做多{lev}x BS={bs:.0f}")
                max_pnl_r = 0; trade_count += 1

        # 平空: 只在非常强底信号或追踪止盈
        if eng.futures_short:
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
            max_pnl_r = max(max_pnl_r, pnl_r)

            # 追踪止盈(盈利超80%才追踪)
            if max_pnl_r >= 0.8:
                trail = max_pnl_r * 0.6
                if pnl_r < trail:
                    eng.close_short(price, dt, f"追踪 max={max_pnl_r*100:.0f}%")
                    max_pnl_r = 0

            # 强底信号翻转
            if bs >= 35:
                eng.close_short(price, dt, f"强底翻转 BS={bs:.0f}")
                max_pnl_r = 0

            # 宽止损-60%
            if pnl_r < -0.6:
                eng.close_short(price, dt, f"止损{pnl_r*100:.0f}%")
                max_pnl_r = 0

        # 平多
        if eng.futures_long:
            pnl_r = eng.futures_long.calc_pnl(price) / eng.futures_long.margin
            max_pnl_r = max(max_pnl_r, pnl_r)
            if max_pnl_r >= 0.5:
                trail = max_pnl_r * 0.6
                if pnl_r < trail:
                    eng.close_long(price, dt, f"追踪 max={max_pnl_r*100:.0f}%")
                    max_pnl_r = 0
            if ts >= 25:
                eng.close_long(price, dt, f"转空 TS={ts}")
                max_pnl_r = 0
            if pnl_r < -0.6:
                eng.close_long(price, dt, f"止损{pnl_r*100:.0f}%")
                max_pnl_r = 0

        if idx % 4 == 0: eng.record_history(dt, price)

    if eng.futures_short: eng.close_short(main_df['close'].iloc[-1], main_df.index[-1], "结束")
    if eng.futures_long: eng.close_long(main_df['close'].iloc[-1], main_df.index[-1], "结束")
    return eng.get_result(main_df)


# ======================================================
#   A3: 巨额空仓(现货全清+单笔大空仓)
# ======================================================
def run_A3(data, signals_all):
    """清仓现货 + 单笔大空仓持数周"""
    eng = FuturesEngine("A3: 巨额空仓", max_leverage=3)
    main_df = data['1h']
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[0]
    short_opened = False; max_pnl_r = 0

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
        eng.check_liquidation(price, dt); eng.charge_funding(price, dt)

        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)
        ts, parts = _calc_top_score(sig, trend)
        bs = _calc_bottom_score(sig, trend)

        if ts >= 15 and not short_opened:
            # 清仓现货
            if eng.spot_eth * price > 1000:
                eng.spot_sell(price, dt, 0.90, f"全清 TS={ts}")

            # 开大空仓
            if not eng.futures_short and ts >= 20:
                margin = eng.available_margin() * 0.7
                lev = 3 if ts >= 35 else 2
                eng.open_short(price, dt, margin, lev,
                    f"大空仓{lev}x TS={ts} {','.join(parts[:3])}")
                max_pnl_r = 0
                short_opened = True

        elif short_opened and ts >= 12 and eng.spot_eth * price > 1000:
            eng.spot_sell(price, dt, 0.7, f"追卖")

        # 追踪止盈(极宽)
        if eng.futures_short:
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
            max_pnl_r = max(max_pnl_r, pnl_r)
            if max_pnl_r >= 1.0:
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
#   A4: 选择性做空(快速减仓+只在最强信号做空)
# ======================================================
def run_A4(data, signals_all):
    """现货快速减仓 + 仅最强信号开空, 长持"""
    eng = FuturesEngine("A4: 选择性做空", max_leverage=3)
    main_df = data['1h']
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[0]
    cooldown_sell = 0; cooldown_short = 0; max_pnl_r = 0; min_r = 0.02

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
        eng.check_liquidation(price, dt); eng.charge_funding(price, dt)
        if cooldown_sell > 0: cooldown_sell -= 1
        if cooldown_short > 0: cooldown_short -= 1

        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)
        sig_4h = get_tf_signal(signals_all, '4h', dt)
        ts, parts = _calc_top_score(sig, trend)
        bs = _calc_bottom_score(sig, trend)
        total = eng.total_value(price)
        eth_r = eng.spot_eth * price / total if total > 0 else 0

        # 现货: 快速减仓, 低门槛(TS>=12)
        if cooldown_sell == 0 and ts >= 12 and eth_r > min_r + 0.02:
            sr = min(0.25 + ts * 0.008, 0.65)
            sr = min(sr, (eth_r - min_r) * 0.95)
            if sr > 0.05:
                eng.spot_sell(price, dt, sr, f"减仓 TS={ts}")
                cooldown_sell = 3  # 短冷却

        # 做空: 极严格 (TS>=30 + 4h确认 + 下跌趋势)
        big_tf = sig_4h['top'] >= 10 or sig_4h.get('exhaust_sell')
        if (cooldown_short == 0 and ts >= 30 and (big_tf or trend['is_downtrend'])
                and not eng.futures_short):
            lev = 3 if ts >= 45 else 2
            margin = eng.available_margin() * 0.5
            eng.open_short(price, dt, margin, lev,
                f"精选做空{lev}x TS={ts} {','.join(parts[:3])}")
            max_pnl_r = 0
            cooldown_short = 24  # 长冷却

        # 平空(保守平仓: 只在强底信号或追踪止盈)
        if eng.futures_short:
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
            max_pnl_r = max(max_pnl_r, pnl_r)
            if max_pnl_r >= 0.6:
                trail = max_pnl_r * 0.6
                if pnl_r < trail:
                    eng.close_short(price, dt, f"追踪 max={max_pnl_r*100:.0f}%")
                    max_pnl_r = 0; cooldown_short = 12
            if bs >= 30:
                eng.close_short(price, dt, f"强底 BS={bs:.0f}")
                max_pnl_r = 0; cooldown_short = 12
            if pnl_r < -0.5:
                eng.close_short(price, dt, f"止损{pnl_r*100:.0f}%")
                max_pnl_r = 0; cooldown_short = 24

        if idx % 4 == 0: eng.record_history(dt, price)

    if eng.futures_short: eng.close_short(main_df['close'].iloc[-1], main_df.index[-1], "结束")
    return eng.get_result(main_df)


# ======================================================
#   A5: 完美时机(减仓+底部加仓)
# ======================================================
def run_A5(data, signals_all):
    """先减仓后在底部精确加仓, 纯现货操作"""
    eng = FuturesEngine("A5: 完美时机", max_leverage=1)
    main_df = data['1h']
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[0]
    cooldown = 0; min_r = 0.02
    phase = 'sell'  # sell / wait / buy

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
        if cooldown > 0: cooldown -= 1

        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)
        ts, parts = _calc_top_score(sig, trend)
        bs = _calc_bottom_score(sig, trend)
        total = eng.total_value(price)
        eth_r = eng.spot_eth * price / total if total > 0 else 0

        if cooldown == 0:
            if phase == 'sell' and ts >= 12 and eth_r > min_r + 0.03:
                sr = min(0.3 + ts * 0.01, 0.7)
                sr = min(sr, (eth_r - min_r) * 0.95)
                if sr > 0.05:
                    eng.spot_sell(price, dt, sr, f"减仓 TS={ts}")
                    cooldown = 2
                if eth_r - sr < 0.08:
                    phase = 'wait'  # ETH占比足够低, 等待底部

            elif phase == 'wait':
                # 等待强底信号
                if bs >= 35:
                    phase = 'buy'

            elif phase == 'buy':
                if bs >= 25 and eng.available_usdt() > 5000:
                    invest = eng.available_usdt() * 0.15
                    eng.spot_buy(price, dt, invest, f"底部加仓 BS={bs:.0f}")
                    cooldown = 12

                # 如果又出现顶信号, 回到sell
                if ts >= 20 and eth_r > 0.1:
                    phase = 'sell'

        if idx % 4 == 0: eng.record_history(dt, price)
    return eng.get_result(main_df)


# ======================================================
#   A6: 终极融合
# ======================================================
def run_A6(data, signals_all):
    """融合最佳元素:
    - A1的快速减仓 + A2的长持空仓 + A4的选择性 + A5的底部加仓"""
    eng = FuturesEngine("A6: 终极融合", max_leverage=3)
    main_df = data['1h']
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[0]
    cd_sell = 0; cd_short = 0; max_pnl_r = 0; min_r = 0.02
    short_count = 0; max_shorts = 3  # 最多3笔空仓

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
        eng.check_liquidation(price, dt); eng.charge_funding(price, dt)
        if cd_sell > 0: cd_sell -= 1
        if cd_short > 0: cd_short -= 1

        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)
        sig_4h = get_tf_signal(signals_all, '4h', dt)
        ts, parts = _calc_top_score(sig, trend)
        bs = _calc_bottom_score(sig, trend)
        total = eng.total_value(price)
        eth_r = eng.spot_eth * price / total if total > 0 else 0

        # === 快速减仓(A1风格, 第一信号大卖) ===
        if cd_sell == 0 and ts >= 12 and eth_r > min_r + 0.02:
            sr = min(0.35 + ts * 0.01, 0.70)
            sr = min(sr, (eth_r - min_r) * 0.95)
            if sr > 0.05:
                eng.spot_sell(price, dt, sr, f"减仓 TS={ts} {','.join(parts[:2])}")
                cd_sell = 2

        # === 精选做空(A2/A4风格, 极高门槛+长持) ===
        big_tf = sig_4h['top'] >= 10 or sig_4h.get('exhaust_sell')
        if (cd_short == 0 and ts >= 28 and (big_tf or trend['is_downtrend'])
                and not eng.futures_short and short_count < max_shorts):
            lev = 3 if ts >= 40 else 2
            margin = eng.available_margin() * (0.6 if ts >= 40 else 0.4)
            eng.open_short(price, dt, margin, lev,
                f"精选做空{lev}x TS={ts} {','.join(parts[:3])}")
            max_pnl_r = 0; short_count += 1
            cd_short = 48  # 2天冷却

        # 追踪止盈(宽松, 让利润奔跑)
        if eng.futures_short:
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
            max_pnl_r = max(max_pnl_r, pnl_r)
            if max_pnl_r >= 0.8:
                trail = max_pnl_r * 0.55
                if pnl_r < trail:
                    eng.close_short(price, dt, f"追踪 max={max_pnl_r*100:.0f}%")
                    max_pnl_r = 0; cd_short = 12
            if bs >= 35:
                eng.close_short(price, dt, f"强底 BS={bs:.0f}")
                max_pnl_r = 0; cd_short = 12
            if pnl_r < -0.55:
                eng.close_short(price, dt, f"止损{pnl_r*100:.0f}%")
                max_pnl_r = 0; cd_short = 24

        # === 底部精确加仓(A5风格) ===
        if (bs >= 35 and eth_r < 0.15 and eng.available_usdt() > 5000
                and cd_sell == 0):
            invest = eng.available_usdt() * 0.10
            eng.spot_buy(price, dt, invest, f"底部加仓 BS={bs:.0f}")
            cd_sell = 16

        if idx % 4 == 0: eng.record_history(dt, price)

    if eng.futures_short: eng.close_short(main_df['close'].iloc[-1], main_df.index[-1], "结束")
    return eng.get_result(main_df)


# ======================================================
#   参数网格搜索(A6框架)
# ======================================================
def run_param_grid(data, signals_all):
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]

    configs = [
        {'name': 'P1: 超低卖门槛',     'sell_th': 8,  'short_th': 25, 'lev': 2, 'cd_s': 48, 'min_r': 0.01},
        {'name': 'P2: 首信号全清',     'sell_th': 12, 'short_th': 30, 'lev': 3, 'cd_s': 48, 'min_r': 0.01},
        {'name': 'P3: 高门槛3x空',    'sell_th': 15, 'short_th': 35, 'lev': 3, 'cd_s': 72, 'min_r': 0.02},
        {'name': 'P4: 纯N风格无现货',  'sell_th': 99, 'short_th': 25, 'lev': 3, 'cd_s': 48, 'min_r': 0.50},
        {'name': 'P5: 中等均衡',       'sell_th': 12, 'short_th': 28, 'lev': 2, 'cd_s': 36, 'min_r': 0.02},
        {'name': 'P6: 低门槛频繁卖',   'sell_th': 8,  'short_th': 30, 'lev': 2, 'cd_s': 24, 'min_r': 0.01},
    ]

    results = []
    for p in configs:
        eng = FuturesEngine(p['name'], max_leverage=p['lev'])
        if p['min_r'] >= 0.5:
            eng.spot_eth = 0  # 纯合约模式
        else:
            eng.spot_eth = eng.initial_eth_value / first_price
        cd_sell = 0; cd_short = 0; max_pnl_r = 0; short_count = 0

        for idx in range(20, len(main_df)):
            dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
            eng.check_liquidation(price, dt); eng.charge_funding(price, dt)
            if cd_sell > 0: cd_sell -= 1
            if cd_short > 0: cd_short -= 1

            trend = get_trend_info(data, dt, price)
            sig = _merge_signals(signals_all, dt)
            sig_4h = get_tf_signal(signals_all, '4h', dt)
            ts, parts = _calc_top_score(sig, trend)
            bs = _calc_bottom_score(sig, trend)
            total = eng.total_value(price)
            eth_r = eng.spot_eth * price / total if total > 0 else 0

            if cd_sell == 0 and ts >= p['sell_th'] and eth_r > p['min_r'] + 0.02:
                sr = min(0.35 + ts * 0.01, 0.70)
                sr = min(sr, (eth_r - p['min_r']) * 0.95)
                if sr > 0.05:
                    eng.spot_sell(price, dt, sr, f"减仓 TS={ts}")
                    cd_sell = 2

            big_tf = sig_4h['top'] >= 10 or sig_4h.get('exhaust_sell')
            if (cd_short == 0 and ts >= p['short_th'] and
                    (big_tf or trend['is_downtrend']) and
                    not eng.futures_short and short_count < 3):
                margin = eng.available_margin() * 0.5
                eng.open_short(price, dt, margin, p['lev'],
                    f"做空{p['lev']}x TS={ts}")
                max_pnl_r = 0; short_count += 1; cd_short = p['cd_s']

            if eng.futures_short:
                pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
                max_pnl_r = max(max_pnl_r, pnl_r)
                if max_pnl_r >= 0.8:
                    trail = max_pnl_r * 0.55
                    if pnl_r < trail:
                        eng.close_short(price, dt, f"追踪")
                        max_pnl_r = 0; cd_short = 12
                if bs >= 35:
                    eng.close_short(price, dt, f"强底")
                    max_pnl_r = 0; cd_short = 12
                if pnl_r < -0.55:
                    eng.close_short(price, dt, f"止损")
                    max_pnl_r = 0; cd_short = 24

            if bs >= 35 and eth_r < 0.15 and eng.available_usdt() > 5000 and cd_sell == 0:
                eng.spot_buy(price, dt, eng.available_usdt() * 0.08, "底部加仓")
                cd_sell = 16

            if idx % 4 == 0: eng.record_history(dt, price)

        if eng.futures_short:
            eng.close_short(main_df['close'].iloc[-1], main_df.index[-1], "结束")
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
        ("A1", run_A1),
        ("A2", run_A2),
        ("A3", run_A3),
        ("A4", run_A4),
        ("A5", run_A5),
        ("A6", run_A6),
    ]

    print(f"\n{'='*140}")
    print(f"  Phase 4: 引擎修正后的真实优化")
    print(f"  核心: 更早更大量卖出ETH + 极少量高确信度长持做空")
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

    print(f"\n>>> 参数网格搜索 (6种配置)...")
    grid = run_param_grid(data, signals_all)
    for r in grid:
        f = r.get('fees', {})
        print(f"    {r['name']:<22} α: {r['alpha']:+.2f}% | 交易: {r['total_trades']}笔 | "
              f"费用: ${f.get('total_costs', 0):,.0f} ({f.get('fee_drag_pct', 0):.2f}%)")
    results.extend(grid)

    # 排名
    ranked = sorted(results, key=lambda x: x['alpha'], reverse=True)
    print(f"\n\n{'='*140}")
    print("             Phase 4 排名 (修正引擎 · 真实收益)")
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

    # 保存
    output = {
        'phase': 4,
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
                        'strategy_futures_v4_result.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果已保存: {path}")
    return output


if __name__ == '__main__':
    run_all()
