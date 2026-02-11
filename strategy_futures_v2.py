"""
合约做空策略 Phase 2 — 基于书中核心原则的进阶优化

分析Phase 1最佳策略P的不足:
1. 交易频繁(196笔), 很多低质量信号
2. 做空入场没有多周期级联确认
3. 固定止盈止损, 错失大行情
4. 没有充分利用书中最强信号(隔堆背离/面积背离)
5. KDJ/CCI/RSI极值区未做核心过滤

新策略设计:
R: 多周期瀑布做空 — 4h/8h定方向, 1h找入场点, 严格级联确认
S: 隔堆+面积背离精确做空 — 只在书中最强信号时做空, 3x杠杆
T: 极值三重确认做空 — KDJ>80+CCI>100+RSI>70三重极值才做空
U: 追踪止盈做空 — 用移动止盈替代固定止盈, 捕获大行情
V: 量价+死叉联合做空 — 价升量减+MACD死叉的双重确认
W: 全书终极融合 — 汇集所有优化精华, 寻找最优组合
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
    build_sell_buy_scores, _merge_signals,
    run_strategy_P, run_baseline_spot
)


# ======================================================
#   辅助: 多周期信号独立获取
# ======================================================
def get_tf_signal(signals_all, tf, dt):
    """获取指定周期的独立信号"""
    if tf not in signals_all:
        return DEFAULT_SIG.copy()
    s = get_signal_at(signals_all[tf], dt)
    return s if s else DEFAULT_SIG.copy()


def get_trend_info(data, dt, price):
    """获取多维度趋势判断"""
    info = {'is_downtrend': True, 'trend_strength': 0}

    # 4h MA50
    df_4h = data.get('4h')
    if df_4h is not None:
        ma50 = df_4h['close'].rolling(50).mean()
        ma20 = df_4h['close'].rolling(20).mean()
        for i in range(len(df_4h) - 1, -1, -1):
            if df_4h.index[i] <= dt:
                m50 = ma50.iloc[i] if not pd.isna(ma50.iloc[i]) else price
                m20 = ma20.iloc[i] if not pd.isna(ma20.iloc[i]) else price
                info['is_downtrend'] = price < m50
                # 趋势强度: 价格偏离MA50的百分比
                info['trend_strength'] = (price - m50) / m50 * 100
                info['ma20_above_ma50'] = m20 > m50
                break

    # 1h MA趋势一致性
    df_1h = data.get('1h')
    if df_1h is not None:
        ma10_1h = df_1h['close'].rolling(10).mean()
        ma30_1h = df_1h['close'].rolling(30).mean()
        for i in range(len(df_1h) - 1, -1, -1):
            if df_1h.index[i] <= dt:
                m10 = ma10_1h.iloc[i] if not pd.isna(ma10_1h.iloc[i]) else price
                m30 = ma30_1h.iloc[i] if not pd.isna(ma30_1h.iloc[i]) else price
                info['short_trend_down'] = m10 < m30
                break

    return info


# ======================================================
#   策略R: 多周期瀑布做空
# ======================================================
def run_strategy_R(data, signals_all):
    """R: 多周期瀑布做空
    核心: 大周期定方向, 小周期找精确入场
    做空条件(全部满足):
      1. 4h或8h出现顶背离(top >= 15)或背驰信号
      2. 2h确认: top >= 10 或死叉
      3. 1h入场: MACD死叉 或 KDJ>70 开始拐头
    做多平空条件:
      1. 4h底信号(bottom >= 30)或底背驰
      2. 1h出现金叉
    """
    eng = FuturesEngine("R: 多周期瀑布做空", max_leverage=3, use_spot=True)
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eng.spot_eth = eng.initial_eth_value / first_price
    cooldown = 0
    min_eth_ratio = 0.05
    pending_short = False  # 大周期信号确认, 等待小周期入场
    pending_strength = 0

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        ind = get_realtime_indicators(main_df, idx)
        eng.check_liquidation(price, dt)
        eng.charge_funding(price, dt)
        if cooldown > 0:
            cooldown -= 1

        trend = get_trend_info(data, dt, price)

        # 各周期独立信号
        sig_4h = get_tf_signal(signals_all, '4h', dt)
        sig_8h = get_tf_signal(signals_all, '8h', dt)
        sig_2h = get_tf_signal(signals_all, '2h', dt)
        sig_1h = get_tf_signal(signals_all, '1h', dt)

        total = eng.total_value(price)
        spot_val = eng.spot_eth * price
        eth_r = spot_val / total if total > 0 else 0

        k_val = ind.get('K')
        dif = ind.get('DIF')
        dea = ind.get('DEA')

        # === 第1层: 大周期信号确认(4h/8h) ===
        large_tf_sell = 0
        if sig_4h['top'] >= 15: large_tf_sell += sig_4h['top'] * 0.4
        if sig_8h['top'] >= 15: large_tf_sell += sig_8h['top'] * 0.3
        if sig_4h.get('exhaust_sell'): large_tf_sell += 20
        if sig_8h.get('exhaust_sell'): large_tf_sell += 15
        if sig_4h.get('sep_divs_top', 0) >= 1: large_tf_sell += 10
        if sig_8h.get('sep_divs_top', 0) >= 1: large_tf_sell += 8

        large_tf_buy = 0
        if sig_4h['bottom'] >= 30: large_tf_buy += sig_4h['bottom'] * 0.3
        if sig_4h.get('exhaust_buy'): large_tf_buy += 15
        if sig_8h['bottom'] >= 30: large_tf_buy += sig_8h['bottom'] * 0.2
        if sig_8h.get('exhaust_buy'): large_tf_buy += 12

        # === 第2层: 2h确认 ===
        mid_confirm_sell = (sig_2h['top'] >= 10 or
                            sig_2h.get('last_cross') == 'death' or
                            sig_2h.get('exhaust_sell'))
        mid_confirm_buy = (sig_2h['bottom'] >= 20 or
                           sig_2h.get('last_cross') == 'golden' or
                           sig_2h.get('exhaust_buy'))

        # === 第3层: 1h精确入场 ===
        # 死叉或KDJ拐头
        is_death_cross = sig_1h.get('last_cross') == 'death'
        kdj_turning_down = k_val and k_val > 60 and dif is not None and dea is not None and dif < dea
        entry_sell = is_death_cross or kdj_turning_down

        is_golden_cross = sig_1h.get('last_cross') == 'golden'
        kdj_turning_up = k_val and k_val < 40 and dif is not None and dea is not None and dif > dea
        entry_buy = is_golden_cross or kdj_turning_up

        if cooldown == 0:
            # 更新pending状态
            if large_tf_sell >= 12 and mid_confirm_sell:
                pending_short = True
                pending_strength = large_tf_sell

            # 做空: 三层确认完成
            if pending_short and entry_sell and not eng.futures_short:
                # 同时减仓
                if eth_r > min_eth_ratio + 0.05:
                    sr = min(0.2 + pending_strength * 0.003, 0.4)
                    sr = min(sr, (eth_r - min_eth_ratio) * 0.9)
                    if sr > 0.05:
                        eng.spot_sell(price, dt, sr,
                            f"减仓 4h={sig_4h['top']:.0f}")

                lev = 3 if pending_strength >= 25 else 2
                margin = eng.available_margin() * (0.4 if pending_strength >= 25 else 0.25)
                eng.open_short(price, dt, margin, lev,
                    f"瀑布做空{lev}x 4h={sig_4h['top']:.0f} 2h确认 1h{'死叉' if is_death_cross else 'KDJ拐'}")
                pending_short = False
                cooldown = 6

            # 平空: 大周期底信号 + 1h入场确认
            if eng.futures_short and large_tf_buy >= 8 and entry_buy:
                eng.close_short(price, dt,
                    f"瀑布平空 4h={sig_4h['bottom']:.0f} 1h{'金叉' if is_golden_cross else 'KDJ↑'}")
                cooldown = 6

            # 做多(保守): 大周期强底信号
            if not eng.futures_long and large_tf_buy >= 20 and entry_buy and not trend['is_downtrend']:
                margin = eng.available_margin() * 0.2
                eng.open_long(price, dt, margin, 2,
                    f"瀑布做多 4h={sig_4h['bottom']:.0f}")
                cooldown = 8

            # 止盈/止损
            if eng.futures_short:
                pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
                if pnl_r > 0.6:
                    eng.close_short(price, dt, f"止盈{pnl_r*100:.0f}%")
                    cooldown = 4
                elif pnl_r < -0.35:
                    eng.close_short(price, dt, f"止损{pnl_r*100:.0f}%")
                    cooldown = 8
                    pending_short = False

            if eng.futures_long:
                pnl_r = eng.futures_long.calc_pnl(price) / eng.futures_long.margin
                if pnl_r > 0.5:
                    eng.close_long(price, dt, f"止盈{pnl_r*100:.0f}%")
                    cooldown = 4
                elif pnl_r < -0.3:
                    eng.close_long(price, dt, f"止损{pnl_r*100:.0f}%")
                    cooldown = 6

        if idx % 4 == 0:
            eng.record_history(dt, price)

    last_p = main_df['close'].iloc[-1]
    last_dt = main_df.index[-1]
    if eng.futures_short: eng.close_short(last_p, last_dt, "结束平空")
    if eng.futures_long: eng.close_long(last_p, last_dt, "结束平多")
    return eng.get_result(main_df)


# ======================================================
#   策略S: 隔堆+面积背离精确做空
# ======================================================
def run_strategy_S(data, signals_all):
    """S: 隔堆+面积背离精确做空
    书中第二/三章核心: 隔堆背离是最可靠的背离信号
    只在出现以下信号时做空(高置信度):
      1. 隔堆背离(separated_top >= 1)
      2. 面积背离(area_top_div >= 1)
      3. 零轴回抽+隔堆(sep_divs_top >= 1 and zero_returns_top >= 1)
      4. DIF/DEA背离(dif_top_div >= 1) + 下降趋势确认
    使用3x杠杆(高确信信号)"""
    eng = FuturesEngine("S: 隔堆精确做空", max_leverage=3, use_spot=True)
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eng.spot_eth = eng.initial_eth_value / first_price
    cooldown = 0
    min_eth_ratio = 0.05

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        ind = get_realtime_indicators(main_df, idx)
        eng.check_liquidation(price, dt)
        eng.charge_funding(price, dt)
        if cooldown > 0:
            cooldown -= 1

        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)

        # 精确信号检测
        has_separated = (sig.get('separated_top', 0) >= 1 or
                         sig.get('sep_divs_top', 0) >= 1)
        has_area_div = sig.get('area_top_div', 0) >= 1
        has_zero_return_sep = (sig.get('sep_divs_top', 0) >= 1 and
                               sig.get('zero_returns_top', 0) >= 1)
        has_dif_div_trend = (sig.get('dif_top_div', 0) >= 1 and
                             trend['is_downtrend'])
        has_double_separated = sig.get('sep_divs_top', 0) >= 2

        # 信号强度打分
        precision_score = 0
        parts = []
        if has_double_separated:
            precision_score += 25; parts.append("双隔堆")
        elif has_separated:
            precision_score += 15; parts.append("隔堆背离")
        if has_zero_return_sep:
            precision_score += 12; parts.append("零轴+隔堆")
        if has_area_div:
            precision_score += 10; parts.append("面积背离")
        if has_dif_div_trend:
            precision_score += 8; parts.append("DIF背离+↓趋势")
        if sig.get('exhaust_sell'):
            precision_score += 15; parts.append("背驰")
        if trend['is_downtrend']:
            precision_score = int(precision_score * 1.2)

        total = eng.total_value(price)
        spot_val = eng.spot_eth * price
        eth_r = spot_val / total if total > 0 else 0

        if cooldown == 0:
            # 做空: 只在精确信号时
            if precision_score >= 15 and not eng.futures_short:
                # 现货减仓
                if eth_r > min_eth_ratio + 0.05:
                    sr = min(0.15 + precision_score * 0.005, 0.5)
                    sr = min(sr, (eth_r - min_eth_ratio) * 0.9)
                    if sr > 0.05:
                        eng.spot_sell(price, dt, sr,
                            f"精确减仓 {','.join(parts[:2])}")

                # 开空: 高确信=3x, 否则2x
                lev = 3 if precision_score >= 25 else 2
                margin = eng.available_margin() * (0.5 if precision_score >= 25 else 0.3)
                eng.open_short(price, dt, margin, lev,
                    f"精确做空{lev}x PS={precision_score} {','.join(parts[:3])}")
                cooldown = 8  # 精确策略冷却更长

            # 底部精确信号
            has_bottom_sep = (sig.get('separated_bottom', 0) >= 1 or
                              sig.get('sep_divs_bottom', 0) >= 1)
            has_bottom_exhaust = sig.get('exhaust_buy')
            bottom_precision = 0
            if has_bottom_sep: bottom_precision += 15
            if has_bottom_exhaust: bottom_precision += 15
            if sig.get('sep_divs_bottom', 0) >= 2: bottom_precision += 10

            if eng.futures_short and bottom_precision >= 12:
                eng.close_short(price, dt,
                    f"底部精确信号 BP={bottom_precision}")
                cooldown = 6

            # 止盈/止损
            if eng.futures_short:
                pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
                if pnl_r > 0.7:
                    eng.close_short(price, dt, f"止盈{pnl_r*100:.0f}%")
                    cooldown = 4
                elif pnl_r < -0.35:
                    eng.close_short(price, dt, f"止损{pnl_r*100:.0f}%")
                    cooldown = 10

        if idx % 4 == 0:
            eng.record_history(dt, price)

    last_p = main_df['close'].iloc[-1]
    last_dt = main_df.index[-1]
    if eng.futures_short: eng.close_short(last_p, last_dt, "结束平空")
    if eng.futures_long: eng.close_long(last_p, last_dt, "结束平多")
    return eng.get_result(main_df)


# ======================================================
#   策略T: 极值三重确认做空
# ======================================================
def run_strategy_T(data, signals_all):
    """T: 极值三重确认做空
    书中第四/五章: KDJ>80超买, CCI>100天线, RSI>70超买
    三重极值 = 超高概率反转区
    做空条件: 至少2个极值指标+1个背离信号
    平仓: 任意指标回到中性区"""
    eng = FuturesEngine("T: 极值三重确认", max_leverage=3, use_spot=True)
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eng.spot_eth = eng.initial_eth_value / first_price
    cooldown = 0
    min_eth_ratio = 0.05

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        ind = get_realtime_indicators(main_df, idx)
        eng.check_liquidation(price, dt)
        eng.charge_funding(price, dt)
        if cooldown > 0:
            cooldown -= 1

        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)

        k_val = ind.get('K')
        rsi6 = ind.get('RSI6')
        cci = ind.get('CCI')

        # 极值检测
        extreme_count = 0
        extreme_parts = []
        if k_val and k_val > 80:
            extreme_count += 1; extreme_parts.append(f"K={k_val:.0f}")
        if cci and cci > 100:
            extreme_count += 1; extreme_parts.append(f"CCI={cci:.0f}")
        if rsi6 and rsi6 > 70:
            extreme_count += 1; extreme_parts.append(f"RSI={rsi6:.0f}")

        # 背离辅助确认
        has_div = (sig['top'] >= 15 or sig.get('exhaust_sell') or
                   sig.get('separated_top', 0) >= 1 or
                   sig.get('area_top_div', 0) >= 1)

        # 底部极值
        bottom_extreme = 0
        if k_val and k_val < 20: bottom_extreme += 1
        if cci and cci < -100: bottom_extreme += 1
        if rsi6 and rsi6 < 30: bottom_extreme += 1

        total = eng.total_value(price)
        spot_val = eng.spot_eth * price
        eth_r = spot_val / total if total > 0 else 0

        if cooldown == 0:
            # 做空: 至少2个极值 + 背离确认
            if extreme_count >= 2 and has_div and not eng.futures_short:
                # 减仓
                if eth_r > min_eth_ratio + 0.05:
                    sr = min(0.15 + extreme_count * 0.1, 0.45)
                    sr = min(sr, (eth_r - min_eth_ratio) * 0.9)
                    if sr > 0.05:
                        eng.spot_sell(price, dt, sr,
                            f"极值减仓 {','.join(extreme_parts)}")

                lev = 3 if extreme_count >= 3 else 2
                margin = eng.available_margin() * (0.5 if extreme_count >= 3 else 0.3)
                eng.open_short(price, dt, margin, lev,
                    f"极值做空{lev}x x{extreme_count} {','.join(extreme_parts)} +背离")
                cooldown = 6

            # 超强极值(3个指标全部超买): 单纯极值也做空
            elif extreme_count >= 3 and not eng.futures_short:
                if eth_r > min_eth_ratio + 0.05:
                    sr = min(0.25, (eth_r - min_eth_ratio) * 0.8)
                    if sr > 0.05:
                        eng.spot_sell(price, dt, sr,
                            f"三极值减仓 {','.join(extreme_parts)}")
                margin = eng.available_margin() * 0.35
                eng.open_short(price, dt, margin, 2,
                    f"三极值做空 {','.join(extreme_parts)}")
                cooldown = 6

            # 平空: 指标回中性 + 底部极值
            if eng.futures_short:
                closed = False
                # 回中性
                neutral = ((k_val and k_val < 50) and
                           (cci is None or cci < 50) and
                           (rsi6 is None or rsi6 < 50))
                if neutral and not closed:
                    pnl = eng.futures_short.calc_pnl(price)
                    if pnl > 0:  # 盈利时才因回中性平仓
                        eng.close_short(price, dt,
                            f"指标回中性 K={k_val:.0f} PnL={pnl:+.0f}")
                        cooldown = 6; closed = True

                # 底部极值触发平仓
                if not closed and eng.futures_short and bottom_extreme >= 2:
                    eng.close_short(price, dt,
                        f"底部极值 K={k_val:.0f}")
                    cooldown = 8; closed = True

                # 止损
                if not closed and eng.futures_short:
                    pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
                    if pnl_r < -0.35:
                        eng.close_short(price, dt, f"止损{pnl_r*100:.0f}%")
                        cooldown = 8

        if idx % 4 == 0:
            eng.record_history(dt, price)

    last_p = main_df['close'].iloc[-1]
    last_dt = main_df.index[-1]
    if eng.futures_short: eng.close_short(last_p, last_dt, "结束平空")
    return eng.get_result(main_df)


# ======================================================
#   策略U: 追踪止盈做空
# ======================================================
def run_strategy_U(data, signals_all):
    """U: 追踪止盈做空
    入场: 与P相同的信号评分系统
    关键改进: 移动止盈(trailing stop)
    - 盈利超过15%后, 追踪止盈线 = 最高盈利 * 60%
    - 盈利超过30%后, 追踪止盈线 = 最高盈利 * 70%
    - 盈利超过50%后, 追踪止盈线 = 最高盈利 * 80%
    这样在大行情中能持仓更久, 捕获更多利润"""
    eng = FuturesEngine("U: 追踪止盈做空", max_leverage=3, use_spot=True)
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eng.spot_eth = eng.initial_eth_value / first_price
    cooldown = 0
    min_eth_ratio = 0.05
    # 追踪止盈状态
    short_max_pnl_ratio = 0  # 空仓最大盈利率
    long_max_pnl_ratio = 0

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        ind = get_realtime_indicators(main_df, idx)
        eng.check_liquidation(price, dt)
        eng.charge_funding(price, dt)
        if cooldown > 0:
            cooldown -= 1

        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)
        sell_s, sell_p, buy_s, buy_p = build_sell_buy_scores(sig, ind, main_df, idx, trend['is_downtrend'])

        total = eng.total_value(price)
        spot_val = eng.spot_eth * price
        eth_r = spot_val / total if total > 0 else 0

        if cooldown == 0:
            # === 做空(与P类似但更严格) ===
            if sell_s >= 15:
                if eth_r > min_eth_ratio + 0.05:
                    sr = min(0.15 + sell_s * 0.005, 0.5)
                    sr = min(sr, (eth_r - min_eth_ratio) * 0.9)
                    if sr > 0.05:
                        eng.spot_sell(price, dt, sr, f"减仓 S={sell_s:.0f}")

                if sell_s >= 22 and not eng.futures_short:
                    lev = 3 if sell_s >= 35 else 2
                    margin = eng.available_margin() * (0.4 if sell_s >= 35 else 0.25)
                    eng.open_short(price, dt, margin, lev,
                        f"做空{lev}x S={sell_s:.0f} {','.join(sell_p[:2])}")
                    short_max_pnl_ratio = 0  # 重置追踪
                    cooldown = 5

            # === 做多 ===
            if buy_s >= 20:
                if eng.futures_short:
                    eng.close_short(price, dt, f"底信号平空 B={buy_s:.0f}")
                    short_max_pnl_ratio = 0
                    cooldown = 4

                if buy_s >= 32 and eth_r < 0.4:
                    invest = eng.available_usdt() * 0.15
                    eng.spot_buy(price, dt, invest, f"加仓 B={buy_s:.0f}")
                    cooldown = 8

        # === 追踪止盈 (每根K线都检查) ===
        if eng.futures_short:
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
            short_max_pnl_ratio = max(short_max_pnl_ratio, pnl_r)

            # 确定追踪比例
            if short_max_pnl_ratio >= 0.5:
                trail_keep = 0.80  # 保留80%的最高利润
            elif short_max_pnl_ratio >= 0.3:
                trail_keep = 0.70
            elif short_max_pnl_ratio >= 0.15:
                trail_keep = 0.60
            else:
                trail_keep = 0  # 还没到追踪条件

            if trail_keep > 0:
                trail_line = short_max_pnl_ratio * trail_keep
                if pnl_r < trail_line:
                    eng.close_short(price, dt,
                        f"追踪止盈 max={short_max_pnl_ratio*100:.0f}% now={pnl_r*100:.0f}%")
                    short_max_pnl_ratio = 0
                    cooldown = 4

            # 固定止损
            if pnl_r < -0.35:
                eng.close_short(price, dt, f"止损{pnl_r*100:.0f}%")
                short_max_pnl_ratio = 0
                cooldown = 6

        if idx % 4 == 0:
            eng.record_history(dt, price)

    last_p = main_df['close'].iloc[-1]
    last_dt = main_df.index[-1]
    if eng.futures_short: eng.close_short(last_p, last_dt, "结束平空")
    if eng.futures_long: eng.close_long(last_p, last_dt, "结束平多")
    return eng.get_result(main_df)


# ======================================================
#   策略V: 量价+死叉联合做空
# ======================================================
def run_strategy_V(data, signals_all):
    """V: 量价+死叉联合做空
    书中核心原则组合:
    做空条件: 价升量减(量价背离) + MACD死叉 + 下降趋势
    这是书中反复强调的最经典组合:
    - 价格新高但量能萎缩 → 上涨力量衰竭
    - MACD出现死叉 → 短期转弱确认
    - 下降趋势中 → 大方向支撑做空
    平空: 地量+金叉 或 底背驰"""
    eng = FuturesEngine("V: 量价+死叉联合", max_leverage=2, use_spot=True)
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eng.spot_eth = eng.initial_eth_value / first_price
    vol_ma20 = main_df['volume'].rolling(20).mean()
    vol_ma5 = main_df['volume'].rolling(5).mean()
    cooldown = 0
    min_eth_ratio = 0.05

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        ind = get_realtime_indicators(main_df, idx)
        eng.check_liquidation(price, dt)
        eng.charge_funding(price, dt)
        if cooldown > 0:
            cooldown -= 1

        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)

        # 量价分析
        avg_vol = vol_ma20.iloc[idx] if idx >= 20 else 1
        vol5 = vol_ma5.iloc[idx] if idx >= 5 else avg_vol
        cur_vol = main_df['volume'].iloc[idx]
        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1
        is_shrinking = vol5 < avg_vol * 0.6
        is_ground_vol = vol_ratio < 0.3

        price_change_5 = 0
        if idx >= 5:
            price_change_5 = (price - main_df['close'].iloc[idx - 5]) / main_df['close'].iloc[idx - 5]
        price_up_vol_down = price_change_5 > 0.01 and is_shrinking
        price_up_vol_down_strong = price_change_5 > 0.02 and vol5 < avg_vol * 0.4

        # MACD交叉
        is_death_cross = sig.get('last_cross') == 'death'
        is_golden_cross = sig.get('last_cross') == 'golden'

        # 综合做空条件
        short_signal = False
        short_strength = 0
        short_parts = []

        # 核心组合: 量价背离 + 死叉
        if price_up_vol_down and is_death_cross:
            short_signal = True
            short_strength += 20
            short_parts.append("价升量减+死叉")

        # 强化: 量价背离 + 背离分数
        if price_up_vol_down and sig['top'] >= 15:
            short_signal = True
            short_strength += 15
            short_parts.append(f"量价+背离{sig['top']:.0f}")

        # 强化: 缩量 + 背驰
        if is_shrinking and sig.get('exhaust_sell'):
            short_signal = True
            short_strength += 18
            short_parts.append("缩量+背驰")

        # 趋势加成
        if trend['is_downtrend']:
            short_strength = int(short_strength * 1.3)
            short_parts.append("↓趋势")

        # 做空信号中叠加其他书中信号
        if short_signal and sig.get('vol_price_up_down', 0) >= 1:
            short_strength += 5; short_parts.append("VP↓")
        if short_signal and sig.get('separated_top', 0) >= 1:
            short_strength += 8; short_parts.append("隔堆")

        total = eng.total_value(price)
        spot_val = eng.spot_eth * price
        eth_r = spot_val / total if total > 0 else 0

        if cooldown == 0:
            if short_signal and short_strength >= 15:
                # 现货减仓
                if eth_r > min_eth_ratio + 0.05:
                    sr = min(0.12 + short_strength * 0.005, 0.4)
                    sr = min(sr, (eth_r - min_eth_ratio) * 0.9)
                    if sr > 0.05:
                        eng.spot_sell(price, dt, sr,
                            f"量价减仓 {','.join(short_parts[:2])}")

                # 开空
                if not eng.futures_short and short_strength >= 18:
                    margin = eng.available_margin() * min(0.15 + short_strength * 0.008, 0.45)
                    eng.open_short(price, dt, margin, 2,
                        f"量价做空 SS={short_strength} {','.join(short_parts[:3])}")
                    cooldown = 6

            # 平空: 地量+金叉 或 底背驰
            buy_trigger = ((is_ground_vol and is_golden_cross) or
                           sig.get('exhaust_buy') or
                           (sig['bottom'] >= 40 and is_golden_cross))
            if eng.futures_short and buy_trigger:
                eng.close_short(price, dt,
                    f"平空 {'地量金叉' if is_ground_vol else '底信号'}")
                cooldown = 6

            # 加仓: 地量+底背驰
            if is_ground_vol and sig.get('exhaust_buy') and eth_r < 0.4:
                invest = eng.available_usdt() * 0.12
                eng.spot_buy(price, dt, invest, f"地量底背驰加仓")
                cooldown = 10

            # 止盈/止损
            if eng.futures_short:
                pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
                if pnl_r > 0.55:
                    eng.close_short(price, dt, f"止盈{pnl_r*100:.0f}%")
                    cooldown = 4
                elif pnl_r < -0.3:
                    eng.close_short(price, dt, f"止损{pnl_r*100:.0f}%")
                    cooldown = 8

        if idx % 4 == 0:
            eng.record_history(dt, price)

    last_p = main_df['close'].iloc[-1]
    last_dt = main_df.index[-1]
    if eng.futures_short: eng.close_short(last_p, last_dt, "结束平空")
    return eng.get_result(main_df)


# ======================================================
#   策略W: 全书终极融合
# ======================================================
def run_strategy_W(data, signals_all):
    """W: 全书终极融合
    集合所有优化思路:
    1. 多周期级联(R): 4h定方向, 1h找入场
    2. 隔堆信号优先(S): 最强信号用最大仓位
    3. 极值区过滤(T): KDJ/CCI/RSI极值确认
    4. 追踪止盈(U): 移动止盈替代固定止盈
    5. 量价确认(V): 价升量减做空最终确认
    6. 渐进减仓(E): 现货分批减仓"""
    eng = FuturesEngine("W: 全书终极融合", max_leverage=3, use_spot=True)
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eng.spot_eth = eng.initial_eth_value / first_price
    vol_ma20 = main_df['volume'].rolling(20).mean()
    vol_ma5 = main_df['volume'].rolling(5).mean()
    cooldown = 0
    min_eth_ratio = 0.05
    short_max_pnl_r = 0

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        ind = get_realtime_indicators(main_df, idx)
        eng.check_liquidation(price, dt)
        eng.charge_funding(price, dt)
        if cooldown > 0:
            cooldown -= 1

        trend = get_trend_info(data, dt, price)
        k_val = ind.get('K')
        rsi6 = ind.get('RSI6')
        cci = ind.get('CCI')

        # 各周期信号
        sig_4h = get_tf_signal(signals_all, '4h', dt)
        sig_2h = get_tf_signal(signals_all, '2h', dt)
        sig_1h = get_tf_signal(signals_all, '1h', dt)
        sig = _merge_signals(signals_all, dt)

        # 量价
        avg_vol = vol_ma20.iloc[idx] if idx >= 20 else 1
        vol5 = vol_ma5.iloc[idx] if idx >= 5 else avg_vol
        cur_vol = main_df['volume'].iloc[idx]
        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1
        is_shrinking = vol5 < avg_vol * 0.6
        is_ground_vol = vol_ratio < 0.3
        price_change_5 = (price - main_df['close'].iloc[idx - 5]) / main_df['close'].iloc[idx - 5] if idx >= 5 else 0
        price_up_vol_down = price_change_5 > 0.01 and is_shrinking

        # ===== 多维度做空评分 =====
        score = 0
        parts = []

        # 维度1: 多周期级联 (R)
        if sig_4h['top'] >= 15: score += 8; parts.append(f"4h背离{sig_4h['top']:.0f}")
        if sig_4h.get('exhaust_sell'): score += 12; parts.append("4h背驰")
        if sig_2h['top'] >= 10 or sig_2h.get('last_cross') == 'death':
            score += 5; parts.append("2h确认")

        # 维度2: 隔堆信号(S) — 权重最高
        if sig.get('sep_divs_top', 0) >= 2: score += 15; parts.append("双隔堆")
        elif sig.get('sep_divs_top', 0) >= 1 and sig.get('zero_returns_top', 0) >= 1:
            score += 12; parts.append("隔堆+零轴")
        elif sig.get('separated_top', 0) >= 1: score += 8; parts.append("隔堆")
        if sig.get('area_top_div', 0) >= 1: score += 6; parts.append("面积背离")
        if sig.get('exhaust_sell'): score += 10; parts.append("背驰")

        # 维度3: 极值区(T)
        extreme_count = 0
        if k_val and k_val > 75: extreme_count += 1
        if cci and cci > 100: extreme_count += 1
        if rsi6 and rsi6 > 65: extreme_count += 1
        if extreme_count >= 2: score += 8; parts.append(f"极值x{extreme_count}")
        elif extreme_count >= 1: score += 3

        # 维度4: 量价确认(V)
        is_death_cross = sig.get('last_cross') == 'death' or sig_1h.get('last_cross') == 'death'
        if price_up_vol_down: score += 6; parts.append("价升量减")
        if price_up_vol_down and is_death_cross: score += 4; parts.append("死叉")
        if sig.get('vol_price_up_down', 0) >= 1: score += 3

        # 趋势加成
        if trend['is_downtrend']:
            score = int(score * 1.25)
            if trend.get('short_trend_down'): score = int(score * 1.1)

        total = eng.total_value(price)
        spot_val = eng.spot_eth * price
        eth_r = spot_val / total if total > 0 else 0

        # ===== 底部信号评分 =====
        buy_score = 0
        if sig.get('exhaust_buy'): buy_score += 15
        if sig.get('sep_divs_bottom', 0) >= 1: buy_score += 10
        if sig['bottom'] >= 30: buy_score += sig['bottom'] * 0.2
        if k_val and k_val < 25: buy_score += 5
        if is_ground_vol: buy_score += 6
        if sig_1h.get('last_cross') == 'golden': buy_score += 4

        if cooldown == 0:
            # === 做空 ===
            if score >= 15:
                # 渐进减仓
                if eth_r > min_eth_ratio + 0.05:
                    sr = min(0.1 + score * 0.004, 0.45)
                    sr = min(sr, (eth_r - min_eth_ratio) * 0.9)
                    if sr > 0.05:
                        eng.spot_sell(price, dt, sr,
                            f"融合减仓 S={score} {','.join(parts[:2])}")

                # 开空
                if not eng.futures_short and score >= 20:
                    if score >= 40:
                        lev = 3; mr = 0.5
                    elif score >= 30:
                        lev = 3; mr = 0.35
                    else:
                        lev = 2; mr = 0.2
                    margin = eng.available_margin() * mr
                    eng.open_short(price, dt, margin, lev,
                        f"融合做空{lev}x S={score} {','.join(parts[:3])}")
                    short_max_pnl_r = 0
                    cooldown = 6

            # === 平空+做多 ===
            if buy_score >= 15 and eng.futures_short:
                eng.close_short(price, dt,
                    f"底信号平空 B={buy_score:.0f}")
                short_max_pnl_r = 0
                cooldown = 6

            if buy_score >= 25 and eth_r < 0.4:
                invest = eng.available_usdt() * 0.12
                eng.spot_buy(price, dt, invest, f"加仓 B={buy_score:.0f}")
                cooldown = 10

        # === 追踪止盈(U) + 固定止损 ===
        if eng.futures_short:
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
            short_max_pnl_r = max(short_max_pnl_r, pnl_r)

            # 追踪止盈
            if short_max_pnl_r >= 0.5:
                trail = short_max_pnl_r * 0.80
            elif short_max_pnl_r >= 0.3:
                trail = short_max_pnl_r * 0.70
            elif short_max_pnl_r >= 0.15:
                trail = short_max_pnl_r * 0.55
            else:
                trail = -999

            if trail > -999 and pnl_r < trail:
                eng.close_short(price, dt,
                    f"追踪止盈 max={short_max_pnl_r*100:.0f}% now={pnl_r*100:.0f}%")
                short_max_pnl_r = 0
                cooldown = 5

            # 固定止损
            if pnl_r < -0.3:
                eng.close_short(price, dt, f"止损{pnl_r*100:.0f}%")
                short_max_pnl_r = 0
                cooldown = 8

        if idx % 4 == 0:
            eng.record_history(dt, price)

    last_p = main_df['close'].iloc[-1]
    last_dt = main_df.index[-1]
    if eng.futures_short: eng.close_short(last_p, last_dt, "结束平空")
    if eng.futures_long: eng.close_long(last_p, last_dt, "结束平多")
    return eng.get_result(main_df)


# ======================================================
#   主入口
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
        ("P-基线", run_strategy_P),
        ("对照", run_baseline_spot),
        ("R", run_strategy_R),
        ("S", run_strategy_S),
        ("T", run_strategy_T),
        ("U", run_strategy_U),
        ("V", run_strategy_V),
        ("W", run_strategy_W),
    ]

    print(f"\n{'='*115}")
    print(f"  合约做空策略 Phase 2 · 基于书中核心原则进阶优化")
    print(f"  初始: 100,000 USDT + 价值100,000 USDT的ETH")
    print(f"{'='*115}")

    results = []
    for name, func in strategies:
        print(f"\n>>> 策略 {name}...")
        r = func(data, signals_all)
        results.append(r)
        liq = f" 强平:{r['liquidations']}次" if r['liquidations'] > 0 else ""
        print(f"    收益: {r['strategy_return']:+.2f}% | α: {r['alpha']:+.2f}% | "
              f"回撤: {r['max_drawdown']:.2f}% | 交易: {r['total_trades']}笔{liq} | "
              f"资产: ${r['final_total']:,.0f}")

    # 排名
    ranked = sorted(results, key=lambda x: x['alpha'], reverse=True)
    print(f"\n\n{'='*145}")
    print("                    Phase 2 合约策略排名 (按超额收益) · 含费用明细")
    print(f"{'='*145}")
    fmt = "{:>3} {:<32} {:>9} {:>9} {:>8} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}"
    print(fmt.format("#", "策略", "收益", "超额α", "回撤", "笔数",
                      "总费用", "滑点", "资金费净", "费用占比", "最终资产"))
    print("-" * 145)
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
            f"${f.get('net_funding', 0):+,.0f}",
            f"{f.get('fee_drag_pct', 0):.2f}%",
            f"${r['final_total']:,.0f}",
        ))
    print("=" * 145)

    # 费用影响分析
    print(f"\n  {'='*80}")
    print("  费用影响分析 (手续费+滑点+资金费率)")
    print(f"  {'='*80}")
    for r in ranked:
        f = r.get('fees', {})
        if f.get('total_costs', 0) > 0:
            print(f"  {r['name']:<30} "
                  f"现货费:{f.get('spot_fees',0):>7,.0f} "
                  f"合约费:{f.get('futures_fees',0):>7,.0f} "
                  f"资金费净:{f.get('net_funding',0):>+7,.0f} "
                  f"滑点:{f.get('slippage_cost',0):>7,.0f} "
                  f"总成本:{f.get('total_costs',0):>7,.0f} "
                  f"({f.get('fee_drag_pct',0):.2f}%)")

    # 找出Phase 1最佳vs Phase 2最佳的对比
    p_baseline = next((r for r in results if 'P-基线' in r['name']), None)
    best_new = ranked[0]
    if p_baseline and best_new['name'] != p_baseline['name']:
        improve = best_new['alpha'] - p_baseline['alpha']
        print(f"\n  Phase 2 最佳策略 vs Phase 1 最佳策略(P):")
        print(f"  {best_new['name']}: α={best_new['alpha']:+.2f}%  vs  P: α={p_baseline['alpha']:+.2f}%  → 提升 {improve:+.2f}%")

    # 保存
    output = {
        'phase': 2,
        'futures_results': [{
            'name': r['name'],
            'summary': {k: v for k, v in r.items() if k not in ('trades', 'history')},
            'trades': r['trades'],
            'history': r['history'],
        } for r in results],
        'best_strategy': ranked[0]['name'],
        'ranking': [{'rank': i + 1, 'name': r['name'], 'alpha': r['alpha'],
                      'return': r['strategy_return'], 'max_dd': r['max_drawdown'],
                      'trades': r['total_trades'],
                      'fees': r.get('fees', {})}
                     for i, r in enumerate(ranked)],
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'strategy_futures_v2_result.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果已保存: {output_path}")
    return output


if __name__ == '__main__':
    run_all()
