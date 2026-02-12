"""
五书融合策略 — 背离 × 均线 × 蜡烛图 × 布林带 × 量价分析

融合五本书的核心理论:
  1. 《背离技术分析》 → MACD/KDJ/CCI/RSI背离 → 捕捉拐点
  2. 《均线技术分析》 → 葛南维八法/排列/交叉 → 确认趋势
  3. 《日本蜡烛图技术》 → K线形态识别 → 微观反转信号
  4. 《布林线》 → %B/Squeeze/带宽 → 波动率+超买超卖
  5.  利弗莫尔/威科夫量价 → OBV/MFI/量价关系 → 资金流确认

融合方式:
  A. 五维加权: 每个系统独立评分后加权汇总
  B. 多数投票: ≥3个系统同向 → 高确信度交易
  C. 层级过滤: 布林带(环境) → 趋势(方向) → 背离+K线(触发) → 量价(确认)
  D. 最强信号跟随: 取五个系统中最强的信号+至少1个验证

初始: 10万USDT + 价值10万USDT的ETH
数据: 币安 ETH/USDT, 1h K线
"""

import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binance_fetcher import fetch_binance_klines
from indicators import add_all_indicators
from strategy_enhanced import analyze_signals_enhanced, get_signal_at, DEFAULT_SIG
from strategy_futures import FuturesEngine
from strategy_futures_v4 import _calc_top_score, _calc_bottom_score
from ma_indicators import add_moving_averages, compute_ma_signals
from candlestick_patterns import compute_candlestick_scores, candle_features
from bollinger_strategy import compute_bollinger_scores, compute_bollinger
from volume_price_strategy import compute_volume_price_scores, compute_volume_indicators


# ======================================================
#   数据获取
# ======================================================
def fetch_data():
    """获取多周期数据"""
    print("获取数据...")
    data = {}
    for tf, days in [('1h', 60), ('4h', 60), ('8h', 60)]:
        df = fetch_binance_klines("ETHUSDT", interval=tf, days=days)
        if df is not None and len(df) > 50:
            df = add_all_indicators(df)
            add_moving_averages(df, timeframe=tf)
            data[tf] = df
            print(f"  {tf}: {len(df)} 条 ({df.index[0]} ~ {df.index[-1]})")
    return data


# ======================================================
#   五维信号计算
# ======================================================
def compute_all_signals(data):
    """预计算所有五个系统的信号"""
    main_df = data['1h']
    signals = {}

    # 1. 背离信号
    print("  [1/5] 计算背离信号...")
    div_signals_1h = analyze_signals_enhanced(main_df, 168)
    div_signals_8h = {}
    if '8h' in data:
        div_signals_8h = analyze_signals_enhanced(data['8h'], 90)
    signals['div_1h'] = div_signals_1h
    signals['div_8h'] = div_signals_8h
    print(f"    1h背离: {len(div_signals_1h)} | 8h背离: {len(div_signals_8h)}")

    # 2. 均线信号
    print("  [2/5] 计算均线信号...")
    ma_signals = compute_ma_signals(main_df, timeframe='1h')
    signals['ma'] = ma_signals
    ma_buy = int((ma_signals['buy_score'] > 15).sum())
    ma_sell = int((ma_signals['sell_score'] > 15).sum())
    print(f"    均线买: {ma_buy} | 均线卖: {ma_sell}")

    # 3. 蜡烛图形态
    print("  [3/5] 扫描蜡烛图形态...")
    cs_sell, cs_buy, cs_names = compute_candlestick_scores(main_df)
    signals['cs_sell'] = cs_sell
    signals['cs_buy'] = cs_buy
    signals['cs_names'] = cs_names
    cs_active = int((cs_sell > 0).sum() + (cs_buy > 0).sum())
    print(f"    K线形态信号: {cs_active}个")

    # 4. 布林带
    print("  [4/5] 计算布林带指标...")
    bb_sell, bb_buy, bb_names = compute_bollinger_scores(main_df)
    signals['bb_sell'] = bb_sell
    signals['bb_buy'] = bb_buy
    signals['bb_names'] = bb_names
    print(f"    布林带信号: {int((bb_sell > 15).sum() + (bb_buy > 15).sum())}个")

    # 5. 量价分析
    print("  [5/5] 计算量价指标...")
    vp_sell, vp_buy, vp_names = compute_volume_price_scores(main_df)
    signals['vp_sell'] = vp_sell
    signals['vp_buy'] = vp_buy
    signals['vp_names'] = vp_names
    print(f"    量价信号: {int((vp_sell > 15).sum() + (vp_buy > 15).sum())}个")

    return signals


def calc_fusion_score(signals, data, idx, dt, config):
    """
    五维融合信号计算

    返回: sell_score, buy_score, reasons_sell, reasons_buy
    """
    mode = config.get('fusion_mode', 'weighted')
    main_df = data['1h']
    price = main_df['close'].iloc[idx]

    # --- 获取五个系统的分数 ---
    # 1. 背离
    sig_1h = get_signal_at(signals['div_1h'], dt) or dict(DEFAULT_SIG)
    sig_8h = get_signal_at(signals['div_8h'], dt) or dict(DEFAULT_SIG)
    merged_div = dict(DEFAULT_SIG)
    merged_div['top'] = 0; merged_div['bottom'] = 0
    for sig_src, w in [(sig_1h, 1.0), (sig_8h, 0.5)]:
        merged_div['top'] += sig_src.get('top', 0) * w
        merged_div['bottom'] += sig_src.get('bottom', 0) * w
        for k in DEFAULT_SIG:
            if isinstance(DEFAULT_SIG[k], bool) and sig_src.get(k):
                merged_div[k] = True
            elif isinstance(DEFAULT_SIG[k], (int, float)) and k not in ('top', 'bottom'):
                merged_div[k] = max(merged_div.get(k, 0), sig_src.get(k, 0))

    # 简化趋势判断
    trend = {'is_downtrend': False, 'is_uptrend': False,
             'ma_bearish': False, 'ma_bullish': False,
             'ma_slope_down': False, 'ma_slope_up': False}
    if idx >= 30:
        c5 = main_df['close'].iloc[max(0, idx - 5):idx].mean()
        c20 = main_df['close'].iloc[max(0, idx - 20):idx].mean()
        if c5 < c20 * 0.99: trend['is_downtrend'] = True
        elif c5 > c20 * 1.01: trend['is_uptrend'] = True

    div_sell, _ = _calc_top_score(merged_div, trend)
    div_buy = _calc_bottom_score(merged_div, trend)

    # 2. 均线
    ma_sell = float(signals['ma']['sell_score'].iloc[idx]) if idx < len(signals['ma']['sell_score']) else 0
    ma_buy = float(signals['ma']['buy_score'].iloc[idx]) if idx < len(signals['ma']['buy_score']) else 0

    # 3. 蜡烛图
    cs_sell = float(signals['cs_sell'].iloc[idx])
    cs_buy = float(signals['cs_buy'].iloc[idx])

    # 4. 布林带
    bb_sell = float(signals['bb_sell'].iloc[idx])
    bb_buy = float(signals['bb_buy'].iloc[idx])

    # 5. 量价
    vp_sell = float(signals['vp_sell'].iloc[idx])
    vp_buy = float(signals['vp_buy'].iloc[idx])

    # --- 权重配置 ---
    w_div = config.get('w_div', 0.30)
    w_ma = config.get('w_ma', 0.20)
    w_cs = config.get('w_cs', 0.15)
    w_bb = config.get('w_bb', 0.20)
    w_vp = config.get('w_vp', 0.15)

    sell_score = 0; buy_score = 0
    r_sell = []; r_buy = []

    if mode == 'weighted':
        # 五维加权
        sell_score = (div_sell * w_div + ma_sell * w_ma + cs_sell * w_cs +
                      bb_sell * w_bb + vp_sell * w_vp)
        buy_score = (div_buy * w_div + ma_buy * w_ma + cs_buy * w_cs +
                     bb_buy * w_bb + vp_buy * w_vp)

        if div_sell > 0: r_sell.append(f"背离{div_sell:.0f}")
        if ma_sell > 0: r_sell.append(f"均线{ma_sell:.0f}")
        if cs_sell > 0: r_sell.append(f"K线{cs_sell:.0f}")
        if bb_sell > 0: r_sell.append(f"BB{bb_sell:.0f}")
        if vp_sell > 0: r_sell.append(f"量价{vp_sell:.0f}")

        if div_buy > 0: r_buy.append(f"背离{div_buy:.0f}")
        if ma_buy > 0: r_buy.append(f"均线{ma_buy:.0f}")
        if cs_buy > 0: r_buy.append(f"K线{cs_buy:.0f}")
        if bb_buy > 0: r_buy.append(f"BB{bb_buy:.0f}")
        if vp_buy > 0: r_buy.append(f"量价{vp_buy:.0f}")

    elif mode == 'vote':
        # 多数投票: ≥3个系统同向
        threshold = config.get('vote_threshold', 15)
        sell_votes = sum(1 for s in [div_sell, ma_sell, cs_sell, bb_sell, vp_sell] if s >= threshold)
        buy_votes = sum(1 for s in [div_buy, ma_buy, cs_buy, bb_buy, vp_buy] if s >= threshold)

        min_votes = config.get('min_votes', 3)

        if sell_votes >= min_votes:
            sell_score = (div_sell + ma_sell + cs_sell + bb_sell + vp_sell) / 5 * (1 + sell_votes * 0.15)
            r_sell.append(f"{sell_votes}票看空")
        if buy_votes >= min_votes:
            buy_score = (div_buy + ma_buy + cs_buy + bb_buy + vp_buy) / 5 * (1 + buy_votes * 0.15)
            r_buy.append(f"{buy_votes}票看涨")

    elif mode == 'layered':
        # 层级过滤: 布林带环境 → 趋势方向 → 触发 → 确认
        # Layer 1: 布林带环境
        bb_bearish = bb_sell >= 20
        bb_bullish = bb_buy >= 20

        # Layer 2: 均线趋势
        ma_bearish = ma_sell >= 15
        ma_bullish = ma_buy >= 15

        # Layer 3: 背离+K线触发
        trigger_sell = (div_sell >= 15) or (cs_sell >= 30)
        trigger_buy = (div_buy >= 15) or (cs_buy >= 30)

        # Layer 4: 量价确认
        vp_confirm_sell = vp_sell >= 10
        vp_confirm_buy = vp_buy >= 10

        if trigger_sell:
            score = div_sell * 0.4 + cs_sell * 0.3
            if bb_bearish: score *= 1.3; r_sell.append("BB看空环境")
            if ma_bearish: score *= 1.2; r_sell.append("均线空头")
            if vp_confirm_sell: score *= 1.15; r_sell.append("量价确认")
            sell_score = score
            if div_sell > 0: r_sell.append(f"背离{div_sell:.0f}")
            if cs_sell > 0: r_sell.append(f"K线{cs_sell:.0f}")

        if trigger_buy:
            score = div_buy * 0.4 + cs_buy * 0.3
            if bb_bullish: score *= 1.3; r_buy.append("BB看涨环境")
            if ma_bullish: score *= 1.2; r_buy.append("均线多头")
            if vp_confirm_buy: score *= 1.15; r_buy.append("量价确认")
            buy_score = score
            if div_buy > 0: r_buy.append(f"背离{div_buy:.0f}")
            if cs_buy > 0: r_buy.append(f"K线{cs_buy:.0f}")

    elif mode == 'strongest':
        # 最强信号 + 验证
        scores_sell = [('背离', div_sell), ('均线', ma_sell), ('K线', cs_sell),
                       ('BB', bb_sell), ('量价', vp_sell)]
        scores_buy = [('背离', div_buy), ('均线', ma_buy), ('K线', cs_buy),
                      ('BB', bb_buy), ('量价', vp_buy)]

        # 卖出: 取最强 + 验证
        sorted_sell = sorted(scores_sell, key=lambda x: -x[1])
        if sorted_sell[0][1] >= 30:
            confirmations = sum(1 for _, s in sorted_sell[1:] if s >= 10)
            if confirmations >= 1:
                sell_score = sorted_sell[0][1] * (1 + confirmations * 0.1)
                r_sell.append(f"最强:{sorted_sell[0][0]}={sorted_sell[0][1]:.0f}")
                r_sell.append(f"{confirmations}个验证")

        sorted_buy = sorted(scores_buy, key=lambda x: -x[1])
        if sorted_buy[0][1] >= 30:
            confirmations = sum(1 for _, s in sorted_buy[1:] if s >= 10)
            if confirmations >= 1:
                buy_score = sorted_buy[0][1] * (1 + confirmations * 0.1)
                r_buy.append(f"最强:{sorted_buy[0][0]}={sorted_buy[0][1]:.0f}")
                r_buy.append(f"{confirmations}个验证")

    elif mode == 'c6_base':
        # === C6底座模式 ===
        # 核心: 背离(70%) + 均线(30%) = C6原始逻辑
        # 新三书(K线/布林/量价)仅作为确认/加成因子, 不参与基础分数
        base_sell = div_sell * 0.7 + ma_sell * 0.3
        base_buy = div_buy * 0.7 + ma_buy * 0.3

        if div_sell > 0: r_sell.append(f"背离{div_sell:.0f}")
        if ma_sell > 0: r_sell.append(f"均线{ma_sell:.0f}")
        if div_buy > 0: r_buy.append(f"背离{div_buy:.0f}")
        if ma_buy > 0: r_buy.append(f"均线{ma_buy:.0f}")

        # 均线排列通用加成(复刻C6逻辑)
        if idx >= 30:
            c5 = main_df['close'].iloc[max(0, idx - 5):idx].mean()
            c20 = main_df['close'].iloc[max(0, idx - 20):idx].mean()
            c60 = main_df['close'].iloc[max(0, idx - 60):idx].mean() if idx >= 60 else c20
            if c5 < c20 < c60:  # 空头排列
                base_sell *= 1.1
                r_sell.append("空头排列")
            elif c5 > c20 > c60:  # 多头排列
                base_buy *= 1.1
                r_buy.append("多头排列")

        # --- 新三书确认层 ---
        # 计算新三书的"确认系数": 有确认=加成, 无确认=保持, 矛盾=轻微削弱
        confirm_mult = config.get('confirm_mult', 0.12)  # 每个确认的加成幅度
        contra_mult = config.get('contra_mult', 0.05)    # 每个反对的削弱幅度

        # 卖出确认
        sell_confirms = 0; sell_contras = 0
        if bb_sell >= 20: sell_confirms += 1; r_sell.append(f"BB确认{bb_sell:.0f}")
        elif bb_buy >= 30: sell_contras += 1
        if vp_sell >= 15: sell_confirms += 1; r_sell.append(f"量价确认{vp_sell:.0f}")
        elif vp_buy >= 25: sell_contras += 1
        if cs_sell >= 30: sell_confirms += 1; r_sell.append(f"K线确认{cs_sell:.0f}")
        elif cs_buy >= 40: sell_contras += 1

        sell_score = base_sell * (1 + sell_confirms * confirm_mult - sell_contras * contra_mult)

        # 买入确认
        buy_confirms = 0; buy_contras = 0
        if bb_buy >= 20: buy_confirms += 1; r_buy.append(f"BB确认{bb_buy:.0f}")
        elif bb_sell >= 30: buy_contras += 1
        if vp_buy >= 15: buy_confirms += 1; r_buy.append(f"量价确认{vp_buy:.0f}")
        elif vp_sell >= 25: buy_contras += 1
        if cs_buy >= 30: buy_confirms += 1; r_buy.append(f"K线确认{cs_buy:.0f}")
        elif cs_sell >= 40: buy_contras += 1

        buy_score = base_buy * (1 + buy_confirms * confirm_mult - buy_contras * contra_mult)

    elif mode == 'c6_bb_boost':
        # === C6底座 + 布林带深度整合 ===
        # 布林带是单书表现最好的(B2: α=+17.72%), 给予更大的融合角色
        base_sell = div_sell * 0.65 + ma_sell * 0.25
        base_buy = div_buy * 0.65 + ma_buy * 0.25

        if div_sell > 0: r_sell.append(f"背离{div_sell:.0f}")
        if ma_sell > 0: r_sell.append(f"均线{ma_sell:.0f}")
        if div_buy > 0: r_buy.append(f"背离{div_buy:.0f}")
        if ma_buy > 0: r_buy.append(f"均线{ma_buy:.0f}")

        # 布林带作为环境加成(不仅确认, 还提供独立的波动率洞察)
        bb_env_sell = 1.0
        bb_env_buy = 1.0
        if bb_sell >= 30:
            bb_env_sell = 1.25  # 布林带强烈看空环境
            r_sell.append(f"BB环境{bb_sell:.0f}")
        elif bb_sell >= 15:
            bb_env_sell = 1.12
            r_sell.append(f"BB辅助{bb_sell:.0f}")
        if bb_buy >= 30:
            bb_env_buy = 1.25
            r_buy.append(f"BB环境{bb_buy:.0f}")
        elif bb_buy >= 15:
            bb_env_buy = 1.12
            r_buy.append(f"BB辅助{bb_buy:.0f}")

        # 布林带也可提供独立信号(当基础分数低但BB很强时)
        if base_sell < 15 and bb_sell >= 40:
            base_sell = max(base_sell, bb_sell * 0.5)
            r_sell.append("BB主导触发")
        if base_buy < 15 and bb_buy >= 40:
            base_buy = max(base_buy, bb_buy * 0.5)
            r_buy.append("BB主导触发")

        sell_score = base_sell * bb_env_sell
        buy_score = base_buy * bb_env_buy

        # 量价微确认
        if vp_sell >= 15: sell_score *= 1.08; r_sell.append(f"量价{vp_sell:.0f}")
        if vp_buy >= 15: buy_score *= 1.08; r_buy.append(f"量价{vp_buy:.0f}")

    elif mode == 'c6_veto':
        # === C6底座 + 新三书否决权 ===
        # 核心逻辑不变, 但当新三书强烈反对时可以否决交易
        base_sell = div_sell * 0.7 + ma_sell * 0.3
        base_buy = div_buy * 0.7 + ma_buy * 0.3

        if div_sell > 0: r_sell.append(f"背离{div_sell:.0f}")
        if ma_sell > 0: r_sell.append(f"均线{ma_sell:.0f}")
        if div_buy > 0: r_buy.append(f"背离{div_buy:.0f}")
        if ma_buy > 0: r_buy.append(f"均线{ma_buy:.0f}")

        # 否决逻辑: 至少2/3个新系统强烈反对 → 大幅削弱信号
        veto_threshold = config.get('veto_threshold', 25)
        sell_vetoes = sum(1 for s in [bb_buy, vp_buy, cs_buy] if s >= veto_threshold)
        buy_vetoes = sum(1 for s in [bb_sell, vp_sell, cs_sell] if s >= veto_threshold)

        if sell_vetoes >= 2:
            sell_score = base_sell * 0.3  # 两个系统否决, 大幅削弱
            r_sell.append(f"被否决({sell_vetoes}/3)")
        else:
            # 正面确认加成
            sell_bonus = 0
            if bb_sell >= 15: sell_bonus += 0.10; r_sell.append(f"BB{bb_sell:.0f}")
            if vp_sell >= 15: sell_bonus += 0.08; r_sell.append(f"量价{vp_sell:.0f}")
            if cs_sell >= 25: sell_bonus += 0.06; r_sell.append(f"K线{cs_sell:.0f}")
            sell_score = base_sell * (1 + sell_bonus)

        if buy_vetoes >= 2:
            buy_score = base_buy * 0.3
            r_buy.append(f"被否决({buy_vetoes}/3)")
        else:
            buy_bonus = 0
            if bb_buy >= 15: buy_bonus += 0.10; r_buy.append(f"BB{bb_buy:.0f}")
            if vp_buy >= 15: buy_bonus += 0.08; r_buy.append(f"量价{vp_buy:.0f}")
            if cs_buy >= 25: buy_bonus += 0.06; r_buy.append(f"K线{cs_buy:.0f}")
            buy_score = base_buy * (1 + buy_bonus)

    return sell_score, buy_score, r_sell, r_buy


# ======================================================
#   策略执行器
# ======================================================
def run_fusion_strategy(data, signals, config, trade_days=None):
    """五书融合策略回测"""
    eng = FuturesEngine(config['name'], max_leverage=config.get('max_lev', 5))
    main_df = data['1h']

    start_dt = None
    if trade_days and trade_days < 60:
        start_dt = main_df.index[-1] - pd.Timedelta(days=trade_days)

    init_idx = 0
    if start_dt:
        init_idx = main_df.index.searchsorted(start_dt)
        if init_idx >= len(main_df): init_idx = 0
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[init_idx]

    eng.max_single_margin = eng.initial_total * config.get('single_pct', 0.15)
    eng.max_margin_total = eng.initial_total * config.get('total_pct', 0.40)
    eng.max_lifetime_margin = eng.initial_total * config.get('lifetime_pct', 5.0)

    sell_threshold = config.get('sell_threshold', 25)
    buy_threshold = config.get('buy_threshold', 25)
    short_threshold = config.get('short_threshold', 35)
    long_threshold = config.get('long_threshold', 35)
    sell_pct = config.get('sell_pct', 0.45)
    margin_use = config.get('margin_use', 0.50)
    lev = config.get('lev', 3)
    cooldown = config.get('cooldown', 4)
    spot_cooldown = config.get('spot_cooldown', 12)
    short_sl = config.get('short_sl', -0.20)
    short_tp = config.get('short_tp', 0.60)
    short_trail = config.get('short_trail', 0.25)
    short_max_hold = config.get('short_max_hold', 72)
    long_sl = config.get('long_sl', -0.15)
    long_tp = config.get('long_tp', 0.50)
    long_trail = config.get('long_trail', 0.20)
    long_max_hold = config.get('long_max_hold', 72)

    short_cd = 0; long_cd = 0; spot_cd = 0
    short_bars = 0; long_bars = 0
    short_max_pnl = 0; long_max_pnl = 0

    for idx in range(60, len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]

        if start_dt and dt < start_dt:
            if idx % 4 == 0: eng.record_history(dt, price)
            continue

        eng.check_liquidation(price, dt)
        short_just_opened = False; long_just_opened = False

        eng.funding_counter += 1
        if eng.funding_counter % 8 == 0:
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

        ss, bs, r_sell, r_buy = calc_fusion_score(signals, data, idx, dt, config)

        in_conflict = False
        if ss > 0 and bs > 0:
            ratio = min(ss, bs) / max(ss, bs)
            in_conflict = ratio >= 0.6 and min(ss, bs) >= 15

        reason_s = ' '.join(r_sell[:3])
        reason_b = ' '.join(r_buy[:3])

        # 卖出现货
        if ss >= sell_threshold and spot_cd == 0 and not in_conflict and eng.spot_eth * price > 500:
            eng.spot_sell(price, dt, sell_pct, f"五书卖 {reason_s}")
            spot_cd = spot_cooldown

        # 开空
        sell_dom = (ss > bs * 1.5) if bs > 0 else True
        if short_cd == 0 and ss >= short_threshold and not eng.futures_short and sell_dom:
            margin = eng.available_margin() * margin_use
            actual_lev = min(lev if ss >= 50 else min(lev, 3) if ss >= 35 else 2, eng.max_leverage)
            eng.open_short(price, dt, margin, actual_lev,
                           f"五书空 {actual_lev}x {reason_s}")
            short_max_pnl = 0; short_bars = 0; short_cd = cooldown; short_just_opened = True

        # 管理空仓
        if eng.futures_short and not short_just_opened:
            short_bars += 1
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
            if pnl_r >= short_tp:
                eng.close_short(price, dt, f"止盈 +{pnl_r*100:.0f}%")
                short_max_pnl = 0; short_cd = cooldown * 2; short_bars = 0
            else:
                if pnl_r > short_max_pnl: short_max_pnl = pnl_r
                if short_max_pnl >= short_trail and eng.futures_short:
                    if pnl_r < short_max_pnl * 0.60:
                        eng.close_short(price, dt, "追踪止盈")
                        short_max_pnl = 0; short_cd = cooldown; short_bars = 0
                if eng.futures_short and bs >= config.get('close_short_bs', 40):
                    bs_dom = (ss < bs * 0.7) if bs > 0 else True
                    if bs_dom:
                        eng.close_short(price, dt, f"反向信号平空 BS={bs:.0f}")
                        short_max_pnl = 0; short_cd = cooldown * 3; short_bars = 0
                if eng.futures_short and pnl_r < short_sl:
                    eng.close_short(price, dt, f"止损 {pnl_r*100:.0f}%")
                    short_max_pnl = 0; short_cd = cooldown * 4; short_bars = 0
                if eng.futures_short and short_bars >= short_max_hold:
                    eng.close_short(price, dt, "超时")
                    short_max_pnl = 0; short_cd = cooldown; short_bars = 0
        elif eng.futures_short and short_just_opened:
            short_bars = 1

        # 买入现货
        if bs >= buy_threshold and spot_cd == 0 and not in_conflict and eng.available_usdt() > 500:
            eng.spot_buy(price, dt, eng.available_usdt() * 0.25,
                         f"五书买 {reason_b}")
            spot_cd = spot_cooldown

        # 开多
        buy_dom = (bs > ss * 1.5) if ss > 0 else True
        if long_cd == 0 and bs >= long_threshold and not eng.futures_long and buy_dom:
            margin = eng.available_margin() * margin_use
            actual_lev = min(lev if bs >= 50 else min(lev, 3) if bs >= 35 else 2, eng.max_leverage)
            eng.open_long(price, dt, margin, actual_lev,
                          f"五书多 {actual_lev}x {reason_b}")
            long_max_pnl = 0; long_bars = 0; long_cd = cooldown; long_just_opened = True

        # 管理多仓
        if eng.futures_long and not long_just_opened:
            long_bars += 1
            pnl_r = eng.futures_long.calc_pnl(price) / eng.futures_long.margin
            if pnl_r >= long_tp:
                eng.close_long(price, dt, f"止盈 +{pnl_r*100:.0f}%")
                long_max_pnl = 0; long_cd = cooldown * 2; long_bars = 0
            else:
                if pnl_r > long_max_pnl: long_max_pnl = pnl_r
                if long_max_pnl >= long_trail and eng.futures_long:
                    if pnl_r < long_max_pnl * 0.60:
                        eng.close_long(price, dt, "追踪止盈")
                        long_max_pnl = 0; long_cd = cooldown; long_bars = 0
                if eng.futures_long and ss >= config.get('close_long_ss', 40):
                    ss_dom = (bs < ss * 0.7) if ss > 0 else True
                    if ss_dom:
                        eng.close_long(price, dt, f"反向信号平多 SS={ss:.0f}")
                        long_max_pnl = 0; long_cd = cooldown * 3; long_bars = 0
                if eng.futures_long and pnl_r < long_sl:
                    eng.close_long(price, dt, f"止损 {pnl_r*100:.0f}%")
                    long_max_pnl = 0; long_cd = cooldown * 4; long_bars = 0
                if eng.futures_long and long_bars >= long_max_hold:
                    eng.close_long(price, dt, "超时")
                    long_max_pnl = 0; long_cd = cooldown; long_bars = 0
        elif eng.futures_long and long_just_opened:
            long_bars = 1

        if idx % 4 == 0: eng.record_history(dt, price)

    last_price = main_df['close'].iloc[-1]
    last_dt = main_df.index[-1]
    if eng.futures_short: eng.close_short(last_price, last_dt, "期末平仓")
    if eng.futures_long: eng.close_long(last_price, last_dt, "期末平仓")

    if start_dt:
        trade_df = main_df[main_df.index >= start_dt]
        if len(trade_df) > 1: return eng.get_result(trade_df)
    return eng.get_result(main_df)


# ======================================================
#   策略变体
# ======================================================
def get_strategies():
    base = {
        'single_pct': 0.15, 'total_pct': 0.40,
        'sell_threshold': 25, 'buy_threshold': 25,
        'short_threshold': 35, 'long_threshold': 35,
        'close_short_bs': 40, 'close_long_ss': 40,
        'sell_pct': 0.45, 'margin_use': 0.50, 'lev': 3,
        'short_sl': -0.20, 'short_tp': 0.60, 'short_trail': 0.25,
        'short_max_hold': 72, 'long_sl': -0.15, 'long_tp': 0.50,
        'long_trail': 0.20, 'long_max_hold': 72,
        'cooldown': 4, 'spot_cooldown': 12, 'max_lev': 5,
        'fusion_mode': 'weighted',
        'w_div': 0.30, 'w_ma': 0.20, 'w_cs': 0.15, 'w_bb': 0.20, 'w_vp': 0.15,
    }

    return [
        # F1: 五维加权(标准)
        {**base, 'name': 'F1: 五维加权'},

        # F2: 多数投票
        {**base, 'name': 'F2: 多数投票',
         'fusion_mode': 'vote', 'min_votes': 3, 'vote_threshold': 15},

        # F3: 层级过滤
        {**base, 'name': 'F3: 层级过滤',
         'fusion_mode': 'layered'},

        # F4: 最强+验证
        {**base, 'name': 'F4: 最强+验证',
         'fusion_mode': 'strongest'},

        # F5: 加权(布林主导)
        {**base, 'name': 'F5: BB主导加权',
         'w_div': 0.15, 'w_ma': 0.15, 'w_cs': 0.10, 'w_bb': 0.40, 'w_vp': 0.20},

        # F6: 激进做空(五维)
        {**base, 'name': 'F6: 激进五维做空',
         'w_div': 0.25, 'w_ma': 0.15, 'w_cs': 0.15, 'w_bb': 0.25, 'w_vp': 0.20,
         'sell_threshold': 18, 'short_threshold': 25,
         'buy_threshold': 25, 'long_threshold': 40,
         'lev': 5, 'max_lev': 5, 'margin_use': 0.70,
         'single_pct': 0.20, 'total_pct': 0.50,
         'sell_pct': 0.55, 'short_sl': -0.30, 'short_tp': 0.80,
         'short_trail': 0.25, 'short_max_hold': 72,
         'cooldown': 4, 'spot_cooldown': 12},

        # F7: 三票做空
        {**base, 'name': 'F7: 三票激进做空',
         'fusion_mode': 'vote', 'min_votes': 3, 'vote_threshold': 12,
         'sell_threshold': 15, 'short_threshold': 20,
         'buy_threshold': 25, 'long_threshold': 40,
         'lev': 5, 'max_lev': 5, 'margin_use': 0.70,
         'single_pct': 0.20, 'total_pct': 0.50,
         'sell_pct': 0.55, 'short_sl': -0.30, 'short_tp': 0.80},

        # F8: 保守共识
        {**base, 'name': 'F8: 保守共识',
         'fusion_mode': 'vote', 'min_votes': 4, 'vote_threshold': 20,
         'sell_threshold': 35, 'buy_threshold': 35,
         'short_threshold': 50, 'long_threshold': 50,
         'lev': 2, 'max_lev': 2, 'margin_use': 0.30,
         'sell_pct': 0.30, 'short_sl': -0.12, 'long_sl': -0.10,
         'spot_cooldown': 24},

        # ========== C6底座模式 (新三书仅做确认) ==========

        # F9: C6底座+三书确认(标准) — 背离70%+均线30%, 新三书加成/削弱
        {**base, 'name': 'F9: C6底座+确认',
         'fusion_mode': 'c6_base',
         'confirm_mult': 0.12, 'contra_mult': 0.05,
         'sell_threshold': 18, 'short_threshold': 25,
         'buy_threshold': 25, 'long_threshold': 40,
         'lev': 5, 'max_lev': 5, 'margin_use': 0.70,
         'single_pct': 0.20, 'total_pct': 0.50,
         'sell_pct': 0.55, 'short_sl': -0.30, 'short_tp': 0.80,
         'short_trail': 0.25, 'short_max_hold': 72,
         'cooldown': 4, 'spot_cooldown': 12},

        # F10: C6底座+强确认 — 确认加成更大
        {**base, 'name': 'F10: C6底座+强确认',
         'fusion_mode': 'c6_base',
         'confirm_mult': 0.18, 'contra_mult': 0.08,
         'sell_threshold': 18, 'short_threshold': 25,
         'buy_threshold': 25, 'long_threshold': 40,
         'lev': 5, 'max_lev': 5, 'margin_use': 0.70,
         'single_pct': 0.20, 'total_pct': 0.50,
         'sell_pct': 0.55, 'short_sl': -0.30, 'short_tp': 0.80,
         'short_trail': 0.25, 'short_max_hold': 72,
         'cooldown': 4, 'spot_cooldown': 12},

        # F11: C6底座+布林深度整合 — BB作为环境因子
        {**base, 'name': 'F11: C6+布林深度',
         'fusion_mode': 'c6_bb_boost',
         'sell_threshold': 18, 'short_threshold': 25,
         'buy_threshold': 25, 'long_threshold': 40,
         'lev': 5, 'max_lev': 5, 'margin_use': 0.70,
         'single_pct': 0.20, 'total_pct': 0.50,
         'sell_pct': 0.55, 'short_sl': -0.30, 'short_tp': 0.80,
         'short_trail': 0.25, 'short_max_hold': 72,
         'cooldown': 4, 'spot_cooldown': 12},

        # F12: C6底座+三书否决权 — 新三书可以否决错误信号
        {**base, 'name': 'F12: C6+三书否决',
         'fusion_mode': 'c6_veto',
         'veto_threshold': 25,
         'sell_threshold': 18, 'short_threshold': 25,
         'buy_threshold': 25, 'long_threshold': 40,
         'lev': 5, 'max_lev': 5, 'margin_use': 0.70,
         'single_pct': 0.20, 'total_pct': 0.50,
         'sell_pct': 0.55, 'short_sl': -0.30, 'short_tp': 0.80,
         'short_trail': 0.25, 'short_max_hold': 72,
         'cooldown': 4, 'spot_cooldown': 12},

        # F13: C6底座+确认(宽松) — 更低的确认门槛
        {**base, 'name': 'F13: C6+宽松确认',
         'fusion_mode': 'c6_base',
         'confirm_mult': 0.15, 'contra_mult': 0.03,
         'sell_threshold': 15, 'short_threshold': 22,
         'buy_threshold': 22, 'long_threshold': 38,
         'lev': 5, 'max_lev': 5, 'margin_use': 0.70,
         'single_pct': 0.20, 'total_pct': 0.50,
         'sell_pct': 0.55, 'short_sl': -0.30, 'short_tp': 0.80,
         'short_trail': 0.25, 'short_max_hold': 72,
         'cooldown': 4, 'spot_cooldown': 10},

        # F14: C6+BB深度+否决混合 — 布林带环境加成 + 否决保护
        {**base, 'name': 'F14: C6+BB+否决',
         'fusion_mode': 'c6_bb_boost',
         'sell_threshold': 16, 'short_threshold': 22,
         'buy_threshold': 23, 'long_threshold': 38,
         'lev': 5, 'max_lev': 5, 'margin_use': 0.75,
         'single_pct': 0.22, 'total_pct': 0.55,
         'sell_pct': 0.60, 'short_sl': -0.30, 'short_tp': 0.80,
         'short_trail': 0.22, 'short_max_hold': 72,
         'cooldown': 3, 'spot_cooldown': 10},

        # F15: C6+BB深度(低频) — 减少交易以降低费用
        {**base, 'name': 'F15: C6+BB低频',
         'fusion_mode': 'c6_bb_boost',
         'sell_threshold': 20, 'short_threshold': 28,
         'buy_threshold': 28, 'long_threshold': 42,
         'lev': 5, 'max_lev': 5, 'margin_use': 0.70,
         'single_pct': 0.20, 'total_pct': 0.50,
         'sell_pct': 0.55, 'short_sl': -0.30, 'short_tp': 0.80,
         'short_trail': 0.25, 'short_max_hold': 72,
         'cooldown': 6, 'spot_cooldown': 16,
         'close_short_bs': 45, 'close_long_ss': 45},

        # F16: C6+BB极致 — 完全复刻C6参数 + BB环境加成
        {**base, 'name': 'F16: C6精确+BB',
         'fusion_mode': 'c6_bb_boost',
         'sell_threshold': 18, 'short_threshold': 25,
         'buy_threshold': 25, 'long_threshold': 40,
         'lev': 5, 'max_lev': 5, 'margin_use': 0.70,
         'single_pct': 0.20, 'total_pct': 0.50,
         'sell_pct': 0.55, 'short_sl': -0.30, 'short_tp': 0.80,
         'short_trail': 0.25, 'short_max_hold': 72,
         'cooldown': 4, 'spot_cooldown': 12,
         'close_short_bs': 40, 'close_long_ss': 40},

        # F17: C6精确+否决 — 完全复刻C6参数 + 三书否决
        {**base, 'name': 'F17: C6精确+否决',
         'fusion_mode': 'c6_veto',
         'veto_threshold': 25,
         'sell_threshold': 18, 'short_threshold': 25,
         'buy_threshold': 25, 'long_threshold': 40,
         'lev': 5, 'max_lev': 5, 'margin_use': 0.70,
         'single_pct': 0.20, 'total_pct': 0.50,
         'sell_pct': 0.55, 'short_sl': -0.30, 'short_tp': 0.80,
         'short_trail': 0.25, 'short_max_hold': 72,
         'cooldown': 4, 'spot_cooldown': 12,
         'close_short_bs': 40, 'close_long_ss': 40},

        # F18: C6精确+确认(低频) — 最大化减少噪音
        {**base, 'name': 'F18: C6精确+确认',
         'fusion_mode': 'c6_base',
         'confirm_mult': 0.15, 'contra_mult': 0.06,
         'sell_threshold': 18, 'short_threshold': 25,
         'buy_threshold': 25, 'long_threshold': 40,
         'lev': 5, 'max_lev': 5, 'margin_use': 0.70,
         'single_pct': 0.20, 'total_pct': 0.50,
         'sell_pct': 0.55, 'short_sl': -0.30, 'short_tp': 0.80,
         'short_trail': 0.25, 'short_max_hold': 72,
         'cooldown': 4, 'spot_cooldown': 12,
         'close_short_bs': 40, 'close_long_ss': 40},
    ]


# ======================================================
#   主函数
# ======================================================
def main(trade_days=None):
    if trade_days is None:
        trade_days = 30

    data = fetch_data()
    if '1h' not in data:
        print("错误: 无法获取1h数据")
        return

    main_df = data['1h']

    print("\n预计算五个信号系统...")
    signals = compute_all_signals(data)

    strategies = get_strategies()
    all_results = []

    start_dt = main_df.index[-1] - pd.Timedelta(days=trade_days)
    trade_start = str(start_dt)[:16]
    trade_end = str(main_df.index[-1])[:16]

    print(f"\n{'=' * 120}")
    print(f"  五书融合策略 · 背离×均线×蜡烛图×布林带×量价 · {len(strategies)}种 · 最近{trade_days}天")
    print(f"  数据: {len(main_df)}根1h K线 | 交易: {trade_start} ~ {trade_end}")
    print(f"  五本书: 《背离技术》《均线技术》《日本蜡烛图技术》《布林线》利弗莫尔量价")
    print(f"{'=' * 120}")

    print(f"\n{'策略':<24} {'α':>8} {'收益':>10} {'BH':>10} {'回撤':>8} "
          f"{'交易':>6} {'强平':>4} {'费用':>10}")
    print('-' * 120)

    for cfg in strategies:
        r = run_fusion_strategy(data, signals, cfg, trade_days=trade_days)
        all_results.append(r)
        fees = r.get('fees', {})
        print(f"  {cfg['name']:<22} {r['alpha']:>+7.2f}% {r['strategy_return']:>+9.2f}% "
              f"{r['buy_hold_return']:>+9.2f}% {r['max_drawdown']:>7.2f}% "
              f"{r['total_trades']:>5} {r['liquidations']:>3} "
              f"${fees.get('total_costs', 0):>9,.0f}")

    ranked = sorted(all_results, key=lambda x: x['alpha'], reverse=True)
    print(f"\n{'=' * 120}")
    print(f"  策略排名(按α排序)")
    print(f"{'=' * 120}")
    for i, r in enumerate(ranked):
        star = ' ★★★' if i == 0 else ' ★★' if i == 1 else ' ★' if i == 2 else ''
        print(f"  #{i + 1}: {r['name']:<24} α={r['alpha']:+.2f}%{star} "
              f"| 收益={r['strategy_return']:+.2f}% 回撤={r['max_drawdown']:.2f}%")

    # 最佳策略交易明细
    best = ranked[0]
    print(f"\n  最佳策略: {best['name']} · 前20笔交易")
    for t in best.get('trades', [])[:20]:
        print(f"  {str(t['time'])[:16]}  {t.get('action',''):<14} @${t.get('price',0):>8,.2f} "
              f"{t.get('leverage',1)}x  ${t.get('total',0):>10,.0f}  {t.get('reason','')[:55]}")

    bf = best.get('fees', {})
    print(f"\n  费用: 现货${bf.get('spot_fees',0):,.0f} 合约${bf.get('futures_fees',0):,.0f} "
          f"资金费${bf.get('net_funding',0):,.0f} 滑点${bf.get('slippage_cost',0):,.0f} "
          f"总${bf.get('total_costs',0):,.0f}")

    # 保存
    output = {
        'description': f'五书融合策略 · 最近{trade_days}天',
        'books': [
            '《背离技术分析》江南小隐',
            '《均线技术分析》邱立波',
            '《日本蜡烛图技术》Steve Nison',
            '《布林线》John Bollinger',
            '利弗莫尔/威科夫量价分析',
        ],
        'run_time': datetime.now().isoformat(),
        'data_range': f"{main_df.index[0]} ~ {main_df.index[-1]}",
        'trade_range': f"{trade_start} ~ {trade_end}",
        'trade_days': trade_days,
        'total_bars': len(main_df),
        'initial_capital': '10万USDT + 价值10万USDT的ETH',
        'timeframe': '1h',
        'results': [{
            'name': r['name'], 'alpha': r['alpha'],
            'strategy_return': r['strategy_return'],
            'buy_hold_return': r['buy_hold_return'],
            'max_drawdown': r['max_drawdown'],
            'total_trades': r['total_trades'],
            'liquidations': r['liquidations'],
            'final_total': r['final_total'],
            'fees': r.get('fees', {}),
            'trades': r.get('trades', []),
            'history': r.get('history', []),
        } for r in all_results],
        'ranking': [{'rank': i + 1, 'name': r['name'], 'alpha': r['alpha']}
                    for i, r in enumerate(ranked)],
        'best_strategy': {
            'name': best['name'], 'alpha': best['alpha'],
            'strategy_return': best['strategy_return'],
            'trades_count': best['total_trades'],
        },
    }

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'five_book_fusion_result.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果已保存: {path}")
    return output


if __name__ == '__main__':
    td = 30
    if len(sys.argv) > 1:
        try: td = max(1, min(60, int(sys.argv[1])))
        except ValueError: pass
    main(trade_days=td)
