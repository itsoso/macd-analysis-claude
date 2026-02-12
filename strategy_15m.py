"""
15分钟周期双向回测策略

基于《背离技术分析》全书策略:
- 15m作为主信号周期 + 1h/4h作为趋势过滤
- 顶背离/背驰 → 卖出现货 + 开空
- 底背离/背驰 → 买入现货 + 开多 + 平空
- 综合MACD/KDJ/CCI/RSI/量价/几何形态
- 完整手续费模型(交易费+资金费率+滑点+强平费)

初始资金: 10万USDT + 价值10万USDT的ETH
回测周期: 30天, 15分钟K线
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
from strategy_futures import FuturesEngine, FuturesPosition
from strategy_futures_v4 import _calc_top_score, _calc_bottom_score


# ======================================================
#   数据获取
# ======================================================
def fetch_data_15m():
    """获取15m主数据 + 辅助周期数据"""
    print("获取数据...")
    data = {}
    configs = [
        ('15m', 30),   # 主周期: 15分钟, 30天
        ('1h', 30),    # 辅助: 趋势判断
        ('4h', 30),    # 辅助: 大趋势
        ('8h', 60),    # 辅助: 大级别信号
    ]
    for tf, days in configs:
        df = fetch_binance_klines("ETHUSDT", interval=tf, days=days)
        if df is not None and len(df) > 50:
            df = add_all_indicators(df)
            data[tf] = df
            print(f"  {tf}: {len(df)} 条 ({df.index[0]} ~ {df.index[-1]})")
    return data


# ======================================================
#   15m趋势判断(用15m自身MA + 1h/4h辅助)
# ======================================================
def get_trend_15m(data, dt, price):
    """基于15m和更大周期判断趋势"""
    info = {
        'is_downtrend': False,
        'is_uptrend': False,
        'trend_strength': 0,
        'short_trend_down': False,
    }

    # 15m MA趋势
    df_15m = data.get('15m')
    if df_15m is not None:
        ma20 = df_15m['close'].rolling(20).mean()   # 5小时
        ma60 = df_15m['close'].rolling(60).mean()   # 15小时
        ma120 = df_15m['close'].rolling(120).mean()  # 30小时
        for i in range(len(df_15m) - 1, -1, -1):
            if df_15m.index[i] <= dt:
                m20 = ma20.iloc[i] if not pd.isna(ma20.iloc[i]) else price
                m60 = ma60.iloc[i] if not pd.isna(ma60.iloc[i]) else price
                m120 = ma120.iloc[i] if not pd.isna(ma120.iloc[i]) else price
                info['is_downtrend'] = price < m60 and m20 < m60
                info['is_uptrend'] = price > m60 and m20 > m60
                info['trend_strength'] = (price - m120) / m120 * 100
                info['short_trend_down'] = m20 < m60
                break

    # 1h大趋势确认
    df_1h = data.get('1h')
    if df_1h is not None:
        ma30_1h = df_1h['close'].rolling(30).mean()
        for i in range(len(df_1h) - 1, -1, -1):
            if df_1h.index[i] <= dt:
                m30 = ma30_1h.iloc[i] if not pd.isna(ma30_1h.iloc[i]) else price
                info['hourly_downtrend'] = price < m30
                info['hourly_uptrend'] = price > m30
                break

    # 4h大趋势确认
    df_4h = data.get('4h')
    if df_4h is not None:
        ma20_4h = df_4h['close'].rolling(20).mean()
        for i in range(len(df_4h) - 1, -1, -1):
            if df_4h.index[i] <= dt:
                m20 = ma20_4h.iloc[i] if not pd.isna(ma20_4h.iloc[i]) else price
                info['h4_downtrend'] = price < m20
                info['h4_uptrend'] = price > m20
                break

    return info


# ======================================================
#   15m实时指标 (KDJ/RSI/CCI)
# ======================================================
def get_15m_realtime(df_15m, idx):
    """获取当前15m K线的实时指标"""
    row = df_15m.iloc[idx]
    r = {}
    for col in ['K', 'D', 'J', 'CCI', 'RSI6', 'RSI12', 'DIF', 'DEA', 'MACD_BAR']:
        r[col] = float(row[col]) if col in df_15m.columns and not pd.isna(row.get(col)) else None
    return r


# ======================================================
#   策略执行器
# ======================================================
def run_strategy_15m(data, signals_15m, signals_1h, signals_8h, config,
                     start_dt=None):
    """
    15分钟双向策略执行器

    决策逻辑(严格前向):
    1. 顶部信号(TS) → 卖现货 + 开空
    2. 底部信号(BS) → 买现货 + 开多 + 平空
    3. 趋势过滤: 用1h/4h确认大趋势方向
    4. 风控: 止损/止盈/追踪/超时

    start_dt: 如果指定, 只在此时间之后执行交易(之前的数据用于信号计算)
    """
    eng = FuturesEngine(config['name'], max_leverage=config.get('max_lev', 5))
    main_df = data['15m']
    # 如果指定了起始时间, 用该时间点的价格作为初始价
    if start_dt:
        init_idx = main_df.index.searchsorted(start_dt)
        if init_idx >= len(main_df):
            init_idx = 0
        eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[init_idx]
    else:
        eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[0]

    # 风控参数
    eng.max_single_margin = eng.initial_total * config.get('single_pct', 0.15)
    eng.max_margin_total = eng.initial_total * config.get('total_pct', 0.40)
    eng.max_lifetime_margin = eng.initial_total * config.get('lifetime_pct', 5.0)

    # 策略参数
    ts_sell = config.get('ts_sell', 15)        # 卖出信号阈值
    ts_short = config.get('ts_short', 22)      # 开空阈值
    bs_buy = config.get('bs_buy', 15)          # 买入信号阈值
    bs_long = config.get('bs_long', 22)        # 开多阈值
    bs_close_short = config.get('bs_close', 30)  # 平空阈值
    ts_close_long = config.get('ts_close', 30)   # 平多阈值
    sell_pct = config.get('sell_pct', 0.50)    # 每次卖出比例
    buy_pct = config.get('buy_pct', 0.30)      # 每次买入使用USDT比例
    margin_use = config.get('margin_use', 0.60)  # 保证金使用比例
    lev = config.get('lev', 3)
    # 做空风控
    short_sl = config.get('short_sl', -0.25)   # 空仓止损 (-25%)
    short_tp = config.get('short_tp', 0.80)    # 空仓止盈 (+80%)
    short_trail_start = config.get('short_trail', 0.30)  # 追踪启动 (+30%)
    short_trail_keep = config.get('short_trail_keep', 0.60)  # 追踪保留60%
    short_max_hold = config.get('short_max_hold', 192)  # 最大持仓48h=192个15m
    # 做多风控
    long_sl = config.get('long_sl', -0.20)     # 多仓止损 (-20%)
    long_tp = config.get('long_tp', 0.60)      # 多仓止盈 (+60%)
    long_trail_start = config.get('long_trail', 0.25)
    long_trail_keep = config.get('long_trail_keep', 0.60)
    long_max_hold = config.get('long_max_hold', 192)
    # 冷却
    cooldown_bars = config.get('cooldown', 4)   # 15m*4=1小时冷却

    short_max_pnl = 0
    long_max_pnl = 0
    short_cd = 0
    long_cd = 0
    spot_cd = 0       # 统一现货冷却(买卖共用)
    spot_cooldown = config.get('spot_cooldown', 48)  # 48*15m=12小时
    short_bars = 0
    long_bars = 0
    prev_sig_id = None  # 上一个信号的时间戳, 用于检测信号切换
    last_spot_action = None  # 上次现货操作('buy'/'sell')
    conflict_threshold = config.get('conflict_threshold', 15)  # TS和BS同时大于此值视为冲突

    for idx in range(60, len(main_df)):  # 从第60个bar开始(前15h用于MA计算)
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]

        # 如果指定了start_dt, 在该时间之前只记录历史不交易
        if start_dt and dt < start_dt:
            if idx % 4 == 0:
                eng.record_history(dt, price)
            continue

        eng.check_liquidation(price, dt)

        short_just_opened = False  # 本bar是否刚开空仓(防同bar开平)
        long_just_opened = False   # 本bar是否刚开多仓(防同bar开平)

        # 资金费率: 15m周期, 每32个bar=8小时
        eng.funding_counter += 1
        if eng.funding_counter % 32 == 0:
            # 重用charge_funding逻辑
            is_negative = (eng.funding_counter * 7 + 3) % 10 < 3
            rate = FuturesEngine.FUNDING_RATE if not is_negative else -FuturesEngine.FUNDING_RATE * 0.5
            if eng.futures_long:
                cost = eng.futures_long.quantity * price * rate
                eng.usdt -= cost
                if cost > 0:
                    eng.total_funding_paid += cost
                else:
                    eng.total_funding_received += abs(cost)
            if eng.futures_short:
                income = eng.futures_short.quantity * price * rate
                eng.usdt += income
                if income > 0:
                    eng.total_funding_received += income
                else:
                    eng.total_funding_paid += abs(income)

        if short_cd > 0:
            short_cd -= 1
        if long_cd > 0:
            long_cd -= 1
        if spot_cd > 0:
            spot_cd -= 1

        # ---- 获取信号 ----
        trend = get_trend_15m(data, dt, price)

        # 15m信号(主信号)
        sig_15m = get_signal_at(signals_15m, dt) or dict(DEFAULT_SIG)
        # 1h信号(辅助)
        sig_1h = get_signal_at(signals_1h, dt) or dict(DEFAULT_SIG)
        # 8h信号(辅助)
        sig_8h = get_signal_at(signals_8h, dt) or dict(DEFAULT_SIG)

        # 融合信号: 15m为主(1.0), 1h辅助(0.4), 8h辅助(0.3)
        merged = dict(DEFAULT_SIG)
        merged['top'] = 0
        merged['bottom'] = 0
        for sig_src, w in [(sig_15m, 1.0), (sig_1h, 0.4), (sig_8h, 0.3)]:
            merged['top'] += sig_src.get('top', 0) * w
            merged['bottom'] += sig_src.get('bottom', 0) * w
            for k in DEFAULT_SIG:
                if isinstance(DEFAULT_SIG[k], bool) and sig_src.get(k):
                    merged[k] = True
                elif isinstance(DEFAULT_SIG[k], (int, float)) and k not in ('top', 'bottom'):
                    merged[k] = max(merged.get(k, 0), sig_src.get(k, 0))
                elif isinstance(DEFAULT_SIG[k], str) and sig_src.get(k):
                    merged[k] = sig_src[k]

        ts, ts_parts = _calc_top_score(merged, trend)
        bs = _calc_bottom_score(merged, trend)

        # 15m实时指标
        rt = get_15m_realtime(main_df, idx)
        kdj_overbought = rt.get('K') and rt['K'] > 80
        kdj_oversold = rt.get('K') and rt['K'] < 20
        rsi_overbought = rt.get('RSI6') and rt['RSI6'] > 70
        rsi_oversold = rt.get('RSI6') and rt['RSI6'] < 30
        cci_high = rt.get('CCI') and rt['CCI'] > 100
        cci_low = rt.get('CCI') and rt['CCI'] < -100
        macd_death = rt.get('DIF') and rt.get('DEA') and rt['DIF'] < rt['DEA']
        macd_golden = rt.get('DIF') and rt.get('DEA') and rt['DIF'] > rt['DEA']

        # ---- 检测信号变化 (15m信号的时间戳是否变化) ----
        cur_sig_id = None
        for t in signals_15m:
            if t <= dt:
                cur_sig_id = t
        signal_changed = (cur_sig_id != prev_sig_id)
        # 信号冲突检测(改进版: 使用比例判断而非绝对阈值)
        # 当两个信号强度接近时才视为冲突; 一方明显主导时允许交易
        if ts > 0 and bs > 0:
            ratio = min(ts, bs) / max(ts, bs)
            # 两者都超过阈值 且 强度比例接近(>60%) → 真正冲突
            in_conflict = (min(ts, bs) >= conflict_threshold and ratio >= 0.6)
        else:
            in_conflict = False

        # ========================================
        # 决策1: 卖出现货 (顶部信号)
        # ========================================
        # 要求: 信号够强 + 冷却结束 + 无冲突 + (信号变了 或 上次是买)
        can_sell = (ts >= ts_sell and spot_cd == 0 and not in_conflict and
                    eng.spot_eth * price > 500 and
                    (signal_changed or last_spot_action != 'sell'))
        if can_sell:
            reason_parts = [f"TS={ts:.0f}"]
            if kdj_overbought:
                reason_parts.append("KDJ超买")
            if rsi_overbought:
                reason_parts.append("RSI超买")
            if cci_high:
                reason_parts.append("CCI>100")
            if ts_parts:
                reason_parts.extend(ts_parts[:2])
            eng.spot_sell(price, dt, sell_pct,
                          f"顶部卖出 {' '.join(reason_parts)}")
            spot_cd = spot_cooldown
            last_spot_action = 'sell'

        # ========================================
        # 决策2: 开空仓 (顶部信号 + 趋势确认)
        # ========================================
        # 开空条件: TS够强 + 冷却完 + 无持仓 + TS明显主导BS(或BS较低)
        ts_dominates = (ts > bs * 1.5) if bs > 0 else True
        bs_conflict_val = config.get('bs_conflict', 20)
        short_ok = (short_cd == 0 and ts >= ts_short and not eng.futures_short
                    and (bs < bs_conflict_val or ts_dominates))
        if short_ok:
            # 书本理论: 顶背离确认后做空, 用趋势和指标过滤
            confirm = 0
            if trend.get('is_downtrend') or trend.get('hourly_downtrend'):
                confirm += 1
            if kdj_overbought or rsi_overbought:
                confirm += 1
            if macd_death:
                confirm += 1
            if cci_high:
                confirm += 1

            # 至少1个确认条件(宽松: 15m信号本身已包含背离)
            if confirm >= 0:
                margin = eng.available_margin() * margin_use
                actual_lev = min(lev, eng.max_leverage)
                if ts >= 40:
                    actual_lev = min(lev, eng.max_leverage)
                elif ts >= 30:
                    actual_lev = min(min(lev, 3), eng.max_leverage)
                else:
                    actual_lev = min(2, eng.max_leverage)

                eng.open_short(price, dt, margin, actual_lev,
                               f"做空 {actual_lev}x TS={ts:.0f} "
                               f"{'↓' if trend.get('is_downtrend') else '→'} "
                               f"{' '.join(ts_parts[:2])}")
                short_max_pnl = 0
                short_bars = 0
                short_cd = cooldown_bars
                short_just_opened = True  # 标记本bar刚开空仓

        # ========================================
        # 决策3: 管理空仓 (风控)
        # ========================================
        if eng.futures_short and not short_just_opened:
            short_bars += 1
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin

            # 硬止盈
            if pnl_r >= short_tp:
                eng.close_short(price, dt, f"止盈 +{pnl_r * 100:.0f}%")
                short_max_pnl = 0
                short_cd = cooldown_bars * 2
                short_bars = 0
            else:
                # 追踪止盈
                if pnl_r > short_max_pnl:
                    short_max_pnl = pnl_r
                if short_max_pnl >= short_trail_start and eng.futures_short:
                    if pnl_r < short_max_pnl * short_trail_keep:
                        eng.close_short(price, dt,
                                        f"追踪止盈 max={short_max_pnl * 100:.0f}% "
                                        f"now={pnl_r * 100:.0f}%")
                        short_max_pnl = 0
                        short_cd = cooldown_bars
                        short_bars = 0

                # 底部信号平空 — 仅当BS明显主导TS时才平(避免TS强势时误平)
                if eng.futures_short and bs >= bs_close_short:
                    # BS必须真正主导: BS > TS*1.3 才算底部确认
                    bs_is_dominant = (ts < bs * 0.7) if bs > 0 else True
                    if bs_is_dominant:
                        eng.close_short(price, dt, f"底部信号平空 BS={bs:.0f} TS={ts:.0f}")
                        short_max_pnl = 0
                        short_cd = cooldown_bars * 3
                        short_bars = 0

                # 硬止损
                if eng.futures_short and pnl_r < short_sl:
                    eng.close_short(price, dt, f"止损 {pnl_r * 100:.0f}%")
                    short_max_pnl = 0
                    short_cd = cooldown_bars * 4
                    short_bars = 0

                # 超时
                if eng.futures_short and short_bars >= short_max_hold:
                    eng.close_short(price, dt, f"超时平仓 {short_bars}bars")
                    short_max_pnl = 0
                    short_cd = cooldown_bars
                    short_bars = 0
        elif eng.futures_short and short_just_opened:
            short_bars = 1  # 刚开仓, 开始计时但跳过风控检查

        # ========================================
        # 决策4: 买入现货 (底部信号)
        # ========================================
        can_buy = (bs >= bs_buy and spot_cd == 0 and not in_conflict and
                   eng.available_usdt() > 500 and
                   (signal_changed or last_spot_action != 'buy'))
        if can_buy:
            reason_parts = [f"BS={bs:.0f}"]
            if kdj_oversold:
                reason_parts.append("KDJ超卖")
            if rsi_oversold:
                reason_parts.append("RSI超卖")
            if cci_low:
                reason_parts.append("CCI<-100")
            buy_amount = eng.available_usdt() * buy_pct
            eng.spot_buy(price, dt, buy_amount,
                         f"底部买入 {' '.join(reason_parts)}")
            spot_cd = spot_cooldown
            last_spot_action = 'buy'

        # ========================================
        # 决策5: 开多仓 (底部信号 + 趋势确认)
        # ========================================
        # 开多条件: BS够强 + 冷却完 + 无持仓 + BS明显主导TS(或TS较低)
        bs_dominates = (bs > ts * 1.5) if ts > 0 else True
        ts_conflict_val = config.get('ts_conflict', 20)
        long_ok = (long_cd == 0 and bs >= bs_long and not eng.futures_long
                   and (ts < ts_conflict_val or bs_dominates))
        if long_ok:
            confirm = 0
            if trend.get('is_uptrend') or trend.get('hourly_uptrend'):
                confirm += 1
            if kdj_oversold or rsi_oversold:
                confirm += 1
            if macd_golden:
                confirm += 1
            if cci_low:
                confirm += 1

            if confirm >= 0:
                margin = eng.available_margin() * margin_use
                actual_lev = min(lev, eng.max_leverage)
                if bs >= 40:
                    actual_lev = min(lev, eng.max_leverage)
                elif bs >= 30:
                    actual_lev = min(min(lev, 3), eng.max_leverage)
                else:
                    actual_lev = min(2, eng.max_leverage)

                eng.open_long(price, dt, margin, actual_lev,
                              f"做多 {actual_lev}x BS={bs:.0f} "
                              f"{'↑' if trend.get('is_uptrend') else '→'}")
                long_max_pnl = 0
                long_bars = 0
                long_cd = cooldown_bars
                long_just_opened = True  # 标记本bar刚开多仓

        # ========================================
        # 决策6: 管理多仓 (风控)
        # ========================================
        if eng.futures_long and not long_just_opened:
            long_bars += 1
            pnl_r = eng.futures_long.calc_pnl(price) / eng.futures_long.margin

            # 硬止盈
            if pnl_r >= long_tp:
                eng.close_long(price, dt, f"止盈 +{pnl_r * 100:.0f}%")
                long_max_pnl = 0
                long_cd = cooldown_bars * 2
                long_bars = 0
            else:
                # 追踪止盈
                if pnl_r > long_max_pnl:
                    long_max_pnl = pnl_r
                if long_max_pnl >= long_trail_start and eng.futures_long:
                    if pnl_r < long_max_pnl * long_trail_keep:
                        eng.close_long(price, dt,
                                       f"追踪止盈 max={long_max_pnl * 100:.0f}% "
                                       f"now={pnl_r * 100:.0f}%")
                        long_max_pnl = 0
                        long_cd = cooldown_bars
                        long_bars = 0

                # 顶部信号平多 — 仅当TS明显主导BS时才平
                if eng.futures_long and ts >= ts_close_long:
                    ts_is_dominant = (bs < ts * 0.7) if ts > 0 else True
                    if ts_is_dominant:
                        eng.close_long(price, dt, f"顶部信号平多 TS={ts:.0f} BS={bs:.0f}")
                        long_max_pnl = 0
                        long_cd = cooldown_bars * 3
                        long_bars = 0

                # 硬止损
                if eng.futures_long and pnl_r < long_sl:
                    eng.close_long(price, dt, f"止损 {pnl_r * 100:.0f}%")
                    long_max_pnl = 0
                    long_cd = cooldown_bars * 4
                    long_bars = 0

                # 超时
                if eng.futures_long and long_bars >= long_max_hold:
                    eng.close_long(price, dt, f"超时平仓 {long_bars}bars")
                    long_max_pnl = 0
                    long_cd = cooldown_bars
                    long_bars = 0
        elif eng.futures_long and long_just_opened:
            long_bars = 1  # 刚开仓, 开始计时但跳过风控检查

        # 更新信号记忆
        prev_sig_id = cur_sig_id

        # 每小时记录一次历史(每4个15m bar)
        if idx % 4 == 0:
            eng.record_history(dt, price)

    # 期末平仓
    last_price = main_df['close'].iloc[-1]
    last_dt = main_df.index[-1]
    if eng.futures_short:
        eng.close_short(last_price, last_dt, "期末平仓")
    if eng.futures_long:
        eng.close_long(last_price, last_dt, "期末平仓")

    # 如果指定了start_dt, 用该时刻的数据作为基准来计算BH收益
    if start_dt:
        trade_df = main_df[main_df.index >= start_dt]
        if len(trade_df) > 1:
            return eng.get_result(trade_df)
    return eng.get_result(main_df)


# ======================================================
#   策略变体定义
# ======================================================
def get_strategies():
    """定义多种策略变体"""
    base = {
        'single_pct': 0.15, 'total_pct': 0.40, 'lifetime_pct': 5.0,
        'ts_sell': 15, 'ts_short': 22, 'bs_buy': 15, 'bs_long': 22,
        'bs_close': 30, 'ts_close': 30, 'bs_conflict': 20, 'ts_conflict': 20,
        'sell_pct': 0.50, 'buy_pct': 0.30, 'margin_use': 0.60, 'lev': 3,
        'short_sl': -0.25, 'short_tp': 0.80, 'short_trail': 0.30,
        'short_trail_keep': 0.60, 'short_max_hold': 192,
        'long_sl': -0.20, 'long_tp': 0.60, 'long_trail': 0.25,
        'long_trail_keep': 0.60, 'long_max_hold': 192,
        'cooldown': 4, 'max_lev': 5,
        'spot_cooldown': 48,       # 48*15m=12h (统一买卖冷却)
        'conflict_threshold': 15,  # TS/BS同时>15时视为信号冲突,不做现货
    }

    strategies = [
        # S1: 书本标准 — 严格遵循书中信号阈值
        {**base, 'name': 'S1: 书本标准(双向)'},

        # S2: 保守做空 — 高阈值开空, 低杠杆
        {**base, 'name': 'S2: 保守做空',
         'ts_short': 30, 'lev': 2, 'short_sl': -0.15, 'short_max_hold': 96,
         'margin_use': 0.40},

        # S3: 激进做空 — 低阈值快开, 高杠杆
        {**base, 'name': 'S3: 激进做空',
         'ts_short': 15, 'lev': 5, 'short_sl': -0.35, 'margin_use': 0.80,
         'single_pct': 0.20, 'total_pct': 0.50},

        # S4: 趋势跟随 — 顺势开仓, 逆势不开
        {**base, 'name': 'S4: 趋势跟随',
         'ts_short': 18, 'bs_long': 18,
         'short_trail': 0.20, 'long_trail': 0.20},

        # S5: 纯合约双向 — 不持现货, 纯合约做多做空
        {**base, 'name': 'S5: 纯合约双向',
         'ts_sell': 999, 'bs_buy': 999,  # 禁用现货买卖
         'ts_short': 18, 'bs_long': 18,
         'lev': 3, 'margin_use': 0.70,
         'single_pct': 0.20, 'total_pct': 0.50},

        # S6: 宽信号快止损 — 更宽松的入场, 更严格的止损
        {**base, 'name': 'S6: 宽信号快止损',
         'ts_short': 12, 'bs_long': 12,
         'short_sl': -0.10, 'long_sl': -0.10,
         'short_tp': 0.40, 'long_tp': 0.30,
         'short_trail': 0.15, 'long_trail': 0.12,
         'cooldown': 2, 'short_max_hold': 96, 'long_max_hold': 96},

        # S7: 大仓低杠 — 大仓位+低杠杆, 降低爆仓风险
        {**base, 'name': 'S7: 大仓低杠',
         'lev': 2, 'single_pct': 0.25, 'total_pct': 0.60,
         'margin_use': 0.80, 'max_lev': 2,
         'short_sl': -0.30, 'long_sl': -0.25},

        # S8: 指标共振 — 更高的阈值确保多指标共振
        {**base, 'name': 'S8: 指标共振',
         'ts_short': 35, 'bs_long': 35, 'ts_sell': 20, 'bs_buy': 20,
         'lev': 4, 'margin_use': 0.70,
         'single_pct': 0.20, 'total_pct': 0.50,
         'short_trail': 0.40, 'long_trail': 0.35},
    ]
    return strategies


# ======================================================
#   主函数
# ======================================================
def main(trade_days=None):
    """
    trade_days: 实际交易天数。如果指定, 获取30天数据用于信号计算,
                但只在最近N天内执行交易。默认None=全部30天。
    """
    data = fetch_data_15m()

    if '15m' not in data:
        print("错误: 无法获取15m数据")
        return

    # 确定交易起始时间
    start_dt = None
    if trade_days and trade_days < 30:
        end_dt = data['15m'].index[-1]
        start_dt = end_dt - pd.Timedelta(days=trade_days)
        bars_in_trade = len(data['15m'][data['15m'].index >= start_dt])
        print(f"\n回测模式: 最近{trade_days}天")
        print(f"  交易起始: {start_dt}")
        print(f"  交易K线数: {bars_in_trade}")
        print(f"  信号计算: 使用全部30天数据")

    # 计算信号(使用全部数据, 确保信号准确)
    print("\n计算信号...")
    signals_15m = analyze_signals_enhanced(data['15m'], 200)
    print(f"  15m: {len(signals_15m)} 个信号点")

    signals_1h = {}
    if '1h' in data:
        signals_1h = analyze_signals_enhanced(data['1h'], 168)
        print(f"  1h: {len(signals_1h)} 个信号点")

    signals_8h = {}
    if '8h' in data:
        signals_8h = analyze_signals_enhanced(data['8h'], 90)
        print(f"  8h: {len(signals_8h)} 个信号点")

    # 运行所有策略
    strategies = get_strategies()
    all_results = []

    trade_label = f"最近{trade_days}天" if trade_days else "30天"
    trade_start_str = str(start_dt)[:16] if start_dt else str(data['15m'].index[0])[:16]
    trade_end_str = str(data['15m'].index[-1])[:16]

    print(f"\n{'=' * 110}")
    print(f"  15分钟双向回测 · {len(strategies)}种策略 · {trade_label}")
    print(f"  信号数据: {data['15m'].index[0]} ~ {data['15m'].index[-1]} ({len(data['15m'])}根K线)")
    print(f"  交易区间: {trade_start_str} ~ {trade_end_str}")
    print(f"  初始: 10万USDT + 价值10万USDT的ETH")
    print(f"{'=' * 110}")

    print(f"\n{'策略':<24} {'α':>8} {'策略收益':>10} {'BH收益':>10} {'回撤':>8} "
          f"{'交易':>6} {'强平':>4} {'费用':>10}")
    print('-' * 110)

    for cfg in strategies:
        r = run_strategy_15m(data, signals_15m, signals_1h, signals_8h, cfg,
                             start_dt=start_dt)
        all_results.append(r)
        fees = r.get('fees', {})
        print(f"  {cfg['name']:<22} {r['alpha']:>+7.2f}% {r['strategy_return']:>+9.2f}% "
              f"{r['buy_hold_return']:>+9.2f}% {r['max_drawdown']:>7.2f}% "
              f"{r['total_trades']:>5} {r['liquidations']:>3} "
              f"${fees.get('total_costs', 0):>9,.0f}")

    # 排名
    ranked = sorted(all_results, key=lambda x: x['alpha'], reverse=True)
    print(f"\n{'=' * 110}")
    print(f"  策略排名")
    print(f"{'=' * 110}")
    for i, r in enumerate(ranked):
        star = ' ★' if i == 0 else ''
        print(f"  #{i + 1}: {r['name']:<24} α={r['alpha']:+.2f}%{star} "
              f"| 收益={r['strategy_return']:+.2f}% 回撤={r['max_drawdown']:.2f}%")

    # 最佳策略详细交易
    best = ranked[0]
    print(f"\n{'=' * 110}")
    print(f"  最佳策略: {best['name']} · 交易明细")
    print(f"{'=' * 110}")
    for t in best.get('trades', []):
        action = t.get('action', '')
        direction = t.get('direction', '')
        reason = t.get('reason', '')
        prc = t.get('price', 0)
        lev = t.get('leverage', 1)
        total = t.get('total', 0)
        print(f"  {t['time'][:16]}  {action:<14} {direction:<6} @${prc:>8,.2f} "
              f"{lev}x  总${total:>10,.0f}  {reason[:55]}")

    # 费用明细
    print(f"\n  费用明细:")
    bf = best.get('fees', {})
    print(f"    现货手续费: ${bf.get('spot_fees', 0):,.2f}")
    print(f"    合约手续费: ${bf.get('futures_fees', 0):,.2f}")
    print(f"    资金费率(净): ${bf.get('net_funding', 0):,.2f}")
    print(f"    滑点成本: ${bf.get('slippage_cost', 0):,.2f}")
    print(f"    强平费: ${bf.get('liquidation_fees', 0):,.2f}")
    print(f"    总费用: ${bf.get('total_costs', 0):,.2f}")

    # 保存结果
    output = {
        'description': f'15分钟双向回测(做多+做空) · {trade_label}',
        'run_time': datetime.now().isoformat(),
        'data_range': f"{data['15m'].index[0]} ~ {data['15m'].index[-1]}",
        'trade_range': f"{trade_start_str} ~ {trade_end_str}",
        'trade_days': trade_days or 30,
        'total_bars': len(data['15m']),
        'initial_capital': '10万USDT + 价值10万USDT的ETH',
        'timeframe': '15m',
        'signal_counts': {
            '15m': len(signals_15m),
            '1h': len(signals_1h),
            '8h': len(signals_8h),
        },
        'results': [{
            'name': r['name'],
            'alpha': r['alpha'],
            'strategy_return': r['strategy_return'],
            'buy_hold_return': r['buy_hold_return'],
            'max_drawdown': r['max_drawdown'],
            'total_trades': r['total_trades'],
            'liquidations': r['liquidations'],
            'final_total': r['final_total'],
            'final_usdt': r['final_usdt'],
            'final_spot_eth': r['final_spot_eth'],
            'fees': r.get('fees', {}),
            'trades': r.get('trades', []),
            'history': r.get('history', []),
        } for r in all_results],
        'ranking': [{'rank': i + 1, 'name': r['name'], 'alpha': r['alpha']}
                    for i, r in enumerate(ranked)],
        'best_strategy': {
            'name': best['name'],
            'alpha': best['alpha'],
            'strategy_return': best['strategy_return'],
            'trades_count': best['total_trades'],
        },
    }

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'strategy_15m_result.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果已保存: {path}")
    return output


if __name__ == '__main__':
    # 支持命令行参数: python strategy_15m.py [天数]
    # 例: python strategy_15m.py 3  → 最近3天回测
    trade_days = None
    if len(sys.argv) > 1:
        try:
            trade_days = int(sys.argv[1])
            if trade_days <= 0 or trade_days > 30:
                print(f"天数范围: 1-30, 输入: {trade_days}")
                trade_days = None
        except ValueError:
            print(f"无效参数: {sys.argv[1]}, 使用默认30天")
    main(trade_days=trade_days)
