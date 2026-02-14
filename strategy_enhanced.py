"""
深度指标增强策略
充分利用书中第二~第五章的所有指标:
- MACD金叉死叉 + 隔堆背离次数 + DIF零轴位置
- KDJ超买超卖区域(>80/<20) + K/D交叉
- CCI天线(+100)/地线(-100)穿越
- RSI超买(>70)/超卖(<30) + 确认突破
- 量价背离(价升量减/地量地价)
- 几何形态(幅度+时间+MA面积)

策略思路:
G: 指标共振策略 — 多指标同时确认才操作
H: MACD金叉死叉增强 — 用交叉信号做精确择时
I: KDJ+RSI极值过滤 — 只在超买超卖极值区操作
J: 量价确认策略 — 量价背离作为额外过滤条件
K: 书中原则严格版 — 严格按书中卖点用背离/买点用背驰
L: 全指标融合增强 — 综合所有改进的最优策略
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
from strategy_compare import BaseStrategy


def fetch_all_data():
    """获取多周期数据"""
    print("获取数据...")
    data = {}
    configs = [('1h', 30), ('2h', 30), ('4h', 30), ('6h', 30), ('8h', 60)]
    for tf, days in configs:
        df = fetch_binance_klines("ETHUSDT", interval=tf, days=days)
        if df is not None and len(df) > 50:
            df = add_all_indicators(df)
            data[tf] = df
            print(f"  {tf}: {len(df)} 条")
    return data


def analyze_signals_enhanced(df, window):
    """增强版信号分析 — 提取全部细粒度指标"""
    signals = {}
    step = max(1, window // 8)
    i = window
    while i < len(df):
        window_df = df.iloc[max(0, i - window):i].copy()
        try:
            analyzer = ComprehensiveAnalyzer(window_df, _skip_copy=True)
            results = analyzer.analyze_all()
            sc = results.get('comprehensive_score', {})
            recs = results.get('trade_recommendations', [])

            # === 基本分数 ===
            sig = {
                'top': sc.get('top_score', 0),
                'bottom': sc.get('bottom_score', 0),
                'top_level': sc.get('top_level', ''),
                'bottom_level': sc.get('bottom_level', ''),
                'exhaust_sell': False,
                'exhaust_buy': False,
                'exhaust_sell_conf': '',    # high/medium
                'exhaust_buy_conf': '',
            }

            # === 背驰信号 + 紧迫度 ===
            for r in recs:
                if r.get('action') == 'SELL_EXHAUSTION':
                    sig['exhaust_sell'] = True
                    sig['exhaust_sell_conf'] = r.get('urgency', 'medium')
                elif r.get('action') == 'BUY_EXHAUSTION':
                    sig['exhaust_buy'] = True
                    sig['exhaust_buy_conf'] = r.get('urgency', 'medium')
                elif r.get('action') == 'SELL':
                    sig['sell_urgency'] = r.get('urgency', 'medium')
                elif r.get('action') == 'BUY':
                    sig['buy_urgency'] = r.get('urgency', 'medium')

            # === MACD详细信号 ===
            macd_data = results.get('macd', {})
            crosses_dict = macd_data.get('crosses', {})
            all_crosses = crosses_dict.get('all_crosses', []) if isinstance(crosses_dict, dict) else []
            # 最近的金叉/死叉
            sig['last_cross'] = ''  # 'golden' or 'death'
            sig['last_cross_above_zero'] = False
            if all_crosses:
                last_c = all_crosses[-1]
                sig['last_cross'] = last_c.get('type', '')
                sig['last_cross_above_zero'] = last_c.get('position') == 'above_zero'

            # MACD隔堆背离次数
            bar_divs = macd_data.get('bar_length_divergence', [])
            sig['separated_top'] = sum(1 for d in bar_divs
                if d.get('subtype') == 'separated' and d.get('direction') == 'top')
            sig['separated_bottom'] = sum(1 for d in bar_divs
                if d.get('subtype') == 'separated' and d.get('direction') == 'bottom')

            # DIF/DEA背离
            dif_divs = macd_data.get('dif_dea_divergence', [])
            sig['dif_top_div'] = sum(1 for d in dif_divs if d.get('direction') == 'top')
            sig['dif_bottom_div'] = sum(1 for d in dif_divs if d.get('direction') == 'bottom')

            # 柱面积背离
            area_divs = macd_data.get('bar_area_divergence', [])
            sig['area_top_div'] = sum(1 for d in area_divs if d.get('direction') == 'top')
            sig['area_bottom_div'] = sum(1 for d in area_divs if d.get('direction') == 'bottom')

            # === 背驰详细 ===
            exh_data = results.get('exhaustion', {})
            sig['zero_returns_top'] = len(exh_data.get('zero_returns_top', []))
            sig['zero_returns_bottom'] = len(exh_data.get('zero_returns_bottom', []))
            sig['sep_divs_top'] = len(exh_data.get('separated_divs_top', []))
            sig['sep_divs_bottom'] = len(exh_data.get('separated_divs_bottom', []))

            # === KDJ ===
            kdj_data = results.get('kdj', {})
            sig['kdj_top_div'] = len(kdj_data.get('top_divergence', []))
            sig['kdj_bottom_div'] = len(kdj_data.get('bottom_divergence', []))

            # === CCI ===
            cci_data = results.get('cci', {})
            sig['cci_top_div'] = len(cci_data.get('top_divergence', []))
            sig['cci_bottom_div'] = len(cci_data.get('bottom_divergence', []))
            # CCI穿越
            cci_crosses = cci_data.get('spring_autumn_crosses', [])
            sig['cci_last_cross'] = ''  # 'spring_up'/'autumn_down'
            if cci_crosses:
                sig['cci_last_cross'] = cci_crosses[-1].get('direction', '')

            # === RSI ===
            rsi_data = results.get('rsi', {})
            sig['rsi_top_div'] = len(rsi_data.get('top_divergence', []))
            sig['rsi_bottom_div'] = len(rsi_data.get('bottom_divergence', []))

            # === 量价 ===
            vol_data = results.get('volume', {})
            sig['vol_price_up_down'] = len(vol_data.get('price_up_volume_down', []))
            sig['vol_ground'] = len(vol_data.get('ground_volume', []))

            # === 形态 ===
            pat_data = results.get('pattern', {})
            sig['pat_amp_top'] = sum(1 for d in pat_data.get('amplitude_divergence', [])
                if d.get('direction') == 'top')
            sig['pat_amp_bottom'] = sum(1 for d in pat_data.get('amplitude_divergence', [])
                if d.get('direction') == 'bottom')

            signals[df.index[i]] = sig
        except Exception:
            pass
        i += step
    return signals


def get_realtime_indicators(df, idx):
    """获取当前K线的实时指标值"""
    row = df.iloc[idx]
    result = {}
    for col in ['K', 'D', 'J', 'CCI', 'RSI6', 'RSI12', 'DIF', 'DEA', 'MACD_BAR',
                 'MA5', 'MA10', 'MA30', 'MA60']:
        if col in df.columns:
            val = row[col]
            result[col] = float(val) if not pd.isna(val) else None
    return result


def get_signal_at(signals_dict, dt):
    """获取指定时间的最近信号"""
    latest = None
    for t, s in signals_dict.items():
        if t <= dt:
            latest = s
        else:
            break
    return latest


DEFAULT_SIG = {
    'top': 0, 'bottom': 0, 'exhaust_sell': False, 'exhaust_buy': False,
    'exhaust_sell_conf': '', 'exhaust_buy_conf': '',
    'last_cross': '', 'last_cross_above_zero': False,
    'separated_top': 0, 'separated_bottom': 0,
    'dif_top_div': 0, 'dif_bottom_div': 0,
    'area_top_div': 0, 'area_bottom_div': 0,
    'zero_returns_top': 0, 'zero_returns_bottom': 0,
    'sep_divs_top': 0, 'sep_divs_bottom': 0,
    'kdj_top_div': 0, 'kdj_bottom_div': 0,
    'cci_top_div': 0, 'cci_bottom_div': 0, 'cci_last_cross': '',
    'rsi_top_div': 0, 'rsi_bottom_div': 0,
    'vol_price_up_down': 0, 'vol_ground': 0,
    'pat_amp_top': 0, 'pat_amp_bottom': 0,
    'top_level': '', 'bottom_level': '',
    'sell_urgency': '', 'buy_urgency': '',
}


def run_strategy_G(data, signals_all):
    """策略G: 指标共振 — 多个指标同时确认才操作"""
    s = BaseStrategy("G: 指标共振")
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eth_qty = s.initial_eth_value / first_price
    usdt = s.initial_usdt
    cooldown = 0
    min_eth_ratio = 0.05

    for idx in range(len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        ind = get_realtime_indicators(main_df, idx)
        if cooldown > 0: cooldown -= 1

        # 融合多周期信号
        sig = {k: 0 if isinstance(v, (int, float)) else v for k, v in DEFAULT_SIG.items()}
        tw = 0
        weights = {'4h': 1.0, '6h': 0.85, '2h': 0.7, '1h': 0.45}
        for tf, w in weights.items():
            if tf in signals_all:
                s_tf = get_signal_at(signals_all[tf], dt) or DEFAULT_SIG
                for k in ['top', 'bottom']:
                    sig[k] += s_tf[k] * w
                for k in ['exhaust_sell', 'exhaust_buy']:
                    if s_tf.get(k): sig[k] = True
                for k in ['separated_top', 'separated_bottom', 'dif_top_div', 'dif_bottom_div',
                           'kdj_top_div', 'kdj_bottom_div', 'cci_top_div', 'cci_bottom_div',
                           'rsi_top_div', 'rsi_bottom_div', 'vol_price_up_down',
                           'area_top_div', 'area_bottom_div', 'pat_amp_top', 'pat_amp_bottom']:
                    sig[k] = max(sig.get(k, 0), s_tf.get(k, 0))
                if s_tf.get('last_cross'): sig['last_cross'] = s_tf['last_cross']
                if s_tf.get('cci_last_cross'): sig['cci_last_cross'] = s_tf['cci_last_cross']
                tw += w
        if tw > 0:
            sig['top'] /= tw
            sig['bottom'] /= tw

        eth_val = eth_qty * price
        total = usdt + eth_val
        eth_r = eth_val / total if total > 0 else 0

        if cooldown == 0:
            # === 卖出: 需要至少3种指标共振 ===
            sell_confirmations = 0
            sell_reasons = []

            if sig['top'] >= 20:
                sell_confirmations += 1; sell_reasons.append(f"综合背离={sig['top']:.0f}")
            if sig['separated_top'] >= 1 or sig['dif_top_div'] >= 1:
                sell_confirmations += 1; sell_reasons.append("MACD背离")
            if sig['kdj_top_div'] >= 1 or (ind.get('K') and ind['K'] > 80):
                sell_confirmations += 1; sell_reasons.append(f"KDJ超买K={ind.get('K',0):.0f}")
            if sig['cci_top_div'] >= 1 or (ind.get('CCI') and ind['CCI'] > 100):
                sell_confirmations += 1; sell_reasons.append(f"CCI={ind.get('CCI',0):.0f}")
            if sig['rsi_top_div'] >= 1 or (ind.get('RSI6') and ind['RSI6'] > 70):
                sell_confirmations += 1; sell_reasons.append(f"RSI={ind.get('RSI6',0):.0f}")
            if sig['vol_price_up_down'] >= 1:
                sell_confirmations += 1; sell_reasons.append("价升量减")
            if sig['area_top_div'] >= 1:
                sell_confirmations += 1; sell_reasons.append("面积背离")
            if sig.get('exhaust_sell'):
                sell_confirmations += 2; sell_reasons.append("背驰")

            if sell_confirmations >= 3 and eth_r > min_eth_ratio + 0.05:
                ratio = min(0.6, 0.15 * sell_confirmations)  # 共振越多卖越多
                available = eth_r - min_eth_ratio
                ratio = min(ratio, available * 0.9)
                if ratio > 0.05:
                    usdt, eth_qty = s.execute_sell(price, dt, usdt, eth_qty, ratio,
                        f"共振{sell_confirmations}: {', '.join(sell_reasons[:3])}")
                    cooldown = 6

            # === 买入: 需要至少4种指标共振(更严格) ===
            buy_confirmations = 0
            buy_reasons = []

            if sig['bottom'] >= 30:
                buy_confirmations += 1; buy_reasons.append(f"综合底背离={sig['bottom']:.0f}")
            if sig['separated_bottom'] >= 1 or sig['dif_bottom_div'] >= 1:
                buy_confirmations += 1; buy_reasons.append("MACD底背离")
            if sig['kdj_bottom_div'] >= 1 or (ind.get('K') and ind['K'] < 20):
                buy_confirmations += 1; buy_reasons.append(f"KDJ超卖K={ind.get('K',0):.0f}")
            if sig['cci_bottom_div'] >= 1 or (ind.get('CCI') and ind['CCI'] < -100):
                buy_confirmations += 1; buy_reasons.append(f"CCI={ind.get('CCI',0):.0f}")
            if sig['rsi_bottom_div'] >= 1 or (ind.get('RSI6') and ind['RSI6'] < 30):
                buy_confirmations += 1; buy_reasons.append(f"RSI={ind.get('RSI6',0):.0f}")
            if sig['vol_ground'] >= 1:
                buy_confirmations += 1; buy_reasons.append("地量信号")
            if sig.get('exhaust_buy'):
                buy_confirmations += 2; buy_reasons.append("背驰")

            if buy_confirmations >= 4 and eth_r < 0.5:
                ratio = min(0.3, 0.08 * buy_confirmations)
                if ratio > 0.05:
                    usdt, eth_qty = s.execute_buy(price, dt, usdt, eth_qty, ratio,
                        f"共振{buy_confirmations}: {', '.join(buy_reasons[:3])}")
                    cooldown = 12

        if idx % 4 == 0:
            s.history.append({'time': dt.isoformat(), 'total': round(usdt + eth_qty * price, 2)})

    return s.calc_result(main_df, first_price, usdt, eth_qty)


def run_strategy_H(data, signals_all):
    """策略H: MACD金叉死叉增强 — 背离+交叉做精确择时"""
    s = BaseStrategy("H: MACD交叉择时")
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eth_qty = s.initial_eth_value / first_price
    usdt = s.initial_usdt
    cooldown = 0
    min_eth_ratio = 0.05

    for idx in range(len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        if cooldown > 0: cooldown -= 1

        sig = {k: 0 if isinstance(v, (int, float)) else v for k, v in DEFAULT_SIG.items()}
        tw = 0
        for tf, w in {'4h': 1.0, '6h': 0.85, '2h': 0.7}.items():
            if tf in signals_all:
                s_tf = get_signal_at(signals_all[tf], dt) or DEFAULT_SIG
                sig['top'] += s_tf.get('top', 0) * w
                sig['bottom'] += s_tf.get('bottom', 0) * w
                for k in ['exhaust_sell', 'exhaust_buy']:
                    if s_tf.get(k): sig[k] = True
                for k in ['separated_top', 'separated_bottom', 'zero_returns_top', 'zero_returns_bottom']:
                    sig[k] = max(sig.get(k, 0), s_tf.get(k, 0))
                if s_tf.get('last_cross'): sig['last_cross'] = s_tf['last_cross']
                if s_tf.get('last_cross_above_zero'): sig['last_cross_above_zero'] = True
                tw += w
        if tw > 0: sig['top'] /= tw; sig['bottom'] /= tw

        eth_val = eth_qty * price
        total = usdt + eth_val
        eth_r = eth_val / total if total > 0 else 0

        if cooldown == 0:
            # 死叉+背离=卖出 (书中: 隔堆背离+DIF返回零轴=卖)
            is_death_cross = sig['last_cross'] == 'death'
            has_top_div = sig['top'] >= 20 or sig['separated_top'] >= 1

            if is_death_cross and has_top_div and eth_r > min_eth_ratio + 0.05:
                ratio = 0.5 if sig.get('exhaust_sell') else (0.35 if sig['top'] >= 40 else 0.2)
                available = eth_r - min_eth_ratio
                ratio = min(ratio, available * 0.9)
                if ratio > 0.05:
                    usdt, eth_qty = s.execute_sell(price, dt, usdt, eth_qty, ratio,
                        f"死叉+背离 T={sig['top']:.0f} sep={sig['separated_top']}")
                    cooldown = 8

            # 仅背驰也卖(不等交叉)
            elif sig.get('exhaust_sell') and eth_r > min_eth_ratio + 0.05:
                ratio = min(0.4, (eth_r - min_eth_ratio) * 0.7)
                if ratio > 0.05:
                    usdt, eth_qty = s.execute_sell(price, dt, usdt, eth_qty, ratio,
                        "背驰紧急卖出")
                    cooldown = 8

            # 纯顶背离渐进减仓
            elif sig['top'] >= 20 and eth_r > min_eth_ratio + 0.1:
                ratio = min(0.15, (eth_r - min_eth_ratio) * 0.3)
                if ratio > 0.05:
                    usdt, eth_qty = s.execute_sell(price, dt, usdt, eth_qty, ratio,
                        f"渐进减仓 T={sig['top']:.0f}")
                    cooldown = 5

            # 金叉+底背驰=买入 (书中: 两次隔堆背离+两次零轴返回+再度底背离)
            is_golden_cross = sig['last_cross'] == 'golden'
            has_exhaustion_buy = sig.get('exhaust_buy')
            has_strong_bottom = sig['separated_bottom'] >= 2 and sig['zero_returns_bottom'] >= 2

            if is_golden_cross and (has_exhaustion_buy or has_strong_bottom) and eth_r < 0.45:
                ratio = 0.3 if has_exhaustion_buy else 0.2
                usdt, eth_qty = s.execute_buy(price, dt, usdt, eth_qty, ratio,
                    f"金叉+底背驰 B={sig['bottom']:.0f}")
                cooldown = 16

        if idx % 4 == 0:
            s.history.append({'time': dt.isoformat(), 'total': round(usdt + eth_qty * price, 2)})

    return s.calc_result(main_df, first_price, usdt, eth_qty)


def run_strategy_I(data, signals_all):
    """策略I: KDJ+RSI极值过滤 — 只在超买超卖极值区操作"""
    s = BaseStrategy("I: KDJ+RSI极值")
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eth_qty = s.initial_eth_value / first_price
    usdt = s.initial_usdt
    cooldown = 0
    min_eth_ratio = 0.05

    for idx in range(len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        ind = get_realtime_indicators(main_df, idx)
        if cooldown > 0: cooldown -= 1

        sig = {k: 0 if isinstance(v, (int, float)) else v for k, v in DEFAULT_SIG.items()}
        tw = 0
        for tf, w in {'4h': 1.0, '6h': 0.85, '2h': 0.7}.items():
            if tf in signals_all:
                s_tf = get_signal_at(signals_all[tf], dt) or DEFAULT_SIG
                sig['top'] += s_tf.get('top', 0) * w
                sig['bottom'] += s_tf.get('bottom', 0) * w
                for k in ['exhaust_sell', 'exhaust_buy']:
                    if s_tf.get(k): sig[k] = True
                tw += w
        if tw > 0: sig['top'] /= tw; sig['bottom'] /= tw

        eth_val = eth_qty * price
        total = usdt + eth_val
        eth_r = eth_val / total if total > 0 else 0

        k_val = ind.get('K')
        rsi = ind.get('RSI6')
        cci = ind.get('CCI')

        if cooldown == 0:
            # 超买区卖出: KDJ K>75 或 RSI>65 或 CCI>100
            is_overbought = (k_val and k_val > 75) or (rsi and rsi > 65) or (cci and cci > 100)
            has_top_signal = sig['top'] >= 15 or sig.get('exhaust_sell')

            if is_overbought and has_top_signal and eth_r > min_eth_ratio + 0.05:
                # 超买程度决定卖出力度
                ob_strength = 0
                if k_val and k_val > 80: ob_strength += 1
                if rsi and rsi > 70: ob_strength += 1
                if cci and cci > 150: ob_strength += 1
                if sig.get('exhaust_sell'): ob_strength += 2

                ratio = min(0.6, 0.15 + 0.1 * ob_strength)
                available = eth_r - min_eth_ratio
                ratio = min(ratio, available * 0.9)
                if ratio > 0.05:
                    usdt, eth_qty = s.execute_sell(price, dt, usdt, eth_qty, ratio,
                        f"超买卖出 K={k_val:.0f} RSI={rsi:.0f} T={sig['top']:.0f}")
                    cooldown = 6

            # 仅背离信号也渐进减仓
            elif sig['top'] >= 20 and eth_r > min_eth_ratio + 0.1:
                ratio = min(0.12, (eth_r - min_eth_ratio) * 0.25)
                if ratio > 0.05:
                    usdt, eth_qty = s.execute_sell(price, dt, usdt, eth_qty, ratio,
                        f"渐进减仓 T={sig['top']:.0f}")
                    cooldown = 5

            # 超卖区买入: KDJ K<25 或 RSI<35 或 CCI<-100
            is_oversold = (k_val and k_val < 25) or (rsi and rsi < 35) or (cci and cci < -100)
            has_bottom_signal = sig.get('exhaust_buy') or sig['bottom'] >= 45

            if is_oversold and has_bottom_signal and eth_r < 0.45:
                os_strength = 0
                if k_val and k_val < 20: os_strength += 1
                if rsi and rsi < 30: os_strength += 1
                if cci and cci < -150: os_strength += 1
                if sig.get('exhaust_buy'): os_strength += 2

                ratio = min(0.35, 0.1 + 0.05 * os_strength)
                usdt, eth_qty = s.execute_buy(price, dt, usdt, eth_qty, ratio,
                    f"超卖买入 K={k_val:.0f} RSI={rsi:.0f} B={sig['bottom']:.0f}")
                cooldown = 12

        if idx % 4 == 0:
            s.history.append({'time': dt.isoformat(), 'total': round(usdt + eth_qty * price, 2)})

    return s.calc_result(main_df, first_price, usdt, eth_qty)


def run_strategy_J(data, signals_all):
    """策略J: 量价确认 — 价升量减确认卖出, 地量信号辅助买入"""
    s = BaseStrategy("J: 量价确认")
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eth_qty = s.initial_eth_value / first_price
    usdt = s.initial_usdt
    cooldown = 0
    min_eth_ratio = 0.05

    # 计算量能指标
    vol_ma20 = main_df['volume'].rolling(20).mean()
    vol_ma5 = main_df['volume'].rolling(5).mean()

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        if cooldown > 0: cooldown -= 1

        sig = {k: 0 if isinstance(v, (int, float)) else v for k, v in DEFAULT_SIG.items()}
        tw = 0
        for tf, w in {'4h': 1.0, '6h': 0.85, '2h': 0.7, '1h': 0.45}.items():
            if tf in signals_all:
                s_tf = get_signal_at(signals_all[tf], dt) or DEFAULT_SIG
                sig['top'] += s_tf.get('top', 0) * w
                sig['bottom'] += s_tf.get('bottom', 0) * w
                for k in ['exhaust_sell', 'exhaust_buy']:
                    if s_tf.get(k): sig[k] = True
                for k in ['vol_price_up_down', 'vol_ground']:
                    sig[k] = max(sig.get(k, 0), s_tf.get(k, 0))
                tw += w
        if tw > 0: sig['top'] /= tw; sig['bottom'] /= tw

        # 实时量能比
        cur_vol = main_df['volume'].iloc[idx]
        avg_vol = vol_ma20.iloc[idx]
        vol5 = vol_ma5.iloc[idx]
        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1
        is_shrinking = vol5 < avg_vol * 0.6  # 近5根量<均量60% = 缩量
        is_ground_vol = vol_ratio < 0.3  # 地量

        # 价格变化
        price_change_5 = (price - main_df['close'].iloc[idx - 5]) / main_df['close'].iloc[idx - 5]

        eth_val = eth_qty * price
        total = usdt + eth_val
        eth_r = eth_val / total if total > 0 else 0

        if cooldown == 0:
            # 价升量减 = 上涨动能不足, 配合背离卖出
            price_up_vol_down = price_change_5 > 0.01 and is_shrinking

            if (sig['top'] >= 15 or sig.get('exhaust_sell') or sig['vol_price_up_down'] >= 1) and eth_r > min_eth_ratio + 0.05:
                ratio = 0.15  # 基础减仓
                if price_up_vol_down: ratio += 0.1  # 价升量减额外加卖
                if sig.get('exhaust_sell'): ratio += 0.2
                if sig['top'] >= 40: ratio += 0.1

                available = eth_r - min_eth_ratio
                ratio = min(ratio, available * 0.9)
                if ratio > 0.05:
                    reason = f"T={sig['top']:.0f}"
                    if price_up_vol_down: reason += " 价升量减"
                    if sig['vol_price_up_down']: reason += " VP↓"
                    usdt, eth_qty = s.execute_sell(price, dt, usdt, eth_qty, ratio, reason)
                    cooldown = 5

            # 地量+底背离 = 可能底部, 小仓位试探
            if is_ground_vol and sig.get('exhaust_buy') and eth_r < 0.4:
                usdt, eth_qty = s.execute_buy(price, dt, usdt, eth_qty, 0.2,
                    f"地量+背驰 vol_r={vol_ratio:.2f}")
                cooldown = 16
            elif sig['bottom'] >= 50 and sig.get('vol_ground', 0) >= 1 and eth_r < 0.35:
                usdt, eth_qty = s.execute_buy(price, dt, usdt, eth_qty, 0.15,
                    f"底背离+地量 B={sig['bottom']:.0f}")
                cooldown = 16

        if idx % 4 == 0:
            s.history.append({'time': dt.isoformat(), 'total': round(usdt + eth_qty * price, 2)})

    return s.calc_result(main_df, first_price, usdt, eth_qty)


def run_strategy_K(data, signals_all):
    """策略K: 书中原则严格版
    卖出用背离(宁早勿晚): MACD隔堆背离1次+DIF返回零轴 → 卖; 或隔堆背离2次以上 → 卖
    买入用背驰(宁迟勿早): 隔堆背离2次以上+DIF两次返回零轴+再度底背离 → 才买
    """
    s = BaseStrategy("K: 书中严格原则")
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eth_qty = s.initial_eth_value / first_price
    usdt = s.initial_usdt
    cooldown = 0
    min_eth_ratio = 0.05

    for idx in range(len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        if cooldown > 0: cooldown -= 1

        # 只看4h和6h (最可靠的周期)
        sig = {k: 0 if isinstance(v, (int, float)) else v for k, v in DEFAULT_SIG.items()}
        for tf in ['4h', '6h']:
            if tf in signals_all:
                s_tf = get_signal_at(signals_all[tf], dt) or DEFAULT_SIG
                for k in DEFAULT_SIG:
                    if isinstance(DEFAULT_SIG[k], (int, float)):
                        sig[k] = max(sig.get(k, 0), s_tf.get(k, 0))
                    elif isinstance(DEFAULT_SIG[k], bool):
                        if s_tf.get(k): sig[k] = True
                    elif isinstance(DEFAULT_SIG[k], str):
                        if s_tf.get(k): sig[k] = s_tf[k]

        eth_val = eth_qty * price
        total = usdt + eth_val
        eth_r = eth_val / total if total > 0 else 0

        if cooldown == 0:
            # === 卖出: 书中规则 ===
            # 条件1: 隔堆背离>=1 + DIF至少返回零轴1次
            sell_cond1 = sig['sep_divs_top'] >= 1 and sig['zero_returns_top'] >= 1
            # 条件2: 隔堆背离>=2 (不要求返回零轴)
            sell_cond2 = sig['sep_divs_top'] >= 2
            # 条件3: 直接背驰
            sell_cond3 = sig.get('exhaust_sell')

            if (sell_cond1 or sell_cond2 or sell_cond3) and eth_r > min_eth_ratio + 0.05:
                if sell_cond3:
                    ratio = 0.6  # 背驰: 大幅卖出
                elif sell_cond2:
                    ratio = 0.4  # 两次隔堆
                else:
                    ratio = 0.3  # 一次隔堆+零轴返回

                available = eth_r - min_eth_ratio
                ratio = min(ratio, available * 0.9)
                if ratio > 0.05:
                    reason = f"sep={sig['sep_divs_top']} zero={sig['zero_returns_top']}"
                    if sell_cond3: reason = "背驰卖出 " + reason
                    usdt, eth_qty = s.execute_sell(price, dt, usdt, eth_qty, ratio, reason)
                    cooldown = 12

            # 渐进减仓(降低等待成本)
            elif sig['top'] >= 25 and eth_r > min_eth_ratio + 0.1:
                ratio = min(0.12, (eth_r - min_eth_ratio) * 0.2)
                if ratio > 0.05:
                    usdt, eth_qty = s.execute_sell(price, dt, usdt, eth_qty, ratio,
                        f"预警减仓 T={sig['top']:.0f}")
                    cooldown = 6

            # === 买入: 书中规则(严格) ===
            # 条件: 隔堆背离>=2 + DIF两次返回零轴 + 底背驰
            buy_cond = (sig['sep_divs_bottom'] >= 2 and sig['zero_returns_bottom'] >= 2
                        and sig.get('exhaust_buy'))

            if buy_cond and eth_r < 0.4:
                usdt, eth_qty = s.execute_buy(price, dt, usdt, eth_qty, 0.3,
                    f"背驰买入 sep={sig['sep_divs_bottom']} zero={sig['zero_returns_bottom']}")
                cooldown = 24

        if idx % 4 == 0:
            s.history.append({'time': dt.isoformat(), 'total': round(usdt + eth_qty * price, 2)})

    return s.calc_result(main_df, first_price, usdt, eth_qty)


def run_strategy_L(data, signals_all):
    """策略L: 全指标融合增强 — 综合所有最佳改进的终极策略

    核心思路:
    1. 渐进减仓为基础(策略E的优势)
    2. 多指标共振增强卖出信号(策略G)
    3. MACD交叉做精确择时(策略H)
    4. KDJ/RSI极值过滤(策略I)
    5. 量价确认(策略J)
    6. 书中原则做最终确认(策略K)
    """
    s = BaseStrategy("L: 全指标终极融合")
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eth_qty = s.initial_eth_value / first_price
    usdt = s.initial_usdt
    cooldown = 0
    min_eth_ratio = 0.05

    # 量能指标
    vol_ma20 = main_df['volume'].rolling(20).mean()
    vol_ma5 = main_df['volume'].rolling(5).mean()

    # 趋势判断
    df_4h = data.get('4h', main_df)
    ma50_4h = df_4h['close'].rolling(50).mean()

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        ind = get_realtime_indicators(main_df, idx)
        if cooldown > 0: cooldown -= 1

        # 趋势判断
        is_downtrend = True
        for i in range(len(df_4h) - 1, -1, -1):
            if df_4h.index[i] <= dt:
                ma_val = ma50_4h.iloc[i] if not pd.isna(ma50_4h.iloc[i]) else price
                is_downtrend = price < ma_val
                break

        # 融合多周期信号
        sig = {k: 0 if isinstance(v, (int, float)) else v for k, v in DEFAULT_SIG.items()}
        tw = 0
        weights = {'4h': 1.0, '6h': 0.85, '8h': 0.5, '2h': 0.7, '1h': 0.3}
        for tf, w in weights.items():
            if tf in signals_all:
                s_tf = get_signal_at(signals_all[tf], dt) or DEFAULT_SIG
                sig['top'] += s_tf.get('top', 0) * w
                sig['bottom'] += s_tf.get('bottom', 0) * w
                for k in ['exhaust_sell', 'exhaust_buy']:
                    if s_tf.get(k): sig[k] = True
                for k in DEFAULT_SIG:
                    if isinstance(DEFAULT_SIG[k], (int, float)) and k not in ('top', 'bottom'):
                        sig[k] = max(sig.get(k, 0), s_tf.get(k, 0))
                    elif isinstance(DEFAULT_SIG[k], str) and s_tf.get(k):
                        sig[k] = s_tf[k]
                tw += w
        if tw > 0: sig['top'] /= tw; sig['bottom'] /= tw

        # 实时指标
        k_val = ind.get('K')
        rsi = ind.get('RSI6')
        cci = ind.get('CCI')
        cur_vol = main_df['volume'].iloc[idx]
        avg_vol = vol_ma20.iloc[idx]
        vol_ratio = cur_vol / avg_vol if avg_vol and avg_vol > 0 else 1
        vol5 = vol_ma5.iloc[idx]
        is_shrinking = vol5 < avg_vol * 0.6 if avg_vol else False
        price_change_5 = (price - main_df['close'].iloc[idx - 5]) / main_df['close'].iloc[idx - 5]
        price_up_vol_down = price_change_5 > 0.01 and is_shrinking

        eth_val = eth_qty * price
        total = usdt + eth_val
        eth_r = eth_val / total if total > 0 else 0

        if cooldown == 0:
            # ============ 多维度卖出评分 ============
            sell_score = 0
            sell_parts = []

            # 1) 综合背离分数
            if sig['top'] >= 15:
                s1 = min(sig['top'] / 3, 10)
                sell_score += s1
                sell_parts.append(f"背离={sig['top']:.0f}")

            # 2) MACD隔堆/DIF背离
            if sig['separated_top'] >= 1 or sig['dif_top_div'] >= 1:
                sell_score += 8
                sell_parts.append("MACD背离")
            if sig['area_top_div'] >= 1:
                sell_score += 4
                sell_parts.append("面积背离")

            # 3) 书中背驰条件
            if sig['sep_divs_top'] >= 1 and sig['zero_returns_top'] >= 1:
                sell_score += 10
                sell_parts.append("隔堆+零轴")
            elif sig['sep_divs_top'] >= 2:
                sell_score += 12
                sell_parts.append("双隔堆")

            # 4) 背驰信号
            if sig.get('exhaust_sell'):
                sell_score += 15
                sell_parts.append("背驰")

            # 5) KDJ超买
            if k_val and k_val > 75:
                sell_score += 5
                sell_parts.append(f"KDJ={k_val:.0f}")

            # 6) RSI超买
            if rsi and rsi > 65:
                sell_score += 4
                sell_parts.append(f"RSI={rsi:.0f}")

            # 7) CCI高位
            if cci and cci > 100:
                sell_score += 4
                sell_parts.append(f"CCI={cci:.0f}")

            # 8) 量价背离
            if price_up_vol_down or sig['vol_price_up_down'] >= 1:
                sell_score += 5
                sell_parts.append("价升量减")

            # 9) 下跌趋势加分
            if is_downtrend:
                sell_score *= 1.3
                sell_parts.append("↓趋势")

            # 10) MACD死叉加分
            if sig.get('last_cross') == 'death':
                sell_score += 5
                sell_parts.append("死叉")

            # 卖出执行
            if sell_score >= 12 and eth_r > min_eth_ratio + 0.05:
                # 分数越高卖越多
                if sell_score >= 35:
                    ratio = 0.55
                elif sell_score >= 25:
                    ratio = 0.35
                elif sell_score >= 18:
                    ratio = 0.2
                else:
                    ratio = 0.12

                available = eth_r - min_eth_ratio
                ratio = min(ratio, available * 0.9)
                if ratio > 0.05:
                    usdt, eth_qty = s.execute_sell(price, dt, usdt, eth_qty, ratio,
                        f"S={sell_score:.0f} {', '.join(sell_parts[:4])}")
                    cooldown = 5

            # ============ 多维度买入评分 ============
            buy_score = 0
            buy_parts = []

            # 1) 综合底背离
            if sig['bottom'] >= 30:
                buy_score += min(sig['bottom'] / 5, 8)
                buy_parts.append(f"底背离={sig['bottom']:.0f}")

            # 2) MACD底背离
            if sig['separated_bottom'] >= 1 or sig['dif_bottom_div'] >= 1:
                buy_score += 6
                buy_parts.append("MACD底背离")

            # 3) 书中背驰条件(严格)
            if sig['sep_divs_bottom'] >= 2 and sig['zero_returns_bottom'] >= 2:
                buy_score += 15
                buy_parts.append("背驰条件")

            # 4) 背驰信号
            if sig.get('exhaust_buy'):
                buy_score += 12
                buy_parts.append("底背驰")

            # 5) KDJ超卖
            if k_val and k_val < 25:
                buy_score += 5
                buy_parts.append(f"KDJ={k_val:.0f}")

            # 6) RSI超卖
            if rsi and rsi < 35:
                buy_score += 4
                buy_parts.append(f"RSI={rsi:.0f}")

            # 7) 地量
            if vol_ratio < 0.3 or sig.get('vol_ground', 0) >= 1:
                buy_score += 5
                buy_parts.append("地量")

            # 8) 金叉
            if sig.get('last_cross') == 'golden':
                buy_score += 4
                buy_parts.append("金叉")

            # 9) 上涨趋势加分
            if not is_downtrend:
                buy_score *= 1.3
                buy_parts.append("↑趋势")
            else:
                buy_score *= 0.7  # 下跌中更谨慎

            # 买入执行(更高门槛)
            if buy_score >= 25 and eth_r < 0.45:
                if buy_score >= 40:
                    ratio = 0.3
                elif buy_score >= 30:
                    ratio = 0.2
                else:
                    ratio = 0.12

                usdt, eth_qty = s.execute_buy(price, dt, usdt, eth_qty, ratio,
                    f"B={buy_score:.0f} {', '.join(buy_parts[:4])}")
                cooldown = 14

        if idx % 4 == 0:
            s.history.append({'time': dt.isoformat(), 'total': round(usdt + eth_qty * price, 2)})

    return s.calc_result(main_df, first_price, usdt, eth_qty)


def run_strategy_E_best(data, signals_all):
    """策略E9复现: 只卖不买 (之前的冠军策略作为对照)"""
    s = BaseStrategy("E9: 只卖不买(对照)")
    main_df = data['1h']
    first_price = main_df['close'].iloc[0]
    eth_qty = s.initial_eth_value / first_price
    usdt = s.initial_usdt
    cooldown = 0
    min_eth_ratio = 0.05

    for idx in range(len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]
        if cooldown > 0: cooldown -= 1

        sig = {k: 0 if isinstance(v, (int, float)) else v for k, v in DEFAULT_SIG.items()}
        tw = 0
        for tf, w in {'4h': 1.0, '6h': 0.85, '2h': 0.7, '1h': 0.45}.items():
            if tf in signals_all:
                s_tf = get_signal_at(signals_all[tf], dt) or DEFAULT_SIG
                sig['top'] += s_tf.get('top', 0) * w
                for k in ['exhaust_sell']:
                    if s_tf.get(k): sig[k] = True
                tw += w
        if tw > 0: sig['top'] /= tw

        eth_val = eth_qty * price
        total = usdt + eth_val
        eth_r = eth_val / total if total > 0 else 0

        if cooldown == 0 and sig['top'] >= 18 and eth_r > min_eth_ratio + 0.05:
            available_r = eth_r - min_eth_ratio
            if sig.get('exhaust_sell') or sig['top'] >= 60:
                sell_r = min(0.55, available_r * 0.9)
            elif sig['top'] >= 40:
                sell_r = min(0.3, available_r * 0.6)
            else:
                sell_r = min(0.18, available_r * 0.4)
            if sell_r > 0.05:
                usdt, eth_qty = s.execute_sell(price, dt, usdt, eth_qty, sell_r,
                    f"减仓 T={sig['top']:.0f}")
                cooldown = 5

        if idx % 4 == 0:
            s.history.append({'time': dt.isoformat(), 'total': round(usdt + eth_qty * price, 2)})

    return s.calc_result(main_df, first_price, usdt, eth_qty)


def run_all():
    """运行全部增强策略"""
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
        ("E9(对照)", run_strategy_E_best),
        ("G", run_strategy_G),
        ("H", run_strategy_H),
        ("I", run_strategy_I),
        ("J", run_strategy_J),
        ("K", run_strategy_K),
        ("L", run_strategy_L),
    ]

    print(f"\n运行 {len(strategies)} 种增强策略...")
    print("=" * 110)

    results = []
    for name, func in strategies:
        print(f"\n>>> 策略 {name}...")
        r = func(data, signals_all)
        results.append(r)
        print(f"    收益: {r['strategy_return']:+.2f}% | 超额: {r['alpha']:+.2f}% | "
              f"回撤: {r['max_drawdown']:.2f}% | 交易: {r['total_trades']}笔 | "
              f"资产: ${r['final_total']:,.0f}")

    # 排名
    print("\n\n" + "=" * 110)
    print("                     增强策略排名 (按超额收益)")
    print("=" * 110)
    fmt = "{:>3} {:<32} {:>10} {:>10} {:>10} {:>10} {:>8} {:>12}"
    print(fmt.format("#", "策略", "策略收益", "买入持有", "超额收益", "最大回撤", "交易数", "最终资产"))
    print("-" * 110)
    for rank, r in enumerate(sorted(results, key=lambda x: x['alpha'], reverse=True), 1):
        star = " ★" if rank == 1 else ""
        print(fmt.format(
            rank, r['name'] + star,
            f"{r['strategy_return']:+.2f}%",
            f"{r['buy_hold_return']:+.2f}%",
            f"{r['alpha']:+.2f}%",
            f"{r['max_drawdown']:.2f}%",
            str(r['total_trades']),
            f"${r['final_total']:,.0f}",
        ))
    print("=" * 110)

    # 保存
    output = {
        'enhanced_results': [{
            'name': r['name'],
            'summary': {k: v for k, v in r.items() if k not in ('trades', 'history')},
            'trades': r['trades'],
            'history': r['history'],
        } for r in results],
        'best_strategy': sorted(results, key=lambda x: x['alpha'], reverse=True)[0]['name'],
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'strategy_enhanced_result.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果已保存: {output_path}")

    return output


if __name__ == '__main__':
    run_all()
