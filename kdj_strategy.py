"""
KDJ波段操作策略 — 基于《随机指标KDJ：波段操作精解》(凌波)

核心指标:
  RSV: 收盘价在N日价格通道中的相对位置(0-100)
  K线: RSV的3日SMA平滑(快线)
  D线: K的3日SMA平滑(慢线)  
  J线: 3K - 2D (超前指标, 可>100或<0)
  KD-MACD柱线: 2*(K-D), 类MACD柱状图

核心策略(书中9章精华):
  第1章: 超买(>80)超卖(<20)区间识别 + 50线多空分界
  第2章: RSV→K→D→J的层级平滑计算
  第3章: K快线波段操作(第一/二买卖点 + 趋势线 + 形态 + 背离 + 钝化)
  第4章: KD交叉(低档金叉 + 高档死叉 + 二次交叉 + 回测不破)
  第5章: KD-MACD柱状线(抽脚缩头 + 杀多棒逼空棒 + 单/双/三峰谷)
  第6章: 四撞顶(底) + KDJ背离 + 多周期共振
  第7章: KDJ与K线/均线/成交量/MACD/布林线配合
  第8章: 多空分界 + 市场强弱 + 左右侧交易
  第9章: KDJ超买系统 + KD-MACD系统

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
from strategy_futures import FuturesEngine


# ======================================================
#   KDJ 指标计算
# ======================================================
def compute_kdj(df, n=9, m1=3, m2=3):
    """
    计算KDJ指标 (书中第2章公式)
    
    RSV = (Close - Low_N) / (High_N - Low_N) * 100
    K = SMA(RSV, m1, 1) = 2/3 * K_prev + 1/3 * RSV
    D = SMA(K, m2, 1) = 2/3 * D_prev + 1/3 * K
    J = 3*K - 2*D
    KD_MACD = 2*(K-D)  (书中第5章)
    """
    high_n = df['high'].rolling(n, min_periods=n).max()
    low_n = df['low'].rolling(n, min_periods=n).min()
    
    hl_range = high_n - low_n
    hl_range = hl_range.replace(0, np.nan)
    
    rsv = (df['close'] - low_n) / hl_range * 100
    rsv = rsv.fillna(50)
    
    # K = SMA(RSV, 3, 1) → 递推: K_t = 2/3 * K_{t-1} + 1/3 * RSV_t
    k_values = np.full(len(df), 50.0)
    d_values = np.full(len(df), 50.0)
    
    alpha_k = 1.0 / m1  # K线平滑系数 (默认 1/3)
    alpha_d = 1.0 / m2  # D线平滑系数 (默认 1/3, 但可以与m1不同)
    
    rsv_arr = rsv.values  # 预提取numpy数组，避免iloc开销
    for i in range(n - 1, len(df)):  # 从n-1开始, 不跳过首个有效RSV
        k_values[i] = (1 - alpha_k) * k_values[i-1] + alpha_k * rsv_arr[i]
        d_values[i] = (1 - alpha_d) * d_values[i-1] + alpha_d * k_values[i]
    
    df['kdj_rsv'] = rsv
    df['kdj_k'] = k_values
    df['kdj_d'] = d_values
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    
    # KD-MACD柱线 (书中第5章)
    df['kd_macd'] = 2 * (df['kdj_k'] - df['kdj_d'])
    
    # K线斜率(趋势) — 直接用df列做shift, 避免index不匹配导致全NaN
    df['kdj_k_slope'] = df['kdj_k'] - df['kdj_k'].shift(3)
    
    # D线斜率
    df['kdj_d_slope'] = df['kdj_d'] - df['kdj_d'].shift(3)
    
    return df


# ======================================================
#   KDJ 信号检测
# ======================================================
def detect_golden_cross(k, d, k_prev, d_prev):
    """检测金叉: K自下而上穿越D (书中第4章第2节)"""
    return k_prev <= d_prev and k > d


def detect_death_cross(k, d, k_prev, d_prev):
    """检测死叉: K自上而下穿越D (书中第4章第3节)"""
    return k_prev >= d_prev and k < d


def detect_divergence_top(df, i, lookback=30):
    """
    顶背离检测 (书中第3章第5节 + 第6章第2节)
    价格创新高, 但K线未创新高
    """
    if i < lookback + 5:
        return 0
    
    close_arr = df['close'].values
    k_arr = df['kdj_k'].values
    
    price = close_arr[i]
    k_val = k_arr[i]
    
    # 在lookback范围内找前一个波峰 (numpy切片 = 视图, 不复制)
    prices_slice = close_arr[i-lookback:i]
    k_slice = k_arr[i-lookback:i]
    
    # 找前一个价格高点
    rel_idx = np.argmax(prices_slice)
    prev_high_price = prices_slice[rel_idx]
    prev_high_k = k_slice[rel_idx]
    
    # 价格创新高(或接近), 但K线更低
    if price >= prev_high_price * 0.998 and k_val < prev_high_k * 0.92:
        # K线在高位(>60)时背离更有意义
        if prev_high_k > 60:
            strength = min(40, (prev_high_k - k_val) * 1.5)
            return strength
    
    return 0


def detect_divergence_bottom(df, i, lookback=30):
    """
    底背离检测 (书中第3章第5节 + 第6章第2节)
    价格创新低, 但K线未创新低
    """
    if i < lookback + 5:
        return 0
    
    close_arr = df['close'].values
    k_arr = df['kdj_k'].values
    
    price = close_arr[i]
    k_val = k_arr[i]
    
    prices_slice = close_arr[i-lookback:i]
    k_slice = k_arr[i-lookback:i]
    
    rel_idx = np.argmin(prices_slice)
    prev_low_price = prices_slice[rel_idx]
    prev_low_k = k_slice[rel_idx]
    
    # 价格创新低(或接近), 但K线更高
    if price <= prev_low_price * 1.002 and k_val > prev_low_k * 1.08:
        # K线在低位(<40)时背离更有意义
        if prev_low_k < 40:
            strength = min(40, (k_val - prev_low_k) * 1.5)
            return strength
    
    return 0


def detect_four_touch(df, i, lookback=40, zone='top'):
    """
    四撞顶/底检测 (书中第6章第1节)
    KDJ指标在超买/超卖区间反复触碰4次以上
    """
    if i < lookback:
        return 0
    
    k_arr = df['kdj_k'].values
    k_vals = k_arr[i-lookback:i+1]  # numpy切片, 不复制
    
    if zone == 'top':
        # 统计进入超买区间(>80)的次数
        touches = 0
        in_zone = False
        for kv in k_vals:
            if kv > 80 and not in_zone:
                touches += 1
                in_zone = True
            elif kv < 70:
                in_zone = False
        
        if touches >= 4 and k_vals[-1] < 80:
            return min(35, touches * 8)
    
    elif zone == 'bottom':
        touches = 0
        in_zone = False
        for kv in k_vals:
            if kv < 20 and not in_zone:
                touches += 1
                in_zone = True
            elif kv > 30:
                in_zone = False
        
        if touches >= 4 and k_vals[-1] > 20:
            return min(35, touches * 8)
    
    return 0


def detect_kd_macd_signal(df, i):
    """
    KD-MACD柱状线信号 (书中第5章)
    抽脚(绿柱缩短=看涨) / 缩头(红柱缩短=看跌)
    """
    if i < 5:
        return 0, 0
    
    sell_score = 0
    buy_score = 0
    
    kd_macd_arr = df['kd_macd'].values
    macd_val = kd_macd_arr[i]
    macd_prev = kd_macd_arr[i-1]
    macd_prev2 = kd_macd_arr[i-2]
    
    # 缩头: 红柱从增长转为缩短 (书中第5章第2节)
    if macd_prev > 0 and macd_prev2 > 0 and macd_val > 0:
        if macd_prev > macd_prev2 and macd_val < macd_prev:
            sell_score += 15  # 红柱缩头, 看跌
    
    # 抽脚: 绿柱从增长转为缩短
    if macd_prev < 0 and macd_prev2 < 0 and macd_val < 0:
        if macd_prev < macd_prev2 and macd_val > macd_prev:
            buy_score += 15  # 绿柱抽脚, 看涨
    
    # 杀多棒: 红柱突然大幅缩短(书中第5章第3节)
    if macd_prev > 5 and macd_val < macd_prev * 0.3:
        sell_score += 10
    
    # 逼空棒: 绿柱突然大幅缩短
    if macd_prev < -5 and macd_val > macd_prev * 0.3:
        buy_score += 10
    
    # 柱线零轴穿越
    if macd_prev <= 0 and macd_val > 0:
        buy_score += 8  # 绿转红
    if macd_prev >= 0 and macd_val < 0:
        sell_score += 8  # 红转绿
    
    return sell_score, buy_score


def detect_second_cross(df, i, lookback=20, direction='golden'):
    """
    二次交叉检测 (书中第4章第4节)
    在同一区域发生两次金叉/死叉, 第二次更可靠
    """
    if i < lookback + 2:
        return 0
    
    k_arr = df['kdj_k'].values
    d_arr = df['kdj_d'].values
    
    cross_count = 0
    
    for j in range(i - lookback, i):
        k = k_arr[j]
        d = d_arr[j]
        k_prev = k_arr[j-1]
        d_prev = d_arr[j-1]
        
        if direction == 'golden':
            if detect_golden_cross(k, d, k_prev, d_prev):
                cross_count += 1
        else:
            if detect_death_cross(k, d, k_prev, d_prev):
                cross_count += 1
    
    # 当前也是交叉
    k = k_arr[i]
    d = d_arr[i]
    k_prev = k_arr[i-1]
    d_prev = d_arr[i-1]
    
    is_current_cross = False
    if direction == 'golden':
        is_current_cross = detect_golden_cross(k, d, k_prev, d_prev)
    else:
        is_current_cross = detect_death_cross(k, d, k_prev, d_prev)
    
    if is_current_cross and cross_count >= 1:
        return min(25, cross_count * 12)
    
    return 0


def detect_pullback_hold(df, i):
    """
    回测不破 / 穿越不过 (书中第4章第5节)
    金叉后K回调到D线附近但未死叉, 然后再次上行 = 强势延续
    死叉后K反弹到D线附近但未金叉, 然后再次下行 = 弱势延续
    """
    if i < 5:
        return 0, 0
    
    sell_score = 0
    buy_score = 0
    
    k_arr = df['kdj_k'].values
    d_arr = df['kdj_d'].values
    
    k = k_arr[i]
    d = d_arr[i]
    k_prev = k_arr[i-1]
    d_prev = d_arr[i-1]
    k_prev2 = k_arr[i-2]
    d_prev2 = d_arr[i-2]
    
    # 回测不破: K回调接近D但不破, 然后再拉升
    # K在D上方运行, K先下降靠近D, 然后重新上升
    if k > d and k_prev > d_prev:
        gap_now = k - d
        gap_prev = k_prev - d_prev
        gap_prev2 = k_prev2 - d_prev2 if k_prev2 > d_prev2 else 0
        
        # 差距先缩小后扩大 = 回测不破
        if gap_prev < gap_prev2 and gap_now > gap_prev and gap_prev < 8:
            if d > 50:  # 在多头区间更有意义
                buy_score += 15
            else:
                buy_score += 8
    
    # 穿越不过: K反弹接近D但不破, 然后再下跌
    if k < d and k_prev < d_prev:
        gap_now = d - k
        gap_prev = d_prev - k_prev
        gap_prev2 = d_prev2 - k_prev2 if d_prev2 > k_prev2 else 0
        
        if gap_prev < gap_prev2 and gap_now > gap_prev and gap_prev < 8:
            if d < 50:
                sell_score += 15
            else:
                sell_score += 8
    
    return sell_score, buy_score


# ======================================================
#   KDJ 综合评分
# ======================================================
def compute_kdj_scores(df):
    """
    计算每根K线的KDJ综合得分
    返回: sell_scores, buy_scores, signal_names (Series)
    """
    df = compute_kdj(df)
    
    n = len(df)
    ss_arr = np.zeros(n)
    bs_arr = np.zeros(n)
    sig_list = [''] * n
    
    # 预提取numpy数组 — 避免主循环内反复调用.iloc (核心优化)
    _k = df['kdj_k'].values
    _d = df['kdj_d'].values
    _j = df['kdj_j'].values
    _kd_macd = df['kd_macd'].values
    _k_slope = df['kdj_k_slope'].values
    
    for i in range(15, n):
        ss = 0  # 卖出分
        bs = 0  # 买入分
        sell_rc = 0  # 卖出理由计数 (替代字符串匹配)
        buy_rc = 0   # 买入理由计数
        reasons = []
        
        k = _k[i]
        d = _d[i]
        j = _j[i]
        k_prev = _k[i-1]
        d_prev = _d[i-1]
        
        # ========== 1. 超买超卖 (第1章+第3章第2节) ==========
        if k > 80 and d > 75:
            ss += 10
            sell_rc += 1
            reasons.append('超买')
            if j > 100:
                ss += 5
                sell_rc += 1
                reasons.append('J>100')
        
        if k < 20 and d < 25:
            bs += 10
            buy_rc += 1
            reasons.append('超卖')
            if j < 0:
                bs += 5
                buy_rc += 1
                reasons.append('J<0')
        
        # ========== 2. 金叉/死叉 (第4章第2-3节) ==========
        is_golden = detect_golden_cross(k, d, k_prev, d_prev)
        is_death = detect_death_cross(k, d, k_prev, d_prev)
        
        if is_golden:
            if k < 20 and d < 25:
                bs += 30
                buy_rc += 1
                reasons.append('低位金叉')
            elif k < 50:
                bs += 20
                buy_rc += 1
                reasons.append('50下金叉')
            elif k < 80:
                bs += 10
                buy_rc += 1
                reasons.append('高位金叉')
            else:
                bs += 5
                buy_rc += 1
                reasons.append('极高金叉')
        
        if is_death:
            if k > 80 and d > 75:
                ss += 30
                sell_rc += 1
                reasons.append('高位死叉')
            elif k > 50:
                ss += 20
                sell_rc += 1
                reasons.append('50上死叉')
            elif k > 20:
                ss += 10
                sell_rc += 1
                reasons.append('低位死叉')
            else:
                ss += 5
                sell_rc += 1
                reasons.append('极低死叉')
        
        # ========== 3. 50线穿越 (第3章第1节) ==========
        k_prev3 = _k[i-3] if i >= 3 else 50
        if k > 50 and k_prev3 < 50:
            bs += 8
            buy_rc += 1
            reasons.append('K突破50')
        if k < 50 and k_prev3 > 50:
            ss += 8
            sell_rc += 1
            reasons.append('K跌破50')
        
        # ========== 4. 背离 (第3章第5节 + 第6章第2节) ==========
        div_top = detect_divergence_top(df, i)
        if div_top > 0:
            ss += div_top
            sell_rc += 1
            reasons.append(f'顶背离{div_top:.0f}')
        
        div_bottom = detect_divergence_bottom(df, i)
        if div_bottom > 0:
            bs += div_bottom
            buy_rc += 1
            reasons.append(f'底背离{div_bottom:.0f}')
        
        # ========== 5. KD-MACD柱线信号 (第5章) ==========
        kd_ss, kd_bs = detect_kd_macd_signal(df, i)
        ss += kd_ss
        bs += kd_bs
        if kd_ss > 0:
            sell_rc += 1
            reasons.append(f'KD柱空{kd_ss}')
        if kd_bs > 0:
            buy_rc += 1
            reasons.append(f'KD柱多{kd_bs}')
        
        # ========== 6. 二次交叉 (第4章第4节) ==========
        second_golden = detect_second_cross(df, i, direction='golden')
        if second_golden > 0:
            bs += second_golden
            buy_rc += 1
            reasons.append('二次金叉')
        
        second_death = detect_second_cross(df, i, direction='death')
        if second_death > 0:
            ss += second_death
            sell_rc += 1
            reasons.append('二次死叉')
        
        # ========== 7. 四撞顶/底 (第6章第1节) ==========
        four_top = detect_four_touch(df, i, zone='top')
        if four_top > 0:
            ss += four_top
            sell_rc += 1
            reasons.append(f'四撞顶{four_top}')
        
        four_bottom = detect_four_touch(df, i, zone='bottom')
        if four_bottom > 0:
            bs += four_bottom
            buy_rc += 1
            reasons.append(f'四撞底{four_bottom}')
        
        # ========== 8. 回测不破 / 穿越不过 (第4章第5节) ==========
        pb_ss, pb_bs = detect_pullback_hold(df, i)
        ss += pb_ss
        bs += pb_bs
        if pb_ss > 0:
            sell_rc += 1
            reasons.append('穿越不过')
        if pb_bs > 0:
            buy_rc += 1
            reasons.append('回测不破')
        
        # ========== 9. K方向加成 (第3章第1节) ==========
        k_slope_val = _k_slope[i]
        k_slope = k_slope_val if not np.isnan(k_slope_val) else 0
        if k_slope < -10 and k > 60:
            ss += 5  # K线快速下降 + 在高位
        if k_slope > 10 and k < 40:
            bs += 5  # K线快速上升 + 在低位
        
        # ========== 10. J线极值强化 ==========
        if j > 110 and k_prev > k:  # J极超买+拐头
            ss += 8
            sell_rc += 1
            reasons.append('J极超买拐头')
        if j < -10 and k_prev < k:  # J极超卖+拐头
            bs += 8
            buy_rc += 1
            reasons.append('J极超卖拐头')
        
        # ========== 11. 多空分界线方向 (第8章第1节) ==========
        if k > 50 and d > 50 and k_slope > 0:
            bs += 3
        if k < 50 and d < 50 and k_slope < 0:
            ss += 3
        
        # 可靠性乘数: 用整数计数替代字符串匹配 (性能优化)
        if sell_rc >= 3:
            ss *= 1.15
        if buy_rc >= 3:
            bs *= 1.15
        
        ss_arr[i] = min(100, ss)
        bs_arr[i] = min(100, bs)
        if reasons:
            sig_list[i] = ' '.join(reasons[:5])
    
    sell_scores = pd.Series(ss_arr, index=df.index)
    buy_scores = pd.Series(bs_arr, index=df.index)
    signal_names = pd.Series(sig_list, index=df.index, dtype=str)
    
    return sell_scores, buy_scores, signal_names


# ======================================================
#   主函数
# ======================================================
def main():
    print("=" * 80)
    print("  《随机指标KDJ：波段操作精解》策略回测")
    print("  凌波 著 · KDJ波段操作 · 9章核心理论完全实现")
    print("=" * 80)
    
    print("\n获取数据...")
    df = fetch_binance_klines("ETHUSDT", interval="1h", days=60)
    if df is None or len(df) < 100:
        print("数据不足"); return
    
    df = add_all_indicators(df)
    
    print("\n计算KDJ信号...")
    sell_scores, buy_scores, signal_names = compute_kdj_scores(df)
    
    sell_active = int((sell_scores > 15).sum())
    buy_active = int((buy_scores > 15).sum())
    print(f"  卖出信号: {sell_active}个 | 买入信号: {buy_active}个")
    print(f"  最大卖出分: {sell_scores.max():.0f} | 最大买入分: {buy_scores.max():.0f}")
    
    # 统计信号类型
    all_reasons = []
    for s in signal_names:
        if s:
            all_reasons.extend(s.split())
    
    from collections import Counter
    reason_counts = Counter(all_reasons)
    print(f"\n  信号类型统计(TOP10):")
    for r, c in reason_counts.most_common(10):
        print(f"    {r}: {c}次")
    
    # 保存结果
    output = {
        'description': 'KDJ波段操作策略 · 《随机指标KDJ》凌波',
        'book': '《随机指标KDJ：波段操作精解》凌波 著',
        'run_time': datetime.now().isoformat(),
        'data_range': f"{df.index[0]} ~ {df.index[-1]}",
        'total_bars': len(df),
        'sell_signals': sell_active,
        'buy_signals': buy_active,
        'max_sell_score': float(sell_scores.max()),
        'max_buy_score': float(buy_scores.max()),
        'signal_types': dict(reason_counts.most_common(20)),
    }
    
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'kdj_result.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果已保存: {path}")
    
    return sell_scores, buy_scores, signal_names


if __name__ == '__main__':
    main()
