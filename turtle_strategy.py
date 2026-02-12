"""
海龟交易策略 — 基于《海龟交易法则（第三版）》(柯蒂斯·费思)

核心系统:
  系统1: 20日突破入市 / 10日突破退出 (短期)
  系统2: 55日突破入市 / 20日突破退出 (长期)

核心概念:
  N (ATR): 真实波动幅度均值, 用于标准化头寸规模和止损
  头寸单位: 1ATR = 账户的1%, 每市场最多4个单位
  逐步建仓: 每突破1/2N添加一个单位
  止损: 入市价±2N
  海龟思维: 掌握优势, 管理风险, 坚定不移, 简单明了

策略要点(14章精华):
  第1章: 流动性风险与价格风险 — 投机者承担价格风险
  第2章: 认知偏差 — 损失厌恶/处置效应/锚定效应/近期偏好
  第3章: 趋势跟踪 + 唐奇安通道突破 + ATR头寸管理
  第4章: 避免结果偏好/近期偏好, 从概率角度思考
  第5章: E-比率(优势比率) + 趋势组合过滤器
  第6章: 支撑阻力位 + 价格不稳定点 + 突破入市
  第7章: 四大风险(衰落/低回报/价格动荡/系统死亡)
  第8章: 破产风险 + 生存第一法则 + 头寸单位规模限制
  第9章: 突破/移动平均/波幅通道/定时退出/简单回顾
  第10章: 6个海龟式交易系统的历史回测
  第11-12章: 过度拟合/交易者效应/稳健性检验
  第13章: 多系统+多市场的防卫体系
  第14章: 心理纪律 — 克服自负, 谦虚, 坚定不移
  附录: 原版海龟交易法则完整版

适配: ETH/USDT 1h K线 (币安)
初始: 10万USDT
"""

import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import datetime
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binance_fetcher import fetch_binance_klines
from indicators import add_all_indicators


# ======================================================
#   海龟核心指标计算
# ======================================================

def compute_atr(df, period=20):
    """
    计算真实波动幅度均值 (ATR/N)
    海龟们称之为N, 现代称ATR
    
    True Range = max(H-L, |H-Cprev|, |L-Cprev|)
    N = ATR = 20日True Range的指数移动平均
    """
    high = df['high']
    low = df['low']
    close_prev = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = true_range.ewm(span=period, adjust=False).mean()
    
    df['turtle_atr'] = atr
    df['turtle_true_range'] = true_range
    return df


def compute_donchian_channels(df, entry_period=20, exit_period=10):
    """
    计算唐奇安通道 (Donchian Channels)
    
    入市通道: N日最高/最低 (系统1: 20日, 系统2: 55日)
    退出通道: M日最高/最低 (系统1: 10日, 系统2: 20日)
    """
    # 入市通道
    df[f'dc_upper_{entry_period}'] = df['high'].rolling(entry_period).max()
    df[f'dc_lower_{entry_period}'] = df['low'].rolling(entry_period).min()
    df[f'dc_mid_{entry_period}'] = (df[f'dc_upper_{entry_period}'] + df[f'dc_lower_{entry_period}']) / 2
    
    # 退出通道
    df[f'dc_upper_{exit_period}'] = df['high'].rolling(exit_period).max()
    df[f'dc_lower_{exit_period}'] = df['low'].rolling(exit_period).min()
    
    return df


def compute_trend_filter(df, short_ma=50, long_ma=300):
    """
    趋势组合过滤器 (第5章)
    
    50日均线 > 300日均线: 只做多
    50日均线 < 300日均线: 只做空
    
    对加密货币1h K线:
    50根1h ≈ 2天, 300根1h ≈ 12.5天
    适配为: 短期50, 长期200 (更适合加密市场波动性)
    """
    df['turtle_ma_short'] = df['close'].rolling(short_ma).mean()
    df['turtle_ma_long'] = df['close'].rolling(long_ma).mean()
    
    df['turtle_trend'] = 'neutral'
    bullish = df['turtle_ma_short'] > df['turtle_ma_long']
    bearish = df['turtle_ma_short'] < df['turtle_ma_long']
    df.loc[bullish, 'turtle_trend'] = 'bullish'
    df.loc[bearish, 'turtle_trend'] = 'bearish'
    
    return df


def compute_all_turtle_indicators(df):
    """计算所有海龟指标"""
    df = compute_atr(df, period=20)
    
    # 系统1: 20日突破入市, 10日突破退出
    df = compute_donchian_channels(df, entry_period=20, exit_period=10)
    
    # 系统2: 55日突破入市, 20日突破退出
    df = compute_donchian_channels(df, entry_period=55, exit_period=20)
    
    # 趋势过滤器
    df = compute_trend_filter(df, short_ma=50, long_ma=200)
    
    # 额外指标: 波动性百分位(用于判断市场状态)
    df['atr_pct'] = df['turtle_atr'] / df['close'] * 100  # ATR占价格百分比
    df['atr_percentile'] = df['turtle_atr'].rolling(200).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    return df


# ======================================================
#   海龟交易信号生成
# ======================================================

def detect_breakout(df, i, period, direction='long'):
    """
    检测唐奇安通道突破 (第3章+附录)
    
    做多: 价格超越N日最高点
    做空: 价格跌破N日最低点
    """
    if i < period + 1:
        return False, 0.0
    
    # 使用shift(1)确保用前一bar的通道
    upper_key = f'dc_upper_{period}'
    lower_key = f'dc_lower_{period}'
    
    if upper_key not in df.columns or lower_key not in df.columns:
        return False, 0.0
    
    close = df['close'].iloc[i]
    high = df['high'].iloc[i]
    low = df['low'].iloc[i]
    
    # 前一根bar的通道值
    prev_upper = df[upper_key].iloc[i - 1]
    prev_lower = df[lower_key].iloc[i - 1]
    
    if pd.isna(prev_upper) or pd.isna(prev_lower):
        return False, 0.0
    
    if direction == 'long' and high > prev_upper:
        strength = (high - prev_upper) / df['turtle_atr'].iloc[i] if df['turtle_atr'].iloc[i] > 0 else 0
        return True, strength
    elif direction == 'short' and low < prev_lower:
        strength = (prev_lower - low) / df['turtle_atr'].iloc[i] if df['turtle_atr'].iloc[i] > 0 else 0
        return True, strength
    
    return False, 0.0


def detect_exit_breakout(df, i, period, direction='long'):
    """
    检测退出突破 (附录)
    
    多头退出: 价格跌破M日最低点
    空头退出: 价格超越M日最高点
    """
    if i < period + 1:
        return False
    
    upper_key = f'dc_upper_{period}'
    lower_key = f'dc_lower_{period}'
    
    if upper_key not in df.columns or lower_key not in df.columns:
        return False
    
    close = df['close'].iloc[i]
    
    prev_lower = df[lower_key].iloc[i - 1]
    prev_upper = df[upper_key].iloc[i - 1]
    
    if pd.isna(prev_lower) or pd.isna(prev_upper):
        return False
    
    if direction == 'long' and close < prev_lower:
        return True
    elif direction == 'short' and close > prev_upper:
        return True
    
    return False


def check_previous_breakout_result(df, i, period, lookback=100):
    """
    检查上一次突破是否盈利 (系统1的过滤规则)
    
    如果上一次突破是盈利的, 当前信号被忽略(系统1独有)
    如果上一次突破亏损, 当前信号有效
    
    Returns: 'profitable', 'losing', 'unknown'
    """
    if i < period + lookback:
        return 'unknown'
    
    upper_key = f'dc_upper_{period}'
    lower_key = f'dc_lower_{period}'
    
    # 向前搜索上一次突破
    for j in range(i - 1, max(i - lookback, period), -1):
        prev_upper = df[upper_key].iloc[j - 1] if j > 0 else np.nan
        prev_lower = df[lower_key].iloc[j - 1] if j > 0 else np.nan
        
        if pd.isna(prev_upper) or pd.isna(prev_lower):
            continue
        
        was_long_breakout = df['high'].iloc[j] > prev_upper
        was_short_breakout = df['low'].iloc[j] < prev_lower
        
        if was_long_breakout or was_short_breakout:
            # 找到上一次突破, 检查后续表现
            entry_price = df['close'].iloc[j]
            atr_at_entry = df['turtle_atr'].iloc[j]
            
            if atr_at_entry <= 0 or pd.isna(atr_at_entry):
                return 'unknown'
            
            # 检查突破后是否在2N止损前获利
            max_adverse = 0
            max_favorable = 0
            
            check_end = min(j + 30, i)  # 最多看30根bar
            for k in range(j + 1, check_end):
                if was_long_breakout:
                    favorable = df['high'].iloc[k] - entry_price
                    adverse = entry_price - df['low'].iloc[k]
                else:
                    favorable = entry_price - df['low'].iloc[k]
                    adverse = df['high'].iloc[k] - entry_price
                
                max_favorable = max(max_favorable, favorable)
                max_adverse = max(max_adverse, adverse)
                
                # 如果先触及2N止损
                if max_adverse >= 2 * atr_at_entry:
                    return 'losing'
            
            # 如果有利变动超过入市风险
            if max_favorable > 2 * atr_at_entry:
                return 'profitable'
            
            return 'losing'
    
    return 'unknown'


def compute_position_size(account_equity, atr_value, risk_per_unit=0.01):
    """
    计算头寸单位规模 (第3章+第8章+附录)
    
    1个头寸单位: 1ATR的波动 = 账户净值的1%
    单位规模 = (账户净值 × 1%) / ATR
    
    对加密货币: 头寸以币数量表示
    """
    if atr_value <= 0 or pd.isna(atr_value):
        return 0
    
    dollar_volatility = atr_value  # 对ETH/USDT, ATR已是美元价值
    unit_size = (account_equity * risk_per_unit) / dollar_volatility
    
    return max(0, unit_size)


# ======================================================
#   海龟评分系统 (适配六书融合框架)
# ======================================================

def compute_turtle_scores(df):
    """
    计算海龟交易策略的买卖评分
    
    评分维度:
    1. 唐奇安通道突破信号 (系统1 + 系统2)
    2. 趋势过滤器 (50/200均线)
    3. ATR波动性分析
    4. 头寸管理信号 (逐步建仓/止损)
    5. 支撑阻力突破强度
    6. 市场状态判断 (平静/波动/趋势)
    
    Returns: sell_scores, buy_scores, signal_names
    """
    n = len(df)
    sell_scores = pd.Series(0.0, index=df.index)
    buy_scores = pd.Series(0.0, index=df.index)
    signal_names = pd.Series('', index=df.index)
    
    # 确保指标已计算
    if 'turtle_atr' not in df.columns:
        df = compute_all_turtle_indicators(df)
    
    for i in range(60, n):
        bs = 0.0  # buy score
        ss = 0.0  # sell score
        reasons = []
        
        close = df['close'].iloc[i]
        atr = df['turtle_atr'].iloc[i]
        trend = df['turtle_trend'].iloc[i] if 'turtle_trend' in df.columns else 'neutral'
        
        if pd.isna(atr) or atr <= 0:
            continue
        
        # ========== 1. 系统1: 20日突破 (第3章) ==========
        s1_long, s1_long_str = detect_breakout(df, i, 20, 'long')
        s1_short, s1_short_str = detect_breakout(df, i, 20, 'short')
        
        if s1_long:
            # 系统1过滤: 检查上一次突破是否盈利
            prev_result = check_previous_breakout_result(df, i, 20)
            if prev_result != 'profitable':
                bs += 25
                reasons.append('S1多突破')
                if s1_long_str > 1.0:
                    bs += 5
                    reasons.append('强突破')
            else:
                bs += 5  # 被过滤但仍有微弱信号
                reasons.append('S1过滤')
        
        if s1_short:
            prev_result = check_previous_breakout_result(df, i, 20)
            if prev_result != 'profitable':
                ss += 25
                reasons.append('S1空突破')
                if s1_short_str > 1.0:
                    ss += 5
                    reasons.append('强突破')
            else:
                ss += 5
                reasons.append('S1过滤')
        
        # ========== 2. 系统2: 55日突破 (附录) ==========
        s2_long, s2_long_str = detect_breakout(df, i, 55, 'long')
        s2_short, s2_short_str = detect_breakout(df, i, 55, 'short')
        
        if s2_long:
            bs += 30  # 系统2: 所有突破都有效, 权重更大
            reasons.append('S2多突破')
            if s2_long_str > 1.5:
                bs += 8
                reasons.append('S2强突')
        
        if s2_short:
            ss += 30
            reasons.append('S2空突破')
            if s2_short_str > 1.5:
                ss += 8
                reasons.append('S2强突')
        
        # ========== 3. 趋势过滤器 (第5章) ==========
        if trend == 'bullish':
            bs += 10
            ss -= 5  # 趋势向上时减弱空头信号
            reasons.append('趋势多')
        elif trend == 'bearish':
            ss += 10
            bs -= 5
            reasons.append('趋势空')
        
        # ========== 4. 退出突破信号 ==========
        # 系统1退出: 10日突破
        exit_long_s1 = detect_exit_breakout(df, i, 10, 'long')
        exit_short_s1 = detect_exit_breakout(df, i, 10, 'short')
        
        if exit_long_s1:
            ss += 15
            reasons.append('S1退出多')
        if exit_short_s1:
            bs += 15
            reasons.append('S1退出空')
        
        # 系统2退出: 20日突破
        exit_long_s2 = detect_exit_breakout(df, i, 20, 'long')
        exit_short_s2 = detect_exit_breakout(df, i, 20, 'short')
        
        if exit_long_s2:
            ss += 20
            reasons.append('S2退出多')
        if exit_short_s2:
            bs += 20
            reasons.append('S2退出空')
        
        # ========== 5. ATR波动性分析 (第7-8章) ==========
        atr_pct = df['atr_pct'].iloc[i] if 'atr_pct' in df.columns else 0
        atr_ptile = df['atr_percentile'].iloc[i] if 'atr_percentile' in df.columns else 0.5
        
        if not pd.isna(atr_ptile):
            if atr_ptile > 0.8:
                # 高波动: 信号更可靠但风险更大
                bs *= 1.1
                ss *= 1.1
                reasons.append('高波动')
            elif atr_ptile < 0.2:
                # 低波动: 可能即将突破 (平静后的风暴)
                bs *= 1.05
                ss *= 1.05
                reasons.append('低波动')
        
        # ========== 6. 价格相对通道位置 ==========
        if i >= 20 and 'dc_upper_20' in df.columns and 'dc_lower_20' in df.columns:
            upper = df['dc_upper_20'].iloc[i]
            lower = df['dc_lower_20'].iloc[i]
            if not pd.isna(upper) and not pd.isna(lower) and upper > lower:
                pos = (close - lower) / (upper - lower)
                if pos > 0.95:
                    bs += 5
                    reasons.append('通道顶')
                elif pos < 0.05:
                    ss += 5
                    reasons.append('通道底')
        
        # ========== 7. 逐步建仓信号 (1/2N间隔) ==========
        if i >= 2:
            price_change = close - df['close'].iloc[i - 1]
            half_n = atr * 0.5
            
            if price_change > half_n:
                bs += 8
                reasons.append('加仓多')
            elif price_change < -half_n:
                ss += 8
                reasons.append('加仓空')
        
        # ========== 8. 止损信号 (2N) ==========
        if i >= 5:
            recent_high = df['high'].iloc[max(0, i-4):i+1].max()
            recent_low = df['low'].iloc[max(0, i-4):i+1].min()
            
            drop_from_high = recent_high - close
            rise_from_low = close - recent_low
            
            if drop_from_high > 2 * atr:
                ss += 12
                reasons.append('止损多')
            if rise_from_low > 2 * atr:
                bs += 12
                reasons.append('止损空')
        
        # ========== 9. 市场状态判断 (第2章) ==========
        if i >= 20:
            recent_range = df['high'].iloc[i-20:i].max() - df['low'].iloc[i-20:i].min()
            expected_range = atr * 20 * 0.5  # 期望范围
            
            if recent_range > expected_range * 1.5:
                reasons.append('波动趋势')
            elif recent_range < expected_range * 0.5:
                reasons.append('平静窄幅')
        
        # ========== 10. 多维信号确认加成 ==========
        buy_count = sum(1 for r in reasons if any(k in r for k in 
            ['多突破', '趋势多', '退出空', '加仓多', '止损空', 'S2多']))
        sell_count = sum(1 for r in reasons if any(k in r for k in 
            ['空突破', '趋势空', '退出多', '加仓空', '止损多', 'S2空']))
        
        if buy_count >= 3:
            bs *= 1.2
        if sell_count >= 3:
            ss *= 1.2
        
        sell_scores.iloc[i] = min(100, max(0, ss))
        buy_scores.iloc[i] = min(100, max(0, bs))
        signal_names.iloc[i] = ' '.join(reasons[:6])
    
    return sell_scores, buy_scores, signal_names


# ======================================================
#   海龟回测引擎
# ======================================================

class TurtleBacktester:
    """
    海龟交易策略回测引擎
    
    完整实现:
    - 双系统 (S1: 20日突破, S2: 55日突破)
    - ATR头寸管理 (1ATR = 1%账户)
    - 逐步建仓 (1/2N间隔, 最多4单位)
    - 止损 (2N)
    - 退出突破 (S1: 10日, S2: 20日)
    - 系统1过滤 (上次盈利突破则忽略)
    - 趋势过滤器
    """
    
    def __init__(self, initial_capital=100000, risk_per_unit=0.01,
                 max_units=4, system='both', use_trend_filter=True):
        self.initial_capital = initial_capital
        self.risk_per_unit = risk_per_unit  # 1ATR = 账户的1%
        self.max_units = max_units
        self.system = system  # 'S1', 'S2', 'both'
        self.use_trend_filter = use_trend_filter
        
        # 状态
        self.capital = initial_capital
        self.position = None  # {'direction', 'units', 'entries', 'stop'}
        self.trades = []
        self.equity_curve = []
        self.max_equity = initial_capital
        self.max_drawdown = 0
    
    def run(self, df):
        """运行完整回测"""
        df = compute_all_turtle_indicators(df)
        n = len(df)
        
        self.capital = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []
        
        for i in range(60, n):
            close = df['close'].iloc[i]
            atr = df['turtle_atr'].iloc[i]
            trend = df['turtle_trend'].iloc[i]
            
            if pd.isna(atr) or atr <= 0:
                self._record_equity(i, close, df)
                continue
            
            # 1. 检查止损
            if self.position:
                self._check_stop_loss(i, df, atr)
            
            # 2. 检查退出信号
            if self.position:
                self._check_exit_signals(i, df)
            
            # 3. 检查入市信号
            if not self.position:
                self._check_entry_signals(i, df, atr, trend)
            
            # 4. 检查逐步建仓
            elif self.position and self.position['units'] < self.max_units:
                self._check_add_units(i, df, atr)
            
            # 5. 记录权益
            self._record_equity(i, close, df)
        
        # 收盘平仓
        if self.position:
            self._close_position(n - 1, df['close'].iloc[-1], 'end_of_test')
        
        return self._generate_results(df)
    
    def _check_entry_signals(self, i, df, atr, trend):
        """检查入市信号"""
        # 系统2: 55日突破 (更强, 优先检查)
        if self.system in ('S2', 'both'):
            s2_long, _ = detect_breakout(df, i, 55, 'long')
            s2_short, _ = detect_breakout(df, i, 55, 'short')
            
            if s2_long and (not self.use_trend_filter or trend != 'bearish'):
                self._open_position(i, df, 'long', atr, 'S2')
                return
            if s2_short and (not self.use_trend_filter or trend != 'bullish'):
                self._open_position(i, df, 'short', atr, 'S2')
                return
        
        # 系统1: 20日突破 (有过滤)
        if self.system in ('S1', 'both'):
            s1_long, _ = detect_breakout(df, i, 20, 'long')
            s1_short, _ = detect_breakout(df, i, 20, 'short')
            
            prev_result = check_previous_breakout_result(df, i, 20)
            
            if s1_long and prev_result != 'profitable':
                if not self.use_trend_filter or trend != 'bearish':
                    self._open_position(i, df, 'long', atr, 'S1')
                    return
            
            if s1_short and prev_result != 'profitable':
                if not self.use_trend_filter or trend != 'bullish':
                    self._open_position(i, df, 'short', atr, 'S1')
                    return
            
            # 系统1被过滤 → 55日突破作为保障(附录)
            if (s1_long or s1_short) and prev_result == 'profitable':
                s2_long_fallback, _ = detect_breakout(df, i, 55, 'long')
                s2_short_fallback, _ = detect_breakout(df, i, 55, 'short')
                
                if s2_long_fallback and (not self.use_trend_filter or trend != 'bearish'):
                    self._open_position(i, df, 'long', atr, 'S1-fallback')
                    return
                if s2_short_fallback and (not self.use_trend_filter or trend != 'bullish'):
                    self._open_position(i, df, 'short', atr, 'S1-fallback')
                    return
    
    def _open_position(self, i, df, direction, atr, system):
        """开仓"""
        entry_price = df['close'].iloc[i]
        unit_size = compute_position_size(self.capital, atr, self.risk_per_unit)
        
        if unit_size <= 0:
            return
        
        cost = unit_size * entry_price
        if cost > self.capital * 0.25:  # 单个单位不超过25%资金
            unit_size = self.capital * 0.25 / entry_price
        
        if direction == 'long':
            stop = entry_price - 2 * atr
        else:
            stop = entry_price + 2 * atr
        
        self.position = {
            'direction': direction,
            'units': 1,
            'entries': [(entry_price, unit_size)],
            'stop': stop,
            'system': system,
            'entry_bar': i,
            'last_add_price': entry_price,
            'atr_at_entry': atr,
        }
    
    def _check_add_units(self, i, df, atr):
        """逐步建仓 (每1/2N添加一个单位)"""
        if not self.position:
            return
        
        close = df['close'].iloc[i]
        direction = self.position['direction']
        last_add = self.position['last_add_price']
        half_n = atr * 0.5
        
        should_add = False
        if direction == 'long' and close >= last_add + half_n:
            should_add = True
        elif direction == 'short' and close <= last_add - half_n:
            should_add = True
        
        if should_add:
            unit_size = compute_position_size(self.capital, atr, self.risk_per_unit)
            if unit_size <= 0:
                return
            
            cost = unit_size * close
            total_exposure = sum(qty * price for price, qty in self.position['entries'])
            if total_exposure + cost > self.capital * 0.8:
                return
            
            self.position['entries'].append((close, unit_size))
            self.position['units'] += 1
            self.position['last_add_price'] = close
            
            # 调整止损 (1/2N滚动)
            if direction == 'long':
                self.position['stop'] = close - 2 * atr
            else:
                self.position['stop'] = close + 2 * atr
    
    def _check_stop_loss(self, i, df, atr):
        """检查止损"""
        if not self.position:
            return
        
        close = df['close'].iloc[i]
        low = df['low'].iloc[i]
        high = df['high'].iloc[i]
        direction = self.position['direction']
        stop = self.position['stop']
        
        if direction == 'long' and low <= stop:
            self._close_position(i, stop, 'stop_loss')
        elif direction == 'short' and high >= stop:
            self._close_position(i, stop, 'stop_loss')
    
    def _check_exit_signals(self, i, df):
        """检查退出信号"""
        if not self.position:
            return
        
        direction = self.position['direction']
        system = self.position['system']
        
        # 系统1: 10日退出, 系统2: 20日退出
        if 'S1' in system:
            if detect_exit_breakout(df, i, 10, direction):
                self._close_position(i, df['close'].iloc[i], 'exit_breakout_S1')
        else:
            if detect_exit_breakout(df, i, 20, direction):
                self._close_position(i, df['close'].iloc[i], 'exit_breakout_S2')
    
    def _close_position(self, i, exit_price, reason):
        """平仓"""
        if not self.position:
            return
        
        direction = self.position['direction']
        total_pnl = 0
        total_quantity = 0
        
        for entry_price, quantity in self.position['entries']:
            if direction == 'long':
                pnl = (exit_price - entry_price) * quantity
            else:
                pnl = (entry_price - exit_price) * quantity
            total_pnl += pnl
            total_quantity += quantity
        
        # 扣除手续费 (0.1% 双边)
        avg_entry = sum(p * q for p, q in self.position['entries']) / total_quantity if total_quantity > 0 else 0
        fee = (avg_entry + exit_price) * total_quantity * 0.001
        net_pnl = total_pnl - fee
        
        self.capital += net_pnl
        
        self.trades.append({
            'entry_bar': self.position['entry_bar'],
            'exit_bar': i,
            'direction': direction,
            'system': self.position['system'],
            'units': self.position['units'],
            'avg_entry': avg_entry,
            'exit_price': exit_price,
            'quantity': total_quantity,
            'gross_pnl': total_pnl,
            'fee': fee,
            'net_pnl': net_pnl,
            'pnl_pct': net_pnl / self.initial_capital * 100,
            'exit_reason': reason,
            'holding_bars': i - self.position['entry_bar'],
            'r_multiple': net_pnl / (self.position['atr_at_entry'] * total_quantity * 2) if total_quantity > 0 and self.position['atr_at_entry'] > 0 else 0,
        })
        
        self.position = None
    
    def _record_equity(self, i, close, df):
        """记录权益曲线"""
        equity = self.capital
        
        if self.position:
            unrealized = 0
            for entry_price, quantity in self.position['entries']:
                if self.position['direction'] == 'long':
                    unrealized += (close - entry_price) * quantity
                else:
                    unrealized += (entry_price - close) * quantity
            equity += unrealized
        
        self.equity_curve.append({
            'bar': i,
            'time': str(df.index[i]) if hasattr(df.index[i], 'isoformat') else str(df.index[i]),
            'equity': equity,
            'capital': self.capital,
        })
        
        if equity > self.max_equity:
            self.max_equity = equity
        
        drawdown = (self.max_equity - equity) / self.max_equity * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    def _generate_results(self, df):
        """生成回测结果"""
        if not self.equity_curve:
            return {}
        
        final_equity = self.equity_curve[-1]['equity']
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100
        
        winning_trades = [t for t in self.trades if t['net_pnl'] > 0]
        losing_trades = [t for t in self.trades if t['net_pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        
        avg_win = np.mean([t['net_pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t['net_pnl'] for t in losing_trades])) if losing_trades else 1
        profit_factor = (sum(t['net_pnl'] for t in winning_trades) / 
                        abs(sum(t['net_pnl'] for t in losing_trades))) if losing_trades and sum(t['net_pnl'] for t in losing_trades) != 0 else float('inf')
        
        # R乘数统计 (第4章)
        r_multiples = [t['r_multiple'] for t in self.trades if not np.isnan(t['r_multiple']) and not np.isinf(t['r_multiple'])]
        avg_r = np.mean(r_multiples) if r_multiples else 0
        
        # 连续盈亏
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        curr_wins = 0
        curr_losses = 0
        for t in self.trades:
            if t['net_pnl'] > 0:
                curr_wins += 1
                curr_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, curr_wins)
            else:
                curr_losses += 1
                curr_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, curr_losses)
        
        # 系统分布
        system_stats = {}
        for t in self.trades:
            sys = t['system']
            if sys not in system_stats:
                system_stats[sys] = {'count': 0, 'wins': 0, 'total_pnl': 0}
            system_stats[sys]['count'] += 1
            if t['net_pnl'] > 0:
                system_stats[sys]['wins'] += 1
            system_stats[sys]['total_pnl'] += t['net_pnl']
        
        result = {
            'description': '海龟交易策略回测 · 《海龟交易法则》费思',
            'book': '《海龟交易法则（第三版）》柯蒂斯·费思 著',
            'run_time': datetime.now().isoformat(),
            'data_range': f"{df.index[0]} ~ {df.index[-1]}",
            'total_bars': len(df),
            'initial_capital': self.initial_capital,
            'final_equity': round(final_equity, 2),
            'total_return_pct': round(total_return, 2),
            'max_drawdown_pct': round(self.max_drawdown, 2),
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate_pct': round(win_rate, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 4) if profit_factor != float('inf') else 'inf',
            'avg_r_multiple': round(avg_r, 4),
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'avg_holding_bars': round(np.mean([t['holding_bars'] for t in self.trades]), 1) if self.trades else 0,
            'system_stats': system_stats,
            'trades': self.trades[-50:],  # 最近50笔交易
            'equity_curve_sample': self.equity_curve[::max(1, len(self.equity_curve) // 200)],
        }
        
        # MAR比率 (第7章)
        if self.max_drawdown > 0:
            # 年化收益(粗略)
            days = len(df) / 24 if len(df) > 24 else 1
            annual_return = total_return * 365 / days
            result['mar_ratio'] = round(annual_return / self.max_drawdown, 4)
        else:
            result['mar_ratio'] = 'N/A'
        
        return result


# ======================================================
#   主函数
# ======================================================

def main(trade_days=None):
    print("=" * 80)
    print("  《海龟交易法则（第三版）》策略回测")
    print("  柯蒂斯·费思 著 · 唐奇安通道突破 + ATR头寸管理")
    print("=" * 80)
    
    days = trade_days or 60
    print(f"\n获取数据 (ETH/USDT 1h, {days}天)...")
    df = fetch_binance_klines("ETHUSDT", interval="1h", days=days)
    if df is None or len(df) < 100:
        print("数据不足"); return None
    
    print(f"  数据: {len(df)} 根K线, {df.index[0]} ~ {df.index[-1]}")
    
    # 1. 计算海龟指标
    print("\n计算海龟指标...")
    df = add_all_indicators(df)
    df = compute_all_turtle_indicators(df)
    
    print(f"  ATR(20): {df['turtle_atr'].iloc[-1]:.2f}")
    print(f"  ATR%: {df['atr_pct'].iloc[-1]:.2f}%")
    print(f"  趋势: {df['turtle_trend'].iloc[-1]}")
    print(f"  20日高: {df['dc_upper_20'].iloc[-1]:.2f}")
    print(f"  20日低: {df['dc_lower_20'].iloc[-1]:.2f}")
    print(f"  55日高: {df['dc_upper_55'].iloc[-1]:.2f}")
    print(f"  55日低: {df['dc_lower_55'].iloc[-1]:.2f}")
    
    # 2. 计算评分信号
    print("\n计算海龟信号评分...")
    sell_scores, buy_scores, signal_names = compute_turtle_scores(df)
    
    sell_active = int((sell_scores > 15).sum())
    buy_active = int((buy_scores > 15).sum())
    print(f"  卖出信号: {sell_active}个 | 买入信号: {buy_active}个")
    print(f"  最大卖出分: {sell_scores.max():.0f} | 最大买入分: {buy_scores.max():.0f}")
    
    # 信号统计
    all_reasons = []
    for s in signal_names:
        if s:
            all_reasons.extend(s.split())
    
    from collections import Counter
    reason_counts = Counter(all_reasons)
    print(f"\n  信号类型统计(TOP10):")
    for r, c in reason_counts.most_common(10):
        print(f"    {r}: {c}次")
    
    # 3. 运行回测
    print("\n" + "=" * 60)
    print("  运行海龟交易回测...")
    print("=" * 60)
    
    bt = TurtleBacktester(
        initial_capital=100000,
        risk_per_unit=0.01,
        max_units=4,
        system='both',
        use_trend_filter=True,
    )
    result = bt.run(df)
    
    if result:
        print(f"\n  初始资金: ${result['initial_capital']:,.0f}")
        print(f"  最终权益: ${result['final_equity']:,.2f}")
        print(f"  总收益率: {result['total_return_pct']:+.2f}%")
        print(f"  最大回撤: {result['max_drawdown_pct']:.2f}%")
        print(f"  总交易数: {result['total_trades']}")
        print(f"  胜率: {result['win_rate_pct']:.1f}%")
        print(f"  盈亏比: {result['profit_factor']}")
        print(f"  平均R乘数: {result['avg_r_multiple']:.4f}")
        print(f"  MAR比率: {result['mar_ratio']}")
        
        if result.get('system_stats'):
            print(f"\n  系统分布:")
            for sys_name, stats in result['system_stats'].items():
                wr = stats['wins'] / stats['count'] * 100 if stats['count'] > 0 else 0
                print(f"    {sys_name}: {stats['count']}笔, 胜率{wr:.0f}%, 盈亏${stats['total_pnl']:+,.0f}")
        
        # 保存结果
        result['signal_types'] = dict(reason_counts.most_common(20))
        
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'turtle_result.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, default=str, indent=2)
        print(f"\n结果已保存: {path}")
    
    return result


if __name__ == '__main__':
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    main(trade_days=days)
