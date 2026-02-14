"""
第四章: 背离与背驰
核心概念:
  - 背离: 趋势衰竭的信号, 可以背离了又背离
  - 背驰: 趋势的最后一次背离, 一旦产生马上反转
  - 背驰是背离量积累到质变的飞跃

操作核心原则:
  - 卖点用背离 (一旦隔堆背离+MACD返回零轴, 就考虑卖出)
  - 买点用背驰 (二次以上隔堆背离+MACD二次返回零轴, 才考虑买入)
"""

import numpy as np
import pandas as pd
import config as cfg
from indicators import (
    calc_macd, find_swing_highs, find_swing_lows,
    identify_bar_groups, find_macd_crosses
)


class ExhaustionAnalyzer:
    """背离与背驰分析器"""

    def __init__(self, df: pd.DataFrame, _skip_copy: bool = False):
        self.df = df if _skip_copy else df.copy()
        self._ensure_macd()
        self.bar_groups = identify_bar_groups(self.df['MACD_BAR'])

    def _ensure_macd(self):
        if 'DIF' not in self.df.columns:
            macd_df = calc_macd(self.df['close'])
            self.df['DIF'] = macd_df['DIF']
            self.df['DEA'] = macd_df['DEA']
            self.df['MACD_BAR'] = macd_df['MACD_BAR']

    # ============================================================
    # 统计MACD黄白线返回零轴次数 (背驰关键条件)
    # ============================================================
    def _count_zero_returns(self, direction: str = 'top') -> list:
        """
        统计DIF线远离零轴后返回零轴的次数和位置

        背驰条件之一:
        - 顶背驰: DIF从高位返回0轴附近
        - 底背驰: DIF从低位返回0轴附近
        """
        dif = self.df['DIF']
        _dif = dif.values
        returns = []

        if direction == 'top':
            # DIF先上升远离零轴, 然后返回零轴附近
            was_high = False
            high_peak = 0
            for i in range(1, len(dif)):
                if _dif[i] > 0.5:  # 远离零轴
                    was_high = True
                    high_peak = max(high_peak, _dif[i])
                elif was_high and abs(_dif[i]) < high_peak * 0.2:
                    # 返回零轴附近
                    returns.append({
                        'idx': i,
                        'dif': _dif[i],
                        'peak': high_peak
                    })
                    was_high = False
                    high_peak = 0
        else:
            # DIF先下降远离零轴, 然后返回零轴附近
            was_low = False
            low_valley = 0
            for i in range(1, len(dif)):
                if _dif[i] < -0.5:
                    was_low = True
                    low_valley = min(low_valley, _dif[i])
                elif was_low and abs(_dif[i]) < abs(low_valley) * 0.2:
                    returns.append({
                        'idx': i,
                        'dif': _dif[i],
                        'valley': low_valley
                    })
                    was_low = False
                    low_valley = 0

        return returns

    # ============================================================
    # 统计隔堆背离次数
    # ============================================================
    def _count_separated_divergences(self, direction: str = 'top') -> list:
        """
        统计隔堆背离的发生位置和次数

        书中指出:
        - 背驰大多发生在第二次隔堆背离处
        - 第一次隔堆背离就引发背驰的概率很小但不可忽视
        """
        df = self.df
        groups = self.bar_groups
        _high = df['high'].values
        _low = df['low'].values

        target_type = 'red' if direction == 'top' else 'green'
        target_groups = [g for g in groups if g['type'] == target_type]

        separated_divs = []

        for i in range(1, len(target_groups)):
            prev_g = target_groups[i - 1]
            curr_g = target_groups[i]

            # 检查中间是否有明显的反色柱堆(隔堆条件)
            between = [g for g in groups
                       if g['start_idx'] > prev_g['end_idx']
                       and g['end_idx'] < curr_g['start_idx']
                       and g['type'] != target_type]

            has_significant_opposite = any(
                g['area'] > min(prev_g['area'], curr_g['area']) * 0.1
                for g in between
            )

            if not has_significant_opposite:
                continue

            # 检查面积/长度是否缩小
            area_ratio = curr_g['area'] / prev_g['area'] if prev_g['area'] > 0 else 1
            length_ratio = curr_g['max_length'] / prev_g['max_length'] if prev_g['max_length'] > 0 else 1

            if area_ratio < 0.9 or length_ratio < 0.9:
                # 确认股价条件
                if direction == 'top':
                    prev_high = np.max(_high[prev_g['start_idx']:prev_g['end_idx'] + 1])
                    curr_high = np.max(_high[curr_g['start_idx']:curr_g['end_idx'] + 1])
                    if curr_high <= prev_high:
                        continue
                else:
                    prev_low = np.min(_low[prev_g['start_idx']:prev_g['end_idx'] + 1])
                    curr_low = np.min(_low[curr_g['start_idx']:curr_g['end_idx'] + 1])
                    if curr_low >= prev_low:
                        continue

                separated_divs.append({
                    'idx': curr_g['end_idx'],
                    'area_ratio': area_ratio,
                    'length_ratio': length_ratio,
                    'prev_group': prev_g,
                    'curr_group': curr_g
                })

        return separated_divs

    # ============================================================
    # 背驰检测 (核心方法)
    # ============================================================
    def detect_exhaustion(self) -> list:
        """
        检测背驰(趋势最后一次背离)

        背驰四大识别特征 (第四章第二节):
        (1) 背驰建立在背离基础上, 先有背离后有背驰
        (2) 背驰大多发生在第二次隔堆背离, 但也可能第一次隔堆就发生
            (条件: 至少一次MACD黄白线返回零轴)
        (3) 第一次隔堆背离产生背驰需要附加条件:
            MACD黄白线至少存在一个返回零轴的过程
        (4) MACD黄白线远离零轴->返回零轴->重新上升/下降不创新高/低
            -> 极易产生背驰

        返回: 背驰信号列表
        """
        signals = []

        # 检测顶背驰
        signals.extend(self._detect_directional_exhaustion('top'))

        # 检测底背驰
        signals.extend(self._detect_directional_exhaustion('bottom'))

        return signals

    def _detect_directional_exhaustion(self, direction: str) -> list:
        """检测指定方向的背驰"""
        signals = []
        df = self.df

        # 获取隔堆背离
        separated_divs = self._count_separated_divergences(direction)
        if not separated_divs:
            return signals

        # 获取零轴返回
        zero_returns = self._count_zero_returns(direction)
        zero_return_indices = [r['idx'] for r in zero_returns]

        # 检查DIF的背离(黄白线不创新高/新低)
        dif = df['DIF']

        for i, div in enumerate(separated_divs):
            div_idx = div['idx']

            # 条件1: 第二次或以上隔堆背离 -> 高概率背驰
            if i >= 1:
                # 检查DIF是否不创新高/新低
                dif_diverges = self._check_dif_divergence(div, direction)

                if dif_diverges:
                    confidence = 'high'
                    desc = f'第{i + 1}次隔堆背离'
                    signals.append(self._create_exhaustion_signal(
                        div, direction, confidence, desc, i + 1, len(zero_returns)
                    ))
                continue

            # 条件2: 第一次隔堆背离 + MACD返回零轴 -> 可能背驰
            has_zero_return_before = any(
                zr < div_idx for zr in zero_return_indices
            )

            if has_zero_return_before:
                dif_diverges = self._check_dif_divergence(div, direction)

                if dif_diverges:
                    confidence = 'medium'
                    desc = '第1次隔堆背离(DIF已返回零轴)'
                    signals.append(self._create_exhaustion_signal(
                        div, direction, confidence, desc, 1, len(zero_returns)
                    ))

        return signals

    def _check_dif_divergence(self, div, direction):
        """检查DIF线是否在该隔堆背离处产生背离"""
        df = self.df
        _dif = df['DIF'].values
        curr_g = div['curr_group']
        prev_g = div['prev_group']

        if direction == 'top':
            dif_prev_max = np.max(_dif[prev_g['start_idx']:prev_g['end_idx'] + 1])
            dif_curr_max = np.max(_dif[curr_g['start_idx']:curr_g['end_idx'] + 1])
            return dif_curr_max < dif_prev_max
        else:
            dif_prev_min = np.min(_dif[prev_g['start_idx']:prev_g['end_idx'] + 1])
            dif_curr_min = np.min(_dif[curr_g['start_idx']:curr_g['end_idx'] + 1])
            return dif_curr_min > dif_prev_min

    def _create_exhaustion_signal(self, div, direction, confidence, desc,
                                    separated_count, zero_return_count):
        """创建背驰信号"""
        df = self.df
        _close = df['close'].values
        idx = div['idx']
        dir_cn = '顶' if direction == 'top' else '底'

        return {
            'type': 'exhaustion',
            'direction': direction,
            'idx': idx,
            'date': df.index[idx],
            'price': _close[idx],
            'confidence': confidence,
            'separated_divergence_count': separated_count,
            'zero_return_count': zero_return_count,
            'area_ratio': div['area_ratio'],
            'length_ratio': div['length_ratio'],
            'description': f'{dir_cn}背驰({confidence}): {desc}, '
                           f'DIF返回零轴{zero_return_count}次'
        }

    # ============================================================
    # 卖点/买点建议 (第四章第三节)
    # ============================================================
    def generate_trade_signals(self) -> dict:
        """
        根据背离/背驰原则生成交易信号

        核心原则:
        卖点用背离:
          条件1: MACD黄白线远离零轴后返回零轴重新上升不创新高 + 一次隔堆背离
          条件2: 虽未返回零轴但已二次以上隔堆背离
          -> 满足任一条件即考虑卖出

        买点用背驰:
          条件1: 二次以上隔堆背离
          条件2: MACD黄白线二次返回零轴后再度底背离
          -> 同时满足两个条件才考虑买入
        """
        sell_signals = []
        buy_signals = []

        exhaustion_signals = self.detect_exhaustion()

        for sig in exhaustion_signals:
            if sig['direction'] == 'top':
                # 卖点用背离原则
                should_sell = False
                reason = ''

                if (sig['zero_return_count'] >= 1 and
                        sig['separated_divergence_count'] >= 1):
                    should_sell = True
                    reason = f'卖出条件1: DIF返回零轴{sig["zero_return_count"]}次 + ' \
                             f'{sig["separated_divergence_count"]}次隔堆背离'

                elif sig['separated_divergence_count'] >= 2:
                    should_sell = True
                    reason = f'卖出条件2: {sig["separated_divergence_count"]}次隔堆背离(未返回零轴)'

                if should_sell:
                    sell_signals.append({
                        **sig,
                        'action': 'sell',
                        'reason': reason,
                        'priority': 'high' if sig['confidence'] == 'high' else 'medium'
                    })

            elif sig['direction'] == 'bottom':
                # 买点用背驰原则 (更严格)
                should_buy = False
                reason = ''

                if (sig['separated_divergence_count'] >= 2 and
                        sig['zero_return_count'] >= 2):
                    should_buy = True
                    reason = f'买入条件: {sig["separated_divergence_count"]}次隔堆背离 + ' \
                             f'DIF {sig["zero_return_count"]}次返回零轴'

                if should_buy:
                    buy_signals.append({
                        **sig,
                        'action': 'buy',
                        'reason': reason,
                        'priority': 'high' if sig['confidence'] == 'high' else 'medium'
                    })

        return {
            'exhaustion_signals': exhaustion_signals,
            'sell_signals': sell_signals,
            'buy_signals': buy_signals,
        }

    def analyze_all(self) -> dict:
        """执行完整的背离背驰分析"""
        trade_signals = self.generate_trade_signals()
        return {
            'exhaustion': trade_signals['exhaustion_signals'],
            'sell_signals': trade_signals['sell_signals'],
            'buy_signals': trade_signals['buy_signals'],
            'zero_returns_top': self._count_zero_returns('top'),
            'zero_returns_bottom': self._count_zero_returns('bottom'),
            'separated_divs_top': self._count_separated_divergences('top'),
            'separated_divs_bottom': self._count_separated_divergences('bottom'),
        }
