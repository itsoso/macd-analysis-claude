"""
第二章: 几何形态背离
包含5种形态背离:
  1. 趋势K线背离 - 上涨中出现长阴/下跌中出现长阳
  2. 趋势幅度背离 - 后一段涨跌幅度小于前一段
  3. 时间度背离   - 后一段持续时间短于前一段
  4. 均线背离     - 股价与长周期均线方向相反(向上/向下交叉背离)
  5. 均线相交面积背离 - 短长期均线间面积逐渐缩小
"""

import numpy as np
import pandas as pd
import config as cfg
from indicators import (
    find_swing_highs, find_swing_lows, identify_trend_segments,
    calc_ma, find_peaks, find_valleys
)


class PatternDivergenceAnalyzer:
    """几何形态背离分析器"""

    def __init__(self, df: pd.DataFrame, _skip_copy: bool = False):
        """
        初始化
        df: 包含 open, high, low, close, volume 及均线列的DataFrame
        """
        self.df = df if _skip_copy else df.copy()
        self._ensure_ma_columns()

    def _ensure_ma_columns(self):
        """确保均线列存在"""
        for p in [cfg.MA_SHORT, cfg.MA_MID, cfg.MA_LONG, cfg.MA_LONG2,
                  cfg.MA_LONG3, cfg.MA_LONG4]:
            col = f'MA{p}'
            if col not in self.df.columns:
                self.df[col] = calc_ma(self.df['close'], p)

    # ============================================================
    # 1. 趋势K线背离 (第二章第一节)
    # 上涨中突然出现长阴K线, 或下跌中突然出现长阳K线
    # ============================================================
    def detect_kline_divergence(self, lookback: int = 20,
                                body_ratio: float = None) -> list:
        """
        检测趋势K线背离

        原理: 在上涨趋势中突然出现一根强力阴线, 或在下跌趋势中突然出现
        一根强力阳线, 说明反趋势力量突然增强, 构成初级背离警示信号。

        参数:
            lookback: 回看周期, 用于判断短期趋势方向
            body_ratio: K线实体相对于价格的占比阈值

        返回: 背离信号列表
        """
        if body_ratio is None:
            body_ratio = cfg.KLINE_DIVERGENCE_BODY_RATIO

        signals = []
        df = self.df
        _close = df['close'].values
        _open = df['open'].values

        for i in range(lookback, len(df)):
            # 计算当前K线的实体大小
            body = abs(_close[i] - _open[i])
            price = _close[i]
            ratio = body / price if price > 0 else 0

            if ratio < body_ratio:
                continue

            # 判断是阴线还是阳线
            is_bearish = _close[i] < _open[i]
            is_bullish = _close[i] > _open[i]

            # 判断近期趋势: 用lookback期内的价格变化
            trend_change = _close[i - 1] - _close[i - lookback]

            # 计算近期平均K线实体
            recent_bodies = np.abs(_close[i - lookback:i] - _open[i - lookback:i])
            avg_body = recent_bodies.mean()

            # 当前K线实体是否显著大于平均水平
            if body < avg_body * 1.5:
                continue

            if trend_change > 0 and is_bearish:
                # 上涨趋势中出现强力阴线 -> 顶部K线背离
                signals.append({
                    'type': 'kline_divergence',
                    'direction': 'top',
                    'idx': i,
                    'date': df.index[i] if hasattr(df.index, 'date') else i,
                    'price': _close[i],
                    'body_size': body,
                    'avg_body': avg_body,
                    'strength': body / avg_body,  # 背离强度
                    'description': '上涨趋势中出现强力阴K线, 趋势K线背离'
                })
            elif trend_change < 0 and is_bullish:
                # 下跌趋势中出现强力阳线 -> 底部K线背离
                signals.append({
                    'type': 'kline_divergence',
                    'direction': 'bottom',
                    'idx': i,
                    'date': df.index[i] if hasattr(df.index, 'date') else i,
                    'price': _close[i],
                    'body_size': body,
                    'avg_body': avg_body,
                    'strength': body / avg_body,
                    'description': '下跌趋势中出现强力阳K线, 趋势K线背离'
                })

        return signals

    # ============================================================
    # 2. 趋势幅度背离 (第二章第二节)
    # 后一上涨/下跌段幅度小于前一段 -> 动能衰竭
    # ============================================================
    def detect_amplitude_divergence(self, order: int = 5,
                                    shrink_ratio: float = None) -> list:
        """
        检测趋势幅度背离

        原理: 股价创新高/新低, 但该段涨跌幅度相比前段减小,
        说明前进动能在衰减, 构成幅度背离。

        参数:
            order: 摆动点识别窗口
            shrink_ratio: 幅度缩小判定比例

        返回: 背离信号列表
        """
        if shrink_ratio is None:
            shrink_ratio = cfg.AMPLITUDE_SHRINK_RATIO

        up_segs, down_segs = identify_trend_segments(self.df, order)
        signals = []

        # 检测上涨幅度顶背离: 后段幅度 < 前段幅度 * shrink_ratio
        for i in range(1, len(up_segs)):
            prev = up_segs[i - 1]
            curr = up_segs[i]

            # 确认创新高
            if curr['end_price'] > prev['end_price']:
                ratio = curr['amplitude'] / prev['amplitude'] if prev['amplitude'] > 0 else 1
                if ratio < shrink_ratio:
                    signals.append({
                        'type': 'amplitude_divergence',
                        'direction': 'top',
                        'idx': curr['end_idx'],
                        'date': self.df.index[curr['end_idx']],
                        'price': curr['end_price'],
                        'prev_amplitude': prev['amplitude'],
                        'curr_amplitude': curr['amplitude'],
                        'ratio': ratio,
                        'severity': 1 - ratio,  # 越接近1越严重
                        'description': f'上涨幅度背离: 当前段幅度仅为前段的{ratio:.1%}'
                    })

        # 检测下跌幅度底背离: 后段幅度 < 前段幅度 * shrink_ratio
        for i in range(1, len(down_segs)):
            prev = down_segs[i - 1]
            curr = down_segs[i]

            # 确认创新低
            if curr['end_price'] < prev['end_price']:
                ratio = curr['amplitude'] / prev['amplitude'] if prev['amplitude'] > 0 else 1
                if ratio < shrink_ratio:
                    signals.append({
                        'type': 'amplitude_divergence',
                        'direction': 'bottom',
                        'idx': curr['end_idx'],
                        'date': self.df.index[curr['end_idx']],
                        'price': curr['end_price'],
                        'prev_amplitude': prev['amplitude'],
                        'curr_amplitude': curr['amplitude'],
                        'ratio': ratio,
                        'severity': 1 - ratio,
                        'description': f'下跌幅度背离: 当前段幅度仅为前段的{ratio:.1%}'
                    })

        return signals

    # ============================================================
    # 3. 时间度背离 (第二章第三节)
    # 后一段持续K线数量(时间度)少于前一段
    # ============================================================
    def detect_time_divergence(self, order: int = 5,
                               shrink_ratio: float = None) -> list:
        """
        检测时间度背离

        原理: 股价创新高/新低, 但该段持续的时间(K线数)相比前段缩短,
        说明原趋势在时间上无法维持, 构成时间度背离。

        参数:
            order: 摆动点识别窗口
            shrink_ratio: 时间度缩小判定比例

        返回: 背离信号列表
        """
        if shrink_ratio is None:
            shrink_ratio = cfg.TIME_SHRINK_RATIO

        up_segs, down_segs = identify_trend_segments(self.df, order)
        signals = []

        # 上涨时间度顶背离
        for i in range(1, len(up_segs)):
            prev = up_segs[i - 1]
            curr = up_segs[i]

            if curr['end_price'] > prev['end_price']:
                ratio = curr['duration'] / prev['duration'] if prev['duration'] > 0 else 1
                if ratio < shrink_ratio:
                    signals.append({
                        'type': 'time_divergence',
                        'direction': 'top',
                        'idx': curr['end_idx'],
                        'date': self.df.index[curr['end_idx']],
                        'price': curr['end_price'],
                        'prev_duration': prev['duration'],
                        'curr_duration': curr['duration'],
                        'ratio': ratio,
                        'severity': 1 - ratio,
                        'description': f'上涨时间度背离: 当前段时间仅为前段的{ratio:.1%}'
                    })

        # 下跌时间度底背离
        for i in range(1, len(down_segs)):
            prev = down_segs[i - 1]
            curr = down_segs[i]

            if curr['end_price'] < prev['end_price']:
                ratio = curr['duration'] / prev['duration'] if prev['duration'] > 0 else 1
                if ratio < shrink_ratio:
                    signals.append({
                        'type': 'time_divergence',
                        'direction': 'bottom',
                        'idx': curr['end_idx'],
                        'date': self.df.index[curr['end_idx']],
                        'price': curr['end_price'],
                        'prev_duration': prev['duration'],
                        'curr_duration': curr['duration'],
                        'ratio': ratio,
                        'severity': 1 - ratio,
                        'description': f'下跌时间度背离: 当前段时间仅为前段的{ratio:.1%}'
                    })

        return signals

    # ============================================================
    # 4. 均线背离 (第二章第四节)
    # 向上交叉背离: 下跌趋势中股价暴涨突破长期均线(均线仍向下) -> 短线卖点
    # 向下交叉背离: 上涨趋势中股价暴跌跌破长期均线(均线仍向上) -> 短线买点
    # ============================================================
    def detect_ma_cross_divergence(self, ma_period: int = None,
                                    velocity_threshold: float = 0.03) -> list:
        """
        检测均线交叉背离

        原理: 股价方向与长周期均线方向相反时发生交叉, 由于长周期均线
        稳定性强, 容易迫使股价回头, 形成短期顶/底。
        暴涨暴跌时效果最显著。

        参数:
            ma_period: 均线周期(默认30)
            velocity_threshold: 暴涨暴跌速率阈值

        返回: 背离信号列表
        """
        if ma_period is None:
            ma_period = cfg.MA_LONG

        ma_col = f'MA{ma_period}'
        if ma_col not in self.df.columns:
            self.df[ma_col] = calc_ma(self.df['close'], ma_period)

        signals = []
        df = self.df
        _close = df['close'].values
        _ma = df[ma_col].values

        for i in range(ma_period + 5, len(df)):
            price = _close[i]
            ma_val = _ma[i]
            prev_price = _close[i - 1]
            prev_ma = _ma[i - 1]

            # 均线方向: 看最近几天的均线斜率
            ma_slope = (_ma[i] - _ma[i - 5]) / _ma[i - 5]

            # 股价穿越均线的速度(衡量暴涨/暴跌程度)
            price_velocity = (price - _close[i - 5]) / _close[i - 5]

            # 向上交叉背离: 股价从下向上穿过均线, 但均线方向向下
            if (prev_price < prev_ma and price > ma_val and
                ma_slope < -0.001 and price_velocity > velocity_threshold):
                signals.append({
                    'type': 'ma_cross_divergence',
                    'direction': 'top',  # 短线卖点
                    'cross_direction': 'upward',
                    'idx': i,
                    'date': df.index[i],
                    'price': price,
                    'ma_value': ma_val,
                    'ma_period': ma_period,
                    'ma_slope': ma_slope,
                    'price_velocity': price_velocity,
                    'is_violent': abs(price_velocity) > velocity_threshold * 2,
                    'description': f'向上交叉背离: 股价暴涨突破MA{ma_period}(均线向下), 短线卖点'
                })

            # 向下交叉背离: 股价从上向下穿过均线, 但均线方向向上
            elif (prev_price > prev_ma and price < ma_val and
                  ma_slope > 0.001 and price_velocity < -velocity_threshold):
                signals.append({
                    'type': 'ma_cross_divergence',
                    'direction': 'bottom',  # 短线买点
                    'cross_direction': 'downward',
                    'idx': i,
                    'date': df.index[i],
                    'price': price,
                    'ma_value': ma_val,
                    'ma_period': ma_period,
                    'ma_slope': ma_slope,
                    'price_velocity': price_velocity,
                    'is_violent': abs(price_velocity) > velocity_threshold * 2,
                    'description': f'向下交叉背离: 股价暴跌跌破MA{ma_period}(均线向上), 短线买点'
                })

        return signals

    # ============================================================
    # 5. 均线相交面积背离 (第二章第五节)
    # 短期均线与长期均线之间的面积逐步缩小
    # ============================================================
    def detect_ma_area_divergence(self, short_period: int = None,
                                   long_period: int = None,
                                   shrink_ratio: float = None) -> list:
        """
        检测均线相交面积背离

        原理: 短期均线与长期均线相交形成的面积逐步萎缩, 说明股价
        离开长期均线的幅度在减小, 原趋势动能在衰减。面积萎缩后
        一旦股价突破长期均线, 极易形成均线扭转。

        参数:
            short_period: 短期均线周期(默认5)
            long_period: 长期均线周期(默认30)
            shrink_ratio: 面积缩小判定比例

        返回: 背离信号列表
        """
        if short_period is None:
            short_period = cfg.MA_SHORT
        if long_period is None:
            long_period = cfg.MA_LONG
        if shrink_ratio is None:
            shrink_ratio = cfg.MA_AREA_SHRINK_RATIO

        short_col = f'MA{short_period}'
        long_col = f'MA{long_period}'

        if short_col not in self.df.columns:
            self.df[short_col] = calc_ma(self.df['close'], short_period)
        if long_col not in self.df.columns:
            self.df[long_col] = calc_ma(self.df['close'], long_period)

        # 计算短长均线差值
        diff = self.df[short_col] - self.df[long_col]
        _diff = diff.values
        _high = self.df['high'].values
        _low = self.df['low'].values
        _close = self.df['close'].values

        # 划分面积区间(按差值正负分段)
        areas = []
        start = long_period
        current_sign = 1 if _diff[start] >= 0 else -1

        for i in range(start + 1, len(diff)):
            new_sign = 1 if _diff[i] >= 0 else -1
            if new_sign != current_sign or i == len(diff) - 1:
                end = i if new_sign != current_sign else i + 1
                segment = np.abs(_diff[start:end])
                area_val = segment.sum()

                area_type = 'above' if current_sign > 0 else 'below'
                areas.append({
                    'type': area_type,
                    'start_idx': start,
                    'end_idx': end - 1,
                    'area': area_val
                })
                start = i
                current_sign = new_sign

        signals = []

        # 比较同类型面积(above与above比, below与below比)
        above_areas = [a for a in areas if a['type'] == 'above']
        below_areas = [a for a in areas if a['type'] == 'below']

        # 上方面积缩小 -> 顶背离
        for i in range(1, len(above_areas)):
            prev = above_areas[i - 1]
            curr = above_areas[i]
            ratio = curr['area'] / prev['area'] if prev['area'] > 0 else 1

            if ratio < shrink_ratio:
                # 确认股价创新高
                prev_high = np.max(_high[prev['start_idx']:prev['end_idx'] + 1])
                curr_high = np.max(_high[curr['start_idx']:curr['end_idx'] + 1])

                if curr_high >= prev_high * 0.98:  # 允许一定容差
                    signals.append({
                        'type': 'ma_area_divergence',
                        'direction': 'top',
                        'idx': curr['end_idx'],
                        'date': self.df.index[curr['end_idx']],
                        'price': _close[curr['end_idx']],
                        'prev_area': prev['area'],
                        'curr_area': curr['area'],
                        'ratio': ratio,
                        'severity': 1 - ratio,
                        'ma_short': short_period,
                        'ma_long': long_period,
                        'description': f'均线相交面积顶背离: MA{short_period}/MA{long_period}面积缩至{ratio:.1%}'
                    })

        # 下方面积缩小 -> 底背离
        for i in range(1, len(below_areas)):
            prev = below_areas[i - 1]
            curr = below_areas[i]
            ratio = curr['area'] / prev['area'] if prev['area'] > 0 else 1

            if ratio < shrink_ratio:
                prev_low = np.min(_low[prev['start_idx']:prev['end_idx'] + 1])
                curr_low = np.min(_low[curr['start_idx']:curr['end_idx'] + 1])

                if curr_low <= prev_low * 1.02:
                    signals.append({
                        'type': 'ma_area_divergence',
                        'direction': 'bottom',
                        'idx': curr['end_idx'],
                        'date': self.df.index[curr['end_idx']],
                        'price': _close[curr['end_idx']],
                        'prev_area': prev['area'],
                        'curr_area': curr['area'],
                        'ratio': ratio,
                        'severity': 1 - ratio,
                        'ma_short': short_period,
                        'ma_long': long_period,
                        'description': f'均线相交面积底背离: MA{short_period}/MA{long_period}面积缩至{ratio:.1%}'
                    })

        return signals

    # ============================================================
    # 综合形态背离分析 (第二章第六节)
    # ============================================================
    def analyze_all(self, order: int = 5) -> dict:
        """
        执行所有几何形态背离分析

        书中原则:
        (1) 多种方法得出结论一致 -> 走势以背离对待
        (2) 存在分歧时: 上涨中卖出->按背离处理; 下跌中买入->暂不按背离处理
        (3) 背离越严重越应警觉

        返回: 各类背离信号的汇总字典
        """
        results = {
            'kline_divergence': self.detect_kline_divergence(),
            'amplitude_divergence': self.detect_amplitude_divergence(order),
            'time_divergence': self.detect_time_divergence(order),
            'ma_cross_divergence': self.detect_ma_cross_divergence(),
            'ma_area_divergence': self.detect_ma_area_divergence(),
        }

        # 统计各时间点的背离信号重合度
        all_signals = []
        for sig_list in results.values():
            all_signals.extend(sig_list)

        # 按位置聚合背离信号
        from collections import defaultdict
        position_signals = defaultdict(list)
        for sig in all_signals:
            # 使用5个K线窗口聚合信号
            bucket = sig['idx'] // 5 * 5
            position_signals[bucket].append(sig)

        # 找出多种背离共振的点
        confluence_signals = []
        for bucket, sigs in position_signals.items():
            top_types = set(s['type'] for s in sigs if s['direction'] == 'top')
            bottom_types = set(s['type'] for s in sigs if s['direction'] == 'bottom')

            if len(top_types) >= 2:
                confluence_signals.append({
                    'direction': 'top',
                    'idx': sigs[0]['idx'],
                    'date': sigs[0]['date'],
                    'price': sigs[0]['price'],
                    'signal_count': len(top_types),
                    'signal_types': list(top_types),
                    'signals': [s for s in sigs if s['direction'] == 'top'],
                    'description': f'多重形态顶背离共振({len(top_types)}种): {", ".join(top_types)}'
                })

            if len(bottom_types) >= 2:
                confluence_signals.append({
                    'direction': 'bottom',
                    'idx': sigs[0]['idx'],
                    'date': sigs[0]['date'],
                    'price': sigs[0]['price'],
                    'signal_count': len(bottom_types),
                    'signal_types': list(bottom_types),
                    'signals': [s for s in sigs if s['direction'] == 'bottom'],
                    'description': f'多重形态底背离共振({len(bottom_types)}种): {", ".join(bottom_types)}'
                })

        results['confluence'] = confluence_signals
        return results
