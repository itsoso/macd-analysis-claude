"""
第三章: MACD技术指标背离
包含4种MACD背离形式:
  1. 黄白线背离     - 股价创新高/低, DIF/DEA不同步
  2. 彩柱线长度背离 - 红/绿柱长度缩短 (当堆/邻堆/隔堆)
  3. 彩柱线面积背离 - 红/绿柱堆面积缩小 (邻堆/隔堆)
  4. 黄白线相交面积背离 - DIF与DEA围合的面积缩小
以及: MACD金叉/死叉分析
"""

import numpy as np
import pandas as pd
import config as cfg
from indicators import (
    calc_macd, find_swing_highs, find_swing_lows,
    identify_bar_groups, find_macd_crosses, identify_trend_segments
)


class MACDDivergenceAnalyzer:
    """MACD技术指标背离分析器"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._ensure_macd()
        self.bar_groups = identify_bar_groups(self.df['MACD_BAR'])
        self.crosses = find_macd_crosses(self.df['DIF'], self.df['DEA'])

    def _ensure_macd(self):
        if 'DIF' not in self.df.columns:
            macd_df = calc_macd(self.df['close'])
            self.df['DIF'] = macd_df['DIF']
            self.df['DEA'] = macd_df['DEA']
            self.df['MACD_BAR'] = macd_df['MACD_BAR']

    # ============================================================
    # MACD金叉/死叉分析 (第三章第一节)
    # ============================================================
    def analyze_crosses(self) -> dict:
        """
        分析MACD金叉死叉

        核心原则:
        - 下跌趋势中只有末期最后一个金叉(底背离后)才有买入价值
        - 上涨趋势中只有末期最后一个死叉(顶背离后)才有卖出价值
        - 上涨趋势的金叉大都可以买入, 越靠前越安全
        - 下跌趋势的死叉都是逃命点
        """
        golden_crosses = [c for c in self.crosses if c['type'] == 'golden']
        death_crosses = [c for c in self.crosses if c['type'] == 'death']

        return {
            'golden_crosses': golden_crosses,
            'death_crosses': death_crosses,
            'all_crosses': self.crosses
        }

    # ============================================================
    # 1. 黄白线背离 (第三章第二节)
    # 股价新高/新低, DIF/DEA不同步
    # ============================================================
    def detect_dif_dea_divergence(self, order: int = 5) -> list:
        """
        检测MACD黄白线背离

        原理: 股价创新高时MACD的DIF/DEA不创新高(顶背离);
        股价创新低时MACD的DIF/DEA不创新低(底背离)。
        背离形成的反差越大, 背离越严重, 反转可能性越大。

        买入宜迟(要求背离严重), 卖出要早(不必苛求非常严重)
        """
        signals = []
        df = self.df

        # 找股价的摆动高低点
        swing_highs = find_swing_highs(df['high'], order)
        swing_lows = find_swing_lows(df['low'], order)

        high_indices = [df.index.get_loc(i) for i in swing_highs[swing_highs].index]
        low_indices = [df.index.get_loc(i) for i in swing_lows[swing_lows].index]

        # 顶背离: 股价创新高, DIF不创新高
        for i in range(1, len(high_indices)):
            idx_prev = high_indices[i - 1]
            idx_curr = high_indices[i]

            price_prev = df['high'].iloc[idx_prev]
            price_curr = df['high'].iloc[idx_curr]
            dif_prev = df['DIF'].iloc[idx_prev]
            dif_curr = df['DIF'].iloc[idx_curr]

            if price_curr > price_prev and dif_curr < dif_prev:
                severity = (dif_prev - dif_curr) / abs(dif_prev) if dif_prev != 0 else 0
                signals.append({
                    'type': 'dif_dea_divergence',
                    'direction': 'top',
                    'idx': idx_curr,
                    'date': df.index[idx_curr],
                    'price': price_curr,
                    'prev_price': price_prev,
                    'prev_idx': idx_prev,
                    'dif_curr': dif_curr,
                    'dif_prev': dif_prev,
                    'severity': abs(severity),
                    'description': f'MACD黄白线顶背离: 股价创新高{price_curr:.2f}, DIF反降'
                })

        # 底背离: 股价创新低, DIF不创新低
        for i in range(1, len(low_indices)):
            idx_prev = low_indices[i - 1]
            idx_curr = low_indices[i]

            price_prev = df['low'].iloc[idx_prev]
            price_curr = df['low'].iloc[idx_curr]
            dif_prev = df['DIF'].iloc[idx_prev]
            dif_curr = df['DIF'].iloc[idx_curr]

            if price_curr < price_prev and dif_curr > dif_prev:
                severity = (dif_curr - dif_prev) / abs(dif_prev) if dif_prev != 0 else 0
                signals.append({
                    'type': 'dif_dea_divergence',
                    'direction': 'bottom',
                    'idx': idx_curr,
                    'date': df.index[idx_curr],
                    'price': price_curr,
                    'prev_price': price_prev,
                    'prev_idx': idx_prev,
                    'dif_curr': dif_curr,
                    'dif_prev': dif_prev,
                    'severity': abs(severity),
                    'description': f'MACD黄白线底背离: 股价创新低{price_curr:.2f}, DIF反升'
                })

        return signals

    # ============================================================
    # 2. 彩柱线长度背离 (第三章第三节)
    # 三种类型: 当堆/邻堆/隔堆
    # 隔堆 > 邻堆 > 当堆 (背离强度)
    # ============================================================
    def detect_bar_length_divergence(self, order: int = 5) -> list:
        """
        检测MACD彩柱线长度背离

        三种背离:
        - 当堆背离: 同一堆内前后K线, 后者创新高/低但柱线缩短
          (适合周线/月线级别, 日线以下可忽略)
        - 邻堆背离: 相邻同色柱堆间, 后堆最长柱<前堆 (中间无/极小反色柱)
        - 隔堆背离: 两同色柱堆间夹有明显反色柱堆, 后堆柱线缩短
          (最严重, 需高度重视)

        重要: 隔堆背离中嵌套的邻堆/当堆背离应区别对待
        """
        signals = []
        df = self.df
        groups = self.bar_groups

        # --- 当堆背离 ---
        signals.extend(self._detect_same_group_divergence(df, order))

        # --- 邻堆背离 & 隔堆背离 ---
        red_groups = [g for g in groups if g['type'] == 'red']
        green_groups = [g for g in groups if g['type'] == 'green']

        # 红柱堆: 检测顶背离
        signals.extend(self._detect_group_length_divergence(
            df, red_groups, groups, 'top'))

        # 绿柱堆: 检测底背离
        signals.extend(self._detect_group_length_divergence(
            df, green_groups, groups, 'bottom'))

        return signals

    def _detect_same_group_divergence(self, df, order):
        """检测当堆背离: 同一柱堆内后续K线创新高/低但柱线缩短"""
        signals = []
        bar = df['MACD_BAR']

        for group in self.bar_groups:
            start = group['start_idx']
            end = group['end_idx']

            if end - start < 2:
                continue

            for i in range(start + 1, end + 1):
                if i >= len(df):
                    break

                abs_curr = abs(bar.iloc[i])
                abs_prev = abs(bar.iloc[i - 1])

                if group['type'] == 'red':
                    # 股价创新高但红柱缩短
                    if (df['high'].iloc[i] > df['high'].iloc[i - 1] and
                            abs_curr < abs_prev * 0.9):
                        signals.append({
                            'type': 'bar_length_divergence',
                            'subtype': 'same_group',  # 当堆
                            'direction': 'top',
                            'idx': i,
                            'date': df.index[i],
                            'price': df['close'].iloc[i],
                            'bar_curr': abs_curr,
                            'bar_prev': abs_prev,
                            'severity': 1 - abs_curr / abs_prev if abs_prev > 0 else 0,
                            'description': '彩柱线当堆顶背离(适合周/月线级别)'
                        })
                elif group['type'] == 'green':
                    # 股价创新低但绿柱缩短
                    if (df['low'].iloc[i] < df['low'].iloc[i - 1] and
                            abs_curr < abs_prev * 0.9):
                        signals.append({
                            'type': 'bar_length_divergence',
                            'subtype': 'same_group',
                            'direction': 'bottom',
                            'idx': i,
                            'date': df.index[i],
                            'price': df['close'].iloc[i],
                            'bar_curr': abs_curr,
                            'bar_prev': abs_prev,
                            'severity': 1 - abs_curr / abs_prev if abs_prev > 0 else 0,
                            'description': '彩柱线当堆底背离(适合周/月线级别)'
                        })

        return signals

    def _detect_group_length_divergence(self, df, same_color_groups, all_groups, direction):
        """检测邻堆/隔堆背离"""
        signals = []

        for i in range(1, len(same_color_groups)):
            prev_g = same_color_groups[i - 1]
            curr_g = same_color_groups[i]

            # 判断是邻堆还是隔堆: 中间是否有明显的反色柱堆
            between_groups = [g for g in all_groups
                              if g['start_idx'] > prev_g['end_idx']
                              and g['end_idx'] < curr_g['start_idx']
                              and g['type'] != curr_g['type']]

            has_significant_opposite = any(
                g['area'] > min(prev_g['area'], curr_g['area']) * 0.1
                for g in between_groups
            )

            subtype = 'separated' if has_significant_opposite else 'adjacent'

            # 比较柱线最大长度
            ratio = curr_g['max_length'] / prev_g['max_length'] if prev_g['max_length'] > 0 else 1

            if ratio >= cfg.BAR_LENGTH_SHRINK_RATIO:
                continue

            # 确认股价创新高/新低
            if direction == 'top':
                prev_high = df['high'].iloc[prev_g['start_idx']:prev_g['end_idx'] + 1].max()
                curr_high = df['high'].iloc[curr_g['start_idx']:curr_g['end_idx'] + 1].max()
                if curr_high <= prev_high:
                    continue
                price = curr_high
            else:
                prev_low = df['low'].iloc[prev_g['start_idx']:prev_g['end_idx'] + 1].min()
                curr_low = df['low'].iloc[curr_g['start_idx']:curr_g['end_idx'] + 1].min()
                if curr_low >= prev_low:
                    continue
                price = curr_low

            subtype_cn = '隔堆' if subtype == 'separated' else '邻堆'
            dir_cn = '顶' if direction == 'top' else '底'

            signals.append({
                'type': 'bar_length_divergence',
                'subtype': subtype,
                'direction': direction,
                'idx': curr_g['max_bar_idx'],
                'date': df.index[curr_g['max_bar_idx']],
                'price': price,
                'prev_max_length': prev_g['max_length'],
                'curr_max_length': curr_g['max_length'],
                'ratio': ratio,
                'severity': 1 - ratio,
                'prev_group': prev_g,
                'curr_group': curr_g,
                'description': f'彩柱线{subtype_cn}{dir_cn}背离: 柱线长度缩至{ratio:.1%}'
            })

        return signals

    # ============================================================
    # 3. 彩柱线面积背离 (第三章第四节)
    # 两种类型: 邻堆面积背离 / 隔堆面积背离
    # ============================================================
    def detect_bar_area_divergence(self) -> list:
        """
        检测MACD彩柱线面积背离

        原理: 对应的彩柱堆面积缩小, 说明DIF与DEA的差值在减小,
        趋势动能在衰歇。面积对比比长度对比能更清楚地看出能量萎缩。

        - 邻堆面积背离: 同一趋势段内 (较弱)
        - 隔堆面积背离: 前后两段趋势间 (较强, 需高度重视)
        - 趋势第三段后的隔堆面积背离尤其重要
        - 二次以上隔堆面积背离时存在绝佳操作机会
        """
        signals = []
        df = self.df
        groups = self.bar_groups

        red_groups = [g for g in groups if g['type'] == 'red']
        green_groups = [g for g in groups if g['type'] == 'green']

        # 红柱面积: 顶背离
        signals.extend(self._detect_group_area_divergence(
            df, red_groups, groups, 'top'))

        # 绿柱面积: 底背离
        signals.extend(self._detect_group_area_divergence(
            df, green_groups, groups, 'bottom'))

        return signals

    def _detect_group_area_divergence(self, df, same_color_groups, all_groups, direction):
        """检测邻堆/隔堆面积背离"""
        signals = []

        for i in range(1, len(same_color_groups)):
            prev_g = same_color_groups[i - 1]
            curr_g = same_color_groups[i]

            # 判断邻堆/隔堆
            between_groups = [g for g in all_groups
                              if g['start_idx'] > prev_g['end_idx']
                              and g['end_idx'] < curr_g['start_idx']
                              and g['type'] != curr_g['type']]

            has_significant_opposite = any(
                g['area'] > min(prev_g['area'], curr_g['area']) * 0.1
                for g in between_groups
            )

            subtype = 'separated' if has_significant_opposite else 'adjacent'

            # 比较面积
            ratio = curr_g['area'] / prev_g['area'] if prev_g['area'] > 0 else 1

            if ratio >= cfg.BAR_AREA_SHRINK_RATIO:
                continue

            # 确认股价创新高/新低
            if direction == 'top':
                prev_high = df['high'].iloc[prev_g['start_idx']:prev_g['end_idx'] + 1].max()
                curr_high = df['high'].iloc[curr_g['start_idx']:curr_g['end_idx'] + 1].max()
                if curr_high <= prev_high:
                    continue
                price = curr_high
            else:
                prev_low = df['low'].iloc[prev_g['start_idx']:prev_g['end_idx'] + 1].min()
                curr_low = df['low'].iloc[curr_g['start_idx']:curr_g['end_idx'] + 1].min()
                if curr_low >= prev_low:
                    continue
                price = curr_low

            subtype_cn = '隔堆' if subtype == 'separated' else '邻堆'
            dir_cn = '顶' if direction == 'top' else '底'

            signals.append({
                'type': 'bar_area_divergence',
                'subtype': subtype,
                'direction': direction,
                'idx': curr_g['end_idx'],
                'date': df.index[curr_g['end_idx']],
                'price': price,
                'prev_area': prev_g['area'],
                'curr_area': curr_g['area'],
                'ratio': ratio,
                'severity': 1 - ratio,
                'prev_group': prev_g,
                'curr_group': curr_g,
                'description': f'彩柱线{subtype_cn}面积{dir_cn}背离: 面积缩至{ratio:.1%}'
            })

        return signals

    # ============================================================
    # 4. 黄白线相交面积背离 (第三章第五节)
    # DIF与DEA围合的面积缩小
    # ============================================================
    def detect_dif_dea_area_divergence(self) -> list:
        """
        检测MACD黄白线相交面积背离

        原理: DIF与DEA围合的面积萎缩, 表明趋势动能衰竭。
        原理与彩柱线面积背离类似, 但在面积比较上不如彩柱线清晰,
        主要作为辅助印证手段。
        """
        signals = []
        df = self.df

        # DIF与DEA的差值就是BAR/2, 面积本质上等同于彩柱面积
        # 但这里单独从黄白线围合角度实现
        diff = df['DIF'] - df['DEA']

        # 分段计算面积
        areas = []
        start = 0
        if len(diff) == 0:
            return signals

        current_sign = 1 if diff.iloc[0] >= 0 else -1

        for i in range(1, len(diff)):
            new_sign = 1 if diff.iloc[i] >= 0 else -1
            if new_sign != current_sign or i == len(diff) - 1:
                end = i if new_sign != current_sign else i + 1
                segment = diff.iloc[start:end].abs()
                area_val = segment.sum()

                areas.append({
                    'type': 'above' if current_sign > 0 else 'below',
                    'start_idx': start,
                    'end_idx': min(end - 1, len(df) - 1),
                    'area': area_val
                })
                start = i
                current_sign = new_sign

        # 同类型面积比较
        above_areas = [a for a in areas if a['type'] == 'above']
        below_areas = [a for a in areas if a['type'] == 'below']

        for area_list, direction in [(above_areas, 'top'), (below_areas, 'bottom')]:
            for i in range(1, len(area_list)):
                prev = area_list[i - 1]
                curr = area_list[i]
                ratio = curr['area'] / prev['area'] if prev['area'] > 0 else 1

                if ratio >= 0.7:
                    continue

                end_idx = min(curr['end_idx'], len(df) - 1)

                # 确认股价条件
                if direction == 'top':
                    prev_high = df['high'].iloc[prev['start_idx']:prev['end_idx'] + 1].max()
                    curr_high = df['high'].iloc[curr['start_idx']:end_idx + 1].max()
                    if curr_high <= prev_high:
                        continue
                else:
                    prev_low = df['low'].iloc[prev['start_idx']:prev['end_idx'] + 1].min()
                    curr_low = df['low'].iloc[curr['start_idx']:end_idx + 1].min()
                    if curr_low >= prev_low:
                        continue

                dir_cn = '顶' if direction == 'top' else '底'
                signals.append({
                    'type': 'dif_dea_area_divergence',
                    'direction': direction,
                    'idx': end_idx,
                    'date': df.index[end_idx],
                    'price': df['close'].iloc[end_idx],
                    'prev_area': prev['area'],
                    'curr_area': curr['area'],
                    'ratio': ratio,
                    'severity': 1 - ratio,
                    'description': f'黄白线相交面积{dir_cn}背离: 面积缩至{ratio:.1%}'
                })

        return signals

    # ============================================================
    # 隔堆背离计数 (第四章需要用到)
    # ============================================================
    def count_separated_divergences(self, direction: str = 'top') -> int:
        """统计隔堆背离次数(背驰判断需要)"""
        length_divs = [s for s in self.detect_bar_length_divergence()
                       if s.get('subtype') == 'separated' and s['direction'] == direction]
        area_divs = [s for s in self.detect_bar_area_divergence()
                     if s.get('subtype') == 'separated' and s['direction'] == direction]
        return max(len(length_divs), len(area_divs))

    # ============================================================
    # MACD黄白线是否曾返回零轴 (第四章背驰条件)
    # ============================================================
    def has_dif_returned_to_zero(self, start_idx: int, end_idx: int) -> bool:
        """检查DIF在指定区间内是否有返回零轴附近的过程"""
        dif_segment = self.df['DIF'].iloc[start_idx:end_idx + 1]
        # 查看是否有穿越零轴或接近零轴(绝对值很小)
        zero_threshold = dif_segment.abs().max() * 0.1
        return (dif_segment.abs() < zero_threshold).any() or \
               (dif_segment.iloc[:-1] * dif_segment.iloc[1:].values < 0).any()

    # ============================================================
    # 综合MACD背离分析 (第三章第六节)
    # ============================================================
    def analyze_all(self) -> dict:
        """
        执行所有MACD背离分析

        书中原则:
        - 用某种方法背离而另一种不背离时:
          上涨卖出->以背离对待; 下跌买入->暂不作为背离
        - 技术指标背离应与形态背离结合使用
        - 下跌趋势中等到各种底背离都明显时再介入
        - 上涨趋势中出现某方面严重顶背离就要立即离开
        """
        results = {
            'crosses': self.analyze_crosses(),
            'dif_dea_divergence': self.detect_dif_dea_divergence(),
            'bar_length_divergence': self.detect_bar_length_divergence(),
            'bar_area_divergence': self.detect_bar_area_divergence(),
            'dif_dea_area_divergence': self.detect_dif_dea_area_divergence(),
            'bar_groups': self.bar_groups,
        }

        # 统计多重MACD背离共振
        all_signals = []
        for key in ['dif_dea_divergence', 'bar_length_divergence',
                     'bar_area_divergence', 'dif_dea_area_divergence']:
            all_signals.extend(results[key])

        from collections import defaultdict
        position_signals = defaultdict(list)
        for sig in all_signals:
            bucket = sig['idx'] // 5 * 5
            position_signals[bucket].append(sig)

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
                    'description': f'多重MACD顶背离共振({len(top_types)}种)'
                })

            if len(bottom_types) >= 2:
                confluence_signals.append({
                    'direction': 'bottom',
                    'idx': sigs[0]['idx'],
                    'date': sigs[0]['date'],
                    'price': sigs[0]['price'],
                    'signal_count': len(bottom_types),
                    'signal_types': list(bottom_types),
                    'description': f'多重MACD底背离共振({len(bottom_types)}种)'
                })

        results['confluence'] = confluence_signals
        return results
