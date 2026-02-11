"""
第五章第四节: 量价背离

核心要点:
- 正常量价关系: 价升量增, 价跌量缩
- 量价背离: 价升量减(上涨末期, 反转前兆) / 价跌量增(各阶段含义不同)
- 上涨末期价升量缩 -> 强烈见顶信号
- 上涨末期价跌量增(放量滞涨) -> 确认反转到来
- 下跌中价跌量增 -> 分歧加大, 不一定见底, 多为反弹
- 下跌末期: 地量地价 -> 底部信号, 之后放量上涨才是启动
"""

import pandas as pd
import numpy as np
from indicators import find_swing_highs, find_swing_lows, identify_trend_segments


class VolumePriceDivergenceAnalyzer:
    """量价背离分析器"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def detect_price_up_volume_down(self, order: int = 5) -> list:
        """
        检测价升量减 (上涨趋势末期最有参考价值)

        原理: 上涨需要资金推动, 创新高时成交量不增反减,
        说明市场犹豫观望, 一旦卖空成为共识, 暴跌随时出现。
        """
        signals = []
        df = self.df
        up_segs, _ = identify_trend_segments(df, order)

        for i in range(1, len(up_segs)):
            prev = up_segs[i - 1]
            curr = up_segs[i]

            if curr['end_price'] <= prev['end_price']:
                continue

            # 比较两段的总成交量
            prev_vol = df['volume'].iloc[prev['start_idx']:prev['end_idx'] + 1].sum()
            curr_vol = df['volume'].iloc[curr['start_idx']:curr['end_idx'] + 1].sum()

            # 标准化: 按K线数量平均
            prev_avg_vol = prev_vol / max(prev['duration'], 1)
            curr_avg_vol = curr_vol / max(curr['duration'], 1)

            if curr_avg_vol < prev_avg_vol * 0.8:
                ratio = curr_avg_vol / prev_avg_vol if prev_avg_vol > 0 else 1
                signals.append({
                    'type': 'volume_price_divergence',
                    'subtype': 'price_up_volume_down',
                    'direction': 'top',
                    'idx': curr['end_idx'],
                    'date': df.index[curr['end_idx']],
                    'price': curr['end_price'],
                    'prev_avg_volume': prev_avg_vol,
                    'curr_avg_volume': curr_avg_vol,
                    'volume_ratio': ratio,
                    'severity': 1 - ratio,
                    'description': f'价升量减: 股价新高{curr["end_price"]:.2f}, 量能萎缩至{ratio:.1%}'
                })

        return signals

    def detect_price_down_volume_up(self, order: int = 5) -> list:
        """
        检测价跌量增

        上涨末期出现: 确认见顶信号(特别是先有价升量减后又价跌量增)
        下跌中出现: 仅为短暂反弹信号, 不代表见底
        """
        signals = []
        df = self.df
        _, down_segs = identify_trend_segments(df, order)

        for i in range(1, len(down_segs)):
            prev = down_segs[i - 1]
            curr = down_segs[i]

            prev_vol = df['volume'].iloc[prev['start_idx']:prev['end_idx'] + 1].sum()
            curr_vol = df['volume'].iloc[curr['start_idx']:curr['end_idx'] + 1].sum()

            prev_avg_vol = prev_vol / max(prev['duration'], 1)
            curr_avg_vol = curr_vol / max(curr['duration'], 1)

            if curr_avg_vol > prev_avg_vol * 1.3:
                ratio = curr_avg_vol / prev_avg_vol if prev_avg_vol > 0 else 1
                signals.append({
                    'type': 'volume_price_divergence',
                    'subtype': 'price_down_volume_up',
                    'direction': 'bottom',
                    'idx': curr['end_idx'],
                    'date': df.index[curr['end_idx']],
                    'price': curr['end_price'],
                    'prev_avg_volume': prev_avg_vol,
                    'curr_avg_volume': curr_avg_vol,
                    'volume_ratio': ratio,
                    'severity': ratio - 1,
                    'description': f'价跌量增: 股价{curr["end_price"]:.2f}, 量能放大至{ratio:.1%}(可能仅是反弹)'
                })

        return signals

    def detect_ground_volume(self, lookback: int = 20) -> list:
        """
        检测地量地价 (下跌末期底部信号)

        地量地价是大底先决条件, 缩量期越长越好, 期间跌幅极小。
        之后突然放量即是启动上涨的迹象。
        """
        signals = []
        df = self.df

        # 计算成交量的滚动最小值和平均值
        vol_min = df['volume'].rolling(window=lookback * 3, min_periods=lookback).min()
        vol_mean = df['volume'].rolling(window=lookback * 3, min_periods=lookback).mean()

        for i in range(lookback * 3, len(df)):
            # 当前量是否接近历史地量
            recent_vol = df['volume'].iloc[i - lookback:i]
            recent_avg = recent_vol.mean()
            overall_avg = df['volume'].iloc[:i].mean()

            if recent_avg < overall_avg * 0.3:
                # 检查近期价格变化幅度(要求极小)
                recent_price = df['close'].iloc[i - lookback:i]
                price_range = (recent_price.max() - recent_price.min()) / recent_price.mean()

                if price_range < 0.1:  # 价格波动小于10%
                    # 检查是否有放量启动迹象
                    if i < len(df) - 1:
                        next_vol = df['volume'].iloc[i]
                        if next_vol > recent_avg * 2:
                            signals.append({
                                'type': 'volume_price_divergence',
                                'subtype': 'ground_volume_breakout',
                                'direction': 'bottom',
                                'idx': i,
                                'date': df.index[i],
                                'price': df['close'].iloc[i],
                                'recent_avg_volume': recent_avg,
                                'breakout_volume': next_vol,
                                'volume_ratio': next_vol / recent_avg,
                                'description': f'地量后放量启动: 成交量暴增{next_vol / recent_avg:.1f}倍'
                            })

        return signals

    def analyze_all(self) -> dict:
        return {
            'price_up_volume_down': self.detect_price_up_volume_down(),
            'price_down_volume_up': self.detect_price_down_volume_up(),
            'ground_volume': self.detect_ground_volume(),
        }
