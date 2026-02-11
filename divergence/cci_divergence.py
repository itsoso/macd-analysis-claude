"""
第五章第二节: CCI指标背离

核心要点:
- CCI是顺势指标, 与KDJ/RSI超买超卖含义相反
- +100(春天线)以上为超买区 -> 买方力量加强, 考虑买入(顺势)
- -100(秋天线)以下为超卖区 -> 卖方力量加强, 考虑卖出(顺势)
- 突破+100春天线跟进, 跌破-100秋天线出来
- 下跌趋势中: 只有底背离后突破+100春天线才值得买入
- 上涨趋势中: 只有顶背离后跌破-100秋天线才必须卖出
- CCI顶背离: 股价创新高, CCI在+100以上不创新高
- CCI底背离: 股价创新低, CCI在-100以下不创新低
- 底背离背离了又背离概率更大, 买入需慎之又慎
"""

import pandas as pd
import numpy as np
import config as cfg
from indicators import calc_cci, find_swing_highs, find_swing_lows


class CCIDivergenceAnalyzer:
    """CCI指标背离分析器"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._ensure_cci()

    def _ensure_cci(self):
        if 'CCI' not in self.df.columns:
            self.df['CCI'] = calc_cci(self.df['high'], self.df['low'], self.df['close'])

    def detect_top_divergence(self, order: int = 5) -> list:
        """
        CCI顶背离: 股价创新高, CCI在+100以上却不创新高
        必须在超买区域(CCI > +100)
        """
        signals = []
        df = self.df

        swing_highs = find_swing_highs(df['high'], order)
        high_indices = [df.index.get_loc(i) for i in swing_highs[swing_highs].index]

        for i in range(1, len(high_indices)):
            idx_prev = high_indices[i - 1]
            idx_curr = high_indices[i]

            price_prev = df['high'].iloc[idx_prev]
            price_curr = df['high'].iloc[idx_curr]

            window = 3
            cci_prev = df['CCI'].iloc[max(0, idx_prev - window):idx_prev + window + 1].max()
            cci_curr = df['CCI'].iloc[max(0, idx_curr - window):idx_curr + window + 1].max()

            # CCI顶背离必须在超买区
            if (price_curr > price_prev and cci_curr < cci_prev and
                    cci_prev > cfg.CCI_SPRING):
                severity = (cci_prev - cci_curr) / abs(cci_prev) if cci_prev != 0 else 0
                signals.append({
                    'type': 'cci_divergence',
                    'direction': 'top',
                    'idx': idx_curr,
                    'date': df.index[idx_curr],
                    'price': price_curr,
                    'cci_prev': cci_prev,
                    'cci_curr': cci_curr,
                    'severity': abs(severity),
                    'description': f'CCI顶背离: 股价新高{price_curr:.2f}, CCI下降{cci_prev:.0f}->{cci_curr:.0f}'
                })

        return signals

    def detect_bottom_divergence(self, order: int = 5) -> list:
        """
        CCI底背离: 股价创新低, CCI在-100以下不创新低
        必须在超卖区域(CCI < -100)
        注意: 底背离背离了又背离概率大, 二次以上底背离更可靠
        """
        signals = []
        df = self.df

        swing_lows = find_swing_lows(df['low'], order)
        low_indices = [df.index.get_loc(i) for i in swing_lows[swing_lows].index]

        divergence_count = 0
        for i in range(1, len(low_indices)):
            idx_prev = low_indices[i - 1]
            idx_curr = low_indices[i]

            price_prev = df['low'].iloc[idx_prev]
            price_curr = df['low'].iloc[idx_curr]

            window = 3
            cci_prev = df['CCI'].iloc[max(0, idx_prev - window):idx_prev + window + 1].min()
            cci_curr = df['CCI'].iloc[max(0, idx_curr - window):idx_curr + window + 1].min()

            # CCI底背离必须在超卖区
            if (price_curr < price_prev and cci_curr > cci_prev and
                    cci_prev < cfg.CCI_AUTUMN):
                divergence_count += 1
                severity = (cci_curr - cci_prev) / abs(cci_prev) if cci_prev != 0 else 0
                signals.append({
                    'type': 'cci_divergence',
                    'direction': 'bottom',
                    'idx': idx_curr,
                    'date': df.index[idx_curr],
                    'price': price_curr,
                    'cci_prev': cci_prev,
                    'cci_curr': cci_curr,
                    'divergence_count': divergence_count,
                    'severity': abs(severity),
                    'is_second_or_more': divergence_count >= 2,
                    'description': f'CCI底背离(第{divergence_count}次): 股价新低{price_curr:.2f}, '
                                   f'CCI上升{cci_prev:.0f}->{cci_curr:.0f}'
                })

        return signals

    def detect_spring_autumn_crosses(self) -> list:
        """检测CCI突破春天线/跌破秋天线"""
        signals = []
        df = self.df
        cci = df['CCI']

        for i in range(1, len(cci)):
            # 突破+100春天线
            if cci.iloc[i - 1] <= cfg.CCI_SPRING and cci.iloc[i] > cfg.CCI_SPRING:
                signals.append({
                    'type': 'cci_spring_cross',
                    'direction': 'up',
                    'idx': i,
                    'date': df.index[i],
                    'price': df['close'].iloc[i],
                    'cci': cci.iloc[i],
                    'description': 'CCI突破+100春天线(顺势买入信号)'
                })
            # 跌破-100秋天线
            elif cci.iloc[i - 1] >= cfg.CCI_AUTUMN and cci.iloc[i] < cfg.CCI_AUTUMN:
                signals.append({
                    'type': 'cci_autumn_cross',
                    'direction': 'down',
                    'idx': i,
                    'date': df.index[i],
                    'price': df['close'].iloc[i],
                    'cci': cci.iloc[i],
                    'description': 'CCI跌破-100秋天线(顺势卖出信号)'
                })

        return signals

    def analyze_all(self) -> dict:
        return {
            'top_divergence': self.detect_top_divergence(),
            'bottom_divergence': self.detect_bottom_divergence(),
            'spring_autumn_crosses': self.detect_spring_autumn_crosses(),
        }
