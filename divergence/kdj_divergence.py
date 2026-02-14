"""
第五章第一节: KDJ指标背离

核心要点:
- KDJ金叉: KD值在20以下K上穿D -> 买入信号
- KDJ死叉: KD值在80以上K下穿D -> 卖出信号
- 上涨趋势中几乎所有金叉都是买入点
- 下跌趋势中只有底背离后的金叉才有买入价值
- 下跌趋势中几乎所有死叉都是卖出点
- 上涨趋势中只有顶背离后的死叉才有卖出价值
- KDJ顶背离: 股价创新高, KD值在80以上不创新高
- KDJ底背离: 股价创新低, KD值在20以下不创新低
"""

import pandas as pd
import numpy as np
import config as cfg
from indicators import calc_kdj, find_kdj_crosses, find_swing_highs, find_swing_lows


class KDJDivergenceAnalyzer:
    """KDJ指标背离分析器"""

    def __init__(self, df: pd.DataFrame, _skip_copy: bool = False):
        self.df = df if _skip_copy else df.copy()
        self._ensure_kdj()

    def _ensure_kdj(self):
        if 'K' not in self.df.columns:
            kdj = calc_kdj(self.df['high'], self.df['low'], self.df['close'])
            self.df['K'] = kdj['K']
            self.df['D'] = kdj['D']
            self.df['J'] = kdj['J']

    def detect_top_divergence(self, order: int = 5) -> list:
        """
        检测KDJ顶背离
        股价创新高, 但KD值在80附近不创新高
        等同于比较前后死叉发生位置的高低
        """
        signals = []
        df = self.df
        _high = df['high'].values
        _k = df['K'].values

        swing_highs = find_swing_highs(df['high'], order)
        high_indices = [df.index.get_loc(i) for i in swing_highs[swing_highs].index]

        for i in range(1, len(high_indices)):
            idx_prev = high_indices[i - 1]
            idx_curr = high_indices[i]

            price_prev = _high[idx_prev]
            price_curr = _high[idx_curr]

            # KD值取附近区域的最大值
            window = 3
            k_prev = np.max(_k[max(0, idx_prev - window):idx_prev + window + 1])
            k_curr = np.max(_k[max(0, idx_curr - window):idx_curr + window + 1])

            if price_curr > price_prev and k_curr < k_prev and k_prev > cfg.KDJ_OVERBOUGHT * 0.8:
                severity = (k_prev - k_curr) / k_prev if k_prev > 0 else 0
                signals.append({
                    'type': 'kdj_divergence',
                    'direction': 'top',
                    'idx': idx_curr,
                    'date': df.index[idx_curr],
                    'price': price_curr,
                    'k_prev': k_prev,
                    'k_curr': k_curr,
                    'severity': abs(severity),
                    'description': f'KDJ顶背离: 股价新高{price_curr:.2f}, K值下降{k_prev:.1f}->{k_curr:.1f}'
                })

        return signals

    def detect_bottom_divergence(self, order: int = 5) -> list:
        """
        检测KDJ底背离
        股价创新低, 但KD值在20附近不创新低
        """
        signals = []
        df = self.df
        _low = df['low'].values
        _k = df['K'].values

        swing_lows = find_swing_lows(df['low'], order)
        low_indices = [df.index.get_loc(i) for i in swing_lows[swing_lows].index]

        for i in range(1, len(low_indices)):
            idx_prev = low_indices[i - 1]
            idx_curr = low_indices[i]

            price_prev = _low[idx_prev]
            price_curr = _low[idx_curr]

            window = 3
            k_prev = np.min(_k[max(0, idx_prev - window):idx_prev + window + 1])
            k_curr = np.min(_k[max(0, idx_curr - window):idx_curr + window + 1])

            if price_curr < price_prev and k_curr > k_prev and k_prev < cfg.KDJ_OVERSOLD * 1.2:
                severity = (k_curr - k_prev) / abs(k_prev) if k_prev != 0 else 0
                signals.append({
                    'type': 'kdj_divergence',
                    'direction': 'bottom',
                    'idx': idx_curr,
                    'date': df.index[idx_curr],
                    'price': price_curr,
                    'k_prev': k_prev,
                    'k_curr': k_curr,
                    'severity': abs(severity),
                    'description': f'KDJ底背离: 股价新低{price_curr:.2f}, K值上升{k_prev:.1f}->{k_curr:.1f}'
                })

        return signals

    def analyze_crosses_with_context(self) -> dict:
        """分析带趋势上下文的KDJ金叉死叉"""
        crosses = find_kdj_crosses(self.df['K'], self.df['D'])
        top_divs = self.detect_top_divergence()
        bottom_divs = self.detect_bottom_divergence()

        return {
            'crosses': crosses,
            'valid_golden': [c for c in crosses if c['type'] == 'golden' and c['valid']],
            'valid_death': [c for c in crosses if c['type'] == 'death' and c['valid']],
        }

    def analyze_all(self) -> dict:
        return {
            'top_divergence': self.detect_top_divergence(),
            'bottom_divergence': self.detect_bottom_divergence(),
            'crosses': self.analyze_crosses_with_context(),
        }
