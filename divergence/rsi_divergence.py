"""
第五章第三节: RSI指标背离

核心要点:
- RSI 80-100: 超强(超买), 考虑卖出
- RSI 50-80:  强势区域, 持有
- RSI 20-50:  弱势区域, 观望
- RSI 0-20:   超弱(超卖), 考虑买入
- RSI顶背离: 股价创新高, RSI值在80以上不创新高
- RSI底背离: 股价创新低, RSI值在20以下不创新低
- 背离后等RSI强势突破50线再买入, 成功率更高
"""

import pandas as pd
import numpy as np
import config as cfg
from indicators import calc_rsi, find_swing_highs, find_swing_lows


class RSIDivergenceAnalyzer:
    """RSI指标背离分析器"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._ensure_rsi()

    def _ensure_rsi(self):
        if 'RSI6' not in self.df.columns:
            self.df['RSI6'] = calc_rsi(self.df['close'], cfg.RSI_SHORT)
        if 'RSI12' not in self.df.columns:
            self.df['RSI12'] = calc_rsi(self.df['close'], cfg.RSI_LONG)

    def detect_top_divergence(self, order: int = 5, rsi_col: str = 'RSI6') -> list:
        """
        RSI顶背离: 股价创新高, RSI在80以上不创新高
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
            rsi_prev = df[rsi_col].iloc[max(0, idx_prev - window):idx_prev + window + 1].max()
            rsi_curr = df[rsi_col].iloc[max(0, idx_curr - window):idx_curr + window + 1].max()

            if (price_curr > price_prev and rsi_curr < rsi_prev and
                    rsi_prev > cfg.RSI_OVERBOUGHT * 0.9):
                severity = (rsi_prev - rsi_curr) / rsi_prev if rsi_prev > 0 else 0
                signals.append({
                    'type': 'rsi_divergence',
                    'direction': 'top',
                    'idx': idx_curr,
                    'date': df.index[idx_curr],
                    'price': price_curr,
                    'rsi_prev': rsi_prev,
                    'rsi_curr': rsi_curr,
                    'severity': abs(severity),
                    'description': f'RSI顶背离: 股价新高{price_curr:.2f}, RSI下降{rsi_prev:.1f}->{rsi_curr:.1f}'
                })

        return signals

    def detect_bottom_divergence(self, order: int = 5, rsi_col: str = 'RSI6') -> list:
        """
        RSI底背离: 股价创新低, RSI在20以下不创新低
        买入建议: 等RSI向上突破50线时再介入
        """
        signals = []
        df = self.df

        swing_lows = find_swing_lows(df['low'], order)
        low_indices = [df.index.get_loc(i) for i in swing_lows[swing_lows].index]

        for i in range(1, len(low_indices)):
            idx_prev = low_indices[i - 1]
            idx_curr = low_indices[i]

            price_prev = df['low'].iloc[idx_prev]
            price_curr = df['low'].iloc[idx_curr]

            window = 3
            rsi_prev = df[rsi_col].iloc[max(0, idx_prev - window):idx_prev + window + 1].min()
            rsi_curr = df[rsi_col].iloc[max(0, idx_curr - window):idx_curr + window + 1].min()

            if (price_curr < price_prev and rsi_curr > rsi_prev and
                    rsi_prev < cfg.RSI_OVERSOLD * 1.2):
                severity = (rsi_curr - rsi_prev) / abs(rsi_prev) if rsi_prev != 0 else 0

                # 检查后续RSI是否突破50线(买入确认)
                rsi_breaks_50 = False
                for j in range(idx_curr, min(idx_curr + 20, len(df))):
                    if df[rsi_col].iloc[j] > 50:
                        rsi_breaks_50 = True
                        break

                signals.append({
                    'type': 'rsi_divergence',
                    'direction': 'bottom',
                    'idx': idx_curr,
                    'date': df.index[idx_curr],
                    'price': price_curr,
                    'rsi_prev': rsi_prev,
                    'rsi_curr': rsi_curr,
                    'severity': abs(severity),
                    'rsi_confirmed': rsi_breaks_50,
                    'description': f'RSI底背离: 股价新低{price_curr:.2f}, RSI上升{rsi_prev:.1f}->{rsi_curr:.1f}'
                                   f'{" (RSI已突破50确认)" if rsi_breaks_50 else ""}'
                })

        return signals

    def analyze_all(self) -> dict:
        return {
            'top_divergence': self.detect_top_divergence(),
            'bottom_divergence': self.detect_bottom_divergence(),
        }
