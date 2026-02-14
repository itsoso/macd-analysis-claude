"""
综合背离分析模块
整合所有背离分析方法, 按照书中原则进行综合判断:

核心原则:
1. 多种方法结论一致 -> 确认背离
2. 结论分歧时: 上涨卖出->按背离处理; 下跌买入->暂不按背离处理
3. 背离越严重越应警觉
4. 卖点用背离(宁早勿晚), 买点用背驰(宁迟勿早)
5. 形态背离 + MACD背离 + 其他指标背离综合判断最可靠
"""

import pandas as pd
from .pattern_divergence import PatternDivergenceAnalyzer
from .macd_divergence import MACDDivergenceAnalyzer
from .exhaustion import ExhaustionAnalyzer
from .kdj_divergence import KDJDivergenceAnalyzer
from .cci_divergence import CCIDivergenceAnalyzer
from .rsi_divergence import RSIDivergenceAnalyzer
from .volume_price_divergence import VolumePriceDivergenceAnalyzer


class ComprehensiveAnalyzer:
    """综合背离分析器"""

    def __init__(self, df: pd.DataFrame, _skip_copy: bool = False):
        # _skip_copy=True: 调用方已保证 df 是独立副本, 跳过 9 次冗余 copy
        # 所有子分析器共享同一份 df, _ensure_* 添加的列互不冲突
        shared_df = df if _skip_copy else df.copy()
        self.df = shared_df
        self.pattern = PatternDivergenceAnalyzer(shared_df, _skip_copy=True)
        self.macd = MACDDivergenceAnalyzer(shared_df, _skip_copy=True)
        self.exhaustion = ExhaustionAnalyzer(shared_df, _skip_copy=True)
        self.kdj = KDJDivergenceAnalyzer(shared_df, _skip_copy=True)
        self.cci = CCIDivergenceAnalyzer(shared_df, _skip_copy=True)
        self.rsi = RSIDivergenceAnalyzer(shared_df, _skip_copy=True)
        self.volume = VolumePriceDivergenceAnalyzer(shared_df, _skip_copy=True)

    def analyze_all(self) -> dict:
        """执行全面综合分析"""
        results = {
            'pattern': self.pattern.analyze_all(),
            'macd': self.macd.analyze_all(),
            'exhaustion': self.exhaustion.analyze_all(),
            'kdj': self.kdj.analyze_all(),
            'cci': self.cci.analyze_all(),
            'rsi': self.rsi.analyze_all(),
            'volume': self.volume.analyze_all(),
        }

        # 综合评分
        results['comprehensive_score'] = self._calculate_comprehensive_score(results)
        results['trade_recommendations'] = self._generate_recommendations(results)

        return results

    def _calculate_comprehensive_score(self, results: dict) -> dict:
        """
        计算综合背离评分

        评分维度:
        - 形态背离 (第二章): 幅度/时间度/均线面积
        - MACD背离 (第三章): 黄白线/彩柱线长度/面积
        - 背驰信号 (第四章): 隔堆背离次数 + 零轴返回次数
        - 其他指标 (第五章): KDJ/CCI/RSI/量价

        满分100分, 分数越高背离越严重
        """
        top_score = 0
        bottom_score = 0
        top_details = []
        bottom_details = []

        # --- 形态背离评分 (最高30分) ---
        pattern = results['pattern']
        for key in ['amplitude_divergence', 'time_divergence', 'ma_area_divergence']:
            for sig in pattern.get(key, []):
                score = min(sig.get('severity', 0) * 15, 10)
                if sig['direction'] == 'top':
                    top_score += score
                    top_details.append(f"{sig['type']}: +{score:.1f}")
                else:
                    bottom_score += score
                    bottom_details.append(f"{sig['type']}: +{score:.1f}")

        # --- MACD背离评分 (最高40分) ---
        macd = results['macd']
        for key in ['dif_dea_divergence', 'bar_length_divergence',
                     'bar_area_divergence', 'dif_dea_area_divergence']:
            for sig in macd.get(key, []):
                base_score = min(sig.get('severity', 0) * 15, 10)
                # 隔堆背离加分
                if sig.get('subtype') == 'separated':
                    base_score *= 1.5
                score = min(base_score, 10)
                if sig['direction'] == 'top':
                    top_score += score
                    top_details.append(f"{sig['type']}({sig.get('subtype', '')}): +{score:.1f}")
                else:
                    bottom_score += score
                    bottom_details.append(f"{sig['type']}({sig.get('subtype', '')}): +{score:.1f}")

        # --- 背驰评分 (最高20分) ---
        exhaustion = results['exhaustion']
        for sig in exhaustion.get('sell_signals', []):
            score = 15 if sig.get('priority') == 'high' else 10
            top_score += score
            top_details.append(f"顶背驰信号: +{score}")

        for sig in exhaustion.get('buy_signals', []):
            score = 15 if sig.get('priority') == 'high' else 10
            bottom_score += score
            bottom_details.append(f"底背驰信号: +{score}")

        # --- 其他指标评分 (最高10分) ---
        for analyzer_key in ['kdj', 'cci', 'rsi']:
            data = results[analyzer_key]
            for sig in data.get('top_divergence', []):
                score = min(sig.get('severity', 0) * 5, 3)
                top_score += score
                top_details.append(f"{sig['type']}: +{score:.1f}")
            for sig in data.get('bottom_divergence', []):
                score = min(sig.get('severity', 0) * 5, 3)
                bottom_score += score
                bottom_details.append(f"{sig['type']}: +{score:.1f}")

        # 量价背离
        for sig in results['volume'].get('price_up_volume_down', []):
            top_score += min(sig.get('severity', 0) * 5, 3)
            top_details.append(f"价升量减: +{min(sig.get('severity', 0) * 5, 3):.1f}")

        return {
            'top_score': min(top_score, 100),
            'bottom_score': min(bottom_score, 100),
            'top_details': top_details,
            'bottom_details': bottom_details,
            'top_level': self._score_to_level(top_score),
            'bottom_level': self._score_to_level(bottom_score),
        }

    @staticmethod
    def _score_to_level(score: float) -> str:
        if score >= 60:
            return '极度严重'
        elif score >= 40:
            return '严重'
        elif score >= 20:
            return '中等'
        elif score >= 10:
            return '轻微'
        else:
            return '无明显背离'

    def _generate_recommendations(self, results: dict) -> list:
        """
        生成交易建议

        原则:
        - 卖出: 出现某一方面严重顶背离就考虑离场
        - 买入: 等到各种底背离都明显时再介入
        """
        recommendations = []
        score = results['comprehensive_score']

        # 卖出建议
        if score['top_score'] >= 40:
            recommendations.append({
                'action': 'SELL',
                'urgency': 'high' if score['top_score'] >= 60 else 'medium',
                'score': score['top_score'],
                'level': score['top_level'],
                'reason': f'综合顶背离评分{score["top_score"]:.0f}/100 ({score["top_level"]})',
                'details': score['top_details']
            })
        elif score['top_score'] >= 20:
            recommendations.append({
                'action': 'CAUTION_SELL',
                'urgency': 'low',
                'score': score['top_score'],
                'level': score['top_level'],
                'reason': f'出现轻微顶背离信号, 评分{score["top_score"]:.0f}/100, 保持警惕',
                'details': score['top_details']
            })

        # 买入建议
        if score['bottom_score'] >= 60:
            recommendations.append({
                'action': 'BUY',
                'urgency': 'high' if score['bottom_score'] >= 70 else 'medium',
                'score': score['bottom_score'],
                'level': score['bottom_level'],
                'reason': f'综合底背离评分{score["bottom_score"]:.0f}/100 ({score["bottom_level"]})',
                'details': score['bottom_details']
            })
        elif score['bottom_score'] >= 30:
            recommendations.append({
                'action': 'WATCH',
                'urgency': 'low',
                'score': score['bottom_score'],
                'level': score['bottom_level'],
                'reason': f'出现底背离信号, 评分{score["bottom_score"]:.0f}/100, 继续观察',
                'details': score['bottom_details']
            })

        # 背驰信号(最高优先级)
        for sig in results['exhaustion'].get('sell_signals', []):
            recommendations.append({
                'action': 'SELL_EXHAUSTION',
                'urgency': 'critical',
                'reason': sig.get('reason', '背驰卖出信号'),
                'idx': sig.get('idx'),
                'date': sig.get('date'),
            })

        for sig in results['exhaustion'].get('buy_signals', []):
            recommendations.append({
                'action': 'BUY_EXHAUSTION',
                'urgency': 'critical',
                'reason': sig.get('reason', '背驰买入信号'),
                'idx': sig.get('idx'),
                'date': sig.get('date'),
            })

        return recommendations

    def print_report(self, results: dict = None):
        """打印综合分析报告"""
        if results is None:
            results = self.analyze_all()

        score = results['comprehensive_score']
        recommendations = results['trade_recommendations']

        print("=" * 70)
        print("               《背离技术分析》综合分析报告")
        print("=" * 70)

        # 综合评分
        print(f"\n【综合评分】")
        print(f"  顶背离评分: {score['top_score']:.0f}/100 ({score['top_level']})")
        if score['top_details']:
            for d in score['top_details'][:5]:
                print(f"    - {d}")
        print(f"  底背离评分: {score['bottom_score']:.0f}/100 ({score['bottom_level']})")
        if score['bottom_details']:
            for d in score['bottom_details'][:5]:
                print(f"    - {d}")

        # 各模块详情
        print(f"\n【形态背离 - 第二章】")
        pattern = results['pattern']
        for key in ['kline_divergence', 'amplitude_divergence', 'time_divergence',
                     'ma_cross_divergence', 'ma_area_divergence']:
            sigs = pattern.get(key, [])
            if sigs:
                print(f"  {key}: {len(sigs)}个信号")
                for s in sigs[-3:]:
                    print(f"    [{s['direction']}] {s['description']}")

        print(f"\n【MACD背离 - 第三章】")
        macd = results['macd']
        for key in ['dif_dea_divergence', 'bar_length_divergence',
                     'bar_area_divergence', 'dif_dea_area_divergence']:
            sigs = macd.get(key, [])
            if sigs:
                print(f"  {key}: {len(sigs)}个信号")
                for s in sigs[-3:]:
                    print(f"    [{s['direction']}] {s['description']}")

        print(f"\n【背离与背驰 - 第四章】")
        exhaustion = results['exhaustion']
        for key in ['exhaustion', 'sell_signals', 'buy_signals']:
            sigs = exhaustion.get(key, [])
            if sigs:
                print(f"  {key}: {len(sigs)}个信号")
                for s in sigs[-3:]:
                    desc = s.get('reason', s.get('description', ''))
                    print(f"    {desc}")

        print(f"\n【其他指标 - 第五章】")
        for name, key in [('KDJ', 'kdj'), ('CCI', 'cci'), ('RSI', 'rsi'), ('量价', 'volume')]:
            data = results[key]
            total = sum(len(v) for v in data.values() if isinstance(v, list))
            if total > 0:
                print(f"  {name}: {total}个信号")
                for k, v in data.items():
                    if isinstance(v, list) and v:
                        for s in v[-2:]:
                            if isinstance(s, dict) and 'description' in s:
                                print(f"    {s['description']}")

        # 交易建议
        print(f"\n{'=' * 70}")
        print(f"【交易建议】")
        if not recommendations:
            print("  当前无明显交易信号")
        for rec in recommendations:
            urgency_map = {'critical': '!!!紧急', 'high': '!!重要', 'medium': '!注意', 'low': '参考'}
            urgency = urgency_map.get(rec.get('urgency', 'low'), '')
            print(f"  {urgency} [{rec['action']}] {rec['reason']}")

        print("=" * 70)
