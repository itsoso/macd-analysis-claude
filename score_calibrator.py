"""
Phase 2a: Score Calibration — SS/BS → 校准概率

将原始 sell_score/buy_score 映射为校准后的概率 p(win) 和期望收益 E[R],
替代硬阈值入场条件。

核心思路:
1. 对每个 (direction, regime) 桶, 按 SS/BS 分档 (每 5 分一档)
2. 计算每档的: 胜率, 平均 R, 净 funding cost
3. 用 isotonic regression 映射 SS → p(win)
4. 入场条件: E[R|ss, regime] > cost_estimate

使用方式:
    # 1. 从回测交易训练校准模型
    calibrator = ScoreCalibrator()
    calibrator.fit_from_trades(trades)

    # 2. 查询入场决策
    ok, info = calibrator.should_enter('short', 'neutral', ss=65, cost=0.002)

    # 3. 保存/加载模型
    calibrator.save('score_calibration_model.json')
    calibrator = ScoreCalibrator.load('score_calibration_model.json')
"""

import json
import os
import sys
import math
import numpy as np
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class IsotonicRegression:
    """
    简易 isotonic regression 实现 (Pool Adjacent Violators Algorithm)。
    不依赖 sklearn, 适合嵌入式使用。

    将单调递增约束拟合 x → y:
        对任意 x1 < x2, 保证 f(x1) <= f(x2)
    """

    def __init__(self):
        self.x_thresholds = []
        self.y_values = []

    def fit(self, x, y, sample_weight=None):
        """
        拟合 isotonic regression。

        Parameters
        ----------
        x : array-like, shape (n,)
            输入特征 (score)
        y : array-like, shape (n,)
            目标值 (win=1, loss=0 或 pnl_r)
        sample_weight : array-like, optional
            样本权重
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(x)
        if n == 0:
            self.x_thresholds = []
            self.y_values = []
            return self

        if sample_weight is None:
            sample_weight = np.ones(n)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)

        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        w_sorted = sample_weight[order]

        # Pool Adjacent Violators (PAV)
        blocks = []
        for i in range(n):
            blocks.append({
                'sum_wy': y_sorted[i] * w_sorted[i],
                'sum_w': w_sorted[i],
                'x_min': x_sorted[i],
                'x_max': x_sorted[i],
            })
            # 合并违反单调性的相邻块
            while len(blocks) >= 2:
                last = blocks[-1]
                prev = blocks[-2]
                val_last = last['sum_wy'] / max(last['sum_w'], 1e-10)
                val_prev = prev['sum_wy'] / max(prev['sum_w'], 1e-10)
                if val_prev > val_last:
                    # 违反单调递增, 合并
                    merged = {
                        'sum_wy': last['sum_wy'] + prev['sum_wy'],
                        'sum_w': last['sum_w'] + prev['sum_w'],
                        'x_min': prev['x_min'],
                        'x_max': last['x_max'],
                    }
                    blocks.pop()
                    blocks.pop()
                    blocks.append(merged)
                else:
                    break

        # 提取分段函数
        self.x_thresholds = []
        self.y_values = []
        for b in blocks:
            val = b['sum_wy'] / max(b['sum_w'], 1e-10)
            mid = (b['x_min'] + b['x_max']) / 2
            self.x_thresholds.append(mid)
            self.y_values.append(float(np.clip(val, 0.0, 1.0)))

        return self

    def predict(self, x):
        """用拟合的分段函数预测。"""
        x = np.asarray(x, dtype=float)
        if len(self.x_thresholds) == 0:
            return np.full_like(x, 0.5)

        result = np.empty_like(x)
        for i, xi in enumerate(x.flat):
            if xi <= self.x_thresholds[0]:
                result.flat[i] = self.y_values[0]
            elif xi >= self.x_thresholds[-1]:
                result.flat[i] = self.y_values[-1]
            else:
                # 线性插值
                idx = np.searchsorted(self.x_thresholds, xi) - 1
                idx = max(0, min(idx, len(self.x_thresholds) - 2))
                x0, x1 = self.x_thresholds[idx], self.x_thresholds[idx + 1]
                y0, y1 = self.y_values[idx], self.y_values[idx + 1]
                t = (xi - x0) / max(x1 - x0, 1e-10)
                result.flat[i] = y0 + t * (y1 - y0)
        return result

    def to_dict(self):
        return {
            'x_thresholds': [round(x, 4) for x in self.x_thresholds],
            'y_values': [round(y, 6) for y in self.y_values],
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.x_thresholds = list(d.get('x_thresholds', []))
        obj.y_values = list(d.get('y_values', []))
        return obj


class ScoreCalibrator:
    """
    分数校准器: 将 SS/BS 映射为校准概率 p(win) 和期望收益 E[R]。

    内部维护:
    - p_win_model: {(direction, regime): IsotonicRegression}  score → p(win)
    - er_model: {(direction, regime): IsotonicRegression}      score → E[R]
    - bin_stats: {(direction, regime): {score_bin: {n, wins, losses, avg_r, avg_mae}}}
    """

    def __init__(self, bin_size=5):
        """
        Parameters
        ----------
        bin_size : int
            分档宽度 (默认每 5 分一档)
        """
        self.bin_size = bin_size
        self.p_win_models = {}
        self.er_models = {}
        self.bin_stats = {}
        self.fit_time = None
        self.total_samples = 0
        self.min_samples_per_bucket = 5

    def fit_from_trades(self, trades, verbose=True):
        """
        从完整的交易记录中训练校准模型。

        Parameters
        ----------
        trades : list[dict]
            交易记录, 每条需包含:
            - action: 平仓动作 (CLOSE_SHORT, CLOSE_LONG, etc.)
            - direction: short / long
            - regime_label: regime 标签
            - ss / bs: 入场时的信号分数
            - pnl: 已实现盈亏
            - margin: 保证金 (用于计算 pnl_r)
        """
        close_keywords = {'CLOSE_SHORT', 'CLOSE_LONG', 'LIQUIDATED'}
        close_cn_keywords = ['平', '止损', '止盈']

        # 收集有效样本
        samples = defaultdict(list)
        for t in trades:
            action = str(t.get('action', ''))
            is_close = action in close_keywords or any(k in action for k in close_cn_keywords)
            if not is_close:
                continue

            direction = str(t.get('direction', 'unknown'))
            regime = str(t.get('regime_label', 'unknown'))
            pnl = float(t.get('pnl', 0))
            margin = float(t.get('margin', 0))

            # 获取入场时的分数
            # 对于 short: 用 ss; 对于 long: 用 bs
            extra = t.get('extra', t)
            if direction == 'short':
                score = float(extra.get('ss', extra.get('sig_entry_score', 0)))
            else:
                score = float(extra.get('bs', extra.get('sig_entry_score', 0)))

            if score <= 0:
                continue

            pnl_r = pnl / max(margin, 1.0) if margin > 0 else 0
            win = 1 if pnl > 0 else 0

            key = (direction, regime)
            samples[key].append({
                'score': score,
                'win': win,
                'pnl_r': pnl_r,
                'mae': abs(float(t.get('min_pnl_r', 0) or 0)),
                'mfe': abs(float(t.get('max_pnl_r', 0) or 0)),
            })

        if verbose:
            print(f"\n{'='*70}")
            print(f"Score Calibration 训练")
            print(f"{'='*70}")

        self.total_samples = sum(len(v) for v in samples.values())

        # 也对纯 direction (跨 regime) 训练一个 fallback 模型
        direction_samples = defaultdict(list)
        for (d, r), s_list in samples.items():
            direction_samples[d].extend(s_list)

        # 训练每个桶的模型
        all_keys = list(samples.keys()) + [(d, '_all') for d in direction_samples.keys()]

        for key in sorted(set(all_keys)):
            direction, regime = key
            if regime == '_all':
                s_list = direction_samples[direction]
            else:
                s_list = samples[key]

            if len(s_list) < self.min_samples_per_bucket:
                if verbose and regime != '_all':
                    print(f"  {direction:>6}/{regime:<18}: 样本不足 ({len(s_list)}), 跳过")
                continue

            scores = np.array([s['score'] for s in s_list])
            wins = np.array([s['win'] for s in s_list])
            pnl_rs = np.array([s['pnl_r'] for s in s_list])

            # Isotonic regression: score → p(win)
            p_model = IsotonicRegression().fit(scores, wins)
            self.p_win_models[key] = p_model

            # Isotonic regression: score → E[R]
            er_model = IsotonicRegression().fit(scores, pnl_rs)
            self.er_models[key] = er_model

            # 分档统计
            bins = {}
            for s in s_list:
                b = int(s['score'] // self.bin_size) * self.bin_size
                if b not in bins:
                    bins[b] = {'n': 0, 'wins': 0, 'losses': 0, 'sum_r': 0, 'sum_mae': 0}
                bins[b]['n'] += 1
                bins[b]['wins'] += s['win']
                bins[b]['losses'] += 1 - s['win']
                bins[b]['sum_r'] += s['pnl_r']
                bins[b]['sum_mae'] += s['mae']
            for b, bv in bins.items():
                bv['avg_r'] = round(bv['sum_r'] / max(bv['n'], 1), 6)
                bv['avg_mae'] = round(bv['sum_mae'] / max(bv['n'], 1), 6)
                bv['win_rate'] = round(bv['wins'] / max(bv['n'], 1), 4)
                del bv['sum_r']
                del bv['sum_mae']
            self.bin_stats[key] = bins

            if verbose:
                n = len(s_list)
                wr = np.mean(wins)
                avg_r = np.mean(pnl_rs)
                p_at_50 = float(p_model.predict(np.array([50.0]))[0]) if len(p_model.x_thresholds) > 0 else 0.5
                er_at_50 = float(er_model.predict(np.array([50.0]))[0]) if len(er_model.x_thresholds) > 0 else 0.0
                print(f"  {direction:>6}/{regime:<18}: n={n:>4}, "
                      f"WR={wr:.2%}, avgR={avg_r:+.4f}, "
                      f"P(win|50)={p_at_50:.3f}, E[R|50]={er_at_50:+.4f}")

        self.fit_time = datetime.now().isoformat()

        if verbose:
            print(f"\n  总样本: {self.total_samples}, 桶数: {len(self.p_win_models)}")
            self._print_reliability_diagram()

    def _print_reliability_diagram(self):
        """打印校准可靠性图 (文本版)。"""
        print(f"\n  Reliability Diagram (校准准确度):")
        print(f"  {'Bucket':>8} {'Pred.P':>8} {'Actual':>8} {'N':>5} {'Gap':>8}")
        for key, bins in sorted(self.bin_stats.items()):
            model = self.p_win_models.get(key)
            if model is None:
                continue
            direction, regime = key
            if regime == '_all':
                continue
            print(f"\n  --- {direction}/{regime} ---")
            for b in sorted(bins.keys()):
                bv = bins[b]
                mid = b + self.bin_size / 2
                pred = float(model.predict(np.array([mid]))[0])
                actual = bv['win_rate']
                gap = pred - actual
                bar = '█' * int(abs(gap) * 50)
                sign = '+' if gap >= 0 else '-'
                print(f"  {b:>3}-{b+self.bin_size:<3} {pred:>7.3f} {actual:>7.3f} {bv['n']:>5} "
                      f"{sign}{abs(gap):.3f} {bar}")

    def should_enter(self, direction, regime, score, cost_estimate=0.002, min_p_win=0.40):
        """
        判断是否应该入场。

        Parameters
        ----------
        direction : str
            'short' or 'long'
        regime : str
            regime 标签
        score : float
            入场信号分数 (SS for short, BS for long)
        cost_estimate : float
            估计的交易成本 (手续费 + 滑点 + funding 作为 R 的比例)
        min_p_win : float
            最低胜率要求

        Returns
        -------
        tuple (bool, dict)
            (是否入场, 诊断信息)
        """
        info = {
            'direction': direction,
            'regime': regime,
            'score': score,
            'cost_estimate': cost_estimate,
        }

        # 查找最佳匹配的模型
        key = (direction, regime)
        fallback_key = (direction, '_all')

        p_model = self.p_win_models.get(key) or self.p_win_models.get(fallback_key)
        er_model = self.er_models.get(key) or self.er_models.get(fallback_key)

        if p_model is None:
            # 没有校准模型, 默认允许
            info['reason'] = 'no_model'
            info['model_key'] = 'none'
            return True, info

        score_arr = np.array([score])
        p_win = float(p_model.predict(score_arr)[0])
        e_r = float(er_model.predict(score_arr)[0]) if er_model else 0.0

        info['p_win'] = round(p_win, 4)
        info['e_r'] = round(e_r, 6)
        info['model_key'] = f"{key[0]}/{key[1]}"

        # 决策: E[R] > cost 且 p(win) > min_p_win
        e_r_net = e_r - cost_estimate
        info['e_r_net'] = round(e_r_net, 6)

        if p_win < min_p_win:
            info['reason'] = f'low_p_win ({p_win:.3f} < {min_p_win})'
            return False, info
        if e_r_net < 0:
            info['reason'] = f'negative_e_r ({e_r_net:+.4f})'
            return False, info

        info['reason'] = 'pass'
        return True, info

    def get_calibrated_score(self, direction, regime, score):
        """
        获取校准后的得分 (p(win) * 100)。

        可用于替代原始 SS/BS 进行阈值比较。
        """
        key = (direction, regime)
        fallback_key = (direction, '_all')
        model = self.p_win_models.get(key) or self.p_win_models.get(fallback_key)
        if model is None:
            return score
        return float(model.predict(np.array([score]))[0]) * 100

    def save(self, path):
        """保存校准模型到 JSON 文件。"""
        data = {
            'version': 1,
            'bin_size': self.bin_size,
            'fit_time': self.fit_time,
            'total_samples': self.total_samples,
            'p_win_models': {},
            'er_models': {},
            'bin_stats': {},
        }
        for key, model in self.p_win_models.items():
            k_str = f"{key[0]}_{key[1]}"
            data['p_win_models'][k_str] = model.to_dict()
        for key, model in self.er_models.items():
            k_str = f"{key[0]}_{key[1]}"
            data['er_models'][k_str] = model.to_dict()
        for key, bins in self.bin_stats.items():
            k_str = f"{key[0]}_{key[1]}"
            data['bin_stats'][k_str] = {str(b): v for b, v in bins.items()}

        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        """从 JSON 文件加载校准模型。"""
        with open(path) as f:
            data = json.load(f)

        obj = cls(bin_size=data.get('bin_size', 5))
        obj.fit_time = data.get('fit_time')
        obj.total_samples = data.get('total_samples', 0)

        for k_str, model_dict in data.get('p_win_models', {}).items():
            parts = k_str.split('_', 1)
            if len(parts) == 2:
                key = (parts[0], parts[1])
                obj.p_win_models[key] = IsotonicRegression.from_dict(model_dict)

        for k_str, model_dict in data.get('er_models', {}).items():
            parts = k_str.split('_', 1)
            if len(parts) == 2:
                key = (parts[0], parts[1])
                obj.er_models[key] = IsotonicRegression.from_dict(model_dict)

        for k_str, bins_dict in data.get('bin_stats', {}).items():
            parts = k_str.split('_', 1)
            if len(parts) == 2:
                key = (parts[0], parts[1])
                obj.bin_stats[key] = {int(b): v for b, v in bins_dict.items()}

        return obj


def main():
    """从回测数据训练校准模型的命令行入口。"""
    import argparse
    parser = argparse.ArgumentParser(description='Score Calibration: SS/BS → 校准概率')
    parser.add_argument('--trades-file', type=str, default=None,
                        help='交易记录 JSON 文件')
    parser.add_argument('--tf', default='1h', help='时间框架')
    parser.add_argument('--days', type=int, default=90, help='回测天数')
    parser.add_argument('--output', type=str, default='score_calibration_model.json',
                        help='输出模型文件')
    parser.add_argument('--bin-size', type=int, default=5, help='分档宽度')
    args = parser.parse_args()

    if args.trades_file:
        with open(args.trades_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            trades = data
        elif isinstance(data, dict):
            trades = data.get('trades', data.get('global_best_trades', []))
        else:
            print("无法识别的文件格式")
            return
    else:
        from mae_calibrator import run_backtest_for_mae
        trades, result = run_backtest_for_mae(tf=args.tf, days=args.days)
        if not trades:
            print("回测未产生交易, 无法训练")
            return

    calibrator = ScoreCalibrator(bin_size=args.bin_size)
    calibrator.fit_from_trades(trades)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    calibrator.save(output_path)
    print(f"\n校准模型已保存到: {output_path}")

    # 展示几个典型分数的校准结果
    print(f"\n{'='*60}")
    print(f"校准结果示例:")
    print(f"{'Score':>7} {'Direction':>10} {'p(win)':>8} {'E[R]':>8}")
    print(f"{'─'*60}")
    for score in [30, 40, 50, 60, 70, 80]:
        for d in ['short', 'long']:
            ok, info = calibrator.should_enter(d, 'neutral', score)
            p = info.get('p_win', '?')
            er = info.get('e_r', '?')
            if isinstance(p, float):
                print(f"  {score:>5} {d:>10} {p:>7.3f} {er:>+7.4f}  {'✓' if ok else '✗'}")


if __name__ == '__main__':
    main()
