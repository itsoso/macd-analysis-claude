"""Score Calibrator — 将 SS/BS 工程分数校准为概率和期望收益

Phase 2a: 用历史交易数据建立 SS → p(win) 和 SS → E[R] 的映射
入场条件从 `ss >= threshold` 变为 `E[R|ss,regime] > cost_estimate`

用法:
  1. calibrator = ScoreCalibrator()
  2. calibrator.fit(trades)        # 用历史交易 fit isotonic regression
  3. p_win = calibrator.p_win('short', 'neutral', ss=45)
  4. e_r = calibrator.expected_r('short', 'neutral', ss=45)
  5. should_open = calibrator.should_enter('short', 'neutral', ss=45, cost=0.002)
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class ScoreCalibrator:
    """Score 校准器 — isotonic regression 将 SS/BS → p(win) 和 E[R]"""

    def __init__(self, min_samples: int = 10, score_key: str = 'ss'):
        """
        Args:
            min_samples: 每个 (direction, regime) 至少需要的样本数
            score_key: 'ss' 表示用 sell_score 做空, buy_score 做多;
                       或 'dominant' 表示取 max(ss, bs)
        """
        self.min_samples = min_samples
        self.score_key = score_key
        # 校准模型: {(direction, regime): {'p_win': IsotonicRegression, 'e_r': IsotonicRegression}}
        self._models: Dict[Tuple[str, str], dict] = {}
        # 原始统计: {(direction, regime): {'scores': [], 'wins': [], 'pnl_r': []}}
        self._raw: Dict[Tuple[str, str], dict] = {}
        # 分桶统计 (用于 reliability diagram)
        self._bucket_stats: Dict[Tuple[str, str], list] = {}
        self._fitted = False

    def fit(self, trades: List[dict], verbose: bool = True) -> 'ScoreCalibrator':
        """用历史交易数据 fit 校准模型

        Args:
            trades: 交易记录列表, 每条包含:
                - action: OPEN_SHORT/OPEN_LONG/CLOSE_SHORT/CLOSE_LONG
                - direction: short/long
                - regime_label: neutral/trend/high_vol/...
                - ss: sell_score at entry
                - bs: buy_score at entry
                - pnl: realized PnL ($)
                - min_pnl_r: MAE (ratio to margin)
                - max_pnl_r: MFE (ratio to margin)
        """
        if not HAS_SKLEARN:
            if verbose:
                print("  [ScoreCalibrator] sklearn 未安装, 使用分桶回退模式")
            return self._fit_bucket_fallback(trades, verbose)

        # 配对 open-close
        paired = self._pair_trades(trades)
        if verbose:
            print(f"  [ScoreCalibrator] 配对交易: {len(paired)} 笔")

        # 按 (direction, regime) 分组
        groups = defaultdict(lambda: {'scores': [], 'wins': [], 'pnl_r': []})
        for p in paired:
            key = (p['direction'], p['regime'])
            score = p['entry_score']
            win = 1.0 if p['pnl'] > 0 else 0.0
            pnl_r = p.get('pnl_r', p['pnl'] / max(p.get('margin', 1), 1))
            groups[key]['scores'].append(score)
            groups[key]['wins'].append(win)
            groups[key]['pnl_r'].append(pnl_r)

        # 也按 direction-only 分组 (fallback for small regime samples)
        dir_groups = defaultdict(lambda: {'scores': [], 'wins': [], 'pnl_r': []})
        for p in paired:
            key = p['direction']
            score = p['entry_score']
            win = 1.0 if p['pnl'] > 0 else 0.0
            pnl_r = p.get('pnl_r', p['pnl'] / max(p.get('margin', 1), 1))
            dir_groups[key]['scores'].append(score)
            dir_groups[key]['wins'].append(win)
            dir_groups[key]['pnl_r'].append(pnl_r)

        self._raw = dict(groups)

        # Fit isotonic regression per group
        for key, data in groups.items():
            n = len(data['scores'])
            direction, regime = key
            if n < self.min_samples:
                # 样本不足, 尝试 direction-only fallback
                fallback = dir_groups.get(direction)
                if fallback and len(fallback['scores']) >= self.min_samples:
                    if verbose:
                        print(f"    {direction}|{regime}: n={n} < {self.min_samples}, "
                              f"回退到 {direction}-all (n={len(fallback['scores'])})")
                    self._models[key] = self._fit_single(
                        fallback['scores'], fallback['wins'], fallback['pnl_r'],
                        f"{direction}|{regime}(fallback)", verbose
                    )
                else:
                    if verbose:
                        print(f"    {direction}|{regime}: n={n} < {self.min_samples}, 跳过")
                continue

            self._models[key] = self._fit_single(
                data['scores'], data['wins'], data['pnl_r'],
                f"{direction}|{regime}", verbose
            )

        self._fitted = True
        return self

    def _fit_single(self, scores, wins, pnl_r, label, verbose):
        """为单个 (direction, regime) fit isotonic regression"""
        X = np.array(scores, dtype=float)
        y_win = np.array(wins, dtype=float)
        y_r = np.array(pnl_r, dtype=float)

        # p(win) calibration
        ir_win = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        ir_win.fit(X, y_win)

        # E[R] calibration — 需要对 outlier 做 winsorize
        y_r_clipped = np.clip(y_r, np.percentile(y_r, 5), np.percentile(y_r, 95))
        ir_r = IsotonicRegression(out_of_bounds='clip')
        ir_r.fit(X, y_r_clipped)

        # 分桶统计
        buckets = self._compute_buckets(X, y_win, y_r)

        if verbose:
            n = len(X)
            wr = np.mean(y_win) * 100
            avg_r = np.mean(y_r) * 100
            print(f"    {label}: n={n} WR={wr:.0f}% avgR={avg_r:+.1f}% "
                  f"score_range=[{X.min():.0f}, {X.max():.0f}]")

        return {
            'p_win': ir_win,
            'e_r': ir_r,
            'n': len(X),
            'buckets': buckets,
        }

    def _compute_buckets(self, scores, wins, pnl_r, bucket_size=5):
        """分桶统计 (供 reliability diagram 和人工审查)"""
        buckets = []
        score_min = int(np.floor(scores.min() / bucket_size) * bucket_size)
        score_max = int(np.ceil(scores.max() / bucket_size) * bucket_size)
        for lo in range(score_min, score_max + 1, bucket_size):
            hi = lo + bucket_size
            mask = (scores >= lo) & (scores < hi)
            n = mask.sum()
            if n > 0:
                buckets.append({
                    'range': f"[{lo},{hi})",
                    'n': int(n),
                    'win_rate': float(np.mean(wins[mask])),
                    'avg_r': float(np.mean(pnl_r[mask])),
                    'median_r': float(np.median(pnl_r[mask])),
                })
        return buckets

    def _fit_bucket_fallback(self, trades, verbose):
        """无 sklearn 时的分桶回退"""
        paired = self._pair_trades(trades)
        groups = defaultdict(lambda: {'scores': [], 'wins': [], 'pnl_r': []})
        for p in paired:
            key = (p['direction'], p['regime'])
            groups[key]['scores'].append(p['entry_score'])
            groups[key]['wins'].append(1.0 if p['pnl'] > 0 else 0.0)
            pnl_r = p.get('pnl_r', p['pnl'] / max(p.get('margin', 1), 1))
            groups[key]['pnl_r'].append(pnl_r)

        for key, data in groups.items():
            X = np.array(data['scores'])
            y_win = np.array(data['wins'])
            y_r = np.array(data['pnl_r'])
            buckets = self._compute_buckets(X, y_win, y_r)
            self._models[key] = {'p_win': None, 'e_r': None, 'n': len(X), 'buckets': buckets}

        self._fitted = True
        return self

    def _pair_trades(self, trades: List[dict]) -> List[dict]:
        """配对 open-close 交易"""
        opens_by_dir = defaultdict(list)
        closes_by_dir = defaultdict(list)

        for t in trades:
            action = t.get('action', '')
            d = t.get('direction', 'unknown')
            if action.startswith('OPEN_'):
                opens_by_dir[d].append(t)
            elif action.startswith('CLOSE_'):
                closes_by_dir[d].append(t)

        paired = []
        for d in opens_by_dir:
            ops = opens_by_dir[d]
            cls = closes_by_dir.get(d, [])
            for i, o in enumerate(ops):
                if i >= len(cls):
                    break
                c = cls[i]
                # entry score = SS for short, BS for long
                if d == 'short':
                    entry_score = float(o.get('ss', 0) or 0)
                else:
                    entry_score = float(o.get('bs', 0) or 0)

                paired.append({
                    'direction': d,
                    'regime': o.get('regime_label', 'unknown'),
                    'entry_score': entry_score,
                    'pnl': float(c.get('pnl', 0) or 0),
                    'pnl_r': float(c.get('min_pnl_r', 0) or 0),  # fallback
                    'margin': float(o.get('margin', 0) or 0),
                    'min_pnl_r': float(c.get('min_pnl_r', 0) or 0),
                    'max_pnl_r': float(c.get('max_pnl_r', 0) or 0),
                })
                # 计算更准确的 pnl_r
                margin = paired[-1]['margin']
                if margin > 0:
                    paired[-1]['pnl_r'] = paired[-1]['pnl'] / margin
        return paired

    def p_win(self, direction: str, regime: str, score: float) -> float:
        """返回校准后的 P(win)

        Args:
            direction: 'short' or 'long'
            regime: regime label
            score: SS (for short) or BS (for long)

        Returns:
            P(win) in [0, 1], 未校准返回 -1
        """
        model = self._get_model(direction, regime)
        if model is None or model.get('p_win') is None:
            return -1.0
        return float(model['p_win'].predict(np.array([[score]]))[0])

    def expected_r(self, direction: str, regime: str, score: float) -> float:
        """返回校准后的 E[R] (期望收益率, ratio to margin)

        Returns:
            E[R], 未校准返回 NaN
        """
        model = self._get_model(direction, regime)
        if model is None or model.get('e_r') is None:
            return float('nan')
        return float(model['e_r'].predict(np.array([[score]]))[0])

    def should_enter(self, direction: str, regime: str, score: float,
                     cost_estimate: float = 0.002,
                     min_p_win: float = 0.40) -> Tuple[bool, dict]:
        """基于校准结果判断是否应该入场

        Args:
            direction: 'short' or 'long'
            regime: regime label
            score: SS or BS
            cost_estimate: 估算单次交易成本 (fee + slippage + avg funding)
            min_p_win: 最低胜率要求

        Returns:
            (should_enter: bool, info: dict)
        """
        p = self.p_win(direction, regime, score)
        er = self.expected_r(direction, regime, score)

        info = {
            'calibrated_p_win': p,
            'calibrated_e_r': er,
            'cost_estimate': cost_estimate,
            'calibrated': p >= 0,
        }

        if p < 0:
            # 未校准, 回退到默认行为 (允许入场)
            info['reason'] = 'uncalibrated'
            return True, info

        # 入场条件: E[R] > cost AND p(win) > min_p_win
        if np.isfinite(er) and er > cost_estimate and p >= min_p_win:
            info['reason'] = f'e_r={er:.3f}>cost={cost_estimate:.3f}, p={p:.2f}>={min_p_win:.2f}'
            return True, info
        elif np.isfinite(er) and er > cost_estimate:
            info['reason'] = f'e_r_ok but p_win={p:.2f}<{min_p_win:.2f}'
            return False, info
        else:
            info['reason'] = f'e_r={er:.3f}<=cost={cost_estimate:.3f}'
            return False, info

    def _get_model(self, direction, regime):
        """获取模型, 支持 fallback"""
        key = (direction, regime)
        if key in self._models:
            return self._models[key]
        # Fallback: 同方向的聚合模型
        for k, m in self._models.items():
            if k[0] == direction and k[1] == '_all':
                return m
        return None

    def get_bucket_stats(self, direction: str, regime: str) -> list:
        """获取分桶统计 (供可视化)"""
        model = self._get_model(direction, regime)
        if model is None:
            return []
        return model.get('buckets', [])

    def summary(self) -> str:
        """打印校准摘要"""
        lines = ["\n  Score Calibration Summary:"]
        lines.append(f"  {'Leg':<25s} {'N':>5s} {'校准WR':>8s} {'bucket范围':>15s}")
        lines.append(f"  {'-'*60}")
        for key, model in sorted(self._models.items()):
            d, r = key
            leg = f"{d}|{r}"
            n = model['n']
            buckets = model.get('buckets', [])
            if buckets:
                wr_range = f"{buckets[0]['win_rate']*100:.0f}-{buckets[-1]['win_rate']*100:.0f}%"
                score_range = f"{buckets[0]['range']}-{buckets[-1]['range']}"
            else:
                wr_range = 'N/A'
                score_range = 'N/A'
            lines.append(f"  {leg:<25s} {n:5d} {wr_range:>8s} {score_range:>15s}")

        # 分桶详情
        for key, model in sorted(self._models.items()):
            d, r = key
            leg = f"{d}|{r}"
            buckets = model.get('buckets', [])
            if buckets:
                lines.append(f"\n  [{leg}] 分桶详情:")
                lines.append(f"  {'Score':>10s} {'N':>5s} {'WR':>7s} {'avgR':>8s} {'medR':>8s}")
                for b in buckets:
                    lines.append(f"  {b['range']:>10s} {b['n']:5d} {b['win_rate']*100:6.0f}% "
                                 f"{b['avg_r']*100:+7.1f}% {b['median_r']*100:+7.1f}%")
        return '\n'.join(lines)

    def save(self, path: str):
        """保存校准结果 (分桶统计, 可 JSON 序列化)"""
        data = {}
        for key, model in self._models.items():
            d, r = key
            data[f"{d}|{r}"] = {
                'n': model['n'],
                'buckets': model.get('buckets', []),
            }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> 'ScoreCalibrator':
        """从文件加载校准结果 (分桶模式, 无需 sklearn)"""
        with open(path) as f:
            data = json.load(f)
        for leg, info in data.items():
            d, r = leg.split('|')
            self._models[(d, r)] = {
                'p_win': None,
                'e_r': None,
                'n': info['n'],
                'buckets': info['buckets'],
            }
        self._fitted = True
        return self
