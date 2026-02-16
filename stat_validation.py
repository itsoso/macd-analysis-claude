"""
Phase 3b: 统计验证工具 — DSR/PSR 计算

Deflated Sharpe Ratio (DSR):
    考虑多次试验后的 SR 通货膨胀, 给出经过 N 次实验校正后的真实 SR 置信度。
    如果 DSR < 0.5, 表示观察到的 Sharpe Ratio 可能只是运气。

Probabilistic Sharpe Ratio (PSR):
    考虑收益分布的偏度和峰度, 给出 SR > 基准值的概率。

Probability of Backtest Overfitting (PBO):
    使用组合对称交叉验证 (CSCV) 估计回测过拟合的概率。
    如果 PBO > 50%, 回测数字不可信。

使用方式:
    from stat_validation import compute_dsr, compute_psr, compute_pbo

    # 单策略验证
    psr = compute_psr(returns_series, sr_benchmark=0)

    # 多实验 DSR
    dsr = compute_dsr(sr_observed=1.5, n_experiments=100,
                      skewness=0.3, kurtosis=4.0, T=252)

    # PBO 估计
    pbo = compute_pbo(returns_matrix)  # N策略 x T期
"""

import numpy as np
import math
from scipy import stats as sp_stats


def compute_psr(returns, sr_benchmark=0.0, annualize=True, periods_per_year=252):
    """
    Probabilistic Sharpe Ratio (PSR).

    计算观测 SR 超过基准 SR 的概率, 考虑有限样本的偏度和峰度。

    Parameters
    ----------
    returns : array-like
        收益序列 (日收益率或 bar 收益率)
    sr_benchmark : float
        基准 Sharpe Ratio (默认 0, 即要求 SR > 0)
    annualize : bool
        是否年化
    periods_per_year : int
        每年的期数 (日收益=252, 1h=8760)

    Returns
    -------
    dict
        包含 psr, sr_observed, sr_std_error, T, skew, kurtosis
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    T = len(returns)

    if T < 10:
        return {'psr': 0.5, 'sr_observed': 0, 'T': T, 'error': 'insufficient_data'}

    # 收益统计
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    if sigma < 1e-10:
        return {'psr': 0.5, 'sr_observed': 0, 'T': T, 'error': 'zero_variance'}

    sr = mu / sigma
    skew = float(sp_stats.skew(returns))
    kurt = float(sp_stats.kurtosis(returns, fisher=True))  # excess kurtosis

    # 年化
    if annualize:
        sr_annual = sr * np.sqrt(periods_per_year)
        sr_bm = sr_benchmark
    else:
        sr_annual = sr
        sr_bm = sr_benchmark

    # SR 的标准误差 (考虑偏度和峰度)
    # 公式来自 Bailey & Lopez de Prado (2014)
    sr_std = np.sqrt(
        (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / max(T - 1, 1)
    )

    if sr_std < 1e-10:
        psr = 0.5
    else:
        z = (sr - sr_bm / np.sqrt(periods_per_year) if annualize else sr - sr_bm) / sr_std
        psr = float(sp_stats.norm.cdf(z))

    return {
        'psr': round(psr, 4),
        'sr_observed': round(sr_annual, 4),
        'sr_per_period': round(sr, 6),
        'sr_std_error': round(sr_std, 6),
        'T': T,
        'skewness': round(skew, 4),
        'excess_kurtosis': round(kurt, 4),
        'annualized': annualize,
        'periods_per_year': periods_per_year,
    }


def compute_dsr(sr_observed, n_experiments, skewness=0.0, kurtosis=3.0,
                T=252, sr_benchmark=0.0):
    """
    Deflated Sharpe Ratio (DSR).

    对多次实验中观测到的最优 SR 进行通货紧缩, 消除 selection bias。

    核心公式:
        E[max(SR)] ≈ (1 - γ) * Φ^{-1}(1 - 1/N) + γ * Φ^{-1}(1 - 1/(N*e))
        其中 γ ≈ 0.5772 (Euler-Mascheroni 常数), N = n_experiments

        DSR = PSR(SR_observed - E[max(SR)])

    Parameters
    ----------
    sr_observed : float
        观测到的最优 Sharpe Ratio (年化)
    n_experiments : int
        总实验次数 (包括所有参数组合)
    skewness : float
        收益分布的偏度
    kurtosis : float
        收益分布的峰度 (非 excess, 即正态=3)
    T : int
        观测期数
    sr_benchmark : float
        基准 SR (默认 0)

    Returns
    -------
    dict
        包含 dsr, sr_haircut, sr_deflated, sr_observed, n_experiments
    """
    if n_experiments <= 1:
        return {
            'dsr': 0.5,
            'sr_haircut': 0.0,
            'sr_deflated': sr_observed,
            'sr_observed': sr_observed,
            'n_experiments': n_experiments,
        }

    # Euler-Mascheroni 常数
    gamma = 0.5772156649

    # 期望最大 SR (来自 N 个独立实验)
    # E[max(Z_1..Z_N)] ≈ (1-γ)*Φ^{-1}(1-1/N) + γ*Φ^{-1}(1-1/(Ne))
    N = float(n_experiments)
    p1 = 1 - 1 / N
    p2 = 1 - 1 / (N * np.e)
    p1 = max(0.001, min(0.999, p1))
    p2 = max(0.001, min(0.999, p2))

    e_max_z = (1 - gamma) * sp_stats.norm.ppf(p1) + gamma * sp_stats.norm.ppf(p2)

    # 转换为 SR 的标准误差口径
    sr_std = np.sqrt(
        (1 - skewness * sr_observed + (kurtosis - 3) / 4 * sr_observed ** 2) / max(T - 1, 1)
    )

    sr_haircut = e_max_z * sr_std * np.sqrt(T)  # 年化口径
    sr_deflated = sr_observed - sr_haircut

    # DSR = P(SR > sr_benchmark | 经过 haircut)
    if sr_std > 0:
        z = (sr_deflated / np.sqrt(252) - sr_benchmark / np.sqrt(252)) / sr_std
        dsr = float(sp_stats.norm.cdf(z))
    else:
        dsr = 0.5

    return {
        'dsr': round(dsr, 4),
        'sr_haircut': round(sr_haircut, 4),
        'sr_deflated': round(sr_deflated, 4),
        'sr_observed': round(sr_observed, 4),
        'n_experiments': n_experiments,
        'e_max_z': round(e_max_z, 4),
        'sr_std_per_period': round(sr_std, 6),
        'T': T,
        'skewness': round(skewness, 4),
        'kurtosis': round(kurtosis, 4),
    }


def compute_pbo(returns_matrix, n_splits=8):
    """
    Probability of Backtest Overfitting (PBO).

    使用组合对称交叉验证 (CSCV) 估计过拟合概率。

    Parameters
    ----------
    returns_matrix : np.ndarray, shape (n_strategies, T)
        N 个策略在 T 个时间步的收益矩阵
    n_splits : int
        时间分割数 (默认 8, 产生 C(8,4)=70 种组合)

    Returns
    -------
    dict
        包含 pbo, n_combos, logit_distribution
    """
    returns_matrix = np.asarray(returns_matrix, dtype=float)
    n_strategies, T = returns_matrix.shape

    if n_strategies < 2 or T < n_splits * 2:
        return {
            'pbo': 0.5,
            'error': 'insufficient_data',
            'n_strategies': n_strategies,
            'T': T,
        }

    # 将时间轴分为 n_splits 段
    split_size = T // n_splits
    splits = []
    for s in range(n_splits):
        start = s * split_size
        end = start + split_size if s < n_splits - 1 else T
        splits.append(returns_matrix[:, start:end])

    # 生成所有 C(n_splits, n_splits//2) 的 IS/OOS 组合
    from itertools import combinations
    half = n_splits // 2
    all_combos = list(combinations(range(n_splits), half))

    logit_values = []
    for is_indices in all_combos:
        oos_indices = tuple(i for i in range(n_splits) if i not in is_indices)

        # IS: 合并选中的分割
        is_returns = np.concatenate([splits[i] for i in is_indices], axis=1)
        oos_returns = np.concatenate([splits[i] for i in oos_indices], axis=1)

        # IS 排名: 按 SR 排序
        is_sr = np.mean(is_returns, axis=1) / np.maximum(np.std(is_returns, axis=1, ddof=1), 1e-10)
        oos_sr = np.mean(oos_returns, axis=1) / np.maximum(np.std(oos_returns, axis=1, ddof=1), 1e-10)

        # IS 最优策略在 OOS 中的排名
        best_is_idx = np.argmax(is_sr)
        # 排名: OOS SR 大于 best_is 在 OOS 中的 SR 的比例
        best_oos_sr = oos_sr[best_is_idx]
        rank_pct = np.mean(oos_sr >= best_oos_sr)  # 0=最差, 1=最佳

        # Logit: 如果 rank_pct > 0.5, IS 最优在 OOS 也不错
        # PBO = P(logit < 0) = P(rank_pct > 0.5) → IS最优在OOS表现差
        if 0 < rank_pct < 1:
            logit = np.log(rank_pct / (1 - rank_pct))
        elif rank_pct >= 1:
            logit = 5.0  # 极好
        else:
            logit = -5.0  # 极差

        logit_values.append(logit)

    logit_values = np.array(logit_values)
    pbo = float(np.mean(logit_values < 0))

    return {
        'pbo': round(pbo, 4),
        'pbo_interpretation': 'overfitting_likely' if pbo > 0.5 else 'acceptable',
        'n_combos': len(all_combos),
        'n_strategies': n_strategies,
        'T': T,
        'n_splits': n_splits,
        'logit_mean': round(float(np.mean(logit_values)), 4),
        'logit_std': round(float(np.std(logit_values)), 4),
        'logit_median': round(float(np.median(logit_values)), 4),
    }


def compute_strategy_stats(returns, periods_per_year=252):
    """
    计算策略的完整统计指标集合。

    Parameters
    ----------
    returns : array-like
        收益序列
    periods_per_year : int
        每年期数

    Returns
    -------
    dict
        完整统计指标
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    T = len(returns)

    if T < 2:
        return {'error': 'insufficient_data', 'T': T}

    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    skew = float(sp_stats.skew(returns))
    kurt = float(sp_stats.kurtosis(returns, fisher=True))

    sr = mu / max(sigma, 1e-10) * np.sqrt(periods_per_year)

    # Sortino Ratio (只用下行偏差)
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else sigma
    sortino = mu / max(downside_std, 1e-10) * np.sqrt(periods_per_year)

    # Max Drawdown
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(np.min(dd))

    # Calmar Ratio
    annual_return = mu * periods_per_year
    calmar = annual_return / max(abs(max_dd), 0.01)

    # Hit Rate
    hit_rate = np.mean(returns > 0)

    # Profit Factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / max(losses, 1e-10)

    # PSR
    psr_result = compute_psr(returns, sr_benchmark=0, periods_per_year=periods_per_year)

    return {
        'T': T,
        'annualized_return': round(annual_return * 100, 2),
        'annualized_volatility': round(sigma * np.sqrt(periods_per_year) * 100, 2),
        'sharpe_ratio': round(sr, 3),
        'sortino_ratio': round(sortino, 3),
        'calmar_ratio': round(calmar, 3),
        'max_drawdown': round(max_dd * 100, 2),
        'hit_rate': round(hit_rate, 4),
        'profit_factor': round(pf, 3),
        'skewness': round(skew, 4),
        'excess_kurtosis': round(kurt, 4),
        'psr': psr_result['psr'],
    }


def main():
    """命令行入口: 从回测历史计算统计验证。"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description='统计验证: DSR/PSR/PBO 计算')
    parser.add_argument('--result-file', type=str, default='optimize_six_book_result.json',
                        help='回测结果 JSON 文件')
    parser.add_argument('--n-experiments', type=int, default=None,
                        help='总实验次数 (默认从 total_variants_tested 读取)')
    args = parser.parse_args()

    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.result_file)
    if not os.path.exists(result_path):
        print(f"文件不存在: {result_path}")
        return

    import os
    with open(result_path) as f:
        data = json.load(f)

    # 提取历史资金曲线
    history = data.get('global_best_history', [])
    if not history:
        print("无历史数据")
        return

    totals = [h['total'] for h in history]
    returns = np.diff(totals) / np.maximum(np.array(totals[:-1]), 1.0)

    print(f"\n{'='*60}")
    print(f"统计验证报告")
    print(f"{'='*60}")

    # 1. 基础统计
    print(f"\n1. 基础统计:")
    stats = compute_strategy_stats(returns, periods_per_year=8760)  # 1h bars
    for k, v in stats.items():
        print(f"   {k}: {v}")

    # 2. PSR
    print(f"\n2. Probabilistic Sharpe Ratio:")
    psr = compute_psr(returns, sr_benchmark=0, periods_per_year=8760)
    print(f"   PSR = {psr['psr']:.4f} (SR > 0 的概率)")
    print(f"   SR observed = {psr['sr_observed']:.4f}")
    print(f"   Skewness = {psr['skewness']:.4f}")
    print(f"   Excess Kurtosis = {psr['excess_kurtosis']:.4f}")

    # 3. DSR
    n_exp = args.n_experiments or data.get('total_variants_tested', 100)
    print(f"\n3. Deflated Sharpe Ratio (N={n_exp} experiments):")
    dsr = compute_dsr(
        sr_observed=psr['sr_observed'],
        n_experiments=n_exp,
        skewness=psr['skewness'],
        kurtosis=psr['excess_kurtosis'] + 3,  # convert to non-excess
        T=psr['T'],
    )
    print(f"   DSR = {dsr['dsr']:.4f}")
    print(f"   SR haircut = {dsr['sr_haircut']:.4f}")
    print(f"   SR deflated = {dsr['sr_deflated']:.4f}")
    print(f"   E[max(Z)] = {dsr['e_max_z']:.4f}")

    if dsr['dsr'] > 0.95:
        print(f"   ✓ DSR > 0.95: 策略 SR 显著超过随机水平")
    elif dsr['dsr'] > 0.50:
        print(f"   △ DSR > 0.50: 策略可能有真实 alpha, 但不确定")
    else:
        print(f"   ✗ DSR < 0.50: 观测到的 SR 可能是运气, 需要更多数据")


if __name__ == '__main__':
    main()
