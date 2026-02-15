#!/usr/bin/env python3
"""月度滚动 Walk-Forward 验证管道

Phase 3a: 6 月 IS + 1 月 OOS, 滚动步长 1 月
Phase 3b: Deflated Sharpe Ratio / Probabilistic Sharpe Ratio 统计验证

指标:
  - 每个 OOS 窗口: Return, PF, WR, MDD, Sharpe
  - 汇总: OOS 窗口胜率, 聚合 Sharpe, DSR, PSR
  - PBO 简化版: 盈利窗口占比

用法:
  cd /path/to/project && .venv/bin/python3 walk_forward_pipeline.py
"""
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from run_p0_oos_validation import load_base_config
from run_p1_p2_sensitivity import prepare_data
from optimize_six_book import _build_tf_score_index, run_strategy_multi_tf

# ── v10.2 + Phase 1 生产配置 ──
V8_OVERRIDES = {
    'regime_short_threshold': 'neutral:60',
    'short_conflict_regimes': 'trend,high_vol,neutral',
    'neutral_struct_discount_0': 0.0,
    'neutral_struct_discount_1': 0.05,
    'neutral_struct_discount_2': 0.15,
    'neutral_struct_discount_3': 0.50,
    'cooldown': 6,
    'use_continuous_trail': True,
    'continuous_trail_start_pnl': 0.05,
    'continuous_trail_max_pb': 0.60,
    'continuous_trail_min_pb': 0.30,
    'short_threshold': 40,
    'long_threshold': 25,
    'short_sl': -0.20,
    'short_tp': 0.60,
    'long_sl': -0.10,
    'long_tp': 0.40,
    'partial_tp_1': 0.15,
    'use_partial_tp_2': True,
    'short_max_hold': 48,
    'short_trail': 0.19,
    'long_trail': 0.12,
    'trail_pullback': 0.50,
    'continuous_trail_max_pb_short': 0.40,
    'use_regime_adaptive_fusion': False,
    'c6_div_weight': 0.70,
}

V10_2_PRODUCTION = {
    'use_soft_veto': True,
    'soft_veto_steepness': 3.0,
    'soft_veto_midpoint': 1.0,
    'soft_struct_min_mult': 0.02,
    'use_leg_risk_budget': True,
    'c6_div_weight': 0.70,
    'use_atr_sl': True,
    'use_regime_adaptive_sl': False,
    'atr_sl_mult': 3.0,
    'atr_sl_floor': -0.25,
    'atr_sl_ceil': -0.06,
    'atr_sl_mult_neutral': 2.0,
    'atr_sl_mult_trend': 3.5,
    'atr_sl_mult_low_vol_trend': 3.0,
    'atr_sl_mult_high_vol': 2.5,
    'atr_sl_mult_high_vol_choppy': 2.0,
    'use_risk_per_trade': True,
    'risk_per_trade_pct': 0.025,
    'risk_stop_mode': 'atr',
    'risk_atr_mult_short': 3.0,
    'risk_atr_mult_long': 2.0,
    'risk_max_margin_pct': 0.40,
    'risk_min_margin_pct': 0.03,
    'risk_budget_neutral_short': 0.10,
    'risk_budget_neutral_long': 0.30,
    'risk_budget_high_vol_short': 0.50,
    'risk_budget_high_vol_long': 0.50,
    'risk_budget_high_vol_choppy_short': 0.20,
    'risk_budget_high_vol_choppy_long': 0.20,
    'risk_budget_trend_short': 0.60,
    'risk_budget_trend_long': 1.20,
    'risk_budget_low_vol_trend_short': 0.50,
    'risk_budget_low_vol_trend_long': 1.20,
    'use_regime_sigmoid': True,
    'tp_disabled_regimes': ['trend', 'low_vol_trend'],
    'use_soft_antisqueeze': True,
    'soft_antisqueeze_w_fz': 0.5,
    'soft_antisqueeze_w_oi': 0.3,
    'soft_antisqueeze_w_imb': 0.2,
    'soft_antisqueeze_midpoint': 1.5,
    'soft_antisqueeze_steepness': 2.0,
    'soft_antisqueeze_max_discount': 0.50,
}


# ==================================================================
# Walk-Forward Engine
# ==================================================================

def run_single_window(all_data, all_signals, needed_tfs, cfg,
                      primary_tf, decision_tfs, start_dt, end_dt):
    """运行单个窗口的回测"""
    tf_score_map = _build_tf_score_index(all_data, all_signals, needed_tfs, cfg)
    primary_df = all_data[primary_tf]
    result = run_strategy_multi_tf(
        primary_df=primary_df, tf_score_map=tf_score_map,
        decision_tfs=decision_tfs, config=cfg,
        primary_tf=primary_tf, trade_days=0,
        trade_start_dt=start_dt, trade_end_dt=end_dt,
    )

    trades = result.get('trades', [])
    closes = [t for t in trades if t.get('action', '').startswith('CLOSE_')]
    n = len(closes)
    wins = sum(1 for t in closes if float(t.get('pnl', 0)) > 0)
    wr = wins / n * 100 if n > 0 else 0
    gp = sum(float(t.get('pnl', 0)) for t in closes if float(t.get('pnl', 0)) > 0)
    gl = abs(sum(float(t.get('pnl', 0)) for t in closes if float(t.get('pnl', 0)) <= 0))
    pf = gp / gl if gl > 0 else 0
    mdd = result.get('max_drawdown', 0)
    ret = result.get('strategy_return', 0)
    calmar = ret / abs(mdd) if mdd != 0 else 0

    # 计算 Sharpe-like ratio (月化)
    pnls = [float(t.get('pnl', 0)) for t in closes]
    if len(pnls) >= 2:
        sharpe = np.mean(pnls) / np.std(pnls, ddof=1) * np.sqrt(len(pnls)) if np.std(pnls) > 0 else 0
    else:
        sharpe = 0

    return {
        'strategy_return': ret,
        'max_drawdown': mdd,
        'total_trades': n,
        'win_rate': wr,
        'profit_factor': pf,
        'calmar': calmar,
        'sharpe': sharpe,
        'gross_profit': gp,
        'gross_loss': gl,
        'pnls': pnls,
    }


def generate_windows(start_year=2024, start_month=1, end_year=2026, end_month=1,
                     is_months=6, oos_months=1, step_months=1):
    """生成 Walk-Forward 窗口列表

    Returns:
        list of (is_start, is_end, oos_start, oos_end) tuples
    """
    windows = []
    current_year = start_year
    current_month = start_month

    while True:
        # IS window
        is_start = f"{current_year}-{current_month:02d}-01"
        is_end_year = current_year + (current_month + is_months - 1) // 12
        is_end_month = (current_month + is_months - 1) % 12 + 1
        is_end = f"{is_end_year}-{is_end_month:02d}-28"  # 月末

        # OOS window
        oos_start_year = current_year + (current_month + is_months) // 12
        oos_start_month_raw = current_month + is_months
        oos_start_year = current_year + (oos_start_month_raw - 1) // 12
        oos_start_month = (oos_start_month_raw - 1) % 12 + 1
        oos_start = f"{oos_start_year}-{oos_start_month:02d}-01"

        oos_end_month_raw = oos_start_month_raw + oos_months
        oos_end_year = current_year + (oos_end_month_raw - 1) // 12
        oos_end_month = (oos_end_month_raw - 1) % 12 + 1
        oos_end = f"{oos_end_year}-{oos_end_month:02d}-28"

        # Check bounds
        oos_end_dt = pd.Timestamp(oos_end)
        end_dt = pd.Timestamp(f"{end_year}-{end_month:02d}-28")
        if oos_end_dt > end_dt:
            break

        windows.append((is_start, is_end, oos_start, oos_end))

        # Step forward
        current_month += step_months
        if current_month > 12:
            current_year += (current_month - 1) // 12
            current_month = (current_month - 1) % 12 + 1

    return windows


# ==================================================================
# DSR / PSR Computation
# ==================================================================

def compute_sharpe_ratio(returns):
    """计算年化 Sharpe Ratio (假设月度收益)"""
    if len(returns) < 2:
        return 0.0
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)
    if std_r == 0:
        return 0.0
    return float(mean_r / std_r * np.sqrt(12))  # 年化


def compute_psr(observed_sharpe, n_obs, skewness=0, kurtosis=3,
                benchmark_sharpe=0):
    """Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012)

    PSR = Φ[(SR^ - SR*) × √(n-1) / √(1 - γ₃·SR^ + (γ₄-1)/4·SR^²)]

    Args:
        observed_sharpe: 观察到的 Sharpe Ratio
        n_obs: 观察样本数
        skewness: 收益偏度 (γ₃)
        kurtosis: 收益峰度 (γ₄), 正态分布=3
        benchmark_sharpe: 基准 Sharpe (默认 0)

    Returns:
        PSR probability
    """
    from scipy.stats import norm
    if n_obs < 2:
        return 0.5

    sr_diff = observed_sharpe - benchmark_sharpe
    denominator = math.sqrt(1 - skewness * observed_sharpe +
                            (kurtosis - 1) / 4 * observed_sharpe ** 2)
    if denominator == 0:
        return 0.5

    z = sr_diff * math.sqrt(n_obs - 1) / denominator
    return float(norm.cdf(z))


def compute_dsr(observed_sharpe, n_obs, n_experiments,
                skewness=0, kurtosis=3, var_sharpes=None):
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014)

    修正多重检验偏差: SR* = √(V[SR]) × ((1-γ)·Z^(-1)[1-1/N] + γ·Z^(-1)[1-1/N·e^(-1)])
    其中 γ ≈ 0.5772 (Euler-Mascheroni), N = 实验次数

    Args:
        observed_sharpe: 观察到的最优 Sharpe
        n_obs: 每次实验的样本数
        n_experiments: 总实验次数
        skewness: 收益偏度
        kurtosis: 收益峰度
        var_sharpes: Sharpe Ratio 的方差 (如已知)

    Returns:
        DSR probability
    """
    from scipy.stats import norm
    if n_experiments <= 1:
        return compute_psr(observed_sharpe, n_obs, skewness, kurtosis)

    # Expected max Sharpe under multiple testing (E0)
    gamma = 0.5772156649  # Euler-Mascheroni constant
    if var_sharpes is None:
        # 估计 Sharpe 方差: Var[SR] ≈ (1 + 0.5·SR²) / T
        var_sharpes = (1 + 0.5 * observed_sharpe ** 2) / max(n_obs, 1)

    std_sr = math.sqrt(max(var_sharpes, 1e-10))

    # Expected max of N i.i.d. draws from N(0, std_sr)
    z_inv = norm.ppf(1 - 1.0 / n_experiments) if n_experiments > 1 else 0
    e0 = std_sr * (
        (1 - gamma) * z_inv +
        gamma * norm.ppf(1 - 1.0 / (n_experiments * math.e))
    )

    # DSR = PSR with benchmark = E0
    return compute_psr(observed_sharpe, n_obs, skewness, kurtosis,
                       benchmark_sharpe=e0)


def main():
    print("=" * 170)
    print("  v11 Phase 3: Walk-Forward 验证 + DSR/PSR 统计分析")
    print("  IS=6月, OOS=1月, 步长=1月 | 区间: 2024-01 ~ 2026-01")
    print("=" * 170)

    # 加载全量数据 (2024-01 ~ 2026-01, 需要 warmup 所以从更早开始)
    print("\n  加载全量数据 (2023-10~2026-02)...")
    all_data, all_signals, needed_tfs, primary_tf, decision_tfs, _, _ = \
        prepare_data('ETHUSDT', '2023-10-01', '2026-02-28')
    print(f"  决策 TF: {decision_tfs}")

    # 生成 Walk-Forward 窗口
    windows = generate_windows(
        start_year=2024, start_month=1,
        end_year=2026, end_month=2,
        is_months=6, oos_months=1, step_months=1
    )
    print(f"\n  Walk-Forward 窗口: {len(windows)} 个")
    for i, (is_s, is_e, oos_s, oos_e) in enumerate(windows):
        print(f"    W{i+1:02d}: IS={is_s}~{is_e} | OOS={oos_s}~{oos_e}")

    # 配置
    cfg = load_base_config()
    cfg.update(V8_OVERRIDES)
    cfg.update(V10_2_PRODUCTION)

    # 运行 Walk-Forward
    print(f"\n{'='*170}")
    print(f"  运行 Walk-Forward ({len(windows)} 个窗口)...")
    print(f"{'='*170}")

    oos_results = []
    for i, (is_s, is_e, oos_s, oos_e) in enumerate(windows):
        is_start_dt = pd.Timestamp(is_s)
        is_end_dt = pd.Timestamp(is_e) + pd.Timedelta(days=3)  # 月末缓冲
        oos_start_dt = pd.Timestamp(oos_s)
        oos_end_dt = pd.Timestamp(oos_e) + pd.Timedelta(days=3)

        # OOS 回测
        print(f"  W{i+1:02d} OOS={oos_s}~{oos_e}...", end=' ', flush=True)
        try:
            oos_r = run_single_window(
                all_data, all_signals, needed_tfs, cfg,
                primary_tf, decision_tfs, oos_start_dt, oos_end_dt
            )
            oos_results.append({
                'window': i + 1,
                'oos_start': oos_s,
                'oos_end': oos_e,
                **oos_r,
            })
            status = "盈利" if oos_r['strategy_return'] > 0 else "亏损"
            print(f"Ret={oos_r['strategy_return']:+.1f}% WR={oos_r['win_rate']:.0f}% "
                  f"PF={oos_r['profit_factor']:.2f} T={oos_r['total_trades']} [{status}]")
        except Exception as e:
            print(f"错误: {e}")
            oos_results.append({
                'window': i + 1, 'oos_start': oos_s, 'oos_end': oos_e,
                'strategy_return': 0, 'win_rate': 0, 'profit_factor': 0,
                'total_trades': 0, 'calmar': 0, 'sharpe': 0,
                'gross_profit': 0, 'gross_loss': 0, 'pnls': [],
                'max_drawdown': 0,
            })

    # ── 汇总 ──
    print(f"\n{'='*170}")
    print(f"  Walk-Forward OOS 窗口汇总")
    print(f"{'='*170}")
    print(f"  {'W#':>4s} {'OOS期间':>20s} {'Ret':>8s} {'WR':>6s} {'PF':>6s} "
          f"{'MDD':>7s} {'Trades':>7s} {'状态':>6s}")
    print(f"  {'-'*70}")

    profitable_windows = 0
    total_windows = len(oos_results)
    oos_returns = []
    all_oos_pnls = []

    for r in oos_results:
        ret = r['strategy_return']
        oos_returns.append(ret)
        all_oos_pnls.extend(r.get('pnls', []))
        status = "盈利" if ret > 0 else "亏损" if ret < 0 else "平手"
        if ret > 0:
            profitable_windows += 1
        print(f"  W{r['window']:02d}  {r['oos_start']}~{r['oos_end']}  "
              f"{ret:+7.1f}% {r['win_rate']:5.0f}% {r['profit_factor']:6.2f} "
              f"{r.get('max_drawdown', 0):6.1f}% {r['total_trades']:6d}  {status}")

    # 汇总统计
    win_pct = profitable_windows / total_windows * 100 if total_windows > 0 else 0
    avg_ret = np.mean(oos_returns) if oos_returns else 0
    median_ret = np.median(oos_returns) if oos_returns else 0
    total_ret = sum(oos_returns)
    cumulative_ret = np.prod([1 + r / 100 for r in oos_returns]) * 100 - 100 if oos_returns else 0

    print(f"\n  === Walk-Forward 汇总 ===")
    print(f"  盈利窗口: {profitable_windows}/{total_windows} ({win_pct:.0f}%)")
    print(f"  平均月度 Ret: {avg_ret:+.1f}%")
    print(f"  中位数月度 Ret: {median_ret:+.1f}%")
    print(f"  累计 Ret: {total_ret:+.1f}% (简单加和)")
    print(f"  复利 Ret: {cumulative_ret:+.1f}%")

    # ── DSR / PSR 计算 ──
    print(f"\n{'='*170}")
    print(f"  Phase 3b: Deflated Sharpe Ratio / PSR 统计验证")
    print(f"{'='*170}")

    if len(oos_returns) >= 3:
        oos_arr = np.array(oos_returns) / 100  # 转为小数
        sr = compute_sharpe_ratio(oos_arr)
        skew = float(pd.Series(oos_arr).skew())
        kurt = float(pd.Series(oos_arr).kurtosis()) + 3  # pandas kurtosis 是 excess

        psr = compute_psr(sr, len(oos_arr), skew, kurt, benchmark_sharpe=0)

        # DSR: 假设历史 ~45 次实验
        n_experiments = 45
        dsr = compute_dsr(sr, len(oos_arr), n_experiments, skew, kurt)

        print(f"\n  OOS Sharpe Ratio (年化): {sr:.2f}")
        print(f"  OOS 收益偏度: {skew:.2f}")
        print(f"  OOS 收益峰度: {kurt:.2f}")
        print(f"  样本数: {len(oos_arr)} 个月度窗口")
        print(f"\n  PSR (P(SR > 0)): {psr*100:.1f}%")
        print(f"  DSR (校正 {n_experiments} 次实验后): {dsr*100:.1f}%")

        # 解读
        print(f"\n  === 统计解读 ===")
        if dsr >= 0.95:
            print(f"  DSR >= 95%: 策略在多重检验后仍然统计显著，alpha 可信")
        elif dsr >= 0.80:
            print(f"  DSR 80-95%: 有一定统计支持，但需要更多 OOS 数据确认")
        elif dsr >= 0.50:
            print(f"  DSR 50-80%: 统计支持薄弱，可能存在过拟合风险")
        else:
            print(f"  DSR < 50%: 策略可能不具备真实 alpha，建议谨慎")

        if win_pct >= 60:
            print(f"  WF 窗口胜率 {win_pct:.0f}% >= 60%: 通过 Walk-Forward 验证")
        else:
            print(f"  WF 窗口胜率 {win_pct:.0f}% < 60%: 未通过 Walk-Forward 验证标准")
    else:
        print("  OOS 窗口数不足，无法计算 DSR/PSR")

    # Trade-level 统计
    if all_oos_pnls:
        pnl_arr = np.array(all_oos_pnls)
        print(f"\n  === Trade-Level OOS 统计 ===")
        print(f"  总交易数: {len(pnl_arr)}")
        print(f"  盈利交易: {(pnl_arr > 0).sum()} ({(pnl_arr > 0).mean()*100:.0f}%)")
        print(f"  平均 PnL: ${np.mean(pnl_arr):+,.0f}")
        print(f"  中位 PnL: ${np.median(pnl_arr):+,.0f}")
        if (pnl_arr > 0).sum() > 0 and (pnl_arr <= 0).sum() > 0:
            avg_win = np.mean(pnl_arr[pnl_arr > 0])
            avg_loss = np.mean(np.abs(pnl_arr[pnl_arr <= 0]))
            print(f"  Avg Win: ${avg_win:+,.0f}")
            print(f"  Avg Loss: ${avg_loss:,.0f}")
            print(f"  Win/Loss Ratio: {avg_win/avg_loss:.2f}")

    print(f"\n{'='*170}")
    print(f"  完成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*170}")


if __name__ == '__main__':
    main()
