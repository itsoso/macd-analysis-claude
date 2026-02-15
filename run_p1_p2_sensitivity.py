"""
P1: 核心参数 ±10% Monte Carlo 扰动 (50组) — 参数稳定性面
P2: short_trail 0.16-0.24 敏感度曲线（步长0.005）

在 2025-01~2026-01 (in-sample) 和 2024-01~2024-12 (OOS) 上都跑。
"""

import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from binance_fetcher import fetch_binance_klines
from indicators import add_all_indicators
from ma_indicators import add_moving_averages
from optimize_six_book import (
    _build_tf_score_index,
    compute_signals_six,
    run_strategy_multi_tf,
)
from backtest_multi_tf_30d_7d import _apply_conservative_risk
from run_p0_oos_validation import V6_OVERRIDES, load_base_config

# ============================================================
#  数据加载 (缓存)
# ============================================================
_DATA_CACHE = {}

def get_tf_data(symbol, tf, fetch_days, warmup_start, end_dt):
    key = (symbol, tf, str(warmup_start), str(end_dt))
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    df = fetch_binance_klines(symbol, interval=tf, days=fetch_days)
    if df is None or len(df) < 120:
        return None
    df = add_all_indicators(df)
    add_moving_averages(df, timeframe=tf)
    df = df[(df.index >= warmup_start) & (df.index <= end_dt)].copy()
    if len(df) < 120:
        return None
    _DATA_CACHE[key] = df
    return df


def run_single_backtest(all_data, all_signals, needed_tfs, cfg, primary_tf,
                        decision_tfs, start_dt, end_dt):
    """运行单次回测，返回核心指标"""
    tf_score_map = _build_tf_score_index(all_data, all_signals, needed_tfs, cfg)
    primary_df = all_data[primary_tf]
    result = run_strategy_multi_tf(
        primary_df=primary_df,
        tf_score_map=tf_score_map,
        decision_tfs=decision_tfs,
        config=cfg,
        primary_tf=primary_tf,
        trade_days=0,
        trade_start_dt=start_dt,
        trade_end_dt=end_dt,
    )

    # 从 trades 计算 WR/PF
    trades = result.get('trades', [])
    opens = [t for t in trades if t.get('action', '').startswith('OPEN_')]
    closes = [t for t in trades if t.get('action', '').startswith('CLOSE_')]

    n_trades = len(closes)
    wins = sum(1 for t in closes if float(t.get('pnl', 0)) > 0)
    wr = wins / n_trades * 100 if n_trades > 0 else 0
    gross_profit = sum(float(t.get('pnl', 0)) for t in closes if float(t.get('pnl', 0)) > 0)
    gross_loss = abs(sum(float(t.get('pnl', 0)) for t in closes if float(t.get('pnl', 0)) <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    return {
        'strategy_return': result.get('strategy_return', 0),
        'max_drawdown': result.get('max_drawdown', 0),
        'total_trades': n_trades,
        'win_rate': wr,
        'profit_factor': pf,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
    }


def prepare_data(symbol, start_date, end_date, warmup_days=60):
    """预加载数据和信号"""
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    warmup_start = start_dt - pd.Timedelta(days=warmup_days)
    now = pd.Timestamp.now().tz_localize(None)
    fetch_days = max(90, int((now - warmup_start).days + 5))

    primary_tf = '1h'
    decision_tfs = ['15m', '1h', '4h', '12h']
    needed_tfs = sorted(set([primary_tf] + decision_tfs))

    all_data = {}
    for tf in needed_tfs:
        df = get_tf_data(symbol, tf, fetch_days, warmup_start, end_dt)
        if df is None:
            raise RuntimeError(f"{tf} 数据不足")
        all_data[tf] = df

    base_cfg = load_base_config()
    all_signals = {
        tf: compute_signals_six(all_data[tf], tf, all_data, max_bars=0)
        for tf in needed_tfs
    }

    return all_data, all_signals, needed_tfs, primary_tf, decision_tfs, start_dt, end_dt


# ============================================================
#  P1: Monte Carlo 参数扰动
# ============================================================
def run_p1_monte_carlo(n_trials=50):
    """Monte Carlo 参数扰动测试"""
    print("\n" + "="*100)
    print("  P1: Monte Carlo 参数扰动 (±10%)")
    print("="*100)

    # 要扰动的核心参数及其 v6.0 基准值
    perturbable_params = {
        'short_threshold': 40,
        'long_threshold': 25,
        'short_sl': -0.20,
        'short_trail': 0.19,
        'trail_pullback': 0.50,
        'short_tp': 0.60,
        'long_sl': -0.10,
        'long_tp': 0.40,
        'long_trail': 0.12,
        'div_weight': 0.55,
        'neutral_struct_discount_0': 0.10,
        'neutral_struct_discount_1': 0.20,
        'short_conflict_discount_mult': 0.60,
    }

    # 准备数据（in-sample）
    print("  加载 in-sample 数据 (2025-01 ~ 2026-01)...")
    is_data, is_signals, needed_tfs, primary_tf, decision_tfs, is_start, is_end = \
        prepare_data('ETHUSDT', '2025-01-01', '2026-01-31')

    print("  加载 OOS 数据 (2024-01 ~ 2024-12)...")
    oos_data, oos_signals, _, _, _, oos_start, oos_end = \
        prepare_data('ETHUSDT', '2024-01-01', '2024-12-31')

    # 基准
    base_cfg = load_base_config()
    print("  运行基准 (v6.0 baseline)...")
    base_is = run_single_backtest(is_data, is_signals, needed_tfs, base_cfg,
                                   primary_tf, decision_tfs, is_start, is_end)
    base_oos = run_single_backtest(oos_data, oos_signals, needed_tfs, base_cfg,
                                    primary_tf, decision_tfs, oos_start, oos_end)

    print(f"  基准 IS:  Ret={base_is['strategy_return']:+.1f}% WR={base_is['win_rate']:.1f}% PF={base_is['profit_factor']:.2f} MDD={base_is['max_drawdown']:.1f}%")
    print(f"  基准 OOS: Ret={base_oos['strategy_return']:+.1f}% WR={base_oos['win_rate']:.1f}% PF={base_oos['profit_factor']:.2f} MDD={base_oos['max_drawdown']:.1f}%")

    # Monte Carlo
    random.seed(42)
    results = []
    for trial in range(n_trials):
        cfg = load_base_config()
        perturbations = {}
        for param, base_val in perturbable_params.items():
            # ±10% 均匀扰动
            pct = random.uniform(-0.10, 0.10)
            new_val = base_val * (1 + pct)
            # 整数参数取整
            if param in ('short_threshold', 'long_threshold'):
                new_val = round(new_val)
            cfg[param] = new_val
            perturbations[param] = new_val

        try:
            is_result = run_single_backtest(is_data, is_signals, needed_tfs, cfg,
                                             primary_tf, decision_tfs, is_start, is_end)
            oos_result = run_single_backtest(oos_data, oos_signals, needed_tfs, cfg,
                                              primary_tf, decision_tfs, oos_start, oos_end)
            results.append({
                'trial': trial,
                **{f'p_{k}': v for k, v in perturbations.items()},
                'is_ret': is_result['strategy_return'],
                'is_wr': is_result['win_rate'],
                'is_pf': is_result['profit_factor'],
                'is_mdd': is_result['max_drawdown'],
                'is_trades': is_result['total_trades'],
                'oos_ret': oos_result['strategy_return'],
                'oos_wr': oos_result['win_rate'],
                'oos_pf': oos_result['profit_factor'],
                'oos_mdd': oos_result['max_drawdown'],
                'oos_trades': oos_result['total_trades'],
            })
            if (trial + 1) % 10 == 0:
                print(f"  [{trial+1}/{n_trials}] IS Ret={is_result['strategy_return']:+.1f}% OOS Ret={oos_result['strategy_return']:+.1f}%")
        except Exception as e:
            print(f"  Trial {trial} failed: {e}")

    # 分析
    rdf = pd.DataFrame(results)
    print(f"\n  Monte Carlo 结果 ({len(rdf)}/{n_trials} 成功):")
    print(f"  {'指标':<12} {'IS p5':>8} {'IS p25':>8} {'IS p50':>8} {'IS p75':>8} {'IS p95':>8} | {'OOS p5':>8} {'OOS p50':>8} {'OOS p95':>8}")
    print(f"  {'-'*95}")
    for metric, label in [('ret', '收益%'), ('wr', 'WR%'), ('pf', 'PF'), ('mdd', 'MDD%')]:
        is_col = f'is_{metric}'
        oos_col = f'oos_{metric}'
        isp = rdf[is_col].quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        oosp = rdf[oos_col].quantile([0.05, 0.50, 0.95])
        print(f"  {label:<12} {isp.iloc[0]:>+7.1f} {isp.iloc[1]:>+7.1f} {isp.iloc[2]:>+7.1f} {isp.iloc[3]:>+7.1f} {isp.iloc[4]:>+7.1f} | {oosp.iloc[0]:>+7.1f} {oosp.iloc[1]:>+7.1f} {oosp.iloc[2]:>+7.1f}")

    # 参数敏感度分析（相关性）
    print(f"\n  参数 vs IS_Ret 相关性（敏感度排序）:")
    corrs = []
    for param in perturbable_params:
        col = f'p_{param}'
        if col in rdf.columns:
            r = rdf[col].corr(rdf['is_ret'])
            corrs.append((param, r))
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for param, r in corrs:
        bar = '█' * int(abs(r) * 50)
        print(f"    {param:<35} r={r:+.3f} {bar}")

    return rdf, base_is, base_oos


# ============================================================
#  P2: short_trail 敏感度曲线
# ============================================================
def run_p2_trail_sensitivity():
    """short_trail 0.16-0.24 敏感度曲线"""
    print("\n" + "="*100)
    print("  P2: short_trail 敏感度曲线 (0.16 ~ 0.24, 步长 0.005)")
    print("="*100)

    is_data, is_signals, needed_tfs, primary_tf, decision_tfs, is_start, is_end = \
        prepare_data('ETHUSDT', '2025-01-01', '2026-01-31')
    oos_data, oos_signals, _, _, _, oos_start, oos_end = \
        prepare_data('ETHUSDT', '2024-01-01', '2024-12-31')

    trail_values = np.arange(0.16, 0.245, 0.005)
    results = []

    for trail in trail_values:
        trail = round(trail, 3)
        cfg = load_base_config()
        cfg['short_trail'] = trail

        is_result = run_single_backtest(is_data, is_signals, needed_tfs, cfg,
                                         primary_tf, decision_tfs, is_start, is_end)
        oos_result = run_single_backtest(oos_data, oos_signals, needed_tfs, cfg,
                                          primary_tf, decision_tfs, oos_start, oos_end)

        results.append({
            'short_trail': trail,
            'is_ret': is_result['strategy_return'],
            'is_wr': is_result['win_rate'],
            'is_pf': is_result['profit_factor'],
            'is_mdd': is_result['max_drawdown'],
            'is_trades': is_result['total_trades'],
            'oos_ret': oos_result['strategy_return'],
            'oos_wr': oos_result['win_rate'],
            'oos_pf': oos_result['profit_factor'],
            'oos_mdd': oos_result['max_drawdown'],
            'oos_trades': oos_result['total_trades'],
        })
        print(f"  trail={trail:.3f} | IS: Ret={is_result['strategy_return']:+.1f}% WR={is_result['win_rate']:.1f}% | OOS: Ret={oos_result['strategy_return']:+.1f}% WR={oos_result['win_rate']:.1f}%")

    rdf = pd.DataFrame(results)
    print(f"\n  short_trail 敏感度汇总:")
    print(f"  {'trail':>6} {'IS_Ret':>8} {'IS_WR':>7} {'IS_PF':>7} {'OOS_Ret':>8} {'OOS_WR':>7} {'OOS_PF':>7} {'悬崖?':>6}")
    print(f"  {'-'*58}")
    for _, row in rdf.iterrows():
        # 检测悬崖: 与上一行的IS_Ret差超过5%
        cliff = ''
        idx = rdf.index[rdf['short_trail'] == row['short_trail']].tolist()[0]
        if idx > 0:
            prev_ret = rdf.iloc[idx-1]['is_ret']
            diff = abs(row['is_ret'] - prev_ret)
            if diff > 5:
                cliff = f'⚠{diff:.0f}%'
        print(f"  {row['short_trail']:>6.3f} {row['is_ret']:>+7.1f}% {row['is_wr']:>6.1f}% {row['is_pf']:>6.2f} {row['oos_ret']:>+7.1f}% {row['oos_wr']:>6.1f}% {row['oos_pf']:>6.2f} {cliff}")

    return rdf


def main():
    export_dir = 'data/backtests/p1_p2_sensitivity'
    os.makedirs(export_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # P2 先跑（快，17个点）
    p2_df = run_p2_trail_sensitivity()
    p2_csv = os.path.join(export_dir, f'p2_trail_sensitivity_{ts}.csv')
    p2_df.to_csv(p2_csv, index=False)
    print(f"\n  P2 结果已保存: {p2_csv}")

    # P1 Monte Carlo（慢，50组×2区间）
    p1_df, base_is, base_oos = run_p1_monte_carlo(n_trials=50)
    p1_csv = os.path.join(export_dir, f'p1_monte_carlo_{ts}.csv')
    p1_df.to_csv(p1_csv, index=False)
    print(f"\n  P1 结果已保存: {p1_csv}")

    print("\n" + "="*100)
    print("  P1+P2 完成")
    print("="*100)


if __name__ == '__main__':
    main()
