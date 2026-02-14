#!/usr/bin/env python3
"""
策略参数快速扫描
==================
信号计算 (~30s) 只做一次, 参数扫描在 score_index + strategy_run (~4s/组) 上迭代。
50 组参数 ≈ 3.5 分钟。

用法:
    python param_sweep.py
    python param_sweep.py --start 2025-01-01 --end 2026-01-31
"""

import argparse
import itertools
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import add_all_indicators
from ma_indicators import add_moving_averages
from signal_core import compute_signals_six_multiprocess
from live_config import StrategyConfig
from kline_store import load_klines
from optimize_six_book import _build_tf_score_index, run_strategy_multi_tf
from signal_core import calc_fusion_score_six_batch

# ── 默认参数 (来自 backtest_multi_tf_daily.py) ──
_LIVE = StrategyConfig()
PRIMARY_TF = _LIVE.timeframe  # '1h'
DECISION_TFS = list(_LIVE.decision_timeframes)
FALLBACK_TFS = list(_LIVE.decision_timeframes_fallback)

TF_HOURS = {
    '10m': 1/6, '15m': 0.25, '30m': 0.5, '1h': 1, '2h': 2, '3h': 3,
    '4h': 4, '6h': 6, '8h': 8, '12h': 12, '16h': 16, '24h': 24,
}


def build_base_config():
    """基线配置 (与 backtest_multi_tf_daily.py 完全一致)"""
    from backtest_multi_tf_daily import _build_default_config
    return _build_default_config()


def scale_config(base, primary_tf):
    """按主周期缩放 hold/cooldown"""
    config = dict(base)
    tf_h = TF_HOURS.get(primary_tf, 1)
    config['short_max_hold'] = max(6, int(config.get('short_max_hold', 72) / tf_h))
    config['long_max_hold'] = max(6, int(config.get('long_max_hold', 72) / tf_h))
    config['cooldown'] = max(1, int(config.get('cooldown', 4) / tf_h))
    config['spot_cooldown'] = max(2, int(config.get('spot_cooldown', 12) / tf_h))
    return config


def run_with_params(all_data, all_signals, score_tfs, decision_tfs,
                    primary_tf, trade_start, trade_end, overrides):
    """用指定参数覆盖运行一次策略, 返回关键指标"""
    base = build_base_config()
    base.update(overrides)
    config = scale_config(base, primary_tf)
    config['name'] = 'param_sweep'

    # 重建评分索引 (fusion weights 可能变了)
    tf_score_index = _build_tf_score_index(all_data, all_signals, score_tfs, config)

    # 运行策略
    result = run_strategy_multi_tf(
        primary_df=all_data[primary_tf],
        tf_score_map=tf_score_index,
        decision_tfs=decision_tfs,
        config=config,
        primary_tf=primary_tf,
        trade_days=0,
        trade_start_dt=pd.Timestamp(trade_start),
        trade_end_dt=pd.Timestamp(trade_end) + pd.Timedelta(hours=23, minutes=59),
    )

    # 提取指标
    trades = result.get('trades', [])

    # contract_pf: 仅合约平仓
    close_actions = {'CLOSE_LONG', 'CLOSE_SHORT', 'LIQUIDATED'}
    close_trades = [t for t in trades if t.get('action') in close_actions]
    wins = [t for t in close_trades if (t.get('pnl') or 0) > 0]
    losses = [t for t in close_trades if (t.get('pnl') or 0) <= 0]

    total_pnl_w = sum(t['pnl'] for t in wins) if wins else 0
    total_pnl_l = abs(sum(t['pnl'] for t in losses)) if losses else 1
    contract_pf = total_pnl_w / total_pnl_l if total_pnl_l > 0 else 0

    # portfolio_pf: 含 PARTIAL_TP / SPOT_SELL
    pnl_actions = {'CLOSE_LONG', 'CLOSE_SHORT', 'LIQUIDATED', 'PARTIAL_TP', 'SPOT_SELL'}
    all_pnl = [t for t in trades if t.get('action') in pnl_actions and t.get('pnl') is not None]
    gp = sum(t['pnl'] for t in all_pnl if (t.get('pnl') or 0) > 0)
    gl = abs(sum(t['pnl'] for t in all_pnl if (t.get('pnl') or 0) < 0))
    portfolio_pf = round(gp / gl, 2) if gl > 0 else 999

    return {
        'return_pct': result.get('strategy_return', 0) * 100,
        'max_dd_pct': result.get('max_drawdown', 0) * 100,
        'total_trades': len(trades),
        'closed': len(close_trades),
        'win_rate': len(wins) / len(close_trades) * 100 if close_trades else 0,
        'profit_factor': contract_pf,
        'portfolio_pf': portfolio_pf,
        'final_equity': result.get('final_total', 0),
        'alpha': result.get('alpha', 0) * 100,
        'fees': result.get('fees', {}).get('total_fees', 0),
    }


# ── 参数空间定义 ──

PARAM_SPACE = {
    # === 优先级 0: 已编码功能开关 A/B (零开发成本) ===
    'use_partial_tp_v3': [False, True],                  # 基线 True — v3早期锁利
    'use_atr_sl': [False, True],                         # 基线 False — ATR自适应止损
    # [已移除] use_short_suppress — A/B+param_sweep验证零效果
    'use_spot_sell_cap': [False, True],                  # 基线 False — SPOT_SELL比例上限
    'use_regime_short_gate': [False, True],              # 基线 False — regime做空门控

    # === 优先级 1: 空头风控 ===
    'short_sl': [-0.12, -0.15, -0.18, -0.25],          # 基线 -0.18
    'short_threshold': [25, 28, 30, 35],                 # 基线 35
    'short_max_hold': [36, 48, 60, 72],                  # 基线 48

    # === 优先级 2: 做多机会 ===
    'long_threshold': [25, 30, 35, 40],                  # 基线 30
    'long_sl': [-0.06, -0.08, -0.10],                   # 基线 -0.08

    # === 优先级 3: 止盈优化 ===
    'short_tp': [0.40, 0.50, 0.60],                     # 基线 0.50
    'long_tp': [0.25, 0.30, 0.40],                      # 基线 0.40
    'short_trail': [0.15, 0.20, 0.25],                  # 基线 0.25
    'long_trail': [0.12, 0.15, 0.20],                   # 基线 0.20
    'trail_pullback': [0.50, 0.55, 0.60],               # 基线 0.60

    # === 优先级 4: 部分止盈 ===
    'partial_tp_1': [0.12, 0.15, 0.20, 0.25],           # 基线 0.15
    'use_partial_tp_2': [False, True],                   # 基线 True
}


def single_param_sweep(all_data, all_signals, score_tfs, decision_tfs,
                       primary_tf, trade_start, trade_end):
    """逐维扫描: 每次只改一个参数, 其他保持基线"""
    base = build_base_config()
    results = []

    # 先跑基线
    print("\n" + "=" * 80)
    print("  基线")
    print("=" * 80)
    t0 = time.time()
    baseline = run_with_params(all_data, all_signals, score_tfs, decision_tfs,
                               primary_tf, trade_start, trade_end, {})
    elapsed = time.time() - t0
    print(f"  基线: 收益={baseline['return_pct']:+.2f}%  DD={baseline['max_dd_pct']:.2f}%  "
          f"WR={baseline['win_rate']:.1f}%  cPF={baseline['profit_factor']:.2f}  "
          f"pPF={baseline['portfolio_pf']:.2f}  "
          f"交易={baseline['total_trades']}  [{elapsed:.1f}s]")
    results.append(('BASELINE', '', baseline))

    # 逐维扫描
    for param_name, values in PARAM_SPACE.items():
        baseline_val = base.get(param_name)
        print(f"\n{'─' * 60}")
        print(f"  扫描: {param_name} (基线={baseline_val})")
        print(f"{'─' * 60}")
        for val in values:
            if val == baseline_val:
                continue  # 跳过基线值
            t0 = time.time()
            overrides = {param_name: val}
            try:
                metrics = run_with_params(all_data, all_signals, score_tfs, decision_tfs,
                                          primary_tf, trade_start, trade_end, overrides)
            except Exception as e:
                print(f"  {param_name}={val}: ERROR: {e}")
                continue
            elapsed = time.time() - t0

            # 与基线对比
            d_ret = metrics['return_pct'] - baseline['return_pct']
            d_dd = metrics['max_dd_pct'] - baseline['max_dd_pct']
            d_pf = metrics['profit_factor'] - baseline['profit_factor']

            d_ppf = metrics['portfolio_pf'] - baseline['portfolio_pf']
            marker = '★' if d_ret > 0 and d_dd <= 0 else ('▲' if d_ret > 0 else '▼')
            print(f"  {marker} {param_name}={val}: 收益={metrics['return_pct']:+.2f}%({d_ret:+.2f})  "
                  f"DD={metrics['max_dd_pct']:.2f}%({d_dd:+.2f})  "
                  f"WR={metrics['win_rate']:.1f}%  cPF={metrics['profit_factor']:.2f}({d_pf:+.2f})  "
                  f"pPF={metrics['portfolio_pf']:.2f}({d_ppf:+.2f})  "
                  f"交易={metrics['total_trades']}  [{elapsed:.1f}s]")
            results.append((param_name, val, metrics))

    return baseline, results


def combo_sweep(all_data, all_signals, score_tfs, decision_tfs,
                primary_tf, trade_start, trade_end, top_params):
    """组合扫描: 用单维扫描找到的最佳值组合测试"""
    print("\n" + "=" * 80)
    print("  组合扫描 — 最佳单维参数组合")
    print("=" * 80)

    # 基线
    baseline = run_with_params(all_data, all_signals, score_tfs, decision_tfs,
                               primary_tf, trade_start, trade_end, {})

    # 逐步累加最佳参数
    current_overrides = {}
    for param_name, best_val, improvement_desc in top_params:
        test_overrides = dict(current_overrides)
        test_overrides[param_name] = best_val
        t0 = time.time()
        metrics = run_with_params(all_data, all_signals, score_tfs, decision_tfs,
                                   primary_tf, trade_start, trade_end, test_overrides)
        elapsed = time.time() - t0

        d_ret = metrics['return_pct'] - baseline['return_pct']
        d_dd = metrics['max_dd_pct'] - baseline['max_dd_pct']
        print(f"  + {param_name}={best_val}: 收益={metrics['return_pct']:+.2f}%({d_ret:+.2f} vs 基线)  "
              f"DD={metrics['max_dd_pct']:.2f}%  cPF={metrics['profit_factor']:.2f}  "
              f"pPF={metrics['portfolio_pf']:.2f}  "
              f"交易={metrics['total_trades']}  [{elapsed:.1f}s]")

        # 只接受不恶化的参数
        if metrics['return_pct'] >= baseline['return_pct'] * 0.95 or d_ret > 0:
            current_overrides[param_name] = best_val
            print(f"    → 接受")
        else:
            print(f"    → 拒绝 (收益下降过多)")

    # 最终组合
    print(f"\n  最终参数组合: {current_overrides}")
    final = run_with_params(all_data, all_signals, score_tfs, decision_tfs,
                            primary_tf, trade_start, trade_end, current_overrides)
    d_ret = final['return_pct'] - baseline['return_pct']
    d_dd = final['max_dd_pct'] - baseline['max_dd_pct']
    print(f"\n  最终结果: 收益={final['return_pct']:+.2f}%({d_ret:+.2f})  "
          f"DD={final['max_dd_pct']:.2f}%({d_dd:+.2f})  "
          f"cPF={final['profit_factor']:.2f}  pPF={final['portfolio_pf']:.2f}  "
          f"交易={final['total_trades']}")
    return current_overrides, final


def main():
    parser = argparse.ArgumentParser(description='策略参数快速扫描')
    parser.add_argument('--start', type=str, default='2025-01-01')
    parser.add_argument('--end', type=str, default='2026-01-31')
    args = parser.parse_args()

    trade_start = args.start
    trade_end = args.end

    print("=" * 80)
    print(f"  策略参数扫描  |  {trade_start} ~ {trade_end}")
    print("=" * 80)

    # ── 1. 加载数据 ──
    print("\n[1/3] 加载数据...")
    t0 = time.time()
    from backtest_multi_tf_daily import fetch_data_for_tf
    all_data = {}
    tf_list = list(dict.fromkeys([PRIMARY_TF, *DECISION_TFS, *FALLBACK_TFS]))
    for tf in tf_list:
        try:
            df = fetch_data_for_tf(tf, days=500, allow_api_fallback=False)
            if df is not None:
                all_data[tf] = df
                print(f"  {tf}: {len(df)} bars")
        except Exception:
            pass

    # 确定可用的决策TF
    available_tfs = [tf for tf in DECISION_TFS if tf in all_data]
    if len(available_tfs) >= 3:
        decision_tfs = available_tfs
    else:
        decision_tfs = [tf for tf in FALLBACK_TFS if tf in all_data]
    score_tfs = list(dict.fromkeys([PRIMARY_TF, *decision_tfs]))
    print(f"  决策TFs: {', '.join(decision_tfs)}")
    print(f"  数据加载: {time.time() - t0:.1f}s")

    # ── 2. 计算信号 (只做一次) ──
    print("\n[2/3] 计算信号 (一次性)...")
    t0 = time.time()
    all_signals = compute_signals_six_multiprocess(all_data, score_tfs)
    signal_time = time.time() - t0
    print(f"  信号计算完成: {signal_time:.1f}s")

    # ── 3. 参数扫描 ──
    print("\n[3/3] 参数扫描...")
    t_sweep = time.time()
    baseline, results = single_param_sweep(
        all_data, all_signals, score_tfs, decision_tfs,
        PRIMARY_TF, trade_start, trade_end,
    )

    # 找出每个参数的最佳值
    print("\n" + "=" * 80)
    print("  单维扫描结果汇总")
    print("=" * 80)

    best_params = {}
    for param_name in PARAM_SPACE:
        param_results = [(v, m) for p, v, m in results if p == param_name]
        if not param_results:
            continue
        # 按综合评分排序: 收益提升 + 回撤改善 + portfolio_PF 提升
        scored = []
        for val, m in param_results:
            # 综合分 = 收益变化% + 回撤改善(负值=改善) * 5 + pPF变化 * 20
            d_ret = m['return_pct'] - baseline['return_pct']
            d_dd = m['max_dd_pct'] - baseline['max_dd_pct']
            d_ppf = m['portfolio_pf'] - baseline['portfolio_pf']
            composite = d_ret - d_dd * 5 + d_ppf * 20
            scored.append((val, m, composite, d_ret, d_dd, d_ppf))
        scored.sort(key=lambda x: x[2], reverse=True)
        best_val, best_m, comp, d_ret, d_dd, d_ppf = scored[0]
        if comp > 0:
            best_params[param_name] = (best_val, d_ret, d_dd, d_ppf, comp)
            print(f"  ★ {param_name}: {best_val} (收益{d_ret:+.2f}% DD{d_dd:+.2f}% pPF{d_ppf:+.2f} 综合{comp:+.1f})")
        else:
            print(f"  · {param_name}: 基线最优")

    # 组合扫描
    if best_params:
        # 按综合分排序
        sorted_params = sorted(best_params.items(), key=lambda x: x[1][4], reverse=True)
        top_params = [(name, val, f"综合{comp:+.1f}") for name, (val, _, _, _, comp) in sorted_params]

        print(f"\n  待组合参数 ({len(top_params)}个):")
        for name, val, desc in top_params:
            print(f"    {name}={val}  ({desc})")

        final_overrides, final_metrics = combo_sweep(
            all_data, all_signals, score_tfs, decision_tfs,
            PRIMARY_TF, trade_start, trade_end, top_params,
        )

        # 输出最终配置
        print("\n" + "=" * 80)
        print("  最终优化配置")
        print("=" * 80)
        base = build_base_config()
        for k, v in final_overrides.items():
            print(f"  {k}: {base.get(k)} → {v}")
    else:
        print("\n  未发现优于基线的参数组合")

    total_time = time.time() - t_sweep
    print(f"\n  参数扫描总耗时: {total_time:.1f}s ({len(results)} 组配置)")


if __name__ == '__main__':
    main()
