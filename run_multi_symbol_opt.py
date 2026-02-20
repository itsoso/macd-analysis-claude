#!/usr/bin/env python3
"""
多标的跨品种参数优化器

目标: 在 4 标的 × 5 年数据上寻找能泛化的参数组合。
方法: 对每组候选参数, 在所有标的上回测, 以聚合 PF 为优化目标。

用法:
    python run_multi_symbol_opt.py                    # 默认 4 标的
    python run_multi_symbol_opt.py --symbols ETHUSDT,BTCUSDT  # 指定标的
    python run_multi_symbol_opt.py --quick             # 快速模式 (少量变体)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

from optimize_six_book import (
    fetch_multi_tf_data,
    compute_signals_six,
    run_strategy,
)
from signal_core import calc_fusion_score_six

# ─── 默认配置 ───────────────────────────────────────
DEFAULT_SYMBOLS = ['ETHUSDT', 'BTCUSDT', 'SOLUSDT', 'BNBUSDT']
TF = '1h'
DAYS = 1900  # ~5 years

# 基线参数 (当前 ETH 最优)
BASELINE = {
    'name': 'baseline',
    'fusion_mode': 'c6_veto_4', 'veto_threshold': 25,
    'single_pct': 0.20, 'total_pct': 0.50, 'lifetime_pct': 5.0,
    'sell_threshold': 18, 'buy_threshold': 25,
    'short_threshold': 25, 'long_threshold': 40,
    'close_short_bs': 40, 'close_long_ss': 40,
    'sell_pct': 0.55, 'margin_use': 0.70, 'lev': 5, 'max_lev': 5,
    'short_sl': -0.15, 'long_sl': -0.08,
    'short_tp': 0.30, 'long_tp': 0.30,
    'short_trail': 0.25, 'long_trail': 0.25,
    'trail_pullback': 0.60,
    'short_max_hold': 72, 'long_max_hold': 72,
    'cooldown': 6,
    'use_atr_sl': True, 'atr_sl_mult': 3.5,
    'use_soft_veto': True, 'soft_veto_steepness': 5.0, 'soft_veto_midpoint': 25.0,
    'use_regime_aware': True,
    'initial_usdt': 100000, 'initial_eth_value': 0,
}


def build_param_grid(quick=False):
    """构建聚焦参数网格 — 只调最敏感的参数。

    Quick (Phase 1): 4 维粗扫 → 定位 long/short/atr/cooldown 最优区域
    Full  (Phase 2): 基于 Phase 1 结论, 锁定已确认参数, 扩展新维度
    """
    if quick:
        # Phase 1: 粗扫 4 个核心维度 (36 组合)
        grid = {
            'long_threshold':  [30, 40, 50],
            'short_threshold': [20, 30],
            'atr_sl_mult':     [2.5, 3.0, 3.5],
            'cooldown':        [4, 6],
        }
    else:
        # Phase 2: 基于 Phase 1 结论 (lon=40, sho=20, atr=3.5, cd=4 最优)
        # 锁定 ATR=3.5 (不敏感), 精细扫描 long/short 邻域 + 3 个新维度
        # 4 × 3 × 4 × 4 × 3 × 2 = 1152 组合, 约 19 小时 (2yr)
        grid = {
            'long_threshold':  [35, 38, 40, 42],
            'short_threshold': [18, 20, 22],
            'sell_threshold':  [14, 16, 18, 20],
            'trail_pullback':  [0.40, 0.50, 0.60, 0.70],
            'cooldown':        [3, 4, 5],
            'lev':             [3, 5],
        }

    keys = list(grid.keys())
    vals = list(grid.values())
    combos = list(product(*vals))

    variants = []
    for combo in combos:
        override = dict(zip(keys, combo))
        cfg = {**BASELINE, **override}
        cfg['name'] = '_'.join(f'{k[:3]}={v}' for k, v in override.items())
        cfg['max_lev'] = cfg['lev']
        variants.append(cfg)

    return variants


def extract_metrics(result):
    """从回测结果中提取关键指标。"""
    trades = result.get('trades', [])
    # 从 trade log 中提取 close 动作的 PnL
    close_pnls = [t.get('pnl', 0) for t in trades
                  if t.get('action', '').startswith('CLOSE') and t.get('pnl', 0) != 0]

    gross_profit = sum(p for p in close_pnls if p > 0)
    gross_loss = abs(sum(p for p in close_pnls if p < 0))

    return {
        'alpha': result.get('alpha', 0),
        'return_pct': result.get('strategy_return', 0),
        'total_trades': result.get('total_trades', 0),
        'max_drawdown': result.get('max_drawdown', 0),
        'wins': sum(1 for p in close_pnls if p > 0),
        'losses': sum(1 for p in close_pnls if p < 0),
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'pf': gross_profit / gross_loss if gross_loss > 0 else 0,
    }


def run_multi_symbol(cfg, symbol_data, symbol_signals):
    """在多个标的上运行同一参数, 返回聚合指标。"""
    per_symbol = {}
    total_profit = 0
    total_loss = 0
    total_trades = 0
    total_wins = 0
    total_losses = 0
    alphas = []

    for sym in symbol_data:
        r = run_strategy(
            symbol_data[sym], symbol_signals[sym], cfg,
            tf=TF, trade_days=DAYS,
        )
        m = extract_metrics(r)
        per_symbol[sym] = m

        total_profit += m['gross_profit']
        total_loss += m['gross_loss']
        total_trades += m['total_trades']
        total_wins += m['wins']
        total_losses += m['losses']
        alphas.append(m['alpha'])

    agg_pf = total_profit / total_loss if total_loss > 0 else 0
    wr = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0

    return {
        'agg_pf': agg_pf,
        'mean_alpha': np.mean(alphas),
        'min_alpha': np.min(alphas),
        'total_trades': total_trades,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'win_rate': wr,
        'per_symbol': per_symbol,
    }


def main():
    parser = argparse.ArgumentParser(description="多标的跨品种参数优化")
    parser.add_argument("--symbols", default=','.join(DEFAULT_SYMBOLS),
                        help="逗号分隔的标的列表")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式 (少量变体)")
    parser.add_argument("--days", type=int, default=DAYS,
                        help="回测天数")
    parser.add_argument("--output-dir", default="logs/multi_symbol_opt")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(',')]

    print("=" * 80)
    print(f"  多标的跨品种参数优化器")
    print(f"  标的: {', '.join(symbols)}")
    print(f"  周期: {TF} | 天数: {args.days}")
    print("=" * 80)

    # ── 加载数据 + 预计算信号 ──
    print(f"\n[1/3] 加载数据 & 计算信号...")
    symbol_data = {}
    symbol_signals = {}

    for sym in symbols:
        print(f"\n  {sym}:")
        data = fetch_multi_tf_data([TF], days=args.days, symbol=sym)
        if TF not in data:
            print(f"    ✗ 无数据, 跳过")
            continue
        symbol_data[sym] = data[TF]
        print(f"    计算六书信号...")
        symbol_signals[sym] = compute_signals_six(data[TF], TF, data)
        print(f"    ✓ {len(data[TF])} bars")

    if len(symbol_data) < 2:
        print("\n✗ 需要至少 2 个标的的数据"); sys.exit(1)

    # ── Baseline ──
    print(f"\n[2/3] Baseline 回测...")
    bl = run_multi_symbol(BASELINE, symbol_data, symbol_signals)
    print(f"  Baseline: agg_PF={bl['agg_pf']:.3f}, mean_α={bl['mean_alpha']:+.1f}%, "
          f"trades={bl['total_trades']}, WR={bl['win_rate']:.1%}")
    for sym, m in bl['per_symbol'].items():
        print(f"    {sym}: α={m['alpha']:+.1f}%, PF={m['pf']:.3f}, "
              f"trades={m['total_trades']}, W/L={m['wins']}/{m['losses']}")

    # ── 参数扫描 ──
    variants = build_param_grid(quick=args.quick)
    print(f"\n[3/3] 参数扫描: {len(variants)} 组合 × {len(symbol_data)} 标的 "
          f"= {len(variants) * len(symbol_data)} 次回测")

    results = []
    t0 = time.time()

    for i, cfg in enumerate(variants, 1):
        r = run_multi_symbol(cfg, symbol_data, symbol_signals)
        results.append({
            'name': cfg['name'],
            'config': {k: v for k, v in cfg.items() if k != 'name'},
            **r,
        })

        elapsed = time.time() - t0
        eta = elapsed / i * (len(variants) - i)

        # 每 10 个打印一次进度, 或者发现更好的
        if i % 10 == 0 or i == len(variants):
            print(f"  [{i:>4d}/{len(variants)}] {cfg['name'][:40]:40s} | "
                  f"PF={r['agg_pf']:.3f} α={r['mean_alpha']:+.1f}% "
                  f"trades={r['total_trades']} WR={r['win_rate']:.1%} "
                  f"| ETA {eta/60:.0f}min")

    elapsed_total = time.time() - t0

    # ── 排序 + 输出 ──
    # 主排序: 聚合 PF, 次排序: mean alpha
    results.sort(key=lambda x: (x['agg_pf'], x['mean_alpha']), reverse=True)

    print(f"\n{'=' * 80}")
    print(f"  TOP 20 参数组合 (按聚合 PF 排序)")
    print(f"{'=' * 80}")
    print(f"{'Rank':>4} {'Name':40s} {'Agg_PF':>7} {'Mean_α':>8} {'Min_α':>8} "
          f"{'Trades':>7} {'WR':>6}")
    print("-" * 80)

    for rank, r in enumerate(results[:20], 1):
        print(f"{rank:>4} {r['name'][:40]:40s} {r['agg_pf']:>7.3f} "
              f"{r['mean_alpha']:>+7.1f}% {r['min_alpha']:>+7.1f}% "
              f"{r['total_trades']:>7} {r['win_rate']:>5.1%}")

    # ── vs Baseline ──
    best = results[0]
    print(f"\n{'=' * 80}")
    print(f"  最优 vs Baseline")
    print(f"{'=' * 80}")
    print(f"  最优: {best['name']}")
    print(f"  聚合 PF:  {best['agg_pf']:.3f}  (Baseline: {bl['agg_pf']:.3f}, "
          f"Δ={best['agg_pf'] - bl['agg_pf']:+.3f})")
    print(f"  平均 α:   {best['mean_alpha']:+.1f}%  (Baseline: {bl['mean_alpha']:+.1f}%)")
    print(f"  最差标的:  {best['min_alpha']:+.1f}%  (Baseline: {bl['min_alpha']:+.1f}%)")
    print(f"  总交易数:  {best['total_trades']}  (Baseline: {bl['total_trades']})")
    print(f"  胜率:      {best['win_rate']:.1%}  (Baseline: {bl['win_rate']:.1%})")

    print(f"\n  逐标的对比:")
    for sym in symbol_data:
        bm = bl['per_symbol'].get(sym, {})
        om = best['per_symbol'].get(sym, {})
        print(f"    {sym}: PF {bm.get('pf',0):.3f}→{om.get('pf',0):.3f}, "
              f"α {bm.get('alpha',0):+.1f}%→{om.get('alpha',0):+.1f}%, "
              f"W/L {bm.get('wins',0)}/{bm.get('losses',0)}"
              f"→{om.get('wins',0)}/{om.get('losses',0)}")

    # ── 保存结果 ──
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(args.output_dir, f'multi_symbol_opt_{ts}.json')

    def _clean(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return str(obj)

    output = {
        'timestamp': ts,
        'symbols': symbols,
        'timeframe': TF,
        'days': args.days,
        'num_variants': len(variants),
        'elapsed_seconds': round(elapsed_total, 1),
        'baseline': {
            'config': {k: v for k, v in BASELINE.items() if k != 'name'},
            **bl,
        },
        'top_20': results[:20],
        'all_results_count': len(results),
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, default=_clean, indent=2)
    print(f"\n结果已保存: {out_path}")
    print(f"总耗时: {elapsed_total/60:.1f} 分钟")


if __name__ == '__main__':
    main()
