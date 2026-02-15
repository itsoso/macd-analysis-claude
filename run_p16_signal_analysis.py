#!/usr/bin/env python3
"""P16: 信号去相关/权重再平衡分析
分析六书信号之间的相关性矩阵，识别冗余信号，
并评估当前融合权重是否合理。

输出:
1. 六书信号之间的相关系数矩阵
2. 各书信号与交易盈利的相关性 (alpha贡献度)
3. 各书在不同regime下的有效性
4. 权重优化建议
"""
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from run_p0_oos_validation import V6_OVERRIDES, load_base_config
from run_p1_p2_sensitivity import prepare_data
from optimize_six_book import _build_tf_score_index, run_strategy_multi_tf

STRATEGY_CONFIG = {
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
}

BOOK_KEYS = ['div', 'ma', 'cs', 'bb', 'vp', 'kdj']
BOOK_SELL_KEYS = [f'book_{k}_sell' for k in BOOK_KEYS]
BOOK_BUY_KEYS = [f'book_{k}_buy' for k in BOOK_KEYS]


def extract_trade_signals(all_data, all_signals, needed_tfs, cfg, primary_tf,
                           decision_tfs, start_dt, end_dt):
    """运行回测并提取每笔交易的六书信号"""
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

    trades = result.get('trades', [])
    opens = [t for t in trades if t.get('action', '').startswith('OPEN_')]
    closes = [t for t in trades if t.get('action', '').startswith('CLOSE_')]

    records = []
    for side in ['short', 'long']:
        side_opens = [o for o in opens if o.get('direction') == side]
        side_closes = [c for c in closes if c.get('direction') == side]

        for i, o in enumerate(side_opens):
            if i >= len(side_closes):
                break
            c = side_closes[i]
            pnl = float(c.get('pnl', 0))
            regime = o.get('regime_label', 'unknown')

            record = {
                'side': side,
                'regime': regime,
                'pnl': pnl,
                'win': int(pnl > 0),
                'ss': float(o.get('ss', 0) or 0),
                'bs': float(o.get('bs', 0) or 0),
            }
            # 提取六书特征
            for bk in BOOK_SELL_KEYS + BOOK_BUY_KEYS:
                record[bk] = float(o.get(bk, 0) or 0)

            records.append(record)

    return pd.DataFrame(records)


def main():
    print("=" * 100)
    print("  P16: 信号去相关/权重再平衡分析")
    print("=" * 100)

    # 合并 IS + OOS 数据
    print("  加载 IS 数据...")
    is_data, is_signals, needed_tfs, primary_tf, decision_tfs, is_start, is_end = \
        prepare_data('ETHUSDT', '2025-01-01', '2026-01-31')
    print("  加载 OOS 数据...")
    oos_data, oos_signals, _, _, _, oos_start, oos_end = \
        prepare_data('ETHUSDT', '2024-01-01', '2024-12-31')

    cfg = load_base_config()
    cfg.update(STRATEGY_CONFIG)

    print("\n  运行 IS 回测...")
    is_df = extract_trade_signals(is_data, is_signals, needed_tfs, cfg, primary_tf,
                                   decision_tfs, is_start, is_end)
    print(f"    IS: {len(is_df)} 笔交易")

    print("  运行 OOS 回测...")
    oos_df = extract_trade_signals(oos_data, oos_signals, needed_tfs, cfg, primary_tf,
                                    decision_tfs, oos_start, oos_end)
    print(f"    OOS: {len(oos_df)} 笔交易")

    all_df = pd.concat([is_df.assign(period='IS'), oos_df.assign(period='OOS')], ignore_index=True)

    # ─── 1. 信号相关矩阵 ───
    print(f"\n{'='*80}")
    print("  1. 六书卖出信号相关矩阵 (短仓)")
    print(f"{'='*80}")
    shorts = all_df[all_df['side'] == 'short']
    if len(shorts) > 5:
        sell_cols = [c for c in BOOK_SELL_KEYS if c in shorts.columns]
        sell_matrix = shorts[sell_cols].corr()
        # 简化列名
        rename = {f'book_{k}_sell': k.upper() for k in BOOK_KEYS}
        sell_matrix = sell_matrix.rename(columns=rename, index=rename)
        print(sell_matrix.round(3).to_string())

    print(f"\n{'='*80}")
    print("  2. 六书买入信号相关矩阵 (多仓)")
    print(f"{'='*80}")
    longs = all_df[all_df['side'] == 'long']
    if len(longs) > 5:
        buy_cols = [c for c in BOOK_BUY_KEYS if c in longs.columns]
        buy_matrix = longs[buy_cols].corr()
        rename_b = {f'book_{k}_buy': k.upper() for k in BOOK_KEYS}
        buy_matrix = buy_matrix.rename(columns=rename_b, index=rename_b)
        print(buy_matrix.round(3).to_string())

    # ─── 2. 信号与盈利的相关性 (Alpha贡献) ───
    print(f"\n{'='*80}")
    print("  3. 信号与盈亏相关性 (Alpha贡献度)")
    print(f"{'='*80}")
    for side_label, side_df, signal_keys in [
        ('SHORT', shorts, BOOK_SELL_KEYS),
        ('LONG', longs, BOOK_BUY_KEYS),
    ]:
        if len(side_df) < 5:
            continue
        print(f"\n  {side_label} (n={len(side_df)}):")
        print(f"  {'信号':<15} {'与PnL相关':>10} {'与WR相关':>10} {'胜率时均值':>12} {'败率时均值':>12} {'Cohen_d':>10}")
        for sk in signal_keys:
            if sk not in side_df.columns:
                continue
            vals = side_df[sk].astype(float)
            corr_pnl = vals.corr(side_df['pnl']) if vals.std() > 0 else 0
            corr_win = vals.corr(side_df['win']) if vals.std() > 0 else 0
            win_mean = side_df.loc[side_df['win'] == 1, sk].mean() if side_df['win'].sum() > 0 else 0
            loss_mean = side_df.loc[side_df['win'] == 0, sk].mean() if (side_df['win'] == 0).sum() > 0 else 0
            pooled_std = vals.std()
            cohen_d = (win_mean - loss_mean) / pooled_std if pooled_std > 0 else 0
            name = sk.replace('book_', '').replace('_sell', '').replace('_buy', '').upper()
            print(f"  {name:<15} {corr_pnl:>+10.3f} {corr_win:>+10.3f} {win_mean:>12.3f} {loss_mean:>12.3f} {cohen_d:>+10.3f}")

    # ─── 3. Regime 细分有效性 ───
    print(f"\n{'='*80}")
    print("  4. 各Regime下信号有效性 (空仓)")
    print(f"{'='*80}")
    for regime in ['neutral', 'trend', 'high_vol', 'low_vol_trend']:
        rdf = shorts[shorts['regime'] == regime]
        if len(rdf) < 3:
            continue
        print(f"\n  Regime: {regime} (n={len(rdf)}, WR={rdf['win'].mean()*100:.0f}%)")
        for sk in BOOK_SELL_KEYS:
            if sk not in rdf.columns:
                continue
            vals = rdf[sk].astype(float)
            corr_win = vals.corr(rdf['win']) if vals.std() > 0 else 0
            name = sk.replace('book_', '').replace('_sell', '').upper()
            print(f"    {name}: corr_win={corr_win:+.3f} mean={vals.mean():.2f} std={vals.std():.2f}")

    # ─── 4. 权重优化建议 ───
    print(f"\n{'='*80}")
    print("  5. 权重优化建议")
    print(f"{'='*80}")
    print("  当前: c6_veto_4 模式 (DIV 70% + MA 30%, BB/VP/CS/KDJ 作为 bonus)")
    print("  基于分析:")
    for sk in BOOK_SELL_KEYS:
        if sk in shorts.columns:
            vals = shorts[sk].astype(float)
            corr = vals.corr(shorts['win']) if vals.std() > 0 else 0
            name = sk.replace('book_', '').replace('_sell', '').upper()
            if corr > 0.1:
                print(f"    {name}: ✅ 正向贡献 (corr={corr:+.3f}) → 建议增加权重")
            elif corr < -0.1:
                print(f"    {name}: ❌ 负向贡献 (corr={corr:+.3f}) → 建议降低权重或反向使用")
            else:
                print(f"    {name}: ○ 中性 (corr={corr:+.3f}) → 保持当前权重")

    # 保存
    export_dir = 'data/backtests/signal_analysis'
    os.makedirs(export_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_df.to_csv(os.path.join(export_dir, f'trade_signals_{ts}.csv'), index=False)
    print(f"\n  交易信号数据已保存到 {export_dir}/")


if __name__ == '__main__':
    main()
