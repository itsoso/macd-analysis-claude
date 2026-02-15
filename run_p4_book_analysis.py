"""
P4: Neutral 各书判别力分析 + 信号相关性矩阵
P4b: MFE/MAE 分析 (按 side+regime)

使用 IS (2025) 和 OOS (2024) 的交易数据。
"""

import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from run_p0_oos_validation import load_base_config
from run_p1_p2_sensitivity import prepare_data, run_single_backtest
from optimize_six_book import (
    _build_tf_score_index,
    run_strategy_multi_tf,
)


def get_trades_with_signals(all_data, all_signals, needed_tfs, cfg, primary_tf,
                             decision_tfs, start_dt, end_dt):
    """运行回测并返回完整的交易记录（含信号细节）"""
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
    return result.get('trades', [])


def pair_trades(trades):
    """配对 OPEN/CLOSE 交易"""
    paired = []
    for direction in ['short', 'long']:
        opens = [t for t in trades if t.get('action') == f'OPEN_{direction.upper()}']
        closes = [t for t in trades if t.get('action') == f'CLOSE_{direction.upper()}']
        for i in range(min(len(opens), len(closes))):
            o = opens[i]
            c = closes[i]
            entry_p = o['exec_price']
            exit_p = c['exec_price']
            if direction == 'short':
                pnl_r = (entry_p - exit_p) / entry_p
            else:
                pnl_r = (exit_p - entry_p) / entry_p

            open_time = pd.Timestamp(o['time'])
            close_time = pd.Timestamp(c['time'])
            hold_bars = int((close_time - open_time).total_seconds() / 3600)

            paired.append({
                'direction': direction,
                'regime': o.get('regime_label', 'unknown'),
                'pnl_r': pnl_r,
                'pnl_usd': c.get('pnl', 0),
                'hold_bars': hold_bars,
                'reason': c.get('reason', ''),
                'win': pnl_r > 0,
                # 六书信号
                'div_sell': o.get('book_div_sell', 0),
                'div_buy': o.get('book_div_buy', 0),
                'ma_sell': o.get('book_ma_sell', 0),
                'ma_buy': o.get('book_ma_buy', 0),
                'cs_sell': o.get('book_cs_sell', 0),
                'cs_buy': o.get('book_cs_buy', 0),
                'bb_sell': o.get('book_bb_sell', 0),
                'bb_buy': o.get('book_bb_buy', 0),
                'vp_sell': o.get('book_vp_sell', 0),
                'vp_buy': o.get('book_vp_buy', 0),
                'kdj_sell': o.get('book_kdj_sell', 0),
                'kdj_buy': o.get('book_kdj_buy', 0),
                'ss': o.get('ss', 0),
                'bs': o.get('bs', 0),
                'struct_confirms': o.get('sig_struct_confirms', -1),
                'struct_discount': o.get('sig_struct_discount_mult', 1.0),
            })
    return pd.DataFrame(paired)


def analyze_book_discriminability(tdf, tag=''):
    """P4: 分析各书在不同 regime 下的判别力"""
    print(f"\n{'='*100}")
    print(f"  P4: 各书判别力分析 — {tag}")
    print(f"{'='*100}")

    for regime in ['neutral', 'trend', 'high_vol', 'low_vol_trend']:
        for side in ['short', 'long']:
            subset = tdf[(tdf['regime'] == regime) & (tdf['direction'] == side)]
            if len(subset) < 5:
                continue

            wins = subset[subset['win']]
            losses = subset[~subset['win']]

            # 对于空单, 看 sell 方向的书; 对于多单, 看 buy 方向
            if side == 'short':
                books = {
                    'DIV_sell': 'div_sell', 'MA_sell': 'ma_sell',
                    'CS_sell': 'cs_sell', 'BB_sell': 'bb_sell',
                    'VP_sell': 'vp_sell', 'KDJ_sell': 'kdj_sell',
                }
            else:
                books = {
                    'DIV_buy': 'div_buy', 'MA_buy': 'ma_buy',
                    'CS_buy': 'cs_buy', 'BB_buy': 'bb_buy',
                    'VP_buy': 'vp_buy', 'KDJ_buy': 'kdj_buy',
                }

            print(f"\n  {regime}|{side} (n={len(subset)}, W={len(wins)}, L={len(losses)}, "
                  f"WR={len(wins)/len(subset)*100:.0f}%)")
            print(f"  {'书':<12} {'Win均值':>9} {'Loss均值':>10} {'差值':>8} {'Cohen_d':>9} {'判别力':>6}")
            print(f"  {'-'*58}")

            for bname, col in books.items():
                win_vals = wins[col].astype(float)
                loss_vals = losses[col].astype(float)

                win_mean = win_vals.mean() if len(win_vals) > 0 else 0
                loss_mean = loss_vals.mean() if len(loss_vals) > 0 else 0
                diff = win_mean - loss_mean

                # Cohen's d
                pooled_std = np.sqrt(
                    ((len(win_vals)-1) * win_vals.std()**2 + (len(loss_vals)-1) * loss_vals.std()**2)
                    / max(1, len(win_vals) + len(loss_vals) - 2)
                ) if len(win_vals) > 1 and len(loss_vals) > 1 else 1
                d = diff / pooled_std if pooled_std > 0 else 0

                # 判别力评级
                if abs(d) >= 0.8:
                    grade = '强★★'
                elif abs(d) >= 0.5:
                    grade = '中★'
                elif abs(d) >= 0.2:
                    grade = '弱'
                else:
                    grade = '无效'

                print(f"  {bname:<12} {win_mean:>8.1f} {loss_mean:>9.1f} {diff:>+7.1f} {d:>+8.2f} {grade:>6}")


def analyze_signal_correlation(tdf, tag=''):
    """信号相关性矩阵"""
    print(f"\n{'='*100}")
    print(f"  P4: 信号相关性矩阵 — {tag}")
    print(f"{'='*100}")

    for regime in ['neutral', 'trend']:
        subset = tdf[tdf['regime'] == regime]
        if len(subset) < 10:
            continue

        # 空单方向
        sell_cols = ['div_sell', 'ma_sell', 'cs_sell', 'bb_sell', 'vp_sell', 'kdj_sell']
        sell_data = subset[sell_cols].astype(float)
        corr = sell_data.corr()

        print(f"\n  {regime} 卖方信号相关性 (n={len(subset)}):")
        labels = ['DIV', 'MA', 'CS', 'BB', 'VP', 'KDJ']
        print(f"  {'':>6}", end='')
        for l in labels:
            print(f" {l:>6}", end='')
        print()
        for i, l in enumerate(labels):
            print(f"  {l:>6}", end='')
            for j in range(len(labels)):
                v = corr.iloc[i, j]
                if i == j:
                    print(f"  {'—':>5}", end='')
                else:
                    print(f" {v:>+5.2f}", end='')
            print()


def analyze_mfe_mae(tdf, tag=''):
    """P4b: MFE/MAE 分析（简化版，用最终 pnl_r 代替真实 MFE/MAE）"""
    print(f"\n{'='*100}")
    print(f"  P4b: 交易盈亏分布 — {tag}")
    print(f"{'='*100}")

    for side in ['short', 'long']:
        side_df = tdf[tdf['direction'] == side]
        for regime in sorted(side_df['regime'].unique()):
            subset = side_df[side_df['regime'] == regime]
            if len(subset) < 3:
                continue

            wins = subset[subset['win']]
            losses = subset[~subset['win']]

            print(f"\n  {side}|{regime} (n={len(subset)})")
            if len(wins) > 0:
                print(f"    赢单: n={len(wins)} avg={wins['pnl_r'].mean()*100:+.1f}% "
                      f"median={wins['pnl_r'].median()*100:+.1f}% "
                      f"max={wins['pnl_r'].max()*100:+.1f}% "
                      f"avg_bars={wins['hold_bars'].mean():.0f}")
            if len(losses) > 0:
                print(f"    亏单: n={len(losses)} avg={losses['pnl_r'].mean()*100:+.1f}% "
                      f"median={losses['pnl_r'].median()*100:+.1f}% "
                      f"min={losses['pnl_r'].min()*100:+.1f}% "
                      f"avg_bars={losses['hold_bars'].mean():.0f}")

            # 按 struct_confirms 分桶
            if 'struct_confirms' in subset.columns:
                sc = subset['struct_confirms'].astype(float)
                if sc.max() >= 0:
                    print(f"    按结构确认数:")
                    for nc in sorted(sc.unique()):
                        if nc < 0:
                            continue
                        nc_sub = subset[sc == nc]
                        nc_wr = nc_sub['win'].mean() * 100
                        nc_avg = nc_sub['pnl_r'].mean() * 100
                        print(f"      confirms={int(nc)}: n={len(nc_sub)} WR={nc_wr:.0f}% avg_pnl={nc_avg:+.1f}%")


def main():
    print("加载数据...")
    is_data, is_signals, needed_tfs, primary_tf, decision_tfs, is_start, is_end = \
        prepare_data('ETHUSDT', '2025-01-01', '2026-01-31')
    oos_data, oos_signals, _, _, _, oos_start, oos_end = \
        prepare_data('ETHUSDT', '2024-01-01', '2024-12-31')

    cfg = load_base_config()

    print("运行 IS 回测...")
    is_trades = get_trades_with_signals(is_data, is_signals, needed_tfs, cfg,
                                         primary_tf, decision_tfs, is_start, is_end)
    is_paired = pair_trades(is_trades)

    print("运行 OOS 回测...")
    oos_trades = get_trades_with_signals(oos_data, oos_signals, needed_tfs, cfg,
                                          primary_tf, decision_tfs, oos_start, oos_end)
    oos_paired = pair_trades(oos_trades)

    print(f"IS 配对交易: {len(is_paired)}笔 | OOS 配对交易: {len(oos_paired)}笔")

    # P4: 各书判别力
    analyze_book_discriminability(is_paired, tag='IS 2025')
    analyze_book_discriminability(oos_paired, tag='OOS 2024')

    # 信号相关性
    analyze_signal_correlation(is_paired, tag='IS 2025')
    analyze_signal_correlation(oos_paired, tag='OOS 2024')

    # P4b: MFE/MAE
    analyze_mfe_mae(is_paired, tag='IS 2025')
    analyze_mfe_mae(oos_paired, tag='OOS 2024')

    # 保存配对交易数据
    export_dir = 'data/backtests/p4_book_analysis'
    os.makedirs(export_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    is_paired.to_csv(os.path.join(export_dir, f'p4_is_paired_trades_{ts}.csv'), index=False)
    oos_paired.to_csv(os.path.join(export_dir, f'p4_oos_paired_trades_{ts}.csv'), index=False)
    print(f"\n  数据已保存到: {export_dir}")


if __name__ == '__main__':
    main()
