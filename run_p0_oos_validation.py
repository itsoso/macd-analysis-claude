"""
P0a: v6.0 参数 2024 OOS 回测
P0b: Regime 统计口径验证（entry_regime vs bar-level 计数）

用途: 用 v6.0 参数原封不动跑 2024-01-01 ~ 2024-12-31（OOS），
      输出按 side+regime 的详细 WR/PF/MFE/MAE 分析。
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import pandas as pd

# 导入核心依赖
from binance_fetcher import fetch_binance_klines
from indicators import add_all_indicators
from ma_indicators import add_moving_averages
from optimize_six_book import (
    _build_tf_score_index,
    compute_signals_six,
    run_strategy_multi_tf,
)
from backtest_multi_tf_30d_7d import _apply_conservative_risk

# ============================================================
#  v6.0 参数覆盖
# ============================================================
V6_OVERRIDES = {
    # 核心 v6.0 参数
    'short_threshold': 40,
    'long_threshold': 25,
    'short_sl': -0.20,
    'short_tp': 0.60,
    'long_sl': -0.10,
    'long_tp': 0.40,
    'short_trail': 0.19,
    'long_trail': 0.12,
    'trail_pullback': 0.50,
    'short_max_hold': 48,
    'long_max_hold': 72,
    # 分段止盈
    'use_partial_tp': True,
    'partial_tp_1': 0.15,
    'partial_tp_1_pct': 0.30,
    'use_partial_tp_2': True,
    'partial_tp_2': 0.50,
    'partial_tp_2_pct': 0.30,
    'use_partial_tp_v3': True,
    'partial_tp_1_early': 0.12,
    'partial_tp_2_early': 0.25,
    # 结构折扣 (v6.0)
    'use_neutral_structural_discount': True,
    'neutral_struct_discount_0': 0.10,
    'neutral_struct_discount_1': 0.20,
    'neutral_struct_discount_2': 1.00,
    # 冲突软折扣 (v6.0)
    'use_short_conflict_soft_discount': True,
    'short_conflict_discount_mult': 0.60,
    'short_conflict_regimes': 'trend,high_vol',
    'short_conflict_div_buy_min': 50.0,
    'short_conflict_ma_sell_min': 12.0,
    # 融合模式
    'fusion_mode': 'c6_veto_4',
    'div_weight': 0.55,
    'kdj_weight': 0.15,
    # Regime
    'use_regime_aware': True,
    'regime_short_threshold': 'neutral:45',
    'use_regime_short_gate': True,
    'regime_short_gate_add': 35,
    'regime_short_gate_regimes': 'low_vol_trend',
    # 多周期
    'use_multi_tf': True,
    'consensus_min_strength': 40,
    'coverage_min': 0.5,
    # 双引擎/微结构/波动目标
    'use_dual_engine': True,
    'use_microstructure': True,
    'use_vol_target': True,
    # 趋势保护
    'use_trend_enhance': True,
    # 保护
    'use_protections': True,
    'prot_loss_streak_limit': 3,
    'prot_loss_streak_cooldown_bars': 24,
    'prot_daily_loss_limit_pct': 0.03,
    'prot_global_dd_limit_pct': 0.12,
    'prot_close_on_global_halt': True,
    # Gate
    'use_live_gate': True,
    # 反向持仓
    'reverse_min_hold_short': 8,
    'reverse_min_hold_long': 8,
    # 硬断路器
    'hard_stop_loss': -0.28,
}


def load_base_config(opt_file='optimize_six_book_result.json'):
    """加载基础配置并覆盖 v6.0 参数"""
    with open(opt_file) as f:
        data = json.load(f)
    cfg = dict((data.get('global_best') or {}).get('config') or {})
    cfg = _apply_conservative_risk(cfg)
    cfg.update(V6_OVERRIDES)
    return cfg


def fetch_tf_data(symbol, tf, fetch_days, warmup_start, end_dt):
    df = fetch_binance_klines(symbol, interval=tf, days=fetch_days)
    if df is None or len(df) < 120:
        return None
    df = add_all_indicators(df)
    add_moving_averages(df, timeframe=tf)
    df = df[(df.index >= warmup_start) & (df.index <= end_dt)].copy()
    if len(df) < 120:
        return None
    return df


def analyze_trades(trades, tag=''):
    """按 side + entry_regime 分析交易"""
    if not trades:
        print(f"  [{tag}] 无交易记录")
        return

    # P0b: 验证每笔交易都有 entry_regime
    regime_missing = 0
    for t in trades:
        extra = t.get('extra') or {}
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except:
                extra = {}
        if not extra.get('sig_regime') and not extra.get('regime'):
            regime_missing += 1

    total = len(trades)
    wins = sum(1 for t in trades if float(t.get('pnl_r', t.get('pnl', 0))) > 0)
    losses = total - wins
    wr = wins / total * 100 if total else 0

    total_pnl = sum(float(t.get('pnl_r', t.get('pnl', 0))) for t in trades)
    gross_profit = sum(float(t.get('pnl_r', t.get('pnl', 0))) for t in trades
                       if float(t.get('pnl_r', t.get('pnl', 0))) > 0)
    gross_loss = abs(sum(float(t.get('pnl_r', t.get('pnl', 0))) for t in trades
                        if float(t.get('pnl_r', t.get('pnl', 0))) <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    print(f"\n{'='*80}")
    print(f"  {tag} 交易汇总")
    print(f"{'='*80}")
    print(f"  总交易: {total}笔 | 胜: {wins} 负: {losses} | WR: {wr:.1f}% | PF: {pf:.2f}")
    print(f"  总PnL%: {total_pnl*100:+.1f}% | 毛利: {gross_profit*100:.1f}% | 毛亏: {gross_loss*100:.1f}%")
    if regime_missing > 0:
        print(f"  ⚠️  P0b: {regime_missing}/{total} 笔交易缺少 entry_regime 标签!")

    # 按 side 分组
    by_side = defaultdict(list)
    for t in trades:
        side = t.get('side', 'unknown')
        if 'short' in str(side).lower() or 'sell' in str(side).lower():
            by_side['short'].append(t)
        elif 'long' in str(side).lower() or 'buy' in str(side).lower():
            by_side['long'].append(t)
        else:
            # 从 reason 推断
            reason = str(t.get('reason', ''))
            if '空' in reason or 'short' in reason.lower():
                by_side['short'].append(t)
            elif '多' in reason or 'long' in reason.lower():
                by_side['long'].append(t)
            else:
                by_side['unknown'].append(t)

    # 按 side+regime 细分
    print(f"\n  {'Side':<8} {'Regime':<16} {'笔数':>5} {'胜':>4} {'负':>4} {'WR%':>7} {'PF':>7} {'AvgWin%':>9} {'AvgLoss%':>10} {'TotalPnl%':>10}")
    print(f"  {'-'*92}")

    regime_trade_count = 0
    for side in ['short', 'long', 'unknown']:
        if side not in by_side:
            continue
        side_trades = by_side[side]

        # 提取 regime
        by_regime = defaultdict(list)
        for t in side_trades:
            extra = t.get('extra') or {}
            if isinstance(extra, str):
                try:
                    extra = json.loads(extra)
                except:
                    extra = {}
            regime = extra.get('sig_regime', extra.get('regime', 'no_regime'))
            by_regime[regime].append(t)

        for regime in sorted(by_regime.keys()):
            rtrades = by_regime[regime]
            n = len(rtrades)
            regime_trade_count += n
            w = sum(1 for t in rtrades if float(t.get('pnl_r', t.get('pnl', 0))) > 0)
            l = n - w
            wr_r = w / n * 100 if n else 0
            gp = sum(float(t.get('pnl_r', t.get('pnl', 0))) for t in rtrades
                     if float(t.get('pnl_r', t.get('pnl', 0))) > 0)
            gl = abs(sum(float(t.get('pnl_r', t.get('pnl', 0))) for t in rtrades
                        if float(t.get('pnl_r', t.get('pnl', 0))) <= 0))
            pf_r = gp / gl if gl > 0 else float('inf')
            avg_win = (gp / w * 100) if w > 0 else 0
            avg_loss = (gl / l * 100) if l > 0 else 0
            total_r = sum(float(t.get('pnl_r', t.get('pnl', 0))) for t in rtrades) * 100
            print(f"  {side:<8} {regime:<16} {n:>5} {w:>4} {l:>4} {wr_r:>6.1f}% {pf_r:>6.2f} {avg_win:>8.1f}% {avg_loss:>9.1f}% {total_r:>+9.1f}%")

    # P0b: 口径验证
    print(f"\n  P0b 口径验证:")
    print(f"    按 entry_regime 汇总交易数: {regime_trade_count}")
    print(f"    总交易数: {total}")
    if regime_trade_count == total:
        print(f"    ✅ 口径一致：每笔交易恰好有 1 个 entry_regime 标签")
    else:
        print(f"    ⚠️  口径不一致！差异 = {regime_trade_count - total}")

    # 持仓 bars 分布
    print(f"\n  持仓时长分布 (bars):")
    for side in ['short', 'long']:
        if side not in by_side:
            continue
        bars_list = []
        for t in by_side[side]:
            bars = t.get('hold_bars', t.get('bars', 0))
            if bars:
                bars_list.append(int(bars))
        if bars_list:
            import numpy as np
            arr = np.array(bars_list)
            print(f"    {side}: n={len(arr)} min={arr.min()} p25={int(np.percentile(arr,25))} "
                  f"median={int(np.median(arr))} p75={int(np.percentile(arr,75))} max={arr.max()}")

    return trades


def main():
    symbol = 'ETHUSDT'
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    warmup_days = 60

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    warmup_start = start_dt - pd.Timedelta(days=warmup_days)
    now = pd.Timestamp.now().tz_localize(None)
    fetch_days = max(90, int((now - warmup_start).days + 5))

    # 时间框架配置
    primary_tf = '1h'
    decision_tfs = ['15m', '1h', '4h', '12h']  # 与当前实盘一致
    needed_tfs = sorted(set([primary_tf] + decision_tfs))

    # 加载 v6.0 配置
    cfg = load_base_config()

    print("=" * 100)
    print("  P0a: v6.0 参数 2024 OOS 回测")
    print("  P0b: Regime 统计口径验证")
    print(f"  区间: {start_date} ~ {end_date} (样本外)")
    print(f"  主周期: {primary_tf} | 决策: {decision_tfs}")
    print(f"  核心参数: ST={cfg['short_threshold']} LT={cfg['long_threshold']} "
          f"SL={cfg['short_sl']} Trail={cfg['short_trail']} PB={cfg['trail_pullback']}")
    print("=" * 100)

    # 获取数据
    all_data = {}
    for tf in needed_tfs:
        print(f"  获取 {tf} 数据...")
        df = fetch_tf_data(symbol, tf, fetch_days, warmup_start, end_dt)
        if df is None:
            print(f"  ❌ {tf} 数据不足!")
            return
        all_data[tf] = df
        print(f"    {tf}: {len(df)} 条 ({df.index[0]} ~ {df.index[-1]})")

    # 计算信号
    print("\n  计算六书信号...")
    all_signals = {
        tf: compute_signals_six(all_data[tf], tf, all_data, max_bars=0)
        for tf in needed_tfs
    }
    tf_score_map = _build_tf_score_index(all_data, all_signals, needed_tfs, cfg)

    # 运行回测
    print("  运行多周期策略回测...")
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

    # 汇总
    ret = result.get('strategy_return', 0)
    bh = result.get('buy_hold_return', 0)
    alpha = result.get('alpha', 0)
    mdd = result.get('max_drawdown', 0)
    total_trades = result.get('total_trades', 0)
    liqs = result.get('liquidations', 0)
    fees = result.get('fees', {})

    print(f"\n{'='*100}")
    print(f"  2024 OOS 回测结果 (v6.0 参数)")
    print(f"{'='*100}")
    print(f"  策略收益: {ret:+.2f}%")
    print(f"  买入持有: {bh:+.2f}%")
    print(f"  Alpha:    {alpha:+.2f}%")
    print(f"  最大回撤: {mdd:.2f}%")
    print(f"  总交易:   {total_trades}笔")
    print(f"  强平次数: {liqs}")
    print(f"  手续费:   ${fees.get('total_costs', 0):,.0f}")

    # 详细交易分析
    trades = result.get('trades', [])
    analyze_trades(trades, tag='2024 OOS (v6.0)')

    # 保存结果
    export_dir = 'data/backtests/p0_oos_validation'
    os.makedirs(export_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存交易明细
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_csv = os.path.join(export_dir, f'p0a_oos_2024_trades_{ts}.csv')
        trades_df.to_csv(trades_csv, index=False)
        print(f"\n  交易明细已保存: {trades_csv}")

    # 保存汇总
    summary = {
        'run_time': ts,
        'period': f'{start_date} ~ {end_date}',
        'type': 'OOS',
        'strategy_return': ret,
        'buy_hold_return': bh,
        'alpha': alpha,
        'max_drawdown': mdd,
        'total_trades': total_trades,
        'liquidations': liqs,
        'params': {k: cfg[k] for k in V6_OVERRIDES},
    }
    summary_file = os.path.join(export_dir, f'p0a_oos_2024_summary_{ts}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"  汇总已保存: {summary_file}")


if __name__ == '__main__':
    main()
