#!/usr/bin/env python3
"""A/B 对比: v2基线 vs v3增强 (同区间、同数据、同信号)

用法:
  python3 ab_test_v3.py
"""
import time, json, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from backtest_multi_tf_daily import (
    _build_default_config, _scale_runtime_config,
    DEFAULT_CONFIG, PRIMARY_TF, DECISION_TFS, AVAILABLE_TFS,
    FALLBACK_DECISION_TFS,
    DEFAULT_TRADE_START, DEFAULT_TRADE_END,
    _build_tf_score_index, _normalize_trade,
    fetch_data_for_tf,
)
from optimize_six_book import run_strategy_multi_tf
from signal_core import compute_signals_six, compute_signals_six_multiprocess
import pandas as pd


TRADE_START = DEFAULT_TRADE_START
TRADE_END = DEFAULT_TRADE_END
INITIAL_CAPITAL = 200000


def _load_data_and_signals():
    """一次性加载数据 + 计算信号, 所有 A/B 变体共享"""
    print("[1/3] 加载数据...")
    t0 = time.time()
    all_data = {}
    history_days = max(560, (pd.Timestamp.now() - pd.Timestamp(TRADE_START)).days + 90)

    for tf in list(dict.fromkeys([PRIMARY_TF, *DECISION_TFS])):
        print(f"  获取 {tf}...")
        df = fetch_data_for_tf(tf, history_days, allow_api_fallback=False)
        if df is not None:
            all_data[tf] = df
            print(f"    {tf}: {len(df)} bars")
    print(f"  数据加载完成 ({time.time()-t0:.1f}s)")

    decision_tfs = [tf for tf in DECISION_TFS if tf in all_data]
    score_tfs = list(dict.fromkeys([PRIMARY_TF, *decision_tfs]))

    print("\n[2/3] 计算信号...")
    t1 = time.time()
    all_signals = compute_signals_six_multiprocess(all_data, score_tfs)
    print(f"  信号计算完成 ({time.time()-t1:.1f}s)")

    # 构建评分索引
    base_cfg = _scale_runtime_config(_build_default_config(), PRIMARY_TF)
    tf_score_index = _build_tf_score_index(all_data, all_signals, score_tfs, base_cfg)
    print(f"  评分索引: {sum(len(v) for v in tf_score_index.values())} 个评分点")

    return all_data, decision_tfs, tf_score_index


def _run_variant(label, cfg_overrides, all_data, decision_tfs, tf_score_index):
    """运行一个配置变体"""
    cfg = _scale_runtime_config(_build_default_config(), PRIMARY_TF)
    cfg.update(cfg_overrides)
    cfg['name'] = f"AB_{label}"

    t0 = time.time()
    result = run_strategy_multi_tf(
        primary_df=all_data[PRIMARY_TF],
        tf_score_map=tf_score_index,
        decision_tfs=decision_tfs,
        config=cfg,
        primary_tf=PRIMARY_TF,
        trade_days=0,
        trade_start_dt=pd.Timestamp(TRADE_START),
        trade_end_dt=pd.Timestamp(TRADE_END) + pd.Timedelta(hours=23, minutes=59),
    )
    elapsed = time.time() - t0

    raw_trades = result.get('trades', [])
    fees = result.get('fees', {})
    final_total = result.get('final_total', 0)
    strategy_return = result.get('strategy_return', 0)
    alpha = result.get('alpha', 0)
    max_drawdown = result.get('max_drawdown', 0)

    # 交易统计
    close_actions = {'CLOSE_LONG', 'CLOSE_SHORT', 'LIQUIDATED'}
    close_trades = [t for t in raw_trades if t.get('action') in close_actions]
    wins = [t for t in close_trades if (t.get('pnl', 0) or 0) > 0]
    losses = [t for t in close_trades if (t.get('pnl', 0) or 0) <= 0]
    win_rate = len(wins) / len(close_trades) * 100 if close_trades else 0

    # 合约PF (仅平仓)
    contract_win = sum(t.get('pnl', 0) for t in wins)
    contract_loss = abs(sum(t.get('pnl', 0) for t in losses))
    contract_pf = contract_win / contract_loss if contract_loss > 0 else 999
    # 组合PF (含 PARTIAL_TP / SPOT_SELL)
    pnl_actions = {'CLOSE_LONG', 'CLOSE_SHORT', 'LIQUIDATED', 'PARTIAL_TP', 'SPOT_SELL'}
    all_pnl_trades = [t for t in raw_trades if t.get('action') in pnl_actions and t.get('pnl') is not None]
    total_win_pnl = sum(t['pnl'] for t in all_pnl_trades if t['pnl'] > 0)
    total_loss_pnl = abs(sum(t['pnl'] for t in all_pnl_trades if t['pnl'] < 0))
    full_pf = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else 999

    # 分类统计
    from collections import defaultdict
    action_stats = defaultdict(lambda: {'count': 0, 'pnl': 0})
    for t in raw_trades:
        a = t.get('action', '')
        d = t.get('direction', '')
        key = f"{a}({'d'})" if a == 'PARTIAL_TP' else a
        action_stats[key]['count'] += 1
        action_stats[key]['pnl'] += (t.get('pnl', 0) or 0)

    total_costs = fees.get('total_costs', 0)

    return {
        'label': label,
        'return_pct': strategy_return,
        'alpha': alpha,
        'max_dd': max_drawdown,
        'total_trades': len(raw_trades),
        'close_trades': len(close_trades),
        'win_count': len(wins),
        'loss_count': len(losses),
        'win_rate': win_rate,
        'contract_pf': contract_pf,
        'full_pf': full_pf,
        'fees': total_costs,
        'final': final_total,
        'action_stats': dict(action_stats),
        'elapsed': elapsed,
    }


def main():
    all_data, decision_tfs, tf_score_index = _load_data_and_signals()

    # 是否跑「实盘口径」：--live 第八轮(降频控损)，--live --spot 第九轮(现货卖出质量)
    run_live_track = '--live' in sys.argv or os.environ.get('RUN_LIVE_AB') == '1'
    run_live_spot = '--spot' in sys.argv or os.environ.get('RUN_LIVE_SPOT') == '1'

    # ── 研究口径基线 (run#43) ──
    stable_base = {
        'short_threshold': 25, 'short_sl': -0.25, 'short_tp': 0.60,
        'long_sl': -0.10, 'short_trail': 0.15, 'long_trail': 0.12,
        'trail_pullback': 0.50,
        'use_partial_tp_v3': True,
        'partial_tp_1_early': 0.12, 'partial_tp_2_early': 0.25,
        'use_spot_sell_cap': False,
        'use_regime_short_gate': True,
        'regime_short_gate_add': 15,
        'regime_short_gate_regimes': 'low_vol_trend',
        'hard_stop_loss': -0.28,
        'use_spot_sell_confirm': False,
    }

    if run_live_track and run_live_spot:
        # ── 第九轮: 实盘口径 现货卖出质量 (Codex) ──
        # 基线 = run#46 (LVT+35)。目标: 高波动期 SPOT_SELL 噪声下降，confirm_ss 下调覆盖 SS 40~70
        # 评估: pPF、CLOSE_SHORT净亏、high_vol SPOT_SELL净值、fee_drag
        live_base = {
            **stable_base,
            'use_microstructure': True,
            'use_dual_engine': True,
            'use_vol_target': True,
            'regime_short_gate_add': 35,
        }
        variants = [
            ("A:run46基线", {**live_base}),
            ("B:确认SS70min2", {**live_base, 'use_spot_sell_confirm': True,
                'spot_sell_confirm_ss': 70, 'spot_sell_confirm_min': 2}),
            ("C:确认SS60min2", {**live_base, 'use_spot_sell_confirm': True,
                'spot_sell_confirm_ss': 60, 'spot_sell_confirm_min': 2}),
            ("D:sell_thr22", {**live_base, 'sell_threshold': 22}),
            ("E:spot_cd18", {**live_base, 'spot_cooldown': 18}),
            ("F:spot_cd24", {**live_base, 'spot_cooldown': 24}),
            ("G:SS70min2+cd18", {**live_base, 'use_spot_sell_confirm': True,
                'spot_sell_confirm_ss': 70, 'spot_sell_confirm_min': 2, 'spot_cooldown': 18}),
        ]
        round_title = "第九轮: 实盘口径 现货卖出质量 | 基线=run#46"
    elif run_live_track:
        # ── 第八轮: 实盘口径 降频控损 (Codex) ──
        # 基线 = run#44，三开关 ON。目标: 空头亏损与交易成本收敛
        live_base = {
            **stable_base,
            'use_microstructure': True,
            'use_dual_engine': True,
            'use_vol_target': True,
        }
        variants = [
            ("A:live基线", {**live_base}),
            ("B:hard_sl-25", {**live_base, 'hard_stop_loss': -0.25}),
            ("C:hard_sl-22", {**live_base, 'hard_stop_loss': -0.22}),
            ("D:LVT+25", {**live_base, 'regime_short_gate_add': 25}),
            ("E:LVT+35", {**live_base, 'regime_short_gate_add': 35}),
            ("F:gate+neutral", {**live_base, 'regime_short_gate_regimes': 'low_vol_trend,neutral'}),
            ("G:确认SS100", {**live_base,
                'use_spot_sell_confirm': True,
                'spot_sell_confirm_ss': 100, 'spot_sell_confirm_min': 3}),
            ("H:sl-22+确认", {**live_base, 'hard_stop_loss': -0.22,
                'use_spot_sell_confirm': True,
                'spot_sell_confirm_ss': 100, 'spot_sell_confirm_min': 3}),
            ("I:LVT25+neutral", {**live_base, 'regime_short_gate_add': 25,
                'regime_short_gate_regimes': 'low_vol_trend,neutral'}),
        ]
        round_title = "第八轮: 实盘口径 降频控损 | 基线=run#44"
    else:
        # ── 第七轮: 研究口径 — 空头尾部 + SPOT_SELL 高分确认 ──
        # 基线 = run#43 稳健档。晋级: MaxDD≥-9%, pPF≥1.9, CLOSE_SHORT 改善, 2024 OOS 不退化
        variants = [
            ("A:稳健档基线", {**stable_base}),
            ("B:hard_sl-25", {**stable_base, 'hard_stop_loss': -0.25}),
            ("C:hard_sl-22", {**stable_base, 'hard_stop_loss': -0.22}),
            ("D:LVT+25", {**stable_base, 'regime_short_gate_add': 25}),
            ("E:LVT+35", {**stable_base, 'regime_short_gate_add': 35}),
            ("F:sl-25+LVT25", {**stable_base, 'hard_stop_loss': -0.25, 'regime_short_gate_add': 25}),
            ("G:sl-22+LVT15", {**stable_base, 'hard_stop_loss': -0.22}),
            ("H:确认SS100", {**stable_base,
                'use_spot_sell_confirm': True,
                'spot_sell_confirm_ss': 100, 'spot_sell_confirm_min': 3}),
            ("I:趋势禁卖", {**stable_base, 'spot_sell_regime_block': 'trend'}),
        ]
        round_title = "第七轮: 空头尾部+SPOT_SELL | 基线=run#43"

    print(f"\n{'='*90}")
    print(f"  {round_title} | {TRADE_START} ~ {TRADE_END} | ${INITIAL_CAPITAL:,}")
    print(f"{'='*90}")

    results = []
    for label, overrides in variants:
        print(f"\n[3/3] 运行 {label}...")
        r = _run_variant(label, overrides, all_data, decision_tfs, tf_score_index)
        results.append(r)
        print(f"  收益: {r['return_pct']:+.2f}%  Alpha: {r['alpha']:+.2f}%  "
              f"回撤: {r['max_dd']:.2f}%  cPF: {r['contract_pf']:.2f}  pPF: {r['full_pf']:.2f}  "
              f"胜率: {r['win_rate']:.1f}%  ({r['elapsed']:.1f}s)")

    # 汇总表
    print(f"\n{'='*120}")
    print(f"  消融对比汇总")
    print(f"{'='*120}")

    header = f"{'指标':<22}" + "".join(f"{r['label']:>20}" for r in results)
    print(header)
    print("-" * (22 + 20 * len(results)))

    rows = [
        ('收益率', 'return_pct', '{:+.2f}%'),
        ('Alpha', 'alpha', '{:+.2f}%'),
        ('最大回撤', 'max_dd', '{:.2f}%'),
        ('合约PF', 'contract_pf', '{:.2f}'),
        ('组合PF', 'full_pf', '{:.2f}'),
        ('胜率', 'win_rate', '{:.1f}%'),
        ('平仓数', 'close_trades', '{:,.0f}'),
        ('总交易', 'total_trades', '{:,.0f}'),
        ('总费用', 'fees', '${:,.0f}'),
        ('期末资金', 'final', '${:,.0f}'),
    ]
    for name, key, fmt in rows:
        vals = [r[key] for r in results]
        row_str = f"  {name:<20}" + "".join(fmt.format(v).rjust(20) for v in vals)
        print(row_str)

    # 各action类型PnL
    print(f"\n  -- 分类型PnL --")
    all_actions = set()
    for r in results:
        all_actions.update(r['action_stats'].keys())
    for act in sorted(all_actions):
        vals = [r['action_stats'].get(act, {'pnl': 0})['pnl'] for r in results]
        cnts = [r['action_stats'].get(act, {'count': 0})['count'] for r in results]
        row_str = f"  {act:<20}"
        for v, c in zip(vals, cnts):
            row_str += f"  {v:>+10,.0f}({c:>3})  "
        print(row_str)

    # vs 基线差异
    base = results[0]
    print(f"\n  -- vs A基线差异 --")
    print(f"  {'变体':<22} {'收益差':>10} {'Alpha差':>10} {'回撤差':>10} {'PF差':>8}")
    for r in results[1:]:
        print(f"  {r['label']:<22} "
              f"{r['return_pct']-base['return_pct']:>+10.2f} "
              f"{r['alpha']-base['alpha']:>+10.2f} "
              f"{r['max_dd']-base['max_dd']:>+10.2f} "
              f"{r['full_pf']-base['full_pf']:>+8.2f}")


if __name__ == '__main__':
    main()
