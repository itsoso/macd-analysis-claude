"""
多周期联合决策 · 60天/30天/7天 真实回测对比

使用 optimize_six_book 优化出的最优策略配置,
在 多周期联合决策模式 下分别在最近60天、30天和7天的真实币安数据上回测。
使用统一的 fuse_tf_scores 融合算法 (与实盘完全一致)。
"""

import pandas as pd
import numpy as np
import json
import sys
import os
import time
import argparse
import socket
import tempfile
import fcntl
from contextlib import contextmanager
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binance_fetcher import fetch_binance_klines
from indicators import add_all_indicators
from ma_indicators import add_moving_averages
from optimize_six_book import (
    compute_signals_six, run_strategy, run_strategy_multi_tf,
    _build_tf_score_index, calc_fusion_score_six, ALL_TIMEFRAMES
)
from strategy_futures import FuturesEngine


def fetch_data_for_tf(tf, days):
    """获取指定时间框架和天数的数据"""
    fetch_days = days + 30
    try:
        df = fetch_binance_klines("ETHUSDT", interval=tf, days=fetch_days)
        if df is not None and len(df) > 50:
            df = add_all_indicators(df)
            add_moving_averages(df, timeframe=tf)
            return df
    except Exception as e:
        print(f"  获取 {tf} 数据失败: {e}")
    return None


def _clean_json(obj):
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    return obj


def _atomic_dump_json(path, data):
    """原子写入 JSON，避免并发写导致半文件。"""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_multi_tf_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(_clean_json(data), f, ensure_ascii=False, default=str, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@contextmanager
def _file_lock(lock_path):
    """进程级排它锁，避免同机并发回测写冲突。"""
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def _slice_train_window(df, test_days=60, train_days=60):
    """
    样本外切片: 训练集使用测试窗口之前的区间，严格不与测试窗口重叠。
    """
    if df is None or len(df) == 0:
        return df
    test_start = df.index[-1] - pd.Timedelta(days=test_days)
    train_start = test_start - pd.Timedelta(days=train_days)
    sliced = df[(df.index >= train_start) & (df.index < test_start)].copy()
    return sliced


def _apply_conservative_risk(config):
    """
    保守仓位约束，降低回测虚高风险。
    """
    safe = dict(config)
    safe["single_pct"] = min(float(safe.get("single_pct", 0.20)), 0.10)
    safe["total_pct"] = min(float(safe.get("total_pct", 0.50)), 0.35)
    safe["margin_use"] = min(float(safe.get("margin_use", 0.70)), 0.50)
    safe["lev"] = min(int(safe.get("lev", 5)), 3)
    safe["max_lev"] = min(int(safe.get("max_lev", safe["lev"])), 3)
    return safe


def _scale_runtime_config(base_config, primary_tf, tf_hours, name):
    """
    按主周期缩放 hold/cooldown，避免不同TF直接复用同一bars参数。
    """
    config = dict(base_config)
    config["name"] = name
    tf_h = tf_hours.get(primary_tf, 1)
    config["short_max_hold"] = max(6, int(config.get("short_max_hold", 72) / tf_h))
    config["long_max_hold"] = max(6, int(config.get("long_max_hold", 72) / tf_h))
    config["cooldown"] = max(1, int(config.get("cooldown", 4) / tf_h))
    config["spot_cooldown"] = max(2, int(config.get("spot_cooldown", 12) / tf_h))
    return config


def _select_oos_base_config(
    opt_result,
    all_data,
    test_days,
    train_days,
    tf_hours,
    fallback_config,
    ref_tf,
    top_n=8,
    conservative=False,
):
    """
    从优化候选中做一次严格样本外筛选:
      - 训练窗口: [T-test_days-train_days, T-test_days)
      - 测试窗口: [T-test_days, T]
    """
    if ref_tf not in all_data:
        return dict(fallback_config), {"mode": "fallback", "reason": f"ref_tf {ref_tf} 不可用"}

    top = opt_result.get("global_top30") or []
    candidates = []

    gb_cfg = (opt_result.get("global_best") or {}).get("config")
    if isinstance(gb_cfg, dict):
        candidates.append(dict(gb_cfg))

    for row in top:
        cfg = row.get("config")
        if isinstance(cfg, dict):
            candidates.append(dict(cfg))
        if len(candidates) >= top_n:
            break

    candidates.append(dict(fallback_config))

    # 去重
    seen = set()
    uniq = []
    for cfg in candidates:
        key = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(cfg)

    train_df = _slice_train_window(all_data[ref_tf], test_days=test_days, train_days=train_days)
    if train_df is None or len(train_df) < 120:
        return dict(fallback_config), {
            "mode": "fallback",
            "reason": f"训练窗口不足: {0 if train_df is None else len(train_df)} bars",
        }

    train_start = train_df.index[0]
    train_end = train_df.index[-1]
    train_data_all = {}
    for tf, df in all_data.items():
        sliced = df[(df.index >= train_start) & (df.index <= train_end)].copy()
        if len(sliced) > 50:
            train_data_all[tf] = sliced

    # 训练信号与配置无关，计算一次复用
    train_signals = compute_signals_six(train_df, ref_tf, train_data_all, max_bars=0)

    scored = []
    for i, cand_cfg in enumerate(uniq):
        merged = {**fallback_config, **cand_cfg}
        if conservative:
            merged = _apply_conservative_risk(merged)
        runtime_cfg = _scale_runtime_config(merged, ref_tf, tf_hours, f"oos_train_{i+1}_{ref_tf}")

        r = run_strategy(train_df, train_signals, runtime_cfg, tf=ref_tf, trade_days=train_days)
        objective = float(r["alpha"]) - max(0.0, abs(float(r["max_drawdown"])) - 20.0) * 0.5
        scored.append(
            {
                "objective": objective,
                "alpha": float(r["alpha"]),
                "max_drawdown": float(r["max_drawdown"]),
                "config": merged,
                "tag": merged.get("tag", f"candidate_{i+1}"),
            }
        )

    scored.sort(key=lambda x: x["objective"], reverse=True)
    best = scored[0]
    meta = {
        "mode": "strict_oos",
        "ref_tf": ref_tf,
        "train_days": train_days,
        "test_days": test_days,
        "candidates": len(scored),
        "selected_tag": best["tag"],
        "selected_alpha_train": round(best["alpha"], 2),
        "selected_mdd_train": round(best["max_drawdown"], 2),
        "selected_objective_train": round(best["objective"], 2),
    }
    return dict(best["config"]), meta


def main(args):
    runner = args.runner
    host = socket.gethostname()
    pid = os.getpid()
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{runner}_{pid}"
    use_oos = not args.disable_oos
    use_conservative = not args.disable_conservative
    use_live_gate = not args.disable_live_gate

    print("=" * 120)
    print("  多周期联合决策 · 60天/30天/7天 真实回测对比")
    print("  数据源: 币安 ETH/USDT 真实K线 · 含手续费/滑点/资金费率")
    print("  融合算法: fuse_tf_scores (回测/实盘统一)")
    print(f"  现实约束: conservative={'ON' if use_conservative else 'OFF'} "
          f"| strict_oos={'ON' if use_oos else 'OFF'} "
          f"| live_gate={'ON' if use_live_gate else 'OFF'}")
    print(f"  运行标识: {run_id} ({runner}@{host})")
    print("=" * 120)

    # ======================================================
    # 从优化结果中加载最优策略配置
    # ======================================================
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'optimize_six_book_result.json')
    if not os.path.exists(result_path):
        print("错误: optimize_six_book_result.json 不存在, 请先运行 optimize_six_book.py")
        return

    with open(result_path, 'r', encoding='utf-8') as f:
        opt_result = json.load(f)

    # 取全局最优配置
    global_best = opt_result.get('global_best', {})
    best_config = global_best.get('config', {})
    best_tf = global_best.get('tf', '1h')
    best_alpha = global_best.get('alpha', 0)

    print(f"\n  全局最优单TF: {best_tf} α={best_alpha:+.2f}%")
    print(f"  融合模式: {best_config.get('fusion_mode', 'c6_veto_4')}")

    # ======================================================
    # 定义多周期组合方案
    # ======================================================
    # 回测用到的所有TF
    all_tfs_needed = ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '24h']

    # 主TF候选
    primary_tf_candidates = ['1h', '2h', '4h']

    # TF组合方案
    multi_tf_combos = [
        ('核心周期', ['30m', '1h', '4h', '8h', '24h']),
        ('全周期',   ['15m', '30m', '1h', '2h', '4h', '8h', '12h', '24h']),
        ('大周期(≥1h)', ['1h', '2h', '4h', '8h', '12h', '24h']),
        ('均衡搭配', ['15m', '1h', '4h', '12h']),
        ('中大周期', ['1h', '2h', '4h', '8h', '12h']),
        ('快慢双层', ['15m', '30m', '4h', '24h']),
    ]

    # 基础参数
    f12_base = {
        'single_pct': 0.20, 'total_pct': 0.50, 'lifetime_pct': 5.0,
        'sell_threshold': best_config.get('sell_threshold', 18),
        'buy_threshold': best_config.get('buy_threshold', 25),
        'short_threshold': best_config.get('short_threshold', 25),
        'long_threshold': best_config.get('long_threshold', 40),
        'close_short_bs': best_config.get('close_short_bs', 40),
        'close_long_ss': best_config.get('close_long_ss', 40),
        'sell_pct': best_config.get('sell_pct', 0.55),
        'margin_use': best_config.get('margin_use', 0.70),
        'lev': best_config.get('lev', 5),
        'max_lev': best_config.get('max_lev', 5),
        'short_sl': best_config.get('short_sl', -0.25),
        'short_tp': best_config.get('short_tp', 0.60),
        'short_trail': best_config.get('short_trail', 0.25),
        'short_max_hold': best_config.get('short_max_hold', 72),
        'long_sl': best_config.get('long_sl', -0.08),
        'long_tp': best_config.get('long_tp', 0.30),
        'long_trail': best_config.get('long_trail', 0.20),
        'long_max_hold': best_config.get('long_max_hold', 72),
        'trail_pullback': best_config.get('trail_pullback', 0.60),
        'cooldown': best_config.get('cooldown', 4),
        'spot_cooldown': best_config.get('spot_cooldown', 12),
        'use_partial_tp': best_config.get('use_partial_tp', False),
        'partial_tp_1': best_config.get('partial_tp_1', 0.20),
        'partial_tp_1_pct': best_config.get('partial_tp_1_pct', 0.30),
        'use_partial_tp_2': best_config.get('use_partial_tp_2', False),
        'partial_tp_2': best_config.get('partial_tp_2', 0.50),
        'partial_tp_2_pct': best_config.get('partial_tp_2_pct', 0.30),
        'use_atr_sl': best_config.get('use_atr_sl', False),
        'atr_sl_mult': best_config.get('atr_sl_mult', 3.0),
        'fusion_mode': best_config.get('fusion_mode', 'c6_veto_4'),
        'veto_threshold': best_config.get('veto_threshold', 25),
        'kdj_bonus': best_config.get('kdj_bonus', 0.09),
        'kdj_weight': best_config.get('kdj_weight', 0.15),
        'kdj_strong_mult': best_config.get('kdj_strong_mult', 1.25),
        'kdj_normal_mult': best_config.get('kdj_normal_mult', 1.12),
        'kdj_reverse_mult': best_config.get('kdj_reverse_mult', 0.70),
        'kdj_gate_threshold': best_config.get('kdj_gate_threshold', 10),
        'veto_dampen': best_config.get('veto_dampen', 0.30),
    }
    if use_conservative:
        f12_base = _apply_conservative_risk(f12_base)

    # ======================================================
    # 获取数据
    # ======================================================
    print(f"\n[1/4] 获取数据 (时间框架: {', '.join(all_tfs_needed)})...")

    all_data = {}
    history_days = 60 + max(args.train_days, 30)
    for tf in all_tfs_needed:
        print(f"  获取 {tf} 数据 ({history_days}天 + 30天缓冲)...")
        df = fetch_data_for_tf(tf, history_days)
        if df is not None:
            all_data[tf] = df
            print(f"    {tf}: {len(df)} 条K线, {df.index[0]} ~ {df.index[-1]}")
        else:
            print(f"    {tf}: 失败!")

    available_tfs = list(all_data.keys())
    print(f"\n  可用时间框架: {', '.join(available_tfs)}")

    # 过滤组合中不可用的TF
    multi_tf_combos = [
        (name, [tf for tf in tfs if tf in available_tfs])
        for name, tfs in multi_tf_combos
    ]
    multi_tf_combos = [(n, t) for n, t in multi_tf_combos if len(t) >= 2]

    primary_tf_candidates = [tf for tf in primary_tf_candidates if tf in available_tfs]

    # ======================================================
    # 计算信号
    # ======================================================
    print(f"\n[2/4] 计算六维信号...")
    all_signals = {}
    for tf in available_tfs:
        print(f"  计算 {tf} 信号...")
        all_signals[tf] = compute_signals_six(all_data[tf], tf, all_data, max_bars=2000)
    print(f"  信号计算完成: {len(all_signals)} 个TF")

    tf_hours = {
        '10m': 1/6, '15m': 0.25, '30m': 0.5, '1h': 1, '2h': 2, '3h': 3,
        '4h': 4, '6h': 6, '8h': 8, '12h': 12, '16h': 16, '24h': 24,
    }

    oos_meta = {
        "mode": "disabled",
        "reason": "disable_oos=true" if not use_oos else "not_applied",
    }
    if use_oos and primary_tf_candidates:
        ref_tf = best_tf if best_tf in available_tfs else primary_tf_candidates[0]
        selected_cfg, oos_meta = _select_oos_base_config(
            opt_result=opt_result,
            all_data=all_data,
            test_days=60,
            train_days=args.train_days,
            tf_hours=tf_hours,
            fallback_config=f12_base,
            ref_tf=ref_tf,
            top_n=args.oos_candidates,
            conservative=use_conservative,
        )
        f12_base = selected_cfg
        print("\n[2.5/4] 严格样本外筛选完成:")
        print(f"  参考TF: {oos_meta.get('ref_tf', ref_tf)}")
        print(f"  候选数: {oos_meta.get('candidates', 0)}")
        print(f"  选中: {oos_meta.get('selected_tag', '-')}")
        print(f"  训练集α: {oos_meta.get('selected_alpha_train', 0):+.2f}%")
        print(f"  训练集回撤: {oos_meta.get('selected_mdd_train', 0):.2f}%")

    # 构建评分索引
    print(f"\n[3/4] 构建TF评分索引...")
    tf_score_index = _build_tf_score_index(all_data, all_signals, available_tfs, f12_base)
    for tf in available_tfs:
        n_scores = len(tf_score_index.get(tf, {}))
        print(f"    {tf}: {n_scores} 个评分点")

    # ======================================================
    # 分别运行 30天 和 7天 回测
    # ======================================================
    periods = [
        {'days': 60, 'label': '最近60天'},
        {'days': 30, 'label': '最近30天'},
        {'days': 7,  'label': '最近7天'},
    ]

    all_period_results = {}

    # 先计算单TF基线 (30天和7天)
    single_tf_baselines = {}
    for period in periods:
        days = period['days']
        baselines = {}
        for ptf in primary_tf_candidates:
            if ptf not in all_data:
                continue
            df = all_data[ptf]
            sigs = all_signals[ptf]
            config = _scale_runtime_config(f12_base, ptf, tf_hours, f'单TF_{ptf}_{days}d')
            r = run_strategy(df, sigs, config, tf=ptf, trade_days=days)
            baselines[ptf] = {
                'alpha': r['alpha'],
                'strategy_return': r['strategy_return'],
                'buy_hold_return': r['buy_hold_return'],
                'max_drawdown': r['max_drawdown'],
                'total_trades': r['total_trades'],
            }
        single_tf_baselines[days] = baselines

    for period in periods:
        days = period['days']
        label = period['label']

        print(f"\n{'=' * 120}")
        print(f"  [回测] {label} — 多周期联合决策 (trade_days={days})")
        print(f"{'=' * 120}")

        # Buy & Hold 基准
        bh_by_tf = {}
        for tf in available_tfs:
            df = all_data[tf]
            end_dt = df.index[-1]
            start_dt = end_dt - pd.Timedelta(days=days)
            start_idx = df.index.searchsorted(start_dt)
            if start_idx >= len(df):
                start_idx = 0
            sp = df['close'].iloc[start_idx]
            ep = df['close'].iloc[-1]
            bh_by_tf[tf] = round((ep / sp - 1) * 100, 2)
        bh_1h = bh_by_tf.get('1h', 0)

        # 单TF基线
        print(f"\n  单TF基线:")
        for ptf in primary_tf_candidates:
            bl = single_tf_baselines.get(days, {}).get(ptf, {})
            if bl:
                print(f"    {ptf}: α={bl['alpha']:+.2f}%  "
                      f"策略={bl['strategy_return']:+.2f}%  "
                      f"BH={bl['buy_hold_return']:+.2f}%  "
                      f"回撤={bl['max_drawdown']:.2f}%")

        # 多周期回测
        print(f"\n  {'方案':<20} {'主TF':>5} {'辅助TFs':<45} {'Alpha':>10} {'策略收益':>12} "
              f"{'BH':>8} {'现金':>8} {'回撤':>8} {'交易':>6} {'vs单TF':>10}")
        print('  ' + '-' * 130)

        period_results = []

        for combo_name, combo_tfs in multi_tf_combos:
            for ptf in primary_tf_candidates:
                if ptf not in all_data:
                    continue

                config = _scale_runtime_config(f12_base, ptf, tf_hours, f'多TF_{combo_name}@{ptf}_{days}d')

                r = run_strategy_multi_tf(
                    all_data[ptf], tf_score_index, combo_tfs, config,
                    primary_tf=ptf, trade_days=days
                )

                fees = r.get('fees', {})
                baseline_alpha = single_tf_baselines.get(days, {}).get(ptf, {}).get('alpha', 0)
                vs_single = r['alpha'] - baseline_alpha

                entry = {
                    'combo_name': combo_name,
                    'primary_tf': ptf,
                    'decision_tfs': combo_tfs,
                    'alpha': r['alpha'],
                    'strategy_return': r['strategy_return'],
                    'buy_hold_return': r['buy_hold_return'],
                    'cash_hold_return': 0.0,
                    'alpha_vs_cash': r['strategy_return'],
                    'max_drawdown': r['max_drawdown'],
                    'total_trades': r['total_trades'],
                    'liquidations': r['liquidations'],
                    'total_cost': fees.get('total_costs', 0),
                    'fees': fees,
                    'vs_single_tf': round(vs_single, 2),
                }
                period_results.append(entry)

                marker = ' ★' if vs_single > 0 else ''
                print(f"  {combo_name:<20} {ptf:>5} {','.join(combo_tfs):<45} "
                      f"{r['alpha']:>+9.2f}% {r['strategy_return']:>+11.2f}% "
                      f"{r['buy_hold_return']:>+7.2f}% {0:>+7.2f}% {r['max_drawdown']:>7.2f}% "
                      f"{r['total_trades']:>5} {vs_single:>+9.2f}%{marker}")

        period_results.sort(key=lambda x: x['alpha'], reverse=True)
        all_period_results[days] = period_results

        # 汇总
        print(f"\n  === {label} 多周期联合决策 TOP5 ===")
        for i, r in enumerate(period_results[:5]):
            print(f"    #{i+1} {r['combo_name']}@{r['primary_tf']} "
                  f"α={r['alpha']:+.2f}% (vs单TF: {r['vs_single_tf']:+.2f}%) "
                  f"回撤={r['max_drawdown']:.2f}% 交易={r['total_trades']}")

    # ======================================================
    # 60天 vs 30天 vs 7天 对比
    # ======================================================
    r60 = all_period_results.get(60, [])
    r30 = all_period_results.get(30, [])
    r7 = all_period_results.get(7, [])

    print(f"\n{'=' * 120}")
    print(f"  60天 vs 30天 vs 7天 多周期联合决策对比")
    print(f"{'=' * 120}")

    print(f"\n  {'方案':<30} {'60天Alpha':>12} {'30天Alpha':>12} {'7天Alpha':>12}")
    print('  ' + '-' * 80)

    for s60 in r60[:15]:
        key = f"{s60['combo_name']}@{s60['primary_tf']}"
        s30_match = next((r for r in r30
                          if r['combo_name'] == s60['combo_name']
                          and r['primary_tf'] == s60['primary_tf']), None)
        s7_match = next((r for r in r7
                         if r['combo_name'] == s60['combo_name']
                         and r['primary_tf'] == s60['primary_tf']), None)
        a30 = f"{s30_match['alpha']:>+11.2f}%" if s30_match else f"{'--':>12}"
        a7 = f"{s7_match['alpha']:>+11.2f}%" if s7_match else f"{'--':>12}"
        print(f"  {key:<30} {s60['alpha']:>+11.2f}% {a30} {a7}")

    # 总体统计
    all_periods_data = [(60, r60, '60天'), (30, r30, '30天'), (7, r7, '7天')]
    valid_periods = [(d, r, l) for d, r, l in all_periods_data if r]
    if len(valid_periods) >= 2:
        print(f"\n  === 总体统计 ===")
        header = f"  {'指标':<25}" + "".join(f"{l:>15}" for _, _, l in valid_periods)
        print(header)
        print('  ' + '-' * (25 + 15 * len(valid_periods)))

        def _stat_line(label, fn):
            return f"  {label:<25}" + "".join(f"{fn(r):>+14.2f}%" for _, r, _ in valid_periods)

        print(_stat_line('平均Alpha', lambda r: np.mean([x['alpha'] for x in r])))
        print(_stat_line('最优Alpha', lambda r: max(x['alpha'] for x in r)))
        print(_stat_line('最差Alpha', lambda r: min(x['alpha'] for x in r)))
        print(f"  {'盈利策略数':<25}" +
              "".join(f"{sum(1 for x in r if x['alpha'] > 0):>15}" for _, r, _ in valid_periods))
        print(f"  {'平均交易数':<25}" +
              "".join(f"{np.mean([x['total_trades'] for x in r]):>15.0f}" for _, r, _ in valid_periods))

    # ======================================================
    # 保存结果
    # ======================================================
    def _format_period_results(results_list):
        return [{
            'rank': i + 1,
            'combo_name': r['combo_name'],
            'primary_tf': r['primary_tf'],
            'decision_tfs': r['decision_tfs'],
            'alpha': r['alpha'],
            'strategy_return': r['strategy_return'],
            'buy_hold_return': r['buy_hold_return'],
            'cash_hold_return': r.get('cash_hold_return', 0.0),
            'alpha_vs_cash': r.get('alpha_vs_cash', r['strategy_return']),
            'max_drawdown': r['max_drawdown'],
            'total_trades': r['total_trades'],
            'liquidations': r.get('liquidations', 0),
            'total_cost': r.get('total_cost', 0),
            'vs_single_tf': r.get('vs_single_tf', 0),
        } for i, r in enumerate(results_list)]

    def _period_summary(results_list):
        if not results_list:
            return {}
        return {
            'avg_alpha': round(np.mean([r['alpha'] for r in results_list]), 2),
            'best_alpha': round(max(r['alpha'] for r in results_list), 2),
            'worst_alpha': round(min(r['alpha'] for r in results_list), 2),
            'profitable_count': sum(1 for r in results_list if r['alpha'] > 0),
            'total_count': len(results_list),
            'avg_trades': round(np.mean([r['total_trades'] for r in results_list]), 1),
        }

    output = {
        'description': '多周期联合决策 · 60天/30天/7天 真实回测对比 (统一fuse_tf_scores)',
        'run_time': datetime.now().isoformat(),
        'data_source': '币安 ETH/USDT 真实K线',
        'fusion_algorithm': 'fuse_tf_scores (回测/实盘统一)',
        'run_meta': {
            'run_id': run_id,
            'runner': runner,
            'host': host,
            'pid': pid,
        },
        'realism': {
            'conservative_risk': use_conservative,
            'strict_oos': use_oos,
            'train_days': args.train_days,
            'oos_candidates': args.oos_candidates,
            'oos_selection': oos_meta,
        },
        'base_config': {
            'best_single_tf': best_tf,
            'best_single_alpha': best_alpha,
            'fusion_mode': f12_base.get('fusion_mode', 'c6_veto_4'),
        },
        'fee_model': {
            'taker_fee': '0.05%',
            'slippage': '0.1%',
            'funding_rate': '±0.01%/8h',
            'liquidation_fee': '0.5%',
        },
        'single_tf_baselines': single_tf_baselines,
        'results_60d': _format_period_results(r60),
        'results_30d': _format_period_results(r30),
        'results_7d': _format_period_results(r7),
    }

    summary = {}
    for key, results in [('60d', r60), ('30d', r30), ('7d', r7)]:
        s = _period_summary(results)
        if s:
            summary[key] = s
    if summary:
        output['summary'] = summary

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(args.results_dir or os.path.join(base_dir, "data", "backtests"))
    latest_path = os.path.join(results_dir, "backtest_multi_tf_30d_7d_result.json")
    runs_dir = os.path.join(results_dir, "multi_tf_60_30_7_runs")
    snapshot_path = os.path.join(runs_dir, f"{run_id}.json")
    index_path = os.path.join(runs_dir, "index.json")
    lock_path = os.path.join(results_dir, ".multi_tf_60_30_7.lock")

    with _file_lock(lock_path):
        _atomic_dump_json(snapshot_path, output)

        index_data = []
        if os.path.exists(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
            except Exception:
                index_data = []

        record = {
            "run_id": run_id,
            "run_time": output["run_time"],
            "runner": runner,
            "host": host,
            "pid": pid,
            "snapshot_file": os.path.basename(snapshot_path),
            "top_60d": {
                "combo": f"{r60[0]['combo_name']}@{r60[0]['primary_tf']}" if r60 else "-",
                "alpha": r60[0]["alpha"] if r60 else 0,
            },
            "top_30d": {
                "combo": f"{r30[0]['combo_name']}@{r30[0]['primary_tf']}" if r30 else "-",
                "alpha": r30[0]["alpha"] if r30 else 0,
            },
            "top_7d": {
                "combo": f"{r7[0]['combo_name']}@{r7[0]['primary_tf']}" if r7 else "-",
                "alpha": r7[0]["alpha"] if r7 else 0,
            },
        }
        index_data = [record] + [x for x in index_data if x.get("run_id") != run_id]
        index_data = index_data[:200]
        _atomic_dump_json(index_path, index_data)

        output["recent_runs"] = index_data[:20]
        if not args.no_update_latest:
            _atomic_dump_json(latest_path, output)

    if args.no_update_latest:
        print(f"\n结果已生成(未覆盖最新): {latest_path}")
    else:
        print(f"\n结果已保存(最新): {latest_path}")
    print(f"结果已归档(快照): {snapshot_path}")
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="多周期联合决策 60/30/7 回测 (冲突安全写入)",
    )
    parser.add_argument(
        "--runner",
        type=str,
        default=os.environ.get("BACKTEST_RUNNER", "local"),
        help="回测执行来源标识 (如 local / online / claude / codex)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "backtests"),
        help="结果输出目录 (默认 data/backtests)",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=60,
        help="严格样本外筛选的训练窗口天数 (默认 60)",
    )
    parser.add_argument(
        "--oos-candidates",
        type=int,
        default=8,
        help="样本外筛选时评估的候选参数数量 (默认 8)",
    )
    parser.add_argument(
        "--disable-oos",
        action="store_true",
        help="关闭严格样本外筛选 (默认开启)",
    )
    parser.add_argument(
        "--disable-conservative",
        action="store_true",
        help="关闭保守仓位约束 (默认开启)",
    )
    parser.add_argument(
        "--no-update-latest",
        action="store_true",
        help="仅生成快照和索引，不覆盖最新结果文件",
    )
    cli_args = parser.parse_args()
    main(cli_args)
