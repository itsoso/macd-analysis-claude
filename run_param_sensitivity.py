#!/usr/bin/env python3
"""参数有效性验证 & 权重优化管道

核心目标:
  1. 六书消融实验 (Ablation Study) — 逐本消除，量化每本书的真实 alpha 贡献
  2. 权重网格搜索 — 系统性搜索最优 DIV/MA 基础权重 + 四书 bonus 组合
  3. 关键参数敏感度分析 — Entry 阈值、Regime 阈值、Veto 参数
  4. Walk-Forward 验证 — 确保优化后的参数在 OOS 上稳健
  5. 统计显著性检验 — DSR/PSR 评估 alpha 可信度

用法:
  cd /path/to/project && .venv/bin/python3 run_param_sensitivity.py
"""
import copy
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from itertools import product

# 确保输出不缓冲
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from run_p0_oos_validation import load_base_config
from run_p1_p2_sensitivity import prepare_data
from optimize_six_book import _build_tf_score_index, run_strategy_multi_tf
from walk_forward_pipeline import (
    V8_OVERRIDES, V10_2_PRODUCTION,
    run_single_window, generate_windows,
    compute_sharpe_ratio, compute_psr, compute_dsr,
)


# ============================================================
# 常量
# ============================================================
PRIMARY_TF = '1h'
DECISION_TFS = ['15m', '1h', '4h', '24h']

# 数据区间
DATA_START = '2023-10-01'
DATA_END = '2026-02-28'

# IS / OOS 分割 (时间正向)
IS_START = '2024-01-01'
IS_END = '2025-06-30'
OOS_START = '2025-07-01'
OOS_END = '2026-01-31'

# Walk-Forward 配置
WF_IS_MONTHS = 6
WF_OOS_MONTHS = 1
WF_STEP_MONTHS = 1

# 输出目录
OUTPUT_DIR = 'param_sensitivity_results'


# ============================================================
# 工具函数
# ============================================================

def get_production_config():
    """获取当前生产配置"""
    cfg = load_base_config()
    cfg.update(V8_OVERRIDES)
    cfg.update(V10_2_PRODUCTION)
    return cfg


def run_backtest(all_data, all_signals, needed_tfs, cfg,
                 start_dt, end_dt):
    """运行单次回测并返回标准化结果"""
    tf_score_map = _build_tf_score_index(all_data, all_signals, needed_tfs, cfg)
    result = run_strategy_multi_tf(
        primary_df=all_data[PRIMARY_TF],
        tf_score_map=tf_score_map,
        decision_tfs=DECISION_TFS,
        config=cfg,
        primary_tf=PRIMARY_TF,
        trade_days=0,
        trade_start_dt=start_dt,
        trade_end_dt=end_dt,
    )

    trades = result.get('trades', [])
    closes = [t for t in trades if t.get('action', '').startswith('CLOSE_')]
    n = len(closes)
    wins = sum(1 for t in closes if float(t.get('pnl', 0)) > 0)
    wr = wins / n * 100 if n > 0 else 0
    gp = sum(float(t.get('pnl', 0)) for t in closes if float(t.get('pnl', 0)) > 0)
    gl = abs(sum(float(t.get('pnl', 0)) for t in closes if float(t.get('pnl', 0)) <= 0))
    pf = gp / gl if gl > 0 else 0
    pnls = [float(t.get('pnl', 0)) for t in closes]

    # 计算期望值 Expectancy
    expectancy = np.mean(pnls) if pnls else 0

    # 计算 Sharpe
    if len(pnls) >= 2 and np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls, ddof=1) * np.sqrt(len(pnls))
    else:
        sharpe = 0

    return {
        'strategy_return': result.get('strategy_return', 0),
        'max_drawdown': result.get('max_drawdown', 0),
        'total_trades': n,
        'win_rate': wr,
        'profit_factor': pf,
        'sharpe': sharpe,
        'expectancy': expectancy,
        'gross_profit': gp,
        'gross_loss': gl,
        'pnls': pnls,
        'alpha': result.get('alpha', 0),
        'calmar': result.get('strategy_return', 0) / abs(result.get('max_drawdown', -1)) if result.get('max_drawdown', 0) != 0 else 0,
    }


def format_result(r, tag=''):
    """格式化单次回测结果"""
    return (f"  {tag:<40s} Ret={r['strategy_return']:+7.1f}%  "
            f"WR={r['win_rate']:5.1f}%  PF={r['profit_factor']:5.2f}  "
            f"T={r['total_trades']:3d}  MDD={r['max_drawdown']:6.1f}%  "
            f"Sharpe={r['sharpe']:5.2f}  E[R]=${r['expectancy']:+,.0f}")


# ============================================================
# Phase 1: 六书消融实验 (Ablation Study)
# ============================================================

def run_ablation_study(all_data, all_signals, needed_tfs, base_cfg,
                       start_dt, end_dt, label=''):
    """逐本消除实验：量化每本书的 alpha 贡献

    方法：
    1. 运行完整六书基准
    2. 每次禁用一本书，看性能下降多少
    3. 下降越大 = 该书贡献越大
    """
    print(f"\n{'='*120}")
    print(f"  Phase 1: 六书消融实验 (Ablation Study) [{label}]")
    print(f"{'='*120}")

    results = {}

    # 1. 基准 (全部六书)
    print("\n  [基准] 完整六书...")
    baseline = run_backtest(all_data, all_signals, needed_tfs, base_cfg,
                            start_dt, end_dt)
    results['baseline'] = baseline
    print(format_result(baseline, '基准(全六书)'))

    # 2. 逐本消除
    # 禁用方法：将对应书的权重/bonus 设为 0
    ablation_configs = {
        'no_DIV': {'c6_div_weight': 0.0},  # DIV=0, MA=100%
        'no_MA': {'c6_div_weight': 1.0},   # DIV=100%, MA=0
        'no_CS': {'cs_bonus': 0.0},
        'no_BB': {'bb_bonus': 0.0},
        'no_VP': {'vp_bonus': 0.0},
        'no_KDJ': {'kdj_bonus': 0.0},
        'no_all_bonus': {'cs_bonus': 0.0, 'bb_bonus': 0.0, 'vp_bonus': 0.0, 'kdj_bonus': 0.0},
    }

    for name, overrides in ablation_configs.items():
        cfg = copy.deepcopy(base_cfg)
        cfg.update(overrides)
        print(f"  [{name}]...")
        r = run_backtest(all_data, all_signals, needed_tfs, cfg,
                         start_dt, end_dt)
        results[name] = r
        # 计算 delta vs baseline
        delta_ret = r['strategy_return'] - baseline['strategy_return']
        delta_pf = r['profit_factor'] - baseline['profit_factor']
        print(format_result(r, name) + f"  ΔRet={delta_ret:+.1f}% ΔPF={delta_pf:+.2f}")

    # 3. 计算各书贡献度
    print(f"\n  === 各书 Alpha 贡献度 (Δ = 移除后的性能变化) ===")
    print(f"  {'书名':<12s} {'ΔRet':>8s} {'ΔPF':>8s} {'ΔWR':>8s} {'ΔSharpe':>10s} {'贡献评级':>10s}")
    print(f"  {'-'*60}")

    contributions = {}
    for name, r in results.items():
        if name == 'baseline':
            continue
        delta_ret = baseline['strategy_return'] - r['strategy_return']  # 注意：正值=移除后变差=有贡献
        delta_pf = baseline['profit_factor'] - r['profit_factor']
        delta_wr = baseline['win_rate'] - r['win_rate']
        delta_sharpe = baseline['sharpe'] - r['sharpe']

        # 综合评分
        score = delta_ret * 0.3 + delta_pf * 10 + delta_sharpe * 5
        if score > 5:
            rating = "★★★ 关键"
        elif score > 1:
            rating = "★★  重要"
        elif score > 0:
            rating = "★   有效"
        else:
            rating = "✗   负贡献"

        contributions[name] = {
            'delta_ret': delta_ret, 'delta_pf': delta_pf,
            'delta_wr': delta_wr, 'delta_sharpe': delta_sharpe,
            'score': score, 'rating': rating,
        }
        print(f"  {name:<12s} {delta_ret:+7.1f}% {delta_pf:+7.2f} "
              f"{delta_wr:+7.1f}% {delta_sharpe:+9.2f}  {rating}")

    return results, contributions


# ============================================================
# Phase 2: 权重网格搜索
# ============================================================

def run_weight_grid_search(all_data, all_signals, needed_tfs, base_cfg,
                           start_dt, end_dt, label=''):
    """系统性搜索 DIV/MA 权重 + 四书 Bonus 最优组合"""
    print(f"\n{'='*120}")
    print(f"  Phase 2: 权重网格搜索 [{label}]")
    print(f"{'='*120}")

    results = []

    # Phase 2a: DIV 权重搜索
    print("\n  [2a] DIV/MA 基础权重搜索...")
    div_weights = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    for div_w in div_weights:
        cfg = copy.deepcopy(base_cfg)
        cfg['c6_div_weight'] = div_w
        tag = f"DIV={div_w:.0%}/MA={1-div_w:.0%}"
        r = run_backtest(all_data, all_signals, needed_tfs, cfg,
                         start_dt, end_dt)
        r['tag'] = tag
        r['div_weight'] = div_w
        results.append(r)
        print(format_result(r, tag))

    # 找最优 DIV 权重
    best_by_pf = max(results, key=lambda x: x['profit_factor'])
    best_by_ret = max(results, key=lambda x: x['strategy_return'])
    best_by_sharpe = max(results, key=lambda x: x['sharpe'])

    print(f"\n  最优 DIV 权重 (按 PF): {best_by_pf['tag']} PF={best_by_pf['profit_factor']:.2f}")
    print(f"  最优 DIV 权重 (按 Ret): {best_by_ret['tag']} Ret={best_by_ret['strategy_return']:+.1f}%")
    print(f"  最优 DIV 权重 (按 Sharpe): {best_by_sharpe['tag']} Sharpe={best_by_sharpe['sharpe']:.2f}")

    # Phase 2b: 四书 Bonus 联合搜索 (使用最优 DIV 权重)
    best_div_w = best_by_pf['div_weight']
    print(f"\n  [2b] 四书 Bonus 联合搜索 (固定 DIV={best_div_w:.0%})...")

    bonus_grid = {
        'cs_bonus': [0.0, 0.03, 0.06, 0.10, 0.15],
        'bb_bonus': [0.0, 0.05, 0.10, 0.15],
        'vp_bonus': [0.0, 0.04, 0.08, 0.12],
        'kdj_bonus': [0.0, 0.05, 0.09, 0.15],
    }

    bonus_results = []
    total_combos = 1
    for vals in bonus_grid.values():
        total_combos *= len(vals)
    print(f"  搜索空间: {total_combos} 种组合")

    count = 0
    for cs_b in bonus_grid['cs_bonus']:
        for bb_b in bonus_grid['bb_bonus']:
            for vp_b in bonus_grid['vp_bonus']:
                for kdj_b in bonus_grid['kdj_bonus']:
                    count += 1
                    cfg = copy.deepcopy(base_cfg)
                    cfg['c6_div_weight'] = best_div_w
                    cfg['cs_bonus'] = cs_b
                    cfg['bb_bonus'] = bb_b
                    cfg['vp_bonus'] = vp_b
                    cfg['kdj_bonus'] = kdj_b

                    tag = f"CS={cs_b:.0%} BB={bb_b:.0%} VP={vp_b:.0%} KDJ={kdj_b:.0%}"
                    r = run_backtest(all_data, all_signals, needed_tfs, cfg,
                                     start_dt, end_dt)
                    r['tag'] = tag
                    r['params'] = {'div_w': best_div_w, 'cs_bonus': cs_b,
                                   'bb_bonus': bb_b, 'vp_bonus': vp_b, 'kdj_bonus': kdj_b}
                    bonus_results.append(r)

                    if count % 50 == 0:
                        print(f"  进度: {count}/{total_combos} ({count/total_combos*100:.0f}%)")

    # 排序
    bonus_results.sort(key=lambda x: x['profit_factor'], reverse=True)

    print(f"\n  === Bonus 搜索 Top 10 (按 PF 排序) ===")
    print(f"  {'#':>3s} {'组合':>45s} {'Ret':>8s} {'PF':>6s} {'WR':>6s} {'T':>5s} {'Sharpe':>8s}")
    print(f"  {'-'*90}")
    for i, r in enumerate(bonus_results[:10]):
        print(f"  {i+1:3d} {r['tag']:>45s} {r['strategy_return']:+7.1f}% "
              f"{r['profit_factor']:5.2f} {r['win_rate']:5.1f}% "
              f"{r['total_trades']:4d} {r['sharpe']:7.2f}")

    return results, bonus_results


# ============================================================
# Phase 3: 关键参数敏感度分析
# ============================================================

def run_param_sensitivity(all_data, all_signals, needed_tfs, base_cfg,
                          start_dt, end_dt, label=''):
    """关键参数逐一扫描，找出敏感度最高的参数"""
    print(f"\n{'='*120}")
    print(f"  Phase 3: 关键参数敏感度分析 [{label}]")
    print(f"{'='*120}")

    # 定义参数空间
    param_space = {
        # Entry 阈值
        'short_threshold': [20, 25, 30, 35, 40, 45, 50, 55, 60],
        'long_threshold': [15, 20, 25, 30, 35, 40, 45],
        'sell_threshold': [10, 12, 15, 18, 22, 25, 30],
        'buy_threshold': [15, 18, 20, 22, 25, 30, 35],

        # 平仓阈值
        'close_short_bs': [25, 30, 35, 40, 45, 50],
        'close_long_ss': [25, 30, 35, 40, 45, 50],

        # Veto 参数
        'veto_threshold': [15, 20, 25, 30, 35, 40],
        'veto_dampen': [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],

        # Soft Veto 参数
        'soft_veto_steepness': [1.0, 2.0, 3.0, 4.0, 5.0],
        'soft_veto_midpoint': [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],

        # Regime 阈值
        'regime_vol_high': [0.012, 0.015, 0.018, 0.020, 0.025, 0.030],
        'regime_trend_strong': [0.008, 0.010, 0.012, 0.015, 0.018, 0.020],

        # Dominance Ratio
        'entry_dominance_ratio': [1.0, 1.2, 1.3, 1.5, 1.8, 2.0],

        # Cooldown
        'cooldown': [2, 4, 6, 8, 10, 12],

        # ATR SL Mult
        'atr_sl_mult': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    }

    all_results = {}

    for param_name, values in param_space.items():
        print(f"\n  ── {param_name} ──")
        current_val = base_cfg.get(param_name, 'N/A')
        print(f"  当前值: {current_val}")

        param_results = []
        for val in values:
            cfg = copy.deepcopy(base_cfg)
            cfg[param_name] = val
            tag = f"{param_name}={val}"
            r = run_backtest(all_data, all_signals, needed_tfs, cfg,
                             start_dt, end_dt)
            r['tag'] = tag
            r['param_value'] = val
            param_results.append(r)

        # 打印结果
        best = max(param_results, key=lambda x: x['profit_factor'])
        worst = min(param_results, key=lambda x: x['profit_factor'])
        pf_range = best['profit_factor'] - worst['profit_factor']
        ret_range = max(r['strategy_return'] for r in param_results) - min(r['strategy_return'] for r in param_results)

        # 敏感度评级
        if pf_range > 0.3 or ret_range > 20:
            sensitivity = "🔴 高敏感"
        elif pf_range > 0.1 or ret_range > 10:
            sensitivity = "🟡 中敏感"
        else:
            sensitivity = "🟢 低敏感"

        print(f"  PF 范围: [{worst['profit_factor']:.2f}, {best['profit_factor']:.2f}] (Δ={pf_range:.2f})")
        print(f"  Ret 范围: [{min(r['strategy_return'] for r in param_results):+.1f}%, {max(r['strategy_return'] for r in param_results):+.1f}%] (Δ={ret_range:.1f}%)")
        print(f"  最优值: {best['param_value']} (PF={best['profit_factor']:.2f}, Ret={best['strategy_return']:+.1f}%)")
        print(f"  敏感度: {sensitivity}")

        for r in param_results:
            marker = " ◀ BEST" if r['param_value'] == best['param_value'] else ""
            marker += " ◀ 当前" if r['param_value'] == current_val else ""
            print(f"    {r['param_value']:>8s} → PF={r['profit_factor']:5.2f}  "
                  f"Ret={r['strategy_return']:+7.1f}%  WR={r['win_rate']:5.1f}%  "
                  f"T={r['total_trades']:3d}{marker}" if isinstance(r['param_value'], str) else
                  f"    {r['param_value']:>8.4f} → PF={r['profit_factor']:5.2f}  "
                  f"Ret={r['strategy_return']:+7.1f}%  WR={r['win_rate']:5.1f}%  "
                  f"T={r['total_trades']:3d}{marker}" if isinstance(r['param_value'], float) else
                  f"    {r['param_value']:>8d} → PF={r['profit_factor']:5.2f}  "
                  f"Ret={r['strategy_return']:+7.1f}%  WR={r['win_rate']:5.1f}%  "
                  f"T={r['total_trades']:3d}{marker}")

        all_results[param_name] = {
            'results': param_results,
            'best': best,
            'pf_range': pf_range,
            'ret_range': ret_range,
            'sensitivity': sensitivity,
            'current_val': current_val,
        }

    # 按敏感度排序
    print(f"\n  === 参数敏感度排名 ===")
    print(f"  {'参数名':>25s} {'当前值':>10s} {'最优值':>10s} {'PF范围':>8s} {'Ret范围':>10s} {'敏感度':>12s}")
    print(f"  {'-'*85}")

    sorted_params = sorted(all_results.items(),
                           key=lambda x: x[1]['pf_range'], reverse=True)
    for param_name, info in sorted_params:
        cv = str(info['current_val'])[:10]
        bv = str(info['best']['param_value'])[:10]
        is_optimal = " ✓" if str(info['current_val']) == str(info['best']['param_value']) else " ★ 需调整"
        print(f"  {param_name:>25s} {cv:>10s} {bv:>10s} "
              f"{info['pf_range']:7.2f} {info['ret_range']:+9.1f}% "
              f"{info['sensitivity']}{is_optimal}")

    return all_results


# ============================================================
# Phase 4: Walk-Forward 交叉验证
# ============================================================

def run_walk_forward_validation(all_data, all_signals, needed_tfs,
                                configs_to_test, config_names):
    """对多组参数配置运行 Walk-Forward 验证"""
    print(f"\n{'='*120}")
    print(f"  Phase 4: Walk-Forward 交叉验证")
    print(f"{'='*120}")

    windows = generate_windows(
        start_year=2024, start_month=1,
        end_year=2026, end_month=2,
        is_months=WF_IS_MONTHS, oos_months=WF_OOS_MONTHS,
        step_months=WF_STEP_MONTHS,
    )
    print(f"  Walk-Forward 窗口: {len(windows)} 个")

    all_wf_results = {}

    for cfg, name in zip(configs_to_test, config_names):
        print(f"\n  ── 测试配置: {name} ──")
        oos_results = []

        for i, (is_s, is_e, oos_s, oos_e) in enumerate(windows):
            oos_start_dt = pd.Timestamp(oos_s)
            oos_end_dt = pd.Timestamp(oos_e) + pd.Timedelta(days=3)

            try:
                oos_r = run_single_window(
                    all_data, all_signals, needed_tfs, cfg,
                    PRIMARY_TF, DECISION_TFS, oos_start_dt, oos_end_dt
                )
                oos_results.append(oos_r)
            except Exception as e:
                oos_results.append({
                    'strategy_return': 0, 'win_rate': 0, 'profit_factor': 0,
                    'total_trades': 0, 'sharpe': 0, 'pnls': [],
                    'max_drawdown': 0, 'calmar': 0,
                    'gross_profit': 0, 'gross_loss': 0,
                })

        # 汇总
        oos_returns = [r['strategy_return'] for r in oos_results]
        profitable_windows = sum(1 for r in oos_returns if r > 0)
        win_pct = profitable_windows / len(oos_returns) * 100 if oos_returns else 0
        avg_ret = np.mean(oos_returns) if oos_returns else 0
        total_trades = sum(r['total_trades'] for r in oos_results)

        # DSR/PSR
        all_pnls = []
        for r in oos_results:
            all_pnls.extend(r.get('pnls', []))

        if len(oos_returns) >= 3:
            oos_arr = np.array(oos_returns) / 100
            sr = compute_sharpe_ratio(oos_arr)
            skew = float(pd.Series(oos_arr).skew())
            kurt = float(pd.Series(oos_arr).kurtosis()) + 3
            psr = compute_psr(sr, len(oos_arr), skew, kurt, benchmark_sharpe=0)
            dsr = compute_dsr(sr, len(oos_arr), 50, skew, kurt)  # 假设50次实验
        else:
            sr = psr = dsr = 0

        print(f"  {name}: 窗口胜率={win_pct:.0f}% ({profitable_windows}/{len(oos_returns)})  "
              f"Avg月Ret={avg_ret:+.1f}%  总交易={total_trades}  "
              f"Sharpe={sr:.2f}  PSR={psr*100:.0f}%  DSR={dsr*100:.0f}%")

        all_wf_results[name] = {
            'oos_results': oos_results,
            'win_pct': win_pct,
            'avg_ret': avg_ret,
            'total_trades': total_trades,
            'sharpe': sr,
            'psr': psr,
            'dsr': dsr,
            'oos_returns': oos_returns,
        }

    # 对比
    print(f"\n  === Walk-Forward 对比 ===")
    print(f"  {'配置':>30s} {'窗口胜率':>10s} {'Avg月Ret':>10s} {'总交易':>8s} "
          f"{'Sharpe':>8s} {'PSR':>6s} {'DSR':>6s}")
    print(f"  {'-'*90}")
    for name, info in all_wf_results.items():
        print(f"  {name:>30s} {info['win_pct']:9.0f}% {info['avg_ret']:+9.1f}% "
              f"{info['total_trades']:7d} {info['sharpe']:7.2f} "
              f"{info['psr']*100:5.0f}% {info['dsr']*100:5.0f}%")

    return all_wf_results


# ============================================================
# Phase 5: 综合优化建议
# ============================================================

def generate_recommendations(ablation, weight_results, sensitivity, wf_results):
    """基于所有分析结果生成综合优化建议"""
    print(f"\n{'='*120}")
    print(f"  Phase 5: 综合优化建议")
    print(f"{'='*120}")

    recs = []

    # 1. 基于消融实验的权重建议
    _, contributions = ablation
    print(f"\n  [1] 基于消融实验的六书权重建议:")
    for name, info in sorted(contributions.items(), key=lambda x: x[1]['score'], reverse=True):
        print(f"      {name:15s} 贡献评分={info['score']:+.1f}  {info['rating']}")
        if info['score'] < 0:
            recs.append(f"⚠️ {name} 对策略有负贡献 (ΔRet={info['delta_ret']:+.1f}%)，建议降低权重或移除")

    # 2. 基于网格搜索的最优权重
    div_results, bonus_results = weight_results
    if bonus_results:
        best = bonus_results[0]
        print(f"\n  [2] 最优权重组合 (IS):")
        print(f"      {best['tag']}  PF={best['profit_factor']:.2f}  Ret={best['strategy_return']:+.1f}%")
        recs.append(f"最优权重: {best['params']}")

    # 3. 需要调整的参数
    if sensitivity:
        print(f"\n  [3] 需要调整的参数:")
        sorted_params = sorted(sensitivity.items(),
                               key=lambda x: x[1]['pf_range'], reverse=True)
        for param_name, info in sorted_params[:5]:
            if str(info['current_val']) != str(info['best']['param_value']):
                rec = (f"  {param_name}: {info['current_val']} → {info['best']['param_value']} "
                       f"(PF +{info['pf_range']:.2f})")
                print(f"      {rec}")
                recs.append(rec)

    # 4. Walk-Forward 验证状态
    if wf_results:
        print(f"\n  [4] Walk-Forward 验证状态:")
        for name, info in wf_results.items():
            status = "✅ 通过" if info['win_pct'] >= 60 else "❌ 未通过"
            dsr_status = "✅" if info['dsr'] >= 0.80 else "⚠️" if info['dsr'] >= 0.50 else "❌"
            print(f"      {name}: WF {status}  DSR {dsr_status} ({info['dsr']*100:.0f}%)")

    print(f"\n  === 建议总结 ===")
    for i, rec in enumerate(recs, 1):
        print(f"  {i}. {rec}")

    return recs


# ============================================================
# Main
# ============================================================

def main():
    start_time = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 120)
    print("  参数有效性验证 & 权重优化管道")
    print("  六书融合策略 v10.2 参数体系系统性分析")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 120)

    # ── 加载数据 ──
    print("\n  加载全量数据...")
    all_data, all_signals, needed_tfs, primary_tf, decision_tfs, _, _ = \
        prepare_data('ETHUSDT', DATA_START, DATA_END)
    print(f"  决策 TF: {decision_tfs}")
    print(f"  数据区间: {DATA_START} ~ {DATA_END}")

    # ── 基础配置 ──
    base_cfg = get_production_config()
    is_start_dt = pd.Timestamp(IS_START)
    is_end_dt = pd.Timestamp(IS_END)
    oos_start_dt = pd.Timestamp(OOS_START)
    oos_end_dt = pd.Timestamp(OOS_END)

    # ═══════════════════════════════════════════
    # Phase 1: 消融实验 (IS + OOS)
    # ═══════════════════════════════════════════
    print(f"\n  IS 区间: {IS_START} ~ {IS_END}")
    print(f"  OOS 区间: {OOS_START} ~ {OOS_END}")

    ablation_is = run_ablation_study(
        all_data, all_signals, needed_tfs, base_cfg,
        is_start_dt, is_end_dt, label='IS'
    )

    ablation_oos = run_ablation_study(
        all_data, all_signals, needed_tfs, base_cfg,
        oos_start_dt, oos_end_dt, label='OOS'
    )

    # ═══════════════════════════════════════════
    # Phase 2: 权重网格搜索 (IS)
    # ═══════════════════════════════════════════
    weight_results = run_weight_grid_search(
        all_data, all_signals, needed_tfs, base_cfg,
        is_start_dt, is_end_dt, label='IS'
    )

    # ═══════════════════════════════════════════
    # Phase 3: 参数敏感度 (IS)
    # ═══════════════════════════════════════════
    sensitivity = run_param_sensitivity(
        all_data, all_signals, needed_tfs, base_cfg,
        is_start_dt, is_end_dt, label='IS'
    )

    # ═══════════════════════════════════════════
    # Phase 4: Walk-Forward 验证
    # ═══════════════════════════════════════════
    # 测试 3 组配置: 当前、优化后(如果有差异)、全bonus=0(纯DIV+MA)
    configs_to_test = [base_cfg]
    config_names = ['当前生产配置(v10.2)']

    # 如果 Phase 2 找到了更好的权重，加入测试
    if weight_results[1]:
        best_bonus = weight_results[1][0]
        opt_cfg = copy.deepcopy(base_cfg)
        opt_cfg.update(best_bonus['params'])
        configs_to_test.append(opt_cfg)
        config_names.append(f'优化权重({best_bonus["tag"]})')

    # 纯双书（无 bonus）基准
    pure_cfg = copy.deepcopy(base_cfg)
    pure_cfg.update({'cs_bonus': 0, 'bb_bonus': 0, 'vp_bonus': 0, 'kdj_bonus': 0})
    configs_to_test.append(pure_cfg)
    config_names.append('纯DIV+MA(无Bonus)')

    wf_results = run_walk_forward_validation(
        all_data, all_signals, needed_tfs,
        configs_to_test, config_names,
    )

    # ═══════════════════════════════════════════
    # Phase 5: 综合建议
    # ═══════════════════════════════════════════
    recommendations = generate_recommendations(
        ablation_is, weight_results, sensitivity, wf_results
    )

    # ── 保存结果 ──
    elapsed = time.time() - start_time
    print(f"\n  总耗时: {elapsed/60:.1f} 分钟")

    summary = {
        'run_time': datetime.now().isoformat(),
        'elapsed_minutes': elapsed / 60,
        'data_range': {'start': DATA_START, 'end': DATA_END},
        'is_range': {'start': IS_START, 'end': IS_END},
        'oos_range': {'start': OOS_START, 'end': OOS_END},
        'recommendations': recommendations,
    }

    # 保存核心结果 (去掉 pnls 避免文件过大)
    def _clean(d):
        if isinstance(d, dict):
            return {k: _clean(v) for k, v in d.items() if k != 'pnls'}
        if isinstance(d, list):
            return [_clean(x) for x in d]
        if isinstance(d, (np.integer, np.floating)):
            return float(d)
        return d

    with open(os.path.join(OUTPUT_DIR, 'param_sensitivity_summary.json'), 'w') as f:
        json.dump(_clean(summary), f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*120}")
    print(f"  分析完成！结果保存至 {OUTPUT_DIR}/")
    print(f"{'='*120}")


if __name__ == '__main__':
    main()
