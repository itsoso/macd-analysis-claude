"""
合约做空策略 — 最终版 (消除事后偏差)

核心原则:
  所有交易决策必须基于"当时时间点"的实盘数据, 不可使用未来信息。

事后偏差清单 (已在此版本中消除):
  ✗ hold∞ (永不平仓) → 因为已知ETH持续下跌 → 替换为信号+止损退出
  ✗ bs_close=9999 (禁用底部信号) → 因为已知$2962不是真底 → 替换为合理阈值
  ✗ single_pct=80% (80%保证金) → 因为已知不会强平 → 替换为标准仓位管理
  ✗ 参数在测试集上优化后在同一数据评测 → 替换为walk-forward验证

验证方法:
  方法A: 书本理论策略 — 参数完全来自《背离技术分析》理论, 不做任何调优
  方法B: Walk-Forward — 前15天训练参数, 后15天验证(样本外测试)
  方法C: 参数敏感性 — 展示参数在合理范围内的表现分布(而非只选最佳)
"""

import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_enhanced import (
    analyze_signals_enhanced, get_realtime_indicators,
    DEFAULT_SIG, fetch_all_data
)
from strategy_futures import FuturesEngine, _merge_signals
from strategy_futures_v2 import get_tf_signal, get_trend_info
from strategy_futures_v4 import _calc_top_score, _calc_bottom_score


# =======================================================
# 通用回测执行器 (严格无前瞻)
# =======================================================
def run_strategy(data, signals_all, config, start_dt=None, end_dt=None):
    """
    通用策略执行器 — 严格前向

    所有决策仅依赖:
      1. 当前bar及之前的价格/指标
      2. 预计算的信号(信号本身只使用历史数据, 已经验证)
      3. 固定的策略参数(不依赖未来)
    """
    eng = FuturesEngine(config['name'], max_leverage=config.get('max_lev', 3))
    main_df = data['1h']
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[0]

    # 风控参数
    eng.max_single_margin = eng.initial_total * config.get('single_pct', 0.15)
    eng.max_margin_total = eng.initial_total * config.get('total_pct', 0.30)
    eng.max_lifetime_margin = eng.initial_total * config.get('lifetime_pct', 2.0)

    # 策略参数 (来自理论/前半段训练, 非当前数据优化)
    ts_sell = config.get('ts_sell', 15)        # 卖出信号阈值
    ts_short = config.get('ts_short', 20)      # 开空信号阈值
    bs_close = config.get('bs_close', 35)      # 底部平仓信号阈值
    sell_pct = config.get('sell_pct', 0.80)    # 每次卖出比例
    margin_use = config.get('margin_use', 0.80) # 可用保证金使用比例
    lev = config.get('lev', 3)                 # 杠杆倍数
    trail_start = config.get('trail_start', 0.5)  # 追踪止盈启动(50%盈利)
    trail_keep = config.get('trail_keep', 0.6)    # 追踪保留(回撤40%则平仓)
    sl = config.get('sl', -0.4)               # 硬止损(-40%)
    tp = config.get('tp', 1.5)                 # 硬止盈(+150%)
    max_hold_bars = config.get('max_hold', 336) # 最大持仓时间(14天*24h)
    cooldown_bars = config.get('cooldown', 8)   # 开仓冷却期(小时)
    require_downtrend = config.get('require_downtrend', True)  # 是否要求下降趋势

    max_pnl_r = 0
    cd = 0
    bars_held = 0

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]
        price = main_df['close'].iloc[idx]

        # 时间范围过滤(walk-forward用)
        if start_dt and dt < start_dt:
            continue
        if end_dt and dt > end_dt:
            continue

        eng.check_liquidation(price, dt)
        eng.charge_funding(price, dt)
        if cd > 0:
            cd -= 1

        # ---- 当前bar的信号 (只用历史数据计算) ----
        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)
        ts, ts_parts = _calc_top_score(sig, trend)
        bs = _calc_bottom_score(sig, trend)

        # ---- 决策1: 卖出现货 ----
        # 条件: 顶部信号足够强 + 还有足够的ETH
        if ts >= ts_sell and eng.spot_eth * price > 1000:
            eng.spot_sell(price, dt, sell_pct,
                          f"信号卖出 TS={ts:.0f} {','.join(ts_parts[:2])}")

        # ---- 决策2: 开空仓 ----
        # 条件: 更强的顶部信号 + 无现有空仓 + 冷却期结束
        # 关键: 当底部信号也很强时(BS >= bs_conflict), 说明信号矛盾, 不开仓
        # (书本理论: 顶底信号矛盾时应观望, 不是任何一方的错)
        bs_conflict = config.get('bs_conflict', 25)
        trend_ok = (not require_downtrend) or trend.get('is_downtrend', False)
        signal_clean = bs < bs_conflict  # 底部信号不能太强, 否则信号矛盾
        if (cd == 0 and ts >= ts_short and trend_ok and signal_clean and
                not eng.futures_short):
            margin = eng.available_margin() * margin_use
            max_allowed_lev = min(lev, eng.max_leverage)
            # 根据信号强度微调杠杆(理论依据: 信号越强确信度越高)
            # 但不超过用户设置的最大杠杆
            if ts >= 35:
                actual_lev = max_allowed_lev  # 强信号: 用满配置杠杆
            elif ts >= 25:
                actual_lev = min(max_allowed_lev, 2)  # 中信号: 最多2x
            else:
                actual_lev = min(max_allowed_lev, 2)  # 弱信号: 最多2x

            eng.open_short(price, dt, margin, actual_lev,
                           f"做空 {actual_lev}x TS={ts:.0f} BS={bs:.0f} {','.join(ts_parts[:3])}")
            max_pnl_r = 0
            bars_held = 0
            cd = cooldown_bars

        # ---- 决策3: 管理空仓 ----
        if eng.futures_short:
            bars_held += 1
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin

            # 3a: 硬止盈
            if pnl_r >= tp:
                eng.close_short(price, dt, f"止盈 +{pnl_r*100:.0f}%")
                max_pnl_r = 0
                cd = cooldown_bars
                bars_held = 0

            # 3b: 追踪止盈
            elif pnl_r > max_pnl_r:
                max_pnl_r = pnl_r
            elif max_pnl_r >= trail_start:
                trail_level = max_pnl_r * trail_keep
                if pnl_r < trail_level and eng.futures_short:
                    eng.close_short(price, dt,
                                    f"追踪止盈 max={max_pnl_r*100:.0f}% now={pnl_r*100:.0f}%")
                    max_pnl_r = 0
                    cd = cooldown_bars
                    bars_held = 0

            # 3c: 底部信号平仓 (关键: 尊重实时信号, 不能因为事后知道会继续跌而忽略)
            if eng.futures_short and bs >= bs_close:
                eng.close_short(price, dt, f"底部信号 BS={bs:.0f}")
                max_pnl_r = 0
                cd = cooldown_bars * 3  # 底部信号后较长冷却, 避免信号矛盾区反复交易
                bars_held = 0

            # 3d: 硬止损
            if eng.futures_short and pnl_r < sl:
                eng.close_short(price, dt, f"止损 {pnl_r*100:.0f}%")
                max_pnl_r = 0
                cd = cooldown_bars * 2  # 止损后更长冷却
                bars_held = 0

            # 3e: 超时平仓 (防止无限持仓)
            if eng.futures_short and bars_held >= max_hold_bars:
                eng.close_short(price, dt, f"超时平仓 {bars_held}h")
                max_pnl_r = 0
                cd = cooldown_bars
                bars_held = 0

        if idx % 4 == 0:
            eng.record_history(dt, price)

    # 结束时平仓
    if eng.futures_short:
        eng.close_short(main_df['close'].iloc[-1], main_df.index[-1], "期末平仓")
    if eng.futures_long:
        eng.close_long(main_df['close'].iloc[-1], main_df.index[-1], "期末平仓")
    return eng.get_result(main_df)


# =======================================================
# 方法A: 书本理论策略 (零优化)
# =======================================================
def method_a_book_theory(data, signals_all):
    """
    参数完全来自《背离技术分析》理论 + 标准风险管理, 不做任何调优。

    理论依据:
      - 顶部背离确认(TS≥15): 隔堆背离+面积缩小+DIF回落 → 卖出信号
      - 强确认(TS≥20): 多周期共振 → 开空
      - 底部信号(BS≥35): 底部背离确认 → 平仓
      - 仓位: 15%单笔(标准风险管理), 最大30%总敞口
      - 杠杆: 2-3x(加密期货标准)
      - 止损: -40%(3x杠杆下ETH反弹13%触发, 合理止损距离)
      - 追踪止盈: 50%利润后开始追踪(标准2:1风险收益比)
    """
    print("\n" + "=" * 100)
    print("  方法A: 书本理论策略 (参数来自理论, 零数据优化)")
    print("=" * 100)

    configs = [
        # A1: 标准保守版 — 完全按书本理论
        # 理论: 背离出现在顶部(趋势还没反转时), 不应要求下降趋势
        # 信号矛盾过滤: 底部信号>25时不开空(矛盾→观望)
        {
            'name': 'A1: 书本标准版',
            'single_pct': 0.15, 'total_pct': 0.30,
            'ts_sell': 15, 'ts_short': 20,
            'bs_close': 35, 'bs_conflict': 25,
            'sell_pct': 0.80, 'margin_use': 0.80,
            'lev': 3,
            'trail_start': 0.5, 'trail_keep': 0.6,
            'sl': -0.40, 'tp': 1.5,
            'max_hold': 336,  # 14天
            'cooldown': 8,
            'require_downtrend': False,  # 书本理论: 背离发生在趋势顶部
        },
        # A2: 中等仓位版
        {
            'name': 'A2: 中等仓位版',
            'single_pct': 0.20, 'total_pct': 0.40,
            'ts_sell': 15, 'ts_short': 20,
            'bs_close': 35, 'bs_conflict': 25,
            'sell_pct': 0.85, 'margin_use': 0.80,
            'lev': 3,
            'trail_start': 0.5, 'trail_keep': 0.6,
            'sl': -0.40, 'tp': 1.5,
            'max_hold': 336,
            'cooldown': 8,
            'require_downtrend': False,
        },
        # A3: 低杠杆安全版
        {
            'name': 'A3: 2x低杠杆版',
            'single_pct': 0.20, 'total_pct': 0.40,
            'ts_sell': 15, 'ts_short': 20,
            'bs_close': 35, 'bs_conflict': 25,
            'sell_pct': 0.85, 'margin_use': 0.80,
            'lev': 2,
            'trail_start': 0.4, 'trail_keep': 0.6,
            'sl': -0.50, 'tp': 1.5,
            'max_hold': 336,
            'cooldown': 8,
            'require_downtrend': False,
        },
        # A4: 需要下降趋势确认版 (保守验证)
        {
            'name': 'A4: 趋势确认版',
            'single_pct': 0.15, 'total_pct': 0.30,
            'ts_sell': 15, 'ts_short': 20,
            'bs_close': 35, 'bs_conflict': 25,
            'sell_pct': 0.80, 'margin_use': 0.80,
            'lev': 3,
            'trail_start': 0.5, 'trail_keep': 0.6,
            'sl': -0.40, 'tp': 1.5,
            'max_hold': 336,
            'cooldown': 8,
            'require_downtrend': True,  # 保守: 等确认后再做空
        },
        # A5: 纯现货策略 (不开空, 作为基准)
        {
            'name': 'A5: 纯现货基准',
            'single_pct': 0.01, 'total_pct': 0.02,
            'ts_sell': 15, 'ts_short': 999,  # 永远不会开空
            'bs_close': 35, 'bs_conflict': 25,
            'sell_pct': 0.85, 'margin_use': 0.80,
            'lev': 1,
            'trail_start': 0.5, 'trail_keep': 0.6,
            'sl': -0.40, 'tp': 1.5,
            'max_hold': 336,
            'cooldown': 8,
            'require_downtrend': False,
        },
    ]

    results = []
    for cfg in configs:
        r = run_strategy(data, signals_all, cfg)
        results.append(r)
        f = r.get('fees', {})
        print(f"  {r['name']:<22} α: {r['alpha']:+.2f}% | 收益: {r['strategy_return']:+.2f}% | "
              f"回撤: {r['max_drawdown']:.2f}% | 交易: {r['total_trades']}笔 | "
              f"费: ${f.get('total_costs', 0):,.0f}")
    return results, configs


# =======================================================
# 方法B: Walk-Forward 验证
# =======================================================
def method_b_walk_forward(data, signals_all):
    """
    前15天训练(寻找最优参数), 后15天验证(样本外测试)。

    训练阶段: 在前15天数据上测试多组参数, 找到最佳组合
    测试阶段: 用训练阶段确定的参数, 在后15天"盲测"
    """
    print("\n" + "=" * 100)
    print("  方法B: Walk-Forward 验证 (前15天训练 → 后15天盲测)")
    print("=" * 100)

    main_df = data['1h']
    total_bars = len(main_df)
    mid_idx = total_bars // 2
    split_dt = main_df.index[mid_idx]
    end_dt = main_df.index[-1]

    print(f"  数据范围: {main_df.index[0]} ~ {end_dt}")
    print(f"  训练集: {main_df.index[0]} ~ {split_dt} ({mid_idx}根K线)")
    print(f"  测试集: {split_dt} ~ {end_dt} ({total_bars - mid_idx}根K线)")

    # ---- 训练阶段: 在前半段搜索参数 ----
    print(f"\n  --- 训练阶段 ---")
    train_params = [
        {'ts_sell': 12, 'ts_short': 18, 'bs_close': 30, 'sl': -0.35, 'trail_start': 0.4},
        {'ts_sell': 15, 'ts_short': 20, 'bs_close': 35, 'sl': -0.40, 'trail_start': 0.5},
        {'ts_sell': 18, 'ts_short': 22, 'bs_close': 40, 'sl': -0.45, 'trail_start': 0.6},
        {'ts_sell': 15, 'ts_short': 25, 'bs_close': 30, 'sl': -0.40, 'trail_start': 0.5},
        {'ts_sell': 12, 'ts_short': 20, 'bs_close': 40, 'sl': -0.50, 'trail_start': 0.4},
        {'ts_sell': 15, 'ts_short': 20, 'bs_close': 45, 'sl': -0.35, 'trail_start': 0.7},
    ]

    base_cfg = {
        'single_pct': 0.15, 'total_pct': 0.30,
        'sell_pct': 0.80, 'margin_use': 0.80,
        'lev': 3, 'trail_keep': 0.6,
        'tp': 1.5, 'max_hold': 336, 'cooldown': 8,
        'require_downtrend': False, 'bs_conflict': 25,
    }

    train_results = []
    for i, params in enumerate(train_params):
        cfg = {**base_cfg, **params, 'name': f'训练#{i+1}'}
        r = run_strategy(data, signals_all, cfg, end_dt=split_dt)
        train_results.append((r, cfg))
        f = r.get('fees', {})
        print(f"    训练#{i+1}: α={r['alpha']:+.2f}% 收益={r['strategy_return']:+.2f}% "
              f"回撤={r['max_drawdown']:.2f}% 交易={r['total_trades']}笔")

    # 选出训练集最佳
    best_train = max(train_results, key=lambda x: x[0]['alpha'])
    best_train_r, best_train_cfg = best_train
    print(f"\n  训练最佳: {best_train_cfg.get('name')} α={best_train_r['alpha']:+.2f}%")
    print(f"  训练最佳参数: ts_sell={best_train_cfg['ts_sell']} ts_short={best_train_cfg['ts_short']} "
          f"bs_close={best_train_cfg['bs_close']} sl={best_train_cfg['sl']} trail={best_train_cfg['trail_start']}")

    # ---- 测试阶段: 用训练参数在后半段盲测 ----
    print(f"\n  --- 测试阶段 (样本外) ---")
    test_cfg = {**best_train_cfg, 'name': 'WF: 样本外测试'}
    test_r = run_strategy(data, signals_all, test_cfg, start_dt=split_dt)
    f = test_r.get('fees', {})
    print(f"  测试结果: α={test_r['alpha']:+.2f}% 收益={test_r['strategy_return']:+.2f}% "
          f"回撤={test_r['max_drawdown']:.2f}% 交易={test_r['total_trades']}笔 费用=${f.get('total_costs', 0):,.0f}")

    # 全量运行(作为参考)
    full_cfg = {**best_train_cfg, 'name': 'WF: 全量参考'}
    full_r = run_strategy(data, signals_all, full_cfg)
    f2 = full_r.get('fees', {})
    print(f"  全量参考: α={full_r['alpha']:+.2f}% 收益={full_r['strategy_return']:+.2f}% "
          f"回撤={full_r['max_drawdown']:.2f}% 交易={full_r['total_trades']}笔 费用=${f2.get('total_costs', 0):,.0f}")

    return {
        'train_best': best_train_r,
        'train_cfg': {k: v for k, v in best_train_cfg.items() if k != 'name'},
        'test_result': test_r,
        'full_result': full_r,
        'split_dt': str(split_dt),
    }


# =======================================================
# 方法C: 参数敏感性分析
# =======================================================
def method_c_sensitivity(data, signals_all):
    """
    展示参数在合理范围内的表现分布, 评估策略鲁棒性。
    不选"最佳", 而是展示"中位数"和"最差"表现。
    """
    print("\n" + "=" * 100)
    print("  方法C: 参数敏感性分析 (展示分布, 而非最佳)")
    print("=" * 100)

    base_cfg = {
        'single_pct': 0.15, 'total_pct': 0.30,
        'ts_sell': 15, 'ts_short': 20,
        'bs_close': 35, 'bs_conflict': 25,
        'sell_pct': 0.80, 'margin_use': 0.80,
        'lev': 3,
        'trail_start': 0.5, 'trail_keep': 0.6,
        'sl': -0.40, 'tp': 1.5,
        'max_hold': 336, 'cooldown': 8,
        'require_downtrend': False,
    }

    sensitivity_results = {}

    # 1. 保证金比例敏感性 (5%~30%, 标准交易范围)
    print(f"\n  1. 保证金比例敏感性 (5%~30%)")
    param_vals = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    alphas = []
    for val in param_vals:
        cfg = {**base_cfg, 'name': f'margin={val*100:.0f}%',
               'single_pct': val, 'total_pct': val * 2}
        r = run_strategy(data, signals_all, cfg)
        alphas.append(r['alpha'])
        print(f"    single={val*100:.0f}%: α={r['alpha']:+.2f}% 收益={r['strategy_return']:+.2f}% "
              f"回撤={r['max_drawdown']:.2f}%")
    sensitivity_results['margin'] = {
        'param': 'single_pct', 'values': param_vals,
        'alphas': alphas,
        'median': float(np.median(alphas)),
        'min': min(alphas), 'max': max(alphas),
    }

    # 2. 止损阈值敏感性
    print(f"\n  2. 止损阈值敏感性 (-25%~-60%)")
    param_vals = [-0.25, -0.30, -0.35, -0.40, -0.50, -0.60]
    alphas = []
    for val in param_vals:
        cfg = {**base_cfg, 'name': f'sl={val*100:.0f}%', 'sl': val}
        r = run_strategy(data, signals_all, cfg)
        alphas.append(r['alpha'])
        print(f"    sl={val*100:.0f}%: α={r['alpha']:+.2f}% 收益={r['strategy_return']:+.2f}% "
              f"回撤={r['max_drawdown']:.2f}%")
    sensitivity_results['stop_loss'] = {
        'param': 'sl', 'values': [v * 100 for v in param_vals],
        'alphas': alphas,
        'median': float(np.median(alphas)),
        'min': min(alphas), 'max': max(alphas),
    }

    # 3. 卖出信号阈值敏感性
    print(f"\n  3. 卖出信号阈值敏感性 (TS=10~25)")
    param_vals = [10, 12, 15, 18, 20, 25]
    alphas = []
    for val in param_vals:
        cfg = {**base_cfg, 'name': f'ts_sell={val}', 'ts_sell': val}
        r = run_strategy(data, signals_all, cfg)
        alphas.append(r['alpha'])
        print(f"    ts_sell={val}: α={r['alpha']:+.2f}% 收益={r['strategy_return']:+.2f}% "
              f"回撤={r['max_drawdown']:.2f}%")
    sensitivity_results['ts_sell'] = {
        'param': 'ts_sell', 'values': param_vals,
        'alphas': alphas,
        'median': float(np.median(alphas)),
        'min': min(alphas), 'max': max(alphas),
    }

    # 4. 底部平仓信号阈值敏感性
    print(f"\n  4. 底部信号阈值敏感性 (BS=25~55)")
    param_vals = [25, 30, 35, 40, 45, 55]
    alphas = []
    for val in param_vals:
        cfg = {**base_cfg, 'name': f'bs_close={val}', 'bs_close': val}
        r = run_strategy(data, signals_all, cfg)
        alphas.append(r['alpha'])
        print(f"    bs_close={val}: α={r['alpha']:+.2f}% 收益={r['strategy_return']:+.2f}% "
              f"回撤={r['max_drawdown']:.2f}%")
    sensitivity_results['bs_close'] = {
        'param': 'bs_close', 'values': param_vals,
        'alphas': alphas,
        'median': float(np.median(alphas)),
        'min': min(alphas), 'max': max(alphas),
    }

    # 5. 追踪止盈启动点敏感性
    print(f"\n  5. 追踪止盈启动点敏感性 (30%~100%)")
    param_vals = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    alphas = []
    for val in param_vals:
        cfg = {**base_cfg, 'name': f'trail={val*100:.0f}%', 'trail_start': val}
        r = run_strategy(data, signals_all, cfg)
        alphas.append(r['alpha'])
        print(f"    trail_start={val*100:.0f}%: α={r['alpha']:+.2f}% 收益={r['strategy_return']:+.2f}% "
              f"回撤={r['max_drawdown']:.2f}%")
    sensitivity_results['trail_start'] = {
        'param': 'trail_start', 'values': [v * 100 for v in param_vals],
        'alphas': alphas,
        'median': float(np.median(alphas)),
        'min': min(alphas), 'max': max(alphas),
    }

    # 6. 杠杆倍数敏感性
    print(f"\n  6. 杠杆倍数敏感性 (1x~5x)")
    param_vals = [1, 2, 3, 4, 5]
    alphas = []
    for val in param_vals:
        cfg = {**base_cfg, 'name': f'{val}x杠杆', 'lev': val, 'max_lev': val}
        r = run_strategy(data, signals_all, cfg)
        alphas.append(r['alpha'])
        print(f"    {val}x: α={r['alpha']:+.2f}% 收益={r['strategy_return']:+.2f}% "
              f"回撤={r['max_drawdown']:.2f}%")
    sensitivity_results['leverage'] = {
        'param': 'leverage', 'values': param_vals,
        'alphas': alphas,
        'median': float(np.median(alphas)),
        'min': min(alphas), 'max': max(alphas),
    }

    # 汇总
    print(f"\n  --- 敏感性汇总 ---")
    for key, s in sensitivity_results.items():
        print(f"  {key:12s}: 中位α={s['median']:+.2f}% 范围=[{s['min']:+.2f}%, {s['max']:+.2f}%] "
              f"(波动{s['max']-s['min']:.2f}%)")

    return sensitivity_results


# =======================================================
# 主入口
# =======================================================
def run_all():
    data = fetch_all_data()

    print("\n计算各周期增强信号...")
    signal_windows = {'1h': 168, '2h': 150, '4h': 120, '6h': 100, '8h': 90}
    signals_all = {}
    for tf, df in data.items():
        w = signal_windows.get(tf, 120)
        if len(df) > w:
            signals_all[tf] = analyze_signals_enhanced(df, w)
            print(f"  {tf}: {len(signals_all[tf])} 个信号点")

    # ============================================
    # 方法A: 书本理论策略
    # ============================================
    results_a, configs_a = method_a_book_theory(data, signals_all)

    # ============================================
    # 方法B: Walk-Forward
    # ============================================
    wf_results = method_b_walk_forward(data, signals_all)

    # ============================================
    # 方法C: 参数敏感性
    # ============================================
    sensitivity = method_c_sensitivity(data, signals_all)

    # ============================================
    # 最终汇总
    # ============================================
    print(f"\n\n{'='*100}")
    print("  最终结论 (无事后偏差)")
    print(f"{'='*100}")

    best_a = max(results_a, key=lambda x: x['alpha'])
    print(f"\n  方法A 书本理论(最佳): {best_a['name']}")
    print(f"    α={best_a['alpha']:+.2f}% | 收益={best_a['strategy_return']:+.2f}% | "
          f"回撤={best_a['max_drawdown']:.2f}% | 交易={best_a['total_trades']}笔")

    test_r = wf_results['test_result']
    print(f"\n  方法B Walk-Forward(样本外):")
    print(f"    α={test_r['alpha']:+.2f}% | 收益={test_r['strategy_return']:+.2f}% | "
          f"回撤={test_r['max_drawdown']:.2f}% | 交易={test_r['total_trades']}笔")

    # 参数鲁棒性
    median_alphas = [s['median'] for s in sensitivity.values()]
    overall_median = np.median(median_alphas)
    print(f"\n  方法C 参数鲁棒性:")
    print(f"    各维度中位α的中位数: {overall_median:+.2f}%")
    for key, s in sensitivity.items():
        print(f"    {key}: 中位{s['median']:+.2f}% [{s['min']:+.2f}% ~ {s['max']:+.2f}%]")

    print(f"\n  对比(不可信的旧结果): Phase 8 hold∞ α=+118.78% ← 严重事后偏差")
    print(f"  真实可预期的α范围: [{min(s['min'] for s in sensitivity.values()):+.2f}% ~ "
          f"{max(s['max'] for s in sensitivity.values()):+.2f}%]")

    # 保存结果
    output = {
        'phase': 'final_v2',
        'description': '消除事后偏差的真实回测',
        'methodology': {
            'method_a': '书本理论参数(零数据优化)',
            'method_b': 'Walk-Forward(前15天训练→后15天盲测)',
            'method_c': '参数敏感性(展示分布而非最佳)',
        },
        'bias_eliminated': [
            'hold∞ (永不平仓) → 改为信号+止损+超时退出',
            'bs_close=9999 (禁用底部信号) → 改为BS≥35理论阈值',
            'single_pct=80% (极端仓位) → 改为15-30%标准仓位',
            '在测试集上优化参数 → Walk-Forward样本外验证',
        ],
        'futures_results': [{
            'name': r['name'],
            'summary': {k: v for k, v in r.items() if k not in ('trades', 'history')},
            'trades': r['trades'],
            'history': r['history'],
        } for r in results_a],
        'walk_forward': {
            'split_dt': wf_results['split_dt'],
            'train_alpha': wf_results['train_best']['alpha'],
            'train_params': wf_results['train_cfg'],
            'test_result': {
                'name': wf_results['test_result']['name'],
                'summary': {k: v for k, v in wf_results['test_result'].items()
                            if k not in ('trades', 'history')},
                'trades': wf_results['test_result']['trades'],
                'history': wf_results['test_result']['history'],
            },
            'full_result': {
                'name': wf_results['full_result']['name'],
                'summary': {k: v for k, v in wf_results['full_result'].items()
                            if k not in ('trades', 'history')},
                'trades': wf_results['full_result']['trades'],
                'history': wf_results['full_result']['history'],
            },
        },
        'sensitivity': sensitivity,
    }

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'strategy_futures_final_result.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果已保存: {path}")
    return output


if __name__ == '__main__':
    run_all()
