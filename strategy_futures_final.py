"""
合约做空策略 Phase 6+7 — 最终优化

目标: 在B1(α=+28.15%)基础上, 通过风控参数优化+策略微调找到最大收益

Phase 6: 风控参数网格搜索
  - max_single_margin: 5%→20% (步长2.5%)
  - max_margin_total: 10%→30%
  - leverage: 2x vs 3x
  - available_margin比例: 0.8-1.0
  - 跨参数组合搜索

Phase 7: 策略结构微调
  - 卖出比例: 0.85-0.98
  - 是否追卖残余
  - 是否开第二笔空仓
  - 是否底部做多
  - 早鸟减仓
"""

import pandas as pd
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_enhanced import (
    analyze_signals_enhanced, get_realtime_indicators,
    DEFAULT_SIG, fetch_all_data
)
from strategy_futures import FuturesEngine, _merge_signals
from strategy_futures_v2 import get_tf_signal, get_trend_info
from strategy_futures_v4 import _calc_top_score, _calc_bottom_score


def run_strategy(data, signals_all, config):
    """通用策略执行器"""
    eng = FuturesEngine(config['name'], max_leverage=config.get('max_lev', 3))
    main_df = data['1h']
    eng.spot_eth = eng.initial_eth_value / main_df['close'].iloc[0]

    # 覆盖风控参数
    if 'single_pct' in config:
        eng.max_single_margin = eng.initial_total * config['single_pct']
    if 'total_pct' in config:
        eng.max_margin_total = eng.initial_total * config['total_pct']
    if 'lifetime_pct' in config:
        eng.max_lifetime_margin = eng.initial_total * config['lifetime_pct']

    # 覆盖可用保证金计算中的USDT比例限制 (默认0.5 → 可调)
    usdt_ratio = config.get('usdt_ratio', 0.5)
    if usdt_ratio != 0.5:
        original_available = eng.available_margin
        def custom_available():
            avail_usdt = eng.available_usdt()
            remaining_cap = eng.max_margin_total - eng.frozen_margin
            lifetime_remain = eng.max_lifetime_margin - eng.lifetime_margin_used
            return max(0, min(avail_usdt * usdt_ratio, remaining_cap,
                              eng.max_single_margin, lifetime_remain))
        eng.available_margin = custom_available

    max_pnl_r = 0
    short_count = 0
    long_count = 0
    cd_short = 0
    cd_long = 0
    early_sold = False
    sell_pct = config.get('sell_pct', 0.92)
    margin_use = config.get('margin_use', 0.95)
    lev_default = config.get('lev', 3)
    ts_sell = config.get('ts_sell', 15)
    ts_short = config.get('ts_short', 20)
    max_shorts = config.get('max_shorts', 1)
    do_long = config.get('do_long', False)
    do_early = config.get('do_early', False)
    do_chase_sell = config.get('do_chase_sell', True)
    short_cd = config.get('short_cd', 48)
    require_downtrend = config.get('require_downtrend', False)
    trail_start = config.get('trail_start', 1.0)
    trail_keep = config.get('trail_keep', 0.55)
    sl = config.get('sl', -0.6)
    min_short_bars = config.get('min_short_bars', 0)
    bs_close = config.get('bs_close', 40)
    bars_since_short = 0

    for idx in range(20, len(main_df)):
        dt = main_df.index[idx]; price = main_df['close'].iloc[idx]
        eng.check_liquidation(price, dt); eng.charge_funding(price, dt)
        if cd_short > 0: cd_short -= 1
        if cd_long > 0: cd_long -= 1

        trend = get_trend_info(data, dt, price)
        sig = _merge_signals(signals_all, dt)
        ts, parts = _calc_top_score(sig, trend)
        bs = _calc_bottom_score(sig, trend)

        # 早鸟减仓
        if do_early and not early_sold:
            ind = get_realtime_indicators(main_df, idx)
            k = ind.get('K', 50); rsi = ind.get('RSI6', 50)
            if ts >= 5 or (k > 75 and rsi > 70):
                if eng.spot_eth * price > 20000:
                    eng.spot_sell(price, dt, 0.35, f"早鸟 K={k:.0f}")
                    early_sold = True

        # 主信号: 清仓现货
        if ts >= ts_sell and eng.spot_eth * price > 500:
            eng.spot_sell(price, dt, sell_pct, f"清仓 TS={ts:.0f}")

        # 追卖残余
        if do_chase_sell and short_count > 0 and eng.spot_eth * price > 500:
            if ts >= 8:
                eng.spot_sell(price, dt, 0.8, "追卖")

        # 开空
        trend_ok = (not require_downtrend) or trend['is_downtrend']
        if (cd_short == 0 and ts >= ts_short and trend_ok and
                not eng.futures_short and short_count < max_shorts):
            margin = eng.available_margin() * margin_use
            lev = min(lev_default, eng.max_leverage)
            if ts >= 35: lev = min(3, eng.max_leverage)
            eng.open_short(price, dt, margin, lev,
                f"做空#{short_count+1} {lev}x TS={ts:.0f}")
            max_pnl_r = 0; short_count += 1; cd_short = short_cd

        # 做多(底部)
        if (do_long and cd_long == 0 and bs >= 35 and
                not eng.futures_long and not eng.futures_short and long_count < 2):
            margin = eng.available_margin() * 0.4
            eng.open_long(price, dt, margin, 2, f"做多 BS={bs:.0f}")
            max_pnl_r = 0; long_count += 1; cd_long = 24

        # 平空
        if eng.futures_short:
            bars_since_short += 1
            pnl_r = eng.futures_short.calc_pnl(price) / eng.futures_short.margin
            max_pnl_r = max(max_pnl_r, pnl_r)
            can_close = bars_since_short >= min_short_bars
            if can_close:
                if max_pnl_r >= trail_start:
                    trail = max_pnl_r * trail_keep
                    if pnl_r < trail:
                        eng.close_short(price, dt, f"追踪 max={max_pnl_r*100:.0f}%")
                        max_pnl_r = 0; cd_short = 24; bars_since_short = 0
                if eng.futures_short and bs >= bs_close:
                    eng.close_short(price, dt, f"强底 BS={bs:.0f}")
                    max_pnl_r = 0; cd_short = 12; bars_since_short = 0
            if eng.futures_short and pnl_r < sl:
                eng.close_short(price, dt, f"止损{pnl_r*100:.0f}%")
                max_pnl_r = 0; cd_short = 36; bars_since_short = 0

        # 平多
        if eng.futures_long:
            pnl_r = eng.futures_long.calc_pnl(price) / eng.futures_long.margin
            max_pnl_r = max(max_pnl_r, pnl_r)
            if max_pnl_r >= 0.25:
                if pnl_r < max_pnl_r * 0.6:
                    eng.close_long(price, dt, "止盈多")
                    max_pnl_r = 0; cd_long = 8
            if ts >= 20:
                eng.close_long(price, dt, "转空")
                max_pnl_r = 0; cd_long = 4
            if pnl_r < -0.4:
                eng.close_long(price, dt, "止损多")
                max_pnl_r = 0; cd_long = 12

        if idx % 4 == 0: eng.record_history(dt, price)

    if eng.futures_short: eng.close_short(main_df['close'].iloc[-1], main_df.index[-1], "结束")
    if eng.futures_long: eng.close_long(main_df['close'].iloc[-1], main_df.index[-1], "结束")
    return eng.get_result(main_df)


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

    # ============================================================
    # Phase 6: 风控参数网格搜索
    # ============================================================
    print(f"\n{'='*140}")
    print(f"  Phase 6: 风控参数网格搜索 (保证金比例×杠杆×卖出比例)")
    print(f"{'='*140}")

    phase6_configs = []

    # 6A: 不同single_max (核心参数, 扩展到50%)
    for sp in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        phase6_configs.append({
            'name': f'6A: single={sp*100:.0f}%',
            'single_pct': sp,
            'total_pct': max(sp * 2.5, 0.15),
            'sell_pct': 0.92,
            'margin_use': 1.0,
            'lev': 3,
        })

    # 6B: 不同杠杆 × 高保证金
    for lev in [2, 3, 4, 5]:
        phase6_configs.append({
            'name': f'6B: {lev}x lev s40%',
            'single_pct': 0.40,
            'total_pct': 1.0,
            'max_lev': lev,
            'lev': lev,
            'sell_pct': 0.92,
            'margin_use': 1.0,
        })

    # 6C: 突破USDT 50%限制 (usdt_ratio 0.5→0.9)
    for ur in [0.5, 0.6, 0.7, 0.8, 0.9]:
        phase6_configs.append({
            'name': f'6C: usdt_r={ur} s50%',
            'single_pct': 0.50,
            'total_pct': 1.0,
            'usdt_ratio': ur,
            'sell_pct': 0.92,
            'margin_use': 1.0,
            'lev': 3,
        })

    # 6D: 极致组合 - 高保证金+高USDT比例+高杠杆
    for sp, ur, lev in [
        (0.40, 0.8, 3), (0.50, 0.8, 3), (0.50, 0.9, 3),
        (0.40, 0.8, 4), (0.50, 0.8, 4), (0.50, 0.9, 4),
        (0.40, 0.8, 5), (0.50, 0.8, 5), (0.50, 0.9, 5),
    ]:
        phase6_configs.append({
            'name': f'6D: s{sp*100:.0f}% ur{ur} {lev}x',
            'single_pct': sp,
            'total_pct': max(sp * 2.5, 1.0),
            'max_lev': lev,
            'lev': lev,
            'usdt_ratio': ur,
            'sell_pct': 0.95,
            'margin_use': 1.0,
        })

    results_p6 = []
    for cfg in phase6_configs:
        r = run_strategy(data, signals_all, cfg)
        results_p6.append(r)
        f = r.get('fees', {})
        print(f"  {r['name']:<26} α: {r['alpha']:+.2f}% | 收益: {r['strategy_return']:+.2f}% | "
              f"回撤: {r['max_drawdown']:.2f}% | 交易: {r['total_trades']}笔 | "
              f"费: ${f.get('total_costs', 0):,.0f} ({f.get('fee_drag_pct', 0):.2f}%)")

    # 找Phase6最佳参数
    best_p6 = max(results_p6, key=lambda x: x['alpha'])
    print(f"\n  Phase 6 最佳: {best_p6['name']} α={best_p6['alpha']:+.2f}%")

    # ============================================================
    # Phase 7: 策略结构微调 (基于Phase 6最佳参数)
    # ============================================================
    print(f"\n{'='*140}")
    print(f"  Phase 7: 策略结构微调 (基于Phase6最佳风控参数)")
    print(f"{'='*140}")

    # 从Phase6最佳中提取参数
    best_cfg = None
    for cfg in phase6_configs:
        if cfg['name'] == best_p6['name']:
            best_cfg = cfg.copy()
            break
    if not best_cfg:
        best_cfg = {'single_pct': 0.15, 'total_pct': 0.30, 'sell_pct': 0.92,
                     'margin_use': 0.95, 'lev': 3}

    phase7_configs = [
        # 7A: 基线(Phase6最佳参数原封不动)
        {**best_cfg, 'name': '7A: P6最佳基线'},

        # 7B: +早鸟减仓
        {**best_cfg, 'name': '7B: +早鸟', 'do_early': True},

        # 7C: +双空仓
        {**best_cfg, 'name': '7C: +双空仓', 'max_shorts': 2, 'short_cd': 48},

        # 7D: +底部做多
        {**best_cfg, 'name': '7D: +做多', 'do_long': True},

        # 7E: +早鸟+双空
        {**best_cfg, 'name': '7E: 早鸟+双空', 'do_early': True, 'max_shorts': 2, 'short_cd': 48},

        # 7F: 全量融合
        {**best_cfg, 'name': '7F: 全量融合', 'do_early': True, 'max_shorts': 2,
         'short_cd': 48, 'do_long': True},

        # 7G: 不追卖(减少费用)
        {**best_cfg, 'name': '7G: 不追卖', 'do_chase_sell': False},

        # 7H: 更宽止损
        {**best_cfg, 'name': '7H: 宽止损-80%', 'sl': -0.8, 'trail_start': 1.2},

        # 7I: 更窄追踪
        {**best_cfg, 'name': '7I: 紧追踪', 'trail_start': 0.6, 'trail_keep': 0.7},

        # 7J: 更高卖出门槛
        {**best_cfg, 'name': '7J: 高卖门槛20', 'ts_sell': 20},

        # 7K: 更低卖出门槛
        {**best_cfg, 'name': '7K: 低卖门槛10', 'ts_sell': 10},

        # 7L: 95%卖出+不追卖
        {**best_cfg, 'name': '7L: 95%卖+不追', 'sell_pct': 0.95, 'do_chase_sell': False},

        # 7M: 满仓100%margin
        {**best_cfg, 'name': '7M: 满仓margin', 'margin_use': 1.0},

        # 7N: 早鸟+不追卖+满仓
        {**best_cfg, 'name': '7N: 早鸟+满仓', 'do_early': True,
         'margin_use': 1.0, 'do_chase_sell': False},

        # 7O: 不需要下降趋势确认
        {**best_cfg, 'name': '7O: 无趋势限制', 'require_downtrend': False, 'margin_use': 1.0},

        # 7P: 98%全清+满仓margin+不追卖
        {**best_cfg, 'name': '7P: 极致卖出', 'sell_pct': 0.98,
         'margin_use': 1.0, 'do_chase_sell': False},

        # 7Q: 低门槛ts=10+双空仓+满仓
        {**best_cfg, 'name': '7Q: 低门槛双空',
         'ts_sell': 10, 'ts_short': 15, 'max_shorts': 2,
         'short_cd': 36, 'margin_use': 1.0},
    ]

    results_p7 = []
    for cfg in phase7_configs:
        r = run_strategy(data, signals_all, cfg)
        results_p7.append(r)
        f = r.get('fees', {})
        print(f"  {r['name']:<26} α: {r['alpha']:+.2f}% | 收益: {r['strategy_return']:+.2f}% | "
              f"回撤: {r['max_drawdown']:.2f}% | 交易: {r['total_trades']}笔 | "
              f"费: ${f.get('total_costs', 0):,.0f} ({f.get('fee_drag_pct', 0):.2f}%)")

    # ============================================================
    # Phase 8: 极致参数搜索 (基于Phase 6+7最佳)
    # ============================================================
    best_so_far = max(results_p6 + results_p7, key=lambda x: x['alpha'])
    best8_cfg = None
    for cfg in phase6_configs + phase7_configs:
        if cfg['name'] == best_so_far['name']:
            best8_cfg = cfg.copy()
            break
    if not best8_cfg:
        best8_cfg = best_cfg.copy()

    print(f"\n{'='*140}")
    print(f"  Phase 8: 极致参数微调 (基于当前冠军 {best_so_far['name']} α={best_so_far['alpha']:+.2f}%)")
    print(f"{'='*140}")

    phase8_configs = []

    # 8A: 在冠军附近做精细网格搜索
    base_sp = best8_cfg.get('single_pct', 0.25)
    base_ur = best8_cfg.get('usdt_ratio', 0.5)
    base_lev = best8_cfg.get('lev', 3)

    # 精细single_pct搜索 (±10%范围)
    for sp_delta in [-0.05, -0.02, 0, 0.02, 0.05, 0.10]:
        sp = min(max(base_sp + sp_delta, 0.05), 0.60)
        for ur in [base_ur, min(base_ur + 0.1, 0.95), min(base_ur + 0.2, 0.95)]:
            cfg_name = f'8A: s{sp*100:.0f}% ur{ur:.1f}'
            phase8_configs.append({
                **best8_cfg,
                'name': cfg_name,
                'single_pct': sp,
                'total_pct': max(sp * 2.5, 1.0),
                'usdt_ratio': ur,
                'margin_use': 1.0,
            })

    # 8B: 杠杆精细搜索 (冠军参数基础上换杠杆)
    for lev in [2, 3, 4, 5]:
        if lev != base_lev:
            phase8_configs.append({
                **best8_cfg,
                'name': f'8B: {lev}x (冠军参数)',
                'max_lev': lev,
                'lev': lev,
                'margin_use': 1.0,
            })

    # 8C: 卖出+保证金+USDT比例 三维联合搜索 (关键组合)
    for sell in [0.90, 0.95, 0.98]:
        for sp in [0.35, 0.45, 0.55]:
            for ur in [0.7, 0.85, 0.95]:
                phase8_configs.append({
                    'name': f'8C: sl{sell*100:.0f} s{sp*100:.0f} u{ur*100:.0f}',
                    'single_pct': sp,
                    'total_pct': max(sp * 2.5, 1.0),
                    'usdt_ratio': ur,
                    'sell_pct': sell,
                    'margin_use': 1.0,
                    'lev': 3,
                    'max_lev': 5,
                })

    # 8E: 持仓时间+底部关闭阈值 (防止过早平仓)
    for min_bars in [0, 48, 96, 168, 336, 9999]:  # 0h, 2d, 4d, 7d, 14d, 不关
        for bsc in [40, 50, 60, 9999]:  # 底部信号阈值, 9999=不根据底部信号关闭
            # 跳过默认组合
            if min_bars == 0 and bsc == 40:
                continue
            phase8_configs.append({
                'name': f'8E: bars{min_bars} bs{bsc}',
                'single_pct': 0.50,
                'total_pct': 1.25,
                'usdt_ratio': 0.65,
                'sell_pct': 0.92,
                'margin_use': 1.0,
                'lev': 3,
                'ts_sell': 10,
                'min_short_bars': min_bars,
                'bs_close': bsc,
                'trail_start': 2.0,  # 更宽追踪
                'trail_keep': 0.5,
            })

    # 8F: 最佳持仓组合 × 不同保证金
    for sp in [0.40, 0.50, 0.60, 0.70, 0.80]:
        phase8_configs.append({
            'name': f'8F: s{sp*100:.0f}% hold∞',
            'single_pct': sp,
            'total_pct': max(sp * 2.5, 1.0),
            'usdt_ratio': min(sp + 0.15, 0.95),
            'sell_pct': 0.92,
            'margin_use': 1.0,
            'lev': 3,
            'ts_sell': 10,
            'min_short_bars': 9999,
            'bs_close': 9999,
            'trail_start': 9999.0,  # 永不追踪止盈 → 持有到结束
        })

    # 去重
    seen_names = set()
    unique_p8 = []
    for cfg in phase8_configs:
        if cfg['name'] not in seen_names:
            seen_names.add(cfg['name'])
            unique_p8.append(cfg)
    phase8_configs = unique_p8

    results_p8 = []
    for cfg in phase8_configs:
        r = run_strategy(data, signals_all, cfg)
        results_p8.append(r)
        f = r.get('fees', {})
        print(f"  {r['name']:<30} α: {r['alpha']:+.2f}% | 收益: {r['strategy_return']:+.2f}% | "
              f"回撤: {r['max_drawdown']:.2f}% | 交易: {r['total_trades']}笔 | "
              f"费: ${f.get('total_costs', 0):,.0f} ({f.get('fee_drag_pct', 0):.2f}%)")

    best_p8 = max(results_p8, key=lambda x: x['alpha'])
    print(f"\n  Phase 8 最佳: {best_p8['name']} α={best_p8['alpha']:+.2f}%")

    # ============================================================
    # 汇总排名
    # ============================================================
    all_results = results_p6 + results_p7 + results_p8
    ranked = sorted(all_results, key=lambda x: x['alpha'], reverse=True)

    print(f"\n\n{'='*140}")
    print("             Phase 6+7+8 终极排名 (所有优化策略)")
    print(f"{'='*140}")
    fmt = "{:>3} {:<30} {:>9} {:>9} {:>8} {:>6} {:>10} {:>9} {:>12}"
    print(fmt.format("#", "策略", "收益", "超额α", "回撤", "笔数", "总费用", "费率%", "最终资产"))
    print("-" * 140)
    for rank, r in enumerate(ranked, 1):
        star = " ★" if rank == 1 else ("  " if rank > 3 else "")
        f = r.get('fees', {})
        print(fmt.format(
            rank, r['name'] + star,
            f"{r['strategy_return']:+.1f}%",
            f"{r['alpha']:+.1f}%",
            f"{r['max_drawdown']:.1f}%",
            str(r['total_trades']),
            f"${f.get('total_costs', 0):,.0f}",
            f"{f.get('fee_drag_pct', 0):.2f}%",
            f"${r['final_total']:,.0f}",
        ))
        if rank >= 20: break
    print("=" * 140)

    # 对比历史最佳
    print(f"\n  历史最佳对比:")
    print(f"  Phase 5 B1(满仓版):  α=+28.15%")
    print(f"  Phase 6+7 冠军:      {ranked[0]['name']} α={ranked[0]['alpha']:+.2f}%")
    improvement = ranked[0]['alpha'] - 28.15
    if improvement > 0:
        print(f"  提升: +{improvement:.2f}% !!!")
    print(f"  最终资产: ${ranked[0]['final_total']:,.0f} (初始$200,000)")

    # 保存 Top 15 结果
    top_results = ranked[:15]
    output = {
        'phase': '6+7',
        'futures_results': [{
            'name': r['name'],
            'summary': {k: v for k, v in r.items() if k not in ('trades', 'history')},
            'trades': r['trades'],
            'history': r['history'],
        } for r in top_results],
        'best_strategy': ranked[0]['name'],
        'ranking': [{'rank': i+1, 'name': r['name'], 'alpha': r['alpha'],
                      'return': r['strategy_return'], 'max_dd': r['max_drawdown'],
                      'trades': r['total_trades'], 'fees': r.get('fees', {})}
                     for i, r in enumerate(ranked[:20])],
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'strategy_futures_final_result.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, default=str, indent=2)
    print(f"\n结果已保存: {path}")
    return output


if __name__ == '__main__':
    run_all()
