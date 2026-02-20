#!/usr/bin/env python3
"""P42: Veto recalibration sweep — soft veto midpoint was 25.0 (effectively disabled!)
Correct range is 0.3-2.0 based on sell_opp normalization to [0, ~3.0].
Also tests hard veto (use_soft_veto=False) and ATR/close thresholds."""
import json, sys, time
sys.path.insert(0, '.')
from optimize_six_book import fetch_multi_tf_data, compute_signals_six, run_strategy

SYMBOLS = ['ETHUSDT', 'BTCUSDT', 'SOLUSDT', 'BNBUSDT']
TF = '1h'

BASE_CFG = {
    'fusion_mode': 'c6_veto_4', 'veto_threshold': 25,
    'single_pct': 0.20, 'total_pct': 0.50, 'lifetime_pct': 5.0,
    'sell_threshold': 14, 'buy_threshold': 25,
    'short_threshold': 18, 'long_threshold': 40,
    'close_short_bs': 40, 'close_long_ss': 40,
    'sell_pct': 0.55, 'margin_use': 0.70, 'lev': 5, 'max_lev': 5,
    'short_sl': -0.15, 'long_sl': -0.08,
    'short_tp': 0.30, 'long_tp': 0.30,
    'short_trail': 0.25, 'long_trail': 0.25,
    'trail_pullback': 0.40,
    'short_max_hold': 72, 'long_max_hold': 72,
    'cooldown': 4,
    'use_atr_sl': True, 'atr_sl_mult': 3.5,
    'use_soft_veto': True, 'soft_veto_steepness': 5.0, 'soft_veto_midpoint': 25.0,
    'use_regime_aware': True,
    'initial_usdt': 100000, 'initial_eth_value': 0,
}

VARIANTS = {
    # A. Baseline (soft veto midpoint=25 = effectively no veto)
    'NoVeto': {},
    # B. Hard veto (original logic: >=2 components → score*0.30)
    'HardV': {'use_soft_veto': False},
    # C. Soft veto with CORRECT midpoint values (0.3-2.0 range)
    'SV03': {'soft_veto_midpoint': 0.3, 'soft_veto_steepness': 5.0},
    'SV05': {'soft_veto_midpoint': 0.5, 'soft_veto_steepness': 5.0},
    'SV08': {'soft_veto_midpoint': 0.8, 'soft_veto_steepness': 5.0},
    'SV10': {'soft_veto_midpoint': 1.0, 'soft_veto_steepness': 5.0},
    'SV15': {'soft_veto_midpoint': 1.5, 'soft_veto_steepness': 5.0},
    'SV20': {'soft_veto_midpoint': 2.0, 'soft_veto_steepness': 5.0},
    # D. Steepness sweep at midpoint=1.0 (default)
    'K3M10': {'soft_veto_midpoint': 1.0, 'soft_veto_steepness': 3.0},
    'K8M10': {'soft_veto_midpoint': 1.0, 'soft_veto_steepness': 8.0},
    # E. Veto dampen floor sweep at midpoint=1.0
    'D20M10': {'soft_veto_midpoint': 1.0, 'veto_dampen': 0.20},
    'D40M10': {'soft_veto_midpoint': 1.0, 'veto_dampen': 0.40},
    'D50M10': {'soft_veto_midpoint': 1.0, 'veto_dampen': 0.50},
    # F. ATR SL multiplier (independent of veto)
    'ATR30': {'atr_sl_mult': 3.0},
    'ATR40': {'atr_sl_mult': 4.0},
    'ATR45': {'atr_sl_mult': 4.5},
    # G. Close thresholds
    'CLS35': {'close_short_bs': 35, 'close_long_ss': 35},
    'CLS45': {'close_short_bs': 45, 'close_long_ss': 45},
    'CLS50': {'close_short_bs': 50, 'close_long_ss': 50},
}

def extract_pf(result):
    trades = result.get('trades', [])
    close_pnls = [t.get('pnl', 0) for t in trades
                  if t.get('action', '').startswith('CLOSE') and t.get('pnl', 0) != 0]
    gp = sum(p for p in close_pnls if p > 0)
    gl = abs(sum(p for p in close_pnls if p < 0))
    return gp, gl

def count_trades(result):
    return result.get('total_trades', 0)

PERIODS = [('IS', 730), ('OOS', 1865), ('5yr', 1900)]

t_start = time.time()
results = {}

for period_name, days in PERIODS:
    print(f"\n[{period_name}] {days}天")
    symbol_data = {}; symbol_signals = {}
    for sym in SYMBOLS:
        data = fetch_multi_tf_data([TF], days=days, symbol=sym)
        symbol_data[sym] = data[TF]
        symbol_signals[sym] = compute_signals_six(data[TF], TF, data)

    results[period_name] = {}
    for vname, voverride in VARIANTS.items():
        cfg = {**BASE_CFG, **voverride}
        total_gp = 0.0; total_gl = 0.0; total_trades = 0
        for sym in SYMBOLS:
            r = run_strategy(symbol_data[sym], symbol_signals[sym], cfg, tf=TF, trade_days=days)
            gp, gl = extract_pf(r)
            total_gp += gp; total_gl += gl
            total_trades += count_trades(r)

        agg_pf = round(total_gp / total_gl, 4) if total_gl > 0 else 0
        results[period_name][vname] = {'pf': agg_pf, 'trades': total_trades}
        base_pf = results[period_name]['NoVeto']['pf']
        delta = agg_pf - base_pf
        marker = " ***" if abs(delta) >= 0.01 else ""
        print(f"  {vname:8s}: PF={agg_pf:.4f}  Δ={delta:+.4f}  T={total_trades}{marker}")

# Summary
print("\n" + "="*60)
print("P42: Veto Recalibration — Summary")
print("="*60)

param_groups = {
    'Veto Mode': ['NoVeto', 'HardV'],
    'Soft Veto Midpoint (k=5)': ['NoVeto', 'SV03', 'SV05', 'SV08', 'SV10', 'SV15', 'SV20'],
    'Steepness @ m=1.0': ['K3M10', 'SV10', 'K8M10'],
    'Dampen Floor @ m=1.0': ['D20M10', 'SV10', 'D40M10', 'D50M10'],
    'ATR SL Multiplier': ['ATR30', 'NoVeto', 'ATR40', 'ATR45'],
    'Close Thresholds': ['CLS35', 'NoVeto', 'CLS45', 'CLS50'],
}

for group_name, variants in param_groups.items():
    print(f"\n  [{group_name}]")
    for pn in ['IS', 'OOS', '5yr']:
        base = results[pn]['NoVeto']['pf']
        vals = []
        for vn in variants:
            info = results[pn].get(vn, {'pf': 0, 'trades': 0})
            d = info['pf'] - base
            vals.append(f"{vn}={info['pf']:.4f}({d:+.4f})")
        print(f"    {pn:4s}: {' | '.join(vals)}")

elapsed = time.time() - t_start
print(f"\nTotal: {elapsed/60:.1f} min, {len(VARIANTS)} variants × {len(PERIODS)} periods")

with open('logs/p42_param_sweep.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("Saved to logs/p42_param_sweep.json")
