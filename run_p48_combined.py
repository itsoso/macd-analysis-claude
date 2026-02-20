#!/usr/bin/env python3
"""P48 combined: MH_S48 + P42 stacking + bonus (ratchet, ss_quality_sl)"""
import json, sys, time
sys.path.insert(0, '.')
from optimize_six_book import fetch_multi_tf_data, compute_signals_six, run_strategy

SYMBOLS = ['ETHUSDT', 'BTCUSDT', 'SOLUSDT', 'BNBUSDT']
TF = '1h'

# v6.1 baseline (CandD + P42 SV20+CLS45)
BASE_CFG = {
    'fusion_mode': 'c6_veto_4', 'veto_threshold': 25,
    'single_pct': 0.20, 'total_pct': 0.50, 'lifetime_pct': 5.0,
    'sell_threshold': 14, 'buy_threshold': 25,
    'short_threshold': 18, 'long_threshold': 40,
    'close_short_bs': 45, 'close_long_ss': 45,
    'sell_pct': 0.55, 'margin_use': 0.70, 'lev': 5, 'max_lev': 5,
    'short_sl': -0.15, 'long_sl': -0.08,
    'short_tp': 0.30, 'long_tp': 0.30,
    'short_trail': 0.25, 'long_trail': 0.25,
    'trail_pullback': 0.40,
    'short_max_hold': 72, 'long_max_hold': 72,
    'cooldown': 4,
    'use_atr_sl': True, 'atr_sl_mult': 3.5,
    'use_soft_veto': True, 'soft_veto_steepness': 5.0, 'soft_veto_midpoint': 2.0,
    'use_regime_aware': True,
    'initial_usdt': 100000, 'initial_eth_value': 0,
}

VARIANTS = {
    'Base': {},
    # P48 alone
    'MH_S48': {'short_max_hold': 48},
    # P48 sensitivity
    'MH_S36': {'short_max_hold': 36},
    'MH_S42': {'short_max_hold': 42},
    'MH_S54': {'short_max_hold': 54},
    # Ratchet trail (existing mechanism, disabled by default)
    'Ratchet': {'use_ratchet_trail': True},
    # Signal quality SL (tighter SL for low-confidence entries)
    'SQSL': {'use_ss_quality_sl': True, 'ss_quality_sl_threshold': 50, 'ss_quality_sl_mult': 0.70},
    'SQSL40': {'use_ss_quality_sl': True, 'ss_quality_sl_threshold': 40, 'ss_quality_sl_mult': 0.70},
    # Combined: MH_S48 + Ratchet
    'MH48+Ratch': {'short_max_hold': 48, 'use_ratchet_trail': True},
    # Combined: MH_S48 + SQSL
    'MH48+SQSL': {'short_max_hold': 48, 'use_ss_quality_sl': True, 'ss_quality_sl_threshold': 50, 'ss_quality_sl_mult': 0.70},
}

def extract_pf(result):
    trades = result.get('trades', [])
    close_pnls = [t.get('pnl', 0) for t in trades
                  if t.get('action', '').startswith('CLOSE') and t.get('pnl', 0) != 0]
    gp = sum(p for p in close_pnls if p > 0)
    gl = abs(sum(p for p in close_pnls if p < 0))
    return gp, gl

PERIODS = [('IS', 730), ('OOS', 1865), ('5yr', 1900)]

t_start = time.time()
results = {}

for period_name, days in PERIODS:
    print(f"\n[{period_name}] {days}天", flush=True)
    symbol_data = {}; symbol_signals = {}
    for sym in SYMBOLS:
        data = fetch_multi_tf_data([TF], days=days, symbol=sym)
        symbol_data[sym] = data[TF]
        symbol_signals[sym] = compute_signals_six(data[TF], TF, data)

    results[period_name] = {}
    for vname, voverride in VARIANTS.items():
        cfg = {**BASE_CFG, **voverride}
        total_gp = 0.0; total_gl = 0.0
        sym_pfs = {}
        for sym in SYMBOLS:
            r = run_strategy(symbol_data[sym], symbol_signals[sym], cfg, tf=TF, trade_days=days)
            gp, gl = extract_pf(r)
            total_gp += gp; total_gl += gl
            sym_pfs[sym] = round(gp / gl, 3) if gl > 0 else 0

        agg_pf = round(total_gp / total_gl, 4) if total_gl > 0 else 0
        results[period_name][vname] = agg_pf
        base_pf = results[period_name]['Base']
        delta = agg_pf - base_pf
        print(f"  {vname:14s}: PF={agg_pf:.4f}  Δ={delta:+.4f}  {sym_pfs}", flush=True)

print("\n" + "="*60)
print("P48 Combined — Summary")
print("="*60)
for pn in ['IS', 'OOS', '5yr']:
    print(f"\n  [{pn}]")
    base = results[pn]['Base']
    for vn in VARIANTS:
        pf = results[pn][vn]
        print(f"    {vn:14s}: PF={pf:.4f}  Δ={pf-base:+.4f}")

elapsed = time.time() - t_start
print(f"\nTotal: {elapsed/60:.1f} min")

with open('logs/p48_combined.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved.")
