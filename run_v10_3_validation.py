#!/usr/bin/env python3
"""v10.3 阶段C: 统计验证优先框架

包含:
1) 扩样基线重测 (2021-01-01 ~ end)
2) 月度滚动 WFO (6m train + 1m test)
3) bootstrap 显著性 (ΔpPF / ΔCalmar)
4) 压力测试 (fee x2, slippage 0.03%/0.05%)
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from backtest_multi_tf_daily import _build_default_config, _scale_runtime_config
from binance_fetcher import (
    fetch_funding_rate_history,
    fetch_mark_price_klines,
    fetch_open_interest_history,
    merge_perp_data_into_klines,
)
from indicators import add_all_indicators
from kline_store import load_klines
from ma_indicators import add_moving_averages
from optimize_six_book import _build_tf_score_index, compute_signals_six, run_strategy_multi_tf
from strategy_futures import FuturesEngine


PRIMARY_TF = "1h"
DECISION_TFS = ["15m", "1h", "4h", "24h"]
NEEDED_TFS = sorted(set([PRIMARY_TF] + DECISION_TFS))
PNL_ACTIONS = {"CLOSE_LONG", "CLOSE_SHORT", "LIQUIDATED", "PARTIAL_TP", "SPOT_SELL"}
CONTRACT_CLOSE_ACTIONS = {"CLOSE_LONG", "CLOSE_SHORT", "LIQUIDATED"}


def _parse_overrides(raw: List[str]) -> Dict:
    out = {}
    for item in raw or []:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
            continue
        if v.startswith("{") or v.startswith("["):
            try:
                out[k] = json.loads(v)
                continue
            except Exception:
                pass
        try:
            out[k] = int(v)
            continue
        except ValueError:
            pass
        try:
            out[k] = float(v)
            continue
        except ValueError:
            pass
        out[k] = v
    return out


def _extract_perp_quality(df: pd.DataFrame) -> Tuple[dict, List[str]]:
    audit = dict(df.attrs.get("perp_data_audit_dict") or {})
    oi_audit = dict(audit.get("open_interest") or {})
    oi_cov = float(oi_audit.get("orig_coverage", 0.0) or 0.0)
    oi_stale = int(oi_audit.get("max_stale_bars", 0) or 0)
    flags = []
    if (oi_cov < 0.20) or (oi_stale > 12):
        for c in ("open_interest", "open_interest_value"):
            if c in df.columns:
                df[c] = np.nan
        flags.append("oi_soft_disabled")
    quality = {
        "audit": audit,
        "oi_orig_coverage": oi_cov,
        "oi_max_stale_bars": oi_stale,
        "oi_soft_disabled": bool("oi_soft_disabled" in flags),
        "quality_flags": flags,
    }
    return quality, flags


def load_all_data(symbol: str, start: str, end: str, warmup_days: int = 120):
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    warmup_start = (start_ts - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    end_s = end_ts.strftime("%Y-%m-%d")

    all_data = {}
    for tf in NEEDED_TFS:
        df = load_klines(
            symbol=symbol,
            interval=tf,
            start=warmup_start,
            end=end_s,
            with_indicators=False,
            allow_api_fallback=False,
        )
        if df is None or len(df) < 200:
            raise RuntimeError(f"{tf} 数据不足，请先执行 run_v10_3_data_backfill.py")
        all_data[tf] = df

    days_full = max(1, int((pd.Timestamp.now().tz_localize(None) - pd.Timestamp(warmup_start)).days + 3))
    mark_df = fetch_mark_price_klines(
        symbol=symbol, interval=PRIMARY_TF, days=days_full, force_api=False, allow_api_fallback=False
    )
    funding_df = fetch_funding_rate_history(
        symbol=symbol, days=days_full, force_api=False, allow_api_fallback=False
    )
    oi_df = fetch_open_interest_history(
        symbol=symbol, interval=PRIMARY_TF, days=30, force_api=False, allow_api_fallback=False
    )
    all_data[PRIMARY_TF] = merge_perp_data_into_klines(all_data[PRIMARY_TF], mark_df, funding_df, oi_df)
    perp_quality, quality_flags = _extract_perp_quality(all_data[PRIMARY_TF])

    for tf in NEEDED_TFS:
        all_data[tf] = add_all_indicators(all_data[tf])
        add_moving_averages(all_data[tf], timeframe=tf)

    all_signals = {
        tf: compute_signals_six(all_data[tf], tf, all_data, max_bars=0)
        for tf in NEEDED_TFS
    }
    return all_data, all_signals, perp_quality, quality_flags


def _calc_pf(trades: list, actions: set) -> float:
    pnls = [float(t.get("pnl") or 0.0) for t in trades if t.get("action") in actions and t.get("pnl") is not None]
    gp = sum(v for v in pnls if v > 0)
    gl = abs(sum(v for v in pnls if v < 0))
    if gl <= 0:
        return 999.0
    return gp / gl


def _history_daily_returns(history: list) -> np.ndarray:
    if not history:
        return np.array([])
    df = pd.DataFrame(history)
    if "time" not in df.columns or "total" not in df.columns:
        return np.array([])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    if df.empty:
        return np.array([])
    daily_total = df["total"].resample("1D").last().dropna()
    if len(daily_total) < 3:
        return np.array([])
    rets = daily_total.pct_change().dropna()
    return rets.values.astype(float)


def _calmar_from_daily_returns(daily_rets: np.ndarray) -> float:
    if daily_rets is None or len(daily_rets) < 10:
        return 0.0
    equity = np.cumprod(1.0 + daily_rets)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-12)
    max_dd = abs(float(np.min(dd)))
    total_ret = float(equity[-1] - 1.0)
    years = max(len(daily_rets) / 365.0, 1e-9)
    if 1.0 + total_ret <= 0:
        cagr = -1.0
    else:
        cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0
    if max_dd <= 1e-12:
        return 0.0
    return cagr / max_dd


def _extract_metrics(result: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    trades = result.get("trades", [])
    strategy_return = float(result.get("strategy_return", 0.0))
    max_dd_pct = float(result.get("max_drawdown", 0.0))
    days = max((end - start).days + 1, 1)
    if 1.0 + strategy_return / 100.0 <= 0:
        cagr = -1.0
    else:
        cagr = (1.0 + strategy_return / 100.0) ** (365.0 / days) - 1.0
    calmar = cagr / (abs(max_dd_pct) / 100.0) if abs(max_dd_pct) > 1e-9 else 0.0
    return {
        "return_pct": strategy_return,
        "max_drawdown_pct": max_dd_pct,
        "contract_pf": round(_calc_pf(trades, CONTRACT_CLOSE_ACTIONS), 4),
        "portfolio_pf": round(_calc_pf(trades, PNL_ACTIONS), 4),
        "calmar": round(calmar, 4),
        "trades": int(len([t for t in trades if t.get("action") in CONTRACT_CLOSE_ACTIONS])),
        "history_daily_returns": _history_daily_returns(result.get("history", [])),
        "pnl_values": np.array(
            [float(t.get("pnl") or 0.0) for t in trades if t.get("action") in PNL_ACTIONS and t.get("pnl") is not None],
            dtype=float,
        ),
    }


@contextmanager
def cost_scenario(fee_mult: float = 1.0, slippage_abs: float | None = None):
    fee_bak = FuturesEngine.TAKER_FEE
    slip_bak = FuturesEngine.SLIPPAGE
    try:
        FuturesEngine.TAKER_FEE = float(fee_bak) * float(fee_mult)
        if slippage_abs is not None:
            FuturesEngine.SLIPPAGE = float(slippage_abs)
        yield
    finally:
        FuturesEngine.TAKER_FEE = fee_bak
        FuturesEngine.SLIPPAGE = slip_bak


def run_once(
    all_data: dict,
    all_signals: dict,
    base_config: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fee_mult: float = 1.0,
    slippage_abs: float | None = None,
    tf_score_map: dict | None = None,
) -> dict:
    cfg = dict(base_config)
    if tf_score_map is None:
        tf_score_map = _build_tf_score_index(all_data, all_signals, NEEDED_TFS, cfg)
    with cost_scenario(fee_mult=fee_mult, slippage_abs=slippage_abs):
        result = run_strategy_multi_tf(
            primary_df=all_data[PRIMARY_TF],
            tf_score_map=tf_score_map,
            decision_tfs=DECISION_TFS,
            config=cfg,
            primary_tf=PRIMARY_TF,
            trade_days=0,
            trade_start_dt=start,
            trade_end_dt=end + pd.Timedelta(hours=23, minutes=59),
        )
    return result


def build_monthly_windows(start: str, end: str, train_months: int = 6, test_months: int = 1):
    s = pd.Timestamp(start).replace(day=1)
    e = pd.Timestamp(end)
    months = pd.date_range(s, e, freq="MS")
    windows = []
    for i in range(train_months, len(months) - test_months + 1):
        train_start = months[i - train_months]
        train_end = months[i] - pd.Timedelta(days=1)
        test_start = months[i]
        if i + test_months < len(months):
            test_end = months[i + test_months] - pd.Timedelta(days=1)
        else:
            test_end = e
        if test_end > e:
            test_end = e
        windows.append((train_start, train_end, test_start, test_end))
    return windows


def bootstrap_delta(base_values: np.ndarray, var_values: np.ndarray, n: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    if len(base_values) < 10 or len(var_values) < 10:
        return {"p_value": 1.0, "ci_low": 0.0, "ci_high": 0.0, "delta_obs": 0.0}
    deltas = np.empty(n, dtype=float)
    for i in range(n):
        b = rng.choice(base_values, size=len(base_values), replace=True)
        v = rng.choice(var_values, size=len(var_values), replace=True)
        b_pf = _calc_pf([{"action": "CLOSE_SHORT", "pnl": x} for x in b], {"CLOSE_SHORT"})
        v_pf = _calc_pf([{"action": "CLOSE_SHORT", "pnl": x} for x in v], {"CLOSE_SHORT"})
        deltas[i] = v_pf - b_pf
    delta_obs = float(np.mean(var_values) - np.mean(base_values))
    if delta_obs >= 0:
        p_val = float(np.mean(deltas <= 0))
    else:
        p_val = float(np.mean(deltas >= 0))
    return {
        "p_value": round(p_val, 6),
        "ci_low": round(float(np.quantile(deltas, 0.05)), 6),
        "ci_high": round(float(np.quantile(deltas, 0.95)), 6),
        "delta_obs": round(delta_obs, 6),
    }


def bootstrap_delta_calmar(base_daily: np.ndarray, var_daily: np.ndarray, n: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed + 7)
    if len(base_daily) < 30 or len(var_daily) < 30:
        return {"p_value": 1.0, "ci_low": 0.0, "ci_high": 0.0, "delta_obs": 0.0}
    deltas = np.empty(n, dtype=float)
    for i in range(n):
        b = rng.choice(base_daily, size=len(base_daily), replace=True)
        v = rng.choice(var_daily, size=len(var_daily), replace=True)
        deltas[i] = _calmar_from_daily_returns(v) - _calmar_from_daily_returns(b)
    delta_obs = _calmar_from_daily_returns(var_daily) - _calmar_from_daily_returns(base_daily)
    if delta_obs >= 0:
        p_val = float(np.mean(deltas <= 0))
    else:
        p_val = float(np.mean(deltas >= 0))
    return {
        "p_value": round(p_val, 6),
        "ci_low": round(float(np.quantile(deltas, 0.05)), 6),
        "ci_high": round(float(np.quantile(deltas, 0.95)), 6),
        "delta_obs": round(float(delta_obs), 6),
    }


def main():
    p = argparse.ArgumentParser(description="v10.3 统计验证管道 (WFO + bootstrap + stress)")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default=pd.Timestamp.now().strftime("%Y-%m-%d"))
    p.add_argument("--train-months", type=int, default=6)
    p.add_argument("--test-months", type=int, default=1)
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--override", action="append", default=[], help="基线覆盖 key=value")
    p.add_argument("--variant-override", action="append", default=[], help="变体覆盖 key=value")
    p.add_argument("--output-dir", default="logs/v10_3_validation")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)

    print("=" * 90)
    print(f"v10.3 验证 | {args.symbol} | {args.start} ~ {args.end}")
    print("=" * 90)

    all_data, all_signals, perp_quality, quality_flags = load_all_data(args.symbol, args.start, args.end)
    print(f"数据质量: flags={quality_flags or ['none']}, oi_cov={perp_quality.get('oi_orig_coverage', 0.0):.2%}, "
          f"oi_max_stale={perp_quality.get('oi_max_stale_bars', 0)}")

    base_cfg = _scale_runtime_config(_build_default_config(), PRIMARY_TF)
    base_cfg.update(_parse_overrides(args.override))
    base_cfg["_perp_data_quality"] = perp_quality
    base_cfg["_data_quality_flags"] = quality_flags

    variant_cfg = dict(base_cfg)
    variant_cfg.update(_parse_overrides(args.variant_override))

    # tf_score_index 构建代价高（searchsorted on timestamps），按配置预构建并复用。
    base_tf_score_map = _build_tf_score_index(all_data, all_signals, NEEDED_TFS, base_cfg)
    var_tf_score_map = _build_tf_score_index(all_data, all_signals, NEEDED_TFS, variant_cfg)

    # C1: 扩样基线重测
    print("\n[1/4] 扩样基线重测...")
    base_res = run_once(all_data, all_signals, base_cfg, start_ts, end_ts, tf_score_map=base_tf_score_map)
    var_res = run_once(all_data, all_signals, variant_cfg, start_ts, end_ts, tf_score_map=var_tf_score_map)
    base_m = _extract_metrics(base_res, start_ts, end_ts)
    var_m = _extract_metrics(var_res, start_ts, end_ts)
    print(f"  baseline: Ret={base_m['return_pct']:+.2f}% pPF={base_m['portfolio_pf']:.3f} "
          f"cPF={base_m['contract_pf']:.3f} Calmar={base_m['calmar']:.3f} MDD={base_m['max_drawdown_pct']:.2f}%")
    print(f"  variant : Ret={var_m['return_pct']:+.2f}% pPF={var_m['portfolio_pf']:.3f} "
          f"cPF={var_m['contract_pf']:.3f} Calmar={var_m['calmar']:.3f} MDD={var_m['max_drawdown_pct']:.2f}%")

    # C2: 月度滚动WFO
    print("\n[2/4] 月度滚动 WFO (6m+1m)...")
    windows = build_monthly_windows(args.start, args.end, args.train_months, args.test_months)
    wfo_rows = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows, 1):
        te_res = run_once(all_data, all_signals, variant_cfg, te_s, te_e, tf_score_map=var_tf_score_map)
        te_m = _extract_metrics(te_res, te_s, te_e)
        wfo_rows.append(
            {
                "window": i,
                "train_start": tr_s.strftime("%Y-%m-%d"),
                "train_end": tr_e.strftime("%Y-%m-%d"),
                "test_start": te_s.strftime("%Y-%m-%d"),
                "test_end": te_e.strftime("%Y-%m-%d"),
                "test_return_pct": te_m["return_pct"],
                "test_portfolio_pf": te_m["portfolio_pf"],
                "test_calmar": te_m["calmar"],
                "test_mdd_pct": te_m["max_drawdown_pct"],
                "test_trades": te_m["trades"],
            }
        )
    wfo_df = pd.DataFrame(wfo_rows)
    win_ratio = float((wfo_df["test_return_pct"] > 0).mean()) if not wfo_df.empty else 0.0
    median_ppf = float(wfo_df["test_portfolio_pf"].median()) if not wfo_df.empty else 0.0
    wfo_pass = (win_ratio >= 0.60) and (median_ppf > 1.2)
    print(f"  WFO窗口: {len(wfo_df)} | 盈利占比={win_ratio:.2%} | 中位pPF={median_ppf:.3f} | pass={wfo_pass}")

    # C3: bootstrap 显著性
    print("\n[3/4] bootstrap 显著性...")
    stat_ppf = bootstrap_delta(base_m["pnl_values"], var_m["pnl_values"], n=args.bootstrap_n)
    stat_calmar = bootstrap_delta_calmar(
        base_m["history_daily_returns"], var_m["history_daily_returns"], n=args.bootstrap_n
    )
    print(f"  ΔpPF p={stat_ppf['p_value']:.4f} CI[{stat_ppf['ci_low']:.4f},{stat_ppf['ci_high']:.4f}]")
    print(f"  ΔCalmar p={stat_calmar['p_value']:.4f} CI[{stat_calmar['ci_low']:.4f},{stat_calmar['ci_high']:.4f}]")

    # C4: 压力测试
    print("\n[4/4] 压力测试...")
    stress_cases = [
        ("baseline_cost", 1.0, None),
        ("fee_x2", 2.0, None),
        ("slippage_0.03pct", 1.0, 0.0003),
        ("slippage_0.05pct", 1.0, 0.0005),
    ]
    stress_rows = []
    for name, fee_mult, slip in stress_cases:
        rs = run_once(
            all_data,
            all_signals,
            variant_cfg,
            start_ts,
            end_ts,
            fee_mult=fee_mult,
            slippage_abs=slip,
            tf_score_map=var_tf_score_map,
        )
        m = _extract_metrics(rs, start_ts, end_ts)
        stress_rows.append(
            {
                "scenario": name,
                "fee_mult": fee_mult,
                "slippage": slip,
                "return_pct": m["return_pct"],
                "portfolio_pf": m["portfolio_pf"],
                "max_drawdown_pct": m["max_drawdown_pct"],
            }
        )
        print(f"  {name:<18} Ret={m['return_pct']:+.2f}% pPF={m['portfolio_pf']:.3f} MDD={m['max_drawdown_pct']:.2f}%")
    stress_df = pd.DataFrame(stress_rows)
    base_stress = stress_df.iloc[0] if not stress_df.empty else None
    pressure_pass = True
    if base_stress is not None:
        base_mdd_abs = abs(float(base_stress["max_drawdown_pct"]))
        for _, row in stress_df.iloc[1:].iterrows():
            if float(row["portfolio_pf"]) <= 1.0:
                pressure_pass = False
            if base_mdd_abs > 1e-9 and abs(float(row["max_drawdown_pct"])) > base_mdd_abs * 1.25:
                pressure_pass = False
    print(f"  压力测试通过: {pressure_pass}")

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "symbol": args.symbol,
        "range": {"start": args.start, "end": args.end},
        "perp_data_quality": perp_quality,
        "baseline_metrics": {k: v for k, v in base_m.items() if k not in ("history_daily_returns", "pnl_values")},
        "variant_metrics": {k: v for k, v in var_m.items() if k not in ("history_daily_returns", "pnl_values")},
        "wfo": {
            "train_months": args.train_months,
            "test_months": args.test_months,
            "windows": int(len(wfo_df)),
            "profit_window_ratio": win_ratio,
            "median_portfolio_pf": median_ppf,
            "pass": bool(wfo_pass),
        },
        "stat_significance": {
            "bootstrap_n": int(args.bootstrap_n),
            "delta_portfolio_pf": stat_ppf,
            "delta_calmar": stat_calmar,
            "pass_rule": "p<=0.10 and direction_consistent",
        },
        "stress_test": {
            "rows": stress_rows,
            "pass": bool(pressure_pass),
            "pass_rule": "pPF>1.0 and MDD not worse than 1.25x baseline",
        },
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.output_dir, f"v10_3_validation_{ts}.json")
    csv_wfo = os.path.join(args.output_dir, f"v10_3_wfo_{ts}.csv")
    csv_stress = os.path.join(args.output_dir, f"v10_3_stress_{ts}.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    wfo_df.to_csv(csv_wfo, index=False)
    stress_df.to_csv(csv_stress, index=False)
    print("\n" + "=" * 90)
    print(f"输出文件:\n  {json_path}\n  {csv_wfo}\n  {csv_stress}")
    print("=" * 90)


if __name__ == "__main__":
    main()
