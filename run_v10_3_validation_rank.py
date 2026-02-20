#!/usr/bin/env python3
"""汇总 v10.3 validation 结果并按稳健性排序。

用法:
  python3 run_v10_3_validation_rank.py
  python3 run_v10_3_validation_rank.py --glob 'logs/v10_3_validation_*/*.json'
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _to_bool(v: Any, default: bool = False) -> bool:
    try:
        return bool(v)
    except Exception:
        return bool(default)


def _format_overrides(v: Any) -> str:
    if not isinstance(v, dict) or not v:
        return ""
    items = sorted(v.items(), key=lambda x: str(x[0]))
    return ", ".join(f"{k}={vv}" for k, vv in items)


def _read_one(path: str) -> Dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None

    base = dict(obj.get("baseline_metrics") or {})
    var = dict(obj.get("variant_metrics") or {})
    wfo = dict(obj.get("wfo") or {})
    stat = dict(obj.get("stat_significance") or {})
    stress = dict(obj.get("stress_test") or {})
    quality = dict(obj.get("perp_data_quality") or {})
    cfg = dict(obj.get("config") or {})

    baseline_ppf = _to_float(base.get("portfolio_pf"))
    variant_ppf = _to_float(var.get("portfolio_pf"))
    baseline_ret = _to_float(base.get("return_pct"))
    variant_ret = _to_float(var.get("return_pct"))
    baseline_calmar = _to_float(base.get("calmar"))
    variant_calmar = _to_float(var.get("calmar"))
    baseline_mdd = _to_float(base.get("max_drawdown_pct"))
    variant_mdd = _to_float(var.get("max_drawdown_pct"))

    row = {
        "file": path,
        "tag": os.path.basename(os.path.dirname(path)),
        "generated_at": obj.get("generated_at", ""),
        "range_start": (obj.get("range") or {}).get("start"),
        "range_end": (obj.get("range") or {}).get("end"),
        "variant_overrides": _format_overrides(cfg.get("variant_overrides")),
        "base_return_pct": baseline_ret,
        "var_return_pct": variant_ret,
        "delta_return_pct": variant_ret - baseline_ret,
        "base_portfolio_pf": baseline_ppf,
        "var_portfolio_pf": variant_ppf,
        "delta_portfolio_pf": variant_ppf - baseline_ppf,
        "base_calmar": baseline_calmar,
        "var_calmar": variant_calmar,
        "delta_calmar": variant_calmar - baseline_calmar,
        "base_mdd_pct": baseline_mdd,
        "var_mdd_pct": variant_mdd,
        "delta_mdd_pct": variant_mdd - baseline_mdd,
        "base_trades": _to_float(base.get("trades")),
        "var_trades": _to_float(var.get("trades")),
        "wfo_windows": int(_to_float(wfo.get("windows"), 0)),
        "wfo_windows_all": int(_to_float(wfo.get("windows_all"), 0)),
        "wfo_profit_ratio": _to_float(wfo.get("profit_window_ratio")),
        "wfo_median_ppf": _to_float(wfo.get("median_portfolio_pf")),
        "wfo_pass": _to_bool(wfo.get("pass")),
        "bootstrap_skipped": _to_bool(stat.get("skipped"), True),
        "stress_skipped": _to_bool(stress.get("skipped"), True),
        "stress_pass": _to_bool(stress.get("pass"), True),
        "quality_flags": ",".join(quality.get("quality_flags") or []),
        "oi_cov": _to_float(quality.get("oi_orig_coverage")),
        "funding_cov": _to_float(quality.get("funding_orig_coverage")),
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="v10.3 validation 结果排行榜")
    ap.add_argument("--glob", default="logs/v10_3_validation_*/*.json")
    ap.add_argument("--output-dir", default="logs")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--min-windows", type=int, default=0, help="仅保留 wfo_windows >= 该值")
    ap.add_argument("--range-start", default="", help="仅保留指定起始日期")
    ap.add_argument("--range-end", default="", help="仅保留指定结束日期")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    rows: List[Dict[str, Any]] = []
    for p in paths:
        r = _read_one(p)
        if r:
            rows.append(r)
    if not rows:
        print("未找到可解析的 validation json")
        return

    df = pd.DataFrame(rows)
    if args.min_windows > 0:
        df = df[df["wfo_windows"] >= int(args.min_windows)]
    if args.range_start:
        df = df[df["range_start"] == args.range_start]
    if args.range_end:
        df = df[df["range_end"] == args.range_end]
    if df.empty:
        print("过滤后无结果")
        return
    df = df.sort_values(
        by=[
            "wfo_pass",
            "wfo_median_ppf",
            "wfo_profit_ratio",
            "delta_portfolio_pf",
            "delta_return_pct",
            "var_portfolio_pf",
        ],
        ascending=[False, False, False, False, False, False],
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f"v10_3_validation_rank_{ts}.csv")
    out_json = os.path.join(args.output_dir, f"v10_3_validation_rank_{ts}.json")
    df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    cols = [
        "tag",
        "variant_overrides",
        "var_return_pct",
        "var_portfolio_pf",
        "delta_return_pct",
        "delta_portfolio_pf",
        "wfo_profit_ratio",
        "wfo_median_ppf",
        "wfo_windows",
        "wfo_pass",
        "quality_flags",
    ]
    print(df[cols].head(args.top).to_string(index=False))
    print("\n输出:")
    print(out_csv)
    print(out_json)


if __name__ == "__main__":
    main()
