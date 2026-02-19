#!/usr/bin/env python3
"""v10.3 参数矩阵快跑（单进程复用数据/信号）

目标:
- 一次加载数据与信号，批量评估参数组合，避免重复预处理
- 先做方向性筛选，再进入完整 WFO/bootstrap 验证
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from itertools import product
from typing import Dict, List

import pandas as pd

from backtest_multi_tf_daily import _build_default_config, _scale_runtime_config
from run_v10_3_validation import (
    PRIMARY_TF,
    _extract_metrics,
    _parse_overrides,
    load_all_data,
    run_once,
)


def _parse_float_list(raw: str) -> List[float]:
    vals = []
    for x in (raw or "").split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    return vals


def _objective(m: Dict) -> float:
    """简单多目标评分: pPF 优先，兼顾 Calmar 与收益。"""
    return (
        float(m.get("portfolio_pf", 0.0)) * 0.60
        + float(m.get("calmar", 0.0)) * 0.25
        + float(m.get("return_pct", 0.0)) / 100.0 * 0.15
    )


def _scalar_metrics(m: Dict) -> Dict:
    return {
        "return_pct": float(m.get("return_pct", 0.0)),
        "max_drawdown_pct": float(m.get("max_drawdown_pct", 0.0)),
        "contract_pf": float(m.get("contract_pf", 0.0)),
        "portfolio_pf": float(m.get("portfolio_pf", 0.0)),
        "calmar": float(m.get("calmar", 0.0)),
        "trades": int(m.get("trades", 0)),
    }


def main():
    p = argparse.ArgumentParser(description="v10.3 参数矩阵快跑")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2026-01-31")
    p.add_argument("--long-thresholds", default="20,25,30")
    p.add_argument("--short-sls", default="-0.16,-0.18,-0.20")
    p.add_argument("--override", action="append", default=[], help="基线覆盖 key=value")
    p.add_argument("--output-dir", default="logs/v10_3_matrix")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)

    print("=" * 90)
    print(f"v10.3 矩阵快跑 | {args.symbol} | {args.start} ~ {args.end}")
    print("=" * 90)

    all_data, all_signals, perp_quality, quality_flags = load_all_data(args.symbol, args.start, args.end)
    print(
        f"数据质量: flags={quality_flags or ['none']}, "
        f"oi_cov={perp_quality.get('oi_orig_coverage', 0.0):.2%}, "
        f"oi_max_stale={perp_quality.get('oi_max_stale_bars', 0)}, "
        f"fr_cov={perp_quality.get('funding_orig_coverage', 0.0):.2%}, "
        f"fr_max_stale={perp_quality.get('funding_max_stale_bars', 0)}"
    )

    base_cfg = _scale_runtime_config(_build_default_config(), PRIMARY_TF)
    base_cfg.update(_parse_overrides(args.override))
    base_cfg["_perp_data_quality"] = perp_quality
    base_cfg["_data_quality_flags"] = quality_flags

    print("\n[1/2] 基线评估...")
    base_res = run_once(all_data, all_signals, base_cfg, start_ts, end_ts)
    base_m = _extract_metrics(base_res, start_ts, end_ts)
    base_m_scalar = _scalar_metrics(base_m)
    print(
        f"  baseline: Ret={base_m['return_pct']:+.2f}% pPF={base_m['portfolio_pf']:.3f} "
        f"cPF={base_m['contract_pf']:.3f} Calmar={base_m['calmar']:.3f} MDD={base_m['max_drawdown_pct']:.2f}%"
    )

    long_thresholds = _parse_float_list(args.long_thresholds)
    short_sls = _parse_float_list(args.short_sls)
    combos = list(product(long_thresholds, short_sls))

    print(f"\n[2/2] 组合评估... 共 {len(combos)} 组")
    rows = []
    for i, (lt, ss) in enumerate(combos, 1):
        cfg = dict(base_cfg)
        cfg["long_threshold"] = int(round(lt))
        cfg["short_sl"] = float(ss)
        res = run_once(all_data, all_signals, cfg, start_ts, end_ts)
        m = _extract_metrics(res, start_ts, end_ts)
        score = _objective(m)
        row = {
            "rank_key_score": score,
            "long_threshold": int(round(lt)),
            "short_sl": float(ss),
            "return_pct": float(m["return_pct"]),
            "portfolio_pf": float(m["portfolio_pf"]),
            "contract_pf": float(m["contract_pf"]),
            "calmar": float(m["calmar"]),
            "max_drawdown_pct": float(m["max_drawdown_pct"]),
            "trades": int(m["trades"]),
            "delta_return_pct": float(m["return_pct"] - base_m["return_pct"]),
            "delta_portfolio_pf": float(m["portfolio_pf"] - base_m["portfolio_pf"]),
            "delta_calmar": float(m["calmar"] - base_m["calmar"]),
        }
        rows.append(row)
        print(
            f"  [{i:02d}/{len(combos)}] lt={int(round(lt)):<2d} ss={ss:+.2f} | "
            f"Ret={m['return_pct']:+6.2f}% pPF={m['portfolio_pf']:.3f} Calmar={m['calmar']:.3f}"
        )

    df = pd.DataFrame(rows).sort_values(
        by=["rank_key_score", "portfolio_pf", "calmar", "return_pct"], ascending=False
    )
    best = df.iloc[0].to_dict() if not df.empty else {}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"v10_3_matrix_{ts}.csv")
    json_path = os.path.join(args.output_dir, f"v10_3_matrix_{ts}.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "symbol": args.symbol,
                "range": {"start": args.start, "end": args.end},
                "base_metrics": base_m_scalar,
                "best": best,
                "rows": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nTop 5:")
    print(df.head(5).to_string(index=False))
    print("\n输出文件:")
    print(f"  {csv_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
