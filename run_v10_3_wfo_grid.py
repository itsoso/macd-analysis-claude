#!/usr/bin/env python3
"""v10.3 WFO 导向网格搜索（复用同一份数据与信号）

目标:
- 不再按全样本收益排序，而按 WFO 稳健性排序
- 输出每个组合的:
  - 全样本指标
  - WFO 盈利窗口占比 / 中位 pPF / 通过标记
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
from optimize_six_book import _build_tf_score_index
from run_v10_3_validation import (
    NEEDED_TFS,
    PRIMARY_TF,
    _extract_metrics,
    _parse_overrides,
    build_monthly_windows,
    load_all_data,
    run_once,
)


def _parse_int_list(raw: str) -> List[int]:
    out = []
    for x in (raw or "").split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(float(x)))
    return out


def _parse_float_list(raw: str) -> List[float]:
    out = []
    for x in (raw or "").split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out


def _parse_bool_list(raw: str) -> List[bool]:
    out: List[bool] = []
    for x in (raw or "").split(","):
        v = x.strip().lower()
        if not v:
            continue
        if v in ("1", "true", "yes", "y", "on"):
            out.append(True)
        elif v in ("0", "false", "no", "n", "off"):
            out.append(False)
    return out


def _normalize_cfg_for_json(cfg: Dict) -> Dict:
    out = {}
    for k, v in cfg.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = str(v)
    return out


def _wfo_score(win_ratio: float, median_ppf: float, median_calmar: float, full_ret: float, pass_flag: bool) -> float:
    bonus = 2.0 if pass_flag else 0.0
    return bonus + median_ppf * 0.7 + win_ratio * 0.8 + median_calmar * 0.15 + (full_ret / 100.0) * 0.2


def _apply_combo(cfg: Dict, long_threshold: int, neutral_short_thr: int, cooldown: int, risk_pct: float) -> Dict:
    c = dict(cfg)
    c["long_threshold"] = int(long_threshold)
    c["cooldown"] = int(cooldown)
    c["risk_per_trade_pct"] = float(risk_pct)
    c["regime_short_threshold"] = f"neutral:{int(neutral_short_thr)}"
    return c


def main():
    p = argparse.ArgumentParser(description="v10.3 WFO 导向网格搜索")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2026-01-31")
    p.add_argument("--train-months", type=int, default=6)
    p.add_argument("--test-months", type=int, default=1)
    p.add_argument("--long-thresholds", default="20,25,30")
    p.add_argument("--neutral-short-thresholds", default="45,60,999")
    p.add_argument("--cooldowns", default="4,6")
    p.add_argument("--risk-pcts", default="0.020,0.025")
    p.add_argument(
        "--close-long-ss-values",
        default="",
        help="close_long_ss 候选列表(逗号分隔)。留空则使用基线值",
    )
    p.add_argument(
        "--neutral-short-budgets",
        default="0.10",
        help="risk_budget_neutral_short 候选列表，逗号分隔",
    )
    p.add_argument(
        "--book-consensus-values",
        default="false",
        help="use_neutral_book_consensus 候选布尔值，逗号分隔，如 false,true",
    )
    p.add_argument(
        "--dynamic-neutral-short-budget-values",
        default="false",
        help="use_neutral_short_dynamic_budget 候选布尔值，逗号分隔，如 false,true",
    )
    p.add_argument(
        "--structure-anchor-sl-values",
        default="",
        help="use_structure_anchor_sl 候选布尔值，逗号分隔。留空则使用基线值",
    )
    p.add_argument("--override", action="append", default=[], help="基线覆盖 key=value")
    p.add_argument(
        "--reuse-tf-score-map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在网格中复用同一份 tf_score_map（默认开启，加速明显）",
    )
    p.add_argument("--output-dir", default="logs/v10_3_wfo_grid")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)

    print("=" * 100)
    print(f"v10.3 WFO网格 | {args.symbol} | {args.start} ~ {args.end}")
    print("=" * 100)

    all_data, all_signals, perp_quality, quality_flags = load_all_data(args.symbol, args.start, args.end)
    windows = build_monthly_windows(args.start, args.end, args.train_months, args.test_months)
    print(
        f"数据质量: flags={quality_flags or ['none']} | "
        f"oi_cov={perp_quality.get('oi_orig_coverage', 0.0):.2%} | "
        f"oi_max_stale={perp_quality.get('oi_max_stale_bars', 0)} | "
        f"fr_cov={perp_quality.get('funding_orig_coverage', 0.0):.2%} | "
        f"fr_max_stale={perp_quality.get('funding_max_stale_bars', 0)}"
    )
    print(f"WFO 窗口数: {len(windows)} (train={args.train_months}m,test={args.test_months}m)")

    base_cfg = _scale_runtime_config(_build_default_config(), PRIMARY_TF)
    base_cfg.update(_parse_overrides(args.override))
    base_cfg["_perp_data_quality"] = perp_quality
    base_cfg["_data_quality_flags"] = quality_flags

    longs = _parse_int_list(args.long_thresholds)
    shorts = _parse_int_list(args.neutral_short_thresholds)
    cds = _parse_int_list(args.cooldowns)
    rps = _parse_float_list(args.risk_pcts)
    cls = _parse_int_list(args.close_long_ss_values)
    if not cls:
        cls = [int(base_cfg.get("close_long_ss", 40))]
    nbs = _parse_float_list(args.neutral_short_budgets)
    bcs = _parse_bool_list(args.book_consensus_values) or [False]
    dyns = _parse_bool_list(args.dynamic_neutral_short_budget_values) or [False]
    sas = _parse_bool_list(args.structure_anchor_sl_values)
    if not sas:
        sas = [bool(base_cfg.get("use_structure_anchor_sl", False))]
    combos = list(product(longs, shorts, cds, rps, nbs, bcs, dyns, cls, sas))

    print(f"组合总数: {len(combos)}")
    shared_tf_score_map = None
    if args.reuse_tf_score_map:
        shared_tf_score_map = _build_tf_score_index(all_data, all_signals, NEEDED_TFS, base_cfg)
        print("tf_score_map: 复用模式 ON")
    else:
        print("tf_score_map: 逐组合重建模式 ON")
    rows = []
    for i, (lt, nst, cd, rp, nb, bc, dyn_budget, cls_v, sas_v) in enumerate(combos, 1):
        cfg = _apply_combo(base_cfg, lt, nst, cd, rp)
        cfg["risk_budget_neutral_short"] = float(nb)
        cfg["use_leg_risk_budget"] = True
        cfg["use_neutral_book_consensus"] = bool(bc)
        cfg["use_neutral_short_dynamic_budget"] = bool(dyn_budget)
        cfg["close_long_ss"] = int(cls_v)
        cfg["use_structure_anchor_sl"] = bool(sas_v)
        tf_score_map = shared_tf_score_map
        if tf_score_map is None:
            tf_score_map = _build_tf_score_index(all_data, all_signals, NEEDED_TFS, cfg)
        full_res = run_once(all_data, all_signals, cfg, start_ts, end_ts, tf_score_map=tf_score_map)
        full_m = _extract_metrics(full_res, start_ts, end_ts)

        wfo_rows = []
        for tr_s, tr_e, te_s, te_e in windows:
            te_res = run_once(all_data, all_signals, cfg, te_s, te_e, tf_score_map=tf_score_map)
            te_m = _extract_metrics(te_res, te_s, te_e)
            wfo_rows.append(
                {
                    "test_return_pct": float(te_m["return_pct"]),
                    "test_portfolio_pf": float(te_m["portfolio_pf"]),
                    "test_calmar": float(te_m["calmar"]),
                }
            )
        wfo_df = pd.DataFrame(wfo_rows)
        win_ratio = float((wfo_df["test_return_pct"] > 0).mean()) if not wfo_df.empty else 0.0
        median_ppf = float(wfo_df["test_portfolio_pf"].median()) if not wfo_df.empty else 0.0
        median_calmar = float(wfo_df["test_calmar"].median()) if not wfo_df.empty else 0.0
        pass_flag = bool(win_ratio >= 0.60 and median_ppf > 1.20)
        score = _wfo_score(win_ratio, median_ppf, median_calmar, float(full_m["return_pct"]), pass_flag)

        row = {
            "wfo_score": score,
            "wfo_pass": pass_flag,
            "wfo_win_ratio": win_ratio,
            "wfo_median_ppf": median_ppf,
            "wfo_median_calmar": median_calmar,
            "long_threshold": int(lt),
            "neutral_short_threshold": int(nst),
            "cooldown": int(cd),
            "risk_per_trade_pct": float(rp),
            "close_long_ss": int(cls_v),
            "use_structure_anchor_sl": bool(sas_v),
            "risk_budget_neutral_short": float(nb),
            "use_neutral_book_consensus": bool(bc),
            "use_neutral_short_dynamic_budget": bool(dyn_budget),
            "full_return_pct": float(full_m["return_pct"]),
            "full_portfolio_pf": float(full_m["portfolio_pf"]),
            "full_contract_pf": float(full_m["contract_pf"]),
            "full_calmar": float(full_m["calmar"]),
            "full_max_drawdown_pct": float(full_m["max_drawdown_pct"]),
            "full_trades": int(full_m["trades"]),
        }
        rows.append(row)
        print(
            f"[{i:02d}/{len(combos)}] lt={lt} nst={nst} cd={cd} rp={rp:.3f} cls={int(cls_v)} "
            f"nbs={nb:.3f} bc={int(bool(bc))} dyn={int(bool(dyn_budget))} sas={int(bool(sas_v))} | "
            f"WFO(win={win_ratio:.1%},medPPF={median_ppf:.3f},pass={pass_flag}) | "
            f"Full(Ret={row['full_return_pct']:+.2f}%,pPF={row['full_portfolio_pf']:.3f})"
        )

    df = pd.DataFrame(rows).sort_values(
        by=["wfo_pass", "wfo_score", "wfo_median_ppf", "full_portfolio_pf"], ascending=[False, False, False, False]
    )
    best = df.iloc[0].to_dict() if not df.empty else {}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"v10_3_wfo_grid_{ts}.csv")
    json_path = os.path.join(args.output_dir, f"v10_3_wfo_grid_{ts}.json")
    df.to_csv(csv_path, index=False)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "symbol": args.symbol,
        "range": {"start": args.start, "end": args.end},
        "grid": {
            "long_thresholds": longs,
            "neutral_short_thresholds": shorts,
            "cooldowns": cds,
            "risk_pcts": rps,
            "close_long_ss_values": cls,
            "neutral_short_budgets": nbs,
            "book_consensus_values": bcs,
            "dynamic_neutral_short_budget_values": dyns,
            "structure_anchor_sl_values": sas,
            "train_months": args.train_months,
            "test_months": args.test_months,
            "num_combos": len(combos),
            "num_windows": len(windows),
        },
        "base_cfg": _normalize_cfg_for_json(
            {
                "use_risk_per_trade": base_cfg.get("use_risk_per_trade"),
                "decision_tfs": base_cfg.get("decision_tfs"),
                "regime_short_threshold": base_cfg.get("regime_short_threshold"),
                "use_regime_sigmoid": base_cfg.get("use_regime_sigmoid"),
            }
        ),
        "best": best,
        "top5": df.head(5).to_dict(orient="records"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nTop 5:")
    print(df.head(5).to_string(index=False))
    print("\n输出文件:")
    print(f"  {csv_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
