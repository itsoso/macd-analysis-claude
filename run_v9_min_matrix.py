#!/usr/bin/env python3
"""
V9 最小实验矩阵（主线口径）

目标:
1) 使用与当前主线一致的决策TF: 15m+1h+4h+24h
2) 仅使用本地K线（不走API回退）
3) 不使用 conservative_risk 口径
4) 一次性跑 6 个变体，输出 IS/OOS 对比

区间:
- IS:  2025-01-01 ~ 2026-01-31
- OOS: 2024-01-01 ~ 2024-12-31
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import pandas as pd

from backtest_multi_tf_daily import (
    _build_default_config,
    fetch_data_for_tf,
    PRIMARY_TF,
    DECISION_TFS,
)
from signal_core import compute_signals_six
from optimize_six_book import _build_tf_score_index, run_strategy_multi_tf


IS_START = "2025-01-01"
IS_END = "2026-01-31"
OOS_START = "2024-01-01"
OOS_END = "2024-12-31"


def _min_required_bars(tf: str) -> int:
    if tf == "24h":
        return 50
    if tf == "4h":
        return 120
    return 300


def prepare_data(start_date: str, end_date: str, warmup_days: int = 90):
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    warmup_start = start_dt - pd.Timedelta(days=warmup_days)
    now = pd.Timestamp.now().tz_localize(None)
    history_days = max(180, int((now - warmup_start).days + 5))

    needed_tfs = sorted(set([PRIMARY_TF] + DECISION_TFS))
    all_data = {}
    for tf in needed_tfs:
        min_bars = _min_required_bars(tf)
        df = fetch_data_for_tf(tf, history_days, allow_api_fallback=False)
        if df is None or len(df) < min_bars:
            raise RuntimeError(f"{tf} 数据不足")
        df = df[(df.index >= warmup_start) & (df.index <= end_dt)].copy()
        if len(df) < min_bars:
            raise RuntimeError(f"{tf} 截取后数据不足")
        all_data[tf] = df

    all_signals = {
        tf: compute_signals_six(all_data[tf], tf, all_data, max_bars=0)
        for tf in needed_tfs
    }
    return all_data, all_signals, needed_tfs, start_dt, end_dt


def _close_only_pf_wr(trades):
    closes = [t for t in trades if str(t.get("action", "")).startswith("CLOSE_")]
    n = len(closes)
    wins = sum(1 for t in closes if float(t.get("pnl", 0.0)) > 0)
    wr = (wins / n * 100.0) if n > 0 else 0.0
    gross_profit = sum(float(t.get("pnl", 0.0)) for t in closes if float(t.get("pnl", 0.0)) > 0)
    gross_loss = abs(sum(float(t.get("pnl", 0.0)) for t in closes if float(t.get("pnl", 0.0)) <= 0))
    pf = (gross_profit / gross_loss) if gross_loss > 1e-12 else 0.0
    return wr, pf, n


def _short_open_stats(trades):
    opens = [t for t in trades if t.get("action") == "OPEN_SHORT"]
    neutral = sum(1 for t in opens if str(t.get("regime_label", "")) == "neutral")
    trend = sum(1 for t in opens if str(t.get("regime_label", "")) == "trend")
    hv = sum(1 for t in opens if str(t.get("regime_label", "")) == "high_vol")
    return len(opens), neutral, trend, hv


def run_once(all_data, all_signals, needed_tfs, cfg, start_dt, end_dt):
    tf_score_index = _build_tf_score_index(all_data, all_signals, needed_tfs, cfg)
    result = run_strategy_multi_tf(
        primary_df=all_data[PRIMARY_TF],
        tf_score_map=tf_score_index,
        decision_tfs=DECISION_TFS,
        config=cfg,
        primary_tf=PRIMARY_TF,
        trade_days=0,
        trade_start_dt=start_dt,
        trade_end_dt=end_dt,
    )
    trades = result.get("trades", [])
    wr, pf, close_n = _close_only_pf_wr(trades)
    short_n, short_neutral, short_trend, short_hv = _short_open_stats(trades)
    return {
        "ret": float(result.get("strategy_return", 0.0)),
        "mdd": float(result.get("max_drawdown", 0.0)),
        "wr": float(wr),
        "cpf_close_only": float(pf),
        "close_n": int(close_n),
        "short_open_n": int(short_n),
        "short_open_neutral": int(short_neutral),
        "short_open_trend": int(short_trend),
        "short_open_high_vol": int(short_hv),
    }


def build_variants(base_cfg: dict, variant_set: str = "full"):
    full = {
        # V0: 基线（当前主线）
        "V0_baseline": {},
        # V1: 保守回退（完全禁止 neutral short）
        "V1_neutral_short_block": {
            "regime_short_threshold": "neutral:999",
        },
        # V2: short 门控加强（仅用现有开关近似硬门控）
        "V2_short_gate_plus": {
            "regime_short_threshold": "neutral:60,trend:55,high_vol:55",
            "short_conflict_regimes": "trend,high_vol,neutral",
        },
        # V3: neutral 信号重加权（现有开关可实现的最小版本）
        "V3_neutral_reweight": {
            "use_regime_adaptive_reweight": True,
            "regime_neutral_ss_dampen": 0.75,
            "regime_neutral_bs_boost": 1.20,
        },
        # V4: V2 + V3 组合
        "V4_gate_plus_reweight": {
            "regime_short_threshold": "neutral:60,trend:55,high_vol:55",
            "short_conflict_regimes": "trend,high_vol,neutral",
            "use_regime_adaptive_reweight": True,
            "regime_neutral_ss_dampen": 0.75,
            "regime_neutral_bs_boost": 1.20,
        },
        # V5: V4 + 更紧 short SL（验证“空单错了应更快止损”）
        "V5_v4_tighter_short_sl": {
            "regime_short_threshold": "neutral:60,trend:55,high_vol:55",
            "short_conflict_regimes": "trend,high_vol,neutral",
            "use_regime_adaptive_reweight": True,
            "regime_neutral_ss_dampen": 0.75,
            "regime_neutral_bs_boost": 1.20,
            "short_sl": -0.16,
        },
    }

    # 快速迭代集：只保留最有解释力的 3 个变体
    fast = {
        "V0_baseline": full["V0_baseline"],
        "V1_neutral_short_block": full["V1_neutral_short_block"],
        "V2_short_gate_plus": full["V2_short_gate_plus"],
    }
    variants = fast if variant_set == "fast" else full

    out = {}
    for name, overrides in variants.items():
        cfg = dict(base_cfg)
        cfg.update(overrides)
        out[name] = cfg
    return out


def parse_args():
    p = argparse.ArgumentParser(description="V9 最小实验矩阵（主线口径）")
    p.add_argument("--is-start", default=IS_START, help=f"IS 开始日期，默认 {IS_START}")
    p.add_argument("--is-end", default=IS_END, help=f"IS 结束日期，默认 {IS_END}")
    p.add_argument("--oos-start", default=OOS_START, help=f"OOS 开始日期，默认 {OOS_START}")
    p.add_argument("--oos-end", default=OOS_END, help=f"OOS 结束日期，默认 {OOS_END}")
    p.add_argument(
        "--variant-set",
        choices=["full", "fast"],
        default="full",
        help="变体集合：full=6个变体，fast=3个核心变体",
    )
    p.add_argument(
        "--min-oos-short-opens",
        type=int,
        default=10,
        help="OOS short 开仓覆盖率告警阈值，默认 10",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 100)
    print("V9 最小实验矩阵（主线口径）")
    print(
        f"PRIMARY={PRIMARY_TF} DECISION={DECISION_TFS} DATA=LOCAL_ONLY "
        f"VARIANTS={args.variant_set}"
    )
    print(f"IS={args.is_start}~{args.is_end} | OOS={args.oos_start}~{args.oos_end}")
    print("=" * 100)

    print("加载 IS 数据...")
    is_data, is_signals, needed_tfs, is_start, is_end = prepare_data(args.is_start, args.is_end)
    print("加载 OOS 数据...")
    oos_data, oos_signals, _, oos_start, oos_end = prepare_data(args.oos_start, args.oos_end)

    base_cfg = _build_default_config()
    variants = build_variants(base_cfg, variant_set=args.variant_set)

    rows = []
    for name, cfg in variants.items():
        print(f"\n[{name}]")
        is_r = run_once(is_data, is_signals, needed_tfs, cfg, is_start, is_end)
        oos_r = run_once(oos_data, oos_signals, needed_tfs, cfg, oos_start, oos_end)
        row = {
            "variant": name,
            "is_ret": is_r["ret"],
            "is_mdd": is_r["mdd"],
            "is_wr": is_r["wr"],
            "is_cpf_close_only": is_r["cpf_close_only"],
            "is_close_n": is_r["close_n"],
            "is_short_open_n": is_r["short_open_n"],
            "is_short_open_neutral": is_r["short_open_neutral"],
            "is_short_open_trend": is_r["short_open_trend"],
            "is_short_open_high_vol": is_r["short_open_high_vol"],
            "oos_ret": oos_r["ret"],
            "oos_mdd": oos_r["mdd"],
            "oos_wr": oos_r["wr"],
            "oos_cpf_close_only": oos_r["cpf_close_only"],
            "oos_close_n": oos_r["close_n"],
            "oos_short_open_n": oos_r["short_open_n"],
            "oos_short_open_neutral": oos_r["short_open_neutral"],
            "oos_short_open_trend": oos_r["short_open_trend"],
            "oos_short_open_high_vol": oos_r["short_open_high_vol"],
        }
        rows.append(row)
        print(
            f"IS ret={is_r['ret']:+.2f}% wr={is_r['wr']:.1f}% cpf={is_r['cpf_close_only']:.2f} "
            f"| OOS ret={oos_r['ret']:+.2f}% wr={oos_r['wr']:.1f}% cpf={oos_r['cpf_close_only']:.2f}"
        )

    df = pd.DataFrame(rows)
    base = df[df["variant"] == "V0_baseline"].iloc[0]
    df["is_dret"] = df["is_ret"] - float(base["is_ret"])
    df["oos_dret"] = df["oos_ret"] - float(base["oos_ret"])

    print("\n" + "=" * 100)
    print("汇总（差值=variant - V0_baseline）")
    print("=" * 100)
    cols = [
        "variant", "is_ret", "is_dret", "is_wr", "is_cpf_close_only",
        "oos_ret", "oos_dret", "oos_wr", "oos_cpf_close_only",
        "oos_short_open_n", "oos_short_open_neutral", "oos_short_open_trend", "oos_short_open_high_vol",
    ]
    print(df[cols].to_string(index=False))

    # 覆盖率告警：如果 OOS 样本里 short 太少，则 short 相关结论不可归因
    base_oos_short = int(base["oos_short_open_n"])
    if base_oos_short < args.min_oos_short_opens:
        print(
            "\n[WARN] OOS short 覆盖率不足: "
            f"baseline oos_short_open_n={base_oos_short} < {args.min_oos_short_opens}.\n"
            "       本次实验不适合评估 short 侧门控/止损/追踪优化。"
        )

    out_dir = os.path.join("data", "backtests", "v9_min_matrix")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(out_dir, f"v9_min_matrix_{ts}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n已保存: {out_csv}")


if __name__ == "__main__":
    main()
