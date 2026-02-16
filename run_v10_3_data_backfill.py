#!/usr/bin/env python3
"""v10.3 阶段A: 一键回补本地数据并输出覆盖报告

目标:
1) K线回补至 2021-01-01 (15m/1h/4h/24h)
2) Mark/Funding 回补至 2021-01-01
3) OI 仅拉取最近30天真实数据 (不做伪历史)
4) 输出数据覆盖报告 JSON
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import List

import pandas as pd

from binance_fetcher import (
    fetch_funding_rate_history,
    fetch_mark_price_klines,
    fetch_open_interest_history,
)
from kline_store import download_klines, load_klines, verify_completeness


def _dtstr(v):
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d %H:%M:%S")
    return str(v)


def _coverage(df: pd.DataFrame | None, label: str) -> dict:
    if df is None or len(df) == 0:
        return {
            "label": label,
            "rows": 0,
            "start": None,
            "end": None,
            "hours_median_gap": None,
            "hours_p95_gap": None,
        }
    gaps_h = df.index.to_series().diff().dropna().dt.total_seconds() / 3600.0
    return {
        "label": label,
        "rows": int(len(df)),
        "start": _dtstr(df.index[0]),
        "end": _dtstr(df.index[-1]),
        "hours_median_gap": float(gaps_h.median()) if len(gaps_h) else None,
        "hours_p95_gap": float(gaps_h.quantile(0.95)) if len(gaps_h) else None,
    }


def _kline_path(symbol: str, interval: str) -> str:
    return os.path.join("data", "klines", symbol.upper(), f"{interval}.parquet")


def _prepare_rebuild_if_needed(symbol: str, intervals: List[str], start: pd.Timestamp):
    for tf in intervals:
        path = _kline_path(symbol, tf)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_parquet(path)
            if df is None or len(df) == 0:
                continue
            local_start = pd.Timestamp(df.index[0]).tz_localize(None) if getattr(df.index, "tz", None) is not None else pd.Timestamp(df.index[0])
            if local_start > start:
                os.remove(path)
                print(f"  [重建] {tf}: 本地起点 {local_start} 晚于目标 {start.date()}，已删除旧文件")
        except Exception as e:
            print(f"  [重建] {tf}: 检查失败({e})，跳过删除")


def main():
    p = argparse.ArgumentParser(description="v10.3 数据回补 (K线+Mark+Funding+OI)")
    p.add_argument("--symbol", default="ETHUSDT", help="交易对 (默认 ETHUSDT)")
    p.add_argument("--start", default="2021-01-01", help="回补起始日")
    p.add_argument(
        "--end",
        default=pd.Timestamp.now().strftime("%Y-%m-%d"),
        help="回补结束日(默认今天)",
    )
    p.add_argument(
        "--intervals",
        default="15m,1h,4h,24h",
        help="K线周期，逗号分隔 (默认 15m,1h,4h,24h)",
    )
    p.add_argument(
        "--report-dir",
        default="logs/data_backfill",
        help="覆盖报告输出目录",
    )
    p.add_argument(
        "--no-force-rebuild-on-gap",
        action="store_true",
        default=False,
        help="关闭自动重建 (默认开启自动重建)",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        default=False,
        help="仅核查本地覆盖率，不触发任何下载/API回补",
    )
    args = p.parse_args()

    symbol = args.symbol.upper()
    intervals = [x.strip() for x in args.intervals.split(",") if x.strip()]
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    now = pd.Timestamp.now().tz_localize(None)
    days_full = max(1, int((now - start).days + 2))

    print("=" * 90)
    print(f"v10.3 数据回补 | {symbol} | {args.start} ~ {args.end}")
    if args.verify_only:
        print("模式: verify-only (仅本地核查)")
    print("=" * 90)

    # A1: K线
    print("\n[1/4] 本地K线处理...")
    if (not args.verify_only) and (not args.no_force_rebuild_on_gap):
        _prepare_rebuild_if_needed(symbol, intervals, start)
    if not args.verify_only:
        download_klines(symbol=symbol, intervals=intervals, start_date=args.start, end_date=args.end)
    verify_ok = verify_completeness(symbol=symbol, start_date=args.start, end_date=args.end)

    kline_cov = {}
    for tf in intervals:
        df = load_klines(
            symbol=symbol,
            interval=tf,
            start=args.start,
            end=args.end,
            with_indicators=False,
            allow_api_fallback=False,
        )
        kline_cov[tf] = _coverage(df, f"kline_{tf}")

    # A2: Mark/Funding
    print("\n[2/4] Mark K线处理...")
    mark_cov = {}
    for tf in ["15m", "1h", "4h"]:
        df_mark = fetch_mark_price_klines(
            symbol=symbol,
            interval=tf,
            days=days_full,
            force_api=not args.verify_only,
            allow_api_fallback=not args.verify_only,
        )
        mark_cov[tf] = _coverage(df_mark, f"mark_{tf}")

    print("\n[3/4] Funding 处理...")
    df_funding = fetch_funding_rate_history(
        symbol=symbol,
        days=days_full,
        force_api=not args.verify_only,
        allow_api_fallback=not args.verify_only,
    )
    funding_cov = _coverage(df_funding, "funding")

    # A3: OI only recent 30d (真实窗口)
    print("\n[4/4] OI 处理 (仅最近30天真实窗口)...")
    oi_cov = {}
    for tf in ["15m", "1h", "4h"]:
        df_oi = fetch_open_interest_history(
            symbol=symbol,
            interval=tf,
            days=30,
            force_api=not args.verify_only,
            allow_api_fallback=not args.verify_only,
        )
        one = _coverage(df_oi, f"oi_{tf}")
        one["binance_limit_note"] = "openInterestHist 仅支持最近30天；不做伪全历史补齐"
        oi_cov[tf] = one

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "symbol": symbol,
        "start": args.start,
        "end": args.end,
        "verify_ok": bool(verify_ok),
        "kline": kline_cov,
        "mark": mark_cov,
        "funding": funding_cov,
        "oi": oi_cov,
    }

    os.makedirs(args.report_dir, exist_ok=True)
    out = os.path.join(
        args.report_dir,
        f"v10_3_backfill_{symbol}_{args.start}_{args.end}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 90)
    print(f"回补完成，覆盖报告: {out}")
    print("=" * 90)


if __name__ == "__main__":
    main()
