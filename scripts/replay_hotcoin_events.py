#!/usr/bin/env python3
"""
HotCoin 事件日志回放工具。

示例:
  python3 scripts/replay_hotcoin_events.py
  python3 scripts/replay_hotcoin_events.py --symbol ETHUSDT --event-type order_result --limit 50
"""

import argparse
import json
import os
from collections import Counter
from datetime import datetime


def _fmt_ts(ts):
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"


def _iter_events(path):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            obj["_line"] = i
            yield obj


def main():
    parser = argparse.ArgumentParser(description="HotCoin 事件日志回放")
    parser.add_argument("--file", default="hotcoin/data/hotcoin_events.jsonl", help="事件日志文件")
    parser.add_argument("--event-type", default="", help="事件类型过滤，逗号分隔")
    parser.add_argument("--symbol", default="", help="交易对过滤，如 ETHUSDT")
    parser.add_argument("--limit", type=int, default=100, help="输出事件条数")
    parser.add_argument("--summary-only", action="store_true", help="仅输出摘要")
    args = parser.parse_args()

    path = args.file
    if not os.path.exists(path):
        print(f"[ERROR] 文件不存在: {path}")
        raise SystemExit(1)

    event_types = {e.strip() for e in args.event_type.split(",") if e.strip()}
    symbol = args.symbol.strip().upper()

    rows = []
    for ev in _iter_events(path):
        et = str(ev.get("event_type", ""))
        payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
        sym = str(payload.get("symbol", "")).upper()

        if event_types and et not in event_types:
            continue
        if symbol and sym != symbol:
            continue
        rows.append(ev)

    if not rows:
        print("[INFO] 无匹配事件")
        return

    type_counter = Counter(str(r.get("event_type", "")) for r in rows)
    symbol_counter = Counter(str((r.get("payload") or {}).get("symbol", "-")) for r in rows)

    print(f"[INFO] 匹配事件: {len(rows)}")
    print("[INFO] 按类型统计:")
    for k, v in type_counter.most_common():
        print(f"  - {k}: {v}")

    print("[INFO] 按币种统计(top10):")
    for k, v in symbol_counter.most_common(10):
        print(f"  - {k}: {v}")

    if args.summary_only:
        return

    tail = rows[-max(1, args.limit):]
    print(f"[INFO] 最近 {len(tail)} 条:")
    for ev in tail:
        ts = _fmt_ts(ev.get("ts"))
        et = ev.get("event_type", "")
        payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
        sym = payload.get("symbol", "-")
        intent = payload.get("intent", "-")
        ok = payload.get("ok")
        line = ev.get("_line")
        print(f"  [{line}] {ts} {et} symbol={sym} intent={intent} ok={ok}")


if __name__ == "__main__":
    main()
