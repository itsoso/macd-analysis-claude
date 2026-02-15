#!/usr/bin/env python3
"""两阶段扫描: Neutral 结构折扣参数 + neutral 做空门槛交互

阶段1:
- 扫描 neutral_struct_discount_0 / neutral_struct_discount_1
- 固定 neutral_struct_activity_thr 与 regime_short_threshold

阶段2:
- 取阶段1 Top-N 组合
- 扫描 neutral_struct_activity_thr × regime_short_threshold
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class RunResult:
    stage: str
    run_id: int
    ret: float
    alpha: float
    mdd: float
    ppf: float
    cpf: float
    winrate: float
    trades: int
    d0: float
    d1: float
    act: float
    nst: float
    notes: str
    elapsed_sec: float

    @property
    def score(self) -> float:
        # 轻量综合评分: 以组合PF为核心，兼顾收益与回撤
        dd_penalty = max(0.0, abs(self.mdd) - 15.0) * 0.03
        return self.ppf + (self.ret / 250.0) - dd_penalty


def _extract(pattern: str, text: str, default: float = 0.0) -> float:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default


def _run_backtest(
    py: str,
    start: str,
    end: str,
    notes: str,
    overrides: dict[str, str | float | int | bool],
    timeout_sec: int,
) -> RunResult | None:
    cmd = [
        py,
        "backtest_multi_tf_daily.py",
        "--start",
        start,
        "--end",
        end,
        "--notes",
        notes,
    ]
    for k, v in overrides.items():
        vv = str(v).lower() if isinstance(v, bool) else str(v)
        cmd.extend(["--override", f"{k}={vv}"])

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return None
    elapsed = time.time() - t0
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")

    run_id = int(_extract(r"run_id=(\d+)", text, 0))
    ret = _extract(r"策略收益:\s+([+\-\d.]+)%", text, 0)
    alpha = _extract(r"Alpha:\s+([+\-\d.]+)%", text, 0)
    mdd = _extract(r"最大回撤:\s+([+\-\d.]+)%", text, 0)
    ppf = _extract(r"组合PF:\s+([\d.]+)", text, 0)
    cpf = _extract(r"合约PF:\s+([\d.]+)", text, 0)
    wr = _extract(r"胜率:\s+([\d.]+)%", text, 0)
    trades = int(_extract(r"交易次数:\s+(\d+)", text, 0))

    if run_id == 0:
        print("  WARN: 未解析到 run_id，可能回测失败")
        return None

    return RunResult(
        stage="",
        run_id=run_id,
        ret=ret,
        alpha=alpha,
        mdd=mdd,
        ppf=ppf,
        cpf=cpf,
        winrate=wr,
        trades=trades,
        d0=float(overrides["neutral_struct_discount_0"]),
        d1=float(overrides["neutral_struct_discount_1"]),
        act=float(overrides["neutral_struct_activity_thr"]),
        nst=float(
            str(overrides["regime_short_threshold"]).split(":")[-1]
            if isinstance(overrides["regime_short_threshold"], str)
            else overrides["regime_short_threshold"]
        ),
        notes=notes,
        elapsed_sec=elapsed,
    )


def _save_csv(path: Path, rows: Iterable[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "stage",
                "run_id",
                "ret",
                "alpha",
                "mdd",
                "ppf",
                "cpf",
                "winrate",
                "trades",
                "d0",
                "d1",
                "activity_thr",
                "neutral_st",
                "score",
                "elapsed_sec",
                "notes",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.stage,
                    r.run_id,
                    r.ret,
                    r.alpha,
                    r.mdd,
                    r.ppf,
                    r.cpf,
                    r.winrate,
                    r.trades,
                    r.d0,
                    r.d1,
                    r.act,
                    r.nst,
                    round(r.score, 6),
                    round(r.elapsed_sec, 2),
                    r.notes,
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="结构折扣两阶段扫描")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-01-31")
    parser.add_argument("--oos-start", default="")
    parser.add_argument("--oos-end", default="")
    parser.add_argument("--topn", type=int, default=3)
    parser.add_argument("--timeout-sec", type=int, default=480)
    parser.add_argument("--quick", action="store_true", help="快速模式，缩小参数网格")
    parser.add_argument("--output", default="logs/sweeps/structural_discount_sweep.csv")
    args = parser.parse_args()

    py = sys.executable
    base_overrides = {
        "use_neutral_structural_discount": True,
        "use_neutral_book_consensus": False,
        "use_short_adverse_exit": False,
        "use_extreme_divergence_short_veto": False,
    }

    if args.quick:
        d0_list = [0.10, 0.15, 0.20]
        d1_list = [0.20, 0.25, 0.35]
        act_list = [8, 10]
        st_list = [45, 50]
    else:
        d0_list = [0.05, 0.10, 0.15, 0.20, 0.30]
        d1_list = [0.15, 0.25, 0.35, 0.50]
        act_list = [5, 8, 10, 15]
        st_list = [40, 45, 50, 55, 60]

    all_rows: list[RunResult] = []

    print("\n=== Stage1: 扫 d0/d1 (固定 activity=10, neutral_st=45) ===")
    stage1_rows: list[RunResult] = []
    stage1_combos = list(itertools.product(d0_list, d1_list))
    for i, (d0, d1) in enumerate(stage1_combos, 1):
        if d1 < d0:
            continue
        ov = dict(base_overrides)
        ov.update(
            {
                "neutral_struct_discount_0": d0,
                "neutral_struct_discount_1": d1,
                "neutral_struct_discount_2": 1.0,
                "neutral_struct_discount_3": 1.0,
                "neutral_struct_discount_4plus": 1.0,
                "neutral_struct_activity_thr": 10,
                "regime_short_threshold": "neutral:45",
            }
        )
        notes = f"struct_stage1 d0={d0} d1={d1}"
        print(f"[S1 {i}/{len(stage1_combos)}] d0={d0:.2f} d1={d1:.2f} ...", end=" ", flush=True)
        rr = _run_backtest(py, args.start, args.end, notes, ov, args.timeout_sec)
        if rr is None:
            print("failed")
            continue
        rr.stage = "S1"
        stage1_rows.append(rr)
        all_rows.append(rr)
        print(f"run#{rr.run_id} ret={rr.ret:+.2f}% pPF={rr.ppf:.2f} MDD={rr.mdd:.2f}%")

    if not stage1_rows:
        print("Stage1 无有效结果，退出")
        return 1

    stage1_top = sorted(stage1_rows, key=lambda x: x.score, reverse=True)[: max(1, args.topn)]
    print("\nStage1 Top:")
    for idx, r in enumerate(stage1_top, 1):
        print(
            f"{idx}. run#{r.run_id} d0={r.d0:.2f} d1={r.d1:.2f} "
            f"ret={r.ret:+.2f}% pPF={r.ppf:.2f} MDD={r.mdd:.2f}% score={r.score:.3f}"
        )

    print("\n=== Stage2: Top-N × activity_thr × neutral_st ===")
    stage2_rows: list[RunResult] = []
    for t_idx, top in enumerate(stage1_top, 1):
        combos2 = list(itertools.product(act_list, st_list))
        for j, (act, st) in enumerate(combos2, 1):
            ov = dict(base_overrides)
            ov.update(
                {
                    "neutral_struct_discount_0": top.d0,
                    "neutral_struct_discount_1": top.d1,
                    "neutral_struct_discount_2": 1.0,
                    "neutral_struct_discount_3": 1.0,
                    "neutral_struct_discount_4plus": 1.0,
                    "neutral_struct_activity_thr": act,
                    "regime_short_threshold": f"neutral:{st}",
                }
            )
            notes = f"struct_stage2 top{t_idx} d0={top.d0} d1={top.d1} act={act} st={st}"
            print(
                f"[S2 top{t_idx} {j}/{len(combos2)}] d0={top.d0:.2f} d1={top.d1:.2f} act={act} st={st} ...",
                end=" ",
                flush=True,
            )
            rr = _run_backtest(py, args.start, args.end, notes, ov, args.timeout_sec)
            if rr is None:
                print("failed")
                continue
            rr.stage = "S2"
            stage2_rows.append(rr)
            all_rows.append(rr)
            print(f"run#{rr.run_id} ret={rr.ret:+.2f}% pPF={rr.ppf:.2f} MDD={rr.mdd:.2f}%")

    final_rank = sorted(all_rows, key=lambda x: x.score, reverse=True)
    print("\n=== FINAL TOP 12 ===")
    for i, r in enumerate(final_rank[:12], 1):
        print(
            f"{i:2d}. {r.stage} run#{r.run_id} d0={r.d0:.2f} d1={r.d1:.2f} "
            f"act={r.act:.0f} st={r.nst:.0f} ret={r.ret:+.2f}% pPF={r.ppf:.2f} "
            f"MDD={r.mdd:.2f}% score={r.score:.3f}"
        )

    # 可选 OOS 验证（只对最终 Top3）
    if args.oos_start and args.oos_end and final_rank:
        print("\n=== OOS Verify (Top3) ===")
        for r in final_rank[:3]:
            ov = dict(base_overrides)
            ov.update(
                {
                    "neutral_struct_discount_0": r.d0,
                    "neutral_struct_discount_1": r.d1,
                    "neutral_struct_discount_2": 1.0,
                    "neutral_struct_discount_3": 1.0,
                    "neutral_struct_discount_4plus": 1.0,
                    "neutral_struct_activity_thr": r.act,
                    "regime_short_threshold": f"neutral:{int(r.nst)}",
                }
            )
            notes = (
                f"struct_oos_from_run{r.run_id} "
                f"d0={r.d0} d1={r.d1} act={r.act} st={int(r.nst)}"
            )
            print(f"[OOS] ref run#{r.run_id} ...", end=" ", flush=True)
            rr = _run_backtest(py, args.oos_start, args.oos_end, notes, ov, args.timeout_sec)
            if rr is None:
                print("failed")
                continue
            rr.stage = "OOS"
            all_rows.append(rr)
            print(f"run#{rr.run_id} ret={rr.ret:+.2f}% pPF={rr.ppf:.2f} MDD={rr.mdd:.2f}%")

    out_path = Path(args.output)
    _save_csv(out_path, all_rows)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

