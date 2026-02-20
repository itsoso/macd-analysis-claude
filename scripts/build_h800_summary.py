#!/usr/bin/env python3
"""
生成 H800 训练汇总（JSON + Markdown）。

输入:
  - data/ml_models/stacking_meta*.json
  - data/gpu_results/*.json
输出:
  - <output_dir>/h800_training_summary_latest.json
  - <output_dir>/h800_training_summary_latest.md
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class StackingGate:
    min_val_auc: float
    min_test_auc: float
    min_oof_auc: float
    max_oof_test_gap: float
    min_samples: int


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _latest_file(pattern: str) -> Optional[Path]:
    files = [Path(p) for p in glob(pattern)]
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def _safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _safe_int(v) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _evaluate_stacking_gate(meta: Dict, gate: StackingGate) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    val_auc = _safe_float(meta.get("val_auc"))
    test_auc = _safe_float(meta.get("test_auc"))
    oof_auc = _safe_float(meta.get("oof_meta_auc"))
    n_samples = _safe_int(meta.get("n_oof_samples"))

    if val_auc is None or val_auc < gate.min_val_auc:
        reasons.append("val_auc")
    if test_auc is None or test_auc < gate.min_test_auc:
        reasons.append("test_auc")
    if oof_auc is None or oof_auc < gate.min_oof_auc:
        reasons.append("oof_auc")
    if n_samples is None or n_samples < gate.min_samples:
        reasons.append("n_oof_samples")
    if oof_auc is not None and test_auc is not None and (oof_auc - test_auc) > gate.max_oof_test_gap:
        reasons.append("overfit_gap")

    return ("PASS" if not reasons else "BLOCKED"), reasons


def _collect_stacking(model_dir: Path, gate: StackingGate) -> List[Dict]:
    metas = sorted(model_dir.glob("stacking_meta*.json"))
    out: List[Dict] = []
    for p in metas:
        try:
            data = _read_json(p)
        except Exception as e:
            out.append({"file": p.name, "status": "READ_ERROR", "error": str(e)})
            continue
        status, reasons = _evaluate_stacking_gate(data, gate)
        val_auc = _safe_float(data.get("val_auc"))
        test_auc = _safe_float(data.get("test_auc"))
        oof_auc = _safe_float(data.get("oof_meta_auc"))
        gap = None
        if oof_auc is not None and test_auc is not None:
            gap = oof_auc - test_auc
        out.append(
            {
                "file": p.name,
                "timeframe": data.get("timeframe"),
                "val_auc": val_auc,
                "test_auc": test_auc,
                "oof_meta_auc": oof_auc,
                "n_oof_samples": _safe_int(data.get("n_oof_samples")),
                "gap_oof_test": gap,
                "status": status,
                "reasons": reasons,
                "alias_of": data.get("alias_of"),
            }
        )
    return out


def _pick_best_result_file(candidates: List[Path], target_tf: str) -> Path:
    """优先挑选包含 target_tf 键的最新文件，否则回退最新文件。"""
    if not candidates:
        raise ValueError("no candidates")
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        try:
            data = _read_json(p)
            if isinstance(data, dict) and target_tf in data:
                return p
        except Exception:
            continue
    return candidates[0]


def _collect_latest_gpu_results(gpu_dir: Path, target_tf: str) -> Dict:
    patterns = {
        "lgb_walkforward": "lgb_walkforward_*.json",
        "lstm_training": "lstm_training_*.json",
        "tft_training": "tft_training_*.json",
        "cross_asset_training": "cross_asset_training_*.json",
        "stacking_ensemble": "stacking_ensemble_*.json",
        "onnx_export": "onnx_export_*.json",
        "tabnet_training": "tabnet_training_*.json",
    }
    result: Dict[str, Dict] = {}
    for name, pat in patterns.items():
        files = [Path(p) for p in glob(str(gpu_dir / pat))]
        if not files:
            continue
        p = _pick_best_result_file(files, target_tf)
        item: Dict = {"file": p.name}
        try:
            item["content"] = _read_json(p)
        except Exception as e:
            item["read_error"] = str(e)
        result[name] = item
    return result


def _alias_info(model_dir: Path) -> Dict:
    alias = model_dir / "stacking_meta.json"
    if not alias.exists():
        return {"exists": False}
    try:
        data = _read_json(alias)
    except Exception as e:
        return {"exists": True, "read_error": str(e)}
    return {
        "exists": True,
        "timeframe": data.get("timeframe"),
        "alias_of": data.get("alias_of"),
        "model_files": data.get("model_files", {}),
    }


def _load_gate_from_env() -> StackingGate:
    return StackingGate(
        min_val_auc=float(os.environ.get("STACKING_MIN_VAL_AUC", os.environ.get("ML_STACKING_MIN_VAL_AUC", "0.53"))),
        min_test_auc=float(os.environ.get("STACKING_MIN_TEST_AUC", os.environ.get("ML_STACKING_MIN_TEST_AUC", "0.52"))),
        min_oof_auc=float(os.environ.get("STACKING_MIN_OOF_AUC", os.environ.get("ML_STACKING_MIN_OOF_AUC", "0.53"))),
        max_oof_test_gap=float(
            os.environ.get("STACKING_MAX_OOF_TEST_GAP", os.environ.get("ML_STACKING_MAX_OOF_TEST_GAP", "0.10"))
        ),
        min_samples=int(os.environ.get("MIN_STACKING_SAMPLES", os.environ.get("ML_STACKING_MIN_OOF_SAMPLES", "20000"))),
    )


def _build_markdown(summary: Dict) -> str:
    lines: List[str] = []
    lines.append("# H800 Nightly Training Summary")
    lines.append("")
    lines.append(f"- generated_at: `{summary['generated_at']}`")
    lines.append(f"- target_tf: `{summary['target_timeframe']}`")
    lines.append(f"- alias: `tf={summary['alias'].get('timeframe')}` `alias_of={summary['alias'].get('alias_of')}`")
    lines.append("")
    gate = summary["stacking_gate"]
    lines.append("## Stacking Gate")
    lines.append("")
    lines.append(
        "- min_val_auc={:.4f}, min_test_auc={:.4f}, min_oof_auc={:.4f}, max_gap={:.4f}, min_samples={}".format(
            gate["min_val_auc"],
            gate["min_test_auc"],
            gate["min_oof_auc"],
            gate["max_oof_test_gap"],
            gate["min_samples"],
        )
    )
    lines.append("")
    lines.append("## Stacking Status")
    lines.append("")
    lines.append("| file | tf | val | test | oof | n_oof | gap | status | reasons |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for item in summary["stacking"]:
        lines.append(
            "| {file} | {tf} | {va} | {ta} | {oa} | {ns} | {gap} | {st} | {rs} |".format(
                file=item.get("file", "-"),
                tf=item.get("timeframe", "-"),
                va=item.get("val_auc", "-"),
                ta=item.get("test_auc", "-"),
                oa=item.get("oof_meta_auc", "-"),
                ns=item.get("n_oof_samples", "-"),
                gap=item.get("gap_oof_test", "-"),
                st=item.get("status", "-"),
                rs=",".join(item.get("reasons", [])),
            )
        )
    lines.append("")
    lines.append("## Promotion Suggestion")
    lines.append("")
    lines.append(f"- production_ready_tfs: `{summary['promotion']['production_ready_tfs']}`")
    lines.append(f"- blocked_tfs: `{summary['promotion']['blocked_tfs']}`")
    lines.append("")
    lines.append("## Latest Result Files")
    lines.append("")
    for k, v in summary["latest_results"].items():
        lines.append(f"- {k}: `{v.get('file')}`")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Build H800 summary json+md")
    parser.add_argument("--output-dir", default="logs/retrain", help="summary output directory")
    parser.add_argument("--target-tf", default=(os.environ.get("ML_STACKING_TIMEFRAME") or "1h").strip())
    parser.add_argument("--model-dir", default="data/ml_models")
    parser.add_argument("--gpu-results-dir", default="data/gpu_results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    gpu_dir = Path(args.gpu_results_dir)

    gate = _load_gate_from_env()
    stacking_items = _collect_stacking(model_dir, gate)
    latest_results = _collect_latest_gpu_results(gpu_dir, args.target_tf)
    alias = _alias_info(model_dir)

    prod_tfs = []
    blocked_tfs = []
    for it in stacking_items:
        tf = it.get("timeframe")
        if not tf:
            continue
        if it.get("status") == "PASS":
            prod_tfs.append(tf)
        elif it.get("status") == "BLOCKED":
            blocked_tfs.append(tf)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_timeframe": args.target_tf,
        "stacking_gate": {
            "min_val_auc": gate.min_val_auc,
            "min_test_auc": gate.min_test_auc,
            "min_oof_auc": gate.min_oof_auc,
            "max_oof_test_gap": gate.max_oof_test_gap,
            "min_samples": gate.min_samples,
        },
        "alias": alias,
        "stacking": stacking_items,
        "latest_results": latest_results,
        "promotion": {
            "production_ready_tfs": sorted(set(prod_tfs)),
            "blocked_tfs": sorted(set(blocked_tfs)),
        },
    }

    json_out = out_dir / "h800_training_summary_latest.json"
    md_out = out_dir / "h800_training_summary_latest.md"
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with md_out.open("w", encoding="utf-8") as f:
        f.write(_build_markdown(summary))

    print(f"[OK] {json_out}")
    print(f"[OK] {md_out}")
    print(
        "[INFO] promotion production_ready_tfs={} blocked_tfs={}".format(
            summary["promotion"]["production_ready_tfs"], summary["promotion"]["blocked_tfs"]
        )
    )


if __name__ == "__main__":
    main()
