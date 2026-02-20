#!/usr/bin/env python3
"""
Stacking 默认别名一致性校验/修复脚本。

目标：
  让 data/ml_models/stacking_meta.json(.pkl) 始终指向指定周期（默认 1h）。

用法：
  python scripts/sync_stacking_alias.py
  python scripts/sync_stacking_alias.py --tf 4h
  python scripts/sync_stacking_alias.py --check-only
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _model_dir() -> Path:
    return _root_dir() / "data" / "ml_models"


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_alias_consistent(model_dir: Path, target_tf: str):
    alias_json = model_dir / "stacking_meta.json"
    alias_pkl = model_dir / "stacking_meta.pkl"
    target_json = model_dir / f"stacking_meta_{target_tf}.json"
    target_pkl = model_dir / f"stacking_meta_{target_tf}.pkl"

    if not target_json.exists() or not target_pkl.exists():
        return False, f"缺少目标周期工件: {target_json.name} / {target_pkl.name}"
    if not alias_json.exists() or not alias_pkl.exists():
        return False, "默认别名缺失"

    try:
        alias_obj = _read_json(alias_json)
        target_obj = _read_json(target_json)
    except Exception as exc:
        return False, f"读取 JSON 失败: {exc}"

    alias_tf = str(alias_obj.get("timeframe", "")).strip()
    target_tf_in_json = str(target_obj.get("timeframe", "")).strip()
    if target_tf_in_json and target_tf_in_json != target_tf:
        return False, f"{target_json.name} 内 timeframe={target_tf_in_json} 与目标 {target_tf} 不一致"

    # 以关键质量字段判断是否与目标版本一致（比时间戳更稳妥）
    keys = ("timeframe", "val_auc", "test_auc", "oof_meta_auc", "n_oof_samples")
    alias_sig = tuple(alias_obj.get(k) for k in keys)
    target_sig = tuple(target_obj.get(k) for k in keys)
    if alias_tf != target_tf or alias_sig != target_sig:
        return False, "默认别名与目标周期工件不一致"

    alias_of = str(alias_obj.get("alias_of", "")).strip()
    if alias_of != target_json.name:
        return False, f"alias_of 不一致 ({alias_of or '缺失'} != {target_json.name})"

    alias_files = dict(alias_obj.get("model_files") or {})
    target_files = dict(target_obj.get("model_files") or {})
    alias_files["meta"] = "stacking_meta.pkl"
    target_files["meta"] = "stacking_meta.pkl"
    if alias_files != target_files:
        return False, "model_files 不一致，默认别名未完整对齐目标周期"

    if _sha256(alias_pkl) != _sha256(target_pkl):
        return False, "stacking_meta.pkl 与目标 pkl 内容不一致"

    return True, "默认别名已与目标周期一致"


def _backup_if_exists(path: Path):
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)


def _repair_alias(model_dir: Path, target_tf: str):
    alias_json = model_dir / "stacking_meta.json"
    alias_pkl = model_dir / "stacking_meta.pkl"
    target_json = model_dir / f"stacking_meta_{target_tf}.json"
    target_pkl = model_dir / f"stacking_meta_{target_tf}.pkl"

    if not target_json.exists() or not target_pkl.exists():
        raise FileNotFoundError(f"目标文件不存在: {target_json.name} / {target_pkl.name}")

    target_obj = _read_json(target_json)
    tf_in_json = str(target_obj.get("timeframe", "")).strip()
    if tf_in_json and tf_in_json != target_tf:
        raise ValueError(
            f"{target_json.name} 内 timeframe={tf_in_json} 与目标 {target_tf} 不一致，已拒绝修复"
        )

    # 先备份再写入，避免误覆盖
    _backup_if_exists(alias_json)
    _backup_if_exists(alias_pkl)

    alias_obj = dict(target_obj)
    alias_obj["timeframe"] = target_tf
    model_files = dict(alias_obj.get("model_files") or {})
    model_files["meta"] = "stacking_meta.pkl"
    alias_obj["model_files"] = model_files
    alias_obj["alias_of"] = target_json.name

    with alias_json.open("w", encoding="utf-8") as f:
        json.dump(alias_obj, f, ensure_ascii=False, indent=2)
    shutil.copy2(target_pkl, alias_pkl)


def main():
    parser = argparse.ArgumentParser(description="Stacking 默认别名一致性校验与自动修复")
    parser.add_argument(
        "--tf",
        default=(os.environ.get("ML_STACKING_TIMEFRAME") or "1h").strip(),
        help="目标周期，默认取 ML_STACKING_TIMEFRAME 或 1h",
    )
    parser.add_argument("--check-only", action="store_true", help="仅校验，不修复")
    args = parser.parse_args()

    target_tf = args.tf.strip()
    if not target_tf:
        print("[FAIL] 目标周期为空")
        return 2

    model_dir = _model_dir()
    if not model_dir.exists():
        print(f"[FAIL] 模型目录不存在: {model_dir}")
        return 2

    ok, msg = _is_alias_consistent(model_dir, target_tf)
    if ok:
        print(f"[OK] {msg} (target_tf={target_tf})")
        return 0

    print(f"[WARN] {msg} (target_tf={target_tf})")
    if args.check_only:
        print("[INFO] check-only 模式，不执行修复")
        return 1

    try:
        _repair_alias(model_dir, target_tf)
        ok2, msg2 = _is_alias_consistent(model_dir, target_tf)
        if ok2:
            print(f"[OK] 修复完成: {msg2} (target_tf={target_tf})")
            return 0
        print(f"[FAIL] 修复后仍不一致: {msg2}")
        return 1
    except Exception as exc:
        print(f"[FAIL] 修复失败: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
