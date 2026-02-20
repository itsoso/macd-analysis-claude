import importlib.util
import json
from pathlib import Path


def _load_sync_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "sync_stacking_alias.py"
    spec = importlib.util.spec_from_file_location("sync_stacking_alias", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_alias_consistency_true(tmp_path):
    mod = _load_sync_module()
    model_dir = tmp_path

    target_json = model_dir / "stacking_meta_1h.json"
    target_pkl = model_dir / "stacking_meta_1h.pkl"
    alias_json = model_dir / "stacking_meta.json"
    alias_pkl = model_dir / "stacking_meta.pkl"

    target_payload = {
        "timeframe": "1h",
        "val_auc": 0.56,
        "test_auc": 0.54,
        "oof_meta_auc": 0.58,
        "n_oof_samples": 24000,
        "model_files": {
            "lgb": "stacking_lgb_1h.txt",
            "meta": "stacking_meta_1h.pkl",
        },
    }

    _write_json(target_json, target_payload)
    target_pkl.write_bytes(b"target-pkl-content")

    alias_payload = dict(target_payload)
    alias_payload["model_files"] = {
        "lgb": "stacking_lgb_1h.txt",
        "meta": "stacking_meta.pkl",
    }
    alias_payload["alias_of"] = "stacking_meta_1h.json"
    _write_json(alias_json, alias_payload)
    alias_pkl.write_bytes(b"target-pkl-content")

    ok, msg = mod._is_alias_consistent(model_dir, "1h")
    assert ok is True, msg


def test_alias_repair_from_mismatch(tmp_path):
    mod = _load_sync_module()
    model_dir = tmp_path

    target_json = model_dir / "stacking_meta_4h.json"
    target_pkl = model_dir / "stacking_meta_4h.pkl"
    alias_json = model_dir / "stacking_meta.json"
    alias_pkl = model_dir / "stacking_meta.pkl"

    target_payload = {
        "timeframe": "4h",
        "val_auc": 0.57,
        "test_auc": 0.55,
        "oof_meta_auc": 0.59,
        "n_oof_samples": 26000,
        "model_files": {
            "lgb": "stacking_lgb_4h.txt",
            "meta": "stacking_meta_4h.pkl",
        },
    }
    _write_json(target_json, target_payload)
    target_pkl.write_bytes(b"target-4h-pkl")

    # 构造错误别名：timeframe、alias_of、pkl 都错
    bad_alias_payload = {
        "timeframe": "1h",
        "val_auc": 0.50,
        "test_auc": 0.50,
        "oof_meta_auc": 0.50,
        "n_oof_samples": 10000,
        "model_files": {"lgb": "stacking_lgb_1h.txt", "meta": "stacking_meta.pkl"},
        "alias_of": "stacking_meta_1h.json",
    }
    _write_json(alias_json, bad_alias_payload)
    alias_pkl.write_bytes(b"bad-pkl")

    ok, _ = mod._is_alias_consistent(model_dir, "4h")
    assert ok is False

    mod._repair_alias(model_dir, "4h")
    ok2, msg2 = mod._is_alias_consistent(model_dir, "4h")
    assert ok2 is True, msg2

    repaired = json.loads(alias_json.read_text(encoding="utf-8"))
    assert repaired["timeframe"] == "4h"
    assert repaired["alias_of"] == "stacking_meta_4h.json"
    assert repaired["model_files"]["meta"] == "stacking_meta.pkl"
