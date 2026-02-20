import importlib.util
import json
import sys
from pathlib import Path


def _load_mod():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "build_h800_summary.py"
    module_name = "build_h800_summary"
    spec = importlib.util.spec_from_file_location(module_name, script)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_pick_best_result_file_prefers_target_tf(tmp_path):
    mod = _load_mod()
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_json(a, {"15m": {"avg_val_auc": 0.5}})
    _write_json(b, {"1h": {"avg_val_auc": 0.6}})

    # 确保 a 更新，但函数仍应优先选含 1h 的 b
    a.touch()
    picked = mod._pick_best_result_file([a, b], "1h")
    assert picked.name == "b.json"


def test_evaluate_stacking_gate_blocked_reason():
    mod = _load_mod()
    gate = mod.StackingGate(
        min_val_auc=0.53,
        min_test_auc=0.52,
        min_oof_auc=0.53,
        max_oof_test_gap=0.10,
        min_samples=20000,
    )
    status, reasons = mod._evaluate_stacking_gate(
        {
            "val_auc": 0.50,
            "test_auc": 0.49,
            "oof_meta_auc": 0.62,
            "n_oof_samples": 5000,
        },
        gate,
    )
    assert status == "BLOCKED"
    assert "val_auc" in reasons
    assert "test_auc" in reasons
    assert "n_oof_samples" in reasons
    assert "overfit_gap" in reasons
