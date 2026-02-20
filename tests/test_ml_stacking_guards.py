import json
import pickle

import numpy as np
import pandas as pd

from ml_live_integration import MLSignalEnhancer


class FakeMetaModel:
    def predict_proba(self, X):
        n = len(X)
        return np.tile(np.array([[0.4, 0.6]], dtype=np.float32), (n, 1))


def _write_stacking_artifacts(model_dir, name, cfg):
    meta_json = model_dir / f"{name}.json"
    meta_pkl = model_dir / f"{name}.pkl"
    with open(meta_json, "w") as f:
        json.dump(cfg, f)
    with open(meta_pkl, "wb") as f:
        pickle.dump(FakeMetaModel(), f)


def test_stacking_quality_gate_rejects_low_auc():
    enhancer = MLSignalEnhancer()
    ok, reason = enhancer._stacking_quality_gate(
        {
            "timeframe": "1h",
            "val_auc": 0.49,
            "test_auc": 0.55,
            "oof_meta_auc": 0.60,
        }
    )
    assert ok is False
    assert "val_auc_too_low" in reason


def test_load_model_skips_low_quality_stacking(tmp_path):
    cfg = {
        "timeframe": "1h",
        "val_auc": 0.49,
        "test_auc": 0.48,
        "oof_meta_auc": 0.70,
        "model_files": {
            "meta": "stacking_meta_1h.pkl",
        },
        "feature_names_73": ["ret_1", "ret_2"],
        "feature_names_94": ["ret_1", "ret_2", "btc_ret_1"],
    }
    _write_stacking_artifacts(tmp_path, "stacking_meta_1h", cfg)

    enhancer = MLSignalEnhancer(model_dir=str(tmp_path))
    enhancer.load_model()

    assert enhancer._stacking_meta_model is None
    assert enhancer._stacking_disabled_reason is not None
    assert "auc" in enhancer._stacking_disabled_reason or "overfit_gap" in enhancer._stacking_disabled_reason


def test_load_model_prefers_target_timeframe_stacking(monkeypatch, tmp_path):
    monkeypatch.setenv("ML_STACKING_TIMEFRAME", "1h")

    cfg_1h = {
        "timeframe": "1h",
        "val_auc": 0.56,
        "test_auc": 0.55,
        "oof_meta_auc": 0.58,
        "model_files": {"meta": "stacking_meta_1h.pkl"},
        "feature_names_73": ["ret_1", "ret_2"],
        "feature_names_94": ["ret_1", "ret_2", "btc_ret_1"],
    }
    cfg_default = {
        "timeframe": "4h",
        "val_auc": 0.58,
        "test_auc": 0.57,
        "oof_meta_auc": 0.59,
        "model_files": {"meta": "stacking_meta.pkl"},
        "feature_names_73": ["ret_1", "ret_2"],
        "feature_names_94": ["ret_1", "ret_2", "btc_ret_1"],
    }

    _write_stacking_artifacts(tmp_path, "stacking_meta_1h", cfg_1h)
    _write_stacking_artifacts(tmp_path, "stacking_meta", cfg_default)

    enhancer = MLSignalEnhancer(model_dir=str(tmp_path))
    enhancer.load_model()

    assert enhancer._stacking_meta_model is not None
    assert enhancer._stacking_config is not None
    assert enhancer._stacking_config.get("timeframe") == "1h"


def test_predict_stacking_from_features_skips_on_low_coverage():
    enhancer = MLSignalEnhancer()
    enhancer._stacking_meta_model = FakeMetaModel()
    enhancer._stacking_config = {
        "feature_names_73": ["f1", "f2", "f3", "f4"],
        "feature_names_94": ["f1", "f2", "f3", "f4", "f5"],
        "feat_mean_73": [0.0, 0.0, 0.0, 0.0],
        "feat_std_73": [1.0, 1.0, 1.0, 1.0],
        "feat_mean_94": [0.0, 0.0, 0.0, 0.0, 0.0],
        "feat_std_94": [1.0, 1.0, 1.0, 1.0, 1.0],
        "model_files": {},
        "extra_features": [],
    }

    # 仅 1 列，故覆盖率远低于默认阈值
    features = pd.DataFrame(np.random.randn(120, 1), columns=["f1"])
    ml_info = {}
    bull_prob = enhancer._predict_stacking_from_features(features, ml_info)

    assert bull_prob is None
    assert "stacking_skipped_reason" in ml_info


def test_can_use_cross_asset_reports_low_coverage():
    enhancer = MLSignalEnhancer()
    enhancer._cross_asset_model = object()
    enhancer._cross_asset_meta = {"feature_names": ["a", "b", "c", "d", "e"]}

    features = pd.DataFrame(np.random.randn(50, 2), columns=["a", "b"])
    ml_info = {}
    ok = enhancer._can_use_cross_asset(features, ml_info)

    assert ok is False
    assert "ca_feature_coverage" in ml_info
    assert "ca_skipped_reason" in ml_info
