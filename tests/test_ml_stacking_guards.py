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


def test_stacking_quality_gate_relative_uses_oof_auc():
    """相对门控改为用 oof_meta_auc（真正 OOF，~24k 样本）而非 val_auc（in-sample，偏高）。
    oof_meta_auc=0.58 > lgb(0.56)-margin(0.01)=0.55 → 应通过；val_auc/test_auc 低不影响相对门控。"""
    enhancer = MLSignalEnhancer()
    enhancer._direction_meta = {"test_auc": 0.56}
    ok, reason = enhancer._stacking_quality_gate(
        {
            "timeframe": "1h",
            "val_auc": 0.57,
            "test_auc": 0.54,
            "oof_meta_auc": 0.58,  # 真正 OOF，高于 LGB-0.01=0.55 → 应通过
            "base_models": ["lgb", "xgboost", "lstm", "tft", "cross_asset_lgb"],
            "extra_features": ["hvol_20"],
            "meta_input_dim": 6,
        }
    )
    assert ok is True, f"oof_meta_auc(0.58)>=lgb(0.56)-margin(0.01) 应通过，reason={reason}"


def test_stacking_quality_gate_relative_blocks_low_oof_auc():
    """oof_meta_auc 低于 LGB-margin 时，相对门控应拒绝（val_auc 高不影响结论）。"""
    enhancer = MLSignalEnhancer()
    enhancer._direction_meta = {"test_auc": 0.56}
    ok, reason = enhancer._stacking_quality_gate(
        {
            "timeframe": "1h",
            "val_auc": 0.83,      # in-sample 偏高，不再参与相对门控
            "test_auc": 0.57,
            "oof_meta_auc": 0.54,  # 真 OOF 低于 LGB(0.56)-0.01=0.55 → 应被拒绝
            "base_models": ["lgb", "xgboost", "lstm", "tft", "cross_asset_lgb"],
            "extra_features": ["hvol_20"],
            "meta_input_dim": 6,
        }
    )
    assert ok is False
    assert "stacking_underperforms_lgb" in reason


def test_stacking_quality_gate_rejects_insufficient_oof_samples():
    enhancer = MLSignalEnhancer()
    enhancer.stacking_min_oof_samples = 20000
    ok, reason = enhancer._stacking_quality_gate(
        {
            "timeframe": "1h",
            "val_auc": 0.57,
            "test_auc": 0.56,
            "oof_meta_auc": 0.58,
            "n_oof_samples": 4200,
            "base_models": ["lgb", "xgboost", "lstm", "tft", "cross_asset_lgb"],
            "extra_features": ["hvol_20"],
            "meta_input_dim": 6,
        }
    )
    assert ok is False
    assert "insufficient_oof_samples" in reason


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


def test_stacking_runs_with_partial_94_coverage():
    """73维覆盖率良好但 94维覆盖率低时，Stacking 应继续运行（不因跨资产缺失而整体中止）。"""
    enhancer = MLSignalEnhancer()
    enhancer._stacking_meta_model = FakeMetaModel()
    enhancer._stacking_config = {
        "feature_names_73": ["f1", "f2", "f3", "f4"],
        "feature_names_94": ["f1", "f2", "f3", "f4", "ca1", "ca2", "ca3"],  # 3 cross-asset missing
        "feat_mean_73": [0.0, 0.0, 0.0, 0.0],
        "feat_std_73": [1.0, 1.0, 1.0, 1.0],
        "feat_mean_94": [0.0] * 7,
        "feat_std_94": [1.0] * 7,
        "model_files": {},
        "extra_features": [],
    }

    # 全部 73维特征都在，跨资产 (ca1/ca2/ca3) 缺失 → 94维覆盖率 4/7 ≈ 0.57 < 0.80
    features = pd.DataFrame(np.random.randn(120, 4), columns=["f1", "f2", "f3", "f4"])
    ml_info = {}
    bull_prob = enhancer._predict_stacking_from_features(features, ml_info)

    # 73维覆盖率 1.0 ≥ 0.90 → 不应跳过 Stacking
    assert "stacking_skipped_reason" not in ml_info, f"不应跳过: {ml_info}"
    # 94维低覆盖率应记录到 stacking_partial_94_coverage
    assert "stacking_partial_94_coverage" in ml_info
    # FakeMetaModel 返回 [[0.4, 0.6]], 结果应为有效值
    assert bull_prob is not None


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


def test_compute_direction_features_augments_cross_asset(monkeypatch):
    enhancer = MLSignalEnhancer()

    def fake_compute_ml_features(df):
        return pd.DataFrame(
            {
                "ret_1": np.linspace(0.0, 0.01, len(df)),
                "ret_5": np.linspace(0.0, 0.02, len(df)),
                "hvol_20": np.full(len(df), 0.01),
            },
            index=df.index,
        )

    monkeypatch.setattr("ml_features.compute_ml_features", fake_compute_ml_features)
    monkeypatch.setattr(enhancer, "_infer_interval_from_df", lambda _df: "1h")

    idx = pd.date_range("2026-01-01", periods=120, freq="1h")

    def fake_load_cross_close(symbol, interval):
        _ = interval
        base = {"BTCUSDT": 100.0, "SOLUSDT": 50.0, "BNBUSDT": 70.0}[symbol]
        return pd.Series(base + np.arange(len(idx)) * 0.1, index=idx)

    monkeypatch.setattr(enhancer, "_load_cross_close_series", fake_load_cross_close)

    df = pd.DataFrame(
        {
            "open": np.ones(len(idx)),
            "high": np.ones(len(idx)),
            "low": np.ones(len(idx)),
            "close": np.ones(len(idx)),
            "volume": np.ones(len(idx)),
        },
        index=idx,
    )
    feat = enhancer._compute_direction_features(df)
    assert feat is not None
    for col in [
        "btc_ret_1", "btc_eth_corr_20", "btc_rel_strength", "btc_vol_ratio",
        "sol_ret_1", "sol_eth_corr_20", "sol_rel_strength", "sol_vol_ratio",
        "bnb_ret_1", "bnb_eth_corr_20", "bnb_rel_strength", "bnb_vol_ratio",
    ]:
        assert col in feat.columns


def test_iter_stacking_candidates_discovers_timeframe_files(tmp_path):
    (tmp_path / "stacking_meta_1h.json").write_text("{}")
    (tmp_path / "stacking_meta_4h.json").write_text("{}")
    (tmp_path / "stacking_meta.json").write_text("{}")
    enhancer = MLSignalEnhancer(model_dir=str(tmp_path), stacking_timeframe="1h")
    cands = list(enhancer._iter_stacking_candidates())
    assert cands[0] == ("stacking_meta_1h.json", "stacking_meta_1h.pkl")
    assert ("stacking_meta_4h.json", "stacking_meta_4h.pkl") in cands
    assert ("stacking_meta.json", "stacking_meta.pkl") == cands[-1]


def test_sanitize_neural_prob_skips_extreme_by_default():
    enhancer = MLSignalEnhancer()
    ml_info = {}
    out = enhancer._sanitize_neural_prob(0.0, "tft", ml_info)
    assert out is None
    assert "tft_skipped_reason" in ml_info


def test_sanitize_neural_prob_keeps_when_drop_disabled(monkeypatch):
    monkeypatch.setenv("ML_DROP_EXTREME_NEURAL_PROBS", "0")
    enhancer = MLSignalEnhancer()
    ml_info = {}
    out = enhancer._sanitize_neural_prob(0.01, "lstm", ml_info)
    assert out is not None
