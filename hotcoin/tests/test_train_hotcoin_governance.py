import json

from hotcoin.ml.train_hotcoin import _write_governance_artifacts


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def test_governance_artifacts_production(tmp_path):
    model_path = tmp_path / "hotness_lgb_15m.txt"
    meta_path = tmp_path / "hotness_lgb_15m_meta.json"
    model_path.write_text("dummy-model", encoding="utf-8")
    _write_json(
        meta_path,
        {
            "n_samples": 25000,
            "n_features": 12,
            "test_auc": 0.61,
            "feature_names": ["f1", "f2"],
        },
    )

    _write_governance_artifacts(
        model_path=str(model_path),
        meta_path=str(meta_path),
        task="hotness",
        interval="15m",
    )

    contract_path = tmp_path / "runtime_contract_hotness_15m.json"
    decision_path = tmp_path / "promotion_decision_hotness_15m.json"
    assert contract_path.exists()
    assert decision_path.exists()

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))

    assert contract["metrics"]["n_samples"] == 25000
    assert contract["metrics"]["test_auc"] == 0.61
    assert decision["approved"] is True
    assert decision["deployment_tier"] == "production"
    assert "all_thresholds_passed" in decision["reasons"]


def test_governance_artifacts_research_only(tmp_path):
    model_path = tmp_path / "trade_lgb_15m.txt"
    meta_path = tmp_path / "trade_lgb_15m_meta.json"
    model_path.write_text("dummy-model", encoding="utf-8")
    _write_json(
        meta_path,
        {
            "n_samples": 8000,
            "n_features": 9,
            "test_auc": 0.52,
            "feature_names": ["a", "b", "c"],
        },
    )

    _write_governance_artifacts(
        model_path=str(model_path),
        meta_path=str(meta_path),
        task="trade",
        interval="15m",
    )

    decision_path = tmp_path / "promotion_decision_trade_15m.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["approved"] is False
    assert decision["deployment_tier"] == "research_only"
    assert "n_samples<20000" in decision["reasons"]
    assert "test_auc<0.55" in decision["reasons"]
