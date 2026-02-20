import numpy as np
import pandas as pd

import train_gpu


def _mock_features(n_rows: int, n_cols: int = 10):
    idx = pd.date_range("2026-01-01", periods=n_rows, freq="1h")
    features = pd.DataFrame(
        np.random.randn(n_rows, n_cols),
        index=idx,
        columns=[f"f{i}" for i in range(n_cols)],
    )
    labels = pd.DataFrame(
        {
            "profitable_long_5": np.random.randint(0, 2, size=n_rows).astype(float),
            "fwd_dir_5": np.random.randint(0, 2, size=n_rows).astype(float),
        },
        index=idx,
    )
    return features, labels


def test_stacking_skips_when_samples_below_threshold(monkeypatch):
    monkeypatch.setattr(train_gpu, "prepare_features", lambda symbol, tf: _mock_features(1200))
    monkeypatch.setattr(train_gpu, "_add_cross_asset_features", lambda features, tf: features)

    result = train_gpu.train_stacking_ensemble(["1h"], min_samples=20000)
    one_h = result["1h"]

    assert one_h["skipped"] is True
    assert "insufficient_samples" in one_h["skip_reason"]
    assert one_h["n_samples"] == 1200


def test_tabnet_skips_when_samples_below_threshold(monkeypatch):
    monkeypatch.setattr(train_gpu, "prepare_features", lambda symbol, tf: _mock_features(800))
    monkeypatch.setattr(train_gpu, "_add_cross_asset_features", lambda features, tf: features)

    result = train_gpu.train_tabnet(["1h"], min_samples=10000)
    one_h = result["1h"]

    assert one_h["skipped"] is True
    assert "insufficient_samples" in one_h["skip_reason"]
    assert one_h["n_samples"] == 800
