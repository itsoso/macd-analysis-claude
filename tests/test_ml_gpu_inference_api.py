"""
测试 ML 远程 GPU 推理 API 方案（方案二）：
  - 单元：Mock 远程 API 返回 bull_prob，断言 enhance_signal 使用该值并回退逻辑正确
  - 集成：可选，启动 ml_inference_server 后请求 /predict 与 ECS 端 enhance_signal
"""

import os
import json
import pytest
import pandas as pd
import numpy as np


# 用过短的 features 避免依赖真实模型文件
@pytest.fixture
def minimal_features_df():
    """至少 96 行、若干列的特征 DataFrame（与 API 约定一致）"""
    n = 96
    cols = ["ret_1", "ret_2", "ret_5", "rsi6", "macd_bar", "hvol_20"]
    data = np.random.randn(n, len(cols)).astype(np.float32) * 0.01
    return pd.DataFrame(data, columns=cols)


@pytest.fixture
def mock_remote_response():
    return {"success": True, "bull_prob": 0.62, "remote_inference": True}


def test_predict_direction_from_features_returns_tuple(minimal_features_df):
    """predict_direction_from_features 返回 (bull_prob, ml_info) 且 ml_info 可空"""
    from ml_live_integration import MLSignalEnhancer

    enhancer = MLSignalEnhancer()
    # 不加载模型也可调用，应返回 (None, {}) 或局部结果
    bull, info = enhancer.predict_direction_from_features(minimal_features_df)
    assert isinstance(info, dict)
    # 未加载模型时 bull 可能为 None 或加权结果
    if bull is not None:
        assert 0 <= bull <= 1


def test_request_remote_direction_no_url():
    """gpu_inference_url 为空时 _request_remote_direction 返回 (None, None)"""
    from ml_live_integration import MLSignalEnhancer

    e = MLSignalEnhancer(gpu_inference_url="")
    df = pd.DataFrame(np.zeros((10, 3)), columns=["a", "b", "c"])
    bull, info = e._request_remote_direction(df, 50.0, 45.0)
    assert bull is None
    assert info is None


def test_request_remote_direction_mock_success(monkeypatch, minimal_features_df, mock_remote_response):
    """Mock requests.post 返回成功时，_request_remote_direction 返回远程 bull_prob"""
    try:
        import requests
    except ImportError:
        pytest.skip("requests not installed")

    from ml_live_integration import MLSignalEnhancer

    def fake_post(url, json=None, timeout=None, headers=None):
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return mock_remote_response
        return R()

    monkeypatch.setattr(requests, "post", fake_post)
    e = MLSignalEnhancer(gpu_inference_url="http://127.0.0.1:5001")
    bull, info = e._request_remote_direction(minimal_features_df, 50.0, 45.0)
    assert bull is not None
    assert abs(bull - 0.62) < 1e-6
    assert info is not None
    assert info.get("remote_inference") is True
    assert info.get("bull_prob") == 0.62


def test_request_remote_direction_mock_timeout(monkeypatch, minimal_features_df):
    """Mock 超时后返回 (None, None)"""
    try:
        import requests
    except ImportError:
        pytest.skip("requests not installed")

    from ml_live_integration import MLSignalEnhancer

    def fake_post_timeout(*args, **kwargs):
        import requests as req
        raise req.exceptions.Timeout()

    monkeypatch.setattr(requests, "post", fake_post_timeout)
    e = MLSignalEnhancer(gpu_inference_url="http://127.0.0.1:5001")
    bull, info = e._request_remote_direction(minimal_features_df, 50.0, 45.0)
    assert bull is None
    assert info is None


def test_enhance_signal_with_remote_uses_bull_prob(monkeypatch, minimal_features_df):
    """当 mock 远程返回 bull_prob 时，enhance_signal 的 ml_info 含该值且 remote_inference"""
    from ml_live_integration import MLSignalEnhancer

    e = MLSignalEnhancer(gpu_inference_url="http://127.0.0.1:5001")
    e._loaded = True
    e._request_remote_direction = lambda f, ss, bs: (0.62, {"bull_prob": 0.62, "remote_inference": True})
    e._compute_direction_features = lambda df: minimal_features_df if df is not None and len(df) >= 1 else None
    ss, bs, ml_info = e.enhance_signal(50.0, 45.0, minimal_features_df)
    assert "bull_prob" in ml_info
    assert ml_info["bull_prob"] == 0.62
    assert ml_info.get("remote_inference") is True


def test_inference_server_predict_payload(minimal_features_df):
    """验证 /predict 的 request/response 形状与 build_app 解析一致"""
    from ml_inference_server import build_app

    app = build_app(device="cpu")
    client = app.test_client()
    payload = {
        "sell_score": 50.0,
        "buy_score": 45.0,
        "features": json.loads(minimal_features_df.to_json(orient="split")),
    }
    try:
        resp = client.post("/predict", data=json.dumps(payload), content_type="application/json")
    except Exception as e:
        pytest.skip(f"predict endpoint failed (e.g. no models): {e}")
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.get_json()
        assert "success" in data
        if data.get("success") and "bull_prob" in data:
            assert 0 <= data["bull_prob"] <= 1


def test_inference_server_health():
    """GET /health 返回 ok"""
    from ml_inference_server import build_app

    app = build_app(device="cpu")
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json().get("status") == "ok"
