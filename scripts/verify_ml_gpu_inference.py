#!/usr/bin/env python3
"""
方案二本地验证脚本：
  1. 启动推理服务（子进程或要求本机已起 ml_inference_server）
  2. 使用 ML_GPU_INFERENCE_URL 调用 enhance_signal，检查 ml_info 含 remote_inference

用法:
  # 终端1: 启动服务（CPU 即可）
  python ml_inference_server.py --port 5001

  # 终端2: 运行本脚本（会请求 http://127.0.0.1:5001）
  python scripts/verify_ml_gpu_inference.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 指向本地推理服务
os.environ["ML_GPU_INFERENCE_URL"] = "http://127.0.0.1:5001"


def main():
    import requests
    import pandas as pd
    import numpy as np

    url = os.environ.get("ML_GPU_INFERENCE_URL", "http://127.0.0.1:5001").rstrip("/") + "/predict"

    # 1. Health
    try:
        r = requests.get(url.replace("/predict", "/health"), timeout=2)
        print("GET /health:", r.status_code, r.json())
    except Exception as e:
        print("无法连接推理服务，请先启动: python ml_inference_server.py --port 5001")
        print("错误:", e)
        return 1

    # 2. 构造最小 features（96 行 × 若干列）
    n = 96
    cols = ["ret_1", "ret_2", "ret_5", "rsi6", "macd_bar", "hvol_20", "funding_rate"]
    if os.path.exists("data/ml_models/ensemble_config.json"):
        with open("data/ml_models/ensemble_config.json") as f:
            cfg = json.load(f)
        cols = cfg.get("feature_names", cols)[:20]
    features_df = pd.DataFrame(
        np.random.randn(n, len(cols)).astype(np.float32) * 0.01,
        columns=cols,
    )
    payload = {
        "sell_score": 50.0,
        "buy_score": 45.0,
        "features": json.loads(features_df.to_json(orient="split")),
    }

    r = requests.post(url, json=payload, timeout=5)
    print("POST /predict:", r.status_code)
    data = r.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))
    if not data.get("success"):
        print("远程推理返回 success=false，可能无模型或报错")
        return 0
    bull = data.get("bull_prob")
    print("bull_prob:", bull)
    if bull is not None:
        print("验证通过: 远程 API 返回 bull_prob")
    return 0


if __name__ == "__main__":
    sys.exit(main())
