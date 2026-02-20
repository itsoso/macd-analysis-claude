"""
ML 方向预测推理服务 — 部署在 GPU 机器，供 ECS 通过 HTTP 调用。

用法:
  python ml_inference_server.py [--host 0.0.0.0] [--port 5001]
  # 使用 GPU: 默认会检测 CUDA，有则用 cuda
  # 强制 CPU: ML_INFERENCE_DEVICE=cpu python ml_inference_server.py

API:
  POST /predict
  Body: { "sell_score": float, "buy_score": float, "features": { "columns", "index", "data" } }
  Response: { "success": true, "bull_prob": float, ... } 或 { "success": false, "error": "..." }
"""

import os
import sys
import json
import logging
import argparse

import pandas as pd

# 项目根为当前工作目录，保证 data/ml_models 可被找到
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# 序列长度：至少 96 以满足 TFT，48 满足 LSTM
FEATURE_MIN_ROWS = 96


def build_app(device: str = "cpu"):
    """创建 Flask 应用并注册路由（延迟导入 Flask，便于测试时 mock）"""
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    # 全局 enhancer，在首次 /predict 或启动时加载
    enhancer = None

    def get_enhancer():
        nonlocal enhancer
        if enhancer is None:
            from ml_live_integration import MLSignalEnhancer

            # GPU 机推理时优先 CUDA
            use_device = os.environ.get("ML_INFERENCE_DEVICE") or device
            env = os.environ.copy()
            if use_device == "cuda":
                env.setdefault("ML_INFERENCE_DEVICE", "cuda")
            e = MLSignalEnhancer()
            # 注入推理设备到内部（当前 MLSignalEnhancer 内部写死 cpu，这里仅做占位；实际 GPU 需改 ml_live_integration）
            e.load_model()
            enhancer = e
            log.info("MLSignalEnhancer 已加载 (device=%s)", use_device)
        return enhancer

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "service": "ml-inference"})

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            body = request.get_json(force=True, silent=False)
        except Exception as e:
            return jsonify({"success": False, "error": f"Invalid JSON: {e}"}), 400

        sell_score = body.get("sell_score", 0.0)
        buy_score = body.get("buy_score", 0.0)
        features_payload = body.get("features")
        if not features_payload or "data" not in features_payload:
            return jsonify({"success": False, "error": "Missing or invalid 'features' (need orient='split')"}), 400

        try:
            # 重建 DataFrame (pandas orient='split': columns, index, data)
            df = pd.DataFrame(
                features_payload["data"],
                index=features_payload.get("index"),
                columns=features_payload.get("columns", []),
            )
        except Exception as e:
            return jsonify({"success": False, "error": f"Failed to build DataFrame: {e}"}), 400

        if len(df) < 1:
            return jsonify({"success": False, "error": "features must have at least 1 row"}), 400

        try:
            en = get_enhancer()
            bull_prob, ml_info = en.predict_direction_from_features(df)
        except Exception as e:
            log.exception("Direction prediction failed")
            return jsonify({"success": False, "error": str(e)}), 500

        if bull_prob is None:
            return jsonify({
                "success": True,
                "bull_prob": 0.5,
                "fallback": "no_model",
                **{k: v for k, v in ml_info.items()},
            })

        out = {
            "success": True,
            "bull_prob": round(float(bull_prob), 4),
            **{k: v for k, v in ml_info.items()},
        }
        return jsonify(out)

    return app


def main():
    parser = argparse.ArgumentParser(description="ML direction inference API (GPU/CPU)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=5001, help="Bind port")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None,
                        help="Inference device (default: cuda if available else cpu)")
    args = parser.parse_args()

    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    if os.environ.get("ML_INFERENCE_DEVICE"):
        device = os.environ.get("ML_INFERENCE_DEVICE")
    log.info("Inference device: %s", device)

    app = build_app(device=device)
    log.info("Listening on %s:%s", args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
