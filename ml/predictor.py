"""
推理接口 — 集成到实盘系统

提供 predict(df, data_all, tf) → buy_prob, sell_prob 接口,
可作为第 7 维信号融入 signal_core.py 的融合评分。
"""

import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ml.config import (
    LOOKBACK, D_MODEL, N_HEADS, D_FF,
    NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DROPOUT, STATIC_DIM,
    FORECAST_HORIZON, ENSEMBLE_TFT_WEIGHT, ENSEMBLE_LGB_WEIGHT,
    DATA_DIR, CHECKPOINT_DIR,
)


class MLPredictor:
    """ML 预测器, 封装 TFT + LightGBM 集成推理"""

    def __init__(self, tf: str = "1h", use_tft: bool = True, use_lgb: bool = True):
        self.tf = tf
        self.use_tft = use_tft
        self.use_lgb = use_lgb
        self._tft_model = None
        self._lgb_model = None
        self._stats = None
        self._meta = None
        self._loaded = False

    def load(self):
        """延迟加载模型"""
        if self._loaded:
            return

        meta_path = os.path.join(DATA_DIR, f"meta_{self.tf}.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"元数据不存在: {meta_path}. 请先运行 ml.prepare_data")
        self._meta = json.load(open(meta_path))

        stats_path = os.path.join(DATA_DIR, f"stats_{self.tf}.parquet")
        if os.path.exists(stats_path):
            self._stats = pd.read_parquet(stats_path)

        # TFT
        if self.use_tft:
            try:
                import torch
                from ml.models import TemporalFusionTransformer

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"tft_best_{self.tf}.pt")
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

                model = TemporalFusionTransformer(
                    n_time_features=ckpt["n_features"],
                    n_static=STATIC_DIM,
                    d_model=D_MODEL,
                    n_heads=N_HEADS,
                    d_ff=D_FF,
                    num_encoder_layers=NUM_ENCODER_LAYERS,
                    num_decoder_layers=NUM_DECODER_LAYERS,
                    dropout=DROPOUT,
                    forecast_horizon=FORECAST_HORIZON,
                ).to(device)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                self._tft_model = model
                self._tft_device = device
                self._tft_feat_cols = ckpt["feat_cols"]
                print(f"[MLPredictor] TFT 已加载 ({device})")
            except Exception as e:
                print(f"[MLPredictor] TFT 加载失败: {e}")
                self.use_tft = False

        # LightGBM
        if self.use_lgb:
            try:
                import lightgbm as lgb
                model_path = os.path.join(CHECKPOINT_DIR, f"lgb_{self.tf}.txt")
                self._lgb_model = lgb.Booster(model_file=model_path)
                print(f"[MLPredictor] LightGBM 已加载")
            except Exception as e:
                print(f"[MLPredictor] LightGBM 加载失败: {e}")
                self.use_lgb = False

        self._loaded = True

    def _prepare_features(self, df, data_all):
        """从 raw DataFrame 提取并标准化特征 (复用 prepare_data 逻辑)"""
        from ml.prepare_data import extract_features, align_cross_tf_features, compute_regime

        features_all = {}
        for t in [self.tf] + [t for t in data_all.keys() if t != self.tf]:
            if t in data_all:
                from ml.prepare_data import extract_features as ef
                features_all[t] = ef({**data_all, self.tf: df}, t)

        main_feats = features_all[self.tf]

        # 跨周期
        cross_feats = align_cross_tf_features(features_all, self.tf)
        result = main_feats.join(cross_feats, how="left")
        result["regime"] = compute_regime(df)

        # 标准化
        if self._stats is not None:
            mean = self._stats["mean"]
            std = self._stats["std"].replace(0, 1)
            cols = [c for c in result.columns if c in mean.index]
            result[cols] = (result[cols] - mean[cols]) / std[cols]

        return result

    def predict(self, df, data_all=None, tf=None):
        """
        预测买卖概率。

        参数:
            df: 主周期 DataFrame (已含指标, 来自 add_all_indicators)
            data_all: 多周期数据 dict (可选, 用于跨周期特征)
            tf: 周期 (可选, 默认用初始化时指定的)

        返回:
            dict: {
                "buy_prob": float (0-1),
                "sell_prob": float (0-1),
                "hold_prob": float (0-1),
                "reg_pred": float (预测收益率),
                "confidence": float (最大概率),
                "source": str ("tft+lgb" / "tft" / "lgb"),
            }
        """
        self.load()

        if data_all is None:
            data_all = {self.tf: df}

        features = self._prepare_features(df, data_all)
        feat_cols = self._meta["feat_cols"]
        time_feat_cols = [c for c in feat_cols if c != "regime"]

        result = {
            "buy_prob": 0.5,
            "sell_prob": 0.5,
            "hold_prob": 0.0,
            "reg_pred": 0.0,
            "confidence": 0.0,
            "source": "none",
        }

        tft_probs = None
        lgb_probs = None

        # TFT 推理
        if self.use_tft and self._tft_model is not None and len(features) >= LOOKBACK:
            import torch
            # 取最后 LOOKBACK 根
            window = features.iloc[-LOOKBACK:]
            x_cols = [c for c in self._tft_feat_cols if c in window.columns]
            x_time = torch.tensor(
                window[x_cols].values.astype(np.float32)
            ).unsqueeze(0).to(self._tft_device)

            # regime static
            regime_val = int(features["regime"].iloc[-1]) if "regime" in features.columns else 1
            x_static = torch.zeros(1, STATIC_DIM, device=self._tft_device)
            x_static[0, regime_val] = 1.0

            with torch.no_grad():
                cls_logits, reg_pred, attn_w, var_w = self._tft_model(x_time, x_static)
            tft_probs = torch.softmax(cls_logits, dim=-1).cpu().numpy()[0]
            result["reg_pred"] = float(reg_pred[0, 0].cpu())

        # LightGBM 推理
        if self.use_lgb and self._lgb_model is not None:
            avail_cols = [c for c in feat_cols if c in features.columns]
            x = features[avail_cols].iloc[-1:].values
            x = np.nan_to_num(x, nan=0.0)
            lgb_probs = self._lgb_model.predict(x)[0]

        # 集成
        if tft_probs is not None and lgb_probs is not None:
            probs = ENSEMBLE_TFT_WEIGHT * tft_probs + ENSEMBLE_LGB_WEIGHT * lgb_probs
            result["source"] = "tft+lgb"
        elif tft_probs is not None:
            probs = tft_probs
            result["source"] = "tft"
        elif lgb_probs is not None:
            probs = lgb_probs
            result["source"] = "lgb"
        else:
            return result

        # probs: [short_prob, hold_prob, long_prob]
        result["sell_prob"] = float(probs[0])
        result["hold_prob"] = float(probs[1])
        result["buy_prob"] = float(probs[2])
        result["confidence"] = float(probs.max())

        return result


# 全局单例 (懒加载)
_predictor = None


def get_predictor(tf: str = "1h") -> MLPredictor:
    global _predictor
    if _predictor is None or _predictor.tf != tf:
        _predictor = MLPredictor(tf=tf)
    return _predictor


def predict(df, data_all=None, tf="1h"):
    """便捷接口: predict(df) → dict"""
    return get_predictor(tf).predict(df, data_all, tf)
