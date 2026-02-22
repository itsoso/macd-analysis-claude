"""
热点币 ML 推理模块

加载训练好的 hotness/trade/pump 模型, 提供实时推理接口。
所有模型默认 shadow 模式 (只记录不影响决策)。

用法:
    from hotcoin.ml.predictor_hot import HotcoinPredictor
    pred = HotcoinPredictor()
    prob = pred.predict_hotness(features_df)
    prob = pred.predict_trade(features_df)
"""

import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("hotcoin.predictor")

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class HotcoinPredictor:
    """热点币 ML 推理器 — 懒加载模型。"""

    def __init__(self, model_dir: str = _MODEL_DIR):
        self.model_dir = model_dir
        self._hotness_model = None
        self._trade_model = None
        self._hotness_meta = None
        self._trade_meta = None

    def _load_model(self, task: str, interval: str = "15m"):
        """懒加载 LightGBM 模型。"""
        model_path = os.path.join(self.model_dir, f"{task}_lgb_{interval}.txt")
        meta_path = model_path.replace(".txt", "_meta.json")

        if not os.path.exists(model_path):
            return None, None

        try:
            import lightgbm as lgb
            model = lgb.Booster(model_file=model_path)
            meta = None
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            log.info("模型已加载: %s (%d features)",
                     model_path, model.num_feature())
            return model, meta
        except Exception as e:
            log.warning("模型加载失败 %s: %s", model_path, e)
            return None, None

    def predict_hotness(self, features: pd.DataFrame, interval: str = "15m") -> Optional[float]:
        """
        预测热度概率 (0-1)。

        Parameters
        ----------
        features : 单行或多行特征 DataFrame (来自 compute_hot_features)

        Returns
        -------
        float or None: 最后一行的热度概率, 模型不可用时返回 None
        """
        if self._hotness_model is None:
            self._hotness_model, self._hotness_meta = self._load_model("hotness", interval)
        if self._hotness_model is None:
            return None

        try:
            X = features.iloc[[-1]].values
            X = np.nan_to_num(X, nan=0.0)

            # 特征数对齐
            n_feat = self._hotness_model.num_feature()
            if X.shape[1] > n_feat:
                X = X[:, :n_feat]
            elif X.shape[1] < n_feat:
                X = np.hstack([X, np.zeros((X.shape[0], n_feat - X.shape[1]))])

            prob = float(self._hotness_model.predict(X)[0])
            return prob
        except Exception as e:
            log.warning("hotness 推理失败: %s", e)
            return None

    def predict_trade(self, features: pd.DataFrame, interval: str = "15m") -> Optional[float]:
        """
        预测交易方向概率 (0-1, >0.5 看多)。

        Returns
        -------
        float or None: 最后一行的交易概率
        """
        if self._trade_model is None:
            self._trade_model, self._trade_meta = self._load_model("trade", interval)
        if self._trade_model is None:
            return None

        try:
            X = features.iloc[[-1]].values
            X = np.nan_to_num(X, nan=0.0)

            n_feat = self._trade_model.num_feature()
            if X.shape[1] > n_feat:
                X = X[:, :n_feat]
            elif X.shape[1] < n_feat:
                X = np.hstack([X, np.zeros((X.shape[0], n_feat - X.shape[1]))])

            prob = float(self._trade_model.predict(X)[0])
            return prob
        except Exception as e:
            log.warning("trade 推理失败: %s", e)
            return None


# 全局单例
_predictor: Optional[HotcoinPredictor] = None


def get_predictor() -> HotcoinPredictor:
    """获取全局 HotcoinPredictor 单例。"""
    global _predictor
    if _predictor is None:
        _predictor = HotcoinPredictor()
    return _predictor
