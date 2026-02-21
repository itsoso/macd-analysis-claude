"""
在线学习 — 增量模型更新

策略:
  1. 收集实盘交易数据 → 本地缓冲
  2. 缓冲满 N 条后触发增量训练
  3. LightGBM 支持 init_model 继续训练
  4. 新模型 AUC 优于旧模型才替换

Phase 4 功能, 前置条件: Phase 3 模型已上线。
"""

import json
import logging
import os
import time
from typing import List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("hotcoin.online")


class OnlineLearner:
    """增量模型更新器。"""

    def __init__(self, model_dir: str, task: str = "trade",
                 buffer_size: int = 500, min_improve_auc: float = 0.005):
        self.model_dir = model_dir
        self.task = task
        self.buffer_size = buffer_size
        self.min_improve_auc = min_improve_auc
        self._buffer: List[dict] = []
        self._current_model = None
        self._current_auc = 0.0

    def add_sample(self, features: dict, label: int, symbol: str = ""):
        """添加一个样本到缓冲区。"""
        self._buffer.append({
            "features": features,
            "label": label,
            "symbol": symbol,
            "ts": time.time(),
        })
        if len(self._buffer) >= self.buffer_size:
            self._trigger_update()
        if len(self._buffer) > self.buffer_size * 3:
            self._buffer = self._buffer[-self.buffer_size:]

    def _trigger_update(self):
        """触发增量训练。"""
        log.info("在线学习: 缓冲区满 (%d 样本), 触发增量更新", len(self._buffer))

        try:
            import lightgbm as lgb
            from sklearn.metrics import roc_auc_score

            all_keys = set()
            for s in self._buffer:
                all_keys.update(s["features"].keys())
            feature_names = sorted(all_keys)
            if not feature_names:
                log.warning("在线学习: 无有效特征, 跳过")
                return

            X = np.array([[s["features"].get(f, 0) for f in feature_names] for s in self._buffer])
            y = np.array([s["label"] for s in self._buffer])

            if len(set(y)) < 2:
                log.warning("在线学习: 标签单一 (全为 %d), 保留 buffer 等待更多样本", int(y[0]))
                return

            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            # 加载现有模型作为 init_model
            model_path = os.path.join(self.model_dir, f"{self.task}_lgb_15m.txt")
            init_model = model_path if os.path.exists(model_path) else None

            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            params = {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.01,  # 较低的学习率用于微调
                "num_leaves": 31,
                "verbose": -1,
            }

            model = lgb.train(
                params, dtrain,
                num_boost_round=100,
                valid_sets=[dval],
                init_model=init_model,
                callbacks=[lgb.early_stopping(20)],
            )

            pred_val = model.predict(X_val)
            new_auc = roc_auc_score(y_val, pred_val) if len(set(y_val)) > 1 else 0.5

            if new_auc > self._current_auc + self.min_improve_auc:
                old_auc = self._current_auc
                model.save_model(model_path)
                self._current_auc = new_auc
                log.info("模型更新成功: AUC %.4f → %.4f", old_auc, new_auc)
            else:
                log.info("模型未更新 (AUC %.4f, 未达改善门槛 +%.3f)",
                         new_auc, self.min_improve_auc)

            self._buffer = []

        except ImportError:
            log.warning("lightgbm 未安装, 跳过在线学习")
        except Exception:
            log.exception("在线学习失败, 保留 %d 条 buffer 数据", len(self._buffer))

    def get_status(self) -> dict:
        return {
            "buffer_size": len(self._buffer),
            "buffer_capacity": self.buffer_size,
            "current_auc": self._current_auc,
            "task": self.task,
        }
