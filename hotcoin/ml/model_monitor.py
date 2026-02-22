"""
模型性能监控 — 追踪 ML 预测准确率 + 漂移检测

功能:
    1. 记录每次 ML 预测 (prob, actual_label)
    2. 滚动计算 AUC / accuracy / calibration
    3. 检测性能漂移 (当前窗口 vs 历史基线)
    4. 输出监控指标供 dashboard 展示

用法:
    from hotcoin.ml.model_monitor import get_monitor
    monitor = get_monitor()
    monitor.record("trade", symbol, prob=0.7, label=1)
    metrics = monitor.get_metrics("trade")
"""

import logging
import time
from collections import deque
from typing import Dict, Optional

import numpy as np

log = logging.getLogger("hotcoin.monitor")


class ModelMonitor:
    """ML 模型性能监控器。"""

    def __init__(self, window_size: int = 500, drift_threshold: float = 0.05):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        # task -> deque of {prob, label, ts, symbol}
        self._records: Dict[str, deque] = {}
        self._baseline_auc: Dict[str, float] = {}

    def record(self, task: str, symbol: str, prob: float, label: int):
        """记录一次预测结果。"""
        if task not in self._records:
            self._records[task] = deque(maxlen=self.window_size * 2)

        self._records[task].append({
            "prob": prob,
            "label": label,
            "symbol": symbol,
            "ts": time.time(),
        })

    def get_metrics(self, task: str) -> dict:
        """计算指定任务的滚动性能指标。"""
        records = self._records.get(task)
        if not records or len(records) < 10:
            return {"status": "insufficient_data", "n_records": len(records) if records else 0}

        recent = list(records)[-self.window_size:]
        probs = np.array([r["prob"] for r in recent])
        labels = np.array([r["label"] for r in recent])

        # 基础指标
        pred_labels = (probs >= 0.5).astype(int)
        accuracy = float(np.mean(pred_labels == labels))
        pos_rate = float(np.mean(labels))
        pred_pos_rate = float(np.mean(pred_labels))

        # AUC (简化计算)
        auc = self._compute_auc(probs, labels)

        # 校准度: 预测概率 vs 实际正例率 (分 5 桶)
        calibration = self._compute_calibration(probs, labels)

        # 漂移检测
        drift = self._detect_drift(task, auc)

        return {
            "task": task,
            "n_records": len(recent),
            "accuracy": round(accuracy, 4),
            "auc": round(auc, 4),
            "pos_rate": round(pos_rate, 4),
            "pred_pos_rate": round(pred_pos_rate, 4),
            "calibration": calibration,
            "drift_detected": drift["detected"],
            "drift_delta": drift["delta"],
            "last_update": recent[-1]["ts"] if recent else 0,
        }

    def get_all_metrics(self) -> Dict[str, dict]:
        """获取所有任务的指标。"""
        return {task: self.get_metrics(task) for task in self._records}

    def set_baseline(self, task: str, auc: float):
        """设置基线 AUC (来自离线训练)。"""
        self._baseline_auc[task] = auc
        log.info("设置 %s 基线 AUC: %.4f", task, auc)

    def _compute_auc(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """简化 AUC 计算 (不依赖 sklearn)。"""
        if len(set(labels)) < 2:
            return 0.5

        pos_probs = probs[labels == 1]
        neg_probs = probs[labels == 0]

        if len(pos_probs) == 0 or len(neg_probs) == 0:
            return 0.5

        # Mann-Whitney U statistic
        n_pos = len(pos_probs)
        n_neg = len(neg_probs)
        concordant = 0
        for p in pos_probs:
            concordant += np.sum(p > neg_probs) + 0.5 * np.sum(p == neg_probs)

        return float(concordant / (n_pos * n_neg))

    def _compute_calibration(self, probs: np.ndarray, labels: np.ndarray) -> list:
        """分桶校准度。"""
        bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
        result = []
        for lo, hi in bins:
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() > 0:
                avg_prob = float(np.mean(probs[mask]))
                avg_label = float(np.mean(labels[mask]))
                result.append({
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "count": int(mask.sum()),
                    "avg_prob": round(avg_prob, 3),
                    "avg_label": round(avg_label, 3),
                    "gap": round(abs(avg_prob - avg_label), 3),
                })
        return result

    def _detect_drift(self, task: str, current_auc: float) -> dict:
        """检测性能漂移。"""
        baseline = self._baseline_auc.get(task)
        if baseline is None:
            return {"detected": False, "delta": 0.0}

        delta = current_auc - baseline
        detected = delta < -self.drift_threshold

        if detected:
            log.warning("模型漂移检测: %s AUC %.4f → %.4f (Δ=%.4f, 阈值=%.4f)",
                        task, baseline, current_auc, delta, self.drift_threshold)

        return {"detected": detected, "delta": round(delta, 4)}


# 全局单例
_monitor: Optional[ModelMonitor] = None


def get_monitor() -> ModelMonitor:
    """获取全局 ModelMonitor 单例。"""
    global _monitor
    if _monitor is None:
        _monitor = ModelMonitor()
    return _monitor
