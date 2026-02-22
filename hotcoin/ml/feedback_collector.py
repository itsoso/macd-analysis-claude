"""
信号反馈收集器 — 追踪交易信号结果, 喂给在线学习

流程:
    1. signal_worker 产出 TradeSignal → feedback_collector.record_signal()
    2. 定时检查信号结果 (N 分钟后价格变化)
    3. 生成标签: 信号后 30min 最大收益 >= 3% → label=1
    4. 喂给 OnlineLearner.add_sample()

用法:
    from hotcoin.ml.feedback_collector import get_feedback_collector
    collector = get_feedback_collector()
    collector.record_signal(symbol, features_dict, signal_action, price_at_signal)
    collector.check_outcomes()  # 定时调用
"""

import logging
import os
import time
from typing import Dict, List, Optional

log = logging.getLogger("hotcoin.feedback")

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class FeedbackCollector:
    """信号反馈收集器。"""

    def __init__(
        self,
        model_dir: str = _MODEL_DIR,
        outcome_window_sec: int = 1800,  # 30 分钟
        min_return_for_success: float = 0.03,  # 3%
    ):
        self.model_dir = model_dir
        self.outcome_window = outcome_window_sec
        self.min_return = min_return_for_success
        self._pending: List[dict] = []  # 等待结果的信号
        self._learner = None
        self._max_pending = 1000

    def _get_learner(self):
        """懒加载 OnlineLearner。"""
        if self._learner is None:
            from hotcoin.ml.online_learning import OnlineLearner
            self._learner = OnlineLearner(
                model_dir=self.model_dir,
                task="trade",
                buffer_size=200,
                min_improve_auc=0.005,
            )
        return self._learner

    def record_signal(
        self,
        symbol: str,
        features: dict,
        action: str,
        price_at_signal: float,
        ml_trade_prob: float = -1.0,
    ):
        """
        记录一个交易信号, 等待后续结果。

        Parameters
        ----------
        symbol : 交易对
        features : 特征字典 (来自 compute_hot_features)
        action : "BUY" / "SELL" / "HOLD"
        price_at_signal : 信号时的价格
        ml_trade_prob : ML 预测概率 (-1 = 不可用)
        """
        if price_at_signal <= 0:
            return

        self._pending.append({
            "symbol": symbol,
            "features": features,
            "action": action,
            "price": price_at_signal,
            "ml_prob": ml_trade_prob,
            "ts": time.time(),
        })

        # 防止内存泄漏
        if len(self._pending) > self._max_pending:
            self._pending = self._pending[-self._max_pending:]

        log.debug("记录信号: %s %s @ %.4f (pending=%d)",
                  symbol, action, price_at_signal, len(self._pending))

    def check_outcomes(self, price_getter=None):
        """
        检查待定信号的结果。

        Parameters
        ----------
        price_getter : callable(symbol) -> float, 获取当前价格
        """
        if not self._pending or price_getter is None:
            return

        now = time.time()
        resolved = []
        remaining = []

        for sig in self._pending:
            elapsed = now - sig["ts"]

            # 还没到检查时间
            if elapsed < self.outcome_window:
                remaining.append(sig)
                continue

            # 超时太久, 丢弃
            if elapsed > self.outcome_window * 3:
                continue

            # 获取当前价格
            try:
                current_price = price_getter(sig["symbol"])
                if current_price is None or current_price <= 0:
                    remaining.append(sig)
                    continue

                ret = (current_price / sig["price"]) - 1
                label = 1 if ret >= self.min_return else 0

                resolved.append({
                    "features": sig["features"],
                    "label": label,
                    "symbol": sig["symbol"],
                    "return": ret,
                    "action": sig["action"],
                    "ml_prob": sig["ml_prob"],
                })

            except Exception as e:
                log.debug("获取 %s 价格失败: %s", sig["symbol"], e)
                remaining.append(sig)

        self._pending = remaining

        # 喂给在线学习
        if resolved:
            learner = self._get_learner()
            n_pos = sum(1 for r in resolved if r["label"] == 1)
            log.info("反馈收集: %d 条结果 (%d 正样本, %d 负样本)",
                     len(resolved), n_pos, len(resolved) - n_pos)

            for r in resolved:
                learner.add_sample(
                    features=r["features"],
                    label=r["label"],
                    symbol=r["symbol"],
                )

    def get_status(self) -> dict:
        """获取收集器状态。"""
        learner_status = self._get_learner().get_status() if self._learner else {}
        return {
            "pending_signals": len(self._pending),
            "outcome_window_sec": self.outcome_window,
            "min_return": self.min_return,
            "learner": learner_status,
        }


# 全局单例
_collector: Optional[FeedbackCollector] = None


def get_feedback_collector() -> FeedbackCollector:
    """获取全局 FeedbackCollector 单例。"""
    global _collector
    if _collector is None:
        _collector = FeedbackCollector()
    return _collector
