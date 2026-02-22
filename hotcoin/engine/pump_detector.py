"""
Pump 阶段检测器

基于 K线 OHLCV + taker_buy_base 数据，检测 5 个 Pump 阶段:
  NORMAL       — 正常波动
  ACCUMULATION — 放量不涨 (吸筹)
  EARLY_PUMP   — 动量 3-10%, 量放大
  MAIN_PUMP    — 动量 >10%, 加速上涨
  DISTRIBUTION — 量大但涨不动, 长上影线 (派发)

评分 4 因子加权:
  成交量 30% | 价格 30% | 订单流 20% | 模式 20%
"""

from dataclasses import dataclass
from enum import Enum
import logging

import numpy as np
import pandas as pd

log = logging.getLogger("hotcoin.pump_detector")


class PumpPhase(Enum):
    NORMAL = "normal"
    ACCUMULATION = "accumulation"
    EARLY_PUMP = "early_pump"
    MAIN_PUMP = "main_pump"
    DISTRIBUTION = "distribution"


@dataclass
class PumpResult:
    phase: PumpPhase
    score: float            # 0-100
    volume_z: float         # 成交量 Z-Score
    momentum: float         # 5-bar 涨幅 %
    acceleration: float     # 动量变化率
    buy_ratio: float        # taker_buy / total_volume
    suggested_action: str   # "WATCH" | "CONSIDER_BUY" | "HOLD" | "PREPARE_SELL" | "SELL"


_MIN_BARS = 30


def detect_pump_phase(df: pd.DataFrame, lookback: int = 100) -> PumpResult:
    """从 K线 DataFrame 检测 Pump 阶段。

    Parameters
    ----------
    df : DataFrame with columns: close, high, low, volume, (optional) taker_buy_base
    lookback : 回溯计算窗口

    Returns
    -------
    PumpResult
    """
    if df is None or len(df) < _MIN_BARS:
        return PumpResult(
            phase=PumpPhase.NORMAL, score=0, volume_z=0,
            momentum=0, acceleration=0, buy_ratio=0.5,
            suggested_action="WATCH",
        )

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    # taker_buy_base 可能缺失
    has_taker = "taker_buy_base" in df.columns
    taker_buy = df["taker_buy_base"].values if has_taker else None

    n = len(close)
    lb = min(lookback, n)

    # ---- 1. 成交量 Z-Score ----
    vol_window = volume[-lb:]
    vol_mean = np.mean(vol_window)
    vol_std = np.std(vol_window)
    current_vol = volume[-1]
    volume_z = (current_vol - vol_mean) / vol_std if vol_std > 0 else 0.0

    # ---- 2. 价格动量 (5-bar 涨幅%) ----
    bars_back = min(5, n - 1)
    ref_price = close[-(bars_back + 1)]
    momentum = ((close[-1] / ref_price) - 1) * 100 if ref_price > 0 else 0.0

    # 前一周期动量 (用于加速度)
    if n > bars_back + 2:
        prev_ref = close[-(bars_back + 2)]
        prev_close = close[-2]
        prev_momentum = ((prev_close / prev_ref) - 1) * 100 if prev_ref > 0 else 0.0
    else:
        prev_momentum = 0.0
    acceleration = momentum - prev_momentum

    # ---- 3. 买卖比 ----
    if taker_buy is not None and current_vol > 0:
        buy_ratio = float(taker_buy[-1] / current_vol)
    else:
        buy_ratio = 0.5

    # ---- 4. 突破检测 ----
    is_breakout = False
    if n >= 21:
        high_20 = np.max(high[-21:-1])
        is_breakout = close[-1] > high_20

    # ---- 评分 ----
    vol_score = min(100, max(0, volume_z * 20))
    if volume_z > 5:
        vol_score = min(100, vol_score + 20)

    price_score = min(50, abs(momentum) * 5)
    if acceleration > 0:
        price_score += min(30, acceleration * 10)
    if is_breakout:
        price_score += 20
    price_score = min(100, max(0, price_score))

    flow_score = min(100, max(0, abs(buy_ratio - 0.5) * 200))
    if buy_ratio < 0.5:
        flow_score = -flow_score  # 卖压为负分

    pattern_score = 0.0
    if vol_score > 50 and price_score > 30:
        pattern_score += 30
    if momentum > 0 and acceleration > 0:
        pattern_score += 25
    if buy_ratio > 0.55:
        pattern_score += 25
    if is_breakout:
        pattern_score += 20
    pattern_score = min(100, pattern_score)

    # 加权综合 (flow_score 可能为负, 用 max(0, ...) 截断)
    total_score = (
        vol_score * 0.30
        + price_score * 0.30
        + max(0, flow_score) * 0.20
        + pattern_score * 0.20
    )
    total_score = max(0, min(100, total_score))

    # ---- 阶段判定 ----
    phase = _determine_phase(total_score, vol_score, price_score, flow_score, momentum, acceleration)
    suggested = _suggest_action(phase, total_score)

    return PumpResult(
        phase=phase,
        score=round(total_score, 1),
        volume_z=round(volume_z, 2),
        momentum=round(momentum, 2),
        acceleration=round(acceleration, 2),
        buy_ratio=round(buy_ratio, 3),
        suggested_action=suggested,
    )


def _determine_phase(
    total: float, vol: float, price: float, flow: float,
    momentum: float, accel: float,
) -> PumpPhase:
    # 派发优先: 强卖压信号不受总分门槛限制
    if momentum < -5:
        return PumpPhase.DISTRIBUTION
    if flow < 0 and price > 30:
        return PumpPhase.DISTRIBUTION
    if total < 30:
        return PumpPhase.NORMAL
    # 主升: 动量 >10% 且加速
    if momentum > 10 and accel > 0:
        return PumpPhase.MAIN_PUMP
    # 早期拉升: 动量 3-10%
    if 3 <= momentum <= 10 and total > 40:
        return PumpPhase.EARLY_PUMP
    # 吸筹: 放量不涨
    if vol > 60 and price < 30:
        return PumpPhase.ACCUMULATION
    # fallback
    if total > 40:
        return PumpPhase.EARLY_PUMP
    return PumpPhase.NORMAL


def _suggest_action(phase: PumpPhase, score: float) -> str:
    if phase == PumpPhase.EARLY_PUMP and score > 50:
        return "CONSIDER_BUY"
    if phase == PumpPhase.MAIN_PUMP:
        return "HOLD"
    if phase == PumpPhase.DISTRIBUTION:
        return "PREPARE_SELL"
    if phase == PumpPhase.ACCUMULATION:
        return "WATCH"
    return "WATCH"


# ---------------------------------------------------------------------------
# ML 增强 Pump 检测 (shadow 模式)
# ---------------------------------------------------------------------------

_ML_MODEL = None
_ML_META = None
_ML_SHADOW = True  # True = 只记录不覆盖规则结果


def _load_pump_model(interval: str = "15m"):
    """懒加载 pump 分类模型。"""
    global _ML_MODEL, _ML_META
    if _ML_MODEL is not None:
        return True

    import os
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "data", f"pump_lgb_{interval}.txt"
    )
    meta_path = model_path.replace(".txt", "_meta.json")

    if not os.path.exists(model_path):
        return False

    try:
        import lightgbm as lgb
        import json
        _ML_MODEL = lgb.Booster(model_file=model_path)
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                _ML_META = json.load(f)
        log.info("Pump ML 模型已加载: %s", model_path)
        return True
    except Exception as e:
        log.warning("Pump ML 模型加载失败: %s", e)
        return False


def detect_pump_phase_ml(
    df: pd.DataFrame,
    interval: str = "15m",
    shadow: bool = True,
) -> PumpResult:
    """
    ML 增强的 Pump 阶段检测。

    先用规则检测, 再用 ML 模型预测, 加权融合。
    shadow=True 时只记录 ML 结果, 返回规则结果。
    """
    rule_result = detect_pump_phase(df)

    if not _load_pump_model(interval):
        return rule_result

    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from hotcoin.ml.train_pump import compute_pump_features

        features = compute_pump_features(df)
        last_row = features.iloc[[-1]].values

        # 处理 NaN
        last_row = np.nan_to_num(last_row, nan=0.0)

        probs = _ML_MODEL.predict(last_row)[0]  # shape: (5,)
        ml_phase_idx = int(np.argmax(probs))
        ml_confidence = float(probs[ml_phase_idx])

        phase_map = {
            0: PumpPhase.NORMAL,
            1: PumpPhase.ACCUMULATION,
            2: PumpPhase.EARLY_PUMP,
            3: PumpPhase.MAIN_PUMP,
            4: PumpPhase.DISTRIBUTION,
        }
        ml_phase = phase_map[ml_phase_idx]

        log.info("Pump ML: %s (conf=%.2f) | Rule: %s (score=%.1f) | shadow=%s",
                 ml_phase.value, ml_confidence,
                 rule_result.phase.value, rule_result.score,
                 shadow)

        if shadow or _ML_SHADOW:
            return rule_result

        # 非 shadow: ML 置信度 > 0.6 时用 ML 结果
        if ml_confidence > 0.6:
            suggested = _suggest_action(ml_phase, rule_result.score)
            return PumpResult(
                phase=ml_phase,
                score=rule_result.score,
                volume_z=rule_result.volume_z,
                momentum=rule_result.momentum,
                acceleration=rule_result.acceleration,
                buy_ratio=rule_result.buy_ratio,
                suggested_action=suggested,
            )

        return rule_result

    except Exception as e:
        log.warning("Pump ML 推理失败: %s", e)
        return rule_result
