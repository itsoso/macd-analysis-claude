#!/usr/bin/env python3
"""
Pump 阶段分类器训练

用历史 K 线训练 LightGBM 多分类模型, 预测 5 个 pump 阶段。

用法:
    python3 hotcoin/ml/train_pump.py --tf 15m --days 365
    python3 hotcoin/ml/train_pump.py --tf 15m --days 365 --symbols ETHUSDT,BTCUSDT
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hotcoin.ml.features_hot import compute_hot_features
from hotcoin.ml.pump_labeler import label_pump_phases

log = logging.getLogger("hotcoin.train_pump")

PHASE_NAMES = ["normal", "accumulation", "early_pump", "main_pump", "distribution"]
DEFAULT_SYMBOLS = ["ETHUSDT", "BTCUSDT", "SOLUSDT", "BNBUSDT"]


def compute_pump_features(df: pd.DataFrame) -> pd.DataFrame:
    """在 hot_features 基础上添加 pump 专用特征。"""
    features = compute_hot_features(df)

    c = df["close"].values
    o = df["open"].values
    h = df["high"].values
    low = df["low"].values
    v = df["volume"].values

    # 成交量加速度: vol_3 / vol_10 的变化率
    vol_3 = pd.Series(v).rolling(3, min_periods=1).mean()
    vol_10 = pd.Series(v).rolling(10, min_periods=1).mean().clip(lower=1)
    vol_ratio = vol_3 / vol_10
    features["vol_acceleration"] = vol_ratio.diff().values

    # 价格加速度: mom_3 - mom_10
    mom_3 = pd.Series(c).pct_change(3)
    mom_10 = pd.Series(c).pct_change(10)
    features["price_acceleration"] = (mom_3 - mom_10).values

    # 上影线占比
    rng = h - low
    rng_safe = np.where(rng > 0, rng, 1)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - low
    features["upper_wick_ratio"] = np.where(rng > 0, upper_wick / rng_safe, 0)
    features["lower_wick_ratio"] = np.where(rng > 0, lower_wick / rng_safe, 0)

    # 突破检测
    close_s = pd.Series(c)
    features["breakout_20"] = (close_s > close_s.rolling(20).max().shift(1)).astype(int).values
    features["breakout_50"] = (close_s > close_s.rolling(50).max().shift(1)).astype(int).values

    return features


def load_and_prepare(symbols: List[str], interval: str, days: int) -> tuple:
    """加载数据, 计算特征和标签。"""
    from binance_fetcher import fetch_binance_klines
    from indicators import add_all_indicators
    from ma_indicators import add_moving_averages

    all_X = []
    all_y = []

    for sym in symbols:
        try:
            df = fetch_binance_klines(sym, interval=interval, days=days)
            if df is None or len(df) < 200:
                log.warning("%s 数据不足, 跳过", sym)
                continue

            df = add_all_indicators(df)
            add_moving_averages(df, timeframe=interval)

            features = compute_pump_features(df)
            labels = label_pump_phases(df)

            # 对齐并去 NaN
            valid = features.dropna().index.intersection(labels.dropna().index)
            if len(valid) < 100:
                log.warning("%s 有效样本不足 (%d), 跳过", sym, len(valid))
                continue

            X = features.loc[valid]
            y = labels.loc[valid]
            all_X.append(X)
            all_y.append(y)
            log.info("%s: %d bars, %d 有效样本", sym, len(df), len(valid))

        except Exception as e:
            log.warning("%s 处理失败: %s", sym, e)

    if not all_X:
        return None, None, None

    X_df = pd.concat(all_X).sort_index()
    y_series = pd.concat(all_y).loc[X_df.index]
    feature_cols = list(all_X[0].columns)

    return X_df, y_series, feature_cols


def train_pump_classifier(
    symbols: List[str],
    interval: str = "15m",
    days: int = 365,
) -> dict:
    """训练 Pump 阶段分类器 (LightGBM multiclass)。"""
    log.info("=" * 60)
    log.info("训练 Pump 阶段分类器")
    log.info("  币种: %s, 周期: %s, 天数: %d", symbols, interval, days)
    log.info("=" * 60)

    X_df, y_series, feature_cols = load_and_prepare(symbols, interval, days)
    if X_df is None:
        log.error("无可用数据")
        return {}

    X = X_df.values
    y = y_series.values

    # 标签分布
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        log.info("  class %d (%s): %d (%.1f%%)",
                 cls, PHASE_NAMES[cls], cnt, cnt / len(y) * 100)

    # 时间分割 70/15/15
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    log.info("分割: train=%d, val=%d, test=%d", len(X_train), len(X_val), len(X_test))

    try:
        import lightgbm as lgb
        from sklearn.metrics import accuracy_score, classification_report

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        params = {
            "objective": "multiclass",
            "num_class": 5,
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 30,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "verbose": -1,
        }

        model = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )

        # 评估
        pred_test = model.predict(X_test)
        pred_labels = np.argmax(pred_test, axis=1)
        acc = accuracy_score(y_test, pred_labels)
        report = classification_report(
            y_test, pred_labels,
            target_names=PHASE_NAMES,
            zero_division=0,
        )
        log.info("Test Accuracy: %.4f", acc)
        log.info("Classification Report:\n%s", report)

        # 保存模型
        out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, f"pump_lgb_{interval}.txt")
        model.save_model(model_path)

        meta = {
            "task": "pump_classifier",
            "interval": interval,
            "n_symbols": len(symbols),
            "symbols": symbols,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "feature_names": feature_cols,
            "phase_names": PHASE_NAMES,
            "test_accuracy": float(acc),
            "label_distribution": {PHASE_NAMES[int(c)]: int(n) for c, n in zip(unique, counts)},
            "trained_at": datetime.now().isoformat(),
        }
        meta_path = model_path.replace(".txt", "_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        log.info("模型已保存: %s", model_path)
        return meta

    except ImportError:
        log.error("lightgbm 未安装, 无法训练")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Pump 阶段分类器训练")
    parser.add_argument("--tf", default="15m", help="K 线周期")
    parser.add_argument("--days", type=int, default=365, help="历史天数")
    parser.add_argument("--symbols", default=None,
                        help="逗号分隔的币种, 默认 ETH/BTC/SOL/BNB")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
    )

    symbols = args.symbols.split(",") if args.symbols else DEFAULT_SYMBOLS
    train_pump_classifier(symbols, args.tf, args.days)


if __name__ == "__main__":
    main()
