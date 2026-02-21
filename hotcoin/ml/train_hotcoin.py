#!/usr/bin/env python3
"""
热点币 ML 训练入口 — H800 GPU 训练

用法:
    # 热度预测模型 (哪些币即将异动)
    python3 hotcoin/ml/train_hotcoin.py --task hotness --symbols MULTI --tf 15m --trials 200

    # 交易方向模型 (候选币的多空判断)
    python3 hotcoin/ml/train_hotcoin.py --task trade --symbols MULTI --tf 15m

    # ETH 零样本迁移验证
    python3 hotcoin/ml/train_hotcoin.py --task transfer_test --symbols MULTI --tf 15m

训练策略:
    Phase 1: 不用 ML, 纯六书快参数 + 动量
    Phase 3: ETH LGB 零样本迁移验证 → 多币种统一训练
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hotcoin.ml.features_hot import (
    compute_hot_features,
    make_hotness_labels,
    make_trade_labels,
)

log = logging.getLogger("hotcoin.train")


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_governance_artifacts(model_path: str, meta_path: str, task: str, interval: str):
    """输出模型契约与晋升门禁结果。"""
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        log.warning("治理产物生成失败: 无法读取 meta (%s)", e)
        return

    model_dir = os.path.dirname(model_path)
    now_iso = datetime.now().isoformat()
    thresholds = {
        "min_samples": 20000,
        "min_test_auc": 0.55,
    }
    n_samples = int(meta.get("n_samples", 0) or 0)
    test_auc = float(meta.get("test_auc", 0.0) or 0.0)
    approved = (n_samples >= thresholds["min_samples"]) and (test_auc >= thresholds["min_test_auc"])
    reasons = []
    if n_samples < thresholds["min_samples"]:
        reasons.append(f"n_samples<{thresholds['min_samples']}")
    if test_auc < thresholds["min_test_auc"]:
        reasons.append(f"test_auc<{thresholds['min_test_auc']}")
    if not reasons:
        reasons.append("all_thresholds_passed")

    runtime_contract = {
        "schema_version": "hotcoin_model_contract_v1",
        "task": task,
        "interval": interval,
        "generated_at": now_iso,
        "model_files": {
            "model": os.path.basename(model_path),
            "meta": os.path.basename(meta_path),
        },
        "hashes": {
            os.path.basename(model_path): _sha256_file(model_path),
            os.path.basename(meta_path): _sha256_file(meta_path),
        },
        "metrics": {
            "n_samples": n_samples,
            "n_features": int(meta.get("n_features", 0) or 0),
            "test_auc": test_auc,
        },
        "feature_names": meta.get("feature_names", []),
    }

    promotion_decision = {
        "schema_version": "hotcoin_promotion_decision_v1",
        "task": task,
        "interval": interval,
        "generated_at": now_iso,
        "metrics": runtime_contract["metrics"],
        "thresholds": thresholds,
        "approved": approved,
        "deployment_tier": "production" if approved else "research_only",
        "reasons": reasons,
    }

    contract_path = os.path.join(model_dir, f"runtime_contract_{task}_{interval}.json")
    decision_path = os.path.join(model_dir, f"promotion_decision_{task}_{interval}.json")
    with open(contract_path, "w", encoding="utf-8") as f:
        json.dump(runtime_contract, f, indent=2, ensure_ascii=False)
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump(promotion_decision, f, indent=2, ensure_ascii=False)
    log.info("治理产物已生成: %s, %s", contract_path, decision_path)

def prepare_multi_symbol_data(symbols: List[str], interval: str = "15m",
                               days: int = 180) -> Dict[str, pd.DataFrame]:
    """获取多币种历史 K 线。"""
    from binance_fetcher import fetch_binance_klines
    from indicators import add_all_indicators
    from ma_indicators import add_moving_averages

    data = {}
    for sym in symbols:
        try:
            df = fetch_binance_klines(sym, interval=interval, days=days)
            if df is None or len(df) < 100:
                log.warning("%s 数据不足 (%d bars), 跳过", sym,
                            len(df) if df is not None else 0)
                continue
            df = add_all_indicators(df)
            add_moving_averages(df, timeframe=interval)
            data[sym] = df
            log.info("%s: %d bars", sym, len(df))
        except Exception as e:
            log.warning("%s 获取失败: %s", sym, e)
    return data


def get_top_symbols(n: int = 100) -> List[str]:
    """获取成交额排名前 N 的 USDT 交易对。"""
    from binance_fetcher import fetch_all_tickers_24h
    tickers = fetch_all_tickers_24h()
    tickers.sort(key=lambda t: float(t.get("quoteVolume", 0)), reverse=True)
    symbols = [t["symbol"] for t in tickers[:n]]
    log.info("Top %d symbols by volume: %s ...", n, symbols[:5])
    return symbols


# ---------------------------------------------------------------------------
# 热度预测训练
# ---------------------------------------------------------------------------

def train_hotness_model(data: Dict[str, pd.DataFrame], interval: str = "15m"):
    """
    训练热度预测模型: 哪些币即将异动?

    标签: 未来 15min 涨幅在全市场排名 >= 90% 分位
    """
    log.info("=" * 60)
    log.info("训练热度预测模型 (hotness)")
    log.info("  币种数: %d, 周期: %s", len(data), interval)
    log.info("=" * 60)

    all_features = []
    all_labels = []

    # 构建横截面收益矩阵
    returns_dict = {}
    for sym, df in data.items():
        features = compute_hot_features(df)
        features["symbol"] = sym
        all_features.append(features)

        ret_15 = df["close"].pct_change(15).shift(-15)
        returns_dict[sym] = ret_15

    if not all_features:
        log.error("无可用数据")
        return

    returns_df = pd.DataFrame(returns_dict)
    hotness_labels = make_hotness_labels(returns_df, window=15, percentile=0.90)

    X_all = pd.concat(all_features, axis=0)
    feature_cols = [c for c in X_all.columns if c != "symbol"]

    combined = []
    for sym in data:
        feat = X_all[X_all["symbol"] == sym][feature_cols]
        labels = hotness_labels.get(sym, pd.Series(0, index=feat.index))
        labels = labels.reindex(feat.index).fillna(0)
        feat = feat.copy()
        feat["label"] = labels.values[:len(feat)]
        combined.append(feat)

    combined_df = pd.concat(combined).dropna()
    combined_df = combined_df.sort_index()
    X = combined_df[feature_cols].values
    y = combined_df["label"].values

    log.info("训练集: %d 样本, %d 特征, label_1=%.2f%%",
             len(X), X.shape[1], y.mean() * 100)

    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        dtrain = lgb.Dataset(X_train, label=y_train)
        dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 50,
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
            valid_sets=[dtest],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )

        pred_test = model.predict(X_test)
        auc = roc_auc_score(y_test, pred_test)
        log.info("Hotness 模型 Test AUC: %.4f", auc)

        # 保存模型
        out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, f"hotness_lgb_{interval}.txt")
        model.save_model(model_path)

        meta = {
            "task": "hotness",
            "interval": interval,
            "n_symbols": len(data),
            "n_samples": len(X),
            "n_features": X.shape[1],
            "feature_names": feature_cols,
            "test_auc": auc,
            "trained_at": datetime.now().isoformat(),
        }
        meta_path = model_path.replace(".txt", "_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        _write_governance_artifacts(model_path, meta_path, task="hotness", interval=interval)

        log.info("模型已保存: %s", model_path)

    except ImportError:
        log.error("lightgbm 未安装, 无法训练")


# ---------------------------------------------------------------------------
# 交易方向训练
# ---------------------------------------------------------------------------

def train_trade_model(data: Dict[str, pd.DataFrame], interval: str = "15m"):
    """
    训练交易方向模型: 候选币的多空判断。

    标签: 进入后 30min 内最大收益 >= 3%
    """
    log.info("=" * 60)
    log.info("训练交易方向模型 (trade)")
    log.info("  币种数: %d, 周期: %s", len(data), interval)
    log.info("=" * 60)

    all_X = []
    all_y = []

    for sym, df in data.items():
        features = compute_hot_features(df)
        feature_cols = [c for c in features.columns]
        labels = make_trade_labels(df, forward_window=30, min_return=0.03)

        valid_idx = features.dropna().index.intersection(labels.dropna().index)
        if len(valid_idx) < 100:
            continue

        features = features.loc[valid_idx]
        labels = labels.loc[valid_idx]

        all_X.append(features[feature_cols])
        all_y.append(labels)

    if not all_X:
        log.error("无可用数据")
        return

    X_df = pd.concat(all_X)
    y_series = pd.concat(all_y)
    X_df = X_df.sort_index()
    y_series = y_series.loc[X_df.index]

    feature_cols = list(all_X[0].columns)
    X = X_df.values
    y = y_series.values

    log.info("训练集: %d 样本, %d 特征, label_1=%.2f%%",
             len(X), X.shape[1], y.mean() * 100)

    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        dtrain = lgb.Dataset(X_train, label=y_train)
        dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 50,
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
            valid_sets=[dtest],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )

        pred_test = model.predict(X_test)
        auc = roc_auc_score(y_test, pred_test)
        log.info("Trade 模型 Test AUC: %.4f", auc)

        out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, f"trade_lgb_{interval}.txt")
        model.save_model(model_path)

        meta = {
            "task": "trade",
            "interval": interval,
            "n_symbols": len(data),
            "n_samples": len(X),
            "n_features": X.shape[1],
            "feature_names": feature_cols,
            "test_auc": auc,
            "trained_at": datetime.now().isoformat(),
        }
        meta_path = model_path.replace(".txt", "_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        _write_governance_artifacts(model_path, meta_path, task="trade", interval=interval)

        log.info("模型已保存: %s", model_path)

    except ImportError:
        log.error("lightgbm 未安装, 无法训练")


# ---------------------------------------------------------------------------
# ETH 迁移验证
# ---------------------------------------------------------------------------

def transfer_test(data: Dict[str, pd.DataFrame], interval: str = "15m"):
    """
    用 ETH 训练的 LGB 模型在热点币上做零样本迁移验证。
    """
    log.info("=" * 60)
    log.info("ETH LGB 零样本迁移验证")
    log.info("=" * 60)

    eth_model_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "data", "ml_models", f"lgb_model_{interval}.txt",
    )
    if not os.path.exists(eth_model_path):
        log.error("ETH 模型不存在: %s", eth_model_path)
        return

    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        eth_model = lgb.Booster(model_file=eth_model_path)
        log.info("ETH 模型已加载: %d 特征", eth_model.num_feature())

        results = {}
        for sym, df in data.items():
            features = compute_hot_features(df)
            labels = make_trade_labels(df, forward_window=30, min_return=0.03)
            valid = features.dropna().index.intersection(labels.dropna().index)
            if len(valid) < 100:
                continue

            X = features.loc[valid].values
            y = labels.loc[valid].values

            # ETH 模型可能特征数不同, 截断或填充
            n_eth_feat = eth_model.num_feature()
            if X.shape[1] > n_eth_feat:
                X = X[:, :n_eth_feat]
            elif X.shape[1] < n_eth_feat:
                pad = np.zeros((X.shape[0], n_eth_feat - X.shape[1]))
                X = np.hstack([X, pad])

            try:
                pred = eth_model.predict(X)
                auc = roc_auc_score(y, pred)
                results[sym] = {"auc": auc, "samples": len(y), "label_rate": y.mean()}
                log.info("  %s: AUC=%.4f (n=%d, label_rate=%.2f%%)",
                         sym, auc, len(y), y.mean() * 100)
            except Exception as e:
                log.warning("  %s 推理失败: %s", sym, e)

        if results:
            avg_auc = np.mean([r["auc"] for r in results.values()])
            log.info("迁移验证平均 AUC: %.4f (%d 币种)", avg_auc, len(results))

    except ImportError:
        log.error("lightgbm 未安装")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="热点币 ML 训练")
    parser.add_argument("--task", choices=["hotness", "trade", "transfer_test"],
                        required=True, help="训练任务")
    parser.add_argument("--symbols", default="MULTI",
                        help="MULTI = top 100 by volume, 或逗号分隔的币种列表")
    parser.add_argument("--tf", default="15m", help="K 线周期")
    parser.add_argument("--days", type=int, default=180, help="历史天数")
    parser.add_argument("--top-n", type=int, default=100, help="MULTI 时取 Top N")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
    )

    if args.symbols == "MULTI":
        symbols = get_top_symbols(args.top_n)
    else:
        symbols = [s.strip() for s in args.symbols.split(",")]

    log.info("准备数据: %d 币种, %s, %d 天", len(symbols), args.tf, args.days)
    data = prepare_multi_symbol_data(symbols, args.tf, args.days)
    log.info("有效币种: %d", len(data))

    if args.task == "hotness":
        train_hotness_model(data, args.tf)
    elif args.task == "trade":
        train_trade_model(data, args.tf)
    elif args.task == "transfer_test":
        transfer_test(data, args.tf)


if __name__ == "__main__":
    main()
