#!/usr/bin/env python3
"""
训练生产模型 — 用 Optuna 最优参数在全量数据上训练 LightGBM + LSTM 集成模型
保存到 data/ml_models/ 供实盘使用
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_gpu import prepare_features, detect_gpu, MODEL_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# Optuna 最优参数
OPTUNA_BEST_PARAMS = {
    'boosting_type': 'gbdt',
    'num_leaves': 34,
    'learning_rate': 0.0102,
    'feature_fraction': 0.573,
    'bagging_fraction': 0.513,
    'bagging_freq': 3,
    'min_child_samples': 56,
    'lambda_l1': 0.0114,
    'lambda_l2': 0.2146,
}
OPTUNA_NUM_BOOST_ROUND = 356


def train_production_lgb(tf='1h'):
    """用 Optuna 最优参数在全量训练集上训练 LightGBM 生产模型"""
    log.info(f"{'='*60}")
    log.info(f"训练 LightGBM 生产模型 — ETHUSDT/{tf}")
    log.info(f"{'='*60}")

    features, labels_df = prepare_features('ETHUSDT', tf)
    label = labels_df['profitable_long_5']

    # 时间切分: 训练80%, 验证10%, 测试10%
    n = len(features)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    purge = 24

    X_train = features.iloc[:train_end]
    y_train = label.iloc[:train_end]
    X_val = features.iloc[train_end + purge:val_end]
    y_val = label.iloc[train_end + purge:val_end]
    X_test = features.iloc[val_end + purge:]
    y_test = label.iloc[val_end + purge:]

    # 过滤 NaN
    valid_tr = y_train.notna()
    valid_va = y_val.notna()
    valid_te = y_test.notna()
    X_train, y_train = X_train[valid_tr], y_train[valid_tr]
    X_val, y_val = X_val[valid_va], y_val[valid_va]
    X_test, y_test = X_test[valid_te], y_test[valid_te]

    log.info(f"训练: {len(X_train)}, 验证: {len(X_val)}, 测试: {len(X_test)}")
    log.info(f"特征数: {features.shape[1]}")

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        **OPTUNA_BEST_PARAMS,
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    t0 = time.time()
    model = lgb.train(
        params, dtrain,
        num_boost_round=OPTUNA_NUM_BOOST_ROUND,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(50),
        ],
    )
    elapsed = time.time() - t0

    # 评估
    from sklearn.metrics import roc_auc_score
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    val_auc = roc_auc_score(y_val, val_pred)
    test_auc = roc_auc_score(y_test, test_pred)

    log.info(f"验证 AUC: {val_auc:.4f}, 测试 AUC: {test_auc:.4f}")
    log.info(f"训练耗时: {elapsed:.1f}s")

    # 保存模型
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'lgb_direction_model.txt')
    model.save_model(model_path)

    # 保存元数据
    feature_names = list(features.columns)
    imp = model.feature_importance(importance_type='gain')
    feature_importance = dict(zip(feature_names, imp.tolist()))

    # 统计特征均值/标准差 (供 LSTM 推理使用)
    feat_mean = X_train.mean().to_dict()
    feat_std = X_train.std().to_dict()

    meta = {
        'feature_names': feature_names,
        'feature_importance': feature_importance,
        'train_metrics': {
            'val_auc': round(val_auc, 4),
            'test_auc': round(test_auc, 4),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'best_iteration': model.best_iteration,
            'num_features': len(feature_names),
        },
        'optuna_params': OPTUNA_BEST_PARAMS,
        'feat_mean': {k: round(v, 8) for k, v in feat_mean.items()},
        'feat_std': {k: round(max(v, 1e-8), 8) for k, v in feat_std.items()},
        'thresholds': {
            'long_threshold': 0.58,
            'short_threshold': 0.42,
        },
    }
    meta_path = model_path + '.meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    log.info(f"模型已保存: {model_path}")
    log.info(f"元数据已保存: {meta_path}")

    # Top features
    top_feats = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
    log.info("Top 15 特征:")
    for name, score in top_feats:
        log.info(f"  {name}: {score:.1f}")

    return {
        'model_path': model_path,
        'val_auc': val_auc,
        'test_auc': test_auc,
        'feature_names': feature_names,
    }


def update_ensemble_config(lgb_result):
    """更新集成配置文件"""
    config = {
        'version': 'v5_gpu_ensemble',
        'trained_at': pd.Timestamp.now().isoformat(),
        'components': {
            'lgb_direction': {
                'model_file': 'lgb_direction_model.txt',
                'weight': 0.65,
                'val_auc': round(lgb_result['val_auc'], 4),
                'test_auc': round(lgb_result['test_auc'], 4),
            },
            'lstm_1h': {
                'model_file': 'lstm_1h.pt',
                'weight': 0.35,
                'test_auc': 0.5366,
                'seq_len': 48,
                'hidden_dim': 128,
                'num_layers': 2,
                'input_dim': len(lgb_result['feature_names']),
            },
            'regime': {
                'model_file': 'trend_regime_model.txt',
                'weight': 0.0,  # regime 不直接参与评分，作为过滤器
            },
        },
        'thresholds': {
            'long_threshold': 0.58,
            'short_threshold': 0.42,
        },
        'feature_names': lgb_result['feature_names'],
    }

    path = os.path.join(MODEL_DIR, 'ensemble_config.json')
    with open(path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    log.info(f"集成配置已更新: {path}")


if __name__ == '__main__':
    log.info("开始训练生产模型...")
    lgb_result = train_production_lgb('1h')
    update_ensemble_config(lgb_result)
    log.info("\n生产模型训练完成！")
