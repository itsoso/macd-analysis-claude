#!/usr/bin/env python3
"""
简化版 LGB 训练脚本 - 跳过 Walk-Forward，直接在全量数据上训练
"""
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import json
import os

# 导入项目模块
from binance_fetcher import fetch_binance_klines
from ml_features import compute_ml_features

def main():
    print('='*60)
    print('LGB 方向模型训练 (简化版)')
    print('='*60)

    # 加载数据
    print('\n加载 ETHUSDT/1h 数据...')
    df = fetch_binance_klines('ETHUSDT', '1h', days=365*5)  # 5年数据
    print(f'数据: {len(df)} 条 ({df.index[0]} ~ {df.index[-1]})')

    # 计算特征
    print('\n计算 ML 特征...')
    features = compute_ml_features(df, '1h')
    print(f'特征: {features.shape}')

    # 生成标签: 未来5小时收益率 > 0.5%
    future_return = df['close'].pct_change(5).shift(-5)
    labels = (future_return > 0.005).astype(int)

    # 对齐索引
    valid_idx = features.index.intersection(labels.index)
    X = features.loc[valid_idx]
    y = labels.loc[valid_idx]

    # 填充缺失值
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    # 去除无效样本
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]

    print(f'有效样本: {len(X)}')

    # 时间序列分割 (80% 训练, 20% 测试)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f'训练集: {len(X_train)} ({X_train.index[0]} ~ {X_train.index[-1]})')
    print(f'测试集: {len(X_test)} ({X_test.index[0]} ~ {X_test.index[-1]})')

    # 训练模型
    print('\n训练 LightGBM...')
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )

    # 评估
    print('\n评估模型...')
    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]

    auc_train = roc_auc_score(y_train, y_pred_train)
    auc_test = roc_auc_score(y_test, y_pred_test)

    print(f'训练集 AUC: {auc_train:.4f}')
    print(f'测试集 AUC: {auc_test:.4f}')

    # 保存模型
    os.makedirs('data/ml_models', exist_ok=True)
    model_path = 'data/ml_models/lgb_direction_model_1h.txt'
    model.booster_.save_model(model_path)
    print(f'\n模型已保存: {model_path}')

    # 保存元数据
    meta = {
        'timeframe': '1h',
        'features': X.shape[1],
        'feature_names': list(X.columns),
        'samples': len(X),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_auc': float(auc_train),
        'test_auc': float(auc_test),
        'best_iteration': model.best_iteration_,
        'training_method': 'simple_split'
    }

    meta_path = model_path + '.meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'元数据已保存: {meta_path}')

    print('\n训练完成!')
    print(f'测试集 AUC: {auc_test:.4f}')

if __name__ == '__main__':
    main()
