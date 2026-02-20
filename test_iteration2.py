"""
测试第2轮迭代：CatBoost + Stacking优化

对比:
- 基线: LightGBM (第1轮的改进模型)
- 改进1: CatBoost GPU
- 改进2: Stacking (LightGBM + CatBoost + XGBoost → LightGBM元学习器)
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import add_all_indicators
from ma_indicators import add_moving_averages
from ml_features import compute_ml_features, compute_profit_labels


def test_catboost_and_stacking():
    """测试 CatBoost 和 Stacking 优化"""
    print("="*60)
    print("第2轮迭代测试：CatBoost + Stacking优化")
    print("="*60)

    # 加载数据
    print("\n1. 加载数据...")
    parquet_path = 'data/klines/ETHUSDT/1h.parquet'
    df = pd.read_parquet(parquet_path)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    df = df.iloc[-180*24:]
    print(f"   数据量: {len(df)} 条")

    # 计算指标和特征
    print("\n2. 计算技术指标和特征...")
    df = add_all_indicators(df)
    df = add_moving_averages(df)
    features = compute_ml_features(df)
    labels = compute_profit_labels(df, horizons=[5], cost_pct=0.0015)
    target_col = 'profitable_long_5'

    # 对齐数据
    valid_idx = features.notna().all(axis=1) & labels[target_col].notna()
    X = features[valid_idx].fillna(0).replace([np.inf, -np.inf], 0)
    y = labels.loc[valid_idx, target_col]

    print(f"   特征数: {X.shape[1]}")
    print(f"   有效样本: {len(X)}, 正类比例: {y.mean():.3f}")

    # 切分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=False
    )

    print(f"\n3. 训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # ========== 基线: LightGBM (第1轮改进模型) ==========
    print("\n" + "="*60)
    print("基线模型: LightGBM")
    print("="*60)

    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 20,
        'learning_rate': 0.05,
        'feature_fraction': 0.65,
        'bagging_fraction': 0.75,
        'bagging_freq': 3,
        'min_child_samples': 50,
        'verbose': -1,
        'seed': 42,
    }

    dtrain_lgb = lgb.Dataset(X_train, label=y_train)
    dval_lgb = lgb.Dataset(X_test, label=y_test, reference=dtrain_lgb)

    model_lgb = lgb.train(
        lgb_params,
        dtrain_lgb,
        num_boost_round=300,
        valid_sets=[dval_lgb],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)]
    )

    y_pred_lgb = model_lgb.predict(X_test)
    auc_lgb = roc_auc_score(y_test, y_pred_lgb)
    acc_lgb = accuracy_score(y_test, (y_pred_lgb > 0.5).astype(int))

    print(f"\nLightGBM 结果:")
    print(f"  AUC: {auc_lgb:.4f}")
    print(f"  Accuracy: {acc_lgb:.4f}")

    # ========== 改进1: CatBoost ==========
    print("\n" + "="*60)
    print("改进1: CatBoost GPU")
    print("="*60)

    try:
        from catboost import CatBoostClassifier, Pool

        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_test, y_test)

        model_cat = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            loss_function='Logloss',
            eval_metric='AUC',
            task_type='CPU',  # 使用 CPU (GPU 可能不可用)
            early_stopping_rounds=30,
            verbose=50,
            random_seed=42,
        )

        model_cat.fit(train_pool, eval_set=val_pool, use_best_model=True)

        y_pred_cat = model_cat.predict_proba(X_test)[:, 1]
        auc_cat = roc_auc_score(y_test, y_pred_cat)
        acc_cat = accuracy_score(y_test, (y_pred_cat > 0.5).astype(int))

        print(f"\nCatBoost 结果:")
        print(f"  AUC: {auc_cat:.4f}")
        print(f"  Accuracy: {acc_cat:.4f}")

        catboost_available = True
    except ImportError:
        print("  CatBoost 未安装，跳过")
        auc_cat = auc_lgb
        acc_cat = acc_lgb
        catboost_available = False

    # ========== 改进2: Stacking (LGB元学习器) ==========
    print("\n" + "="*60)
    print("改进2: Stacking Ensemble (LGB元学习器)")
    print("="*60)

    # 生成基模型预测 (OOF)
    print("生成基模型预测...")

    # LightGBM 预测
    meta_train_lgb = np.zeros(len(X_train))
    meta_test_lgb = y_pred_lgb

    # 5-Fold CV 生成 OOF
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=False)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        dtrain_fold = lgb.Dataset(X_fold_train, label=y_fold_train)
        model_fold = lgb.train(lgb_params, dtrain_fold, num_boost_round=100, callbacks=[lgb.log_evaluation(0)])

        meta_train_lgb[val_idx] = model_fold.predict(X_fold_val)

    print(f"  LightGBM OOF AUC: {roc_auc_score(y_train, meta_train_lgb):.4f}")

    # CatBoost 预测 (如果可用)
    if catboost_available:
        meta_train_cat = np.zeros(len(X_train))
        meta_test_cat = y_pred_cat

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]

            train_pool_fold = Pool(X_fold_train, y_fold_train)
            model_fold = CatBoostClassifier(
                iterations=100, learning_rate=0.05, depth=6,
                task_type='CPU', verbose=False, random_seed=42
            )
            model_fold.fit(train_pool_fold)

            meta_train_cat[val_idx] = model_fold.predict_proba(X_fold_val)[:, 1]

        print(f"  CatBoost OOF AUC: {roc_auc_score(y_train, meta_train_cat):.4f}")

        # 元特征矩阵
        meta_X_train = np.column_stack([meta_train_lgb, meta_train_cat])
        meta_X_test = np.column_stack([meta_test_lgb, meta_test_cat])
    else:
        # 只用 LightGBM
        meta_X_train = meta_train_lgb.reshape(-1, 1)
        meta_X_test = meta_test_lgb.reshape(-1, 1)

    # 训练元学习器 (LightGBM)
    print("\n训练元学习器 (LightGBM)...")
    meta_params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 15,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 3,
        'min_child_samples': 50,
        'verbose': -1,
        'seed': 42,
    }

    dmeta_train = lgb.Dataset(meta_X_train, label=y_train)
    meta_model = lgb.train(meta_params, dmeta_train, num_boost_round=100, callbacks=[lgb.log_evaluation(0)])

    y_pred_stack = meta_model.predict(meta_X_test)
    auc_stack = roc_auc_score(y_test, y_pred_stack)
    acc_stack = accuracy_score(y_test, (y_pred_stack > 0.5).astype(int))

    print(f"\nStacking 结果:")
    print(f"  AUC: {auc_stack:.4f}")
    print(f"  Accuracy: {acc_stack:.4f}")

    # ========== 对比结果 ==========
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    print(f"LightGBM (基线):  AUC={auc_lgb:.4f}, Acc={acc_lgb:.4f}")
    if catboost_available:
        print(f"CatBoost:         AUC={auc_cat:.4f}, Acc={acc_cat:.4f} ({(auc_cat/auc_lgb-1)*100:+.2f}%)")
    print(f"Stacking:         AUC={auc_stack:.4f}, Acc={acc_stack:.4f} ({(auc_stack/auc_lgb-1)*100:+.2f}%)")

    # 保存结果
    result = {
        'lgb_auc': float(auc_lgb),
        'lgb_acc': float(acc_lgb),
        'catboost_auc': float(auc_cat) if catboost_available else None,
        'catboost_acc': float(acc_cat) if catboost_available else None,
        'stacking_auc': float(auc_stack),
        'stacking_acc': float(acc_stack),
        'auc_improvement_catboost': float(auc_cat - auc_lgb) if catboost_available else None,
        'auc_improvement_stacking': float(auc_stack - auc_lgb),
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
    }

    import json
    with open('test_iteration2_result.json', 'w') as f:
        json.dump(result, f, indent=2)

    print("\n✓ 结果已保存到 test_iteration2_result.json")

    return result


if __name__ == '__main__':
    result = test_catboost_and_stacking()
