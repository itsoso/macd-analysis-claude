"""
测试第1轮迭代：高频微结构特征对模型性能的影响

对比:
- 基线: 原始特征 (73维)
- 改进: 原始特征 + 高频微结构特征 (73+8=81维)
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

from binance_fetcher import fetch_binance_klines
from indicators import add_all_indicators
from ma_indicators import add_moving_averages
from ml_features import compute_ml_features, compute_profit_labels


def test_microstructure_features():
    """测试高频微结构特征的效果"""
    print("="*60)
    print("第1轮迭代测试：高频微结构特征")
    print("="*60)

    # 加载数据 (直接从本地 Parquet)
    print("\n1. 加载数据...")
    parquet_path = 'data/klines/ETHUSDT/1h.parquet'
    df = pd.read_parquet(parquet_path)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    # 取最近 180 天
    df = df.iloc[-180*24:]
    print(f"   数据量: {len(df)} 条")
    print(f"   时间范围: {df.index[0]} ~ {df.index[-1]}")

    # 计算指标
    print("\n2. 计算技术指标...")
    df = add_all_indicators(df)
    df = add_moving_averages(df)

    # 计算特征
    print("\n3. 计算 ML 特征...")
    features = compute_ml_features(df)

    # 识别新增的高频微结构特征
    microstructure_features = [
        'ofi', 'ofi_ma5', 'ofi_std5', 'cum_ofi', 'cum_ofi_slope',
        'large_trade_ratio', 'buy_sell_pressure',
        'vwap_dist_change', 'vwap_dist_ma5', 'above_vwap_streak'
    ]

    # 基线特征 (排除微结构特征)
    baseline_features = [c for c in features.columns if c not in microstructure_features]
    print(f"   基线特征: {len(baseline_features)} 维")
    print(f"   微结构特征: {len([c for c in microstructure_features if c in features.columns])} 维")
    print(f"   总特征: {features.shape[1]} 维")

    # 计算标签
    print("\n4. 计算标签...")
    labels = compute_profit_labels(df, horizons=[5], cost_pct=0.0015)
    target_col = 'profitable_long_5'

    # 对齐数据
    valid_idx = features.notna().all(axis=1) & labels[target_col].notna()
    X = features[valid_idx]
    y = labels.loc[valid_idx, target_col]

    print(f"   有效样本: {len(X)}")
    print(f"   正类比例: {y.mean():.3f}")

    # 切分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=False  # 时间序列不打乱
    )

    print(f"\n5. 训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # ========== 基线模型 (不含微结构特征) ==========
    print("\n" + "="*60)
    print("基线模型 (原始特征)")
    print("="*60)

    X_train_baseline = X_train[baseline_features].fillna(0).replace([np.inf, -np.inf], 0)
    X_test_baseline = X_test[baseline_features].fillna(0).replace([np.inf, -np.inf], 0)

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

    dtrain_baseline = lgb.Dataset(X_train_baseline, label=y_train)
    dval_baseline = lgb.Dataset(X_test_baseline, label=y_test, reference=dtrain_baseline)

    model_baseline = lgb.train(
        lgb_params,
        dtrain_baseline,
        num_boost_round=300,
        valid_sets=[dval_baseline],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)]
    )

    y_pred_baseline = model_baseline.predict(X_test_baseline)
    auc_baseline = roc_auc_score(y_test, y_pred_baseline)
    acc_baseline = accuracy_score(y_test, (y_pred_baseline > 0.5).astype(int))

    print(f"\n基线模型结果:")
    print(f"  AUC: {auc_baseline:.4f}")
    print(f"  Accuracy: {acc_baseline:.4f}")

    # ========== 改进模型 (含微结构特征) ==========
    print("\n" + "="*60)
    print("改进模型 (原始特征 + 高频微结构特征)")
    print("="*60)

    X_train_improved = X_train.fillna(0).replace([np.inf, -np.inf], 0)
    X_test_improved = X_test.fillna(0).replace([np.inf, -np.inf], 0)

    dtrain_improved = lgb.Dataset(X_train_improved, label=y_train)
    dval_improved = lgb.Dataset(X_test_improved, label=y_test, reference=dtrain_improved)

    model_improved = lgb.train(
        lgb_params,
        dtrain_improved,
        num_boost_round=300,
        valid_sets=[dval_improved],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)]
    )

    y_pred_improved = model_improved.predict(X_test_improved)
    auc_improved = roc_auc_score(y_test, y_pred_improved)
    acc_improved = accuracy_score(y_test, (y_pred_improved > 0.5).astype(int))

    print(f"\n改进模型结果:")
    print(f"  AUC: {auc_improved:.4f}")
    print(f"  Accuracy: {acc_improved:.4f}")

    # ========== 对比结果 ==========
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    print(f"基线 AUC:  {auc_baseline:.4f}")
    print(f"改进 AUC:  {auc_improved:.4f}")
    print(f"提升:      {(auc_improved - auc_baseline):.4f} ({(auc_improved/auc_baseline - 1)*100:+.2f}%)")
    print()
    print(f"基线 Acc:  {acc_baseline:.4f}")
    print(f"改进 Acc:  {acc_improved:.4f}")
    print(f"提升:      {(acc_improved - acc_baseline):.4f} ({(acc_improved/acc_baseline - 1)*100:+.2f}%)")

    # 特征重要性分析
    print("\n" + "="*60)
    print("微结构特征重要性")
    print("="*60)

    feat_imp = pd.DataFrame({
        'feature': model_improved.feature_name(),
        'importance': model_improved.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    micro_imp = feat_imp[feat_imp['feature'].isin(microstructure_features)]
    if len(micro_imp) > 0:
        print("\n微结构特征排名:")
        for idx, row in micro_imp.iterrows():
            rank = feat_imp.index.get_loc(idx) + 1
            print(f"  #{rank:2d} {row['feature']:20s} {row['importance']:10.1f}")
    else:
        print("  未使用微结构特征")

    # 保存结果
    result = {
        'baseline_auc': float(auc_baseline),
        'improved_auc': float(auc_improved),
        'auc_improvement': float(auc_improved - auc_baseline),
        'baseline_acc': float(acc_baseline),
        'improved_acc': float(acc_improved),
        'acc_improvement': float(acc_improved - acc_baseline),
        'n_features_baseline': len(baseline_features),
        'n_features_improved': X_train_improved.shape[1],
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
        'microstructure_features': micro_imp.to_dict('records') if len(micro_imp) > 0 else [],
    }

    import json
    with open('test_iteration1_result.json', 'w') as f:
        json.dump(result, f, indent=2)

    print("\n✓ 结果已保存到 test_iteration1_result.json")

    return result


if __name__ == '__main__':
    result = test_microstructure_features()
