"""
测试第3轮迭代：特征工程优化

策略:
- 添加更多高质量特征
- 特征交互 (feature interactions)
- 特征选择优化
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import add_all_indicators
from ma_indicators import add_moving_averages
from ml_features import compute_ml_features, compute_profit_labels


def add_advanced_features(features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """添加高级特征"""
    feat = features.copy()

    # 1. 特征交互 (Feature Interactions)
    if 'ret_1' in feat.columns and 'hvol_5' in feat.columns:
        # 收益率 / 波动率 (Sharpe-like)
        feat['ret_vol_ratio'] = feat['ret_1'] / (feat['hvol_5'] + 1e-6)

    if 'ofi' in feat.columns and 'ret_1' in feat.columns:
        # OFI 与收益率的交互
        feat['ofi_ret_interaction'] = feat['ofi'] * feat['ret_1']

    if 'rsi6' in feat.columns and 'rsi12' in feat.columns:
        # RSI 差异
        feat['rsi_diff'] = feat['rsi6'] - feat['rsi12']

    if 'kdj_k' in feat.columns and 'kdj_d' in feat.columns:
        # KDJ 动量
        feat['kdj_momentum'] = feat['kdj_k'] - feat['kdj_d']

    # 2. 价格动量的高阶特征
    if 'ret_1' in feat.columns:
        # 收益率的滚动统计
        ret_1 = feat['ret_1']
        feat['ret_1_ma5'] = ret_1.rolling(5).mean()
        feat['ret_1_std5'] = ret_1.rolling(5).std()
        feat['ret_1_skew10'] = ret_1.rolling(10).skew()
        feat['ret_1_kurt10'] = ret_1.rolling(10).kurt()

    # 3. 趋势强度特征
    if 'close' in df.columns:
        close = df['close']
        # 线性回归斜率 (趋势强度)
        for window in [10, 20]:
            slopes = []
            for i in range(len(close)):
                if i < window:
                    slopes.append(np.nan)
                else:
                    y = close.iloc[i-window:i].values
                    x = np.arange(window)
                    slope = np.polyfit(x, y, 1)[0] / close.iloc[i]
                    slopes.append(slope)
            feat[f'trend_slope_{window}'] = slopes

    # 4. 波动率聚类特征
    if 'hvol_20' in feat.columns:
        hvol = feat['hvol_20']
        # 波动率分位数
        feat['hvol_percentile'] = hvol.rolling(60).rank(pct=True)
        # 波动率状态 (低/中/高)
        feat['hvol_regime'] = pd.cut(hvol, bins=3, labels=[0, 1, 2]).astype(float)

    # 5. 量价背离特征
    if 'ret_1' in feat.columns and 'vol_change' in feat.columns:
        # 价格上涨但成交量下降 (背离)
        price_up = (feat['ret_1'] > 0).astype(int)
        vol_down = (feat['vol_change'] < 0).astype(int)
        feat['price_vol_divergence'] = price_up * vol_down

    # 6. 支撑/阻力位特征
    if 'close' in df.columns:
        close = df['close']
        high = df['high']
        low = df['low']

        # 距离最近高点/低点的距离
        feat['dist_to_recent_high_20'] = (close - high.rolling(20).max()) / close
        feat['dist_to_recent_low_20'] = (close - low.rolling(20).min()) / close

        # 突破信号
        feat['breakout_high'] = (close > high.rolling(20).max().shift(1)).astype(float)
        feat['breakout_low'] = (close < low.rolling(20).min().shift(1)).astype(float)

    # 7. 时间衰减特征 (最近的信号更重要)
    if 'ofi' in feat.columns:
        # 指数加权移动平均
        feat['ofi_ewm'] = feat['ofi'].ewm(span=10).mean()

    if 'ret_1' in feat.columns:
        feat['ret_ewm'] = feat['ret_1'].ewm(span=10).mean()

    return feat


def test_feature_engineering():
    """测试特征工程优化"""
    print("="*60)
    print("第3轮迭代测试：特征工程优化")
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
    print("\n2. 计算技术指标和基础特征...")
    df = add_all_indicators(df)
    df = add_moving_averages(df)
    features_base = compute_ml_features(df)
    print(f"   基础特征: {features_base.shape[1]} 维")

    # 添加高级特征
    print("\n3. 添加高级特征...")
    features_advanced = add_advanced_features(features_base, df)
    print(f"   高级特征: {features_advanced.shape[1]} 维")
    print(f"   新增特征: {features_advanced.shape[1] - features_base.shape[1]} 维")

    # 计算标签
    labels = compute_profit_labels(df, horizons=[5], cost_pct=0.0015)
    target_col = 'profitable_long_5'

    # 对齐数据
    valid_idx = features_advanced.notna().all(axis=1) & labels[target_col].notna()
    X_base = features_base[valid_idx].fillna(0).replace([np.inf, -np.inf], 0)
    X_advanced = features_advanced[valid_idx].fillna(0).replace([np.inf, -np.inf], 0)
    y = labels.loc[valid_idx, target_col]

    print(f"\n4. 有效样本: {len(X_advanced)}, 正类比例: {y.mean():.3f}")

    # 切分数据
    X_base_train, X_base_test, y_train, y_test = train_test_split(
        X_base, y, test_size=0.3, random_state=42, shuffle=False
    )
    X_adv_train, X_adv_test, _, _ = train_test_split(
        X_advanced, y, test_size=0.3, random_state=42, shuffle=False
    )

    print(f"   训练集: {len(X_base_train)}, 测试集: {len(X_base_test)}")

    # ========== 基线: 基础特征 ==========
    print("\n" + "="*60)
    print("基线模型: 基础特征 (76维)")
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

    dtrain_base = lgb.Dataset(X_base_train, label=y_train)
    dval_base = lgb.Dataset(X_base_test, label=y_test, reference=dtrain_base)

    model_base = lgb.train(
        lgb_params,
        dtrain_base,
        num_boost_round=300,
        valid_sets=[dval_base],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)]
    )

    y_pred_base = model_base.predict(X_base_test)
    auc_base = roc_auc_score(y_test, y_pred_base)
    acc_base = accuracy_score(y_test, (y_pred_base > 0.5).astype(int))

    print(f"\n基线模型结果:")
    print(f"  AUC: {auc_base:.4f}")
    print(f"  Accuracy: {acc_base:.4f}")

    # ========== 改进: 高级特征 ==========
    print("\n" + "="*60)
    print(f"改进模型: 高级特征 ({X_advanced.shape[1]}维)")
    print("="*60)

    dtrain_adv = lgb.Dataset(X_adv_train, label=y_train)
    dval_adv = lgb.Dataset(X_adv_test, label=y_test, reference=dtrain_adv)

    model_adv = lgb.train(
        lgb_params,
        dtrain_adv,
        num_boost_round=300,
        valid_sets=[dval_adv],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)]
    )

    y_pred_adv = model_adv.predict(X_adv_test)
    auc_adv = roc_auc_score(y_test, y_pred_adv)
    acc_adv = accuracy_score(y_test, (y_pred_adv > 0.5).astype(int))

    print(f"\n改进模型结果:")
    print(f"  AUC: {auc_adv:.4f}")
    print(f"  Accuracy: {acc_adv:.4f}")

    # ========== 对比结果 ==========
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    print(f"基线 AUC:  {auc_base:.4f}")
    print(f"改进 AUC:  {auc_adv:.4f}")
    print(f"提升:      {(auc_adv - auc_base):.4f} ({(auc_adv/auc_base - 1)*100:+.2f}%)")
    print()
    print(f"基线 Acc:  {acc_base:.4f}")
    print(f"改进 Acc:  {acc_adv:.4f}")
    print(f"提升:      {(acc_adv - acc_base):.4f} ({(acc_adv/acc_base - 1)*100:+.2f}%)")

    # 特征重要性分析
    print("\n" + "="*60)
    print("新增特征重要性 Top 10")
    print("="*60)

    feat_imp = pd.DataFrame({
        'feature': model_adv.feature_name(),
        'importance': model_adv.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    new_features = [c for c in X_advanced.columns if c not in X_base.columns]
    new_feat_imp = feat_imp[feat_imp['feature'].isin(new_features)]

    if len(new_feat_imp) > 0:
        print("\n新增特征排名:")
        for idx, row in new_feat_imp.head(10).iterrows():
            rank = feat_imp.index.get_loc(idx) + 1
            print(f"  #{rank:2d} {row['feature']:30s} {row['importance']:10.1f}")
    else:
        print("  未使用新增特征")

    # 保存结果
    result = {
        'baseline_auc': float(auc_base),
        'improved_auc': float(auc_adv),
        'auc_improvement': float(auc_adv - auc_base),
        'baseline_acc': float(acc_base),
        'improved_acc': float(acc_adv),
        'acc_improvement': float(acc_adv - acc_base),
        'n_features_baseline': X_base.shape[1],
        'n_features_improved': X_advanced.shape[1],
        'n_new_features': len(new_features),
        'n_samples_train': len(X_base_train),
        'n_samples_test': len(X_base_test),
        'top_new_features': new_feat_imp.head(10).to_dict('records') if len(new_feat_imp) > 0 else [],
    }

    import json
    with open('test_iteration3_result.json', 'w') as f:
        json.dump(result, f, indent=2)

    print("\n✓ 结果已保存到 test_iteration3_result.json")

    return result


if __name__ == '__main__':
    result = test_feature_engineering()
