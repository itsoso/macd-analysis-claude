"""
CatBoost GPU 模型实现

CatBoost 特点:
1. 对称树 (Oblivious Trees) - 更快的推理速度
2. 有序提升 (Ordered Boosting) - 减少过拟合
3. 原生支持类别特征
4. GPU 加速
"""

import os
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("catboost not installed. Run: pip install catboost")

logger = logging.getLogger(__name__)


class CatBoostPredictor:
    """CatBoost 分类器封装"""

    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.05,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
        bagging_temperature: float = 1.0,
        border_count: int = 128,
        task_type: str = 'GPU',
        devices: str = '0',
        verbose: int = 50,
    ):
        """
        参数:
            iterations: 迭代次数
            learning_rate: 学习率
            depth: 树深度
            l2_leaf_reg: L2 正则化
            random_strength: 随机强度
            bagging_temperature: Bagging 温度
            border_count: 分箱数量
            task_type: 'GPU' 或 'CPU'
            devices: GPU 设备 ID
            verbose: 日志频率
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("catboost not installed")

        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.border_count = border_count
        self.task_type = task_type
        self.devices = devices
        self.verbose = verbose
        self.model = None
        self.feature_names = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 30,
    ) -> Dict:
        """
        训练 CatBoost 模型

        参数:
            X_train, y_train: 训练集
            X_val, y_val: 验证集 (如果为 None，自动从训练集切分 20%)
            early_stopping_rounds: 早停轮数

        返回:
            训练历史字典
        """
        self.feature_names = X_train.columns.tolist()

        # 处理缺失值和无穷值
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        if X_val is not None:
            X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 如果没有验证集，自动切分
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )

        logger.info(f"训练集: {len(X_train)}, 验证集: {len(X_val)}")
        logger.info(f"正类比例: 训练={y_train.mean():.3f}, 验证={y_val.mean():.3f}")

        # 创建 Pool
        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)

        # 初始化模型
        self.model = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            random_strength=self.random_strength,
            bagging_temperature=self.bagging_temperature,
            border_count=self.border_count,
            loss_function='Logloss',
            eval_metric='AUC',
            task_type=self.task_type,
            devices=self.devices,
            early_stopping_rounds=early_stopping_rounds,
            verbose=self.verbose,
            random_seed=42,
        )

        # 训练
        self.model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
        )

        # 评估
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        auc = roc_auc_score(y_val, y_pred_proba)
        acc = accuracy_score(y_val, y_pred)

        logger.info(f"验证集 AUC: {auc:.4f}, Accuracy: {acc:.4f}")

        return {
            'val_auc': auc,
            'val_acc': acc,
            'best_iteration': self.model.best_iteration_,
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        if self.model is None:
            raise ValueError("模型未训练")

        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测类别"""
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性"""
        if self.model is None:
            raise ValueError("模型未训练")

        importance = self.model.get_feature_importance()
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
        }).sort_values('importance', ascending=False)

        return df

    def save(self, path: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未训练")

        self.model.save_model(path)
        logger.info(f"模型已保存: {path}")

    def load(self, path: str):
        """加载模型"""
        self.model = CatBoostClassifier()
        self.model.load_model(path)
        logger.info(f"模型已加载: {path}")


def train_catboost_walk_forward(
    df: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    target_col: str = 'profitable_long_5',
    train_window: int = 2000,
    test_window: int = 500,
    step: int = 250,
    **catboost_kwargs,
) -> Tuple[CatBoostPredictor, Dict]:
    """
    Walk-Forward 验证训练 CatBoost

    参数:
        df: 原始数据 (用于对齐索引)
        features: 特征 DataFrame
        labels: 标签 DataFrame
        target_col: 目标列名
        train_window: 训练窗口大小
        test_window: 测试窗口大小
        step: 滑动步长
        **catboost_kwargs: CatBoost 参数

    返回:
        (最终模型, 评估结果字典)
    """
    # 对齐数据
    valid_idx = features.notna().all(axis=1) & labels[target_col].notna()
    X = features[valid_idx]
    y = labels.loc[valid_idx, target_col]

    logger.info(f"有效样本数: {len(X)}, 正类比例: {y.mean():.3f}")

    # Walk-Forward 切分
    results = []
    start_idx = 0

    while start_idx + train_window + test_window <= len(X):
        train_end = start_idx + train_window
        test_end = train_end + test_window

        X_train = X.iloc[start_idx:train_end]
        y_train = y.iloc[start_idx:train_end]
        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

        logger.info(f"\n=== Fold {len(results)+1}: 训练 [{start_idx}:{train_end}], 测试 [{train_end}:{test_end}] ===")

        # 训练模型
        model = CatBoostPredictor(**catboost_kwargs)
        model.fit(X_train, y_train, early_stopping_rounds=30)

        # 测试
        y_pred_proba = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, (y_pred_proba > 0.5).astype(int))

        logger.info(f"测试集 AUC: {auc:.4f}, Accuracy: {acc:.4f}")

        results.append({
            'fold': len(results) + 1,
            'train_start': start_idx,
            'train_end': train_end,
            'test_start': train_end,
            'test_end': test_end,
            'auc': auc,
            'acc': acc,
        })

        start_idx += step

    # 用全部数据训练最终模型
    logger.info("\n=== 训练最终模型 (全部数据) ===")
    final_model = CatBoostPredictor(**catboost_kwargs)
    final_model.fit(X, y, early_stopping_rounds=50)

    # 汇总结果
    summary = {
        'n_folds': len(results),
        'mean_auc': np.mean([r['auc'] for r in results]),
        'std_auc': np.std([r['auc'] for r in results]),
        'mean_acc': np.mean([r['acc'] for r in results]),
        'folds': results,
    }

    logger.info(f"\nWalk-Forward 平均 AUC: {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}")

    return final_model, summary
