"""
TabNet 模型实现 - 表格数据专用深度学习模型

TabNet 特点:
1. 可解释的注意力机制 (Feature Selection)
2. 稀疏特征选择 (Sparse Feature Learning)
3. 端到端学习，无需手工特征工程
4. 性能优于传统 GBDT (在某些任务上)

参考: https://arxiv.org/abs/1908.07442
"""

import os
import json
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    logging.warning("pytorch-tabnet not installed. Run: pip install pytorch-tabnet")

logger = logging.getLogger(__name__)


class TabNetPredictor:
    """TabNet 分类器封装"""

    def __init__(
        self,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        gamma: float = 1.5,
        n_independent: int = 2,
        n_shared: int = 2,
        lambda_sparse: float = 1e-3,
        optimizer_params: Optional[Dict] = None,
        scheduler_params: Optional[Dict] = None,
        mask_type: str = 'entmax',
        device_name: str = 'auto',
    ):
        """
        参数:
            n_d: 决策层维度
            n_a: 注意力层维度
            n_steps: 决策步骤数 (类似树的深度)
            gamma: 稀疏正则化系数
            n_independent: 独立 GLU 层数
            n_shared: 共享 GLU 层数
            lambda_sparse: 稀疏损失权重
            mask_type: 'sparsemax' 或 'entmax' (更稀疏)
            device_name: 'auto', 'cpu', 'cuda'
        """
        if not TABNET_AVAILABLE:
            raise ImportError("pytorch-tabnet not installed")

        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.lambda_sparse = lambda_sparse
        self.mask_type = mask_type

        if optimizer_params is None:
            optimizer_params = {'lr': 2e-3}
        if scheduler_params is None:
            scheduler_params = {'step_size': 10, 'gamma': 0.9}

        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.device_name = device_name
        self.model = None
        self.feature_names = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        max_epochs: int = 100,
        patience: int = 15,
        batch_size: int = 1024,
        virtual_batch_size: int = 128,
    ) -> Dict:
        """
        训练 TabNet 模型

        参数:
            X_train, y_train: 训练集
            X_val, y_val: 验证集 (如果为 None，自动从训练集切分 20%)
            max_epochs: 最大训练轮数
            patience: 早停耐心值
            batch_size: 批大小
            virtual_batch_size: 虚拟批大小 (Ghost Batch Normalization)

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

        # 初始化模型
        self.model = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            lambda_sparse=self.lambda_sparse,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=self.optimizer_params,
            scheduler_params=self.scheduler_params,
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type=self.mask_type,
            device_name=self.device_name,
            verbose=1,
            seed=42,
        )

        # 训练
        self.model.fit(
            X_train.values, y_train.values,
            eval_set=[(X_val.values, y_val.values)],
            eval_name=['val'],
            eval_metric=['auc', 'accuracy'],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            num_workers=0,
            drop_last=False,
        )

        # 评估
        y_pred_proba = self.model.predict_proba(X_val.values)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        auc = roc_auc_score(y_val, y_pred_proba)
        acc = accuracy_score(y_val, y_pred)

        logger.info(f"验证集 AUC: {auc:.4f}, Accuracy: {acc:.4f}")

        return {
            'val_auc': auc,
            'val_acc': acc,
            'history': self.model.history,
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        if self.model is None:
            raise ValueError("模型未训练")

        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.model.predict_proba(X.values)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测类别"""
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性"""
        if self.model is None:
            raise ValueError("模型未训练")

        # TabNet 的特征重要性来自注意力掩码
        importance = self.model.feature_importances_
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
        }).sort_values('importance', ascending=False)

        return df

    def save(self, path: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未训练")

        # 保存模型权重
        self.model.save_model(path)

        # 保存元数据
        meta_path = path.replace('.zip', '_meta.json')
        meta = {
            'n_d': self.n_d,
            'n_a': self.n_a,
            'n_steps': self.n_steps,
            'gamma': self.gamma,
            'n_independent': self.n_independent,
            'n_shared': self.n_shared,
            'lambda_sparse': self.lambda_sparse,
            'mask_type': self.mask_type,
            'feature_names': self.feature_names,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"模型已保存: {path}")

    def load(self, path: str):
        """加载模型"""
        # 加载元数据
        meta_path = path.replace('.zip', '_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            self.feature_names = meta['feature_names']
            self.n_d = meta['n_d']
            self.n_a = meta['n_a']
            self.n_steps = meta['n_steps']
            self.gamma = meta['gamma']
            self.n_independent = meta['n_independent']
            self.n_shared = meta['n_shared']
            self.lambda_sparse = meta['lambda_sparse']
            self.mask_type = meta['mask_type']

        # 初始化模型
        self.model = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            lambda_sparse=self.lambda_sparse,
            mask_type=self.mask_type,
            device_name=self.device_name,
        )

        # 加载权重
        self.model.load_model(path)
        logger.info(f"模型已加载: {path}")


def train_tabnet_walk_forward(
    df: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    target_col: str = 'profitable_long_5',
    train_window: int = 2000,
    test_window: int = 500,
    step: int = 250,
    **tabnet_kwargs,
) -> Tuple[TabNetPredictor, Dict]:
    """
    Walk-Forward 验证训练 TabNet

    参数:
        df: 原始数据 (用于对齐索引)
        features: 特征 DataFrame
        labels: 标签 DataFrame
        target_col: 目标列名
        train_window: 训练窗口大小
        test_window: 测试窗口大小
        step: 滑动步长
        **tabnet_kwargs: TabNet 参数

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
        model = TabNetPredictor(**tabnet_kwargs)
        model.fit(X_train, y_train, max_epochs=50, patience=10)

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
    final_model = TabNetPredictor(**tabnet_kwargs)
    final_model.fit(X, y, max_epochs=100, patience=20)

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
