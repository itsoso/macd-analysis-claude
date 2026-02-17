"""
ETH/USDT 时间序列预测模型 v2

核心改进 (vs v1):
  1. 利润化标签: 扣手续费后仍盈利才算正类, 解决"涨0.01%也算涨"的噪声
  2. 特征精选: 相关性去冗余 + 重要性 top-30, 避免 69 特征 / 720 样本过拟合
  3. 强正则化: num_leaves=15, min_child_samples=50, DART + 高 dropout
  4. 多尺度集成: 3h/5h/12h/24h 四个模型投票, 跨尺度共识更稳健
  5. Stacking: LightGBM + XGBoost + Ridge 三模型元学习器
  6. 扩展窗口: 训练窗口随时间扩展而非固定 720, 用更多历史
"""

import os
import json
import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')

logger = logging.getLogger(__name__)


# ================================================================
# 配置
# ================================================================
@dataclass
class MLConfig:
    """ML 模型配置"""
    # 预测目标
    target_horizon: int = 5
    classification: bool = True
    use_profit_labels: bool = True    # v2: 利润化标签
    cost_pct: float = 0.0015          # 单次手续费+滑点

    # LightGBM 参数 (v2: 强正则但不过度抑制学习)
    lgb_params: Dict = field(default_factory=lambda: {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 20,              # v2: 31→20 (15太少)
        'learning_rate': 0.05,         # 保持原速，靠正则防过拟合
        'feature_fraction': 0.65,      # 适度随机
        'bagging_fraction': 0.75,
        'bagging_freq': 3,
        'min_child_samples': 30,       # v2: 20→30 (50太严格)
        'lambda_l1': 0.5,             # 适度 L1
        'lambda_l2': 1.0,             # 强 L2
        'is_unbalance': True,         # v2: 自动处理类别不平衡
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    })
    lgb_num_boost_round: int = 300
    lgb_early_stopping_rounds: int = 30

    # 特征精选
    use_feature_selection: bool = True
    feature_corr_threshold: float = 0.85
    feature_top_n: int = 30            # v2: 69→30 核心特征

    # Walk-forward 参数
    train_window: int = 1200           # v2: 720→1200 (50天) 更多数据
    min_train_window: int = 720        # 最小训练窗口
    expanding_window: bool = True      # v2: 扩展窗口而非固定
    val_window: int = 168
    retrain_interval: int = 120        # v2: 168→120 更频繁重训练
    purge_gap: int = 24
    min_train_samples: int = 500

    # 多尺度集成
    use_multi_horizon: bool = True     # v2: 多尺度
    horizons: List = field(default_factory=lambda: [3, 5, 12, 24])
    horizon_weights: Dict = field(default_factory=lambda: {3: 0.15, 5: 0.35, 12: 0.30, 24: 0.20})

    # Stacking (WF 中默认关闭以保持速度, 生产模型用 stacking)
    use_stacking: bool = False
    stacking_models: List = field(default_factory=lambda: ['lgb', 'xgb', 'ridge'])

    # LSTM 参数 (保留，可选)
    use_lstm: bool = False
    lstm_seq_len: int = 48
    lstm_hidden_dim: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_lr: float = 0.001
    lstm_epochs: int = 30
    lstm_batch_size: int = 64

    # 集成
    lgb_weight: float = 0.7
    lstm_weight: float = 0.3

    # 信号生成 (v2: 更宽的中性区间)
    long_threshold: float = 0.58       # v2: 0.60→0.58
    short_threshold: float = 0.42      # v2: 0.40→0.42
    score_scale: float = 100.0

    # 路径
    model_dir: str = 'data/ml_models'


# ================================================================
# LightGBM 模型 (强正则化版)
# ================================================================
class LGBPredictor:
    """LightGBM 方向预测器"""

    def __init__(self, config: MLConfig):
        self.config = config
        self.model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        self.train_metrics: Dict = {}

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict:
        self.feature_names = list(X_train.columns)
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        if X_val is not None:
            X_val = X_val.replace([np.inf, -np.inf], np.nan)

        dtrain = lgb.Dataset(X_train, label=y_train)
        valid_sets = [dtrain]
        valid_names = ['train']
        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            valid_sets.append(dval)
            valid_names.append('val')

        callbacks = [lgb.log_evaluation(period=0)]
        if X_val is not None:
            callbacks.append(lgb.early_stopping(self.config.lgb_early_stopping_rounds))

        self.model = lgb.train(
            self.config.lgb_params, dtrain,
            num_boost_round=self.config.lgb_num_boost_round,
            valid_sets=valid_sets, valid_names=valid_names,
            callbacks=callbacks,
        )

        imp = self.model.feature_importance(importance_type='gain')
        self.feature_importance = dict(zip(self.feature_names, imp.tolist()))
        self.train_metrics = {
            'best_iteration': self.model.best_iteration,
            'num_features': len(self.feature_names),
            'train_samples': len(X_train),
            'val_samples': len(X_val) if X_val is not None else 0,
        }
        return self.train_metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("模型未训练")
        X = X[self.feature_names].replace([np.inf, -np.inf], np.nan)
        return self.model.predict(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        result = np.zeros(len(proba), dtype=int)
        result[proba >= self.config.long_threshold] = 1
        result[proba <= self.config.short_threshold] = -1
        return result

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        meta = {
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'train_metrics': self.train_metrics,
        }
        with open(path + '.meta.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)

    def load(self, path: str):
        self.model = lgb.Booster(model_file=path)
        meta_path = path + '.meta.json'
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.feature_names = meta.get('feature_names', [])
            self.feature_importance = meta.get('feature_importance', {})
            self.train_metrics = meta.get('train_metrics', {})

    def top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        return sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:n]


# ================================================================
# Stacking 集成 (LightGBM + XGBoost + Ridge)
# ================================================================
class StackingPredictor:
    """
    三模型 Stacking 集成:
      - LightGBM (DART, 强正则)
      - XGBoost (保守配置)
      - Ridge Logistic (线性基线)
    元学习器: 简单加权平均 (避免在小数据上再训练元模型过拟合)
    """

    def __init__(self, config: MLConfig):
        self.config = config
        self.models: Dict = {}
        self.weights: Dict = {'lgb': 0.45, 'xgb': 0.35, 'ridge': 0.20}
        self.feature_names: List[str] = []
        self._trained = False

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict:
        self.feature_names = list(X_train.columns)
        X_tr = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_vl = X_val.replace([np.inf, -np.inf], np.nan).fillna(0) if X_val is not None else None
        metrics = {}

        # 1. LightGBM
        lgb_model = LGBPredictor(self.config)
        lgb_metrics = lgb_model.train(X_train, y_train, X_val, y_val)
        self.models['lgb'] = lgb_model
        metrics['lgb'] = lgb_metrics

        # 2. XGBoost
        try:
            import xgboost as xgb
            dtrain = xgb.DMatrix(X_tr.values, label=y_train.values,
                                 feature_names=self.feature_names)
            xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 4,             # 浅树
                'learning_rate': 0.03,
                'subsample': 0.7,
                'colsample_bytree': 0.6,
                'min_child_weight': 50,
                'reg_alpha': 1.0,
                'reg_lambda': 1.0,
                'seed': 123,
                'verbosity': 0,
            }
            evals = [(dtrain, 'train')]
            if X_vl is not None and y_val is not None:
                dval = xgb.DMatrix(X_vl.values, label=y_val.values,
                                   feature_names=self.feature_names)
                evals.append((dval, 'val'))

            xgb_model = xgb.train(
                xgb_params, dtrain, num_boost_round=150,
                evals=evals, early_stopping_rounds=20,
                verbose_eval=False,
            )
            self.models['xgb'] = xgb_model
            metrics['xgb'] = {'best_iteration': xgb_model.best_iteration}
        except Exception as e:
            logger.warning(f"XGBoost 训练失败: {e}")
            self.weights['lgb'] += self.weights.get('xgb', 0)
            self.weights['xgb'] = 0

        # 3. Ridge Logistic Regression
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            self._ridge_scaler = StandardScaler()
            X_scaled = self._ridge_scaler.fit_transform(X_tr.values)
            ridge = LogisticRegression(
                penalty='l2', C=0.1, solver='lbfgs',
                max_iter=500, random_state=42,
            )
            ridge.fit(X_scaled, y_train.values)
            self.models['ridge'] = ridge
            metrics['ridge'] = {'C': 0.1}
        except Exception as e:
            logger.warning(f"Ridge 训练失败: {e}")
            self.weights['lgb'] += self.weights.get('ridge', 0)
            self.weights['ridge'] = 0

        # 归一化权重
        total_w = sum(self.weights.values())
        if total_w > 0:
            self.weights = {k: v / total_w for k, v in self.weights.items()}

        # 在验证集上自适应调整权重
        if X_val is not None and y_val is not None and len(y_val) > 20:
            self._calibrate_weights(X_val, y_val)

        self._trained = True
        metrics['weights'] = dict(self.weights)
        return metrics

    def _calibrate_weights(self, X_val: pd.DataFrame, y_val: pd.Series):
        """根据验证集 AUC 自适应调整权重"""
        from sklearn.metrics import roc_auc_score
        aucs = {}
        for name in self.models:
            try:
                proba = self._predict_single(name, X_val)
                valid = y_val.notna()
                aucs[name] = roc_auc_score(y_val[valid], proba[valid.values])
            except Exception:
                aucs[name] = 0.5

        # AUC 越高权重越大 (相对于 0.5 基线)
        excess = {k: max(0, v - 0.5) for k, v in aucs.items()}
        total = sum(excess.values())
        if total > 0.01:
            self.weights = {k: v / total for k, v in excess.items()}
        # 兜底: 如果所有模型都很差, 等权
        else:
            n = len(self.models)
            self.weights = {k: 1.0 / n for k in self.models}

    def _predict_single(self, name: str, X: pd.DataFrame) -> np.ndarray:
        X_clean = X[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)

        if name == 'lgb':
            return self.models['lgb'].predict_proba(X)
        elif name == 'xgb':
            import xgboost as xgb
            dtest = xgb.DMatrix(X_clean.values, feature_names=self.feature_names)
            return self.models['xgb'].predict(dtest)
        elif name == 'ridge':
            X_scaled = self._ridge_scaler.transform(X_clean.values)
            return self.models['ridge'].predict_proba(X_scaled)[:, 1]
        return np.full(len(X), 0.5)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("模型未训练")
        proba = np.zeros(len(X))
        for name, weight in self.weights.items():
            if name in self.models and weight > 0:
                proba += weight * self._predict_single(name, X)
        return np.clip(proba, 0, 1)

    def top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        if 'lgb' in self.models:
            return self.models['lgb'].top_features(n)
        return []


# ================================================================
# 多尺度集成预测器
# ================================================================
class MultiHorizonPredictor:
    """
    多时间尺度集成: 训练 3h/5h/12h/24h 四个模型, 加权投票。

    原理: 不同时间尺度捕捉不同市场特征
      - 3h: 短期动量/反转
      - 5h: 中短期趋势
      - 12h: 日内周期
      - 24h: 中期趋势
    只有多个尺度共识看涨/看跌时, 才给出强信号。
    """

    def __init__(self, config: MLConfig):
        self.config = config
        self.horizon_models: Dict[int, StackingPredictor] = {}
        self.selected_features: Dict[int, List[str]] = {}
        self._trained = False

    def train(self, features: pd.DataFrame, labels_df: pd.DataFrame,
              X_val: Optional[pd.DataFrame] = None,
              y_val_df: Optional[pd.DataFrame] = None) -> Dict:
        """为每个时间尺度训练独立模型"""
        from ml_features import select_features

        metrics = {}
        horizons = self.config.horizons

        for h in horizons:
            target_col = f'profitable_long_{h}'
            if target_col not in labels_df.columns:
                target_col = f'fwd_dir_{h}'
            if target_col not in labels_df.columns:
                continue

            y = labels_df[target_col]
            valid = y.notna()

            # 特征精选 (每个尺度独立)
            if self.config.use_feature_selection:
                sel = select_features(
                    features[valid], y[valid],
                    corr_threshold=self.config.feature_corr_threshold,
                    importance_top_n=self.config.feature_top_n,
                )
                self.selected_features[h] = sel
                X_train_h = features[sel]
            else:
                self.selected_features[h] = list(features.columns)
                X_train_h = features

            # 训练
            y_val_h = None
            X_val_h = None
            if y_val_df is not None and target_col in y_val_df.columns:
                y_val_h = y_val_df[target_col]
                X_val_h = X_val[self.selected_features[h]] if X_val is not None else None

            model = StackingPredictor(self.config) if self.config.use_stacking else LGBPredictor(self.config)

            valid_train = y.notna()
            m = model.train(
                X_train_h[valid_train], y[valid_train],
                X_val_h, y_val_h,
            )
            self.horizon_models[h] = model
            metrics[f'h{h}'] = m

        self._trained = True
        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """多尺度加权投票"""
        if not self._trained:
            raise RuntimeError("模型未训练")

        weights = self.config.horizon_weights
        proba = np.zeros(len(X))
        total_w = 0

        for h, model in self.horizon_models.items():
            w = weights.get(h, 0.25)
            X_h = X[self.selected_features[h]]
            proba += w * model.predict_proba(X_h)
            total_w += w

        if total_w > 0:
            proba /= total_w

        return np.clip(proba, 0, 1)

    def top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """合并所有尺度的特征重要性"""
        merged = {}
        for h, model in self.horizon_models.items():
            for feat, score in model.top_features(50):
                merged[feat] = merged.get(feat, 0) + score
        return sorted(merged.items(), key=lambda x: x[1], reverse=True)[:n]


# ================================================================
# 集成预测器 (统一入口)
# ================================================================
class EnsemblePredictor:
    """
    统一预测器入口。

    v2: 支持多尺度 + stacking 模式
    """

    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or MLConfig()
        self.lgb = LGBPredictor(self.config)
        self.multi_horizon: Optional[MultiHorizonPredictor] = None
        self.stacking: Optional[StackingPredictor] = None
        self.lstm = None
        self.is_trained = False
        self._selected_features: List[str] = []

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              labels_df: Optional[pd.DataFrame] = None,
              y_val_df: Optional[pd.DataFrame] = None) -> Dict:
        results = {}

        # 特征精选
        if self.config.use_feature_selection:
            from ml_features import select_features
            self._selected_features = select_features(
                X_train, y_train,
                corr_threshold=self.config.feature_corr_threshold,
                importance_top_n=self.config.feature_top_n,
            )
            X_train_sel = X_train[self._selected_features]
            X_val_sel = X_val[self._selected_features] if X_val is not None else None
        else:
            self._selected_features = list(X_train.columns)
            X_train_sel = X_train
            X_val_sel = X_val

        # 多尺度模式
        if self.config.use_multi_horizon and labels_df is not None:
            self.multi_horizon = MultiHorizonPredictor(self.config)
            mh_metrics = self.multi_horizon.train(
                X_train, labels_df.loc[X_train.index],
                X_val, y_val_df,
            )
            results['multi_horizon'] = mh_metrics
        # Stacking 模式
        elif self.config.use_stacking:
            self.stacking = StackingPredictor(self.config)
            st_metrics = self.stacking.train(X_train_sel, y_train, X_val_sel, y_val)
            results['stacking'] = st_metrics
        # 单 LightGBM
        else:
            lgb_metrics = self.lgb.train(X_train_sel, y_train, X_val_sel, y_val)
            results['lgb'] = lgb_metrics

        self.is_trained = True
        return results

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("模型未训练")

        if self.multi_horizon:
            return self.multi_horizon.predict_proba(X)
        elif self.stacking:
            X_sel = X[self._selected_features]
            return self.stacking.predict_proba(X_sel)
        else:
            X_sel = X[self._selected_features]
            return self.lgb.predict_proba(X_sel)

    def predict_scores(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        proba = self.predict_proba(X)
        scale = self.config.score_scale
        direction = np.zeros(len(proba), dtype=int)
        direction[proba >= self.config.long_threshold] = 1
        direction[proba <= self.config.short_threshold] = -1
        return {
            'bull_prob': proba,
            'ml_long_score': proba * scale,
            'ml_short_score': (1 - proba) * scale,
            'direction': direction,
        }

    def save(self, model_dir: Optional[str] = None):
        model_dir = model_dir or self.config.model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.lgb.save(os.path.join(model_dir, 'lgb_model.txt'))
        with open(os.path.join(model_dir, 'ensemble_config.json'), 'w') as f:
            json.dump({
                'target_horizon': self.config.target_horizon,
                'lgb_weight': self.config.lgb_weight,
                'lstm_weight': self.config.lstm_weight if self.lstm else 0,
                'use_lstm': self.lstm is not None,
                'long_threshold': self.config.long_threshold,
                'short_threshold': self.config.short_threshold,
                'selected_features': self._selected_features,
            }, f, indent=2)

    def load(self, model_dir: Optional[str] = None):
        model_dir = model_dir or self.config.model_dir
        self.lgb.load(os.path.join(model_dir, 'lgb_model.txt'))
        config_path = os.path.join(model_dir, 'ensemble_config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            self._selected_features = cfg.get('selected_features', [])
            self.config.lgb_weight = cfg.get('lgb_weight', 0.7)
        self.is_trained = True

    def top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        if self.multi_horizon:
            return self.multi_horizon.top_features(n)
        elif self.stacking:
            return self.stacking.top_features(n)
        return self.lgb.top_features(n)


# ================================================================
# Walk-Forward 训练引擎 v2
# ================================================================
class WalkForwardEngine:
    """
    Walk-forward v2: 支持扩展窗口 + 多尺度 + 利润化标签。
    """

    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or MLConfig()
        self.fold_results: List[Dict] = []

    def run(self, features: pd.DataFrame, labels: pd.Series,
            labels_df: Optional[pd.DataFrame] = None,
            verbose: bool = True) -> pd.DataFrame:
        cfg = self.config
        n = len(features)
        all_preds = pd.Series(index=features.index, dtype=float)
        all_preds[:] = np.nan

        start_idx = cfg.min_train_window + cfg.val_window + cfg.purge_gap
        if start_idx >= n:
            raise ValueError(f"数据不足: 需要至少 {start_idx} 条, 只有 {n} 条")

        # v2: 特征精选只做一次 (在第一个训练窗口上), 后续 fold 复用
        self._cached_features = None
        if cfg.use_feature_selection:
            from ml_features import select_features
            init_end = start_idx - cfg.purge_gap - cfg.val_window
            init_start = max(0, init_end - cfg.train_window)
            X_init = features.iloc[init_start:init_end]
            y_init = labels.iloc[init_start:init_end]
            valid = y_init.notna()
            self._cached_features = select_features(
                X_init[valid], y_init[valid],
                corr_threshold=cfg.feature_corr_threshold,
                importance_top_n=cfg.feature_top_n,
            )
            if verbose:
                print(f"  特征精选: {features.shape[1]} → {len(self._cached_features)} 核心特征")

        fold = 0
        cursor = start_idx

        while cursor < n:
            train_end = cursor - cfg.purge_gap - cfg.val_window
            # v2: 扩展窗口 (使用更多历史)
            if cfg.expanding_window:
                train_start = max(0, train_end - cfg.train_window)
            else:
                train_start = max(0, train_end - cfg.min_train_window)

            val_start = train_end + cfg.purge_gap
            val_end = cursor
            test_start = cursor
            test_end = min(n, cursor + cfg.retrain_interval)

            if train_end - train_start < cfg.min_train_samples:
                cursor += cfg.retrain_interval
                continue

            X_train = features.iloc[train_start:train_end]
            y_train = labels.iloc[train_start:train_end]
            X_val = features.iloc[val_start:val_end]
            y_val = labels.iloc[val_start:val_end]
            X_test = features.iloc[test_start:test_end]

            # 过滤 NaN
            valid_train = y_train.notna()
            X_train = X_train[valid_train]
            y_train = y_train[valid_train]
            valid_val = y_val.notna()
            X_val = X_val[valid_val]
            y_val = y_val[valid_val]

            if len(X_train) < cfg.min_train_samples:
                cursor += cfg.retrain_interval
                continue

            # 构建多尺度标签
            train_labels_df = None
            val_labels_df = None
            if cfg.use_multi_horizon and labels_df is not None:
                train_labels_df = labels_df.iloc[train_start:train_end][valid_train]
                val_labels_df = labels_df.iloc[val_start:val_end][valid_val] if len(X_val) > 10 else None

            # 训练 (用缓存的精选特征, 避免每 fold 重复)
            import dataclasses
            fold_cfg = dataclasses.replace(cfg, use_feature_selection=False)

            if self._cached_features:
                X_train_sel = X_train[self._cached_features]
                X_val_sel = X_val[self._cached_features] if len(X_val) > 10 else None
            else:
                X_train_sel = X_train
                X_val_sel = X_val if len(X_val) > 10 else None

            model = EnsemblePredictor(fold_cfg)
            metrics = model.train(
                X_train_sel, y_train,
                X_val_sel,
                y_val if len(y_val) > 10 else None,
                labels_df=train_labels_df,
                y_val_df=val_labels_df,
            )

            # OOS 预测
            X_test_sel = X_test[self._cached_features] if self._cached_features else X_test
            proba = model.predict_proba(X_test_sel)
            all_preds.iloc[test_start:test_end] = proba

            # 验证集 AUC
            val_auc = self._compute_auc(y_val, model.predict_proba(X_val_sel)) if len(X_val) > 10 and X_val_sel is not None else 0

            fold_info = {
                'fold': fold,
                'train_range': f"{features.index[train_start]} ~ {features.index[train_end - 1]}",
                'test_range': f"{features.index[test_start]} ~ {features.index[min(test_end - 1, n - 1)]}",
                'train_size': len(X_train),
                'val_auc': round(val_auc, 4),
                'test_size': test_end - test_start,
                'top_features': model.top_features(10),
            }
            self.fold_results.append(fold_info)

            if verbose:
                print(f"  Fold {fold}: train={len(X_train)} "
                      f"val_AUC={val_auc:.4f} test={test_end - test_start} "
                      f"[{features.index[test_start].strftime('%m-%d')} ~ "
                      f"{features.index[min(test_end - 1, n - 1)].strftime('%m-%d')}]")

            fold += 1
            cursor += cfg.retrain_interval

        # 构建结果
        result_df = pd.DataFrame({
            'bull_prob': all_preds,
            'ml_long_score': all_preds * cfg.score_scale,
            'ml_short_score': (1 - all_preds) * cfg.score_scale,
        }, index=features.index)

        result_df['direction'] = 0
        result_df.loc[result_df['bull_prob'] >= cfg.long_threshold, 'direction'] = 1
        result_df.loc[result_df['bull_prob'] <= cfg.short_threshold, 'direction'] = -1

        return result_df

    @staticmethod
    def _compute_auc(y_true: pd.Series, y_pred: np.ndarray) -> float:
        try:
            from sklearn.metrics import roc_auc_score
            valid = y_true.notna()
            if valid.sum() < 10:
                return 0.5
            return roc_auc_score(y_true[valid], y_pred[valid.values])
        except Exception:
            return 0.5

    def summary(self) -> Dict:
        if not self.fold_results:
            return {}
        aucs = [f['val_auc'] for f in self.fold_results]
        return {
            'total_folds': len(self.fold_results),
            'avg_val_auc': round(np.mean(aucs), 4),
            'min_val_auc': round(np.min(aucs), 4),
            'max_val_auc': round(np.max(aucs), 4),
            'std_val_auc': round(np.std(aucs), 4),
        }
