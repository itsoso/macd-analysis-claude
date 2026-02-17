"""
ML 分位数回归预测器

核心理念 (来自 GPT Pro 建议):
  预测收益的分布而非方向 → 输出 5 个分位点 (q05, q25, q50, q75, q95)
  这直接给出:
    - 期望收益: q50 (中位数)
    - 风险估计: (q75-q25) 作为 IQR, (q95-q05) 作为尾部范围
    - 尾部风险: q05 = 95% VaR (做多时的最大可能亏损)
    - 交易门槛: 只有 q25 > 交易成本 才做多 (75% 概率盈利)
    - Kelly 仓位: w = μ/σ² 用分位数直接估计

同时增强衍生品特征:
  funding_z, oi_z, basis_z, taker_imbalance → 合约专属 alpha

训练方式:
  5 个独立的 LightGBM quantile 模型, 各自优化对应的 pinball loss
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


QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
MODEL_DIR = os.path.join('data', 'ml_models')


@dataclass
class QuantileConfig:
    """分位数回归配置"""
    horizons: List[int] = field(default_factory=lambda: [5, 12])  # 预测 5h, 12h
    quantiles: List[float] = field(default_factory=lambda: [0.05, 0.25, 0.50, 0.75, 0.95])

    # LightGBM 参数 (适度正则, 回归任务允许更多叶子)
    lgb_base_params: Dict = field(default_factory=lambda: {
        'boosting_type': 'gbdt',
        'num_leaves': 20,
        'learning_rate': 0.05,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.75,
        'bagging_freq': 3,
        'min_child_samples': 40,
        'lambda_l1': 0.5,
        'lambda_l2': 1.0,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    })
    num_boost_round: int = 150          # 减少最大轮数 (加速)
    early_stopping_rounds: int = 20

    # Walk-forward (每 10 天重训, 平衡精度和速度)
    min_train_window: int = 1200
    expanding: bool = True
    val_window: int = 168
    retrain_interval: int = 240       # 10 天重训 (120→240, 减少 fold 数)
    purge_gap: int = 36

    # 交易决策参数 (GPT Pro 建议: 成本感知门槛)
    roundtrip_cost: float = 0.003     # 来回手续费+滑点 0.3%
    funding_per_hour: float = 0.00001  # 平均 funding 成本/小时
    min_edge_ratio: float = 2.0       # 最低要求: E[return]/cost > 2
    kelly_fraction: float = 0.25      # 凯利比例的使用分数 (1/4 Kelly, 保守)
    max_position: float = 0.30        # 最大仓位占比

    model_dir: str = 'data/ml_models'


def compute_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    增强版特征集: regime 特征 + 衍生品特征 (15 维)

    比 regime-only 的 10 维多了 5 个衍生品专属特征:
      - funding_z: 资金费率偏离 (拥挤信号)
      - oi_z: 持仓量偏离 (加/减仓信号)
      - basis_z: 基差偏离 (期现溢价)
      - taker_imbalance: 主动买卖失衡 (方向压力)
      - volume_surprise: 成交量异常度 (事件探测)
    """
    from ml_regime import compute_regime_features
    feat = compute_regime_features(df)

    close = df['close'].astype(float)
    volume = df['volume'].astype(float)
    lookback = 60

    # ── 衍生品特征 ──
    if 'funding_rate' in df.columns:
        fr = df['funding_rate'].astype(float).fillna(0)
        fr_mean = fr.rolling(lookback, min_periods=10).mean()
        fr_std = fr.rolling(lookback, min_periods=10).std().replace(0, np.nan)
        feat['funding_z'] = ((fr - fr_mean) / fr_std).clip(-5, 5).fillna(0)
    else:
        feat['funding_z'] = 0.0

    if 'open_interest_value' in df.columns:
        oi = df['open_interest_value'].astype(float)
        oi_pct = oi.pct_change()
        oi_mean = oi_pct.rolling(lookback, min_periods=10).mean()
        oi_std = oi_pct.rolling(lookback, min_periods=10).std().replace(0, np.nan)
        feat['oi_z'] = ((oi_pct - oi_mean) / oi_std).clip(-5, 5).fillna(0)
    else:
        feat['oi_z'] = 0.0

    # basis proxy: (close - VWAP) / close
    if 'vwap' in df.columns:
        basis = (close - df['vwap'].astype(float)) / close
    else:
        vwap = (df['quote_volume'].astype(float) / volume).replace(0, np.nan)
        basis = (close - vwap) / close
    basis_mean = basis.rolling(lookback, min_periods=10).mean()
    basis_std = basis.rolling(lookback, min_periods=10).std().replace(0, np.nan)
    feat['basis_z'] = ((basis - basis_mean) / basis_std).clip(-5, 5).fillna(0)

    # taker imbalance
    if 'taker_buy_base' in df.columns:
        tbr = df['taker_buy_base'].astype(float) / volume.replace(0, np.nan)
        feat['taker_imbalance'] = ((tbr.fillna(0.5) - 0.5) * 2).clip(-1, 1)
    elif 'taker_buy_quote' in df.columns and 'quote_volume' in df.columns:
        tbr = df['taker_buy_quote'].astype(float) / df['quote_volume'].astype(float).replace(0, np.nan)
        feat['taker_imbalance'] = ((tbr.fillna(0.5) - 0.5) * 2).clip(-1, 1)
    else:
        feat['taker_imbalance'] = 0.0

    # volume surprise (z-score of log volume)
    log_vol = np.log1p(volume)
    vol_mean = log_vol.rolling(lookback, min_periods=10).mean()
    vol_std = log_vol.rolling(lookback, min_periods=10).std().replace(0, np.nan)
    feat['volume_surprise'] = ((log_vol - vol_mean) / vol_std).clip(-5, 5).fillna(0)

    return feat


def compute_return_labels(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """
    计算对数收益标签 (回归目标, 非分类)

    返回: 每个 horizon 的 log return
    """
    labels = pd.DataFrame(index=df.index)
    close = df['close'].astype(float)

    for h in horizons:
        labels[f'log_ret_{h}'] = np.log(close.shift(-h) / close)

    return labels


class QuantilePredictor:
    """
    分位数回归预测器

    对每个 horizon, 训练 5 个 LightGBM quantile 模型 (q05/q25/q50/q75/q95)
    """

    def __init__(self, config: Optional[QuantileConfig] = None):
        self.config = config or QuantileConfig()
        self.models: Dict[str, lgb.Booster] = {}  # key: f"h{h}_q{int(q*100)}"
        self.feature_names: List[str] = []
        self._trained = False

    def train(self, X_train: pd.DataFrame, y_dict: Dict[int, pd.Series],
              X_val: Optional[pd.DataFrame] = None,
              y_val_dict: Optional[Dict[int, pd.Series]] = None) -> Dict:
        """
        训练所有分位数模型

        参数:
            y_dict: {horizon: log_return_series}
        """
        self.feature_names = list(X_train.columns)
        X_tr = X_train.replace([np.inf, -np.inf], np.nan)
        X_vl = X_val.replace([np.inf, -np.inf], np.nan) if X_val is not None else None

        metrics = {}
        cfg = self.config

        for h in cfg.horizons:
            y = y_dict.get(h)
            if y is None:
                continue

            valid = y.notna()
            if valid.sum() < 500:
                continue

            for q in cfg.quantiles:
                key = f"h{h}_q{int(q * 100)}"

                params = dict(cfg.lgb_base_params)
                params['objective'] = 'quantile'
                params['alpha'] = q
                params['metric'] = 'quantile'

                dtrain = lgb.Dataset(X_tr[valid], label=y[valid])
                valid_sets = [dtrain]
                valid_names = ['train']
                callbacks = [lgb.log_evaluation(period=0)]

                if X_vl is not None and y_val_dict and h in y_val_dict:
                    y_v = y_val_dict[h]
                    v_valid = y_v.notna()
                    if v_valid.sum() > 10:
                        dval = lgb.Dataset(X_vl[v_valid], label=y_v[v_valid], reference=dtrain)
                        valid_sets.append(dval)
                        valid_names.append('val')
                        callbacks.append(lgb.early_stopping(cfg.early_stopping_rounds))

                model = lgb.train(
                    params, dtrain,
                    num_boost_round=cfg.num_boost_round,
                    valid_sets=valid_sets, valid_names=valid_names,
                    callbacks=callbacks,
                )
                self.models[key] = model
                metrics[key] = {'best_iter': model.best_iteration}

        self._trained = True
        return metrics

    def predict(self, X: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        预测所有 horizon 的分位数

        返回: {horizon: DataFrame(columns=[q05,q25,q50,q75,q95])}
        """
        if not self._trained:
            raise RuntimeError("模型未训练")

        X_clean = X[self.feature_names].replace([np.inf, -np.inf], np.nan)
        results = {}

        for h in self.config.horizons:
            df_q = pd.DataFrame(index=X.index)
            for q in self.config.quantiles:
                key = f"h{h}_q{int(q * 100)}"
                if key in self.models:
                    df_q[f'q{int(q * 100):02d}'] = self.models[key].predict(X_clean)
                else:
                    df_q[f'q{int(q * 100):02d}'] = 0.0
            results[h] = df_q

        return results

    def predict_latest(self, X: pd.DataFrame) -> Dict:
        """
        预测最新一条数据, 返回交易决策相关信息

        返回:
            {
                'horizon_5': {'q05': -.02, 'q25': -.005, 'q50': .003, 'q75': .01, 'q95': .03,
                              'expected_return': .003, 'iqr': .015, 'tail_risk': .02},
                'horizon_12': {...},
                'decision': {
                    'action': 'long'/'short'/'hold',
                    'confidence': float,
                    'kelly_weight': float,
                    'edge_ratio': float,
                },
            }
        """
        all_preds = self.predict(X)
        result = {}
        cfg = self.config

        for h, df_q in all_preds.items():
            row = df_q.iloc[-1]
            q05 = float(row['q05'])
            q25 = float(row['q25'])
            q50 = float(row['q50'])
            q75 = float(row['q75'])
            q95 = float(row['q95'])

            iqr = q75 - q25
            tail_range = q95 - q05

            result[f'horizon_{h}'] = {
                'q05': round(q05, 6),
                'q25': round(q25, 6),
                'q50': round(q50, 6),
                'q75': round(q75, 6),
                'q95': round(q95, 6),
                'expected_return': round(q50, 6),
                'iqr': round(iqr, 6),
                'tail_risk': round(q05, 6),  # 做多的最大可能亏损
                'upside': round(q95, 6),
            }

        # 交易决策 (用主 horizon)
        main_h = cfg.horizons[0]
        if f'horizon_{main_h}' in result:
            info = result[f'horizon_{main_h}']
            result['decision'] = self._make_decision(info, main_h)

        return result

    def _make_decision(self, quantiles: Dict, horizon: int) -> Dict:
        """
        基于分位数输出做交易决策 (GPT Pro 建议的核心逻辑)

        做多条件: q25 > 交易成本 (即 75% 概率盈利)
        做空条件: q75 < -交易成本
        仓位: 受限 Kelly = k * μ/σ²
        """
        cfg = self.config
        q05 = quantiles['q05']
        q25 = quantiles['q25']
        q50 = quantiles['q50']
        q75 = quantiles['q75']
        q95 = quantiles['q95']

        # 总成本 = 来回手续费 + 持仓期间 funding
        total_cost = cfg.roundtrip_cost + cfg.funding_per_hour * horizon

        # 期望收益 (q50) 和风险 (IQR 的一半作为 σ 近似)
        mu = q50
        sigma = max((q75 - q25) / 1.35, 1e-6)  # IQR / 1.35 ≈ std for normal

        # 净期望收益
        mu_net_long = mu - total_cost
        mu_net_short = -mu - total_cost

        # 风险调整后的 edge ratio
        edge_long = mu_net_long / total_cost if total_cost > 0 else 0
        edge_short = mu_net_short / total_cost if total_cost > 0 else 0

        # Kelly 仓位
        kelly_long = max(0, mu_net_long / (sigma ** 2 + 1e-8))
        kelly_short = max(0, mu_net_short / (sigma ** 2 + 1e-8))

        # 应用 Kelly 分数 (1/4 Kelly 保守策略)
        pos_long = min(kelly_long * cfg.kelly_fraction, cfg.max_position)
        pos_short = min(kelly_short * cfg.kelly_fraction, cfg.max_position)

        # 决策
        action = 'hold'
        confidence = 0.0
        kelly_weight = 0.0

        # 做多: q25 > cost (75% 概率覆盖成本) 且 edge > 2
        if q25 > total_cost and edge_long >= cfg.min_edge_ratio:
            action = 'long'
            confidence = min(1.0, edge_long / 5.0)
            kelly_weight = pos_long

        # 做空: q75 < -cost (75% 概率覆盖成本)
        elif q75 < -total_cost and edge_short >= cfg.min_edge_ratio:
            action = 'short'
            confidence = min(1.0, edge_short / 5.0)
            kelly_weight = pos_short

        return {
            'action': action,
            'confidence': round(confidence, 4),
            'kelly_weight': round(kelly_weight, 4),
            'edge_ratio_long': round(edge_long, 4),
            'edge_ratio_short': round(edge_short, 4),
            'mu_net_long': round(mu_net_long, 6),
            'mu_net_short': round(mu_net_short, 6),
            'sigma': round(sigma, 6),
            'total_cost': round(total_cost, 6),
        }

    def save(self, model_dir: Optional[str] = None):
        model_dir = model_dir or self.config.model_dir
        os.makedirs(model_dir, exist_ok=True)
        for key, model in self.models.items():
            model.save_model(os.path.join(model_dir, f'quantile_{key}.txt'))
        with open(os.path.join(model_dir, 'quantile_config.json'), 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'horizons': self.config.horizons,
                'quantiles': self.config.quantiles,
            }, f, indent=2)

    def load(self, model_dir: Optional[str] = None):
        model_dir = model_dir or self.config.model_dir
        cfg_path = os.path.join(model_dir, 'quantile_config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            self.feature_names = cfg.get('feature_names', [])
            self.config.horizons = cfg.get('horizons', self.config.horizons)
            self.config.quantiles = cfg.get('quantiles', self.config.quantiles)

        for h in self.config.horizons:
            for q in self.config.quantiles:
                key = f"h{h}_q{int(q * 100)}"
                path = os.path.join(model_dir, f'quantile_{key}.txt')
                if os.path.exists(path):
                    self.models[key] = lgb.Booster(model_file=path)
        self._trained = len(self.models) > 0


class QuantileWalkForward:
    """Walk-forward 验证分位数预测器"""

    def __init__(self, config: Optional[QuantileConfig] = None):
        self.config = config or QuantileConfig()
        self.fold_results: List[Dict] = []

    def run(self, features: pd.DataFrame, labels_df: pd.DataFrame,
            verbose: bool = True) -> Dict[int, pd.DataFrame]:
        """运行 walk-forward, 返回每个 horizon 的 OOS 分位数预测"""
        cfg = self.config
        n = len(features)

        # 初始化输出
        all_preds = {}
        for h in cfg.horizons:
            all_preds[h] = pd.DataFrame(
                index=features.index,
                columns=[f'q{int(q * 100):02d}' for q in cfg.quantiles],
                dtype=float,
            )

        start_idx = cfg.min_train_window + cfg.val_window + cfg.purge_gap
        if start_idx >= n:
            raise ValueError(f"数据不足: 需要 {start_idx}, 只有 {n}")

        fold = 0
        cursor = start_idx

        while cursor < n:
            train_end = cursor - cfg.purge_gap - cfg.val_window
            train_start = 0 if cfg.expanding else max(0, train_end - cfg.min_train_window)
            val_start = train_end + cfg.purge_gap
            val_end = cursor
            test_start = cursor
            test_end = min(n, cursor + cfg.retrain_interval)

            if train_end - train_start < 500:
                cursor += cfg.retrain_interval
                continue

            # 数据分割
            X_tr = features.iloc[train_start:train_end]
            X_val = features.iloc[val_start:val_end]
            X_test = features.iloc[test_start:test_end]

            y_tr_dict = {}
            y_val_dict = {}
            for h in cfg.horizons:
                col = f'log_ret_{h}'
                if col in labels_df.columns:
                    y_tr_dict[h] = labels_df[col].iloc[train_start:train_end]
                    y_val_dict[h] = labels_df[col].iloc[val_start:val_end]

            # 训练
            model = QuantilePredictor(cfg)
            metrics = model.train(X_tr, y_tr_dict, X_val, y_val_dict)

            # OOS 预测
            preds = model.predict(X_test)
            for h, df_q in preds.items():
                for col in df_q.columns:
                    all_preds[h].loc[X_test.index, col] = df_q[col].values

            # 评估: Pinball Loss on val
            val_loss = self._eval_quantile_loss(model, X_val, y_val_dict)

            fold_info = {
                'fold': fold,
                'train_size': train_end - train_start,
                'test_size': test_end - test_start,
                'val_pinball_loss': val_loss,
            }
            self.fold_results.append(fold_info)

            if verbose:
                print(f"  Fold {fold}: train={train_end - train_start} "
                      f"pinball_loss={val_loss:.6f} test={test_end - test_start} "
                      f"[{features.index[test_start].strftime('%m-%d')} ~ "
                      f"{features.index[min(test_end - 1, n - 1)].strftime('%m-%d')}]")

            fold += 1
            cursor += cfg.retrain_interval

        return all_preds

    def _eval_quantile_loss(self, model: QuantilePredictor,
                            X_val: pd.DataFrame,
                            y_val_dict: Dict[int, pd.Series]) -> float:
        """计算验证集平均 pinball loss"""
        preds = model.predict(X_val)
        total_loss = 0
        count = 0

        for h, df_q in preds.items():
            if h not in y_val_dict:
                continue
            y = y_val_dict[h]
            valid = y.notna()
            if valid.sum() < 10:
                continue

            for q in self.config.quantiles:
                col = f'q{int(q * 100):02d}'
                pred = df_q.loc[valid, col].values
                actual = y[valid].values
                residual = actual - pred
                loss = np.mean(np.where(residual >= 0, q * residual, (q - 1) * residual))
                total_loss += loss
                count += 1

        return total_loss / max(count, 1)

    def summary(self) -> Dict:
        if not self.fold_results:
            return {}
        losses = [f['val_pinball_loss'] for f in self.fold_results]
        return {
            'total_folds': len(self.fold_results),
            'avg_pinball_loss': round(np.mean(losses), 6),
            'std_pinball_loss': round(np.std(losses), 6),
        }


def evaluate_quantile_quality(labels_df: pd.DataFrame,
                               pred_dict: Dict[int, pd.DataFrame]) -> Dict:
    """
    评估分位数预测质量:
      1. Coverage: 实际收益落在 q05-q95 之间的比例 (应 ~90%)
      2. Calibration: 每个分位数的实际覆盖率
      3. Sharpness: 预测区间的平均宽度 (越窄越好, 在覆盖率达标前提下)
      4. 方向准确率: q50 > 0 时实际也上涨的比例
    """
    results = {}

    for h, df_q in pred_dict.items():
        col = f'log_ret_{h}'
        if col not in labels_df.columns:
            continue

        y = labels_df[col]
        valid = y.notna() & df_q['q50'].notna()
        if valid.sum() < 30:
            continue

        actual = y[valid].values
        q05 = df_q.loc[valid, 'q05'].values
        q25 = df_q.loc[valid, 'q25'].values
        q50 = df_q.loc[valid, 'q50'].values
        q75 = df_q.loc[valid, 'q75'].values
        q95 = df_q.loc[valid, 'q95'].values

        # Coverage
        cov_90 = np.mean((actual >= q05) & (actual <= q95))
        cov_50 = np.mean((actual >= q25) & (actual <= q75))

        # Calibration
        cal = {}
        for q_val, q_col in [(0.05, q05), (0.25, q25), (0.50, q50), (0.75, q75), (0.95, q95)]:
            cal[f'q{int(q_val*100):02d}_actual_below'] = round(np.mean(actual < q_col), 3)

        # Sharpness
        avg_width_90 = np.mean(q95 - q05)
        avg_width_50 = np.mean(q75 - q25)

        # Direction accuracy
        dir_acc = np.mean((q50 > 0) == (actual > 0))

        # Pinball loss
        pinball = 0
        for q, qp in zip([0.05, 0.25, 0.50, 0.75, 0.95], [q05, q25, q50, q75, q95]):
            r = actual - qp
            pinball += np.mean(np.where(r >= 0, q * r, (q - 1) * r))
        pinball /= 5

        results[f'h{h}'] = {
            'coverage_90': round(cov_90, 3),
            'coverage_50': round(cov_50, 3),
            'calibration': cal,
            'sharpness_90': round(avg_width_90, 6),
            'sharpness_50': round(avg_width_50, 6),
            'direction_accuracy': round(dir_acc, 3),
            'avg_pinball_loss': round(pinball, 6),
            'n_samples': int(valid.sum()),
        }

    return results


def run_quantile_backtest():
    """运行完整的分位数回归回测"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=365)
    parser.add_argument('--tf', type=str, default='1h')
    args = parser.parse_args()

    print("=" * 80)
    print(f"  ML 分位数回归预测器 + 成本感知决策")
    print(f"  时间框架: {args.tf} | 天数: {args.days}")
    print(f"  输出: q05/q25/q50/q75/q95 收益分布")
    print("=" * 80)

    # 1. 获取数据
    print(f"\n[1/5] 获取数据...")
    from optimize_six_book import fetch_multi_tf_data
    all_data = fetch_multi_tf_data([args.tf], days=args.days)
    df = all_data[args.tf]
    print(f"  数据: {len(df)} 条K线")

    # 2. 特征和标签
    print(f"\n[2/5] 计算增强特征 (15 维, 含衍生品)...")
    features = compute_enhanced_features(df)
    cfg = QuantileConfig()
    labels = compute_return_labels(df, cfg.horizons)
    print(f"  特征: {features.shape[1]} 维")
    print(f"  标签: {list(labels.columns)}")
    for h in cfg.horizons:
        col = f'log_ret_{h}'
        ret = labels[col].dropna()
        print(f"  h{h}: mean={ret.mean():.5f} std={ret.std():.5f}")

    # 3. Walk-forward
    print(f"\n[3/5] Walk-forward 分位数回归...")
    wf = QuantileWalkForward(cfg)
    pred_dict = wf.run(features, labels, verbose=True)

    summary = wf.summary()
    print(f"\n  Walk-forward 摘要:")
    for k, v in summary.items():
        print(f"    {k}: {v}")

    # 4. 评估分位数质量
    print(f"\n[4/5] 分位数预测质量评估...")
    quality = evaluate_quantile_quality(labels, pred_dict)
    for h_key, metrics in quality.items():
        print(f"\n  {h_key}:")
        print(f"    90% 覆盖率: {metrics['coverage_90']:.1%} (理想=90%)")
        print(f"    50% 覆盖率: {metrics['coverage_50']:.1%} (理想=50%)")
        print(f"    方向准确率: {metrics['direction_accuracy']:.1%}")
        print(f"    区间宽度(90%): {metrics['sharpness_90']:.4%}")
        print(f"    区间宽度(50%): {metrics['sharpness_50']:.4%}")
        print(f"    Pinball Loss: {metrics['avg_pinball_loss']:.6f}")
        print(f"    校准:")
        for k, v in metrics['calibration'].items():
            print(f"      {k}: {v}")

    # 5. 模拟交易决策
    print(f"\n[5/5] 成本感知交易决策模拟...")
    simulate_quantile_trading(df, pred_dict, labels, cfg)

    return pred_dict, quality


def simulate_quantile_trading(df: pd.DataFrame,
                               pred_dict: Dict[int, pd.DataFrame],
                               labels: pd.DataFrame,
                               config: QuantileConfig):
    """模拟基于分位数的成本感知交易"""
    main_h = config.horizons[0]
    df_q = pred_dict[main_h]
    col = f'log_ret_{main_h}'

    valid = df_q['q50'].notna() & labels[col].notna()
    q25 = df_q.loc[valid, 'q25'].values
    q50 = df_q.loc[valid, 'q50'].values
    q75 = df_q.loc[valid, 'q75'].values
    actual = labels.loc[valid, col].values

    cost = config.roundtrip_cost + config.funding_per_hour * main_h

    # 策略: q25 > cost → long, q75 < -cost → short, else hold
    long_mask = q25 > cost
    short_mask = q75 < -cost
    hold_mask = ~long_mask & ~short_mask

    n = len(actual)
    long_n = long_mask.sum()
    short_n = short_mask.sum()
    hold_n = hold_mask.sum()

    print(f"  决策分布: Long={long_n} ({long_n/n:.1%}) "
          f"Short={short_n} ({short_n/n:.1%}) Hold={hold_n} ({hold_n/n:.1%})")

    # 收益
    long_ret = actual[long_mask] - cost
    short_ret = -actual[short_mask] - cost

    all_returns = np.concatenate([long_ret, short_ret]) if len(long_ret) + len(short_ret) > 0 else np.array([0])

    if len(all_returns) > 0:
        print(f"  交易次数: {len(all_returns)}")
        print(f"  累计净收益: {all_returns.sum():.4%}")
        print(f"  平均每笔: {all_returns.mean():.4%}")
        print(f"  胜率: {(all_returns > 0).mean():.1%}")
        print(f"  盈亏比: {all_returns[all_returns > 0].mean() / abs(all_returns[all_returns < 0].mean()):.2f}"
              if (all_returns < 0).any() and (all_returns > 0).any() else "")

    # 对比: 买入持有
    bh_ret = actual.sum()
    print(f"  对比买入持有: {bh_ret:.4%}")


if __name__ == '__main__':
    run_quantile_backtest()
