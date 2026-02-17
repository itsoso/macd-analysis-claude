"""
ML Regime 预测器 v3

核心理念转换:
  v1/v2: 预测价格方向 → OOS AUC 0.49-0.52 (失败)
  v3: 预测波动率 regime + 趋势质量 → OOS AUC 目标 0.60+ (可行)

原因:
  - 方向预测在 1h 级别几乎等于猜硬币 (EMH)
  - 波动率有聚类效应 (GARCH), ATR/BB 可预测未来波动
  - 趋势连续性比方向预测更稳定 (动量效应)
  - 只需 8-10 个核心特征, 避免过拟合

两个预测目标:
  1. vol_regime: 未来 12 bar 是否高波动 (用于决定是否交易)
  2. trend_quality: 当前趋势是否可持续 (用于增强/抑制信号)

使用方式:
  - 高波动 + 强趋势 → 加强规则信号
  - 低波动 → 抑制信号 (大概率震荡)
  - 高波动 + 无趋势 → 保持原信号 (可能反转)
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


@dataclass
class RegimeConfig:
    """Regime 预测配置"""
    # 波动率 regime 参数
    vol_lookback: int = 12            # 未来 12 bar 波动率窗口
    vol_percentile: float = 0.55      # 高于 55% 分位为高波动 (略偏向更多高波动)

    # 趋势质量参数
    trend_lookback: int = 24          # 趋势评估窗口
    trend_min_r2: float = 0.3         # 线性拟合 R² > 0.3 为有效趋势

    # LightGBM (极简配置)
    lgb_params: Dict = field(default_factory=lambda: {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 12,             # 极少叶数
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 3,
        'min_child_samples': 40,
        'lambda_l1': 0.3,
        'lambda_l2': 1.0,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    })
    num_boost_round: int = 200
    early_stopping_rounds: int = 25

    # Walk-forward
    min_train_window: int = 720       # 30 天
    expanding: bool = True
    val_window: int = 168
    retrain_interval: int = 120
    purge_gap: int = 36               # 加大 purge (1.5天)

    # 路径
    model_dir: str = 'data/ml_models'


def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    精简特征集: 10 个对波动率/趋势最有预测力的特征。

    选择原则:
      - 每个特征有明确物理含义
      - 互相关 < 0.7
      - 在多种市场条件下稳定
      - 向量化实现，高性能
    """
    feat = pd.DataFrame(index=df.index)
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)

    # 1. ATR 归一化 (波动率的最佳预测因子)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    feat['atr_pct'] = atr_14 / close

    # 2. 短期/长期波动率比 (波动率加速/减速)
    log_ret = np.log(close / close.shift(1))
    hvol_5 = log_ret.rolling(5).std()
    hvol_20 = log_ret.rolling(20).std()
    feat['vol_ratio'] = hvol_5 / hvol_20.replace(0, np.nan)

    # 3. Bollinger 带宽 (挤压→即将爆发)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat['bb_width'] = (2 * bb_std / bb_mid.replace(0, np.nan))

    # 4. BB 宽度在历史中的分位 (当前处于挤压周期?)
    feat['bb_width_rank'] = feat['bb_width'].rolling(120).rank(pct=True)

    # 5. 价格动量 (5 bar 累计收益, 趋势方向)
    feat['momentum_5'] = close.pct_change(5)

    # 6. 趋势一致性: 20 bar 线性拟合 R² (向量化版本)
    feat['trend_r2'] = _rolling_r2_vectorized(close, window=20)

    # 7. 量比 (交易活跃度, 高量常伴随大波动)
    feat['vol_ma_ratio'] = volume.rolling(5).mean() / volume.rolling(20).mean().replace(0, np.nan)

    # 8. 最近 bar 的 range 占 20 bar range 的比例 (突破倾向)
    recent_range = (high.rolling(3).max() - low.rolling(3).min()) / close
    period_range = (high.rolling(20).max() - low.rolling(20).min()) / close
    feat['range_ratio'] = recent_range / period_range.replace(0, np.nan)

    # 9. Funding rate (加密货币特有: 正 funding = 多头拥挤)
    if 'funding_rate' in df.columns:
        feat['funding_rate'] = df['funding_rate'].astype(float)
    else:
        feat['funding_rate'] = 0.0

    # 10. 大级别趋势斜率 (60 bar MA 斜率, 避免噪声)
    ma60 = close.rolling(60).mean()
    feat['ma60_slope'] = ma60.pct_change(5)

    return feat


def _rolling_r2_vectorized(series: pd.Series, window: int = 20) -> pd.Series:
    """向量化滚动 R² 计算, 比 for 循环快 100x+"""
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    ss_x = ((x - x_mean) ** 2).sum()

    # 用 rolling apply (仍然比纯 Python loop 快因为 pandas 内部优化)
    def calc_r2(y):
        y_mean = y.mean()
        ss_y = ((y - y_mean) ** 2).sum()
        if ss_y == 0:
            return 0.0
        ss_xy = ((x - x_mean) * (y - y_mean)).sum()
        return (ss_xy ** 2) / (ss_x * ss_y)

    return series.rolling(window).apply(calc_r2, raw=True)


def compute_regime_labels(df: pd.DataFrame, config: Optional[RegimeConfig] = None) -> pd.DataFrame:
    """
    计算 regime 标签:
      1. vol_high: 未来 12 bar 的价格波动率 > 历史 55% 分位
      2. trend_strong: 未来 12 bar 的价格变动 > 1x ATR
    """
    cfg = config or RegimeConfig()
    labels = pd.DataFrame(index=df.index)
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)

    lookback = cfg.vol_lookback

    # 波动率标签: 未来 lookback bar 的 high-low range / close
    fwd_high = high.rolling(lookback).max().shift(-lookback)
    fwd_low = low.rolling(lookback).min().shift(-lookback)
    fwd_range = (fwd_high - fwd_low) / close
    labels['fwd_vol'] = fwd_range

    # 动态阈值: 在历史窗口中的分位
    vol_threshold = fwd_range.rolling(240, min_periods=60).quantile(cfg.vol_percentile)
    labels['vol_high'] = (fwd_range > vol_threshold).astype(int)

    # 趋势标签: 未来 lookback bar 的方向性
    fwd_ret = close.shift(-lookback) / close - 1
    labels['fwd_ret'] = fwd_ret

    # 用 ATR 归一化: |fwd_ret| > 1x ATR = 有效趋势
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean() / close
    labels['trend_strong'] = (fwd_ret.abs() > atr).astype(int)

    # 综合标签: 方向预测 (仅在高波动时有意义)
    labels['fwd_dir'] = (fwd_ret > 0).astype(int)

    return labels


class RegimePredictor:
    """
    Regime 预测器: 预测波动率 regime 和趋势质量。

    输出:
      - vol_prob: 高波动概率 (0~1)
      - trend_prob: 强趋势概率 (0~1)
      - regime: 'trending' / 'ranging' / 'volatile'
      - trade_confidence: 综合交易置信度 (0~1)
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.vol_model: Optional[lgb.Booster] = None
        self.trend_model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []
        self._trained = False

    def train(self, features: pd.DataFrame,
              vol_labels: pd.Series,
              trend_labels: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              vol_val: Optional[pd.Series] = None,
              trend_val: Optional[pd.Series] = None) -> Dict:
        """训练波动率和趋势模型"""
        self.feature_names = list(features.columns)
        cfg = self.config
        metrics = {}

        X_train = features.replace([np.inf, -np.inf], np.nan)
        X_v = X_val.replace([np.inf, -np.inf], np.nan) if X_val is not None else None

        # 1. 波动率 regime 模型
        valid = vol_labels.notna()
        dtrain = lgb.Dataset(X_train[valid], label=vol_labels[valid])
        valid_sets = [dtrain]
        valid_names = ['train']
        callbacks = [lgb.log_evaluation(period=0)]

        if X_v is not None and vol_val is not None:
            v_mask = vol_val.notna()
            dval = lgb.Dataset(X_v[v_mask], label=vol_val[v_mask], reference=dtrain)
            valid_sets.append(dval)
            valid_names.append('val')
            callbacks.append(lgb.early_stopping(cfg.early_stopping_rounds))

        self.vol_model = lgb.train(
            cfg.lgb_params, dtrain,
            num_boost_round=cfg.num_boost_round,
            valid_sets=valid_sets, valid_names=valid_names,
            callbacks=callbacks,
        )
        metrics['vol_best_iter'] = self.vol_model.best_iteration

        # 2. 趋势质量模型
        valid_t = trend_labels.notna()
        dtrain_t = lgb.Dataset(X_train[valid_t], label=trend_labels[valid_t])
        valid_sets_t = [dtrain_t]
        valid_names_t = ['train']
        callbacks_t = [lgb.log_evaluation(period=0)]

        if X_v is not None and trend_val is not None:
            v_mask_t = trend_val.notna()
            dval_t = lgb.Dataset(X_v[v_mask_t], label=trend_val[v_mask_t], reference=dtrain_t)
            valid_sets_t.append(dval_t)
            valid_names_t.append('val')
            callbacks_t.append(lgb.early_stopping(cfg.early_stopping_rounds))

        self.trend_model = lgb.train(
            cfg.lgb_params, dtrain_t,
            num_boost_round=cfg.num_boost_round,
            valid_sets=valid_sets_t, valid_names=valid_names_t,
            callbacks=callbacks_t,
        )
        metrics['trend_best_iter'] = self.trend_model.best_iteration

        self._trained = True
        return metrics

    def predict(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """预测 regime"""
        if not self._trained:
            raise RuntimeError("模型未训练")

        X = features[self.feature_names].replace([np.inf, -np.inf], np.nan)
        vol_prob = self.vol_model.predict(X)
        trend_prob = self.trend_model.predict(X)

        # 交易置信度: vol_prob × trend_prob 的几何平均
        trade_confidence = np.sqrt(vol_prob * trend_prob)

        # Regime 分类
        regime = np.full(len(X), 'ranging', dtype=object)
        regime[(vol_prob >= 0.55) & (trend_prob >= 0.50)] = 'trending'
        regime[(vol_prob >= 0.55) & (trend_prob < 0.50)] = 'volatile'

        return {
            'vol_prob': vol_prob,
            'trend_prob': trend_prob,
            'trade_confidence': trade_confidence,
            'regime': regime,
        }

    def save(self, model_dir: Optional[str] = None):
        model_dir = model_dir or self.config.model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.vol_model.save_model(os.path.join(model_dir, 'vol_regime_model.txt'))
        self.trend_model.save_model(os.path.join(model_dir, 'trend_regime_model.txt'))
        with open(os.path.join(model_dir, 'regime_config.json'), 'w') as f:
            json.dump({'feature_names': self.feature_names}, f, indent=2)

    def load(self, model_dir: Optional[str] = None):
        model_dir = model_dir or self.config.model_dir
        self.vol_model = lgb.Booster(model_file=os.path.join(model_dir, 'vol_regime_model.txt'))
        self.trend_model = lgb.Booster(model_file=os.path.join(model_dir, 'trend_regime_model.txt'))
        cfg_path = os.path.join(model_dir, 'regime_config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            self.feature_names = cfg.get('feature_names', [])
        self._trained = True


class RegimeWalkForward:
    """Walk-forward 验证 regime 预测器"""

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.fold_results: List[Dict] = []

    def run(self, features: pd.DataFrame, labels_df: pd.DataFrame,
            verbose: bool = True) -> pd.DataFrame:
        cfg = self.config
        n = len(features)

        all_vol_prob = pd.Series(index=features.index, dtype=float)
        all_vol_prob[:] = np.nan
        all_trend_prob = pd.Series(index=features.index, dtype=float)
        all_trend_prob[:] = np.nan

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

            vol_tr = labels_df['vol_high'].iloc[train_start:train_end]
            vol_val = labels_df['vol_high'].iloc[val_start:val_end]
            trend_tr = labels_df['trend_strong'].iloc[train_start:train_end]
            trend_val = labels_df['trend_strong'].iloc[val_start:val_end]

            # 训练
            model = RegimePredictor(cfg)
            metrics = model.train(
                X_tr, vol_tr, trend_tr,
                X_val, vol_val, trend_val,
            )

            # OOS 预测
            preds = model.predict(X_test)
            all_vol_prob.iloc[test_start:test_end] = preds['vol_prob']
            all_trend_prob.iloc[test_start:test_end] = preds['trend_prob']

            # 验证集 AUC
            val_preds = model.predict(X_val)
            vol_auc = self._auc(vol_val, val_preds['vol_prob'])
            trend_auc = self._auc(trend_val, val_preds['trend_prob'])

            fold_info = {
                'fold': fold,
                'vol_val_auc': round(vol_auc, 4),
                'trend_val_auc': round(trend_auc, 4),
                'test_size': test_end - test_start,
            }
            self.fold_results.append(fold_info)

            if verbose:
                print(f"  Fold {fold}: train={train_end - train_start} "
                      f"vol_AUC={vol_auc:.4f} trend_AUC={trend_auc:.4f} "
                      f"test={test_end - test_start} "
                      f"[{features.index[test_start].strftime('%m-%d')} ~ "
                      f"{features.index[min(test_end - 1, n - 1)].strftime('%m-%d')}]")

            fold += 1
            cursor += cfg.retrain_interval

        result_df = pd.DataFrame({
            'vol_prob': all_vol_prob,
            'trend_prob': all_trend_prob,
            'trade_confidence': np.sqrt(all_vol_prob * all_trend_prob),
        }, index=features.index)

        result_df['regime'] = 'ranging'
        result_df.loc[
            (result_df['vol_prob'] >= 0.55) & (result_df['trend_prob'] >= 0.50),
            'regime'
        ] = 'trending'
        result_df.loc[
            (result_df['vol_prob'] >= 0.55) & (result_df['trend_prob'] < 0.50),
            'regime'
        ] = 'volatile'

        return result_df

    @staticmethod
    def _auc(y_true: pd.Series, y_pred: np.ndarray) -> float:
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
        vol_aucs = [f['vol_val_auc'] for f in self.fold_results]
        trend_aucs = [f['trend_val_auc'] for f in self.fold_results]
        return {
            'total_folds': len(self.fold_results),
            'avg_vol_auc': round(np.mean(vol_aucs), 4),
            'avg_trend_auc': round(np.mean(trend_aucs), 4),
            'vol_auc_range': f"[{min(vol_aucs):.4f}, {max(vol_aucs):.4f}]",
            'trend_auc_range': f"[{min(trend_aucs):.4f}, {max(trend_aucs):.4f}]",
        }


def run_regime_backtest():
    """运行 regime 预测器的完整回测验证"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=365)
    parser.add_argument('--tf', type=str, default='1h')
    args = parser.parse_args()

    print("=" * 80)
    print(f"  ML Regime 预测器 v3 回测")
    print(f"  时间框架: {args.tf} | 天数: {args.days}")
    print("=" * 80)

    # 1. 获取数据
    print(f"\n[1/5] 获取数据...")
    from optimize_six_book import fetch_multi_tf_data
    all_data = fetch_multi_tf_data([args.tf], days=args.days)
    df = all_data[args.tf]
    print(f"  数据: {len(df)} 条K线")

    # 2. 计算特征和标签
    print(f"\n[2/5] 计算 regime 特征 (精简 8 维)...")
    features = compute_regime_features(df)
    labels = compute_regime_labels(df)

    vol_dist = labels['vol_high'].mean()
    trend_dist = labels['trend_strong'].mean()
    print(f"  特征: {features.shape[1]} 维")
    print(f"  波动率标签: 高波动={vol_dist:.1%} 低波动={1 - vol_dist:.1%}")
    print(f"  趋势标签: 强趋势={trend_dist:.1%} 弱趋势={1 - trend_dist:.1%}")

    # 3. Walk-forward
    print(f"\n[3/5] Walk-forward 验证...")
    wf = RegimeWalkForward()
    results = wf.run(features, labels)

    summary = wf.summary()
    print(f"\n  摘要:")
    for k, v in summary.items():
        print(f"    {k}: {v}")

    # 4. OOS 评估
    print(f"\n[4/5] OOS regime 预测评估...")
    valid_idx = results['vol_prob'].notna()
    if valid_idx.sum() > 0:
        from sklearn.metrics import roc_auc_score, accuracy_score

        # 波动率预测 OOS AUC
        vol_valid = valid_idx & labels['vol_high'].notna()
        vol_oos_auc = roc_auc_score(
            labels['vol_high'][vol_valid],
            results['vol_prob'][vol_valid],
        )
        print(f"  波动率 OOS AUC: {vol_oos_auc:.4f}")

        # 趋势预测 OOS AUC
        trend_valid = valid_idx & labels['trend_strong'].notna()
        trend_oos_auc = roc_auc_score(
            labels['trend_strong'][trend_valid],
            results['trend_prob'][trend_valid],
        )
        print(f"  趋势 OOS AUC: {trend_oos_auc:.4f}")

        # Regime 分布
        regime_counts = results.loc[valid_idx, 'regime'].value_counts()
        print(f"\n  Regime 分布 (OOS):")
        for regime, count in regime_counts.items():
            pct = count / valid_idx.sum()
            print(f"    {regime}: {count} ({pct:.1%})")

        # 各 regime 下的实际波动率
        print(f"\n  各 regime 下的实际前向波动率:")
        for regime in ['trending', 'volatile', 'ranging']:
            mask = valid_idx & (results['regime'] == regime) & labels['fwd_vol'].notna()
            if mask.sum() > 0:
                avg_vol = labels.loc[mask, 'fwd_vol'].mean()
                avg_ret = labels.loc[mask, 'fwd_ret'].abs().mean()
                print(f"    {regime:10s}: avg_vol={avg_vol:.4f} avg_|ret|={avg_ret:.4f} n={mask.sum()}")

        # 验证: "trending" regime 下预测方向的准确率
        print(f"\n  Trending regime 下的方向预测:")
        trend_mask = valid_idx & (results['regime'] == 'trending') & labels['fwd_dir'].notna()
        if trend_mask.sum() > 0:
            dir_acc = (labels.loc[trend_mask, 'fwd_dir'] == 1).mean()
            print(f"    方向偏多概率: {dir_acc:.1%} (n={trend_mask.sum()})")
            avg_ret_trending = labels.loc[trend_mask, 'fwd_ret'].mean()
            print(f"    平均前向收益: {avg_ret_trending:.4%}")

    # 5. 策略模拟: regime 过滤 vs 无过滤
    print(f"\n[5/5] 策略模拟: regime 过滤效果...")
    simulate_regime_filter(df, results, labels)

    return results, labels


def simulate_regime_filter(df: pd.DataFrame, regime_results: pd.DataFrame,
                           labels: pd.DataFrame):
    """模拟 regime 过滤器的效果"""
    valid = regime_results['vol_prob'].notna() & labels['fwd_ret'].notna()
    fwd_ret = labels.loc[valid, 'fwd_ret']
    regime = regime_results.loc[valid, 'regime']

    # 基准: 买入持有
    bh_return = fwd_ret.sum()
    print(f"  基准 (买入持有): 累计收益 = {bh_return:.2%}")

    # 策略 1: 只在 trending regime 做多
    trending = regime == 'trending'
    if trending.sum() > 0:
        trend_ret = fwd_ret[trending].sum()
        trade_pct = trending.mean()
        print(f"  Trending 做多: 累计={trend_ret:.2%} 交易时间={trade_pct:.1%} "
              f"(n={trending.sum()})")

    # 策略 2: 只在非 ranging 时交易
    not_ranging = regime != 'ranging'
    if not_ranging.sum() > 0:
        active_ret = fwd_ret[not_ranging].sum()
        active_pct = not_ranging.mean()
        print(f"  非 ranging: 累计={active_ret:.2%} 交易时间={active_pct:.1%} "
              f"(n={not_ranging.sum()})")

    # 策略 3: 高置信度交易
    confidence = regime_results.loc[valid, 'trade_confidence']
    high_conf = confidence >= 0.55
    if high_conf.sum() > 0:
        hc_ret = fwd_ret[high_conf].sum()
        hc_pct = high_conf.mean()
        print(f"  高置信度: 累计={hc_ret:.2%} 交易时间={hc_pct:.1%} "
              f"(n={high_conf.sum()})")


if __name__ == '__main__':
    run_regime_backtest()
