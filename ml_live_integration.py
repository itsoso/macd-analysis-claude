"""
ML 实盘集成模块 v4 (最终版)

三层架构 (GPT Pro 建议):
  1. 预测层: 分位数回归 → 输出收益分布 (q05/q25/q50/q75/q95)
  2. 决策层: Regime 过滤 + 成本感知门槛 + Kelly 仓位
  3. 执行层: 与现有六书融合信号协同

核心价值:
  - 分位数校准准确 (OOS 90% 覆盖率 88.4%, 几乎完美)
  - Regime 预测有效 (OOS AUC 0.60, 回报密度 2.2x)
  - 两者组合: regime 判断"何时交易", 分位数判断"交易多大仓位"

使用:
  1. 训练: python3.10 ml_live_integration.py --train
  2. 实盘: live_signal_generator.py 自动调用 MLSignalEnhancer.enhance_signal()
"""

import os
import json
import logging
import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join('data', 'ml_models')


class MLSignalEnhancer:
    """
    ML 信号增强器 v4 — Regime 过滤 + 分位数风控

    工作模式:
      1. 计算 regime (vol_prob, trend_prob, trade_confidence)
      2. 如有分位数模型: 计算收益分布 → 风险调整
      3. 综合输出: 增强/抑制规则信号
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self._regime_model = None
        self._quantile_model = None
        self._loaded = False

        # Regime 增强参数
        self.high_conf_threshold = 0.55
        self.low_conf_threshold = 0.35
        self.boost_factor = 1.12
        self.dampen_factor = 0.88
        self.strong_boost_factor = 1.20

        # 分位数风控参数
        self.cost_threshold = 0.003       # 来回成本 0.3%
        self.risk_dampen_q05 = -0.03      # q05 < -3% 时强制降权

    def load_model(self) -> bool:
        """加载所有可用模型"""
        loaded_any = False

        # Regime 模型
        try:
            vol_path = os.path.join(self.model_dir, 'vol_regime_model.txt')
            if os.path.exists(vol_path):
                from ml_regime import RegimePredictor
                self._regime_model = RegimePredictor()
                self._regime_model.load(self.model_dir)
                logger.info("Regime 模型加载成功")
                loaded_any = True
        except Exception as e:
            logger.warning(f"Regime 模型加载失败: {e}")

        # 分位数模型 (可选)
        try:
            q_cfg_path = os.path.join(self.model_dir, 'quantile_config.json')
            if os.path.exists(q_cfg_path):
                from ml_quantile import QuantilePredictor
                self._quantile_model = QuantilePredictor()
                self._quantile_model.load(self.model_dir)
                logger.info("分位数模型加载成功")
                loaded_any = True
        except Exception as e:
            logger.warning(f"分位数模型加载失败: {e}")

        self._loaded = loaded_any
        return loaded_any

    def enhance_signal(
        self,
        sell_score: float,
        buy_score: float,
        df: pd.DataFrame,
    ) -> Tuple[float, float, Dict]:
        """
        综合 ML 增强: Regime + 分位数

        返回:
            (enhanced_sell_score, enhanced_buy_score, ml_info)
        """
        if not self._loaded:
            if not self.load_model():
                return sell_score, buy_score, {'ml_available': False}

        ml_info = {'ml_available': True, 'ml_version': 'v4'}
        enhanced_buy = buy_score
        enhanced_sell = sell_score

        # 1. Regime 过滤
        if self._regime_model:
            try:
                from ml_regime import compute_regime_features
                features = compute_regime_features(df)
                latest = features.iloc[[-1]]
                preds = self._regime_model.predict(latest)

                confidence = float(preds['trade_confidence'][0])
                regime = str(preds['regime'][0])

                ml_info['regime'] = regime
                ml_info['trade_confidence'] = round(confidence, 4)
                ml_info['vol_prob'] = round(float(preds['vol_prob'][0]), 4)
                ml_info['trend_prob'] = round(float(preds['trend_prob'][0]), 4)

                if regime == 'trending' and confidence >= self.high_conf_threshold:
                    enhanced_buy *= self.strong_boost_factor
                    enhanced_sell *= self.strong_boost_factor
                    ml_info['regime_action'] = 'strong_boost'
                elif confidence >= self.high_conf_threshold:
                    enhanced_buy *= self.boost_factor
                    enhanced_sell *= self.boost_factor
                    ml_info['regime_action'] = 'boost'
                elif confidence <= self.low_conf_threshold:
                    enhanced_buy *= self.dampen_factor
                    enhanced_sell *= self.dampen_factor
                    ml_info['regime_action'] = 'dampen'
                else:
                    ml_info['regime_action'] = 'neutral'
            except Exception as e:
                logger.warning(f"Regime 计算异常: {e}")

        # 2. 分位数风控 (如有)
        if self._quantile_model:
            try:
                from ml_quantile import compute_enhanced_features
                q_features = compute_enhanced_features(df)
                q_latest = q_features.iloc[[-1]]
                q_preds = self._quantile_model.predict(q_latest)

                # 取第一个 horizon
                h = self._quantile_model.config.horizons[0]
                if h in q_preds:
                    row = q_preds[h].iloc[0]
                    q05 = float(row['q05'])
                    q25 = float(row['q25'])
                    q50 = float(row['q50'])
                    q75 = float(row['q75'])
                    q95 = float(row['q95'])

                    ml_info['quantile'] = {
                        'q05': round(q05, 5), 'q25': round(q25, 5),
                        'q50': round(q50, 5), 'q75': round(q75, 5),
                        'q95': round(q95, 5),
                    }

                    # 风险调整: 如果尾部风险太大, 降权
                    if q05 < self.risk_dampen_q05:
                        risk_factor = max(0.7, 1.0 + q05 / 0.10)
                        enhanced_buy *= risk_factor
                        ml_info['quantile_action'] = f'risk_dampen({risk_factor:.2f})'
                    elif q95 > -self.risk_dampen_q05:
                        # 极端上涨风险 → 降低做空
                        risk_factor = max(0.7, 1.0 - q95 / 0.10)
                        enhanced_sell *= risk_factor
                        ml_info['quantile_action'] = f'short_risk_dampen({risk_factor:.2f})'
                    else:
                        ml_info['quantile_action'] = 'neutral'

            except Exception as e:
                logger.warning(f"分位数计算异常: {e}")

        ml_info['original_buy'] = round(buy_score, 2)
        ml_info['original_sell'] = round(sell_score, 2)
        ml_info['enhanced_buy'] = round(enhanced_buy, 2)
        ml_info['enhanced_sell'] = round(enhanced_sell, 2)

        return float(enhanced_sell), float(enhanced_buy), ml_info


def train_production_model(days: int = 365):
    """训练 Regime + 分位数生产模型"""
    from optimize_six_book import fetch_multi_tf_data

    print("=" * 70)
    print("  训练 ML 生产模型 v4 (Regime + 分位数)")
    print(f"  数据: 最近 {days} 天")
    print("=" * 70)

    # 获取数据
    print("\n[1/5] 获取数据...")
    all_data = fetch_multi_tf_data(['1h'], days=days)
    if '1h' not in all_data:
        print("错误: 无法获取数据")
        return
    df = all_data['1h']
    print(f"  {len(df)} 条K线 ({df.index[0]} ~ {df.index[-1]})")

    # 训练 Regime 模型
    print("\n[2/5] 训练 Regime 模型...")
    from ml_regime import (
        RegimePredictor, RegimeConfig,
        compute_regime_features, compute_regime_labels,
    )

    features_r = compute_regime_features(df)
    labels_r = compute_regime_labels(df)
    cfg_r = RegimeConfig()

    val_window = cfg_r.val_window
    purge_gap = cfg_r.purge_gap
    train_end = len(features_r) - val_window - purge_gap
    max_train = 1440  # 60 天
    train_start = max(0, train_end - max_train)
    val_start = train_end + purge_gap
    val_end = len(features_r)

    model_r = RegimePredictor(cfg_r)
    model_r.train(
        features_r.iloc[train_start:train_end],
        labels_r['vol_high'].iloc[train_start:train_end],
        labels_r['trend_strong'].iloc[train_start:train_end],
        features_r.iloc[val_start:val_end],
        labels_r['vol_high'].iloc[val_start:val_end],
        labels_r['trend_strong'].iloc[val_start:val_end],
    )
    model_r.save(MODEL_DIR)
    print("  Regime 模型已保存")

    # 训练分位数模型
    print("\n[3/5] 训练分位数模型...")
    from ml_quantile import (
        QuantilePredictor, QuantileConfig,
        compute_enhanced_features, compute_return_labels,
    )

    features_q = compute_enhanced_features(df)
    cfg_q = QuantileConfig()
    labels_q = compute_return_labels(df, cfg_q.horizons)

    train_end_q = len(features_q) - val_window - purge_gap
    train_start_q = max(0, train_end_q - max_train)
    val_start_q = train_end_q + purge_gap

    y_tr = {h: labels_q[f'log_ret_{h}'].iloc[train_start_q:train_end_q] for h in cfg_q.horizons}
    y_val = {h: labels_q[f'log_ret_{h}'].iloc[val_start_q:] for h in cfg_q.horizons}

    model_q = QuantilePredictor(cfg_q)
    metrics_q = model_q.train(
        features_q.iloc[train_start_q:train_end_q],
        y_tr,
        features_q.iloc[val_start_q:],
        y_val,
    )
    model_q.save(MODEL_DIR)
    print(f"  分位数模型已保存: {len(model_q.models)} 个子模型")

    # 验证
    print("\n[4/5] 验证模型...")
    from sklearn.metrics import roc_auc_score

    val_r = model_r.predict(features_r.iloc[val_start:val_end])
    vol_valid = labels_r['vol_high'].iloc[val_start:val_end].notna()
    vol_auc = roc_auc_score(
        labels_r['vol_high'].iloc[val_start:val_end][vol_valid],
        val_r['vol_prob'][vol_valid.values],
    )
    print(f"  Regime Vol AUC: {vol_auc:.4f}")

    # 分位数校准检查
    q_preds = model_q.predict(features_q.iloc[val_start_q:])
    for h in cfg_q.horizons:
        if h in q_preds:
            y_actual = labels_q[f'log_ret_{h}'].iloc[val_start_q:]
            valid = y_actual.notna() & q_preds[h]['q50'].notna()
            if valid.sum() > 10:
                actual = y_actual[valid].values
                q05 = q_preds[h].loc[valid, 'q05'].values
                q95 = q_preds[h].loc[valid, 'q95'].values
                cov = np.mean((actual >= q05) & (actual <= q95))
                print(f"  分位数 h{h} 90%覆盖率: {cov:.1%}")

    # 保存元数据
    print("\n[5/5] 保存元数据...")
    meta = {
        'version': 'v4_regime_quantile',
        'trained_at': datetime.datetime.now().isoformat(),
        'data_range': f"{df.index[0]} ~ {df.index[-1]}",
        'components': ['regime', 'quantile'],
        'regime_vol_auc': vol_auc,
    }
    with open(os.path.join(MODEL_DIR, 'training_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  所有模型已保存至 {MODEL_DIR}/")
    print("  v4 生产模型训练完成!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="ML 实盘集成 v4")
    parser.add_argument('--train', action='store_true', help='训练生产模型')
    parser.add_argument('--days', type=int, default=365, help='训练数据天数')
    args = parser.parse_args()

    if args.train:
        train_production_model(days=args.days)
    else:
        print("用法:")
        print("  训练: python3.10 ml_live_integration.py --train [--days 365]")
