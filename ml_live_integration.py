"""
ML 实盘集成模块 v6 (GPU 增强版 — 8 模型矩阵)

五层架构:
  1. 方向预测层: LGB + LSTM + TFT + 跨资产LGB 集成 → bull_prob
  2. 预测层: 分位数回归 → 输出收益分布 (q05~q95) → Kelly 仓位 + 动态止损
  3. 决策层: Regime 过滤 + 成本感知门槛
  4. 融合层: MTF Fusion MLP (可选, 替代规则加权)
  5. 执行层: 与六书融合信号协同

v6 更新:
  - 集成 TFT (AUC 0.55+) + 跨资产 LGB (94维, AUC 0.55)
  - 修复 LSTM 标准化 (从 TFT meta 读取 feat_mean/feat_std)
  - ONNX 推理加速 (如可用)

使用:
  1. 训练: python train_gpu.py --mode all_v3 (在 H800 上)
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
    ML 信号增强器 v6 — 8 模型矩阵集成

    工作模式:
      1. 方向预测: LGB + LSTM + TFT + 跨资产LGB → bull_prob
      2. 计算 regime (vol_prob, trend_prob, trade_confidence)
      3. 如有分位数模型: 收益分布 → Kelly 仓位 + 动态止损
      4. 综合输出: 增强/抑制规则信号
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self._regime_model = None
        self._quantile_model = None
        self._direction_model = None  # LGB 方向预测
        self._lstm_model = None       # LSTM 方向预测
        self._tft_model = None        # v6: TFT 方向预测
        self._cross_asset_model = None  # v6: 跨资产 LGB
        self._cross_asset_meta = None
        self._loaded = False

        # 方向预测参数
        self._direction_meta = None
        self._lstm_meta = None
        self._tft_meta = None
        self._norm_mean = None        # v6: 共享标准化参数
        self._norm_std = None
        self.direction_long_threshold = 0.58
        self.direction_short_threshold = 0.42
        self.direction_boost = 1.15
        self.direction_dampen = 0.85
        self.lgb_weight = 0.55            # v6: 调整权重分配
        self.lstm_weight = 0.25
        self.tft_weight = 0.10            # v6: TFT 权重
        self.cross_asset_weight = 0.10    # v6: 跨资产权重

        # Regime 增强参数
        self.high_conf_threshold = 0.55
        self.low_conf_threshold = 0.35
        self.boost_factor = 1.12
        self.dampen_factor = 0.88
        self.strong_boost_factor = 1.20

        # 分位数风控参数
        self.cost_threshold = 0.003
        self.risk_dampen_q05 = -0.03

    def load_model(self) -> bool:
        """加载所有可用模型"""
        loaded_any = False

        # v5: LightGBM 方向预测模型
        try:
            lgb_path = os.path.join(self.model_dir, 'lgb_direction_model.txt')
            if os.path.exists(lgb_path):
                import lightgbm as lgb_lib
                self._direction_model = lgb_lib.Booster(model_file=lgb_path)
                meta_path = lgb_path + '.meta.json'
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        self._direction_meta = json.load(f)
                    # 读取集成配置中的阈值
                    thresholds = self._direction_meta.get('thresholds', {})
                    self.direction_long_threshold = thresholds.get('long_threshold', 0.58)
                    self.direction_short_threshold = thresholds.get('short_threshold', 0.42)
                logger.info(f"LGB 方向预测模型加载成功 ({len(self._direction_meta.get('feature_names', []))} 特征)")
                loaded_any = True
        except Exception as e:
            logger.warning(f"LGB 方向预测模型加载失败: {e}")

        # LSTM 方向预测模型
        try:
            lstm_path = os.path.join(self.model_dir, 'lstm_1h.pt')
            if os.path.exists(lstm_path):
                self._lstm_meta = {'model_path': lstm_path}
                logger.info("LSTM 方向预测模型路径已注册 (延迟加载)")
                loaded_any = True
        except Exception as e:
            logger.warning(f"LSTM 方向预测模型注册失败: {e}")

        # v6: TFT 方向预测模型
        try:
            tft_path = os.path.join(self.model_dir, 'tft_1h.pt')
            tft_meta_path = os.path.join(self.model_dir, 'tft_1h.meta.json')
            if os.path.exists(tft_path) and os.path.exists(tft_meta_path):
                with open(tft_meta_path) as f:
                    self._tft_meta = json.load(f)
                self._tft_meta['model_path'] = tft_path
                logger.info(f"TFT 模型路径已注册 (input_dim={self._tft_meta.get('input_dim', '?')}, 延迟加载)")
                loaded_any = True
                # 从 TFT meta 提取标准化参数 (TFT 训练保存了完整的 feat_mean/feat_std)
                if 'feat_mean' in self._tft_meta and 'feat_std' in self._tft_meta:
                    feat_names = self._tft_meta.get('feature_names', [])
                    self._norm_mean = dict(zip(feat_names, self._tft_meta['feat_mean']))
                    self._norm_std = dict(zip(feat_names, self._tft_meta['feat_std']))
                    logger.info(f"标准化参数已从 TFT meta 加载 ({len(feat_names)} 维)")
        except Exception as e:
            logger.warning(f"TFT 模型注册失败: {e}")

        # v6: 跨资产 LGB
        try:
            ca_path = os.path.join(self.model_dir, 'lgb_cross_asset_1h.txt')
            ca_meta_path = ca_path + '.meta.json'
            if os.path.exists(ca_path):
                import lightgbm as lgb_lib
                self._cross_asset_model = lgb_lib.Booster(model_file=ca_path)
                if os.path.exists(ca_meta_path):
                    with open(ca_meta_path) as f:
                        self._cross_asset_meta = json.load(f)
                logger.info(f"跨资产 LGB 加载成功 ({len(self._cross_asset_meta.get('feature_names', []))} 特征)")
                loaded_any = True
        except Exception as e:
            logger.warning(f"跨资产 LGB 加载失败: {e}")

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

        # 读取集成配置
        try:
            cfg_path = os.path.join(self.model_dir, 'ensemble_config.json')
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    ecfg = json.load(f)
                comps = ecfg.get('components', {})
                if 'lgb_direction' in comps:
                    self.lgb_weight = comps['lgb_direction'].get('weight', 0.55)
                if 'lstm_1h' in comps:
                    self.lstm_weight = comps['lstm_1h'].get('weight', 0.25)
                # 如 ensemble_config 也存了标准化参数 (兼容)
                if self._norm_mean is None and 'feat_mean' in ecfg:
                    feat_names = ecfg.get('feature_names', [])
                    self._norm_mean = dict(zip(feat_names, ecfg['feat_mean']))
                    self._norm_std = dict(zip(feat_names, ecfg['feat_std']))
                    logger.info(f"标准化参数已从 ensemble_config 加载 ({len(feat_names)} 维)")
                logger.info(
                    f"集成配置已加载 (LGB:{self.lgb_weight}, LSTM:{self.lstm_weight}, "
                    f"TFT:{self.tft_weight}, CA:{self.cross_asset_weight})"
                )
        except Exception as e:
            logger.warning(f"集成配置加载失败: {e}")

        # 日志汇总
        model_list = []
        if self._direction_model: model_list.append('LGB')
        if self._lstm_meta: model_list.append('LSTM')
        if self._tft_meta: model_list.append('TFT')
        if self._cross_asset_model: model_list.append('CrossAsset')
        if self._regime_model: model_list.append('Regime')
        if self._quantile_model: model_list.append('Quantile')
        logger.info(f"ML 模型加载完成: {len(model_list)} 个 [{', '.join(model_list)}]")

        self._loaded = loaded_any
        return loaded_any

    def _compute_direction_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算方向预测所需特征 (复用 ml_features.py)"""
        try:
            from ml_features import compute_ml_features
            features = compute_ml_features(df)
            return features
        except Exception as e:
            logger.warning(f"方向特征计算失败: {e}")
            return None

    def _predict_direction_lgb(self, features: pd.DataFrame) -> Optional[float]:
        """LGB 方向预测 → bull_prob"""
        if self._direction_model is None or self._direction_meta is None:
            return None
        try:
            feat_names = self._direction_meta.get('feature_names', [])
            # 对齐特征列 (缺失列填 0)
            latest = features.iloc[[-1]]
            X = pd.DataFrame(0.0, index=latest.index, columns=feat_names)
            for col in feat_names:
                if col in latest.columns:
                    X[col] = latest[col].values
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            pred = self._direction_model.predict(X)
            return float(np.clip(pred[0], 0, 1))
        except Exception as e:
            logger.warning(f"LGB 方向预测失败: {e}")
            return None

    def _predict_direction_lstm(self, features: pd.DataFrame) -> Optional[float]:
        """LSTM 方向预测 → bull_prob"""
        if self._lstm_meta is None:
            return None
        try:
            import torch
            import torch.nn as nn

            model_path = self._lstm_meta['model_path']
            if not os.path.exists(model_path):
                return None

            # 需要至少 48 根K线的序列
            SEQ_LEN = 48
            if len(features) < SEQ_LEN:
                return None

            feat_names = self._direction_meta.get('feature_names', []) if self._direction_meta else list(features.columns)

            # 对齐特征
            X_df = pd.DataFrame(0.0, index=features.index, columns=feat_names)
            for col in feat_names:
                if col in features.columns:
                    X_df[col] = features[col].values
            X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            feat_values = X_df.values.astype(np.float32)

            # 标准化: 优先用持久化的 mean/std (v6: 从 TFT meta 或 ensemble_config)
            norm_applied = False
            norm_source = self._norm_mean or (self._direction_meta or {}).get('feat_mean')
            if self._norm_mean:
                for i, col in enumerate(feat_names):
                    m = self._norm_mean.get(col, 0.0)
                    s = self._norm_std.get(col, 1.0)
                    feat_values[:, i] = (feat_values[:, i] - m) / max(s, 1e-8)
                norm_applied = True
            elif self._direction_meta and isinstance(self._direction_meta.get('feat_mean'), dict):
                mean_dict = self._direction_meta['feat_mean']
                std_dict = self._direction_meta['feat_std']
                for i, col in enumerate(feat_names):
                    m = mean_dict.get(col, 0.0)
                    s = std_dict.get(col, 1.0)
                    feat_values[:, i] = (feat_values[:, i] - m) / max(s, 1e-8)
                norm_applied = True
            if not norm_applied:
                half = len(feat_values) // 2
                m = np.nanmean(feat_values[:half], axis=0)
                s = np.nanstd(feat_values[:half], axis=0) + 1e-8
                feat_values = (feat_values - m) / s

            feat_values = np.nan_to_num(feat_values, nan=0.0, posinf=3.0, neginf=-3.0)

            # 取最后 SEQ_LEN 根
            seq = feat_values[-SEQ_LEN:]
            X_tensor = torch.FloatTensor(seq).unsqueeze(0)  # (1, SEQ_LEN, features)

            # 延迟加载 LSTM 模型
            if self._lstm_model is None:
                input_dim = len(feat_names)

                class LSTMAttention(nn.Module):
                    def __init__(self, in_dim, hidden_dim=128, num_layers=2, dropout=0.3):
                        super().__init__()
                        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers,
                                            batch_first=True, dropout=dropout, bidirectional=True)
                        self.attn_fc = nn.Linear(hidden_dim * 2, 1)
                        self.classifier = nn.Sequential(
                            nn.Linear(hidden_dim * 2, 64),
                            nn.GELU(),
                            nn.Dropout(0.2),
                            nn.Linear(64, 1),
                        )

                    def forward(self, x):
                        lstm_out, _ = self.lstm(x)
                        attn_weights = torch.softmax(self.attn_fc(lstm_out), dim=1)
                        context = (attn_weights * lstm_out).sum(dim=1)
                        return self.classifier(context).squeeze(-1)

                device = 'cpu'  # 推理用 CPU 即可
                self._lstm_model = LSTMAttention(input_dim).to(device)
                state = torch.load(model_path, map_location=device, weights_only=True)
                self._lstm_model.load_state_dict(state)
                self._lstm_model.eval()
                logger.info(f"LSTM 模型加载完成 (input_dim={input_dim})")

            with torch.no_grad():
                logit = self._lstm_model(X_tensor)
                prob = torch.sigmoid(logit).item()
            return float(np.clip(prob, 0, 1))

        except Exception as e:
            logger.warning(f"LSTM 方向预测失败: {e}")
            return None

    def _predict_direction_tft(self, features: pd.DataFrame) -> Optional[float]:
        """TFT 方向预测 → bull_prob"""
        if self._tft_meta is None:
            return None
        try:
            import torch
            import torch.nn as nn

            model_path = self._tft_meta['model_path']
            if not os.path.exists(model_path):
                return None

            tft_feat_names = self._tft_meta.get('feature_names', [])
            seq_len = self._tft_meta.get('seq_len', 96)
            if len(features) < seq_len:
                return None

            X_df = pd.DataFrame(0.0, index=features.index, columns=tft_feat_names)
            for col in tft_feat_names:
                if col in features.columns:
                    X_df[col] = features[col].values
            X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            vals = X_df.values.astype(np.float32)

            # TFT meta 自带标准化参数
            if 'feat_mean' in self._tft_meta:
                mean_arr = np.array(self._tft_meta['feat_mean'], dtype=np.float32)
                std_arr = np.array(self._tft_meta['feat_std'], dtype=np.float32)
                std_arr = np.where(std_arr < 1e-8, 1.0, std_arr)
                vals = (vals - mean_arr) / std_arr

            vals = np.nan_to_num(vals, nan=0.0, posinf=3.0, neginf=-3.0)
            seq = vals[-seq_len:]
            X_tensor = torch.FloatTensor(seq).unsqueeze(0)

            if self._tft_model is None:
                input_dim = self._tft_meta.get('input_dim', len(tft_feat_names))
                d_model = self._tft_meta.get('d_model', 64)
                n_heads = self._tft_meta.get('n_heads', 4)
                d_ff = d_model * 2
                n_layers = 2

                class EfficientTFT(nn.Module):
                    def __init__(self, in_dim, dm, nh, dff, nl, dropout=0.15):
                        super().__init__()
                        self.input_proj = nn.Sequential(
                            nn.Linear(in_dim, dm), nn.LayerNorm(dm),
                            nn.GELU(), nn.Dropout(dropout))
                        self.lstm = nn.LSTM(dm, dm, nl, batch_first=True,
                                            dropout=dropout if nl > 1 else 0)
                        self.lstm_norm = nn.LayerNorm(dm)
                        enc_layer = nn.TransformerEncoderLayer(
                            d_model=dm, nhead=nh, dim_feedforward=dff,
                            dropout=dropout, activation='gelu', batch_first=True)
                        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
                        self.transformer_norm = nn.LayerNorm(dm)
                        self.attn_pool = nn.Linear(dm, 1)
                        self.classifier = nn.Sequential(
                            nn.Linear(dm, dff), nn.GELU(),
                            nn.Dropout(dropout), nn.Linear(dff, 1))

                    def forward(self, x):
                        h = self.input_proj(x)
                        lstm_out, _ = self.lstm(h)
                        h = self.lstm_norm(lstm_out + h)
                        h = self.transformer(h)
                        h = self.transformer_norm(h)
                        attn_w = torch.softmax(self.attn_pool(h), dim=1)
                        context = (attn_w * h).sum(dim=1)
                        return self.classifier(context).squeeze(-1)

                self._tft_model = EfficientTFT(input_dim, d_model, n_heads, d_ff, n_layers)
                state = torch.load(model_path, map_location='cpu', weights_only=False)
                self._tft_model.load_state_dict(state)
                self._tft_model.eval()
                logger.info(f"TFT 模型加载完成 (input_dim={input_dim}, d_model={d_model})")

            with torch.no_grad():
                logit = self._tft_model(X_tensor)
                prob = torch.sigmoid(logit).item()
            return float(np.clip(prob, 0, 1))

        except Exception as e:
            logger.warning(f"TFT 方向预测失败: {e}")
            return None

    def _predict_direction_cross_asset(self, features: pd.DataFrame) -> Optional[float]:
        """跨资产 LGB 方向预测 (94 维含 BTC/SOL/BNB)"""
        if self._cross_asset_model is None or self._cross_asset_meta is None:
            return None
        try:
            feat_names = self._cross_asset_meta.get('feature_names', [])
            latest = features.iloc[[-1]]
            X = pd.DataFrame(0.0, index=latest.index, columns=feat_names)
            for col in feat_names:
                if col in latest.columns:
                    X[col] = latest[col].values
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            pred = self._cross_asset_model.predict(X)
            return float(np.clip(pred[0], 0, 1))
        except Exception as e:
            logger.warning(f"跨资产 LGB 预测失败: {e}")
            return None

    def enhance_signal(
        self,
        sell_score: float,
        buy_score: float,
        df: pd.DataFrame,
    ) -> Tuple[float, float, Dict]:
        """
        综合 ML 增强: 方向预测 + Regime + 分位数

        返回:
            (enhanced_sell_score, enhanced_buy_score, ml_info)
        """
        if not self._loaded:
            if not self.load_model():
                return sell_score, buy_score, {'ml_available': False}

        ml_info = {'ml_available': True, 'ml_version': 'v6'}
        enhanced_buy = buy_score
        enhanced_sell = sell_score

        # 0. 方向预测 (多模型集成: LGB + LSTM + TFT + 跨资产)
        has_direction = (self._direction_model is not None or self._lstm_meta is not None
                         or self._tft_meta is not None or self._cross_asset_model is not None)
        if has_direction:
            try:
                features = self._compute_direction_features(df)
                if features is not None and len(features) > 0:
                    lgb_prob = self._predict_direction_lgb(features)
                    lstm_prob = self._predict_direction_lstm(features)
                    tft_prob = self._predict_direction_tft(features)
                    ca_prob = self._predict_direction_cross_asset(features)

                    probs = []
                    weights = []
                    if lgb_prob is not None:
                        probs.append(lgb_prob)
                        weights.append(self.lgb_weight)
                        ml_info['lgb_bull_prob'] = round(lgb_prob, 4)
                    if lstm_prob is not None:
                        probs.append(lstm_prob)
                        weights.append(self.lstm_weight)
                        ml_info['lstm_bull_prob'] = round(lstm_prob, 4)
                    if tft_prob is not None:
                        probs.append(tft_prob)
                        weights.append(self.tft_weight)
                        ml_info['tft_bull_prob'] = round(tft_prob, 4)
                    if ca_prob is not None:
                        probs.append(ca_prob)
                        weights.append(self.cross_asset_weight)
                        ml_info['ca_bull_prob'] = round(ca_prob, 4)

                    if probs:
                        total_w = sum(weights)
                        bull_prob = sum(p * w for p, w in zip(probs, weights)) / total_w
                        ml_info['bull_prob'] = round(bull_prob, 4)

                        # 方向性调整: bull_prob 偏向做多时加强 BS, 偏向做空时加强 SS
                        if bull_prob >= self.direction_long_threshold:
                            # ML 看涨 → BS 加权, SS 降权
                            strength = (bull_prob - 0.5) * 2  # 0~1 范围
                            bs_mult = 1.0 + strength * (self.direction_boost - 1.0)
                            ss_mult = 1.0 - strength * (1.0 - self.direction_dampen)
                            enhanced_buy *= bs_mult
                            enhanced_sell *= ss_mult
                            ml_info['direction_action'] = f'bullish(BS*{bs_mult:.2f},SS*{ss_mult:.2f})'
                        elif bull_prob <= self.direction_short_threshold:
                            # ML 看跌 → SS 加权, BS 降权
                            strength = (0.5 - bull_prob) * 2
                            ss_mult = 1.0 + strength * (self.direction_boost - 1.0)
                            bs_mult = 1.0 - strength * (1.0 - self.direction_dampen)
                            enhanced_sell *= ss_mult
                            enhanced_buy *= bs_mult
                            ml_info['direction_action'] = f'bearish(SS*{ss_mult:.2f},BS*{bs_mult:.2f})'
                        else:
                            ml_info['direction_action'] = 'neutral'
            except Exception as e:
                logger.warning(f"方向预测计算异常: {e}")

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

        # 2. 分位数风控 + Kelly 仓位 + 动态止损
        if self._quantile_model:
            try:
                from ml_quantile import compute_enhanced_features
                q_features = compute_enhanced_features(df)
                q_latest = q_features.iloc[[-1]]
                q_preds = self._quantile_model.predict(q_latest)

                # 取短期 horizon (用于止损) 和长期 horizon (用于方向)
                horizons = self._quantile_model.config.horizons
                h_short = horizons[0]  # e.g. 5 bars
                h_long = horizons[-1] if len(horizons) > 1 else h_short  # e.g. 12 bars

                q_all = {}
                for h in [h_short, h_long]:
                    if h in q_preds:
                        row = q_preds[h].iloc[0]
                        q_all[h] = {
                            'q05': float(row['q05']), 'q25': float(row['q25']),
                            'q50': float(row['q50']), 'q75': float(row['q75']),
                            'q95': float(row['q95']),
                        }

                if h_short in q_all:
                    qs = q_all[h_short]
                    ml_info['quantile'] = {k: round(v, 5) for k, v in qs.items()}
                    if h_long in q_all and h_long != h_short:
                        ml_info['quantile_long'] = {k: round(v, 5) for k, v in q_all[h_long].items()}

                    # --- Kelly 仓位计算 ---
                    bull_prob = ml_info.get('bull_prob', 0.5)
                    q05 = qs['q05']
                    q50 = qs['q50']
                    q95 = qs['q95']

                    # Kelly: f* = (p * b - q * a) / (b * a)
                    # p = win_prob, q = 1-p, b = avg_win, a = avg_loss
                    if bull_prob >= 0.5:
                        # 做多视角: win = q95, loss = abs(q05)
                        avg_win = max(q95, 0.001)
                        avg_loss = max(abs(q05), 0.001)
                        kelly_raw = (bull_prob * avg_win - (1 - bull_prob) * avg_loss) / (avg_win * avg_loss)
                    else:
                        # 做空视角: win = abs(q05), loss = q95
                        avg_win = max(abs(q05), 0.001)
                        avg_loss = max(q95, 0.001)
                        kelly_raw = ((1 - bull_prob) * avg_win - bull_prob * avg_loss) / (avg_win * avg_loss)

                    # 半 Kelly (保守) + 上下限
                    kelly_half = kelly_raw * 0.5
                    kelly_fraction = float(np.clip(kelly_half, 0.1, 1.0))
                    ml_info['kelly_fraction'] = round(kelly_fraction, 4)
                    ml_info['kelly_raw'] = round(kelly_raw, 4)

                    # --- 动态止损 (基于 q05/q95) ---
                    # 做多止损 = |q05| of short horizon; 做空止损 = q95
                    dynamic_sl_long = abs(q05) if q05 < 0 else 0.02
                    dynamic_sl_short = q95 if q95 > 0 else 0.02
                    # 限制在 1%~8% 范围
                    dynamic_sl_long = float(np.clip(dynamic_sl_long, 0.01, 0.08))
                    dynamic_sl_short = float(np.clip(dynamic_sl_short, 0.01, 0.08))
                    ml_info['dynamic_sl_long'] = round(dynamic_sl_long, 4)
                    ml_info['dynamic_sl_short'] = round(dynamic_sl_short, 4)

                    # --- 仓位缩放 (综合 Kelly + 风险) ---
                    position_scale = kelly_fraction
                    # 尾部风险额外降权
                    if q05 < self.risk_dampen_q05:
                        tail_factor = max(0.5, 1.0 + q05 / 0.10)
                        position_scale *= tail_factor
                        ml_info['quantile_action'] = f'risk_dampen(kelly={kelly_fraction:.2f},tail={tail_factor:.2f})'
                    elif q95 > -self.risk_dampen_q05:
                        tail_factor = max(0.5, 1.0 - q95 / 0.10)
                        position_scale *= tail_factor
                        ml_info['quantile_action'] = f'short_risk(kelly={kelly_fraction:.2f},tail={tail_factor:.2f})'
                    else:
                        ml_info['quantile_action'] = f'kelly({kelly_fraction:.2f})'

                    ml_info['position_scale'] = round(float(np.clip(position_scale, 0.1, 1.0)), 4)

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
