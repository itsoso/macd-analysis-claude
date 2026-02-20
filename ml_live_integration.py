"""
ML 实盘集成模块 v6 (GPU 增强版 — 8 模型矩阵 + Stacking Ensemble)

五层架构:
  1. 方向预测层: Stacking(LGB+XGB+LSTM+TFT→Meta) 或 LGB+LSTM+TFT+跨资产LGB 加权
  2. 预测层: 分位数回归 → 输出收益分布 (q05~q95) → Kelly 仓位 + 动态止损
  3. 决策层: Regime 过滤 + 成本感知门槛
  4. 融合层: MTF Fusion MLP (可选, 替代规则加权)
  5. 执行层: 与六书融合信号协同

v6 更新:
  - 集成 TFT (AUC 0.55+) + 跨资产 LGB (94维, AUC 0.55)
  - 修复 LSTM 标准化 (从 TFT meta 读取 feat_mean/feat_std)
  - ONNX 推理加速 (如可用)
  - Stacking Ensemble: 4 基模型 OOF → LogisticRegression 元学习器

使用:
  1. 训练: python train_gpu.py --mode all_v3 (在 H800 上)
  2. Stacking: python train_gpu.py --mode stacking --tf 1h
     ⚠ 三轮迭代结论: 小数据集(~4k)下 Stacking 严重过拟合(AUC -13%)，建议样本 20000+ 再训；当前优先用加权 LGB+LSTM+TFT。
  3. 实盘: live_signal_generator.py 自动调用 MLSignalEnhancer.enhance_signal()
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


def _env_flag(name: str, default: bool) -> bool:
    """解析布尔环境变量。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "y")


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


class MLSignalEnhancer:
    """
    ML 信号增强器 v6 — 8 模型矩阵集成 + Stacking Ensemble

    工作模式:
      1. 方向预测: Stacking(优先) 或 LGB+LSTM+TFT+跨资产LGB → bull_prob
      2. 计算 regime (vol_prob, trend_prob, trade_confidence)
      3. 如有分位数模型: 收益分布 → Kelly 仓位 + 动态止损
      4. 综合输出: 增强/抑制规则信号
    """

    def __init__(
        self,
        model_dir: str = MODEL_DIR,
        gpu_inference_url: str = "",
        inference_device: str = "",
    ):
        self.model_dir = model_dir
        self.gpu_inference_url = (gpu_inference_url or "").strip() or os.environ.get("ML_GPU_INFERENCE_URL", "").strip()
        self.gpu_inference_timeout = int(os.environ.get("ML_GPU_INFERENCE_TIMEOUT", "5"))
        self._regime_model = None
        self._quantile_model = None
        self._direction_model = None  # LGB 方向预测
        self._lstm_model = None       # LSTM 方向预测
        self._tft_model = None        # v6: TFT 方向预测
        self._tft_onnx_session = None  # v6: TFT ONNX 推理会话（可选加速）
        self._cross_asset_model = None  # v6: 跨资产 LGB
        self._cross_asset_meta = None
        self._loaded = False

        # Stacking ensemble
        self._stacking_meta_model = None  # LogisticRegression 元学习器
        self._stacking_config = None      # stacking_meta.json 配置
        self._stacking_lgb = None         # stacking 专用 LGB
        self._stacking_xgb = None         # stacking 专用 XGBoost
        self._stacking_lstm = None        # stacking 专用 LSTM
        self._stacking_tft = None         # stacking 专用 TFT

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
        # 推理设备: 显式参数 > 环境变量 > 自动探测
        requested_device = (inference_device or os.environ.get("ML_INFERENCE_DEVICE", "")).strip().lower()
        if requested_device not in ("", "cpu", "cuda"):
            logger.warning(f"未知推理设备 '{requested_device}'，将使用自动探测")
            requested_device = ""

        cuda_available = False
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False

        if requested_device == "cuda" and not cuda_available:
            logger.warning("请求 CUDA 推理但当前不可用，回退到 CPU")
            requested_device = "cpu"

        if requested_device:
            self._inference_device = requested_device
        else:
            self._inference_device = "cuda" if cuda_available else "cpu"
        self.lstm_weight = 0.25
        self.tft_weight = 0.10            # v6: TFT 权重
        self.cross_asset_weight = 0.10    # v6: 跨资产权重

        # Stacking / 跨资产门控参数
        self.stacking_enabled = _env_flag("ML_ENABLE_STACKING", True)
        self.stacking_target_timeframe = (
            os.environ.get("ML_STACKING_TIMEFRAME")
            or os.environ.get("ML_TIMEFRAME")
            or "1h"
        ).strip()
        self.stacking_min_val_auc = float(os.environ.get("ML_STACKING_MIN_VAL_AUC", "0.53"))
        self.stacking_min_test_auc = float(os.environ.get("ML_STACKING_MIN_TEST_AUC", "0.52"))
        self.stacking_min_oof_auc = float(os.environ.get("ML_STACKING_MIN_OOF_AUC", "0.53"))
        self.stacking_max_oof_test_gap = float(os.environ.get("ML_STACKING_MAX_OOF_TEST_GAP", "0.10"))
        self.stacking_min_feature_coverage_73 = _clamp01(
            float(os.environ.get("ML_STACKING_MIN_FEATURE_COVERAGE_73", "0.90"))
        )
        self.stacking_min_feature_coverage_94 = _clamp01(
            float(os.environ.get("ML_STACKING_MIN_FEATURE_COVERAGE_94", "0.80"))
        )
        self.cross_asset_min_feature_coverage = _clamp01(
            float(os.environ.get("ML_CROSS_ASSET_MIN_FEATURE_COVERAGE", "0.80"))
        )
        self._stacking_disabled_reason = None

        # Regime 增强参数
        self.high_conf_threshold = 0.55
        self.low_conf_threshold = 0.35
        self.boost_factor = 1.12
        self.dampen_factor = 0.88
        self.strong_boost_factor = 1.20

        # 分位数风控参数
        self.cost_threshold = 0.003
        self.risk_dampen_q05 = -0.03

    @staticmethod
    def _feature_coverage(features: pd.DataFrame, required_cols) -> Tuple[int, int, float]:
        """返回 (命中特征数, 需要特征数, 覆盖率)。"""
        cols = list(required_cols or [])
        total = len(cols)
        if total == 0:
            return 0, 0, 1.0
        present = sum(1 for c in cols if c in features.columns)
        return present, total, present / total

    def _iter_stacking_candidates(self):
        """按优先级返回可尝试的 stacking 配置文件名。"""
        candidates = []
        tf = (self.stacking_target_timeframe or "").strip()
        if tf:
            candidates.append((f"stacking_meta_{tf}.json", f"stacking_meta_{tf}.pkl"))
        candidates.append(("stacking_meta.json", "stacking_meta.pkl"))

        seen = set()
        for pair in candidates:
            if pair not in seen:
                seen.add(pair)
                yield pair

    def _stacking_quality_gate(self, cfg: Dict) -> Tuple[bool, str]:
        """检查 stacking 模型是否达到上线质量门槛。"""
        if not self.stacking_enabled:
            return False, "disabled_by_env"

        tf_cfg = str(cfg.get("timeframe", "")).strip()
        if self.stacking_target_timeframe and tf_cfg and tf_cfg != self.stacking_target_timeframe:
            return False, f"timeframe_mismatch({tf_cfg}!={self.stacking_target_timeframe})"

        def _to_float(v):
            try:
                return float(v)
            except Exception:
                return None

        val_auc = _to_float(cfg.get("val_auc"))
        test_auc = _to_float(cfg.get("test_auc"))
        oof_auc = _to_float(cfg.get("oof_meta_auc"))

        if val_auc is not None and val_auc < self.stacking_min_val_auc:
            return False, f"val_auc_too_low({val_auc:.4f}<{self.stacking_min_val_auc:.4f})"
        if test_auc is not None and test_auc < self.stacking_min_test_auc:
            return False, f"test_auc_too_low({test_auc:.4f}<{self.stacking_min_test_auc:.4f})"
        if oof_auc is not None and oof_auc < self.stacking_min_oof_auc:
            return False, f"oof_auc_too_low({oof_auc:.4f}<{self.stacking_min_oof_auc:.4f})"
        if oof_auc is not None and test_auc is not None:
            gap = oof_auc - test_auc
            if gap > self.stacking_max_oof_test_gap:
                return False, f"overfit_gap_too_large({gap:.4f}>{self.stacking_max_oof_test_gap:.4f})"
        return True, ""

    def _can_use_cross_asset(self, features: pd.DataFrame, ml_info: Optional[Dict] = None) -> bool:
        """跨资产分支特征覆盖检查，覆盖率不足则跳过。"""
        if self._cross_asset_model is None or self._cross_asset_meta is None:
            return False
        feat_names = self._cross_asset_meta.get('feature_names', [])
        present, total, coverage = self._feature_coverage(features, feat_names)
        if ml_info is not None:
            ml_info['ca_feature_coverage'] = round(coverage, 3)
        if coverage < self.cross_asset_min_feature_coverage:
            if ml_info is not None:
                ml_info['ca_skipped_reason'] = f'low_feature_coverage({present}/{total})'
            return False
        return True

    def load_model(self) -> bool:
        """加载所有可用模型"""
        loaded_any = False

        # v5: LightGBM 方向预测模型 (优先 1h 版本: 80维, AUC 0.5655)
        try:
            lgb_path_1h = os.path.join(self.model_dir, 'lgb_direction_model_1h.txt')
            lgb_path_old = os.path.join(self.model_dir, 'lgb_direction_model.txt')
            lgb_path = lgb_path_1h if os.path.exists(lgb_path_1h) else lgb_path_old
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
                feat_cnt = len((self._direction_meta or {}).get('feature_names', []))
                logger.info(f"LGB 方向预测模型加载成功 ({feat_cnt} 特征)")
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
                feat_cnt = len((self._cross_asset_meta or {}).get('feature_names', []))
                logger.info(f"跨资产 LGB 加载成功 ({feat_cnt} 特征)")
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

        # Stacking ensemble 元学习器（含 timeframe / 质量门控）
        try:
            if not self.stacking_enabled:
                self._stacking_disabled_reason = "disabled_by_env"
                logger.info("Stacking 已禁用 (ML_ENABLE_STACKING=0)")
            else:
                for meta_name, fallback_pkl_name in self._iter_stacking_candidates():
                    stacking_meta_path = os.path.join(self.model_dir, meta_name)
                    if not os.path.exists(stacking_meta_path):
                        continue

                    with open(stacking_meta_path) as f:
                        cfg = json.load(f)

                    model_files = cfg.get('model_files', {}) or {}
                    pkl_name = model_files.get('meta', fallback_pkl_name) or fallback_pkl_name
                    stacking_pkl_path = os.path.join(self.model_dir, pkl_name)
                    if not os.path.exists(stacking_pkl_path):
                        logger.warning(f"Stacking 配置存在但元学习器文件缺失: {stacking_pkl_path}")
                        self._stacking_disabled_reason = "meta_model_missing"
                        continue

                    quality_ok, reason = self._stacking_quality_gate(cfg)
                    if not quality_ok:
                        self._stacking_disabled_reason = reason
                        logger.warning(f"跳过 Stacking ({meta_name}): {reason}")
                        continue

                    import pickle
                    with open(stacking_pkl_path, 'rb') as f:
                        self._stacking_meta_model = pickle.load(f)
                    self._stacking_config = cfg
                    self._stacking_disabled_reason = None

                    # 加载 stacking 专用 LGB
                    lgb_file = model_files.get('lgb', '')
                    lgb_stk_path = os.path.join(self.model_dir, lgb_file)
                    if lgb_file and os.path.exists(lgb_stk_path):
                        import lightgbm as lgb_lib
                        self._stacking_lgb = lgb_lib.Booster(model_file=lgb_stk_path)

                    # 加载 stacking 专用 XGBoost
                    xgb_file = model_files.get('xgboost', '')
                    xgb_stk_path = os.path.join(self.model_dir, xgb_file)
                    if xgb_file and os.path.exists(xgb_stk_path):
                        import xgboost as xgb_lib
                        self._stacking_xgb = xgb_lib.Booster()
                        self._stacking_xgb.load_model(xgb_stk_path)

                    logger.info(
                        "Stacking 元学习器加载成功 "
                        "(tf=%s, val_auc=%s, test_auc=%s, source=%s)",
                        cfg.get('timeframe', '?'),
                        cfg.get('val_auc', '?'),
                        cfg.get('test_auc', '?'),
                        meta_name,
                    )
                    loaded_any = True
                    break

                if self._stacking_meta_model is None and self._stacking_disabled_reason is None:
                    self._stacking_disabled_reason = "artifact_not_found"
        except Exception as e:
            self._stacking_disabled_reason = f"load_error:{e}"
            logger.warning(f"Stacking 元学习器加载失败: {e}")

        # 日志汇总
        model_list = []
        if self._direction_model: model_list.append('LGB')
        if self._lstm_meta: model_list.append('LSTM')
        if self._tft_meta: model_list.append('TFT')
        if self._cross_asset_model: model_list.append('CrossAsset')
        if self._regime_model: model_list.append('Regime')
        if self._quantile_model: model_list.append('Quantile')
        if self._stacking_meta_model: model_list.append('Stacking')
        logger.info(f"ML 模型加载完成: {len(model_list)} 个 [{', '.join(model_list)}]")

        # 预热 Stacking LSTM/TFT（避免首次推理延迟加载失败）
        if self._stacking_meta_model is not None and self._stacking_config is not None:
            self._warmup_stacking_submodels()

        self._loaded = loaded_any
        return loaded_any

    def _warmup_stacking_submodels(self) -> None:
        """预热 Stacking LSTM/TFT 子模型，避免首次推理延迟加载失败"""
        if self._stacking_meta_model is None or self._stacking_config is None:
            return
        cfg = self._stacking_config
        feat_names_73 = cfg.get('feature_names_73', [])
        feat_names_94 = cfg.get('feature_names_94', [])
        all_cols = list(dict.fromkeys(feat_names_73 + feat_names_94))
        n_rows = 96  # TFT 最长序列
        fake_df = pd.DataFrame(0.0, index=range(n_rows), columns=all_cols)
        try:
            self._predict_stacking_lstm(fake_df, cfg)
            logger.info("Stacking LSTM 预热完成")
        except Exception as e:
            logger.warning(f"Stacking LSTM 预热失败 (仍用延迟加载): {e}")
        try:
            self._predict_stacking_tft(fake_df, cfg)
            logger.info("Stacking TFT 预热完成")
        except Exception as e:
            logger.warning(f"Stacking TFT 预热失败 (仍用延迟加载): {e}")

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

            # lstm_1h.pt 用 stacking feature_names_73 (80维) 训练；优先用该特征集和配套标准化
            if self._stacking_config and self._stacking_config.get('feature_names_73'):
                feat_names = self._stacking_config['feature_names_73']
                feat_mean_arr = np.array(self._stacking_config.get('feat_mean_73', [0.0] * len(feat_names)), dtype=np.float32)
                feat_std_arr  = np.array(self._stacking_config.get('feat_std_73',  [1.0] * len(feat_names)), dtype=np.float32)
                use_stacking_norm = True
            else:
                feat_names = self._direction_meta.get('feature_names', []) if self._direction_meta else list(features.columns)
                use_stacking_norm = False

            # 对齐特征
            X_df = pd.DataFrame(0.0, index=features.index, columns=feat_names)
            for col in feat_names:
                if col in features.columns:
                    X_df[col] = features[col].values
            X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            feat_values = X_df.values.astype(np.float32)

            # 标准化
            if use_stacking_norm:
                feat_values = (feat_values - feat_mean_arr) / np.maximum(feat_std_arr, 1e-8)
            elif self._direction_meta and isinstance(self._direction_meta.get('feat_mean'), dict):
                mean_dict = self._direction_meta['feat_mean']
                std_dict = self._direction_meta['feat_std']
                for i, col in enumerate(feat_names):
                    m = mean_dict.get(col, 0.0)
                    s = std_dict.get(col, 1.0)
                    feat_values[:, i] = (feat_values[:, i] - m) / max(s, 1e-8)
            else:
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

                dev = self._inference_device
                self._lstm_model = LSTMAttention(input_dim).to(dev)
                state = torch.load(model_path, map_location=dev, weights_only=True)
                self._lstm_model.load_state_dict(state)
                self._lstm_model.eval()
                logger.info(f"LSTM 模型加载完成 (input_dim={input_dim}, device={dev})")

            with torch.no_grad():
                logit = self._lstm_model(X_tensor.to(self._inference_device))
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

            # 优先尝试 ONNX 推理（CPU 友好，无需 PyTorch）
            onnx_path = model_path.replace('.pt', '.onnx')
            if os.path.exists(onnx_path):
                try:
                    import onnxruntime as ort
                    if self._tft_onnx_session is None:
                        self._tft_onnx_session = ort.InferenceSession(
                            onnx_path, providers=['CPUExecutionProvider'])
                        logger.info(f"TFT ONNX 加载完成 (seq={seq_len}, features={seq.shape[-1]})")
                    ort_input = {self._tft_onnx_session.get_inputs()[0].name: seq[np.newaxis].astype(np.float32)}
                    logit = float(self._tft_onnx_session.run(None, ort_input)[0][0, 0])
                    prob = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
                    return float(np.clip(prob, 0, 1))
                except Exception as onnx_err:
                    logger.debug(f"TFT ONNX 推理失败，回退 PyTorch: {onnx_err}")
                    self._tft_onnx_session = None  # 清除，下次重试

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
                dev = self._inference_device
                state = torch.load(model_path, map_location=dev, weights_only=False)
                self._tft_model.load_state_dict(state)
                self._tft_model.to(dev)
                self._tft_model.eval()
                logger.info(f"TFT 模型加载完成 (input_dim={input_dim}, d_model={d_model}, device={dev})")

            with torch.no_grad():
                logit = self._tft_model(X_tensor.to(self._inference_device))
                prob = torch.sigmoid(logit).item()
            return float(np.clip(prob, 0, 1))

        except Exception as e:
            logger.warning(f"TFT 方向预测失败: {e}")
            return None

    def _predict_direction_cross_asset(self, features: pd.DataFrame) -> Optional[float]:
        """跨资产 LGB 方向预测 (94 维含 BTC/SOL/BNB)"""
        if not self._can_use_cross_asset(features):
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

    def _predict_stacking(self, df: pd.DataFrame, ml_info: Dict) -> Optional[float]:
        """Stacking ensemble 预测: 4 基模型 → 元学习器 → bull_prob"""
        features = self._compute_direction_features(df)
        if features is None or len(features) == 0:
            return None
        return self._predict_stacking_from_features(features, ml_info)

    def _predict_stacking_from_features(self, features: pd.DataFrame, ml_info: Dict) -> Optional[float]:
        """Stacking 预测，使用已计算好的 features（供 GPU 推理 API 复用）"""
        if self._stacking_meta_model is None or self._stacking_config is None:
            return None
        if features is None or len(features) == 0:
            return None

        cfg = self._stacking_config
        feat_names_73 = cfg.get('feature_names_73', [])
        feat_names_94 = cfg.get('feature_names_94', [])

        present_73, total_73, cov_73 = self._feature_coverage(features, feat_names_73)
        present_94, total_94, cov_94 = self._feature_coverage(features, feat_names_94)
        ml_info['stacking_feature_coverage_73'] = round(cov_73, 3)
        ml_info['stacking_feature_coverage_94'] = round(cov_94, 3)
        if cov_73 < self.stacking_min_feature_coverage_73 or cov_94 < self.stacking_min_feature_coverage_94:
            ml_info['stacking_skipped_reason'] = (
                "low_feature_coverage("
                f"73:{present_73}/{total_73},94:{present_94}/{total_94})"
            )
            return None

        # 对齐 73 维特征
        latest = features.iloc[[-1]]
        X_73 = pd.DataFrame(0.0, index=latest.index, columns=feat_names_73)
        for col in feat_names_73:
            if col in latest.columns:
                X_73[col] = latest[col].values
        X_73 = X_73.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)

        # 基模型 1: LGB
        lgb_prob = 0.5
        if self._stacking_lgb is not None:
            try:
                lgb_prob = float(np.clip(self._stacking_lgb.predict(X_73)[0], 0, 1))
                ml_info['stacking_lgb_prob'] = round(lgb_prob, 4)
            except Exception as e:
                logger.warning(f"Stacking LGB 失败: {e}")

        # 基模型 2: XGBoost
        xgb_prob = 0.5
        if self._stacking_xgb is not None:
            try:
                import xgboost as xgb_lib
                dmat = xgb_lib.DMatrix(X_73)
                xgb_prob = float(np.clip(self._stacking_xgb.predict(dmat)[0], 0, 1))
                ml_info['stacking_xgb_prob'] = round(xgb_prob, 4)
            except Exception as e:
                logger.warning(f"Stacking XGBoost 失败: {e}")

        # 基模型 3: LSTM (延迟加载)
        lstm_prob = 0.5
        try:
            lstm_prob = self._predict_stacking_lstm(features, cfg)
            if lstm_prob is not None:
                ml_info['stacking_lstm_prob'] = round(lstm_prob, 4)
            else:
                lstm_prob = 0.5
        except Exception as e:
            logger.warning(f"Stacking LSTM 失败: {e}")

        # 基模型 4: TFT (延迟加载)
        tft_prob = 0.5
        try:
            tft_prob = self._predict_stacking_tft(features, cfg)
            if tft_prob is not None:
                ml_info['stacking_tft_prob'] = round(tft_prob, 4)
            else:
                tft_prob = 0.5
        except Exception as e:
            logger.warning(f"Stacking TFT 失败: {e}")

        # 组装元特征
        meta_X = np.array([[lgb_prob, xgb_prob, lstm_prob, tft_prob]])

        # 附加特征 (hvol_20 等)
        extra_features = cfg.get('extra_features', [])
        if 'hvol_20' in extra_features and 'hvol_20' in features.columns:
            hvol = float(features['hvol_20'].iloc[-1])
            if np.isnan(hvol) or np.isinf(hvol):
                hvol = 0.0
            meta_X = np.hstack([meta_X, [[hvol]]])

        # 元学习器预测
        bull_prob = float(self._stacking_meta_model.predict_proba(meta_X)[0, 1])
        bull_prob = float(np.clip(bull_prob, 0, 1))
        ml_info['stacking_bull_prob'] = round(bull_prob, 4)
        ml_info['stacking_mode'] = True
        return bull_prob

    def predict_direction_from_features(self, features: pd.DataFrame) -> Tuple[Optional[float], Dict]:
        """
        仅做方向预测，输入为已计算好的 features（供 GPU 推理 API 与 ECS 远程调用复用）。
        返回 (bull_prob, ml_info)，ml_info 仅含方向相关键。
        """
        ml_info: Dict = {}
        if features is None or len(features) == 0:
            return None, ml_info

        bull_prob = None
        if self._stacking_meta_model is None and self._stacking_disabled_reason:
            ml_info['stacking_disabled_reason'] = self._stacking_disabled_reason

        if self._stacking_meta_model is not None:
            try:
                bull_prob = self._predict_stacking_from_features(features, ml_info)
            except Exception as e:
                logger.warning(f"Stacking 预测失败: {e}")

        if bull_prob is None:
            has_direction = (
                self._direction_model is not None or self._lstm_meta is not None
                or self._tft_meta is not None or self._cross_asset_model is not None
            )
            if has_direction:
                try:
                    lgb_prob = self._predict_direction_lgb(features)
                    lstm_prob = self._predict_direction_lstm(features)
                    tft_prob = self._predict_direction_tft(features)
                    ca_prob = None
                    if self._can_use_cross_asset(features, ml_info):
                        ca_prob = self._predict_direction_cross_asset(features)
                    probs, weights = [], []
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
                except Exception as e:
                    logger.warning(f"方向预测计算异常: {e}")

        if bull_prob is not None:
            ml_info['bull_prob'] = round(bull_prob, 4)
        return bull_prob, ml_info

    # 远程 GPU 推理：发送特征行数（满足 TFT 96 / LSTM 48）
    FEATURE_ROWS_FOR_REMOTE = 96

    def _request_remote_direction(
        self, features: pd.DataFrame, sell_score: float, buy_score: float
    ) -> Tuple[Optional[float], Optional[Dict]]:
        """
        请求远程 GPU 推理 API 获取 bull_prob。失败返回 (None, None)。
        """
        if not self.gpu_inference_url or features is None or len(features) == 0:
            return None, None
        if len(features) < self.FEATURE_ROWS_FOR_REMOTE:
            return None, None
        try:
            import requests
        except ImportError:
            logger.warning("requests 未安装，无法使用远程 GPU 推理")
            return None, None

        # 取最后 FEATURE_ROWS_FOR_REMOTE 行
        tail = features.tail(self.FEATURE_ROWS_FOR_REMOTE)
        payload = {
            "sell_score": float(sell_score),
            "buy_score": float(buy_score),
            "features": json.loads(tail.to_json(orient="split")),
        }
        try:
            r = requests.post(
                self.gpu_inference_url.rstrip("/") + "/predict",
                json=payload,
                timeout=self.gpu_inference_timeout,
                headers={"Content-Type": "application/json"},
            )
            r.raise_for_status()
            data = r.json()
            if not data.get("success"):
                logger.warning("远程推理返回 success=false: %s", data.get("error", ""))
                return None, None
            bull = data.get("bull_prob")
            if bull is None:
                return None, None
            bull = float(bull)
            ml_info = {k: v for k, v in data.items() if k not in ("success", "bull_prob")}
            ml_info["bull_prob"] = round(bull, 4)
            ml_info["remote_inference"] = True
            return bull, ml_info
        except requests.exceptions.Timeout:
            logger.warning("远程 GPU 推理超时 (%ss)，回退本地", self.gpu_inference_timeout)
            return None, None
        except requests.exceptions.RequestException as e:
            logger.warning("远程 GPU 推理请求失败: %s，回退本地", e)
            return None, None
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("远程推理响应解析失败: %s，回退本地", e)
            return None, None

    def _predict_stacking_lstm(self, features: pd.DataFrame, cfg: Dict) -> Optional[float]:
        """Stacking 专用 LSTM 推理"""
        import torch
        import torch.nn as nn

        model_file = cfg.get('model_files', {}).get('lstm', '')
        model_path = os.path.join(self.model_dir, model_file)
        if not model_file or not os.path.exists(model_path):
            return None

        feat_names = cfg.get('feature_names_73', [])
        SEQ_LEN = 48
        if len(features) < SEQ_LEN:
            return None

        # 对齐特征
        X_df = pd.DataFrame(0.0, index=features.index, columns=feat_names)
        for col in feat_names:
            if col in features.columns:
                X_df[col] = features[col].values
        X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        vals = X_df.values.astype(np.float32)

        # 标准化
        feat_mean = np.array(cfg.get('feat_mean_73', [0.0] * len(feat_names)), dtype=np.float32)
        feat_std = np.array(cfg.get('feat_std_73', [1.0] * len(feat_names)), dtype=np.float32)
        vals = (vals - feat_mean) / np.maximum(feat_std, 1e-8)
        vals = np.clip(np.nan_to_num(vals, nan=0.0, posinf=3.0, neginf=-3.0), -5, 5)

        seq = vals[-SEQ_LEN:]
        X_tensor = torch.FloatTensor(seq).unsqueeze(0)

        if self._stacking_lstm is None:
            input_dim = len(feat_names)

            class LSTMAttention(nn.Module):
                def __init__(self, in_dim, hidden=128, layers=2, drop=0.3):
                    super().__init__()
                    self.lstm = nn.LSTM(in_dim, hidden, layers,
                                        batch_first=True, dropout=drop, bidirectional=True)
                    self.attn_fc = nn.Linear(hidden * 2, 1)
                    self.classifier = nn.Sequential(
                        nn.Linear(hidden * 2, 64), nn.GELU(), nn.Dropout(0.2),
                        nn.Linear(64, 1))

                def forward(self, x):
                    out, _ = self.lstm(x)
                    w = torch.softmax(self.attn_fc(out), dim=1)
                    ctx = (w * out).sum(dim=1)
                    return self.classifier(ctx).squeeze(-1)

            dev = self._inference_device
            self._stacking_lstm = LSTMAttention(input_dim).to(dev)
            state = torch.load(model_path, map_location=dev, weights_only=True)
            self._stacking_lstm.load_state_dict(state)
            self._stacking_lstm.eval()
            logger.info(f"Stacking LSTM 加载完成 (input_dim={input_dim}, device={dev})")

        with torch.no_grad():
            logit = self._stacking_lstm(X_tensor.to(self._inference_device))
            prob = torch.sigmoid(logit).item()
        return float(np.clip(prob, 0, 1))

    def _predict_stacking_tft(self, features: pd.DataFrame, cfg: Dict) -> Optional[float]:
        """Stacking 专用 TFT 推理"""
        import torch
        import torch.nn as nn

        model_file = cfg.get('model_files', {}).get('tft', '')
        model_path = os.path.join(self.model_dir, model_file)
        if not model_file or not os.path.exists(model_path):
            return None

        feat_names = cfg.get('feature_names_94', [])
        SEQ_LEN = 96
        if len(features) < SEQ_LEN:
            return None

        # 对齐特征 (94 维含跨资产)
        X_df = pd.DataFrame(0.0, index=features.index, columns=feat_names)
        for col in feat_names:
            if col in features.columns:
                X_df[col] = features[col].values
        X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        vals = X_df.values.astype(np.float32)

        # 标准化
        feat_mean = np.array(cfg.get('feat_mean_94', [0.0] * len(feat_names)), dtype=np.float32)
        feat_std = np.array(cfg.get('feat_std_94', [1.0] * len(feat_names)), dtype=np.float32)
        vals = (vals - feat_mean) / np.maximum(feat_std, 1e-8)
        vals = np.clip(np.nan_to_num(vals, nan=0.0, posinf=3.0, neginf=-3.0), -5, 5)

        seq = vals[-SEQ_LEN:]
        X_tensor = torch.FloatTensor(seq).unsqueeze(0)

        if self._stacking_tft is None:
            input_dim = len(feat_names)

            class EfficientTFT(nn.Module):
                def __init__(self, in_dim, d_model=64, n_heads=4, d_ff=128, n_layers=2, drop=0.15):
                    super().__init__()
                    self.input_proj = nn.Sequential(
                        nn.Linear(in_dim, d_model), nn.LayerNorm(d_model),
                        nn.GELU(), nn.Dropout(drop))
                    self.lstm = nn.LSTM(d_model, d_model, n_layers,
                                        batch_first=True, dropout=drop if n_layers > 1 else 0)
                    self.lstm_norm = nn.LayerNorm(d_model)
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                        dropout=drop, activation='gelu', batch_first=True)
                    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
                    self.transformer_norm = nn.LayerNorm(d_model)
                    self.attn_pool = nn.Linear(d_model, 1)
                    self.classifier = nn.Sequential(
                        nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(drop),
                        nn.Linear(d_ff, 1))

                def forward(self, x):
                    h = self.input_proj(x)
                    lstm_out, _ = self.lstm(h)
                    h = self.lstm_norm(lstm_out + h)
                    h = self.transformer(h)
                    h = self.transformer_norm(h)
                    w = torch.softmax(self.attn_pool(h), dim=1)
                    ctx = (w * h).sum(dim=1)
                    return self.classifier(ctx).squeeze(-1)

            dev = self._inference_device
            self._stacking_tft = EfficientTFT(input_dim).to(dev)
            state = torch.load(model_path, map_location=dev, weights_only=True)
            self._stacking_tft.load_state_dict(state)
            self._stacking_tft.eval()
            logger.info(f"Stacking TFT 加载完成 (input_dim={input_dim}, device={dev})")

        with torch.no_grad():
            logit = self._stacking_tft(X_tensor.to(self._inference_device))
            prob = torch.sigmoid(logit).item()
        return float(np.clip(prob, 0, 1))

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
        if self._stacking_meta_model is None and self._stacking_disabled_reason:
            ml_info['stacking_disabled_reason'] = self._stacking_disabled_reason
        enhanced_buy = buy_score
        enhanced_sell = sell_score

        # 0. 方向预测: 优先远程 GPU API → 否则本地 Stacking(优先) → 多模型加权(fallback)
        bull_prob = None
        has_direction = (
            self._direction_model is not None or self._lstm_meta is not None
            or self._tft_meta is not None or self._cross_asset_model is not None
        )

        # 统一计算一次方向特征，避免远程失败后本地 fallback 重复计算
        direction_features = None
        if self.gpu_inference_url or self._stacking_meta_model is not None or has_direction:
            direction_features = self._compute_direction_features(df)

        if self.gpu_inference_url and direction_features is not None and len(direction_features) >= 1:
            bull_prob, remote_ml = self._request_remote_direction(
                direction_features, sell_score, buy_score
            )
            if bull_prob is not None and remote_ml:
                ml_info.update(remote_ml)

        if bull_prob is None and self._stacking_meta_model is not None and direction_features is not None:
            # Stacking ensemble — 4 基模型 → 元学习器
            try:
                bull_prob = self._predict_stacking_from_features(direction_features, ml_info)
            except Exception as e:
                logger.warning(f"Stacking 预测失败，退回加权集成: {e}")
                bull_prob = None

        if bull_prob is None and has_direction and direction_features is not None:
            # Fallback: 多模型加权集成 (LGB + LSTM + TFT + 跨资产)
            if len(direction_features) > 0:
                try:
                    lgb_prob = self._predict_direction_lgb(direction_features)
                    lstm_prob = self._predict_direction_lstm(direction_features)
                    tft_prob = self._predict_direction_tft(direction_features)
                    ca_prob = None
                    if self._can_use_cross_asset(direction_features, ml_info):
                        ca_prob = self._predict_direction_cross_asset(direction_features)

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
                except Exception as e:
                    logger.warning(f"方向预测计算异常: {e}")

        if bull_prob is not None:
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
