"""
实时信号生成器
桥接现有策略代码（optimize_six_book.py 中的信号计算）与实盘引擎

工作流程:
1. 获取最新K线数据 (Binance API)
2. 计算六维指标 (复用现有代码)
3. 融合评分 (calc_fusion_score_six)
4. 输出交易信号 (sell_score, buy_score, 推荐动作)
"""

import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binance_fetcher import (
    fetch_binance_klines,
    fetch_mark_price_klines,
    fetch_funding_rate_history,
    fetch_open_interest_history,
    merge_perp_data_into_klines,
)
from indicators import add_all_indicators
from strategy_enhanced import analyze_signals_enhanced, get_signal_at, DEFAULT_SIG
from strategy_futures_v4 import _calc_top_score, _calc_bottom_score
from ma_indicators import add_moving_averages, compute_ma_signals
from candlestick_patterns import compute_candlestick_scores
from bollinger_strategy import compute_bollinger_scores
from volume_price_strategy import compute_volume_price_scores
from kdj_strategy import compute_kdj_scores
from signal_core import calc_fusion_score_six, compute_signals_six
from multi_tf_consensus import compute_weighted_consensus, fuse_tf_scores, neural_fuse_tf_scores


class SignalResult:
    """信号计算结果"""

    def __init__(self):
        self.timestamp: str = ""
        self.price: float = 0
        self.sell_score: float = 0
        self.buy_score: float = 0
        self.components: Dict = {}
        self.action: str = "HOLD"  # HOLD / OPEN_LONG / OPEN_SHORT / CLOSE_LONG / CLOSE_SHORT
        self.reason: str = ""
        self.conflict: bool = False
        self.bar_index: int = 0
        self.regime_label: str = "neutral"
        self.atr_pct: float = 0.0  # ATR 占价格比, 用于波动率过滤

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "price": self.price,
            "sell_score": self.sell_score,
            "buy_score": self.buy_score,
            "components": self.components,
            "regime_label": self.regime_label,
            "action": self.action,
            "reason": self.reason,
            "conflict": self.conflict,
        }

    def __repr__(self):
        return (f"Signal({self.timestamp} price={self.price:.2f} "
                f"SS={self.sell_score:.1f} BS={self.buy_score:.1f} "
                f"action={self.action})")


class LiveSignalGenerator:
    """
    实时信号生成器
    复用 optimize_six_book.py 中的信号计算逻辑
    """

    # 各时间框架推荐最大回看天数(避免数据量过大导致计算超时)
    _MAX_LOOKBACK_DAYS = {
        '1m': 2, '3m': 3, '5m': 5, '10m': 7, '15m': 15,
        '30m': 25, '1h': 60, '2h': 60, '4h': 90,
        '8h': 120, '12h': 120, '1d': 180,
    }

    def __init__(self, config, logger=None):
        """
        config: StrategyConfig 实例
        """
        self.config = config
        self.logger = logger
        self.symbol = config.symbol
        self.timeframe = config.timeframe
        # 根据 timeframe 自动限制 lookback_days，防止小周期数据过多
        max_days = self._MAX_LOOKBACK_DAYS.get(
            config.timeframe, config.lookback_days
        )
        self.lookback_days = min(config.lookback_days, max_days)

        # 缓存数据
        self._df: Optional[pd.DataFrame] = None
        self._signals: Optional[Dict] = None
        self._data_all: Dict = {}
        self._last_refresh: float = 0
        self._refresh_interval = self._calc_refresh_interval()

        # 信号历史
        self._signal_history: list = []

        # ML 增强器 (懒加载)
        self._ml_enhancer = None
        self._ml_status_logged = False

    def _calc_refresh_interval(self) -> float:
        """根据时间框架计算数据刷新间隔(秒)"""
        tf = self.timeframe
        minutes_map = {
            '1m': 1, '3m': 3, '5m': 5, '10m': 10, '15m': 15,
            '30m': 30, '1h': 60, '2h': 120, '3h': 180, '4h': 240,
            '6h': 360, '8h': 480, '12h': 720, '16h': 960, '24h': 1440,
            '1d': 1440,
        }
        minutes = minutes_map.get(tf, 60)
        # 在K线结束前10秒刷新数据，确保拿到最新收盘数据
        return max(30, minutes * 60 - 10)

    def _build_runtime_config_dict(self) -> Dict:
        """将 StrategyConfig 转为 dict，供 regime/fusion 共用。"""
        cfg = dict(getattr(self.config, "__dict__", {}))
        cfg.setdefault('use_regime_aware', True)
        return cfg

    def _infer_regime_label(self, df: pd.DataFrame, idx: int) -> str:
        """与回测同源的 regime 推断（只用 idx-1 及之前数据）。"""
        if idx < 2:
            return 'neutral'
        try:
            # 延迟导入，避免启动阶段不必要的重依赖
            from optimize_six_book import _compute_regime_controls
            ctl = _compute_regime_controls(df, idx - 1, self._build_runtime_config_dict())
            return str((ctl or {}).get('regime_label', 'neutral') or 'neutral')
        except Exception:
            return 'neutral'

    def _build_fusion_config(self, regime_label: str = 'neutral') -> Dict:
        """构造与回测一致的融合参数字典。"""
        cfg = {
            'fusion_mode': self.config.fusion_mode,
            'veto_threshold': self.config.veto_threshold,
            'kdj_bonus': self.config.kdj_bonus,
            'kdj_weight': getattr(self.config, 'kdj_weight', 0.15),
            'div_weight': getattr(self.config, 'div_weight', 0.55),
            'kdj_strong_mult': getattr(self.config, 'kdj_strong_mult', 1.25),
            'kdj_normal_mult': getattr(self.config, 'kdj_normal_mult', 1.12),
            'kdj_reverse_mult': getattr(self.config, 'kdj_reverse_mult', 0.70),
            'kdj_gate_threshold': getattr(self.config, 'kdj_gate_threshold', 10),
            'veto_dampen': getattr(self.config, 'veto_dampen', 0.30),
            'bb_bonus': getattr(self.config, 'bb_bonus', 0.10),
            'vp_bonus': getattr(self.config, 'vp_bonus', 0.08),
            'cs_bonus': getattr(self.config, 'cs_bonus', 0.06),
            # 与回测保持一致：注入当前 regime，触发动态融合分支
            '_regime_label': regime_label,
        }
        if getattr(self.config, 'use_regime_adaptive_fusion', False):
            cfg['fusion_mode'] = 'regime_adaptive'
        return cfg

    # ============================================================
    # 数据获取
    # ============================================================
    def refresh_data(self, force: bool = False) -> bool:
        """
        刷新K线数据和指标计算
        返回: True 如果数据已更新
        """
        now = time.time()
        if not force and (now - self._last_refresh) < self._refresh_interval:
            return False

        try:
            if self.logger:
                self.logger.info(f"刷新 {self.symbol} {self.timeframe} 数据...")

            # 获取主周期数据
            df = fetch_binance_klines(
                self.symbol, interval=self.timeframe,
                days=self.lookback_days
            )
            if df is None or len(df) < 50:
                if self.logger:
                    self.logger.error(
                        f"数据不足: {len(df) if df is not None else 0} bars"
                    )
                return False

            # ── V9: 获取衍生品数据 (Mark Price / Funding Rate / OI) ──
            # 非阻塞: 任何衍生品数据获取失败不影响主信号
            _perp_days = min(self.lookback_days, 90)
            mark_df = None
            funding_df = None
            oi_df = None
            try:
                mark_df = fetch_mark_price_klines(
                    self.symbol, interval=self.timeframe, days=_perp_days
                )
            except Exception as _e:
                if self.logger:
                    self.logger.warning(f"Mark Price 获取失败 (非致命): {_e}")
            try:
                funding_df = fetch_funding_rate_history(
                    self.symbol, days=_perp_days
                )
            except Exception as _e:
                if self.logger:
                    self.logger.warning(f"Funding Rate 获取失败 (非致命): {_e}")
            try:
                oi_df = fetch_open_interest_history(
                    self.symbol, interval=self.timeframe, days=_perp_days
                )
            except Exception as _e:
                if self.logger:
                    self.logger.warning(f"OI 获取失败 (非致命): {_e}")

            # 合并衍生品数据到主 DataFrame
            if any(x is not None for x in (mark_df, funding_df, oi_df)):
                df = merge_perp_data_into_klines(df, mark_df, funding_df, oi_df)
                _perp_cols = [c for c in ('mark_high', 'mark_low', 'funding_rate',
                                          'open_interest', 'taker_buy_quote')
                              if c in df.columns and df[c].notna().any()]
                if self.logger:
                    self.logger.info(f"衍生品数据已合并: {_perp_cols}")

            # 计算指标
            df = add_all_indicators(df)
            add_moving_averages(df, timeframe=self.timeframe)

            self._df = df
            self._data_all[self.timeframe] = df

            # 获取8h辅助数据 (如果主周期不是 8h/12h/16h/24h)
            if self.timeframe not in ('8h', '12h', '16h', '24h'):
                try:
                    df_8h = fetch_binance_klines(
                        self.symbol, interval='8h',
                        days=self.lookback_days
                    )
                    if df_8h is not None and len(df_8h) > 20:
                        df_8h = add_all_indicators(df_8h)
                        self._data_all['8h'] = df_8h
                except Exception:
                    pass  # 8h数据失败不影响主信号

            # 计算六维信号 (限制最多1500根K线，避免小周期超时)
            self._signals = compute_signals_six(
                df, self.timeframe, self._data_all,
                max_bars=1500
            )

            self._last_refresh = now

            if self.logger:
                self.logger.info(
                    f"数据刷新完成: {len(df)} bars, "
                    f"最新时间: {df.index[-1]}"
                )

            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"数据刷新失败: {e}\n{traceback.format_exc()}")
            return False

    # ============================================================
    # 信号计算
    # ============================================================
    def compute_latest_signal(self) -> Optional[SignalResult]:
        """
        计算最新K线的交易信号
        返回: SignalResult 或 None
        """
        if self._df is None or self._signals is None:
            self.refresh_data(force=True)
            if self._df is None:
                return None

        df = self._df
        signals = self._signals
        idx = len(df) - 1  # 最新一根已收盘K线

        if idx < 30:
            return None

        try:
            dt = df.index[idx]
            regime_label = self._infer_regime_label(df, idx)
            config_dict = self._build_fusion_config(regime_label)
            sell_score, buy_score = calc_fusion_score_six(
                signals, df, idx, dt, config_dict
            )
            # 与回测 P9 逻辑对齐（仅当未启用 P18 时）
            if (not getattr(self.config, 'use_regime_adaptive_fusion', False)
                    and getattr(self.config, 'use_regime_adaptive_reweight', False)
                    and regime_label == 'neutral'):
                sell_score *= float(getattr(self.config, 'regime_neutral_ss_dampen', 0.85))
                buy_score *= float(getattr(self.config, 'regime_neutral_bs_boost', 1.10))

            # ── v11: Soft Anti-Squeeze (微结构连续降权) ──
            _antisq_info = {}
            if getattr(self.config, 'use_soft_antisqueeze', False):
                sell_score, buy_score, _antisq_info = self._apply_soft_antisqueeze(
                    df, idx, sell_score, buy_score
                )

            # ── ML 增强 (第七维度, 可选) ──
            _ml_info = {}
            _ml_enabled = bool(getattr(self.config, 'use_ml_enhancement', False))
            if _ml_enabled:
                try:
                    if self._ml_enhancer is None:
                        from ml_live_integration import MLSignalEnhancer
                        gpu_url = getattr(self.config, "ml_gpu_inference_url", "") or ""
                        self._ml_enhancer = MLSignalEnhancer(
                            gpu_inference_url=gpu_url,
                            stacking_timeframe=self.timeframe,
                        )
                        if self.logger:
                            self.logger.info(
                                "[ML CONFIG] enabled=True shadow=%s tf=%s gpu_url=%s",
                                getattr(self.config, 'ml_enhancement_shadow_mode', True),
                                self.timeframe,
                                bool(gpu_url),
                            )
                    _ml_ss, _ml_bs, _ml_info = self._ml_enhancer.enhance_signal(
                        sell_score, buy_score, df
                    )
                    _ml_shadow = getattr(self.config, 'ml_enhancement_shadow_mode', True)
                    _ml_info['shadow_mode'] = _ml_shadow
                    if self.logger:
                        # v5: 记录完整 ML 决策信息
                        _bull = _ml_info.get('bull_prob', '-')
                        _lgb = _ml_info.get('lgb_bull_prob', '-')
                        _lstm = _ml_info.get('lstm_bull_prob', '-')
                        _tft = _ml_info.get('tft_bull_prob', '-')
                        _ca = _ml_info.get('ca_bull_prob', '-')
                        _dir_act = _ml_info.get('direction_action', '-')
                        _regime = _ml_info.get('regime', '?')
                        _conf = _ml_info.get('trade_confidence', 0)
                        _mode = 'shadow' if _ml_shadow else 'LIVE'
                        _kelly = _ml_info.get('kelly_fraction', '-')
                        _pos_scale = _ml_info.get('position_scale', '-')
                        _q_act = _ml_info.get('quantile_action', '-')
                        _ver = _ml_info.get('ml_version', 'v?')
                        _stk_dis = _ml_info.get('stacking_disabled_reason', '-')
                        _stk_skip = _ml_info.get('stacking_skipped_reason', '-')
                        _ca_skip = _ml_info.get('ca_skipped_reason', '-')
                        _cov73 = _ml_info.get('stacking_feature_coverage_73', '-')
                        _cov94 = _ml_info.get('stacking_feature_coverage_94', '-')
                        _ca_cov = _ml_info.get('ca_feature_coverage', '-')
                        _remote = _ml_info.get('remote_inference', False)
                        self.logger.info(
                            f"[ML {_mode} {_ver}] bull_prob={_bull} "
                            f"(LGB={_lgb}, LSTM={_lstm}, TFT={_tft}, CA={_ca}) "
                            f"dir={_dir_act} regime={_regime} "
                            f"conf={_conf:.3f} "
                            f"kelly={_kelly} pos_scale={_pos_scale} "
                            f"remote={_remote} "
                            f"stk_dis={_stk_dis} stk_skip={_stk_skip} "
                            f"stk_cov=({_cov73},{_cov94}) ca_cov={_ca_cov} ca_skip={_ca_skip} "
                            f"q_act={_q_act} "
                            f"SS {sell_score:.1f}→{_ml_ss:.1f} "
                            f"BS {buy_score:.1f}→{_ml_bs:.1f}"
                        )
                    if not _ml_shadow:
                        sell_score, buy_score = _ml_ss, _ml_bs
                except Exception as _ml_err:
                    if self.logger:
                        self.logger.warning(f"ML 增强失败: {_ml_err}")
                    _ml_info = {'ml_error': str(_ml_err)[:120]}
            elif self.logger and not self._ml_status_logged:
                self.logger.info("[ML CONFIG] enabled=False (use_ml_enhancement=False)")
                self._ml_status_logged = True

            # 提取各维度分数
            components = self._extract_components(signals, idx, dt)
            if _antisq_info:
                components['antisq_long_penalty'] = _antisq_info.get('long_penalty', 0)
                components['antisq_short_penalty'] = _antisq_info.get('short_penalty', 0)
            if _ml_info:
                components['ml_bull_prob'] = _ml_info.get('bull_prob', 0.5)
                components['ml_lgb_prob'] = _ml_info.get('lgb_bull_prob', '-')
                components['ml_lstm_prob'] = _ml_info.get('lstm_bull_prob', '-')
                components['ml_tft_prob'] = _ml_info.get('tft_bull_prob', '-')
                components['ml_ca_prob'] = _ml_info.get('ca_bull_prob', '-')
                components['ml_direction'] = _ml_info.get('direction_action', 'neutral')
                components['ml_regime'] = _ml_info.get('regime', '-')
                components['ml_confidence'] = _ml_info.get('trade_confidence', 0)
                components['ml_shadow'] = _ml_info.get('shadow_mode', True)
                components['ml_version'] = _ml_info.get('ml_version', '-')
                # v5: Kelly 仓位 + 动态止损
                components['ml_kelly_fraction'] = _ml_info.get('kelly_fraction', '-')
                components['ml_position_scale'] = _ml_info.get('position_scale', '-')
                components['ml_dynamic_sl_long'] = _ml_info.get('dynamic_sl_long', '-')
                components['ml_dynamic_sl_short'] = _ml_info.get('dynamic_sl_short', '-')
                components['ml_quantile_action'] = _ml_info.get('quantile_action', '-')
                components['ml_remote_inference'] = _ml_info.get('remote_inference', False)
                components['ml_stacking_disabled_reason'] = _ml_info.get('stacking_disabled_reason', '')
                components['ml_stacking_skipped_reason'] = _ml_info.get('stacking_skipped_reason', '')
                components['ml_ca_skipped_reason'] = _ml_info.get('ca_skipped_reason', '')
                components['ml_stacking_cov_73'] = _ml_info.get('stacking_feature_coverage_73', '')
                components['ml_stacking_cov_94'] = _ml_info.get('stacking_feature_coverage_94', '')
                components['ml_ca_cov'] = _ml_info.get('ca_feature_coverage', '')
                components['ml_error'] = _ml_info.get('ml_error', '')

            # 构建结果
            result = SignalResult()
            result.timestamp = str(dt)
            result.price = float(df['close'].iloc[idx])
            result.sell_score = float(sell_score)
            result.buy_score = float(buy_score)
            result.components = components
            result.regime_label = regime_label
            result.bar_index = idx
            result.conflict = (sell_score > 15 and buy_score > 15 and
                              abs(sell_score - buy_score) < 10)

            # 计算 ATR 百分比 (14 bar ATR / 当前价格)
            atr_period = 14
            if idx >= atr_period:
                high = df['high'].iloc[idx - atr_period + 1:idx + 1]
                low = df['low'].iloc[idx - atr_period + 1:idx + 1]
                prev_close = df['close'].iloc[idx - atr_period:idx]
                tr = pd.concat([
                    high - low,
                    (high - prev_close.values).abs(),
                    (low - prev_close.values).abs(),
                ], axis=1).max(axis=1)
                atr_val = tr.mean()
                result.atr_pct = float(atr_val / result.price) if result.price > 0 else 0.0
            else:
                result.atr_pct = 0.0

            # 记录历史
            self._signal_history.append(result.to_dict())
            if len(self._signal_history) > 500:
                self._signal_history = self._signal_history[-500:]

            return result

        except Exception as e:
            if self.logger:
                self.logger.error(f"信号计算失败: {e}\n{traceback.format_exc()}")
            return None

    def _extract_components(self, signals: dict, idx: int, dt) -> dict:
        """提取六维信号分量"""
        components = {}

        # 背离
        sig_main = get_signal_at(signals.get('div', {}), dt) or dict(DEFAULT_SIG)
        trend = {'is_downtrend': False, 'is_uptrend': False,
                 'ma_bearish': False, 'ma_bullish': False,
                 'ma_slope_down': False, 'ma_slope_up': False}
        if idx >= 30:
            df = self._df
            c5 = df['close'].iloc[max(0, idx - 5):idx].mean()
            c20 = df['close'].iloc[max(0, idx - 20):idx].mean()
            if c5 < c20 * 0.99:
                trend['is_downtrend'] = True
                trend['ma_bearish'] = True
            elif c5 > c20 * 1.01:
                trend['is_uptrend'] = True
                trend['ma_bullish'] = True

        div_sell, _ = _calc_top_score(sig_main, trend)
        div_buy = _calc_bottom_score(sig_main, trend)
        components['div_s'] = div_sell
        components['div_b'] = div_buy

        # 均线
        def _safe_get(series, idx):
            try:
                return float(series.iloc[idx]) if idx < len(series) else 0
            except (IndexError, TypeError, ValueError):
                return 0

        components['ma_s'] = _safe_get(signals['ma']['sell_score'], idx)
        components['ma_b'] = _safe_get(signals['ma']['buy_score'], idx)

        # 其他4维
        components['cs_s'] = _safe_get(signals['cs_sell'], idx)
        components['cs_b'] = _safe_get(signals['cs_buy'], idx)
        components['bb_s'] = _safe_get(signals['bb_sell'], idx)
        components['bb_b'] = _safe_get(signals['bb_buy'], idx)
        components['vp_s'] = _safe_get(signals['vp_sell'], idx)
        components['vp_b'] = _safe_get(signals['vp_buy'], idx)
        components['kdj_s'] = _safe_get(signals['kdj_sell'], idx)
        components['kdj_b'] = _safe_get(signals['kdj_buy'], idx)

        return components

    # ============================================================
    # v11: Soft Anti-Squeeze
    # ============================================================
    def _apply_soft_antisqueeze(self, df, idx, sell_score, buy_score):
        """
        计算微结构拥挤度，对 SS/BS 施加连续惩罚。
        与回测 optimize_six_book.py 中的 soft anti-squeeze 逻辑一致，
        但这里直接作用于分数（因为实盘不经过 margin_mult）。

        多头拥挤 → 做空有风险 → 降低 sell_score
        空头拥挤 → 做多有风险 → 降低 buy_score
        """
        import math
        info = {'long_penalty': 0.0, 'short_penalty': 0.0,
                'funding_z': 0.0, 'oi_z': 0.0, 'taker_imb': 0.0}

        try:
            # 获取微结构数据 — 复用 _build_microstructure_features 逻辑
            lookback = max(8, int(getattr(self.config, 'micro_lookback_bars', 48)))
            minp = max(5, lookback // 3)

            close = pd.to_numeric(df['close'], errors='coerce')
            volume = pd.to_numeric(df.get('volume', pd.Series(0.0, index=df.index)), errors='coerce')
            quote_volume = df.get('quote_volume')
            if quote_volume is None:
                quote_volume = close * volume
            quote_volume = pd.to_numeric(quote_volume, errors='coerce').replace(0, np.nan)

            taker_buy_quote = df.get('taker_buy_quote')
            if taker_buy_quote is None:
                taker_buy_quote = quote_volume * 0.5
            taker_buy_quote = pd.to_numeric(taker_buy_quote, errors='coerce')

            taker_buy_ratio = (taker_buy_quote / quote_volume).clip(0.0, 1.0).fillna(0.5)
            taker_imb = float(((taker_buy_ratio - 0.5) * 2.0).clip(-1.0, 1.0).iloc[idx])

            # OI z-score
            oi_series = None
            for col in ('open_interest', 'oi'):
                if col in df.columns:
                    oi_series = pd.to_numeric(df[col], errors='coerce')
                    break
            if oi_series is None:
                oi_series = quote_volume
            oi_mean = oi_series.rolling(lookback, min_periods=minp).mean()
            oi_std = oi_series.rolling(lookback, min_periods=minp).std(ddof=0).replace(0, np.nan)
            oi_z = float(((oi_series - oi_mean) / oi_std).clip(-5.0, 5.0).fillna(0.0).iloc[idx])

            # Funding z-score
            if 'funding_rate' in df.columns:
                funding_rate = pd.to_numeric(df['funding_rate'], errors='coerce').fillna(0.0)
            else:
                funding_rate = pd.Series(0.0, index=df.index)
            fr_mean = funding_rate.rolling(lookback, min_periods=minp).mean()
            fr_std = funding_rate.rolling(lookback, min_periods=minp).std(ddof=0).replace(0, np.nan)
            funding_z = float(((funding_rate - fr_mean) / fr_std).clip(-5.0, 5.0).fillna(0.0).iloc[idx])

            info['funding_z'] = funding_z
            info['oi_z'] = oi_z
            info['taker_imb'] = taker_imb

            # Sigmoid penalty 计算
            _w_fz = float(getattr(self.config, 'soft_antisqueeze_w_fz', 0.5))
            _w_oi = float(getattr(self.config, 'soft_antisqueeze_w_oi', 0.3))
            _w_imb = float(getattr(self.config, 'soft_antisqueeze_w_imb', 0.2))
            _mid = float(getattr(self.config, 'soft_antisqueeze_midpoint', 1.5))
            _steep = float(getattr(self.config, 'soft_antisqueeze_steepness', 2.0))
            _max_disc = float(getattr(self.config, 'soft_antisqueeze_max_discount', 0.50))

            # 多头拥挤评分
            _long_crowd = _w_fz * max(0, funding_z) + _w_oi * max(0, oi_z) + _w_imb * max(0, taker_imb / 0.12)
            _long_penalty = 1.0 / (1.0 + math.exp(-_steep * (_long_crowd - _mid)))

            # 空头拥挤评分
            _short_crowd = _w_fz * max(0, -funding_z) + _w_oi * max(0, oi_z) + _w_imb * max(0, -taker_imb / 0.12)
            _short_penalty = 1.0 / (1.0 + math.exp(-_steep * (_short_crowd - _mid)))

            info['long_penalty'] = _long_penalty
            info['short_penalty'] = _short_penalty

            # 多头拥挤 → 做空有风险 → 降低 sell_score
            if _long_penalty > 0.05:
                sell_score *= (1.0 - _long_penalty * _max_disc)
            # 空头拥挤 → 做多有风险 → 降低 buy_score
            if _short_penalty > 0.05:
                buy_score *= (1.0 - _short_penalty * _max_disc)

            if self.logger and (_long_penalty > 0.1 or _short_penalty > 0.1):
                self.logger.info(
                    f"[AntiSqueeze] fz={funding_z:.2f} oi_z={oi_z:.2f} imb={taker_imb:.3f} "
                    f"long_pen={_long_penalty:.3f} short_pen={_short_penalty:.3f}"
                )

        except Exception as e:
            if self.logger:
                self.logger.warning(f"[AntiSqueeze] 计算失败(非致命): {e}")

        return sell_score, buy_score, info

    # ============================================================
    # 交易决策
    # ============================================================
    def evaluate_action(self, signal: SignalResult,
                        has_long: bool = False,
                        has_short: bool = False,
                        long_pnl_ratio: float = 0,
                        short_pnl_ratio: float = 0,
                        long_bars: int = 0,
                        short_bars: int = 0,
                        long_max_pnl: float = 0,
                        short_max_pnl: float = 0,
                        short_cooldown: int = 0,
                        long_cooldown: int = 0) -> SignalResult:
        """
        根据信号和当前持仓状态，决定交易动作
        返回: 更新了 action 和 reason 的 SignalResult
        """
        ss = signal.sell_score
        bs = signal.buy_score
        cfg = self.config

        reasons = []

        # --- 平仓检查 (优先级最高) ---

        # 空仓平仓
        if has_short:
            # 止盈
            if short_pnl_ratio >= cfg.short_tp:
                signal.action = "CLOSE_SHORT"
                signal.reason = f"止盈 pnl={short_pnl_ratio:.1%} >= tp={cfg.short_tp:.1%}"
                return signal
            # 追踪止盈
            if (cfg.short_trail > 0 and short_max_pnl >= cfg.short_trail and
                    short_pnl_ratio < short_max_pnl * cfg.trail_pullback):
                signal.action = "CLOSE_SHORT"
                signal.reason = (f"追踪止盈 max={short_max_pnl:.1%} "
                                f"cur={short_pnl_ratio:.1%}")
                return signal
            # 反向信号 (需满足最小持仓时间 + 净差额要求, 与回测对齐)
            _close_margin = getattr(cfg, 'close_signal_margin', 20)
            if (bs >= cfg.close_short_bs and bs > ss * 0.7
                    and (bs - ss) >= _close_margin
                    and short_bars >= cfg.reverse_min_hold_short):
                signal.action = "CLOSE_SHORT"
                signal.reason = (f"反向信号 BS={bs:.1f} >= {cfg.close_short_bs} "
                                 f"margin={bs - ss:.1f} bars={short_bars}")
                return signal
            # 止损
            if short_pnl_ratio < cfg.short_sl:
                signal.action = "CLOSE_SHORT"
                signal.reason = f"止损 pnl={short_pnl_ratio:.1%} < sl={cfg.short_sl:.1%}"
                return signal
            # 超时
            if cfg.short_max_hold > 0 and short_bars >= cfg.short_max_hold:
                signal.action = "CLOSE_SHORT"
                signal.reason = f"超时 bars={short_bars} >= max={cfg.short_max_hold}"
                return signal

        # 多仓平仓
        if has_long:
            if long_pnl_ratio >= cfg.long_tp:
                signal.action = "CLOSE_LONG"
                signal.reason = f"止盈 pnl={long_pnl_ratio:.1%} >= tp={cfg.long_tp:.1%}"
                return signal
            if (cfg.long_trail > 0 and long_max_pnl >= cfg.long_trail and
                    long_pnl_ratio < long_max_pnl * cfg.trail_pullback):
                signal.action = "CLOSE_LONG"
                signal.reason = (f"追踪止盈 max={long_max_pnl:.1%} "
                                f"cur={long_pnl_ratio:.1%}")
                return signal
            # 反向信号 (需满足最小持仓时间 + 净差额要求, 与回测对齐)
            _close_margin = getattr(cfg, 'close_signal_margin', 20)
            if (ss >= cfg.close_long_ss and ss > bs * 0.7
                    and (ss - bs) >= _close_margin
                    and long_bars >= cfg.reverse_min_hold_long):
                signal.action = "CLOSE_LONG"
                signal.reason = (f"反向信号 SS={ss:.1f} >= {cfg.close_long_ss} "
                                 f"margin={ss - bs:.1f} bars={long_bars}")
                return signal
            if long_pnl_ratio < cfg.long_sl:
                signal.action = "CLOSE_LONG"
                signal.reason = f"止损 pnl={long_pnl_ratio:.1%} < sl={cfg.long_sl:.1%}"
                return signal
            if cfg.long_max_hold > 0 and long_bars >= cfg.long_max_hold:
                signal.action = "CLOSE_LONG"
                signal.reason = f"超时 bars={long_bars} >= max={cfg.long_max_hold}"
                return signal

        # --- 开仓检查 ---

        # 波动率过滤: ATR 不足以覆盖开平费用时拒绝开仓
        min_atr_pct = getattr(cfg, 'min_atr_pct_to_open', 0.003)
        if hasattr(signal, 'atr_pct') and signal.atr_pct > 0 and signal.atr_pct < min_atr_pct:
            signal.action = "HOLD"
            signal.reason = (f"波动率不足 ATR={signal.atr_pct:.4f} "
                             f"< {min_atr_pct} SS={ss:.1f} BS={bs:.1f}")
            return signal

        # 开仓比率阈值 (可配置, 默认 1.3; 原硬编码 1.5 过于严格)
        open_dominance = getattr(cfg, 'open_dominance_ratio', 1.3)

        # 开空
        if (not has_short and short_cooldown <= 0 and
                ss >= cfg.short_threshold and ss > bs * open_dominance):
            signal.action = "OPEN_SHORT"
            signal.reason = f"开空 SS={ss:.1f} >= {cfg.short_threshold}"
            return signal

        # 开多
        if (not has_long and long_cooldown <= 0 and
                bs >= cfg.long_threshold and bs > ss * open_dominance):
            signal.action = "OPEN_LONG"
            signal.reason = f"开多 BS={bs:.1f} >= {cfg.long_threshold}"
            return signal

        signal.action = "HOLD"
        signal.reason = f"持续观察 SS={ss:.1f} BS={bs:.1f}"
        return signal

    # ============================================================
    # 部分止盈检查
    # ============================================================
    def check_partial_tp(self, side: str, pnl_ratio: float,
                         partial_done_1: bool = False,
                         partial_done_2: bool = False) -> Optional[dict]:
        """
        检查是否触发部分止盈
        返回: {"level": 1 or 2, "close_pct": 0.30} 或 None
        """
        cfg = self.config

        if not cfg.use_partial_tp:
            return None

        if side == "SHORT":
            if not partial_done_1 and pnl_ratio >= cfg.partial_tp_1:
                return {"level": 1, "close_pct": cfg.partial_tp_1_pct}
            if (cfg.use_partial_tp_2 and not partial_done_2 and
                    partial_done_1 and pnl_ratio >= cfg.partial_tp_2):
                return {"level": 2, "close_pct": cfg.partial_tp_2_pct}

        elif side == "LONG":
            if not partial_done_1 and pnl_ratio >= cfg.partial_tp_1:
                return {"level": 1, "close_pct": cfg.partial_tp_1_pct}
            if (cfg.use_partial_tp_2 and not partial_done_2 and
                    partial_done_1 and pnl_ratio >= cfg.partial_tp_2):
                return {"level": 2, "close_pct": cfg.partial_tp_2_pct}

        return None

    # ============================================================
    # 实时价格信号 (不等待K线收盘)
    # ============================================================
    def compute_realtime_risk_signal(self, current_price: float) -> Optional[dict]:
        """
        基于当前实时价格的紧急风险信号
        用于K线未收盘时的实时监控
        """
        if self._df is None:
            return None

        df = self._df
        last_close = float(df['close'].iloc[-1])

        # 快速计算价格变动
        price_change = (current_price - last_close) / last_close

        alerts = []

        # 极端价格变动 (超过3%短时波动)
        if abs(price_change) > 0.03:
            alerts.append({
                "type": "EXTREME_MOVE",
                "message": f"价格剧烈波动 {price_change:+.2%}",
                "severity": "HIGH"
            })

        # 超过5%波动
        if abs(price_change) > 0.05:
            alerts.append({
                "type": "FLASH_CRASH",
                "message": f"闪崩/暴涨警告 {price_change:+.2%}",
                "severity": "CRITICAL"
            })

        return alerts if alerts else None

    # ============================================================
    # 工具方法
    # ============================================================
    def get_signal_history(self, last_n: int = 50) -> list:
        """获取最近的信号历史"""
        return self._signal_history[-last_n:]

    def get_current_data_info(self) -> dict:
        """获取当前数据状态"""
        if self._df is None:
            return {"status": "no_data"}

        df = self._df
        return {
            "status": "ok",
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "total_bars": len(df),
            "latest_bar": str(df.index[-1]),
            "latest_close": float(df['close'].iloc[-1]),
            "last_refresh": datetime.fromtimestamp(
                self._last_refresh).isoformat() if self._last_refresh else None,
            "refresh_interval_sec": self._refresh_interval,
        }

    def needs_refresh(self) -> bool:
        """检查是否需要刷新数据"""
        if self._df is None:
            return True
        return (time.time() - self._last_refresh) >= self._refresh_interval

    # ============================================================
    # 多周期信号生成
    # ============================================================
    def compute_multi_tf_signal(self, tf: str) -> dict:
        """
        为指定时间框架计算信号 (独立于主时间框架)
        返回: {"tf": str, "ok": bool, "action": str, "sell_score": float, "buy_score": float, ...}
        """
        try:
            max_days = self._MAX_LOOKBACK_DAYS.get(tf, 60)
            lookback = min(self.config.lookback_days, max_days)

            # 获取K线数据
            df = fetch_binance_klines(self.symbol, interval=tf, days=lookback)
            if df is None or len(df) < 50:
                return {"tf": tf, "ok": False, "error": f"数据不足({len(df) if df is not None else 0} bars)"}

            # 计算指标
            df = add_all_indicators(df)
            add_moving_averages(df, timeframe=tf)
            data_all = {tf: df}

            # 获取8h辅助数据
            if tf not in ('8h', '12h', '16h', '24h'):
                try:
                    df_8h = fetch_binance_klines(self.symbol, interval='8h', days=lookback)
                    if df_8h is not None and len(df_8h) > 20:
                        df_8h = add_all_indicators(df_8h)
                        data_all['8h'] = df_8h
                except Exception:
                    pass

            # 计算六维信号
            signals = compute_signals_six(df, tf, data_all, max_bars=1500)

            # 计算融合分数
            idx = len(df) - 1
            regime_label = self._infer_regime_label(df, idx)
            config_dict = self._build_fusion_config(regime_label)
            sell_score, buy_score = calc_fusion_score_six(
                signals, df, idx, df.index[idx], config_dict
            )
            if (not getattr(self.config, 'use_regime_adaptive_fusion', False)
                    and getattr(self.config, 'use_regime_adaptive_reweight', False)
                    and regime_label == 'neutral'):
                sell_score *= float(getattr(self.config, 'regime_neutral_ss_dampen', 0.85))
                buy_score *= float(getattr(self.config, 'regime_neutral_bs_boost', 1.10))
            sell_score = float(sell_score)
            buy_score = float(buy_score)

            # 判断动作
            action = "HOLD"
            if sell_score >= self.config.short_threshold and sell_score > buy_score * 1.5:
                action = "OPEN_SHORT"
            elif buy_score >= self.config.long_threshold and buy_score > sell_score * 1.5:
                action = "OPEN_LONG"

            return {
                "tf": tf,
                "ok": True,
                "action": action,
                "sell_score": sell_score,
                "buy_score": buy_score,
                "price": float(df['close'].iloc[idx]),
                "timestamp": str(df.index[idx]),
                "bars": len(df),
                "regime_label": regime_label,
            }

        except Exception as e:
            return {"tf": tf, "ok": False, "error": str(e)}

    def compute_multi_tf_consensus(self, decision_tfs: List[str]) -> dict:
        """
        并行计算多个时间框架的信号，用连续分数融合生成共识决策。

        与回测引擎 (calc_multi_tf_consensus) 共用同一套融合算法
        (fuse_tf_scores)，避免离散化信息损失。

        参数:
            decision_tfs: 参与决策的时间框架列表

        返回:
            dict - 包含 consensus (共识决策) 和 tf_results (各TF信号)
        """
        if self.logger:
            self.logger.info(f"多周期共识: 计算 {','.join(decision_tfs)} ...")

        start_time = time.time()
        results = []

        # 并行获取各TF信号
        with ThreadPoolExecutor(max_workers=min(4, len(decision_tfs))) as executor:
            futures = {
                executor.submit(self.compute_multi_tf_signal, tf): tf
                for tf in decision_tfs
            }
            for future in as_completed(futures):
                tf = futures[future]
                try:
                    result = future.result(timeout=120)
                    results.append(result)
                except Exception as e:
                    results.append({"tf": tf, "ok": False, "error": str(e)})

        # ── 从各TF结果中提取连续分数 ──
        tf_scores = {}
        tf_timestamps = {}
        for r in results:
            if r.get("ok") and "sell_score" in r and "buy_score" in r:
                tf_scores[r["tf"]] = (r["sell_score"], r["buy_score"])
                if r.get("timestamp"):
                    tf_timestamps[r["tf"]] = r["timestamp"]

        # ── TF 时间戳对齐校验: 各周期最新 K 线时间差超过自身周期的 10% 时告警 ──
        _TF_MINUTES = {'15m': 15, '30m': 30, '1h': 60, '4h': 240, '8h': 480, '24h': 1440}
        if len(tf_timestamps) >= 2 and self.logger:
            import pandas as _pd
            _ts_parsed = {}
            for _tf, _ts in tf_timestamps.items():
                try:
                    _ts_parsed[_tf] = _pd.Timestamp(_ts)
                except Exception:
                    pass
            if len(_ts_parsed) >= 2:
                _ts_vals = sorted(_ts_parsed.values())
                _max_gap_minutes = (_ts_vals[-1] - _ts_vals[0]).total_seconds() / 60
                # 以最小参与周期作为容忍阈值
                _min_tf_min = min(_TF_MINUTES.get(tf, 60) for tf in _ts_parsed)
                _tolerance = _min_tf_min * 1.5
                if _max_gap_minutes > _tolerance:
                    _stale_tfs = [tf for tf, ts in _ts_parsed.items()
                                  if (_ts_vals[-1] - ts).total_seconds() / 60 > _tolerance]
                    self.logger.warning(
                        f"[多周期对齐] TF 时间戳偏差 {_max_gap_minutes:.0f}min "
                        f"(容忍 {_tolerance:.0f}min)，滞后 TF: {_stale_tfs}。"
                        "共识决策可能基于过时数据。"
                    )

        # ── 用统一融合算法计算共识 (与回测一致) ──
        fuse_config = {
            'short_threshold': self.config.short_threshold,
            'long_threshold': self.config.long_threshold,
            'coverage_min': getattr(self.config, 'coverage_min', 0.5),
        }
        if getattr(self.config, 'ml_use_neural_fusion', False):
            consensus = neural_fuse_tf_scores(tf_scores, decision_tfs, fuse_config)
            if self.logger and consensus.get('neural_prob') is not None:
                self.logger.info(f"[ML Neural Fusion] prob={consensus['neural_prob']:.4f}")
        else:
            consensus = fuse_tf_scores(tf_scores, decision_tfs, fuse_config)

        elapsed = time.time() - start_time
        if self.logger:
            decision = consensus.get("decision", {})
            self.logger.info(
                f"多周期共识完成 ({elapsed:.1f}s): "
                f"{decision.get('label', '?')} "
                f"strength={decision.get('strength', 0)} "
                f"direction={decision.get('direction', '?')} "
                f"coverage={consensus.get('coverage', 0):.0%} "
                f"fused_ss={consensus.get('weighted_ss', 0):.1f} "
                f"fused_bs={consensus.get('weighted_bs', 0):.1f}"
            )

        return {
            "consensus": consensus,
            "tf_results": results,
            "elapsed_sec": round(elapsed, 1),
        }
