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

from binance_fetcher import fetch_binance_klines
from indicators import add_all_indicators
from strategy_enhanced import analyze_signals_enhanced, get_signal_at, DEFAULT_SIG
from strategy_futures_v4 import _calc_top_score, _calc_bottom_score
from ma_indicators import add_moving_averages, compute_ma_signals
from candlestick_patterns import compute_candlestick_scores
from bollinger_strategy import compute_bollinger_scores
from volume_price_strategy import compute_volume_price_scores
from kdj_strategy import compute_kdj_scores
from signal_core import calc_fusion_score_six, compute_signals_six
from multi_tf_consensus import compute_weighted_consensus, fuse_tf_scores


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

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "price": self.price,
            "sell_score": self.sell_score,
            "buy_score": self.buy_score,
            "components": self.components,
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
            # 将 StrategyConfig 转为 dict 供 calc_fusion_score_six 使用
            config_dict = {
                'fusion_mode': self.config.fusion_mode,
                'veto_threshold': self.config.veto_threshold,
                'kdj_bonus': self.config.kdj_bonus,
            }

            dt = df.index[idx]
            sell_score, buy_score = calc_fusion_score_six(
                signals, df, idx, dt, config_dict
            )

            # 提取各维度分数
            components = self._extract_components(signals, idx, dt)

            # 构建结果
            result = SignalResult()
            result.timestamp = str(dt)
            result.price = float(df['close'].iloc[idx])
            result.sell_score = float(sell_score)
            result.buy_score = float(buy_score)
            result.components = components
            result.bar_index = idx
            result.conflict = (sell_score > 15 and buy_score > 15 and
                              abs(sell_score - buy_score) < 10)

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
            # 反向信号
            if bs >= cfg.close_short_bs and bs > ss * 0.7:
                signal.action = "CLOSE_SHORT"
                signal.reason = f"反向信号 BS={bs:.1f} >= {cfg.close_short_bs}"
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
            if ss >= cfg.close_long_ss and ss > bs * 0.7:
                signal.action = "CLOSE_LONG"
                signal.reason = f"反向信号 SS={ss:.1f} >= {cfg.close_long_ss}"
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

        # 开空
        if (not has_short and short_cooldown <= 0 and
                ss >= cfg.short_threshold and ss > bs * 1.5):
            signal.action = "OPEN_SHORT"
            signal.reason = f"开空 SS={ss:.1f} >= {cfg.short_threshold}"
            return signal

        # 开多
        if (not has_long and long_cooldown <= 0 and
                bs >= cfg.long_threshold and bs > ss * 1.5):
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
            config_dict = {
                'fusion_mode': self.config.fusion_mode,
                'veto_threshold': self.config.veto_threshold,
                'kdj_bonus': self.config.kdj_bonus,
            }
            sell_score, buy_score = calc_fusion_score_six(
                signals, df, idx, df.index[idx], config_dict
            )
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
        for r in results:
            if r.get("ok") and "sell_score" in r and "buy_score" in r:
                tf_scores[r["tf"]] = (r["sell_score"], r["buy_score"])

        # ── 用统一融合算法计算共识 (与回测一致) ──
        fuse_config = {
            'short_threshold': self.config.short_threshold,
            'long_threshold': self.config.long_threshold,
        }
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
