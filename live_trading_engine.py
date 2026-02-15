"""
实盘交易引擎
协调: 信号生成 → 风控检查 → 订单执行 → 绩效追踪 → 通知
支持: Paper / Testnet / Small Live / Scale Up 四个阶段
"""

import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timedelta
from typing import Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from live_config import LiveTradingConfig, TradingPhase, StrategyConfig
from trading_logger import TradingLogger
from notifier import create_notifier
from risk_manager import RiskManager
from order_manager import create_order_manager, BinanceAPIError
from live_signal_generator import LiveSignalGenerator, SignalResult
from performance_tracker import PerformanceTracker


class Position:
    """持仓状态"""

    def __init__(self, side: str, entry_price: float, quantity: float,
                 margin: float, leverage: int, entry_time: str,
                 order_id: str = ""):
        self.side = side               # LONG / SHORT
        self.entry_price = entry_price
        self.quantity = quantity
        self.margin = margin
        self.leverage = leverage
        self.entry_time = entry_time
        self.order_id = order_id
        self.bars_held = 0
        self.max_pnl_ratio = 0.0
        self.partial_tp_1_done = False
        self.partial_tp_2_done = False
        self.entry_fee = 0.0
        # 入场决策详情
        self.entry_reason: str = ""          # 开仓原因
        self.entry_signal: dict = {}         # 信号详情 {buy_score, sell_score, components, ...}
        self.entry_consensus: dict = {}      # 多周期共识 {direction, strength, label, ...}

    def calc_pnl(self, current_price: float) -> float:
        """计算未实现盈亏"""
        if self.side == "LONG":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def calc_pnl_ratio(self, current_price: float) -> float:
        """计算盈亏比例"""
        if self.margin == 0:
            return 0
        return self.calc_pnl(current_price) / self.margin

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "margin": self.margin,
            "leverage": self.leverage,
            "entry_time": self.entry_time,
            "bars_held": self.bars_held,
            "max_pnl_ratio": self.max_pnl_ratio,
            "partial_tp_1_done": self.partial_tp_1_done,
            "partial_tp_2_done": self.partial_tp_2_done,
            "entry_reason": self.entry_reason,
            "entry_signal": self.entry_signal,
            "entry_consensus": self.entry_consensus,
        }


class LiveTradingEngine:
    """
    实盘交易引擎 - 系统核心
    支持 Paper / Testnet / Small Live / Scale Up 四个阶段
    """

    def __init__(self, config: LiveTradingConfig):
        self.config = config
        self.phase = config.phase
        self.running = False

        # --- 创建子系统 ---
        # 日志
        self.logger = TradingLogger(
            log_dir=config.log_dir,
            name=f"live_{config.phase.value}"
        )

        # 通知
        self.notifier = create_notifier(config.telegram)

        # 风控
        risk_state_file = os.path.join(config.data_dir, "risk_state.json")
        self.risk_manager = RiskManager(
            config=config.risk,
            initial_capital=config.initial_capital,
            logger=self.logger,
            notifier=self.notifier,
            state_file=risk_state_file,
        )

        # 订单管理
        self.order_manager = create_order_manager(
            config, self.logger, self.notifier
        )

        # 信号生成
        self.signal_generator = LiveSignalGenerator(
            config=config.strategy,
            logger=self.logger,
        )

        # 绩效追踪
        self.tracker = PerformanceTracker(
            initial_capital=config.initial_capital,
            data_dir=config.data_dir,
        )

        # --- 交易状态 ---
        self.usdt = config.initial_capital
        self.frozen_margin = 0.0
        self.positions: Dict[str, Position] = {}  # "LONG" / "SHORT" -> Position
        self.short_cooldown = 0
        self.long_cooldown = 0
        self.total_bars = 0
        self._last_signal: Optional[SignalResult] = None
        self._last_balance_log = 0
        self._last_daily_summary = ""
        self._cumulative_funding = 0.0

        # --- 多周期共识状态 ---
        self._last_consensus: Optional[dict] = None
        self._last_consensus_time: float = 0
        self._use_multi_tf = config.strategy.use_multi_tf
        preferred_tfs = list(getattr(config.strategy, "decision_timeframes", []) or [])
        fallback_tfs = list(getattr(config.strategy, "decision_timeframes_fallback", []) or [])
        if len(preferred_tfs) >= 2:
            self._decision_tfs = preferred_tfs
            self._decision_tfs_source = "preferred"
        else:
            self._decision_tfs = fallback_tfs
            self._decision_tfs_source = "fallback"
        if self._use_multi_tf and len(self._decision_tfs) < 2:
            self._use_multi_tf = False
            self._decision_tfs_source = "disabled_insufficient_tfs"
        self._consensus_min_strength = config.strategy.consensus_min_strength
        self._consensus_position_scale = config.strategy.consensus_position_scale

        # --- v11: Score Calibrator ---
        self._score_calibrator = None
        self._score_cal_stats = {'evaluated': 0, 'allowed': 0, 'blocked': 0}
        if getattr(config.strategy, 'use_score_calibration', False):
            self._init_score_calibrator()

        # 加载持久化状态
        self._load_state()

    # ============================================================
    # 主运行循环
    # ============================================================
    def run(self):
        """启动交易引擎主循环"""
        self.running = True

        # 注册信号处理
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        self.logger.info("=" * 60)
        self.logger.info(f"  实盘交易引擎启动")
        self.logger.info(f"  阶段: {self.phase.value}")
        self.logger.info(f"  交易对: {self.config.strategy.symbol}")
        self.logger.info(f"  主时间框架: {self.config.strategy.timeframe}")
        if self._use_multi_tf:
            self.logger.info(f"  🔗 多周期决策: 启用")
            self.logger.info(f"  决策TFs: {','.join(self._decision_tfs)}")
            if self._decision_tfs_source == "fallback":
                self.logger.warning("  多周期TF使用回退配置（preferred不足2个）")
            self.logger.info(f"  最低共识强度: {self._consensus_min_strength}")
        else:
            self.logger.info(f"  多周期决策: 关闭 (单TF模式)")
        self.logger.info(f"  执行交易: {self.config.execute_trades}")
        self.logger.info(f"  初始资金: ${self.config.initial_capital:.2f}")
        self.logger.info(f"  杠杆: {self.config.strategy.leverage}x")
        self.logger.info(f"  融合模式: {self.config.strategy.fusion_mode}")
        self.logger.info("=" * 60)

        # 通知启动
        self.notifier.notify_system("START", (
            f"阶段: {self.phase.value}\n"
            f"交易对: {self.config.strategy.symbol}\n"
            f"时间框架: {self.config.strategy.timeframe}\n"
            f"执行交易: {self.config.execute_trades}\n"
            f"初始资金: ${self.config.initial_capital:.2f}"
        ))

        # API 连接测试
        if not self.order_manager.test_connection():
            self.logger.error("API 连接测试失败，引擎未启动")
            self.notifier.notify_system("ERROR", "API 连接测试失败")
            return

        # 如果是真实交易阶段，设置杠杆和保证金模式
        if self.config.execute_trades:
            self._setup_exchange()

        # 首次加载数据
        self.signal_generator.refresh_data(force=True)

        try:
            while self.running:
                self._tick()
                self._sleep_until_next_check()
        except Exception as e:
            self.logger.error(f"引擎异常退出: {e}\n{traceback.format_exc()}")
            self.notifier.notify_error(e, "引擎主循环")
        finally:
            self._on_shutdown()

    def _tick(self):
        """每个检查周期的核心逻辑"""
        try:
            # 1. 检查是否需要刷新数据 (新K线收盘)
            data_refreshed = self.signal_generator.refresh_data()

            if data_refreshed:
                self.total_bars += 1

                # 递减冷却
                if self.short_cooldown > 0:
                    self.short_cooldown -= 1
                if self.long_cooldown > 0:
                    self.long_cooldown -= 1

                # 更新持仓 bars_held
                for pos in self.positions.values():
                    pos.bars_held += 1

                # 2. 计算信号
                sig = self.signal_generator.compute_latest_signal()
                if sig is None:
                    return

                # 3. 获取当前价格
                current_price = sig.price

                # 4. 更新持仓盈亏
                self._update_positions_pnl(current_price)

                # 5. 风控检查
                equity = self._calc_equity(current_price)
                risk_actions = self.risk_manager.check_positions(
                    current_price, equity,
                    [p.to_dict() for p in self.positions.values()]
                )

                # 执行风控动作
                for action in risk_actions:
                    self._execute_risk_action(action, current_price)

                # 6. 检查部分止盈
                self._check_partial_take_profits(current_price)

                # 7. 评估交易动作 (先用单TF逻辑处理平仓)
                sig = self.signal_generator.evaluate_action(
                    sig,
                    has_long="LONG" in self.positions,
                    has_short="SHORT" in self.positions,
                    long_pnl_ratio=(self.positions["LONG"].calc_pnl_ratio(current_price)
                                   if "LONG" in self.positions else 0),
                    short_pnl_ratio=(self.positions["SHORT"].calc_pnl_ratio(current_price)
                                    if "SHORT" in self.positions else 0),
                    long_bars=(self.positions["LONG"].bars_held
                              if "LONG" in self.positions else 0),
                    short_bars=(self.positions["SHORT"].bars_held
                               if "SHORT" in self.positions else 0),
                    long_max_pnl=(self.positions["LONG"].max_pnl_ratio
                                 if "LONG" in self.positions else 0),
                    short_max_pnl=(self.positions["SHORT"].max_pnl_ratio
                                  if "SHORT" in self.positions else 0),
                    short_cooldown=self.short_cooldown,
                    long_cooldown=self.long_cooldown,
                )

                # 7b. 多周期共识门控 —— 开仓决策需要共识确认
                if self._use_multi_tf and sig.action in ("OPEN_LONG", "OPEN_SHORT"):
                    sig = self._apply_multi_tf_gate(sig)

                # 7c. v11 Score Calibration 门控
                if (getattr(self.config.strategy, 'use_score_calibration', False)
                        and sig.action in ("OPEN_LONG", "OPEN_SHORT")):
                    sig = self._apply_score_calibration_gate(sig)

                self._last_signal = sig

                # 8. 记录信号日志
                log_extra = {}
                if self._last_consensus:
                    cs = self._last_consensus.get("consensus", {})
                    cd = cs.get("decision", {})
                    log_extra = {
                        "consensus_label": cd.get("label", ""),
                        "consensus_strength": cd.get("strength", 0),
                        "consensus_direction": cd.get("direction", ""),
                        "fused_ss": round(cs.get("weighted_ss", 0), 1),
                        "fused_bs": round(cs.get("weighted_bs", 0), 1),
                        "coverage": cs.get("coverage", 0),
                    }
                self.logger.log_signal(
                    sell_score=sig.sell_score,
                    buy_score=sig.buy_score,
                    components=sig.components,
                    conflict=sig.conflict,
                    action_taken=sig.action,
                    timestamp=sig.timestamp,
                )

                # 9. 执行交易动作
                if sig.action != "HOLD":
                    self._execute_action(sig, current_price)

                # 9b. 反手逻辑 — 平仓后立即检查反方向开仓
                #     当 CLOSE_LONG 由"反向信号"触发时，检查 OPEN_SHORT
                #     当 CLOSE_SHORT 由"反向信号"触发时，检查 OPEN_LONG
                if sig.action in ("CLOSE_LONG", "CLOSE_SHORT") and "反向信号" in sig.reason:
                    reverse_sig = self._try_reverse_open(sig, current_price)
                    if reverse_sig and reverse_sig.action != "HOLD":
                        self._execute_action(reverse_sig, current_price)
                        # 更新信号记录用于通知
                        self.logger.log_signal(
                            sell_score=reverse_sig.sell_score,
                            buy_score=reverse_sig.buy_score,
                            components=reverse_sig.components,
                            conflict=reverse_sig.conflict,
                            action_taken=reverse_sig.action,
                            timestamp=reverse_sig.timestamp,
                        )
                        sig = reverse_sig

                # 10. 通知信号
                if self.config.telegram.notify_signals and sig.action != "HOLD":
                    self.notifier.notify_signal(
                        sig.sell_score, sig.buy_score,
                        sig.action, current_price,
                        self.config.strategy.symbol,
                    )

            # 定时任务
            now = time.time()

            # 余额日志
            if now - self._last_balance_log >= self.config.balance_log_interval_sec:
                self._log_balance()
                self._last_balance_log = now

            # 每日总结
            today = datetime.now().strftime("%Y-%m-%d")
            if today != self._last_daily_summary and datetime.now().hour >= 23:
                self._send_daily_summary(today)
                self._last_daily_summary = today

        except BinanceAPIError as e:
            self.logger.error(f"Binance API 错误: {e}")
            if e.code in (-1021, -1022):  # 时间同步/签名错误
                time.sleep(5)
        except Exception as e:
            self.logger.error(f"Tick 异常: {e}\n{traceback.format_exc()}")
            self.notifier.notify_error(e, "tick 循环")

    # ============================================================
    # 多周期共识门控
    # ============================================================
    def _apply_multi_tf_gate(self, sig: SignalResult,
                             reverse_mode: bool = False) -> SignalResult:
        """
        多周期共识门控: 开仓信号需要多周期共识确认

        参数:
          reverse_mode: 反手模式 — 平仓后立即反向开仓时使用。
                        放宽 "大小周期反向" 的限制，允许强信号通过。

        规则 (常规模式):
          - 共识 actionable=True 且方向一致 → 放行开仓
          - 共识 actionable=True 但方向相反 → 阻止
          - 共识 actionable=False → 阻止
          - 共识 strength >= consensus_min_strength → 放行

        规则 (反手模式，额外放宽):
          - "大小周期反向" → 允许通过 (因为刚刚平仓，短线反转信号可信)
          - 仅当 coverage 不足时才阻止
        """
        try:
            # 调用多周期信号计算
            multi_result = self.signal_generator.compute_multi_tf_consensus(
                self._decision_tfs
            )
            self._last_consensus = multi_result
            self._last_consensus_time = time.time()

            consensus = multi_result.get("consensus", {})
            decision = consensus.get("decision", {})
            direction = decision.get("direction", "hold")
            strength = decision.get("strength", 0)
            actionable = decision.get("actionable", False)
            label = decision.get("label", "")

            # 判断开仓方向是否与共识一致
            sig_direction = "long" if sig.action == "OPEN_LONG" else "short"

            fused_ss = round(consensus.get("weighted_ss", 0), 1)
            fused_bs = round(consensus.get("weighted_bs", 0), 1)
            coverage = consensus.get("coverage", 0)

            mode_tag = "[多周期门控-反手]" if reverse_mode else "[多周期门控]"
            self.logger.info(
                f"{mode_tag} 单TF建议={sig.action} | "
                f"共识={label} direction={direction} "
                f"strength={strength} actionable={actionable} "
                f"fused_ss={fused_ss} fused_bs={fused_bs} "
                f"coverage={coverage:.0%}"
            )

            # ── 反手模式: 宽松处理 ──
            if reverse_mode:
                # 反手时只在 coverage 不足时阻止
                if coverage < 0.3:
                    sig.action = "HOLD"
                    sig.reason = (f"反手被阻止: 覆盖率不足 "
                                  f"{coverage:.0%} < 30%")
                    self.logger.info(f"{mode_tag} ⛔ {sig.reason}")
                    return sig

                # 反手放行 — 附加信息，标记为反手开仓
                sig.reason = (f"{sig.reason} | 反手放行: {label} "
                              f"strength={strength}")
                self.logger.info(
                    f"{mode_tag} ✅ 反手开仓确认: {sig.action} "
                    f"(共识={direction}, 反手模式放宽门控)"
                )
                return sig

            # ── 常规模式 ──

            # 规则 0 (新增): 强信号覆盖 — 当 fused SS/BS 超强时
            #   允许通过，即使 actionable=False 或方向不一致
            strong_override = False
            if sig_direction == "short" and fused_ss >= 70:
                strong_override = True
            elif sig_direction == "long" and fused_bs >= 70:
                strong_override = True

            if strong_override and not actionable:
                sig.reason = (f"{sig.reason} | 强信号覆盖: {label} "
                              f"fused_{sig_direction[0]}s={fused_ss if sig_direction == 'short' else fused_bs:.1f}>=70")
                self.logger.info(
                    f"{mode_tag} ⚡ 强信号覆盖放行: {sig.action} "
                    f"fused_ss={fused_ss} fused_bs={fused_bs}"
                )
                return sig

            # 规则 1: 共识不可操作 → 阻止
            if not actionable:
                sig.action = "HOLD"
                sig.reason = (f"多周期共识阻止: {label} "
                              f"(strength={strength}, 需>={self._consensus_min_strength})")
                self.logger.info(f"{mode_tag} ⛔ 开仓被阻止: {sig.reason}")
                return sig

            # 规则 2: 共识方向与单TF方向不一致 → 阻止
            if direction != sig_direction:
                sig.action = "HOLD"
                sig.reason = (f"多周期共识方向不一致: 单TF={sig_direction} "
                              f"vs 共识={direction} ({label})")
                self.logger.info(f"{mode_tag} ⛔ 方向不一致: {sig.reason}")
                return sig

            # 规则 3: 共识强度不够 → 阻止
            if strength < self._consensus_min_strength:
                sig.action = "HOLD"
                sig.reason = (f"多周期共识强度不足: {strength} "
                              f"< {self._consensus_min_strength} ({label})")
                self.logger.info(f"{mode_tag} ⛔ 强度不足: {sig.reason}")
                return sig

            # 通过所有门控 → 放行，附加共识信息
            sig.reason = (f"{sig.reason} | 多周期确认: {label} "
                          f"strength={strength}")
            self.logger.info(
                f"{mode_tag} ✅ 开仓确认: {sig.action} "
                f"strength={strength}"
            )

            return sig

        except Exception as e:
            # 纸上/非执行模式: fail-open (不阻止交易, 降级为单TF)
            # 执行交易模式: fail-closed (阻止开仓, 安全第一)
            is_paper = (self.phase == TradingPhase.PAPER) or (not self.config.execute_trades)
            if is_paper:
                self.logger.warning(
                    f"[多周期门控] 计算异常，降级为单TF模式 (paper): {e}"
                )
                return sig
            else:
                self.logger.error(
                    f"[多周期门控] 计算异常，fail-closed 阻止开仓: {e}"
                )
                sig.action = "HOLD"
                sig.reason = f"多周期计算异常 fail-closed: {e}"
                return sig

    # ============================================================
    # v11: Score Calibration 门控
    # ============================================================
    def _init_score_calibrator(self):
        """加载 Score Calibration 模型"""
        try:
            from score_calibrator import ScoreCalibrator
            model_path = getattr(self.config.strategy, 'score_calibration_model_path', '')
            if not model_path:
                # 自动查找默认路径
                base_dir = os.path.dirname(os.path.abspath(__file__))
                for candidate in ('score_calibration.json', 'data/score_calibration.json'):
                    p = os.path.join(base_dir, candidate)
                    if os.path.exists(p):
                        model_path = p
                        break
            if model_path and os.path.exists(model_path):
                cal = ScoreCalibrator()
                cal.load(model_path)
                self._score_calibrator = cal
                self.logger.info(f"[ScoreCal] 校准模型已加载: {model_path}")
            else:
                self.logger.warning(
                    "[ScoreCal] 未找到校准模型文件, Score Calibration 将以 shadow 模式运行 "
                    "(仅记录, 不拦截)"
                )
        except Exception as e:
            self.logger.warning(f"[ScoreCal] 初始化失败(非致命): {e}")

    def _apply_score_calibration_gate(self, sig: SignalResult) -> SignalResult:
        """
        Score Calibration 门控:
        - 有校准模型时: 使用 should_enter() 判断
        - shadow_mode=True 时: 只记录不拦截
        - 无模型时: 透传
        """
        cfg = self.config.strategy
        shadow_mode = getattr(cfg, 'score_calibration_shadow_mode', True)
        cost = float(getattr(cfg, 'score_calibration_cost', 0.0015))
        min_p = float(getattr(cfg, 'score_calibration_min_p_win', 0.48))

        if self._score_calibrator is None:
            return sig

        try:
            direction = 'short' if sig.action == 'OPEN_SHORT' else 'long'
            score = sig.sell_score if direction == 'short' else sig.buy_score
            regime = sig.regime_label

            self._score_cal_stats['evaluated'] += 1
            ok, info = self._score_calibrator.should_enter(direction, regime, score, cost, min_p)

            p_win = info.get('calibrated_p_win', 0)
            e_r = info.get('calibrated_e_r', 0)

            if ok:
                self._score_cal_stats['allowed'] += 1
                self.logger.info(
                    f"[ScoreCal] ✅ {direction} {regime} score={score:.1f} "
                    f"p_win={p_win:.3f} E[R]={e_r:.4f} → 放行"
                )
            else:
                self._score_cal_stats['blocked'] += 1
                if shadow_mode:
                    self.logger.info(
                        f"[ScoreCal] 👻 SHADOW: {direction} {regime} score={score:.1f} "
                        f"p_win={p_win:.3f} E[R]={e_r:.4f} → 本应拦截(shadow模式放行)"
                    )
                else:
                    sig.action = "HOLD"
                    sig.reason = (
                        f"ScoreCal拦截: {direction} {regime} score={score:.1f} "
                        f"p_win={p_win:.3f}<{min_p} E[R]={e_r:.4f}"
                    )
                    self.logger.info(
                        f"[ScoreCal] ⛔ 拦截: {direction} {regime} score={score:.1f} "
                        f"p_win={p_win:.3f} E[R]={e_r:.4f}"
                    )
        except Exception as e:
            self.logger.warning(f"[ScoreCal] 评估失败(非致命, 放行): {e}")

        return sig

    # ============================================================
    # 反手 (Reverse) 逻辑
    # ============================================================
    def _try_reverse_open(self, close_sig: SignalResult, price: float):
        """
        平仓后立即检查反方向开仓 (反手)。

        当 CLOSE_LONG 由 "反向信号" 触发时 (SS >= close_long_ss)，
        检查是否应该 OPEN_SHORT；反之亦然。

        反手开仓使用宽松的多周期门控:
          - 跳过 "大小周期反向" 的硬拒绝
          - 只要信号足够强 (SS >= short_threshold) 且无冷却即可执行
        """
        from copy import deepcopy
        sig = deepcopy(close_sig)

        ss = sig.sell_score
        bs = sig.buy_score
        cfg = self.config.strategy

        if close_sig.action == "CLOSE_LONG":
            # 尝试反手做空
            if (ss >= cfg.short_threshold and
                    ss > bs * 1.2 and  # 宽松比率 (原始 1.5 太严)
                    self.short_cooldown <= 0 and
                    "SHORT" not in self.positions):
                sig.action = "OPEN_SHORT"
                sig.reason = f"反手做空 SS={ss:.1f} >= {cfg.short_threshold} (平多后)"
                self.logger.info(
                    f"[反手] ✅ 平多后反手做空: SS={ss:.1f} BS={bs:.1f}"
                )
                # 反手时仍然通过多周期门控，但使用宽松模式
                if self._use_multi_tf:
                    sig = self._apply_multi_tf_gate(sig, reverse_mode=True)
                return sig
            else:
                self.logger.info(
                    f"[反手] ⏭️ 平多后不满足反手条件: SS={ss:.1f} BS={bs:.1f} "
                    f"需要 SS>={cfg.short_threshold} & SS>BS*1.2"
                )

        elif close_sig.action == "CLOSE_SHORT":
            # 尝试反手做多
            if (bs >= cfg.long_threshold and
                    bs > ss * 1.2 and  # 宽松比率
                    self.long_cooldown <= 0 and
                    "LONG" not in self.positions):
                sig.action = "OPEN_LONG"
                sig.reason = f"反手做多 BS={bs:.1f} >= {cfg.long_threshold} (平空后)"
                self.logger.info(
                    f"[反手] ✅ 平空后反手做多: BS={bs:.1f} SS={ss:.1f}"
                )
                if self._use_multi_tf:
                    sig = self._apply_multi_tf_gate(sig, reverse_mode=True)
                return sig
            else:
                self.logger.info(
                    f"[反手] ⏭️ 平空后不满足反手条件: BS={bs:.1f} SS={ss:.1f} "
                    f"需要 BS>={cfg.long_threshold} & BS>SS*1.2"
                )

        return None

    # ============================================================
    # 交易执行
    # ============================================================
    def _execute_action(self, sig: SignalResult, price: float):
        """执行交易动作"""
        action = sig.action
        symbol = self.config.strategy.symbol

        if action == "OPEN_LONG":
            self._open_position("LONG", price, sig.reason, sig=sig)

        elif action == "OPEN_SHORT":
            self._open_position("SHORT", price, sig.reason, sig=sig)

        elif action == "CLOSE_LONG":
            self._close_position("LONG", price, sig.reason)

        elif action == "CLOSE_SHORT":
            self._close_position("SHORT", price, sig.reason)

    def _open_position(self, side: str, price: float, reason: str,
                       sig: 'SignalResult | None' = None):
        """开仓"""
        symbol = self.config.strategy.symbol
        cfg = self.config.strategy
        leverage = self.risk_manager.constrain_leverage(cfg.leverage)

        # 计算保证金
        equity = self._calc_equity(price)
        raw_margin = equity * cfg.margin_use * cfg.single_pct

        # 多周期共识强度缩放仓位
        if (self._use_multi_tf and self._consensus_position_scale
                and self._last_consensus):
            consensus_strength = (self._last_consensus
                                  .get("consensus", {})
                                  .get("decision", {})
                                  .get("strength", 50))
            # strength 40-100 映射到 0.5-1.0 的仓位比例
            scale = max(0.5, min(1.0, consensus_strength / 100))
            raw_margin *= scale
            self.logger.info(
                f"[仓位缩放] 共识强度={consensus_strength} → "
                f"仓位比例={scale:.0%}"
            )

        margin = self.risk_manager.constrain_margin(
            raw_margin, equity, self.frozen_margin
        )

        if margin <= 0:
            self.logger.info(f"可用保证金不足，跳过 {side} 开仓")
            return

        # 风控检查
        allowed, deny_reason = self.risk_manager.can_open_position(
            side, margin, equity, self.frozen_margin
        )
        if not allowed:
            self.logger.info(f"风控拒绝开仓: {deny_reason}")
            return

        # 计算数量
        quantity = (margin * leverage) / price

        if self.config.execute_trades:
            # --- 真实下单 ---
            try:
                if side == "LONG":
                    result = self.order_manager.market_open_long(
                        symbol, quantity, reason
                    )
                else:
                    result = self.order_manager.market_open_short(
                        symbol, quantity, reason
                    )

                # 获取实际成交价格
                actual_price = float(result.get("avgPrice", price))
                order_id = str(result.get("orderId", ""))

                # 滑点检查
                ok, slip = self.risk_manager.check_slippage(price, actual_price, side)
                if not ok:
                    # 滑点过大，立即平仓
                    self.logger.warning(f"滑点过大 {slip:.3%}，立即平仓")
                    if side == "LONG":
                        self.order_manager.market_close_long(symbol, quantity, "滑点过大")
                    else:
                        self.order_manager.market_close_short(symbol, quantity, "滑点过大")
                    return

                entry_price = actual_price

            except Exception as e:
                self.logger.error(f"下单失败: {e}")
                self.notifier.notify_error(e, f"开仓 {side}")
                return
        else:
            # --- Paper / 模拟 ---
            # 模拟滑点
            slippage = 0.001
            if side == "LONG":
                entry_price = price * (1 + slippage)
            else:
                entry_price = price * (1 - slippage)
            order_id = f"PAPER_{self.total_bars}"

        # 计算手续费
        notional = quantity * entry_price
        fee = notional * 0.0005  # Taker fee

        # 创建持仓
        pos = Position(
            side=side, entry_price=entry_price,
            quantity=quantity, margin=margin,
            leverage=leverage,
            entry_time=datetime.now().isoformat(),
            order_id=order_id,
        )
        pos.entry_fee = fee
        # 保存入场决策详情
        pos.entry_reason = reason
        if sig is not None:
            pos.entry_signal = {
                "buy_score": sig.buy_score,
                "sell_score": sig.sell_score,
                "components": sig.components,
                "conflict": sig.conflict,
                "price": sig.price,
                "timestamp": sig.timestamp,
            }
        if self._last_consensus:
            consensus = self._last_consensus.get("consensus", {})
            decision = consensus.get("decision", {})
            tf_scores = consensus.get("tf_scores", {})
            pos.entry_consensus = {
                "direction": decision.get("direction", ""),
                "strength": decision.get("strength", 0),
                "label": decision.get("label", ""),
                "actionable": decision.get("actionable", False),
                "tf_scores": tf_scores,
                "decision_tfs": self._decision_tfs,
                "weighted_ss": consensus.get("weighted_ss", 0),
                "weighted_bs": consensus.get("weighted_bs", 0),
                "coverage": consensus.get("coverage", 0),
            }
        self.positions[side] = pos

        # 更新资金
        self.usdt -= (margin + fee)
        self.frozen_margin += margin

        # 日志
        self.logger.log_trade(
            action=f"OPEN_{side}",
            symbol=self.config.strategy.symbol,
            side=side, price=entry_price,
            qty=quantity, margin=margin,
            leverage=leverage, fee=fee,
            reason=reason, order_id=order_id,
        )

        # 通知
        self.notifier.notify_trade(
            action=f"OPEN_{side}",
            symbol=self.config.strategy.symbol,
            side=side, price=entry_price,
            qty=quantity, margin=margin,
            leverage=leverage, fee=fee,
            reason=reason,
        )

        # 设置冷却
        if side == "SHORT":
            self.short_cooldown = self.config.strategy.cooldown
        else:
            self.long_cooldown = self.config.strategy.cooldown

        # 记录
        self.tracker.record_trade({
            "action": f"OPEN_{side}", "side": side,
            "price": entry_price, "qty": quantity,
            "margin": margin, "leverage": leverage,
            "fee": fee, "pnl": 0, "reason": reason,
            "expected_price": price, "actual_price": entry_price,
        })

        # 在交易所设置止损单 (实盘安全网)
        if self.config.execute_trades:
            self._place_exchange_stop_loss(side, entry_price, quantity)

        self._save_state()

    def _close_position(self, side: str, price: float, reason: str,
                        partial_pct: float = 1.0):
        """平仓 (支持部分平仓)"""
        if side not in self.positions:
            return

        pos = self.positions[side]
        symbol = self.config.strategy.symbol
        close_qty = pos.quantity * partial_pct

        if self.config.execute_trades:
            try:
                if side == "LONG":
                    result = self.order_manager.market_close_long(
                        symbol, close_qty, reason
                    )
                else:
                    result = self.order_manager.market_close_short(
                        symbol, close_qty, reason
                    )
                actual_price = float(result.get("avgPrice", price))
                order_id = str(result.get("orderId", ""))
            except Exception as e:
                self.logger.error(f"平仓失败: {e}")
                self.notifier.notify_error(e, f"平仓 {side}")
                return
        else:
            # 模拟滑点
            slippage = 0.001
            if side == "LONG":
                actual_price = price * (1 - slippage)  # 平多是卖
            else:
                actual_price = price * (1 + slippage)  # 平空是买
            order_id = f"PAPER_CLOSE_{self.total_bars}"

        # 计算盈亏
        if side == "LONG":
            pnl = (actual_price - pos.entry_price) * close_qty
        else:
            pnl = (pos.entry_price - actual_price) * close_qty

        fee = close_qty * actual_price * 0.0005
        margin_released = pos.margin * partial_pct
        net_pnl = pnl - fee

        # 更新资金
        self.usdt += margin_released + pnl - fee
        self.frozen_margin -= margin_released

        # 日志
        action = f"CLOSE_{side}" if partial_pct >= 1.0 else f"PARTIAL_TP_{side}"
        self.logger.log_trade(
            action=action,
            symbol=symbol, side=side,
            price=actual_price, qty=close_qty,
            margin=margin_released,
            leverage=pos.leverage, fee=fee,
            pnl=net_pnl, reason=reason,
            order_id=order_id,
        )

        # 通知
        self.notifier.notify_trade(
            action=action, symbol=symbol,
            side=side, price=actual_price,
            qty=close_qty, margin=margin_released,
            leverage=pos.leverage, fee=fee,
            pnl=net_pnl, reason=reason,
        )

        # 记录绩效
        self.tracker.record_trade({
            "action": action, "side": side,
            "price": actual_price, "qty": close_qty,
            "margin": margin_released,
            "leverage": pos.leverage,
            "fee": fee, "pnl": net_pnl, "reason": reason,
            "expected_price": price, "actual_price": actual_price,
            "bars_held": pos.bars_held,
        })

        # 更新风控
        if partial_pct >= 1.0:
            equity = self._calc_equity(price)
            self.risk_manager.on_trade_closed(net_pnl, fee, equity)
            del self.positions[side]

            # 撤销交易所止损单
            if self.config.execute_trades:
                try:
                    self.order_manager.cancel_all_orders(symbol)
                except Exception:
                    pass

            # 设置冷却
            if side == "SHORT":
                self.short_cooldown = self.config.strategy.cooldown
            else:
                self.long_cooldown = self.config.strategy.cooldown
        else:
            # 部分平仓 - 更新持仓
            pos.quantity -= close_qty
            pos.margin -= margin_released

        self.tracker.record_equity(self._calc_equity(price))
        self._save_state()

    def _check_partial_take_profits(self, price: float):
        """检查部分止盈"""
        for side, pos in list(self.positions.items()):
            pnl_ratio = pos.calc_pnl_ratio(price)

            tp_info = self.signal_generator.check_partial_tp(
                side, pnl_ratio,
                pos.partial_tp_1_done,
                pos.partial_tp_2_done,
            )

            if tp_info:
                level = tp_info["level"]
                close_pct = tp_info["close_pct"]
                reason = f"部分止盈 Level {level} pnl={pnl_ratio:.1%}"

                self._close_position(side, price, reason, partial_pct=close_pct)

                if level == 1:
                    pos.partial_tp_1_done = True
                elif level == 2:
                    pos.partial_tp_2_done = True

    def _place_exchange_stop_loss(self, side: str, entry_price: float,
                                  quantity: float):
        """在交易所层面设置止损单 (安全网)"""
        cfg = self.config.strategy
        symbol = cfg.symbol

        try:
            if side == "LONG":
                sl_price = entry_price * (1 + cfg.long_sl)
                self.order_manager.place_stop_loss(
                    symbol, side, sl_price, quantity, "交易所安全止损"
                )
            else:
                sl_price = entry_price * (1 - cfg.short_sl)
                self.order_manager.place_stop_loss(
                    symbol, side, sl_price, quantity, "交易所安全止损"
                )
            self.logger.info(f"交易所止损单已设置: {side} SL={sl_price:.2f}")
        except Exception as e:
            self.logger.warning(f"设置交易所止损单失败: {e}")

    # ============================================================
    # 辅助方法
    # ============================================================
    def _update_positions_pnl(self, price: float):
        """更新持仓最大盈亏比例"""
        for pos in self.positions.values():
            ratio = pos.calc_pnl_ratio(price)
            if ratio > pos.max_pnl_ratio:
                pos.max_pnl_ratio = ratio

    def _calc_equity(self, price: float) -> float:
        """计算总权益"""
        equity = self.usdt + self.frozen_margin
        for pos in self.positions.values():
            equity += pos.calc_pnl(price)
        return equity

    def _execute_risk_action(self, action: dict, price: float):
        """执行风控动作"""
        if action["action"] == "CLOSE":
            side = action["side"]
            reason = action.get("reason", "风控平仓")
            if side in self.positions:
                self._close_position(side, price, reason)

    def _setup_exchange(self):
        """设置交易所参数 (杠杆、保证金模式)"""
        symbol = self.config.strategy.symbol
        leverage = self.risk_manager.constrain_leverage(
            self.config.strategy.leverage
        )

        try:
            self.order_manager.set_margin_type(symbol, "ISOLATED")
            self.logger.info(f"保证金模式: ISOLATED")
        except Exception as e:
            self.logger.warning(f"设置保证金模式失败: {e}")

        try:
            self.order_manager.set_leverage(symbol, leverage)
            self.logger.info(f"杠杆设置: {leverage}x")
        except Exception as e:
            self.logger.warning(f"设置杠杆失败: {e}")

    def _log_balance(self):
        """记录余额快照"""
        try:
            price = self.signal_generator._df['close'].iloc[-1] if \
                self.signal_generator._df is not None else 0
            equity = self._calc_equity(price)
            unrealized = sum(p.calc_pnl(price) for p in self.positions.values())

            positions_info = []
            for pos in self.positions.values():
                positions_info.append({
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "qty": pos.quantity,
                    "pnl": pos.calc_pnl(price),
                })

            self.logger.log_balance(
                total_equity=equity,
                usdt=self.usdt,
                unrealized_pnl=unrealized,
                frozen_margin=self.frozen_margin,
                available_margin=self.usdt - self.frozen_margin,
                positions=positions_info,
            )

            self.tracker.record_equity(equity)

            # 定期保存状态到文件，供 Web 控制面板读取
            self._save_state()

        except Exception:
            pass

    def _send_daily_summary(self, date: str):
        """发送每日总结"""
        try:
            daily = self.tracker.get_daily_summary(date)
            summary = self.tracker.get_summary()

            price = (self.signal_generator._df['close'].iloc[-1]
                    if self.signal_generator._df is not None else 0)
            equity = self._calc_equity(price)

            positions_info = []
            for pos in self.positions.values():
                positions_info.append({
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "pnl": pos.calc_pnl(price),
                })

            self.notifier.notify_daily_summary(
                date=date, equity=equity,
                daily_pnl=daily["pnl"],
                daily_return=daily["pnl"] / self.config.initial_capital,
                trades_count=daily["trades"],
                wins=daily["wins"],
                losses=daily["losses"],
                max_drawdown=summary["max_drawdown"],
                positions=positions_info,
                extra={
                    "总收益率": f"{summary['total_return']:.2%}",
                    "总胜率": f"{summary['win_rate']:.0%}",
                    "总交易": summary["total_trades"],
                },
            )

            self.tracker.record_daily(date, daily)

        except Exception as e:
            self.logger.error(f"每日总结发送失败: {e}")

    def _sleep_until_next_check(self):
        """等待到下次检查"""
        interval = self.config.signal_check_interval_sec
        time.sleep(interval)

    # ============================================================
    # 状态持久化
    # ============================================================
    def _save_state(self):
        """保存引擎状态"""
        price = 0
        try:
            if self.signal_generator._df is not None:
                price = self.signal_generator._df['close'].iloc[-1]
        except Exception:
            pass

        state = {
            "phase": self.phase.value,
            "initial_capital": self.config.initial_capital,
            "usdt": self.usdt,
            "frozen_margin": self.frozen_margin,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "short_cooldown": self.short_cooldown,
            "long_cooldown": self.long_cooldown,
            "total_bars": self.total_bars,
            "cumulative_funding": self._cumulative_funding,
            "equity": self._calc_equity(price),
            "unrealized_pnl": sum(p.calc_pnl(price) for p in self.positions.values()),
            "symbol": self.config.strategy.symbol,
            "timeframe": self.config.strategy.timeframe,
            "leverage": self.config.strategy.leverage,
            "execute_trades": self.config.execute_trades,
            "running": self.running,
            "pid": os.getpid(),
            "saved_at": datetime.now().isoformat(),
            "use_multi_tf": self._use_multi_tf,
            "decision_tfs": self._decision_tfs if self._use_multi_tf else [],
            "last_consensus": {
                "decision": (self._last_consensus or {}).get("consensus", {}).get("decision", {}),
                "time": datetime.fromtimestamp(self._last_consensus_time).isoformat()
                        if self._last_consensus_time else None,
            } if self._last_consensus else None,
        }
        os.makedirs(self.config.data_dir, exist_ok=True)
        filepath = os.path.join(self.config.data_dir, "engine_state.json")
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        # 同步保存风控状态
        try:
            self.risk_manager._save_state()
        except Exception:
            pass

    def _load_state(self):
        """加载引擎状态"""
        filepath = os.path.join(self.config.data_dir, "engine_state.json")
        if not os.path.exists(filepath):
            return

        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            self.usdt = state.get("usdt", self.config.initial_capital)
            self.frozen_margin = state.get("frozen_margin", 0)
            self.short_cooldown = state.get("short_cooldown", 0)
            self.long_cooldown = state.get("long_cooldown", 0)
            self.total_bars = state.get("total_bars", 0)
            self._cumulative_funding = state.get("cumulative_funding", 0)

            # 恢复持仓
            for side, pos_data in state.get("positions", {}).items():
                self.positions[side] = Position(
                    side=pos_data["side"],
                    entry_price=pos_data["entry_price"],
                    quantity=pos_data["quantity"],
                    margin=pos_data["margin"],
                    leverage=pos_data["leverage"],
                    entry_time=pos_data.get("entry_time", ""),
                )
                self.positions[side].bars_held = pos_data.get("bars_held", 0)
                self.positions[side].max_pnl_ratio = pos_data.get("max_pnl_ratio", 0)
                self.positions[side].partial_tp_1_done = pos_data.get("partial_tp_1_done", False)
                self.positions[side].partial_tp_2_done = pos_data.get("partial_tp_2_done", False)
                # 恢复入场决策详情
                self.positions[side].entry_reason = pos_data.get("entry_reason", "")
                self.positions[side].entry_signal = pos_data.get("entry_signal", {})
                self.positions[side].entry_consensus = pos_data.get("entry_consensus", {})

            self.logger.info(
                f"状态已恢复: usdt=${self.usdt:.2f} "
                f"frozen=${self.frozen_margin:.2f} "
                f"positions={list(self.positions.keys())} "
                f"bars={self.total_bars}"
            )

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"状态加载失败，使用默认值: {e}")

    # ============================================================
    # 生命周期
    # ============================================================
    def _handle_shutdown(self, signum, frame):
        """优雅关机"""
        self.logger.info(f"收到关机信号 {signum}，开始优雅关机...")
        self.running = False

    def _on_shutdown(self):
        """关机清理"""
        self.logger.info("引擎关闭中...")
        self._save_state()

        # 发送通知
        summary = self.tracker.get_summary()
        self.notifier.notify_system("STOP", (
            f"引擎已停止\n"
            f"总交易: {summary['total_trades']}\n"
            f"总盈亏: ${summary['total_pnl']:+.2f}\n"
            f"胜率: {summary['win_rate']:.0%}"
        ))

        self.logger.info("引擎已安全关闭")

    # ============================================================
    # 外部控制接口
    # ============================================================
    def get_status(self) -> dict:
        """获取引擎状态"""
        price = 0
        if self.signal_generator._df is not None:
            price = float(self.signal_generator._df['close'].iloc[-1])

        return {
            "phase": self.phase.value,
            "running": self.running,
            "execute_trades": self.config.execute_trades,
            "initial_capital": self.config.initial_capital,
            "usdt": self.usdt,
            "equity": self._calc_equity(price),
            "frozen_margin": self.frozen_margin,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "short_cooldown": self.short_cooldown,
            "long_cooldown": self.long_cooldown,
            "total_bars": self.total_bars,
            "last_signal": self._last_signal.to_dict() if self._last_signal else None,
            "risk_status": self.risk_manager.get_status_report(),
            "data_info": self.signal_generator.get_current_data_info(),
            "performance": self.tracker.get_summary(),
            "multi_tf": {
                "enabled": self._use_multi_tf,
                "decision_tfs": self._decision_tfs,
                "consensus_min_strength": self._consensus_min_strength,
                "last_consensus": (self._last_consensus or {}).get("consensus", {}).get("decision", {}),
            },
        }

    def kill_switch(self, reason: str = "手动触发"):
        """一键平仓"""
        self.risk_manager.activate_kill_switch(reason)

        # 平仓所有持仓
        if self.signal_generator._df is not None:
            price = float(self.signal_generator._df['close'].iloc[-1])
        else:
            price = self.order_manager.get_current_price(
                self.config.strategy.symbol
            )

        for side in list(self.positions.keys()):
            self._close_position(side, price, f"Kill Switch: {reason}")

        # 如果是真实交易，也在交易所层面平仓
        if self.config.execute_trades:
            self.order_manager.close_all_positions(
                self.config.strategy.symbol
            )

    def resume(self, reason: str = "人工审查后恢复"):
        """恢复交易"""
        self.risk_manager.resume_trading(reason)
