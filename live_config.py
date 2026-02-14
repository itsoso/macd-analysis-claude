"""
实盘交易配置管理
支持 Phase 0-4 的多阶段配置，安全参数约束
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class TradingPhase(Enum):
    """交易阶段"""
    PAPER = "paper"           # Phase 1: 纸上交易
    TESTNET = "testnet"       # Phase 2: 测试网实盘
    SMALL_LIVE = "small_live" # Phase 3: 小资金实盘
    SCALE_UP = "scale_up"     # Phase 4: 逐步加仓


@dataclass
class APIConfig:
    """Binance API 配置"""
    api_key: str = ""
    api_secret: str = ""
    # Testnet 配置
    testnet_api_key: str = ""
    testnet_api_secret: str = ""
    # 端点
    mainnet_base_url: str = "https://fapi.binance.com"
    testnet_base_url: str = "https://testnet.binancefuture.com"
    mainnet_ws_url: str = "wss://fstream.binance.com"
    testnet_ws_url: str = "wss://stream.binancefuture.com"

    def get_base_url(self, phase: TradingPhase) -> str:
        if phase in (TradingPhase.PAPER, TradingPhase.TESTNET):
            return self.testnet_base_url
        return self.mainnet_base_url

    def get_ws_url(self, phase: TradingPhase) -> str:
        if phase in (TradingPhase.PAPER, TradingPhase.TESTNET):
            return self.testnet_ws_url
        return self.mainnet_ws_url

    def get_credentials(self, phase: TradingPhase) -> tuple:
        if phase in (TradingPhase.PAPER, TradingPhase.TESTNET):
            return self.testnet_api_key, self.testnet_api_secret
        return self.api_key, self.api_secret


@dataclass
class TelegramConfig:
    """Telegram 通知配置"""
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""
    # 通知级别控制
    notify_trades: bool = True       # 每笔交易通知
    notify_signals: bool = False     # 信号通知（可能很频繁）
    notify_risk: bool = True         # 风险事件通知
    notify_daily_summary: bool = True  # 每日总结
    notify_errors: bool = True       # 错误通知


@dataclass
class RiskConfig:
    """风控配置 - 按阶段递进"""
    # --- 基础风控 ---
    max_leverage: int = 2                  # 最大杠杆
    max_single_position_pct: float = 0.20  # 单仓最大占比
    max_total_margin_pct: float = 0.50     # 总仓位最大占比
    reserve_pct: float = 0.30             # 资金储备比例（至少保留30%）

    # --- 熔断机制 ---
    max_daily_loss_pct: float = 0.05       # 日最大亏损比例 → 暂停
    max_weekly_loss_pct: float = 0.10      # 周最大亏损比例 → 暂停
    max_consecutive_losses: int = 5        # 连续亏损次数 → 暂停
    max_drawdown_pct: float = 0.15         # 最大回撤 → 暂停

    # --- 订单安全 ---
    order_timeout_sec: int = 30            # 订单超时时间
    max_retry_count: int = 3              # 下单重试次数
    max_slippage_pct: float = 0.005        # 最大可接受滑点 (0.5%)

    # --- 冷却期 ---
    cooldown_after_liquidation_bars: int = 24  # 强平后冷却K线数
    cooldown_after_stop_loss_bars: int = 6     # 止损后冷却K线数

    # --- Kill Switch ---
    kill_switch_enabled: bool = True       # 是否启用一键平仓
    emergency_close_all: bool = False      # 紧急平仓标志


# ============================================================
# 各阶段预设配置
# ============================================================

PHASE_CONFIGS = {
    TradingPhase.PAPER: {
        "description": "Phase 1: 纸上交易 - 仅记录信号，不执行交易",
        "execute_trades": False,
        "risk": RiskConfig(
            max_leverage=5,          # Paper 不限制
            max_daily_loss_pct=1.0,  # Paper 不熔断
            max_weekly_loss_pct=1.0,
            max_consecutive_losses=999,
            max_drawdown_pct=1.0,
        ),
        "initial_capital": 100000,   # 模拟资金
        "leverage": 5,
    },
    TradingPhase.TESTNET: {
        "description": "Phase 2: 测试网实盘 - 真实API，模拟资金",
        "execute_trades": True,
        "risk": RiskConfig(
            max_leverage=5,
            max_daily_loss_pct=0.20,
            max_weekly_loss_pct=0.30,
            max_consecutive_losses=10,
            max_drawdown_pct=0.30,
        ),
        "initial_capital": 10000,    # Testnet 模拟资金
        "leverage": 5,
    },
    TradingPhase.SMALL_LIVE: {
        "description": "Phase 3: 小资金实盘 - $500-1000, 2x杠杆",
        "execute_trades": True,
        "risk": RiskConfig(
            max_leverage=2,           # 严格限制2x
            max_single_position_pct=0.15,
            max_total_margin_pct=0.40,
            reserve_pct=0.50,         # 保留50%
            max_daily_loss_pct=0.05,
            max_weekly_loss_pct=0.10,
            max_consecutive_losses=5,
            max_drawdown_pct=0.10,    # 10%回撤暂停
            max_slippage_pct=0.003,
        ),
        "initial_capital": 500,      # 起始 $500
        "leverage": 2,               # 2x 杠杆
    },
    TradingPhase.SCALE_UP: {
        "description": "Phase 4: 逐步加仓 - 盈利后渐进放大",
        "execute_trades": True,
        "risk": RiskConfig(
            max_leverage=3,
            max_single_position_pct=0.20,
            max_total_margin_pct=0.50,
            reserve_pct=0.30,
            max_daily_loss_pct=0.05,
            max_weekly_loss_pct=0.08,
            max_consecutive_losses=5,
            max_drawdown_pct=0.15,
        ),
        "initial_capital": 1000,
        "leverage": 3,
    },
}


# ── 策略参数版本 ──
# 通过环境变量 STRATEGY_VERSION=v1/v2/v3/v4 可切换
# 默认使用 v4 (run31 — param_sweep 系统性优化)
STRATEGY_PARAM_VERSIONS = {
    "v1": {  # run28 基线参数 (趋势v3 + LiveGate + Regime)
        "short_threshold": 25,
        "long_threshold": 40,
        "short_sl": -0.25,
        "short_tp": 0.60,
        "long_tp": 0.30,
        "partial_tp_1": 0.20,
        "use_partial_tp_2": False,
        "short_max_hold": 72,
    },
    "v2": {  # run29 优化参数 (2026-02-14 参数扫描)
        "short_threshold": 35,
        "long_threshold": 30,
        "short_sl": -0.18,
        "short_tp": 0.50,
        "long_tp": 0.40,
        "partial_tp_1": 0.15,
        "use_partial_tp_2": True,
        "short_max_hold": 48,
    },
    "v3": {  # run30 早期锁利 (v3分段止盈 A/B验证)
        "short_threshold": 35,
        "long_threshold": 30,
        "short_sl": -0.18,
        "short_tp": 0.50,
        "long_tp": 0.40,
        "partial_tp_1": 0.12,    # 更早锁利 (+12%即触发)
        "use_partial_tp_2": True,
        "short_max_hold": 48,
    },
    "v4": {  # v5.1: P0前视修复 + 参数重优化 (run#190/221)
        "short_threshold": 40,   # 保持 (扫描验证 st=40 最稳健)
        "long_threshold": 30,    # 保持 (v2一致)
        "short_sl": -0.20,       # v5.1: 放宽止损, P0修复后需更大呼吸空间 (v4旧:-0.16)
        "short_tp": 0.60,        # 更大目标利润 (v2:0.50)
        "long_sl": -0.10,        # 多头略放宽 (v2:-0.08)
        "long_tp": 0.40,         # 保持 (v2一致)
        "partial_tp_1": 0.15,    # 保持v2 (v3 early由use_partial_tp_v3控制)
        "use_partial_tp_2": True,
        "short_max_hold": 48,    # 保持
        "short_trail": 0.20,     # v5.1: 放宽追踪, 让TP1后剩余仓位充分发展 (v4旧:0.15)
        "long_trail": 0.12,      # 多头追踪 (v2:0.20)
        "trail_pullback": 0.50,  # 稍微收紧 (v2:0.60)
    },
}
_ACTIVE_VERSION = os.environ.get("STRATEGY_VERSION", "v4")


def get_strategy_version() -> str:
    """返回当前生效的策略参数版本"""
    return _ACTIVE_VERSION


def _resolve_param(field_name: str, v2_default):
    """根据 STRATEGY_VERSION 环境变量解析参数值"""
    ver = _ACTIVE_VERSION
    params = STRATEGY_PARAM_VERSIONS.get(ver, STRATEGY_PARAM_VERSIONS["v2"])
    return params.get(field_name, v2_default)


@dataclass
class StrategyConfig:
    """策略参数配置 - 从优化结果加载

    版本切换: 设置环境变量 STRATEGY_VERSION=v1/v2/v3/v4
    默认: v4 (run31 — param_sweep 系统性优化)
    """
    symbol: str = "ETHUSDT"
    timeframe: str = "1h"
    # 信号阈值
    sell_threshold: float = 18
    buy_threshold: float = 25
    short_threshold: float = field(default_factory=lambda: _resolve_param("short_threshold", 35))
    long_threshold: float = field(default_factory=lambda: _resolve_param("long_threshold", 30))
    close_short_bs: float = 40
    close_long_ss: float = 40
    sell_pct: float = 0.55
    # 止损止盈
    short_sl: float = field(default_factory=lambda: _resolve_param("short_sl", -0.18))
    short_tp: float = field(default_factory=lambda: _resolve_param("short_tp", 0.50))
    long_sl: float = field(default_factory=lambda: _resolve_param("long_sl", -0.08))
    long_tp: float = field(default_factory=lambda: _resolve_param("long_tp", 0.40))
    short_trail: float = field(default_factory=lambda: _resolve_param("short_trail", 0.25))
    long_trail: float = field(default_factory=lambda: _resolve_param("long_trail", 0.20))
    trail_pullback: float = field(default_factory=lambda: _resolve_param("trail_pullback", 0.60))
    # 部分止盈
    use_partial_tp: bool = True
    partial_tp_1: float = field(default_factory=lambda: _resolve_param("partial_tp_1", 0.15))
    partial_tp_1_pct: float = 0.30
    use_partial_tp_2: bool = field(default_factory=lambda: _resolve_param("use_partial_tp_2", True))
    partial_tp_2: float = 0.50
    partial_tp_2_pct: float = 0.30
    use_atr_sl: bool = False            # ATR自适应止损(A/B证明对ETH无效,保留代码供其他币种)
    atr_sl_mult: float = 2.5            # ATR倍数: 2.5倍ATR作为止损距离
    atr_sl_floor: float = -0.25         # ATR止损下限(最宽, 高波动时)
    atr_sl_ceil: float = -0.15          # ATR止损上限(最窄, 低波动时)
    # [已移除] use_short_suppress: A/B+param_sweep双重验证完全零效果(SS>=42已覆盖)
    # Regime-aware 做空抑制: 在 trend/low_vol_trend regime 中提高做空门槛
    # 数据支持: run#32 regime分析显示 73% 止损亏损来自这两个 regime
    # Codex: 实盘口径主线; 第八轮 E 变体 LVT+35 最优, 不改为 hard_stop=-0.22
    hard_stop_loss: float = -0.28          # 硬断路器 -28%(研究口径 C 变体 -0.22 仅限 run#43 轨)
    use_regime_short_gate: bool = True    # 启用 LVT 做空门控
    regime_short_gate_add: float = 35     # 实盘口径 E 变体: +35 (run#44 基线 -87k → -45k)
    regime_short_gate_regimes: str = 'low_vol_trend'  # 仅 low_vol_trend
    # Codex run#62 主线: confirm35x3 + block high_vol
    use_spot_sell_confirm: bool = True     # run#62: 启用高分确认过滤
    spot_sell_confirm_ss: float = 35       # run#62: SS>=35 需额外确认 (原 100)
    spot_sell_confirm_min: int = 3         # 至少满足 3 项确认条件
    use_spot_sell_cap: bool = False        # 不启用 (run#67 验证退化)
    spot_sell_max_pct: float = 0.30        # 单笔卖出比例上限
    # run#85: 高波动+趋势段禁止 SPOT_SELL (run#62: 仅 high_vol; 趋势段误卖实锤)
    spot_sell_regime_block: str = 'high_vol,trend'
    # 趋势保护（现货底仓）
    use_trend_enhance: bool = True
    trend_floor_ratio: float = 0.50
    min_base_eth_ratio: float = 0.0
    # True: 仅 trend 引擎启用趋势保护（兼容旧口径）; False: 解耦（仅看EMA趋势）
    trend_enhance_engine_gate: bool = False
    # 仓位管理
    leverage: int = 5
    max_lev: int = 5
    margin_use: float = 0.70
    single_pct: float = 0.20
    total_pct: float = 0.50
    # 冷却
    cooldown: int = 4
    spot_cooldown: int = 12
    # NoTP 提前退出（长短独立 + regime 白名单）
    # 兼容旧参数: no_tp_exit_bars / no_tp_exit_min_pnl / no_tp_exit_regimes
    no_tp_exit_bars: int = 0           # 0=关闭 (旧参数, 保留兼容)
    no_tp_exit_min_pnl: float = 0.03   # 旧参数, 保留兼容
    no_tp_exit_regimes: str = 'neutral'  # 旧参数, 保留兼容
    no_tp_exit_short_bars: int = 0
    no_tp_exit_short_min_pnl: float = 0.03
    no_tp_exit_short_loss_floor: float = -0.03
    no_tp_exit_short_regimes: str = 'neutral'
    no_tp_exit_long_bars: int = 0
    no_tp_exit_long_min_pnl: float = 0.03
    no_tp_exit_long_loss_floor: float = -0.03
    no_tp_exit_long_regimes: str = 'neutral'
    # 融合模式
    fusion_mode: str = "c6_veto_4"
    veto_threshold: float = 25
    kdj_bonus: float = 0.09
    kdj_weight: float = 0.15
    div_weight: float = 0.55
    kdj_strong_mult: float = 1.25
    kdj_normal_mult: float = 1.12
    kdj_reverse_mult: float = 0.70
    kdj_gate_threshold: float = 10
    veto_dampen: float = 0.30
    bb_bonus: float = 0.10
    vp_bonus: float = 0.08
    cs_bonus: float = 0.06
    # 数据参数
    lookback_days: int = 60
    # 最大持仓K线数
    short_max_hold: int = field(default_factory=lambda: _resolve_param("short_max_hold", 48))
    long_max_hold: int = 72

    # ── 多周期联合决策 ──
    use_multi_tf: bool = True                    # 是否启用多周期共识
    decision_timeframes: List[str] = field(      # 参与决策的时间框架
        default_factory=lambda: ['15m', '1h', '4h', '24h']
    )
    decision_timeframes_fallback: List[str] = field(  # 回退时间框架(可配置)
        default_factory=lambda: ['15m', '30m', '1h', '4h', '8h', '24h']
    )
    consensus_min_strength: int = 40             # 共识最低强度才可开仓 (0-100)
    coverage_min: float = 0.5                    # 多周期覆盖率下限
    consensus_position_scale: bool = True        # 是否按共识强度缩放仓位

    # ── 微结构增强(资金费率/基差/OI代理) ──
    use_microstructure: bool = True
    micro_lookback_bars: int = 48
    micro_imbalance_threshold: float = 0.08
    micro_oi_trend_z: float = 0.8
    micro_basis_extreme_z: float = 1.2
    micro_basis_crowded_z: float = 2.2
    micro_funding_extreme: float = 0.0006
    micro_participation_trend: float = 1.15
    micro_funding_proxy_mult: float = 0.35
    micro_score_boost: float = 0.08
    micro_score_dampen: float = 0.10
    micro_margin_mult_step: float = 0.06
    micro_mode_override: bool = True

    # ── 双引擎(趋势/反转) ──
    use_dual_engine: bool = True
    entry_dominance_ratio: float = 1.5
    trend_engine_entry_mult: float = 0.95
    trend_engine_exit_mult: float = 1.05
    trend_engine_hold_mult: float = 1.35
    trend_engine_risk_mult: float = 1.10
    trend_engine_dominance_ratio: float = 1.35
    reversion_engine_entry_mult: float = 1.12
    reversion_engine_exit_mult: float = 0.90
    reversion_engine_hold_mult: float = 0.70
    reversion_engine_risk_mult: float = 0.75
    reversion_engine_dominance_ratio: float = 1.75

    # ── 波动目标仓位 ──
    use_vol_target: bool = True
    vol_target_annual: float = 0.85
    vol_target_lookback_bars: int = 48
    vol_target_min_scale: float = 0.45
    vol_target_max_scale: float = 1.35

    # ── Regime 可调阈值 (供参数扫描) ──
    regime_vol_high: float = 0.020      # 高波动判定阈值
    regime_vol_low: float = 0.007       # 低波动判定阈值
    regime_trend_strong: float = 0.015  # 强趋势判定阈值
    regime_trend_weak: float = 0.006    # 弱趋势判定阈值
    regime_atr_high: float = 0.018      # ATR高波动阈值
    regime_lookback_bars: int = 48      # Regime 回望窗口
    regime_atr_bars: int = 14           # Regime ATR 计算周期

    # ── 分段止盈增强(v3 早期锁利) ──
    # A/B 测试: TP1+12%/TP2+25% 比 TP1+15%/TP2+50% 收益+205pp, 回撤更优
    partial_tp_1_early: float = 0.12    # v3触发TP1 (+12% vs 默认+15%)
    partial_tp_2_early: float = 0.25    # v3触发TP2 (+25% vs 默认+50%)
    use_partial_tp_v3: bool = True      # 启用v3分段止盈(早期锁利, A/B已验证)

    @classmethod
    def from_optimize_result(cls, filepath: str, timeframe: str = None) -> 'StrategyConfig':
        """从优化结果 JSON 加载最优配置"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        config = cls()

        # 尝试从 global_best 加载
        if 'global_best' in data and 'config' in data['global_best']:
            best_config = data['global_best']['config']
            if timeframe is None and 'tf' in data['global_best']:
                timeframe = data['global_best']['tf']
        # 或者按 timeframe 查找
        elif 'results' in data and timeframe:
            for r in data['results']:
                if r.get('tf') == timeframe:
                    best_config = r.get('config', {})
                    break
            else:
                raise ValueError(f"未找到 timeframe={timeframe} 的优化结果")
        else:
            best_config = data

        # 映射配置字段
        aliases = {
            "lev": "leverage",
            "decision_tfs": "decision_timeframes",
        }
        for key, value in best_config.items():
            target_key = aliases.get(key, key)
            if hasattr(config, target_key):
                setattr(config, target_key, value)

        if isinstance(config.decision_timeframes, str):
            config.decision_timeframes = [
                x.strip() for x in config.decision_timeframes.split(",") if x.strip()
            ]

        if timeframe:
            config.timeframe = timeframe

        return config


@dataclass
class LiveTradingConfig:
    """完整实盘配置"""
    phase: TradingPhase = TradingPhase.PAPER
    api: APIConfig = field(default_factory=APIConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    # 运行参数
    execute_trades: bool = False
    initial_capital: float = 100000
    log_dir: str = "logs/live"
    data_dir: str = "data/live"
    state_file: str = "data/live/trading_state.json"

    # 监控间隔
    signal_check_interval_sec: int = 10   # 检查信号频率
    balance_log_interval_sec: int = 300   # 余额日志频率 (5min)
    health_check_interval_sec: int = 60   # 健康检查频率

    def __post_init__(self):
        """根据 phase 自动设置默认值"""
        phase_cfg = PHASE_CONFIGS.get(self.phase, {})
        if phase_cfg:
            self.execute_trades = phase_cfg.get("execute_trades", False)
            self.risk = phase_cfg.get("risk", self.risk)
            if self.initial_capital == 100000:  # 未手动设置
                self.initial_capital = phase_cfg.get("initial_capital", 100000)

    @classmethod
    def load(cls, config_path: str = "live_trading_config.json") -> 'LiveTradingConfig':
        """从 JSON 文件加载配置"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r') as f:
            data = json.load(f)

        config = cls()
        config.phase = TradingPhase(data.get("phase", "paper"))

        # API
        api_data = data.get("api", {})
        config.api = APIConfig(**{k: v for k, v in api_data.items()
                                  if k in APIConfig.__dataclass_fields__})

        # Telegram
        tg_data = data.get("telegram", {})
        config.telegram = TelegramConfig(**{k: v for k, v in tg_data.items()
                                            if k in TelegramConfig.__dataclass_fields__})

        # Risk (阶段默认 + 覆盖)
        phase_cfg = PHASE_CONFIGS.get(config.phase, {})
        config.risk = phase_cfg.get("risk", RiskConfig())
        risk_data = data.get("risk", {})
        for k, v in risk_data.items():
            if hasattr(config.risk, k):
                setattr(config.risk, k, v)

        # Strategy
        strategy_data = data.get("strategy", {})
        if "optimize_result_file" in strategy_data:
            tf = strategy_data.get("timeframe", None)
            config.strategy = StrategyConfig.from_optimize_result(
                strategy_data["optimize_result_file"], tf
            )
        else:
            config.strategy = StrategyConfig(**{k: v for k, v in strategy_data.items()
                                                if k in StrategyConfig.__dataclass_fields__})

        # 运行参数
        config.execute_trades = data.get("execute_trades",
                                         phase_cfg.get("execute_trades", False))
        config.initial_capital = data.get("initial_capital",
                                          phase_cfg.get("initial_capital", 100000))
        config.log_dir = data.get("log_dir", config.log_dir)
        config.data_dir = data.get("data_dir", config.data_dir)

        # 安全校验
        config._validate()
        return config

    def _validate(self):
        """安全校验"""
        phase_cfg = PHASE_CONFIGS.get(self.phase, {})
        max_lev = phase_cfg.get("risk", RiskConfig()).max_leverage

        # 杠杆不能超过阶段限制
        if self.strategy.leverage > max_lev:
            print(f"[SAFETY] 杠杆 {self.strategy.leverage}x 超过阶段限制 "
                  f"{max_lev}x，已强制调整")
            self.strategy.leverage = max_lev

        # Phase 3 严格限制
        if self.phase == TradingPhase.SMALL_LIVE:
            if self.initial_capital > 2000:
                print(f"[SAFETY] Phase 3 初始资金 ${self.initial_capital} "
                      f"超过 $2000 限制，已调整")
                self.initial_capital = 2000
            if self.strategy.leverage > 2:
                self.strategy.leverage = 2

        # Paper 模式不能执行真实交易
        if self.phase == TradingPhase.PAPER:
            self.execute_trades = False

    def save_template(self, path: str = "live_trading_config.json"):
        """保存配置模板"""
        template = {
            "phase": self.phase.value,
            "api": {
                "api_key": "YOUR_API_KEY",
                "api_secret": "YOUR_API_SECRET",
                "testnet_api_key": "YOUR_TESTNET_API_KEY",
                "testnet_api_secret": "YOUR_TESTNET_API_SECRET",
            },
            "telegram": {
                "enabled": False,
                "bot_token": "YOUR_BOT_TOKEN",
                "chat_id": "YOUR_CHAT_ID",
                "notify_trades": True,
                "notify_risk": True,
                "notify_daily_summary": True,
            },
            "strategy": {
                "optimize_result_file": "optimize_six_book_result.json",
                "symbol": "ETHUSDT",
                "timeframe": "1h",
                "use_multi_tf": True,
                "decision_timeframes": ["15m", "1h", "4h", "24h"],
                "decision_timeframes_fallback": ["15m", "30m", "1h", "4h", "8h", "24h"],
                "consensus_min_strength": 40,
                "coverage_min": 0.5,
                "use_microstructure": True,
                "use_dual_engine": True,
                "use_vol_target": True,
                "vol_target_annual": 0.85,
            },
            "risk": {
                "max_leverage": 2,
                "max_daily_loss_pct": 0.05,
                "max_consecutive_losses": 5,
            },
            "initial_capital": 500,
            "log_dir": "logs/live",
            "data_dir": "data/live",
        }
        with open(path, 'w') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        print(f"配置模板已保存: {path}")


def create_default_config(phase: TradingPhase = TradingPhase.PAPER) -> LiveTradingConfig:
    """创建指定阶段的默认配置"""
    config = LiveTradingConfig(phase=phase)
    config.__post_init__()
    return config
