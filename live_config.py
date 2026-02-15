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
    "v4": {  # v8.0 (B3+P13): v7.0 + 连续追踪止盈
        # ── 基础参数 (继承 v6.0) ──
        "short_threshold": 40,
        "long_threshold": 25,
        "short_sl": -0.20,
        "short_tp": 0.60,
        "long_sl": -0.10,
        "long_tp": 0.40,
        "partial_tp_1": 0.15,
        "use_partial_tp_2": True,
        "short_max_hold": 48,
        "short_trail": 0.19,
        "long_trail": 0.12,
        "trail_pullback": 0.50,
        # ── v7.0 B3: neutral short 入场质量修正 ──
        "cooldown": 6,                          # B3: 冷却 4→6
        "regime_short_threshold": "neutral:60",  # B3: neutral 空单门槛 45→60
        "short_conflict_regimes": "trend,high_vol,neutral",  # B3: 冲突折扣扩展到 neutral
        "neutral_struct_discount_0": 0.0,        # B3: 0本→禁止
        "neutral_struct_discount_1": 0.05,       # B3: 1本→5%
        "neutral_struct_discount_2": 0.15,       # B3: 2本→15%
        "neutral_struct_discount_3": 0.50,       # B3: 3本→50%
        # ── v8.0 P13: 连续追踪止盈 (替代离散门槛) ──
        # P13 A/B: IS PF 0.91→1.08, OOS WR +3.9%, OOS PF 2.33→2.51
        # P15 Walk-Forward: 3/6窗口盈利, 均值Ret +9.6%/Q, WR>54%
        "use_continuous_trail": True,
        "continuous_trail_start_pnl": 0.05,      # 利润>=5%开始追踪
        "continuous_trail_max_pb": 0.60,          # 低利润时最宽回撤容忍 60%
        "continuous_trail_min_pb": 0.30,          # 高利润时最紧回撤容忍 30%
    },
    "v5": {  # v8.1 (P20): v8.0 + 空头追踪收紧
        # ── 基础参数 (继承 v8.0) ──
        "short_threshold": 40,
        "long_threshold": 25,
        "short_sl": -0.20,
        "short_tp": 0.60,
        "long_sl": -0.10,
        "long_tp": 0.40,
        "partial_tp_1": 0.15,
        "use_partial_tp_2": True,
        "short_max_hold": 48,
        "short_trail": 0.19,
        "long_trail": 0.12,
        "trail_pullback": 0.50,
        # ── v7.0 B3: neutral short 入场质量修正 ──
        "cooldown": 6,
        "regime_short_threshold": "neutral:60",
        "short_conflict_regimes": "trend,high_vol,neutral",
        "neutral_struct_discount_0": 0.0,
        "neutral_struct_discount_1": 0.05,
        "neutral_struct_discount_2": 0.15,
        "neutral_struct_discount_3": 0.50,
        # ── v8.0 P13: 连续追踪止盈 ──
        "use_continuous_trail": True,
        "continuous_trail_start_pnl": 0.05,
        "continuous_trail_max_pb": 0.60,
        "continuous_trail_min_pb": 0.30,
        # ── v8.1 P20: 空头追踪收紧 ──
        # P20 A/B: IS +5.5%, PF 0.92→0.96, OOS 无回退
        # 空头追踪回撤容忍从 60%→40%, 更早锁定利润
        "continuous_trail_max_pb_short": 0.40,
    },
}
_ACTIVE_VERSION = os.environ.get("STRATEGY_VERSION", "v5")


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
    # neutral 体制分层 SPOT_SELL: A/B(run#496-499) 显示主样本负贡献，默认关闭
    use_neutral_spot_sell_layer: bool = False
    neutral_spot_sell_confirm_thr: float = 10.0
    neutral_spot_sell_min_confirms_any: int = 2
    neutral_spot_sell_strong_confirms: int = 4
    neutral_spot_sell_full_ss_min: float = 70.0
    neutral_spot_sell_weak_ss_min: float = 55.0
    neutral_spot_sell_weak_pct_cap: float = 0.15
    neutral_spot_sell_block_ss_min: float = 70.0
    use_spot_sell_cap: bool = False        # 不启用 (run#67 验证退化)
    spot_sell_max_pct: float = 0.30        # 单笔卖出比例上限
    # run#85: 高波动+趋势段禁止 SPOT_SELL (run#62: 仅 high_vol; 趋势段误卖实锤)
    spot_sell_regime_block: str = 'high_vol,trend'
    # 停滞再入场: A/B(run#496-499) 触发极少且主样本负贡献，默认关闭
    use_stagnation_reentry: bool = False
    stagnation_reentry_days: float = 10.0
    stagnation_reentry_regimes: str = 'trend,low_vol_trend'
    stagnation_reentry_min_spot_ratio: float = 0.30
    stagnation_reentry_buy_pct: float = 0.20
    stagnation_reentry_min_usdt: float = 500.0
    stagnation_reentry_cooldown_days: float = 3.0
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
    cooldown: int = 6  # v7.0 B3: 4→6, 近似 ghost cooldown (IS+0.9%, OOS+5.6%)
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
    # 止损后冷却倍数（v5.2: 支持配置化，默认保持原始值4x）
    short_sl_cd_mult: int = 4          # 空头止损后 cooldown*4
    long_sl_cd_mult: int = 4           # 多头止损后 cooldown*4
    # 反向平仓最小持仓bars。用于抑制短周期来回反手造成的手续费拖累。
    # 2025-01~2026-01 A/B: 8 bars 在收益/组合PF上优于0/2/6/12。
    reverse_min_hold_short: int = 8
    reverse_min_hold_long: int = 8
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

    # ── S1: Regime做空门槛覆盖 ──
    # v7.0 B3: neutral 门槛提高到 60 (P4实证neutral short无alpha, B3 OOS +5.6%)
    # 回退 B1b: 设为 'neutral:999' 可完全禁止 neutral 空单
    # 格式 "regime:threshold,..." 例 "neutral:60" → neutral中SS须>=60才开空
    regime_short_threshold: str = 'neutral:60'

    # ── S1.5: neutral 体制信号质量门控（结构性过滤，减少震荡假突破） ──
    # 逻辑: 方向强度 + 共识链条 + 大周期不冲突 + 信号连续性
    # 当前结论: 样本内提升明显，但 2024 OOS 退化，暂保持默认关闭（实验开关）
    use_neutral_quality_gate: bool = False
    neutral_min_score_gap: float = 12.0
    neutral_min_strength: float = 45.0
    neutral_min_streak: int = 2
    neutral_nochain_extra_gap: float = 20.0
    neutral_large_conflict_ratio: float = 1.10

    # ── 信号置信度学习层（实验） ──
    # 默认关闭：先用于回测复盘与策略迭代，确认稳健后再启用实盘
    use_confidence_learning: bool = False
    confidence_min_raw: float = 0.42
    confidence_min_posterior: float = 0.47
    confidence_min_samples: int = 8
    confidence_block_after_samples: int = 30
    confidence_threshold_gain: float = 0.35
    confidence_threshold_min_mult: float = 0.88
    confidence_threshold_max_mult: float = 1.22
    confidence_prior_alpha: float = 2.0
    confidence_prior_beta: float = 2.0
    confidence_win_pnl_r: float = 0.03
    confidence_loss_pnl_r: float = -0.03
    # 回测日志输出（不影响实盘执行）
    print_signal_features: bool = True
    signal_replay_top_n: int = 10

    # ── Neutral 六书共识门控 ──────────────────────────────────────────
    # 核心: neutral 体制中 divergence 占融合权重70%但判别力≈0,
    #       CS(d=0.40)/KDJ(d=0.42) 才是真正有效确认书。
    #       要求多本书独立确认方向, 而非依赖单一 SS 阈值。
    use_neutral_book_consensus: bool = False      # 二元门控 (已弃用, 用渐进折扣替代)
    neutral_book_sell_threshold: float = 10.0     # 卖方书"活跃"阈值
    neutral_book_buy_threshold: float = 10.0      # 买方书"活跃"阈值
    neutral_book_min_confirms: int = 2            # 最少确认书数 (5本结构书)
    neutral_book_max_conflicts: int = 4           # 最大允许冲突书数
    neutral_book_cs_kdj_threshold_adj: float = 0.0  # CS+KDJ双确认时阈值调整

    # ── Neutral 结构质量渐进折扣 ──
    # 核心改进: 不阻止交易(避免蝴蝶效应), 而是根据结构书独立确认数量
    # 渐进折扣 SS/BS, 让弱共识信号自然被现有阈值过滤。
    # 5本结构书(CS/KDJ/MA/BB/VP), 排除在neutral中无判别力的divergence。
    use_neutral_structural_discount: bool = True
    neutral_struct_activity_thr: float = 10.0     # 书"活跃"阈值
    neutral_struct_discount_0: float = 0.0        # v7.0 B3: 0本确认→直接不做 (P4: confirms与WR无单调关系)
    neutral_struct_discount_1: float = 0.05       # v7.0 B3: 1本确认→5%仓位
    neutral_struct_discount_2: float = 0.15       # v7.0 B3: 2本确认→15%仓位
    neutral_struct_discount_3: float = 0.50       # v7.0 B3: 3本确认→50%仓位
    neutral_struct_discount_4plus: float = 1.00   # 4-5本: 极强共识→全额
    # 结构折扣生效的 regime（仅 neutral: trend/high_vol扩展测试 ret下降19% →回退）
    structural_discount_short_regimes: str = 'neutral'
    structural_discount_long_regimes: str = 'neutral'

    # ── neutral short 结构确认器（减少震荡期错空） ──
    use_neutral_short_structure_gate: bool = False
    neutral_short_structure_large_tfs: str = '4h,24h'
    neutral_short_structure_need_min_tfs: int = 1
    neutral_short_structure_min_agree: int = 1
    neutral_short_structure_div_gap: float = 8.0
    neutral_short_structure_ma_gap: float = 5.0
    neutral_short_structure_vp_gap: float = 4.0
    neutral_short_structure_fail_open: bool = True
    neutral_short_structure_soften_weak: bool = True
    neutral_short_structure_soften_mult: float = 1.10

    # ── 趋势/高波动下空单冲突软折扣（策略层） ──
    # 目标: 避免“卖出极强但买方div也很强”的高冲突做空放大尾部止损。
    # 默认关闭，仅用于A/B验证。
    use_short_conflict_soft_discount: bool = True  # v7.0 B3: 扩展到 neutral
    short_conflict_regimes: str = 'trend,high_vol,neutral'  # v7.0 B3: +neutral
    short_conflict_div_buy_min: float = 50.0
    short_conflict_ma_sell_min: float = 12.0
    short_conflict_discount_mult: float = 0.60      # Codex验证最优值 (0.50太激进)

    # ── P7: trend/high_vol short 大周期方向门控 ──
    # 当 4h+24h 加权方向看多时, 在 trend/high_vol regime 中阻止开空
    # P0 数据: OOS trend空WR=46.7%(PF=0.84), high_vol空WR=33.3%(PF=0.47)
    use_trend_short_large_tf_gate: bool = False
    trend_short_large_tf_gate_regimes: str = 'trend,high_vol'
    trend_short_large_tf_bs_ratio: float = 1.0  # large_bs > large_ss * ratio → block

    # ── P8: trend/high_vol short 结构确认硬门槛 ──
    # 要求至少 N 本结构书确认才允许开空 (与结构折扣独立, 这是硬阻止)
    use_trend_short_min_confirms: bool = False
    trend_short_min_confirms_regimes: str = 'trend,high_vol'
    trend_short_min_confirms_n: int = 3

    # ── P6: Ghost cooldown ──
    # 被门控过滤的强信号也触发部分冷却, 防止连续尝试低质量入场
    use_ghost_cooldown: bool = False
    ghost_cooldown_bars: int = 3

    # ── P10: Fast-fail 快速亏损退出 ──
    # 如果 trend/high_vol 空单在前N个bar就亏损超阈值, 提前离场
    # 依据: P0 显示 trend/high_vol 止损在 2-11 bars 内触发
    use_fast_fail_exit: bool = False
    fast_fail_max_bars: int = 3
    fast_fail_loss_threshold: float = -0.05
    fast_fail_regimes: str = 'trend,high_vol'

    # ── P9: Regime-adaptive SS/BS 后置调权 ──
    # neutral 中 DIV 判别力为负(d=-0.64), 做空 SS 不可靠 → 衰减
    # 做多 BS 在 neutral 中可靠(WR=85%) → 增强
    use_regime_adaptive_reweight: bool = False
    regime_neutral_ss_dampen: float = 0.85      # neutral 中 SS 打85折
    regime_neutral_bs_boost: float = 1.10       # neutral 中 BS 增10%

    # ── P12: 动态 Regime 阈值 ──
    # 替代固定 vol/trend 阈值, 使用滚动窗口百分位数
    use_dynamic_regime_thresholds: bool = False
    dynamic_regime_lookback_bars: int = 2160    # 90天 * 24h = 2160 bars
    dynamic_regime_vol_quantile: float = 0.80   # 波动率 80th percentile
    dynamic_regime_trend_quantile: float = 0.80 # 趋势强度 80th percentile

    # ── P13: 追踪止盈连续化 ──
    # 替代离散门槛触发, 改为连续动态回撤容忍
    use_continuous_trail: bool = False
    continuous_trail_start_pnl: float = 0.05     # 利润>=5%开始追踪
    continuous_trail_max_pb: float = 0.60         # 低利润时最宽回撤容忍 (多单)
    continuous_trail_min_pb: float = 0.30         # 高利润时最紧回撤容忍
    continuous_trail_max_pb_short: float = 0.60   # P20: 空单最宽回撤容忍 (默认同多, 可收紧到0.40)

    # ── P18: Regime-Adaptive 六维融合权重 ──
    # 核心改造: 在回测主循环中根据 regime 动态重新融合六书分数
    # 解决: neutral 中 DIV alpha 为负 (d=-0.64) 但占基数 70% 的结构性问题
    # neutral: DIV 大幅降权(25%), CS/KDJ 大幅升权(bonus 15%)
    # trend: DIV 保留高权重(60%), 背离在趋势末端有效
    # high_vol: VP 升权(12%), 量价在高波中更有效
    use_regime_adaptive_fusion: bool = False
    # 说明: 信号层会将 div_w/ma_w 归一化后用于基数融合，因此这里直接填“目标占比”
    regime_trend_div_w: float = 0.60
    regime_trend_ma_w: float = 0.40
    regime_low_vol_trend_div_w: float = 0.60
    regime_low_vol_trend_ma_w: float = 0.40
    regime_neutral_div_w: float = 0.25
    regime_neutral_ma_w: float = 0.75
    regime_high_vol_div_w: float = 0.45
    regime_high_vol_ma_w: float = 0.55
    regime_high_vol_choppy_div_w: float = 0.30
    regime_high_vol_choppy_ma_w: float = 0.70

    # ── P24: Regime-Adaptive 止损 ──
    # 空单止损按 regime 差异化: neutral 收紧(错了快认输), trend 保持(给空间)
    # Claude 独创建议: -20% 对 neutral 空单过宽, crypto 空头挤压效应显著
    use_regime_adaptive_sl: bool = False
    regime_neutral_short_sl: float = -0.12       # neutral 空单止损 -12%
    regime_trend_short_sl: float = -0.20         # trend 空单止损 -20% (保持)
    regime_low_vol_trend_short_sl: float = -0.20
    regime_high_vol_short_sl: float = -0.15      # high_vol 空单止损 -15%
    regime_high_vol_choppy_short_sl: float = -0.15

    # ── V9: 回测口径真实化 (Perp) ──
    # 强平检测优先使用 mark 价格序列（若本地K线包含相关列）
    use_mark_price_for_liquidation: bool = False
    mark_price_col: str = 'mark_price'
    mark_high_col: str = 'mark_high'
    mark_low_col: str = 'mark_low'
    # 资金费率优先使用真实 funding_rate 列（若存在），否则回退旧模型
    use_real_funding_rate: bool = False
    funding_rate_col: str = 'funding_rate'
    funding_interval_hours: float = 8.0
    funding_interval_hours_col: str = 'funding_interval_hours'

    # ── V9: Leg 级风险预算（regime × direction） ──
    # 默认关闭；开启后只影响开仓保证金分配，不改变信号本身
    use_leg_risk_budget: bool = False
    risk_budget_neutral_long: float = 1.00
    risk_budget_neutral_short: float = 1.00
    risk_budget_trend_long: float = 1.00
    risk_budget_trend_short: float = 1.00
    risk_budget_low_vol_trend_long: float = 1.00
    risk_budget_low_vol_trend_short: float = 1.00
    risk_budget_high_vol_long: float = 1.00
    risk_budget_high_vol_short: float = 1.00
    risk_budget_high_vol_choppy_long: float = 1.00
    risk_budget_high_vol_choppy_short: float = 1.00

    # ── P21: Risk-per-trade (R) 仓位模型 ──
    # GPT Pro 最高价值建议: 用固定风险百分比 + 止损距离反推仓位, 消灭"少数大亏单主导"
    # 替代现有 margin_use * available_margin 的固定比例仓位
    # risk_per_trade: 每笔最大风险 = equity × R%
    # 仓位 = risk_amount / (stop_distance_pct × entry_price), 再 clip 到 margin 上限
    # 优势: 尾部可控, 几何增长对大亏极敏感, 控制单笔尾部后可安全加大风险预算
    use_risk_per_trade: bool = False
    risk_per_trade_pct: float = 0.015    # 每笔最大风险占 equity 的 1.5%
    risk_stop_mode: str = 'atr'          # 止损距离计算方式: 'atr' 或 'fixed'
    risk_atr_mult_short: float = 2.5     # 空单: 止损 = entry ± ATR × 倍数
    risk_atr_mult_long: float = 2.0      # 多单: 止损 = entry ± ATR × 倍数
    risk_fixed_stop_short: float = 0.04  # 空单固定止损距离 4% (当 mode='fixed')
    risk_fixed_stop_long: float = 0.03   # 多单固定止损距离 3% (当 mode='fixed')
    risk_max_margin_pct: float = 0.50    # 仓位上限 = equity × 50% (防极端)
    risk_min_margin_pct: float = 0.05    # 仓位下限 = equity × 5% (避免过小)

    # ── P23: 加权结构确认 (基于 Cohen's d 先验) ──
    # 替代简单计数: sum(1 if feat>thr) → sum(alpha_weight * strength) - penalty
    # 权重基于 P16 的 Cohen's d 分析: MA/CS 判别力最强, KDJ 中等, BB/VP 弱
    use_weighted_confirms: bool = False
    wc_ma_sell_w: float = 1.5     # MA 卖方权重 (Cohen's d 较大)
    wc_cs_sell_w: float = 1.4     # CS 卖方权重
    wc_kdj_sell_w: float = 1.0    # KDJ 卖方权重
    wc_vp_sell_w: float = 0.6     # VP 卖方权重 (较弱)
    wc_bb_sell_w: float = 0.3     # BB 卖方权重 (偏负/无效)
    wc_ma_buy_w: float = 1.5      # MA 买方权重
    wc_cs_buy_w: float = 1.4      # CS 买方权重
    wc_kdj_buy_w: float = 1.0     # KDJ 买方权重
    wc_vp_buy_w: float = 0.6      # VP 买方权重
    wc_bb_buy_w: float = 0.3      # BB 买方权重
    wc_min_score: float = 2.0     # 加权确认最低分 (替代 min_confirms)
    wc_conflict_penalty_scale: float = 0.5  # 冲突方向惩罚系数
    wc_struct_discount_thr_0: float = 0.5   # 加权分 < 0.5 → 最大折扣
    wc_struct_discount_thr_1: float = 1.5   # 加权分 < 1.5 → 折扣1
    wc_struct_discount_thr_2: float = 2.5   # 加权分 < 2.5 → 折扣2
    wc_struct_discount_thr_3: float = 3.5   # 加权分 < 3.5 → 折扣3

    # ── 多头冲突软折扣（neutral/low_vol_trend） ──
    # 目标: 买入信号中若卖方divergence过强，先减仓再观察，降低中性体制假突破亏损。
    # 默认关闭，仅用于A/B验证。
    use_long_conflict_soft_discount: bool = False
    long_conflict_regimes: str = 'neutral,low_vol_trend'
    long_conflict_div_sell_min: float = 50.0
    long_conflict_ma_buy_min: float = 12.0
    long_conflict_discount_mult: float = 0.50

    # ── Long 高置信错单候选门控（仅A/B验证，默认关闭） ──
    # 候选A: low_vol_trend 且置信度过高（历史错单集中）时，抑制开多
    use_long_high_conf_gate_a: bool = False
    long_high_conf_gate_a_conf_min: float = 0.85
    long_high_conf_gate_a_regime: str = 'low_vol_trend'
    # 候选B: neutral + 高置信 + VP买方过热（历史错单集中）时，抑制开多
    use_long_high_conf_gate_b: bool = False    # OOS正向但主区间回撤，先保留为可选开关
    long_high_conf_gate_b_conf_min: float = 0.90
    long_high_conf_gate_b_regime: str = 'neutral'
    long_high_conf_gate_b_vp_buy_min: float = 30.0

    # ── 空单逆势防守退出（结构化风控） ──
    # 目标: 在空单亏损扩张且多头共识抬升时提前离场，降低 -20% 类尾部止损频率
    # 默认关闭：先用于A/B验证，不直接改变现有基线
    use_short_adverse_exit: bool = False
    short_adverse_min_bars: int = 8
    short_adverse_loss_r: float = -0.08
    short_adverse_bs: float = 55.0
    short_adverse_bs_dom_ratio: float = 0.85
    short_adverse_ss_cap: float = 95.0
    short_adverse_require_bs_dom: bool = False
    short_adverse_ma_conflict_gap: float = 8.0
    short_adverse_conflict_thr: float = 10.0
    short_adverse_min_conflicts: int = 3
    short_adverse_need_cs_kdj: bool = True
    short_adverse_large_bs_min: float = 35.0
    short_adverse_large_ratio: float = 0.55
    short_adverse_need_chain_long: bool = True
    short_adverse_regimes: str = 'trend,low_vol_trend,high_vol'

    # ── 极端 divergence 做空否决（结构化过滤） ──
    use_extreme_divergence_short_veto: bool = False
    extreme_div_short_threshold: float = 85.0
    extreme_div_short_confirm_thr: float = 10.0
    extreme_div_short_min_confirms: int = 3
    extreme_div_short_regimes: str = 'trend,high_vol'

    # ── S2: 保本止损 — TP1触发后将SL移至保本, 防止盈利全部回吐 ──
    # 最新消融: 单开与组合均拉低收益, 默认关闭
    use_breakeven_after_tp1: bool = False
    breakeven_buffer: float = 0.01      # 允许入场价下方1%才触发保本止损

    # ── S3: 棘轮追踪止损 — 利润越高回撤容忍越小 ──
    # 最新消融: 与保本止损叠加时显著压制利润, 默认关闭
    use_ratchet_trail: bool = False
    # 格式 "threshold:pullback,..." — 到达利润阈值后pullback收紧
    ratchet_trail_tiers: str = '0.20:0.50,0.30:0.40,0.40:0.30'

    # ── S5: 信号质量止损 — 弱信号入场使用更紧SL, 减少低质量交易亏损 ──
    # 当前区间收益导向下默认关闭, 作为风控备选开关
    use_ss_quality_sl: bool = False
    ss_quality_sl_threshold: float = 50  # SS/BS低于此值视为弱信号
    ss_quality_sl_mult: float = 0.70     # 弱信号SL *= 0.70 (如-0.20→-0.14)

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
