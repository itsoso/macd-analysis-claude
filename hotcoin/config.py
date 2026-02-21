"""
热点币系统全局配置

独立于 live_config.py (ETH 交易配置)，仅控制热点币发现与交易。
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

STABLECOINS: Set[str] = {
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "USDPUSDT", "DAIUSDT",
    "FDUSDUSDT", "EURUSDT", "GBPUSDT", "AEURUSDT",
}

BLACKLIST_SYMBOLS: Set[str] = {
    s.strip() for s in os.environ.get("HOTCOIN_BLACKLIST", "").split(",")
} - {""}

BASE_WS_URL = "wss://stream.binance.com:9443"
BASE_REST_URL = "https://api.binance.com"


def _safe_float(env_key: str, default: float) -> float:
    raw = os.environ.get(env_key, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_env_bool(env_key: str) -> Any:
    """
    解析布尔环境变量。
    返回 True/False，若未设置或无法识别则返回 None。
    """
    raw = os.environ.get(env_key)
    if raw is None:
        return None
    val = raw.strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return None


# ---------------------------------------------------------------------------
# Discovery 层配置
# ---------------------------------------------------------------------------

@dataclass
class DiscoveryConfig:
    # 全市场 Ticker 扫描
    ticker_interval_sec: float = 1.0

    # 量价异动检测
    volume_surge_ratio: float = 5.0       # 1min 量 / 20min 均量 倍数阈值
    price_surge_5m_pct: float = 0.03      # 5min 涨幅阈值
    min_quote_volume_24h: float = 500_000  # 最低 24h 成交额 (USDT)
    max_price_change_24h: float = 0.30    # FOMO 过滤: 24h 涨幅上限

    # 候选池
    pool_max_size: int = 20
    pool_enter_score: float = 40.0        # 入池最低热度
    pool_exit_score: float = 20.0         # 出池热度
    pool_exit_hold_sec: int = 600         # 低分持续 N 秒后出池
    pool_cooling_sec: int = 1800          # 止损后冷却时间

    # 新币上线监控
    listing_poll_sec: int = 300
    announcement_poll_sec: int = 60

    # 六维热度权重 (Phase 1 仅 w4+w5+w6 生效)
    w_announcement: float = 0.20
    w_social: float = 0.15
    w_sentiment: float = 0.10
    w_momentum: float = 0.25
    w_liquidity: float = 0.20
    w_risk_penalty: float = 0.10


# ---------------------------------------------------------------------------
# Trading 层配置
# ---------------------------------------------------------------------------

@dataclass
class TradingConfig:
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h"])
    signal_loop_sec: float = 10.0
    max_signal_workers: int = 5

    # 入场
    min_consensus_strength: int = 20      # 多周期共识最低强度
    entry_confirm_tf: str = "15m"         # 方向确认周期
    entry_trigger_tf: str = "5m"          # 触发周期

    # 出场
    default_sl_pct: float = -0.05         # 默认止损 -5%
    atr_sl_mult: float = 1.5             # ATR 止损倍数 (比 ETH 的 2.5 更紧)
    take_profit_tiers: List[tuple] = field(default_factory=lambda: [
        (0.05, 0.30),   # 涨 5% 卖 30%
        (0.10, 0.30),   # 涨 10% 卖 30%
    ])
    trailing_stop_pct: float = 0.03
    max_hold_minutes: int = 240           # 最长持仓 4h
    black_swan_pct: float = -0.20         # 15min 跌超此值立即止损


# ---------------------------------------------------------------------------
# Execution 层配置
# ---------------------------------------------------------------------------

@dataclass
class ExecutionConfig:
    enable_order_execution: bool = False
    max_concurrent_positions: int = 5
    max_single_position_pct: float = 0.10  # 单币最大 10% 资金
    max_total_exposure_pct: float = 0.40   # 总敞口 40%
    max_sector_exposure_pct: float = 0.20  # 同板块上限
    initial_capital: float = field(default_factory=lambda: _safe_float("HOTCOIN_CAPITAL", 1000))

    # 风控
    daily_max_loss_pct: float = 0.05       # 日亏损 -5% 停新仓
    total_drawdown_halt_pct: float = 0.15  # 总回撤 -15% 全清仓
    single_coin_max_loss_pct: float = 0.05 # 单币最大亏损
    cooling_after_halt_sec: int = 86400    # 全清仓后冷却 24h

    # 下单
    order_timeout_sec: float = 3.0         # 限价单超时转市价
    use_paper_trading: bool = True         # 默认纸面交易


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------

@dataclass
class HotCoinConfig:
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    db_path: str = os.path.join(os.path.dirname(__file__), "data", "hotcoins.db")
    log_level: str = os.environ.get("HOTCOIN_LOG_LEVEL", "INFO")


def load_config() -> HotCoinConfig:
    """加载配置，环境变量可覆盖关键字段。"""
    cfg = HotCoinConfig()

    # 1) 优先读取 DB 中保存的热点币配置
    try:
        import config_store
        saved = config_store.get_hotcoin_config()
        if isinstance(saved, dict) and saved:
            _merge_hotcoin_config(cfg, saved)
    except Exception:
        # DB 不可用时保持默认配置
        pass

    # 2) 环境变量覆盖（最高优先级）
    paper = _parse_env_bool("HOTCOIN_PAPER")
    if paper is not None:
        cfg.execution.use_paper_trading = paper
    capital = os.environ.get("HOTCOIN_CAPITAL")
    if capital:
        try:
            cfg.execution.initial_capital = float(capital)
        except ValueError:
            pass
    execute = _parse_env_bool("HOTCOIN_EXECUTE")
    if execute is not None:
        cfg.execution.enable_order_execution = execute
    db_path = os.environ.get("HOTCOIN_DB_PATH")
    if db_path:
        cfg.db_path = db_path
    log_level = os.environ.get("HOTCOIN_LOG_LEVEL")
    if log_level:
        cfg.log_level = log_level
    return cfg


def _coerce_value(value: Any, default: Any) -> Any:
    """按默认值类型进行弱类型转换。"""
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(value)
        except Exception:
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except Exception:
            return default
    if isinstance(default, list):
        return value if isinstance(value, list) else default
    if isinstance(default, str):
        return str(value)
    return value


def _merge_section(target: Any, updates: Dict[str, Any]):
    if not isinstance(updates, dict):
        return
    for key, value in updates.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        setattr(target, key, _coerce_value(value, current))


def _merge_hotcoin_config(cfg: HotCoinConfig, saved: Dict[str, Any]):
    if isinstance(saved.get("discovery"), dict):
        _merge_section(cfg.discovery, saved["discovery"])
    if isinstance(saved.get("trading"), dict):
        _merge_section(cfg.trading, saved["trading"])
    if isinstance(saved.get("execution"), dict):
        _merge_section(cfg.execution, saved["execution"])
    if "db_path" in saved:
        cfg.db_path = _coerce_value(saved["db_path"], cfg.db_path)
    if "log_level" in saved:
        cfg.log_level = _coerce_value(saved["log_level"], cfg.log_level)
