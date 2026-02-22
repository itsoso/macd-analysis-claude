"""
候选池管理 — 热点币的入池/出池/冷却/持久化

存储: SQLite (hotcoins.db)
  coin_pool:    当前候选池状态
  heat_history: 热度评分历史
"""

import copy
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from hotcoin.config import DiscoveryConfig

log = logging.getLogger("hotcoin.pool")


@dataclass
class HotCoin:
    symbol: str
    heat_score: float = 0.0
    source: str = "unknown"          # "momentum" | "listing" | "social" | "mixed"
    discovered_at: float = 0.0       # unix ts
    # 量价维度
    price_change_1m: float = 0.0
    price_change_5m: float = 0.0
    price_change_15m: float = 0.0
    price_change_1h: float = 0.0
    volume_surge_ratio: float = 0.0
    quote_volume_24h: float = 0.0
    price_change_24h: float = 0.0
    # 社交维度 (Phase 2 填充)
    mention_velocity: float = 0.0
    sentiment: float = 0.0
    kol_mentions: List[str] = field(default_factory=list)
    # 公告维度
    listing_open_time: int = 0
    has_listing_signal: bool = False
    # 元信息
    has_futures: bool = False
    coin_age_days: int = -1
    # 状态管理
    status: str = "watching"         # "watching" | "analyzing" | "trading" | "cooling"
    last_score_update: float = 0.0
    low_score_since: float = 0.0     # 低于出池门槛的起始时间
    cooling_until: float = 0.0       # 冷却到期时间

    # 六维热度评分分项
    score_announcement: float = 0.0
    score_social: float = 0.0
    score_sentiment: float = 0.0
    score_momentum: float = 0.0
    score_liquidity: float = 0.0
    score_risk_penalty: float = 0.0

    # Pump + 信号 + 预警字段
    pump_phase: str = "normal"
    pump_score: float = 0.0
    alert_level: str = "NONE"        # "NONE" | "L1" | "L2" | "L3"
    alert_score: float = 0.0
    active_signals: str = ""         # "S1@5m,S3@15m" (逗号分隔)
    active_filters: str = ""         # "F2" (逗号分隔)


class CandidatePool:
    """线程安全的候选币池, 带 SQLite 持久化。"""

    def __init__(self, db_path: str, config: DiscoveryConfig):
        self.config = config
        self._lock = threading.Lock()
        self._coins: Dict[str, HotCoin] = {}
        self._heat_hist_ts: Dict[str, float] = {}

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._init_tables()
        self._load_from_db()

    def _init_tables(self):
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS coin_pool (
                symbol TEXT PRIMARY KEY,
                heat_score REAL,
                source TEXT,
                status TEXT,
                data_json TEXT,
                updated_at REAL
            );
            CREATE TABLE IF NOT EXISTS heat_history (
                symbol TEXT,
                ts REAL,
                score REAL,
                components TEXT,
                PRIMARY KEY (symbol, ts)
            );
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                side TEXT,
                price REAL,
                qty REAL,
                quote_qty REAL,
                pnl REAL,
                signal_detail TEXT,
                ts REAL
            );
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                ts REAL,
                action TEXT,
                strength INTEGER,
                confidence REAL,
                components TEXT
            );
        """)
        self._db.commit()

    def _load_from_db(self):
        """启动时从 SQLite 恢复候选池。"""
        rows = self._db.execute(
            "SELECT symbol, heat_score, source, status, data_json FROM coin_pool"
        ).fetchall()
        for sym, score, source, status, data_json in rows:
            try:
                d = json.loads(data_json) if data_json else {}
                coin = HotCoin(symbol=sym, heat_score=score, source=source, status=status, **{
                    k: v for k, v in d.items()
                    if k in HotCoin.__dataclass_fields__ and k not in ("symbol", "heat_score", "source", "status")
                })
                self._coins[sym] = coin
            except Exception:
                log.warning("恢复 %s 失败, 跳过", sym)
        if self._coins:
            log.info("从 DB 恢复 %d 个候选币", len(self._coins))

    def _persist(self, coin: HotCoin, *, commit: bool = True):
        """写入 / 更新单个币种到 DB。commit=False 时由调用方负责批量提交。"""
        data = {k: v for k, v in asdict(coin).items()
                if k not in ("symbol", "heat_score", "source", "status", "kol_mentions")}
        data["kol_mentions"] = coin.kol_mentions
        self._db.execute(
            """INSERT OR REPLACE INTO coin_pool (symbol, heat_score, source, status, data_json, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (coin.symbol, coin.heat_score, coin.source, coin.status,
             json.dumps(data, default=str), time.time()),
        )
        if commit:
            self._db.commit()

    _HEAT_HISTORY_INTERVAL = 60  # 每个 symbol 最多 60 秒写一条

    def record_heat_history(self, coin: HotCoin):
        now = time.time()
        with self._lock:
            last_ts = self._heat_hist_ts.get(coin.symbol, 0)
            if now - last_ts < self._HEAT_HISTORY_INTERVAL:
                return
            components = json.dumps({
                "announcement": coin.score_announcement,
                "social": coin.score_social,
                "sentiment": coin.score_sentiment,
                "momentum": coin.score_momentum,
                "liquidity": coin.score_liquidity,
                "risk_penalty": coin.score_risk_penalty,
            })
            self._db.execute(
                "INSERT OR IGNORE INTO heat_history (symbol, ts, score, components) VALUES (?, ?, ?, ?)",
                (coin.symbol, now, coin.heat_score, components),
            )
            self._db.commit()
            self._heat_hist_ts[coin.symbol] = now

    # ------------------------------------------------------------------
    # 入池接口
    # ------------------------------------------------------------------

    def on_anomaly(self, signal):
        """量价异动检测到的币种入池。"""
        with self._lock:
            now = time.time()
            coin = self._coins.get(signal.symbol)
            if coin and coin.status == "cooling" and now < coin.cooling_until:
                return  # 冷却中, 不入池

            if coin is None:
                coin = HotCoin(
                    symbol=signal.symbol,
                    source="momentum",
                    discovered_at=now,
                )
                self._coins[signal.symbol] = coin

            coin.volume_surge_ratio = signal.volume_surge_ratio
            coin.price_change_5m = signal.price_change_5m
            coin.price_change_24h = signal.price_change_24h
            coin.quote_volume_24h = signal.quote_volume_24h
            if coin.source == "listing":
                coin.source = "mixed"
            elif coin.source == "unknown":
                coin.source = "momentum"
            coin.status = "watching"
            self._persist(coin)

    def on_social_mention(self, symbol: str, source: str = "twitter",
                          kol_id: str = "", mention_count_1h: int = 0,
                          sentiment: float = 0.0):
        """社交提及信号入池 (Phase 2)。"""
        with self._lock:
            now = time.time()
            coin = self._coins.get(symbol)
            if coin and coin.status == "cooling" and now < coin.cooling_until:
                return
            if coin is None:
                coin = HotCoin(
                    symbol=symbol,
                    source="social",
                    discovered_at=now,
                )
                self._coins[symbol] = coin

            coin.mention_velocity = mention_count_1h
            if sentiment != 0.0:
                coin.sentiment = sentiment
            if kol_id and kol_id not in coin.kol_mentions:
                coin.kol_mentions.append(kol_id)
            if coin.source in ("momentum", "listing"):
                coin.source = "mixed"
            elif coin.source == "unknown":
                coin.source = "social"
            self._persist(coin)

    def on_listing(self, symbol: str, open_time: int):
        """新币上线信号入池。"""
        with self._lock:
            coin = self._coins.get(symbol)
            if coin is None:
                coin = HotCoin(
                    symbol=symbol,
                    source="listing",
                    discovered_at=time.time(),
                )
                self._coins[symbol] = coin

            coin.has_listing_signal = True
            coin.listing_open_time = open_time
            if coin.source == "momentum":
                coin.source = "mixed"
            elif coin.source == "unknown":
                coin.source = "listing"
            coin.status = "watching"
            self._persist(coin)
            log.info("新币入池: %s (openTime=%d)", symbol, open_time)

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def get_top(self, n: int = 10, min_score: float = 0.0) -> List[HotCoin]:
        """返回热度最高的 N 个候选币 (返回浅拷贝)。"""
        with self._lock:
            eligible = [copy.copy(c) for c in self._coins.values()
                        if c.heat_score >= min_score
                        and c.status not in ("cooling",)]
            eligible.sort(key=lambda c: c.heat_score, reverse=True)
            return eligible[:n]

    def get_all(self) -> List[HotCoin]:
        """返回所有候选币 (浅拷贝, 线程安全)。"""
        with self._lock:
            return [copy.copy(c) for c in self._coins.values()]

    def get(self, symbol: str) -> Optional[HotCoin]:
        with self._lock:
            c = self._coins.get(symbol)
            return copy.copy(c) if c else None

    def update_coin(self, coin: HotCoin):
        with self._lock:
            self._coins[coin.symbol] = coin
            self._persist(coin)

    def update_status(self, symbol: str, status: str):
        """仅更新 status 字段, 不覆盖实时行情数据。"""
        with self._lock:
            coin = self._coins.get(symbol)
            if coin:
                coin.status = status
                self._persist(coin)

    def update_coins_batch(self, coins: List[HotCoin]):
        """批量更新评分字段 — 不覆盖实时行情数据, 单次 commit。"""
        _SCORE_FIELDS = (
            "heat_score", "score_announcement", "score_social",
            "score_sentiment", "score_momentum", "score_liquidity",
            "score_risk_penalty", "last_score_update", "low_score_since",
            "pump_phase", "pump_score", "alert_level", "alert_score",
            "active_signals", "active_filters",
        )
        with self._lock:
            for coin in coins:
                existing = self._coins.get(coin.symbol)
                if existing is None:
                    self._coins[coin.symbol] = coin
                else:
                    for f in _SCORE_FIELDS:
                        setattr(existing, f, getattr(coin, f))
                self._persist(self._coins[coin.symbol], commit=False)
            self._db.commit()

    def set_cooling(self, symbol: str):
        """交易止损后, 设置冷却。"""
        with self._lock:
            coin = self._coins.get(symbol)
            if coin:
                coin.status = "cooling"
                coin.cooling_until = time.time() + self.config.pool_cooling_sec
                self._persist(coin)
                log.info("%s 进入冷却 (%ds)", symbol, self.config.pool_cooling_sec)

    def remove_expired(self):
        """清除低分超时和冷却到期的币种。"""
        now = time.time()
        to_remove = []
        with self._lock:
            for sym, coin in list(self._coins.items()):
                if coin.status == "cooling" and now >= coin.cooling_until:
                    to_remove.append(sym)
                elif (coin.heat_score < self.config.pool_exit_score
                      and coin.low_score_since > 0
                      and now - coin.low_score_since > self.config.pool_exit_hold_sec):
                    to_remove.append(sym)

            if not to_remove:
                return

            for sym in to_remove:
                self._coins.pop(sym, None)
                self._heat_hist_ts.pop(sym, None)

            try:
                for sym in to_remove:
                    self._db.execute("DELETE FROM coin_pool WHERE symbol = ?", (sym,))
                self._db.commit()
            except Exception:
                log.exception("出池 DB 写入失败, 下次循环重试")

            log.info("出池: %s", ", ".join(to_remove))

    @property
    def size(self) -> int:
        return len(self._coins)

    def close(self):
        self._db.close()
