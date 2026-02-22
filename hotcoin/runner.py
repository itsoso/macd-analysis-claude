#!/usr/bin/env python3
"""
热点币系统主入口 — asyncio 事件循环

启动方式:
    python -m hotcoin.runner                  # 默认纸面交易
    HOTCOIN_PAPER=0 python -m hotcoin.runner  # 实盘 (小资金灰度)
"""

import asyncio
import gzip
import json
import logging
import signal
import shutil
import sys
import os
import threading
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hotcoin.config import load_config, HotCoinConfig
from hotcoin.discovery.ticker_stream import TickerStream
from hotcoin.discovery.anomaly_detector import AnomalyDetector
from hotcoin.discovery.candidate_pool import CandidatePool
from hotcoin.discovery.hot_ranker import HotRanker
from hotcoin.discovery.listing_monitor import ListingMonitor
from hotcoin.discovery.social_twitter import TwitterMonitor
from hotcoin.discovery.social_binance_sq import BinanceSquareMonitor
from hotcoin.discovery.filters import CoinFilter
from hotcoin.engine.signal_dispatcher import SignalDispatcher
from hotcoin.execution.spot_engine import HotCoinSpotEngine

log = logging.getLogger("hotcoin")


def _apply_state_hysteresis(
    current_state: str,
    current_recovery_ok_cycles: int,
    raw_state: str,
    raw_reasons,
    degraded_recovery_cycles: int,
    blocked_recovery_cycles: int,
    allow_recovery_progress: bool = True,
):
    """对状态恢复做滞后控制：非 tradeable -> tradeable 需要连续健康周期确认。"""
    valid_states = {"tradeable", "degraded", "blocked"}
    raw_state = raw_state if raw_state in valid_states else "degraded"
    current_state = current_state if current_state in valid_states else raw_state
    reasons = list(raw_reasons or [])
    recovery_ok_cycles = max(0, int(current_recovery_ok_cycles or 0))

    if raw_state != "tradeable":
        # 劣化状态立即生效，恢复计数清零
        return raw_state, reasons, 0, None

    if current_state == "tradeable":
        return "tradeable", (reasons or ["healthy"]), 0, None

    required = blocked_recovery_cycles if current_state == "blocked" else degraded_recovery_cycles
    required = max(1, int(required))
    if allow_recovery_progress:
        recovery_ok_cycles += 1

    if recovery_ok_cycles >= required:
        return "tradeable", ["recovered_to_tradeable"], 0, None

    pending = {
        "target_state": "tradeable",
        "ok_cycles": recovery_ok_cycles,
        "required_cycles": required,
        "source_state": current_state,
    }
    reasons.append(f"recovery_pending:{recovery_ok_cycles}/{required}")
    return current_state, reasons, recovery_ok_cycles, pending


def _event_archive_base(events_file: str) -> tuple[str, str]:
    directory = os.path.dirname(events_file) or "."
    filename = os.path.basename(events_file)
    if filename.endswith(".jsonl"):
        stem = filename[:-6]
    else:
        stem = filename
    return directory, stem


def _list_event_archives(events_file: str) -> list[str]:
    directory, stem = _event_archive_base(events_file)
    out = []
    try:
        for name in os.listdir(directory):
            if not name.startswith(stem + "."):
                continue
            if not (name.endswith(".jsonl") or name.endswith(".jsonl.gz")):
                continue
            full = os.path.join(directory, name)
            if os.path.abspath(full) == os.path.abspath(events_file):
                continue
            out.append(full)
    except Exception:
        return []
    def _mtime(path: str) -> float:
        try:
            return os.path.getmtime(path)
        except Exception:
            return 0.0

    out.sort(key=_mtime, reverse=True)
    return out


def _prune_event_archives(events_file: str, keep: int):
    keep = max(0, int(keep))
    archives = _list_event_archives(events_file)
    for p in archives[keep:]:
        try:
            os.remove(p)
        except Exception:
            log.warning("删除旧事件归档失败: %s", p, exc_info=True)


def _rotate_event_log_file(
    events_file: str,
    max_bytes: int,
    keep_archives: int,
    compress_archive: bool = True,
    now_ts: float | None = None,
) -> str | None:
    try:
        if not os.path.exists(events_file):
            return None
        size = os.path.getsize(events_file)
    except Exception:
        return None

    max_bytes = max(1, int(max_bytes))
    if size < max_bytes:
        return None

    directory, stem = _event_archive_base(events_file)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(now_ts or time.time()))
    rotated = os.path.join(directory, f"{stem}.{ts}.jsonl")
    idx = 1
    while os.path.exists(rotated) or os.path.exists(rotated + ".gz"):
        rotated = os.path.join(directory, f"{stem}.{ts}_{idx}.jsonl")
        idx += 1

    os.replace(events_file, rotated)
    final_path = rotated
    if compress_archive:
        gz_path = rotated + ".gz"
        with open(rotated, "rb") as src, gzip.open(gz_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.remove(rotated)
        final_path = gz_path

    _prune_event_archives(events_file, keep_archives)
    return final_path


class HotCoinRunner:
    """主调度器: 协调 Discovery → Trading → Execution。"""

    TICKER_STALE_DEGRADED_SEC = 90
    TICKER_STALE_BLOCKED_SEC = 300
    ORDER_ERROR_DEGRADED_5M = 3
    ORDER_ERROR_BLOCKED_5M = 10
    DEGRADED_RECOVERY_CONFIRM_CYCLES = 3
    BLOCKED_RECOVERY_CONFIRM_CYCLES = 6
    EVENT_LOG_ROTATE_MAX_BYTES = 20 * 1024 * 1024
    EVENT_LOG_ARCHIVE_KEEP = 14
    EVENT_LOG_COMPRESS_ARCHIVE = True

    def __init__(self, config: HotCoinConfig):
        self.config = config
        self._shutdown = asyncio.Event()
        self._status_file = os.path.join(os.path.dirname(__file__), "data", "hotcoin_runtime_status.json")
        self._events_file = os.path.join(os.path.dirname(__file__), "data", "hotcoin_events.jsonl")
        self._event_lock = threading.Lock()
        self._last_engine_state = ""
        self._effective_engine_state = "unknown"
        self._recovery_ok_cycles = 0
        self._state_recovery_pending = None

        # Discovery 层
        self.pool = CandidatePool(config.db_path, config.discovery)
        self.coin_filter = CoinFilter(config.discovery)
        self.anomaly_detector = AnomalyDetector(config.discovery)
        # Trading 层 (先创建, 以便取 coin_age_fn)
        self.dispatcher = SignalDispatcher(config.trading, config.discovery)
        self.spot_engine = HotCoinSpotEngine(config, self.pool, event_sink=self._emit_event)

        self.ranker = HotRanker(config.discovery,
                                coin_age_fn=self.spot_engine.executor.get_coin_age_days)
        self.ticker_stream = TickerStream(config.discovery, self.anomaly_detector, self.pool)
        self.listing_monitor = ListingMonitor(config.discovery, self.pool)
        self.twitter_monitor = TwitterMonitor(self.pool)
        self.square_monitor = BinanceSquareMonitor(self.pool)

        # (Trading 层 dispatcher + spot_engine 已在上方创建)

    def _emit_event(self, event_type: str, payload: dict, trace_id: str = ""):
        event = {
            "ts": time.time(),
            "event_type": str(event_type),
            "payload": payload if isinstance(payload, dict) else {"raw": payload},
        }
        if trace_id:
            event["trace_id"] = trace_id
        try:
            os.makedirs(os.path.dirname(self._events_file), exist_ok=True)
            line = json.dumps(event, ensure_ascii=False) + "\n"
            with self._event_lock:
                rotated = _rotate_event_log_file(
                    events_file=self._events_file,
                    max_bytes=self.EVENT_LOG_ROTATE_MAX_BYTES,
                    keep_archives=self.EVENT_LOG_ARCHIVE_KEEP,
                    compress_archive=self.EVENT_LOG_COMPRESS_ARCHIVE,
                )
                if rotated:
                    log.info("事件日志已轮转: %s", rotated)
                with open(self._events_file, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception:
            log.exception("写入事件日志失败: %s", self._events_file)

    def _latest_ticker_age_sec(self, now: float) -> float:
        tickers = self.ticker_stream.tickers_ref
        if not tickers:
            return -1.0
        latest_event_ts = max((t.event_time or 0) for t in tickers.values()) / 1000.0
        if latest_event_ts <= 0:
            return -1.0
        return max(0.0, now - latest_event_ts)

    def _compute_engine_state(self, current_prices: dict, runtime_metrics: dict):
        now = time.time()
        age_sec = self._latest_ticker_age_sec(now)
        freshness = {
            "ws_connected": bool(self.ticker_stream.tickers_ref),
            "latest_ticker_age_sec": round(age_sec, 2) if age_sec >= 0 else None,
            "priced_symbols": len(current_prices),
        }

        reasons = []
        severity = 0  # 0=tradeable,1=degraded,2=blocked

        risk_summary = self.spot_engine.risk.get_summary()
        if bool(risk_summary.get("halted")):
            severity = 2
            reasons.append("risk_halted")

        if age_sec < 0:
            severity = max(severity, 1)
            reasons.append("ticker_unavailable")
        elif age_sec >= self.TICKER_STALE_BLOCKED_SEC:
            severity = max(severity, 2)
            reasons.append(f"ticker_stale>{self.TICKER_STALE_BLOCKED_SEC}s")
        elif age_sec >= self.TICKER_STALE_DEGRADED_SEC:
            severity = max(severity, 1)
            reasons.append(f"ticker_stale>{self.TICKER_STALE_DEGRADED_SEC}s")

        order_errors_5m = int(runtime_metrics.get("order_errors_5m", 0) or 0)
        if order_errors_5m >= self.ORDER_ERROR_BLOCKED_5M:
            severity = max(severity, 2)
            reasons.append(f"order_errors_5m>={self.ORDER_ERROR_BLOCKED_5M}")
        elif order_errors_5m >= self.ORDER_ERROR_DEGRADED_5M:
            severity = max(severity, 1)
            reasons.append(f"order_errors_5m>={self.ORDER_ERROR_DEGRADED_5M}")

        state = "tradeable" if severity == 0 else ("degraded" if severity == 1 else "blocked")
        return state, reasons, freshness

    def _apply_engine_state_hysteresis(
        self,
        raw_state: str,
        raw_reasons,
        allow_recovery_progress: bool = True,
    ):
        state, reasons, ok_cycles, pending = _apply_state_hysteresis(
            current_state=self._effective_engine_state,
            current_recovery_ok_cycles=self._recovery_ok_cycles,
            raw_state=raw_state,
            raw_reasons=raw_reasons,
            degraded_recovery_cycles=self.DEGRADED_RECOVERY_CONFIRM_CYCLES,
            blocked_recovery_cycles=self.BLOCKED_RECOVERY_CONFIRM_CYCLES,
            allow_recovery_progress=allow_recovery_progress,
        )
        self._effective_engine_state = state
        self._recovery_ok_cycles = ok_cycles
        self._state_recovery_pending = pending
        return state, reasons, pending

    async def run(self):
        """主事件循环。"""
        log.info("=" * 60)
        log.info("热点币系统启动")
        log.info("  纸面交易: %s", self.config.execution.use_paper_trading)
        log.info("  初始资金: $%.0f", self.config.execution.initial_capital)
        log.info("  候选池上限: %d", self.config.discovery.pool_max_size)
        log.info("  最大并发仓位: %d", self.config.execution.max_concurrent_positions)
        log.info("  信号执行: %s", self.config.execution.enable_order_execution)
        if self.config.execution.enable_order_execution:
            log.info("  执行模式: %s", "PAPER" if self.config.execution.use_paper_trading else "LIVE")
        log.info("=" * 60)
        self._write_status_snapshot(candidates=[], signals=[])

        # 关键任务: 崩溃则触发系统关闭
        critical_names = {"ticker_stream", "main_loop"}
        tasks = [
            asyncio.create_task(self.ticker_stream.run(self._shutdown), name="ticker_stream"),
            asyncio.create_task(self.listing_monitor.run(self._shutdown), name="listing_monitor"),
            asyncio.create_task(self.square_monitor.run(self._shutdown), name="square_monitor"),
            asyncio.create_task(self._main_loop(), name="main_loop"),
        ]
        if self.twitter_monitor.enabled:
            tasks.append(asyncio.create_task(self.twitter_monitor.run(self._shutdown), name="twitter_monitor"))

        for t in tasks:
            t.add_done_callback(lambda fut, crit=critical_names: self._on_task_done(fut, crit))

        try:
            await self._shutdown.wait()
        finally:
            log.info("开始优雅关闭...")
            for t in tasks:
                t.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=15,
                )
            except asyncio.TimeoutError:
                log.warning("子任务取消超时 (15s), 强制继续")
            self.dispatcher.shutdown()
            self._write_stopped_status()
            self.pool.close()
            log.info("热点币系统已关闭")

    def _on_task_done(self, fut: asyncio.Task, critical_names: set):
        """子任务完成/崩溃回调。"""
        name = fut.get_name()
        if fut.cancelled():
            return
        exc = fut.exception()
        if exc:
            log.error("子任务 %s 崩溃: %s", name, exc, exc_info=exc)
            if name in critical_names:
                log.critical("关键任务 %s 崩溃, 触发系统关闭", name)
                self._shutdown.set()
            else:
                log.warning("非关键任务 %s 崩溃, 系统继续运行", name)

    async def _main_loop(self):
        """10s 周期: 评分 → 筛选 → 信号 → (执行)。"""
        interval = self.config.trading.signal_loop_sec
        reconcile_every = 30  # 每 30 轮 (~5min) 对账一次
        loop_count = 0
        _slow_threshold = max(interval * 3, 30)
        while not self._shutdown.is_set():
            cycle_id = uuid.uuid4().hex[:12]
            loop_count += 1
            _cycle_t0 = time.time()
            try:
                # 0) 清除冷却到期 / 低分超时的币种
                self.pool.remove_expired()

                # 1) 更新六维热度评分
                self.ranker.update_scores(self.pool)

                # 2) 过滤 + 取 Top N
                candidates = self.pool.get_top(
                    n=self.config.execution.max_concurrent_positions * 2,
                    min_score=self.config.discovery.pool_enter_score,
                )
                candidates = self.coin_filter.apply(candidates)

                if candidates:
                    symbols = [c.symbol for c in candidates]
                    log.info("[%s] 候选池 Top %d: %s", cycle_id[:8], len(candidates),
                             ", ".join(f"{s}({c.heat_score:.0f})" for s, c in zip(symbols, candidates)))

                # 3) 信号计算
                signals = await self.dispatcher.compute_signals(candidates, cycle_id=cycle_id)
                for sig in signals:
                    if sig.action != "HOLD":
                        log.info("信号 %s %s | strength=%d confidence=%.2f | %s",
                                 sig.action, sig.symbol, sig.strength,
                                 sig.confidence, sig.reason)

                # 3.5) 回写 Pump + Alert + Signals 到候选池
                for sig in signals:
                    coin = self.pool.get(sig.symbol)
                    if coin is None:
                        continue
                    if sig.pump_phase:
                        coin.pump_phase = sig.pump_phase
                    if sig.pump_score > 0:
                        coin.pump_score = sig.pump_score
                    if sig.alert_level:
                        coin.alert_level = sig.alert_level
                    if sig.alert_score > 0:
                        coin.alert_score = sig.alert_score
                    # signals/filters 每轮全量更新 (可以从有→无)
                    coin.active_signals = ",".join(sig.active_signals) if sig.active_signals else ""
                    coin.active_filters = ",".join(sig.active_filters) if sig.active_filters else ""
                    self.pool.update_coin(coin)

                current_prices = {
                    sym: ticker.close
                    for sym, ticker in self.ticker_stream.tickers_ref.items()
                    if ticker.close > 0
                }

                runtime_metrics_gate = self.spot_engine.executor.get_runtime_metrics(window_sec=300)
                raw_state_gate, raw_state_reasons_gate, freshness = self._compute_engine_state(
                    current_prices=current_prices,
                    runtime_metrics=runtime_metrics_gate,
                )
                engine_state, state_reasons, recovery_pending = self._apply_engine_state_hysteresis(
                    raw_state=raw_state_gate,
                    raw_reasons=raw_state_reasons_gate,
                    allow_recovery_progress=True,
                )
                allow_open = engine_state == "tradeable"
                if engine_state != self._last_engine_state:
                    log.info("引擎状态切换: %s -> %s (%s)",
                             self._last_engine_state or "unknown",
                             engine_state,
                             ", ".join(state_reasons) if state_reasons else "ok")
                    self._last_engine_state = engine_state
                self._emit_event("candidate_snapshot", {
                    "pool_size": self.pool.size,
                    "candidate_count": len(candidates),
                    "top_symbols": [c.symbol for c in candidates[:10]],
                    "engine_state": engine_state,
                    "state_recovery_pending": recovery_pending,
                }, trace_id=cycle_id)
                self._emit_event("signal_snapshot", {
                    "signal_count": len(signals),
                    "actionable_count": len([s for s in signals if s.action != "HOLD"]),
                    "signals": [
                        {"symbol": s.symbol, "action": s.action, "strength": s.strength}
                        for s in signals if s.action != "HOLD"
                    ],
                    "engine_state": engine_state,
                    "allow_open": allow_open,
                    "state_recovery_pending": recovery_pending,
                }, trace_id=cycle_id)

                # 4) 可选执行
                if self.config.execution.enable_order_execution:
                    self.spot_engine.process_signals(
                        signals, allow_open=allow_open, current_prices=current_prices)

                    if loop_count % reconcile_every == 0:
                        try:
                            await asyncio.to_thread(self.spot_engine.reconcile_open_orders)
                        except Exception:
                            log.debug("对账异常", exc_info=True)

                    can_check_positions = bool(current_prices) and (
                        freshness.get("latest_ticker_age_sec") is None
                        or freshness.get("latest_ticker_age_sec") <= self.TICKER_STALE_BLOCKED_SEC
                    )
                    if can_check_positions:
                        self.spot_engine.check_positions(current_prices)
                    elif self.spot_engine.num_positions > 0:
                        log.warning("主循环: 价格数据不可用/过旧, 跳过持仓检查 (state=%s, 持仓=%d)",
                                    engine_state, self.spot_engine.num_positions)

                runtime_metrics = self.spot_engine.executor.get_runtime_metrics(window_sec=300)
                raw_state, raw_state_reasons, freshness = self._compute_engine_state(
                    current_prices=current_prices,
                    runtime_metrics=runtime_metrics,
                )
                engine_state, state_reasons, recovery_pending = self._apply_engine_state_hysteresis(
                    raw_state=raw_state,
                    raw_reasons=raw_state_reasons,
                    allow_recovery_progress=False,
                )
                self._write_status_snapshot(
                    candidates=candidates,
                    signals=signals,
                    engine_state=engine_state,
                    state_reasons=state_reasons,
                    state_recovery_pending=recovery_pending,
                    freshness=freshness,
                    runtime_metrics=runtime_metrics,
                )

            except (MemoryError, SystemExit):
                log.critical("致命异常, 触发系统关闭", exc_info=True)
                self._shutdown.set()
                return
            except Exception:
                log.exception("主循环异常 (pool=%d, positions=%d)",
                              self.pool.size, self.spot_engine.num_positions)

            _cycle_elapsed = time.time() - _cycle_t0
            if _cycle_elapsed > _slow_threshold:
                log.warning("主循环 [%s] 耗时 %.1fs 超过慢阈值 %.0fs (pool=%d)",
                            cycle_id[:8], _cycle_elapsed, _slow_threshold, self.pool.size)
            elif loop_count % 60 == 0:
                log.info("主循环健康: 第%d轮 耗时 %.1fs", loop_count, _cycle_elapsed)

            await asyncio.sleep(interval)

    def _write_status_snapshot(
        self,
        candidates,
        signals,
        engine_state: str = "unknown",
        state_reasons=None,
        state_recovery_pending=None,
        freshness=None,
        runtime_metrics=None,
    ):
        state_reasons = list(state_reasons or [])
        state_recovery_pending = state_recovery_pending if isinstance(state_recovery_pending, dict) else None
        freshness = freshness if isinstance(freshness, dict) else {}
        runtime_metrics = runtime_metrics if isinstance(runtime_metrics, dict) else {}
        actionable = [s for s in signals if s.action != "HOLD"]
        anomaly_coins = [c for c in candidates if c.source in ("momentum", "mixed")]
        precheck_stats = {}
        try:
            precheck_stats = self.spot_engine.executor.get_precheck_stats()
        except Exception:
            pass
        risk_summary = {}
        try:
            risk_summary = self.spot_engine.risk.get_summary()
        except Exception:
            pass
        payload = {
            "running": True,
            "ts": time.time(),
            "engine_state": engine_state,
            "engine_state_reasons": state_reasons,
            "state_recovery_pending": state_recovery_pending,
            "can_open_new_positions": engine_state == "tradeable",
            "freshness": freshness,
            "execution_metrics": runtime_metrics,
            "risk_halted": bool(risk_summary.get("halted")),
            "event_log_file": self._events_file,
            "pool_size": self.pool.size,
            "candidates": [
                {
                    "symbol": c.symbol,
                    "heat_score": c.heat_score,
                    "source": c.source,
                    "status": c.status,
                    "price_change_5m": c.price_change_5m,
                    "price_change_24h": c.price_change_24h,
                    "quote_volume_24h": c.quote_volume_24h,
                    "volume_surge_ratio": c.volume_surge_ratio,
                    "score_announcement": c.score_announcement,
                    "score_social": c.score_social,
                    "score_sentiment": c.score_sentiment,
                    "score_momentum": c.score_momentum,
                    "score_liquidity": c.score_liquidity,
                    "score_risk_penalty": c.score_risk_penalty,
                    "has_listing_signal": c.has_listing_signal,
                    "pump_phase": c.pump_phase,
                    "pump_score": c.pump_score,
                    "alert_level": c.alert_level,
                    "alert_score": c.alert_score,
                    "active_signals": c.active_signals,
                    "active_filters": c.active_filters,
                }
                for c in candidates[:20]
            ],
            "ws_connected": bool(self.ticker_stream.tickers_ref),
            "paper": self.config.execution.use_paper_trading,
            "execution_enabled": bool(self.config.execution.enable_order_execution),
            "anomaly_count": len(anomaly_coins),
            "active_signals": len(actionable),
            "positions": self.spot_engine.num_positions,
            "recent_signals": [
                {
                    "symbol": s.symbol,
                    "action": s.action,
                    "strength": s.strength,
                    "confidence": s.confidence,
                    "reason": s.reason,
                    "computed_at": s.computed_at,
                }
                for s in actionable[:20]
            ],
            "recent_anomalies": [
                {
                    "symbol": c.symbol,
                    "volume_surge_ratio": c.volume_surge_ratio,
                    "price_change_5m": c.price_change_5m,
                    "computed_at": c.last_score_update or c.discovered_at,
                }
                for c in anomaly_coins[:20]
            ],
            "precheck_stats": precheck_stats,
            "hot_posts": self._collect_hot_posts(),
        }
        self._write_json_atomic(payload)

    def _collect_hot_posts(self) -> list:
        """合并币安广场 + Twitter 最近帖子, 按时间倒序返回。"""
        posts = []
        try:
            posts.extend(self.square_monitor.get_recent_posts(30))
        except Exception:
            pass
        try:
            posts.extend(self.twitter_monitor.get_recent_posts(30))
        except Exception:
            pass
        posts.sort(key=lambda p: p.get("ts", 0), reverse=True)
        return posts[:50]

    def _write_stopped_status(self):
        self._write_json_atomic({
            "running": False,
            "ts": time.time(),
            "engine_state": "stopped",
            "can_open_new_positions": False,
            "message": "热点币系统已停止",
        })

    def _write_json_atomic(self, payload: dict):
        try:
            os.makedirs(os.path.dirname(self._status_file), exist_ok=True)
            tmp = f"{self._status_file}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, self._status_file)
        except Exception:
            log.exception("写入运行状态失败: %s", self._status_file)


def _setup_logging(level: str):
    fmt = "%(asctime)s [%(name)s] %(levelname)s  %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, stream=sys.stdout)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def main():
    config = load_config()
    _setup_logging(config.log_level)

    runner = HotCoinRunner(config)
    try:
        from hotcoin.web.routes import set_runner
        set_runner(runner)
    except Exception:
        pass
    loop = asyncio.new_event_loop()

    def _handle_signal(*_):
        log.info("收到关闭信号")
        runner._shutdown.set()

    for sig_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig_name, _handle_signal)

    try:
        loop.run_until_complete(runner.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
