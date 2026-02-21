#!/usr/bin/env python3
"""
热点币系统主入口 — asyncio 事件循环

启动方式:
    python -m hotcoin.runner                  # 默认纸面交易
    HOTCOIN_PAPER=0 python -m hotcoin.runner  # 实盘 (小资金灰度)
"""

import asyncio
import json
import logging
import signal
import sys
import os
import threading
import time

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


class HotCoinRunner:
    """主调度器: 协调 Discovery → Trading → Execution。"""

    TICKER_STALE_DEGRADED_SEC = 90
    TICKER_STALE_BLOCKED_SEC = 300
    ORDER_ERROR_DEGRADED_5M = 3
    ORDER_ERROR_BLOCKED_5M = 10

    def __init__(self, config: HotCoinConfig):
        self.config = config
        self._shutdown = asyncio.Event()
        self._status_file = os.path.join(os.path.dirname(__file__), "data", "hotcoin_runtime_status.json")
        self._events_file = os.path.join(os.path.dirname(__file__), "data", "hotcoin_events.jsonl")
        self._event_lock = threading.Lock()
        self._last_engine_state = ""

        # Discovery 层
        self.pool = CandidatePool(config.db_path, config.discovery)
        self.coin_filter = CoinFilter(config.discovery)
        self.anomaly_detector = AnomalyDetector(config.discovery)
        self.ranker = HotRanker(config.discovery)
        self.ticker_stream = TickerStream(config.discovery, self.anomaly_detector, self.pool)
        self.listing_monitor = ListingMonitor(config.discovery, self.pool)
        self.twitter_monitor = TwitterMonitor(self.pool)
        self.square_monitor = BinanceSquareMonitor(self.pool)

        # Trading 层
        self.dispatcher = SignalDispatcher(config.trading, config.discovery)
        self.spot_engine = HotCoinSpotEngine(config, self.pool, event_sink=self._emit_event)

    def _emit_event(self, event_type: str, payload: dict):
        event = {
            "ts": time.time(),
            "event_type": str(event_type),
            "payload": payload if isinstance(payload, dict) else {"raw": payload},
        }
        try:
            os.makedirs(os.path.dirname(self._events_file), exist_ok=True)
            line = json.dumps(event, ensure_ascii=False) + "\n"
            with self._event_lock:
                with open(self._events_file, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception:
            log.exception("写入事件日志失败: %s", self._events_file)

    def _latest_ticker_age_sec(self, now: float) -> float:
        tickers = self.ticker_stream.tickers
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
            "ws_connected": bool(self.ticker_stream.tickers),
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
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
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
        while not self._shutdown.is_set():
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
                    log.info("候选池 Top %d: %s", len(candidates),
                             ", ".join(f"{s}({c.heat_score:.0f})" for s, c in zip(symbols, candidates)))

                # 3) 信号计算
                signals = await self.dispatcher.compute_signals(candidates)
                for sig in signals:
                    if sig.action != "HOLD":
                        log.info("信号 %s %s | strength=%d confidence=%.2f | %s",
                                 sig.action, sig.symbol, sig.strength,
                                 sig.confidence, sig.reason)

                current_prices = {
                    sym: ticker.close
                    for sym, ticker in self.ticker_stream.tickers.items()
                    if ticker.close > 0
                }

                runtime_metrics_gate = self.spot_engine.executor.get_runtime_metrics(window_sec=300)
                engine_state, state_reasons, freshness = self._compute_engine_state(
                    current_prices=current_prices,
                    runtime_metrics=runtime_metrics_gate,
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
                })
                self._emit_event("signal_snapshot", {
                    "signal_count": len(signals),
                    "actionable_count": len([s for s in signals if s.action != "HOLD"]),
                    "engine_state": engine_state,
                    "allow_open": allow_open,
                })

                # 4) 可选执行
                if self.config.execution.enable_order_execution:
                    self.spot_engine.process_signals(signals, allow_open=allow_open)
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
                engine_state, state_reasons, freshness = self._compute_engine_state(
                    current_prices=current_prices,
                    runtime_metrics=runtime_metrics,
                )
                self._write_status_snapshot(
                    candidates=candidates,
                    signals=signals,
                    engine_state=engine_state,
                    state_reasons=state_reasons,
                    freshness=freshness,
                    runtime_metrics=runtime_metrics,
                )

            except Exception:
                log.exception("主循环异常 (pool=%d, positions=%d)",
                              self.pool.size, self.spot_engine.num_positions)

            await asyncio.sleep(interval)

    def _write_status_snapshot(
        self,
        candidates,
        signals,
        engine_state: str = "unknown",
        state_reasons=None,
        freshness=None,
        runtime_metrics=None,
    ):
        state_reasons = list(state_reasons or [])
        freshness = freshness if isinstance(freshness, dict) else {}
        runtime_metrics = runtime_metrics if isinstance(runtime_metrics, dict) else {}
        actionable = [s for s in signals if s.action != "HOLD"]
        anomaly_coins = [c for c in candidates if c.source in ("momentum", "mixed")]
        precheck_stats = {}
        try:
            precheck_stats = self.spot_engine.executor.get_precheck_stats()
        except Exception:
            precheck_stats = {}
        risk_summary = self.spot_engine.risk.get_summary()
        payload = {
            "running": True,
            "ts": time.time(),
            "engine_state": engine_state,
            "engine_state_reasons": state_reasons,
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
                    "score_momentum": c.score_momentum,
                    "score_liquidity": c.score_liquidity,
                    "score_risk_penalty": c.score_risk_penalty,
                    "has_listing_signal": c.has_listing_signal,
                }
                for c in candidates[:20]
            ],
            "ws_connected": bool(self.ticker_stream.tickers),
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
        }
        self._write_json_atomic(payload)

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
