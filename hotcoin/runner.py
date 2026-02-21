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

    def __init__(self, config: HotCoinConfig):
        self.config = config
        self._shutdown = asyncio.Event()
        self._status_file = os.path.join(os.path.dirname(__file__), "data", "hotcoin_runtime_status.json")

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
        self.spot_engine = HotCoinSpotEngine(config, self.pool)

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

        tasks = [
            asyncio.create_task(self.ticker_stream.run(self._shutdown), name="ticker_stream"),
            asyncio.create_task(self.listing_monitor.run(self._shutdown), name="listing_monitor"),
            asyncio.create_task(self.square_monitor.run(self._shutdown), name="square_monitor"),
            asyncio.create_task(self._main_loop(), name="main_loop"),
        ]
        if self.twitter_monitor.enabled:
            tasks.append(asyncio.create_task(self.twitter_monitor.run(self._shutdown), name="twitter_monitor"))

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

    async def _main_loop(self):
        """10s 周期: 评分 → 筛选 → 信号 → (执行)。"""
        interval = self.config.trading.signal_loop_sec
        while not self._shutdown.is_set():
            try:
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

                # 4) 可选执行
                if self.config.execution.enable_order_execution:
                    self.spot_engine.process_signals(signals)
                    current_prices = {
                        sym: ticker.close
                        for sym, ticker in self.ticker_stream.tickers.items()
                        if ticker.close > 0
                    }
                    if current_prices:
                        self.spot_engine.check_positions(current_prices)
                self._write_status_snapshot(candidates=candidates, signals=signals)

            except Exception:
                log.exception("主循环异常")

            await asyncio.sleep(interval)

    def _write_status_snapshot(self, candidates, signals):
        actionable = [s for s in signals if s.action != "HOLD"]
        payload = {
            "running": True,
            "ts": time.time(),
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
            "anomaly_count": len([c for c in candidates if c.source in ("momentum", "mixed")]),
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
            "recent_anomalies": [],
        }
        self._write_json_atomic(payload)

    def _write_stopped_status(self):
        self._write_json_atomic({
            "running": False,
            "ts": time.time(),
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
