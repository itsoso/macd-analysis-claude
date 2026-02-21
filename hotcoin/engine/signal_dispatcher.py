"""
多币信号调度器

使用 asyncio + ThreadPoolExecutor 并发为候选池中的多个币种计算信号。
信号计算本身是同步阻塞 (涉及网络 IO + pandas), 通过线程池并行。
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List

from hotcoin.config import TradingConfig, DiscoveryConfig
from hotcoin.discovery.candidate_pool import HotCoin
from hotcoin.engine.signal_worker import TradeSignal, compute_signal_for_symbol
from hotcoin.engine.hot_coin_params import HOT_COIN_TIMEFRAMES

log = logging.getLogger("hotcoin.dispatch")


class SignalDispatcher:
    """多币种信号并发调度。"""

    SIGNAL_TIMEOUT_SEC = 60

    def __init__(self, trading_config: TradingConfig, discovery_config: DiscoveryConfig):
        self.config = trading_config
        self.discovery_config = discovery_config
        self._executor = ThreadPoolExecutor(
            max_workers=trading_config.max_signal_workers,
            thread_name_prefix="sig-worker",
        )

    async def compute_signals(self, candidates: List[HotCoin],
                              cycle_id: str = "") -> List[TradeSignal]:
        """并发计算候选币信号, 返回所有结果 (含 HOLD)。单币超时保护。"""
        if not candidates:
            return []

        loop = asyncio.get_running_loop()
        tasks = []
        for coin in candidates:
            tfs = self.config.timeframes or HOT_COIN_TIMEFRAMES
            future = loop.run_in_executor(
                self._executor,
                compute_signal_for_symbol,
                coin.symbol,
                tfs,
            )
            tasks.append(asyncio.wait_for(future, timeout=self.SIGNAL_TIMEOUT_SEC))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        signals = []
        for coin, result in zip(candidates, results):
            if isinstance(result, asyncio.TimeoutError):
                log.warning("%s 信号计算超时 (>%ds)", coin.symbol, self.SIGNAL_TIMEOUT_SEC)
                signals.append(TradeSignal(
                    symbol=coin.symbol, action="HOLD",
                    reason=f"timeout >{self.SIGNAL_TIMEOUT_SEC}s",
                    trace_id=cycle_id,
                ))
            elif isinstance(result, Exception):
                log.error("%s 信号计算异常: %s", coin.symbol, result, exc_info=result)
                signals.append(TradeSignal(
                    symbol=coin.symbol, action="HOLD",
                    reason=f"exception: {type(result).__name__}: {result}",
                    trace_id=cycle_id,
                ))
            else:
                result.trace_id = cycle_id
                signals.append(result)

        actionable = [s for s in signals if s.action != "HOLD"]
        if actionable:
            log.info("信号结果 [%s]: %d/%d 可操作 — %s",
                     cycle_id[:8] if cycle_id else "-",
                     len(actionable), len(signals),
                     ", ".join(f"{s.symbol}:{s.action}" for s in actionable))

        return signals

    def shutdown(self, wait: bool = True, timeout: float = 30):
        """优雅关闭线程池。wait=True 时最多等 timeout 秒, 防止死锁。"""
        import threading

        log.info("SignalDispatcher 关闭中 (wait=%s, timeout=%.0fs)", wait, timeout)
        if not wait:
            self._executor.shutdown(wait=False, cancel_futures=True)
            return

        done = threading.Event()

        def _shutdown_worker():
            self._executor.shutdown(wait=True, cancel_futures=True)
            done.set()

        t = threading.Thread(target=_shutdown_worker, daemon=True)
        t.start()
        if done.wait(timeout=timeout):
            log.info("SignalDispatcher 所有 worker 已退出")
        else:
            log.warning("SignalDispatcher shutdown 超时 (%.0fs), 强制放弃", timeout)
