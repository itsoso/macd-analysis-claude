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

    def __init__(self, trading_config: TradingConfig, discovery_config: DiscoveryConfig):
        self.config = trading_config
        self.discovery_config = discovery_config
        self._executor = ThreadPoolExecutor(
            max_workers=trading_config.max_signal_workers,
            thread_name_prefix="sig-worker",
        )

    async def compute_signals(self, candidates: List[HotCoin]) -> List[TradeSignal]:
        """并发计算候选币信号, 返回所有结果 (含 HOLD)。"""
        if not candidates:
            return []

        loop = asyncio.get_running_loop()
        tasks = []
        for coin in candidates:
            tfs = self.config.timeframes or HOT_COIN_TIMEFRAMES
            task = loop.run_in_executor(
                self._executor,
                compute_signal_for_symbol,
                coin.symbol,
                tfs,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        signals = []
        for coin, result in zip(candidates, results):
            if isinstance(result, Exception):
                log.warning("%s 信号计算异常: %s", coin.symbol, result)
                signals.append(TradeSignal(
                    symbol=coin.symbol, action="HOLD",
                    reason=f"exception: {result}",
                ))
            else:
                signals.append(result)

        actionable = [s for s in signals if s.action != "HOLD"]
        if actionable:
            log.info("信号结果: %d/%d 可操作 — %s",
                     len(actionable), len(signals),
                     ", ".join(f"{s.symbol}:{s.action}" for s in actionable))

        return signals

    def shutdown(self):
        self._executor.shutdown(wait=False)
