"""R8 改进测试: PnL exec_price, pool 出池, cleanup 安全, 预检重置, 缓存批量淘汰。"""

import time
import threading

from hotcoin.config import DiscoveryConfig, HotCoinConfig
from hotcoin.discovery.candidate_pool import CandidatePool, HotCoin
from hotcoin.discovery.ticker_stream import TickerStream, MiniTicker
from hotcoin.execution.order_executor import OrderExecutor


# -- close_position 优先用 exec_price --

def test_close_uses_exec_price_for_pnl():
    """平仓时 PnL 应基于实际成交价 (exec_price), 而非 hint price。"""
    from hotcoin.execution.spot_engine import HotCoinSpotEngine

    class _FakePool:
        anomalies = []
        def on_anomaly(self, a): pass
        def set_cooling(self, s): pass

    config = HotCoinConfig()
    config.execution.use_paper_trading = True
    engine = HotCoinSpotEngine(config, pool=_FakePool())

    engine.risk.open_position("TESTUSDT", "BUY", 100.0, 1.0, 100.0)

    engine._close_position("TESTUSDT", price=110.0, reason="test")

    assert engine.risk.num_positions == 0
    trades = engine.pnl._trades
    assert len(trades) == 1
    assert trades[0].exit_price == 110.0


def test_partial_close_uses_exec_price():
    """部分平仓也应优先用 exec_price。"""
    from hotcoin.execution.spot_engine import HotCoinSpotEngine

    class _FakePool:
        anomalies = []
        def on_anomaly(self, a): pass
        def set_cooling(self, s): pass

    config = HotCoinConfig()
    config.execution.use_paper_trading = True
    engine = HotCoinSpotEngine(config, pool=_FakePool())

    engine.risk.open_position("TESTUSDT", "BUY", 100.0, 2.0, 200.0)
    engine._partial_close("TESTUSDT", price=120.0, pct=0.5, reason="tp1")

    pos = engine.risk.positions.get("TESTUSDT")
    assert pos is not None
    assert pos.qty < 2.0


# -- candidate_pool.remove_expired 安全 --

def test_remove_expired_handles_cooling(tmp_path):
    """冷却到期的币种正确出池。"""
    db = str(tmp_path / "pool.db")
    pool = CandidatePool(db, DiscoveryConfig())

    coin = HotCoin(symbol="OLDUSDT", status="cooling", cooling_until=time.time() - 1)
    pool.update_coin(coin)
    assert pool.size == 1

    pool.remove_expired()
    assert pool.size == 0


def test_remove_expired_handles_low_score(tmp_path):
    """低分超时出池。"""
    db = str(tmp_path / "pool.db")
    config = DiscoveryConfig()
    config.pool_exit_hold_sec = 0  # 立即出池
    pool = CandidatePool(db, config)

    coin = HotCoin(
        symbol="WEAKUSDT", heat_score=0.5, status="watching",
        low_score_since=time.time() - 100,
    )
    pool.update_coin(coin)
    assert pool.size == 1

    pool.remove_expired()
    assert pool.size == 0


def test_remove_expired_no_error_on_empty(tmp_path):
    """空池调用 remove_expired 不异常。"""
    db = str(tmp_path / "pool.db")
    pool = CandidatePool(db, DiscoveryConfig())
    pool.remove_expired()
    assert pool.size == 0


# -- ticker_stream._cleanup_stale 异常安全 --

class _FakeDetector:
    def detect(self, sym, ticker):
        return None


class _FakePoolForStream:
    def on_anomaly(self, a): pass


def test_cleanup_stale_handles_bad_event_time():
    """event_time 异常时 cleanup 不崩溃。"""
    stream = TickerStream(DiscoveryConfig(), _FakeDetector(), _FakePoolForStream())

    bad_ticker = MiniTicker(
        symbol="BADUSDT", close=1.0, open_price=1.0,
        high=1.0, low=1.0, base_volume=0, quote_volume=0,
        event_time=0,
    )
    stream._tickers["BADUSDT"] = bad_ticker
    stream._cleanup_stale(time.time())
    assert "BADUSDT" not in stream._tickers


# -- precheck stats 每小时重置 --

def test_precheck_stats_reset_after_ttl():
    """预检统计在 TTL 到期后自动清零。"""
    config = HotCoinConfig()
    config.execution.use_paper_trading = True
    executor = OrderExecutor(config.execution)

    executor._record_precheck_failure("ETHUSDT", "MIN_NOTIONAL")
    assert executor._precheck_failures.get("MIN_NOTIONAL") == 1

    executor._precheck_stats_reset_at = time.time() - 7200

    executor._record_precheck_failure("BTCUSDT", "LOT_SIZE")
    assert executor._precheck_failures.get("MIN_NOTIONAL") is None
    assert executor._precheck_failures.get("LOT_SIZE") == 1


def test_precheck_stats_no_reset_within_ttl():
    """TTL 内不重置。"""
    config = HotCoinConfig()
    config.execution.use_paper_trading = True
    executor = OrderExecutor(config.execution)

    executor._record_precheck_failure("ETHUSDT", "MIN_NOTIONAL")
    executor._record_precheck_failure("ETHUSDT", "MIN_NOTIONAL")
    assert executor._precheck_failures.get("MIN_NOTIONAL") == 2


# -- K 线缓存批量淘汰 --

def test_kline_cache_batch_prune():
    """缓存超限时批量淘汰而非逐条。"""
    from hotcoin.engine.signal_worker import (
        _kline_cache, _kline_cache_lock, _KLINE_CACHE_MAX, _KLINE_CACHE_PRUNE_BATCH,
        clear_kline_cache,
    )
    import pandas as pd

    clear_kline_cache()

    with _kline_cache_lock:
        for i in range(_KLINE_CACHE_MAX + 10):
            key = (f"SYM{i}USDT", "1m")
            df = pd.DataFrame({"close": [1.0]})
            _kline_cache[key] = (df, time.time() - (_KLINE_CACHE_MAX + 10 - i))

    assert len(_kline_cache) > _KLINE_CACHE_MAX

    from hotcoin.engine.signal_worker import _get_cached_klines
    import unittest.mock as mock

    fake_df = pd.DataFrame({"close": range(100)})
    with mock.patch("binance_fetcher.fetch_binance_klines", return_value=fake_df):
        _get_cached_klines("NEWUSDT", "5m", 7, 50)

    assert len(_kline_cache) <= _KLINE_CACHE_MAX
    clear_kline_cache()
