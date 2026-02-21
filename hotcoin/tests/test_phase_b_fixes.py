"""
Phase B 修复项测试:
  B-9  ExchangeInfoCache 刷新失败后 5min 重试
  B-7  SignalDispatcher shutdown 超时
  C-7  K线 bar-level 缓存
  C-5  PnLTracker 线程安全
"""

import threading
import time

import pytest


# ---- B-9: ExchangeInfoCache retry ----

def test_exchange_cache_retries_after_failure(monkeypatch):
    from hotcoin.execution.order_executor import ExchangeInfoCache

    cache = ExchangeInfoCache()
    call_count = {"n": 0}

    def _failing_refresh(self):
        call_count["n"] += 1
        raise RuntimeError("network error")

    monkeypatch.setattr(ExchangeInfoCache, "_refresh", _failing_refresh)

    cache.get("ETHUSDT")
    assert call_count["n"] == 1
    assert cache._consecutive_failures == 1

    cache._last_attempt = time.time() - cache.RETRY_INTERVAL - 1
    cache.get("ETHUSDT")
    assert call_count["n"] == 2
    assert cache._consecutive_failures == 2


def test_exchange_cache_reset_on_success(monkeypatch):
    from hotcoin.execution.order_executor import ExchangeInfoCache

    cache = ExchangeInfoCache()
    cache._consecutive_failures = 5
    cache._last_attempt = 0

    def _ok_refresh(self):
        self._last_refresh = time.time()
        self._last_attempt = time.time()
        self._consecutive_failures = 0

    monkeypatch.setattr(ExchangeInfoCache, "_refresh", _ok_refresh)
    cache.get("ETHUSDT")
    assert cache._consecutive_failures == 0


# ---- B-7: SignalDispatcher shutdown timeout ----

def test_dispatcher_shutdown_timeout():
    from hotcoin.config import TradingConfig, DiscoveryConfig
    from hotcoin.engine.signal_dispatcher import SignalDispatcher

    cfg = TradingConfig(max_signal_workers=1)
    dcfg = DiscoveryConfig()
    disp = SignalDispatcher(cfg, dcfg)

    t0 = time.time()
    disp.shutdown(wait=True, timeout=1)
    elapsed = time.time() - t0
    assert elapsed < 5


def test_dispatcher_shutdown_nowait():
    from hotcoin.config import TradingConfig, DiscoveryConfig
    from hotcoin.engine.signal_dispatcher import SignalDispatcher

    cfg = TradingConfig(max_signal_workers=1)
    dcfg = DiscoveryConfig()
    disp = SignalDispatcher(cfg, dcfg)

    t0 = time.time()
    disp.shutdown(wait=False)
    elapsed = time.time() - t0
    assert elapsed < 2


# ---- C-7: K-line bar-level cache ----

def test_kline_cache_hit(monkeypatch):
    import pandas as pd
    from hotcoin.engine import signal_worker

    signal_worker.clear_kline_cache()

    fetch_count = {"n": 0}
    fake_df = pd.DataFrame({"close": range(100)})

    def _fake_fetch(symbol, interval=None, days=None):
        fetch_count["n"] += 1
        return fake_df

    monkeypatch.setattr("hotcoin.engine.signal_worker.fetch_binance_klines", _fake_fetch,
                         raising=False)
    monkeypatch.setitem(
        signal_worker.__dict__, "fetch_binance_klines", _fake_fetch
    ) if hasattr(signal_worker, "fetch_binance_klines") else None

    import importlib
    import sys
    if "binance_fetcher" not in sys.modules:
        import types
        mod = types.ModuleType("binance_fetcher")
        mod.fetch_binance_klines = _fake_fetch
        sys.modules["binance_fetcher"] = mod

    monkeypatch.setattr("binance_fetcher.fetch_binance_klines", _fake_fetch)

    df1 = signal_worker._get_cached_klines("TESTUSDT", "5m", days=7, min_bars=50)
    assert fetch_count["n"] == 1
    assert df1 is not None

    df2 = signal_worker._get_cached_klines("TESTUSDT", "5m", days=7, min_bars=50)
    assert fetch_count["n"] == 1  # cache hit, no extra fetch

    signal_worker.clear_kline_cache()


def test_kline_cache_eviction(monkeypatch):
    import pandas as pd
    from hotcoin.engine import signal_worker

    signal_worker.clear_kline_cache()

    fake_df = pd.DataFrame({"close": range(100)})

    def _fake_fetch(symbol, interval=None, days=None):
        return fake_df

    monkeypatch.setattr("binance_fetcher.fetch_binance_klines", _fake_fetch)

    old_max = signal_worker._KLINE_CACHE_MAX
    monkeypatch.setattr(signal_worker, "_KLINE_CACHE_MAX", 3)

    for i in range(5):
        signal_worker._get_cached_klines(f"SYM{i}USDT", "1m", days=1, min_bars=10)

    assert len(signal_worker._kline_cache) <= 3

    monkeypatch.setattr(signal_worker, "_KLINE_CACHE_MAX", old_max)
    signal_worker.clear_kline_cache()


# ---- C-5: PnLTracker thread safety ----

def test_pnl_tracker_concurrent_writes():
    import tempfile
    from hotcoin.execution.pnl_tracker import PnLTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = PnLTracker(data_dir=tmpdir)
        errors = []

        def _writer(thread_id):
            try:
                for i in range(20):
                    tracker.record_trade(
                        symbol=f"T{thread_id}USDT", side="BUY",
                        entry_price=100.0, exit_price=101.0 + i * 0.1,
                        qty=1.0, entry_time=time.time() - 60,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        summary = tracker.get_summary()
        assert summary["total_trades"] == 100


def test_pnl_tracker_summary_snapshot_isolation():
    import tempfile
    from hotcoin.execution.pnl_tracker import PnLTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = PnLTracker(data_dir=tmpdir)
        for i in range(10):
            tracker.record_trade(
                symbol="ETHUSDT", side="BUY",
                entry_price=100.0, exit_price=110.0,
                qty=1.0, entry_time=time.time() - 60,
            )

        summary = tracker.get_summary()
        assert summary["total_trades"] == 10
        assert summary["wins"] == 10
        assert summary["win_rate"] == 1.0
