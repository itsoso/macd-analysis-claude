"""
CandidatePool 单元测试
"""

import os
import tempfile
import time
import pytest

from hotcoin.config import DiscoveryConfig
from hotcoin.discovery.candidate_pool import CandidatePool, HotCoin
from hotcoin.discovery.anomaly_detector import AnomalySignal


@pytest.fixture
def pool():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_hotcoins.db")
        config = DiscoveryConfig()
        p = CandidatePool(db_path, config)
        yield p
        p.close()


def test_on_anomaly(pool):
    signal = AnomalySignal(
        symbol="TESTUSDT",
        volume_surge_ratio=8.0,
        price_change_5m=0.05,
        price_change_24h=0.10,
        quote_volume_24h=2_000_000,
        detected_at=time.time(),
    )
    pool.on_anomaly(signal)
    assert pool.size == 1
    coin = pool.get("TESTUSDT")
    assert coin is not None
    assert coin.source == "momentum"


def test_on_listing(pool):
    pool.on_listing("NEWUSDT", open_time=1000000)
    assert pool.size == 1
    coin = pool.get("NEWUSDT")
    assert coin is not None
    assert coin.has_listing_signal is True
    assert coin.source == "listing"


def test_mixed_source(pool):
    pool.on_listing("MIXUSDT", open_time=0)
    signal = AnomalySignal(
        symbol="MIXUSDT",
        volume_surge_ratio=6.0,
        price_change_5m=0.04,
        price_change_24h=0.08,
        quote_volume_24h=1_500_000,
        detected_at=time.time(),
    )
    pool.on_anomaly(signal)
    coin = pool.get("MIXUSDT")
    assert coin.source == "mixed"


def test_get_top(pool):
    for i in range(5):
        coin = HotCoin(symbol=f"T{i}USDT", heat_score=float(i * 20))
        pool.update_coin(coin)

    top = pool.get_top(n=3, min_score=20)
    assert len(top) == 3
    assert top[0].heat_score == 80.0


def test_cooling(pool):
    signal = AnomalySignal(
        symbol="COOLTEST",
        volume_surge_ratio=6.0,
        price_change_5m=0.04,
        price_change_24h=0.08,
        quote_volume_24h=1_000_000,
        detected_at=time.time(),
    )
    pool.on_anomaly(signal)
    pool.set_cooling("COOLTEST")

    coin = pool.get("COOLTEST")
    assert coin.status == "cooling"

    # cooling 中不应该再入池
    pool.on_anomaly(signal)
    coin = pool.get("COOLTEST")
    assert coin.status == "cooling"


def test_social_mention(pool):
    pool.on_social_mention("SOCUSDT", source="twitter", kol_id="kol1", mention_count_1h=5)
    coin = pool.get("SOCUSDT")
    assert coin is not None
    assert coin.source == "social"
    assert coin.mention_velocity == 5
    assert "kol1" in coin.kol_mentions
