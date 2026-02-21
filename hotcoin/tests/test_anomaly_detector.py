"""
AnomalyDetector 单元测试
"""

import time
import pytest

from hotcoin.config import DiscoveryConfig
from hotcoin.discovery.anomaly_detector import AnomalyDetector


class MockTicker:
    def __init__(self, **kwargs):
        self.symbol = kwargs.get("symbol", "TESTUSDT")
        self.volume_1m = kwargs.get("volume_1m", 100000)
        self.avg_volume_20m = kwargs.get("avg_volume_20m", 10000)
        self.price_change_5m = kwargs.get("price_change_5m", 0.05)
        self.quote_volume = kwargs.get("quote_volume", 1_000_000)
        self.price_change_24h = kwargs.get("price_change_24h", 0.10)


def test_detects_anomaly():
    config = DiscoveryConfig()
    detector = AnomalyDetector(config)

    ticker = MockTicker(
        volume_1m=100000,
        avg_volume_20m=10000,   # 10x surge
        price_change_5m=0.05,    # +5%
        quote_volume=2_000_000,  # $2M
        price_change_24h=0.10,   # +10%
    )
    signal = detector.detect("PEPEUSDT", ticker)
    assert signal is not None
    assert signal.symbol == "PEPEUSDT"
    assert signal.volume_surge_ratio == 10.0


def test_filters_low_volume():
    config = DiscoveryConfig()
    detector = AnomalyDetector(config)

    ticker = MockTicker(
        volume_1m=100000,
        avg_volume_20m=10000,
        price_change_5m=0.05,
        quote_volume=100_000,  # too low
        price_change_24h=0.10,
    )
    signal = detector.detect("LOWVOLSDT", ticker)
    assert signal is None


def test_filters_fomo():
    config = DiscoveryConfig()
    detector = AnomalyDetector(config)

    ticker = MockTicker(
        volume_1m=100000,
        avg_volume_20m=10000,
        price_change_5m=0.05,
        quote_volume=2_000_000,
        price_change_24h=0.50,  # +50% — FOMO
    )
    signal = detector.detect("FOMOUSDT", ticker)
    assert signal is None


def test_filters_stablecoins():
    config = DiscoveryConfig()
    detector = AnomalyDetector(config)

    ticker = MockTicker(
        volume_1m=100000,
        avg_volume_20m=10000,
        price_change_5m=0.05,
        quote_volume=2_000_000,
        price_change_24h=0.01,
    )
    signal = detector.detect("USDCUSDT", ticker)
    assert signal is None


def test_cooldown():
    config = DiscoveryConfig()
    detector = AnomalyDetector(config)
    detector._cooldown_sec = 60

    ticker = MockTicker(
        volume_1m=100000,
        avg_volume_20m=10000,
        price_change_5m=0.05,
        quote_volume=2_000_000,
        price_change_24h=0.10,
    )

    sig1 = detector.detect("COOLSDT", ticker)
    assert sig1 is not None

    sig2 = detector.detect("COOLSDT", ticker)
    assert sig2 is None  # cooldown
