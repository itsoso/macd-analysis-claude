"""R11 改进验证: ranker 异常容错、pnl_tracker 指标增强、allocator 不变式"""

import time
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# 1. hot_ranker: 单个币评分异常不中断整体
# ---------------------------------------------------------------------------
class TestHotRankerExceptionSafety:
    def test_single_coin_error_does_not_block_others(self):
        from hotcoin.config import DiscoveryConfig
        from hotcoin.discovery.hot_ranker import HotRanker
        from hotcoin.discovery.candidate_pool import HotCoin

        cfg = DiscoveryConfig()
        ranker = HotRanker(cfg)

        coins = [
            HotCoin(symbol="AAAUSDT", price_change_5m=0.05, quote_volume_24h=1e6),
            HotCoin(symbol="BBBUSDT", price_change_5m=0.03, quote_volume_24h=2e6),
        ]

        original_compute = ranker._compute_score
        call_count = {"n": 0}

        def _bad_first_coin(coin):
            call_count["n"] += 1
            if coin.symbol == "AAAUSDT":
                raise ValueError("模拟评分错误")
            return original_compute(coin)

        ranker._compute_score = _bad_first_coin

        class FakePool:
            def __init__(self, coins):
                self._coins = coins
            def get_all(self):
                return list(self._coins)
            def update_coins_batch(self, coins):
                self._updated = coins
            def record_heat_history(self, coin):
                pass

        pool = FakePool(coins)
        ranker.update_scores(pool)

        assert call_count["n"] == 2
        assert len(pool._updated) == 1
        assert pool._updated[0].symbol == "BBBUSDT"
        assert pool._updated[0].heat_score > 0


# ---------------------------------------------------------------------------
# 2. pnl_tracker: profit_factor 和 max_drawdown
# ---------------------------------------------------------------------------
class TestPnLTrackerEnhanced:
    def test_profit_factor_and_max_drawdown(self):
        from hotcoin.execution.pnl_tracker import PnLTracker
        import tempfile, os

        tracker = PnLTracker(data_dir=tempfile.mkdtemp())

        now = time.time()
        tracker.record_trade("A", "BUY", 100, 110, 1, now - 60, "tp")
        tracker.record_trade("B", "BUY", 100, 90, 1, now - 50, "sl")
        tracker.record_trade("C", "BUY", 100, 120, 1, now - 40, "tp2")

        s = tracker.get_summary()
        assert s["total_trades"] == 3
        assert s["wins"] == 2
        assert s["losses"] == 1

        assert "profit_factor" in s
        gross_profit = 10 + 20
        gross_loss = 10
        expected_pf = gross_profit / gross_loss
        assert s["profit_factor"] == round(expected_pf, 2)

        assert "max_drawdown" in s
        assert s["max_drawdown"] >= 0

    def test_profit_factor_no_losses(self):
        from hotcoin.execution.pnl_tracker import PnLTracker
        import tempfile

        tracker = PnLTracker(data_dir=tempfile.mkdtemp())
        now = time.time()
        tracker.record_trade("X", "BUY", 10, 15, 1, now - 10, "tp")

        s = tracker.get_summary()
        assert s["profit_factor"] == "inf"
        assert s["max_drawdown"] == 0

    def test_max_drawdown_calculation(self):
        from hotcoin.execution.pnl_tracker import PnLTracker
        import tempfile

        tracker = PnLTracker(data_dir=tempfile.mkdtemp())
        now = time.time()
        tracker.record_trade("A", "BUY", 100, 120, 1, now - 60, "tp")
        tracker.record_trade("B", "BUY", 100, 80, 1, now - 50, "sl")
        tracker.record_trade("C", "BUY", 100, 85, 1, now - 40, "sl")

        s = tracker.get_summary()
        # peak after A: +20, then B: -20, C: -15 → cumulative: 20, 0, -15
        # peak=20, max_dd = 20 - (-15) = 35
        assert s["max_drawdown"] == 35.0


# ---------------------------------------------------------------------------
# 3. capital_allocator: heat_score 边界 (补充 R10 的 clamp)
# ---------------------------------------------------------------------------
class TestAllocatorNegativeHeat:
    def test_negative_heat_score_clamped(self):
        from hotcoin.execution.capital_allocator import CapitalAllocator
        from hotcoin.config import ExecutionConfig

        cfg = ExecutionConfig(
            initial_capital=10000,
            max_concurrent_positions=5,
            max_total_exposure_pct=0.5,
            max_single_position_pct=0.2,
            use_paper_trading=True,
            enable_order_execution=False,
        )
        alloc = CapitalAllocator(cfg)
        result = alloc.allocate_single(heat_score=-50, liquidity_score=80,
                                       current_positions=0)
        zero_result = alloc.allocate_single(heat_score=0, liquidity_score=80,
                                            current_positions=0)
        assert result == zero_result, "负 heat_score 应被 clamp 到 0"
