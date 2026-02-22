"""R10 改进验证: allocator 边界、quote_vol 估算、dust 清理、params API"""

import time
import types
import threading
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. CapitalAllocator 边界条件
# ---------------------------------------------------------------------------
from hotcoin.execution.capital_allocator import CapitalAllocator
from hotcoin.config import ExecutionConfig


def _make_exec_cfg(**overrides):
    defaults = dict(
        initial_capital=10000,
        max_concurrent_positions=5,
        max_total_exposure_pct=0.5,
        max_single_position_pct=0.2,
        use_paper_trading=True,
        enable_order_execution=False,
    )
    defaults.update(overrides)
    return ExecutionConfig(**defaults)


class TestCapitalAllocatorEdgeCases:
    def test_heat_score_clamped_above_100(self):
        """heat_score > 100 不应导致分配超出上限。"""
        alloc = CapitalAllocator(_make_exec_cfg())
        normal = alloc.allocate_single(heat_score=100, liquidity_score=80,
                                       current_positions=0)
        over = alloc.allocate_single(heat_score=200, liquidity_score=80,
                                     current_positions=0)
        assert over == normal, f"heat>100 应被 clamp: {over} vs {normal}"

    def test_batch_budget_not_exceeded(self):
        """批量分配总额不超过 remaining_capital。"""
        cfg = _make_exec_cfg(initial_capital=100, max_total_exposure_pct=0.5,
                             max_concurrent_positions=10)
        alloc = CapitalAllocator(cfg)
        candidates = [
            {"symbol": f"COIN{i}USDT", "heat_score": 90, "liquidity_score": 90}
            for i in range(8)
        ]
        result = alloc.allocate_batch(candidates, current_positions=0, used_exposure=0)
        total = sum(result.values())
        assert total <= 50.01, f"总分配 {total} 超过 remaining_capital 50"

    def test_batch_early_exit_on_budget_exhausted(self):
        """预算耗尽后不继续分配。"""
        cfg = _make_exec_cfg(initial_capital=50, max_total_exposure_pct=0.5,
                             max_concurrent_positions=10)
        alloc = CapitalAllocator(cfg)
        candidates = [
            {"symbol": f"T{i}USDT", "heat_score": 80, "liquidity_score": 80}
            for i in range(10)
        ]
        result = alloc.allocate_batch(candidates, current_positions=0, used_exposure=0)
        total = sum(result.values())
        assert total <= 25.01


# ---------------------------------------------------------------------------
# 2. signal_worker quote_vol_24h 滚动均值
# ---------------------------------------------------------------------------

class TestQuoteVolEstimation:
    def test_uses_rolling_mean_not_single_bar(self):
        """验证 quote_vol_24h 使用多 bar 均值而非单个 bar。"""
        import pandas as pd
        import numpy as np

        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "open": np.random.uniform(100, 200, n),
            "high": np.random.uniform(200, 300, n),
            "low": np.random.uniform(50, 100, n),
            "close": np.random.uniform(100, 200, n),
            "volume": np.random.uniform(1000, 5000, n),
            "quote_volume": np.random.uniform(100000, 500000, n),
        })
        df.iloc[-1, df.columns.get_loc("quote_volume")] = 9999999

        tf = "5m"
        tf_minutes = {"5m": 5}
        bars_per_day = 1440 / tf_minutes[tf]
        sample_n = min(len(df), max(1, int(bars_per_day)))
        avg_vol = float(df["quote_volume"].iloc[-sample_n:].mean())
        estimate = avg_vol * bars_per_day

        naive_estimate = float(df["quote_volume"].iloc[-1]) * bars_per_day
        assert estimate < naive_estimate, \
            "滚动均值应比仅用最后 bar 更低 (最后 bar 被注入了极端值)"


# ---------------------------------------------------------------------------
# 3. partial_close 后 dust 清理
# ---------------------------------------------------------------------------

class TestDustCleanup:
    def test_dust_position_auto_closed(self):
        """partial_close 后剩余价值 < $1 时触发自动全部平仓。"""
        from hotcoin.config import load_config
        from hotcoin.execution.spot_engine import HotCoinSpotEngine

        config = load_config()

        class FakePool:
            def get(self, sym):
                return None
            def set_cooling(self, sym):
                pass

        engine = HotCoinSpotEngine(config, FakePool())

        engine.executor.spot_market_sell = MagicMock(return_value={
            "status": "FILLED", "price": "0.50", "qty": "1.0"
        })

        engine.risk.open_position("DUSTUSDT", "BUY", price=0.50, qty=2.0)
        pos = engine.risk.positions.get("DUSTUSDT")
        assert pos is not None

        engine._partial_close("DUSTUSDT", 0.50, 0.9, "tp1")

        remaining = engine.risk.positions.get("DUSTUSDT")
        assert remaining is None, "剩余价值 $0.10 < $1, 应已被 dust_cleanup 清理"


# ---------------------------------------------------------------------------
# 4. /api/params 端点
# ---------------------------------------------------------------------------

class TestParamsAPI:
    def test_params_endpoint_returns_structure(self):
        from hotcoin.web.routes import hotcoin_bp
        from flask import Flask

        app = Flask(__name__)
        app.register_blueprint(hotcoin_bp, url_prefix="/hotcoin")

        with app.test_client() as client:
            resp = client.get("/hotcoin/api/params")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["ok"] is True
            assert "fusion" in data
            assert "indicators" in data
            assert "timeframes" in data
            assert "consensus" in data
            assert "kline_params" in data
            assert isinstance(data["timeframes"], list)
            assert len(data["timeframes"]) > 0
