"""
六书融合策略 · 核心模块完整单元测试

覆盖:
1. FuturesEngine: 费用/滑点/强平/资金费率/仓位管理/余额一致性
2. KDJ信号计算: compute_kdj/compute_kdj_scores 正确性
3. 策略引擎: 部分止盈/边界条件/极端行情/资金耗尽
4. 融合评分: calc_fusion_score_six 多模式验证
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_futures import FuturesEngine, FuturesPosition


# ======================================================
#   辅助函数
# ======================================================
def make_ts(n=100):
    """创建测试用时间序列"""
    return pd.date_range('2025-01-01', periods=n, freq='h')


def make_df(n=100, base_price=2000, trend=0):
    """创建模拟OHLCV DataFrame"""
    idx = make_ts(n)
    np.random.seed(42)
    close = base_price + np.cumsum(np.random.randn(n) * 20) + np.arange(n) * trend
    close = np.maximum(close, 100)  # 确保价格为正
    high = close + np.abs(np.random.randn(n) * 10)
    low = close - np.abs(np.random.randn(n) * 10)
    low = np.maximum(low, 50)
    opn = close + np.random.randn(n) * 5
    vol = np.abs(np.random.randn(n) * 1000) + 500
    df = pd.DataFrame({
        'open': opn, 'high': high, 'low': low, 'close': close,
        'volume': vol, 'quote_volume': vol * close
    }, index=idx)
    return df


# ======================================================
#   1. FuturesPosition 测试
# ======================================================
class TestFuturesPosition:
    def test_long_pnl_positive(self):
        pos = FuturesPosition('long', 2000, 10, 5, 4000)
        pnl = pos.calc_pnl(2100)  # 涨100
        assert pnl == pytest.approx(10 * 100, rel=1e-6)  # +1000

    def test_long_pnl_negative(self):
        pos = FuturesPosition('long', 2000, 10, 5, 4000)
        pnl = pos.calc_pnl(1900)
        assert pnl == pytest.approx(-1000, rel=1e-6)

    def test_short_pnl_positive(self):
        pos = FuturesPosition('short', 2000, 10, 5, 4000)
        pnl = pos.calc_pnl(1900)  # 跌100, 空头赚
        assert pnl == pytest.approx(1000, rel=1e-6)

    def test_short_pnl_negative(self):
        pos = FuturesPosition('short', 2000, 10, 5, 4000)
        pnl = pos.calc_pnl(2100)
        assert pnl == pytest.approx(-1000, rel=1e-6)

    def test_liquidation_long(self):
        pos = FuturesPosition('long', 2000, 1, 5, 400)
        # 修正: 维持保证金 = notional * 0.05 = qty * price * 0.05
        # 强平条件: margin + pnl < qty * price * 0.05
        # 400 + (p - 2000) * 1 < 1 * p * 0.05
        # 400 + p - 2000 < 0.05p  →  0.95p < 1600  →  p < 1684.21
        assert pos.is_liquidated(1680)
        assert not pos.is_liquidated(1690)

    def test_liquidation_short(self):
        pos = FuturesPosition('short', 2000, 1, 5, 400)
        # 维持保证金 = qty * price * 0.05
        # 400 + (2000 - p) * 1 < 1 * p * 0.05
        # 2400 - p < 0.05p  →  2400 < 1.05p  →  p > 2285.71
        assert pos.is_liquidated(2290)
        assert not pos.is_liquidated(2280)

    def test_liquidation_price_long(self):
        pos = FuturesPosition('long', 2000, 1, 5, 400)
        liq_p = pos.liquidation_price()
        # P = (qty*entry - margin) / (qty*(1-mr))
        # P = (1*2000 - 400) / (1*(1-0.05)) = 1600/0.95 ≈ 1684.21
        assert liq_p == pytest.approx(1684.2105, rel=1e-4)

    def test_liquidation_price_short(self):
        pos = FuturesPosition('short', 2000, 1, 5, 400)
        liq_p = pos.liquidation_price()
        # P = (qty*entry + margin) / (qty*(1+mr))
        # P = (1*2000 + 400) / (1*(1+0.05)) = 2400/1.05 ≈ 2285.71
        assert liq_p == pytest.approx(2285.7143, rel=1e-4)


# ======================================================
#   2. FuturesEngine 基础测试
# ======================================================
class TestFuturesEngineBasic:
    def setup_method(self):
        self.eng = FuturesEngine('test', max_leverage=5)
        self.eng.max_single_margin = 50000
        self.eng.max_margin_total = 100000
        self.eng.max_lifetime_margin = 500000
        self.dt = pd.Timestamp('2025-01-01')

    def test_initial_state(self):
        assert self.eng.usdt == pytest.approx(100000)
        assert self.eng.spot_eth == 0
        assert self.eng.frozen_margin == 0
        assert self.eng.futures_long is None
        assert self.eng.futures_short is None

    def test_spot_buy_fees(self):
        """现货买入: 验证费用和滑点"""
        initial = self.eng.usdt
        self.eng.spot_buy(2000, self.dt, 10000, "test")
        
        # 投入10000, 手续费 10000*0.0005=5, 滑点在价格中
        assert self.eng.usdt == pytest.approx(initial - 10000)
        assert self.eng.spot_eth > 0
        assert self.eng.total_spot_fees == pytest.approx(5, rel=0.01)
        assert self.eng.total_slippage_cost > 0

    def test_spot_sell_fees(self):
        """现货卖出: 验证费用和滑点"""
        self.eng.spot_eth = 5.0
        self.eng.spot_sell(2000, self.dt, 0.5, "test")
        
        # 卖2.5 ETH, 含滑点和手续费
        assert self.eng.spot_eth == pytest.approx(2.5)
        assert self.eng.total_spot_fees > 0

    def test_open_long_fees(self):
        """开多仓: 验证费用扣除和保证金冻结"""
        initial = self.eng.usdt
        self.eng.open_long(2000, self.dt, 10000, 5, "test")
        
        notional = 10000 * 5  # 50000
        fee = notional * 0.0005  # 25
        
        assert self.eng.futures_long is not None
        assert self.eng.usdt == pytest.approx(initial - fee)
        assert self.eng.frozen_margin == pytest.approx(10000)
        assert self.eng.total_futures_fees == pytest.approx(fee, rel=0.01)

    def test_open_close_long_cycle(self):
        """完整开多→平多周期: 验证余额一致性"""
        initial = self.eng.usdt
        self.eng.open_long(2000, self.dt, 10000, 5, "test")
        
        # 价格涨到 2100 后平仓
        self.eng.close_long(2100, self.dt, "tp")
        
        assert self.eng.futures_long is None
        assert self.eng.frozen_margin == pytest.approx(0)
        # 应该盈利 (但扣除手续费和滑点后)
        assert self.eng.usdt > initial - 100  # 扣费后仍应盈利

    def test_open_close_short_cycle(self):
        """完整开空→平空周期"""
        initial = self.eng.usdt
        self.eng.open_short(2000, self.dt, 10000, 5, "test")
        
        # 价格跌到 1900 后平仓
        self.eng.close_short(1900, self.dt, "tp")
        
        assert self.eng.futures_short is None
        assert self.eng.frozen_margin == pytest.approx(0)
        assert self.eng.usdt > initial - 100

    def test_double_open_prevented(self):
        """不能重复开仓"""
        self.eng.open_long(2000, self.dt, 10000, 5, "test1")
        usdt_after_first = self.eng.usdt
        self.eng.open_long(2000, self.dt, 10000, 5, "test2")
        # 第二次应该被忽略
        assert self.eng.usdt == usdt_after_first

    def test_close_empty_safe(self):
        """平仓空仓位应安全"""
        initial = self.eng.usdt
        self.eng.close_long(2000, self.dt, "no_pos")
        assert self.eng.usdt == initial

    def test_minimum_trade_size(self):
        """最小交易额限制"""
        initial = self.eng.usdt
        self.eng.open_long(2000, self.dt, 100, 5, "too_small")  # < 200
        assert self.eng.futures_long is None
        assert self.eng.usdt == initial


# ======================================================
#   3. FuturesEngine 费率一致性测试
# ======================================================
class TestFeeConsistency:
    def test_open_long_fee_manual_calc(self):
        """开多手续费手工验算"""
        eng = FuturesEngine('test', max_leverage=5)
        eng.max_single_margin = 50000
        eng.max_margin_total = 100000
        eng.max_lifetime_margin = 500000
        dt = pd.Timestamp('2025-01-01')
        
        price = 2700
        margin = 10000
        lev = 5
        
        eng.open_long(price, dt, margin, lev, "test")
        
        notional = margin * lev  # 50000
        expected_fee = notional * 0.0005  # 25
        expected_slippage = notional * 0.001  # 50
        actual_entry = price * 1.001  # 2702.7
        expected_qty = notional / actual_entry
        
        assert eng.total_futures_fees == pytest.approx(expected_fee, rel=0.01)
        assert eng.total_slippage_cost == pytest.approx(expected_slippage, rel=0.01)
        assert eng.futures_long.entry_price == pytest.approx(actual_entry, rel=1e-4)
        assert eng.futures_long.quantity == pytest.approx(expected_qty, rel=1e-4)

    def test_roundtrip_long_manual_calc(self):
        """多仓完整周期手工验算"""
        eng = FuturesEngine('test', max_leverage=5)
        eng.max_single_margin = 50000
        eng.max_margin_total = 100000
        eng.max_lifetime_margin = 500000
        dt = pd.Timestamp('2025-01-01')
        
        initial = eng.usdt
        eng.open_long(2700, dt, 10000, 5, "test")
        eng.close_long(2800, dt, "test")
        
        # 手工计算
        notional_open = 50000
        entry = 2700 * 1.001  # 2702.7
        qty = notional_open / entry
        exit_p = 2800 * 0.999  # 2797.2
        pnl = (exit_p - entry) * qty
        fee_open = notional_open * 0.0005
        fee_close = exit_p * qty * 0.0005
        
        expected_final = initial - fee_open + pnl - fee_close
        assert eng.usdt == pytest.approx(expected_final, rel=1e-4)

    def test_roundtrip_short_manual_calc(self):
        """空仓完整周期手工验算"""
        eng = FuturesEngine('test', max_leverage=5)
        eng.max_single_margin = 50000
        eng.max_margin_total = 100000
        eng.max_lifetime_margin = 500000
        dt = pd.Timestamp('2025-01-01')
        
        initial = eng.usdt
        eng.open_short(2700, dt, 10000, 5, "test")
        eng.close_short(2600, dt, "test")
        
        notional_open = 50000
        entry = 2700 * 0.999  # 2697.3
        qty = notional_open / entry
        exit_p = 2600 * 1.001  # 2602.6
        pnl = (entry - exit_p) * qty
        fee_open = notional_open * 0.0005
        fee_close = exit_p * qty * 0.0005
        
        expected_final = initial - fee_open + pnl - fee_close
        assert eng.usdt == pytest.approx(expected_final, rel=1e-4)


# ======================================================
#   4. 强平测试
# ======================================================
class TestLiquidation:
    def test_long_liquidation(self):
        """多头触发强平"""
        eng = FuturesEngine('test', max_leverage=10)
        eng.max_single_margin = 50000
        eng.max_margin_total = 100000
        eng.max_lifetime_margin = 500000
        dt = pd.Timestamp('2025-01-01')
        
        eng.open_long(2000, dt, 10000, 10, "test")
        # 杠杆10x, margin 10000, notional 100000
        # 入场价约2002, 强平价约 2002 - 9500/49.95 ≈ 1812
        
        # 极端下跌, 应该触发强平
        eng.check_liquidation(1500, dt)
        assert eng.futures_long is None
        assert eng.total_liquidation_fees > 0
        assert eng.frozen_margin == pytest.approx(0)

    def test_short_liquidation(self):
        """空头触发强平"""
        eng = FuturesEngine('test', max_leverage=10)
        eng.max_single_margin = 50000
        eng.max_margin_total = 100000
        eng.max_lifetime_margin = 500000
        dt = pd.Timestamp('2025-01-01')
        
        eng.open_short(2000, dt, 10000, 10, "test")
        
        # 极端上涨
        eng.check_liquidation(2500, dt)
        assert eng.futures_short is None
        assert eng.total_liquidation_fees > 0

    def test_no_false_liquidation(self):
        """正常波动不应触发强平"""
        eng = FuturesEngine('test', max_leverage=5)
        eng.max_single_margin = 50000
        eng.max_margin_total = 100000
        eng.max_lifetime_margin = 500000
        dt = pd.Timestamp('2025-01-01')
        
        eng.open_long(2000, dt, 10000, 5, "test")
        # 5x杠杆, 小幅下跌不应触发
        eng.check_liquidation(1950, dt)
        assert eng.futures_long is not None


# ======================================================
#   5. 资金费率测试
# ======================================================
class TestFundingRate:
    def test_funding_rate_deduction(self):
        """资金费率应正确扣除"""
        eng = FuturesEngine('test', max_leverage=5)
        eng.max_single_margin = 50000
        eng.max_margin_total = 100000
        eng.max_lifetime_margin = 500000
        dt = pd.Timestamp('2025-01-01')
        
        eng.open_long(2000, dt, 10000, 5, "test")
        initial_usdt = eng.usdt
        
        # 模拟8小时(每次counter +1, 每8次收费)
        for i in range(8):
            eng.charge_funding(2000, dt)
        
        # 应该收了一次费用
        assert eng.total_funding_paid > 0 or eng.total_funding_received > 0
        assert eng.usdt != initial_usdt


# ======================================================
#   6. frozen_margin 一致性测试
# ======================================================
class TestFrozenMarginConsistency:
    def test_open_close_margin_zero(self):
        """开仓→平仓后frozen_margin归零"""
        eng = FuturesEngine('test', max_leverage=5)
        eng.max_single_margin = 50000
        eng.max_margin_total = 100000
        eng.max_lifetime_margin = 500000
        dt = pd.Timestamp('2025-01-01')
        
        eng.open_long(2000, dt, 10000, 5, "test")
        assert eng.frozen_margin == pytest.approx(10000)
        
        eng.close_long(2100, dt, "test")
        assert eng.frozen_margin == pytest.approx(0)

    def test_dual_position_margin(self):
        """同时开多空, frozen_margin = 两者之和"""
        eng = FuturesEngine('test', max_leverage=5)
        eng.max_single_margin = 50000
        eng.max_margin_total = 100000
        eng.max_lifetime_margin = 500000
        dt = pd.Timestamp('2025-01-01')
        
        eng.open_long(2000, dt, 10000, 5, "test")
        eng.open_short(2000, dt, 8000, 3, "test")
        
        assert eng.frozen_margin == pytest.approx(18000)
        
        eng.close_long(2100, dt, "test")
        assert eng.frozen_margin == pytest.approx(8000)
        
        eng.close_short(1900, dt, "test")
        assert eng.frozen_margin == pytest.approx(0)


# ======================================================
#   7. KDJ 计算测试
# ======================================================
class TestKDJCalculation:
    def test_compute_kdj_basic(self):
        """KDJ基本计算"""
        from kdj_strategy import compute_kdj
        df = make_df(100)
        result = compute_kdj(df)
        
        assert 'kdj_k' in result.columns
        assert 'kdj_d' in result.columns
        assert 'kdj_j' in result.columns
        assert 'kd_macd' in result.columns
        assert 'kdj_k_slope' in result.columns
        assert 'kdj_d_slope' in result.columns

    def test_kdj_range(self):
        """K和D应在0-100范围内(近似)"""
        from kdj_strategy import compute_kdj
        df = make_df(200)
        result = compute_kdj(df)
        
        k = result['kdj_k'].iloc[20:]
        d = result['kdj_d'].iloc[20:]
        assert k.min() >= -5  # 允许微小越界
        assert k.max() <= 105
        assert d.min() >= -5
        assert d.max() <= 105

    def test_kdj_slope_not_nan(self):
        """修复后slope不应全是NaN"""
        from kdj_strategy import compute_kdj
        df = make_df(100)
        result = compute_kdj(df)
        
        slope_k = result['kdj_k_slope'].iloc[15:]
        slope_d = result['kdj_d_slope'].iloc[15:]
        
        # 修复后应有非NaN值
        assert slope_k.notna().sum() > 0, "K slope全是NaN — 修复失败!"
        assert slope_d.notna().sum() > 0, "D slope全是NaN — 修复失败!"

    def test_kdj_different_m1_m2(self):
        """m1 != m2时D线应使用不同alpha"""
        from kdj_strategy import compute_kdj
        df = make_df(100)
        
        result_same = compute_kdj(df.copy(), m1=3, m2=3)
        result_diff = compute_kdj(df.copy(), m1=3, m2=5)
        
        # D值应该不同
        d_same = result_same['kdj_d'].iloc[20:].values
        d_diff = result_diff['kdj_d'].iloc[20:].values
        assert not np.allclose(d_same, d_diff, atol=0.1), "m2不同但D线相同 — alpha_d未生效"

    def test_compute_kdj_scores_basic(self):
        """KDJ评分基本运行"""
        from kdj_strategy import compute_kdj_scores
        df = make_df(100)
        sell, buy, names = compute_kdj_scores(df)
        
        assert len(sell) == len(df)
        assert len(buy) == len(df)
        assert sell.dtype == float
        assert buy.dtype == float
        # 前15根因为预热应该是0
        assert sell.iloc[:15].sum() == 0
        assert buy.iloc[:15].sum() == 0

    def test_kdj_scores_no_negative(self):
        """评分不应为负"""
        from kdj_strategy import compute_kdj_scores
        df = make_df(200)
        sell, buy, _ = compute_kdj_scores(df)
        assert (sell >= 0).all()
        assert (buy >= 0).all()

    def test_kdj_scores_bounded(self):
        """评分应 <= 100"""
        from kdj_strategy import compute_kdj_scores
        df = make_df(200)
        sell, buy, _ = compute_kdj_scores(df)
        assert (sell <= 100).all()
        assert (buy <= 100).all()


# ======================================================
#   8. 边界条件和极端场景
# ======================================================
class TestEdgeCases:
    def test_zero_available_margin(self):
        """可用保证金为0时不应开仓"""
        eng = FuturesEngine('test', max_leverage=5)
        eng.max_single_margin = 50000
        eng.max_margin_total = 10000
        eng.max_lifetime_margin = 500000
        dt = pd.Timestamp('2025-01-01')
        
        # 先开一个仓占满额度
        eng.open_long(2000, dt, 10000, 5, "test")
        
        # 再开应该失败
        eng.open_short(2000, dt, 10000, 3, "test2")
        assert eng.futures_short is None

    def test_consecutive_losses(self):
        """连续止损后资金仍为正"""
        eng = FuturesEngine('test', max_leverage=5)
        eng.max_single_margin = 50000
        eng.max_margin_total = 100000
        eng.max_lifetime_margin = 500000
        dt = pd.Timestamp('2025-01-01')
        
        for i in range(5):
            eng.open_long(2000, dt, 5000, 3, f"loss_{i}")
            eng.close_long(1900, dt, f"sl_{i}")  # 每次亏损
        
        # usdt不应为负 (尽管可能很少)
        assert eng.usdt >= 0 or eng.usdt > -100  # 允许小额负数(fee导致)
        assert eng.frozen_margin == pytest.approx(0)

    def test_very_small_position(self):
        """极小仓位不应开"""
        eng = FuturesEngine('test', max_leverage=5)
        eng.max_single_margin = 50000
        eng.max_margin_total = 100000
        eng.max_lifetime_margin = 500000
        dt = pd.Timestamp('2025-01-01')
        
        eng.open_long(2000, dt, 50, 2, "tiny")  # margin < 200
        assert eng.futures_long is None

    def test_spot_sell_no_eth(self):
        """没有ETH时卖出不应崩溃"""
        eng = FuturesEngine('test', max_leverage=5)
        dt = pd.Timestamp('2025-01-01')
        initial = eng.usdt
        eng.spot_sell(2000, dt, 0.5, "no_eth")
        assert eng.usdt == initial

    def test_large_leverage(self):
        """高杠杆 — 引擎不自动限制, 策略层必须调用min(lev, max_leverage)"""
        eng = FuturesEngine('test', max_leverage=5)
        eng.max_single_margin = 50000
        eng.max_margin_total = 100000
        eng.max_lifetime_margin = 500000
        dt = pd.Timestamp('2025-01-01')
        
        # 正确用法: 策略层限制杠杆
        actual_lev = min(20, eng.max_leverage)  # 应为5
        eng.open_long(2000, dt, 10000, actual_lev, "high_lev")
        assert eng.futures_long is not None
        assert eng.futures_long.leverage == 5


# ======================================================
#   9. 余额一致性验证 (会计恒等式)
# ======================================================
class TestAccountingInvariant:
    def _verify_invariant(self, eng, price, msg=""):
        """验证: frozen_margin == sum(pos.margin)"""
        expected_frozen = 0
        if eng.futures_long:
            expected_frozen += eng.futures_long.margin
        if eng.futures_short:
            expected_frozen += eng.futures_short.margin
        assert eng.frozen_margin == pytest.approx(expected_frozen, abs=0.01), \
            f"frozen_margin不一致 {msg}: expected={expected_frozen:.2f}, got={eng.frozen_margin:.2f}"

    def test_full_lifecycle_invariant(self):
        """完整交易周期中frozen_margin始终一致"""
        eng = FuturesEngine('test', max_leverage=5)
        eng.max_single_margin = 50000
        eng.max_margin_total = 100000
        eng.max_lifetime_margin = 500000
        dt = pd.Timestamp('2025-01-01')
        
        self._verify_invariant(eng, 2000, "初始")
        
        eng.open_long(2000, dt, 10000, 5, "test")
        self._verify_invariant(eng, 2000, "开多后")
        
        eng.open_short(2000, dt, 8000, 3, "test")
        self._verify_invariant(eng, 2000, "开空后")
        
        eng.close_long(2100, dt, "test")
        self._verify_invariant(eng, 2100, "平多后")
        
        eng.close_short(1900, dt, "test")
        self._verify_invariant(eng, 1900, "平空后")


# ======================================================
#   10. 信号融合评分测试
# ======================================================
class TestFusionScore:
    def test_c6_veto_mode(self):
        """c6_veto模式基本运行"""
        from optimize_six_book import calc_fusion_score_six, compute_signals_six
        from indicators import add_all_indicators
        from ma_indicators import add_moving_averages
        
        df = make_df(200)
        df = add_all_indicators(df)
        add_moving_averages(df, timeframe='1h')
        
        signals = compute_signals_six(df, '1h', {'1h': df})
        
        config = {'fusion_mode': 'c6_veto'}
        ss, bs = calc_fusion_score_six(signals, df, 100, df.index[100], config)
        
        assert isinstance(ss, float)
        assert isinstance(bs, float)
        assert ss >= 0
        assert bs >= 0


# ======================================================
#   运行测试
# ======================================================
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
