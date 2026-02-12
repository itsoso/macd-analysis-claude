"""
实盘交易系统单元测试
覆盖: 配置管理 / 风控 / 订单管理 / 信号生成 / 绩效追踪 / 引擎集成
"""

import json
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from live_config import (
    LiveTradingConfig, TradingPhase, StrategyConfig,
    RiskConfig, APIConfig, TelegramConfig,
    create_default_config, PHASE_CONFIGS,
)
from trading_logger import TradingLogger, TRADE, SIGNAL, RISK, BALANCE, FUNDING
from notifier import TelegramNotifier, DummyNotifier, create_notifier
from risk_manager import RiskManager, RiskState
from order_manager import PaperOrderManager, BinanceAPIError
from performance_tracker import PerformanceTracker
from live_trading_engine import Position


# ============================================================
# 1. 配置管理测试
# ============================================================
class TestConfig:
    """配置管理测试"""

    def test_default_config_paper(self):
        """Paper 阶段默认配置"""
        config = create_default_config(TradingPhase.PAPER)
        assert config.phase == TradingPhase.PAPER
        assert config.execute_trades is False
        assert config.initial_capital == 100000

    def test_default_config_small_live(self):
        """Small Live 阶段默认配置"""
        config = create_default_config(TradingPhase.SMALL_LIVE)
        assert config.phase == TradingPhase.SMALL_LIVE
        assert config.execute_trades is True
        assert config.initial_capital == 500
        assert config.risk.max_leverage == 2

    def test_leverage_safety_clamp(self):
        """杠杆安全限制"""
        config = create_default_config(TradingPhase.SMALL_LIVE)
        config.strategy.leverage = 10  # 尝试设置过高杠杆
        config._validate()
        assert config.strategy.leverage == 2  # 被限制为2x

    def test_capital_safety_clamp(self):
        """资金安全限制"""
        config = create_default_config(TradingPhase.SMALL_LIVE)
        config.initial_capital = 5000  # 超过 Phase 3 限制
        config._validate()
        assert config.initial_capital == 2000  # 被限制为 $2000

    def test_paper_no_execute(self):
        """Paper 模式强制不执行交易"""
        config = create_default_config(TradingPhase.PAPER)
        config.execute_trades = True  # 尝试强制开启
        config._validate()
        assert config.execute_trades is False

    def test_strategy_config_fields(self):
        """策略配置字段完整性"""
        cfg = StrategyConfig()
        assert cfg.symbol == "ETHUSDT"
        assert cfg.fusion_mode == "c6_veto_4"
        assert cfg.short_sl < 0
        assert cfg.long_tp > 0
        assert cfg.leverage > 0

    def test_all_phases_have_config(self):
        """所有阶段都有预设配置"""
        for phase in TradingPhase:
            assert phase in PHASE_CONFIGS

    def test_save_and_load_template(self):
        """配置模板保存和加载"""
        config = create_default_config(TradingPhase.PAPER)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name

        try:
            config.save_template(path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert "phase" in data
            assert "api" in data
            assert "telegram" in data
        finally:
            os.unlink(path)


# ============================================================
# 2. 风控管理测试
# ============================================================
class TestRiskManager:
    """风控管理器测试"""

    def _create_rm(self, initial=10000, **kwargs):
        config = RiskConfig(**kwargs)
        return RiskManager(config, initial)

    def test_can_open_basic(self):
        """基本开仓检查"""
        rm = self._create_rm()
        ok, reason = rm.can_open_position("LONG", 1000, 10000, 0)
        assert ok is True

    def test_single_position_limit(self):
        """单仓限制"""
        rm = self._create_rm(initial=10000, max_single_position_pct=0.10)
        ok, reason = rm.can_open_position("LONG", 2000, 10000, 0)
        assert ok is False
        assert "单仓" in reason

    def test_total_margin_limit(self):
        """总仓位限制"""
        rm = self._create_rm(initial=10000, max_total_margin_pct=0.30)
        ok, reason = rm.can_open_position("LONG", 2000, 10000, 2000)
        assert ok is False

    def test_reserve_limit(self):
        """资金储备限制"""
        rm = self._create_rm(initial=10000, reserve_pct=0.50,
                             max_total_margin_pct=0.80)
        # equity=10000, frozen=4000, 再开 2000 → 可用=10000-4000-2000=4000 < 50%=5000
        ok, reason = rm.can_open_position("LONG", 2000, 10000, 4000)
        assert ok is False
        assert "储备" in reason

    def test_kill_switch_blocks_trading(self):
        """Kill Switch 阻止开仓"""
        rm = self._create_rm()
        rm.activate_kill_switch("测试")
        ok, reason = rm.can_open_position("LONG", 1000, 10000, 0)
        assert ok is False
        assert "Kill Switch" in reason

    def test_pause_blocks_trading(self):
        """暂停状态阻止开仓"""
        rm = self._create_rm()
        rm.state.is_paused = True
        rm.state.pause_reason = "测试暂停"
        ok, reason = rm.can_open_position("LONG", 1000, 10000, 0)
        assert ok is False

    def test_consecutive_loss_circuit_breaker(self):
        """连续亏损熔断"""
        rm = self._create_rm(max_consecutive_losses=3)
        for _ in range(3):
            rm.on_trade_closed(pnl=-100, fee=5, current_equity=9500)
        assert rm.state.is_paused is True
        assert "CONSECUTIVE_LOSS" in rm.state.pause_reason

    def test_daily_loss_circuit_breaker(self):
        """日亏损熔断"""
        rm = self._create_rm(initial=10000, max_daily_loss_pct=0.05)
        rm.on_trade_closed(pnl=-600, fee=5, current_equity=9400)
        assert rm.state.is_paused is True
        assert "DAILY" in rm.state.pause_reason

    def test_drawdown_circuit_breaker(self):
        """回撤熔断"""
        rm = self._create_rm(initial=10000, max_drawdown_pct=0.10)
        rm.state.peak_equity = 10000
        rm.on_trade_closed(pnl=-1100, fee=0, current_equity=8900)
        assert rm.state.is_paused is True

    def test_resume_trading(self):
        """恢复交易"""
        rm = self._create_rm(max_consecutive_losses=2)
        rm.on_trade_closed(pnl=-100, fee=0, current_equity=9900)
        rm.on_trade_closed(pnl=-100, fee=0, current_equity=9800)
        assert rm.state.is_paused is True
        rm.resume_trading("人工审查")
        assert rm.state.is_paused is False
        assert rm.state.consecutive_losses == 0

    def test_constrain_leverage(self):
        """杠杆约束"""
        rm = self._create_rm(max_leverage=3)
        assert rm.constrain_leverage(5) == 3
        assert rm.constrain_leverage(2) == 2

    def test_constrain_margin(self):
        """保证金约束"""
        rm = self._create_rm(
            initial=10000,
            max_single_position_pct=0.20,
            max_total_margin_pct=0.50,
            reserve_pct=0.30,
        )
        margin = rm.constrain_margin(5000, 10000, 0)
        assert margin <= 2000  # max single = 20%

    def test_slippage_check_reject(self):
        """滑点过大被拒绝"""
        rm = self._create_rm(max_slippage_pct=0.003)
        # 10/2500 = 0.4% > 0.3% 限制
        ok, slip = rm.check_slippage(2500, 2510, "LONG")
        assert ok is False
        assert slip > 0.003

    def test_slippage_check_accept(self):
        """滑点在可接受范围"""
        rm = self._create_rm(max_slippage_pct=0.005)
        # 5/2500 = 0.2% < 0.5% 限制
        ok, slip = rm.check_slippage(2500, 2505, "LONG")
        assert ok is True

    def test_state_persistence(self):
        """状态持久化"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            rm1 = RiskManager(RiskConfig(), 10000, state_file=path)
            rm1.on_trade_closed(pnl=500, fee=10, current_equity=10490)
            rm1.on_trade_closed(pnl=-200, fee=10, current_equity=10280)

            rm2 = RiskManager(RiskConfig(), 10000, state_file=path)
            assert rm2.state.total_trades == 2
            assert rm2.state.total_wins == 1
            assert rm2.state.total_losses == 1
        finally:
            os.unlink(path)

    def test_win_resets_consecutive(self):
        """盈利重置连续亏损计数"""
        rm = self._create_rm(max_consecutive_losses=5)
        rm.on_trade_closed(pnl=-100, fee=0, current_equity=9900)
        rm.on_trade_closed(pnl=-100, fee=0, current_equity=9800)
        assert rm.state.consecutive_losses == 2
        rm.on_trade_closed(pnl=200, fee=0, current_equity=10000)
        assert rm.state.consecutive_losses == 0


# ============================================================
# 3. Paper 订单管理测试
# ============================================================
class TestPaperOrderManager:
    """Paper 订单管理器测试"""

    def test_simulate_open_long(self):
        """模拟开多"""
        om = PaperOrderManager(initial_capital=100000)
        result = om.market_open_long("ETHUSDT", 10.0, "test")
        assert result["status"] == "FILLED"
        assert result["paper"] is True
        assert "PAPER" in result["orderId"]

    def test_simulate_open_short(self):
        """模拟开空"""
        om = PaperOrderManager(initial_capital=100000)
        result = om.market_open_short("ETHUSDT", 5.0, "test")
        assert result["status"] == "FILLED"

    def test_order_history(self):
        """订单历史记录"""
        om = PaperOrderManager(initial_capital=100000)
        om.market_open_long("ETHUSDT", 10.0, "test1")
        om.market_close_long("ETHUSDT", 10.0, "test2")
        history = om.get_order_history()
        assert len(history) == 2

    def test_test_connection(self):
        """连接测试"""
        om = PaperOrderManager()
        assert om.test_connection() is True


# ============================================================
# 4. Position 测试
# ============================================================
class TestPosition:
    """持仓对象测试"""

    def test_long_pnl_positive(self):
        """多仓盈利"""
        pos = Position("LONG", 2500, 10, 5000, 5, "2025-01-01")
        pnl = pos.calc_pnl(2600)
        assert pnl == 1000  # (2600-2500) * 10

    def test_long_pnl_negative(self):
        """多仓亏损"""
        pos = Position("LONG", 2500, 10, 5000, 5, "2025-01-01")
        pnl = pos.calc_pnl(2400)
        assert pnl == -1000

    def test_short_pnl_positive(self):
        """空仓盈利"""
        pos = Position("SHORT", 2500, 10, 5000, 5, "2025-01-01")
        pnl = pos.calc_pnl(2400)
        assert pnl == 1000

    def test_short_pnl_negative(self):
        """空仓亏损"""
        pos = Position("SHORT", 2500, 10, 5000, 5, "2025-01-01")
        pnl = pos.calc_pnl(2600)
        assert pnl == -1000

    def test_pnl_ratio(self):
        """盈亏比例"""
        pos = Position("LONG", 2500, 10, 5000, 5, "2025-01-01")
        ratio = pos.calc_pnl_ratio(2750)
        assert ratio == pytest.approx(0.5, abs=0.01)  # 2500/5000 = 50%

    def test_to_dict(self):
        """序列化"""
        pos = Position("LONG", 2500, 10, 5000, 5, "2025-01-01")
        d = pos.to_dict()
        assert d["side"] == "LONG"
        assert d["entry_price"] == 2500
        assert d["quantity"] == 10
        assert d["leverage"] == 5


# ============================================================
# 5. 绩效追踪测试
# ============================================================
class TestPerformanceTracker:
    """绩效追踪器测试"""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _create_tracker(self):
        return PerformanceTracker(initial_capital=10000, data_dir=self._tmpdir)

    def test_record_trade(self):
        """记录交易"""
        t = self._create_tracker()
        t.record_trade({
            "action": "CLOSE_LONG", "side": "LONG",
            "price": 2600, "qty": 10, "pnl": 500, "fee": 10,
        })
        assert len(t.trades) == 1

    def test_summary_basic(self):
        """基本统计"""
        t = self._create_tracker()
        t.record_trade({"action": "CLOSE_LONG", "pnl": 500, "fee": 10})
        t.record_trade({"action": "CLOSE_SHORT", "pnl": -200, "fee": 10})
        t.record_trade({"action": "CLOSE_LONG", "pnl": 300, "fee": 10})

        s = t.get_summary()
        assert s["total_trades"] == 3
        assert s["wins"] == 2
        assert s["losses"] == 1
        assert s["win_rate"] == pytest.approx(2 / 3, abs=0.01)
        assert s["total_pnl"] == 600  # 500 - 200 + 300

    def test_drawdown_tracking(self):
        """回撤追踪"""
        t = self._create_tracker()
        t.record_equity(10000)
        t.record_equity(10500)
        t.record_equity(9500)
        # drawdown = (10500-9500)/10500 ≈ 9.52%
        assert t.max_drawdown > 0.09

    def test_slippage_tracking(self):
        """滑点追踪"""
        t = self._create_tracker()
        t.record_trade({
            "action": "CLOSE_LONG",
            "pnl": 100, "fee": 5,
            "expected_price": 2500,
            "actual_price": 2495,
        })
        assert len(t.slippage_records) == 1
        assert t.slippage_records[0]["slippage_pct"] == pytest.approx(0.002, abs=0.001)

    def test_open_trades_not_in_summary(self):
        """OPEN 交易不计入统计"""
        t = self._create_tracker()
        t.record_trade({"action": "OPEN_LONG", "pnl": 0, "fee": 10})
        s = t.get_summary()
        assert s["total_trades"] == 0  # OPEN 不算

    def test_persistence(self):
        """持久化"""
        t1 = PerformanceTracker(10000, self._tmpdir)
        t1.record_trade({"action": "CLOSE_LONG", "pnl": 500, "fee": 10})
        t1._save()

        t2 = PerformanceTracker(10000, self._tmpdir)
        assert len(t2.trades) == 1
        assert t2.total_fees == 10


# ============================================================
# 6. 日志系统测试
# ============================================================
class TestTradingLogger:
    """日志系统测试"""

    def test_create_logger(self):
        """创建日志器"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradingLogger(log_dir=tmpdir, name="test_logger")
            assert logger is not None

    def test_log_trade(self):
        """记录交易日志"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradingLogger(log_dir=tmpdir, name="test_trade")
            logger.log_trade(
                action="OPEN_LONG", symbol="ETHUSDT", side="LONG",
                price=2500, qty=10, margin=5000, leverage=5,
                fee=25, reason="test"
            )
            assert logger._trade_count == 1

    def test_log_signal(self):
        """记录信号日志"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradingLogger(log_dir=tmpdir, name="test_signal")
            logger.log_signal(
                sell_score=35.5, buy_score=12.3,
                components={"div_s": 30, "ma_s": 10, "kdj_s": 5,
                            "bb_s": 3, "vp_s": 2, "cs_s": 0,
                            "div_b": 5, "ma_b": 3, "kdj_b": 2,
                            "bb_b": 1, "vp_b": 0, "cs_b": 0},
                conflict=False, action_taken="OPEN_SHORT"
            )
            assert logger._signal_count == 1

    def test_log_risk(self):
        """记录风险日志"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradingLogger(log_dir=tmpdir, name="test_risk")
            logger.log_risk("STOP_LOSS", "止损触发", 0.15, 0.30, "CLOSE")
            assert logger._risk_event_count == 1

    def test_log_balance(self):
        """记录余额日志"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradingLogger(log_dir=tmpdir, name="test_balance")
            logger.log_balance(
                total_equity=10500, usdt=8000,
                unrealized_pnl=500, frozen_margin=2000,
                available_margin=6000
            )

    def test_custom_levels(self):
        """自定义日志级别"""
        assert TRADE == 25
        assert SIGNAL == 24
        assert RISK == 23
        assert BALANCE == 22
        assert FUNDING == 21


# ============================================================
# 7. 通知系统测试
# ============================================================
class TestNotifier:
    """通知系统测试"""

    def test_dummy_notifier(self):
        """空通知器不抛异常"""
        n = DummyNotifier()
        n.notify_trade("OPEN_LONG", "ETHUSDT", "LONG", 2500, 10)
        n.notify_risk("STOP_LOSS", "test")
        n.notify_daily_summary("2025-01-01", 10000, 500, 0.05, 10, 7, 3)
        n.notify_system("START", "test")
        n.notify_error(Exception("test"))
        assert n.test_connection() is True

    def test_create_notifier_disabled(self):
        """禁用时创建空通知器"""
        cfg = TelegramConfig(enabled=False)
        n = create_notifier(cfg)
        assert isinstance(n, DummyNotifier)

    def test_telegram_notifier_disabled(self):
        """Telegram 无 token 时自动禁用"""
        n = TelegramNotifier(bot_token="", chat_id="", enabled=True)
        assert n.enabled is False


# ============================================================
# 8. RiskState 序列化测试
# ============================================================
class TestRiskState:
    """风控状态序列化"""

    def test_to_dict_and_back(self):
        """序列化/反序列化"""
        state = RiskState()
        state.peak_equity = 15000
        state.daily_pnl = -200
        state.consecutive_losses = 3
        state.total_trades = 50
        state.total_pnl = 2500

        d = state.to_dict()
        state2 = RiskState.from_dict(d)
        assert state2.peak_equity == 15000
        assert state2.daily_pnl == -200
        assert state2.consecutive_losses == 3
        assert state2.total_trades == 50


# ============================================================
# 9. API 错误测试
# ============================================================
class TestBinanceAPIError:
    """API 错误"""

    def test_error_str(self):
        e = BinanceAPIError(-1021, "Timestamp not valid")
        assert "-1021" in str(e)
        assert "Timestamp" in str(e)


# ============================================================
# 10. 集成测试 - 完整交易流程
# ============================================================
class TestIntegration:
    """集成测试"""

    def test_full_trade_cycle(self):
        """完整交易周期: 风控检查 → 开仓 → 持仓 → 平仓"""
        # 1. 创建风控
        rm = RiskManager(
            RiskConfig(max_leverage=5, max_single_position_pct=0.20),
            initial_capital=10000
        )

        # 2. 检查开仓
        ok, _ = rm.can_open_position("SHORT", 1500, 10000, 0)
        assert ok is True

        # 3. 创建持仓
        pos = Position("SHORT", 2500, 6, 1500, 5, "2025-01-01")
        assert pos.calc_pnl(2400) == 600   # 盈利
        assert pos.calc_pnl(2600) == -600  # 亏损

        # 4. 平仓并更新风控
        pnl = pos.calc_pnl(2400)  # 600
        rm.on_trade_closed(pnl=pnl, fee=10, current_equity=10590)

        assert rm.state.total_trades == 1
        assert rm.state.total_wins == 1
        assert rm.state.consecutive_losses == 0

    def test_circuit_breaker_recovery(self):
        """熔断 → 恢复 → 继续交易"""
        rm = RiskManager(
            RiskConfig(max_consecutive_losses=2),
            initial_capital=10000
        )

        # 连续亏损触发熔断
        rm.on_trade_closed(-200, 5, 9795)
        rm.on_trade_closed(-200, 5, 9590)
        assert rm.state.is_paused is True

        # 恢复
        rm.resume_trading("审查完毕")
        assert rm.state.is_paused is False

        # 可以继续开仓
        ok, _ = rm.can_open_position("LONG", 1000, 9590, 0)
        assert ok is True

    def test_performance_compare(self):
        """绩效对比"""
        with tempfile.TemporaryDirectory() as tmpdir:
            t = PerformanceTracker(10000, tmpdir)
            t.record_trade({"action": "CLOSE_LONG", "pnl": 500, "fee": 10})
            t.record_trade({"action": "CLOSE_SHORT", "pnl": 300, "fee": 10})
            t.record_equity(10780)

            backtest = {
                "total_return": 0.10,
                "win_rate": 0.65,
                "max_drawdown": 0.08,
                "total_trades": 20,
            }
            comparison = t.compare_with_backtest(backtest)
            assert "return" in comparison
            assert "win_rate" in comparison
            assert "slippage" in comparison


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
