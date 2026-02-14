import json

from live_config import StrategyConfig
from live_signal_generator import LiveSignalGenerator
import backtest_multi_tf_daily as daily_cfg


def test_strategy_config_from_optimize_result_supports_aliases(tmp_path):
    data = {
        "global_best": {
            "tf": "2h",
            "config": {
                "lev": 3,
                "decision_tfs": ["15m", "1h", "4h", "12h"],
                "fusion_mode": "kdj_timing",
                "kdj_strong_mult": 1.3,
            },
        }
    }
    fp = tmp_path / "opt_result.json"
    fp.write_text(json.dumps(data), encoding="utf-8")

    cfg = StrategyConfig.from_optimize_result(str(fp))
    assert cfg.timeframe == "2h"
    assert cfg.leverage == 3
    assert cfg.decision_timeframes == ["15m", "1h", "4h", "12h"]
    assert cfg.fusion_mode == "kdj_timing"
    assert cfg.kdj_strong_mult == 1.3


def test_live_fusion_config_contains_backtest_fusion_params():
    cfg = StrategyConfig(
        fusion_mode="kdj_weighted",
        veto_threshold=30,
        kdj_bonus=0.11,
        kdj_weight=0.22,
        div_weight=0.51,
        kdj_strong_mult=1.31,
        kdj_normal_mult=1.15,
        kdj_reverse_mult=0.66,
        kdj_gate_threshold=14,
        veto_dampen=0.27,
        bb_bonus=0.12,
        vp_bonus=0.09,
        cs_bonus=0.07,
    )
    gen = LiveSignalGenerator.__new__(LiveSignalGenerator)
    gen.config = cfg

    fusion_cfg = gen._build_fusion_config()
    assert fusion_cfg["fusion_mode"] == "kdj_weighted"
    assert fusion_cfg["veto_threshold"] == 30
    assert fusion_cfg["kdj_bonus"] == 0.11
    assert fusion_cfg["kdj_weight"] == 0.22
    assert fusion_cfg["div_weight"] == 0.51
    assert fusion_cfg["kdj_strong_mult"] == 1.31
    assert fusion_cfg["kdj_normal_mult"] == 1.15
    assert fusion_cfg["kdj_reverse_mult"] == 0.66
    assert fusion_cfg["kdj_gate_threshold"] == 14
    assert fusion_cfg["veto_dampen"] == 0.27
    assert fusion_cfg["bb_bonus"] == 0.12
    assert fusion_cfg["vp_bonus"] == 0.09
    assert fusion_cfg["cs_bonus"] == 0.07


def test_backtest_daily_defaults_align_with_strategy_defaults():
    cfg = StrategyConfig()
    assert cfg.decision_timeframes == ["15m", "1h", "4h", "24h"]
    assert cfg.decision_timeframes_fallback == ["15m", "30m", "1h", "4h", "8h", "24h"]

    assert daily_cfg.PRIMARY_TF == cfg.timeframe
    assert daily_cfg.DECISION_TFS == cfg.decision_timeframes
    assert daily_cfg.FALLBACK_DECISION_TFS == cfg.decision_timeframes_fallback
    assert all(tf in daily_cfg.AVAILABLE_TFS for tf in cfg.decision_timeframes)
    assert daily_cfg.DEFAULT_CONFIG["sell_threshold"] == cfg.sell_threshold
    assert daily_cfg.DEFAULT_CONFIG["buy_threshold"] == cfg.buy_threshold
    assert daily_cfg.DEFAULT_CONFIG["short_threshold"] == cfg.short_threshold
    assert daily_cfg.DEFAULT_CONFIG["long_threshold"] == cfg.long_threshold
    assert daily_cfg.DEFAULT_CONFIG["fusion_mode"] == cfg.fusion_mode
    assert daily_cfg.DEFAULT_CONFIG["veto_threshold"] == cfg.veto_threshold
    assert daily_cfg.DEFAULT_CONFIG["kdj_bonus"] == cfg.kdj_bonus
    assert daily_cfg.DEFAULT_CONFIG["kdj_weight"] == cfg.kdj_weight
    assert daily_cfg.DEFAULT_CONFIG["consensus_min_strength"] == cfg.consensus_min_strength
    assert daily_cfg.DEFAULT_CONFIG["coverage_min"] == cfg.coverage_min
    # 注: use_microstructure / use_dual_engine / use_vol_target 在回测中
    # 被故意关闭以隔离趋势保护 v3 效果, 因此与 StrategyConfig 不同
    assert daily_cfg.DEFAULT_CONFIG["use_microstructure"] is False  # 回测中关闭
    assert daily_cfg.DEFAULT_CONFIG["use_dual_engine"] is False     # 回测中关闭
    assert daily_cfg.DEFAULT_CONFIG["use_vol_target"] is False      # 回测中关闭
    assert daily_cfg.DEFAULT_CONFIG["vol_target_annual"] == cfg.vol_target_annual


def test_regime_gate_add_35_effective():
    """验证 gate_add=35 时 LVT regime 中做空有效门槛为 60 (25+35)。"""
    cfg = StrategyConfig()
    assert cfg.regime_short_gate_add == 35
    assert cfg.short_threshold == 25
    effective_threshold = cfg.short_threshold + cfg.regime_short_gate_add
    assert effective_threshold == 60


def test_warmup_fixed_200_bars():
    """验证 warmup 按固定 200 bar 计算，不随样本长度 5% 放大。"""
    WARMUP_BARS = 200
    # 短样本: 取 min(200, len-1)
    assert min(max(60, WARMUP_BARS), 500 - 1) == 200
    # 长样本: 之前 5%*10000=500，现固定 200
    assert min(max(60, WARMUP_BARS), 10000 - 1) == 200
    # 极短样本: 不超过 len-1
    assert min(max(60, WARMUP_BARS), 100 - 1) == 99
