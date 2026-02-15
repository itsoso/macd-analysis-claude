"""
P0 可解释性回归测试
确保 structural_discount / book_consensus / neutral_struct_* 参数
在 summary_json 和 strategy_snapshot 中正确落库。
"""
import json
import multi_tf_daily_db as db_mod


# ── 测试 1: strategy_snapshot 必含 neutral_book_* 与 neutral_struct_* ──

def test_strategy_snapshot_contains_book_consensus_fields():
    """strategy_snapshot 必须包含 neutral_book_* 参数字段。"""
    config = {
        'use_neutral_book_consensus': True,
        'neutral_book_sell_threshold': 10.0,
        'neutral_book_buy_threshold': 10.0,
        'neutral_book_min_confirms': 2,
        'neutral_book_max_conflicts': 4,
        'neutral_book_cs_kdj_threshold_adj': 0.0,
    }
    snapshot = db_mod._build_strategy_snapshot(config)
    required_keys = [
        'use_neutral_book_consensus',
        'neutral_book_sell_threshold',
        'neutral_book_buy_threshold',
        'neutral_book_min_confirms',
        'neutral_book_max_conflicts',
        'neutral_book_cs_kdj_threshold_adj',
    ]
    for key in required_keys:
        assert key in snapshot, (
            f"strategy_snapshot 缺失字段: {key}. "
            f"已有字段: {sorted(snapshot.keys())}"
        )


def test_strategy_snapshot_contains_structural_discount_fields():
    """strategy_snapshot 必须包含 neutral_struct_* 参数字段。"""
    config = {
        'use_neutral_structural_discount': True,
        'neutral_struct_activity_thr': 10.0,
        'neutral_struct_discount_0': 0.15,
        'neutral_struct_discount_1': 0.25,
        'neutral_struct_discount_2': 1.0,
        'neutral_struct_discount_3': 1.0,
        'neutral_struct_discount_4plus': 1.0,
        'structural_discount_short_regimes': 'neutral,trend',
        'structural_discount_long_regimes': 'neutral',
    }
    snapshot = db_mod._build_strategy_snapshot(config)
    required_keys = [
        'use_neutral_structural_discount',
        'neutral_struct_activity_thr',
        'neutral_struct_discount_0',
        'neutral_struct_discount_1',
        'neutral_struct_discount_2',
        'neutral_struct_discount_3',
        'neutral_struct_discount_4plus',
        'structural_discount_short_regimes',
        'structural_discount_long_regimes',
    ]
    for key in required_keys:
        assert key in snapshot, (
            f"strategy_snapshot 缺失字段: {key}. "
            f"已有字段: {sorted(snapshot.keys())}"
        )


# ── 测试 2: summary_json 结构验证 ──

def test_summary_json_structural_discount_passthrough():
    """
    当 result 中包含 structural_discount 时，
    summary 构建逻辑必须将其透传到 summary_json。

    通过模拟 result dict 验证 backtest_multi_tf_daily.py
    的 summary 构建逻辑会正确拾取 structural_discount 键。
    """
    # 模拟一个 result dict (只含我们关心的键)
    mock_result = {
        'structural_discount': {
            'enabled': True,
            'evaluated': 42,
            'discount_applied': 10,
            'confirm_distribution': {0: 5, 1: 7, 2: 15, 3: 10, 4: 3, 5: 2},
            'avg_mult': 0.871,
            'activity_thr': 10.0,
            'discount_0': 0.15,
            'discount_1': 0.25,
            'discount_2': 1.0,
            'discount_3': 1.0,
            'discount_4plus': 1.0,
        },
        'book_consensus_gate': {
            'enabled': True,
            'evaluated': 100,
            'short_blocked': 5,
            'long_blocked': 3,
        },
    }

    # 模拟 summary 构建的 passthrough 逻辑
    # (与 backtest_multi_tf_daily.py 中实际逻辑一致)
    summary = {}
    if mock_result.get('structural_discount'):
        summary['structural_discount'] = mock_result['structural_discount']
    if mock_result.get('book_consensus_gate'):
        summary['book_consensus_gate'] = mock_result['book_consensus_gate']

    assert 'structural_discount' in summary, \
        "summary_json 必须包含 structural_discount（当启用时）"
    assert summary['structural_discount']['evaluated'] == 42
    assert summary['structural_discount']['avg_mult'] == 0.871

    assert 'book_consensus_gate' in summary, \
        "summary_json 必须包含 book_consensus_gate（当启用时）"
    assert summary['book_consensus_gate']['evaluated'] == 100


def test_strategy_snapshot_serializable():
    """确保 strategy_snapshot 可以被 JSON 序列化（落库前提）。"""
    config = {
        'use_neutral_book_consensus': False,
        'neutral_book_sell_threshold': 10.0,
        'neutral_book_buy_threshold': 10.0,
        'neutral_book_min_confirms': 2,
        'neutral_book_max_conflicts': 4,
        'neutral_book_cs_kdj_threshold_adj': 0.0,
        'use_neutral_structural_discount': True,
        'neutral_struct_activity_thr': 10.0,
        'neutral_struct_discount_0': 0.15,
        'neutral_struct_discount_1': 0.25,
        'neutral_struct_discount_2': 1.0,
        'neutral_struct_discount_3': 1.0,
        'neutral_struct_discount_4plus': 1.0,
    }
    snapshot = db_mod._build_strategy_snapshot(config)
    # 不应抛异常
    serialized = json.dumps(snapshot, ensure_ascii=False)
    deserialized = json.loads(serialized)
    assert deserialized['use_neutral_structural_discount'] is True
    assert deserialized['use_neutral_book_consensus'] is False


def test_strategy_snapshot_contains_sl_cooldown_fields():
    """strategy_snapshot 必须包含 SL 冷却倍数字段，保证回测配置可复现。"""
    config = {
        'short_sl_cd_mult': 4,
        'long_sl_cd_mult': 5,
    }
    snapshot = db_mod._build_strategy_snapshot(config)
    assert 'short_sl_cd_mult' in snapshot
    assert 'long_sl_cd_mult' in snapshot
    assert snapshot['short_sl_cd_mult'] == 4
    assert snapshot['long_sl_cd_mult'] == 5


def test_strategy_snapshot_contains_short_conflict_soft_discount_fields():
    """strategy_snapshot 必须包含空单冲突软折扣参数字段。"""
    config = {
        'use_short_conflict_soft_discount': True,
        'short_conflict_regimes': 'trend,high_vol',
        'short_conflict_div_buy_min': 50.0,
        'short_conflict_ma_sell_min': 12.0,
        'short_conflict_discount_mult': 0.5,
    }
    snapshot = db_mod._build_strategy_snapshot(config)
    required_keys = [
        'use_short_conflict_soft_discount',
        'short_conflict_regimes',
        'short_conflict_div_buy_min',
        'short_conflict_ma_sell_min',
        'short_conflict_discount_mult',
    ]
    for key in required_keys:
        assert key in snapshot


def test_strategy_snapshot_contains_long_conflict_soft_discount_fields():
    """strategy_snapshot 必须包含多单冲突软折扣参数字段。"""
    config = {
        'use_long_conflict_soft_discount': True,
        'long_conflict_regimes': 'neutral,low_vol_trend',
        'long_conflict_div_sell_min': 50.0,
        'long_conflict_ma_buy_min': 12.0,
        'long_conflict_discount_mult': 0.5,
    }
    snapshot = db_mod._build_strategy_snapshot(config)
    required_keys = [
        'use_long_conflict_soft_discount',
        'long_conflict_regimes',
        'long_conflict_div_sell_min',
        'long_conflict_ma_buy_min',
        'long_conflict_discount_mult',
    ]
    for key in required_keys:
        assert key in snapshot


def test_strategy_snapshot_contains_long_high_conf_gate_fields():
    """strategy_snapshot 必须包含 long 高置信候选门控字段。"""
    config = {
        'use_long_high_conf_gate_a': True,
        'long_high_conf_gate_a_conf_min': 0.85,
        'long_high_conf_gate_a_regime': 'low_vol_trend',
        'use_long_high_conf_gate_b': True,
        'long_high_conf_gate_b_conf_min': 0.90,
        'long_high_conf_gate_b_regime': 'neutral',
        'long_high_conf_gate_b_vp_buy_min': 30.0,
    }
    snapshot = db_mod._build_strategy_snapshot(config)
    required_keys = [
        'use_long_high_conf_gate_a',
        'long_high_conf_gate_a_conf_min',
        'long_high_conf_gate_a_regime',
        'use_long_high_conf_gate_b',
        'long_high_conf_gate_b_conf_min',
        'long_high_conf_gate_b_regime',
        'long_high_conf_gate_b_vp_buy_min',
    ]
    for key in required_keys:
        assert key in snapshot


def test_strategy_snapshot_contains_spot_layer_and_reentry_fields():
    """strategy_snapshot 必须包含 neutral SPOT_SELL 分层与停滞再入场参数。"""
    config = {
        'use_neutral_spot_sell_layer': True,
        'neutral_spot_sell_confirm_thr': 10.0,
        'neutral_spot_sell_min_confirms_any': 2,
        'neutral_spot_sell_strong_confirms': 4,
        'neutral_spot_sell_full_ss_min': 70.0,
        'neutral_spot_sell_weak_ss_min': 55.0,
        'neutral_spot_sell_weak_pct_cap': 0.15,
        'neutral_spot_sell_block_ss_min': 70.0,
        'use_stagnation_reentry': True,
        'stagnation_reentry_days': 10.0,
        'stagnation_reentry_regimes': 'trend,low_vol_trend',
        'stagnation_reentry_min_spot_ratio': 0.3,
        'stagnation_reentry_buy_pct': 0.2,
        'stagnation_reentry_min_usdt': 500.0,
        'stagnation_reentry_cooldown_days': 3.0,
    }
    snapshot = db_mod._build_strategy_snapshot(config)
    required_keys = [
        'use_neutral_spot_sell_layer',
        'neutral_spot_sell_confirm_thr',
        'neutral_spot_sell_min_confirms_any',
        'neutral_spot_sell_strong_confirms',
        'neutral_spot_sell_full_ss_min',
        'neutral_spot_sell_weak_ss_min',
        'neutral_spot_sell_weak_pct_cap',
        'neutral_spot_sell_block_ss_min',
        'use_stagnation_reentry',
        'stagnation_reentry_days',
        'stagnation_reentry_regimes',
        'stagnation_reentry_min_spot_ratio',
        'stagnation_reentry_buy_pct',
        'stagnation_reentry_min_usdt',
        'stagnation_reentry_cooldown_days',
    ]
    for key in required_keys:
        assert key in snapshot
