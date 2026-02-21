from hotcoin.runner import _apply_state_hysteresis


def test_degraded_recovery_requires_multiple_cycles():
    state = "degraded"
    ok_cycles = 0

    state, reasons, ok_cycles, pending = _apply_state_hysteresis(
        current_state=state,
        current_recovery_ok_cycles=ok_cycles,
        raw_state="tradeable",
        raw_reasons=[],
        degraded_recovery_cycles=3,
        blocked_recovery_cycles=6,
        allow_recovery_progress=True,
    )
    assert state == "degraded"
    assert ok_cycles == 1
    assert pending is not None
    assert "recovery_pending:1/3" in reasons

    state, reasons, ok_cycles, pending = _apply_state_hysteresis(
        current_state=state,
        current_recovery_ok_cycles=ok_cycles,
        raw_state="tradeable",
        raw_reasons=[],
        degraded_recovery_cycles=3,
        blocked_recovery_cycles=6,
        allow_recovery_progress=True,
    )
    assert state == "degraded"
    assert ok_cycles == 2
    assert pending is not None
    assert "recovery_pending:2/3" in reasons

    state, reasons, ok_cycles, pending = _apply_state_hysteresis(
        current_state=state,
        current_recovery_ok_cycles=ok_cycles,
        raw_state="tradeable",
        raw_reasons=[],
        degraded_recovery_cycles=3,
        blocked_recovery_cycles=6,
        allow_recovery_progress=True,
    )
    assert state == "tradeable"
    assert ok_cycles == 0
    assert pending is None
    assert reasons == ["recovered_to_tradeable"]


def test_blocked_recovery_uses_blocked_threshold():
    state = "blocked"
    ok_cycles = 5
    state, reasons, ok_cycles, pending = _apply_state_hysteresis(
        current_state=state,
        current_recovery_ok_cycles=ok_cycles,
        raw_state="tradeable",
        raw_reasons=[],
        degraded_recovery_cycles=3,
        blocked_recovery_cycles=6,
        allow_recovery_progress=True,
    )
    assert state == "tradeable"
    assert ok_cycles == 0
    assert pending is None
    assert reasons == ["recovered_to_tradeable"]


def test_non_tradeable_state_applies_immediately_and_resets_counter():
    state, reasons, ok_cycles, pending = _apply_state_hysteresis(
        current_state="degraded",
        current_recovery_ok_cycles=2,
        raw_state="blocked",
        raw_reasons=["risk_halted"],
        degraded_recovery_cycles=3,
        blocked_recovery_cycles=6,
        allow_recovery_progress=True,
    )
    assert state == "blocked"
    assert ok_cycles == 0
    assert pending is None
    assert reasons == ["risk_halted"]


def test_recovery_counter_does_not_increment_when_progress_disabled():
    state, reasons, ok_cycles, pending = _apply_state_hysteresis(
        current_state="degraded",
        current_recovery_ok_cycles=1,
        raw_state="tradeable",
        raw_reasons=[],
        degraded_recovery_cycles=3,
        blocked_recovery_cycles=6,
        allow_recovery_progress=False,
    )
    assert state == "degraded"
    assert ok_cycles == 1
    assert pending is not None
    assert pending["ok_cycles"] == 1
    assert pending["required_cycles"] == 3
    assert "recovery_pending:1/3" in reasons
