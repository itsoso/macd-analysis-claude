import sys
import types

from hotcoin import config as hot_cfg


def _with_fake_config_store(monkeypatch, saved):
    fake = types.SimpleNamespace(get_hotcoin_config=lambda: saved)
    monkeypatch.setitem(sys.modules, "config_store", fake)


def test_hotcoin_paper_env_overrides_saved_false(monkeypatch):
    _with_fake_config_store(monkeypatch, {"execution": {"use_paper_trading": False}})
    monkeypatch.setenv("HOTCOIN_PAPER", "1")
    cfg = hot_cfg.load_config()
    assert cfg.execution.use_paper_trading is True


def test_hotcoin_paper_env_can_disable(monkeypatch):
    _with_fake_config_store(monkeypatch, {"execution": {"use_paper_trading": True}})
    monkeypatch.setenv("HOTCOIN_PAPER", "0")
    cfg = hot_cfg.load_config()
    assert cfg.execution.use_paper_trading is False


def test_hotcoin_execute_env_true_false(monkeypatch):
    _with_fake_config_store(monkeypatch, {})
    monkeypatch.setenv("HOTCOIN_EXECUTE", "true")
    cfg_true = hot_cfg.load_config()
    assert cfg_true.execution.enable_order_execution is True

    monkeypatch.setenv("HOTCOIN_EXECUTE", "false")
    cfg_false = hot_cfg.load_config()
    assert cfg_false.execution.enable_order_execution is False
