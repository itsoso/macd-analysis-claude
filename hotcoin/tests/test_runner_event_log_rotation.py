import gzip
import os

from hotcoin.runner import _list_event_archives, _prune_event_archives, _rotate_event_log_file


def test_rotate_event_log_file_noop_when_under_limit(tmp_path):
    events = tmp_path / "hotcoin_events.jsonl"
    events.write_text('{"a":1}\n', encoding="utf-8")

    out = _rotate_event_log_file(
        events_file=str(events),
        max_bytes=1024,
        keep_archives=3,
        compress_archive=True,
    )
    assert out is None
    assert events.exists()
    assert events.read_text(encoding="utf-8") == '{"a":1}\n'


def test_rotate_event_log_file_with_compress(tmp_path):
    events = tmp_path / "hotcoin_events.jsonl"
    payload = ('{"event":"x"}\n' * 20)
    events.write_text(payload, encoding="utf-8")

    rotated = _rotate_event_log_file(
        events_file=str(events),
        max_bytes=32,
        keep_archives=5,
        compress_archive=True,
        now_ts=1700000000,
    )
    assert rotated is not None
    assert rotated.endswith(".jsonl.gz")
    assert os.path.exists(rotated)
    assert not events.exists()

    with gzip.open(rotated, "rt", encoding="utf-8") as f:
        assert f.read() == payload


def test_prune_event_archives_keep_n(tmp_path):
    events = tmp_path / "hotcoin_events.jsonl"
    events.write_text("", encoding="utf-8")

    created = []
    for i in range(5):
        p = tmp_path / f"hotcoin_events.20240101_00000{i}.jsonl.gz"
        p.write_bytes(b"x")
        os.utime(p, (1700000000 + i, 1700000000 + i))
        created.append(str(p))

    _prune_event_archives(str(events), keep=2)
    remain = _list_event_archives(str(events))
    assert len(remain) == 2
    assert set(remain).issubset(set(created))
