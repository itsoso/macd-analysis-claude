import pytest

pytest.importorskip("flask")

from app import app


def _login(client):
    with client.session_transaction() as sess:
        sess["logged_in"] = True
        sess["username"] = "tester"


def test_hotcoin_config_post_rejects_non_object_payload():
    client = app.test_client()
    _login(client)

    resp = client.post(
        "/api/live/hotcoin_config",
        json=["not", "an", "object"],
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "JSON 对象" in data["message"]
