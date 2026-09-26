import json

import mem.db as db


def test_sqlite_persists_users_sessions_and_events(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATABASE_URL", None)
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "app.db"))
    db.init_db()

    user = db.create_user("Reader", "scrypt$hash", 12345)
    assert user["username"] == "reader"
    assert db.get_user_by_username("READER")["user_id"] == 12345

    db.create_session("token-hash", user["pk"], 9999999999)
    assert db.get_session_user("token-hash")["username"] == "reader"
    db.delete_session("token-hash")
    assert db.get_session_user("token-hash") is None

    db.insert_event(12345, "trace-1", "turn", json.dumps({"question": "hello"}))
    events = db.list_events(12345)
    assert events[0]["data"] == {"question": "hello"}
    assert db.event_counts(12345) == {"turn": 1}
    db.delete_user_events(12345)
    assert db.list_events(12345) == []
