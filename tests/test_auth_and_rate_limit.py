import pytest

from mem.auth import hash_password, verify_password


def test_password_hash_verifies_correct_password():
    stored = hash_password("correct-horse-battery-staple")
    assert verify_password("correct-horse-battery-staple", stored)


def test_password_hash_rejects_wrong_password():
    stored = hash_password("correct-horse-battery-staple")
    assert not verify_password("wrong-password", stored)


@pytest.mark.parametrize("stored", ["", "broken", "scrypt$not-hex$nope", "bcrypt$a$b"])
def test_password_hash_rejects_malformed_hash_safely(stored):
    assert not verify_password("anything", stored)


def test_rate_limit_allows_requests_under_limit(monkeypatch):
    import web_app

    web_app._hits.clear()
    monkeypatch.setattr(web_app.time, "monotonic", lambda: 100.0)
    web_app.check_rate_limit("test:under", 2)
    web_app.check_rate_limit("test:under", 2)


def test_rate_limit_blocks_requests_over_limit(monkeypatch):
    from fastapi import HTTPException

    import web_app

    web_app._hits.clear()
    monkeypatch.setattr(web_app.time, "monotonic", lambda: 100.0)
    web_app.check_rate_limit("test:over", 1)
    with pytest.raises(HTTPException) as exc:
        web_app.check_rate_limit("test:over", 1)
    assert exc.value.status_code == 429


def test_rate_limit_window_resets_after_60_seconds(monkeypatch):
    import web_app

    web_app._hits.clear()
    now = [100.0]
    monkeypatch.setattr(web_app.time, "monotonic", lambda: now[0])
    web_app.check_rate_limit("test:window", 1)
    now[0] = 160.0
    web_app.check_rate_limit("test:window", 1)
