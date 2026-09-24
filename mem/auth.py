"""Username/password login using only the standard library.

- Passwords: scrypt with a random salt per user.
- Login sessions: a random token in an HttpOnly cookie. Only the SHA-256 of the
  token is stored, so a leaked database can't be used to log in.
"""

import asyncio
import hashlib
import hmac
import os
import secrets
import time

from fastapi import HTTPException, Request, Response

from mem import db

SESSION_COOKIE = "mem_session"
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_DAYS", "30")) * 86400
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "0") == "1"  # set to 1 behind HTTPS


def _scrypt(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32, maxmem=64 * 1024 * 1024
    )


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    return f"scrypt${salt.hex()}${_scrypt(password, salt).hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_hex, digest_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        candidate = _scrypt(password, bytes.fromhex(salt_hex)).hex()
    except ValueError:
        return False
    return hmac.compare_digest(candidate, digest_hex)


# Verified against when the username doesn't exist, so login takes the same
# time either way.
DUMMY_HASH = hash_password("not-a-real-password")


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def start_session(response: Response, user_pk: int):
    token = secrets.token_urlsafe(32)
    await asyncio.to_thread(
        db.create_session, hash_token(token), user_pk, time.time() + SESSION_TTL_SECONDS
    )
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=COOKIE_SECURE,
    )


async def end_session(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        await asyncio.to_thread(db.delete_session, hash_token(token))
    response.delete_cookie(SESSION_COOKIE)


async def current_user(request: Request) -> dict:
    """FastAPI dependency. Returns {"pk", "username", "user_id"} or raises 401.
    user_id is the id used in Qdrant; it always comes from the server-side
    session, never from anything the client sends."""
    token = request.cookies.get(SESSION_COOKIE)
    user = None
    if token:
        user = await asyncio.to_thread(db.get_session_user, hash_token(token))
    if not user:
        raise HTTPException(status_code=401, detail="Not signed in")
    return user
