"""SQLite storage for accounts, login sessions and the memory activity log.

Uses only the standard library. Set DB_PATH to change the file location.
On hosts with an ephemeral disk (for example Render's free tier) the file is
wiped on every deploy: attach a persistent disk or move this to Postgres.
"""

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.getenv("DB_PATH", "data/app.db")


@contextmanager
def _db():
    folder = os.path.dirname(DB_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _db() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                mem_user_id INTEGER NOT NULL UNIQUE,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_pk INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                trace_id TEXT,
                ts TEXT NOT NULL,
                type TEXT NOT NULL,
                data TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_user ON memory_events(user_id, id);
            """
        )


# ---------- users ----------
def create_user(username: str, password_hash: str, mem_user_id: int) -> dict | None:
    """Returns the new user, or None if the username is taken."""
    with _db() as c:
        try:
            cur = c.execute(
                "INSERT INTO users (username, password_hash, mem_user_id, created_at) "
                "VALUES (?, ?, ?, ?)",
                (username, password_hash, mem_user_id, time.time()),
            )
        except sqlite3.IntegrityError:
            return None
        return {"pk": cur.lastrowid, "username": username, "user_id": mem_user_id}


def get_user_by_username(username: str) -> dict | None:
    with _db() as c:
        row = c.execute(
            "SELECT id AS pk, username, password_hash, mem_user_id AS user_id "
            "FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return dict(row) if row else None


# ---------- login sessions ----------
def create_session(token_hash: str, user_pk: int, expires_at: float):
    with _db() as c:
        c.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
        c.execute(
            "INSERT INTO sessions (token_hash, user_pk, expires_at) VALUES (?, ?, ?)",
            (token_hash, user_pk, expires_at),
        )


def get_session_user(token_hash: str) -> dict | None:
    with _db() as c:
        row = c.execute(
            "SELECT u.id AS pk, u.username, u.mem_user_id AS user_id "
            "FROM sessions s JOIN users u ON u.id = s.user_pk "
            "WHERE s.token_hash = ? AND s.expires_at > ?",
            (token_hash, time.time()),
        ).fetchone()
        return dict(row) if row else None


def delete_session(token_hash: str):
    with _db() as c:
        c.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))


# ---------- memory activity events ----------
def insert_event(user_id: int, trace_id: str | None, type_: str, data_json: str):
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _db() as c:
        c.execute(
            "INSERT INTO memory_events (user_id, trace_id, ts, type, data) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, trace_id, ts, type_, data_json),
        )


def list_events(user_id: int, after_id: int = 0, limit: int = 100) -> list[dict]:
    """after_id == 0: the latest `limit` events. Otherwise events newer than after_id.
    Always returned oldest-first."""
    with _db() as c:
        if after_id > 0:
            rows = c.execute(
                "SELECT * FROM memory_events WHERE user_id = ? AND id > ? "
                "ORDER BY id ASC LIMIT ?",
                (user_id, after_id, limit),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM (SELECT * FROM memory_events WHERE user_id = ? "
                "ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
                (user_id, limit),
            ).fetchall()
    return [
        {
            "id": r["id"],
            "trace_id": r["trace_id"],
            "ts": r["ts"],
            "type": r["type"],
            "data": json.loads(r["data"]),
        }
        for r in rows
    ]


def event_counts(user_id: int) -> dict[str, int]:
    with _db() as c:
        rows = c.execute(
            "SELECT type, COUNT(*) AS n FROM memory_events WHERE user_id = ? GROUP BY type",
            (user_id,),
        ).fetchall()
    return {r["type"]: r["n"] for r in rows}


def delete_user_events(user_id: int):
    with _db() as c:
        c.execute("DELETE FROM memory_events WHERE user_id = ?", (user_id,))


init_db()
