import asyncio
import logging
import os
import re
import secrets
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import dspy
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from mem import db
from mem.auth import (
    DUMMY_HASH,
    current_user,
    end_session,
    hash_password,
    start_session,
    verify_password,
)
from mem.events import TraceRef, log_event
from mem.generate_embeddings import generate_embeddings
from mem.response_generator import bound_transcript, create_response_generator, model
from mem.startup import validate_startup_config
from mem.update_memory import update_memories
from mem.vectordb import (
    EmbeddedMemory,
    create_memory_collection,
    delete_records,
    delete_user_records,
    fetch_all_user_records,
    get_all_categories,
    insert_memories,
)

BASE_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)

ALLOW_SIGNUP = os.getenv("ALLOW_SIGNUP", "1") == "1"
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "500"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
AUTH_RATE_LIMIT_PER_MINUTE = int(os.getenv("AUTH_RATE_LIMIT_PER_MINUTE", "10"))
USERNAME_RE = re.compile(r"[A-Za-z0-9_.-]{3,32}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await validate_startup_config()
    await create_memory_collection()
    yield


app = FastAPI(title="Recall Terminal", lifespan=lifespan)


# ---------- models ----------
class SignupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    answer: str
    trace_id: str


class MemoryUpdateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


# ---------- rate limiting (in-process sliding window) ----------
_hits: dict[str, deque[float]] = defaultdict(deque)


def check_rate_limit(key: str, limit: int):
    now = time.monotonic()
    window = _hits[key]
    while window and now - window[0] >= 60:
        window.popleft()
    if len(window) >= limit:
        raise HTTPException(status_code=429, detail="Too many requests. Slow down.")
    window.append(now)
    if len(_hits) > 10_000:  # drop idle keys
        for k in [k for k, v in _hits.items() if not v or now - v[-1] > 60]:
            del _hits[k]


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


# ---------- per-user chat sessions (transcript cache) ----------
@dataclass
class ChatSession:
    user_id: int
    past_messages: list[dict[str, str]] = field(default_factory=list)
    existing_categories: list[str] = field(default_factory=list)
    response_generator: dspy.ReAct | None = None
    trace: TraceRef = field(default_factory=TraceRef)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    memory_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = field(default_factory=time.monotonic)


sessions: dict[int, ChatSession] = {}


def prune_sessions():
    now = time.monotonic()
    for uid in [u for u, s in sessions.items() if now - s.last_used > SESSION_TTL_SECONDS]:
        del sessions[uid]
    while len(sessions) >= MAX_SESSIONS:
        oldest = min(sessions, key=lambda u: sessions[u].last_used)
        del sessions[oldest]


async def get_session(user_id: int) -> ChatSession:
    session = sessions.get(user_id)
    if session is None:
        prune_sessions()
        session = ChatSession(user_id=user_id)
        session.existing_categories = await get_all_categories(user_id=user_id)
        session.response_generator = create_response_generator(
            user_id, trace=session.trace
        )
        sessions[user_id] = session
    session.last_used = time.monotonic()
    return session


async def save_memory_safely(
    session: ChatSession, messages: list[dict[str, str]], trace_id: str
):
    """Runs after the response is sent; failures never affect the user's answer."""
    async with session.memory_lock:
        try:
            await update_memories(
                user_id=session.user_id, messages=messages, trace_id=trace_id
            )
            session.existing_categories = await get_all_categories(
                user_id=session.user_id
            )
        except Exception as e:
            logger.exception("Memory update failed")
            await log_event(
                session.user_id,
                trace_id,
                "error",
                message=f"{type(e).__name__}: {str(e)[:200]}",
            )


# ---------- basic routes ----------
@app.get("/healthz")
async def health_check():
    return {"status": "ok"}


@app.exception_handler(Exception)
async def handle_unexpected_error(request, exception):
    logger.exception("Unhandled request error", exc_info=exception)
    return JSONResponse(
        status_code=500,
        content={"detail": "The assistant could not complete that request."},
    )


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


# ---------- auth ----------
@app.post("/api/auth/signup")
async def signup(body: SignupRequest, request: Request, response: Response):
    if not ALLOW_SIGNUP:
        raise HTTPException(status_code=403, detail="Sign-ups are disabled")
    check_rate_limit(f"auth:{client_ip(request)}", AUTH_RATE_LIMIT_PER_MINUTE)
    username = body.username.strip()
    if not USERNAME_RE.fullmatch(username):
        raise HTTPException(
            status_code=422,
            detail="Username must be 3-32 characters: letters, numbers, . _ -",
        )
    password_hash = await asyncio.to_thread(hash_password, body.password)
    # Random 56-bit id used in Qdrant. Random (not 1, 2, 3...) so that if the
    # database is ever reset, new accounts can't inherit old vector records.
    user = await asyncio.to_thread(
        db.create_user, username, password_hash, secrets.randbits(56) or 1
    )
    if user is None:
        raise HTTPException(status_code=409, detail="Username already taken")
    await start_session(response, user["pk"])
    return {"username": user["username"]}


@app.post("/api/auth/login")
async def login(body: LoginRequest, request: Request, response: Response):
    check_rate_limit(f"auth:{client_ip(request)}", AUTH_RATE_LIMIT_PER_MINUTE)
    user = await asyncio.to_thread(db.get_user_by_username, body.username.strip())
    stored = user["password_hash"] if user else DUMMY_HASH
    ok = await asyncio.to_thread(verify_password, body.password, stored)
    if not user or not ok:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    await start_session(response, user["pk"])
    return {"username": user["username"]}


@app.post("/api/auth/logout")
async def logout(request: Request, response: Response):
    await end_session(request, response)
    return {"ok": True}


@app.get("/api/auth/me")
async def me(user: dict = Depends(current_user)):
    return {"username": user["username"]}


# ---------- chat ----------
@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(current_user),
):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty")

    user_id = user["user_id"]
    check_rate_limit(f"chat:{user_id}", RATE_LIMIT_PER_MINUTE)
    session = await get_session(user_id)

    async with session.lock:
        trace_id = uuid.uuid4().hex[:12]
        session.trace.id = trace_id
        try:
            with dspy.context(lm=model):
                result = await session.response_generator.acall(
                    transcript=session.past_messages,
                    question=question,
                    existing_categories=session.existing_categories,
                )
        except Exception as e:
            logger.exception("Chat generation failed")
            kind = type(e).__name__
            rate_limited = (
                "RateLimit" in kind
                or "RESOURCE_EXHAUSTED" in str(e)
                or "429" in str(e)
            )
            # Shows up on the dashboard so a demo failure is diagnosable
            # without opening the server logs.
            await log_event(
                user_id,
                trace_id,
                "error",
                message=f"{kind}: {str(e)[:200]}",
                question=question[:600],
            )
            if rate_limited:
                raise HTTPException(
                    status_code=429,
                    detail="The AI model's rate limit was hit. Wait a minute and try again.",
                )
            raise HTTPException(
                status_code=502,
                detail=f"The AI model request failed ({kind}). Details are in the activity log.",
            )

        answer = result.response
        save = bool(result.save_memory)
        session.past_messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        session.past_messages = bound_transcript(session.past_messages)

        await log_event(
            user_id,
            trace_id,
            "turn",
            question=question[:600],
            answer=answer[:600],
            save_memory=save,
        )
        if save:
            background_tasks.add_task(
                save_memory_safely,
                session,
                list(session.past_messages[-6:]),
                trace_id,
            )

    return ChatResponse(answer=answer, trace_id=trace_id)


@app.get("/api/chat/history")
async def chat_history(user: dict = Depends(current_user)):
    session = sessions.get(user["user_id"])
    return {"messages": session.past_messages if session else []}


@app.post("/api/chat/reset")
async def chat_reset(user: dict = Depends(current_user)):
    """Start a fresh conversation. Long-term memory is kept."""
    session = sessions.get(user["user_id"])
    if session:
        session.past_messages = []
    return {"ok": True}


# ---------- dashboard ----------
@app.get("/api/dashboard/events")
async def dashboard_events(
    after_id: int = 0, limit: int = 100, user: dict = Depends(current_user)
):
    limit = max(1, min(limit, 200))
    events = await asyncio.to_thread(
        db.list_events, user["user_id"], max(after_id, 0), limit
    )
    return {"events": events}


@app.get("/api/dashboard/memories")
async def dashboard_memories(user: dict = Depends(current_user)):
    records = await fetch_all_user_records(user["user_id"])
    records.sort(key=lambda r: r.date, reverse=True)
    counts = await asyncio.to_thread(db.event_counts, user["user_id"])
    return {
        "memories": [
            {
                "point_id": r.point_id,
                "text": r.memory_text,
                "categories": r.categories,
                "date": r.date,
                "source_text": r.source_text,
            }
            for r in records
        ],
        "counts": counts,
    }


@app.delete("/api/memory")
async def delete_my_memory(user: dict = Depends(current_user)):
    """Delete this account's stored memories and its activity log."""
    user_id = user["user_id"]
    await delete_user_records(user_id)
    await asyncio.to_thread(db.delete_user_events, user_id)
    sessions.pop(user_id, None)
    return {"deleted": True}


async def _get_user_memory(user_id: int, point_id: str):
    records = await fetch_all_user_records(user_id)
    return next((record for record in records if record.point_id == point_id), None)


@app.put("/api/memory/{point_id}")
async def update_my_memory(
    point_id: str,
    body: MemoryUpdateRequest,
    user: dict = Depends(current_user),
):
    record = await _get_user_memory(user["user_id"], point_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Memory text cannot be empty")
    vector = (await generate_embeddings([text], task="document"))[0]
    updated = EmbeddedMemory(
        user_id=record.user_id,
        memory_text=text,
        categories=record.categories,
        date=datetime.now(timezone.utc).isoformat(timespec="minutes"),
        embedding=vector,
        source_text=record.source_text,
    )
    await insert_memories([updated], point_ids=[point_id])
    return {"updated": True}


@app.delete("/api/memory/{point_id}")
async def delete_one_memory(point_id: str, user: dict = Depends(current_user)):
    record = await _get_user_memory(user["user_id"], point_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    await delete_records([point_id])
    return {"deleted": True}


@app.get("/api/memory/export")
async def export_my_memories(user: dict = Depends(current_user)):
    """Download this account's stored long-term memories as JSON."""
    records = await fetch_all_user_records(user["user_id"])
    records.sort(key=lambda r: r.date, reverse=True)
    return {
        "memories": [
            {
                "text": r.memory_text,
                "categories": r.categories,
                "date": r.date,
                "source_text": r.source_text,
            }
            for r in records
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        proxy_headers=True,
        forwarded_allow_ips="*",  # behind Render's proxy; gives real client IPs
    )
