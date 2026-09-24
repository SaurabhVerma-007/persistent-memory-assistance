import asyncio
import hashlib
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import dspy
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from mem.response_generator import create_response_generator, model
from mem.update_memory import update_memories
from mem.vectordb import (
    create_memory_collection,
    delete_user_records,
    get_all_categories,
)

BASE_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)

MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "500"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
MAX_TRANSCRIPT_MESSAGES = int(os.getenv("MAX_TRANSCRIPT_MESSAGES", "20"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
CLIENT_COOKIE = "mem_client_id"
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "0") == "1"  # set to 1 on Render (HTTPS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_memory_collection()
    yield


app = FastAPI(title="Mem0 Memory Chatbot", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str


@dataclass
class ChatSession:
    user_id: int
    past_messages: list[dict[str, str]] = field(default_factory=list)
    existing_categories: list[str] = field(default_factory=list)
    response_generator: dspy.ReAct | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    memory_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = field(default_factory=time.monotonic)


sessions: dict[str, ChatSession] = {}


# ---------- identity ----------
def _is_valid_client_id(value: str) -> bool:
    return len(value) == 32 and all(c in "0123456789abcdef" for c in value)


def user_id_from_client_id(client_id: str) -> int:
    """Stable integer user id (56 bits) derived from the random cookie value,
    so memories survive restarts and new sessions."""
    digest = hashlib.sha256(client_id.encode()).digest()
    return int.from_bytes(digest[:7], "big")


def resolve_client_id(request: Request, response: Response) -> str:
    client_id = request.cookies.get(CLIENT_COOKIE, "")
    if not _is_valid_client_id(client_id):
        client_id = uuid.uuid4().hex
        response.set_cookie(
            CLIENT_COOKIE,
            client_id,
            max_age=60 * 60 * 24 * 365,
            httponly=True,
            samesite="lax",
            secure=COOKIE_SECURE,
        )
    return client_id


# ---------- rate limiting (in-process, per IP) ----------
_hits: dict[str, deque[float]] = defaultdict(deque)


def check_rate_limit(key: str):
    now = time.monotonic()
    window = _hits[key]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Too many requests. Slow down.")
    window.append(now)
    if len(_hits) > 10_000:  # drop idle keys
        for k in [k for k, v in _hits.items() if not v or now - v[-1] > 60]:
            del _hits[k]


# ---------- sessions ----------
def prune_sessions():
    now = time.monotonic()
    for sid in [s for s, v in sessions.items() if now - v.last_used > SESSION_TTL_SECONDS]:
        del sessions[sid]
    while len(sessions) >= MAX_SESSIONS:
        oldest = min(sessions, key=lambda s: sessions[s].last_used)
        del sessions[oldest]


async def create_session(user_id: int) -> tuple[str, ChatSession]:
    prune_sessions()
    session = ChatSession(user_id=user_id)
    session.existing_categories = await get_all_categories(user_id=user_id)
    session.response_generator = create_response_generator(user_id)
    session_id = uuid.uuid4().hex
    sessions[session_id] = session
    return session_id, session


async def get_session(session_id: str | None, user_id: int) -> tuple[str, ChatSession]:
    session = sessions.get(session_id) if session_id else None
    if session is not None and session.user_id == user_id:
        session.last_used = time.monotonic()
        return session_id, session
    return await create_session(user_id)


async def save_memory_safely(session: ChatSession, messages: list[dict[str, str]]):
    """Runs after the response is sent; failures never affect the user's answer."""
    async with session.memory_lock:
        try:
            await update_memories(user_id=session.user_id, messages=messages)
            session.existing_categories = await get_all_categories(
                user_id=session.user_id
            )
        except Exception:
            logger.exception("Memory update failed")


# ---------- routes ----------
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


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty")

    check_rate_limit(request.client.host if request.client else "unknown")
    client_id = resolve_client_id(request, response)
    session_id, session = await get_session(
        body.session_id, user_id_from_client_id(client_id)
    )

    async with session.lock:
        with dspy.context(lm=model):
            result = await session.response_generator.acall(
                transcript=session.past_messages,
                question=question,
                existing_categories=session.existing_categories,
            )

        answer = result.response
        session.past_messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        session.past_messages = session.past_messages[-MAX_TRANSCRIPT_MESSAGES:]

        if result.save_memory:
            background_tasks.add_task(
                save_memory_safely, session, list(session.past_messages[-6:])
            )

    return ChatResponse(answer=answer, session_id=session_id)


@app.delete("/api/memory")
async def delete_my_memory(request: Request):
    """Delete every stored memory for the current browser."""
    client_id = request.cookies.get(CLIENT_COOKIE, "")
    if not _is_valid_client_id(client_id):
        return {"deleted": False}
    user_id = user_id_from_client_id(client_id)
    await delete_user_records(user_id)
    for sid in [s for s, v in sessions.items() if v.user_id == user_id]:
        del sessions[sid]
    return {"deleted": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        proxy_headers=True,
        forwarded_allow_ips="*",  # behind Render's proxy; gives real client IPs
    )