import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import dspy
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from mem.generate_embeddings import generate_embeddings
from mem.response_generator import ResponseGenerator, model
from mem.update_memory import update_memories
from mem.vectordb import (
    create_memory_collection,
    get_all_categories,
    search_memories,
    stringify_retrieved_point,
)


BASE_DIR = Path(__file__).parent
app = FastAPI(title="Mem0 Memory Chatbot")
sessions: dict[str, "ChatSession"] = {}
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def initialize_qdrant():
    await create_memory_collection()


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


def create_response_generator(user_id: int) -> dspy.ReAct:
    async def fetch_similar_memories(search_text: str, categories: list[str]):
        search_vector = (await generate_embeddings([search_text]))[0]
        memories = await search_memories(
            search_vector,
            user_id=user_id,
            categories=None if not categories else categories,
        )
        return {
            "memories": [stringify_retrieved_point(memory) for memory in memories]
        }

    return dspy.ReAct(ResponseGenerator, tools=[fetch_similar_memories], max_iters=2)


async def create_session() -> tuple[str, ChatSession]:
    session_id = uuid.uuid4().hex
    user_id = uuid.uuid4().int % 2_000_000_000
    session = ChatSession(user_id=user_id)
    await create_memory_collection()
    session.existing_categories = await get_all_categories(user_id=user_id)
    session.response_generator = create_response_generator(user_id)
    sessions[session_id] = session
    return session_id, session


async def get_session(session_id: str | None) -> tuple[str, ChatSession]:
    if session_id is not None and session_id in sessions:
        return session_id, sessions[session_id]
    return await create_session()


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
async def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty")

    session_id, session = await get_session(request.session_id)
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

        if result.save_memory:
            await update_memories(
                user_id=session.user_id,
                messages=session.past_messages[-6:],
            )
            session.existing_categories = await get_all_categories(
                user_id=session.user_id
            )

    return ChatResponse(answer=answer, session_id=session_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )