from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import mem.update_memory as update_memory
from mem.vectordb import RetrievedMemory


class ScriptedReAct:
    """Replace the model with a deterministic action while exercising its tools."""

    action = None

    def __init__(self, signature, tools, max_iters):
        self.tools = tools

    async def acall(self, **_kwargs):
        if self.action == "ADD":
            await self.tools[0]("I enjoy trail running", ["hobbies"])
        elif self.action == "UPDATE":
            await self.tools[1](0, "I enjoy trail running in the Alps", ["hobbies"])
        elif self.action == "DELETE":
            await self.tools[2]([0])
        else:
            await self.tools[3]()
        return SimpleNamespace(summary=self.action)


def existing_memory(point_id="point-123", text="I enjoy trail running"):
    return RetrievedMemory(
        point_id=point_id,
        user_id=7,
        memory_text=text,
        categories=["hobbies"],
        date="2026-01-01T00:00+00:00",
        score=0.9,
    )


@pytest.fixture
def memory_tools(monkeypatch):
    ScriptedReAct.action = None
    monkeypatch.setattr(update_memory.dspy, "ReAct", ScriptedReAct)
    monkeypatch.setattr(update_memory, "generate_embeddings", AsyncMock(return_value=[[0.1, 0.2]]))
    insert = AsyncMock()
    delete = AsyncMock()
    log = AsyncMock()
    monkeypatch.setattr(update_memory, "insert_memories", insert)
    monkeypatch.setattr(update_memory, "delete_records", delete)
    monkeypatch.setattr(update_memory, "log_event", log)
    return insert, delete, log


@pytest.mark.asyncio
async def test_new_fact_adds_memory(memory_tools):
    insert, delete, _log = memory_tools
    ScriptedReAct.action = "ADD"

    result = await update_memory.update_memories_agent(
        user_id=7,
        messages=[{"role": "user", "content": "I enjoy trail running"}],
        existing_memories=[],
    )

    assert result == "ADD"
    insert.assert_awaited_once()
    assert insert.await_args.kwargs == {}
    assert insert.await_args.args[0][0].memory_text == "I enjoy trail running"
    assert insert.await_args.args[0][0].user_id == 7
    delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_enriched_fact_updates_existing_point(memory_tools):
    insert, delete, _log = memory_tools
    ScriptedReAct.action = "UPDATE"

    result = await update_memory.update_memories_agent(
        user_id=7,
        messages=[{"role": "user", "content": "I enjoy trail running in the Alps"}],
        existing_memories=[existing_memory()],
    )

    assert result == "UPDATE"
    assert insert.await_args.kwargs["point_ids"] == ["point-123"]
    assert insert.await_args.args[0][0].memory_text == "I enjoy trail running in the Alps"
    delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_contradiction_deletes_old_memory(memory_tools):
    insert, delete, _log = memory_tools
    ScriptedReAct.action = "DELETE"

    result = await update_memory.update_memories_agent(
        user_id=7,
        messages=[{"role": "user", "content": "I no longer enjoy trail running"}],
        existing_memories=[existing_memory()],
    )

    assert result == "DELETE"
    delete.assert_awaited_once_with(["point-123"])
    insert.assert_not_awaited()


@pytest.mark.asyncio
async def test_already_known_fact_is_noop(memory_tools):
    insert, delete, log = memory_tools
    ScriptedReAct.action = "NOOP"

    result = await update_memory.update_memories_agent(
        user_id=7,
        messages=[{"role": "user", "content": "I enjoy trail running"}],
        existing_memories=[existing_memory()],
    )

    assert result == "NOOP"
    log.assert_any_await(7, None, "noop")
    insert.assert_not_awaited()
    delete.assert_not_awaited()
