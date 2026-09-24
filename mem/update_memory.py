import logging
from datetime import datetime, timezone

import dspy
from pydantic import BaseModel

from mem.config import (
    UPDATER_MAX_TOKENS,
    UPDATER_MODEL,
    UPDATER_TEMPERATURE,
    make_lm,
)
from mem.events import log_event
from mem.generate_embeddings import generate_embeddings
from mem.vectordb import (
    EmbeddedMemory,
    RetrievedMemory,
    delete_records,
    insert_memories,
    search_memories,
)

logger = logging.getLogger(__name__)

dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)

updater_lm = make_lm(
    UPDATER_MODEL, temperature=UPDATER_TEMPERATURE, max_tokens=UPDATER_MAX_TOKENS
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="minutes")


class MemoryWithIds(BaseModel):
    memory_id: int
    memory_text: str
    memory_categories: list[str]


class UpdateMemorySignature(dspy.Signature):
    """
    You will be given the conversation between user and assistant and some similar memories from the database. Your goal is to decide how to combine the new memories into the database with the existing memories.

    Actions meaning:
    - ADD: add new memories into the database as a new memory
    - UPDATE: update an existing memory with richer information.
    - DELETE: remove memory items from the database that aren't required anymore due to new information
    - NOOP: No need to take any action

    Only store facts the USER stated about themselves. If no action is required you can finish.

    Think less and do actions.
    """

    messages: list[dict] = dspy.InputField()
    existing_memories: list[MemoryWithIds] = dspy.InputField()
    summary: str = dspy.OutputField(
        description="Summarize what you did. Very short (less than 10 words)"
    )


async def update_memories_agent(
    user_id: int,
    messages: list[dict],
    existing_memories: list[RetrievedMemory],
    trace_id: str | None = None,
):
    def id_error(memory_id: int) -> str | None:
        if not existing_memories:
            return "There are no existing memories to change. Use add_memory instead."
        if not 0 <= memory_id < len(existing_memories):
            return f"Invalid memory_id {memory_id}. Valid ids: 0 to {len(existing_memories) - 1}."
        return None

    async def add_memory(memory_text: str, categories: list[str]) -> str:
        """
        Add a new memory to the database.

        Args:
        memory_text: A simple, atomic fact about the user.
        categories: Use existing categories or create new ones if required.
        """
        embeddings = await generate_embeddings([memory_text], task="document")
        await insert_memories(
            [
                EmbeddedMemory(
                    user_id=user_id,
                    memory_text=memory_text,
                    categories=categories,
                    date=_now(),
                    embedding=embeddings[0],
                )
            ]
        )
        await log_event(
            user_id, trace_id, "add", text=memory_text, categories=categories
        )
        return f"Memory: '{memory_text}' was added to DB"

    async def update(memory_id: int, updated_memory_text: str, categories: list[str]):
        """
        Replace an existing memory with richer or corrected information.

        Args:
        memory_id: integer id of the memory to replace (from existing_memories)
        updated_memory_text: Simple atomic factoid that replaces the old memory
        categories: Use existing categories or create new ones if required
        """
        if err := id_error(memory_id):
            return err
        old = existing_memories[memory_id]
        # Embed first, then overwrite the same point: a failed embedding call
        # can no longer lose the old memory.
        embeddings = await generate_embeddings([updated_memory_text], task="document")
        await insert_memories(
            [
                EmbeddedMemory(
                    user_id=user_id,
                    memory_text=updated_memory_text,
                    categories=categories,
                    date=_now(),
                    embedding=embeddings[0],
                )
            ],
            point_ids=[old.point_id],
        )
        await log_event(
            user_id,
            trace_id,
            "update",
            old_text=old.memory_text,
            new_text=updated_memory_text,
            categories=categories,
        )
        return f"Memory {memory_id} has been updated to: '{updated_memory_text}'"

    async def noop():
        """
        Call this if no action is required
        """
        await log_event(user_id, trace_id, "noop")
        return "No action done"

    async def delete(memory_ids: list[int]):
        """
        Remove memories that are contradicted or no longer needed.

        Args:
        memory_ids: list of integer memory_ids (from existing_memories) to remove
        """
        valid = sorted({i for i in memory_ids if not id_error(i)})
        if not valid:
            return "No valid memory_ids given; nothing was deleted."
        # Map the model's list indices to real Qdrant point ids.
        await delete_records([existing_memories[i].point_id for i in valid])
        for i in valid:
            await log_event(
                user_id,
                trace_id,
                "delete",
                text=existing_memories[i].memory_text,
                categories=existing_memories[i].categories,
            )
        return f"Memories {valid} deleted"

    memory_updater = dspy.ReAct(
        UpdateMemorySignature, tools=[add_memory, update, delete, noop], max_iters=3
    )
    memory_ids = [
        MemoryWithIds(
            memory_id=idx, memory_text=m.memory_text, memory_categories=m.categories
        )
        for idx, m in enumerate(existing_memories)
    ]

    with dspy.context(lm=updater_lm):
        out = await memory_updater.acall(
            messages=messages, existing_memories=memory_ids
        )
    await log_event(user_id, trace_id, "summary", text=out.summary)
    return out.summary


async def update_memories(
    user_id: int, messages: list[dict], trace_id: str | None = None
):
    latest_user_message = [x["content"] for x in messages if x["role"] == "user"][-1]
    embedding = (await generate_embeddings([latest_user_message], task="query"))[0]

    retrieved_memories = await search_memories(search_vector=embedding, user_id=user_id)
    await log_event(
        user_id,
        trace_id,
        "candidates",
        memories=[
            {
                "text": m.memory_text,
                "score": round(m.score, 3),
                "categories": m.categories,
            }
            for m in retrieved_memories
        ],
    )

    return await update_memories_agent(
        user_id=user_id,
        existing_memories=retrieved_memories,
        messages=messages,
        trace_id=trace_id,
    )


async def test():
    messages = [{"role": "user", "content": "I want to go Tokyo"}]
    response = await update_memories(user_id=1, messages=messages)
    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(test())
