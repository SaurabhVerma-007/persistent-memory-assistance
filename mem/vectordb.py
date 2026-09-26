import asyncio
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import Distance, Filter, VectorParams

from mem.config import (
    COLLECTION_NAME,
    EMBED_DIM,
    MEMORY_SCORE_THRESHOLD,
    MEMORY_TOP_K,
    QDRANT_API_KEY,
    QDRANT_URL,
)

client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


class EmbeddedMemory(BaseModel):
    user_id: int
    memory_text: str
    categories: list[str]
    date: str
    embedding: list[float]
    source_text: str | None = None


class RetrievedMemory(BaseModel):
    point_id: str
    user_id: int
    memory_text: str
    categories: list[str]
    date: str
    score: float
    source_text: str | None = None


async def _ensure_indexes():
    # Safe to run every startup; repairs collections that are missing an index.
    for field_name, schema in (
        ("user_id", models.PayloadSchemaType.INTEGER),
        ("categories", models.PayloadSchemaType.KEYWORD),
    ):
        try:
            await client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema,
            )
        except Exception as e:  # already exists / transient
            print(f"Index '{field_name}' not created: {e}")


async def create_memory_collection():
    if not await client.collection_exists(COLLECTION_NAME):
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.DOT),
        )
        print(f"Collection '{COLLECTION_NAME}' created")
    else:
        info = await client.get_collection(COLLECTION_NAME)
        size = getattr(info.config.params.vectors, "size", None)
        if size is not None and size != EMBED_DIM:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' has vector size {size}, "
                f"but EMBED_DIM is {EMBED_DIM}. Use a different QDRANT_COLLECTION."
            )
        print(f"Collection '{COLLECTION_NAME}' exists")
    await _ensure_indexes()


async def insert_memories(
    memories: list[EmbeddedMemory], point_ids: Optional[list[str]] = None
) -> list[str]:
    """Upsert memories. Pass point_ids to overwrite existing points (used by update)."""
    ids = point_ids or [uuid4().hex for _ in memories]
    if len(ids) != len(memories):
        raise ValueError("point_ids must match memories in length")
    await client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=point_id,
                payload={
                    "user_id": memory.user_id,
                    "categories": memory.categories,
                    "memory_text": memory.memory_text,
                    "date": memory.date,
                    "source_text": memory.source_text,
                },
                vector=memory.embedding,
            )
            for point_id, memory in zip(ids, memories)
        ],
    )
    return ids


async def search_memories(
    search_vector: list[float],
    user_id: int,
    categories: Optional[list[str]] = None,
    limit: int = MEMORY_TOP_K,
    score_threshold: float = MEMORY_SCORE_THRESHOLD,
):
    must_conditions: list[models.Condition] = [
        models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
    ]
    if categories:
        must_conditions.append(
            models.FieldCondition(key="categories", match=models.MatchAny(any=categories))
        )
    outs = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=search_vector,
        with_payload=True,
        query_filter=Filter(must=must_conditions),
        score_threshold=score_threshold,
        limit=limit,
    )
    return [convert_retrieved_records(p) for p in outs.points if p is not None]


async def delete_user_records(user_id):
    await client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=Filter(
                must=[
                    models.FieldCondition(
                        key="user_id", match=models.MatchValue(value=user_id)
                    )
                ]
            )
        ),
    )


async def delete_records(point_ids):
    await client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.PointIdsList(points=point_ids),
    )


async def fetch_all_user_records(user_id, limit: int = 1000):
    records, _ = await client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                models.FieldCondition(
                    key="user_id", match=models.MatchValue(value=user_id)
                )
            ]
        ),
        limit=limit,
        with_payload=True,
    )
    return [convert_retrieved_records(r) for r in records]


def convert_retrieved_records(point) -> RetrievedMemory:
    return RetrievedMemory(
        point_id=str(point.id),
        user_id=point.payload["user_id"],
        memory_text=point.payload["memory_text"],
        categories=point.payload["categories"],
        date=point.payload["date"],
        score=getattr(point, "score", 0.0) or 0.0,  # scroll results have no score
        source_text=point.payload.get("source_text"),
    )


async def get_all_categories(user_id):
    """Unique categories for this user, via Qdrant's facet on the indexed field."""
    facet_result = await client.facet(
        collection_name=COLLECTION_NAME,
        key="categories",
        facet_filter=Filter(
            must=[
                models.FieldCondition(
                    key="user_id", match=models.MatchValue(value=user_id)
                )
            ]
        ),
        limit=1000,
    )
    return [hit.value for hit in facet_result.hits]


def stringify_retrieved_point(retrieved_memory: RetrievedMemory):
    return (
        f"{retrieved_memory.memory_text} "
        f"(Categories: {retrieved_memory.categories}) "
        f"Relevance: {retrieved_memory.score:.2f}"
    )


if __name__ == "__main__":
    asyncio.run(create_memory_collection())
