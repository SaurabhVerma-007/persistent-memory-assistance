import asyncio
import os

import numpy as np
from google import genai
from google.genai import errors, types

from mem.config import EMBED_DIM, EMBED_PROVIDER, LOCAL_EMBED_MODEL

_gemini_client: genai.Client | None = None
_local_model = None

_GEMINI_TASK_TYPES = {
    "document": "RETRIEVAL_DOCUMENT",
    "query": "RETRIEVAL_QUERY",
}


def _normalize(vectors) -> list[list[float]]:
    """L2-normalize so that DOT distance in Qdrant behaves like cosine similarity.
    Gemini's truncated (non-3072) embeddings are not normalized by the API."""
    arr = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).tolist()


async def _embed_gemini(strings: list[str], task: str):
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    config = types.EmbedContentConfig(
        task_type=_GEMINI_TASK_TYPES[task],
        output_dimensionality=EMBED_DIM,
    )
    for attempt in range(4):
        try:
            out = await _gemini_client.aio.models.embed_content(
                model="gemini-embedding-001",
                contents=strings,
                config=config,
            )
            return [item.values for item in out.embeddings]
        except errors.APIError as e:
            # Free tiers rate-limit (429); retry with backoff on transient errors.
            if e.code in (429, 500, 503) and attempt < 3:
                await asyncio.sleep(2**attempt)
                continue
            raise


def _embed_local_sync(strings: list[str], task: str):
    global _local_model
    if _local_model is None:
        from fastembed import TextEmbedding  # pip/uv add fastembed

        _local_model = TextEmbedding(model_name=LOCAL_EMBED_MODEL)
    embed = _local_model.query_embed if task == "query" else _local_model.passage_embed
    return [v.tolist() for v in embed(strings)]


async def _embed_local(strings: list[str], task: str):
    # fastembed is synchronous; keep the event loop free.
    return await asyncio.to_thread(_embed_local_sync, strings, task)


async def generate_embeddings(strings: list[str], task: str = "document"):
    """Return normalized embeddings.

    task="document" when storing memories, task="query" when searching.
    """
    if task not in _GEMINI_TASK_TYPES:
        raise ValueError("task must be 'document' or 'query'")
    if not strings:
        return []

    if EMBED_PROVIDER == "local":
        raw = await _embed_local(strings, task)
    else:
        raw = await _embed_gemini(strings, task)

    if len(raw[0]) != EMBED_DIM:
        raise ValueError(
            f"Embedding size {len(raw[0])} != EMBED_DIM {EMBED_DIM}. "
            "Set EMBED_DIM to match the model (and use a matching collection)."
        )
    return _normalize(raw)


if __name__ == "__main__":
    texts = ["Hello how are you", "I like Machine Learning"]
    vecs = asyncio.run(generate_embeddings(texts))
    print(len(vecs), "vectors of size", len(vecs[0]))