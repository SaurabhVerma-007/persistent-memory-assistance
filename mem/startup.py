"""Fast-fail validation for external services required by a selected run."""

import os
from urllib.parse import urlparse

from mem.config import CHAT_MODEL, EMBED_PROVIDER, QDRANT_URL, UPDATER_MODEL
from mem.vectordb import client


async def validate_startup_config() -> None:
    """Check required credentials and confirm that Qdrant is reachable."""
    required_gemini = EMBED_PROVIDER == "gemini" or any(
        model.startswith("gemini/") for model in (CHAT_MODEL, UPDATER_MODEL)
    )
    if required_gemini and not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "Startup configuration error: GEMINI_API_KEY is required by the selected "
            "chat or embedding provider."
        )

    parsed = urlparse(QDRANT_URL)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(
            "Startup configuration error: QDRANT_URL must be an absolute HTTP(S) URL."
        )
    try:
        await client.get_collections()
    except Exception as exc:
        raise RuntimeError(
            f"Startup configuration error: cannot reach Qdrant at {QDRANT_URL}. "
            "Check QDRANT_URL, QDRANT_API_KEY, and network access."
        ) from exc
