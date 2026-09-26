"""Central runtime configuration. Every value can be overridden with an env var.

LLM names are LiteLLM model strings, for example:
    gemini/<model-name>            (uses GEMINI_API_KEY)
    groq/openai/gpt-oss-20b        (uses GROQ_API_KEY)
    openrouter/<vendor>/<model>    (uses OPENROUTER_API_KEY)
Check AI Studio / the provider's model list for currently available names.
"""

import os

# --- LLMs ---------------------------------------------------------------
# GEMINI_MODEL is still honoured so existing setups keep working.
CHAT_MODEL = (
    os.getenv("CHAT_MODEL") or os.getenv("GEMINI_MODEL") or "gemini/gemini-2.5-flash"
)
# The updater makes several small tool-calling steps per memory save.
# Point it at a different provider to split free-tier quotas.
UPDATER_MODEL = os.getenv("UPDATER_MODEL") or CHAT_MODEL

# Gemini 2.5+ "thinking" tokens count toward max_tokens, so don't set these too low.
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "8000"))
UPDATER_MAX_TOKENS = int(os.getenv("UPDATER_MAX_TOKENS", "4000"))
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "1.0"))
UPDATER_TEMPERATURE = float(os.getenv("UPDATER_TEMPERATURE", "0.3"))

# --- Embeddings ---------------------------------------------------------
# "gemini" (API, free tier) or "local" (fastembed, runs on CPU, no API key).
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "gemini").lower()
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768" if EMBED_PROVIDER == "gemini" else "384"))

# --- Qdrant -------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
# Provider + dimension are in the default name so incompatible vectors
# never share a collection.
COLLECTION_NAME = (
    os.getenv("QDRANT_COLLECTION") or f"memories_{EMBED_PROVIDER}_{EMBED_DIM}_norm"
)

# --- Retrieval ----------------------------------------------------------
# Starting values. Vectors are normalized, so DOT == cosine similarity.
# Print a few real scores (stringify_retrieved_point shows them) and tune.
MEMORY_SCORE_THRESHOLD = float(os.getenv("MEMORY_SCORE_THRESHOLD", "0.5"))
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "3"))
MAX_TRANSCRIPT_MESSAGES = max(2, int(os.getenv("MAX_TRANSCRIPT_MESSAGES", "20")))


def make_lm(model: str, *, temperature: float, max_tokens: int):
    """Build a dspy.LM. Gemini gets its key explicitly; other providers
    are read from their standard env vars by LiteLLM."""
    import dspy

    kwargs = {}
    if model.startswith("gemini/"):
        kwargs["api_key"] = os.environ.get("GEMINI_API_KEY")
    return dspy.LM(
        model=model, temperature=temperature, max_tokens=max_tokens, **kwargs
    )
