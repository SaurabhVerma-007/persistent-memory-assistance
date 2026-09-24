"""Activity log for the dashboard.

Event types written by the pipeline:
    turn        the user's message, the answer, and whether a save was requested
    retrieve    a memory search: query, category filter, and what came back
    candidates  the existing memories shown to the updater
    add / update / delete / noop   the updater's decisions
    summary     the updater's one-line summary
    error       a failed memory update
"""

import asyncio
import json
import logging
from dataclasses import dataclass

from mem import db

logger = logging.getLogger(__name__)


@dataclass
class TraceRef:
    """Mutable holder so a long-lived tool closure can see the current turn's id."""

    id: str | None = None


async def log_event(user_id: int, trace_id: str | None, type_: str, **data):
    """Never raises: a logging failure must not break the chat."""
    try:
        await asyncio.to_thread(
            db.insert_event,
            user_id,
            trace_id,
            type_,
            json.dumps(data, ensure_ascii=False, default=str),
        )
    except Exception:
        logger.exception("Could not log %s event", type_)
