import warnings

warnings.filterwarnings("ignore")

import dspy
from rich.console import Console
from rich.rule import Rule

from mem.config import (
    CHAT_MAX_TOKENS,
    CHAT_MODEL,
    CHAT_TEMPERATURE,
    MAX_TRANSCRIPT_MESSAGES,
    make_lm,
)
from mem.events import TraceRef, log_event
from mem.generate_embeddings import generate_embeddings
from mem.update_memory import update_memories
from mem.vectordb import RetrievedMemory, get_all_categories, search_memories, stringify_retrieved_point

console = Console(log_path=False)
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)


def bound_transcript(messages: list[dict]) -> list[dict]:
    """Keep the newest complete turns within the configured message limit."""
    limit = MAX_TRANSCRIPT_MESSAGES - MAX_TRANSCRIPT_MESSAGES % 2
    return messages[-limit:]


model = make_lm(CHAT_MODEL, temperature=CHAT_TEMPERATURE, max_tokens=CHAT_MAX_TOKENS)


class ResponseGenerator(dspy.Signature):
    """
    You will be given a past conversation transcript between user and an AI agent. Also the latest question by the user.

    Relevant saved memories are retrieved before you answer and provided separately. You MUST use them when they answer a question about the user or an earlier conversation. Do not claim you do not know a personal fact when a matching saved memory is provided. Ignore retrieved memories that are unrelated to the question. If the provided memories do not answer the question, you may use the search tool to search with a more specific query.

    You are also provided a list of existing categories in the memory database to quickly search across categories. You can select multiple categories as a list to do your searches. If you select no categories (keep it empty). If you keep categories as empty, we will simply search across the entire database - that is fine too.

    The retrieved information may or may not contain the information user wants. React accordingly.

    You must output the final response, and also decide the latest interaction needs to be recorded into the memory database. New memories are meant to store new information that the user provides.

    While responding, you must be aware that you are continuously learning new memories about the user, so if retrieved memories do not directly address the user's question, mention what you know, acknowledge the gaps in your knowledge, and ask the user for information.

    If you retrieved records using the search tools, and the information was already present, no need to save a new memory. Only save memory if the new information is richer than what you retrieved or didn't find.

    New memories should be made when the USER provides new info. It is not to save information about the the AI or the assistant.
    """

    transcript: list[dict] = dspy.InputField()
    existing_categories: list[str] = dspy.InputField()
    retrieved_memories: list[str] = dspy.InputField(
        description="Relevant saved user memories retrieved for this question. Use matching facts when answering."
    )
    question: str = dspy.InputField()
    response: str = dspy.OutputField()
    save_memory: bool = dspy.OutputField(
        description="True if a new memory record needs to be created for the latest interaction"
    )


def create_response_generator(
    user_id: int, verbose: bool = False, trace: TraceRef | None = None
) -> dspy.ReAct:
    """Shared by the CLI (run_chat) and the web app.

    `trace` is a mutable holder; the web app sets trace.id for each turn so
    retrieval events on the dashboard are grouped under the message that
    caused them.
    """

    async def fetch_similar_memories(search_text: str, categories: list[str]):
        """
        Search memories from vector database if conversation requires additional context.

        Args:
        - search_text : The string to embed and do vector similarity search
        - categories : List of strings taken from existing_categories. Use an empty list ( [] ) if you want to search across all categories.
        """
        if verbose:
            console.log("Search text: ", search_text)
            console.log("Categories: ", categories)

        memories = await retrieve_relevant_memories(
            user_id=user_id,
            search_text=search_text,
            categories=categories,
            trace_id=trace.id if trace else None,
        )
        memories_str = [stringify_retrieved_point(m_) for m_ in memories]
        if verbose:
            console.log("Retrieved memories:\n- " + "\n- ".join(memories_str))
        return {"memories": memories_str}

    return dspy.ReAct(ResponseGenerator, tools=[fetch_similar_memories], max_iters=2)


async def retrieve_relevant_memories(
    user_id: int,
    search_text: str,
    *,
    categories: list[str] | None = None,
    trace_id: str | None = None,
) -> list[RetrievedMemory]:
    """Search Qdrant and log the result before the response model runs."""
    search_vector = (await generate_embeddings([search_text], task="query"))[0]
    memories = await search_memories(
        search_vector,
        user_id=user_id,
        categories=categories or None,
    )
    await log_event(
        user_id,
        trace_id,
        "retrieve",
        query=search_text,
        categories=categories or [],
        results=[
            {
                "text": memory.memory_text,
                "score": round(memory.score, 3),
                "categories": memory.categories,
            }
            for memory in memories
        ],
    )
    return memories


async def run_chat(user_id):
    response_generator = create_response_generator(user_id, verbose=True)
    past_messages = []

    existing_categories = await get_all_categories(user_id=user_id)

    console.print("Let's begin to chat!", style="bold green")

    while True:
        question = console.input("[bold cyan]> [/bold cyan]")
        console.print(Rule(style="grey50"))

        with console.status("[bold green] Working..."):
            past_messages = bound_transcript(past_messages)
            retrieved_memories = await retrieve_relevant_memories(
                user_id=user_id,
                search_text=question,
            )
            with dspy.context(lm=model):
                out = await response_generator.acall(
                    transcript=past_messages,
                    question=question,
                    existing_categories=existing_categories,
                    retrieved_memories=[
                        stringify_retrieved_point(memory)
                        for memory in retrieved_memories
                    ],
                )

            response = out.response

            past_messages.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response},
                ]
            )
            past_messages = bound_transcript(past_messages)

            if out.save_memory:
                # Blocking here to show the workflow; the web app runs this
                # in the background.
                console.log("Trying to update memory...")
                try:
                    update_result = await update_memories(
                        user_id=user_id,
                        messages=past_messages[-6:],
                    )
                    console.log(update_result, style="italic")
                    existing_categories = await get_all_categories(user_id=user_id)
                except Exception as e:
                    console.log(f"Memory update failed: {e}", style="bold red")

        console.print(f"\nAI: {response}\n\n", style="bold green")
