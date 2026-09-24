import os

from google import genai
from mem0 import Memory


# ============================================================
# Configuration
# ============================================================

USER_ID = "avb"
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

MODEL = "gemini-3.5-flash-lite"


# ============================================================
# Gemini client
# ============================================================

gemini = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# Mem0 + local Qdrant
# ============================================================

memory = Memory.from_config(
    {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "mem0_local",
                "host": "localhost",
                "port": 6333,
                "embedding_model_dims": 768,
            },
        },

        "llm": {
            "provider": "gemini",
            "config": {
                "model": MODEL,
                "api_key": GEMINI_API_KEY,
                "temperature": 0.1,
                "max_tokens": 2000,
            },
        },

        "embedder": {
            "provider": "gemini",
            "config": {
                "model": "models/gemini-embedding-001",
                "api_key": GEMINI_API_KEY,
                "embedding_dims": 768,
            },
        },
    }
)


# ============================================================
# Conversation history
# ============================================================

conversation = []


# ============================================================
# Helper: retrieve memories
# ============================================================

def get_relevant_memories(query: str) -> list[str]:
    result = memory.search(
        query,
        user_id=USER_ID,
        limit=5,
    )

    return [
        item["memory"]
        for item in result.get("results", [])
    ]


# ============================================================
# Chat loop
# ============================================================

print("=" * 50)
print("      Gemini + Mem0 OSS + Qdrant")
print("=" * 50)
print("Memory store : local Qdrant")
print("Qdrant       : http://localhost:6333")
print("Type 'exit' or 'quit' to stop.")
print()


while True:

    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break


    # --------------------------------------------------------
    # 1. Retrieve long-term memories from Qdrant
    # --------------------------------------------------------

    memories = get_relevant_memories(user_input)

    print("\nRetrieved memories:")

    if memories:
        for item in memories:
            print(f"  - {item}")
    else:
        print("  - None")


    # --------------------------------------------------------
    # 2. Build memory context
    # --------------------------------------------------------

    if memories:

        memory_context = "\n".join(
            f"- {item}"
            for item in memories
        )

    else:

        memory_context = "No relevant long-term memories."


    # --------------------------------------------------------
    # 3. Build conversation context
    # --------------------------------------------------------

    recent_history = "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in conversation[-10:]
    )


    prompt = f"""
You are a helpful AI assistant.

Use relevant long-term memories when they help answer
the user's question.

Do not invent facts.

LONG-TERM MEMORIES:
{memory_context}

RECENT CONVERSATION:
{recent_history}

USER:
{user_input}
"""


    # --------------------------------------------------------
    # 4. Ask Gemini
    # --------------------------------------------------------

    response = gemini.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    answer = response.text.strip()


    # --------------------------------------------------------
    # 5. Display answer
    # --------------------------------------------------------

    print(f"\nAssistant: {answer}\n")


    # --------------------------------------------------------
    # 6. Update short-term conversation history
    # --------------------------------------------------------

    conversation.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    conversation.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


    # --------------------------------------------------------
    # 7. Store the interaction in Mem0
    # --------------------------------------------------------

    memory.add(
        conversation[-2:],
        user_id=USER_ID,
    )
