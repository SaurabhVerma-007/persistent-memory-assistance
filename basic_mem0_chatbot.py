from openai import OpenAI
from mem0 import MemoryClient
import os

"""
Required environment variables:

GEMINI_API_KEY=your-gemini-key
MEM0_API_KEY=your-mem0-key
"""

user_id = "avb"

# Mem0 Cloud memory
memory = MemoryClient()

# Gemini through Google's OpenAI-compatible API
client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

messages = []

while True:
    user_input = input("User: ").strip()

    if not user_input:
        continue

    if user_input.lower() in {"exit", "quit"}:
        break

    messages.append({
        "role": "user",
        "content": user_input,
    })

    # Retrieve relevant long-term memories
    related_memories = memory.search(
        user_input,
        user_id=user_id,
    )

    print("\nMemories:")
    print(related_memories)

    related_memories_text = "\n - ".join(
        m["memory"] for m in related_memories
    )

    system_message = [
        {
            "role": "system",
            "content": f"""Answer the user's question honestly.

Here is relevant information that previous interactions
with this user have taught us:

- {related_memories_text}
""",
        }
    ]

    response = client.chat.completions.create(
        model="gemini-3.5-flash-lite",
        messages=system_message + messages,
        reasoning_effort="low",
    )

    answer = response.choices[0].message.content

    messages.append({
        "role": "assistant",
        "content": answer,
    })

    print(f"\nAssistant: {answer}\n")

    # Store the latest user/assistant exchange in Mem0
    memory.add(
        messages[-2:],
        user_id=user_id,
    )