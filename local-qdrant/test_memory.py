import os

from mem0 import Memory


GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

config = {
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
            "model": "gemini-3.5-flash-lite",
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


memory = Memory.from_config(config)

print("Mem0 + Qdrant initialized successfully!")

result = memory.add(
    "My name is Rahul and I am a computer science student.",
    user_id="avb",
)

print("\nADD RESULT:")
print(result)

result = memory.search(
    "What is my name?",
    user_id="avb",
)

print("\nSEARCH RESULT:")
print(result)