import asyncio
import os

from google import genai
from google.genai import types

client = None


async def generate_embeddings(strings: list[str]):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    out = await client.aio.models.embed_content(
        model="gemini-embedding-001",
        contents=strings,
        config=types.EmbedContentConfig(output_dimensionality=768),
    )
    embeddings = [item.values for item in out.embeddings]
    return embeddings

if __name__ == "__main__":
    texts = [
        "Hello how are you",
        "I like Machine Learning"
    ]
    asyncio.run(generate_embeddings(texts))
