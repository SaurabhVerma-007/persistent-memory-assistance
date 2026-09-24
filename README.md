# Mem0 Memory Chatbots

This project contains a few small experiments with long-term memory for conversational AI. The examples use [Mem0](https://github.com/mem0ai/mem0), [Qdrant](https://qdrant.tech/), OpenAI embeddings, Gemini, and DSPy.

There are three separate implementations:

1. **Custom memory pipeline** - the primary example in `main.py` and `mem/`. It uses DSPy to decide when to search and update memory, OpenAI embeddings, and a Qdrant collection named `memories`.
2. **Mem0 OSS with local Qdrant** - `local-qdrant/local_chatbot.py`. Mem0 manages memory extraction and storage while Gemini provides the language model and embeddings.
3. **Mem0 Cloud example** - `basic_mem0_chatbot.py`. This uses the hosted Mem0 service through `MemoryClient`.

The implementations use different models, collections, and vector dimensions. They are examples to compare, not one shared runtime.

## Prerequisites

- Python 3.10 or newer for the root project
- Python 3.12 or newer for the `local-qdrant` workspace member
- [uv](https://docs.astral.sh/uv/)
- A Qdrant server running at `http://localhost:6333`
- Network access to the model APIs used by the selected example

This repository does not include a Qdrant startup configuration. Start Qdrant separately and make sure the HTTP API is available on port `6333` before running a chatbot.

## Installation

From this directory, install the project and its workspace dependencies:

```powershell
uv sync
```

The code reads API keys from the process environment. Copying `.env.example` alone is not enough because the application does not load `.env` files automatically.

For the custom pipeline, set the OpenAI key:

```powershell
$env:OPENAI_API_KEY = "your-openai-api-key"
```

For the Gemini-based examples, set:

```powershell
$env:GEMINI_API_KEY = "your-gemini-api-key"
```

The Mem0 Cloud example also requires a Mem0 API key:

```powershell
$env:MEM0_API_KEY = "your-mem0-api-key"
```

These environment variables only apply to the current PowerShell session. Use your normal secret-management approach for persistent configuration.

## Quick Start: Custom Pipeline

The root implementation stores memories in Qdrant and uses user IDs to isolate records.

1. Start Qdrant at `http://localhost:6333`.
2. Create the `memories` collection and its payload indexes:

   ```powershell
   uv run python -m mem.vectordb
   ```

3. Start the chat:

   ```powershell
   uv run python main.py
   ```

By default, the chat uses user ID `1`. Pass an integer to use a different user:

```powershell
uv run python main.py 42
```

Example session:

```text
> I like hiking in the mountains.
AI: ...

> What hobbies do I have?
AI: ...
```

The assistant decides whether a turn contains information worth saving. Press `Ctrl+C` to stop the custom chat.

## How the Custom Pipeline Works

The main flow is:

1. `main.py` parses an optional integer user ID and starts the asynchronous chat loop.
2. `mem/response_generator.py` uses DSPy with `gpt-5-mini` to generate a response, decide whether memory is needed, and choose memory categories.
3. When searching, `mem/generate_embeddings.py` creates 64-dimensional `text-embedding-3-small` vectors.
4. `mem/vectordb.py` searches Qdrant for up to two matching memories for the current user, with a score threshold of `0.1`.
5. When a turn should be saved, `mem/update_memory.py` decides whether to add, update, delete, or ignore memory records.

The custom collection is configured with dot-product similarity and stores each memory's user ID, text, categories, timestamp, and vector.

## Alternative Implementations

### Mem0 OSS with Local Qdrant

This example uses Mem0's configuration API with:

- Qdrant collection: `mem0_local`
- Qdrant host: `localhost:6333`
- Gemini chat model: `gemini-3.5-flash-lite`
- Gemini embedding model: `models/gemini-embedding-001`
- Embedding size: 768

Run it from the project root:

```powershell
uv run python local-qdrant/local_chatbot.py
```

It keeps the last ten messages in short-term conversation context, retrieves up to five relevant long-term memories, and accepts `exit` or `quit` to stop.

### Mem0 Cloud

`basic_mem0_chatbot.py` uses `MemoryClient` instead of the local Qdrant configuration. It requires both `MEM0_API_KEY` and `GEMINI_API_KEY`:

```powershell
uv run python basic_mem0_chatbot.py
```

This example also accepts `exit` or `quit`.

## Testing

The local-Qdrant directory contains an executable integration smoke test:

```powershell
uv run python local-qdrant/test_memory.py
```

The test requires a running Qdrant server, `GEMINI_API_KEY`, network access to Gemini, and a working Mem0 installation. It adds a sample memory and searches for it. There is currently no configured unit-test runner or test suite for the custom pipeline.

## Persistence and Collections

Qdrant is an external service. Whether memories survive a restart depends on how that Qdrant server is configured and where it stores its data.

Do not use the same collection for the two local implementations:

| Implementation | Collection | Embeddings | User identifier |
| --- | --- | --- | --- |
| Custom pipeline | `memories` | OpenAI `text-embedding-3-small`, 64 dimensions | Integer, for example `1` or `42` |
| Mem0 OSS local example | `mem0_local` | Gemini `models/gemini-embedding-001`, 768 dimensions | String, currently `avb` |

The vector dimensions and storage behavior are incompatible. Keeping the collections separate avoids querying vectors with the wrong schema.

## Configuration Notes and Limitations

- The Qdrant URL, collection names, model names, and embedding dimensions are currently hardcoded in the Python files.
- `.env.example` lists `GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `GEMINI_MODEL`, and `QDRANT_COLLECTION`, but the checked-in applications currently read only the API keys described above. The other variables are not wired into the runtime.
- The custom chat has no `exit` or `quit` command; stop it with `Ctrl+C`.
- The custom chat keeps the complete current transcript in memory for the session.
- The examples perform network calls synchronously or directly inside the chat workflow, so response and memory-update latency depends on the model and Qdrant services.
- The local-Qdrant package's console script is separate from `local-qdrant/local_chatbot.py`; run the chatbot file directly as shown above.

## Project Layout

```text
main.py                         Custom pipeline entry point
mem/
  response_generator.py         DSPy chat loop and memory decisions
  generate_embeddings.py        OpenAI embedding generation
  vectordb.py                   Qdrant collection and search operations
  update_memory.py              Memory add/update/delete decisions
basic_mem0_chatbot.py           Mem0 Cloud example
local-qdrant/
  local_chatbot.py              Mem0 OSS + local Qdrant chat
  test_memory.py                Local integration smoke test
  pyproject.toml                Local-Qdrant package metadata
.env.example                    Example variable names (not auto-loaded)
```